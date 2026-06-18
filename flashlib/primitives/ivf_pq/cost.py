"""Cost model for IVF-PQ -- kmeans (build) + knn (coarse) + LUT + ADC fine-scan.

The estimate is a two-phase tree mirroring an end-to-end ``flash_ivf_pq``
call:

* **build** -- one-time:
    - ``kmeans``   : coarse quantizer trained on a ``min(M, nlist*256)`` sample.
    - ``assign``   : one x²-free pass over all ``M`` rows vs ``nlist`` centroids.
    - ``pq_train`` : ``m`` sub-quantizers (``ksub=256`` each) trained on a
                     residual sub-sample as one batched k-means.
    - ``encode``   : encode all ``M`` residuals to ``(M, m)`` uint8 codes.
    - ``layout``   : bincount + cumsum + argsort + cell-contiguous reorder
                     of the compact code array.
* **search** -- per query batch (the steady-state cost). Two strategies,
  routed exactly like the runtime ``_pick_regime`` (enough work, then the
  dsub/qpl/m crossover; large dsub or large ``m`` keep the LUT ahead):

  - **No-LUT decode + GEMM** (enough work *and* the crossover favours GEMM):
      - ``coarse``  : ``flash_knn`` top-``nprobe`` over the ``nlist`` centroids.
      - ``group``   : inverse map -- argsort the ``nq*nprobe`` ``(query, list)``
                      pairs so each list's probing queries are contiguous.
      - ``fine``    : cluster-centric decode of the probed PQ codes + a
                      tensor-core cross term (ADC as a GEMM, **no LUT**).
      - ``rerank``  : exact ADC over an oversampled candidate pool.
    Builds no LUT, so peak memory is just the ``nq*nprobe*k`` partials.

  - **ADC LUT scan** (small batches):
      - ``coarse``  : as above.
      - ``lut``     : build the ``(nq, P, m, 256)`` tables (``P = nprobe`` if
                      ``by_residual`` else ``1``), flash-tiled over queries.
      - ``fine``    : fused ragged-code scan, ``m``-way LUT gather/candidate.

The shape contract is ``shape = (M, D)`` (the database) with the search
workload supplied via ``params``:

    params = {"nlist":.., "nprobe":.., "k":.., "nq":.., "m":..,
              "nbits":.., "by_residual":.., "niter":.., "pq_niter":..}
"""
import math

from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype in ("fp32", "float32", "tf32", "float") else 2


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _pq_dims(D: int, m: int) -> tuple:
    """Pure-python mirror of torch_fallback._pq_dims (no torch import)."""
    m = max(1, int(m))
    if m > D:
        m = int(D)
    dsub = int(math.ceil(D / m))
    while m * dsub < 16:
        dsub += 1
    return m, dsub, m * dsub


def common(shape, params):
    M, D = shape
    params = params or {}
    nlist = int(params.get("nlist", 1024))
    nlist = max(1, min(nlist, M))
    nprobe = int(params.get("nprobe", 8))
    nprobe = max(1, min(nprobe, nlist))
    k = int(params.get("k", 10))
    nq = int(params.get("nq", 10_000))
    niter = int(params.get("niter", params.get("max_iters", 20)))
    pq_niter = int(params.get("pq_niter", 25))
    nbits = int(params.get("nbits", 8))
    ksub = 1 << nbits
    by_residual = bool(params.get("by_residual", True))
    m, dsub, Dp = _pq_dims(int(D), int(params.get("m", 8)))
    train_size = int(params.get("train_size") or min(M, nlist * 256))
    train_size = max(min(train_size, M), nlist)
    pq_train_size = int(params.get("pq_train_size") or min(M, max(ksub * 16, 4096)))
    pq_train_size = max(min(pq_train_size, M), ksub)
    return (M, D, nlist, nprobe, k, nq, niter, pq_niter, nbits, ksub,
            by_residual, m, dsub, Dp, train_size, pq_train_size)


# ── build phase ───────────────────────────────────────────────────────────
def _build_subops(M, nlist, niter, pq_niter, ksub, m, dsub, Dp,
                  train_size, pq_train_size, dtype, device):
    db = _dtype_bytes(dtype)

    km = _est("kmeans", shape=(train_size, Dp),
              params={"K": nlist, "max_iters": niter},
              dtype=dtype, device=device)
    km.op_name = "ivf_pq.build.kmeans"

    # Full-database assignment: one x²-free (M, nlist, Dp) pass.
    assign_flops = 2 * M * nlist * Dp
    assign_bytes = (M * Dp + nlist * Dp) * db + M * 4
    a_rt, a_bound = roofline(assign_flops, assign_bytes, dtype, device,
                             op_type="kmeans", n_launches=1)
    assign = Estimate(
        op_name="ivf_pq.build.assign", runtime_ms=a_rt,
        flops=assign_flops, bytes_moved=assign_bytes,
        memory_peak_gb=M * Dp * db / 1e9, bound=a_bound,
        confidence="calibrated", n_kernel_launches=1,
        notes=[f"assign M={M} rows to nlist={nlist} centroids (x²-free)"],
        dtype=dtype, device=device,
    )

    # PQ codebook training: m batched k-means, ksub centroids, dsub dims.
    pqt_flops = m * 2 * pq_train_size * ksub * dsub * pq_niter
    pqt_bytes = m * pq_train_size * dsub * db * pq_niter + m * ksub * dsub * db
    p_rt, p_bound = roofline(pqt_flops, pqt_bytes, dtype, device,
                             op_type="kmeans", n_launches=pq_niter)
    pq_train = Estimate(
        op_name="ivf_pq.build.pq_train", runtime_ms=p_rt,
        flops=pqt_flops, bytes_moved=pqt_bytes,
        memory_peak_gb=pq_train_size * Dp * db / 1e9, bound=p_bound,
        confidence="roofline", n_kernel_launches=pq_niter,
        notes=[f"train m={m} sub-quantizers (ksub={ksub}, dsub={dsub}) "
               "as one batched k-means"],
        dtype=dtype, device=device,
    )

    # Encode all residuals: m batched assigns of (M, ksub, dsub).
    enc_flops = 2 * M * ksub * Dp
    enc_bytes = M * Dp * db + m * ksub * dsub * db + M * m
    e_rt, e_bound = roofline(enc_flops, enc_bytes, dtype, device,
                             op_type="kmeans", n_launches=1)
    encode = Estimate(
        op_name="ivf_pq.build.encode", runtime_ms=e_rt,
        flops=enc_flops, bytes_moved=enc_bytes,
        memory_peak_gb=M * Dp * db / 1e9, bound=e_bound,
        confidence="roofline", n_kernel_launches=1,
        notes=[f"encode M={M} residuals -> (M, m={m}) uint8 codes"],
        dtype=dtype, device=device,
    )

    # CSR layout: bincount + cumsum + argsort + reorder of (M, m) uint8 codes.
    layout_bytes = M * m * 2 + M * 4 * 3
    l_rt, l_bound = roofline(0.0, layout_bytes, dtype, device,
                             op_type="elementwise", n_launches=4)
    layout = Estimate(
        op_name="ivf_pq.build.layout", runtime_ms=l_rt,
        flops=0.0, bytes_moved=layout_bytes,
        memory_peak_gb=M * m / 1e9, bound=l_bound,
        confidence="roofline", n_kernel_launches=4,
        notes=["bincount + cumsum + argsort + cell-contiguous code reorder"],
        dtype=dtype, device=device,
    )
    return [km, assign, pq_train, encode, layout]


# Mirror of triton.search._pick_regime: no-LUT decode+GEMM is chosen once
# there are enough queries (to amortise grouping) AND candidate comparisons
# (to repay its floor) AND the dsub/qpl/m crossover favours GEMM -- short
# sub-vectors decode cheaply, but large dsub or large m keep the LUT ahead.
# See that module for the sweep that calibrates these.
_GEMM_MIN_NQ = 256
_GEMM_MIN_WORK = 2_000_000
_DSUB_LUT_MIN = 9
_QPL_LUT_SLOPE = 4.0
_M_LUT_MIN = 48
_DSUB_LUT_ALWAYS = 48


def _route_is_gemm(nq, nprobe, M, nlist, dsub, m):
    """True iff search routes to no-LUT decode+GEMM (mirror of _pick_regime)."""
    work = nq * nprobe * (M / max(nlist, 1))
    if nq < _GEMM_MIN_NQ or work < _GEMM_MIN_WORK:
        return False
    if dsub < _DSUB_LUT_MIN:
        return True
    if m >= _M_LUT_MIN or dsub >= _DSUB_LUT_ALWAYS:
        return False
    return nq * nprobe / max(nlist, 1) > _QPL_LUT_SLOPE * dsub


# ── search phase ──────────────────────────────────────────────────────────
def _search_subops_gemm(M, nlist, nprobe, k, nq, by_residual, m, dsub, Dp,
                        dtype, device):
    """No-LUT cluster-centric decode + tensor-core GEMM (+ exact re-rank)."""
    db = _dtype_bytes(dtype)
    cand = nq * nprobe * (M / max(nlist, 1))
    topk_pad = _next_pow2(k)
    over_pad = _next_pow2(k * 2)

    coarse = _est("knn", shape=(1, nq, nlist, Dp), params={"k": nprobe},
                  dtype=dtype, device=device)
    coarse.op_name = "ivf_pq.search.coarse"

    # Inverse map: argsort/bincount/cumsum over the nq*nprobe (query,list) pairs.
    P = nq * nprobe
    group_bytes = P * 4 * 4
    g_rt, g_bound = roofline(0.0, group_bytes, dtype, device,
                             op_type="elementwise", n_launches=3)
    group = Estimate(
        op_name="ivf_pq.search.group", runtime_ms=g_rt,
        flops=0.0, bytes_moved=group_bytes,
        memory_peak_gb=P * 4 * 2 / 1e9, bound=g_bound,
        confidence="roofline", n_kernel_launches=3,
        notes=[f"inverse map: argsort {P} (query,list) pairs by list"],
        dtype=dtype, device=device,
    )

    # Fine scan: decode codes (cand*m uint8, shared across the query tile) and
    # a WGMMA cross term (2*cand*Dp MACs). Partials are the only big alloc.
    fine_flops = 2 * cand * Dp
    fine_bytes = cand * m + nq * nprobe * Dp * db + nq * nprobe * topk_pad * 8
    f_rt, f_bound = roofline(fine_flops, fine_bytes, dtype, device,
                             op_type="ivf_pq_search", n_launches=1)
    fine = Estimate(
        op_name="ivf_pq.search.fine", runtime_ms=f_rt,
        flops=fine_flops, bytes_moved=fine_bytes,
        memory_peak_gb=nq * nprobe * topk_pad * 8 / 1e9, bound=f_bound,
        confidence="calibrated", n_kernel_launches=1,
        suggested_config={"nprobe": nprobe, "k": k, "m": m},
        notes=[
            f"no-LUT decode+GEMM: ~{cand:.0f} candidates decoded + tensor-core "
            f"cross term (nq={nq}, nprobe={nprobe}, M/nlist={M/max(nlist,1):.0f}); "
            "no ADC LUT, no (nq x candidates) HBM matrix",
        ],
        dtype=dtype, device=device,
    )

    # Exact ADC re-rank of an oversampled pool (~k*2 candidates/query).
    rr_flops = 2 * nq * over_pad * Dp
    rr_bytes = nq * over_pad * m + nq * Dp * db + nq * over_pad * 4
    r_rt, r_bound = roofline(rr_flops, rr_bytes, dtype, device,
                             op_type="ivf_pq_search", n_launches=1)
    rerank = Estimate(
        op_name="ivf_pq.search.rerank", runtime_ms=r_rt,
        flops=rr_flops, bytes_moved=rr_bytes,
        memory_peak_gb=nq * over_pad * 4 / 1e9, bound=r_bound,
        confidence="roofline", n_kernel_launches=1,
        notes=[f"exact ADC re-rank of ~{over_pad} candidates/query "
               "(tf32 GEMM selects, exact decode ranks)"],
        dtype=dtype, device=device,
    )
    return [coarse, group, fine, rerank]


def _search_subops(M, nlist, nprobe, k, nq, ksub, by_residual, m, dsub, Dp,
                   dtype, device):
    # Route exactly like the runtime (triton.search._pick_regime).
    if _route_is_gemm(nq, nprobe, M, nlist, dsub, m):
        return _search_subops_gemm(M, nlist, nprobe, k, nq, by_residual,
                                   m, dsub, Dp, dtype, device)

    db = _dtype_bytes(dtype)
    P = nprobe if by_residual else 1

    coarse = _est("knn", shape=(1, nq, nlist, Dp), params={"k": nprobe},
                  dtype=dtype, device=device)
    coarse.op_name = "ivf_pq.search.coarse"

    # LUT build: (nq, P, m, ksub) entries, each a dsub-dim squared distance.
    # Total work/traffic is unchanged by tiling, but the *live* table is
    # bounded: search tiles over query blocks (flash-attention style) so
    # only a (q_tile, P, m, ksub) table exists at once, capped at ~2 GiB.
    _LUT_BUDGET_GB = 2.0
    lut_flops = 2 * nq * P * m * ksub * dsub
    lut_bytes = nq * P * m * ksub * 4 + nq * Dp * db + (nq * P * Dp * db if by_residual else 0)
    full_lut_gb = nq * P * m * ksub * 4 / 1e9
    lut_peak_gb = min(full_lut_gb, _LUT_BUDGET_GB)
    lu_rt, lu_bound = roofline(lut_flops, lut_bytes, dtype, device,
                               op_type="ivf_pq_search", n_launches=1)
    lut = Estimate(
        op_name="ivf_pq.search.lut", runtime_ms=lu_rt,
        flops=lut_flops, bytes_moved=lut_bytes,
        memory_peak_gb=lut_peak_gb, bound=lu_bound,
        confidence="roofline", n_kernel_launches=1,
        notes=[f"ADC tables (nq={nq}, P={P}, m={m}, ksub={ksub}); "
               f"query-tiled, live LUT <= {lut_peak_gb:.2f}GB "
               f"(full {full_lut_gb:.2f}GB never materialised)"],
        dtype=dtype, device=device,
    )

    # Fine scan: scan nprobe lists of avg length M/nlist; m gathers per candidate.
    cand = nq * nprobe * (M / max(nlist, 1))
    topk_pad = _next_pow2(k)
    fine_flops = cand * m
    fine_bytes = cand * m + cand * m * 4 + nq * nprobe * topk_pad * 8
    f_rt, f_bound = roofline(fine_flops, fine_bytes, dtype, device,
                             op_type="ivf_pq_search", n_launches=1)
    fine = Estimate(
        op_name="ivf_pq.search.fine", runtime_ms=f_rt,
        flops=fine_flops, bytes_moved=fine_bytes,
        memory_peak_gb=nq * nprobe * topk_pad * 8 / 1e9, bound=f_bound,
        confidence="calibrated", n_kernel_launches=1,
        suggested_config={"nprobe": nprobe, "k": k, "m": m},
        notes=[
            f"fused ADC scan: ~{cand:.0f} candidates x m={m} LUT gathers "
            f"(nq={nq}, nprobe={nprobe}, M/nlist={M/max(nlist,1):.0f}); "
            "no (nq x candidates) HBM matrix",
        ],
        dtype=dtype, device=device,
    )
    return [coarse, lut, fine]


def _compose(name, subops, *, tol, dtype, device, notes, suggested):
    total_rt = sum(s.runtime_ms for s in subops)
    flops = sum(s.flops for s in subops)
    bytes_moved = sum(s.bytes_moved for s in subops)
    dominant = max(subops, key=lambda s: s.runtime_ms)
    return Estimate(
        op_name=name, runtime_ms=total_rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=max((s.memory_peak_gb for s in subops), default=0.0),
        bound=dominant.bound, confidence="calibrated",
        n_kernel_launches=sum(s.n_kernel_launches for s in subops),
        suggested_config=suggested, subops=subops, notes=notes,
        tol=tol, dtype=dtype, device=device,
    )


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """End-to-end IVF-PQ cost (build + search) as a sub-op tree."""
    (M, D, nlist, nprobe, k, nq, niter, pq_niter, nbits, ksub,
     by_residual, m, dsub, Dp, train_size, pq_train_size) = common(shape, params)
    build = _compose(
        "ivf_pq.build",
        _build_subops(M, nlist, niter, pq_niter, ksub, m, dsub, Dp,
                      train_size, pq_train_size, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"one-time build: M={M}, D={D}, nlist={nlist}, m={m}, "
               f"dsub={dsub}, nbits={nbits}, by_residual={by_residual}"],
        suggested={"nlist": nlist, "m": m, "niter": niter},
    )
    search = _compose(
        "ivf_pq.search",
        _search_subops(M, nlist, nprobe, k, nq, ksub, by_residual, m, dsub, Dp,
                       dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"per-batch search: nq={nq}, k={k}, nprobe={nprobe}, m={m}"],
        suggested={"nprobe": nprobe, "k": k},
    )
    code_bytes = m
    return _compose(
        "ivf_pq", [build, search], tol=tol, dtype=dtype, device=device,
        notes=[
            f"M={M}, D={D}, nlist={nlist}, nprobe={nprobe}, k={k}, nq={nq}, "
            f"m={m}, nbits={nbits}",
            f"compression: {4*D}B fp32 vector -> {code_bytes}B PQ code "
            f"({4.0*D/max(code_bytes,1):.0f}x)",
            "recall fixed by (nlist, nprobe, m, codebooks); ADC distances",
        ],
        suggested={"nlist": nlist, "nprobe": nprobe, "k": k, "m": m},
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Suggest ``(nlist, nprobe, m)`` -- FAISS-style ``nlist ~ sqrt(M)``."""
    (M, D, nlist, nprobe, k, _nq, _ni, _pqi, _nb, _ks,
     _byr, m, _ds, _Dp, _ts, _pts) = common(shape, params)
    sqrt_nlist = int(max(1, round(M ** 0.5)))
    return {
        "nlist": params.get("nlist", sqrt_nlist) if params else sqrt_nlist,
        "nprobe": nprobe,
        "m": m,
        "k": k,
    }


# ── GPU op-name shim ───────────────────────────────────────────────────────
# IVF-PQ has a single GPU backend (Triton); this is the canonical op_name the
# runtime reports on CUDA (see impl.route_op_name) and what
# info.estimate("ivf_pq_triton") resolves to. The torch fallback is a CPU
# reference, not a Pareto variant, so it is intentionally not registered.
def estimate_ivf_pq_triton(shape, params=None, tol=None, dtype="float32",
                           device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "ivf_pq_triton"
    est.tol = tol
    return est
