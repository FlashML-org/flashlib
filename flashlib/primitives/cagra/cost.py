"""Cost model for CAGRA -- knn-graph build + greedy-traversal search.

The estimate is a two-phase tree mirroring an end-to-end ``flash_cagra``
call:

* **build** -- one-time:
    - ``knn``    : exact kNN graph over the database
                   (reuses :mod:`flashlib.primitives.knn.cost` at
                   ``k = intermediate_graph_degree + 1``).
    - ``prune``  : detour pruning -- per vertex a ``(igd, igd)``
                   pairwise-distance tile + detour count
                   (``M * igd^2 * (2D + 3)`` flops, gemm-class).
    - ``merge``  : reverse-edge grouping + dedupe (argsorts + gathers
                   over ``M * igd`` edges; bandwidth-bound).
* **search** -- per query batch (the steady-state cost):
    - ``traverse`` : per query ~``iters`` iterations, each scoring
                     ``search_width * graph_degree`` candidates
                     (``2D`` flops each) after gathering their rows
                     from HBM. Random-gather bound: the useful bytes
                     are ``nq * iters * sw * gd * D * dtype_bytes``
                     with cache-line rounding, at gather (not
                     streaming) efficiency.

Iteration count: traversal converges when the ``itopk`` window is fully
visited; empirically ``iters ~ itopk / search_width + slack``, the
model uses that directly.

The shape contract is ``shape = (M, D)`` (the database) with the search
workload supplied via ``params``:

    params = {"graph_degree":.., "intermediate_graph_degree":..,
              "itopk":.., "search_width":.., "k":.., "nq":..}
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.info.dispatch import estimate as _est


def _dtype_bytes(dtype: str) -> int:
    return 4 if dtype in ("fp32", "float32", "tf32", "float") else 2


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def common(shape, params):
    M, D = shape
    params = params or {}
    gd = int(params.get("graph_degree", 32))
    gd = _next_pow2(max(2, min(gd, M - 1)))
    igd = int(params.get("intermediate_graph_degree", min(2 * gd, 96)))
    igd = max(gd, min(igd, 128, M - 1))
    k = int(params.get("k", 10))
    itopk = int(params.get("itopk_size", params.get("itopk", max(64, k))))
    sw = int(params.get("search_width", 1))
    nq = int(params.get("nq", 10_000))
    return M, D, gd, igd, itopk, sw, k, nq


# ── build phase ───────────────────────────────────────────────────────────
def _build_subops(M, D, gd, igd, dtype, device):
    db = _dtype_bytes(dtype)

    knn = _est("knn", shape=(1, M, M, D), params={"k": igd + 1},
               dtype=dtype, device=device)
    knn.op_name = "cagra.build.knn"

    # Detour prune: per chunk row a (igd, D) gather, an (igd, igd) gram,
    # and the detour-count reduction.
    prune_flops = M * igd * igd * (2 * D + 3)
    prune_bytes = M * igd * (D * db + 8) + M * gd * 4
    p_rt, p_bound = roofline(prune_flops, prune_bytes, dtype, device,
                             op_type="gemm", n_launches=max(1, M // 4096))
    prune = Estimate(
        op_name="cagra.build.prune", runtime_ms=p_rt,
        flops=prune_flops, bytes_moved=prune_bytes,
        memory_peak_gb=4096 * igd * igd * 4 / 1e9, bound=p_bound,
        confidence="roofline", n_kernel_launches=max(1, M // 4096),
        notes=[f"detour prune igd={igd} -> gd={gd} "
               f"((igd,igd) pairwise tile per vertex)"],
        dtype=dtype, device=device,
    )

    # Reverse merge: argsort + bincount + scatter over M*gd edges, then a
    # (M, ~3gd) dedupe pass.
    merge_bytes = M * gd * 8 * 6 + M * 3 * gd * 8 * 3
    m_rt, m_bound = roofline(0.0, merge_bytes, dtype, device,
                             op_type="elementwise", n_launches=12)
    merge = Estimate(
        op_name="cagra.build.merge", runtime_ms=m_rt,
        flops=0.0, bytes_moved=merge_bytes,
        memory_peak_gb=M * 3 * gd * 8 / 1e9, bound=m_bound,
        confidence="roofline", n_kernel_launches=12,
        notes=["reverse-edge grouping + dedupe + pad"],
        dtype=dtype, device=device,
    )
    return [knn, prune, merge]


# ── search phase ──────────────────────────────────────────────────────────
def _search_subops(M, D, gd, itopk, sw, k, nq, dtype, device):
    db = _dtype_bytes(dtype)

    itopk_pad = _next_pow2(itopk)
    iters = itopk_pad // max(sw, 1) + 16
    cand_per_iter = sw * gd

    flops = 2.0 * nq * iters * cand_per_iter * D
    # gather reads round to 128B lines; graph rows are coalesced gd*4.
    line = 128
    row_bytes = ((D * db + line - 1) // line) * line
    bytes_moved = nq * iters * (cand_per_iter * row_bytes + sw * gd * 4) \
        + nq * k * 12
    t_rt, t_bound = roofline(flops, bytes_moved, dtype, device,
                             op_type="cagra_search", n_launches=1)
    traverse = Estimate(
        op_name="cagra.search.traverse", runtime_ms=t_rt,
        flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=nq * k * 12 / 1e9, bound=t_bound,
        confidence="calibrated", n_kernel_launches=1,
        suggested_config={"itopk": itopk, "search_width": sw, "k": k},
        notes=[
            f"greedy traversal ~{iters} iters x {cand_per_iter} candidates "
            f"per query (random-gather bound; no HBM intermediates)",
        ],
        dtype=dtype, device=device,
    )
    return [traverse]


def _compose(name, subops, *, tol, dtype, device, notes, suggested):
    total_rt = sum(s.runtime_ms for s in subops)
    flops = sum(s.flops for s in subops)
    bytes_moved = sum(s.bytes_moved for s in subops)
    dominant = max(subops, key=lambda s: s.runtime_ms)
    return Estimate(
        op_name=name, runtime_ms=total_rt, flops=flops,
        bytes_moved=bytes_moved,
        memory_peak_gb=max((s.memory_peak_gb for s in subops), default=0.0),
        bound=dominant.bound, confidence="calibrated",
        n_kernel_launches=sum(s.n_kernel_launches for s in subops),
        suggested_config=suggested, subops=subops, notes=notes,
        tol=tol, dtype=dtype, device=device,
    )


def estimate(shape, params=None, tol=None, dtype="float32", device="H100",
             **_):
    """End-to-end CAGRA cost (build + search) as a sub-op tree."""
    M, D, gd, igd, itopk, sw, k, nq = common(shape, params)
    build = _compose(
        "cagra.build",
        _build_subops(M, D, gd, igd, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"one-time build: M={M}, D={D}, gd={gd}, igd={igd}"],
        suggested={"graph_degree": gd, "intermediate_graph_degree": igd},
    )
    search = _compose(
        "cagra.search",
        _search_subops(M, D, gd, itopk, sw, k, nq, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"per-batch search: nq={nq}, k={k}, itopk={itopk}, "
               f"search_width={sw}"],
        suggested={"itopk": itopk, "search_width": sw, "k": k},
    )
    return _compose(
        "cagra", [build, search], tol=tol, dtype=dtype, device=device,
        notes=[
            f"M={M}, D={D}, gd={gd}, itopk={itopk}, k={k}, nq={nq}",
            "recall governed by (graph_degree, itopk, search_width); "
            "benchmark the recall/QPS frontier, not fixed params",
        ],
        suggested={"graph_degree": gd, "itopk": itopk, "k": k},
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100",
              **_):
    """Suggest ``(graph_degree, itopk)`` for a target recall band."""
    M, _D, gd, igd, itopk, sw, k, _nq = common(shape, params)
    return {
        "graph_degree": gd,
        "intermediate_graph_degree": igd,
        "itopk": itopk,
        "search_width": sw,
        "k": k,
    }


# ── GPU op-name shim ───────────────────────────────────────────────────────
def estimate_cagra_triton(shape, params=None, tol=None, dtype="float32",
                          device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "cagra_triton"
    est.tol = tol
    return est
