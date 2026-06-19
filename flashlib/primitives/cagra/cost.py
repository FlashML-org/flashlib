"""Cost model for CAGRA -- graph build (kNN + optimize) + greedy traversal search.

A two-phase tree mirroring an end-to-end ``flash_cagra`` call:

* **build** -- one-time:
    - ``knn_graph`` : the initial degree-``d_init`` kNN graph, routed
      exactly like :mod:`...cagra.triton.build_graph` -- ``flash_knn``
      (brute force) for small ``N``, ``flash_ivf_pq`` self-query for large
      ``N`` (delegated to those primitives' own cost models).
    - ``prune``     : rank-based detour pruning, one CTA per node doing an
      ``O(d_init^2)`` membership reduction (``O(d_init^3)`` element-ops).
    - ``reverse``   : reverse-edge build + merge to the final out-degree
      (sorts/scatters, bandwidth-bound).
* **search** -- per query batch (steady state): a single CTA per query
  runs up to ``max_iterations`` greedy iterations, each expanding
  ``search_width`` frontier nodes -> ``graph_degree`` true squared-L2
  distance evaluations + an on-chip sorted-buffer merge. No ``(nq x
  candidates)`` matrix ever hits HBM.

Shape contract: ``shape = (N, D)`` (the database) with the workload in
``params``:

    params = {"graph_degree":.., "intermediate_degree":.., "itopk_size":..,
              "search_width":.., "max_iterations":.., "k":.., "nq":..,
              "build_algo":.., "nlist":.., "nprobe":..}
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


# Mirror of triton.build_graph._BRUTEFORCE_MAX_N.
_BRUTEFORCE_MAX_N = 50_000


def common(shape, params):
    N, D = shape
    params = params or {}
    gd = int(params.get("graph_degree", 64))
    gd = max(1, min(gd, max(1, N - 1)))
    d_init = int(params.get("intermediate_degree", 2 * gd))
    d_init = max(gd, min(d_init, max(1, N - 1)))
    k = int(params.get("k", 10))
    nq = int(params.get("nq", 10_000))
    itopk = max(int(params.get("itopk_size", 64)), k, gd)
    itopk = _next_pow2(itopk)
    sw = max(1, int(params.get("search_width", 1)))
    max_iters = int(params.get("max_iterations", 0)) or (math.ceil(itopk / sw) + 16)
    algo = params.get("build_algo", "auto")
    if algo == "auto":
        algo = "bruteforce" if N <= _BRUTEFORCE_MAX_N else "ivf_pq"
    nlist = int(params.get("nlist", min(N, max(32, int(N ** 0.5) * 2))))
    nprobe = int(params.get("nprobe", min(nlist, max(16, nlist // 25))))
    return (N, D, gd, d_init, k, nq, itopk, sw, max_iters, algo, nlist, nprobe)


# ── build phase ─────────────────────────────────────────────────────────────
def _build_subops(N, D, gd, d_init, algo, nlist, nprobe, dtype, device):
    db = _dtype_bytes(dtype)

    if algo == "bruteforce":
        knn = _est("knn", shape=(1, N, N, D), params={"k": d_init + 1},
                   dtype=dtype, device=device)
    else:
        knn = _est("ivf_pq", shape=(N, D),
                   params={"nlist": nlist, "nprobe": nprobe, "k": d_init + 1,
                           "nq": N, "m": max(1, min(D, 64))},
                   dtype=dtype, device=device)
    knn.op_name = f"cagra.build.knn_graph[{algo}]"

    # Detour prune: one CTA/node, O(d_init^2) rank look-ups (O(d_init^3) ops).
    prune_flops = float(N) * d_init * d_init
    prune_bytes = float(N) * d_init * d_init * 4 + N * d_init * 4
    pr_rt, pr_bound = roofline(prune_flops, prune_bytes, dtype, device,
                               op_type="elementwise", n_launches=1)
    prune = Estimate(
        op_name="cagra.build.prune", runtime_ms=pr_rt,
        flops=prune_flops, bytes_moved=prune_bytes,
        memory_peak_gb=N * _next_pow2(d_init) * 4 / 1e9, bound=pr_bound,
        confidence="roofline", n_kernel_launches=1,
        notes=[f"rank-based detour prune: N={N} CTAs x d_init={d_init}^2 "
               "rank look-ups -> top graph_degree"],
        dtype=dtype, device=device,
    )

    # Reverse edges + merge: sorts/scatters over the N*graph_degree edge set.
    rev_bytes = float(N) * gd * 4 * 6
    rv_rt, rv_bound = roofline(0.0, rev_bytes, dtype, device,
                               op_type="elementwise", n_launches=6)
    reverse = Estimate(
        op_name="cagra.build.reverse_merge", runtime_ms=rv_rt,
        flops=0.0, bytes_moved=rev_bytes,
        memory_peak_gb=N * gd * 8 / 1e9, bound=rv_bound,
        confidence="roofline", n_kernel_launches=6,
        notes=["reverse-edge build (sort by dst) + interleave/dedup merge "
               f"to out-degree {gd}"],
        dtype=dtype, device=device,
    )
    return [knn, prune, reverse]


# ── search phase ────────────────────────────────────────────────────────────
def _search_subops(N, D, gd, k, nq, itopk, sw, max_iters, dtype, device):
    db = _dtype_bytes(dtype)

    # Distance evals: nq * iters * search_width * graph_degree D-dim squared-L2.
    n_dist = float(nq) * max_iters * sw * gd
    dist_flops = 2.0 * n_dist * D
    # Dataset gathers dominate traffic (graph-scattered, little reuse).
    dist_bytes = n_dist * D * db + float(nq) * max_iters * sw * gd * 4
    d_rt, d_bound = roofline(dist_flops, dist_bytes, dtype, device,
                             op_type="elementwise", n_launches=1)
    expand = Estimate(
        op_name="cagra.search.expand", runtime_ms=d_rt,
        flops=dist_flops, bytes_moved=dist_bytes,
        memory_peak_gb=nq * itopk * 8 / 1e9, bound=d_bound,
        confidence="roofline", n_kernel_launches=1,
        suggested_config={"itopk_size": itopk, "search_width": sw, "k": k},
        notes=[f"greedy expand: nq={nq} x iters={max_iters} x sw={sw} x "
               f"deg={gd} true squared-L2 ({D}-dim); single CTA/query, "
               "no (nq x candidates) HBM matrix"],
        dtype=dtype, device=device,
    )

    # On-chip buffer maintenance: per merge an O(itopk^2) dedup + 2*itopk sort.
    merge_flops = float(nq) * max_iters * sw * (itopk * itopk + 2 * itopk * math.log2(max(2, 2 * itopk)))
    merge_bytes = float(nq) * itopk * 8
    m_rt, m_bound = roofline(merge_flops, merge_bytes, dtype, device,
                             op_type="elementwise", n_launches=1)
    merge = Estimate(
        op_name="cagra.search.buffer_merge", runtime_ms=m_rt,
        flops=merge_flops, bytes_moved=merge_bytes,
        memory_peak_gb=nq * itopk * 8 / 1e9, bound=m_bound,
        confidence="roofline", n_kernel_launches=1,
        notes=[f"on-chip sorted top-{itopk} buffer: per-merge O(itopk^2) "
               "dedup + bitonic sort (registers, no HBM)"],
        dtype=dtype, device=device,
    )
    return [expand, merge]


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
    """End-to-end CAGRA cost (build + search) as a sub-op tree."""
    (N, D, gd, d_init, k, nq, itopk, sw, max_iters, algo, nlist, nprobe) = common(
        shape, params)
    build = _compose(
        "cagra.build",
        _build_subops(N, D, gd, d_init, algo, nlist, nprobe, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"one-time build via {algo}: N={N}, D={D}, "
               f"d_init={d_init} -> graph_degree={gd}"],
        suggested={"graph_degree": gd, "intermediate_degree": d_init},
    )
    search = _compose(
        "cagra.search",
        _search_subops(N, D, gd, k, nq, itopk, sw, max_iters, dtype, device),
        tol=tol, dtype=dtype, device=device,
        notes=[f"per-batch search: nq={nq}, k={k}, itopk={itopk}, "
               f"search_width={sw}, max_iters={max_iters}"],
        suggested={"itopk_size": itopk, "search_width": sw, "k": k},
    )
    graph_gb = N * gd * 4 / 1e9
    return _compose(
        "cagra", [build, search], tol=tol, dtype=dtype, device=device,
        notes=[
            f"N={N}, D={D}, graph_degree={gd}, itopk_size={itopk}, "
            f"search_width={sw}, k={k}, nq={nq}",
            f"graph storage: {graph_gb:.3f} GB ({gd} int32 edges/node) "
            f"+ dataset {N * D * _dtype_bytes(dtype) / 1e9:.3f} GB",
            "recall set by graph quality (graph_degree / build) + search "
            "budget (itopk_size / search_width); true squared-L2 distances",
        ],
        suggested={"graph_degree": gd, "itopk_size": itopk,
                   "search_width": sw, "k": k},
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Suggest ``(graph_degree, itopk_size)`` -- cuVS-style defaults."""
    (N, D, gd, d_init, k, _nq, itopk, sw, _mi, _algo, _nl, _np) = common(shape, params)
    return {
        "graph_degree": gd,
        "intermediate_degree": d_init,
        "itopk_size": itopk,
        "search_width": sw,
        "k": k,
    }


# ── GPU op-name shim ─────────────────────────────────────────────────────────
# CAGRA has a single GPU backend (Triton); the torch fallback is a CPU
# reference, not a Pareto variant, so only the Triton estimate is registered.
def estimate_cagra_triton(shape, params=None, tol=None, dtype="float32",
                          device="H100", **_):
    est = estimate(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = "cagra_triton"
    est.tol = tol
    return est
