"""Cost models for K-Means -- smart dispatcher + per-backend.

``estimate(...)`` mirrors the runtime ``flash_kmeans`` dispatcher: it picks
a backend by ``(shape, tol, backend)`` and returns that backend's estimate.
``estimate_kmeans_triton`` and ``estimate_kmeans_cutedsl`` are exposed
separately so ``info.variants("kmeans", ...)`` can compare the alternatives
without running anything.

The assign step dominates on every shape we benchmark (it's a
``(N, K, D)`` GEMM equivalent with an in-register epilogue), so we
model it as a calibrated KMeans-class op via
:data:`flashlib.info.roofline._SUSTAINED_TFLOPS[("kmeans", dtype, dev)]`
and add a small bandwidth-bound update pass for the Lloyd centroid step.

Triton vs CuteDSL: per ``benchmarks/results/boundaries_kmeans.md`` the
two paths land within ±5 % of each other for every shape in the
FA3-eligible regime (D <= 512, D % 16 = 0, B = 1); we model them with
the same compute budget. CuteDSL gets ~15-30 % wins at the
``(K=4096, D=512)`` extreme corner only -- those are encoded as a
shape-conditional ``eff_factor``.
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline


def _shape(shape):
    if len(shape) == 2:
        N, D = shape
        return 1, N, D
    return shape


def common(shape, params):
    B, N, D = _shape(shape)
    params = params or {}
    K = params.get("K", params.get("n_clusters", 10))
    niter = params.get("max_iters", params.get("niter", 25))
    return B, N, D, K, niter


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Smart dispatcher cost -- picks the routed backend.

    Triton is the default (broadest coverage: any B, any D, all metrics).
    CuteDSL is reachable only when the FA3 kernel constraints are met
    (B=1, 16 <= D <= 512, D % 16 = 0); otherwise the cost model falls
    back to Triton. On a Trainium device (``device`` starting ``trn``),
    the NKI backend is modelled instead.
    """
    if str(device).lower().startswith("trn"):
        est = estimate_kmeans_trainium(shape, params=params, tol=tol,
                                       dtype=dtype, device=device)
        est.tol = tol
        return est
    est = estimate_kmeans_triton(shape, params=params, tol=tol, dtype=dtype,
                                  device=device)
    est.op_name = "kmeans_triton"
    est.tol = tol
    return est


def _assign_flops_bytes(B, N, D, K, dtype_bytes):
    """Per-iteration assign + update FLOP + byte counts."""
    # x²-free assign: 2*N*K*D dot products (one CTA per (n, k_block) tile).
    assign_flops = 2 * B * N * K * D
    # Lloyd update: per-point write of the assigned cluster index (N*4 bytes)
    # and a reduction over D for each cluster. Reduction is N*D fp32 ops.
    update_flops = B * N * D
    flops = assign_flops + update_flops
    # Bytes: X read once per iter (large), C read once per iter (small),
    # N*4 assignment writes + K*D centroid write back.
    bytes_moved = (B * N * D + B * K * D) * dtype_bytes + B * N * 4 + B * K * D * 4
    return flops, bytes_moved


def estimate_kmeans_triton(shape, params=None, tol=None, dtype="float32",
                           device="H100", **_):
    """Triton split-D + heuristic backend.

    Uses the calibrated ``("kmeans", dtype, device)`` sustained TFLOPS
    when present (see :data:`flashlib.info.roofline._SUSTAINED_TFLOPS`):
    bf16 lands at ~700 TF effective on H200, fp32 at ~320 TF.
    """
    B, N, D, K, niter = common(shape, params)
    dtype_bytes = 4 if dtype in ("fp32", "float32", "tf32") else 2
    flops_iter, bytes_iter = _assign_flops_bytes(B, N, D, K, dtype_bytes)
    flops = niter * flops_iter
    bytes_moved = niter * bytes_iter
    n_launches = 2 * niter      # assign + update per iter
    rt, bound = roofline(flops, bytes_moved, dtype, device, op_type="kmeans",
                          n_launches=n_launches)
    return Estimate(
        op_name="kmeans_triton",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=B * N * D * dtype_bytes / 1e9,
        bound=bound, confidence="calibrated", n_kernel_launches=n_launches,
        suggested_config={
            "variant": "split_d" if D > 256 else "default",
            "BN": 128, "BK": 64 if K <= 256 else 128,
        },
        subops=[],
        notes=[
            f"B={B}, N={N}, D={D}, K={K}, niter={niter}",
            "Triton x²-free assign + sorted Lloyd update; "
            "calibrated against boundaries_kmeans.md.",
        ],
        expected_residual=None, precision_tier="exact", tol=tol,
    )


def estimate_kmeans_cutedsl(shape, params=None, tol=None, dtype="float32",
                            device="H100", **_):
    """Hopper FA3-style fused TMA+WGMMA assign.

    Hardware constraints: B=1, 16 <= D <= 512, D % 16 = 0. The cost
    model only diverges from Triton at the (K >= 1024, D >= 256)
    corner where CuteDSL pulls ~15-25 % ahead -- everywhere else the
    two paths are tied within measurement noise (boundaries_kmeans.md).
    """
    B, N, D, K, niter = common(shape, params)
    # bf16 storage by default in the cutedsl path
    dtype_bytes = 2
    flops_iter, bytes_iter = _assign_flops_bytes(B, N, D, K, dtype_bytes)
    flops = niter * flops_iter
    bytes_moved = niter * bytes_iter
    n_launches = niter           # fused assign + update -> 1 launch / iter
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="kmeans",
                          n_launches=n_launches)
    # Empirical: cutedsl wins ~20 % only at (K >= 1024, D >= 256); else parity.
    if K >= 1024 and D >= 256:
        rt = rt * 0.80
        note_speedup = "cutedsl ~20% faster than triton at this corner"
    else:
        note_speedup = "cutedsl ~ triton on this shape (boundaries_kmeans.md)"
    return Estimate(
        op_name="kmeans_cutedsl",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=B * N * D * 2 / 1e9,
        bound=bound, confidence="measured", n_kernel_launches=n_launches,
        suggested_config={"BM": 128, "BN": 256, "use_ws": False},
        subops=[],
        notes=[
            f"B={B}, N={N}, D={D}, K={K}, niter={niter}",
            "Hopper TMA+WGMMA fused assign; B=1, 16<=D<=512, 16|D required.",
            note_speedup,
        ],
        expected_residual=None, precision_tier="exact", tol=tol,
    )


def estimate_kmeans_trainium(shape, params=None, tol=None, dtype="bf16",
                             device="trn2", **_):
    """AWS Trainium NKI backend (single NeuronCore).

    The flash assign is a ``2*N*K*D`` matmul on the PE array with a
    streaming argmax epilogue; the Lloyd update is a second one-hot matmul
    of the same order (``~2*N*K*D``), so per-iter FLOPs are modelled at
    ``2x`` the assign. bf16 inputs / fp32 accumulate. Calibrated sustained
    throughput lives in :data:`flashlib.info.roofline._SUSTAINED_TFLOPS`
    under ``("kmeans", "bf16", "trn2")``; utilization is argmax-epilogue-
    bound at small D and matmul-bound (~47% of the single-core bf16 peak)
    at D=512.
    """
    B, N, D, K, niter = common(shape, params)
    dtype_bytes = 2
    assign_flops, bytes_iter = _assign_flops_bytes(B, N, D, K, dtype_bytes)
    # assign matmul + one-hot-matmul update (both ~2*N*K*D on the PE array).
    flops_iter = 2 * assign_flops
    flops = niter * flops_iter
    bytes_moved = niter * bytes_iter
    n_launches = niter          # one fused assign+update kernel per iter
    rt, bound = roofline(flops, bytes_moved, "bf16", device, op_type="kmeans",
                          n_launches=n_launches)
    return Estimate(
        op_name="kmeans_trainium",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=B * N * D * 2 / 1e9,
        bound=bound, confidence="measured", n_kernel_launches=n_launches,
        suggested_config={"MT": 128, "BK": 512},
        subops=[],
        notes=[
            f"B={B}, N={N}, D={D}, K={K}, niter={niter}",
            "Trainium NKI nc_matmul assign + one-hot-matmul update; "
            "B=1, euclidean, D<=512, bf16. Single NeuronCore.",
        ],
        expected_residual=None, precision_tier="exact", tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    """Recommend a backend / variant based on measured cross-overs."""
    B, N, D, K, niter = common(shape, params)
    backend = "triton"
    fa3_eligible = (B == 1 and 16 <= D <= 512 and D % 16 == 0)
    if fa3_eligible and K >= 1024 and D >= 256:
        backend = "cutedsl"
    return {
        "backend": backend,
        "variant": "split_d" if D > 256 else "default",
    }
