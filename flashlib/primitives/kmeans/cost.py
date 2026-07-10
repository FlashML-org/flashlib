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
from flashlib.info.roofline import LAUNCH_OVERHEAD_MS, roofline


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
        est = estimate_kmeans_nki(
            shape, params=params, tol=tol, dtype=dtype, device=device)
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


# Inclusive (N_min, N_max, D_min, D_max, K_min, K_max) calibration region for
# the general sorted path. The exact N=262144,D=256,K=4096 point is overridden
# by grouped reduction after passing its stronger sorted-vs-grouped gate.
_NKI_SORTED_AUTO_RANGES = (
    (262144, 1 << 20, 256, 512, 1, 4096),
)
_NKI_GROUPED_AUTO_SHAPE = (262144, 256, 4096)


def estimate_kmeans_nki(shape, params=None, tol=None, dtype="bf16",
                             device="trn2", **_):
    """AWS Trainium NKI backend scoped to one LNC2.

    D<=128 uses the measured transposed one-hot TensorEngine update. The fixed
    262144x256, K=4096 target uses fused low6/high6 group-reduce; wider D uses
    sorted or O(N*D) fp32 atomic update with XLA index_add as fallback.
    """
    options = params or {}
    B, N, D, K, niter = common(shape, options)
    full_chip = bool(options.get("full_chip", False))
    requested_update = options.get(
        "update_mode", options.get("use_nki_update"))
    grouped_auto_eligible = (
        B == 1 and (N, D, K) == _NKI_GROUPED_AUTO_SHAPE)
    grouped_kernel_supported = (
        B == 1
        and K == 4096
        and D <= 512
        and ((N + 255) // 256) * 256 < (1 << 24)
        and ((D + 255) // 256) * 128 <= 511
    )
    grouped_fallback = (
        requested_update == "grouped" and not grouped_kernel_supported)
    grouped_selected = grouped_kernel_supported and (
        requested_update == "grouped"
        or (
            requested_update not in (
                "atomic", "onehot", "sorted", "index_add", False)
            and grouped_auto_eligible
        )
    )
    sorted_auto_eligible = any(
        B == 1
        and n0 <= N <= n1 and d0 <= D <= d1 and k0 <= K <= k1
        for n0, n1, d0, d1, k0, k1 in _NKI_SORTED_AUTO_RANGES
    )
    # Padding adds sentinel K and can widen the radix outside the measured
    # <=64 region (notably padded K=4096).
    max_sort_key = K if N % 256 else K - 1
    sort_bits = max(1, int(max_sort_key).bit_length())
    sort_passes = (sort_bits + 6) // 7
    sort_radix_bits = (sort_bits + sort_passes - 1) // sort_passes
    sorted_auto_eligible &= (
        sort_passes <= 2 and (1 << sort_radix_bits) <= 64
    )
    sorted_kernel_supported = (
        B == 1
        and ((N + 255) // 256) * 256 < (1 << 24)
        and sort_passes <= 3
    )
    sorted_fallback = (
        requested_update == "sorted" and not sorted_kernel_supported)
    sorted_selected = not grouped_selected and sorted_kernel_supported and (
        requested_update in ("sorted", "grouped") or (
            requested_update not in (
                "atomic", "onehot", "grouped", "index_add", False)
            and sorted_auto_eligible
        )
    )
    if requested_update == "onehot" and D > 128:
        raise ValueError("Trainium onehot update requires D <= 128")
    assign_flops = 2 * B * N * K * D
    if K >= 32768:
        # Bounded-residency K=65536 kernel, k_resident=4096/point_lanes=4.
        # Actual-value LNC2 proxy rates; four synchronized scopes scale 3.9x+.
        large_rates = {128: 16.9, 256: 32.6, 512: 64.8}
        if D in large_rates:
            assign_tf = large_rates[D]
        elif D < 256:
            assign_tf = 16.9 + (D - 128) * (32.6 - 16.9) / 128
        else:
            assign_tf = 32.6 + (D - 256) * (64.8 - 32.6) / 256
    else:
        # Measured aggregate one-LNC2 rates at D=64/128/256/512.
        assign_tf = max(12.0, min(70.0, 0.137 * D))
    if full_chip:
        assign_tf *= 4.0
    assign_ms_iter = assign_flops / (assign_tf * 1e12) * 1e3

    grouped_extra = 0
    if grouped_selected:
        update_mode = (
            "grouped" if grouped_auto_eligible else "grouped_experimental")
        Np = ((N + 255) // 256) * 256
        max_blocks = (Np + 64 * 511) // 512
        bucket_rows = max_blocks * 512
        update_flops = (
            2 * B * max_blocks * 512 * 64 * D
            + 2 * B * 64 * max_blocks * 64 * D
        )
        scale_n = B * N / 262144.0
        update_ms_iter = (
            1.0 + 23.5 * scale_n
            + 1.0 + 5.2 * scale_n * D / 256.0
        )
        d_per = ((D + 255) // 256) * 128
        grouped_extra = 2 * (
            bucket_rows * 2 * 4
            + max_blocks * 64 * (d_per + 1) * 4
            + max_blocks * 3 * 4
        )
        grouped_bytes = (
            16 * B * N
            + 2 * B * N * D
            + 8 * B * max_blocks * 64 * D
        )
        bytes_iter = (
            (B * N * D + B * K * D) * 2
            + B * N * 4 + B * K * D * 4 + grouped_bytes
        )
        bound = "memory"
        launches_per_iter = 2
    elif sorted_selected:
        update_mode = (
            "sorted" if sorted_auto_eligible else "sorted_experimental")
        update_flops = B * N * D
        # Actual-value XLA calibration at N=131072,K=4096: radix ~25 ms and
        # the atomic-free lower-bound/segmented half ~6 ms for 128 feature
        # columns. LNC2 repeats the private sort but shards feature columns.
        radix_factor = 0.90 if K <= 1024 else (1.0 if K <= 4096 else 1.75)
        radix_ms = 0.000195 * B * N * radix_factor
        d_per_core = ((D + 255) // 256) * 128
        segmented_ms = 2.3 + 0.204e-6 * B * N * d_per_core
        # The fused kernel overlaps the final radix traffic with setup for the
        # lower-bound/scan stage (about 3 ms on the D=256 LNC2 target).
        update_ms_iter = radix_ms + 0.65 * segmented_ms
        sorted_bytes = (
            24 * B * N                 # pair ping-pong/histogram passes
            + 12 * B * N * D           # gather, prefix write/read
            + 8 * B * K * D
        )
        bytes_iter = (
            (B * N * D + B * K * D) * 2
            + B * N * 4 + B * K * D * 4 + sorted_bytes
        )
        bound = "memory"
        launches_per_iter = 2
    elif requested_update in ("index_add", False):
        update_mode = "index_add"
        update_flops = B * N * D
        # Warm compatibility-path calibration at N=131072,D=128,K=4096.
        update_ms_iter = 4.7e-5 * B * N * D
        bytes_iter = (
            (B * N * D + B * K * D) * 2
            + B * N * 12 + B * K * D * 8
        )
        bound = "memory"
        launches_per_iter = 2
    elif requested_update == "onehot" or (
        requested_update != "atomic" and D <= 128 and K < 32768
    ):
        update_mode = "onehot_transposed"
        update_flops = assign_flops
        update_tf = max(12.0, min(26.0, 0.203 * D))
        update_ms_iter = update_flops / (update_tf * 1e12) * 1e3
        bytes_iter = (
            (B * N * D + B * K * D) * 2
            + B * N * 4 + B * K * D * 4
        )
        bound = "compute"
        launches_per_iter = 2
    else:
        update_mode = "atomic"
        update_flops = B * N * D
        atomic_bytes = 12 * B * N * D + 8 * B * N
        if K >= 32768:
            # K=65536, N=262144 actual-value LNC2 Lloyd/uniform calibration.
            atomic_base = {128: 69.3, 256: 96.2, 512: 146.4}
            if D in atomic_base:
                base_ms = atomic_base[D]
            elif D < 256:
                base_ms = 69.3 + (D - 128) * (96.2 - 69.3) / 128
            else:
                base_ms = 96.2 + (D - 256) * (146.4 - 96.2) / 256
            if full_chip:
                # End-to-end residual after subtracting measured assign at
                # N=10M,K=65536 (chunked atomic + collective + finalize).
                full_update = {128: 1230.0, 256: 1581.0, 512: 2212.0}
                update_ms_iter = full_update.get(
                    D, 1581.0 * D / 256.0) * B * N / 10_000_000
            else:
                update_ms_iter = base_ms * B * N / 262144.0
        else:
            # fp32 source read + atomic RMW (~12 B/feature), calibrated at
            # 7.5 GB/s effective random-update bandwidth for K=4096.
            update_ms_iter = atomic_bytes / 7.5e9 * 1e3
        bytes_iter = (
            (B * N * D + B * K * D) * 2
            + B * N * 4 + B * K * D * 4 + atomic_bytes
        )
        bound = "memory"
        launches_per_iter = 2

    n_launches = launches_per_iter * niter
    # Each Lloyd iteration is an explicit XLA step boundary. The warm
    # N=32768,D=128,K=1024 loop measures ~1.5 ms/iter beyond kernel work.
    xla_step_ms = 1.5
    rt = niter * (assign_ms_iter + update_ms_iter + xla_step_ms)
    rt = max(rt, n_launches * LAUNCH_OVERHEAD_MS)
    flops = niter * (assign_flops + update_flops)
    bytes_moved = niter * bytes_iter
    sorted_extra = (
        2 * B * N * 128 * 4 + 2 * B * N * 2 * 4
        if sorted_selected else 0
    )
    return Estimate(
        op_name="kmeans_nki",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(
            B * N * D * 6 + B * K * D * 4 + B * N * 4
            + sorted_extra + grouped_extra
        ) / 1e9,
        bound=bound, confidence="measured", n_kernel_launches=n_launches,
        suggested_config={
            "MT": 128, "BK": 512, "lnc_width": 2,
            "lnc2_scopes": 4 if full_chip else 1,
            "large_k_assign": K >= 32768,
            "k_resident": 4096 if K >= 32768 else K,
            "point_lanes": 4 if K >= 32768 else 1,
            "assign_hfu_percent": (
                {128: 10.1, 256: 19.5, 512: 38.9}.get(D)
                if K >= 32768 and full_chip else None),
            "update": update_mode,
            "grouped_auto_eligible": grouped_auto_eligible,
            "sorted_auto_eligible": sorted_auto_eligible,
            "fold_csq": (D % 128) <= 126 and (D % 128) != 0,
        },
        subops=[],
        notes=[
            f"B={B}, N={N}, D={D}, K={K}, niter={niter}",
            f"Trainium NKI on {'four LNC2 scopes' if full_chip else 'one LNC2'}: "
            f"nc_matmul assign + {update_mode} "
            "update; B=1, euclidean, D<=512, bf16 rank inputs/fp32 sums.",
            (
                "Fused group-reduce is auto-selected at its calibrated "
                "6+6-bit target (>=1.25x vs sorted on uniform and Lloyd IDs)."
                if grouped_selected and grouped_auto_eligible
                else (
                    "Explicit grouped request is unsupported at this shape; "
                    "the estimate includes its sorted/atomic fallback."
                    if grouped_fallback
                    else
                    (
                        "Radix sorted update is auto-selected in this "
                        "calibrated shape range."
                        if sorted_selected and sorted_auto_eligible
                        else (
                            "Explicit sorted request is unsupported at this "
                            "shape; the estimate includes its fallback."
                            if sorted_fallback
                            else (
                                "Shape is optimized, but an explicit update "
                                "mode overrides automatic routing."
                                if grouped_auto_eligible or sorted_auto_eligible
                                else "Grouped/sorted updates remain explicit "
                                "outside measured production ranges."
                            )
                        )
                    )
                )
            ),
            (
                "K=65536 exact full-chip path is calibrated at "
                "N=10M: 3.71/4.16/4.80 s for D=128/256/512. "
                "Assign HFU is 10.1/19.5/38.9%; D128 is Vector "
                "MAX8/FIND_INDEX8 bound."
                if K >= 32768 and full_chip else
                "Large-K full-chip calibration is not selected."
            ),
        ],
        expected_residual=1e-3, precision_tier="fast", tol=tol,
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
