"""Cost models for flash_knn -- dispatcher + per-backend.

Shape: ``(B, N, M, D)`` or ``(N, M, D)``. ``estimate(...)`` mirrors
:func:`flashlib.primitives.knn.flash_knn_dispatch` so agents see what
would actually run at this shape.

Op names emitted
----------------

* ``knn_triton``      -- Triton fused dispatcher. The shape-only
                         heuristic in
                         :mod:`flashlib.primitives.knn.triton.dispatch`
                         picks ``BN / BM / M_PER_SPLIT / NUM_STAGES_PIPE``
                         and the kernel mode (insert vs the small-Q
                         Pattern-A sortmerge corner). One unified
                         x²-free insert kernel covers everything from
                         ``Q=1`` search to ``100K × 100K`` build, plus
                         a sortmerge variant for the small-Q +
                         medium-K Pattern-A regime.
* ``knn_cutedsl_fa3`` -- Hopper FA3 fully-fused (opt-in; never
                         auto-routed -- requires explicit
                         ``backend="cutedsl"``).
* ``knn_torch``       -- pure-torch reference (full ``(B, N, M)``
                         materialised).

Performance anchoring
---------------------

The Triton path is calibrated against measured runs on H200, e.g.
``(1, 8192, 65536, 256) k=16`` runs at ~2.64 ms. The model splits
the budget into:

* the x²-free dot kernel (compute-bound, calibrated against
  :data:`flashlib.info.roofline._SUSTAINED_TFLOPS[("knn", dtype, dev)]`)
* the external gather pass (bandwidth-bound, ``BN × K × D`` HBM bytes).
"""
from flashlib.info.estimate import Estimate
from flashlib.info.roofline import roofline
from flashlib.primitives.knn.impl import route_op_name as _route_op_name


def _shape(shape):
    if len(shape) == 4:
        return shape
    if len(shape) == 3:
        N, M, D = shape
        return 1, N, M, D
    raise ValueError("flash_knn shape must be (B, N, M, D) or (N, M, D)")


def common(shape, params):
    B, N, M, D = _shape(shape)
    params = params or {}
    k = params.get("k", 10)
    return B, N, M, D, k


def estimate(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    B, N, M, D, k = common(shape, params)
    if str(device).lower().startswith("trn"):
        chosen = (
            "knn_nki"
            if B == 1 and M <= 16384 and D <= 512 and k <= 64
            else "knn_torch"
        )
    else:
        chosen = _route_op_name(B=B, N=N, M=M, D=D, k=k)
    fn = {
        "knn_triton":      estimate_knn_triton,
        "knn_cutedsl_fa3": estimate_knn_cutedsl_fa3,
        "knn_nki":         estimate_knn_nki,
        "knn_torch":       estimate_knn_torch,
    }[chosen]
    est = fn(shape, params=params, tol=tol, dtype=dtype, device=device)
    est.op_name = chosen
    est.tol = tol
    return est


def _residual(dtype, tol):
    """Map ``(dtype, tol)`` to an expected residual + precision tier.

    The KNN kernel can take a bf16 / fp16 storage cast when the caller
    opts in via ``tol >= 1e-3`` (DBSCAN + UMAP do this); otherwise the
    fp32-input path is exact in the score and only the gather
    introduces a tiny rounding error.
    """
    if dtype in ("bf16", "bfloat16"):
        return 1e-3, "fast"
    if dtype in ("fp16", "float16"):
        return 8e-4, "fast"
    return 1e-7, "exact"


def _heuristic_BN(B: int, N: int, M: int, K: int) -> int:
    """Mirror the BN bucket from
    :func:`flashlib.primitives.knn.triton.dispatch._heuristic_config`.

    Build regime (large N*B) uses ``BN=128`` so a single CTA owns the
    whole query batch; search regime uses BN scaled by N for WGMMA
    utilisation.
    """
    NB = N * B
    if NB >= 50_000:
        return 128            # build: BN=128 covers all query rows
    if NB >= 30_000:
        return 64
    if NB <= 8:
        return 8              # Pattern-A small-Q
    if NB <= 32:
        return 16
    if NB <= 128:
        return 32
    return 64


def _knn_compute_bytes(B, N, M, D, K, dtype_bytes):
    """Algorithmic FLOP + byte counts for the x²-free insert kernel.

    FLOPs: ``2 * B * N * M * D`` (one dot product per (n, m) pair, x²-free
    so no extra reduction) plus a small ``4 * B * N * M`` for the
    per-tile top-K insert (uniform branch + scalar compare).

    Bytes (algorithmic lower bound for a fully fused kernel):
        X read once     -> ``B * N * D * dtype_bytes``
        C read once     -> ``B * M * D * dtype_bytes``
        idx/dist out    -> ``B * N * K * (4 + 4)`` (int32 idx + fp32 dist)
    Real HBM traffic exceeds this when X / C don't fit L2 -- the
    `eff_factor` on top of the roofline absorbs that.
    """
    distance_flops = 2 * B * N * M * D
    topk_flops     = 4 * B * N * M
    flops          = distance_flops + topk_flops
    bytes_moved    = (B * N * D + B * M * D) * dtype_bytes + B * N * K * 8
    return flops, bytes_moved


def _regime(B: int, N: int) -> str:
    """Return ``"knn_build"`` for B*N >= 50K, else ``"knn_search"``.

    Mirrors the gate inside
    :func:`flashlib.primitives.knn.triton.dispatch._heuristic_config`:
    the build path uses ``BN=128`` and saturates WGMMA on the N axis;
    the search path under-saturates and runs at ~5-7× lower effective
    TFLOPS.
    """
    return "knn_build" if B * N >= 50_000 else "knn_search"


def estimate_knn_triton(shape, params=None, tol=None, dtype="float32",
                        device="H100", **_):
    """Triton fused KNN -- on-chip top-K, no HBM cross.

    Single iterative-insert kernel (plus a sortmerge variant on the
    small-Q + medium-K Pattern-A corner), x²-free score, indices-only
    output. The runtime dispatcher picks the routing inside
    :func:`_heuristic_config`:

    * ``N*B >= 50_000`` -> ``BN=128`` build regime; ``ctas_no_split >=
      NUM_SMS * 8`` triggers the single-pass-per-CTA path.
    * Otherwise -> M-split flash-decode with BN/BM/wave-count scaled
      by ``(N, M, K)`` and the ``NB``-bucket table; Pattern-A fast
      paths cover ``B*N <= 8`` small-Q corners.

    Both paths are 2 launches (kernel + external gather); the gather
    is cheap (BN × K × D bandwidth-bound) and writes the user-visible
    squared L2 distances by re-computing ``(x − c[idx])²`` directly.

    The roofline picks ``op_class='knn_build'`` (B*N >= 50K) or
    ``'knn_search'`` (small-Q regime); see :func:`_regime`. The two
    regimes are calibrated separately in
    :data:`flashlib.info.roofline._SUSTAINED_TFLOPS` (~600 TF bf16
    build vs ~80 TF bf16 search on H200).
    """
    B, N, M, D, k = common(shape, params)
    dtype_bytes = 2 if dtype in ("fp16", "float16", "bf16", "bfloat16") else 4
    flops, bytes_moved = _knn_compute_bytes(B, N, M, D, k, dtype_bytes)
    op_class = _regime(B, N)

    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type=op_class, n_launches=1)

    # External gather adds a small bw-bound pass: BN * K * D fp32 reads
    # from C + BN * K fp32 writes. Negligible relative to the insert
    # kernel except at very small N where it can be ~10% of the budget.
    BN = _heuristic_BN(B, N, M, k)
    gather_bytes = B * N * k * D * dtype_bytes + B * N * k * 4
    gather_rt, _ = roofline(2 * B * N * k * D, gather_bytes, dtype,
                            device, op_type="elementwise", n_launches=1)
    rt = rt + gather_rt

    res, tier = _residual(dtype, tol)
    return Estimate(
        op_name="knn_triton",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(B * N * D + B * M * D) * dtype_bytes / 1e9,
        bound=bound, confidence="calibrated", n_kernel_launches=2,
        suggested_config={
            "BN": BN,
            "kernel": "sortmerge" if (B * N <= 8 and k <= 16) else "insert",
            "regime": op_class,
        },
        subops=[],
        notes=[
            f"B={B}, N={N}, M={M}, D={D}, k={k}, dtype={dtype}, regime={op_class}",
            f"x²-free Triton insert kernel; BN={BN}; "
            f"gather adds ~{gather_rt:.2f} ms.",
        ],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def estimate_knn_cutedsl_fa3(shape, params=None, tol=None, dtype="float32",
                             device="H100", **_):
    """FA3-style fully-fused TMA + WGMMA + register top-K.

    NOT auto-selected -- requires explicit ``backend="cutedsl"``. The
    heuristic mode pays ~5-8 s of CuteDSL compile per shape (single
    config from :func:`_heuristic_fa3_config`); the autotune mode
    sweeps the full grid (multi-minute first call) and caches the
    winner. Pairs with the same external gather pass for distances.

    Empirical: same compute budget as the Triton path on the shapes
    where both are viable, so the cost model uses the same calibrated
    roofline and adds the ~5 s first-call compile note. The compile
    cost is amortised over subsequent same-shape calls.
    """
    B, N, M, D, k = common(shape, params)
    dtype_bytes = 2 if dtype in ("fp16", "float16", "bf16", "bfloat16") else 4
    flops, bytes_moved = _knn_compute_bytes(B, N, M, D, k, dtype_bytes)
    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type=_regime(B, N), n_launches=2)
    res, tier = _residual(dtype, tol)
    return Estimate(
        op_name="knn_cutedsl_fa3",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(B * N * D + B * M * D) * dtype_bytes / 1e9,
        bound=bound, confidence="calibrated", n_kernel_launches=2,
        suggested_config={"strategy": "perthread" if k <= 16 else "sortmerge"},
        subops=[],
        notes=[
            f"B={B}, N={N}, M={M}, D={D}, k={k}, dtype={dtype}",
            "FA3-style fused TMA+WGMMA; opt-in via backend='cutedsl' "
            "(first call per shape compiles, ~5-8 s).",
        ],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def estimate_knn_nki(shape, params=None, tol=None, dtype="bf16",
                          device="trn2", **_):
    """AWS Trainium (NKI) flash-knn: ``nc_matmul`` cross term + on-chip top-k
    via ``max8`` / ``nc_match_replace8``, plus an fp32 gather for true
    distances. Never materialises the ``(N, M)`` matrix.

    The production route is the measured full-score variant for ``M<=16384``.
    Ranking and true-distance recovery are fused into one device-resident NKI
    call on one LNC2; the latter indirectly gathers ``N*k*D`` fp32 values. No
    ``N*M`` HBM matrix or NumPy round-trip is allocated.
    """
    B, N, M, D, k = common(shape, params)
    base_pad = ((k + 7) // 8) * 8
    score_bf16 = 32 <= base_pad <= 56
    rank_kpad = ((k + 15) // 8) * 8 if score_bf16 else base_pad
    candidate_k = rank_kpad if score_bf16 else k
    rank_flops = 2 * B * N * M * D
    distance_flops = 3 * B * N * candidate_k * D
    flops = rank_flops + distance_flops
    # fp32/TF32 rank operands + int32 candidates; fp32 indirect gather +
    # distances.
    bytes_moved = (
        (B * N * D + B * M * D) * 4
        + B * N * candidate_k * 4
        + B * N * candidate_k * D * 4
        + B * N * candidate_k * 4
    )
    op_class = "knn_build" if B * N >= 1024 else "knn_search"
    # Warm one-LNC2 calibration: rank throughput rises with query occupancy
    # (~2.0 TF at Q=1024, 4.7 TF at Q=4096, approaching 8-13 TF at Q>=8192).
    rank_tf = min(13.0, max(2.0, 1.1 + 0.00088 * B * N))
    rank_ms = rank_flops / (rank_tf * 1e12) * 1e3
    # The fused epilogue is indirect-gather limited; the calibrated fixed term
    # includes XLA dispatch/output synchronization at the reference shape.
    gather_rate = 6.5e9 if score_bf16 else 5.5e9
    distance_ms = distance_flops / 3.0 / gather_rate * 1e3 + 0.85
    # fp32 max8/match-replace rescans the full score row once per 8 results.
    rounds = rank_kpad // 8
    scan_ms_per_round = 0.19 if score_bf16 else 0.70
    score_scan_ms = (
        max(rounds - 2, 0) * scan_ms_per_round
        * (B * N / 4096.0) * (M / 8192.0)
    )
    rt = max(rank_ms + distance_ms + score_scan_ms, 0.15)
    bound = (
        "memory" if distance_ms + score_scan_ms > rank_ms else "compute"
    )
    # Exhaustive scan with FP32 distance recovery, but TF32/BF16 rank scores
    # can change the candidate set for exact near ties.
    res, tier = 1e-4, "fast"
    return Estimate(
        op_name="knn_nki",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(
            (B * N * D + B * M * D) * 4
            + B * N * candidate_k * 8
        ) / 1e9,
        bound=bound, confidence="calibrated", n_kernel_launches=1,
        suggested_config={
            "regime": op_class,
            "variant": "full_score",
            "score_dtype": (
                "bf16" if 32 <= ((k + 7) // 8) * 8 <= 56 else "fp32"
            ),
            "candidate_k": (
                ((k + 15) // 8) * 8
                if 32 <= ((k + 7) // 8) * 8 <= 56
                else k
            ),
            "lnc_width": 2,
        },
        subops=[],
        notes=[
            f"B={B}, N={N}, M={M}, D={D}, k={k}, dtype={dtype}",
            "Trainium NKI on one LNC2: TF32 rank + fp32 indirect "
            "gather-distance; "
            "M<=16384, B==1, k<=64 (else routes to torch).",
        ],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def estimate_knn_torch(shape, params=None, tol=None, dtype="float32",
                       device="H100", **_):
    """Pure-torch reference -- materialises the full ``(B, N, M)`` matrix."""
    B, N, M, D, k = common(shape, params)
    dtype_bytes = 2 if dtype in ("fp16", "float16", "bf16", "bfloat16") else 4
    flops, bytes_moved = _knn_compute_bytes(B, N, M, D, k, dtype_bytes)
    # Reference path actually pays an extra (B, N, M) materialisation pass;
    # double the bytes side and use the elementwise calibration (no on-chip
    # top-K, so no compute-side speedup vs roofline).
    bytes_moved = bytes_moved + B * N * M * 4
    rt, bound = roofline(flops, bytes_moved, dtype, device,
                          op_type="elementwise", n_launches=1)
    res, tier = _residual(dtype, tol)
    return Estimate(
        op_name="knn_torch",
        runtime_ms=rt, flops=flops, bytes_moved=bytes_moved,
        memory_peak_gb=(B * N * D + B * M * D + B * N * M) * 4 / 1e9,
        bound=bound, confidence="roofline", n_kernel_launches=1,
        suggested_config={}, subops=[],
        notes=[
            f"B={B}, N={N}, M={M}, D={D}, k={k}, dtype={dtype}",
            "Torch reference (full (B, N, M) materialised); "
            "correctness baseline only.",
        ],
        expected_residual=res, precision_tier=tier, tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="float32", device="H100", **_):
    B, N, M, D, k = common(shape, params)
    if str(device).lower().startswith("trn"):
        chosen = (
            "knn_nki"
            if B == 1 and M <= 16384 and D <= 512 and k <= 64
            else "knn_torch"
        )
    else:
        chosen = _route_op_name(B=B, N=N, M=M, D=D, k=k)
    if chosen == "knn_nki":
        return {
            "variant": "full_score",
            "score_dtype": (
                "bf16" if 32 <= ((k + 7) // 8) * 8 <= 56 else "fp32"
            ),
            "candidate_k": (
                ((k + 15) // 8) * 8
                if 32 <= ((k + 7) // 8) * 8 <= 56
                else k
            ),
            "rank_input": "tf32",
            "lnc_width": 2,
            "regime": "build" if B * N >= 1024 else "search",
        }
    return {
        "variant": chosen,
        "BN": _heuristic_BN(B, N, M, k),
        "regime": "build" if N * B >= 50_000 else "search",
    }
