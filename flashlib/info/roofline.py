"""Roofline model + measured-throughput overrides. Pure stdlib (no torch).

Two stages of estimation, in order:

1. **Calibrated lookup** -- if a measured ``(op_class, dtype, device)``
   sustained TFLOPS / GB/s has been recorded in :data:`_SUSTAINED_TFLOPS`
   / :data:`_SUSTAINED_BW_TBS`, use that. Source: the benchmark suite
   under ``benchmarks/results/``; the recorded numbers are end-to-end
   wall-clock medians on warmed kernels with cuda-synced timing.

2. **Roofline fallback** -- if no calibration is registered for the
   ``(op_class, dtype, device)`` triple, fall back to
   ``peak_compute(dtype, device) * default_efficiency(op_class)``
   and ``peak_bw(device) * default_efficiency(op_class)``.

The roofline ``runtime_ms`` is then ``max(t_compute, t_memory)``; the
``bound`` is whichever side dominates.

The module is import-cheap (no GPU, no torch). Callers that want to
*also* probe the actual hardware should call :func:`detect_device`,
which performs a deferred ``import torch``.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------
# Hardware peaks (per GPU). Add new SKUs by appending to the dict.
# Numbers are vendor-spec dense peaks for matmul tensor cores; the bandwidth
# is HBM B/s (advertised, not achievable).
# ----------------------------------------------------------------------------
_HARDWARE: dict[str, dict[str, float]] = {
    "H100": {
        "fp64_tflops": 33.5,    # FP64 tensor core
        "fp32_tflops": 51.0,    # FP32 simt
        "tf32_tflops": 417.0,   # TF32 tensor core (sparse off)
        "fp16_tflops": 1979.0,  # FP16 tensor core (sparse off)
        "bf16_tflops": 1979.0,  # BF16 tensor core (sparse off)
        "int8_tflops": 3958.0,  # INT8 tensor core (sparse off)
        "bw_tbs":       3.10,
    },
    "H200": {
        "fp64_tflops": 33.5,
        "fp32_tflops": 51.0,
        "tf32_tflops": 417.0,
        "fp16_tflops": 1979.0,
        "bf16_tflops": 1979.0,
        "int8_tflops": 3958.0,
        "bw_tbs":       4.80,
    },
    "A100": {
        "fp64_tflops": 19.5,
        "fp32_tflops": 19.5,
        "tf32_tflops": 156.0,
        "fp16_tflops": 312.0,
        "bf16_tflops": 312.0,
        "int8_tflops": 624.0,
        "bw_tbs":       2.04,
    },
}


# Per-op-class default efficiency (achieved fraction of peak) when no
# calibration is registered. These are intentionally conservative and
# only used as a *fallback*; calibrated entries in
# ``_SUSTAINED_TFLOPS`` / ``_SUSTAINED_BW_TBS`` override these.
#
# Rationale for the numbers:
#   gemm-class kernels: WGMMA-driven tl.dot kernels at large M/N/K
#     consistently hit ~40% of vendor peak for fp16/bf16 and ~50% for
#     tf32 (FP32 inputs go through the TF32 path when allow_tf32=True).
#   elementwise / reduction: bandwidth-bound; mature Triton kernels
#     sustain ~80% of HBM peak on H100/H200.
#   solver: cuSOLVER and Triton-Jacobi-class kernels are
#     launch/serialization-bound at the sizes we care about; treating
#     them at 25% of FP32 peak roughly matches what we see on H200.
#   reduction (cov_gemm / scaler / etc.): asymmetric tensor-core matmul
#     -> mostly bw-bound, ~70% of HBM peak.
_DEFAULT_EFFICIENCY: dict[str, float] = {
    "gemm":         0.40,
    "elementwise":  0.80,
    "reduction":    0.70,
    "solver":       0.25,
    "knn":          0.45,  # WGMMA + on-chip top-K; reduced by epilogue overhead
    "topk":         0.45,
    "spmv":         0.70,  # sparse: HBM-bound, near-elementwise
    "rng":          0.60,  # streaming PRNG kernels
    # IVF-Flat fine-scan: online path streams cell-contiguous candidates
    # from HBM with an on-chip top-K epilogue -> bandwidth-bound; batch path
    # groups queries per list into a tensor-core GEMM (read reuse) ->
    # WGMMA-bound. Build reuses the kmeans op-class; "ivf_flat" is the
    # dispatcher-level default.
    "ivf_flat":         0.55,
    "ivf_flat_search":  0.60,  # coalesced list reads, top-K epilogue derate
    "ivf_flat_build":   0.45,  # dominated by the kmeans assign pass
    # IVF-PQ ADC fine-scan: streams compressed uint8 codes and gathers an
    # m-entry distance from the per-(query,list) LUT per candidate -> strongly
    # bandwidth-/gather-bound (the LUT gathers, not the codes, dominate). Build
    # adds PQ codebook training + encode on top of the kmeans assign pass.
    "ivf_pq":           0.50,
    "ivf_pq_search":    0.55,  # LUT-gather + top-K epilogue derate
    "ivf_pq_build":     0.45,
}


# ----------------------------------------------------------------------------
# Calibrated sustained throughput per (op_class, dtype, device).
#
# Sourced from end-to-end measured wall-clock medians under
# benchmarks/results/{boundaries_gemm,boundaries_kmeans,vs_cuml_full}.md
# at the H100/H200 reference shapes. When a primitive runs at a known
# sustained TFLOPS we record it here; the roofline then computes
# t_compute = flops / (TFLOPS_calibrated * 1e12) rather than
# t_compute = flops / (peak_compute * default_efficiency * 1e12).
#
# The key is (op_class, canonical_dtype, device). Canonical dtype names
# are 'fp32', 'tf32', 'bf16', 'fp16', 'fp64', 'int8'. ``canonicalize_dtype``
# normalises user input. If the device key isn't present, look-up falls
# through to the default-efficiency formula.
# ----------------------------------------------------------------------------
# Sustained throughput is reported as an **average across a primitive's
# typical operating range** -- not a peak. We deliberately set values
# conservatively (closer to the geometric mean of measured numbers
# across build / search regimes) so users don't get hard-edged
# over-confident predictions. Shape-specific regimes (KNN search vs
# build, KMeans tiny-N vs large-N) layer additional ``eff_factor``
# adjustments in their cost.py.
_SUSTAINED_TFLOPS: dict[tuple[str, str, str], float] = {
    # ─── GEMM (measured at N=K=M=8192, see benchmarks/results/boundaries_gemm.md) ───
    ("gemm", "fp32",  "H200"):  50.0,    # cuBLAS fp32 (no TC)
    ("gemm", "tf32",  "H200"): 386.0,    # cuBLAS TF32 / Triton fused
    ("gemm", "bf16",  "H200"): 816.0,    # cuBLAS bf16; Triton 3xbf16 ≈ 228
    ("gemm", "fp16",  "H200"): 892.0,    # cuBLAS fp16
    ("gemm", "fp64",  "H200"):  60.0,    # cuBLAS fp64
    # ─── KMeans assign (Triton kernel) ───────────────────────────────────
    # Geometric mean across the K=64 / K=256 regimes from
    # full_speedup_report.md: 200 TF fp32 effective at large-K shapes,
    # bw-bound at smaller K (the bw-side calibration in
    # ``_SUSTAINED_BW_TBS`` handles those).
    ("kmeans", "bf16", "H200"): 400.0,
    ("kmeans", "fp32", "H200"): 200.0,
    # ─── KNN insert / sortmerge kernels (x²-free score) ───────────────────
    # Two distinct regimes -- the KNN cost.py picks the appropriate
    # op_class based on (B*N) saturating the SMs:
    #   "knn_build"  -- B*N >= 50_000 (BN=128 build regime): hits ~600 TF
    #                   bf16 effective; search-tier shapes fall back here.
    #   "knn_search" -- small-Q + large-M regime: ~80 TF bf16 effective due
    #                   to under-saturated WGMMA on the (N) axis.
    ("knn_build",  "bf16", "H200"): 600.0,
    ("knn_build",  "fp32", "H200"): 260.0,
    ("knn_search", "bf16", "H200"):  80.0,
    ("knn_search", "fp32", "H200"):  40.0,
    # ─── IVF-Flat fused fine-scan ─────────────────────────────────────────
    # The online path is bandwidth-bound (the ``_SUSTAINED_BW_TBS`` entry
    # below dominates the estimate); these compute numbers track the
    # analogous knn_search regime and matter only for the batch GEMM path.
    ("ivf_flat_search", "bf16", "H200"): 90.0,
    ("ivf_flat_search", "fp32", "H200"): 45.0,
}


# Calibrated sustained HBM bandwidth per (op_class, device). Same source.
# For bandwidth-bound ops we record the achieved GB/s the kernel sustains.
#
# Lower than vendor peak for two structural reasons:
#   1. Mature single-launch streaming kernels (StandardScaler / cov_gemm)
#      sustain ~75% of HBM peak; that's reflected in 'elementwise'.
#   2. Multi-iteration kernels (KMeans Lloyd, k-NN insert at large NB)
#      re-read X from HBM each iteration -- L2 spillover puts effective
#      bw closer to 25-35 % of peak, reflected in 'kmeans' / 'knn'.
_SUSTAINED_BW_TBS: dict[tuple[str, str], float] = {
    # single-launch streaming kernels (StandardScaler, cov_gemm-style)
    ("elementwise", "H200"): 3.6,    # ~75% of 4.8
    ("elementwise", "H100"): 2.4,    # ~78% of 3.10
    ("reduction",   "H200"): 3.4,
    ("reduction",   "H100"): 2.3,
    # Multi-iteration assign kernels -- X re-read each iter; sustained
    # bw measured at ~1.1 TB/s on H200 for KMeans (500K, 64, K=64, 25 iter)
    # from benchmarks/results/full_speedup_report.md.
    ("kmeans",      "H200"): 1.1,
    ("kmeans",      "H100"): 0.8,
    # KNN insert at large NB shows the same L2-spill pattern.
    ("knn_build",   "H200"): 1.4,
    ("knn_build",   "H100"): 1.0,
    ("knn_search",  "H200"): 2.8,   # search is more bw-limited (low D, low N)
    ("knn_search",  "H100"): 1.9,
    # IVF-Flat fine-scan has two regimes. ONLINE (tiny nq, elementwise
    # kernel): bandwidth-bound coalesced list reads + per-(query,probe)
    # top-K epilogue, ~0.6-1.0 TB/s on H200. BATCH (large nq, group-by-list
    # tensor-core GEMM): each list's vectors are read once from HBM and
    # reused across all queries probing it, so *effective* candidate-read
    # bandwidth measured via benchmarks/vs_cuml/ivf_flat.py is 16-31 TB/s --
    # far above HBM peak (~4.8 TB/s) because it is no longer HBM-bound but
    # WGMMA-bound. We keep the conservative online BW here (lower bound for
    # batch); H100 scaled by the HBM-peak ratio (3.1/4.8).
    ("ivf_flat_search", "H200"): 1.0,
    ("ivf_flat_search", "H100"): 0.7,
    # IVF-PQ ADC fine-scan: the steady-state cost is the random LUT gathers
    # (4 B each) plus the coalesced uint8 code stream. Effective sustained
    # bandwidth is gather-limited, conservatively ~0.9 TB/s on H200 (online
    # elementwise path); H100 scaled by the HBM-peak ratio (3.1/4.8). Backfill
    # from benchmarks/vs_cuml/ivf_pq.py once measured.
    ("ivf_pq_search", "H200"): 0.9,
    ("ivf_pq_search", "H100"): 0.6,
}


# ----------------------------------------------------------------------------
# Public surface
# ----------------------------------------------------------------------------

_DTYPE_ALIASES: dict[str, str] = {
    # floats
    "float32": "fp32",  "float":   "fp32",  "f32":   "fp32",  "single": "fp32",
    "float64": "fp64",  "double":  "fp64",  "f64":   "fp64",
    "float16": "fp16",  "half":    "fp16",  "f16":   "fp16",
    "bfloat16": "bf16", "bf16":    "bf16",  "bf":    "bf16",
    "tf32":    "tf32",
    "int8":    "int8",  "uint8":   "int8",
    # already canonical
    "fp32": "fp32", "fp64": "fp64", "fp16": "fp16",
}


def canonicalize_dtype(dtype: str | None) -> str:
    """Normalise a user-provided dtype string to flashlib's canonical form.

    Examples: ``"float32"`` / ``"f32"`` -> ``"fp32"``,
    ``"bfloat16"`` -> ``"bf16"``. Unknown values fall through to ``"fp32"``.
    """
    if dtype is None:
        return "fp32"
    key = str(dtype).lower().strip()
    return _DTYPE_ALIASES.get(key, key if key in _HARDWARE.get("H100", {}) else "fp32")


def detect_device(default: str = "H100") -> str:
    """Probe the actual CUDA device via a deferred torch import.

    Returns a key matching :data:`_HARDWARE` (``"H100"``, ``"H200"``,
    ``"A100"``) by sniffing ``torch.cuda.get_device_name(0)``. Falls
    back to ``default`` when torch/CUDA is unavailable. Used by
    :func:`flashlib.info.dispatch.estimate` when the caller doesn't
    pin a device.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return default
        name = torch.cuda.get_device_name(0).upper()
    except Exception:
        return default
    if "H200" in name:
        return "H200"
    if "H100" in name:
        return "H100"
    if "A100" in name:
        return "A100"
    return default


def get_compute_peak(dtype: str, op_type: str, device: str) -> float:
    """Return vendor-peak TFLOPS for the given canonical dtype on device.

    ``op_type='solver'`` always returns FP32 peak (cuSOLVER and the
    flashlib Jacobi kernel are FP32-only by design).
    """
    hw = _HARDWARE.get(device, _HARDWARE["H100"])
    if op_type == "solver":
        return hw["fp32_tflops"]
    cdtype = canonicalize_dtype(dtype)
    return hw.get(f"{cdtype}_tflops", hw["fp32_tflops"])


def get_bandwidth_peak(device: str) -> float:
    """Return advertised HBM TB/s for the device."""
    hw = _HARDWARE.get(device, _HARDWARE["H100"])
    return hw["bw_tbs"]


def get_sustained_throughput(op_class: str, dtype: str,
                              device: str) -> float | None:
    """Calibrated sustained TFLOPS, or ``None`` if no calibration is recorded.

    Lookup key is ``(op_class, canonical_dtype, device)``.
    """
    return _SUSTAINED_TFLOPS.get((op_class, canonicalize_dtype(dtype), device))


def get_sustained_bandwidth(op_class: str, device: str) -> float | None:
    """Calibrated sustained HBM TB/s, or ``None`` if no calibration."""
    return _SUSTAINED_BW_TBS.get((op_class, device))


def default_efficiency(op_class: str) -> float:
    """Achieved fraction of peak for an op-class when no calibration exists."""
    return _DEFAULT_EFFICIENCY.get(op_class, 0.50)


# Per-kernel-launch fixed overhead on PyTorch + Triton, observed wall-clock.
# Includes the host-side dispatch + cuda launch + the small per-iter
# autograd / dtype-check work that happens around every kernel call. End-to-end
# warm-kernel "tick" measured at ~40-60 us on PyTorch 2.11 + Triton 3.x;
# we use 50 us as the round number. Cold first-call autotune compile is
# *not* modelled here (it's a separate one-time hit); the floor reflects
# what an agent should expect once the kernel is warmed.
LAUNCH_OVERHEAD_MS: float = 0.050


def roofline(flops: float, bytes_moved: float, dtype: str, device: str,
             op_type: str = "gemm",
             efficiency: float | None = None,
             n_launches: int = 1) -> tuple[float, str]:
    """Estimate runtime (ms) and bottleneck via roofline.

    Order of precedence:

    1. If ``efficiency`` is provided -> use it for both compute and bw.
    2. Else look up calibrated sustained TFLOPS / TB/s for the
       ``(op_type, dtype, device)`` triple and use those directly
       (without further derating).
    3. Else fall back to ``peak * default_efficiency(op_type)``.

    A launch-latency floor of
    ``n_launches * LAUNCH_OVERHEAD_MS`` is added so small-shape
    primitives (where each kernel does ~tens of µs of work but the
    primitive dispatches dozens of launches) don't predict
    sub-launch-overhead times. The returned ``bound`` switches to
    ``'latency'`` when this floor dominates.

    Returns ``(runtime_ms, bound)`` where ``bound`` is one of
    ``'compute' | 'memory' | 'latency'``.
    """
    # ----- compute side -----
    sustained_tf = (
        None if efficiency is not None
        else get_sustained_throughput(op_type, dtype, device)
    )
    if sustained_tf is not None:
        eff_compute_tf = sustained_tf
    else:
        eff = efficiency if efficiency is not None else default_efficiency(op_type)
        eff_compute_tf = max(get_compute_peak(dtype, op_type, device) * eff, 1e-6)
    t_compute_ms = (flops / 1e12 / eff_compute_tf) * 1000.0

    # ----- memory side -----
    sustained_bw = (
        None if efficiency is not None
        else get_sustained_bandwidth(op_type, device)
    )
    if sustained_bw is not None:
        eff_bw_tbs = sustained_bw
    else:
        eff = efficiency if efficiency is not None else default_efficiency(op_type)
        eff_bw_tbs = max(get_bandwidth_peak(device) * eff, 1e-6)
    t_memory_ms = (bytes_moved / 1e12 / eff_bw_tbs) * 1000.0

    # ----- latency floor -----
    t_launch_ms = max(n_launches, 1) * LAUNCH_OVERHEAD_MS

    runtime_ms = max(t_compute_ms, t_memory_ms, t_launch_ms)
    if runtime_ms == t_launch_ms and t_launch_ms > max(t_compute_ms, t_memory_ms):
        bound = "latency"
    elif t_compute_ms > t_memory_ms:
        bound = "compute"
    else:
        bound = "memory"
    return runtime_ms, bound


def list_devices() -> list[str]:
    """Return registered device SKUs."""
    return sorted(_HARDWARE.keys())


def list_op_classes() -> list[str]:
    """Return op-classes with registered default efficiency."""
    return sorted(_DEFAULT_EFFICIENCY.keys())


def is_calibrated(op_class: str, dtype: str, device: str) -> bool:
    """True iff at least one of sustained-TFLOPS or sustained-bw is recorded."""
    return (
        get_sustained_throughput(op_class, dtype, device) is not None
        or get_sustained_bandwidth(op_class, device) is not None
    )
