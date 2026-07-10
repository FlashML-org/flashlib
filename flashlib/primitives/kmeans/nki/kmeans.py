"""Trainium (NKI) flash-kmeans Lloyd loop.

``nki_kmeans_Euclid`` is the end-to-end driver behind
``flash_kmeans(x, backend="nki")``. It picks one of two paths:

torch-neuronx / XLA path (preferred, ``_nki_kmeans_xla``)
-------------------------------------------------------------
* **assign** -- the NKI flash kernel via ``nki.jit`` (streams centroids,
  never materialises the N x K matrix; PE-array bf16 FLOPs).
* **update** -- production routes use the transposed one-hot TensorEngine
  kernel, fused low6/high6 group-reduce at its calibrated target, stable radix
  segmented reduction in its wider large-N/D range, or fp32 ``atomic_rmw``;
  ``index_add`` remains the compatibility fallback.
  ``torch.sort`` itself is rejected by the Neuron compiler (``NCC_EVRF029``),
  so sorted mode uses a custom dense-integer radix implementation.
* the loop runs in torch-xla, so kernel/graph **compilation is cached**
  across Lloyd iterations (no per-iter recompile). ``tol == 0.0`` stays
  sync-free (no device->host scalar read mid-loop).

Requires torch-xla (on Amazon Linux 2023 apply
``tools/neuron/patch_torch_xla_glibc.py`` first).

Device-side one-hot-matmul update (default for D <= 128)
--------------------------------------------------------
Instead of the ``index_add`` scatter, the M-step runs as a matmul
(``onehot^T @ x`` for sums, ``onehot^T @ valid`` for counts) -- the shape the
PE array wants -- as a *second* cached ``nki.jit`` kernel (fusing it into the
assign graph is what actually hung the compiler; two separate cached kernels
compile fine). The key to making it win is the operand orientation: the naive
one-hot-as-stationary form tiles K by 128 and issues 4x the assign's matmul
tiles, so its Neuron compile is minutes-long and loses despite a faster steady
state. The **transposed** form (``update_body_transposed``, requires D <= 128)
makes ``x`` the stationary operand and the one-hot the 512-wide *moving*
operand, so K is tiled by 512, the tile count matches the assign, and it
compiles in ~seconds. Result: scatter-free and ~10x faster end-to-end than
``index_add`` (15 iters, N=131072, K=4096, D=128: ~1.1 s vs ~11.8 s, warm
compile cache). It returns sums/counts transposed; the finalize un-transposes.
For D > 128, the ``N=262144,D=256,K=4096,LNC2`` target uses a specialized
group-by: one low-six-bit bucket scatter, 512-row high-six-bit local matmuls,
and a TensorEngine merge of block partials. It emits no sorted permutation and
measures 30.7--33.9 ms across uniform/hotspot/zipf/Lloyd-like assignments,
versus 57.0 ms for the stable sorted update. The wider calibrated sorted path
removes feature atomics with two radix passes followed by indirect segmented
scans; other wide shapes use O(N*D) fp32 ``atomic_rmw``. ``index_add`` remains
the final compatibility fallback.

LNC2 sharding and full-chip workers
-----------------------------------
Each custom call is launched across the configured logical-NeuronCore width
(``[nl.nc(2)]`` for LNC2). The width is read from
``neuron-ls.logical_neuroncore_config`` and can be overridden with
``FLASHLIB_NKI_NSHARDS``. This is distinct from the four independent
LNC2 execution scopes exposed by the chip; full-chip throughput uses four
processes pinned with ``NEURON_RT_VISIBLE_CORES=0..3``.

The two stages shard on *different* axes, each embarrassingly parallel so
**no cross-core reduction** is needed:

* **assign** shards the point (M) axis -- each core argmins a disjoint block
  of points against the full (replicated) centroid set.
* **transposed update** shards the centroid (K) axis -- each core owns a
  disjoint block of 512-wide K-tiles, reducing ``onehot^T@x`` into its own
  PSUM and writing a disjoint output-column slice.
* **atomic update** shards feature columns -- each core atomically accumulates
  disjoint statistics columns, so there is no inter-core write conflict.
* **sorted update** independently builds the same private permutation on each
  core, then shards feature columns for disjoint segmented-sum stores.
* **grouped update** independently builds low-digit buckets, then shards
  feature columns through private block partials and disjoint final stores.

Shared padding keeps the two kernels' tensors aligned: points are padded to a
multiple of ``128 * n_shards`` and centroids to ``512 * n_shards``. Results are
bit-identical to the single-core path (the split is over independent output
rows/cols, not a floating-point reduction). Assign scales ~1.97x inside LNC2.
Four synchronized, device-resident workers also retain near-linear chip-level
throughput on the measured Lloyd shape. If an LNC2 launch fails, the driver
retries at width one.

For K=65536, centroids are resident in 4096-column chunks and points are
streamed through fixed 32768-row compiled calls. Four N-sharded LNC2 workers
process N=10M exactly in 3.71/4.16/4.80 seconds per Lloyd iteration for
D=128/256/512; assign alone is 2.48/2.57/2.59 seconds. Corresponding dense
assign HFU is 10.1/19.5/38.9%; D128 remains Vector max/find-index bound even
though Tensor and Vector overlap for essentially the entire kernel.

baremetal fallback (``_nki_kmeans_baremetal``)
---------------------------------------------------
Where torch-xla is unavailable: one fused ``nki.baremetal`` kernel
(assign + one-hot-matmul update) per iter with host finalize. Correct but
recompiles per call, so use modest ``max_iters``. (Single-core: the fused
kernel mixes the M-contracting assign and N-contracting update, so it is not
SPMD-sharded; the perf path is the torch-xla loop above.)

v1 scope: euclidean, ``B == 1``, ``D <= 512``, bf16 matmul inputs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from flashlib.primitives.kmeans.nki.assign import (
    _as_bf16, _ceil, _to_f32_np, _try_init_nki, _BK, _PMAX, _N_SHARDS,
)


def _build_step_kernel():
    """Build (once) the fused assign+update baremetal kernel."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    from flashlib.primitives.kmeans.nki.assign_kernel import flash_assign_body
    from flashlib.primitives.kmeans.nki.update_kernel import update_body

    def _lloyd_step(xT, x, valid, cT, csq):
        # xT [Dp, Np] bf16, x [Np, D] bf16, valid [Np, 1] bf16,
        # cT [Dp, Kp] bf16, csq [1, Kp] fp32.
        Np = xT.shape[1]
        D = x.shape[1]
        Kp = cT.shape[1]
        cluster_ids = nl.ndarray((Np, 1), dtype=nl.int32, buffer=nl.shared_hbm)
        flash_assign_body(xT, cT, csq, cluster_ids)
        sums = nl.ndarray((Kp, D), dtype=nl.float32, buffer=nl.shared_hbm)
        cnts = nl.ndarray((Kp, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        update_body(x, cluster_ids, valid, sums, cnts)
        return cluster_ids, sums, cnts

    return nki.baremetal()(_lloyd_step)


_STEP_KERNEL = None


def _step_kernel():
    global _STEP_KERNEL
    if _STEP_KERNEL is None:
        _STEP_KERNEL = _build_step_kernel()
    return _STEP_KERNEL


def _xla_available() -> bool:
    try:
        import torch_xla.core.xla_model  # noqa: F401
        return True
    except Exception:
        return False


_LNC_WIDTH: Optional[int] = None
_SORTED_AUTO_MIN_N = 262144
_SORTED_AUTO_MAX_N = 1 << 20
_SORTED_AUTO_MIN_D = 256
_SORTED_AUTO_MAX_RADIX = 64
_GROUPED_AUTO_SHAPE = (262144, 256, 4096)
_LARGE_K_N_CHUNK = 32768
_LARGE_K_UPDATE_N_CHUNK = 262144


def _grouped_auto_eligible(N: int, D: int, K: int,
                           n_shards: int) -> bool:
    """Measured >=1.25x winner over sorted on the fixed 6+6-bit target."""
    return n_shards == 2 and (N, D, K) == _GROUPED_AUTO_SHAPE


def _sorted_auto_eligible(N: int, D: int, K: int,
                          n_shards: int) -> bool:
    """Conservative range where sorted beat atomic by at least 1.25x.

    The gate also rejects a padded power-of-two K when its sentinel would
    double the radix width (for example K=4096 with a padded point axis).
    """
    if not (
        n_shards == 2
        and _SORTED_AUTO_MIN_N <= N <= _SORTED_AUTO_MAX_N
        and _SORTED_AUTO_MIN_D <= D <= _BK
        and 0 < K <= 4096
    ):
        return False
    has_padding = N % (_PMAX * n_shards) != 0
    max_key = K if has_padding else K - 1
    key_bits = max(1, int(max_key).bit_length())
    n_passes = (key_bits + 6) // 7
    radix_bits = (key_bits + n_passes - 1) // n_passes
    return n_passes <= 2 and (1 << radix_bits) <= _SORTED_AUTO_MAX_RADIX


def _probe_usable_cores(max_probe: int = 16) -> int:
    """Compatibility name for querying the configured physical-core LNC width.

    ``nl.nc(n)`` inside one XLA custom call spans the ``n`` physical cores of
    that call's LNC; it does *not* span the four logical LNC devices shown by
    ``neuron-ls`` on a full Trn2 chip. Read the explicit
    ``logical-neuroncore-config`` value instead of launching marker programs
    and misinterpreting their count as schedulable logical devices.
    """
    import re
    import subprocess

    try:
        proc = subprocess.run(
            ["neuron-ls"], capture_output=True, text=True, timeout=2,
            check=False)
        match = re.search(
            r"logical-neuroncore-config:\s*(\d+)", proc.stdout)
        if match:
            return max(1, min(max_probe, int(match.group(1))))
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return max(1, min(max_probe, _N_SHARDS))


def _neuron_n_shards(default: int = _N_SHARDS) -> int:
    """Physical-core width used by one LNC custom call (1 or 2 here)."""
    global _LNC_WIDTH
    import os

    override = (
        os.environ.get("FLASHLIB_NKI_LNC_WIDTH")
        or os.environ.get("FLASHLIB_NKI_NSHARDS")
    )
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    if _LNC_WIDTH is None:
        try:
            _LNC_WIDTH = _probe_usable_cores()
        except Exception:                                           # noqa: BLE001
            _LNC_WIDTH = default
    return max(1, _LNC_WIDTH)


def _launch(kernel, n_shards: int, *args):
    """Launch a ``nki.jit`` kernel single- or multi-core (``[nl.nc(n)]``)."""
    if n_shards > 1:
        import neuronxcc.nki.language as nl
        return kernel[nl.nc(n_shards)](*args)
    return kernel(*args)


_COMPILED_LARGE_CALLS: dict = {}


def _compiled_large_call(key, name: str, kernel, n_shards: int):
    """Compile one fixed-shape NKI chunk call and reuse it at every offset."""
    compiled = _COMPILED_LARGE_CALLS.get(key)
    if compiled is None:
        import torch_xla

        def step(*args):
            return _launch(kernel, n_shards, *args)

        compiled = torch_xla.compile(
            step, full_graph=True, name=name)
        _COMPILED_LARGE_CALLS[key] = compiled
    return compiled


_JIT_ASSIGN: dict = {}


def _jit_assign_kernel(n_shards: int = 1, *, D: Optional[int] = None,
                       K: Optional[int] = None, fold_csq: bool = False,
                       score_bf16: bool = False, large_k: bool = False,
                       k_resident: int = 4096, point_lanes: int = 4):
    """Cached shape/precision-specific ``nki.jit`` flash-assign kernel."""
    key = (
        D, K, fold_csq, score_bf16, n_shards,
        large_k, k_resident, point_lanes)
    k = _JIT_ASSIGN.get(key)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.assign_kernel import (
            _flash_assign_kernel, make_assign_kernel,
            make_assign_kernel_large,
        )
        if large_k:
            if fold_csq:
                raise ValueError(
                    "large-K assign does not support folded centroid norms")
            if score_bf16:
                raise ValueError(
                    "large-K assign requires FP32 score reduction")
            body = make_assign_kernel_large(
                n_shards, k_resident=k_resident,
                point_lanes=point_lanes)
        elif n_shards == 1 and not fold_csq and not score_bf16:
            body = _flash_assign_kernel
        else:
            body = make_assign_kernel(
                n_shards, fold_csq=fold_csq, score_bf16=score_bf16)
        k = nki.jit(body)
        _JIT_ASSIGN[key] = k
    return k


_JIT_UPDATE: dict = {}


def _jit_update_kernel(Kp: int):
    """Cached ``nki.jit`` one-hot-matmul update kernel (the M-step), keyed by
    the padded centroid count ``Kp``.

    This is the fast, device-side, scatter-free update. It is a *separate*
    kernel from the assign (rather than fused into one) on purpose: the update
    body compiles fine standalone under ``nki.jit`` (~2-8 s, then cached across
    Lloyd iterations), but *fusing* it with the assign into one graph hangs the
    Neuron compiler -- the assign contracts on D while the update contracts on
    N, and reconciling both matmul orientations (plus the intra-kernel HBM
    round-trip of the cluster ids) in a single custom-call blows up the
    layout-assignment pass. Two cached kernels are same-performance (a fused
    kernel would only save ~us of launch overhead against ~tens of ms of update
    compute) and strictly lower-risk. ``cid`` flows assign->update device-to-
    device with no host sync.

    ``Kp`` is baked into the kernel (via ``make_update_kernel``) rather than
    passed as a call argument, so the compiled kernel takes only tensors and is
    reused across iterations -- a plain ``int`` arg is not part of the jit
    shape-cache key and would trigger a full recompile every iteration.
    """
    k = _JIT_UPDATE.get(Kp)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.update_kernel import make_update_kernel
        k = nki.jit(make_update_kernel(Kp))
        _JIT_UPDATE[Kp] = k
    return k


_JIT_UPDATE_T: dict = {}


def _jit_update_kernel_transposed(Kp: int, n_shards: int = 1):
    """Cached ``nki.jit`` *transposed* one-hot-matmul update (D<=128), keyed by
    ``(Kp, n_shards)``. Returns ``(sumsT[D,Kp], cntsT[1,Kp])``.

    This is the update variant that actually wins: swapping the matmul operands
    (x stationary, one-hot the 512-wide moving operand) makes the K axis
    512-wide, so the update matches the assign's matmul-tile count and compiles
    in ~seconds (vs ~1.5 min for the 128-wide-K bodies) while keeping the fast
    PSUM-accumulate steady state -- see ``update_kernel.update_body_transposed``.

    ``n_shards`` > 1 shards the centroid (K) axis across NeuronCores (launch
    with ``[nl.nc(n_shards)]``); the shards own disjoint 512-wide K-tiles and
    write disjoint output columns, so no cross-core reduction is needed.
    """
    key = (Kp, n_shards)
    k = _JIT_UPDATE_T.get(key)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.update_kernel import make_update_kernel_transposed
        k = nki.jit(make_update_kernel_transposed(Kp, n_shards))
        _JIT_UPDATE_T[key] = k
    return k


_JIT_UPDATE_ATOMIC: dict = {}


def _jit_update_kernel_atomic(
    Kp: int, n_shards: int = 1, *, clear_stats: bool = True,
):
    """Cached O(N*D) atomic scatter update for Trainium2.

    The kernel shards feature columns across the two physical cores in an
    LNC2 launch, so each core owns disjoint HBM addresses. Counts are
    duplicated per core and column zero is consumed by the driver.
    """
    key = (Kp, n_shards, clear_stats)
    k = _JIT_UPDATE_ATOMIC.get(key)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.update_kernel import (
            make_update_kernel_atomic,
        )
        k = nki.jit(make_update_kernel_atomic(
            Kp, n_shards, clear_stats=clear_stats))
        _JIT_UPDATE_ATOMIC[key] = k
    return k


_JIT_UPDATE_SORTED: dict = {}


def _jit_update_kernel_sorted(N: int, K: int, Kp: int,
                              n_shards: int = 1):
    """Cached opt-in radix-sort + segmented-scan update.

    ``N`` and ``K`` are part of the key because they control the padding
    sentinel, radix width, and valid count mask.  The returned custom call
    consumes padded ``cid[Np]`` and fp32 ``x[Np,Dp]`` tensors.
    """
    key = (N, K, Kp, n_shards)
    k = _JIT_UPDATE_SORTED.get(key)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.update_kernel import (
            make_update_kernel_sorted,
        )
        k = nki.jit(make_update_kernel_sorted(N, K, Kp, n_shards))
        _JIT_UPDATE_SORTED[key] = k
    return k


_JIT_UPDATE_GROUPED: dict = {}


def _jit_update_kernel_grouped(N: int, K: int, n_shards: int = 1):
    """Cached low6 bucket + high6 partial-reduce update for K=4096."""
    key = (N, K, n_shards)
    k = _JIT_UPDATE_GROUPED.get(key)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.nki.grouped_update_kernel import (
            make_grouped_update_kernel,
        )
        k = nki.jit(make_grouped_update_kernel(N, K, n_shards))
        _JIT_UPDATE_GROUPED[key] = k
    return k


def _nki_kmeans_xla(x, n_clusters, max_iters, tol, init_centroids,
                         verbose, use_nki_update=True,
                         distributed: bool = False):
    """Lloyd on torch-neuronx/XLA: NKI flash assign (``nki.jit``) + a device
    update, graph compilation cached across iterations (no per-iter recompile).
    ``cid`` flows assign->update device-to-device (no host sync); ``tol == 0.0``
    stays sync-free.

    ``use_nki_update=True`` routes D<=128 to the measured transposed one-hot
    TensorEngine update, the fixed 6+6-bit target to grouped reduction, the
    wider calibrated range to sorted reduction, and other wider D to the
    O(N*D) fp32 atomic scatter. Explicit strings ``"onehot"``, ``"atomic"``,
    ``"grouped"``, ``"sorted"``, and ``"index_add"`` support A/B runs.
    Empty clusters keep their previous centroids, and mutable statistics are
    cleared inside the kernel on every invocation.
    """
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    return_on_device = x.device.type == "xla"
    B, N, D = xb.shape
    if B != 1:
        raise NotImplementedError("NKI KMeans supports B == 1 (v1)")
    if D > _BK:
        raise NotImplementedError("NKI KMeans supports D <= 512 (v1)")
    K = int(n_clusters)

    # A rank-local 2.5M x 512 FP32 tensor exceeds Neuron's 4 GiB
    # single-tensor limit. Large-K Lloyd stores points in BF16 and only casts
    # one update chunk to FP32 at a time.
    x2d = xb[0].to(
        dev, dtype=torch.bfloat16 if K >= 32768 else torch.float32)
    if init_centroids is None:
        idx = torch.randint(0, N, (K,), device=dev)
        cent0 = x2d[idx].float().clone()
    else:
        cent0 = init_centroids.reshape(K, D).to(dev, dtype=torch.float32).clone()
    if distributed:
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError(
                "distributed Trainium KMeans requires an initialized "
                "XLA process group")
        cent0 = xm.all_reduce(xm.REDUCE_SUM, cent0)
        cent0 = cent0 / dist.get_world_size()
    xm.mark_step()
    xm.wait_device_ops()

    # ``nl.nc(2)`` is LNC2: two physical NeuronCore-v3 cores sharing one
    # logical XLA device. It is not the four-LNC count reported by neuron-ls.
    # The transposed TensorEngine update is the measured winner for D<=128.
    # For wider features it cannot satisfy the stationary-free limit. The
    # fixed 6+6-bit target uses grouped reduction, the wider calibrated range
    # uses radix reduction, and other shapes use atomic.
    n_shards = _neuron_n_shards()
    if isinstance(use_nki_update, str):
        modes = [use_nki_update]
        if use_nki_update == "grouped":
            modes.extend(["sorted", "atomic", "index_add"])
        elif use_nki_update == "sorted":
            modes.append("onehot" if D <= _PMAX else "atomic")
            modes.append("index_add")
    elif use_nki_update:
        if K >= 32768:
            modes = ["atomic"]
        elif D <= _PMAX:
            modes = ["onehot", "atomic"]
        elif _grouped_auto_eligible(N, D, K, n_shards):
            modes = ["grouped", "sorted", "atomic"]
        elif _sorted_auto_eligible(N, D, K, n_shards):
            modes = ["sorted", "atomic"]
        else:
            modes = ["atomic"]
        modes.append("index_add")
    else:
        modes = ["index_add"]

    last_error = None
    done = False
    for update_mode in modes:
        attempts = (
            [n_shards]
            if distributed
            else ([n_shards, 1] if n_shards > 1 else [1])
        )
        for attempt in attempts:
            try:
                ids_t, cent, it = _run_xla_lloyd(
                    dev, x2d, cent0.clone(), N, D, K, attempt,
                    max_iters, tol, update_mode, verbose,
                    distributed=distributed)
                done = True
                break
            except Exception as e:                                  # noqa: BLE001
                last_error = e
                if verbose:
                    print(f"NKI: update={update_mode}, nl.nc({attempt}) "
                          f"failed ({e!r})")
        if done:
            break
    if not done:
        assert last_error is not None
        raise last_error

    cent_t = cent.to(dtype=x.dtype)[None, :, :]
    ids_t = ids_t[None, :]
    if not return_on_device:
        cent_t = cent_t.cpu()
        ids_t = ids_t.cpu()
    if squeeze:
        ids_t = ids_t.squeeze(0)
        cent_t = cent_t.squeeze(0)
    return ids_t, cent_t, it


def _run_xla_lloyd(dev, x2d, cent, N, D, K, n_shards, max_iters, tol,
                   update_mode, verbose, distributed: bool = False):
    """One full Lloyd run at a fixed NeuronCore shard count.

    Padding is shared between the two kernels so their tensors line up: the
    point axis is padded to a multiple of ``128 * n_shards`` (the assign shards
    M by ``n_shards``; the update reads the same padded ``cid``/``x``) and the
    centroid axis to a multiple of ``512 * n_shards`` (the transposed update
    shards K by ``n_shards``; the assign streams all K per core). ``n_shards``
    logical cores run each kernel with no cross-core reduction -- the M-split
    (assign) and K-split (update) are both embarrassingly parallel.

    Returns ``(cluster_ids[N] int32 (device), centroids[K,D] fp32 (device),
    n_iter)``.
    """
    import torch_xla.core.xla_model as xm

    large_k = K >= 32768
    k_resident = 4096
    point_lanes = 4 if large_k else 1
    Np = _ceil(
        N,
        _LARGE_K_UPDATE_N_CHUNK
        if large_k else _PMAX * n_shards * point_lanes)
    Dp = _ceil(D, _PMAX)
    Kp = _ceil(
        K, k_resident if large_k else _BK * n_shards)

    # Fold the centroid norm into two spare contraction rows only when both fit
    # in the existing 128-wide tile. A bf16 high+residual pair preserves the
    # fp32 norm closely; a single bf16 norm row caused excessive near-tie
    # assignment changes. Exact multiples and D%128==127 keep the explicit
    # fp32 epilogue rather than paying another matmul tile.
    fold_csq = (Dp - D) >= 2
    score_bf16 = False
    jit_assign = _jit_assign_kernel(
        n_shards, D=D, K=K, fold_csq=fold_csq,
        score_bf16=score_bf16, large_k=large_k,
        k_resident=k_resident, point_lanes=point_lanes)
    if isinstance(update_mode, bool):
        update_mode = "atomic" if update_mode else "index_add"
    if update_mode not in (
        "atomic", "onehot", "grouped", "sorted", "index_add"
    ):
        raise ValueError(
            "NKI KMeans update mode must be atomic, onehot, grouped, "
            "sorted, or index_add")
    if update_mode == "onehot" and D > _PMAX:
        raise ValueError("transposed one-hot update requires D <= 128")
    if update_mode == "sorted" and Np >= (1 << 24):
        raise ValueError("sorted update requires padded N < 2**24")
    if update_mode == "grouped" and K != 4096:
        raise ValueError("grouped update requires K == 4096")
    if update_mode == "grouped" and Np >= (1 << 24):
        raise ValueError("grouped update requires padded N < 2**24")
    jit_update_atomic = (
        _jit_update_kernel_atomic(Kp, n_shards)
        if update_mode == "atomic" else None
    )
    jit_update_atomic_accum = (
        _jit_update_kernel_atomic(
            Kp, n_shards, clear_stats=False)
        if update_mode == "atomic" and large_k else None
    )
    jit_update_t = (
        _jit_update_kernel_transposed(Kp, n_shards)
        if update_mode == "onehot" else None
    )
    jit_update_sorted = (
        _jit_update_kernel_sorted(N, K, Kp, n_shards)
        if update_mode == "sorted" else None
    )
    jit_update_grouped = (
        _jit_update_kernel_grouped(N, K, n_shards)
        if update_mode == "grouped" else None
    )
    large_assign_call = (
        _compiled_large_call(
            ("assign", D, Kp, n_shards, k_resident, point_lanes),
            f"kmeans_large_assign_d{D}_k{Kp}_nc{n_shards}",
            jit_assign, n_shards)
        if large_k else None
    )
    large_atomic_clear_call = (
        _compiled_large_call(
            ("atomic_clear", D, Kp, n_shards),
            f"kmeans_large_atomic_clear_d{D}_k{Kp}_nc{n_shards}",
            jit_update_atomic, n_shards)
        if large_k and jit_update_atomic is not None else None
    )
    large_atomic_accum_call = (
        _compiled_large_call(
            ("atomic_accum", D, Kp, n_shards),
            f"kmeans_large_atomic_accum_d{D}_k{Kp}_nc{n_shards}",
            jit_update_atomic_accum, n_shards)
        if large_k and jit_update_atomic_accum is not None else None
    )

    xbf = x2d.to(torch.bfloat16)
    import torch.nn.functional as F

    # xT (D-major, padded, bf16) is constant across iterations. Functional
    # pad/concat avoids an XLA dynamic-update-slice graph around every custom
    # call. In folded mode two appended constant rows multiply the bf16
    # high/residual representation of -0.5*c_sq.
    if large_k:
        x_eff = None
        xT = None
    else:
        x_eff = (
            torch.cat((xbf, torch.ones(
                N, 2, device=dev, dtype=torch.bfloat16)), dim=1)
            if fold_csq else xbf
        )
        xT = F.pad(
            x_eff.t(), (0, Np - N, 0, Dp - x_eff.shape[1]))
    ones_n = torch.ones(N, device=dev, dtype=torch.float32)  # index_add fallback

    # N-major points are constant across iterations. The atomic source appends
    # a validity/count column and pads its width for an even LNC2 feature split.
    if update_mode == "atomic":
        atomic_combined = (
            _ceil(D + 1, n_shards) // n_shards <= _BK)
        D_update = _ceil(D + 1, n_shards) if atomic_combined else D
        x_pad = (
            None if large_k else
            torch.zeros(Np, D_update, device=dev, dtype=torch.float32))
        if not large_k:
            x_pad[:N, :D] = x2d
        if atomic_combined:
            if not large_k:
                x_pad[:N, D] = 1.0
            count_pad = None
        else:
            # D=512 already fills the maximum free dimension. Accumulate
            # counts in a second tiny atomic pass, one private column/core.
            count_pad = torch.zeros(
                Np, n_shards, device=dev, dtype=torch.float32)
            count_pad[:N, :] = 1.0
        valid = None
    elif update_mode == "sorted":
        atomic_combined = False
        count_pad = None
        D_update = _ceil(D, _PMAX * n_shards)
        x_pad = torch.zeros(
            Np, D_update, device=dev, dtype=torch.float32)
        x_pad[:N, :D] = x2d
        valid = None
    elif update_mode == "grouped":
        atomic_combined = False
        count_pad = None
        D_update = _ceil(D, _PMAX * n_shards)
        if D_update // n_shards > 511:
            raise ValueError(
                "grouped update requires <=511 feature columns per core")
        x_pad = torch.zeros(
            Np, D_update, device=dev, dtype=torch.bfloat16)
        x_pad[:N, :D] = xbf
        valid = None
    else:
        atomic_combined = False
        count_pad = None
        D_update = D
        x_pad = torch.zeros(Np, D_update, device=dev, dtype=torch.bfloat16)
        x_pad[:N, :D] = xbf
        valid = torch.zeros(Np, 1, device=dev, dtype=torch.bfloat16)
        valid[:N] = 1.0

    if large_k:
        xT_chunks = []
        for n0 in range(0, Np, _LARGE_K_N_CHUNK):
            real = max(0, min(_LARGE_K_N_CHUNK, N - n0))
            x_part = xbf[n0:n0 + real, :]
            xT_chunks.append(F.pad(
                x_part.t(),
                (0, _LARGE_K_N_CHUNK - real, 0, Dp - D)))

        if update_mode != "atomic" or not atomic_combined:
            raise ValueError(
                "large-K streamed update currently requires combined "
                "LNC2 atomic statistics")
    else:
        assert xT is not None
        assert x_pad is not None
        xT_chunks = None
    if large_k:
        # Materialize each fixed-shape point shard once. Otherwise XLA sees a
        # different dynamic-slice graph at every chunk offset.
        xm.mark_step()
        xbf = None
        x_eff = None
        xT = None
        x_pad = None

    cid = torch.zeros(N, device=dev, dtype=torch.long)
    it = 0
    for it in range(max_iters):
        cent_bf = cent.to(torch.bfloat16)
        csq_valid = (cent * cent).sum(1)
        csq = F.pad(
            csq_valid, (0, Kp - K), value=3.0e38).unsqueeze(0)
        if fold_csq:
            cent_pad = F.pad(cent_bf, (0, 0, 0, Kp - K))
            bias_f = -0.5 * csq_valid
            bias_hi_valid = bias_f.to(torch.bfloat16)
            bias_lo_valid = (bias_f - bias_hi_valid.float()).to(
                torch.bfloat16)
            bias_hi = F.pad(
                bias_hi_valid,
                (0, Kp - K), value=-1.0e30)
            bias_lo = F.pad(
                bias_lo_valid, (0, Kp - K), value=0.0)
            cent_eff = torch.cat(
                (cent_pad, bias_hi.unsqueeze(1), bias_lo.unsqueeze(1)),
                dim=1)
            cT = F.pad(
                cent_eff.t(), (0, 0, 0, Dp - (D + 2)))
        else:
            cT = F.pad(cent_bf.t(), (0, Kp - K, 0, Dp - D))

        out_chunks = None
        id_update_chunks = None
        if large_k:
            assert xT_chunks is not None
            assert large_assign_call is not None
            out_chunks = []
            for xT_chunk in xT_chunks:
                out_chunks.append(
                    large_assign_call(xT_chunk, cT, csq))
            xm.wait_device_ops()
            out = torch.cat(out_chunks, dim=0)
            id_update_chunks = [
                out[n0:n0 + _LARGE_K_UPDATE_N_CHUNK, 0].contiguous()
                for n0 in range(0, Np, _LARGE_K_UPDATE_N_CHUNK)
            ]
            xm.mark_step()
        else:
            out = _launch(
                jit_assign, n_shards, xT, cT, csq)       # (Np, 1) int32

        # M-step. Grouped avoids a full permutation at its calibrated target;
        # sorted and atomic cover wider shapes, while index_add is the final
        # framework fallback.
        sums = counts = None
        stats_reduced = False
        if jit_update_atomic is not None:
            stats0 = torch.zeros(
                Kp, D_update, device=dev, dtype=torch.float32)
            if large_k:
                assert id_update_chunks is not None
                stats = stats0
                for chunk_idx, id_chunk in enumerate(id_update_chunks):
                    n0 = chunk_idx * _LARGE_K_UPDATE_N_CHUNK
                    real = max(
                        0, min(_LARGE_K_UPDATE_N_CHUNK, N - n0))
                    extra = torch.zeros(
                        real, D_update - D,
                        device=dev, dtype=torch.float32)
                    extra[:, 0] = 1.0
                    source = torch.cat(
                        (x2d[n0:n0 + real, :].float(), extra), dim=1)
                    x_chunk = F.pad(
                        source,
                        (0, 0, 0, _LARGE_K_UPDATE_N_CHUNK - real))
                    call = (
                        large_atomic_clear_call
                        if chunk_idx == 0
                        else large_atomic_accum_call)
                    assert call is not None
                    stats = call(stats, id_chunk, x_chunk)
                    xm.wait_device_ops()
            else:
                stats = _launch(
                    jit_update_atomic, n_shards,
                    stats0, out[:, 0], x_pad)
            if distributed and atomic_combined:
                # Coalesce feature sums and counts into one CCOM transfer.
                stats = xm.all_reduce(xm.REDUCE_SUM, stats)
                stats_reduced = True
            sums = stats[:K, :D]
            if atomic_combined:
                counts = stats[:K, D]
            else:
                assert count_pad is not None
                count_stats = _launch(
                    jit_update_atomic, n_shards,
                    torch.zeros(
                        Kp, n_shards, device=dev, dtype=torch.float32),
                    out[:, 0], count_pad)
                counts = count_stats[:K, 0]
        elif jit_update_sorted is not None:
            sorted_sums, sorted_counts = _launch(
                jit_update_sorted, n_shards, out[:, 0], x_pad)
            sums = sorted_sums[:K, :D]
            counts = sorted_counts[:K, 0]
        elif jit_update_grouped is not None:
            grouped_sums, grouped_counts = _launch(
                jit_update_grouped, n_shards, out[:, 0], x_pad)
            sums = grouped_sums[:, :D]
            counts = grouped_counts[:, 0]
        elif jit_update_t is not None:
            assert valid is not None
            sumsT, cntsT = _launch(jit_update_t, n_shards, x_pad, out, valid)
            sums = sumsT[:, :K].t()                            # [D,Kp]->[K,D]
            counts = cntsT[0, :K]
        cid = out[:N, 0].to(torch.long)
        if sums is None:
            # Legacy reduce-by-key via index_add (scatter) -- O(N*D).
            sums = torch.zeros(K, D, device=dev, dtype=torch.float32).index_add_(0, cid, x2d)
            counts = torch.zeros(K, device=dev, dtype=torch.float32).index_add_(0, cid, ones_n)
        if distributed and not stats_reduced:
            # Each rank owns a disjoint N shard. Reduce only compact KxD
            # statistics; assignment IDs remain rank-local throughout Lloyd.
            sums = xm.all_reduce(xm.REDUCE_SUM, sums)
            counts = xm.all_reduce(xm.REDUCE_SUM, counts)
        new_cent = sums / counts.clamp(min=1.0).unsqueeze(1)
        empty = (counts == 0.0).unsqueeze(1)
        new_cent = torch.where(empty, cent, new_cent)
        need_shift = tol > 0.0 or verbose
        shift = (new_cent - cent).norm(dim=1).max() if need_shift else None
        cent = new_cent
        xm.mark_step()                                          # execute this iter
        if verbose:
            assert shift is not None
            print(f"Iter {it} (nki/xla, lnc={n_shards}, "
                  f"update={update_mode}), "
                  f"shift={float(shift):.6f}")
        if tol > 0.0 and shift is not None and float(shift) < tol:
            break

    return cid.to(torch.int32), cent, it + 1


def nki_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    use_nki_update=True,
    **_,
):
    """Lloyd k-means on Trainium (squared-L2). See module docstring.

    Uses the torch-xla path (NKI flash assign + a device update, cached
    compile) when torch-xla is importable, else the baremetal fallback. Force
    the driver with ``backend="xla"`` or ``backend="baremetal"``.
    ``use_nki_update=True`` selects the measured one-hot/grouped/sorted/atomic
    route for the shape. Pass ``"onehot"``, ``"atomic"``, ``"grouped"``,
    ``"sorted"``, or ``"index_add"`` for explicit A/B runs.
    """
    if not _try_init_nki():
        raise RuntimeError("NKI backend unavailable: NKI did not import")
    use_xla = backend == "xla" or (backend is None and _xla_available())
    if use_xla:
        return _nki_kmeans_xla(
            x, n_clusters, max_iters, tol, init_centroids, verbose,
            use_nki_update=use_nki_update)
    return _nki_kmeans_baremetal(
        x, n_clusters, max_iters, tol, init_centroids, verbose)


def nki_kmeans_Euclid_distributed(
    local_x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    use_nki_update=True,
):
    """Exact N-sharded Lloyd worker for a torchrun XLA process group.

    Every rank owns disjoint points and returns rank-local IDs. Centroid
    statistics are all-reduced on device each iteration, so all ranks return
    identical centroids without gathering the N-axis.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError(
            "call under torchrun after dist.init_process_group('xla')")
    return _nki_kmeans_xla(
        local_x, n_clusters, max_iters, tol,
        init_centroids, verbose,
        use_nki_update=use_nki_update,
        distributed=True)


def _nki_kmeans_baremetal(
    x, n_clusters, max_iters, tol, init_centroids, verbose,
):
    """Baremetal fallback: fused assign+update kernel per iter, host finalize."""
    from flashlib.primitives.kmeans.nki.update_kernel import finalize as _finalize

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    B, N, D = xb.shape
    if B != 1:
        raise NotImplementedError("NKI KMeans supports B == 1 (v1)")
    if D > _BK:
        raise NotImplementedError("NKI KMeans supports D <= 512 (v1)")
    K = int(n_clusters)

    x_f32 = _to_f32_np(xb[0])                       # (N, D)
    if init_centroids is None:
        idx = torch.randint(0, N, (K,), device="cpu").numpy()
        cent = x_f32[idx].copy()                    # (K, D) fp32
    else:
        cent = _to_f32_np(init_centroids.reshape(K, D)).copy()

    Np, Dp, Kp = _ceil(N, _PMAX), _ceil(D, _PMAX), _ceil(K, _BK)

    # Constant per-iteration inputs (built once).
    xT = np.zeros((Dp, Np), np.float32)
    xT[:D, :N] = x_f32.T
    xT = _as_bf16(xT)
    x_pad = np.zeros((Np, D), np.float32)
    x_pad[:N] = x_f32
    x_pad = _as_bf16(x_pad)
    valid = np.zeros((Np, 1), np.float32)
    valid[:N] = 1.0
    valid = _as_bf16(valid)

    kernel = _step_kernel()
    cluster_ids = np.zeros(N, np.int32)
    it = 0
    for it in range(max_iters):
        cT = np.zeros((Dp, Kp), np.float32)
        cT[:D, :K] = cent.T
        csq = np.full((1, Kp), 3.0e38, np.float32)
        csq[0, :K] = (cent ** 2).sum(1)
        cids, sums, cnts = kernel(xT, x_pad, valid, _as_bf16(cT), csq)
        cluster_ids = np.asarray(cids[:N, 0], np.int32)
        cent_new, shift = _finalize(sums, cnts, cent)
        if verbose:
            print(f"Iter {it} (nki), shift={float(shift.max()):.6f}")
        cent = cent_new
        if tol > 0.0 and float(shift.max()) < tol:
            break

    ids_t = torch.from_numpy(cluster_ids).to(torch.int32).to(x.device)[None, :]
    cent_t = torch.from_numpy(cent).to(x.dtype).to(x.device)[None, :, :]
    if squeeze:
        ids_t = ids_t.squeeze(0)
        cent_t = cent_t.squeeze(0)
    return ids_t, cent_t, it + 1
