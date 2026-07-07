"""Trainium (NKI) flash-kmeans Lloyd loop.

``trainium_kmeans_Euclid`` is the end-to-end driver behind
``flash_kmeans(x, backend="trainium")``. It picks one of two paths:

torch-neuronx / XLA path (preferred, ``_trainium_kmeans_xla``)
-------------------------------------------------------------
* **assign** -- the NKI flash kernel via ``nki.jit`` (streams centroids,
  never materialises the N x K matrix; PE-array bf16 FLOPs).
* **update** -- ``torch.zeros(K, D).index_add_(0, cid, x)`` for the sums
  and ``index_add_`` of ones for the counts: an O(N*D) reduce-by-key. This
  is a *scatter* (Trainium has no hardware sort -- ``torch.sort`` is rejected
  by the Neuron compiler, ``NCC_EVRF029`` -- so the Triton sort+segment-sum
  strategy is impossible), but its XLA graph is small and compiles in a few
  seconds. ``use_nki_update=True`` swaps in a scatter-free one-hot-matmul NKI
  kernel instead (see below).
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
For D > 128 (``x`` would exceed the 128 stationary-free limit) the loop falls
back to ``index_add``.

baremetal fallback (``_trainium_kmeans_baremetal``)
---------------------------------------------------
Where torch-xla is unavailable: one fused ``nki.baremetal`` kernel
(assign + one-hot-matmul update) per iter with host finalize. Correct but
recompiles per call, so use modest ``max_iters``.

v1 scope: euclidean, ``B == 1``, ``D <= 512``, bf16 matmul inputs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from flashlib.primitives.kmeans.trainium.assign import (
    _as_bf16, _ceil, _to_f32_np, _try_init_nki, _BK, _PMAX,
)


def _build_step_kernel():
    """Build (once) the fused assign+update baremetal kernel."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    from flashlib.primitives.kmeans.trainium.assign_kernel import flash_assign_body
    from flashlib.primitives.kmeans.trainium.update_kernel import update_body

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


_JIT_ASSIGN = None


def _jit_assign_kernel():
    global _JIT_ASSIGN
    if _JIT_ASSIGN is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.trainium.assign_kernel import _flash_assign_kernel
        _JIT_ASSIGN = nki.jit(_flash_assign_kernel)
    return _JIT_ASSIGN


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
        from flashlib.primitives.kmeans.trainium.update_kernel import make_update_kernel
        k = nki.jit(make_update_kernel(Kp))
        _JIT_UPDATE[Kp] = k
    return k


_JIT_UPDATE_T: dict = {}


def _jit_update_kernel_transposed(Kp: int):
    """Cached ``nki.jit`` *transposed* one-hot-matmul update (D<=128), keyed by
    ``Kp``. Returns ``(sumsT[D,Kp], cntsT[1,Kp])``.

    This is the update variant that actually wins: swapping the matmul operands
    (x stationary, one-hot the 512-wide moving operand) makes the K axis
    512-wide, so the update matches the assign's matmul-tile count and compiles
    in ~seconds (vs ~1.5 min for the 128-wide-K bodies) while keeping the fast
    PSUM-accumulate steady state -- see ``update_kernel.update_body_transposed``.
    """
    k = _JIT_UPDATE_T.get(Kp)
    if k is None:
        import neuronxcc.nki as nki
        from flashlib.primitives.kmeans.trainium.update_kernel import make_update_kernel_transposed
        k = nki.jit(make_update_kernel_transposed(Kp))
        _JIT_UPDATE_T[Kp] = k
    return k


def _trainium_kmeans_xla(x, n_clusters, max_iters, tol, init_centroids,
                         verbose, use_nki_update: bool = True):
    """Lloyd on torch-neuronx/XLA: NKI flash assign (``nki.jit``) + a device
    update, graph compilation cached across iterations (no per-iter recompile).
    ``cid`` flows assign->update device-to-device (no host sync); ``tol == 0.0``
    stays sync-free.

    Two update paths, selected by ``use_nki_update`` (default ``True``):

    * ``True`` **and D <= 128** (the default winner): a second cached
      ``nki.jit`` **transposed** one-hot-matmul kernel -- ``x`` is the
      stationary operand and the one-hot is the 512-wide *moving* operand, so
      the K axis is tiled by 512 and the update matches the assign's matmul-
      tile count (compiles in ~seconds, not minutes) while keeping the fast
      PSUM-accumulate steady state. It is scatter-free (``onehot^T @ x``) and
      measured ~10x faster end-to-end than ``index_add`` (e.g. 15 iters at
      N=131072, K=4096, D=128: ~1.1 s vs ~11.8 s with a warm Neuron compile
      cache). Returns sums/counts transposed; the finalize un-transposes.
    * ``False`` (or D > 128, where the transpose can't run since ``x`` would
      exceed the 128 stationary-free limit): ``torch...index_add_`` reduce-by-
      key. A scatter Trainium does weakly, but a small fast graph. NOTE: the
      GPU sort+segment-sum update is impossible here -- the Neuron compiler
      rejects sort (``NCC_EVRF029``).

    History: the NKI update was believed to "hang" under ``nki.jit``. It was
    really (a) *fusing* assign+update into one graph blowing up layout
    assignment (fixed: two separate cached kernels), (b) passing ``Kp`` as an
    int arg, not part of the jit shape-cache key, forcing a recompile every
    iteration (fixed: bake ``Kp`` via the ``make_update_kernel*`` factories),
    and (c) the 128-wide-K one-hot orientation issuing 4x the assign's matmul
    tiles -> minutes-long compile (fixed: the transposed 512-wide-K operands
    above). If the NKI update fails to build, this falls back to ``index_add``.
    """
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    jit_assign = _jit_assign_kernel()

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    B, N, D = xb.shape
    if B != 1:
        raise NotImplementedError("trainium kmeans supports B == 1 (v1)")
    if D > _BK:
        raise NotImplementedError("trainium kmeans supports D <= 512 (v1)")
    K = int(n_clusters)
    Np, Dp, Kp = _ceil(N, _PMAX), _ceil(D, _PMAX), _ceil(K, _BK)
    # The NKI update that wins end-to-end is the transposed one (D <= 128): it
    # compiles in ~seconds like the assign. For D > 128 no NKI update beats the
    # index_add scatter (their compile dwarfs the steady-state gain), so use it.
    use_nki_t = use_nki_update and D <= _PMAX
    jit_update = _jit_update_kernel_transposed(Kp) if use_nki_t else None

    x2d = xb[0].to(dev, dtype=torch.float32)                     # (N, D) fp32
    if init_centroids is None:
        idx = torch.randint(0, N, (K,), device="cpu")
        cent = x2d[idx.to(dev)].clone()                         # (K, D) fp32
    else:
        cent = init_centroids.reshape(K, D).to(dev, dtype=torch.float32).clone()

    # xT (D-major, padded, bf16) is constant across iterations -> build once.
    xbf = x2d.to(torch.bfloat16)
    xT = torch.zeros(Dp, Np, device=dev, dtype=torch.bfloat16)
    xT[:D, :N] = xbf.t()
    ones_n = torch.ones(N, device=dev, dtype=torch.float32)  # index_add fallback

    # For the NKI one-hot-matmul update: N-major padded points + a valid mask
    # (0 on pad rows so padding contributes 0 to both sums and counts). Built
    # once (constant across iterations), like xT.
    x_pad = torch.zeros(Np, D, device=dev, dtype=torch.bfloat16)
    x_pad[:N] = xbf
    valid = torch.zeros(Np, 1, device=dev, dtype=torch.bfloat16)
    valid[:N] = 1.0

    cid = torch.zeros(N, device=dev, dtype=torch.long)
    it = 0
    for it in range(max_iters):
        cT = torch.zeros(Dp, Kp, device=dev, dtype=torch.bfloat16)
        cT[:D, :K] = cent.to(torch.bfloat16).t()
        csq = torch.full((1, Kp), 3.0e38, device=dev, dtype=torch.float32)
        csq[0, :K] = (cent * cent).sum(1)

        out = jit_assign(xT, cT, csq)                          # (Np, 1) int32

        # M-step: transposed one-hot-matmul update (device-side, scatter-free).
        # The padded cluster ids ``out`` feed straight in; the valid mask zeroes
        # padded rows. Returns sums/counts transposed (sumsT[D,Kp], cntsT[1,Kp]);
        # the .t()/index un-transpose folds into the finalize. Falls back to the
        # index_add scatter if the update kernel cannot build.
        sums = counts = None
        if jit_update is not None:
            try:
                sumsT, cntsT = jit_update(x_pad, out, valid)   # [D,Kp],[1,Kp]
                sums = sumsT[:, :K].t()
                counts = cntsT[0, :K]
            except Exception as e:                                  # noqa: BLE001
                if verbose:
                    print(f"trainium: NKI update unavailable ({e!r}); "
                          f"falling back to index_add")
                jit_update = None
        cid = out[:N, 0].to(torch.long)
        if sums is None:
            # Legacy reduce-by-key via index_add (scatter) -- O(N*D).
            sums = torch.zeros(K, D, device=dev, dtype=torch.float32).index_add_(0, cid, x2d)
            counts = torch.zeros(K, device=dev, dtype=torch.float32).index_add_(0, cid, ones_n)
        new_cent = sums / counts.clamp(min=1.0).unsqueeze(1)
        empty = (counts == 0.0).unsqueeze(1)
        new_cent = torch.where(empty, cent, new_cent)
        shift = (new_cent - cent).norm(dim=1).max()
        cent = new_cent
        xm.mark_step()                                          # execute this iter
        if verbose:
            print(f"Iter {it} (trainium/xla), shift={float(shift):.6f}")
        if tol > 0.0 and float(shift) < tol:
            break

    ids_t = cid.to(torch.int32).cpu()[None, :]
    cent_t = cent.to(torch.float32).cpu().to(x.dtype)[None, :, :]
    if squeeze:
        ids_t = ids_t.squeeze(0)
        cent_t = cent_t.squeeze(0)
    return ids_t, cent_t, it + 1


def trainium_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: Optional[torch.Tensor] = None,
    verbose: bool = False,
    backend: Optional[str] = None,
    use_nki_update: bool = True,
    **_,
):
    """Lloyd k-means on Trainium (squared-L2). See module docstring.

    Uses the torch-xla path (NKI flash assign + a device update, cached
    compile) when torch-xla is importable, else the baremetal fallback. Force
    the driver with ``backend="xla"`` or ``backend="baremetal"``.
    ``use_nki_update=True`` (default) uses the scatter-free transposed one-hot-
    matmul NKI update for D <= 128 (~10x faster end-to-end than the index_add
    scatter); set ``False`` (or run with D > 128) to use ``index_add``. See
    ``_trainium_kmeans_xla``.
    """
    if not _try_init_nki():
        raise RuntimeError("trainium backend unavailable: NKI did not import")
    use_xla = backend == "xla" or (backend is None and _xla_available())
    if use_xla:
        return _trainium_kmeans_xla(
            x, n_clusters, max_iters, tol, init_centroids, verbose,
            use_nki_update=use_nki_update)
    return _trainium_kmeans_baremetal(
        x, n_clusters, max_iters, tol, init_centroids, verbose)


def _trainium_kmeans_baremetal(
    x, n_clusters, max_iters, tol, init_centroids, verbose,
):
    """Baremetal fallback: fused assign+update kernel per iter, host finalize."""
    from flashlib.primitives.kmeans.trainium.update_kernel import finalize as _finalize

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    B, N, D = xb.shape
    if B != 1:
        raise NotImplementedError("trainium kmeans supports B == 1 (v1)")
    if D > _BK:
        raise NotImplementedError("trainium kmeans supports D <= 512 (v1)")
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
            print(f"Iter {it} (trainium), shift={float(shift.max()):.6f}")
        cent = cent_new
        if tol > 0.0 and float(shift.max()) < tol:
            break

    ids_t = torch.from_numpy(cluster_ids).to(torch.int32).to(x.device)[None, :]
    cent_t = torch.from_numpy(cent).to(x.dtype).to(x.device)[None, :, :]
    if squeeze:
        ids_t = ids_t.squeeze(0)
        cent_t = cent_t.squeeze(0)
    return ids_t, cent_t, it + 1
