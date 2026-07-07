"""Trainium (NKI) flash-knn -- Python driver / wrapper.

Public API:
    trainium_knn_available() -> bool
    trainium_knn(x, c, k, *, return_distances=True) -> (vals, idxs) | idxs

The NKI kernel (:mod:`knn_kernel`) returns nearest-corpus **indices** only
(ranked by the x2-free sim, which equals true-distance order). This module
handles torch<->numpy marshalling, padding/transpose to the PE-array tile
layout, and the true-distance epilogue.

True distances are recovered here (not in the kernel) by gathering ``c[idx]``
and computing ``(x - c)^2`` in fp32 -- the kernel ranks on a bf16 matmul, and
recovering distance as ``||x||^2 - sim`` would catastrophically cancel for the
near neighbours that matter. After the accurate recompute we re-sort each
query's k by ``(dist, idx)`` to honour the API's ascending-distance, index-
tie-break contract (and to repair any bf16 near-tie flips in the kernel order).

v1 scope: squared-L2, ``B == 1``, corpus ``M <= 16384`` (the ``max8`` element/
partition limit -- single-shot top-k, no cross-tile merge), bf16 matmul inputs.
Larger corpora fall back to another backend in the dispatcher.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Reuse the kmeans trainium driver helpers (NKI import guard, ceil, casts).
from flashlib.primitives.kmeans.trainium.assign import (
    _as_bf16, _ceil, _to_f32_np, _try_init_nki,
)

_PMAX = 128
_BK = 512
_MAX_M = 16384        # max8 elements-per-partition limit (single-shot v1)
_N_SHARDS = 2         # NeuronCores to shard the query axis over (trn2: 2 LNC)


def trainium_knn_available() -> bool:
    return _try_init_nki()


_MC_KERNEL_CACHE: dict = {}


def _mc_kernel(kpad: int, n_shards: int):
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.trainium.knn_kernel import make_knn_kernel
    key = (kpad, n_shards)
    k = _MC_KERNEL_CACHE.get(key)
    if k is None:
        k = nki.baremetal()(make_knn_kernel(kpad, n_shards))
        _MC_KERNEL_CACHE[key] = k
    return k


def _knn_np(q_f32: np.ndarray, c_f32: np.ndarray, k: int,
            n_shards: int = _N_SHARDS) -> np.ndarray:
    """Run the baremetal knn kernel; return int64 indices [Nq, k] (nearest
    first by the x2-free sim). ``q_f32`` [Nq, D], ``c_f32`` [M, D]. Shards the
    query axis over ``n_shards`` NeuronCores (~2x on trn2), single-core fallback."""
    import neuronxcc.nki.language as nl
    from flashlib.primitives.knn.trainium.knn_kernel import knn_kernel_baremetal

    Nq, D = q_f32.shape
    M = c_f32.shape[0]
    Dp, Mp, kpad = _ceil(D, _PMAX), _ceil(M, _BK), _ceil(k, 8)
    cT = np.zeros((Dp, Mp), np.float32)
    cT[:D, :M] = c_f32.T
    csq = np.full((1, Mp), 3.0e38, np.float32)   # pad corpus never wins the max
    csq[0, :M] = (c_f32 ** 2).sum(1)
    cT_b, csq = _as_bf16(cT), csq

    def _run(shards):
        Qp = _ceil(Nq, _PMAX * shards)
        qT = np.zeros((Dp, Qp), np.float32)
        qT[:D, :Nq] = q_f32.T
        qT_b = _as_bf16(qT)
        if shards > 1:
            out = _mc_kernel(kpad, shards)[nl.nc(shards)](qT_b, cT_b, csq)
        else:
            out = knn_kernel_baremetal(qT_b, cT_b, csq, kpad)
        return np.asarray(out[:Nq, :k], dtype=np.int64)

    if n_shards > 1:
        try:
            return _run(n_shards)
        except Exception:  # noqa: BLE001 -- fewer cores than requested, etc.
            pass
    return _run(1)


def trainium_knn(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    return_distances: bool = True,
):
    """Exact brute-force k-NN on Trainium (squared-L2), flash-style (never
    materialises the Nq x M distance matrix).

    Parameters
    ----------
    x : (B, Nq, D) or (Nq, D)   query points.
    c : (B, M, D) or (M, D)     corpus points.
    k : int                     neighbours per query (k <= 64, k <= M).

    Returns ``(vals, idxs)`` (both ``(B, Nq, k)``; ``vals`` = true squared-L2
    fp32, ascending; ties by index) or just ``idxs`` when
    ``return_distances=False`` -- shapes match :func:`flash_knn_dispatch`.
    """
    if not _try_init_nki():
        raise RuntimeError("trainium knn backend unavailable: NKI did not import")

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    cb = c.unsqueeze(0) if c.ndim == 2 else c
    B, Nq, D = xb.shape
    M = cb.shape[1]
    if B != 1:
        raise NotImplementedError("trainium knn supports B == 1 (v1)")
    if M > _MAX_M:
        raise NotImplementedError(
            f"trainium knn v1 supports corpus M <= {_MAX_M} (max8 limit); got {M}")
    if k > M:
        raise ValueError(f"k ({k}) must be <= corpus size M ({M})")

    q_np = _to_f32_np(xb[0])                       # (Nq, D)
    c_np = _to_f32_np(cb[0])                        # (M, D)
    idx = _knn_np(q_np, c_np, k)                    # (Nq, k) int64, sim order

    # True-distance epilogue in fp32 (accurate; no ||x||^2 - sim cancellation).
    gathered = c_np[idx]                            # (Nq, k, D)
    dist = ((q_np[:, None, :] - gathered) ** 2).sum(-1)   # (Nq, k) fp32
    # Re-sort each query's k by (dist, idx) to guarantee the ordering contract
    # and repair any bf16 near-tie flips from the kernel's sim ranking.
    order = np.lexsort((idx, dist), axis=1)         # stable, dist primary
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(dist, order, axis=1)

    idxs_t = torch.from_numpy(idx.astype(np.int32)).to(x.device)
    if not return_distances:
        return idxs_t if squeeze else idxs_t.unsqueeze(0)
    vals_t = torch.from_numpy(dist.astype(np.float32)).to(x.device)
    if squeeze:
        return vals_t, idxs_t
    return vals_t.unsqueeze(0), idxs_t.unsqueeze(0)
