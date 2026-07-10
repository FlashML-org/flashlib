"""Trainium (NKI) flash-knn -- Python driver / wrapper.

Public API:
    nki_knn_available() -> bool
    nki_knn(x, c, k, *, return_distances=True) -> (vals, idxs) | idxs

The preferred XLA path fuses exhaustive TF32 ranking, fp32 indirect
gather-distance, and the small candidate reorder into one NKI custom call.
Inputs and outputs remain device-resident; only CPU callers copy the final
``(N, k)`` result back. The NumPy/baremetal path is a compatibility fallback.

v1 scope: squared-L2, ``B == 1``, corpus ``M <= 16384`` (the ``max8``
elements-per-partition limit), and ``k <= 64``. Larger shapes route elsewhere.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# Reuse the kmeans nki driver helpers (NKI import guard, ceil, casts).
from flashlib.primitives.kmeans.nki.assign import (
    _ceil, _to_f32_np, _try_init_nki,
)

_PMAX = 128
_BK = 512
_MAX_M = 16384        # max8 elements-per-partition limit (single-shot v1)
_N_SHARDS = 2         # physical-core width of one Trn2 LNC2 execution scope


def nki_knn_available() -> bool:
    return _try_init_nki()


_MC_KERNEL_CACHE: dict = {}


def _mc_kernel(kpad: int, n_shards: int, *, score_bf16: bool = True,
               fold_csq: bool = True):
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import make_knn_kernel
    key = (kpad, n_shards, score_bf16, fold_csq)
    k = _MC_KERNEL_CACHE.get(key)
    if k is None:
        k = nki.baremetal()(make_knn_kernel(
            kpad, n_shards, score_bf16, fold_csq))
        _MC_KERNEL_CACHE[key] = k
    return k


def _xla_available() -> bool:
    try:
        import torch_xla.core.xla_model  # noqa: F401
        return True
    except Exception:
        return False


_JIT_KERNEL_CACHE: dict = {}


def _jit_kernel(kpad: int, n_shards: int, *, variant: str = "full",
                score_bf16: bool = True, fold_csq: bool = True):
    """Cached shape-policy-specific ``nki.jit`` KNN kernel.

    Unlike ``nki.baremetal`` (which recompiles on *every* call -- ~12 s each,
    which otherwise dominates and hides the multi-core win), the jit kernel is
    compiled once inside the torch-xla graph and reused across calls, so the
    warm per-call cost is the actual device time.
    """
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import (
        make_knn_hierarchical_kernel, make_knn_kernel,
    )
    key = (variant, kpad, n_shards, score_bf16, fold_csq)
    k = _JIT_KERNEL_CACHE.get(key)
    if k is None:
        if variant == "full":
            body = make_knn_kernel(
                kpad, n_shards, score_bf16, fold_csq)
        elif variant == "hierarchical":
            body = make_knn_hierarchical_kernel(
                kpad, n_shards, score_bf16, fold_csq)
        else:
            raise ValueError(f"unknown NKI KNN kernel variant {variant!r}")
        k = nki.jit(body)
        _JIT_KERNEL_CACHE[key] = k
    return k


_JIT_MERGE_CACHE: dict = {}


def _jit_merge_kernel(kpad: int, n_shards: int):
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import (
        make_knn_candidate_merge_kernel,
    )
    key = (kpad, n_shards)
    k = _JIT_MERGE_CACHE.get(key)
    if k is None:
        k = nki.jit(make_knn_candidate_merge_kernel(kpad, n_shards))
        _JIT_MERGE_CACHE[key] = k
    return k


_JIT_DISTANCE_CACHE: dict = {}


def _jit_distance_kernel(
    kpad: int, n_shards: int, candidate_k: Optional[int] = None,
):
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import (
        make_knn_distance_kernel,
    )
    key = (kpad, n_shards, candidate_k)
    k = _JIT_DISTANCE_CACHE.get(key)
    if k is None:
        k = nki.jit(make_knn_distance_kernel(
            kpad, n_shards, candidate_k))
        _JIT_DISTANCE_CACHE[key] = k
    return k


_JIT_FUSED_CACHE: dict = {}


def _jit_fused_kernel(
    kpad: int,
    n_shards: int,
    *,
    score_bf16: bool,
    fold_csq: bool,
    candidate_k: Optional[int] = None,
):
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import (
        make_knn_fused_kernel,
    )
    key = (kpad, n_shards, score_bf16, fold_csq, candidate_k)
    kernel = _JIT_FUSED_CACHE.get(key)
    if kernel is None:
        kernel = nki.jit(make_knn_fused_kernel(
            kpad, n_shards, score_bf16, fold_csq, candidate_k))
        _JIT_FUSED_CACHE[key] = kernel
    return kernel


def _launch(kernel, n_shards: int, *args):
    if n_shards > 1:
        import neuronxcc.nki.language as nl
        return kernel[nl.nc(n_shards)](*args)
    return kernel(*args)


def _kernel_policy(
    M: int, D: int, k: int,
) -> tuple[str, bool, bool, int]:
    """Return ``(variant, score_bf16, fold_csq, candidate_pad)``."""
    import os

    base_pad = _ceil(k, 8)
    # A raw bf16 score row at k=32 measured 0.9888 recall. Selecting eight
    # extra candidates and fp32-reranking them restored 1.0 recall while
    # remaining ~18% faster than an fp32 score scan. k>56 has no room to
    # oversample under the k<=64 kernel limit, so it stays fp32.
    score_bf16 = 32 <= base_pad <= 56
    override = os.environ.get("FLASHLIB_NKI_KNN_SCORE_BF16", "").lower()
    if override in ("0", "false", "off"):
        score_bf16 = False
    candidate_pad = _ceil(k + 8, 8) if score_bf16 else base_pad
    # Hierarchical local-top-k is exact with respect to the rank score and is
    # retained for future M>16384 work, but on the current M<=16384 shapes its
    # extra candidate writes/merge measured 2.6x slower. Keep it opt-in for
    # A/B; the no-regression production route stays full-score.
    variant_override = os.environ.get(
        "FLASHLIB_NKI_KNN_VARIANT", "").lower()
    variant = "hierarchical" if variant_override == "hierarchical" else "full"
    return variant, score_bf16, True, candidate_pad


def _nki_knn_xla(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    *,
    return_distances: bool,
    n_shards: Optional[int] = None,
):
    """Device-resident rank, fp32 gather-distance, and lexicographic reorder."""
    import torch.nn.functional as F
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    cb = c.unsqueeze(0) if c.ndim == 2 else c
    B, Nq, D = xb.shape
    M = cb.shape[1]
    if B != 1 or cb.shape[0] != 1:
        raise NotImplementedError("NKI KNN supports B == 1")

    q = xb[0].to(dev, dtype=torch.float32)
    corpus = cb[0].to(dev, dtype=torch.float32)
    if n_shards is None:
        from flashlib.primitives.kmeans.nki.kmeans import _neuron_n_shards
        n_shards = _neuron_n_shards()

    # Float32 NKI matmul inputs use Trainium's higher-precision TF32 path.
    # It measured slightly faster end-to-end than bf16 here (no cast/residual
    # packing) and materially improves candidate recall on near ties.
    Dp = _ceil(D + 1, _PMAX)
    Qp = _ceil(Nq, _PMAX * n_shards)
    Mp = _ceil(M, _BK)
    variant, score_bf16, fold_csq, kpad = _kernel_policy(Mp, D, k)
    distance_candidates = kpad if score_bf16 else k

    q_eff = torch.cat((
        q,
        torch.ones(Nq, 1, device=dev, dtype=torch.float32),
    ), dim=1)
    qT = F.pad(
        q_eff.t(), (0, Qp - Nq, 0, Dp - (D + 1)))

    corpus_pad_rank = F.pad(corpus, (0, 0, 0, Mp - M))
    bias = -0.5 * (corpus * corpus).sum(1)
    bias_valid = bias
    if Mp > M:
        # A huge bf16 value (for example -3e38) can overflow the PE fp32
        # accumulator and max8 may treat the resulting NaN as a winner. Use a
        # finite, data-dependent lower bound instead:
        # q.c - .5||c||² >= -D*max|q|*max|c| - .5*D*max|c|².
        q_abs = q.abs().max()
        c_abs = corpus.abs().max()
        lower = -D * q_abs * c_abs - 0.5 * D * c_abs * c_abs
        pad_bias = torch.clamp(
            lower - lower.abs() * 0.01 - 1.0, min=-1.0e30)
        bias_row = torch.cat((
            bias_valid,
            pad_bias.expand(Mp - M),
        ))
    else:
        bias_row = bias_valid
    corpus_eff = torch.cat(
        (corpus_pad_rank, bias_row.unsqueeze(1)), dim=1)
    cT = F.pad(
        corpus_eff.t(), (0, 0, 0, Dp - (D + 1)))
    csq = torch.zeros(1, Mp, device=dev, dtype=torch.float32)

    raw_idx = None
    if variant != "full":
        rank_kernel = _jit_kernel(
            kpad, n_shards, variant=variant,
            score_bf16=score_bf16, fold_csq=fold_csq)
        cand_score, cand_idx = _launch(
            rank_kernel, n_shards, qT, cT, csq)
        pos = _launch(
            _jit_merge_kernel(kpad, n_shards),
            n_shards, cand_score)
        raw_idx = torch.gather(cand_idx, 1, pos.to(torch.int64))

    # True fp32 distance recovery is another NKI custom call. Corpus padding
    # keeps indirect loads safe for padded query rows; real rows always select
    # real candidates because k <= M and pad rank scores are -inf.
    q_pad = F.pad(q, (0, 0, 0, Qp - Nq))
    corpus_pad = F.pad(corpus, (0, 0, 0, Mp - M))
    if variant == "full":
        sorted_dist, sorted_pos, raw_idx = _launch(
            _jit_fused_kernel(
                kpad, n_shards, score_bf16=score_bf16,
                fold_csq=fold_csq, candidate_k=distance_candidates),
            n_shards, qT, cT, csq, q_pad, corpus_pad)
    else:
        assert raw_idx is not None
        sorted_dist, sorted_pos = _launch(
            _jit_distance_kernel(
                kpad, n_shards, distance_candidates), n_shards,
            q_pad, corpus_pad, raw_idx)
    vals = sorted_dist[:Nq, :k]
    idx = torch.gather(
        raw_idx[:Nq], 1, sorted_pos[:Nq, :k].to(torch.int64),
    ).to(torch.int32)
    xm.mark_step()

    if x.device.type != "xla":
        vals = vals.cpu()
        idx = idx.cpu()
    if squeeze:
        return (vals, idx) if return_distances else idx
    vals = vals.unsqueeze(0)
    idx = idx.unsqueeze(0)
    return (vals, idx) if return_distances else idx


def _knn_np(q_f32: np.ndarray, c_f32: np.ndarray, k: int,
            n_shards: Optional[int] = None) -> np.ndarray:
    """Run the knn kernel; return int64 indices [Nq, k] (nearest first by the
    x2-free sim). ``q_f32`` [Nq, D], ``c_f32`` [M, D]. Shards the query axis
    over all usable NeuronCores (probed; ~1.9x on trn2's 2 logical cores),
    single-core fallback. ``n_shards=None`` auto-detects the usable core count.

    Operands are **augmented** so the ``nc_matmul`` output is the rank score
    directly: append a contraction row carrying ``-0.5||c||^2`` to the corpus
    and a constant ``1`` row to the queries, so
    ``<q',c'> = <q,c> - 0.5||c||^2`` (monotone in the true sim, same argmax).
    This lets the kernel skip the per-tile ``2*cross - c_sq`` Vector epilogue
    (``fold_csq=True``) -- measured ~1.3x. The kernel keeps the score row in
    FP32 in this compatibility path. The driver recomputes true FP32 distances
    at the returned indices and re-sorts below.
    """
    import neuronxcc.nki.language as nl

    Nq, D = q_f32.shape
    M = c_f32.shape[0]
    # Float32 operands select the higher-precision TF32 matmul path. Fold the
    # exact fp32 norm into one extra contraction row.
    norm_rows = 1
    Dp, Mp, kpad = (
        _ceil(D + norm_rows, _PMAX), _ceil(M, _BK), _ceil(k, 8)
    )
    cT = np.zeros((Dp, Mp), np.float32)
    cT[:D, :M] = c_f32.T
    bias = -0.5 * (c_f32 ** 2).sum(1)
    cT[D, :M] = bias
    if Mp > M:
        q_abs = float(np.abs(q_f32).max())
        c_abs = float(np.abs(c_f32).max())
        lower = -D * q_abs * c_abs - 0.5 * D * c_abs * c_abs
        cT[D, M:] = max(
            lower - abs(lower) * 0.01 - 1.0, -1.0e30)
    csq_np = np.zeros((1, Mp), np.float32)        # unused (fold_csq=True)
    variant, _, fold_csq, _ = _kernel_policy(Mp, D, k)
    # Compatibility path returns only k candidates to its NumPy epilogue;
    # retain fp32 scores rather than requiring an oversized candidate result.
    score_bf16 = False

    def _qT(shards):
        Qp = _ceil(Nq, _PMAX * shards)
        qT = np.zeros((Dp, Qp), np.float32)
        qT[:D, :Nq] = q_f32.T
        qT[D:D + norm_rows, :Nq] = 1.0            # pairs with norm rows
        return qT

    # Preferred path: cached nki.jit on all usable NeuronCores. The kernel is
    # compiled once (per (kpad, shards)) inside torch-xla and reused, so warm
    # calls pay only device time -- and query-sharding gives ~1.9x on trn2's 2
    # logical cores. (nki.baremetal recompiles every call, ~12 s, which would
    # dwarf and hide the multi-core win; it is kept only as the no-torch-xla
    # fallback below.)
    if _xla_available():
        try:
            import torch
            import torch_xla.core.xla_model as xm
            from flashlib.primitives.kmeans.nki.kmeans import _neuron_n_shards

            dev = xm.xla_device()
            shards = n_shards if n_shards else _neuron_n_shards()
            cT_d = torch.from_numpy(cT).to(dev, torch.float32)
            csq_d = torch.from_numpy(csq_np).to(dev)
            qT_d = torch.from_numpy(_qT(shards)).to(dev, torch.float32)
            variants = [variant]
            if variant != "full":
                variants.append("full")
            for selected in variants:
                try:
                    kern = _jit_kernel(
                        kpad, shards, variant=selected,
                        score_bf16=score_bf16, fold_csq=fold_csq)
                    if selected == "full":
                        out = (
                            kern[nl.nc(shards)](qT_d, cT_d, csq_d)
                            if shards > 1 else kern(qT_d, cT_d, csq_d)
                        )
                    else:
                        result = (
                            kern[nl.nc(shards)](qT_d, cT_d, csq_d)
                            if shards > 1 else kern(qT_d, cT_d, csq_d)
                        )
                        cand_score, cand_idx = result
                        merge = _jit_merge_kernel(kpad, shards)
                        pos = (
                            merge[nl.nc(shards)](cand_score)
                            if shards > 1 else merge(cand_score)
                        )
                        out = torch.gather(
                            cand_idx, 1, pos.to(torch.int64))
                    xm.mark_step()
                    return out[:Nq, :k].to(torch.int64).cpu().numpy()
                except Exception:                                  # noqa: BLE001
                    if selected == variants[-1]:
                        raise
        except Exception:  # noqa: BLE001 -- torch-xla missing/unpatched, etc.
            pass

    # Fallback: baremetal (numpy in/out; recompiles per call).
    cT_b = cT
    shards = n_shards if n_shards else _N_SHARDS

    def _run(sh):
        qT_b = _qT(sh)
        kernel = _mc_kernel(
            kpad, sh, score_bf16=score_bf16, fold_csq=fold_csq)
        if sh > 1:
            out = kernel[nl.nc(sh)](qT_b, cT_b, csq_np)
        else:
            out = kernel(qT_b, cT_b, csq_np)
        return np.asarray(out[:Nq, :k], dtype=np.int64)

    if shards > 1:
        try:
            return _run(shards)
        except Exception:  # noqa: BLE001 -- fewer cores than requested, etc.
            pass
    return _run(1)


def nki_knn(
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
        raise RuntimeError("NKI KNN backend unavailable: NKI did not import")

    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    cb = c.unsqueeze(0) if c.ndim == 2 else c
    B, Nq, D = xb.shape
    M = cb.shape[1]
    if B != 1 or cb.shape[0] != 1:
        raise NotImplementedError("NKI KNN supports B == 1 (v1)")
    if cb.shape[2] != D:
        raise ValueError(
            f"query/corpus feature dimensions differ: {D} vs {cb.shape[2]}")
    if D > 512:
        raise NotImplementedError("NKI KNN supports D <= 512 (v1)")
    if Nq == 0 or M == 0:
        raise ValueError("NKI KNN requires non-empty query and corpus")
    if M > _MAX_M:
        raise NotImplementedError(
            f"NKI KNN v1 supports corpus M <= {_MAX_M} (max8 limit); got {M}")
    if k < 1 or k > 64 or k > M:
        raise ValueError(
            f"k must satisfy 1 <= k <= min(64, M={M}); got {k}")

    if _xla_available():
        try:
            return _nki_knn_xla(
                x, c, k, return_distances=return_distances)
        except Exception:                                          # noqa: BLE001
            # Keep the proven NumPy/baremetal route as a compatibility
            # fallback for older Neuron SDKs that cannot compile the
            # indirect-gather epilogue.
            pass

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
