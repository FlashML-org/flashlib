"""IVF-PQ index build (Triton/GPU path).

The build reuses three already-tuned flashlib primitives and adds only
the PQ-specific encode + cell-contiguous layout:

* coarse quantizer training -> :func:`flashlib.primitives.kmeans.flash_kmeans`
  on a sample of the database (FAISS-style ``min(M, nlist * 256)`` rows).
* full-database assignment -> :func:`...kmeans.triton.assign.euclid_assign_triton`
  (the x²-free nearest-centroid kernel; one pass over the database).
* PQ codebook training -> :func:`...kmeans.batch_kmeans_Euclid` run as a
  **single batched k-means of ``B = m`` problems**, one per sub-quantizer,
  each clustering the residual sub-vectors into ``ksub = 256`` centroids.
  This is the key reuse: training all ``m`` codebooks is just one
  batched call into the existing Lloyd kernel.

PQ-specific work:

    residual = X - centroids[assign]          # (M, Dp), or X if not by_residual
    codes[:, s] = nearest sub-centroid of residual sub-vector s   # (M, m) uint8
    counts   = bincount(assign)
    offsets  = [0, cumsum(counts)]            # (nlist + 1,)
    order    = argsort(assign, stable=True)   # stored-row -> original id
    codes    = codes[order]                   # cell-contiguous codes

Padding: ``D`` is zero-padded to ``Dp = m * dsub`` (``>= 16``) so every
sub-vector has a uniform width and the coarse kernels keep a contraction
dim >= 16. Zero columns add 0 to every squared-L2 distance.
"""
from __future__ import annotations

from typing import Optional

import torch

from flashlib.primitives.ivf_pq.index import IvfPqIndex
from flashlib.primitives.ivf_pq.torch_fallback import _pad_features, _pq_dims
from flashlib.primitives.kmeans import batch_kmeans_Euclid, flash_kmeans
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton


def _train_pq_codebooks_triton(
    resid: torch.Tensor, m: int, dsub: int, ksub: int, *, niter: int, seed: int
) -> torch.Tensor:
    """Train ``m`` sub-quantizers as one ``B=m`` batched k-means.

    ``resid`` is ``(N, m*dsub)`` fp32 (CUDA); returns ``(m, ksub, dsub)``.
    Deterministic: sub-centroids are initialised from a seeded random
    sample of the residual sub-vectors.
    """
    N = resid.shape[0]
    resid_sub = resid.reshape(N, m, dsub).permute(1, 0, 2).contiguous()   # (m, N, dsub)

    g = torch.Generator(device=resid.device).manual_seed(seed)
    sel = torch.randint(0, N, (m, ksub), generator=g, device=resid.device)  # (m, ksub)
    init = torch.gather(resid_sub, 1, sel[..., None].expand(-1, -1, dsub)).contiguous()

    _, codebooks, _ = batch_kmeans_Euclid(
        resid_sub, ksub, max_iters=niter, init_centroids=init,
    )                                                                       # (m, ksub, dsub)
    return codebooks.to(torch.float32).contiguous()


def _encode_pq_triton(
    resid: torch.Tensor, codebooks: torch.Tensor, m: int, dsub: int
) -> torch.Tensor:
    """Encode residual rows to ``(N, m)`` uint8 codes via the x²-free assign."""
    N = resid.shape[0]
    resid_sub = resid.reshape(N, m, dsub).permute(1, 0, 2).contiguous()   # (m, N, dsub)
    codes_mN = euclid_assign_triton(resid_sub, codebooks)                 # (m, N) int32
    return codes_mN.t().contiguous().to(torch.uint8)                      # (N, m)


def ivf_pq_build_triton(
    X: torch.Tensor,
    nlist: int,
    *,
    m: int = 8,
    nbits: int = 8,
    metric: str = "l2",
    by_residual: bool = True,
    nprobe: int = 8,
    niter: int = 20,
    pq_niter: int = 25,
    train_size: Optional[int] = None,
    pq_train_size: Optional[int] = None,
    seed: int = 0,
) -> IvfPqIndex:
    """Build an IVF-PQ index on the GPU. Returns :class:`IvfPqIndex`."""
    if metric != "l2":
        raise NotImplementedError(
            f"ivf_pq currently supports metric='l2' only (got {metric!r})"
        )
    if nbits != 8:
        raise NotImplementedError(f"ivf_pq supports nbits=8 only (got {nbits})")
    if not X.is_cuda or X.ndim != 2:
        raise ValueError("ivf_pq_build_triton requires a 2D CUDA tensor")

    M, D = X.shape
    nlist = int(min(nlist, M))
    if nlist < 1:
        raise ValueError("nlist must be >= 1")
    m, dsub, Dp = _pq_dims(int(D), m)
    ksub = 1 << nbits                                            # 256
    Xp = _pad_features(X.to(torch.float32), Dp).contiguous()    # (M, Dp)

    # ── coarse quantizer: k-means on a sample ──────────────────────────
    train_size = int(train_size or min(M, nlist * 256))
    train_size = max(min(train_size, M), nlist)
    g = torch.Generator(device=X.device).manual_seed(seed)
    sample_idx = torch.randperm(M, generator=g, device=X.device)[:train_size]
    sample = Xp.index_select(0, sample_idx)

    _, centroids, _ = flash_kmeans(
        sample, nlist, max_iters=niter, metric="euclidean",
    )
    centroids = centroids.to(torch.float32).contiguous()        # (nlist, Dp)

    # ── assign every database row to its nearest centroid ──────────────
    assign = euclid_assign_triton(
        Xp.unsqueeze(0), centroids.unsqueeze(0),
    ).squeeze(0).to(torch.int64)                                # (M,)

    # ── residuals + PQ codebooks (trained on a residual subsample) ─────
    resid_all = (
        Xp - centroids.index_select(0, assign) if by_residual else Xp
    )                                                           # (M, Dp)
    pq_train_size = int(pq_train_size or min(M, max(ksub * 16, 4096)))
    pq_train_size = max(min(pq_train_size, M), ksub)
    pq_idx = torch.randperm(M, generator=g, device=X.device)[:pq_train_size]
    resid_train = resid_all.index_select(0, pq_idx).contiguous()
    codebooks = _train_pq_codebooks_triton(
        resid_train, m, dsub, ksub, niter=pq_niter, seed=seed + 1,
    )                                                           # (m, ksub, dsub)

    # ── encode all residuals to PQ codes ───────────────────────────────
    codes = _encode_pq_triton(resid_all, codebooks, m, dsub)    # (M, m) uint8

    # ── CSR cell-contiguous layout ─────────────────────────────────────
    counts = torch.bincount(assign, minlength=nlist)            # (nlist,)
    offsets = torch.zeros(nlist + 1, dtype=torch.int64, device=X.device)
    offsets[1:] = counts.cumsum(0)
    order = torch.argsort(assign, stable=True)                  # (M,) int64
    codes_sorted = codes.index_select(0, order).contiguous()    # (M, m)
    max_list_len = int(counts.max().item())                     # one sync at build

    return IvfPqIndex(
        centroids=centroids,
        pq_codebooks=codebooks,
        codes=codes_sorted,
        ids=order.to(torch.int64),
        list_offsets=offsets,
        metric=metric,
        by_residual=bool(by_residual),
        D=int(D),
        Dp=int(Dp),
        dsub=int(dsub),
        m=int(m),
        nbits=int(nbits),
        nlist=int(nlist),
        nprobe=int(nprobe),
        max_list_len=max_list_len,
    )


__all__ = ["ivf_pq_build_triton"]
