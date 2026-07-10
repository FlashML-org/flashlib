"""Trainium (NKI) flash-kmeans assign -- Python wrappers / driver.

Public API:
    nki_available() -> bool
    nki_assign_euclid(x, centroids, out=None) -> cluster_ids
    nki_info() -> str

The heavy lifting is in :mod:`assign_kernel` (the NKI kernel). This module
handles torch <-> numpy marshalling, padding to the PE-array tile
constraints, and driver selection.

Execution paths
---------------
* torch-neuronx / XLA (preferred for CPU and XLA inputs): the ``@nki.jit``
  kernel runs inside the XLA graph and caches compilation. On Amazon Linux
  2023 (glibc 2.34) the Neuron
  torch-xla wheel needs the one-line ELF fix in
  ``tools/neuron/patch_torch_xla_glibc.py`` (its ``_XLAC`` imports
  ``hypot@GLIBC_2.35``); after that, ``torch_xla`` loads and runs on the
  NeuronCore.
* baremetal: compatibility fallback when torch-neuronx is unavailable.
  ``nki.baremetal`` compiles + runs directly with NumPy inputs/outputs.

v1 scope: euclidean metric, ``B == 1``, ``D <= 512``. bf16 matmul inputs
with fp32 ``c_sq`` and fp32 accumulation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

_BK = 512      # centroid tile (moving free) -- Kp padded to a multiple of this
_PMAX = 128    # partition / M-tile size -- N and D padded to a multiple of this
_N_SHARDS = 2  # fallback physical-core width for one LNC2 execution scope

_NKI_AVAILABLE: Optional[bool] = None
_NKI_IMPORT_ERROR: Optional[Exception] = None


def _try_init_nki() -> bool:
    """Return True iff the NKI toolchain imports (cached)."""
    global _NKI_AVAILABLE, _NKI_IMPORT_ERROR
    if _NKI_AVAILABLE is not None:
        return _NKI_AVAILABLE
    try:
        import ml_dtypes  # noqa: F401
        import neuronxcc.nki  # noqa: F401
        import neuronxcc.nki.language  # noqa: F401
        import neuronxcc.nki.isa  # noqa: F401

        _NKI_AVAILABLE = True
    except Exception as e:  # pragma: no cover - depends on local install
        _NKI_IMPORT_ERROR = e
        _NKI_AVAILABLE = False
    return _NKI_AVAILABLE


def nki_available() -> bool:
    return _try_init_nki()


def _ceil(a: int, b: int) -> int:
    return ((a + b - 1) // b) * b


def _to_f32_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()


def _as_bf16(a: np.ndarray) -> np.ndarray:
    import ml_dtypes
    return a.astype(ml_dtypes.bfloat16)


def prep_assign(x_f32: np.ndarray, c_f32: np.ndarray, n_shards: int = 1):
    """Pad + transpose to the kernel's operand layout.

    Returns ``(xT_bf16 [Dp, Np], cT_bf16 [Dp, Kp], csq_f32 [1, Kp])``. With
    ``n_shards`` > 1 the point axis is padded to a multiple of
    ``n_shards * _PMAX`` so each NeuronCore gets an equal M-tile block.
    """
    N, D = x_f32.shape
    K = c_f32.shape[0]
    Np, Dp, Kp = _ceil(N, _PMAX * n_shards), _ceil(D, _PMAX), _ceil(K, _BK)

    xT = np.zeros((Dp, Np), np.float32)
    xT[:D, :N] = x_f32.T
    cT = np.zeros((Dp, Kp), np.float32)
    cT[:D, :K] = c_f32.T
    csq = np.full((1, Kp), 3.0e38, np.float32)   # pad centroids never win argmax
    csq[0, :K] = (c_f32 ** 2).sum(1)
    return _as_bf16(xT), _as_bf16(cT), csq


_MC_KERNEL_CACHE: dict = {}


def _mc_kernel(n_shards: int):
    import neuronxcc.nki as nki
    from flashlib.primitives.kmeans.nki.assign_kernel import make_assign_kernel
    k = _MC_KERNEL_CACHE.get(n_shards)
    if k is None:
        k = nki.baremetal()(make_assign_kernel(n_shards))
        _MC_KERNEL_CACHE[n_shards] = k
    return k


def _assign_np(x_f32: np.ndarray, c_f32: np.ndarray,
               n_shards: int = _N_SHARDS) -> np.ndarray:
    """Run the baremetal assign kernel; return int32 cluster ids [N].

    Shards the point axis over ``n_shards`` NeuronCores (~2x on trn2). Falls
    back to single core if the multi-core launch fails (e.g. fewer logical
    cores available than requested).
    """
    import neuronxcc.nki.language as nl
    from flashlib.primitives.kmeans.nki.assign_kernel import (
        flash_assign_kernel_baremetal,
    )
    N = x_f32.shape[0]
    if n_shards > 1:
        try:
            xT, cT, csq = prep_assign(x_f32, c_f32, n_shards=n_shards)
            out = _mc_kernel(n_shards)[nl.nc(n_shards)](xT, cT, csq)
            return np.asarray(out[:N, 0], dtype=np.int32)
        except Exception:
            pass  # fall back to single core
    xT, cT, csq = prep_assign(x_f32, c_f32)
    out = flash_assign_kernel_baremetal(xT, cT, csq)   # [Np, 1] int32
    return np.asarray(out[:N, 0], dtype=np.int32)


def nki_assign_euclid(
    x: torch.Tensor,
    centroids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Nearest-centroid assignment on Trainium (squared-L2).

    Parameters
    ----------
    x : (B, N, D) or (N, D) tensor.
    centroids : (B, K, D) or (K, D) tensor.

    Returns cluster ids as an int32 tensor shaped like x without its D axis.
    """
    if not _try_init_nki():
        raise RuntimeError(
            f"NKI backend unavailable: NKI toolchain did not import "
            f"({_NKI_IMPORT_ERROR!r})"
        )
    squeeze = x.ndim == 2
    xb = x.unsqueeze(0) if squeeze else x
    cb = centroids.unsqueeze(0) if centroids.ndim == 2 else centroids
    B, N, D = xb.shape
    if B != 1:
        raise NotImplementedError("NKI KMeans supports B == 1 (v1)")
    if D > _BK:
        raise NotImplementedError("NKI KMeans supports D <= 512 (v1)")

    # Prefer the cached XLA custom call even for CPU callers. Mixing an active
    # torch-xla runtime with nki.baremetal in the same process tears down NRT
    # state on current Neuron SDKs; this also avoids per-call baremetal compile.
    try:
        import torch.nn.functional as F
        import torch_xla.core.xla_model as xm
        from flashlib.primitives.kmeans.nki.kmeans import (
            _compiled_large_call, _jit_assign_kernel, _launch,
            _LARGE_K_N_CHUNK, _neuron_n_shards,
        )

        dev = xm.xla_device()
        return_on_device = x.device.type == "xla"
        K = cb.shape[-2]
        large_k = K >= 32768
        x2d = xb[0].to(
            dev, dtype=torch.bfloat16 if large_k else torch.float32)
        c2d = cb[0].to(dev, dtype=torch.float32)
        n_shards = _neuron_n_shards()
        k_resident = 4096
        point_lanes = 4 if large_k else 1
        Np = _ceil(
            N, _LARGE_K_N_CHUNK
            if large_k else _PMAX * n_shards * point_lanes)
        Dp = _ceil(D, _PMAX)
        Kp = _ceil(K, k_resident if large_k else _BK)
        fold_csq = (Dp - D) >= 2 and not large_k

        xbf = x2d.to(torch.bfloat16)
        cbf = c2d.to(torch.bfloat16)
        csq_valid = (c2d * c2d).sum(1)
        csq = F.pad(
            csq_valid, (0, Kp - K), value=3.0e38).unsqueeze(0)
        if fold_csq:
            x_eff = torch.cat((
                xbf,
                torch.ones(N, 2, device=dev, dtype=torch.bfloat16),
            ), dim=1)
            c_pad = F.pad(cbf, (0, 0, 0, Kp - K))
            bias = -0.5 * csq_valid
            bias_hi_valid = bias.to(torch.bfloat16)
            bias_lo_valid = (
                bias - bias_hi_valid.float()).to(torch.bfloat16)
            bias_hi = F.pad(
                bias_hi_valid, (0, Kp - K), value=-1.0e30)
            bias_lo = F.pad(
                bias_lo_valid, (0, Kp - K), value=0.0)
            c_eff = torch.cat((
                c_pad, bias_hi.unsqueeze(1), bias_lo.unsqueeze(1),
            ), dim=1)
            xT = F.pad(
                x_eff.t(), (0, Np - N, 0, Dp - (D + 2)))
            cT = F.pad(
                c_eff.t(), (0, 0, 0, Dp - (D + 2)))
        else:
            xT = (
                None if large_k else
                F.pad(xbf.t(), (0, Np - N, 0, Dp - D)))
            cT = F.pad(
                F.pad(cbf, (0, 0, 0, Kp - K)).t(),
                (0, 0, 0, Dp - D))

        kernel = _jit_assign_kernel(
            n_shards, D=D, K=K, fold_csq=fold_csq,
            score_bf16=False, large_k=large_k,
            k_resident=k_resident, point_lanes=point_lanes)
        if large_k:
            call = _compiled_large_call(
                ("assign_api", D, Kp, n_shards),
                f"kmeans_assign_api_d{D}_k{Kp}_nc{n_shards}",
                kernel, n_shards)
            chunks = []
            for n0 in range(0, Np, _LARGE_K_N_CHUNK):
                real = max(0, min(_LARGE_K_N_CHUNK, N - n0))
                x_chunk = F.pad(
                    xbf[n0:n0 + real, :].t(),
                    (0, _LARGE_K_N_CHUNK - real, 0, Dp - D))
                xm.mark_step()
                chunks.append(call(x_chunk, cT, csq))
            xm.wait_device_ops()
            raw = torch.cat(chunks, dim=0)
        else:
            raw = _launch(kernel, n_shards, xT, cT, csq)
        ids_t = raw[:N, 0].to(torch.int32)
        xm.mark_step()
        if not return_on_device:
            ids_t = ids_t.cpu()
        if out is not None:
            out.copy_(ids_t.to(out.device).view(out.shape))
            return out
        return ids_t if squeeze else ids_t.unsqueeze(0)
    except Exception:
        # Compatibility fallback for installations without torch-neuronx.
        pass

    ids = _assign_np(_to_f32_np(xb[0]), _to_f32_np(cb[0]))
    ids_t = torch.from_numpy(ids).to(x.device)
    if out is not None:
        out.copy_(ids_t.view(out.shape))
        return out
    return ids_t if squeeze else ids_t.unsqueeze(0)


def nki_info() -> str:
    if _try_init_nki():
        import neuronxcc
        return f"Trainium NKI assign available (neuronxcc=={neuronxcc.__version__})"
    return f"Trainium NKI UNAVAILABLE: {_NKI_IMPORT_ERROR!r}"
