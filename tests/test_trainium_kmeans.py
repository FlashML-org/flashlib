"""Trainium (NKI) flash-kmeans parity tests.

Unlike ``test_backend_parity.py`` (CUDA-guarded), these run on an AWS
NeuronDevice host (Trainium/Inferentia) and are skipped everywhere else.
They exercise the ``nki.baremetal`` path -- the standalone assign kernel
against a torch argmin reference, and the end-to-end Lloyd loop against a
numpy reference. NKI compiles per call, so shapes are kept small.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HAS_NEURON = bool(glob.glob("/dev/neuron*"))
if not _HAS_NEURON:
    pytest.skip("Trainium tests require a NeuronDevice (/dev/neuron*)",
                allow_module_level=True)

try:
    from flashlib.primitives.kmeans.trainium import trainium_available
    if not trainium_available():
        raise ImportError
except Exception:  # pragma: no cover
    pytest.skip("NKI toolchain (neuronxcc) unavailable", allow_module_level=True)

SEED = 0


def _torch_argmin(x, c):
    xs = (x * x).sum(-1, keepdim=True)
    cs = (c * c).sum(-1).unsqueeze(0)
    cross = x.float() @ c.float().t()
    return (xs + cs - 2.0 * cross).argmin(-1).to(torch.int32)


def test_trainium_assign_matches_torch_reference():
    torch.manual_seed(SEED)
    N, D, K = 512, 128, 256
    x = torch.randn(N, D, dtype=torch.bfloat16)
    c = torch.randn(K, D, dtype=torch.bfloat16)

    from flashlib.primitives.kmeans.trainium import trainium_assign_euclid
    ids = trainium_assign_euclid(x, c).cpu()
    ref = _torch_argmin(x, c).cpu()

    # bf16 near-ties can flip the argmin; allow a small rate (as the CuteDSL
    # parity test does).
    mismatch = (ids != ref).float().mean().item()
    assert mismatch < 3e-2, f"trainium assign mismatch {mismatch:.4f}"


def test_trainium_kmeans_converges_like_numpy():
    rng = np.random.default_rng(SEED)
    N, D, K, iters = 512, 64, 16, 4
    x = rng.standard_normal((N, D), dtype=np.float32)
    cent0 = x[rng.integers(0, N, K)].copy()

    from flashlib.primitives.kmeans.trainium import trainium_kmeans_Euclid
    ids, cent, nit = trainium_kmeans_Euclid(
        torch.from_numpy(x), K, max_iters=iters,
        init_centroids=torch.from_numpy(cent0))
    cent = cent.cpu().numpy()

    # numpy Lloyd reference from the same init
    ref = cent0.copy()
    for _ in range(iters):
        cid = ((x[:, None] - ref[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            m = cid == k
            if m.any():
                ref[k] = x[m].mean(0)

    def inertia(cc):
        return ((x[:, None] - cc[None]) ** 2).sum(-1).min(1).sum()

    ratio = inertia(cent) / inertia(ref)
    assert nit == iters
    assert ratio < 1.02, f"trainium kmeans inertia ratio {ratio:.4f} too high"


def test_trainium_transposed_update_matches_index_add():
    """The default (transposed one-hot-matmul) XLA update should track the
    ``index_add`` reference. Same init + iters, so the only difference is the
    M-step implementation; centroids should agree up to bf16 rounding of the
    sums (the assign uses bf16 either way, so cluster ids match)."""
    pytest.importorskip("torch_xla")
    rng = np.random.default_rng(SEED)
    N, D, K, iters = 1024, 64, 32, 5           # D <= 128 -> transposed path
    x = torch.from_numpy(rng.standard_normal((N, D), dtype=np.float32))
    cent0 = torch.from_numpy(x.numpy()[rng.integers(0, N, K)].copy())

    from flashlib.primitives.kmeans.trainium.kmeans import _trainium_kmeans_xla
    _, cent_nki, _ = _trainium_kmeans_xla(
        x, K, iters, 0.0, cent0.clone(), False, use_nki_update=True)
    _, cent_ia, _ = _trainium_kmeans_xla(
        x, K, iters, 0.0, cent0.clone(), False, use_nki_update=False)
    cent_nki = cent_nki.cpu().numpy()
    cent_ia = cent_ia.cpu().numpy()

    def inertia(cc):
        return ((x.numpy()[:, None] - cc[None]) ** 2).sum(-1).min(1).sum()

    # Inertia (clustering quality) must match closely; per-centroid means can
    # differ by the bf16 quantisation of the summed points.
    ratio = inertia(cent_nki) / inertia(cent_ia)
    assert 0.98 < ratio < 1.02, f"transposed-update inertia ratio {ratio:.4f}"


def test_flash_kmeans_routes_to_trainium():
    """The dispatcher auto-selects the trainium backend on a Neuron host."""
    from flashlib.primitives.kmeans.impl import _route
    from flashlib import _hw
    backend, _ = _route(B=1, N=1000, D=64, K=16, metric="euclidean",
                        hw=_hw.current())
    assert backend == "trainium"
