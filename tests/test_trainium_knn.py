"""Trainium (NKI) flash-knn parity tests.

Run on an AWS NeuronDevice host (Trainium/Inferentia); skipped everywhere
else. They exercise the ``nki.baremetal`` path (the NKI top-k kernel + the
fp32 distance epilogue) against a torch brute-force reference, using a
set-overlap recall metric to tolerate bf16 near-ties (as the CUDA knn parity
tests do). NKI compiles per shape, so sizes are kept small.
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
    from flashlib.primitives.knn.trainium import trainium_knn_available
    if not trainium_knn_available():
        raise ImportError
except Exception:  # pragma: no cover
    pytest.skip("NKI toolchain (neuronxcc) unavailable", allow_module_level=True)

SEED = 0


def _torch_topk(q, c, k):
    """Exact top-k by squared-L2 (fp32): returns (dist[N,k], idx[N,k])."""
    d2 = ((q[:, None, :] - c[None]) ** 2).sum(-1)
    vals, idx = torch.topk(d2, k, dim=1, largest=False, sorted=True)
    return vals, idx.to(torch.int32)


def _recall(idx, ref):
    idx = np.asarray(idx); ref = np.asarray(ref)
    return float(np.mean([len(set(idx[i]) & set(ref[i])) / idx.shape[1]
                          for i in range(idx.shape[0])]))


@pytest.mark.parametrize("Nq,M,D,k", [
    (300, 2000, 64, 10),     # search-ish: modest Q, mid corpus
    (1024, 4096, 128, 32),   # build-ish: large Q, larger k
    (64, 512, 16, 8),        # small D (padded to 128 partition internally)
])
def test_trainium_knn_matches_torch_reference(Nq, M, D, k):
    torch.manual_seed(SEED)
    q = torch.randn(Nq, D)
    c = torch.randn(M, D)

    from flashlib.primitives.knn.trainium import trainium_knn
    vals, idx = trainium_knn(q, c, k)
    ref_vals, ref_idx = _torch_topk(q, c, k)

    # Set-overlap recall (bf16 ranking can flip near-ties).
    rec = _recall(idx.cpu().numpy(), ref_idx.cpu().numpy())
    assert rec > 0.95, f"trainium knn recall@{k} = {rec:.4f}"

    # Distances are true fp32 squared-L2 (gathered), ascending.
    v = vals.cpu().numpy()
    assert np.all(np.diff(v, axis=1) >= -1e-3), "distances not ascending"
    # Returned vals equal the exact distance at the returned indices.
    d2 = ((q[:, None, :] - c[None]) ** 2).sum(-1).cpu().numpy()
    true = np.take_along_axis(d2, idx.cpu().numpy().astype(np.int64), axis=1)
    assert np.abs(v - true).max() < 1e-2


def test_trainium_knn_indices_only():
    torch.manual_seed(SEED)
    q = torch.randn(200, 32)
    c = torch.randn(1500, 32)
    from flashlib.primitives.knn.trainium import trainium_knn
    idx = trainium_knn(q, c, 8, return_distances=False)
    assert idx.shape == (200, 8) and idx.dtype == torch.int32
    vals, idx2 = trainium_knn(q, c, 8, return_distances=True)
    assert bool((idx == idx2).all())


def test_flash_knn_routes_to_trainium():
    """The dispatcher auto-selects the trainium backend on a Neuron host."""
    from flashlib.primitives.knn.impl import _route
    from flashlib import _hw
    assert _route(B=1, N=1000, D=64, k=16, hw=_hw.current()) == "trainium"
    # k > 64 and B > 1 fall back to the torch reference.
    assert _route(B=1, N=1000, D=64, k=128, hw=_hw.current()) == "torch"
