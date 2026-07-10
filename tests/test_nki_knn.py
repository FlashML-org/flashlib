"""Trainium (NKI) flash-knn parity tests.

Run on an AWS NeuronDevice host (Trainium/Inferentia); skipped everywhere
else. They exercise the cached XLA/NKI rank + fp32 distance epilogue against a
torch brute-force reference, including non-tile shapes, ties, LNC1/LNC2, and
device-resident I/O. NKI compiles per shape, so sizes are kept modest.
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
    from flashlib.primitives.knn.nki import nki_knn_available
    if not nki_knn_available():
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
    idx = np.asarray(idx)
    ref = np.asarray(ref)
    return float(np.mean([len(set(idx[i]) & set(ref[i])) / idx.shape[1]
                          for i in range(idx.shape[0])]))


@pytest.mark.parametrize("Nq,M,D,k", [
    (300, 2000, 64, 10),     # search-ish: modest Q, mid corpus
    (1024, 4096, 128, 32),   # build-ish: large Q, larger k
    (64, 512, 16, 8),        # small D (padded to 128 partition internally)
    (96, 777, 256, 17),      # non-tile corpus and non-8-multiple k
    (32, 513, 512, 9),       # maximum supported D + padding boundary
])
def test_nki_knn_matches_torch_reference(Nq, M, D, k):
    torch.manual_seed(SEED)
    q = torch.randn(Nq, D)
    c = torch.randn(M, D)

    from flashlib.primitives.knn.nki import nki_knn
    vals, idx = nki_knn(q, c, k)
    ref_vals, ref_idx = _torch_topk(q, c, k)

    # TF32 ranking is exhaustive; the large-k bf16 score variant oversamples
    # and fp32-reranks candidates to preserve this recall floor.
    rec = _recall(idx.cpu().numpy(), ref_idx.cpu().numpy())
    assert rec >= 0.99, f"NKI KNN recall@{k} = {rec:.4f}"

    # Distances are true fp32 squared-L2 (gathered), ascending.
    v = vals.cpu().numpy()
    assert np.all(np.diff(v, axis=1) >= -1e-3), "distances not ascending"
    # Returned vals equal the exact distance at the returned indices.
    d2 = ((q[:, None, :] - c[None]) ** 2).sum(-1).cpu().numpy()
    true = np.take_along_axis(d2, idx.cpu().numpy().astype(np.int64), axis=1)
    assert np.abs(v - true).max() < 1e-2


def test_nki_knn_indices_only():
    torch.manual_seed(SEED)
    q = torch.randn(200, 32)
    c = torch.randn(1500, 32)
    from flashlib.primitives.knn.nki import nki_knn
    idx = nki_knn(q, c, 8, return_distances=False)
    assert idx.shape == (200, 8) and idx.dtype == torch.int32
    vals, idx2 = nki_knn(q, c, 8, return_distances=True)
    assert bool((idx == idx2).all())


def test_nki_knn_duplicate_ties_are_index_ordered():
    torch.manual_seed(SEED)
    D, M, k = 64, 512, 8
    q = torch.zeros(8, D)
    c = torch.randn(M, D) + 4.0
    c[0].zero_()
    c[1].zero_()  # exact duplicate / exact-distance tie

    from flashlib.primitives.knn.nki import nki_knn
    vals, idx = nki_knn(q, c, k)
    assert bool((idx[:, 0] == 0).all() and (idx[:, 1] == 1).all())
    ties = vals[:, 1:] == vals[:, :-1]
    assert bool((~ties | (idx[:, 1:] >= idx[:, :-1])).all())


def test_nki_knn_xla_resident_and_lnc_consistent():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.knn.nki.knn import _nki_knn_xla

    torch.manual_seed(SEED)
    q = torch.randn(256, 64)
    c = torch.randn(1024, 64)
    vals1, idx1 = _nki_knn_xla(
        q, c, 10, return_distances=True, n_shards=1)
    vals2, idx2 = _nki_knn_xla(
        q, c, 10, return_distances=True, n_shards=2)
    assert torch.equal(idx1, idx2)
    assert torch.allclose(vals1, vals2, atol=1e-5, rtol=0)

    dev = xm.xla_device()
    vals_d, idx_d = _nki_knn_xla(
        q.to(dev), c.to(dev), 10, return_distances=True, n_shards=2)
    assert vals_d.device.type == "xla" and idx_d.device.type == "xla"
    assert torch.equal(idx_d.cpu(), idx2)


def test_flash_knn_routes_to_nki():
    """The dispatcher auto-selects the NKI backend on a Neuron host."""
    from flashlib.primitives.knn.impl import _route
    from flashlib import _hw
    assert _route(B=1, N=1000, D=64, k=16, hw=_hw.current()) == "nki"
    # k > 64 and B > 1 fall back to the torch reference.
    assert _route(B=1, N=1000, D=64, k=128, hw=_hw.current()) == "torch"
    with pytest.raises(ValueError, match="backend must be one of"):
        _route(
            B=1, N=1000, D=64, k=16,
            backend="trainium", hw=_hw.current())
