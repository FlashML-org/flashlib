"""Blackwell tcgen05 KNN search kernel (BlackwellKnnSearchTC) tests.

Covers the kernel's whole envelope -- D in {64..512, %64}, k <= 16, arbitrary
Q/M including the padded tails (Q % 128 != 0 via TMA OOB zero-fill, M %
64 != 0 via the +INF-padded c_sq) -- plus the flash_knn auto-routing gates.

Skipped off-Blackwell or when cutlass-dsl is unavailable.
"""
from __future__ import annotations

import pytest
import torch

from flashlib import _hw

bw = pytest.importorskip(
    "flashlib.primitives.knn.cutedsl.blackwell_impl",
    reason="cutlass-dsl unavailable")

requires_blackwell = pytest.mark.skipif(
    not (torch.cuda.is_available() and _hw.current().is_blackwell
         and bw.blackwell_available()),
    reason="needs a Blackwell GPU + cutlass-dsl",
)


def _brute(q: torch.Tensor, db: torch.Tensor, k: int):
    d = (q.float() ** 2).sum(-1)[:, None] + (db.float() ** 2).sum(-1)[None, :] \
        - 2.0 * q.float() @ db.float().T
    return torch.topk(d.clamp_min(0), k, dim=-1, largest=False, sorted=True)


def _assert_topk(q, db, dist, idx, k, atol=1e-2):
    """Tie-tolerant: every returned index must be within atol of the true
    kth distance, and the returned distance must match the exact one."""
    ref_vals, _ = _brute(q, db, k)
    kth = ref_vals[:, -1:]
    chosen = ((q.float()[:, None, :] - db.float()[idx.long()]) ** 2).sum(-1)
    assert bool((idx >= 0).all()), "negative index returned"
    assert bool((idx < db.shape[0]).all()), "index out of range"
    within = chosen <= kth + kth.abs() * 1e-3 + atol
    assert bool(within.all()), f"recall {within.float().mean():.4f} < 1"
    assert (dist - chosen).abs().max().item() <= atol


@requires_blackwell
@pytest.mark.parametrize("Q,M,D,k", [
    (4096, 65536, 128, 10),     # large-Q core shape
    (128, 8192, 128, 10),       # small everything
    (100, 100000, 128, 10),     # Q tail (100 % 128 != 0)
    (4097, 65537, 128, 10),     # Q and M tails together
    (1024, 131072, 256, 10),    # D=256
    (256, 65536, 384, 16),      # D=384, k at the gate
    (512, 131072, 512, 10),     # D at SEARCH_TC_DMAX
    (333, 55555, 192, 7),       # everything irregular
    (2048, 65536, 64, 10),      # D=64 (single k-tile)
    (16, 262144, 128, 1),       # k=1, deep splits
])
def test_search_tc_exact(Q, M, D, k):
    torch.manual_seed(0)
    q = torch.randn(Q, D, device="cuda", dtype=torch.bfloat16)
    db = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    dist, idx = bw.knn_search_tc(q, db, k)
    torch.cuda.synchronize()
    _assert_topk(q, db, dist, idx, k)


@requires_blackwell
@pytest.mark.parametrize("S", [1, 2, 3, 7, 37, 296])
def test_search_tc_split_counts(S):
    """Range-based splits must cover every db tile for any S (incl. odd
    tile counts per split and S not dividing the tile count)."""
    torch.manual_seed(1)
    Q, M, D, k = 256, 99777, 128, 10   # 1560 tiles - prime-ish, has M tail
    q = torch.randn(Q, D, device="cuda", dtype=torch.bfloat16)
    db = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    dist, idx = bw.knn_search_tc(q, db, k, num_splits=S)
    torch.cuda.synchronize()
    _assert_topk(q, db, dist, idx, k)


@requires_blackwell
def test_search_tc_matches_triton_lane():
    """tc lane and the Triton lane must agree through the public API."""
    torch.manual_seed(2)
    q = torch.randn(1, 2048, 128, device="cuda", dtype=torch.bfloat16)
    db = torch.randn(1, 65536, 128, device="cuda", dtype=torch.bfloat16)
    v_auto, i_auto = __import__("flashlib").flash_knn(q, db, k=10)
    v_tri, i_tri = __import__("flashlib").flash_knn(q, db, k=10,
                                                    backend="triton")
    # distances must agree to bf16 noise; indices may swap only across ties
    assert (v_auto - v_tri).abs().max().item() < 1e-2


@requires_blackwell
def test_search_tc_supported_gates():
    assert bw.search_tc_supported(128, 10)
    assert bw.search_tc_supported(512, 16)
    assert not bw.search_tc_supported(512, 17)      # k gate
    assert not bw.search_tc_supported(576, 10)      # D gate
    assert not bw.search_tc_supported(100, 10)      # D % 64
    assert not bw.search_tc_supported(32, 10)       # D < 64


@requires_blackwell
def test_autopick_routes_search_to_cutedsl():
    """The dispatcher's auto path must pick CuteDSL for the tc envelope and
    Triton outside it."""
    from flashlib.primitives.knn.impl import _cutedsl_autopick
    hw = _hw.current()
    mk = lambda Q, M, D: (torch.randn(1, Q, D, device="cuda",
                                      dtype=torch.bfloat16),
                          torch.randn(1, M, D, device="cuda",
                                      dtype=torch.bfloat16))
    x, c = mk(2048, 65536, 256)
    assert _cutedsl_autopick(x, c, 10, hw)
    x, c = mk(2048, 65536, 100)            # D not a multiple of 64
    assert not _cutedsl_autopick(x, c, 10, hw)
    x, c = mk(2048, 65536, 128)
    assert not _cutedsl_autopick(x, c, 32, hw)   # k > 16 stays Triton
    x, c = mk(1000, 2000, 128)
    assert not _cutedsl_autopick(x, c, 10, hw)   # tiny corpus stays Triton
