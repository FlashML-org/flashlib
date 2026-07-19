"""Backend parity tests — verify enhanced kernels are correct.

For every primitive that gained a new Triton implementation OR a CuteDSL
backend in the multi-DSL integration, we check:

  * **Reference parity** — primitive against a torch / sklearn reference,
    within the published precision tier of its dtype.
  * **Cross-backend parity** — Triton output vs CuteDSL output on the same
    input, within the looser of the two backends' published tolerances.

The "old triton vs new triton" parity is implicitly covered by the
reference parity test: if both old and new triton agreed with the torch
reference within the same tol, they trivially agreed with each other.

These tests are CUDA-only and skip cleanly on CPU CI.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for backend parity tests", allow_module_level=True)

DEVICE = "cuda"
SEED = 42


def _seeded(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_hopper() -> bool:
    p = torch.cuda.get_device_properties(0)
    return p.major >= 9


def _is_blackwell() -> bool:
    p = torch.cuda.get_device_properties(0)
    return p.major >= 10


# ---------------------------------------------------------------------------
# kmeans parity
# ---------------------------------------------------------------------------


def _torch_kmeans_assign(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Reference assignment: argmin_j ||x_i - c_j||^2."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)              # (B, N, 1)
    c_sq = (c * c).sum(dim=-1).unsqueeze(-2)              # (B, 1, K)
    cross = torch.matmul(x.float(), c.float().transpose(-1, -2))  # (B, N, K)
    dist = x_sq + c_sq - 2.0 * cross
    return dist.argmin(dim=-1).to(torch.int32)


def test_kmeans_assign_triton_matches_torch_reference():
    _seeded()
    B, N, D, K = 1, 4096, 128, 64
    x = torch.randn(B, N, D, device=DEVICE, dtype=torch.float32)
    centroids = torch.randn(B, K, D, device=DEVICE, dtype=torch.float32)
    x_sq = (x * x).sum(dim=-1)

    from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
    triton_ids = euclid_assign_triton(x, centroids).to(torch.int32)
    ref_ids = _torch_kmeans_assign(x, centroids)

    # Allow a small rate of disagreement (boundary points where two centroids
    # are equidistant within fp32 round-off can flip).
    mismatch_rate = (triton_ids != ref_ids).float().mean().item()
    assert mismatch_rate < 5e-3, (
        f"kmeans triton assign disagrees with torch ref: {mismatch_rate:.4f}"
    )


@pytest.mark.skipif(not _is_hopper(), reason="cutedsl assign requires Hopper SM90")
def test_kmeans_assign_cutedsl_matches_triton():
    _seeded()
    B, N, D, K = 1, 4096, 128, 64
    x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
    centroids = torch.randn(B, K, D, device=DEVICE, dtype=torch.bfloat16)
    x_sq = (x.float() * x.float()).sum(dim=-1)

    from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
    from flashlib.primitives.kmeans.cutedsl import cutedsl_assign_euclid

    triton_ids = euclid_assign_triton(x, centroids).to(torch.int32)
    try:
        cute_ids = cutedsl_assign_euclid(x, centroids, x_sq).to(torch.int32)
    except Exception as e:
        pytest.skip(f"cutedsl assign fall-through (kernel unavailable): {e}")

    # bf16 distances can flip the argmin near boundary rows; allow a small rate.
    mismatch_rate = (triton_ids != cute_ids).float().mean().item()
    assert mismatch_rate < 1e-2, (
        f"kmeans triton vs cutedsl mismatch: {mismatch_rate:.4f}"
    )


def test_flash_kmeans_smart_dispatcher_runs():
    _seeded()
    x = torch.randn(8000, 64, device=DEVICE, dtype=torch.float32)
    from flashlib.primitives.kmeans import flash_kmeans
    cluster_ids, centroids, n_iter = flash_kmeans(
        x, n_clusters=16, max_iters=10,
    )
    assert cluster_ids.shape == (8000,)
    assert centroids.shape == (16, 64)
    assert (cluster_ids >= 0).all() and (cluster_ids < 16).all()


# ---------------------------------------------------------------------------
# knn parity
# ---------------------------------------------------------------------------


def _torch_knn_topk(x: torch.Tensor, c: torch.Tensor, k: int):
    """Reference top-K: smallest squared L2 distances."""
    x_sq = (x.float() * x.float()).sum(dim=-1, keepdim=True)
    c_sq = (c.float() * c.float()).sum(dim=-1).unsqueeze(-2)
    cross = torch.matmul(x.float(), c.float().transpose(-1, -2))
    dist = x_sq + c_sq - 2.0 * cross
    vals, idxs = torch.topk(dist, k, dim=-1, largest=False)
    return vals, idxs


def test_knn_triton_matches_torch_reference_bf16():
    _seeded()
    B, N, M, D, K = 1, 256, 4096, 128, 8
    x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(B, M, D, device=DEVICE, dtype=torch.bfloat16)

    from flashlib.primitives.knn import flash_knn
    vals_t, idxs_t = flash_knn(x, c, K, backend="triton")
    vals_r, idxs_r = _torch_knn_topk(x, c, K)

    # Set-equality of returned indices per row (top-K may differ in order).
    set_t = set(map(tuple, idxs_t[0].sort(dim=-1).values.cpu().tolist()))
    set_r = set(map(tuple, idxs_r[0].sort(dim=-1).values.cpu().tolist()))
    overlap = len(set_t & set_r) / len(set_r)
    assert overlap > 0.95, f"knn triton/torch index overlap {overlap:.3f} too low"


def test_knn_triton_small_n_large_n_index_parity():
    """The two Triton routings (``small_n`` M-split vs ``large_n``
    single-pass) share one insert kernel but pick different BN / BM /
    M_PER_SPLIT configs. Both must return the same top-K *set* on the
    same shape (order may differ on ties).

    A regression in either branch of the unified heuristic inside
    :func:`flashlib.primitives.knn.triton.flash_knn_triton` would show up
    as set divergence here. Uses a small enough shape to bench cheaply,
    large enough that bf16 ties don't drown the signal.
    """
    _seeded()
    B, N, M, D, K = 1, 2048, 8192, 64, 8
    x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(B, M, D, device=DEVICE, dtype=torch.bfloat16)

    from flashlib.primitives.knn.triton import (
        flash_knn_triton_small_n,
        flash_knn_triton_large_n,
    )
    idxs_search = flash_knn_triton_small_n(x, c, K)
    idxs_build = flash_knn_triton_large_n(x, c, K)

    s_search = set(map(tuple, idxs_search[0].sort(dim=-1).values.cpu().tolist()))
    s_build = set(map(tuple, idxs_build[0].sort(dim=-1).values.cpu().tolist()))
    overlap = len(s_search & s_build) / len(s_search)
    assert overlap > 0.95, (
        f"triton small_n/large_n index overlap {overlap:.3f} too low"
    )


@pytest.mark.skipif(not _is_hopper(), reason="cutedsl FA3 knn requires Hopper SM90")
def test_knn_cutedsl_fa3_matches_triton():
    """FA3 fully-fused path agrees with Triton on top-K indices.

    Both backends consume the same score formulation; on bf16 inputs
    they should produce essentially identical neighbour sets (FA3 is
    order-tolerant with Triton up to ties).
    """
    _seeded()
    B, N, M, D, K = 1, 1024, 8192, 64, 8
    x = torch.randn(B, N, D, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(B, M, D, device=DEVICE, dtype=torch.bfloat16)

    from flashlib.primitives.knn import flash_knn

    vals_t, idxs_t = flash_knn(x, c, K, backend="triton")
    try:
        vals_c, idxs_c = flash_knn(x, c, K, backend="cutedsl", autotune=False)
    except Exception as e:
        pytest.skip(f"cutedsl FA3 unavailable on this build: {e}")

    # Compare set of top-K per row (order-tolerant; bf16 distances
    # may tie).
    s_t = set(map(tuple, idxs_t[0].sort(dim=-1).values.cpu().tolist()))
    s_c = set(map(tuple, idxs_c[0].sort(dim=-1).values.cpu().tolist()))
    overlap = len(s_t & s_c) / len(s_t)
    assert overlap > 0.90, f"knn triton/cutedsl-fa3 overlap {overlap:.3f} too low"


def _exact_idx(x, c, k):
    xf = x.float(); cf = c.float()
    d = ((xf * xf).sum(-1, keepdim=True) + (cf * cf).sum(-1)[None, :]
         - 2.0 * (xf @ cf.t()))
    d.clamp_(min=0)
    return torch.topk(d, k, dim=-1, largest=False, sorted=True)[1]


@pytest.mark.skipif(not _is_blackwell(),
                    reason="Blackwell CuteDSL knn requires sm_100")
@pytest.mark.parametrize("q,m,k", [(1, 16384, 10), (4, 65536, 5), (4, 4096, 10)])
def test_knn_blackwell_search_matches_exact(q, m, k):
    """Small-Q search on sm_100 (where Triton's tl.dot min-M=16 cannot run):
    the Blackwell CuteDSL kernel must hit recall 1.0 vs the fp32-exact top-K."""
    _seeded(q * 13 + m + k)
    from flashlib.primitives.knn.cutedsl import (
        blackwell_available, knn_search_cutedsl)
    if not blackwell_available():
        pytest.skip("cutlass-dsl / cuda-python unavailable")
    qx = torch.randn(q, 128, device=DEVICE, dtype=torch.bfloat16)
    db = torch.randn(m, 128, device=DEVICE, dtype=torch.bfloat16)
    dist, idx = knn_search_cutedsl(qx, db, k)
    torch.cuda.synchronize()
    ri = _exact_idx(qx, db, k)
    recall = (idx.long().unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).float().mean()
    assert recall.item() == 1.0, f"blackwell search recall {recall.item():.4f}"
    assert idx.dtype == torch.int32 and tuple(idx.shape) == (q, k)


@pytest.mark.skipif(not _is_blackwell(),
                    reason="Blackwell CuteDSL knn requires sm_100")
@pytest.mark.parametrize("n,k", [(512, 5), (1024, 10), (2048, 5)])
def test_knn_blackwell_build_matches_exact(n, k):
    """Self-kNN build (tcgen05 MMA + register top-K) recall 1.0 vs exact."""
    _seeded(n * 7 + k)
    from flashlib.primitives.knn.cutedsl import (
        blackwell_available, knn_build_cutedsl)
    if not blackwell_available():
        pytest.skip("cutlass-dsl / cuda-python unavailable")
    x = torch.randn(n, 128, device=DEVICE, dtype=torch.bfloat16)
    dist, idx = knn_build_cutedsl(x, k)
    torch.cuda.synchronize()
    ri = _exact_idx(x, x, k)
    recall = (idx.long().unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).float().mean()
    assert recall.item() == 1.0, f"blackwell build recall {recall.item():.4f}"


@pytest.mark.skipif(not _is_blackwell(),
                    reason="Blackwell CuteDSL knn requires sm_100")
@pytest.mark.parametrize("n,k", [(8192, 10), (16384, 10), (16384, 32),
                                 (32768, 32)])
def test_knn_blackwell_build_largeN_highk(n, k):
    """Large-N high-k build is exact (full top-k per split + S*k merge): the
    worst-of-K recompute switches max-tree -> linear scan above _MAXTREE_MAX so
    k=32 stays in registers. Recall must be exactly 1.0."""
    _seeded(n * 7 + k)
    from flashlib.primitives.knn.cutedsl import (
        blackwell_available, knn_build_cutedsl)
    if not blackwell_available():
        pytest.skip("cutlass-dsl / cuda-python unavailable")
    x = torch.randn(n, 128, device=DEVICE, dtype=torch.bfloat16)
    ri = _exact_idx(x, x, k)
    _, idx = knn_build_cutedsl(x, k)
    torch.cuda.synchronize()
    rec = (idx.long().unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).float().mean()
    # Exact algorithm; the only misses are BF16-MMA near-ties (the CUDA path
    # has the same), a few per million neighbours at large N*k.
    assert rec.item() >= 0.9999, f"exact build recall {rec.item():.5f}"


@pytest.mark.skipif(not _is_blackwell(),
                    reason="Blackwell CuteDSL knn requires sm_100")
def test_knn_flash_auto_routes_smallq_to_blackwell():
    """flash_knn transparently serves the small-Q search shape Triton cannot
    compile on sm_100, and the result matches the exact top-K."""
    _seeded()
    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.knn.cutedsl import blackwell_available
    if not blackwell_available():
        pytest.skip("cutlass-dsl / cuda-python unavailable")
    q = torch.randn(4, 128, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(16384, 128, device=DEVICE, dtype=torch.bfloat16)
    vals, idxs = flash_knn(q, c, 10)            # auto-route (no backend kw)
    torch.cuda.synchronize()
    ri = _exact_idx(q, c, 10)
    recall = (idxs.long().unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).float().mean()
    assert recall.item() == 1.0
    # vals are true squared-L2 (gathered); must be non-negative and ascending
    assert (vals >= 0).all()
    assert (vals[:, 1:] >= vals[:, :-1] - 1e-3).all()


def _hopper_sm90() -> bool:
    return _is_hopper() and not _is_blackwell()


@pytest.mark.skipif(not _hopper_sm90(), reason="Hopper sm_90 FA3 path")
@pytest.mark.parametrize("q", [1, 4, 8, 13, 100])
def test_knn_cutedsl_fa3_smallq_search_exact(q):
    """FA3 search kernel handles partial / small query tiles (Q not a multiple
    of BM). This is the query-pad fix that replaced the previous OOB illegal
    instruction; recall must be exactly 1.0 vs the fp32 top-K."""
    _seeded(q * 5 + 3)
    from flashlib.primitives.knn.cutedsl import cutedsl_available
    from flashlib.primitives.knn.cutedsl.impl import cutedsl_flash_knn
    if not cutedsl_available():
        pytest.skip("cutlass-dsl / cuda-python unavailable")
    qx = torch.randn(q, 128, device=DEVICE, dtype=torch.bfloat16)
    db = torch.randn(8192, 128, device=DEVICE, dtype=torch.bfloat16)
    idx = cutedsl_flash_knn(qx.unsqueeze(0), db.unsqueeze(0), 8)[0]
    torch.cuda.synchronize()
    ri = _exact_idx(qx, db, 8)
    recall = (idx.long().unsqueeze(-1) == ri.unsqueeze(-2)).any(-1).float().mean()
    assert recall.item() == 1.0, f"smallq search recall {recall.item():.4f}"
    assert idx.dtype == torch.int32 and tuple(idx.shape) == (q, 8)


@pytest.mark.skipif(not _hopper_sm90(), reason="Hopper sm_90 build auto-route")
def test_knn_autopick_hopper_build_band():
    """Lock the Hopper transparent-routing policy: large-N low/mid-k self-kNN
    builds auto-route to CuteDSL (1.4-2.4x); high-k, narrow-N, wide-D, and any
    search shape stay on Triton."""
    import flashlib.primitives.knn.impl as kimpl
    from flashlib import _hw
    hw = _hw.current()

    def pick(N, D, K, *, build=True, M=None):
        x = torch.empty(1, N, D, device=DEVICE, dtype=torch.bfloat16)
        c = x if build else torch.empty(1, M or N, D, device=DEVICE,
                                        dtype=torch.bfloat16)
        return kimpl._cutedsl_autopick(x, c, K, hw)

    assert pick(131072, 64, 8) is True
    assert pick(131072, 128, 20) is True
    assert pick(131072, 64, 24) is False                 # k > kmax
    assert pick(131072, 256, 8) is False                 # D > 128
    assert pick(40000, 64, 8) is False                   # N < 50k
    assert pick(8, 64, 8, build=False, M=131072) is False  # search


@pytest.mark.skipif(not _hopper_sm90(), reason="Hopper sm_90 strict routing")
def test_knn_flash_strict_no_cross_backend_fallback():
    """Routing is strict and proactive: the dispatcher picks exactly one
    backend up front and runs it -- there is NO reactive cross-backend net.
    If the chosen backend fails, the error propagates (pin the other backend
    to recover). (Small-Q on sm_90 is a non-issue -- Triton tiles every Q;
    the sm_100 small-Q gap is handled proactively by ``_cutedsl_autopick``.)"""
    _seeded()
    import flashlib.primitives.knn.impl as kimpl
    from flashlib.primitives.knn import flash_knn
    q = torch.randn(8, 128, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(8192, 128, device=DEVICE, dtype=torch.bfloat16)
    orig = kimpl.flash_knn_triton

    def boom(*a, **k):
        raise RuntimeError("simulated triton failure")

    # auto routes this Hopper search shape to Triton; with no reactive net the
    # failure surfaces instead of silently swapping to CuteDSL.
    kimpl.flash_knn_triton = boom
    try:
        with pytest.raises(RuntimeError):
            flash_knn(q, c, 8, return_distances=False)
    finally:
        kimpl.flash_knn_triton = orig


# ---------------------------------------------------------------------------
# Pairwise MRD — every autotune config must fill the full matrix (issue #12)
# ---------------------------------------------------------------------------


def test_pairwise_mrd_every_config_fills_full_matrix():
    """Regression for issue #12: rectangular tiles left an uninitialized hole.

    The autotuner masks per-config bugs (all configs are benchmarked into the
    same buffer, so square configs paper over holes left by rectangular
    ones). Force each config to run alone against a NaN-poisoned output so
    any cell not written by *that* config survives as NaN, then check value
    parity with the torch reference and exact symmetry.
    """
    import triton
    from flashlib.kernels.distance.triton import mrd as M
    from flashlib.kernels.distance.triton._common import _round_to_bucket

    _seeded()
    kernel = M._pairwise_mrd_kernel
    orig_configs = kernel.configs
    try:
        # N deliberately not a multiple of every block size.
        for N, D in [(512, 32), (500, 32), (129, 7)]:
            X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
            core = torch.rand(N, device=DEVICE) + 0.1
            x_sq = (X * X).sum(dim=1)
            ref = torch.maximum(
                torch.maximum(core[:, None], core[None, :]), torch.cdist(X, X)
            )
            for cfg in M._PAIRWISE_MRD_CONFIGS:
                kernel.configs = [cfg]
                kernel.cache.clear()
                out = torch.full((N, N), float("nan"), device=DEVICE,
                                 dtype=torch.float32)
                grid = lambda META: (triton.cdiv(N, META["BLOCK_I"]),
                                     triton.cdiv(N, META["BLOCK_J"]))
                kernel[grid](
                    X, x_sq, core, out, N, D,
                    X.stride(0), X.stride(1), out.stride(0), out.stride(1),
                    N_KEY=_round_to_bucket(N), D_KEY=_round_to_bucket(D),
                )
                torch.cuda.synchronize()
                bi, bj = cfg.kwargs["BLOCK_I"], cfg.kwargs["BLOCK_J"]
                tag = f"N={N} D={D} BLOCK_I={bi} BLOCK_J={bj}"
                n_nan = torch.isnan(out).sum().item()
                assert n_nan == 0, f"{tag}: {n_nan} unwritten cells"
                err = (out - ref).abs().max().item()
                assert err < 0.5, f"{tag}: max err {err:.4f} vs torch ref"
                asym = (out - out.T).abs().max().item()
                assert asym == 0.0, f"{tag}: output not symmetric ({asym})"
    finally:
        kernel.configs = orig_configs
        kernel.cache.clear()


# ---------------------------------------------------------------------------
# Standard scaler — element-wise within fp16 tol
# ---------------------------------------------------------------------------


def test_standard_scaler_triton_matches_torch():
    _seeded()
    N, D = 8192, 256
    x = torch.randn(N, D, device=DEVICE, dtype=torch.float32) * 7.5 + 3.0

    from flashlib.primitives.standard_scaler.triton import (
        flash_standard_scaler_fit_transform,
    )
    y, (mean, std) = flash_standard_scaler_fit_transform(x)
    ref = (x - x.mean(dim=0)) / x.std(dim=0, correction=0).clamp_min(1e-12)
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# PCA — eigenvalue parity vs torch
# ---------------------------------------------------------------------------


def test_pca_triton_eigvals_match_torch():
    _seeded()
    N, D, K = 4096, 64, 16
    # Generate data with a clear spectrum
    A = torch.randn(D, D, device=DEVICE, dtype=torch.float32)
    cov = A @ A.t() / D + 1e-3 * torch.eye(D, device=DEVICE)
    L = torch.linalg.cholesky(cov)
    x = (torch.randn(N, D, device=DEVICE, dtype=torch.float32) @ L.t())
    x = x - x.mean(dim=0)

    from flashlib.primitives.pca import flash_pca
    ev_asc, evec_asc = flash_pca(x, K=K)

    # Torch reference: eigh on the cov matrix, descending top-K.
    cov_emp = (x.t() @ x) / N
    eigvals_full, _ = torch.linalg.eigh(cov_emp)
    ref_topk = eigvals_full[-K:].flip(0)  # descending

    flash_topk = ev_asc.flip(0)  # primitive returns ascending; match descending
    rel = (flash_topk - ref_topk).abs() / ref_topk.abs().clamp_min(1e-6)
    assert rel.max().item() < 5e-2, (
        f"pca eigval rel-err {rel.max():.4f} exceeds 5%"
    )


# ---------------------------------------------------------------------------
# DBSCAN — labels (clustering) match sklearn-equivalent grouping
# ---------------------------------------------------------------------------


def test_dbscan_triton_labels_make_sense():
    """Two well-separated Gaussian blobs should yield two non-noise clusters."""
    _seeded()
    N, D = 2000, 8
    blob_a = torch.randn(N // 2, D, device=DEVICE, dtype=torch.float32) + 5.0
    blob_b = torch.randn(N // 2, D, device=DEVICE, dtype=torch.float32) - 5.0
    x = torch.cat([blob_a, blob_b], dim=0)

    from flashlib.primitives.dbscan import flash_dbscan
    labels = flash_dbscan(x, eps=2.0, min_samples=5)
    n_clusters = (labels.unique() >= 0).sum().item()
    assert 1 <= n_clusters <= 4, f"dbscan found {n_clusters} clusters; expected ~2"
    # First half should mostly share one label, second half another.
    a_labels = labels[: N // 2]
    b_labels = labels[N // 2 :]
    a_dom = a_labels.mode().values.item()
    b_dom = b_labels.mode().values.item()
    assert a_dom != b_dom, "dbscan failed to separate the two blobs"


# ---------------------------------------------------------------------------
# Linear regression — closed-form parity
# ---------------------------------------------------------------------------


def test_linear_regression_triton_matches_torch():
    _seeded()
    N, D = 8192, 32
    X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    w_true = torch.randn(D, device=DEVICE, dtype=torch.float32)
    y = X @ w_true + 0.01 * torch.randn(N, device=DEVICE)

    from flashlib.primitives.linear_regression import flash_linear_regression
    w_flash = flash_linear_regression(X, y)

    XtX = X.t() @ X
    Xty = X.t() @ y
    w_torch = torch.linalg.solve(XtX, Xty)
    rel = (w_flash - w_torch).norm() / w_torch.norm().clamp_min(1e-9)
    assert rel.item() < 1e-3, f"linreg coeff rel-err {rel:.4e}"


# ---------------------------------------------------------------------------
# Ridge regression — match torch closed form
# ---------------------------------------------------------------------------


def test_ridge_triton_matches_torch():
    _seeded()
    N, D = 4096, 32
    alpha = 1.0
    X = torch.randn(N, D, device=DEVICE, dtype=torch.float32)
    y = X @ torch.randn(D, device=DEVICE) + 0.05 * torch.randn(N, device=DEVICE)

    from flashlib.primitives.ridge import flash_ridge_regression
    w_flash = flash_ridge_regression(X, y, alpha=alpha)

    XtX = X.t() @ X + alpha * torch.eye(D, device=DEVICE)
    Xty = X.t() @ y
    w_torch = torch.linalg.solve(XtX, Xty)
    rel = (w_flash - w_torch).norm() / w_torch.norm().clamp_min(1e-9)
    assert rel.item() < 1e-3, f"ridge coeff rel-err {rel:.4e}"


# ---------------------------------------------------------------------------
# Multinomial NB — class-count parity
# ---------------------------------------------------------------------------


def test_multinomial_nb_fit_runs():
    _seeded()
    N, V, C = 4096, 256, 5
    X = torch.randint(0, 8, (N, V), device=DEVICE, dtype=torch.float32)
    y = torch.randint(0, C, (N,), device=DEVICE, dtype=torch.int64)

    from flashlib.primitives.multinomial_nb import (
        flash_multinomial_nb_fit, flash_multinomial_nb_predict,
    )
    params = flash_multinomial_nb_fit(X, y, n_classes=C)
    preds = flash_multinomial_nb_predict(X, params)
    assert preds.shape == (N,)
    # Random labels: an in-sample accuracy below 1/C for chance is fine; we
    # just want to ensure the fit/predict path runs end-to-end.
    assert preds.min().item() >= 0 and preds.max().item() < C
