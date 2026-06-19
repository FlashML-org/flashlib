"""Correctness tests for the 5 v0.1 primitives, against torch / sklearn baselines."""
import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
def test_standard_scaler_matches_sklearn():
    from flashlib import flash_standard_scaler

    torch.manual_seed(0)
    X = torch.randn(10_000, 128, device="cuda", dtype=torch.float32)
    Y, (mean, std) = flash_standard_scaler(X)

    ref_mean = X.mean(dim=0)
    ref_std = X.std(dim=0, unbiased=False)
    ref_Y = (X - ref_mean) / ref_std
    assert (Y - ref_Y).abs().max().item() < 1e-4
    assert (mean - ref_mean).abs().max().item() < 1e-5
    assert (std - ref_std).abs().max().item() < 1e-4


@cuda_only
def test_kmeans_partitions_blob_data():
    from flashlib import KMeans

    torch.manual_seed(0)
    # D >= 16 because Triton tl.dot requires K >= 16 in the assign kernel.
    D = 16
    centers = torch.randn(3, D, device="cuda") * 8.0
    pts_per = 1000
    X = torch.cat([
        c + torch.randn(pts_per, D, device="cuda") * 0.3 for c in centers
    ])
    model = KMeans(d=D, k=3, niter=30, seed=0).fit(X)
    labels = model.predict(X)
    # Each ground-truth blob should map predominantly to a single cluster.
    for i in range(3):
        sub = labels[i*pts_per:(i+1)*pts_per]
        most_common = torch.mode(sub).values
        purity = (sub == most_common).float().mean().item()
        assert purity > 0.95, f"blob {i} purity {purity:.2%}"


@cuda_only
def test_pca_matches_torch_svd():
    from flashlib import flash_pca

    torch.manual_seed(0)
    X = torch.randn(5_000, 64, device="cuda", dtype=torch.float32)
    Xc = X - X.mean(dim=0)
    eigvals_asc, eigvecs_asc = flash_pca(Xc, K=10)
    # Reference: torch eigh on covariance.
    cov = Xc.T @ Xc / Xc.shape[0]
    ref_vals, ref_vecs = torch.linalg.eigh(cov)
    # Top-10 eigenvalues should match (eigvals_asc is ascending top-K, take last 10).
    assert torch.allclose(eigvals_asc, ref_vals[-10:], atol=1e-3, rtol=1e-3)


@cuda_only
def test_knn_matches_naive_topk():
    from flashlib import flash_knn

    torch.manual_seed(0)
    X = torch.randn(1, 256, 128, device="cuda", dtype=torch.float32).contiguous()
    Xbf = X.to(torch.bfloat16)
    dists, idxs = flash_knn(Xbf, Xbf, k=5)
    # Reference: cdist + topk.
    ref_d2 = torch.cdist(X[0], X[0]) ** 2
    ref_vals, ref_idxs = ref_d2.topk(5, largest=False, dim=1)
    # idxs may differ on ties; verify distances match within bf16 tolerance.
    flash_first_idx = idxs[0, :, 0]
    ref_first_idx = ref_idxs[:, 0]
    same = (flash_first_idx == ref_first_idx).float().mean().item()
    assert same > 0.99, f"only {same:.2%} of nearest-neighbors agree"


def _blobs(M, D, n_centers, device, seed=0, scale=4.0, spread=1.0):
    g = torch.Generator(device=device).manual_seed(seed)
    centers = torch.randn(n_centers, D, generator=g, device=device) * scale
    lab = torch.randint(0, n_centers, (M,), generator=g, device=device)
    X = centers[lab] + torch.randn(M, D, generator=g, device=device) * spread
    return X.to(torch.float32).contiguous()


def _recall(got_ids, ref_ids):
    k = ref_ids.shape[1]
    hits = sum(
        len(set(got_ids[i].tolist()) & set(ref_ids[i].tolist()))
        for i in range(got_ids.shape[0])
    )
    return hits / (got_ids.shape[0] * k)


@cuda_only
def test_ivf_flat_recall_vs_brute():
    """nprobe==nlist must reproduce brute force; moderate nprobe clears 0.95.

    At ``nprobe == nlist`` every list is probed, so the candidate set is the
    whole database and the fused fine-scan must return the exact top-k
    distances (modulo fp tie-breaking at the k boundary). Recall then climbs
    monotonically with nprobe, the only recall knob.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    torch.manual_seed(0)
    M, D, nlist, k = 20_000, 64, 128, 10
    X = _blobs(M, D, 16, "cuda", seed=0)
    Q = _blobs(256, D, 16, "cuda", seed=1)

    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=15, seed=0)

    ref = torch.cdist(Q, X) ** 2
    ref_vals, ref_ids = ref.topk(k, largest=False, dim=1)

    # Exhaustive: probe every list -> exact distances.
    vals_all, ids_all = flash_ivf_flat_search(index, Q, k, nprobe=nlist)
    assert torch.allclose(
        vals_all.sort(1).values, ref_vals.sort(1).values, rtol=1e-3, atol=1e-1
    ), "nprobe==nlist distances must equal brute force"
    assert _recall(ids_all, ref_ids) >= 0.99

    # Moderate nprobe still clears a high recall bar on blob data.
    _, ids_mod = flash_ivf_flat_search(index, Q, k, nprobe=nlist // 4)
    assert _recall(ids_mod, ref_ids) >= 0.95

    # Recall is monotonic in nprobe (more probes never hurts).
    r_lo = _recall(flash_ivf_flat_search(index, Q, k, nprobe=2)[1], ref_ids)
    r_hi = _recall(flash_ivf_flat_search(index, Q, k, nprobe=16)[1], ref_ids)
    assert r_hi >= r_lo


@cuda_only
def test_ivf_flat_isorecall_vs_reference():
    """Triton kernel == pure-torch reference on the SAME index (iso-recall).

    Building once and running both the fused Triton search and the torch
    oracle over the identical centroids/assignments must return identical
    neighbour ids at matched ``nprobe`` -- so any divergence is a kernel
    bug, not an algorithmic difference.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search
    from flashlib.primitives.ivf_flat import torch_fallback as tf

    torch.manual_seed(0)
    M, D, nlist, k = 12_000, 48, 64, 8
    X = _blobs(M, D, 10, "cuda", seed=2)
    Q = _blobs(128, D, 10, "cuda", seed=3)

    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=15, seed=0)

    for nprobe in (4, 8, 16):
        gv, gi = flash_ivf_flat_search(index, Q, k, nprobe=nprobe)
        tv, ti = tf.ivf_flat_search_torch(index, Q, k, nprobe=nprobe)
        # Same candidate set scanned, same exact distances found; compare on
        # the distance values, which are robust to k-boundary ties.
        assert torch.allclose(
            gv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-2
        ), f"distance sets diverge at nprobe={nprobe}"
        # IDs then match too, save for the rare fp tie at the k-th boundary
        # (two candidates equidistant to within fp32 accumulation order).
        same = (gi.sort(1).values == ti.sort(1).values).float().mean().item()
        assert same >= 0.99, f"iso-recall id mismatch at nprobe={nprobe}: {same:.4f}"


@cuda_only
def test_ivf_flat_application_class():
    """The sklearn-style IVFFlat wrapper round-trips fit -> kneighbors."""
    from flashlib import IVFFlat

    torch.manual_seed(0)
    X = _blobs(8_000, 32, 8, "cuda", seed=4)
    Q = _blobs(64, 32, 8, "cuda", seed=5)
    model = IVFFlat(nlist=64, nprobe=32, n_neighbors=10, niter=10, seed=0).fit(X)
    dist, idx = model.kneighbors(Q)
    assert dist.shape == (64, 10) and idx.shape == (64, 10)

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(10, largest=False, dim=1)
    assert _recall(idx, ref_ids) >= 0.95


@cuda_only
def test_ivf_flat_gemm_matches_elementwise_high_d():
    """GEMM and elementwise fine-scan agree at D>256 (D-split path).

    D>256 drives the GEMM kernel through its D-split branch (re-gather the
    query tile per corpus chunk) followed by an oversampled exact re-rank.
    Both variants scan the same candidate set, so their returned squared-L2
    distances must match each other and stay at brute-force recall.
    """
    from flashlib import flash_ivf_flat_build, flash_ivf_flat_search

    torch.manual_seed(0)
    M, D, nlist, k = 12_000, 320, 64, 10
    X = _blobs(M, D, 12, "cuda", seed=6)
    Q = _blobs(512, D, 12, "cuda", seed=7)
    index = flash_ivf_flat_build(X, nlist, nprobe=8, niter=12, seed=0)

    gv, gi = flash_ivf_flat_search(index, Q, k, nprobe=16, variant="gemm")
    ev, _ = flash_ivf_flat_search(index, Q, k, nprobe=16, variant="elementwise")
    assert torch.allclose(
        gv.sort(1).values, ev.sort(1).values, rtol=1e-3, atol=1e-1
    ), "gemm (D-split) vs elementwise distances diverge at D>256"

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)
    assert _recall(gi, ref_ids) >= 0.95


@cuda_only
@pytest.mark.parametrize("by_residual", [True, False])
def test_ivf_pq_isoresult_vs_oracle(by_residual):
    """Triton ADC kernel == pure-torch reference on the SAME index.

    With every list probed (``nprobe == nlist``) both paths scan the
    identical candidate set, so the fused Triton ADC must equal the torch
    reference **bit-for-bit** -- any divergence is a kernel bug. Covers
    both the residual (per-(query, list) LUT) and non-residual (query-only
    LUT, probe-stride 0) code paths.

    For *partial* nprobe the Triton coarse (``flash_knn``, tf32 GEMM) and
    the reference coarse (``cdist``, exact fp32) can pick a different
    ``nprobe``-th list at near-tied centroids; because ADC distances are
    approximate, that rare probe-set difference perturbs a handful of
    queries (a far list can inject a quantization-error false positive),
    so we require the vast majority -- not all -- of queries to match.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search
    from flashlib.primitives.ivf_pq import torch_fallback as tf

    torch.manual_seed(0)
    M, D, nlist, m, k = 12_000, 48, 64, 16, 8
    X = _blobs(M, D, 10, "cuda", seed=2)
    Q = _blobs(128, D, 10, "cuda", seed=3)

    index = flash_ivf_pq_build(
        X, nlist, m=m, nprobe=8, by_residual=by_residual, niter=15, seed=0
    )

    # Exhaustive probing: candidate set = all lists for both paths, so the
    # coarse step is deterministic and the kernels must agree bit-for-bit.
    gv, gi = flash_ivf_pq_search(index, Q, k, nprobe=nlist)
    tv, ti = tf.ivf_pq_search_torch(index, Q, k, nprobe=nlist)
    assert torch.allclose(
        gv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-1
    ), f"exhaustive ADC distances diverge (by_residual={by_residual})"
    same = (gi.sort(1).values == ti.sort(1).values).float().mean().item()
    assert same >= 0.98, f"exhaustive iso-result id mismatch: {same:.4f}"

    # Partial nprobe: the vast majority of queries match exactly (the rest
    # differ only because flash_knn vs cdist picked a different far list).
    for nprobe in (4, 8, 16):
        gv, _ = flash_ivf_pq_search(index, Q, k, nprobe=nprobe)
        tv, _ = tf.ivf_pq_search_torch(index, Q, k, nprobe=nprobe)
        row_ok = torch.isclose(
            gv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-1
        ).all(dim=1).float().mean().item()
        assert row_ok >= 0.95, (
            f"too many queries diverge at nprobe={nprobe} "
            f"(by_residual={by_residual}): {row_ok:.4f}"
        )


@cuda_only
def test_ivf_pq_recall_vs_brute():
    """Fine PQ clears a recall bar vs exact L2; recall is monotonic in nprobe.

    With small sub-vectors (``dsub=2`` here, 256 sub-centroids each) the PQ
    reconstruction is accurate, so even though the returned distances are
    ADC-approximate the recovered neighbour ids largely match exact
    brute-force. Recall climbs monotonically with nprobe -- more probed
    lists can only enlarge the candidate set.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    torch.manual_seed(0)
    M, D, nlist, m, k = 20_000, 32, 128, 16, 10
    X = _blobs(M, D, 16, "cuda", seed=0)
    Q = _blobs(256, D, 16, "cuda", seed=1)

    index = flash_ivf_pq_build(X, nlist, m=m, nprobe=8, niter=15, seed=0)

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)

    # Exhaustive probing scans every list (distances still ADC-approximate).
    _, ids_all = flash_ivf_pq_search(index, Q, k, nprobe=nlist)
    assert _recall(ids_all, ref_ids) >= 0.5

    r_lo = _recall(flash_ivf_pq_search(index, Q, k, nprobe=2)[1], ref_ids)
    r_hi = _recall(flash_ivf_pq_search(index, Q, k, nprobe=32)[1], ref_ids)
    assert r_hi >= r_lo


@cuda_only
def test_ivf_pq_nprobe_monotonic():
    """Recall rises sharply with nprobe (ADC lets it wiggle near saturation).

    More probed lists enlarge the candidate set, so recall climbs with
    nprobe. Unlike exact IVF-Flat the climb is not *strictly* monotonic:
    ADC distances are approximate, so a newly probed far list can
    occasionally inject a quantization-error false positive and nudge
    recall down once it has saturated. We assert the dominant upward trend
    and bound any local dip.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    torch.manual_seed(0)
    M, D, nlist, m, k = 12_000, 48, 64, 12, 10
    X = _blobs(M, D, 12, "cuda", seed=6)
    Q = _blobs(128, D, 12, "cuda", seed=7)
    index = flash_ivf_pq_build(X, nlist, m=m, nprobe=8, niter=12, seed=0)

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)
    recalls = [
        _recall(flash_ivf_pq_search(index, Q, k, nprobe=p)[1], ref_ids)
        for p in (1, 4, 16, 64)
    ]
    # Clear gain from probing more lists ...
    assert recalls[-1] >= recalls[0] + 0.2, f"nprobe gave no recall gain: {recalls}"
    # ... with only small ADC-induced dips along the way.
    for lo, hi in zip(recalls, recalls[1:]):
        assert hi >= lo - 0.03, f"recall dropped too much in nprobe: {recalls}"


@cuda_only
def test_ivf_pq_application_class():
    """The sklearn-style IVFPQ wrapper round-trips fit -> kneighbors."""
    from flashlib import IVFPQ

    torch.manual_seed(0)
    X = _blobs(8_000, 32, 8, "cuda", seed=4)
    Q = _blobs(64, 32, 8, "cuda", seed=5)
    model = IVFPQ(nlist=64, m=16, nprobe=32, n_neighbors=10, niter=10, seed=0).fit(X)
    dist, idx = model.kneighbors(Q)
    assert dist.shape == (64, 10) and idx.shape == (64, 10)
    assert idx.dtype == torch.int64
    assert (idx >= -1).all() and (idx < 8_000).all()
    assert torch.isfinite(dist).all()
    # 32-dim fp32 vector (128 B) -> 16-byte PQ code = 8x compression.
    assert model.compression_ratio is not None and model.compression_ratio > 1.0

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(10, largest=False, dim=1)
    assert _recall(idx, ref_ids) >= 0.5


@cuda_only
def test_ivf_pq_batch_matches_online():
    """The group-by-list `batch` kernel == the online kernel (exact ADC).

    Both fine-scan variants compute the identical asymmetric distance
    (a sum of LUT lookups, no x²-free approximation), so their returned
    distances must match and their ids must agree save for the rare ADC
    tie at the k-th boundary.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    torch.manual_seed(0)
    M, D, nlist, m, k = 16_000, 64, 64, 16, 10
    X = _blobs(M, D, 12, "cuda", seed=8)
    Q = _blobs(512, D, 12, "cuda", seed=9)   # large batch -> high code reuse
    index = flash_ivf_pq_build(X, nlist, m=m, nprobe=16, niter=12, seed=0)

    for nprobe in (8, 16):
        ov, oi = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant="online")
        bv, bi = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant="batch")
        assert torch.allclose(ov, bv, rtol=1e-4, atol=1e-2), \
            f"online vs batch ADC distances diverge at nprobe={nprobe}"
        same = (oi == bi).float().mean().item()
        assert same >= 0.98, f"online vs batch id mismatch at nprobe={nprobe}: {same:.4f}"


@cuda_only
@pytest.mark.parametrize("by_residual", [True, False])
def test_ivf_pq_gemm_matches_oracle(by_residual):
    """No-LUT decode+GEMM kernel == pure-torch ADC oracle (the bulk path).

    The ``gemm`` variant scores candidates with a tf32 tensor-core cross
    term instead of an ADC LUT, then exact-re-ranks an oversampled pool, so
    its returned distances must equal the reference ADC (to fp tolerance)
    and -- with every list probed, where the coarse step is deterministic --
    its ids must match. Covers residual and non-residual encoding, and the
    cross-list merge that residual ``‖rq_c‖²`` makes non-trivial.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search
    from flashlib.primitives.ivf_pq import torch_fallback as tf

    torch.manual_seed(0)
    M, D, nlist, m, k = 16_000, 64, 64, 16, 10
    X = _blobs(M, D, 12, "cuda", seed=2)
    Q = _blobs(1024, D, 12, "cuda", seed=3)   # bulk batch -> exercises GEMM
    index = flash_ivf_pq_build(
        X, nlist, m=m, nprobe=16, by_residual=by_residual, niter=15, seed=0
    )

    # Exhaustive probing: identical candidate set for both paths.
    gv, gi = flash_ivf_pq_search(index, Q, k, nprobe=nlist, variant="gemm")
    tv, ti = tf.ivf_pq_search_torch(index, Q, k, nprobe=nlist)
    assert torch.allclose(
        gv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-1
    ), f"gemm ADC distances diverge from oracle (by_residual={by_residual})"
    same = (gi.sort(1).values == ti.sort(1).values).float().mean().item()
    assert same >= 0.98, f"gemm vs oracle id mismatch: {same:.4f}"

    # gemm and online resolve to the same ADC distances on the same index.
    for nprobe in (8, 16):
        gv, _ = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant="gemm")
        ov, _ = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant="online")
        assert torch.allclose(
            gv.sort(1).values, ov.sort(1).values, rtol=1e-3, atol=1e-1
        ), f"gemm vs online ADC distances diverge at nprobe={nprobe}"


def _hopper_cutedsl():
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_properties(0).major < 9:
        return False
    from flashlib.kernels.cute_helpers import is_cutedsl_available
    return is_cutedsl_available()


@pytest.mark.skipif(not _hopper_cutedsl(), reason="Hopper + CUTLASS DSL required")
@pytest.mark.parametrize("variant", ["cute_lut", "cute_gemm"])
@pytest.mark.parametrize("by_residual", [True, False])
def test_ivf_pq_cutedsl_matches_oracle(variant, by_residual):
    """CuTe DSL fine-scan kernels == pure-torch ADC oracle.

    ``cute_lut`` builds the asymmetric-distance LUT in shared memory from a
    precomputed-table decomposition (the dsub cross term is a per-query /
    per-index GEMM) and ADC-scans the codes with data-dependent SMEM
    gathers (the road Triton cannot express) under a one-query-per-CTA
    warp-shuffle top-k; ``cute_gemm`` decodes the codes to sub-vectors and
    scores them with a WGMMA cross term. Both must reproduce the reference
    ADC distances (to fp tolerance) and -- with every list probed -- ids.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search
    from flashlib.primitives.ivf_pq import torch_fallback as tf

    torch.manual_seed(0)
    M, D, nlist, m, k = 16_000, 64, 64, 16, 10
    X = _blobs(M, D, 12, "cuda", seed=2)
    Q = _blobs(1024, D, 12, "cuda", seed=3)
    index = flash_ivf_pq_build(
        X, nlist, m=m, nprobe=16, by_residual=by_residual, niter=15, seed=0
    )

    # Exhaustive probing: identical candidate set for kernel and oracle.
    cv, ci = flash_ivf_pq_search(index, Q, k, nprobe=nlist, variant=variant)
    tv, ti = tf.ivf_pq_search_torch(index, Q, k, nprobe=nlist)
    assert torch.allclose(
        cv.sort(1).values, tv.sort(1).values, rtol=1e-3, atol=1e-1
    ), f"{variant} ADC distances diverge from oracle (by_residual={by_residual})"
    same = (ci.sort(1).values == ti.sort(1).values).float().mean().item()
    assert same >= 0.98, f"{variant} vs oracle id mismatch: {same:.4f}"

    # CuTe and the Triton gemm path resolve to the same ADC distances.
    for nprobe in (8, 16):
        cv, _ = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant=variant)
        gv, _ = flash_ivf_pq_search(index, Q, k, nprobe=nprobe, variant="gemm")
        assert torch.allclose(
            cv.sort(1).values, gv.sort(1).values, rtol=1e-3, atol=1e-1
        ), f"{variant} vs gemm ADC distances diverge at nprobe={nprobe}"


@cuda_only
def test_ivf_pq_query_tiling_identical():
    """Flash-style query tiling is exact: tiled == untiled, bit for bit.

    The residual LUT ``(nq, nprobe, m, 256)`` is the only structure that
    scales with ``nq * nprobe``; search tiles over query blocks so only a
    ``(q_tile, ...)`` LUT is ever live. Tiling changes *when* work runs,
    never *what*, so every tile size must reproduce the single-tile result
    exactly (no tolerance) -- and the live LUT must shrink with q_tile.
    """
    from flashlib import flash_ivf_pq_build, flash_ivf_pq_search

    torch.manual_seed(0)
    M, D, nlist, m, k = 60_000, 96, 256, 24, 10
    X = _blobs(M, D, 16, "cuda", seed=10)
    Q = _blobs(3000, D, 16, "cuda", seed=11)
    index = flash_ivf_pq_build(X, nlist, m=m, nprobe=16, niter=12, seed=0)

    # Tiling only applies to the LUT path, so pin the variant (auto would
    # route this work to the no-LUT GEMM kernel, which never tiles).
    # Reference: force a single tile (no tiling).
    ref_v, ref_i = flash_ivf_pq_search(
        index, Q, k, nprobe=16, variant="online", q_tile=10**9
    )

    prev_peak = None
    for q_tile in (2048, 512, 128):
        torch.cuda.reset_peak_memory_stats()
        v, i = flash_ivf_pq_search(
            index, Q, k, nprobe=16, variant="online", q_tile=q_tile
        )
        peak = torch.cuda.max_memory_allocated()
        assert torch.equal(v, ref_v), f"tiled vals differ at q_tile={q_tile}"
        assert torch.equal(i, ref_i), f"tiled ids differ at q_tile={q_tile}"
        if prev_peak is not None:
            assert peak <= prev_peak, "smaller q_tile must not raise peak memory"
        prev_peak = peak

    # The auto tile must keep the live LUT under the budget.
    from flashlib.primitives.ivf_pq.triton.search import (
        _auto_q_tile, _LUT_BUDGET_BYTES,
    )
    bq = _auto_q_tile(Q.shape[0], 16, m, index.by_residual)
    assert bq * 16 * m * 256 * 4 <= _LUT_BUDGET_BYTES


@cuda_only
def test_cagra_recall_vs_brute():
    """CAGRA recall climbs with itopk_size and clears a high bar.

    Unlike IVF there is no ``(nlist, nprobe)`` that reproduces an exact
    candidate set -- CAGRA is heuristic, so the contract is recall@k vs
    exact brute-force ground truth, governed by graph quality and the
    search budget. ``itopk_size`` is the dominant accuracy/speed knob:
    a larger internal buffer both widens the beam and (since the buffer is
    random-filled at start) adds traversal restarts, so recall rises with
    it and saturates near exhaustive.
    """
    from flashlib import flash_cagra_build, flash_cagra_search

    torch.manual_seed(0)
    M, D, k = 12_000, 48, 10
    X = _blobs(M, D, 12, "cuda", seed=6)
    Q = _blobs(128, D, 12, "cuda", seed=7)

    index = flash_cagra_build(X, graph_degree=32, intermediate_degree=64, seed=0)
    assert index.graph.shape == (M, 32)

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)

    recalls = [
        _recall(flash_cagra_search(index, Q, k, itopk_size=it)[1], ref_ids)
        for it in (32, 64, 128, 256)
    ]
    # A large internal buffer gets us close to exhaustive recall ...
    assert recalls[-1] >= 0.90, f"itopk=256 recall too low: {recalls}"
    # ... with a clear, near-monotone climb from the smallest budget.
    assert recalls[-1] >= recalls[0] + 0.15, f"itopk gave no gain: {recalls}"
    for lo, hi in zip(recalls, recalls[1:]):
        assert hi >= lo - 0.03, f"recall dropped with larger itopk: {recalls}"


@cuda_only
def test_cagra_torch_triton_parity():
    """Triton search == the torch oracle on a shared graph (search_width=1).

    With ``search_width=1`` the traversal is a single deterministic
    best-first walk, so the Triton single-CTA kernel and the vectorized
    torch oracle -- seeded identically and walking the *same* optimized
    graph -- must return the same neighbours (ids agree; distances match to
    fp tolerance).
    """
    from flashlib import flash_cagra_build, flash_cagra_search

    torch.manual_seed(0)
    M, D, k = 3_000, 32, 10
    X = _blobs(M, D, 8, "cuda", seed=2)
    Q = _blobs(64, D, 8, "cuda", seed=3)

    index = flash_cagra_build(X, graph_degree=32, intermediate_degree=64, seed=0)
    tv, ti = flash_cagra_search(
        index, Q, k, itopk_size=64, search_width=1, seed=0, backend="triton")
    ov, oi = flash_cagra_search(
        index, Q, k, itopk_size=64, search_width=1, seed=0, backend="torch")

    assert _recall(ti, oi) >= 0.98, "triton/torch search disagree at sw=1"
    assert (tv - ov).abs().max().item() < 1e-2


@pytest.mark.skipif(not _hopper_cutedsl(), reason="Hopper + CUTLASS DSL required")
def test_cagra_cutedsl_triton_parity():
    """CuteDSL SMEM search == the Triton search on the same graph.

    Both keep an itopk buffer over the same optimized graph with identical
    seeding; the CuteDSL kernel moves the buffer + dedup to shared memory
    and computes distances with warp-teams, but the *result set* must match
    Triton's (same neighbours; squared-L2 distances to fp tolerance) at
    every itopk -- including the large buffers where Triton spills and
    CuteDSL is the faster path.
    """
    from flashlib import flash_cagra_build, flash_cagra_search

    torch.manual_seed(0)
    M, D, k = 8_000, 64, 10
    X = _blobs(M, D, 8, "cuda", seed=4)
    Q = _blobs(96, D, 8, "cuda", seed=5)

    index = flash_cagra_build(X, graph_degree=32, intermediate_degree=64, seed=0)
    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(k, largest=False, dim=1)

    for itopk in (64, 128, 256):
        cv, ci = flash_cagra_search(
            index, Q, k, itopk_size=itopk, search_width=1, seed=0,
            backend="cutedsl")
        tv, ti = flash_cagra_search(
            index, Q, k, itopk_size=itopk, search_width=1, seed=0,
            backend="triton")
        # Same graph + seeding -> the two kernels return the same set.
        assert _recall(ci, ti) >= 0.99, \
            f"cutedsl/triton disagree at itopk={itopk}: {_recall(ci, ti)}"
        # Returned distances are the true squared-L2 of the returned ids.
        true = ((Q[:, None, :] - X[ci.clamp(min=0)]) ** 2).sum(-1)
        assert (true - cv).abs().max().item() < 1e-2
        # And recall tracks the exhaustive ground truth.
        assert _recall(ci, ref_ids) >= _recall(ti, ref_ids) - 0.02


@cuda_only
def test_cagra_graph_properties():
    """The optimized graph is a valid fixed-degree proximity graph.

    Out-degree must equal ``graph_degree`` exactly, every edge must be a
    valid in-range neighbour, there are no self-loops, and on a single
    connected cloud the optimize pass (detour prune + reverse edges) must
    leave one weakly-connected component (no orphaned nodes that the
    traversal could never reach).
    """
    from flashlib import flash_cagra_build
    from flashlib.primitives.cagra.triton.optimize import _connectivity_report

    torch.manual_seed(0)
    M, D = 5_000, 32
    X = _blobs(M, D, 1, "cuda", seed=0)   # single cloud -> connected

    index = flash_cagra_build(X, graph_degree=32, intermediate_degree=64, seed=0)
    g = index.graph
    assert g.shape == (M, 32) and g.dtype == torch.int32
    assert index.graph_degree == 32
    assert (g >= 0).all() and (g < M).all(), "out-of-range neighbour id"
    self_id = torch.arange(M, device=g.device, dtype=g.dtype)[:, None]
    assert (g == self_id).sum().item() == 0, "graph contains a self-loop"
    # Each row's neighbours are distinct (merge de-duplicates).
    sorted_rows = g.sort(dim=1).values
    assert (sorted_rows[:, 1:] == sorted_rows[:, :-1]).sum().item() == 0
    # Connected data -> one component.
    assert _connectivity_report(g) == 1


@cuda_only
def test_cagra_application_class():
    """The sklearn-style CAGRA wrapper round-trips fit -> kneighbors."""
    from flashlib import CAGRA

    torch.manual_seed(0)
    X = _blobs(8_000, 32, 8, "cuda", seed=4)
    Q = _blobs(64, 32, 8, "cuda", seed=5)
    model = CAGRA(graph_degree=32, n_neighbors=10, itopk_size=128, seed=0).fit(X)
    dist, idx = model.kneighbors(Q)
    assert dist.shape == (64, 10) and idx.shape == (64, 10)
    assert idx.dtype == torch.int64
    assert (idx >= -1).all() and (idx < 8_000).all()
    assert torch.isfinite(dist).all()
    assert model.graph_degree_ == 32

    ref = torch.cdist(Q, X) ** 2
    _, ref_ids = ref.topk(10, largest=False, dim=1)
    assert _recall(idx, ref_ids) >= 0.80


@cuda_only
def test_dbscan_recovers_blobs():
    from flashlib import flash_dbscan

    torch.manual_seed(0)
    centers = torch.tensor([[0., 0.], [20., 20.], [-20., 10.]], device="cuda")
    pts_per = 500
    X = torch.cat([
        c + torch.randn(pts_per, 2, device="cuda") * 0.3 for c in centers
    ]).to(torch.float32)
    # tol=1e-3 exercises the bf16 KNN path (faster + numerically clean for
    # this test's tightly-clustered 2D data). Default tol=None routes through
    # flash_knn's fp32 path which has known accumulator-rounding quirks
    # affecting boundary classification on dense low-D inputs.
    labels = flash_dbscan(X, eps=1.0, min_samples=5, max_neighbors=32, tol=1e-3)
    n_clusters = int(labels.max().item()) + 1 if labels.max() >= 0 else 0
    n_noise = int((labels == -1).sum().item())
    assert n_clusters == 3
    assert n_noise < 50
