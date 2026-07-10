"""Trainium (NKI) flash-kmeans parity tests.

Unlike ``test_backend_parity.py`` (CUDA-guarded), these run on an AWS
NeuronDevice host (Trainium/Inferentia) and are skipped everywhere else.
They exercise cached XLA/NKI assignment and end-to-end Lloyd loops against
torch/NumPy references, including one-hot and atomic updates plus LNC parity.
NKI compiles per shape, so sizes are kept small.
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
    from flashlib.primitives.kmeans.nki import nki_available
    if not nki_available():
        raise ImportError
except Exception:  # pragma: no cover
    pytest.skip("NKI toolchain (neuronxcc) unavailable", allow_module_level=True)

SEED = 0


def _torch_argmin(x, c):
    xs = (x * x).sum(-1, keepdim=True)
    cs = (c * c).sum(-1).unsqueeze(0)
    cross = x.float() @ c.float().t()
    return (xs + cs - 2.0 * cross).argmin(-1).to(torch.int32)


def test_nki_assign_matches_torch_reference():
    torch.manual_seed(SEED)
    N, D, K = 512, 128, 256
    x = torch.randn(N, D, dtype=torch.bfloat16)
    c = torch.randn(K, D, dtype=torch.bfloat16)

    from flashlib.primitives.kmeans.nki import nki_assign_euclid
    ids = nki_assign_euclid(x, c).cpu()
    ref = _torch_argmin(x, c).cpu()

    # bf16 near-ties can flip the argmin; allow a small rate (as the CuteDSL
    # parity test does).
    mismatch = (ids != ref).float().mean().item()
    assert mismatch < 3e-2, f"nki assign mismatch {mismatch:.4f}"


def test_nki_chunked_assign_cross_chunk_tie_and_lnc_parity():
    """Bounded-SBUF large-K assign must preserve global IDs and tie order."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.assign_kernel import (
        make_assign_kernel,
        make_assign_kernel_large,
    )

    rng = np.random.default_rng(SEED)
    N, D, K = 512, 128, 2048
    x = rng.standard_normal((N, D), dtype=np.float32)
    c = rng.standard_normal((K, D), dtype=np.float32)
    c[1024] = c[3]  # same vector in the next resident K chunk
    x[0] = c[3]
    dev = xm.xla_device()
    xT = torch.from_numpy(x).to(
        dev, dtype=torch.bfloat16).t().contiguous()
    cT = torch.from_numpy(c).to(
        dev, dtype=torch.bfloat16).t().contiguous()
    csq = torch.from_numpy(
        np.square(c, dtype=np.float32).sum(
            axis=1, keepdims=True).T).to(dev)

    got = {}
    for n_shards in (1, 2):
        resident = nki.jit(make_assign_kernel(n_shards))
        chunked = nki.jit(make_assign_kernel_large(
            n_shards, k_resident=1024, point_lanes=2))
        if n_shards == 2:
            ref = resident[nl.nc(2)](xT, cT, csq)
            out = chunked[nl.nc(2)](xT, cT, csq)
        else:
            ref = resident(xT, cT, csq)
            out = chunked(xT, cT, csq)
        xm.mark_step()
        ref_np, out_np = ref.cpu().numpy(), out.cpu().numpy()
        assert np.array_equal(out_np, ref_np)
        assert int(out_np[0, 0]) == 3
        got[n_shards] = out_np
    assert np.array_equal(got[1], got[2])


def test_nki_large_assign_65k_sampled_reference():
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.assign_kernel import (
        make_assign_kernel_large,
    )

    rng = np.random.default_rng(SEED + 1)
    N, D, K = 1024, 128, 65536
    x = rng.standard_normal((N, D), dtype=np.float32)
    c = rng.standard_normal((K, D), dtype=np.float32)
    dev = xm.xla_device()
    x_bf = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    c_bf = torch.from_numpy(c).to(torch.bfloat16).float().numpy()
    xT = torch.from_numpy(x).to(
        dev, dtype=torch.bfloat16).t().contiguous()
    cT = torch.from_numpy(c).to(
        dev, dtype=torch.bfloat16).t().contiguous()
    csq_np = np.square(c, dtype=np.float32).sum(
        axis=1, keepdims=True).T
    kernel = nki.jit(make_assign_kernel_large(
        2, k_resident=4096, point_lanes=4))
    got = kernel[nl.nc(2)](
        xT, cT, torch.from_numpy(csq_np).to(dev))
    xm.mark_step()
    got_np = got.cpu().numpy()[:, 0]

    sample = 32
    ref = (
        2.0 * x_bf[:sample] @ c_bf.T - csq_np
    ).argmax(axis=1)
    assert np.array_equal(got_np[:sample], ref)
    assert 0 <= int(got_np.min()) <= int(got_np.max()) < K

    from flashlib.primitives.kmeans.nki import nki_assign_euclid
    api_ids = nki_assign_euclid(
        torch.from_numpy(x), torch.from_numpy(c)).cpu().numpy()
    assert np.array_equal(api_ids[:sample], ref)


def test_nki_kmeans_converges_like_numpy():
    rng = np.random.default_rng(SEED)
    N, D, K, iters = 512, 64, 16, 4
    x = rng.standard_normal((N, D), dtype=np.float32)
    cent0 = x[rng.integers(0, N, K)].copy()

    from flashlib.primitives.kmeans.nki import nki_kmeans_Euclid
    ids, cent, nit = nki_kmeans_Euclid(
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
    assert ratio < 1.02, f"NKI KMeans inertia ratio {ratio:.4f} too high"


def test_nki_transposed_update_matches_index_add():
    """The default (transposed one-hot-matmul) XLA update should track the
    ``index_add`` reference. Same init + iters, so the only difference is the
    M-step implementation; centroids should agree up to bf16 rounding of the
    sums (the assign uses bf16 either way, so cluster ids match)."""
    pytest.importorskip("torch_xla")
    rng = np.random.default_rng(SEED)
    N, D, K, iters = 1024, 64, 32, 5           # D <= 128 -> transposed path
    x = torch.from_numpy(rng.standard_normal((N, D), dtype=np.float32))
    cent0 = torch.from_numpy(x.numpy()[rng.integers(0, N, K)].copy())

    from flashlib.primitives.kmeans.nki.kmeans import _nki_kmeans_xla
    _, cent_nki, _ = _nki_kmeans_xla(
        x, K, iters, 0.0, cent0.clone(), False, use_nki_update=True)
    _, cent_ia, _ = _nki_kmeans_xla(
        x, K, iters, 0.0, cent0.clone(), False, use_nki_update=False)
    cent_nki = cent_nki.cpu().numpy()
    cent_ia = cent_ia.cpu().numpy()

    def inertia(cc):
        return ((x.numpy()[:, None] - cc[None]) ** 2).sum(-1).min(1).sum()

    # Inertia (clustering quality) must match closely; per-centroid means can
    # differ by the bf16 quantisation of the summed points.
    ratio = inertia(cent_nki) / inertia(cent_ia)
    assert 0.98 < ratio < 1.02, f"transposed-update inertia ratio {ratio:.4f}"


def test_nki_atomic_update_repeated_ids_empty_clusters_and_reset():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.kmeans.nki.kmeans import (
        _jit_update_kernel_atomic, _launch,
    )

    rng = np.random.default_rng(SEED)
    N, D, Kp = 512, 129, 1024
    x = rng.standard_normal((N, D), dtype=np.float32)
    # Deliberately use only seven clusters: repeated ids + many empty clusters.
    cid = (np.arange(N, dtype=np.int32) % 7).astype(np.int32)
    src = np.zeros((N, 130), np.float32)
    src[:, :D] = x
    src[:, D] = 1.0
    ref = np.zeros((Kp, 130), np.float32)
    np.add.at(ref, cid, src)

    dev = xm.xla_device()
    src_d = torch.from_numpy(src).to(dev)
    cid_d = torch.from_numpy(cid).to(dev)
    got = {}
    for n_shards in (1, 2):
        kernel = _jit_update_kernel_atomic(Kp, n_shards)
        # Invoke twice: mutable XLA buffers may be reused, so the kernel must
        # clear statistics rather than accumulating across calls.
        for _ in range(2):
            out = _launch(
                kernel, n_shards,
                torch.zeros(Kp, 130, device=dev), cid_d, src_d)
            xm.mark_step()
            got[n_shards] = out.cpu().numpy()
        assert np.allclose(got[n_shards], ref, atol=2e-5, rtol=0)
        assert np.count_nonzero(got[n_shards][7:, D]) == 0
    assert np.array_equal(got[1], got[2])


def test_nki_atomic_update_d512_combines_count_on_lnc2():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.kmeans.nki.kmeans import (
        _jit_update_kernel_atomic, _launch,
    )

    rng = np.random.default_rng(SEED)
    N, D, K, Kp = 512, 512, 37, 1024
    ids = rng.integers(0, K, N, dtype=np.int32)
    src = np.zeros((N, 514), dtype=np.float32)
    src[:, :D] = rng.standard_normal((N, D), dtype=np.float32)
    src[:, D] = 1.0
    ref = np.zeros((Kp, 514), dtype=np.float32)
    np.add.at(ref, ids, src)

    dev = xm.xla_device()
    out = _launch(
        _jit_update_kernel_atomic(Kp, 2), 2,
        torch.zeros(Kp, 514, dtype=torch.float32, device=dev),
        torch.from_numpy(ids).to(dev),
        torch.from_numpy(src).to(dev))
    xm.mark_step()
    assert np.allclose(out.cpu().numpy(), ref, atol=2e-5, rtol=0)


def test_nki_atomic_update_chunk_accumulation():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.kmeans.nki.kmeans import _launch
    from flashlib.primitives.kmeans.nki.update_kernel import (
        make_update_kernel_atomic,
    )
    import neuronxcc.nki as nki

    rng = np.random.default_rng(SEED)
    N, split, D, K, Kp = 1024, 512, 129, 37, 1024
    ids = rng.integers(0, K, N, dtype=np.int32)
    src = np.zeros((N, 130), dtype=np.float32)
    src[:, :D] = rng.standard_normal((N, D), dtype=np.float32)
    src[:, D] = 1.0
    ref = np.zeros((Kp, 130), dtype=np.float32)
    np.add.at(ref, ids, src)
    dev = xm.xla_device()
    ids_d = torch.from_numpy(ids).to(dev)
    src_d = torch.from_numpy(src).to(dev)
    clear = nki.jit(make_update_kernel_atomic(
        Kp, 2, clear_stats=True))
    accum = nki.jit(make_update_kernel_atomic(
        Kp, 2, clear_stats=False))
    stats = _launch(
        clear, 2, torch.zeros(Kp, 130, device=dev),
        ids_d[:split], src_d[:split])
    stats = _launch(
        accum, 2, stats, ids_d[split:], src_d[split:])
    xm.mark_step()
    assert np.allclose(stats.cpu().numpy(), ref, atol=2e-5, rtol=0)


@pytest.mark.parametrize("K", [4096, 16384, 65536])
def test_nki_radix_sort_stable_permutation_and_padding(K):
    """Indirect scatter destinations must be unique and preserve equal-key
    order, including a non-128-multiple real point count."""
    import neuronxcc.nki as nki
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.sort_kernel import (
        make_radix_sort_kernel,
        radix_sort_reference,
    )

    rng = np.random.default_rng(SEED)
    N, Np = 130, 256
    cid = np.zeros(Np, dtype=np.int32)
    cid[:N] = rng.integers(0, K, N, dtype=np.int32)
    # Force repeated keys whose input order is easy to identify.
    cid[:12] = np.array([7, 3, 7, 7, 3, 0, 7, 3, 0, 7, 3, 0])
    dev = xm.xla_device()
    kernel = nki.jit(make_radix_sort_kernel(N, K))
    got = kernel(torch.from_numpy(cid).to(dev))
    xm.mark_step()
    got = got.cpu().numpy()
    ref = radix_sort_reference(cid, N, K)
    assert np.array_equal(got, ref)
    assert np.array_equal(np.sort(got[:N, 1]), np.arange(N))
    assert np.all(got[N:, 0] == K)  # padded sentinel rows sort last


@pytest.mark.parametrize(
    "K,has_padding,expected",
    [
        (1024, False, (2, 5, 32)),
        (4096, False, (2, 6, 64)),
        (16384, False, (2, 7, 128)),
        (65536, False, (3, 6, 64)),
        (1024, True, (2, 6, 64)),
        (4096, True, (2, 7, 128)),
        (16384, True, (3, 5, 32)),
        (65536, True, (3, 6, 64)),
    ],
)
def test_nki_radix_config_boundaries(K, has_padding, expected):
    from flashlib.primitives.kmeans.nki.sort_kernel import radix_config
    assert radix_config(K, has_padding) == expected


def _grouped_test_ids(distribution, N, K, rng):
    if distribution == "uniform":
        return rng.integers(0, K, N, dtype=np.int32)
    if distribution == "hotspot":
        ids = rng.integers(0, K, N, dtype=np.int32)
        ids[rng.random(N) < 0.9] = 0
        return ids
    weights = 1.0 / np.power(np.arange(1, K + 1), 1.2)
    weights /= weights.sum()
    return rng.choice(K, size=N, p=weights).astype(np.int32)


def test_nki_grouped_bucket_alignment_padding_and_distributions():
    """The one-pass bucketizer must preserve every row and leave explicit
    sentinel holes up to each 512-row boundary."""
    import neuronxcc.nki as nki
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.grouped_update_kernel import (
        grouped_bucket_reference,
        make_grouped_bucket_kernel,
    )

    N, Np, K = 385, 512, 4096
    rng = np.random.default_rng(SEED)
    dev = xm.xla_device()
    kernel = nki.jit(make_grouped_bucket_kernel(N, K))
    for distribution in ("uniform", "hotspot", "zipf"):
        ids = np.zeros(Np, dtype=np.int32)
        ids[:N] = _grouped_test_ids(distribution, N, K, rng)
        got_pairs, got_blocks, got_meta = kernel(
            torch.from_numpy(ids).to(dev))
        xm.mark_step()
        got_pairs = got_pairs.cpu().numpy()
        got_blocks = got_blocks.cpu().numpy()
        got_meta = got_meta.cpu().numpy()
        ref_pairs, ref_blocks, ref_meta = grouped_bucket_reference(
            ids, N, K)
        assert np.array_equal(got_meta, ref_meta)
        assert np.array_equal(got_blocks, ref_blocks)
        populated = np.zeros(got_pairs.shape[0], dtype=bool)
        for block, (_, n_rows) in enumerate(got_blocks):
            populated[block * 512:block * 512 + n_rows] = True
        assert np.array_equal(
            got_pairs[populated], ref_pairs[populated])
        assert populated.sum() == Np
        assert np.all((got_meta[0] % 512) == 0)


def test_nki_grouped_update_distributions_empty_padding_reset_and_lnc():
    """Validate exact counts, BF16 sum quality, and feature-sharded parity."""
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.grouped_update_kernel import (
        make_grouped_update_kernel,
    )

    N, Np, D, K = 385, 512, 256, 4096
    rng = np.random.default_rng(SEED)
    points = np.zeros((Np, D), dtype=np.float32)
    points[:N] = rng.standard_normal((N, D), dtype=np.float32)
    # The grouped path contracts BF16 point values; form the reference from
    # those exact quantized values rather than the original FP32 source.
    points_bf = (
        torch.from_numpy(points).to(torch.bfloat16).float().numpy())
    dev = xm.xla_device()
    points_d = torch.from_numpy(points).to(
        dev, dtype=torch.bfloat16)
    kernels = {
        ns: nki.jit(make_grouped_update_kernel(N, K, ns))
        for ns in (1, 2)
    }

    for distribution in ("uniform", "hotspot", "zipf"):
        ids = np.zeros(Np, dtype=np.int32)
        ids[:N] = _grouped_test_ids(distribution, N, K, rng)
        ids_d = torch.from_numpy(ids).to(dev)
        ref_sums = np.zeros((K, D), dtype=np.float32)
        ref_counts = np.zeros(K, dtype=np.int32)
        np.add.at(ref_sums, ids[:N], points_bf[:N])
        np.add.at(ref_counts, ids[:N], 1)

        got = {}
        for ns in (1, 2):
            for _ in range(2):  # private-HBM scratch must not leak state
                if ns == 2:
                    sums, counts = kernels[ns][nl.nc(2)](
                        ids_d, points_d)
                else:
                    sums, counts = kernels[ns](ids_d, points_d)
                xm.mark_step()
                got[ns] = (
                    sums.cpu().numpy(), counts.cpu().numpy())
            sums_np, counts_np = got[ns]
            assert np.allclose(
                sums_np, ref_sums, atol=3e-4, rtol=2e-5)
            for shard in range(ns):
                assert np.array_equal(
                    counts_np[:, shard], ref_counts)
            assert np.count_nonzero(sums_np[ref_counts == 0]) == 0

            # Assigned-cluster inertia is sensitive to centroid-mean errors
            # while avoiding a prohibitive N x 4096 distance matrix.
            live = ref_counts > 0
            cent = np.zeros_like(ref_sums)
            ref_cent = np.zeros_like(ref_sums)
            cent[live] = sums_np[live] / ref_counts[live, None]
            ref_cent[live] = (
                ref_sums[live] / ref_counts[live, None])
            inertia = np.square(
                points_bf[:N] - cent[ids[:N]]).sum()
            ref_inertia = np.square(
                points_bf[:N] - ref_cent[ids[:N]]).sum()
            assert 0.9999 < inertia / ref_inertia < 1.0001

        assert np.array_equal(got[1][1][:, 0], got[2][1][:, 0])
        assert np.allclose(
            got[1][0], got[2][0], atol=3e-4, rtol=0)


@pytest.mark.parametrize(
    "N,D,K,n_shards,expected",
    [
        (262144, 256, 4096, 2, True),
        (262144, 512, 1024, 2, True),
        (262145, 256, 1024, 2, True),
        (262145, 256, 4096, 2, False),
        (131072, 256, 4096, 2, False),
        (262144, 128, 4096, 2, False),
        (262144, 256, 4096, 1, False),
        (262144, 256, 16384, 2, False),
    ],
)
def test_nki_sorted_auto_range(N, D, K, n_shards, expected):
    from flashlib.primitives.kmeans.nki.kmeans import (
        _sorted_auto_eligible,
    )
    assert _sorted_auto_eligible(N, D, K, n_shards) is expected


@pytest.mark.parametrize(
    "N,D,K,n_shards,expected",
    [
        (262144, 256, 4096, 2, True),
        (262143, 256, 4096, 2, False),
        (262144, 257, 4096, 2, False),
        (262144, 256, 2048, 2, False),
        (262144, 256, 4096, 1, False),
    ],
)
def test_nki_grouped_auto_target(N, D, K, n_shards, expected):
    from flashlib.primitives.kmeans.nki.kmeans import (
        _grouped_auto_eligible,
    )
    assert _grouped_auto_eligible(N, D, K, n_shards) is expected


def test_nki_sorted_update_padding_empty_clusters_reset_and_lnc():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.kmeans import (
        _jit_update_kernel_sorted, _launch,
    )

    rng = np.random.default_rng(SEED)
    N, Np, D, Dp, K = 385, 512, 256, 256, 37
    cid = np.zeros(Np, dtype=np.int32)
    # Only seven live clusters exercises repeated IDs and empty cluster rows.
    cid[:N] = np.arange(N, dtype=np.int32) % 7
    src = np.zeros((Np, Dp), dtype=np.float32)
    src[:N, :D] = rng.standard_normal((N, D), dtype=np.float32)
    ref_sums = np.zeros((1024, Dp), dtype=np.float32)
    ref_counts = np.zeros(1024, dtype=np.int32)
    np.add.at(ref_sums, cid[:N], src[:N])
    np.add.at(ref_counts, cid[:N], 1)

    dev = xm.xla_device()
    cid_d = torch.from_numpy(cid).to(dev)
    src_d = torch.from_numpy(src).to(dev)
    got = {}
    for n_shards in (1, 2):
        Kp = 512 * n_shards
        kernel = _jit_update_kernel_sorted(N, K, Kp, n_shards)
        # Invoke twice to catch stale private/shared HBM state.
        for _ in range(2):
            sums, counts = _launch(
                kernel, n_shards, cid_d, src_d)
            xm.mark_step()
            got[n_shards] = (sums.cpu().numpy(), counts.cpu().numpy())
        sums_np, counts_np = got[n_shards]
        assert np.allclose(
            sums_np, ref_sums[:Kp], atol=2e-5, rtol=0)
        for shard in range(n_shards):
            assert np.array_equal(counts_np[:, shard], ref_counts[:Kp])
        assert np.count_nonzero(sums_np[7:]) == 0
    assert np.array_equal(got[1][0], got[2][0][:512])


def test_nki_sorted_lloyd_matches_index_add_inertia():
    pytest.importorskip("torch_xla")
    rng = np.random.default_rng(SEED)
    N, D, K, iters = 512, 129, 37, 3
    x = torch.from_numpy(rng.standard_normal((N, D), dtype=np.float32))
    init = x[:K].clone()

    from flashlib.primitives.kmeans.nki.kmeans import _nki_kmeans_xla
    ids_s, cent_s, _ = _nki_kmeans_xla(
        x, K, iters, 0.0, init.clone(), False,
        use_nki_update="sorted")
    ids_r, cent_r, _ = _nki_kmeans_xla(
        x, K, iters, 0.0, init.clone(), False,
        use_nki_update="index_add")
    assert torch.equal(ids_s.cpu(), ids_r.cpu())

    data = x.numpy()
    def inertia(cc):
        return ((data[:, None] - cc[None]) ** 2).sum(-1).min(1).sum()

    ratio = inertia(cent_s.cpu().numpy()) / inertia(cent_r.cpu().numpy())
    assert 0.999 < ratio < 1.001


def test_nki_kmeans_wide_d_xla_resident():
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.kmeans.nki.kmeans import _nki_kmeans_xla

    rng = np.random.default_rng(SEED)
    N, D, K = 256, 256, 16
    x = torch.from_numpy(rng.standard_normal((N, D), dtype=np.float32))
    init = x[:K].clone()
    dev = xm.xla_device()
    ids, cent, nit = _nki_kmeans_xla(
        x.to(dev), K, 2, 0.0, init.to(dev), False,
        use_nki_update="atomic")
    assert ids.device.type == "xla" and cent.device.type == "xla"
    assert nit == 2 and bool(torch.isfinite(cent.cpu()).all())


def test_flash_kmeans_routes_to_nki():
    """The dispatcher auto-selects the NKI backend on a Neuron host."""
    from flashlib.primitives.kmeans.impl import _route
    from flashlib import _hw
    backend, _ = _route(B=1, N=1000, D=64, K=16, metric="euclidean",
                        hw=_hw.current())
    assert backend == "nki"
    with pytest.raises(ValueError, match="backend must be one of"):
        _route(
            B=1, N=1000, D=64, K=16,
            metric="euclidean", backend="trainium")


def test_nki_multicore_matches_singlecore():
    """The multi-core Lloyd (assign sharded over points, transposed update
    over centroids, ``[nl.nc(n)]``) must be *bit-identical* to single core:
    the split is over independent output rows/cols, not a float reduction, so
    cluster ids and centroids should match exactly (not just in inertia)."""
    pytest.importorskip("torch_xla")
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.kmeans.nki.kmeans import (
        _run_xla_lloyd, _neuron_n_shards,
    )

    n_shards = _neuron_n_shards()
    if n_shards < 2:
        pytest.skip("single logical NeuronCore -- nothing to shard")

    rng = np.random.default_rng(SEED)
    N, D, K, iters = 4096, 64, 128, 5      # D <= 128 -> transposed update path
    x = rng.standard_normal((N, D), dtype=np.float32)
    cent0 = x[rng.integers(0, N, K)].copy()

    dev = xm.xla_device()
    x2d = torch.from_numpy(x).to(dev, dtype=torch.float32)
    c0 = torch.from_numpy(cent0).to(dev, dtype=torch.float32)

    ids1, cent1, _ = _run_xla_lloyd(dev, x2d, c0.clone(), N, D, K, 1,
                                    iters, 0.0, "onehot", False)
    idsN, centN, _ = _run_xla_lloyd(dev, x2d, c0.clone(), N, D, K, n_shards,
                                    iters, 0.0, "onehot", False)

    assert bool((ids1.cpu() == idsN.cpu()).all()), "cluster ids differ across shards"
    assert torch.equal(cent1.cpu(), centN.cpu()), "centroids differ across shards"
