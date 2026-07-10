"""Micro-benchmark: Trainium (NKI) flash-kmeans assign bf16 utilization.

Measures the sustained bf16 PE-array throughput of the standalone flash
assign kernel (the FLOP-dominant Lloyd step) via ``nki.benchmark`` and
reports achieved TFLOPs = ``2 * N * K * D / latency`` against the
single-NeuronCore bf16 peak.

Run on a Trainium instance inside the Neuron venv, e.g.::

    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    python benchmarks/micro/bench_kmeans_nki.py

The assign kernel is compiled once per shape by ``nki.benchmark`` (warmup
+ timed iters), so this reports kernel time without the per-call
baremetal recompile that the end-to-end Lloyd loop pays.

Baseline vs torch-on-Trainium
-----------------------------
``torch_xla_materialize_assign_ms`` times the *naive* assign that
materialises the full ``(N, K)`` distance matrix in device HBM, run on the
same NeuronCore via torch-neuronx / XLA. This is the true apples-to-apples
"flash vs materialise" comparison (same hardware). Measured (assign step,
median warm iter):

    Shape (N, D, K)      | torch-xla (materialise) | NKI (flash) | speedup
    ---------------------+-------------------------+-------------+--------
    262144 x 128 x 4096  |        59.9 ms          |   31.3 ms   |  1.9x
    262144 x 256 x 4096  |        43.4 ms          |   31.4 ms   |  1.4x
    131072 x 512 x 4096  |        32.5 ms          |   15.8 ms   |  2.1x
    262144 x 256 x 16384 |       206.5 ms          |  125.1 ms   |  1.7x

Beyond the ~1.4-2.1x speedup, the flash kernel is ``O(N + K)`` in device
memory vs the materialise path's ``O(N * K)`` -- e.g. N=1M, K=16384 is a
64 GB distance matrix that the naive path cannot hold but flash streams.

Multi-core (more compute)
-------------------------
Sharding the point (M) axis across NeuronCores (``make_assign_kernel(nc)``
launched with ``[nl.nc(nc)]``) is embarrassingly parallel -- no cross-core
reduction. Measured device latency (NCL(50), neuron-bench) at
N=262144, D=256, K=4096:

    one NC      (nc=1)  : 31.4 ms
    one LNC2    (nc=2)  : 15.7 ms   -> 2.0x

Each operator invocation uses one LNC2 execution scope. A trn2.3xlarge has
four such LNC2 scopes on its chip; a separate multi-process experiment is
needed to measure full-chip throughput. With four synchronized,
device-resident workers (``NEURON_RT_VISIBLE_CORES=0..3``), the
``N=32768,D=128,K=1024`` one-hot Lloyd loop measured 2.15--2.25 ms/iteration
per worker, i.e. near-linear aggregate scaling. Note:
intra-kernel tweaks (swapping the argmax op to Vector-only or manual
PSUM double-buffering) did *not* help -- the epilogue is element-throughput
bound across the Vector+GpSimd engines and the compiler already overlaps
them, so "more cores" is the effective lever, not "different ops".

NOTE: ``nki.benchmark`` spawns ``neuron-bench --fixed-nc-count 1`` and
cannot benchmark a multi-core NEFF (or share the device with an in-process
torch-xla runtime). To time multi-core, save the NEFF
(``nki.baremetal(save_neff_name=...)(make_assign_kernel(nc))[nl.nc(nc)]``)
and run ``neuron-bench exec --fixed-nc-count <nc> ... file.neff``. Run the
torch-xla baseline in a *separate* process from the NKI sweep.

On Amazon Linux 2023, torch-xla needs the one-line ELF fix in
``tools/neuron/patch_torch_xla_glibc.py`` first (glibc 2.34 vs the wheel's
``hypot@GLIBC_2.35``).
"""
from __future__ import annotations

import numpy as np

# Dense bf16 peak denominator per physical NeuronCore-v3.
_TRN2_CORE_BF16_TFLOPS = 83.375

_SHAPES = [
    # (N, D, K)
    (262144, 64, 1024),
    (262144, 128, 4096),
    (262144, 256, 4096),
    (1048576, 128, 4096),
    (262144, 256, 16384),
    (131072, 512, 4096),
]


def _bench_one(N, D, K, warmup=5, iters=20):
    import ml_dtypes  # noqa: F401
    import neuronxcc.nki as nki

    from flashlib.primitives.kmeans.nki.assign_kernel import _flash_assign_kernel
    from flashlib.primitives.kmeans.nki.assign import prep_assign

    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D), dtype=np.float32)
    c = rng.standard_normal((K, D), dtype=np.float32)
    xT, cT, csq = prep_assign(x, c)

    kernel = nki.benchmark(warmup=warmup, iters=iters)(_flash_assign_kernel)
    kernel(xT, cT, csq)
    lat = kernel.benchmark_result.nc_latency
    p50_us = float(lat.get_latency_percentile(50))
    p90_us = float(lat.get_latency_percentile(90))

    flops = 2.0 * N * K * D
    tflops = flops / (p50_us * 1e-6) / 1e12
    pct = 100.0 * tflops / _TRN2_CORE_BF16_TFLOPS
    return p50_us, p90_us, tflops, pct


def _ceil(n, m):
    return ((n + m - 1) // m) * m


def _assignment_ids(distribution, N, K, rng):
    """Generate update workloads with distinct contention/occupancy profiles."""
    if distribution == "uniform":
        return rng.integers(0, K, N, dtype=np.int32)
    if distribution == "hotspot":
        ids = rng.integers(0, K, N, dtype=np.int32)
        ids[rng.random(N) < 0.90] = 0
        return ids
    if distribution == "zipf":
        weights = 1.0 / np.power(np.arange(1, K + 1), 1.2)
        weights /= weights.sum()
        return rng.choice(K, size=N, p=weights).astype(np.int32)
    if distribution == "lloyd":
        # A converged Lloyd iteration generally has non-uniform Voronoi-cell
        # occupancy but no point-order grouping. Lognormal cell volumes are a
        # reproducible proxy without materialising an N x K CPU matrix.
        weights = rng.lognormal(mean=0.0, sigma=0.8, size=K)
        weights /= weights.sum()
        return rng.choice(K, size=N, p=weights).astype(np.int32)
    raise ValueError(
        "distribution must be uniform, hotspot, zipf, or lloyd")


def bench_sorted_update_stages(
    N=131072,
    D=128,
    K=4096,
    distributions=("uniform", "hotspot", "zipf", "lloyd"),
    warmup=3,
    iters=10,
    n_shards=1,
):
    """Benchmark radix-only, segmented-only, and complete sorted updates.

    This uses cached ``nki.jit`` XLA custom calls because neuron-bench replaces
    input values with zeros; doing that would make every requested distribution
    a 100% hotspot and invalidate contention results. p50/p90 include the XLA
    step boundary while tensors remain device-resident. The first call's wall
    time includes compilation.
    """
    import time
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.sort_kernel import (
        make_radix_sort_kernel,
        radix_sort_reference,
    )
    from flashlib.primitives.kmeans.nki.update_kernel import (
        make_segmented_update_kernel,
        make_update_kernel_atomic,
        make_update_kernel_sorted,
        make_update_kernel_transposed,
    )

    if D > 512:
        raise ValueError("Trainium KMeans supports D <= 512")

    if n_shards not in (1, 2):
        raise ValueError("n_shards must be 1 or 2")
    Np = _ceil(N, 128 * n_shards)
    if Np >= (1 << 24):
        raise ValueError("sorted update requires padded N < 2**24")
    Dp = _ceil(D, 128 * n_shards)
    Kp = _ceil(K, 512 * n_shards)
    rng = np.random.default_rng(0)
    src = np.zeros((Np, Dp), dtype=np.float32)
    src[:N, :D] = rng.standard_normal((N, D), dtype=np.float32)

    radix_kernel = nki.jit(make_radix_sort_kernel(N, K))
    segment_kernel = nki.jit(
        make_segmented_update_kernel(N, K, Kp, n_shards))
    sorted_kernel = nki.jit(
        make_update_kernel_sorted(N, K, Kp, n_shards))

    def radix_step(ids):
        return radix_kernel(ids)

    def segment_step(points, pairs):
        if n_shards > 1:
            return segment_kernel[nl.nc(n_shards)](points, pairs)
        return segment_kernel(points, pairs)

    def sorted_step(ids, points):
        if n_shards > 1:
            return sorted_kernel[nl.nc(n_shards)](ids, points)
        return sorted_kernel(ids, points)

    tag = f"n{N}_d{D}_k{K}_nc{n_shards}"
    radix_bench = torch_xla.compile(
        radix_step, full_graph=True, name=f"kmeans_radix_{tag}")
    segment_bench = torch_xla.compile(
        segment_step, full_graph=True, name=f"kmeans_segment_{tag}")
    sorted_bench = torch_xla.compile(
        sorted_step, full_graph=True, name=f"kmeans_sorted_{tag}")

    atomic_count_kernel = None
    atomic_count_src = None
    if _ceil(D + 1, n_shards) // n_shards <= 512:
        atomic_width = _ceil(D + 1, n_shards)
        atomic_src = np.zeros((Np, atomic_width), dtype=np.float32)
        atomic_src[:N, :D] = src[:N, :D]
        atomic_src[:N, D] = 1.0
        atomic_kernel = nki.jit(
            make_update_kernel_atomic(Kp, n_shards))
    else:
        atomic_width = D
        atomic_src = src[:, :D]
        atomic_kernel = nki.jit(
            make_update_kernel_atomic(Kp, n_shards))
        atomic_count_kernel = nki.jit(
            make_update_kernel_atomic(Kp, n_shards))
        atomic_count_src = np.zeros(
            (Np, n_shards), dtype=np.float32)
        atomic_count_src[:N] = 1.0

    if atomic_count_kernel is None:
        def atomic_step(stats, ids, points):
            if n_shards > 1:
                return atomic_kernel[nl.nc(n_shards)](
                    stats, ids, points)
            return atomic_kernel(stats, ids, points)
    else:
        def atomic_step(stats, count_stats, ids, points, count_points):
            if n_shards > 1:
                sums = atomic_kernel[nl.nc(n_shards)](
                    stats, ids, points)
                counts = atomic_count_kernel[nl.nc(n_shards)](
                    count_stats, ids, count_points)
            else:
                sums = atomic_kernel(stats, ids, points)
                counts = atomic_count_kernel(
                    count_stats, ids, count_points)
            return sums, counts

    atomic_bench = torch_xla.compile(
        atomic_step, full_graph=True, name=f"kmeans_atomic_{tag}")

    onehot_bench = None
    if D <= 128:
        onehot_kernel = nki.jit(
            make_update_kernel_transposed(Kp, n_shards))

        def onehot_step(points, ids, valid):
            if n_shards > 1:
                return onehot_kernel[nl.nc(n_shards)](
                    points, ids, valid)
            return onehot_kernel(points, ids, valid)

        onehot_bench = torch_xla.compile(
            onehot_step, full_graph=True,
            name=f"kmeans_onehot_{tag}")
    dev = xm.xla_device()
    src_d = torch.from_numpy(src).to(dev)
    atomic_src_d = torch.from_numpy(atomic_src).to(dev)
    atomic_count_src_d = (
        torch.from_numpy(atomic_count_src).to(dev)
        if atomic_count_src is not None else None
    )
    onehot_x_d = src_d[:, :D].to(torch.bfloat16)
    valid_d = torch.zeros(
        Np, 1, dtype=torch.bfloat16, device=dev)
    valid_d[:N] = 1
    xm.mark_step()
    xm.wait_device_ops()

    rows = []
    compiled = set()

    def time_actual(name, kernel, args):
        def execute():
            result = kernel(*args)
            xm.wait_device_ops()
            return result

        compile_wall = 0.0
        if name not in compiled:
            t0 = time.perf_counter()
            execute()
            compile_wall = time.perf_counter() - t0
            compiled.add(name)
        for _ in range(warmup):
            execute()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            execute()
            samples.append((time.perf_counter() - t0) * 1e6)
        return (
            float(np.percentile(samples, 50)),
            float(np.percentile(samples, 90)),
            compile_wall,
        )

    print(
        f"{'distribution':>12} {'stage':>16} {'p50(us)':>10} "
        f"{'p90(us)':>10} {'compile+run(s)':>15}")
    for distribution in distributions:
        ids = np.zeros(Np, dtype=np.int32)
        ids[:N] = _assignment_ids(distribution, N, K, rng)
        pairs = radix_sort_reference(ids, N, K)
        ids_d = torch.from_numpy(ids).to(dev)
        pairs_d = torch.from_numpy(pairs).to(dev)
        atomic_stats_d = torch.zeros(
            Kp, atomic_width, dtype=torch.float32, device=dev)
        atomic_count_stats_d = (
            torch.zeros(
                Kp, n_shards, dtype=torch.float32, device=dev)
            if atomic_count_kernel is not None else None
        )
        xm.mark_step()
        xm.wait_device_ops()

        atomic_args = (
            (atomic_stats_d, ids_d, atomic_src_d)
            if atomic_count_kernel is None
            else (
                atomic_stats_d, atomic_count_stats_d, ids_d,
                atomic_src_d, atomic_count_src_d,
            )
        )
        calls = [
            ("radix", radix_bench, (ids_d,)),
            ("segmented", segment_bench, (src_d, pairs_d)),
            ("sorted_full", sorted_bench, (ids_d, src_d)),
            (
                "atomic",
                atomic_bench,
                atomic_args,
            ),
        ]
        if onehot_bench is not None:
            calls.append(
                (
                    "onehot",
                    onehot_bench,
                    (onehot_x_d, ids_d[:, None], valid_d),
                ))

        for name, kernel, args in calls:
            p50, p90, compile_wall = time_actual(
                name, kernel, args)
            rows.append({
                "distribution": distribution,
                "stage": name,
                "p50_us": p50,
                "p90_us": p90,
                "compile_wall_s": compile_wall,
            })
            print(
                f"{distribution:>12} {name:>16} {p50:>10.1f} "
                f"{p90:>10.1f} {compile_wall:>15.2f}")

    print("crossover: distribution  atomic/sorted  onehot-gate  production")
    for distribution in distributions:
        stage_rows = {
            row["stage"]: row
            for row in rows if row["distribution"] == distribution
        }
        sorted_row = stage_rows["sorted_full"]
        speedup = (
            stage_rows["atomic"]["p50_us"] / sorted_row["p50_us"])
        onehot_gate = (
            "onehot" not in stage_rows
            or sorted_row["p50_us"] <= stage_rows["onehot"]["p50_us"]
        )
        production_gate = speedup >= 1.25 and onehot_gate
        sorted_row["speedup_vs_atomic"] = speedup
        sorted_row["production_gate"] = production_gate
        print(
            f"           {distribution:>12} {speedup:>14.2f}x "
            f"{str(onehot_gate):>12} {str(production_gate):>11}")
    return rows


def bench_grouped_update_stages(
    N=262144,
    D=256,
    K=4096,
    distributions=("uniform", "hotspot", "zipf", "lloyd"),
    warmup=3,
    iters=10,
    n_shards=2,
):
    """Benchmark the fused low6/high6 group-reduce and its three stages.

    The standalone local/merge stages use one ``D / n_shards`` feature slice,
    matching the work performed by each physical core in the complete LNC
    launch.  The production decision is intentionally strict: grouped must be
    at least 1.25x faster than sorted for both uniform and Lloyd-like IDs and
    must not regress on hotspot IDs.
    """
    import time
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.grouped_update_kernel import (
        make_grouped_bucket_kernel,
        make_grouped_local_kernel,
        make_grouped_merge_kernel,
        make_grouped_update_kernel,
    )
    from flashlib.primitives.kmeans.nki.update_kernel import (
        make_update_kernel_atomic,
        make_update_kernel_groupby_baseline,
        make_update_kernel_sorted,
    )

    if K != 4096:
        raise ValueError("first grouped-update benchmark requires K=4096")
    if n_shards not in (1, 2):
        raise ValueError("n_shards must be 1 or 2")
    Np = _ceil(N, 128 * n_shards)
    Dp = _ceil(D, 128 * n_shards)
    if Dp // n_shards + 1 > 512:
        raise ValueError("grouped per-core feature shard must be <= 511")
    Kp = _ceil(K, 512 * n_shards)
    if Kp != K:
        raise ValueError("grouped update does not emit padded cluster rows")

    rng = np.random.default_rng(0)
    src = np.zeros((Np, Dp), dtype=np.float32)
    src[:N, :D] = rng.standard_normal((N, D), dtype=np.float32)
    d_per = Dp // n_shards
    valid = np.zeros((Np, 1), dtype=np.float32)
    valid[:N] = 1.0
    atomic_width = _ceil(D + 1, n_shards)
    atomic_src = np.zeros((Np, atomic_width), dtype=np.float32)
    atomic_src[:, :D] = src[:, :D]
    atomic_src[:, D:D + 1] = valid

    bucket_kernel = nki.jit(make_grouped_bucket_kernel(N, K))
    local_kernel = nki.jit(make_grouped_local_kernel(Np, K))
    merge_kernel = nki.jit(make_grouped_merge_kernel(Np, K))
    grouped_kernel = nki.jit(
        make_grouped_update_kernel(N, K, n_shards))
    direct_kernel = nki.jit(
        make_update_kernel_groupby_baseline(K, n_shards))
    sorted_kernel = nki.jit(
        make_update_kernel_sorted(N, K, Kp, n_shards))
    atomic_kernel = nki.jit(
        make_update_kernel_atomic(Kp, n_shards))

    def bucket_step(ids):
        return bucket_kernel(ids)

    def local_step(points, pairs, block_metadata):
        return local_kernel(points, pairs, block_metadata)

    def merge_step(partial_sums, partial_counts, block_keys):
        return merge_kernel(
            partial_sums, partial_counts, block_keys)

    def grouped_step(ids, points):
        if n_shards > 1:
            return grouped_kernel[nl.nc(n_shards)](ids, points)
        return grouped_kernel(ids, points)

    def direct_step(points, ids, point_valid):
        if n_shards > 1:
            return direct_kernel[nl.nc(n_shards)](
                points, ids, point_valid)
        return direct_kernel(points, ids, point_valid)

    def sorted_step(ids, points):
        if n_shards > 1:
            return sorted_kernel[nl.nc(n_shards)](ids, points)
        return sorted_kernel(ids, points)

    def atomic_step(stats, ids, points):
        if n_shards > 1:
            return atomic_kernel[nl.nc(n_shards)](
                stats, ids, points)
        return atomic_kernel(stats, ids, points)

    tag = f"n{N}_d{D}_k{K}_nc{n_shards}"
    compiled_steps = {
        "bucket": torch_xla.compile(
            bucket_step, full_graph=True,
            name=f"kmeans_grouped_bucket_{tag}"),
        "local_partial": torch_xla.compile(
            local_step, full_graph=True,
            name=f"kmeans_grouped_local_{tag}"),
        "merge": torch_xla.compile(
            merge_step, full_graph=True,
            name=f"kmeans_grouped_merge_{tag}"),
        "grouped_full": torch_xla.compile(
            grouped_step, full_graph=True,
            name=f"kmeans_grouped_full_{tag}"),
        "sorted_full": torch_xla.compile(
            sorted_step, full_graph=True,
            name=f"kmeans_grouped_sorted_{tag}"),
        "atomic": torch_xla.compile(
            atomic_step, full_graph=True,
            name=f"kmeans_grouped_atomic_{tag}"),
        "direct_onehot": torch_xla.compile(
            direct_step, full_graph=True,
            name=f"kmeans_grouped_direct_{tag}"),
    }

    dev = xm.xla_device()
    src_f32_d = torch.from_numpy(src).to(dev)
    src_bf16_d = src_f32_d.to(torch.bfloat16)
    src_shard_d = torch.from_numpy(src[:, :d_per]).to(
        dev, dtype=torch.bfloat16)
    valid_d = torch.from_numpy(valid).to(
        dev, dtype=torch.bfloat16)
    atomic_src_d = torch.from_numpy(atomic_src).to(dev)
    xm.mark_step()
    xm.wait_device_ops()

    compiled = set()

    def time_actual(name, kernel, args):
        def execute():
            result = kernel(*args)
            xm.wait_device_ops()
            return result

        compile_wall = 0.0
        if name not in compiled:
            t0 = time.perf_counter()
            result = execute()
            compile_wall = time.perf_counter() - t0
            compiled.add(name)
        else:
            result = execute()
        for _ in range(warmup):
            result = execute()
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            result = execute()
            samples.append((time.perf_counter() - t0) * 1e6)
        return {
            "p50_us": float(np.percentile(samples, 50)),
            "p90_us": float(np.percentile(samples, 90)),
            "compile_wall_s": compile_wall,
        }, result

    rows = []
    print(
        f"{'distribution':>12} {'stage':>16} {'p50(us)':>10} "
        f"{'p90(us)':>10} {'compile+run(s)':>15}")
    for distribution in distributions:
        ids = np.zeros(Np, dtype=np.int32)
        ids[:N] = _assignment_ids(distribution, N, K, rng)
        ids_d = torch.from_numpy(ids).to(dev)
        ids_col_d = ids_d[:, None]
        atomic_stats_d = torch.zeros(
            Kp, atomic_width, dtype=torch.float32, device=dev)
        xm.mark_step()
        xm.wait_device_ops()

        bucket_stats, bucket_out = time_actual(
            "bucket", compiled_steps["bucket"], (ids_d,))
        pairs_d, block_metadata_d, _ = bucket_out
        local_stats, local_out = time_actual(
            "local_partial", compiled_steps["local_partial"],
            (src_shard_d, pairs_d, block_metadata_d))
        partial_sums_d, partial_counts_d, block_keys_d = local_out
        merge_stats, _ = time_actual(
            "merge", compiled_steps["merge"],
            (partial_sums_d, partial_counts_d, block_keys_d))

        measured = [
            ("bucket", bucket_stats),
            ("local_partial", local_stats),
            ("merge", merge_stats),
        ]
        for name, args in (
            ("grouped_full", (ids_d, src_bf16_d)),
            ("sorted_full", (ids_d, src_f32_d)),
            ("atomic", (atomic_stats_d, ids_d, atomic_src_d)),
            ("direct_onehot", (src_bf16_d, ids_col_d, valid_d)),
        ):
            stats, _ = time_actual(
                name, compiled_steps[name], args)
            measured.append((name, stats))

        for name, stats in measured:
            row = {
                "distribution": distribution,
                "stage": name,
                **stats,
            }
            rows.append(row)
            print(
                f"{distribution:>12} {name:>16} "
                f"{stats['p50_us']:>10.1f} "
                f"{stats['p90_us']:>10.1f} "
                f"{stats['compile_wall_s']:>15.2f}")

    by_distribution = {
        distribution: {
            row["stage"]: row
            for row in rows
            if row["distribution"] == distribution
        }
        for distribution in distributions
    }
    gate_distributions = [
        name for name in ("uniform", "lloyd")
        if name in by_distribution
    ]
    speedup_gate = (
        len(gate_distributions) == 2
        and all(
            by_distribution[name]["sorted_full"]["p50_us"]
            / by_distribution[name]["grouped_full"]["p50_us"] >= 1.25
            for name in gate_distributions
        )
    )
    hotspot_gate = (
        "hotspot" not in by_distribution
        or by_distribution["hotspot"]["grouped_full"]["p50_us"]
        <= by_distribution["hotspot"]["sorted_full"]["p50_us"]
    )
    production_gate = speedup_gate and hotspot_gate
    print("grouped production gate:")
    for name in gate_distributions:
        speedup = (
            by_distribution[name]["sorted_full"]["p50_us"]
            / by_distribution[name]["grouped_full"]["p50_us"])
        print(f"  {name:>8}: {speedup:.2f}x vs sorted (need >=1.25x)")
    if "hotspot" in by_distribution:
        regression = (
            by_distribution["hotspot"]["grouped_full"]["p50_us"]
            / by_distribution["hotspot"]["sorted_full"]["p50_us"])
        print(f"   hotspot: {regression:.2f}x sorted latency (need <=1.00x)")
    print(f"  auto-route eligible: {production_gate}")
    return rows, production_gate


def profile_sorted_update(
    N=131072,
    D=128,
    K=4096,
    distribution="uniform",
    working_directory="/tmp",
):
    """Create Neuron Profile artifacts for the complete sorted update.

    NKI profile execution substitutes zero-valued inputs. Control flow is
    fixed, but indirect DMA addresses remain data-dependent, so use this trace
    for engine/spill analysis and the actual-value benchmark above for timing.
    ``distribution`` is retained only in the artifact label.
    """
    import os
    import neuronxcc.nki as nki

    from flashlib.primitives.kmeans.nki.update_kernel import (
        make_update_kernel_sorted,
    )

    Np, Dp, Kp = _ceil(N, 128), _ceil(D, 128), _ceil(K, 512)
    rng = np.random.default_rng(0)
    ids = np.zeros(Np, dtype=np.int32)
    ids[:N] = _assignment_ids(distribution, N, K, rng)
    src = np.zeros((Np, Dp), dtype=np.float32)
    src[:N, :D] = rng.standard_normal((N, D), dtype=np.float32)
    tag = f"kmeans_sorted_n{N}_d{D}_k{K}_{distribution}"
    profile_dir = os.path.abspath(working_directory)
    os.makedirs(profile_dir, exist_ok=True)
    profiler = nki.profile(
        working_directory=profile_dir,
        save_neff_name=f"{tag}.neff",
        save_trace_name=f"{tag}.ntff",
        overwrite=True,
    )(make_update_kernel_sorted(N, K, Kp))
    profiler(ids, src)
    print(
        f"profile reports: {profile_dir}/json_reports/{tag}_profile_*.json "
        "(inspect summary for SBUF/PSUM spill and engine utilization)")
    return tag


def profile_grouped_update(
    N=262144,
    D=256,
    K=4096,
    n_shards=2,
    working_directory="/tmp",
):
    """Create Neuron Profile artifacts for the complete grouped update.

    Profiling substitutes zero-valued inputs, so all IDs hit one low bucket.
    Use the actual-value benchmark for latency; use this trace for engine,
    DGE/HBM, and spill analysis.
    """
    import os
    import ml_dtypes
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    from flashlib.primitives.kmeans.nki.grouped_update_kernel import (
        make_grouped_update_kernel,
    )

    if K != 4096:
        raise ValueError("first grouped-update profile requires K=4096")
    Np = _ceil(N, 128 * n_shards)
    Dp = _ceil(D, 128 * n_shards)
    ids = np.zeros(Np, dtype=np.int32)
    src = np.zeros((Np, Dp), dtype=ml_dtypes.bfloat16)
    tag = f"kmeans_grouped_n{N}_d{D}_k{K}_nc{n_shards}"
    profile_dir = os.path.abspath(working_directory)
    os.makedirs(profile_dir, exist_ok=True)
    profiler = nki.profile(
        working_directory=profile_dir,
        save_neff_name=f"{tag}.neff",
        save_trace_name=f"{tag}.ntff",
        overwrite=True,
    )(make_grouped_update_kernel(N, K, n_shards))
    if n_shards > 1:
        profiler[nl.nc(n_shards)](ids, src)
    else:
        profiler(ids, src)
    print(
        f"profile reports: {profile_dir}/json_reports/"
        f"{tag}_profile_*.json")
    return tag


def torch_xla_materialize_assign_ms(N, D, K, warmup=3, reps=10):
    """Naive torch assign (materialise the (N,K) distance matrix) on Trainium.

    Runs on the NeuronCore via torch-neuronx / XLA -- the same-hardware
    baseline for the flash kernel. ``.item()`` forces full execution so XLA
    cannot DCE the unused output. Run in a separate process from the NKI
    sweep (device sharing conflict). Requires the AL2023 torch-xla fix
    (``tools/neuron/patch_torch_xla_glibc.py``).
    """
    import time
    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    x = torch.randn(N, D, device=dev)
    c = torch.randn(K, D, device=dev)

    def step():
        d = (x * x).sum(1, keepdim=True) + (c * c).sum(1)[None, :] - 2.0 * (x @ c.t())
        return int(d.argmin(1).sum().item())

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    return (time.perf_counter() - t0) / reps * 1e3


def bench_lloyd_update_ab(N=131072, D=128, K=4096, iters=15):
    """End-to-end A/B of sorted, one-hot, atomic, and index_add updates.

    Reports total wall (including disk-cached compile lookup) per mode.

    Measured on trn2 (warm Neuron compile cache), N=131072, D=128, K=4096,
    15 iters::

        index_add scatter   : ~11.8 s
        transposed NKI matmul: ~1.1 s   -> ~10x

    The win is the M-step: the scatter is data-dependent (Trainium has no
    hardware sort; ``torch.sort`` is rejected, ``NCC_EVRF029``), while the
    transposed one-hot matmul (x stationary, one-hot the 512-wide moving
    operand -> assign-like matmul-tile count and ~seconds compile) runs on the
    PE array. Requires torch-xla.
    """
    import time
    import numpy as np
    import torch

    from flashlib.primitives.kmeans.nki.kmeans import nki_kmeans_Euclid

    x = torch.from_numpy(np.random.default_rng(0).standard_normal((N, D), dtype=np.float32))
    init = x[:K].clone()
    out = {}
    modes = ("index_add", "onehot", "atomic", "sorted")
    for mode in modes:
        t0 = time.perf_counter()
        nki_kmeans_Euclid(x, K, max_iters=iters, init_centroids=init.clone(),
                               use_nki_update=mode)
        out[mode] = time.perf_counter() - t0
    label = {
        "index_add": "index_add scatter",
        "onehot": "transposed one-hot",
        "atomic": "NKI atomic scatter",
        "sorted": "radix request*",
    }
    for mode in modes:
        print(f"  {label[mode]:>22}: {out[mode]:6.1f} s "
              f"({out[mode]/iters*1e3:5.0f} ms/iter incl compile)")
    winner = min(modes, key=out.get)
    print(f"  winner: {label[winner]} "
          f"({out['index_add']/out[winner]:.1f}x vs index_add)")
    print("  * sorted request has atomic/index_add safety fallbacks")
    return out


def bench_multicore_scaling(N=262144, D=128, K=4096, iters=20):
    """Warm per-iteration time of the full Lloyd loop on 1 vs all logical
    NeuronCores (assign sharded over points, transposed update over centroids).

    Both kernels are launched ``[nl.nc(n_shards)]``; the split is over
    independent output rows/cols so results are bit-identical to single core.
    Measured on trn2.3xlarge (2 logical cores, LNC=2), warm compile cache:

        N=131072 D=128 K=4096 : nc1 24.5 ms/it -> nc2 13.8 ms/it (1.77x)
        N=262144 D=128 K=4096 : nc1 50.2 ms/it -> nc2 29.9 ms/it (1.68x)

    and the assign kernel alone scales ~1.97x (near-linear). Requires torch-xla.
    """
    import time
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.kmeans import _run_xla_lloyd

    dev = xm.xla_device()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D), dtype=np.float32)
    cent0 = x[rng.integers(0, N, K)].copy()
    x2d = torch.from_numpy(x).to(dev, dtype=torch.float32)
    c0 = torch.from_numpy(cent0).to(dev, dtype=torch.float32)

    row = {}
    for ns in (1, 2):
        _run_xla_lloyd(
            dev, x2d, c0.clone(), N, D, K, ns, 3, 0.0, "onehot", False)
        t = time.perf_counter()
        ids, _, _ = _run_xla_lloyd(dev, x2d, c0.clone(), N, D, K, ns, iters,
                                   0.0, "onehot", False)
        _ = ids.cpu()
        row[ns] = (time.perf_counter() - t) / iters * 1e3
    print(f"  N={N} D={D} K={K}: nc1={row[1]:.1f} ms/it -> "
          f"nc2={row[2]:.1f} ms/it ({row[1]/row[2]:.2f}x)")
    return row


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sorted-update", action="store_true",
        help="benchmark radix/segmented/full update stages instead of assign")
    parser.add_argument(
        "--profile-sorted", action="store_true",
        help="emit Neuron Profile artifacts for the complete sorted update")
    parser.add_argument(
        "--grouped-update", action="store_true",
        help="benchmark fused low6/high6 group-reduce stages")
    parser.add_argument(
        "--profile-grouped", action="store_true",
        help="emit Neuron Profile artifacts for the grouped update")
    parser.add_argument("--N", type=int, default=131072)
    parser.add_argument("--D", type=int, default=128)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--n-shards", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--distributions", nargs="+",
        default=["uniform", "hotspot", "zipf", "lloyd"])
    parser.add_argument("--profile-dir", default="/tmp")
    args = parser.parse_args()

    if args.profile_sorted:
        profile_sorted_update(
            args.N, args.D, args.K, args.distributions[0],
            args.profile_dir)
        return
    if args.profile_grouped:
        profile_grouped_update(
            args.N, args.D, args.K, args.n_shards,
            args.profile_dir)
        return
    if args.grouped_update:
        bench_grouped_update_stages(
            args.N, args.D, args.K, args.distributions,
            args.warmup, args.iters, args.n_shards)
        return
    if args.sorted_update:
        bench_sorted_update_stages(
            args.N, args.D, args.K, args.distributions,
            args.warmup, args.iters, args.n_shards)
        return

    print(f"{'N':>8} {'D':>5} {'K':>7} {'p50(us)':>10} {'p90(us)':>10} "
          f"{'TFLOPs':>8} {'%peak':>7}")
    for N, D, K in _SHAPES:
        try:
            p50, p90, tflops, pct = _bench_one(N, D, K)
            print(f"{N:>8} {D:>5} {K:>7} {p50:>10.1f} {p90:>10.1f} "
                  f"{tflops:>8.1f} {pct:>6.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"{N:>8} {D:>5} {K:>7}  FAILED: {str(e).splitlines()[0][:60]}")
    print("\nLloyd update A/B (full loop, transposed NKI matmul vs index_add):")
    try:
        bench_lloyd_update_ab()
    except Exception as e:  # noqa: BLE001
        print(f"  update A/B skipped: {str(e).splitlines()[0][:60]}")
    print("\nMulti-core scaling (full Lloyd iter, 1 vs all logical cores):")
    try:
        bench_multicore_scaling()
    except Exception as e:  # noqa: BLE001
        print(f"  multicore scaling skipped: {str(e).splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
