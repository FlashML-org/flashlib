"""Large-shape exact KMeans benchmarks for one full Trainium2 chip.

The production target is N=10M, K=65536, D in {128, 256, 512}.  A single
exact Lloyd assign contains hundreds of TFLOPs, so this harness uses scalable
proxy shapes, reports extrapolated lower bounds, and only launches the final
shape after every smaller gate passes.

Measured full-chip target (warm, four N-sharded LNC2 ranks):

    D       assign       full Lloyd iteration      assign HFU
    128     2.483 s            3.713 s               10.1%
    256     2.575 s            4.155 s               19.5%
    512     2.589 s            4.801 s               38.9%

Examples::

    # One LNC2 exact-assign proxy
    python benchmarks/micro/bench_kmeans_nki_large.py \
        --assign --N 32768 --D 128 --K 65536

    # Four-LNC2 XLA collective baseline
    torchrun --nproc_per_node=4 \
        benchmarks/micro/bench_kmeans_nki_large.py \
        --all-reduce --D 256 --K 65536
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np


_CHIP_BF16_TFLOPS = 667.0
_PHYSICAL_CORES = 8
_LNC2_SCOPES = 4
_SBUF_BYTES_PER_CORE = 28 * 1024**2
_TARGET_N = 10_000_000
_TARGET_K = 65_536
_TARGET_D = (128, 256, 512)
_ASSIGN_N_CHUNK = 32768


def _ceil(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def shape_report(N: int, D: int, K: int, *, world_size: int = 4):
    """Static FLOP/memory feasibility report without allocating tensors."""
    flops = 2 * N * K * D
    target_peak_s = flops / (_CHIP_BF16_TFLOPS * 1e12)
    centroid_bf16 = K * D * 2
    point_bf16 = N * D * 2
    point_fp32 = N * D * 4
    stats_fp32 = K * (D + 1) * 4
    local_n = _ceil(
        math.ceil(N / world_size), _ASSIGN_N_CHUNK)
    return {
        "N": N,
        "D": D,
        "K": K,
        "flops": flops,
        "peak_lower_bound_s": target_peak_s,
        "peak_lower_bound_min": target_peak_s / 60,
        "point_bf16_gib": point_bf16 / 1024**3,
        "point_fp32_gib": point_fp32 / 1024**3,
        "centroid_bf16_mib": centroid_bf16 / 1024**2,
        "stats_fp32_mib_per_rank": stats_fp32 / 1024**2,
        "rank_local_N_padded": local_n,
        "full_centroid_sbuf_mib": centroid_bf16 / 1024**2,
        "full_centroid_fits_one_core_sbuf": (
            centroid_bf16 < _SBUF_BYTES_PER_CORE),
    }


def print_target_reports():
    for D in _TARGET_D:
        print(json.dumps(
            shape_report(_TARGET_N, D, _TARGET_K),
            sort_keys=True))


def bench_assign(
    N: int,
    D: int,
    K: int,
    *,
    n_shards: int = 2,
    k_resident: int = 4096,
    point_lanes: int = 4,
    warmup: int = 2,
    iters: int = 5,
):
    """Actual-value, device-resident exact assign timing on one LNC."""
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl

    from flashlib.primitives.kmeans.nki.assign_kernel import (
        make_assign_kernel,
        make_assign_kernel_large,
    )

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed and not dist.is_initialized():
        dist.init_process_group("xla")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    multiple = 128 * n_shards * (point_lanes if K >= 32768 else 1)
    Np = _ceil(
        N, _ASSIGN_N_CHUNK if K >= 32768 else multiple)
    Kp = _ceil(K, k_resident if K >= 32768 else 512 * n_shards)
    Dp = _ceil(D, 128)
    rng = np.random.default_rng(0)
    x = np.zeros((Dp, Np), dtype=np.float32)
    c = np.zeros((Dp, Kp), dtype=np.float32)
    x[:D, :N] = rng.standard_normal((D, N), dtype=np.float32)
    c[:D, :K] = rng.standard_normal((D, K), dtype=np.float32)
    csq = np.full((1, Kp), 3.0e38, dtype=np.float32)
    csq[:, :K] = np.square(c[:D, :K], dtype=np.float32).sum(
        axis=0, keepdims=True)

    dev = xm.xla_device()
    x_d = torch.from_numpy(x).to(dev, dtype=torch.bfloat16)
    c_d = torch.from_numpy(c).to(dev, dtype=torch.bfloat16)
    q_d = torch.from_numpy(csq).to(dev)
    x_chunks = (
        [
            x_d[:, n0:n0 + _ASSIGN_N_CHUNK].contiguous()
            for n0 in range(0, Np, _ASSIGN_N_CHUNK)
        ]
        if K >= 32768 else [x_d]
    )
    body = (
        make_assign_kernel_large(
            n_shards, k_resident=k_resident,
            point_lanes=point_lanes)
        if K >= 32768 else make_assign_kernel(n_shards)
    )
    kernel = nki.jit(body)

    def step(x_arg, c_arg, q_arg):
        if n_shards > 1:
            return kernel[nl.nc(n_shards)](x_arg, c_arg, q_arg)
        return kernel(x_arg, c_arg, q_arg)

    compiled = torch_xla.compile(
        step, full_graph=True,
        name=(
            f"kmeans_large_assign_nc{_ASSIGN_N_CHUNK if K >= 32768 else Np}_"
            f"d{D}_k{K}_"
            f"kr{k_resident}_l{point_lanes}_nc{n_shards}"))

    def execute():
        if K >= 32768:
            outputs = []
            for x_chunk in x_chunks:
                outputs.append(compiled(x_chunk, c_d, q_d))
            xm.wait_device_ops()
            return outputs
        out_tensor = compiled(x_d, c_d, q_d)
        xm.wait_device_ops()
        return [out_tensor]

    xm.mark_step()
    xm.wait_device_ops()
    t0 = time.perf_counter()
    outputs = execute()
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        outputs = execute()
    if distributed:
        dist.barrier()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        outputs = execute()
        samples.append((time.perf_counter() - t0) * 1e3)
    p50_ms = float(np.percentile(samples, 50))
    p90_ms = float(np.percentile(samples, 90))
    tflops = 2 * N * K * D / (p50_ms * 1e-3) / 1e12
    result = {
        "stage": "assign",
        "rank": rank,
        "world_size": world,
        "N": N,
        "D": D,
        "K": K,
        "n_shards": n_shards,
        "k_resident": k_resident,
        "point_lanes": point_lanes,
        "compile_s": compile_s,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "lnc_tflops": tflops,
        "lnc_hfu_percent": (
            100 * tflops / (_CHIP_BF16_TFLOPS / _LNC2_SCOPES)),
        "id_checksum_1k": int(torch.cat(
            outputs, dim=0)[:min(N, 1024)].cpu().numpy().astype(
                np.int64).sum()),
    }
    if K == _TARGET_K and D in _TARGET_D:
        result["target_fullchip_extrapolated_s"] = (
            2 * _TARGET_N * K * D
            / (tflops * _LNC2_SCOPES * 1e12))
    print(json.dumps(result, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return result


def bench_distributed_all_reduce(
    D: int,
    K: int,
    *,
    warmup: int = 2,
    iters: int = 5,
):
    """Benchmark the Kx(D+1) statistics collective under torchrun."""
    import torch
    import torch.distributed as dist
    import torch_xla
    import torch_xla.core.xla_model as xm

    if not dist.is_initialized():
        dist.init_process_group("xla")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != _LNC2_SCOPES:
        raise ValueError(
            f"full-chip benchmark requires world_size=4, got {world}")
    dev = xm.xla_device()
    stats = torch.full(
        (K, D + 1), float(rank + 1),
        dtype=torch.float32, device=dev)

    def step(value):
        return xm.all_reduce(xm.REDUCE_SUM, value)

    compiled = torch_xla.compile(
        step, full_graph=True,
        name=f"kmeans_stats_allreduce_k{K}_d{D}_w{world}")
    dist.barrier()
    t0 = time.perf_counter()
    out = compiled(stats)
    xm.wait_device_ops()
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        out = compiled(stats)
        xm.wait_device_ops()
    dist.barrier()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = compiled(stats)
        xm.wait_device_ops()
        samples.append((time.perf_counter() - t0) * 1e3)
    dist.barrier()
    expected = world * (world + 1) / 2
    check = float(out[0, 0].cpu())
    row = {
        "stage": "all_reduce",
        "rank": rank,
        "world_size": world,
        "K": K,
        "D_plus_count": D + 1,
        "bytes": K * (D + 1) * 4,
        "compile_s": compile_s,
        "p50_ms": float(np.percentile(samples, 50)),
        "p90_ms": float(np.percentile(samples, 90)),
        "correct": check == expected,
        "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    dist.destroy_process_group()
    return row


def bench_distributed_lloyd(
    local_N: int,
    D: int,
    K: int,
    *,
    iters: int = 3,
    update_mode: str = "atomic",
):
    """Run exact N-sharded Lloyd across all four LNC2 scopes."""
    import torch
    import torch.distributed as dist
    import torch_xla.core.xla_model as xm

    from flashlib.primitives.kmeans.nki.kmeans import (
        nki_kmeans_Euclid_distributed,
    )

    if not dist.is_initialized():
        dist.init_process_group("xla")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != _LNC2_SCOPES:
        raise ValueError("distributed Lloyd requires four LNC2 ranks")
    dev = xm.xla_device()
    local_rng = np.random.default_rng(100 + rank)
    init_rng = np.random.default_rng(17)
    local_x = torch.from_numpy(local_rng.standard_normal(
        (local_N, D), dtype=np.float32)).to(
            dev, dtype=torch.bfloat16 if K >= 32768 else torch.float32)
    init = torch.from_numpy(init_rng.standard_normal(
        (K, D), dtype=np.float32)).to(dev)
    xm.mark_step()
    xm.wait_device_ops()

    # Compile every rank before entering the synchronized measurement.
    for _ in range(2):
        ids, cent, _ = nki_kmeans_Euclid_distributed(
            local_x, K, max_iters=iters, tol=0.0,
            init_centroids=init, use_nki_update=update_mode)
        xm.wait_device_ops()
    dist.barrier()
    t0 = time.perf_counter()
    ids, cent, nit = nki_kmeans_Euclid_distributed(
        local_x, K, max_iters=iters, tol=0.0,
        init_centroids=init, use_nki_update=update_mode)
    xm.wait_device_ops()
    elapsed = time.perf_counter() - t0
    dist.barrier()
    # All ranks must own identical centroids after each statistics all-reduce.
    checksum = cent.float().sum()
    gathered = xm.all_gather(checksum.reshape(1), dim=0)
    xm.mark_step()
    gathered_np = gathered.cpu().numpy()
    row = {
        "stage": "distributed_lloyd",
        "rank": rank,
        "world_size": world,
        "global_N": local_N * world,
        "local_N": local_N,
        "D": D,
        "K": K,
        "update": update_mode,
        "iterations": nit,
        "ms_per_iter": elapsed / nit * 1e3,
        "centroids_equal": bool(np.allclose(
            gathered_np, gathered_np[0], atol=1e-3, rtol=0)),
        "local_id_checksum_1k": int(
            ids[:min(local_N, 1024)].cpu().numpy().astype(
                np.int64).sum()),
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--assign", action="store_true")
    parser.add_argument("--all-reduce", action="store_true")
    parser.add_argument("--lloyd", action="store_true")
    parser.add_argument(
        "--launch-full-chip", action="store_true",
        help="launch four torchrun ranks and retry transient cache races")
    parser.add_argument("--N", type=int, default=32768)
    parser.add_argument("--D", type=int, choices=_TARGET_D, default=128)
    parser.add_argument("--K", type=int, default=_TARGET_K)
    parser.add_argument("--n-shards", type=int, choices=(1, 2), default=2)
    parser.add_argument("--k-resident", type=int, default=4096)
    parser.add_argument("--point-lanes", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--update", choices=("atomic", "sorted", "index_add"),
        default="atomic")
    args = parser.parse_args()
    if args.launch_full_chip:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("--launch-full-chip is only valid in the parent")
        forwarded = [
            arg for arg in sys.argv[1:]
            if arg != "--launch-full-chip"
        ]
        command = [
            os.path.join(
                os.path.dirname(sys.executable), "torchrun"),
            "--standalone", "--nproc_per_node=4",
            os.path.abspath(__file__),
            *forwarded,
        ]
        env = os.environ.copy()
        env.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
        env.setdefault("NEURON_RT_VISIBLE_CORES", "0-3")
        for attempt in range(1, 4):
            result = subprocess.run(command, env=env, check=False)
            if result.returncode == 0:
                return
            print(
                f"full-chip attempt {attempt}/3 failed; retrying cached "
                "graphs", flush=True)
        raise SystemExit(result.returncode)
    if args.report or not (args.assign or args.all_reduce or args.lloyd):
        print_target_reports()
    if args.assign:
        bench_assign(
            args.N, args.D, args.K, n_shards=args.n_shards,
            k_resident=args.k_resident, point_lanes=args.point_lanes,
            warmup=args.warmup, iters=args.iters)
    if args.all_reduce:
        bench_distributed_all_reduce(
            args.D, args.K, warmup=args.warmup, iters=args.iters)
    if args.lloyd:
        bench_distributed_lloyd(
            args.N, args.D, args.K,
            iters=args.iters, update_mode=args.update)


if __name__ == "__main__":
    main()
