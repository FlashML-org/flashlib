"""Isolated CAGRA-kernel driver for ncu roofline profiling.

Builds (and disk-caches) a CAGRA index, then launches exactly one custom
CAGRA Triton kernel so ``ncu --launch-count`` can capture it cleanly:

    sudo -E /opt/nvidia/nsight-compute/<ver>/ncu \\
        --target-processes all --launch-count 1 \\
        --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\\
sm__warps_active.avg.pct_of_peak_sustained_active \\
        python -m benchmarks.profile_cagra --kernel search --N 1000000 --D 128 \\
            --itopk 64 --nq 4096

``--kernel`` is ``search`` (``_cagra_search_kernel``) or ``detour``
(``_detour_prune_kernel``). The index/initial-graph are cached under
``$FLASHLIB_BENCH_CACHE`` (default ``~/.cache/flashlib_bench``) so repeated
ncu passes don't rebuild.
"""
from __future__ import annotations

import argparse
import os

import torch


def _cache_dir() -> str:
    d = os.environ.get("FLASHLIB_BENCH_CACHE",
                       os.path.expanduser("~/.cache/flashlib_bench"))
    os.makedirs(d, exist_ok=True)
    return d


def _data(N, D, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(N, D, generator=g).to("cuda")


def get_index(N, D, gd, inter, seed=0):
    from flashlib.primitives.cagra.index import CagraIndex
    path = os.path.join(_cache_dir(), f"cagra_idx_{N}_{D}_{gd}_{inter}.pt")
    if os.path.exists(path):
        blob = torch.load(path)
        return CagraIndex(
            dataset=blob["dataset"].cuda(), graph=blob["graph"].cuda(),
            graph_degree=gd, intermediate_degree=inter, metric="l2",
            D=D, Dp=D, build_algo=blob.get("algo", "ivf_pq"),
        )
    from flashlib import flash_cagra_build
    X = _data(N, D, seed)
    idx = flash_cagra_build(X, graph_degree=gd, intermediate_degree=inter, seed=seed)
    torch.save({"dataset": idx.dataset.cpu(), "graph": idx.graph.cpu(),
                "algo": idx.build_algo}, path)
    return idx


def get_initial_graph(N, D, inter, seed=0):
    path = os.path.join(_cache_dir(), f"cagra_ginit_{N}_{D}_{inter}.pt")
    if os.path.exists(path):
        return torch.load(path).cuda()
    from flashlib.primitives.cagra.triton.build_graph import build_initial_graph
    X = _data(N, D, seed)
    _, g_init, _ = build_initial_graph(X, inter, seed=seed)
    torch.save(g_init.cpu(), path)
    return g_init


def _search_fn(args):
    if args.kernel == "search_cutedsl":
        from flashlib.primitives.cagra.cutedsl.search_kernel import (
            cagra_search_cutedsl as fn,
        )
    else:
        from flashlib.primitives.cagra.triton.search import (
            cagra_search_triton as fn,
        )
    return fn


def run_search(args):
    fn = _search_fn(args)
    idx = get_index(args.N, args.D, args.gd, args.inter, args.seed)
    g = torch.Generator(device="cpu").manual_seed(1)
    Q = torch.randn(args.nq, args.D, generator=g).to("cuda")
    kw = {}
    if args.kernel == "search_cutedsl" and args.threads > 0:
        kw["threads"] = args.threads
    # Warm (compile) outside the profiled region; ncu --launch-count picks
    # the steady-state launch.
    for _ in range(args.warmup):
        fn(idx, Q, args.k, itopk_size=args.itopk,
           search_width=args.sw, max_iterations=args.max_iters, **kw)
    torch.cuda.synchronize()
    if args.bench > 0:
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.bench):
            fn(idx, Q, args.k, itopk_size=args.itopk,
               search_width=args.sw, max_iterations=args.max_iters, **kw)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / args.bench
        print(f"{args.kernel} N={args.N} D={args.D} itopk={args.itopk} "
              f"thr={args.threads} nq={args.nq}: {ms:.3f} ms/iter  "
              f"{args.nq / ms * 1e3:,.0f} QPS")
    fn(idx, Q, args.k, itopk_size=args.itopk,
       search_width=args.sw, max_iterations=args.max_iters, **kw)
    torch.cuda.synchronize()


def run_detour(args):
    from flashlib.primitives.cagra.triton.optimize import _detour_prune_triton
    g_init = get_initial_graph(args.N, args.D, args.inter, args.seed)
    for _ in range(args.warmup):
        _detour_prune_triton(g_init, args.gd)
    torch.cuda.synchronize()
    _detour_prune_triton(g_init, args.gd)
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="isolated CAGRA kernel for ncu")
    ap.add_argument("--kernel",
                    choices=["search", "search_cutedsl", "detour"],
                    default="search")
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--gd", type=int, default=64)
    ap.add_argument("--inter", type=int, default=128)
    ap.add_argument("--itopk", type=int, default=64)
    ap.add_argument("--sw", type=int, default=1)
    ap.add_argument("--max-iters", dest="max_iters", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--bench", type=int, default=0,
                    help="if >0, time this many search iters and print QPS")
    ap.add_argument("--threads", type=int, default=0,
                    help="CTA width for the cutedsl kernel (0 = default 128)")
    args = ap.parse_args()

    if args.kernel in ("search", "search_cutedsl"):
        run_search(args)
    else:
        run_detour(args)


if __name__ == "__main__":
    main()
