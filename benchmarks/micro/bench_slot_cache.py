"""slot_cache: every selection strategy over (num_cached, K, miss rate).

Part 1 pins FreeToken's actual operating points; part 2 sweeps wide enough to show
where each strategy stops working. Methodology matches bench_topk_vs_seq: CUDA-graph
replay, routing from a device pool, achieved miss rate read back from stats.

    python benchmarks/micro/bench_slot_cache.py
"""
from __future__ import annotations

import torch

from dataclasses import dataclass
from functools import partial

from flashlib.kernels.slot_cache import N_STATS, lru_ensure

CALLS, POOL, COLD = 32, 64, 1 << 21
ITERS, WARMUP = 60, 15
IMPLS = tuple((s, partial(lru_ensure, strategy=s)) for s in ("seq", "topk", "insert"))


@dataclass
class _State:
    """The nine tensors lru_ensure operates on. A real caller aliases its own."""

    num_total: int
    num_cached: int
    k: int
    device: torch.device

    def __post_init__(self):
        dev, plan = self.device, min(self.k, self.num_cached)
        self.slot_of_id = torch.full((self.num_total,), -1, dtype=torch.int32, device=dev)
        self.id_of_slot = torch.full((self.num_cached,), -1, dtype=torch.int32, device=dev)
        self.lru_usage = torch.zeros(self.num_cached, dtype=torch.int64, device=dev)
        self.lru_step = torch.zeros((), dtype=torch.int64, device=dev)
        self.src_indices = torch.empty(plan, dtype=torch.int32, device=dev)
        self.dst_indices = torch.empty(plan, dtype=torch.int32, device=dev)
        self.num_copy = torch.zeros((), dtype=torch.int64, device=dev)
        self.out = torch.empty(self.k, dtype=torch.int32, device=dev)
        self.stats = torch.zeros(N_STATS, dtype=torch.int64, device=dev)

    def call(self, impl, q):
        impl(q, self.slot_of_id, self.id_of_slot, self.lru_usage, self.lru_step,
             self.out, self.src_indices, self.dst_indices, self.num_copy,
             stats=self.stats)


def run(impl, cache, k, hot):
    dev = torch.device("cuda")
    st = _State(COLD, cache, k, dev)
    gen = torch.Generator().manual_seed(0)

    hot_ids = torch.arange(CALLS * hot, dtype=torch.int32).reshape(CALLS, hot)
    pool = torch.empty(POOL, CALLS, k, dtype=torch.int32)
    for t in range(POOL):
        parts = [hot_ids] if hot else []
        if hot < k:
            parts.append(torch.randint(CALLS * k, COLD, (CALLS, k - hot),
                                       generator=gen, dtype=torch.int32))
        pool[t] = torch.cat(parts, dim=1)
    pool = pool.to(dev)
    qbuf = torch.zeros(CALLS, k, dtype=torch.int32, device=dev)
    views = [qbuf[i] for i in range(CALLS)]

    def call():
        for q in views:
            st.call(impl, q)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        qbuf.copy_(pool[0])
        call()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    for t in range(WARMUP):
        qbuf.copy_(pool[t % POOL])
        g.replay()
    torch.cuda.synchronize()
    st.stats.zero_()

    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for t in range(ITERS):
        qbuf.copy_(pool[t % POOL])
        g.replay()
    b.record()
    torch.cuda.synchronize()
    us = a.elapsed_time(b) * 1000 / (ITERS * CALLS)
    return us, int(st.stats[1].item()) / max(int(st.stats[2].item()), 1)


def row(cache, k, hot, label=""):
    cells, miss = {}, 0.0
    for name, impl in IMPLS:
        try:
            cells[name], miss = run(impl, cache, k, hot)
        except Exception:
            cells[name] = float("inf")
    best = min(cells, key=cells.get)
    vals = " ".join(f"{cells[n]:>8.2f}" if cells[n] < 1e9 else f"{'FAIL':>8}"
                    for n, _ in IMPLS)
    print(f"{label:>10} {cache:>7} {k:>4} {miss:>6.2f} {vals}  {best}")


def header(title):
    print(f"\n=== {title} ===")
    print(f"{'':>10} {'cache':>7} {'K':>4} {'miss':>6} "
          f"{'seq':>8} {'topk':>8} {'insert':>8}  best")


def main() -> None:
    header("1. warm working point (36 partitions x 128 ids)")
    for cache in (2073, 4608):
        for k in (8, 4):
            for hot, tag in ((k, "warm"), (k - 1, "steady"), (k // 2, "half"), (0, "cold")):
                row(cache, k, hot, tag)

    header("2a. general: num_cached sweep (K=8)")
    for cache in (1024, 4096, 16384, 65536, 262144):
        for hot, tag in ((8, "0% miss"), (4, "50% miss"), (0, "100% miss")):
            row(cache, 8, hot, tag)

    header("2b. general: K sweep (num_cached=4096)")
    for k in (8, 32, 128):
        for hot, tag in ((k, "0% miss"), (k // 2, "50% miss"), (0, "100% miss")):
            row(4096, k, hot, tag)


if __name__ == "__main__":
    main()
