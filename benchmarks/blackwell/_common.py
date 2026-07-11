"""Shared utilities for the Blackwell flash-kmeans assign benchmarks.

`loom.bench` (cake's timing backend) is not a flashlib dependency, so these
benchmarks use a local cold-L2 CUDA-events timer that mirrors cake's
`bench_gpu_time(cold_l2=True)`: the L2 cache is overwritten before every timed
iteration so each launch reads its inputs from HBM, not from a warm cache.

The shape grid is restricted to the regime the new Blackwell CuteDSL assign
kernel supports (B=1, bf16, D in {64, 128, 256}, N % 128 == 0, K % 256 == 0),
which is also a subset of cake's `flash_kmeans_assign` contract so every shape
runs on both engines.
"""
from __future__ import annotations

import statistics
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Shape grid (N, D, K): B=1, bf16, N % 128 == 0, K % 256 == 0, D in {64,128,256}
# ---------------------------------------------------------------------------
SHAPES: tuple[tuple[int, int, int], ...] = (
    # small / cake-catalog cross-reference shapes
    (2048, 128, 4096),
    (4096, 256, 4096),
    (16384, 128, 256),
    # mid N, headline D=128
    (65536, 128, 1024),
    (65536, 128, 4096),
    (262144, 128, 1024),
    (1048576, 128, 1024),
    # D=64
    (65536, 64, 1024),
    (262144, 64, 4096),
    # D=256
    (65536, 256, 1024),
    (65536, 256, 4096),
)


def device_info() -> tuple[str, tuple[int, int]]:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    return name, cap


def make_l2_flush_buffer(device: str | torch.device = "cuda") -> torch.Tensor:
    """A scratch buffer >= 2x the device L2 so zeroing it evicts the cache."""
    props = torch.cuda.get_device_properties(device)
    l2 = int(getattr(props, "l2_cache_size", 0) or 0)
    nbytes = max(l2 * 2, 256 * 1024 * 1024)
    return torch.empty(nbytes // 4, dtype=torch.int32, device=device)


def bench_gpu_ms(
    fn: Callable[[], object],
    *,
    warmup: int = 5,
    iters: int = 30,
    cold_l2: bool = True,
    flush_buf: torch.Tensor | None = None,
    device: str | torch.device = "cuda",
) -> tuple[float, list[float]]:
    """Median GPU time (ms) over ``iters``, flushing the L2 each iter.

    Returns ``(median_ms, sorted_times_ms)``. The L2 flush happens *before*
    ``start.record()`` so it is never included in the timed window.
    """
    if cold_l2 and flush_buf is None:
        flush_buf = make_l2_flush_buffer(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(iters):
        if cold_l2 and flush_buf is not None:
            flush_buf.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return statistics.median(times), times


def bench_gpu_ms_multi(
    fns: dict,
    *,
    warmup: int = 5,
    iters: int = 50,
    cold_l2: bool = True,
    flush_buf: torch.Tensor | None = None,
    device: str | torch.device = "cuda",
    order: list | None = None,
) -> dict:
    """Interleaved cold-L2 timing of several callables.

    Each iteration times every callable back-to-back (L2 flushed before
    each) so all engines experience the *same* clock/thermal state -- the
    GPU here does not allow clock locking, and timing all iters of engine A
    before engine B makes A look fast (cool) and B slow (hot). Interleaving
    removes that bias. Returns ``{name: (median_ms, sorted_times_ms)}``.
    """
    names = order or list(fns.keys())
    if cold_l2 and flush_buf is None:
        flush_buf = make_l2_flush_buffer(device)
    for nm in names:
        for _ in range(warmup):
            fns[nm]()
    torch.cuda.synchronize()
    samples: dict = {nm: [] for nm in names}
    for _ in range(iters):
        for nm in names:
            if cold_l2 and flush_buf is not None:
                flush_buf.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fns[nm]()
            end.record()
            end.synchronize()
            samples[nm].append(start.elapsed_time(end))
    out = {}
    for nm in names:
        t = sorted(samples[nm])
        out[nm] = (statistics.median(t), t)
    return out


def make_inputs(
    N: int,
    D: int,
    K: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Seeded ``(x, centroids)`` of shape ``(1, N, D)`` / ``(1, K, D)``."""
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    x = torch.randn((1, N, D), dtype=dtype, device=device, generator=g).contiguous()
    c = torch.randn((1, K, D), dtype=dtype, device=device, generator=g).contiguous()
    return x, c


def reference_assign(x: torch.Tensor, c: torch.Tensor, *, chunk: int = 8192) -> torch.Tensor:
    """``argmax(x @ cᵀ - 0.5·‖c‖²)`` == ``argmin ‖x - c‖²`` (the x²-free rank).

    Computed in fp32, chunked over N to bound memory. Returns ``(B, N)`` int32.
    """
    xf = x.float()
    cf = c.float()
    c_sq = (cf * cf).sum(-1)  # (B, K)
    B, N, _ = x.shape
    out = torch.empty((B, N), dtype=torch.int32, device=x.device)
    for b in range(B):
        ct = cf[b].t().contiguous()
        bias = 0.5 * c_sq[b]
        for s in range(0, N, chunk):
            q = xf[b, s : s + chunk]
            sc = q @ ct - bias[None, :]
            out[b, s : s + q.shape[0]] = sc.argmax(-1).int()
    return out


def correctness(
    ids: torch.Tensor,
    ref: torch.Tensor,
    x: torch.Tensor,
    c: torch.Tensor,
    *,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> dict:
    """Exact-id match rate plus a tie-tolerant correctness.

    Near-equal centroid distances make exact argmin disagree under bf16
    rounding (a genuine tie, not a bug). ``tie_ok`` is the fraction of points
    whose *chosen* centroid is no farther than the reference's beyond
    ``atol + rtol·d_ref`` -- i.e. correct up to ties.
    """
    a = ids.view(-1).long()
    b = ref.view(-1).long()
    exact = (a == b).float().mean().item()

    xf = x.float().reshape(-1, x.shape[-1])
    cf = c.float().reshape(-1, c.shape[-1])
    da = ((xf - cf[a]) ** 2).sum(-1)
    db = ((xf - cf[b]) ** 2).sum(-1)
    bad = (da - db) > (atol + rtol * db)
    tie_ok = 1.0 - bad.float().mean().item()
    return {
        "exact_match": exact,
        "tie_ok": tie_ok,
        "mismatch": int((a != b).sum().item()),
        "bad": int(bad.sum().item()),
    }


def tflops(N: int, D: int, K: int, ms: float, *, B: int = 1) -> float:
    """Cross-term GEMM FLOPs (2·B·N·K·D) over the kernel time."""
    return 2.0 * B * N * K * D / ms / 1e9
