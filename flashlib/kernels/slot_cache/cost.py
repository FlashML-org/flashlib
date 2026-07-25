"""Cost model + strategy routing for ``lru_ensure``.

Fitted to H100 measurements, not roofline: the kernel is one CTA on one SM and is
latency-bound end to end -- quadrupling the thread count changes nothing, so peak
bandwidth and peak FLOPs both predict the wrong thing. Hence ``bound="latency"`` and
``confidence="measured"``.

    T(num_cached, K, m) = L + P1(K) + [m > 0] * (E(C) + m * S(C))

    L      ~ 0.98us     launch, independent of everything (an empty kernel measures this)
    P1(K)  = 0.43 / 0.49 / 1.21 / 3.84 / 57.5 / 985us at K = 8 / 32 / 64 / 128 / 256 / 512
             -- the [K, K] dedup block; the cliff at 256 is a register spill
    E(C)   ~ 0.6-1.0us  entering the victim scan (load + fence)
    S(C)   = 0.44 + 0.026 * (C / 256) us per miss, and *not* reducible by more warps
"""
from __future__ import annotations

from flashlib.info.estimate import Estimate

# P1(K), measured; interpolate in log2(K) between these and extrapolate as K^2 past 128.
_PHASE1_US = {8: 0.43, 16: 0.46, 32: 0.49, 64: 1.21, 128: 3.84, 256: 57.5, 512: 985.0}
_LAUNCH_US = 0.98
_ENTER_US = 0.8

# num_cached past which the register-resident strategies stop paying (and, further up,
# stop compiling); the streaming one degrades ~1.6x per doubling instead of ~2.2x.
STREAMING_THRESHOLD = 24_000


def _phase1_us(k: int) -> float:
    if k in _PHASE1_US:
        return _PHASE1_US[k]
    lo = max((x for x in _PHASE1_US if x <= k), default=8)
    hi = min((x for x in _PHASE1_US if x >= k), default=512)
    if lo == hi:
        return _PHASE1_US[lo] * (k / lo) ** 2
    f = (k - lo) / (hi - lo)
    return _PHASE1_US[lo] + f * (_PHASE1_US[hi] - _PHASE1_US[lo])


def _per_miss_us(num_cached: int) -> float:
    return 0.44 + 0.026 * (num_cached / 256.0)


def estimate(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    """``shape = (num_total, num_cached, K)``; ``params={"num_missing": m}`` optional.

    ``m`` defaults to ``K / 8`` -- a 12.5% miss rate, roughly what a warm MoE expert
    cache runs at.
    """
    if len(shape) != 3:
        raise ValueError("slot_cache shape must be (num_total, num_cached, K)")
    num_total, num_cached, k = shape
    params = params or {}
    m = params.get("num_missing", max(1.0, k / 8.0))

    us = _LAUNCH_US + _phase1_us(k)
    if m > 0:
        us += _ENTER_US + m * _per_miss_us(num_cached)

    # Metadata only: the payload is the caller's, and the kernel never reads it.
    bytes_moved = 4 * k + 12 * num_cached + 8 * k
    return Estimate(
        op_name="slot_cache",
        runtime_ms=us / 1000.0,
        flops=0,
        bytes_moved=bytes_moved,
        memory_peak_gb=(4 * num_total + 12 * num_cached) / 1e9,
        bound="latency",
        confidence="measured",
        n_kernel_launches=1,
        suggested_config=recommend(shape, params, tol, dtype, device),
        notes=[
            f"num_total={num_total} num_cached={num_cached} K={k} num_missing={m:g}",
            "single CTA; more warps does not help (measured flat from 4 to 16)",
            f"{100 * _LAUNCH_US / us:.0f}% of this is launch overhead",
        ],
        expected_residual=None,
        precision_tier=None,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    """Which selection strategy ``lru_ensure`` will use for this shape, and why.

    Informational: the router picks the same thing on its own. Pass the result's
    ``strategy`` back via ``lru_ensure(..., strategy=...)`` only to override it.

    Measured crossovers (H100):

    * ``num_cached >= 24K`` -- only ``lru_ensure_insert`` still scales; the other two hold
      the cache in registers and spill (``lru_ensure_topk`` stops compiling past ~64K).
      At 65K slots it is 2.2x ``topk`` and 3.3x the sequential scan.
    * below that, cost is decided by the miss count: the sequential scan is
      ``floor + slope * m`` while the select is flat, crossing near ``m = 0.5 * K``.
    * miss rate unknown -> ``lru_ensure_auto`` branches at runtime, measured within 1-4%
      of whichever would have won.
    """
    _, num_cached, k = shape
    params = params or {}
    m = params.get("num_missing")

    if num_cached >= STREAMING_THRESHOLD:
        return {"strategy": "insert", "block_c_tile": 2048,
                "why": "register-resident strategies spill past ~24K slots"}
    if m is None:
        return {"strategy": "auto", "thresh": max(2, k // 4),
                "why": "miss count unknown; the kernel branches on it at runtime"}
    if m < 0.5 * k:
        return {"strategy": "seq", "why": f"m={m:g} below the ~0.5*K crossover"}
    return {"strategy": "topk", "why": f"m={m:g} above the ~0.5*K crossover"}
