"""Cost model + strategy routing for ``lru_ensure``.

Fitted to H100 measurements, not roofline: the kernel is one CTA on one SM and is
latency-bound end to end -- quadrupling the thread count changes nothing, so peak
bandwidth and peak FLOPs both predict the wrong thing. Hence ``bound="latency"`` and
``confidence="measured"``.

The numbers describe **CUDA-graph replay**, which is how this kernel is meant to run. An
eager Triton launch adds ~20us of host-side dispatch -- several times the kernel itself,
and enough to hide the entire difference between the two strategies. Add it back if the
caller launches eagerly.

    T(num_cached, K, m) = P1(K) + SELECT(num_cached, m)

    P1(K)   the ``[K, K]`` dedup block: flat to K=128, then spills hard (+55us at 256)
    SELECT  ~1.3us when m == 0 -- both strategies early out -- and otherwise
              seq     base(C) + slope(C) * m, both terms growing with C
              insert  flat(C), independent of m but roughly linear in MAX_K
"""
from __future__ import annotations

from flashlib.info.estimate import Estimate

# Routing, mirrored in the kernel module. Streaming needs *both*: a cache too large for a
# register-resident scan, and a query narrow enough that its MAX_K-wide insert loop stays
# cheap. Measured crossovers below.
STREAMING_THRESHOLD = 40_000
STREAMING_MAX_K = 8

# Phase 1 over the K=8 baseline, at num_cached=512, m=1.
_P1_EXTRA_US = {8: 0.0, 16: 0.05, 32: 0.02, 64: 1.25, 128: 4.17, 256: 55.3, 512: 195.7}

# Phase 2, K=8. seq: (base, slope) of base + slope * m. The 4x jumps at 24K and 48K are
# where the [BLOCK_C] key block spills to local memory.
_SEQ_US = {
    512: (1.6, 0.56), 1024: (1.8, 0.63), 2048: (2.0, 0.70), 4096: (2.9, 0.89),
    8192: (3.0, 1.26), 16384: (3.6, 2.11), 24576: (8.0, 6.22), 32768: (8.4, 6.22),
    49152: (21.7, 14.8), 65536: (23.0, 14.9), 131072: (21.9, 30.0),
}
# insert: flat in m, ~linear in num_cached (it tiles, so it never spills).
_INSERT_US = {
    512: 5.3, 1024: 6.0, 2048: 6.9, 4096: 11.8, 8192: 19.3, 16384: 31.8,
    24576: 39.9, 32768: 50.8, 49152: 65.9, 65536: 75.0, 131072: 103.8,
}
_EARLY_OUT_US = 1.33  # m == 0: phase 2 is skipped entirely
_EAGER_LAUNCH_US = 20.0  # Triton's host-side dispatch, if not replaying a graph


def _interp(table, x, log=True):
    """Linear between the bracketing measured points (in log2 x by default)."""
    if x in table:
        return table[x]
    lo = max((k for k in table if k <= x), default=min(table))
    hi = min((k for k in table if k >= x), default=max(table))
    if lo == hi:
        return table[lo]
    import math
    f = ((math.log2(x) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) if log
         else (x - lo) / (hi - lo))
    a, b = table[lo], table[hi]
    if isinstance(a, tuple):
        return tuple(p + f * (q - p) for p, q in zip(a, b))
    return a + f * (b - a)


def _select_us(strategy, num_cached, k, m):
    if m <= 0:
        return _EARLY_OUT_US
    if strategy == "insert":
        # MAX_K is the next power of two at or above 8; the loop cost tracks it.
        max_k = max(8, 1 << (k - 1).bit_length())
        return _interp(_INSERT_US, num_cached) * (max_k / 8.0)
    base, slope = _interp(_SEQ_US, num_cached)
    return base + slope * m


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

    rec = recommend(shape, params, tol, dtype, device)
    us = _interp(_P1_EXTRA_US, k) + _select_us(rec["strategy"], num_cached, k, m)

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
        suggested_config=rec,
        notes=[
            f"num_total={num_total} num_cached={num_cached} K={k} num_missing={m:g}",
            f"strategy={rec['strategy']}; single CTA, and more warps does not help "
            f"(measured flat from 4 to 16)",
            f"CUDA-graph replay; an eager launch adds ~{_EAGER_LAUNCH_US:.0f}us of "
            f"Triton dispatch on top",
        ],
        expected_residual=None,
        precision_tier=None,
        tol=tol,
    )


def recommend(shape, params=None, tol=None, dtype="int32", device="H100", **_):
    """Which selection strategy ``lru_ensure`` will use for this shape, and why.

    Informational: the router picks the same thing from the same two constants. Pass the
    result's ``strategy`` back via ``lru_ensure(..., strategy=...)`` only to override it.

    The router sees only the shape, but the real crossover also moves with the miss
    count, so when ``num_missing`` is known this reports whether the other strategy
    would have been cheaper. Measured on H100:

    * ``num_cached < 40K`` -- the sequential scan wins at every miss count. Its block is
      register-resident, and below that size it does not spill.
    * ``num_cached >= 40K`` **and** ``K <= 8`` -- streaming wins once the miss count
      passes roughly ``K/2``; below that the scan is still ahead, so a shape-only router
      is choosing on the worst case, which streaming bounds far better (flat vs
      ``15us * m`` at 64K slots).
    * ``K > 8`` -- the scan wins regardless of cache size: streaming's insert loop is
      linear in MAX_K and reaches 1.5ms at K=128, 36x the scan.
    """
    _, num_cached, k = shape
    params = params or {}
    m = params.get("num_missing")

    streaming = num_cached >= STREAMING_THRESHOLD and k <= STREAMING_MAX_K
    if streaming:
        why = f"num_cached>={STREAMING_THRESHOLD} with K<={STREAMING_MAX_K}"
    elif num_cached >= STREAMING_THRESHOLD:
        why = f"K={k} above {STREAMING_MAX_K}: the insert loop is linear in MAX_K"
    else:
        why = f"num_cached<{STREAMING_THRESHOLD}: the scan does not spill yet"

    out = {"strategy": "insert" if streaming else "seq", "why": why}
    if streaming:
        out["block_c_tile"] = 2048
    if m is not None:
        other = "seq" if streaming else "insert"
        mine_us = _select_us(out["strategy"], num_cached, k, m)
        other_us = _select_us(other, num_cached, k, m)
        if other_us < mine_us:
            out["note"] = (f"at num_missing={m:g}, strategy={other!r} would be "
                           f"{mine_us / other_us:.1f}x faster; pass it explicitly")
    return out
