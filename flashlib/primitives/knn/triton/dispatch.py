"""flash_knn -- host-side dispatcher: kernel/config selection + 2-stage launch.

Backs the public :func:`flashlib.primitives.knn.flash_knn` entry point.

Top-level wrappers
------------------

* :func:`flash_knn_triton` -- one-shot kNN, returns ``(B, N, k)`` int32
  indices. The shape-only heuristic now handles both build (large-N,
  single-pass per CTA) and search (small-N, M-split flash-decode)
  without any host-side SM-saturation gate -- ``ctas_no_split`` is
  inspected after BN is picked, so build shapes (``BN=128`` ->
  ``ctas_no_split`` already saturates 132 SMs) and search shapes
  (small BN -> M-split for tail-fill) both fall out of the same
  decision tree.
* :func:`flash_knn_triton_small_n` / :func:`flash_knn_triton_large_n`
  -- explicit overrides for callers (e.g. K-means assign) that know
  their shape category at construction time.
* :func:`_heuristic_config`, :func:`_autotune` -- shape-only heuristic
  (default, fast first call) + opt-in brute-force autotune (slower
  first call, cached steady-state). Both share the same kernel grid;
  ``force_path="large_n"`` collapses M-splits down to one per CTA.

Two kernels live behind this dispatcher:

* :mod:`insert`    -- iterative argmin-insert top-K. Picked by the
  heuristic on virtually every shape.
* :mod:`sortmerge` -- packed-uint64 sort-merge top-K with the IEEE-
  sortable u32 transform. Routed to for the **Pattern-A small-Q**
  corner (``B*N <= 8 or (B*N <= 16 and K in {32, 64})``) at medium-K
  + ``M <= 200K`` where the wider sort per chunk beats insert's
  argmin loop on this regime's autotune data. Otherwise insert wins;
  sortmerge is kept in :func:`_gen_configs` so the offline tuner can
  keep verifying that.

Pipelining depth
----------------
Both kernels take a ``NUM_STAGES_PIPE`` constexpr for the M-loop's
``tl.range`` pipelining factor. The dispatcher tunes it per shape from
a 201-shape ns-sweep (upstream ``bench/sweep_ns_by_k.py``): K<=8 +
small tile wants ``ns=3`` to hide HBM latency, K>=64 + big tile wants
``ns=2`` (registers tight), wide-D wants ``ns=2`` uniformly, narrow
tile + anything wants ``ns=1`` to avoid spills. See the comments
inside :func:`_heuristic_config` for the empirical breakdown.

Distance recovery
-----------------
The kernels emit **signed shifted scores** ``s = c_sq - 2*<x, c>``, not
true squared L2. ``argmin-k`` over ``s`` matches ``argmin-k`` over true
squared L2 (since ``-||x||^2`` is a per-row constant), but the returned
score is not a valid distance. The :func:`flash_knn` public wrapper
calls :func:`flashlib.kernels.distance.triton_knn_gather_sqdist` on the
returned indices to write the true ``||x - c[idx]||^2`` per neighbour.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from flashlib import _hw
from flashlib.primitives.knn.triton._common import _next_pow2, _bench_quick
from flashlib.primitives.knn.triton.sortmerge import _flash_knn_sortmerge_kernel
from flashlib.primitives.knn.triton.insert import _flash_knn_insert_kernel


# ── SRAM estimation ────────────────────────────────────────────────────


def _estimate_sram(bn, bm, D, K, d_inner, num_stages, *,
                   sortmerge=False, dtype_bytes=2):
    """Per-CTA SRAM estimate (bytes) for the unified kernel — coarse.

    Counts only what Triton **definitely** keeps in SMEM: the x tile
    + c tile (per iteration of the inner D loop). Everything else
    (cross fp32 accumulator, score tile, topK heap, sort scratch) is
    register-resident at the tile sizes used by the dispatcher's
    heuristic, and the NUM_STAGES_PIPE multi-buffering of the c tile
    is sometimes elided by Triton's pipeliner. Including any of these
    in the estimate produces 50-200 KB of over-count that **wrongly
    shrinks** configs the kernel could actually launch — a bf16
    regression demonstrated by the ``D=256`` cells in
    ``benchmarks/results/micro_knn_dtype_smem_fit.md``.

    This estimator is therefore **optimistic on purpose**. Its only
    job is to bias the autotune candidate filter away from configs
    that obviously can't fit (e.g. ``BN=128, BM=128, D=256, fp32``
    needs >232 KB by raw arithmetic). The source of truth for "does
    this fit?" is ``_run``'s runtime ``OutOfResources`` catch + shrink
    loop. ``dtype_bytes`` is still threaded through so fp32 inputs
    correctly double the x/c contribution.
    """
    x_sub = bn * d_inner * dtype_bytes
    c_sub = bm * d_inner * dtype_bytes
    c_sq = bm * 4
    base = x_sub + c_sub + c_sq
    if sortmerge:
        chunk = bn * bm * 8
        base += chunk
    return base


def _smem_limit(device) -> int:
    """Per-block dynamic shared-memory budget for ``device``.

    Mirrors :func:`flashlib.primitives.kmeans.triton.assign._smem_limit`
    — Triton uses opt-in dynamic SMEM; prefer that attribute when
    available, fall back to static limit, and finally to a conservative
    48 KiB for old PyTorch builds.
    """
    props = torch.cuda.get_device_properties(device)
    for attr in (
        "shared_memory_per_block_optin",
        "max_shared_memory_per_block_optin",
        "shared_memory_per_block",
        "max_shared_memory_per_block",
    ):
        v = getattr(props, attr, None)
        if v:
            return int(v)
    return 48 * 1024


def _min_bn(device) -> int:
    """Smallest legal BN: ``tl.dot`` needs M >= 16 on Blackwell (Triton pads M=8 to 16 only on sm_90)."""
    return 16 if _hw.current(device.index).is_blackwell else 8


def _fit_config_to_smem(cfg: dict, D: int, K: int,
                        dtype_bytes: int, smem_limit: int, min_bn: int = 8) -> dict:
    """Shrink ``(BN, BM, NUM_STAGES_PIPE)`` until the kernel fits SMEM.

    Returns ``cfg`` unchanged when it already fits. Otherwise enumerates
    power-of-two reductions on the three SMEM-bearing axes and picks the
    one that maximises ``BN * BM * num_stages`` (work-per-program tile)
    among those that fit, breaking ties towards the original aspect ratio.

    Why only these three axes?
      * ``D_INNER``: changing it flips between single-D-tile and D-split
        paths, which would also change the kernel's perf profile. We
        keep the heuristic's D-split decision and shrink the cheaper
        axes first.
      * ``TOPK_PAD``: determined by ``K``; cannot be reduced without
        breaking correctness.
      * ``kernel_mode``, ``num_warps``, ``M_PER_SPLIT``, ``NUM_SPLITS``:
        these don't affect per-CTA SMEM. Stable.

    Pattern (and rationale) mirrors :func:`flashlib.primitives.kmeans.\
triton.assign._fit_config_to_smem`.
    """
    BN0 = int(cfg["BN"])
    BM0 = int(cfg["BM"])
    D_INNER = int(cfg["D_INNER"])
    NS0 = int(cfg.get("NUM_STAGES_PIPE", 2))
    sortmerge = cfg.get("kernel_mode") == "sortmerge"

    cur = _estimate_sram(BN0, BM0, D, K, D_INNER, NS0,
                         sortmerge=sortmerge, dtype_bytes=dtype_bytes)
    if cur <= smem_limit:
        return cfg

    def _pow2_down_to(v, lo):
        out = []
        x = v
        while x >= lo:
            out.append(x)
            x //= 2
        return out

    # BN must stay >= min_bn (matches _gen_configs lowest BN rung);
    # BM must stay >= 32 for the inner loop to keep WGMMA shape;
    # NUM_STAGES_PIPE must stay >= 1.
    bn_cands = _pow2_down_to(BN0, min_bn)
    bm_cands = _pow2_down_to(BM0, 32)
    ns_cands = list(range(NS0, 0, -1))

    best = None
    best_key = None
    for bn in bn_cands:
        for bm in bm_cands:
            # sortmerge requires BM == TOPK_PAD; do not shrink BM there.
            if sortmerge and bm != BM0:
                continue
            for ns in ns_cands:
                if _estimate_sram(bn, bm, D, K, D_INNER, ns,
                                  sortmerge=sortmerge,
                                  dtype_bytes=dtype_bytes) > smem_limit:
                    continue
                aspect_penalty = abs((bn / max(bm, 1)) - (BN0 / max(BM0, 1)))
                # Prefer: larger total tile work, closer aspect ratio,
                # larger BN (more N-parallelism), larger NS (better pipelining).
                key = (bn * bm * ns, -aspect_penalty, bn, ns)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (bn, bm, ns)

    if best is None:
        raise RuntimeError(
            f"flash_knn: cannot fit kernel into shared memory "
            f"(D={D}, K={K}, D_INNER={D_INNER}, dtype_bytes={dtype_bytes}, "
            f"smem_limit={smem_limit}). Original config "
            f"BN={BN0}, BM={BM0}, NS={NS0} needs {cur} bytes."
        )

    bn, bm, ns = best
    out = dict(cfg)
    out["BN"] = bn
    out["BM"] = bm
    out["NUM_STAGES_PIPE"] = ns
    return out


def _pipe_stages_for(K, d_inner, D):
    """``NUM_STAGES_PIPE`` candidates for the autotune M-loop sweep.

    The K-step inner loop adds register pressure that scales with K, so
    deeper pipelining (which double-/triple-buffers the next C-tile) is
    only profitable when K is small. Empirically:

      * D-split path (``d_inner < D``) -- kernel uses ns=1 by construction.
      * K >= 64 -- only try {1, 2}; ns >= 3 spills registers and tanks perf
        (probe showed ns=3 going from 750 -> 1220 us at K=128).
      * K <= 32 -- try {1, 2, 3} so big pipelines can hide HBM latency
        for huge-M shapes.
    """
    if d_inner < D:
        return [1]
    if K >= 64:
        return [1, 2]
    return [1, 2, 3]


def _gen_configs(D, K, *, dtype_bytes=2, smem_limit=None, min_bn=8):
    """Generate candidate kernel configs (sortmerge + insert), SRAM-validated.

    Covers ``BN ∈ {8, 16, 32, 64, 128}`` × ``num_warps ∈ {2, 4}`` × the
    BM / D_INNER / NUM_STAGES_PIPE combinations that fit per-CTA SMEM
    on H200.

    The ``BN = 8`` rung unlocks the small-N + medium-K Pattern-A regime
    (``B*N <= 8`` + ``K ∈ {16, 32, 64, 128}``) where a small row tile +
    sortmerge ``BM = max(32, K)`` wins over insert. ``tl.dot`` silently
    pads the M dimension to 16 on Triton 3.x sm_90, so BN=8 still
    compiles there; ``min_bn=16`` drops that rung where it does not
    (Blackwell).

    ``dtype_bytes`` and ``smem_limit`` parameterise the SMEM check so
    fp32 inputs don't slip through with a config that OOR's at launch.
    Defaults preserve the pre-fix behaviour (bf16 / 220 KB) for any
    caller that didn't yet plumb dtype awareness through.
    """
    # 220 KB stays the default — slightly below the 227 KB H200 opt-in
    # limit to leave room for static SMEM the compiler adds.
    max_sram = 220_000 if smem_limit is None else smem_limit
    d_pad = _next_pow2(D)
    topk_pad_sm = max(32, _next_pow2(K))
    topk_pad_ins = _next_pow2(K)
    d_inner_cands = [d_pad] if D <= 256 else [128]

    configs = []

    # sort-merge: BM == TOPK_PAD; BN ∈ {8, 16, 32} (BN >= 64 never wins
    # for sortmerge in autotune data).
    bm_sm = topk_pad_sm
    for bn in [b for b in (8, 16, 32) if b >= min_bn]:
        for nw in [2, 4]:
            for d_inner in d_inner_cands:
                num_d_iters = math.ceil(D / d_inner)
                ns = 2 if num_d_iters == 1 else 1
                sram = _estimate_sram(
                    bn, bm_sm, D, K, d_inner, ns,
                    sortmerge=True, dtype_bytes=dtype_bytes,
                )
                if sram <= max_sram:
                    for ns_pipe in _pipe_stages_for(K, d_inner, D):
                        configs.append({
                            "BN": bn, "BM": bm_sm, "D_INNER": d_inner,
                            "num_warps": nw,
                            "TOPK_PAD": topk_pad_sm,
                            "kernel_mode": "sortmerge",
                            "NUM_STAGES_PIPE": ns_pipe,
                        })

    # iterative insert: BM decoupled from K, BN ∈ {8, 16, 32, 64, 128}.
    for bn in [b for b in (8, 16, 32, 64, 128) if b >= min_bn]:
        for nw in [2, 4]:
            for bm in [64, 128, 256]:
                for d_inner in d_inner_cands:
                    num_d_iters = math.ceil(D / d_inner)
                    ns = 2 if num_d_iters == 1 else 1
                    sram = _estimate_sram(
                        bn, bm, D, K, d_inner, ns,
                        sortmerge=False, dtype_bytes=dtype_bytes,
                    )
                    if sram <= max_sram:
                        for ns_pipe in _pipe_stages_for(K, d_inner, D):
                            configs.append({
                                "BN": bn, "BM": bm, "D_INNER": d_inner,
                                "num_warps": nw,
                                "TOPK_PAD": topk_pad_ins,
                                "kernel_mode": "insert",
                                "NUM_STAGES_PIPE": ns_pipe,
                            })

    return configs


def _gen_m_splits(M, BM, *, B=1, N=1, BN=16):
    """M_PER_SPLIT candidates targeting {1, 2, 4, 8, 16} waves on 132 SMs.

    The wave count is critical -- Stage-2 reduce scales linearly with
    NUM_SPLITS, while too few splits leaves SMs idle. The autotune
    sweep confirms that wave=2 is the winner on huge-M + medium-N
    shapes (e.g. ``1×128×10M×64×10``), which the original {1, 4, 16}
    grid missed.

    Capped at 4096×BM tiles (``MAX_MPS_TILES``) to bound the static
    M-loop trip count for fast Triton compilation. Also includes the
    single-pass case when M itself fits the cap (subsumes the large-N
    kernel as ``num_splits = 1``).
    """
    NUM_SMS = 132
    MAX_MPS_TILES = 4096
    num_n_tiles = max(1, (N + BN - 1) // BN)
    max_mps = min(M, MAX_MPS_TILES * BM)
    candidates = set()
    for waves in [1, 2, 4, 8, 16]:
        target_splits = max(2, (NUM_SMS * waves) // (num_n_tiles * B))
        target_splits = min(target_splits, max(2, M // BM))
        mps = (M + target_splits - 1) // target_splits
        mps = ((mps + BM - 1) // BM) * BM
        mps = max(mps, BM)
        mps = min(mps, max_mps)
        candidates.add(mps)
    if M <= max_mps:
        single = ((M + BM - 1) // BM) * BM
        candidates.add(single)
    return sorted(candidates)


# ── shape-only heuristic ───────────────────────────────────────────────


def _heuristic_config(B, N, M, D, K, *, force_path=None,
                      dtype_bytes=2, smem_limit=None, min_bn=8):
    """Pick a kernel config from shape alone -- no autotune.

    When ``dtype_bytes`` and ``smem_limit`` are supplied (the dispatcher
    plumbs them from ``x.element_size()`` and the device's opt-in SMEM
    cap), the picked config is post-processed by :func:`_fit_config_to_smem`
    so that fp32 inputs at large D never produce a config that OOR's at
    launch.

    Derived from a 92-shape autotune sweep on H200/bf16. The data
    showed clean decision boundaries on three axes:

      * **BN by NB-bucket + M-bump**:
        NB <= 8   -> BN=8;
        NB <= 32  -> BN=16 (32 at M>=5M);
        NB <= 256 -> BN=64 (128 at M>=5M + narrow-D + small-K);
        NB <= 2K  -> BN=64 (128 at M>=5M);
        NB >= 8K  -> BN=128 (64 for D>=256 + K>=32).

        The M-bump rule fixes the ``(1, 128, 10M, 64, 10)`` regression
        identified by flashlib -- at M=10M each extra N-tile costs
        ~300us of c-replication HBM traffic.

      * **BM by K**:
        K<=4         -> 128/256 (256 only at very-small M + small NB);
        K=5..16      -> 128 default (256 at huge-M);
        K=32         -> 128 default (256 at NB<=8 + M<=200K);
        K>=64        -> 64 default (sortmerge at NB<=8).

      * **kernel_mode**:
        ``sortmerge`` only when ``NB <= 8`` and ``K ∈ {32, 64, 128}``
        and ``D <= 256`` and ``M <= 200K`` (Pattern-A1), plus a
        secondary ``NB <= 16 + K ∈ {32, 64} + M <= 200K`` corner
        (Pattern-A2). All other shapes use ``insert``.

      * **num_warps**: 4 everywhere except tiny tiles (BN=8 + BM<=64 +
        K<=4 + huge M, where nw=2 wins by reducing register pressure)
        and the BN=16 + huge-M + small-K corner where nw=2 frees a
        warp set for the topK epilogue.

      * **target waves**: 1 for build (NB>=8K) and very-large K;
        2 for medium/large M; 4 for small M + small K.

      * **NUM_STAGES_PIPE**: 2 by default; 3 for BN>=64 + BM<=128 +
        K<=8 (small tile + tiny K hides HBM latency) and for
        BN ∈ [16, 32] + BM=256 + K>=64 (big tile keeps K loop fed);
        1 for narrow-tile + any K (avoid register spills) and for the
        D-split path; 2 for D_INNER>=256 and for BN>=128. Without
        this rule, K>=64 + D=256 shapes hit catastrophic regressions
        (e.g. ``(1,16,100K,256,128)`` 439us -> 2429us at ns=1, a 5.5x
        slowdown); conversely K<=32 + huge-M shapes save 20-45% vs
        ns=2 (e.g. ``(1,1,10M,128,4)`` 1426us -> 785us at ns=1).

    Args:
        B, N, M, D, K: shape parameters.
        force_path: ``"large_n"`` for single-pass (one M-split per CTA),
            ``None`` for the wave-targeted M-split.
        min_bn: lower bound on ``BN`` (16 on Blackwell, see :func:`_min_bn`).

    Returns:
        Config dict: ``BN, BM, D_INNER, TOPK_PAD, kernel_mode,
        num_warps, M_PER_SPLIT, NUM_SPLITS, NUM_STAGES_PIPE``.
    """
    NUM_SMS = 132
    NB = N * B
    TOPK_PAD = _next_pow2(K)
    D_INNER = _next_pow2(D) if D <= 256 else 128
    is_large_n = (force_path == "large_n")
    # MAX_MPS_TILES bounds the per-CTA M-loop iteration count. The
    # original cap of 512 was too tight for huge M with BM=128 (cap
    # = 65K), which forced fewer splits than autotune wanted. Bump
    # to 4096 so we never bottleneck on this cap.
    MAX_MPS_TILES = 4096

    # ───────────────────────────────────────────────────────────────
    # Step 1: kernel_mode, BN, BM, num_warps
    # ───────────────────────────────────────────────────────────────
    if NB <= 8:
        # Tiny query: BN=8 default.
        BN = 8
        # Sortmerge fast-path: K ∈ {32, 64, 128}, M <= 200K. Now
        # extended to ANY D (autotune ``(1,1,100K,1024,32)`` picks
        # BN=8 BM=32 sort).
        if (not is_large_n) and K in (32, 64, 128) and M <= 200_000:
            BM = max(32, _next_pow2(K))
            sram = _estimate_sram(8, BM, D, K, D_INNER,
                                  num_stages=(2 if D_INNER >= D else 1),
                                  sortmerge=True)
            if sram <= 220_000:
                kernel_mode = "sortmerge"
                num_warps = 2 if BM >= 128 else 4
            else:
                kernel_mode = "insert"
                BM = 256 if M <= 200_000 else 128
                num_warps = 4
        else:
            kernel_mode = "insert"
            if K <= 4:
                # K=4 + D>=256 + huge M -> BM=64 nw=2 (autotune
                # ``(1,1,1M,256,4)`` picks this). HBM-bound; BM=256
                # adds wasted SMEM.
                if D >= 256 and M >= 500_000:
                    BM = 64
                else:
                    BM = 256
            elif K <= 16:
                # D>=256 + K=16 -> BM=128 (autotune (1,1,100K,256,16))
                if D >= 256 and K >= 16:
                    BM = 128
                else:
                    BM = 256 if (M >= 500_000 or K >= 16) else 128
            elif K == 32:
                # autotune (1,1,1M,256,32) prefers BM=128 at D>=256.
                # At narrower D, BM=256 wins.
                if D >= 256 and M >= 500_000:
                    BM = 128
                else:
                    BM = 256
            elif K <= 64:
                BM = 64
            else:  # K=128
                BM = 256
            # nw=2 for small/medium-D BM=64 cases
            if BM == 64 and K <= 4 and D >= 512 and M >= 500_000:
                # D >= 512 hits the D-split path (D_INNER capped at 128);
                # nw=2 was calibrated for this regime at (1, 1, 1M, 256, 4)
                # but later re-benching at D=256 showed nw=4 ns=1 wins by
                # 1.48-1.81x there (single-D-iter, multibuffered C). Keep
                # nw=2 only for the D-split path; D=256 falls to the
                # default nw=4 below and gets the ns=1 override in Step 3.
                num_warps = 2
            elif BM >= 128 and K <= 4 and D <= 64 and M >= 5_000_000:
                # NB<=8 + K<=4 + narrow-D + huge-M: nw=2 wins by 20-30 %.
                # 4 warps split the SM register file too thinly for the
                # K=4 argmin-insert epilogue at BM>=128, so the compiler
                # spills; halving the warp count doubles regs/warp and
                # lets ILP recover. Verified at D=64 across M ∈ [5M, 60M]
                # (D=128 strictly prefers nw=4 at this BM; do not bump
                # the threshold without re-benching).
                num_warps = 2
            else:
                num_warps = 4

    elif NB <= 32:
        kernel_mode = "insert"
        # Sortmerge also wins at NB <= 16 + K ∈ {32, 64} (autotune
        # ``(1,16,100K,256,32)`` picks BN=8 BM=32 sortmerge).
        if (not is_large_n) and NB <= 16 and K in (32, 64) and M <= 200_000:
            BN = 8
            BM = max(32, _next_pow2(K))
            sram = _estimate_sram(8, BM, D, K, D_INNER,
                                  num_stages=(2 if D_INNER >= D else 1),
                                  sortmerge=True)
            if sram <= 220_000:
                kernel_mode = "sortmerge"
                num_warps = 4
            else:
                BN, BM = 16, 256
        if kernel_mode == "insert":
            if M >= 5_000_000 and N >= 32:
                BN = 32
            else:
                BN = max(8, min(16, _next_pow2(N)))
            if K <= 4:
                BM = 128
            elif K <= 16:
                BM = 256 if M <= 2_000_000 else 128
            else:  # K >= 32
                BM = 256
            if K <= 4 and M >= 500_000:
                num_warps = 2
            elif K <= 16 and M >= 5_000_000 and BN <= 16:
                num_warps = 2
            elif K <= 16 and NB >= 17 and M >= 5_000_000:
                # autotune (1,32,10M,64,8) BN=32 nw=2 wins
                num_warps = 2
            else:
                num_warps = 4

    elif NB <= 256:
        # Medium queries: BN must scale with M to control
        # c-replication, but ONLY when K is very small (K<=4) or
        # very large (K>=32). At K ∈ {5..16} the autotune consistently
        # picks BN=64 even at M=10M, because K=8..16 hits a sweet
        # spot where the topk-insert + Stage-2 cost of 1-wave
        # BN=128 outweighs the c-HBM savings.
        kernel_mode = "insert"
        big_NB = (NB >= 192)

        # K=128 special-case: medium NB + K=128, autotune picks
        # BN=8 BM=256 insert (NOT sortmerge!). The K-step argmin loop
        # at BN=64 K=128 is too expensive -- better to chop N into
        # many small tiles.
        if K >= 128 and M <= 200_000:
            BN = 8
            BM = 256
        elif M >= 5_000_000 and D <= 64 and (K <= 4 or K >= 32):
            BN = 128
        elif big_NB and M >= 5_000_000:
            BN = 128
        elif big_NB and D >= 256 and M <= 200_000 and K <= 4:
            BN = 128
        elif D >= 256 and K <= 16 and (NB >= 64 and M >= 500_000):
            # D=256 + K<=16 + (medium NB + medium M): BN=128 wins.
            # autotune (1,128,1M,256,8) picks BN=128 BM=64.
            BN = 128
        else:
            BN = 64
        BN = min(BN, max(8, _next_pow2(N)))
        # BM
        if K >= 128:
            BM = 256  # already covered above, keep consistent here
        elif K <= 4 and D <= 128 and M <= 200_000:
            BM = 256
        elif D >= 256 and K <= 16:
            BM = 64
        elif K <= 32:
            BM = 64 if (BN == 64 and K == 32 and D >= 256) else 128
        elif K <= 64:
            BM = 128
        else:
            BM = 128
        num_warps = 4

    elif NB <= 2048:
        kernel_mode = "insert"
        if M >= 5_000_000:
            BN = 128
        elif M >= 500_000 and D >= 256 and K <= 16:
            BN = 128
        else:
            BN = 64
        BN = min(BN, max(8, _next_pow2(N)))
        if K <= 4 and M >= 500_000:
            BM = 64
        elif K <= 16 and M >= 500_000 and D >= 128:
            BM = 64
        elif K >= 64:
            BM = 64
        else:
            BM = 128
        num_warps = 4

    else:  # NB >= 8K (build / large-eval regime)
        kernel_mode = "insert"
        # autotune patterns:
        #   NB=50K-100K D<=128 K<=8  -> BN=128 (halves HBM)
        #   NB=10K     K>=8           -> BN=64  (more topk work per row)
        #   NB>=10K    D>=256 K<=16   -> BN=128
        #   NB>=10K    D>=256 K>=32   -> BN=64
        if D >= 256 and K <= 16:
            BN = 128
        elif D >= 256 and K >= 32:
            BN = 64
        elif K <= 4:
            BN = 128
        elif K <= 8 and NB >= 30_000:
            # Larger NB -> BN=128 saves enough HBM to overcome topk cost
            BN = 128
        else:
            BN = 64
        if K <= 4 and D >= 256:
            BM = 64  # SMEM pressure at D=256 + BM=128
        elif K <= 4:
            BM = 128 if (D <= 64 or BN == 128) else 64
        elif K <= 8 and BN == 128:
            BM = 128
        elif K >= 64 and BN == 64 and D <= 128 and NB >= 30_000:
            # flashlib carve-out: NB in [30K, ~200K] + BN=64 + K>=64 +
            # D<=128 wins big at BM=128 splits=1 vs BM=64 splits=2.
            # Empirically at (1, 48K, 48K, 64, 64) the upstream rule
            # (BM=64, target_splits=2) runs 17.95 ms, while BM=128
            # splits=1 ns=2 runs 11.71 ms (-35%); at (1, 64K, 64K, 64,
            # 64) 29.20 ms -> 18.91 ms (-35%). Bounded on K>=64 (K=32
            # autotune at this NB shows BM=64 splits=1 wins, ~1 ms over
            # BM=128 splits=1) and on NB>=30K so the upstream rule
            # still applies to smaller-NB shapes where its M-split
            # target was directly autotune-derived.
            BM = 128
        else:
            BM = 64
        num_warps = 4
    BN = max(BN, min_bn)

    # ───────────────────────────────────────────────────────────────
    # Step 2: M_PER_SPLIT (target waves)
    # ───────────────────────────────────────────────────────────────
    num_n_tiles = max(1, math.ceil(N / BN))
    ctas_no_split = num_n_tiles * B

    # Workaround for an insert-kernel correctness issue with the
    # D-split path: when ``M_PER_SPLIT == BM`` (single M-loop
    # iteration) and the d-loop has >=4 chunks, the result is wrong.
    # Bumping the minimum mps to 2*BM keeps the M-loop at >=2 iters.
    needs_min_2_iters = (D_INNER < D and (D + D_INNER - 1) // D_INNER >= 4)
    min_mps = (2 * BM) if needs_min_2_iters else BM

    if is_large_n:
        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
    elif ctas_no_split >= NUM_SMS * 8:
        # Massively oversaturated (e.g. N=100K, BN=64, ctas=1563): one
        # split fully utilises SMs.
        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
    elif K >= 32 and BN == 64 and D <= 128 and NB >= 30_000 and not is_large_n:
        # flashlib carve-out (pairs with the BM=128 K>=64 rule above
        # plus K=32 BM=64 standalone): the K>=32 build regime at
        # NB>=30K wants splits=1 -- 2 splits doubles the K-step
        # argmin work per row and the Stage-2 reduce, neither of
        # which the autotune at NB=10K K=8 accounted for. Empirically
        # saves ~6 ms on (1, 48K, 48K, 64, 64) and ~10 ms on
        # (1, 64K, 64K, 64, 64) vs target_splits=2, and ~1 ms on
        # (1, 56K, 56K, 64, 32) vs target_splits=2.
        M_PER_SPLIT = ((M + BM - 1) // BM) * BM
    else:
        if ctas_no_split >= NUM_SMS * 4:
            # ctas >= 528: 2 splits for tail-fill at very large builds
            # (autotune (1,100K,100K,64,4) BN=128 ctas=782 -> 2 splits)
            target_splits = 2
        elif ctas_no_split >= NUM_SMS * 2:
            # ctas in [264, 528): single split already 2-4 waves; adding
            # splits hurts more than it helps. autotune
            # (1,50K,50K,64,8) BN=128 ctas=391 -> 1 split.
            target_splits = 1
        elif ctas_no_split >= NUM_SMS:
            # ctas in [132, 264): 3 splits for better tail. autotune
            # (1,10K,10K,64,16) BN=64 ctas=157 -> 3 splits.
            target_splits = 3
        else:
            # Under-saturated: target_waves by K (autotune-derived).
            # Use FLOOR div: snaps to integer wave count, which matches
            # autotune choices on shapes like NB=1024 K=32 ctas=16
            # (16 splits = 2 waves, 17 splits = 2.06 waves but bad
            # tail).
            if kernel_mode == "sortmerge":
                target_waves = 2
            elif K >= 128 and NB >= 64:
                # K=128 + medium NB: 2w wins (autotune
                # (1,128,100K,128,128) picks 16 splits at ctas=16).
                target_waves = 2
            elif K >= 128:
                target_waves = 1
            elif K >= 64 and ctas_no_split <= 2:
                # K=64 + tiny ctas: 1 wave wins (autotune
                # (1,128,100K,128,64) picks 66 splits = 1w).
                target_waves = 1
            elif K >= 64:
                target_waves = 2
            elif K <= 4 and ctas_no_split == 1 and (NB <= 32 or M <= 1_000_000):
                # K=4 single-CTA-stack: 4 waves at small NB or small M.
                # autotune NB=1 K=4 -> 521 splits; (1,128,10M,64,4)
                # however picks 264 splits (2w) -- too much M per CTA
                # for more splits to help.
                target_waves = 4
            elif K <= 8 and ctas_no_split >= 8 and ctas_no_split <= 32:
                target_waves = 4 if M >= 500_000 else 2
            elif K <= 8 and ctas_no_split == 1 and M >= 5_000_000 and BN <= 32:
                target_waves = 4
            elif B >= 2 and N <= 8 and K <= 8:
                # Batched B>=2 + single-query + K<=8: 4w wins
                target_waves = 4
            else:
                target_waves = 2
            target_splits = max(1, NUM_SMS * target_waves // ctas_no_split)
            target_splits = min(target_splits, max(1, math.ceil(M / BM)))
        mps_raw = math.ceil(M / target_splits)
        mps = math.ceil(mps_raw / BM) * BM
        mps = max(min_mps, min(mps, ((M + BM - 1) // BM) * BM))
        mps = min(mps, MAX_MPS_TILES * BM)
        M_PER_SPLIT = mps

    NUM_SPLITS = math.ceil(M / M_PER_SPLIT)

    # ───────────────────────────────────────────────────────────────
    # Step 3: NUM_STAGES_PIPE (M-loop pipeline depth)
    # ───────────────────────────────────────────────────────────────
    # Derived from a 201-shape × 4-ns sweep on H200 (upstream
    # ``bench/sweep_ns_by_k.py``), cross-checked against direct probes
    # on N>=64 build shapes. Three competing effects:
    #
    #   * Each extra pipeline stage adds another buffered C-tile,
    #     costing ``BM * D_INNER * 2`` bytes of SMEM/registers. Wide
    #     D-tiles amortise this well; narrow ones don't.
    #   * The K-step argmin inner loop holds live state proportional
    #     to ``BN * TOPK_PAD * 8`` bytes. For narrow rows (BN<=16) +
    #     large K, registers get tight and deeper pipelines spill.
    #   * Medium-N tiles (BN>=64) have abundant compute per tile to
    #     amortise prefetch, but get little benefit at very large K
    #     because the K loop dominates anyway.
    #
    # Empirical winners (mode_ns from the 201-shape sweep):
    #
    #   D_INNER >= 256 (wide D)               -> ns=2  (uniform across K)
    #   BN >= 128 (huge build N)              -> ns=2  (registers tight)
    #   BN >= 64 + BM <= 128 + K <= 8         -> ns=3  (small tile + tiny K)
    #   BN >= 64                              -> ns=2  (BM=256 SMEM-bound)
    #   BN ∈ [16,32] + BM=256 + K >= 64       -> ns=3  (big-tile large-K)
    #   otherwise (narrow tile + any K)       -> ns=1  (avoid spills)
    #
    # Without this rule, K>=64 + D=256 shapes hit catastrophic
    # regressions (e.g. (1,16,100K,256,128) 439us -> 2429us at ns=1,
    # a 5.5x slowdown). Conversely K<=32 + huge-M shapes save 20-45%
    # vs ns=2 (e.g. (1,1,10M,128,4) 1426us -> 785us at ns=1).
    if D_INNER < D:
        NUM_STAGES_PIPE = 1  # D-split path is hard-coded to ns=1
    elif (BN == 8 and BM == 64 and D_INNER == 256
          and K <= 4 and M >= 500_000):
        # NB<=8 + K<=4 + D=256 + BM=64 + huge-M: ns=1 wins by 1.48-1.81x
        # over the default ns=2. Direct verification at K ∈ {1, 2, 4} ×
        # M ∈ {1M, 5M, 10M, 30M}: (nw=4, ns=1) lifts %peak HBM from
        # 39-48 % (ns=2) to 57-85 % (ns=1). The ns=2 default below was
        # calibrated for wider-N tiles; at BN=8 the deeper pipeline
        # spends too many registers on prefetch and the K=4 epilogue
        # spills. Pairs with the nw=4 default the rule above falls into
        # at D=256.
        NUM_STAGES_PIPE = 1
    elif D_INNER >= 256:
        NUM_STAGES_PIPE = 2
    elif BN >= 128:
        NUM_STAGES_PIPE = 2
    elif BN >= 64 and BM <= 128 and K <= 8:
        NUM_STAGES_PIPE = 3
    elif BN >= 64:
        NUM_STAGES_PIPE = 2
    elif BN >= 16 and BM >= 256 and K >= 64:
        NUM_STAGES_PIPE = 3
    else:
        NUM_STAGES_PIPE = 1

    cfg = {
        "BN": BN, "BM": BM, "D_INNER": D_INNER,
        "TOPK_PAD": (
            max(32, _next_pow2(K)) if kernel_mode == "sortmerge"
            else _next_pow2(K)
        ),
        "kernel_mode": kernel_mode, "num_warps": num_warps,
        "M_PER_SPLIT": M_PER_SPLIT,
        "NUM_SPLITS": NUM_SPLITS,
        "NUM_STAGES_PIPE": NUM_STAGES_PIPE,
    }

    # Dtype-aware SMEM shrink. The shape-only heuristic above was tuned
    # on bf16 and may pick (BN=128, BM=128, D_INNER=128) for shapes that
    # cleanly fit ~210 KB at 2 bytes/elt but blow past the 227 KB H200
    # opt-in cap at 4 bytes/elt. The fitter shrinks BN/BM/NUM_STAGES_PIPE
    # by powers of 2 until the kernel fits — leaving D_INNER and
    # kernel_mode (which encode different perf regimes) alone.
    if smem_limit is not None:
        cfg = _fit_config_to_smem(cfg, D, K, dtype_bytes, smem_limit, min_bn=min_bn)

    return cfg


# ── autotune (opt-in) ──────────────────────────────────────────────────


_autotune_cache: dict = {}


def _autotune(x, c, k, *, force_path=None):
    """Brute-force autotune over the candidate config grid.

    Cached per ``(B, N, M, D, k, dtype, force_path)`` shape; first call
    costs ~30-60 s of compile time, subsequent calls hit the cache in
    sub-ms.

    The candidate grid is dtype-aware: ``_gen_configs`` filters out
    configs whose SMEM exceeds the device opt-in limit at this dtype,
    so fp32 inputs at large D never have an unrunnable config in the
    search space.
    """
    B, N, D = x.shape
    M = c.shape[1]
    dtype_bytes = x.element_size()
    key = (B, N, M, D, k, x.dtype, force_path)

    if key in _autotune_cache:
        return _autotune_cache[key]

    smem_limit = _smem_limit(x.device)
    configs = _gen_configs(D, k,
                           dtype_bytes=dtype_bytes,
                           smem_limit=min(smem_limit, 220_000),
                           min_bn=_min_bn(x.device))
    if not configs:
        raise RuntimeError(
            f"No valid flash_knn config for D={D}, K={k}, "
            f"dtype={x.dtype} (element_size={dtype_bytes})"
        )

    # Drop configs with BN > next_pow2(N) -- they would just pad N to BN
    # and run the same kernel slower.
    max_bn = max(16, _next_pow2(N))
    configs = [cfg for cfg in configs if cfg["BN"] <= max_bn]

    best_time = float('inf')
    best_cfg = None
    max_partial_bytes = 4 * 1024**3

    for cfg in configs:
        bn = cfg["BN"]
        bm = cfg["BM"]
        d_inner = cfg["D_INNER"]
        topk_pad = cfg["TOPK_PAD"]
        kernel_mode = cfg["kernel_mode"]
        nw = cfg["num_warps"]
        ns_pipe = cfg.get("NUM_STAGES_PIPE", 2)

        if force_path == "large_n":
            mps_list = [((M + bm - 1) // bm) * bm]
        else:
            mps_list = _gen_m_splits(M, bm, B=B, N=N, BN=bn)

        for mps in mps_list:
            num_splits = math.ceil(M / mps)
            partial_bytes = B * num_splits * N * k * 8
            if partial_bytes > max_partial_bytes:
                continue

            try:
                pv = torch.empty((B, N, num_splits, k),
                                 device=x.device, dtype=torch.float32)
                pi = torch.empty((B, N, num_splits, k),
                                 device=x.device, dtype=torch.int32)
                grid = (num_splits, math.ceil(N / bn), B)
                pv_s0, pv_s1, pv_s2, pv_s3 = pv.stride()
                pi_s0, pi_s1, pi_s2, pi_s3 = pi.stride()

                if kernel_mode == "sortmerge":
                    def run(grid=grid, bn=bn, bm=bm, d_inner=d_inner,
                            topk_pad=topk_pad, mps=mps, nw=nw,
                            num_splits=num_splits, pv=pv, pi=pi,
                            ns_pipe=ns_pipe,
                            pv_s0=pv_s0, pv_s1=pv_s1, pv_s2=pv_s2, pv_s3=pv_s3,
                            pi_s0=pi_s0, pi_s1=pi_s1, pi_s2=pi_s2, pi_s3=pi_s3):
                        _flash_knn_sortmerge_kernel[grid](
                            x, c, pv, pi,
                            x.stride(0), x.stride(1), x.stride(2),
                            c.stride(0), c.stride(1), c.stride(2),
                            pv_s0, pv_s2, pv_s1, pv_s3,
                            pi_s0, pi_s2, pi_s1, pi_s3,
                            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
                            BN=bn, BM=bm, D_INNER=d_inner,
                            TOPK_PAD=topk_pad,
                            NUM_STAGES_PIPE=ns_pipe,
                            num_warps=nw,
                        )
                        if num_splits > 1:
                            pv_flat = pv.view(B, N, -1)
                            pv_flat.topk(k, dim=-1, largest=False)
                else:
                    max_steps = min(k, bm)
                    def run(grid=grid, bn=bn, bm=bm, d_inner=d_inner,
                            topk_pad=topk_pad, mps=mps, nw=nw,
                            num_splits=num_splits, pv=pv, pi=pi,
                            max_steps=max_steps, ns_pipe=ns_pipe,
                            pv_s0=pv_s0, pv_s1=pv_s1, pv_s2=pv_s2, pv_s3=pv_s3,
                            pi_s0=pi_s0, pi_s1=pi_s1, pi_s2=pi_s2, pi_s3=pi_s3):
                        _flash_knn_insert_kernel[grid](
                            x, c, pv, pi,
                            x.stride(0), x.stride(1), x.stride(2),
                            c.stride(0), c.stride(1), c.stride(2),
                            pv_s0, pv_s2, pv_s1, pv_s3,
                            pi_s0, pi_s2, pi_s1, pi_s3,
                            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
                            BN=bn, BM=bm, D_INNER=d_inner,
                            TOPK_PAD=topk_pad, MAX_STEPS=max_steps,
                            NUM_STAGES_PIPE=ns_pipe,
                            num_warps=nw,
                        )
                        if num_splits > 1:
                            pv_flat = pv.view(B, N, -1)
                            pv_flat.topk(k, dim=-1, largest=False)

                t = _bench_quick(run)
                if t < best_time:
                    best_time = t
                    best_cfg = {**cfg, "M_PER_SPLIT": mps,
                                "NUM_SPLITS": num_splits,
                                "NUM_STAGES_PIPE": ns_pipe}

                del pv, pi
            except Exception:
                continue

    if best_cfg is None:
        raise RuntimeError(
            f"All flash_knn configs failed for B={B}, N={N}, M={M}, D={D}, K={k}")

    _autotune_cache[key] = best_cfg
    return best_cfg


# ── runner ─────────────────────────────────────────────────────────────


_OOR_FALLBACK_CACHE: dict = {}


def _try_launch(x, c, cfg, B, N, M, D, k):
    """Single launch attempt; returns idxs or raises ``triton.compiler.errors.OutOfResources``.

    Factored out of ``_run`` so the OOR-shrink retry loop can call it
    repeatedly with a shrunk cfg.
    """
    bn = cfg["BN"]
    bm = cfg["BM"]
    d_inner = cfg["D_INNER"]
    topk_pad = cfg["TOPK_PAD"]
    mps = cfg["M_PER_SPLIT"]
    num_splits = cfg["NUM_SPLITS"]
    kernel_mode = cfg["kernel_mode"]
    nw = cfg["num_warps"]
    num_stages_pipe = cfg.get("NUM_STAGES_PIPE", 2)

    partial_vals = torch.empty((B, N, num_splits, k),
                               device=x.device, dtype=torch.float32)
    partial_idxs = torch.empty((B, N, num_splits, k),
                               device=x.device, dtype=torch.int32)
    grid = (num_splits, math.ceil(N / bn), B)

    pv_s0, pv_s1, pv_s2, pv_s3 = partial_vals.stride()
    pi_s0, pi_s1, pi_s2, pi_s3 = partial_idxs.stride()

    if kernel_mode == "sortmerge":
        _flash_knn_sortmerge_kernel[grid](
            x, c, partial_vals, partial_idxs,
            x.stride(0), x.stride(1), x.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            pv_s0, pv_s2, pv_s1, pv_s3,
            pi_s0, pi_s2, pi_s1, pi_s3,
            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
            BN=bn, BM=bm, D_INNER=d_inner,
            TOPK_PAD=topk_pad,
            NUM_STAGES_PIPE=num_stages_pipe,
            num_warps=nw,
        )
    else:
        max_steps = min(k, bm)
        _flash_knn_insert_kernel[grid](
            x, c, partial_vals, partial_idxs,
            x.stride(0), x.stride(1), x.stride(2),
            c.stride(0), c.stride(1), c.stride(2),
            pv_s0, pv_s2, pv_s1, pv_s3,
            pi_s0, pi_s2, pi_s1, pi_s3,
            N=N, M=M, D=D, K=k, M_PER_SPLIT=mps,
            BN=bn, BM=bm, D_INNER=d_inner,
            TOPK_PAD=topk_pad, MAX_STEPS=max_steps,
            NUM_STAGES_PIPE=num_stages_pipe,
            num_warps=nw,
        )

    if num_splits == 1:
        return partial_idxs[:, :, 0, :].contiguous()

    pv = partial_vals.view(B, N, -1)
    pi = partial_idxs.view(B, N, -1)
    _, sel = pv.topk(k, dim=-1, largest=False, sorted=True)
    out_idxs = pi.gather(-1, sel.to(torch.int64)).to(torch.int32)
    return out_idxs


def _shrink_cfg_on_oor(cfg: dict, min_bn: int = 8) -> Optional[dict]:
    """Halve the cheapest SMEM-bearing axis. ``None`` when nothing left.

    Order:
      1. NUM_STAGES_PIPE 3 → 2 → 1 (drops the c-tile pipeline buffer
         — usually free or nearly-free at this depth).
      2. Tend toward a near-square tile by **halving the larger of
         (BN, BM)** first. WGMMA shape efficiency drops rapidly once
         BM<32 on Hopper (MMA atom is 64x16x16), so we prefer to
         shrink BN when BN > BM. For BN=128 BM=64 (typical build
         regime) this gives BN=64 BM=64 in one step, which empirically
         matches the kmeans-style ``_fit_config_to_smem`` choice and
         beats BN=128 BM=32 by ~20 % at D=1024 fp32.
      3. Floors: BN ≥ ``min_bn`` (8, or 16 on Blackwell), BM ≥ 32
         (both required by the MMA atom).

    sortmerge requires ``BM == TOPK_PAD``; skip BM shrinks and only
    halve BN.

    Also recomputes ``M_PER_SPLIT`` so it remains a multiple of the
    new BM (the kernel's inner ``tl.range`` walk requires this).
    """
    ns = int(cfg.get("NUM_STAGES_PIPE", 2))
    bm = int(cfg["BM"])
    bn = int(cfg["BN"])
    sortmerge = cfg.get("kernel_mode") == "sortmerge"

    out = dict(cfg)
    if ns > 1:
        out["NUM_STAGES_PIPE"] = ns - 1
        return out
    if sortmerge:
        if bn > min_bn:
            out["BN"] = bn // 2
            return out
        return None
    # Non-sortmerge: prefer shrinking the larger axis.
    if bn > bm and bn > min_bn:
        out["BN"] = bn // 2
        return out
    if bm > 32:
        out["BM"] = bm // 2
        new_bm = out["BM"]
        out["M_PER_SPLIT"] = ((int(out["M_PER_SPLIT"]) + new_bm - 1)
                              // new_bm) * new_bm
        return out
    if bn > min_bn:
        out["BN"] = bn // 2
        return out
    return None


def _run(x: torch.Tensor, c: torch.Tensor, k: int, *,
         force_path: Optional[str] = None,
         autotune: bool = False) -> torch.Tensor:
    """Shared dispatcher -- launch stage 1, optional stage 2 reduce, return idxs.

    Returns indices only ``(B, N, k) int32``. The wrapper at
    :func:`flashlib.primitives.knn.flash_knn` calls the gather kernel to
    produce true squared distances per neighbour.

    On ``OutOfResources`` at launch, we shrink one SMEM-bearing axis
    (NS → BM → BN) and retry, caching the surviving config per
    ``(D, K, dtype, force_path)`` so subsequent calls with the same
    shape skip the failed compiles. This is the source of truth for
    "does this fit?"; ``_estimate_sram`` only provides a coarse hint
    for the autotune candidate filter.
    """
    assert x.is_cuda and c.is_cuda
    assert x.dtype == c.dtype
    B, N, D = x.shape
    M = c.shape[1]
    assert c.shape == (B, M, D)
    assert 1 <= k <= M

    min_bn = _min_bn(x.device)
    if autotune:
        cfg = _autotune(x, c, k, force_path=force_path)
    else:
        cfg = _heuristic_config(
            B, N, M, D, k,
            force_path=force_path,
            dtype_bytes=x.element_size(),
            smem_limit=min(_smem_limit(x.device), 220_000),
            min_bn=min_bn,
        )

    # Surviving-config cache keyed on the parts the dispatcher can't
    # vary (D, K, dtype, force_path). If a previous call already
    # shrunk past an OOR for this shape, start with that smaller cfg.
    fb_key = (D, k, x.dtype, force_path,
              cfg["kernel_mode"], cfg.get("D_INNER"))
    if fb_key in _OOR_FALLBACK_CACHE:
        cached = _OOR_FALLBACK_CACHE[fb_key]
        # Use the cached cfg only if it's smaller than the current one
        # on at least one SMEM-bearing axis (otherwise the heuristic
        # already picked something safe for this shape).
        sm_cur = (cfg["BN"], cfg["BM"], cfg.get("NUM_STAGES_PIPE", 2))
        sm_cached = (cached["BN"], cached["BM"],
                     cached.get("NUM_STAGES_PIPE", 2))
        if sm_cached < sm_cur:
            cfg = {**cfg, **{k_: cached[k_] for k_ in
                              ("BN", "BM", "NUM_STAGES_PIPE")}}
            # Re-align mps to new BM
            new_bm = cfg["BM"]
            cfg["M_PER_SPLIT"] = ((cfg["M_PER_SPLIT"] + new_bm - 1)
                                  // new_bm) * new_bm
            cfg["NUM_SPLITS"] = math.ceil(M / cfg["M_PER_SPLIT"])

    # Import lazily so we don't pay it on every dispatch.
    try:
        from triton.runtime.errors import OutOfResources
    except ImportError:
        try:
            from triton.compiler.errors import OutOfResources
        except ImportError:
            OutOfResources = RuntimeError  # safest fallback

    last_err = None
    initial_cfg = cfg
    shrunk = False
    for _attempt in range(8):
        try:
            result = _try_launch(x, c, cfg, B, N, M, D, k)
            # Cache the surviving cfg so subsequent calls with the
            # same (D, K, dtype, ...) shape skip the failed compile.
            if shrunk:
                _OOR_FALLBACK_CACHE[fb_key] = {
                    "BN": cfg["BN"], "BM": cfg["BM"],
                    "NUM_STAGES_PIPE": cfg.get("NUM_STAGES_PIPE", 2),
                }
            return result
        except OutOfResources as e:
            last_err = e
            new_cfg = _shrink_cfg_on_oor(cfg, min_bn)
            if new_cfg is None:
                break
            cfg = new_cfg
            shrunk = True
            continue

    raise last_err if last_err is not None else RuntimeError(
        "flash_knn: exhausted OOR shrink attempts")


# ── public API ─────────────────────────────────────────────────────────


def flash_knn_triton_small_n(x: torch.Tensor, c: torch.Tensor, k: int,
                              *, autotune: bool = False) -> torch.Tensor:
    """Force the M-split (search) path -- ``(B, N, k) int32`` indices."""
    return _run(x, c, k, force_path=None, autotune=autotune)


def flash_knn_triton_large_n(x: torch.Tensor, c: torch.Tensor, k: int,
                              *, autotune: bool = False) -> torch.Tensor:
    """Force the single-pass (build) path -- ``(B, N, k) int32`` indices."""
    return _run(x, c, k, force_path="large_n", autotune=autotune)


def flash_knn_triton(x: torch.Tensor, c: torch.Tensor, k: int,
                     *, autotune: bool = False) -> torch.Tensor:
    """Universal Triton dispatch -- the heuristic picks BN/BM/mode/
    mps/ns_pipe based on shape, including whether to single-pass or
    M-split. The per-CTA-count check inside :func:`_heuristic_config`
    handles both build and eval shapes without a host-side forced path.

    Neither path materialises an N×M cross or distance matrix to HBM,
    and neither computes or loads ``x_sq`` -- the two hard contracts
    flashlib's KNN imposes on every Triton entry point here.

    Args:
        x: ``(B, N, D)`` bf16 / fp16 / fp32 query tensor.
        c: ``(B, M, D)`` corpus, same dtype.
        k: number of neighbours.
        autotune: ``False`` (default) uses the shape-only heuristic --
            first call pays a single Triton compile (~0.5 s). ``True``
            runs the full brute-force sweep + caches per shape (~30-90 s
            first call).

    Returns:
        idxs: ``(B, N, k)`` int32 -- nearest-neighbour indices, sorted
        by ascending true squared L2 distance (ties broken by index).
    """
    return _run(x, c, k, force_path=None, autotune=autotune)
