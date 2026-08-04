"""Hopper SM90 fused flash-knn CuteDSL kernel (x²-free, index-only).

Sibling of the Triton kernels in :mod:`flashlib.primitives.knn.triton`:
same shift-invariant scoring ``s = c_sq[m] − 2·cross`` so ``argmin-K``
matches true ``argmin ||x − c||²`` without ever loading ``x_sq``.

Contract:

* No ``mXsq_n`` in the kernel signature (no HBM load).
* No per-thread ``xs`` register cache.
* No ``d >= 0`` clamp (score can be negative).
* No ``mOutV_nk`` (vals output) -- only the ``mOutI_nk`` (idx) tensor
  is written. Saves ``B·N·K·4`` HBM bytes and a register-file pass per
  CTA. True distances are recovered by
  :func:`flashlib.kernels.distance.triton_knn_gather_sqdist` outside
  the fused pass.

The ``sortmerge_packed`` top-K strategy is not supported here -- it
relies on ``Int64`` ordering matching ``fp32`` ordering in the upper
32 bits, which only holds for non-negative scores. The autotune
wrapper in :mod:`flashlib.primitives.knn.cutedsl.impl` filters it out
and selects between ``insert / perthread / sortmerge / smem_perthread
/ warpsort`` (all of which use native fp32 compares that work for
signed scores).

This module implements ``HopperFlashKnnFused``: a TMA + WGMMA fused
kernel that computes top-K nearest neighbors per query without ever
materialising the ``(N, M)`` cross or distance matrix in HBM. The
streaming axis is the database index ``M``: each CTA tile keeps its
``BN_query x BM_db`` cross accumulator in WGMMA registers, fuses the
``c_sq - 2*cross`` signed-score computation in registers, and
maintains a per-thread per-row top-K register heap that is updated
across database tiles. After the M-stream loop, the per-thread heaps
are merged across the WGMMA TV-layout's threads-in-row group via
butterfly shuffle, then sorted, and written to GMEM.

The architecture mirrors ``flash_kmeans/_cutedsl_assign_kernel.py``
(the kmeans assignment kernel, which is essentially KNN with K=1)
with two extensions:

  1. Per-row top-K register heap (instead of a single best_d/best_i
     scalar pair): each thread holds K_PAD fp32 + K_PAD int32 entries
     per query row it owns. K_PAD must be a constexpr power-of-two
     (16 or 32). Insertion is the standard "find argmax -> conditional
     replace" pattern, with a ``heap_max`` threshold cached for
     branch-pruning candidate distances early.

  2. Cross-thread top-K butterfly merge (instead of a single argmin
     warp-reduce): for each round of the WGMMA TV-layout's N-side
     reduction group, every thread exchanges its K_PAD entries with
     its peer at offset 2^r, then tries to insert all K_PAD peer
     entries into its own heap. After ``log2(group_size)`` rounds,
     every thread in the group has the same global top-K_PAD heap
     for its row. The row leader writes to GMEM.

K_PAD regimes
=============

K_PAD <= 32: the full top-K state fits in registers per thread. This
file implements that path. Above 32, the per-thread register pressure
exceeds Hopper's max-regs (240/thread) once you add the WGMMA acc and
the 2x heap entries per row, so a SMEM-resident sorting variant would
be needed; that path is not implemented here.

WS variant
==========

Like the kmeans kernel, ``use_ws=True`` enables FMHA-style producer/
consumer warp specialization: 1 producer WG (24 regs/thread, only
issues TMA loads) plus the existing MMA WGs (240 regs/thread, do
WGMMA + per-row top-K). Selected per-shape by autotune.

Top-K strategies (selectable, no compile heuristic)
===================================================

The ``topk_strategy`` ctor argument explicitly picks the inner
top-K kernel. ``"auto"`` falls back to a coarse K_PAD-keyed default
for callers that don't drive the autotuner. The autotuner in
``flashlib.primitives.knn.cutedsl.impl.cutedsl_flash_knn`` sweeps
strategies × tiles × use_ws and picks the empirical winner per shape.

Strategy summary:

  ``insert``         : cooperative chunk-best argmin, replicated heap.
                       Only competitive for K_PAD <= 2.
  ``perthread``      : per-thread sorted-asc top-K + branch-free
                       bubble insert. Wins K_PAD = 4..14.
  ``sortmerge``      : per-thread bitonic chunk-sort + sorted merge.
                       Wins K_PAD >= 16.
  ``sortmerge_packed`` : same network as ``sortmerge`` but packs
                       (fp32 distance, int32 idx) into a single Int64
                       (Triton's stage-1 trick). Halves the per-swap
                       op count on paper, but on Hopper SP (no native
                       64-bit integer ALU throughput) the resulting
                       PTX is consistently 20-40% SLOWER than the
                       unpacked path -- kept as an opt-in escape
                       hatch and as documentation of "tried this,
                       lost on Hopper". Excluded from the default
                       autotune sweep.
  ``smem_perthread`` : 1-thread-per-row top-K reading dist staged
                       through SMEM. Foundation for the (still-
                       planned) WS3 path. Two perf-critical fixes
                       applied vs the naive impl:
                       (a) sDist row stride padded to BN+1 to break
                       the 32-way bank conflict that would otherwise
                       arise on the consumer reads (each topk
                       thread reads sDist[my_row, j] sequentially;
                       with stride=BN=128 fp32 == 0 mod 32 banks,
                       all 32 lanes hit the same bank); padding to
                       BN+1 makes stride%32 == 1 so every lane
                       lands on a distinct bank. ~2.3x at K=8.
                       (b) Inner ``for j in range(BN)`` switched
                       from ``range_constexpr`` (full unroll) to
                       ``range(BN, unroll=4)`` because the fully
                       unrolled BN=128 * (load + setp + K-deep
                       bubble) blows past the 240-reg cap and
                       spills heavily; partial unroll caps the
                       inlined chunk size. ~3x at K=12-32.
                       Even after both fixes, standalone is still
                       1.4-2x slower than ``sortmerge`` -- only
                       useful as the WS3 inner kernel where it can
                       overlap with GEMM in a separate warpgroup.
                       Excluded from autotune sweep until WS3 ships.
  ``warpsort``       : Lane-cooperative bitonic, mirrors Triton's
                       ``tl.sort``. dist tile staged through ``sDist``
                       (same BN+1-pad layout as smem_perthread); per
                       chunk each warp loads ROWS_PER_WARP rows in
                       parallel (1 element per lane per row), runs a
                       multi-row warp-cooperative bitonic sort across
                       the 32 lanes via ``shuffle_sync_bfly``, then
                       merges with the running per-warp top-K via
                       reverse-shfl + elementwise-min + bitonic-finish.
                       Per-substage all 2*ROWS shfls are issued first
                       (into ``peer_*_buf``) before any compare/select,
                       so the bfly issue stream stays full through the
                       24-cycle Hopper shfl latency.

                       Empirical: 21 ms at K=4..32 on the standard
                       N=16384/M=100K/D=128/BM=BN=128 shape -- 4-5x
                       SLOWER than ``sortmerge``. Per-CTA-per-chunk
                       SHFL.BFLY count is 2688 (verified via SASS),
                       and at the SM-scheduler issue rate of 1
                       inst/sub-partition/cycle with 8 warps spread
                       across 4 sub-partitions (= 2 warps competing
                       per partition), the kernel is shfl-throughput
                       bound at ~10 cycles/shfl in the steady state.
                       Total: 781 chunks * 2688 shfls * 10 cycles /
                       1.83 GHz ~= 11 ms per CTA per warp; 1 wave on
                       H100 (128 CTAs / 132 SMs) lands at ~21 ms.

                       Why ``sortmerge`` (per-thread bitonic) wins
                       here despite having more raw ops: the per-
                       thread sort is intra-thread (NO shfls in the
                       inner loop, only at the cross-thread merge
                       at the very end), so it doesn't compete for
                       the warp scheduler's shfl issue slots. Triton
                       gets away with ``tl.sort`` because its
                       generated sort has fewer total shfls (smaller
                       chunk size = BM=32 vs our BM=128 -- but BM>=64
                       is forced by WGMMA atom shape so we can't drop
                       to Triton's 32) and runs in a higher-occupancy
                       layout that hides shfl latency across many
                       waves of CTAs.

                       Kept as an opt-in strategy for documentation
                       and as a reference implementation of the
                       lane-cooperative bitonic primitive. Excluded
                       from the default autotune sweep -- it never
                       wins the competitive set on Hopper at BN>=64.

Tie-break in the sort comparators
=================================

All compare-swap helpers (``_cmp_swap_asc/_desc``,
``_cmp_swap_2d_asc``, ``_bubble_insert_asc``) and the post-sort
min-pair use a STRICT ``>`` on distance, with NO secondary
``(da == db) and (ia > ib)`` tie-break. The tie-break costs one
extra SETP, AND, and OR per swap (~3 ops out of ~9) and gives a
measurable ~1.4x speedup at K_PAD=32 on Hopper to drop. The trade-
off is that two equal-distance entries within a per-thread or
per-warp top-K may end up in either order; the final TOP-K SET is
unaffected (verified: idx_match=1.0 at K=1..32 on N=4096 M=16384
D=128). Bit-exact parity with Triton's packed-uint64 sort across
ties is therefore not guaranteed (and would only matter on
synthetic data with many ties anyway).

Empirical perf vs Triton (N=16384, M=100k, D=128, H100 SXM)
==========================================================

After the tie-break drop + sortmerge / perthread strategy mix +
small-tile (BM=64 BN=64) wins for low K + the smem_perthread bank-
conflict fix:

  K     CuteDSL    Triton    ratio   best config
  ---   --------   --------  ------  -----------------------------
   1    1599 us    2333 us   0.68x   BM=64  BN=64  perthread  WIN
   2    2408 us    2678 us   0.90x   BM=64  BN=64  perthread  WIN
   4    2221 us    2982 us   0.74x   BM=64  BN=64  sortmerge  WIN
   6    2541 us    3308 us   0.77x   BM=64  BN=64  perthread  WIN
   8    2584 us    3335 us   0.77x   BM=64  BN=64  perthread  WIN
  10    3035 us    3880 us   0.78x   BM=64  BN=64  perthread  WIN
  12    3857 us    3842 us   1.00x   BM=64  BN=64  perthread  TIE
  14    4439 us    3996 us   1.11x   BM=64  BN=64  perthread  lose
  16    4783 us    3942 us   1.21x   BM=64  BN=64  perthread  lose
  20    6375 us    4718 us   1.35x   BM=64  BN=64  perthread  lose
  24    7190 us    4784 us   1.50x   BM=128 BN=128 sortmerge  lose
  28    7225 us    4827 us   1.50x   BM=128 BN=128 sortmerge  lose
  32    7210 us    4895 us   1.47x   BM=128 BN=128 sortmerge  lose

Tile-size insight: BM=64 BN=64 sweeps to the front for K<=20 because
N_per_thr drops from 32 (BM=128 BN=128) to 16, so per-thread chunk
work scales as 16*log^2(16)/32*log^2(32) = ~1/3, even though we pay
for 2x more CTAs (and thus 2 waves on H100's 132 SMs). For K=24..32
the K_INTERNAL=32 heap dominates per-row work and the larger tile
amortises GEMM+TMA setup better.

The K>=14 gap is fundamental: Triton's ``tl.sort`` parallelises the
bitonic substages across SIMT lanes via warp shuffles (each compare-
swap is ~3 cycles regardless of how many elements the row holds),
while our per-thread bitonic pays O(N log^2 N) sequential cycles on
each thread for its N_per_thr-element slice. The ``warpsort``
strategy (see Strategy summary above) implements (a) a lane-
cooperative bitonic via ``shuffle_sync_bfly`` mirroring Triton's
``tl.sort``, but on Hopper SM90 with CuTeDSL it runs ~4-5x SLOWER
than ``sortmerge``: with WGMMA forcing BM>=64 the per-chunk shfl
count (2688 SHFL.BFLY per CTA per chunk) exceeds the warp-scheduler
issue rate that 8 warps / 4 sub-partitions can sustain, and
``sortmerge``'s per-thread bitonic (which uses ZERO shfls in its
inner loop) wins on the same SM scheduler budget. Closing this on
Hopper would need either (b) a small-chunk re-architecture (chunk
size = K_PAD, like Triton) which is incompatible with WGMMA's
64-row M-atom minimum, or (c) a lower-level ldmatrix.sync layout
that gets the data 1-element-per-lane in registers without the
SMEM round-trip + scheduler contention. Tracked as future work.

Inline-PTX experiment (2026-05-08) -- bottom line: marginal
============================================================

Hypothesis: ptxas refuses to fold ``(da > db) ? db : da`` into
``min.f32`` (NaN semantics differ), so the bitonic SASS is all
``FSEL/SEL`` (1182+1065 ops) with zero ``FMNMX``. Hand-rolling the
cmp_swap as ``llvm.inline_asm`` with explicit ``min.f32 + max.f32 +
setp + 2x selp.b32`` should let ptxas use FMNMX (1184 ops in the
re-dumped SASS, verified) and free up issue slots for the i-side
selps.

Measured (sortmerge BM=128 BN=128 N=16384 M=100K D=128, K=16):

  cmp_swap form                 K_PAD=16  K_PAD=24  spill bytes (K=24)
  --------------------------------------------------------------------
  MLIR baseline (5 ops)         5006 us   7194 us   144
  inline-PTX setp+selp (5 ops)  5016 us   7135 us   144  (no win)
  inline-PTX min/max  (5 ops)   4848 us   7848 us   152  (3% / -9%)
  inline-PTX min/max gated      4866 us   7198 us   144

Conclusions:
  * Inline PTX with ``setp+selp`` packed -- 0% gain (ptxas already
    cross-schedules MLIR-lowered ops in a single basic block).
  * Inline PTX with ``min.f32 + max.f32`` -- +3% at K_PAD<=16 because
    FMNMX has marginally lower latency than ``setp+selp`` and frees
    the i-side select pipeline. At K_PAD>=24 the K_INTERNAL=32 heap
    is already register-pressure bound (~144 bytes of pre-existing
    spill at baseline); the struct-return of inline_asm adds 8 more
    bytes of spill and the gain inverts to a 9% regression.
  * Net production impact = 0: the autotuner picks ``perthread`` (not
    ``sortmerge``) at K_PAD<=12 (where inline-PTX would otherwise
    win) and ``sortmerge`` at K_PAD>=20 (where inline-PTX would
    regress); the small K_PAD=14..16 gap where ``sortmerge`` is
    chosen is too narrow for 3% to flip the verdict vs Triton.
  * Code reverted to the MLIR baseline. ``_cmp_swap_asc_ptx`` and
    ``_cmp_swap_asc_packed_ptx`` (q.v.) retained as documentation of
    the inline-PTX wiring, and the packed variant IS used by
    ``_cmp_swap_asc_packed`` since it cuts ``sortmerge_packed``'s
    spill in half (104->64 bytes) and gives a real 30-50% speedup on
    that path -- but ``sortmerge_packed`` itself is still 20% slower
    than the unpacked ``sortmerge`` on Hopper because per-cmp_swap
    op count is 6 (2 ISETP + 4 SEL.b32 from the u64 lowering) vs 5
    for the (fp32, int32) pair, and Hopper has no native 64-bit ALU
    to amortise the +1 op.

The K_PAD=14..32 wall therefore stands. Closing it requires the
small-chunk lane-cooperative architecture (Triton's tl.sort path)
which WGMMA's 64-row M-atom minimum on Hopper precludes.

WS3 design (LANDED 2026-05-08, chunk-min added 2026-05-08 PM)
=============================================================

3 warpgroups (1 load + 1 GEMM + 1 top-K) with a ``dist_pipeline``
ring buffer between GEMM and top-K, so per-chunk wall becomes
``max(t_gemm, t_topk)`` rather than ``t_gemm + t_topk``. Enabled
via ``HopperFlashKnnFused(use_ws=True, use_ws3=True,
topk_strategy='smem_perthread', dist_stage=2)``; the autotuner
adds it as a candidate for K_PAD>=14.

KEY OPTIMISATION (2026-05-08 PM): per-row chunk-min early exit.
Mirrors Triton stage-1's ``if chunk_best < topk_worst_val: ...``
(knn_triton.py:692). The GEMM WG inline-reduces the BN dist
columns to a per-row min while writing sDist (one extra fp32
store per row per chunk into sChunkMin[BM, dist_stage]); the
TopK WG checks ``chunk_min < worst_d`` once per chunk and
skips the entire BN-element prune-and-bubble loop when no
element of the chunk can possibly improve. Cost per pruned
chunk per row drops from ``BN ld + BN setp + (rare) K-deep
bubble inserts`` (~192 ops) down to ``1 ld + 1 setp + 1 BRA``
(~3 ops) -- a 64x reduction. At steady state with random
data >99% of chunks past the first ~K are pruned, so this
cuts the K=20..32 WS3 wall by 12-30% (e.g. K=24 6024 -> 5372
us, K=20 5877 -> 5104 us, K=4 5653 -> 3670 us).

Why it works (the earlier "shelved" analysis was wrong on tile
choice, not on architecture): the prior estimate had t_topk =
7.3ms for ``smem_perthread`` at BM=64 BN=64. That's correct for
that tile -- but at BM=128 BN=64 the per-CTA work doubles, the
CTA count halves, the pipeline overhead amortises, and the wall
collapses to ~6.0ms at K=24 vs the WS2 best of 7.2ms (1.20x
speedup). At BM=64 BN=64 WS3 still loses (~10.8ms vs ~7.2ms);
this is why the autotuner only enables WS3 with BM=128 BN=64.

Constraints (enforced in ``__init__``):
  * ``mma_warp_groups == 1`` so total CTA = 3*128 = 384 threads
    fits the 64K register file. Practically: BM=64, or BM=128
    with BN<=64 (BM=128 BN>=128 has atom_layout=(2,1,1) -> 2 mma
    WGs -> 4 WGs total which doesn't fit).
  * ``topk_strategy == 'smem_perthread'`` -- the only inner top-K
    that doesn't depend on the WGMMA output TV layout for its
    inputs. The dist tile is staged through SMEM in row-major,
    1-thread-per-row, so the top-K WG never touches WGMMA-shaped
    fragments.

dist_stage tuning (REVISED 2026-05-08 EVE after ncu profiling):
  * ``dist_stage=1`` blocks GEMM on every chunk waiting for topK
    drain -- 1.5x worse than dist_stage>=2.
  * ``dist_stage=2`` was the prior default.
  * ``dist_stage=3`` is the NEW default for K_PAD>=14, reliably
    4-11% faster than stage=2 across K=4..32. Why: ncu showed the
    GEMM WG was stalled 49% on shared-store scoreboard (waiting
    for the 2-stage ring to drain); a third buffer keeps GEMM
    running while TopK consumes the other two, restoring overlap.
  * ``dist_stage=4`` fails (CUDA_ERROR_INVALID_VALUE -- 228 KB
    SMEM budget exceeded at BM=128 BN=64).

NCU PROFILE FINDINGS (2026-05-08 EVE, K=24 BM=128 BN=64):

    Metric                    WS3 chunk-min  Triton _flash_knn_kernel
    ------------------------- -------------- --------------------------
    Duration                  7.07 ms        6.07 ms
    Issued IPC                0.23           1.59
    Eligible warps/scheduler  0.27           0.55
    Warp cycles/inst          9.79 cy        4.71 cy
    Active threads/warp avg   18.18 / 32     32.00 / 32 (perfect)
    Branch efficiency         97.75%         100.00%
    Achieved occupancy        14.05%         11.67%
    Shared-store conflicts    73.86% 3.8-way 10.20% 3.7-way
    Top stall                 L1TEX 4.8/9.8  Fixed-lat 1.8/4.7

The dominant inefficiency is the WGMMA m64n64 fp32 acc TV layout
(8 rows x 4 col-pairs per warp) hammering the same 14 banks
during ``_stage_dist_to_smem`` (stride-2 col pattern within
each 4-thread row group). Triton's ``tl.dot`` keeps acc in
registers and never has to relay through SMEM, so it dodges
this entirely.

Tried-and-failed: K_SW128 swizzle on sDist. Reduced shared-store
conflicts from 73.86% to 48.7% but added 37% more instructions
(per-access XOR), making the kernel 19% SLOWER (7.07 -> 8.46 ms).
The standard CUTLASS atom swizzles are tuned for A/B WGMMA OPERAND
loads, not accumulator stores with the stride-2 col pattern --
they only break the in-atom conflicts, not the cross-atom-along-N
patterns that dominate at BN=64.

** MODE H: cross-WG worst_d feedback (2026-05-08 EVE -- BIG WIN) **

The breakthrough was realising that we don't need to FIX the
bank conflicts -- we can just SKIP the SMEM writes entirely for
the 99% of chunks that the consumer would prune anyway. The
mechanism (``Mode H'' in the dispatch table):

  1. Allocate ``sWorstD[BM]`` fp32 in SMEM, init to +inf.
  2. After each chunk the TopK WG processes, write
     ``sWorstD[my_row] = topk_d[K-1]`` (the heap's K-th-worst,
     which is exactly the threshold a new candidate must beat).
  3. The GEMM WG, after WGMMA but BEFORE staging dist to sDist,
     does a 2-pass write:
       Pass 1: compute per-thread per-row min of d (no STS).
               cross-shfl-reduce over 4-threads-in-row.
       Pass 2: read sWorstD[m_local]; if row_min >= worst_d_stale,
               SKIP the row's 16 STS + d-compute. Else write all
               16 d values to sDist (existing path).
     chunk_min is ALWAYS written to sChunkMin (the consumer's
     existing prune gate handles correctness on skipped rows).

Stale-read safety: heap top is monotonically NON-INCREASING
(each insert reduces or keeps), so any sWorstD value GEMM reads
is a CONSERVATIVE upper bound on the current worst_d. GEMM may
process chunks that could have been pruned (correct, just
slightly slower) but never SKIPS a chunk that the consumer
would have processed. Single fp32 LDS/STS are atomic on aligned
addresses -- no torn reads.

NCU comparison at K=24 (post Mode H):

    Metric                    Pre-Mode-H    Mode H stg=3   delta
    ------------------------- ------------- -------------- -------
    Bank conflicts (STS)      153 M         8.7 M          -94%
    Shared-store wavefronts   208 M         34 M           -84%
    short_scoreboard stall    1.13          0.78           -31%
    Wall time (ncu single)    7468 us       6558 us        -12%

Empirical (H200 N=16384 M=100K D=128, post Mode H + dist_stage=3):

    K     Pre-Mode-H stg3   Mode H stg=3   Triton    vs Triton
    ----- ----------------- -------------- --------- -----------
    4     3303 us           2582 us        2987 us   1.16x WIN
    8     3775 us           3068 us        3337 us   1.09x WIN
    12    4481 us           3796 us        3847 us   1.01x WIN
    16    4489 us           3788 us        3946 us   1.04x WIN
    20    5658 us           4991 us        4710 us   0.94x (close)
    24    5657 us           4990 us        4787 us   0.96x (close)
    28    5679 us           5000 us        4823 us   0.96x (close)
    32    5668 us           4997 us        4888 us   0.98x (close)

** TILE-SIZE SWEEP (2026-05-09) -- SMALLER tiles often WIN BIG **

After Mode H closed the SMEM-relay bottleneck, the next
critical insight was that the "default" BM=128 BN=64 was
NOT optimal across shapes. SWEEP across (BM, BN) in
{64, 128} x {64, 128, 256} revealed:

   Shape                     Old: BM128 BN64 stg3   NEW best
   ------------------------  ---------------------  --------------------
   D=64 N=32K M=200K K=4     6908us (1.22x)         BM64 BN64 stg3 6016us (1.40x)
   D=128 N=4K  M=1M   K=16   20488us (1.24x)        BM64 BN64 stg3 14643us (1.73x)
   D=128 N=4K  M=1M   K=4    20833us (0.98x)        BM64 BN128 stg3 11969us (1.70x)
   D=256 N=8K  M=200K K=4    -- (BM=128+D=256 OOM)  BM64 BN64 stg3 3346us (1.84x)
                             P_w fallback 5386us    -- 250.7 TFLOPS!
   D=256 N=8K  M=200K K=16   P_w fallback 9409us    BM64 BN64 stg3 4275us (1.66x)
   D=256 N=8K  M=200K K=24   P_w fallback 34276us   BM128 BN64 stg3 6866us (1.18x)

Why smaller tile wins:
  1. More CTAs => better wave fill on 132 SMs. BM=64 produces
     4x more CTAs than BM=128 at fixed N, slashing wave-tail
     latency.
  2. Less SMEM => can ALWAYS fit dist_stage=3 (deeper pipeline
     => better GEMM/TopK overlap). At D=256 BM=128 BN=128 OOMs;
     BM=64 BN=64 has plenty of headroom.
  3. Lower per-CTA register pressure => higher SM occupancy.

Why the original "big tile = good" intuition failed: that's
the right intuition for raw matmul where GEMM throughput
dominates. After Mode H eliminated the SMEM bank-conflict
wavefronts, the bottleneck shifted to wave-tail latency
(idle SMs waiting for last CTAs to finish), which smaller
tiles + more waves alleviate.

Best per-shape autotune picks (now sweeps {64,128}x{64,128}
x stage{2,3}):

   Shape                K=4         K=16        K=24         K=32
   ------------------- ----------- ----------- ------------ -----------
   D=64 N=32K M=200K   BM64 BN64   BM64 BN64   BM128 BN64   BM128 BN64
   D=128 N=16K M=100K  BM128 BN64  BM128 BN64  BM128 BN64   BM128 BN64
   D=128 N=4K  M=1M    BM64 BN128  BM64 BN64   BM64 BN64    BM64 BN64
   D=256 N=8K  M=200K  BM64 BN64   BM64 BN64   BM128 BN64   BM128 BN64

Pattern: small-K + long-M + large-D => BM=64 BN=64;
medium-batch + small-D => BM=128 BN=64. autotune picks
the right one per shape automatically.

Peak achieved TFLOPS: 250.7 (25.4% of 989 H200 BF16 peak)
at CLIP-style shape K=4. Up from 184.7 TFLOPS (18.7% peak)
before tile-size sweep -- a 36% local TFLOPS jump.
"""
from __future__ import annotations

from typing import Tuple, Type

import math
import os

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# Low-level MLIR helpers used by the packed-Int64 sort path so we can
# bit-cast fp32 -> i32 in the chunk-build inner loop (avoiding the
# recast_tensor write pattern, which the compiler turns into RMW
# half-stores). ``arith_helper.bitcast`` returns an MLIR value; we wrap
# back into a CuTeDSL Numeric.
from cutlass._mlir.extras import types as _mlir_types  # type: ignore
from cutlass._mlir.dialects import llvm as _llvm  # type: ignore

try:  # cutlass-dsl <= 4.5.x
    from cutlass.base_dsl._mlir_helpers import arith as _arith_helper  # type: ignore
except ImportError:  # cutlass-dsl >= 4.6 moved _mlir_helpers to the top level
    from cutlass._mlir_helpers import arith as _arith_helper  # type: ignore


def _bitcast_f32_to_i32(val_f32):
    """Reinterpret an fp32 CuTeDSL value's bit pattern as int32."""
    return cutlass.Int32(
        _arith_helper.bitcast(val_f32.ir_value(), _mlir_types.i32())
    )


def _bitcast_i32_to_u32(val_i32):
    """Reinterpret an int32 CuTeDSL value's bit pattern as uint32."""
    return cutlass.Uint32(
        _arith_helper.bitcast(val_i32.ir_value(), _mlir_types.i32())
    )


# ----------------------------------------------------------------------
# Inline PTX cmp_swap helpers
# ----------------------------------------------------------------------
#
# Why inline PTX: the per-thread bitonic ``_cmp_swap_asc`` lowers
# through MLIR ``arith.cmpf + arith.select`` to PTX ``setp.gt.f32 +
# 4x selp``. ptxas refuses to fold ``(a > b) ? b : a`` into
# ``min.f32`` because of NaN semantics (min.f32 returns the non-NaN
# while the ternary preserves NaN), so we never see FMNMX in the SASS
# even though the bitonic network has thousands of these comparisons.
#
# Hand-rolling the cmp_swap as a single ``llvm.inline_asm`` block
# does two things ptxas can't reach from MLIR-lowered code:
#   1. Uses ``min.f32`` / ``max.f32`` directly for the d outputs --
#      asserting "no NaN" by construction (distances are >=0 finite
#      bf16 GEMM accumulations) so the IEEE-min vs ternary semantics
#      gap doesn't matter.
#   2. Packs all 5 PTX ops into one block so ptxas treats them as a
#      single scheduling unit, which empirically lets it co-issue the
#      i-side selps with the d-side min/max instead of serialising
#      them through one issue queue.
#
# Op count (per cmp_swap) is unchanged at 5 PTX instructions, but the
# scheduling gain is what matters: on H200 SM90a sortmerge BM=128
# BN=128 K=16, this brings the per-chunk sort+merge body from ~5.0 ms
# to ~3.X ms (matches Triton's tl.sort cost on the same shape).
def _cmp_swap_asc_ptx(da, ia, db, ib):
    """Branch-free ascending compare-swap returning (d_min, i_min, d_max, i_max).

    ``da, db`` must be ``cutlass.Float32`` and ``ia, ib`` must be
    ``cutlass.Int32``. Tie-break is dropped (strict ``>``); see
    ``_cmp_swap_asc`` for the rationale (it's safe in our bitonic
    network because the cross-thread butterfly merge is also a
    value-only sort).

    Empirical SASS comparison on H200 SM90a sortmerge BM=128 BN=128:
      * Pure-MLIR lowering -> 1182 FSEL + 1065 SEL + 446 FSETP.GT
        + 224 FSETP.GEU = 5 SASS ops/cmp_swap. Sortmerge K=16 5.0 ms.
      * min.f32/max.f32 + selp variant (commented out below) ->
        1184 FMNMX + 1253 SEL + 158 FSEL + 702 FSETP.GT. Same op
        count but FMNMX has higher latency on the FMA pipe than
        FSEL, so the bitonic critical path lengthens. Net win ~3%.
      * setp+selp packed in one inline-asm block (current) -> same
        op count as the MLIR baseline but ptxas sees the 5 ops as
        one scheduling unit, which lets it intermingle the d-side
        and i-side selps with the next cmp_swap's setp instead of
        serialising. Net win ~10-15%.
    """
    struct_ty = _llvm.StructType.get_literal(
        [_mlir_types.f32(), _mlir_types.i32(),
         _mlir_types.f32(), _mlir_types.i32()]
    )
    # 5 PTX ops per cmp_swap with min.f32 / max.f32 for the d outputs.
    # Same op count as setp+4xselp but the d-side dependency chain
    # is min/max -> done (1c) instead of setp (1c) -> selp (1c) =
    # 2c, which loosens the critical path for the next cmp_swap that
    # depends on the d output. The IEEE min/max NaN semantics differ
    # from the ternary's (we return non-NaN; ternary returns NaN if
    # da is NaN), but distances from non-negative bf16 GEMM are
    # always finite so the difference is unobservable.
    asm = (
        "{\n\t"
        ".reg .pred p;\n\t"
        "min.f32 $0, $4, $6;\n\t"           # d_min = fminnum(da, db)
        "max.f32 $2, $4, $6;\n\t"           # d_max = fmaxnum(da, db)
        "setp.gt.f32 p, $4, $6;\n\t"        # p = (da > db)
        "selp.b32 $1, $7, $5, p;\n\t"       # i_min = p ? ib : ia
        "selp.b32 $3, $5, $7, p;\n\t"       # i_max = p ? ia : ib
        "}"
    )
    res = _llvm.inline_asm(
        struct_ty,
        [
            da.ir_value(), ia.ir_value(),
            db.ir_value(), ib.ir_value(),
        ],
        asm,
        "=f,=r,=f,=r,f,r,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    d_min = cutlass.Float32(_llvm.extractvalue(_mlir_types.f32(), res, [0]))
    i_min = cutlass.Int32(_llvm.extractvalue(_mlir_types.i32(), res, [1]))
    d_max = cutlass.Float32(_llvm.extractvalue(_mlir_types.f32(), res, [2]))
    i_max = cutlass.Int32(_llvm.extractvalue(_mlir_types.i32(), res, [3]))
    return d_min, i_min, d_max, i_max


def _cmp_swap_asc_packed_ptx(pa, pb):
    """Branch-free ascending compare-swap on packed Int64 returning (p_min, p_max).

    Inputs/outputs are ``cutlass.Int64`` carrying the packed
    (fp32_distance_bits << 32) | uint32_index bit pattern (Triton's
    flash_knn stage1 layout). Single 64-bit unsigned compare gives
    the value-asc + smaller-index-tie-break semantics for free.

    On Hopper SM90 there's no native 64-bit ALU, so the packed compare
    lowers to ``ISETP.GT.U32 + ISETP.GT.U32.EX`` (carry-chained) and
    each ``selp.b64`` lowers to 2 SEL.b32 (low + high halves). Total
    SASS per cmp_swap: 2 ISETP + 4 SEL = 6 ops, vs the unpacked
    (d, i) version's 1 FSETP + 4 SEL = 5 ops. So packed costs +1 op
    per swap on Hopper -- IT'S NOT A WIN ON HOPPER. We keep this
    helper as documentation of the 64-bit inline-PTX path; on future
    arch with a native 64-bit ALU (or where the MLIR Int64 lowering
    spills more, e.g. K_PAD>=24 with sortmerge_packed) this can
    reduce spill pressure (~40 bytes less than the MLIR baseline at
    K_PAD=32).
    """
    asm = (
        "{\n\t"
        ".reg .pred p;\n\t"
        # 64-bit unsigned compare; ptxas lowers to ISETP.GT.U32 +
        # ISETP.GT.U32.EX (carry-chained) at SASS level.
        "setp.gt.u64 p, $2, $3;\n\t"
        "selp.b64 $0, $3, $2, p;\n\t"   # p_min = p ? pb : pa
        "selp.b64 $1, $2, $3, p;\n\t"   # p_max = p ? pa : pb
        "}"
    )
    struct_ty = _llvm.StructType.get_literal(
        [_mlir_types.i64(), _mlir_types.i64()]
    )
    res = _llvm.inline_asm(
        struct_ty,
        [pa.ir_value(), pb.ir_value()],
        asm,
        "=l,=l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
    )
    p_min = cutlass.Int64(_llvm.extractvalue(_mlir_types.i64(), res, [0]))
    p_max = cutlass.Int64(_llvm.extractvalue(_mlir_types.i64(), res, [1]))
    return p_min, p_max


# maxtree worst-of-K: K at/above which the streaming running-max beats the
# balanced max-tree (the max-tree's K live leaves spill in CuteDSL/MLIR). H100
# router-tile crossover; see ``HopperFlashKnnFused._worst_row``.
_MAXTREE_STREAM_KMIN = 11


class HopperFlashKnnFused:
    """Hopper TMA+WGMMA fused KNN kernel with register-resident top-K."""

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        m_block_size: int,
        n_block_size: int,
        k_pad: int,
        use_ws: bool = False,
        topk_strategy: str = "auto",
        use_ws3: bool = False,
        dist_stage: int = 2,
        use_acc_pipeline: bool = False,
        use_ws4: bool = False,
    ):
        """Configure the kernel.

        Parameters
        ----------
        acc_dtype:
            WGMMA accumulator dtype. Must be ``cutlass.Float32`` for
            numerically stable distance computation.
        m_block_size:
            CTA tile in the queries (M-of-GEMM) dim. Must be 64 or 128.
            Each CTA processes this many query rows.
        n_block_size:
            CTA tile in the database (N-of-GEMM, streamed) dim. Must be
            64, 128, or 256. Streamed via TMA; this is the per-tile
            mini-batch of database points contracting against the loaded
            X tile.
        k_pad:
            Per-thread per-row top-K register heap size. Must be 16 or
            32 (power-of-two for fast argmax loops; both fit in 240
            regs/thread when combined with the WGMMA acc).
        use_ws:
            FMHA-style producer/consumer warp specialization.
        """
        if acc_dtype is not cutlass.Float32:
            raise TypeError("acc_dtype must be Float32 for numerically stable knn dist")
        if m_block_size not in (64, 128, 256):
            raise ValueError("m_block_size must be 64, 128, or 256")
        if n_block_size not in (64, 128, 256):
            raise ValueError("n_block_size must be 64, 128, or 256")
        if not (1 <= k_pad <= 32):
            raise ValueError("k_pad must be between 1 and 32 (register-heap variant)")

        self.acc_dtype = acc_dtype
        self.tile_shape_mnk = (m_block_size, n_block_size, 1)
        # WGMMA atom shape for fp16/bf16 has M=64 fixed; we tile in M
        # via atom_layout. BM=64 / 128 / 256 → (1,1,1) / (2,1,1) / (4,1,1).
        #
        # WS3 special-case: WS3 needs total CTA = 3 WGs (load+gemm+topk),
        # so the GEMM stage can only have ONE warp group. For BM=128
        # BN>=128 we'd normally pick atom_layout=(2,1,1) (2 parallel
        # WGMMA m64nNk16 atoms = 2 WGs). For WS3 we instead force
        # atom_layout=(1,1,1) and let the single GEMM WG issue the two
        # m64 WGMMAs SEQUENTIALLY (one for rows 0-63, one for 64-127).
        # Costs: 2x WGMMA latency in the GEMM critical path. Wins:
        # unlocks BN>=128 for WS3, halving the chunk count and
        # doubling arithmetic intensity per chunk.
        if m_block_size == 256:
            self.atom_layout_mnk = (4, 1, 1)
        elif (
            m_block_size == 128
            and n_block_size >= 128
            and not bool(use_ws3)
        ):
            self.atom_layout_mnk = (2, 1, 1)
        else:
            self.atom_layout_mnk = (1, 1, 1)
        self.cluster_shape_mn = (1, 1)
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.k_pad = int(k_pad)
        # k_pad rounded up to next pow2 for the bitonic networks. Slots in
        # [k_pad, k_pad_pow2) are kept at +inf and never written out.
        self.k_pad_pow2 = self._next_pow2(self.k_pad)
        # Top-K strategy. The autotuner sweeps strategies + tiles +
        # use_ws and picks the empirical winner per shape; the only
        # compile-time gating here is the legacy ``"auto"`` fallback
        # used by paths that haven't migrated to the autotuner yet.
        #
        # Strategies:
        #   ``insert``    : cooperative chunk-best argmin + replicated
        #                   per-thread heap (no post-loop merge).
        #   ``perthread`` : CUTLASS-style per-thread sorted-asc top-K
        #                   + branch-free bubble insert + butterfly.
        #                   Mirrors ``Sm90TopKSoftmaxColReduction`` in
        #                   ``include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp``.
        #   ``sortmerge`` : per-thread bitonic chunk-sort + merge into
        #                   running sorted top-K + butterfly.
        #
        # Note re Triton's ``tl.sort``: it's also a software bitonic
        # (see ``triton/python/triton/language/standard.py::_compare_and_swap``)
        # but it vectorises across SIMT lanes via reshape-as-hypercube
        # + the ``xor_sum`` swap trick, so each compare-swap becomes a
        # single warp shuffle. CuteDSL has no layout-aware sort
        # primitive, so a per-thread bitonic emits the entire network
        # sequentially. Closing the K>=24 gap requires either a manual
        # lane-cooperative bitonic across threads-in-row using
        # ``cute.arch.shuffle_sync_bfly``, or warp specialisation that
        # gives top-K its own warpgroup with a row-major 1-thread-per-row
        # layout (no shuffles) overlapped with GEMM via SMEM dist
        # staging.
        valid_strategies = (
            "insert", "perthread",
            "sortmerge", "sortmerge_packed",
            "smem_perthread", "warpsort", "auto",
            # Maxtree top-K ported from the Blackwell BUILD kernel:
            #   ``maxtree``      : register per-thread unsorted heap +
            #                   group-min-4 prune + max-tree (the
            #                   register analog of ``perthread``).
            #   ``smem_maxtree`` : 1-thread-per-row WS3/WS4 variant of
            #                   ``smem_perthread`` with the same maxtree
            #                   inner loop.
            "maxtree", "smem_maxtree",
        )
        if topk_strategy not in valid_strategies:
            raise ValueError(
                f"topk_strategy must be one of {valid_strategies}, got {topk_strategy!r}"
            )
        if topk_strategy == "auto":
            # Empirical default kept ONLY for callers that don't yet drive
            # the autotuner. Production should pass an explicit strategy.
            # NOTE: ``sortmerge_packed`` (Int64 packed bitonic) is
            # implemented and correct, but on Hopper SM (which lacks
            # native 64-bit integer ALU throughput) the resulting
            # bitonic network is consistently 20-40% slower than the
            # plain (fp32, int32) ``sortmerge`` once the tie-break is
            # dropped from the comparator. We keep ``sortmerge_packed``
            # as an opt-in autotune candidate but the default for
            # K_PAD>16 stays on ``sortmerge``.
            if self.k_pad <= 2:
                self.topk_strategy = "insert"
            elif self.k_pad <= 16:
                self.topk_strategy = "perthread"
            else:
                self.topk_strategy = "sortmerge"
        else:
            self.topk_strategy = topk_strategy

        # smem_perthread / smem_maxtree / warpsort use CTA-wide
        # ``cute.arch.sync_threads`` for the SMEM dist relayout. That
        # barrier includes producer warps when use_ws=True, which
        # deadlocks with the producer/consumer pipeline. WS3 (separate
        # top-K warpgroup with named barriers) lifts this restriction
        # (see ``use_ws3``).
        if (
            self.topk_strategy in ("smem_perthread", "smem_maxtree", "warpsort")
            and use_ws and not use_ws3
        ):
            raise ValueError(
                f"topk_strategy={self.topk_strategy!r} is incompatible "
                "with use_ws=True (uses CTA-wide sync_threads which "
                "deadlocks with the producer/consumer pipeline). Set "
                "use_ws=False, set use_ws3=True, or pick another "
                "strategy."
            )
        # ``smem_maxtree`` is only wired into the WS3 / WS4 top-K warpgroup
        # (the 1-thread-per-row layout the maxtree inner loop needs); it has
        # no non-WS code path. Require use_ws3 so misconfigurations fail
        # loudly at construction rather than tracing the wrong kernel.
        if self.topk_strategy == "smem_maxtree" and not use_ws3:
            raise ValueError(
                "topk_strategy='smem_maxtree' requires use_ws3=True "
                "(WS3/WS4 top-K warpgroup). Use 'maxtree' for the non-WS / "
                "WS2 register path."
            )

        self.use_ws = bool(use_ws)
        self.use_ws3 = bool(use_ws3)
        # maxtree worst-of-K recompute mode. "auto" (default): K-adaptive --
        # the streaming running-max (2 live scalars) for K>=_MAXTREE_STREAM_KMIN
        # where the balanced max-tree's K live leaves spill in CuteDSL/MLIR,
        # else the shallower max-tree. (The Blackwell BUILD kernel hit the same
        # spill and switched to streaming; Hopper rides the same MLIR backend.)
        # "0"/"1" pin tree/stream for A/B.
        self._worst_stream_mode = os.environ.get(
            "FLASHLIB_KNN_WORST_STREAM", "auto")
        # Optimization B: WS4 architecture (2 GEMM warpgroups + 1
        # TopK warpgroup + 1 Load warpgroup). Each GEMM WG runs on
        # a different SM sub-partition's WGMMA pipe, so the GEMM
        # throughput potentially doubles. The two GEMM WGs alternate
        # processing chunks (GEMM-A: 0,2,4,...; GEMM-B: 1,3,5,...);
        # each has its own ``c_pipeline`` (Load -> GEMM) and
        # ``dist_pipeline`` (GEMM -> TopK). The single TopK WG
        # consumes from both dist pipelines in chunk order.
        # Constraints:
        #   * Requires use_ws3 = True (uses the same ws3 pipeline
        #     primitives + Mode H sWorstD feedback).
        #   * SMEM doubles for sC and sDist (need ~2x), so c_stage
        #     and dist_stage are conservative (2 each typically).
        #   * Register budget: load=24 + gemm*2 + topk <= 512 per
        #     thread average (64K reg / 128 threads / 4 WGs). With
        #     gemm=192, topk=96 we get 24+192+192+96=504. Tight.
        # WS4 SMEM gate: doubles sC + sDist. The largest tile (BM=128
        # BN=128) doesn't fit (sC=2*64KB = 128KB, sDist=2*68KB = 136KB,
        # plus sX = 32KB → > 228KB SMEM cap). Cap at BM*BN <= 128*64
        # for safety (smaller tiles always fit).
        if (
            bool(use_ws4)
            and bool(use_ws3)
            and (m_block_size * n_block_size) <= 128 * 64
        ):
            self.use_ws4 = True
        else:
            self.use_ws4 = False
        # Optimization A: WGMMA pipelining (2-stage acc ping-pong).
        # Hides WGMMA latency by issuing chunk N+1's WGMMA while
        # still doing chunk N's d-compute / SMEM staging. Requires
        # 2x acc registers, gated to tiles where doubled acc fits
        # within the 255-reg/thread Hopper cap. Per-thread acc =
        # BM*BN/128 fp32; need <= 128 (= 128 reg) doubled, so
        # require BM*BN <= 128*128 = 16384. Empirically the
        # crucial gate is BM*BN <= 128*64 = 8192 because BM=128
        # BN=128 hits the 256-reg ceiling (128 fp32 acc *2) which
        # is hard up against the 255 cap and OOMs other state.
        # Auto-disabled when not WS3 (no benefit -- WS2 has no
        # GEMM/TopK pipeline to hide behind).
        if (
            bool(use_acc_pipeline)
            and bool(use_ws3)
            and (m_block_size * n_block_size) <= 128 * 64
        ):
            self.use_acc_pipeline = True
        else:
            self.use_acc_pipeline = False
        # WS3: 3 warp groups (load, GEMM, top-K). Top-K runs in its OWN
        # warpgroup with the smem_perthread strategy reading dist from
        # a multi-staged SMEM ring buffer; the GEMM WG produces the
        # dist tile per chunk and signals via a per-WG mbarrier
        # pipeline. Constraints:
        #   * Only ``smem_perthread`` topk strategy makes sense here
        #     (the per-thread bubble is the only one that doesn't need
        #     the WGMMA-TV layout for its inputs).
        #   * The GEMM WG owns the WGMMA atom layout, so we only allow
        #     mma_warp_groups == 1 to keep the total WG count at 3
        #     (load + gemm + topk = 3 * 128 = 384 threads / CTA). For
        #     BM=128 with atom_layout=(2,1,1) we'd need 4 WGs total,
        #     which is fine register-wise but pushes SMEM tighter --
        #     supported but not the default.
        #   * Multi-stage dist SMEM is required for any overlap. With
        #     ``dist_stage=1`` the GEMM WG would block on every chunk
        #     waiting for the TopK WG to consume the previous one; the
        #     default ``dist_stage=2`` lets one chunk's GEMM run in
        #     parallel with the previous chunk's TopK.
        if self.use_ws3:
            if not self.use_ws:
                raise ValueError("use_ws3=True requires use_ws=True")
            if self.topk_strategy not in ("smem_perthread", "smem_maxtree"):
                raise ValueError(
                    f"use_ws3=True requires topk_strategy in "
                    f"('smem_perthread', 'smem_maxtree'), "
                    f"got {self.topk_strategy!r}"
                )
            if self.mma_warp_groups != 1:
                raise ValueError(
                    f"use_ws3=True requires mma_warp_groups==1 (BM=64 or "
                    f"BM=128 with BN=64). Got mma_warp_groups="
                    f"{self.mma_warp_groups} (BM={m_block_size} "
                    f"BN={n_block_size})."
                )
        if self.use_ws4:
            # WS4 reuses ws3's pipeline machinery + 1 extra GEMM WG.
            # Tile-size gate: each GEMM WG's acc must fit in 192 reg
            # budget. acc per thread = BM*BN/128 fp32. Need acc <=
            # 128 reg (leaving ~64 for other state). So BM*BN <=
            # 128*128 = 16384 -- excludes very large tiles only.
            if (m_block_size * n_block_size) > 128 * 128:
                raise ValueError(
                    f"use_ws4=True requires BM*BN <= 128*128 (acc fits "
                    f"in 192 reg). Got BM={m_block_size} BN={n_block_size}."
                )

        # SMEM dist staging size (fp32 elements). Allocated for
        # smem_perthread strategy; placeholder 1 element otherwise (the
        # MemRange API in some CuTeDSL versions disallows zero-sized).
        #
        # Two layouts coexist:
        #
        # (A) Padded row-major (legacy non-WS3 paths).
        #     BN+1 row stride breaks the consumer-side 32-way bank
        #     conflict on ``sDist[my_row, j]`` reads (each top-K thread
        #     reads 0..BN-1; with stride=BN=128 fp32 and 32 banks of 4
        #     bytes, all 32 threads in a warp hit bank j%32 because
        #     128 % 32 == 0). Padding to BN+1 makes stride%32 == 1, so
        #     each thread lands on a distinct bank.
        #
        # (B) CUTLASS K_SW128 XOR-swizzled layout (WS3 path).
        #     The legacy (A) padding only fixes the CONSUMER-side
        #     conflict but the WGMMA ``_stage_dist_to_smem`` PRODUCER
        #     write hits a 3.8-way conflict (ncu reports 73.86% of
        #     shared store wavefronts conflict, est. 22% local speedup
        #     if fixed).  The K_SW128 swizzle XORs row bits into col
        #     offsets within each 8x32-fp32 atom, breaking BOTH the
        #     producer and consumer access patterns to a true
        #     permutation of banks, killing all conflicts at the cost
        #     of one extra index XOR per access (free in HW).
        #
        # WS3 multi-stages the dist tile: total size grows by
        # ``dist_stage`` so GEMM and TopK can ping-pong on different
        # buffers.
        if self.topk_strategy in ("smem_perthread", "smem_maxtree", "warpsort"):
            self.dist_smem_row_stride = n_block_size + 1
            self.dist_stage = max(1, int(dist_stage)) if self.use_ws3 else 1
            self.dist_smem_size = (
                m_block_size * self.dist_smem_row_stride * self.dist_stage
            )
            self.use_dist_swizzle = False
        else:
            self.dist_smem_row_stride = 1
            self.dist_stage = 1
            self.dist_smem_size = 1
            self.use_dist_swizzle = False
        # WS3 per-row chunk-min SMEM. Sized BM x dist_stage fp32. The
        # GEMM WG fills sChunkMin[my_row, stage] = min over BN cols of
        # the dist row before signalling the dist_pipeline producer;
        # the TopK WG checks ``chunk_min < worst_d`` once per chunk
        # and skips the entire BN-element bubble-insert loop when
        # pruned. This mirrors Triton stage-1's ``if chunk_best <
        # topk_worst_val`` early exit. At steady state >99% of chunks
        # get pruned, taking the per-chunk topK cost from BN ld+setp
        # ops down to 1 ld+setp -- the dominant smem_perthread
        # speedup at K_PAD>=14.
        if self.use_ws3:
            self.chunk_min_smem_size = m_block_size * self.dist_stage
            # WS3 Mode-H cross-WG worst_d feedback. Sized BM fp32 (one
            # entry per row, NOT per stage -- the value is logically
            # "current worst_d" not "per-chunk"). TopK WG writes its
            # heap[K-1] back here after each consumed chunk; GEMM WG
            # reads it BEFORE writing sDist for that row. If
            # row_chunk_min >= sWorstD[m_local] the GEMM WG SKIPS the
            # 16 STS for that row's portion of sDist (and the d
            # compute that would feed them), eliminating ~99% of the
            # 73.86% bank-conflict-bound shared stores ncu identified.
            #
            # Stale-read safety: heap top is monotonically non-
            # increasing (each insert reduces or keeps), so any read
            # of sWorstD by GEMM gives a value >= current worst_d.
            # That's a CONSERVATIVE bound: GEMM may process chunks
            # that could have been pruned (correct, just slower) but
            # never skips chunks it should process (correctness OK).
            # Single fp32 STS/LDS are atomic on aligned addresses, no
            # tearing. Init to +inf so chunk 0 always gets processed.
            self.worst_d_smem_size = m_block_size
        else:
            self.chunk_min_smem_size = 1
            self.worst_d_smem_size = 1
        self.num_load_warp_groups = 1 if self.use_ws else 0
        # WS3 adds one extra warp group dedicated to top-K. Total
        # threads_per_cta = (load + gemm + topk) * 128.
        # WS4 adds an EXTRA gemm WG (gemm_b) for 2-WG WGMMA pipe
        # parallelism: load + gemm_a + gemm_b + topk = 4 * 128.
        self.num_topk_warp_groups = 1 if self.use_ws3 else 0
        self.num_extra_gemm_warp_groups = 1 if self.use_ws4 else 0
        self.threads_per_cta = (
            (
                self.num_load_warp_groups
                + self.mma_warp_groups
                + self.num_extra_gemm_warp_groups
                + self.num_topk_warp_groups
            )
            * self.num_threads_per_warp_group
        )
        self.load_warp_group_id = 0
        # In WS3: load=0, gemm=1, topk=2.
        # In WS4: load=0, gemm_a=1, gemm_b=2, topk=3.
        self.gemm_warp_group_id = self.num_load_warp_groups
        self.gemm_b_warp_group_id = (
            self.num_load_warp_groups + self.mma_warp_groups
        )
        self.topk_warp_group_id = (
            self.num_load_warp_groups
            + self.mma_warp_groups
            + self.num_extra_gemm_warp_groups
        )
        self.num_consumer_warps = self.mma_warp_groups * 4

        # Register quotas. WS4 sums to 504/512 budget per thread:
        #   load(24) + gemm_a(192) + gemm_b(192) + topk(96) = 504.
        # WS3 sums to 432/512 budget per thread:
        #   load(24) + gemm(240) + topk(168) = 432.
        # Non-WS uses 240 mma reg with no special split.
        self.num_regs_load = 24
        if self.use_ws4:
            # Two GEMM WGs each at 192. Verified to fit BM<=128 BN<=64
            # tile state (acc 32-64 fp32 + xs/cs/loop state ~50 = ~110
            # actual usage; 192 budget gives 80 reg headroom for spill
            # avoidance). Larger tiles (BM=128 BN=128, acc=128 fp32)
            # would exceed 192 -- gated below with a check.
            self.num_regs_mma = 192
        else:
            self.num_regs_mma = 240
        # WS3 register budget: 1 CTA = 384 threads x avg-regs/thread
        # must fit in the SM's 64K register file. Splitting:
        #   * Load WG     : 24 regs * 128 = 3072
        #   * GEMM WG     : 240 regs * 128 = 30720
        #   * TopK WG     : N regs * 128 = 128N
        # Sum <= 65536 -> N <= 248. We pick 168 (the next setmaxregister
        # tier below 184 on Hopper) which leaves headroom for occupancy
        # and matches the actual TopK-WG live-set:
        #   heap_d  + heap_i = 2 * K_PAD * 4 bytes = 256 bytes max @
        #     K=32 -> 64 regs
        #   pending_d/pending_i + worst_d + loop state ~= 16 regs
        #   tile coordinates ~= 8 regs
        # Total ~88 regs needed; 168 gives ~2x headroom (compiler may
        # prefer to keep some intermediates live). Bigger value just
        # reduces occupancy without helping the inner loop.
        if self.use_ws4:
            # WS4: 96 reg for TopK (was 168 in WS3). The TopK
            # warpgroup state is small (~70 reg actual), so 96 is
            # sufficient with some headroom. Verified to fit K<=32.
            self.num_regs_topk = 96
        else:
            self.num_regs_topk = 168 if self.use_ws3 else 0

        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: Tuple[int, int],
    ) -> Tuple[cute.CopyAtom, cute.Tensor]:
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile,
        )

    # ----------------------------------------------------------------------
    # Host-side entry
    # ----------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,         # (N, D)         queries fp16/bf16, k-major
        c: cute.Tensor,         # (M, D)         database fp16/bf16, k-major
        c_sq: cute.Tensor,      # (M,)           fp32
        out_idxs: cute.Tensor,  # (N, K_PAD)     int32 — indices only
        stream: cuda.CUstream,
    ):
        x_dtype = x.element_type
        c_dtype = c.element_type
        if cutlass.const_expr(x_dtype != c_dtype):
            raise TypeError("x and c dtype must match")
        if cutlass.const_expr(x_dtype.width != 16):
            raise TypeError("x dtype must be fp16 or bf16")
        self.x_dtype = x_dtype
        self.c_dtype = c_dtype

        x_layout = utils.LayoutEnum.from_tensor(x)
        c_layout = utils.LayoutEnum.from_tensor(c)
        if cutlass.const_expr(x_layout.sm90_mma_major_mode() != cute.nvgpu.warpgroup.OperandMajorMode.K):
            raise RuntimeError("x must be k-major (D contiguous)")
        if cutlass.const_expr(c_layout.sm90_mma_major_mode() != cute.nvgpu.warpgroup.OperandMajorMode.K):
            raise RuntimeError("c must be k-major (D contiguous)")

        D = x.shape[1]
        self.tile_shape_mnk = (self.tile_shape_mnk[0], self.tile_shape_mnk[1], D)

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            x_dtype, c_dtype,
            x_layout.sm90_mma_major_mode(),
            c_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )

        x_bytes_per_stage = (
            self.tile_shape_mnk[0] * self.tile_shape_mnk[2] * x_dtype.width // 8
        )
        c_bytes_per_stage = (
            self.tile_shape_mnk[1] * self.tile_shape_mnk[2] * c_dtype.width // 8
        )
        mbar_helpers_bytes = 1024
        # Reserve SMEM for sDist + sChunkMin + sWorstD (WS3-only) so
        # the c_stage choice doesn't collide with the dist staging
        # ring buffer. Without this, BM=128 BN=128 WS3 would set
        # c_stage=4 (sC=128KB) and then OOM when sDist (132KB at
        # dist_stage=2) gets allocated. fp32 = 4 B/elem.
        ws3_aux_bytes = (
            self.dist_smem_size * 4
            + self.chunk_min_smem_size * 4
            + self.worst_d_smem_size * 4
            # 3x buffer_align for sDist/sChunkMin/sWorstD struct
            # alignment padding (cute.struct.Align[..., 1024])
            + 3 * 1024
            if self.use_ws3 else 0
        )
        # WS4 doubles sDist + sChunkMin (one ring per GEMM WG).
        # sWorstD remains shared. Also doubles sC requirement so we
        # halve the c_stage budget below.
        ws4_aux_bytes = (
            self.dist_smem_size * 4 + self.chunk_min_smem_size * 4
            + 2 * 1024  # struct.Align padding for sDist_b, sChunkMin_b
            if self.use_ws4 else 0
        )
        budget = (
            self.smem_capacity
            - mbar_helpers_bytes
            - x_bytes_per_stage
            - ws3_aux_bytes
            - ws4_aux_bytes
        )
        # WS4 alternates the c_pipeline between sC_a and sC_b, so
        # the per-GEMM-WG c_stage works on HALF of the c byte
        # budget. Both pipelines are sized identically.
        c_stage_divisor = 2 * c_bytes_per_stage if self.use_ws4 else c_bytes_per_stage
        c_stage = budget // c_stage_divisor
        c_stage = max(1, min(4, c_stage))
        self.c_stage = c_stage
        self.x_stage = 1

        self.x_smem_layout_staged = sm90_utils.make_smem_layout_a(
            x_layout, self.tile_shape_mnk, x_dtype, self.x_stage
        )
        self.c_smem_layout_staged = sm90_utils.make_smem_layout_b(
            c_layout, self.tile_shape_mnk, c_dtype, self.c_stage
        )

        # NOTE on sDist swizzle attempt: K_SW128 swizzle was tried for
        # use_ws3 to break the 73.86% shared-store bank conflicts ncu
        # measured. Result: conflicts dropped to 48.7% (better) but
        # the per-access XOR added 37% more instructions, slowing the
        # kernel from 7.07ms to 8.46ms. The WGMMA m64n64 fp32 TV
        # layout (8 rows x 4 col-pairs per warp) doesn't compose
        # cleanly with the standard CUTLASS atom swizzles -- those
        # are tuned for A/B WGMMA OPERAND loads, not for accumulator
        # stores with stride-2 col pattern. Reverted to the simple
        # BN+1 padded layout. Future direction: vectorize SMEM reads
        # (ldmatrix or ld.shared.v4.f32) to cut access count rather
        # than fixing per-access bank conflicts.
        self.dist_smem_layout_staged = None

        tma_atom_x, tma_tensor_x = self._make_tma_atoms_and_tensors(
            x, self.x_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
        )
        tma_atom_c, tma_tensor_c = self._make_tma_atoms_and_tensors(
            c, self.c_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
        )

        N = x.shape[0]
        num_m_tiles = (N + self.tile_shape_mnk[0] - 1) // self.tile_shape_mnk[0]
        grid = (num_m_tiles, 1, 1)

        # Optional dist_smem for the SMEM-relayout top-K paths
        # (smem_perthread). Sized BM x BN fp32 -- a single chunk's
        # distances staged through SMEM so the top-K warp(s) can read
        # one row entirely from a single thread (no cross-thread
        # shuffles in the top-K loop). Only allocated when the chosen
        # strategy needs it; for register-only paths the field shrinks
        # to 0 bytes so smem budget stays the same.
        # WS3 needs an extra mbarrier pipeline for the GEMM->TopK dist
        # handoff. Each stage reserves 2 mbarriers (1 producer-arrived,
        # 1 consumer-arrived); the count below matches PipelineAsync's
        # internal storage layout.
        dist_pipeline_storage = (
            self.dist_stage * 2 if self.use_ws3 else 0
        )
        # WS4 doubles c_pipeline storage (1 per GEMM WG, alternating
        # chunks via the load WG) and dist_pipeline storage (1 per
        # GEMM WG), for two independent producer/consumer queues.
        ws4_extra_c_pipe_slots = self.c_stage * 2 if self.use_ws4 else 0
        ws4_extra_dist_pipe_slots = (
            dist_pipeline_storage if self.use_ws4 else 0
        )
        ws4_extra_sC_size = (
            cute.cosize(self.c_smem_layout_staged) if self.use_ws4 else 1
        )
        ws4_extra_sDist_size = (
            self.dist_smem_size if self.use_ws4 else 1
        )
        ws4_extra_chunkmin_size = (
            self.chunk_min_smem_size if self.use_ws4 else 1
        )

        @cute.struct
        class SharedStorage:
            c_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.c_stage * 2]
            x_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.x_stage * 2]
            dist_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, max(1, dist_pipeline_storage)
            ]
            # WS4-only secondary pipelines (size = 1 placeholder when WS3-only)
            c_pipeline_b_array_ptr: cute.struct.MemRange[
                cutlass.Int64, max(1, ws4_extra_c_pipe_slots)
            ]
            dist_pipeline_b_array_ptr: cute.struct.MemRange[
                cutlass.Int64, max(1, ws4_extra_dist_pipe_slots)
            ]
            sX: cute.struct.Align[
                cute.struct.MemRange[x_dtype, cute.cosize(self.x_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[c_dtype, cute.cosize(self.c_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # WS4-only sC_b for the second GEMM WG's C tiles
            sC_b: cute.struct.Align[
                cute.struct.MemRange[c_dtype, ws4_extra_sC_size],
                self.buffer_align_bytes,
            ]
            sDist: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.dist_smem_size],
                self.buffer_align_bytes,
            ]
            # WS4-only sDist_b for the second GEMM WG's dist tiles
            sDist_b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, ws4_extra_sDist_size],
                self.buffer_align_bytes,
            ]
            sChunkMin: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.chunk_min_smem_size],
                self.buffer_align_bytes,
            ]
            sChunkMin_b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, ws4_extra_chunkmin_size],
                self.buffer_align_bytes,
            ]
            sWorstD: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.worst_d_smem_size],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        if cutlass.const_expr(self.use_ws4):
            self.kernel_ws4(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out_idxs,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )
        elif cutlass.const_expr(self.use_ws3):
            self.kernel_ws3(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out_idxs,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )
        elif cutlass.const_expr(self.use_ws):
            self.kernel_ws(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out_idxs,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )
        else:
            self.kernel(
                tma_atom_x, tma_tensor_x,
                tma_atom_c, tma_tensor_c,
                c_sq, out_idxs,
                self.tiled_mma,
                self.x_smem_layout_staged,
                self.c_smem_layout_staged,
            ).launch(
                grid=grid,
                block=[self.threads_per_cta, 1, 1],
                stream=stream,
            )

    # ----------------------------------------------------------------------
    # Non-warp-specialised device kernel
    # ----------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_md: cute.Tensor,
        mCsq_m: cute.Tensor,
        mOutI_nk: cute.Tensor,    # (N, K_PAD) int32  — indices only
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        BM = self.tile_shape_mnk[0]   # = BN_query (queries per CTA)
        BN = self.tile_shape_mnk[1]   # = BM_db (database streamed per tile)
        N_total = mX_nd.shape[0]      # global query count
        M_total = mC_md.shape[0]      # global database count
        cta_m_offset = bidx * BM      # this CTA's query row offset
        K_PAD = self.k_pad

        # --- SMEM + pipelines ----------------------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype, cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=cute.size_in_bytes(
                self.c_dtype, cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        num_c_tiles = (M_total + BN - 1) // BN
        gC_md = cute.local_tile(
            mC_md, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), (None, 0),
        )

        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x, 0, cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(
                cute.local_tile(
                    mX_nd,
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),
        )
        tma_cS, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_md, 0, 2),
        )

        thr_mma = tiled_mma.get_slice(tidx)
        tCsX = thr_mma.partition_A(sX)
        tCsC = thr_mma.partition_B(sC)
        tCrX = tiled_mma.make_fragment_A(tCsX)
        tCrC = tiled_mma.make_fragment_B(tCsC)

        cP = cute.make_identity_tensor((BM, BN))
        ptPcP = thr_mma.partition_C(cP)

        gC_fake = cute.make_identity_tensor((BM, BN))
        tCgC_fake = thr_mma.partition_C(gC_fake)
        acc_shape = tCgC_fake.shape
        acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- Issue X TMA + prefetch C tiles --------------------------------
        x_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.x_stage
        )
        c_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.c_stage
        )
        prefetch_c_tile_cnt = cutlass.max(cutlass.min(self.c_stage, num_c_tiles), 0)

        if warp_idx == 0:
            x_pipeline.producer_acquire(x_producer_state)
            cute.copy(
                tma_atom_x,
                tma_xG[(None, bidx)],
                tma_xS[(None, x_producer_state.index)],
                tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
            )
            x_pipeline.producer_commit(x_producer_state)
            x_producer_state.advance()

            for c_pre in cutlass.range(prefetch_c_tile_cnt, unroll=1):
                c_pipeline.producer_acquire(c_producer_state)
                cute.copy(
                    tma_atom_c,
                    tma_cG[(None, c_producer_state.count)],
                    tma_cS[(None, c_producer_state.index)],
                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                )
                c_pipeline.producer_commit(c_producer_state)
                c_producer_state.advance()

        # --- Per-thread state ---------------------------------------------
        acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
        acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
        ptPcP_mn = cute.make_tensor(
            ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
        )
        M_per_thr = cute.size(acc_mn, mode=[0])
        N_per_thr = cute.size(acc_mn, mode=[1])

        # x_sq is dropped — the score is c_sq[m] − 2·cross.

        # Per-thread per-row top-K register state.
        #
        # ROWS_OWNED is the number of query rows this thread maintains
        # top-K state for:
        #   * insert / perthread / sortmerge -> M_per_thr (each thread
        #     owns the rows from its WGMMA TV slice).
        #   * smem_perthread -> rows_owned_smem (each thread owns one
        #     row of the BM tile if its tidx < BM, zero otherwise --
        #     state is sized 1 unconditionally and gated at runtime).
        #
        # K_INTERNAL is the heap width: K_PAD for everything except
        # sortmerge / sortmerge_packed which need the next pow2 for the
        # bitonic network.
        K_INTERNAL = self.k_pad_pow2 if cutlass.const_expr(
            self.topk_strategy in ("sortmerge", "sortmerge_packed")
        ) else K_PAD
        if cutlass.const_expr(self.topk_strategy == "smem_perthread"):
            ROWS_OWNED = 1
        else:
            ROWS_OWNED = M_per_thr

        # warpsort: per-warp top-K state distributed across 32 lanes.
        # Each warp handles ROWS_PER_WARP = BM / num_consumer_warps
        # rows; lane l holds the l-th smallest distance of each row in
        # an array of length ROWS_PER_WARP per lane. Lanes K_PAD..31
        # are kept at +INF so they sort to the tail and never get
        # picked into the top-K. K_PAD must be a power of 2 in [1, 32]
        # for warpsort (we sort exactly 32 elements per warp; K_PAD<32
        # is correct via INF padding but wastes some sort work).
        ROWS_PER_WARP = BM // self.num_consumer_warps
        if cutlass.const_expr(self.topk_strategy == "warpsort"):
            heap_d_arr = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Float32
            )
            heap_i_arr = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Int32
            )
            # Per-warp chunk scratch: ROWS_PER_WARP entries per lane,
            # one per row processed in parallel by this warp. Updated
            # per sub-chunk and consumed by the merge.
            chunk_d_arr = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Float32
            )
            chunk_i_arr = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Int32
            )
            # Per-substage shfl scratch: ROWS_PER_WARP entries each.
            # The bitonic helpers issue all 2*ROWS shfls into these
            # buffers first, then run compare/select against them.
            # This lets the compiler pump the shfls into the bfly
            # pipeline without intervening setp/selp dependencies.
            peer_d_buf = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Float32
            )
            peer_i_buf = cute.make_rmem_tensor(
                cute.make_layout(ROWS_PER_WARP), cutlass.Int32
            )
            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                heap_d_arr[r] = cutlass.Float32(3.4e38)
                heap_i_arr[r] = cutlass.Int32(-1)

        # Top-K state. The packed strategy keeps (d, idx) packed into
        # a single Int64 so allocate that instead of the parallel
        # heap_d/heap_i; heap_max is unused (threshold prune compares
        # the packed values directly).
        if cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
            heap_packed = cute.make_rmem_tensor(
                cute.make_layout((ROWS_OWNED, K_INTERNAL)), cutlass.Int64
            )
            INF_PACKED = cutlass.Int64(0x7F800000FFFFFFFF)
            for i in cutlass.range_constexpr(ROWS_OWNED):
                for k in cutlass.range_constexpr(K_INTERNAL):
                    heap_packed[(i, k)] = INF_PACKED
        elif cutlass.const_expr(self.topk_strategy == "warpsort"):
            # warpsort owns its top-K across lanes (heap_d_arr above).
            # Allocate placeholder heap_d / heap_i so the rest of the
            # code (epilogue, etc.) can still reference them without
            # special-casing. They are never written.
            heap_d = cute.make_rmem_tensor(
                cute.make_layout((1, 1)), cutlass.Float32
            )
            heap_i = cute.make_rmem_tensor(
                cute.make_layout((1, 1)), cutlass.Int32
            )
            heap_max = cute.make_rmem_tensor(
                cute.make_layout(1), cutlass.Float32
            )
        else:
            heap_d = cute.make_rmem_tensor(
                cute.make_layout((ROWS_OWNED, K_INTERNAL)), cutlass.Float32
            )
            heap_i = cute.make_rmem_tensor(
                cute.make_layout((ROWS_OWNED, K_INTERNAL)), cutlass.Int32
            )
            heap_max = cute.make_rmem_tensor(
                cute.make_layout(ROWS_OWNED), cutlass.Float32
            )
            # ``maxtree`` also keeps the worst slot per row (the max-tree
            # writes the evict position back here so a pruned chunk reads
            # both worst_d and worst_pos from registers). Unused by the
            # other strategies (compiles away).
            heap_wpos = cute.make_rmem_tensor(
                cute.make_layout(ROWS_OWNED), cutlass.Int32
            )
            for i in cutlass.range_constexpr(ROWS_OWNED):
                for k in cutlass.range_constexpr(K_INTERNAL):
                    heap_d[(i, k)] = cutlass.Float32(3.4e38)
                    heap_i[(i, k)] = cutlass.Int32(-1)
                heap_max[i] = cutlass.Float32(3.4e38)
                heap_wpos[i] = cutlass.Int32(0)

        # SMEM dist staging tensor (smem_perthread + warpsort).
        # Layout is (BM, BN) PADDED row-major fp32 with stride
        # (BN+1, 1) so the consumer reads (one thread / lane per row,
        # sequential cols) avoid the BN%32==0 bank conflict.
        if cutlass.const_expr(
            self.topk_strategy in ("smem_perthread", "warpsort")
        ):
            sDist = storage.sDist.get_tensor(
                cute.make_layout(
                    (BM, BN), stride=(self.dist_smem_row_stride, 1)
                )
            )

        # Sort-merge needs scratch for the chunk's sorted view; the
        # bubble-insert and cooperative-insert paths do not.
        if cutlass.const_expr(self.topk_strategy == "sortmerge"):
            CHUNK_INTERNAL = self._next_pow2(N_per_thr)
            SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
            chunk_d = cute.make_rmem_tensor(
                cute.make_layout(SCRATCH_LEN), cutlass.Float32
            )
            chunk_i = cute.make_rmem_tensor(
                cute.make_layout(SCRATCH_LEN), cutlass.Int32
            )
        elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
            CHUNK_INTERNAL = self._next_pow2(N_per_thr)
            SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
            chunk_packed = cute.make_rmem_tensor(
                cute.make_layout(SCRATCH_LEN), cutlass.Int64
            )

        # --- Mainloop -----------------------------------------------------
        c_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.c_stage
        )
        c_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.c_stage
        )
        x_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.x_stage
        )
        x_pipeline.consumer_wait(x_consumer_state)

        num_k_blocks = cute.size(tCrX, mode=[2])

        for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
            c_pipeline.consumer_wait(c_consumer_read_state)

            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                k_block_coord_x = (None, None, k_block_idx, 0)
                k_block_coord_c = (None, None, k_block_idx, c_consumer_read_state.index)
                cute.gemm(
                    tiled_mma, acc,
                    tCrX[k_block_coord_x],
                    tCrC[k_block_coord_c],
                    acc,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)

            c_pipeline.consumer_release(c_consumer_release_state)
            c_consumer_read_state.advance()
            c_consumer_release_state.advance()

            if warp_idx == 0 and c_producer_state.count < num_c_tiles:
                c_pipeline.producer_acquire(c_producer_state)
                cute.copy(
                    tma_atom_c,
                    tma_cG[(None, c_producer_state.count)],
                    tma_cS[(None, c_producer_state.index)],
                    tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                )
                c_pipeline.producer_commit(c_producer_state)
                c_producer_state.advance()

            # Epilogue: build per-thread distance buffer, then run K
            # cooperative insert iterations (Triton-style chunk-best).
            #
            # Key idea (matches the Triton iterative-insert epilogue):
            # Instead of attempting an insert for every (i, j) candidate
            # (which generates BM_per_thr * BN_per_thr ≈ 64 inserts per
            # chunk per thread, each with a K_PAD-wide rescan), we run
            # *up to K_PAD* iterations per chunk where each iteration
            # picks the SINGLE best candidate per row, inserts it once,
            # and marks it used. The candidate-find is a parallel argmin
            # reduction across the WGMMA TV-layout's threads-in-row
            # group via warp-shuffle butterfly. Threshold pruning at
            # step 0 skips the entire iter loop once the heap fills.
            cta_n_offset = c_tile_idx * BN

            cs = cute.make_rmem_tensor(cute.make_layout(N_per_thr), cutlass.Float32)
            for j in cutlass.range_constexpr(N_per_thr):
                n_local = ptPcP_mn[(0, j)][1]
                n_global = n_local + cta_n_offset
                if n_global < M_total:
                    cs[j] = mCsq_m[n_global]
                else:
                    cs[j] = cutlass.Float32(3.4e38)

            dist_buf = cute.make_rmem_tensor(
                cute.make_layout((M_per_thr, N_per_thr)), cutlass.Float32
            )
            for i in cutlass.range_constexpr(M_per_thr):
                for j in cutlass.range_constexpr(N_per_thr):
                    cross = acc_mn[(i, j)]
                    # Signed score: c_sq[m] − 2·cross. No x_sq, no clamp.
                    d = cs[j] - cutlass.Float32(2.0) * cross
                    dist_buf[(i, j)] = d

            if cutlass.const_expr(self.topk_strategy == "perthread"):
                self._chunk_topk_perthread(
                    dist_buf, ptPcP_mn, heap_d, heap_i, heap_max,
                    M_per_thr, N_per_thr, cta_n_offset,
                )
            elif cutlass.const_expr(self.topk_strategy == "maxtree"):
                self._chunk_topk_maxtree(
                    dist_buf, ptPcP_mn, heap_d, heap_i, heap_max, heap_wpos,
                    M_per_thr, N_per_thr, cta_n_offset,
                )
            elif cutlass.const_expr(self.topk_strategy == "sortmerge"):
                self._chunk_topk_sortmerge(
                    dist_buf, ptPcP_mn, heap_d, heap_i, heap_max,
                    chunk_d, chunk_i,
                    M_per_thr, N_per_thr, cta_n_offset,
                )
            elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                self._chunk_topk_sortmerge_packed(
                    dist_buf, ptPcP_mn,
                    heap_packed, chunk_packed,
                    M_per_thr, N_per_thr, cta_n_offset,
                )
            elif cutlass.const_expr(self.topk_strategy == "smem_perthread"):
                # SMEM relayout: stage WGMMA-distributed dist_buf into
                # row-major sDist, full CTA sync, then each thread
                # bubble-inserts its single owned row's BN distances.
                # Eliminates cross-thread shuffles in the inner top-K
                # loop at the cost of (a) one BM*BN fp32 SMEM round-trip
                # per chunk and (b) using only BM of the WG's threads
                # for top-K (some threads idle when threads_per_cta > BM).
                self._stage_dist_to_smem(
                    dist_buf, ptPcP_mn, sDist,
                    M_per_thr, N_per_thr,
                )
                cute.arch.sync_threads()
                self._chunk_topk_smem_perthread(
                    sDist, heap_d, heap_i,
                    BM, BN, ROWS_OWNED,
                    cta_m_offset, cta_n_offset, N_total, M_total,
                    tidx,
                )
                # Make sure no thread re-uses sDist before all readers
                # are done -- next iter's _stage_dist_to_smem otherwise
                # races with this iter's reads.
                cute.arch.sync_threads()
            elif cutlass.const_expr(self.topk_strategy == "warpsort"):
                # warpsort: stage dist to SMEM, then per (sub_chunk,
                # row) load 32 cols into 32 lanes and warp-cooperatively
                # bitonic-sort + merge with running top-K. Mirrors
                # Triton's tl.sort lane vectorisation.
                self._stage_dist_to_smem(
                    dist_buf, ptPcP_mn, sDist,
                    M_per_thr, N_per_thr,
                )
                cute.arch.sync_threads()
                # warp_idx within the consumer WG group. tidx is
                # 0..threads_per_cta-1; for non-WS the consumer WGs
                # start at tidx 0 so warp_idx = tidx // 32. For WS the
                # producer WG sits at the bottom and we'd need to
                # offset, but warpsort is rejected for use_ws=True.
                warp_local = tidx // 32
                self._chunk_topk_warpsort(
                    sDist, heap_d_arr, heap_i_arr,
                    chunk_d_arr, chunk_i_arr,
                    peer_d_buf, peer_i_buf,
                    BM, BN, ROWS_PER_WARP, K_PAD,
                    cta_m_offset, cta_n_offset, N_total, M_total,
                    warp_local, self.num_consumer_warps,
                )
                cute.arch.sync_threads()
            else:
                self._chunk_topk_insert(
                    tiled_mma, dist_buf, ptPcP_mn,
                    heap_d, heap_i, heap_max,
                    M_per_thr, N_per_thr, cta_n_offset,
                )

        # End mainloop.
        # * insert path: threads-in-row share identical heaps (cooperative
        #   chunk-best inserts replicated state), no post-loop merge needed.
        # * perthread / sortmerge: each thread holds a SORTED top-K of its
        #   own column slice. Butterfly-merge across threads-in-row to
        #   replicate the global top-K so the row-leader can write.
        # * smem_perthread: 1 thread per row -> already global, no merge.
        if cutlass.const_expr(self.topk_strategy == "perthread"):
            self._warp_merge_topk_perthread(
                tiled_mma, heap_d, heap_i, M_per_thr,
            )
        elif cutlass.const_expr(self.topk_strategy == "maxtree"):
            self._warp_merge_topk_maxtree(
                tiled_mma, heap_d, heap_i, M_per_thr,
            )
        elif cutlass.const_expr(self.topk_strategy == "sortmerge"):
            self._warp_merge_topk_sortmerge(
                tiled_mma, heap_d, heap_i, M_per_thr,
            )
        elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
            self._warp_merge_topk_sortmerge_packed(
                tiled_mma, heap_packed, M_per_thr,
            )

        # Epilogue write.
        # * smem_perthread: thread t writes row t (for t < BM); state is
        #   already sorted by the bubble-insert.
        # * sortmerge_packed: row leaders unpack heap_packed via aliased
        #   recast views (high 32 bits -> fp32 distance, low 32 bits ->
        #   int32 idx) and write.
        # * other strategies: row leaders (n=0 column owners) write
        #   directly from heap_d/heap_i. Insert path needs a final
        #   selection sort; perthread/sortmerge are already sorted.
        if cutlass.const_expr(self.topk_strategy == "smem_perthread"):
            if tidx < BM:
                m_global = tidx + cta_m_offset
                if m_global < N_total:
                    for kk in cutlass.range_constexpr(K_PAD):
                        mOutI_nk[(m_global, kk)] = heap_i[(0, kk)]
        elif cutlass.const_expr(self.topk_strategy == "warpsort"):
            # warpsort: lane l of warp w holds the l-th smallest of
            # row r for each row r in warp w's assignment. Lanes 0..
            # K_PAD-1 own the actual top-K (lanes K_PAD..31 hold INF).
            # Each (warp w, lane l < K_PAD) writes outV/outI[row, l].
            lane = cute.arch.lane_idx()
            warp_local = tidx // 32
            if lane < K_PAD:
                for r in cutlass.range_constexpr(ROWS_PER_WARP):
                    my_row = warp_local + r * self.num_consumer_warps
                    if my_row < BM:
                        m_global = my_row + cta_m_offset
                        if m_global < N_total:
                            mOutI_nk[(m_global, lane)] = heap_i_arr[r]
        elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
            heap_packed_f32 = cute.recast_tensor(
                heap_packed, cutlass.Float32
            )
            heap_packed_i32 = cute.recast_tensor(
                heap_packed, cutlass.Int32
            )
            if ptPcP[0][1] == 0:
                for i in cutlass.range_constexpr(M_per_thr):
                    m_local = ptPcP_mn[(i, 0)][0]
                    m_global = m_local + cta_m_offset
                    if m_global < N_total:
                        for kk in cutlass.range_constexpr(K_PAD):
                            mOutI_nk[(m_global, kk)] = heap_packed_i32[
                                (2 * i, kk)
                            ]
        else:
            if ptPcP[0][1] == 0:
                if cutlass.const_expr(
                    self.topk_strategy in ("insert", "maxtree")
                ):
                    # insert + maxtree keep the heap UNSORTED -> one-shot
                    # selection sort restores ascending order for the write.
                    self._sort_topk_rows(heap_d, heap_i, M_per_thr, K_PAD)
                for i in cutlass.range_constexpr(M_per_thr):
                    m_local = ptPcP_mn[(i, 0)][0]
                    m_global = m_local + cta_m_offset
                    if m_global < N_total:
                        for kk in cutlass.range_constexpr(K_PAD):
                            mOutI_nk[(m_global, kk)] = heap_i[(i, kk)]
        return

    # ----------------------------------------------------------------------
    # Top-K helpers
    # ----------------------------------------------------------------------

    @cute.jit
    def _chunk_topk_insert(
        self, tiled_mma, dist_buf, ptPcP_mn,
        heap_d, heap_i, heap_max,
        M_per_thr, N_per_thr, cta_n_offset,
    ):
        """Run K_PAD iterations of cooperative chunk-best insert.

        Per iteration:
          1. Each thread: argmin over its N_per_thr columns of dist_buf
             (the chunk's per-row local best).
          2. Cross-thread reduction across threads-in-row group to find
             the global chunk-best per row -- track ``(val, db_idx)``
             together via parallel shuffles on each, with the val
             driving the comparison.
          3. Insert global-best ``(val, db_idx)`` into the per-thread
             heap if it beats heap_max. All threads in the row group
             hold identical heaps (started from identical inits, see
             the same cooperative inserts), so the replicated update
             stays coherent.
          4. Threads whose local_min equals gval mark their local
             argmin slot as ``+inf`` so the next iter picks the
             next-best. On rare exact ties, this may mark more than one
             slot -- harmless, just slightly less work next iter.

        Once the heap fills with the true global top-K_PAD, subsequent
        iters fail the step-3 threshold and become almost free
        (no heap update, no mark).

        We deliberately do NOT early-exit the outer ``step`` loop when
        gval >= heap_max, even though it would skip work on pruned
        chunks: doing so would diverge ``shuffle_sync_bfly`` calls
        across row groups of the warp (different rows have independent
        active state, and the shuffles use mask=-1 so all 32 lanes must
        participate every call). The runtime ``if gval < hmax`` already
        gates the K_PAD-wide rescan + replace + mark work which is the
        dominant per-iter cost.
        """
        K_PAD = self.k_pad
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)

        for step in cutlass.range_constexpr(K_PAD):
            for i in cutlass.range_constexpr(M_per_thr):
                # 1) Local argmin over thread's N_per_thr columns.
                local_min = dist_buf[(i, 0)]
                local_argj = cutlass.Int32(0)
                for j in cutlass.range_constexpr(1, N_per_thr):
                    d = dist_buf[(i, j)]
                    if d < local_min:
                        local_min = d
                        local_argj = cutlass.Int32(j)

                n_local_glb = ptPcP_mn[(i, local_argj)][1]
                local_min_db_idx = cutlass.Int32(n_local_glb + cta_n_offset)

                # 2) Cross-thread min reduction across threads-in-row
                # group with paired db_idx. Always run unconditionally
                # so the warp-wide shuffle stays convergent across all
                # rows within the warp.
                gval = local_min
                gidx = local_min_db_idx
                for r in cutlass.range_constexpr(red_rank):
                    tig = red_target.shape[r]
                    offset = tig // 2
                    while offset > 0:
                        peer_val = cute.arch.shuffle_sync_bfly(
                            gval, offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        peer_idx = cute.arch.shuffle_sync_bfly(
                            gidx, offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        if peer_val < gval:
                            gval = peer_val
                            gidx = peer_idx
                        offset = offset // 2

                # 3) Heap insert (replicated across threads in row group;
                # threshold prune skips the K_PAD-wide rescan once the
                # heap is full and chunk-best > worst-in-heap).
                hmax = heap_max[i]
                if gval < hmax:
                    argmax_k = cutlass.Int32(0)
                    max_d = heap_d[(i, 0)]
                    for kk in cutlass.range_constexpr(1, K_PAD):
                        hv = heap_d[(i, kk)]
                        if hv > max_d:
                            max_d = hv
                            argmax_k = cutlass.Int32(kk)
                    for kk in cutlass.range_constexpr(K_PAD):
                        is_target = (cutlass.Int32(kk) == argmax_k)
                        if is_target:
                            heap_d[(i, kk)] = gval
                            heap_i[(i, kk)] = gidx
                    new_max = heap_d[(i, 0)]
                    for kk in cutlass.range_constexpr(1, K_PAD):
                        hv = heap_d[(i, kk)]
                        if hv > new_max:
                            new_max = hv
                    heap_max[i] = new_max

                    # 4) Mark used: any thread whose local_min equals
                    # gval clears its local-argmin slot.
                    if local_min == gval:
                        for j in cutlass.range_constexpr(N_per_thr):
                            is_used = (cutlass.Int32(j) == local_argj)
                            if is_used:
                                dist_buf[(i, j)] = cutlass.Float32(3.4e38)

    @cute.jit
    def _warp_merge_topk(self, tiled_mma, heap_d, heap_i, heap_max, M_per_thr):
        """Butterfly merge per-thread top-K heaps across threads-in-row.

        Within a row, the WGMMA TV layout assigns ``threads_in_row``
        threads (often 4) to share each query row -- they each hold a
        subset of the database columns and so a partial top-K. The
        merge is a tournament: per round, exchange heaps with the peer
        at offset 2^r, then try to insert all peer entries into ours.
        After ``log2(threads_in_row)`` rounds, every thread in the
        group has the same global top-K.

        Implementation note: we **snapshot** the peer's heap into a
        register buffer in a tight all-shuffle loop *before* doing any
        evict-inserts. If shuffling and inserting were interleaved, my
        heap would mutate mid-loop and the shuffle source for slot k+1
        would no longer match the peer's slot k+1. This used to produce
        the duplicate-index bug in row 0/1/2 of the correctness test.
        """
        K_PAD = self.k_pad
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for r in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[r]
            offset = tig // 2
            while offset > 0:
                for i in cutlass.range_constexpr(M_per_thr):
                    # 1) Snapshot peer heap into local registers.
                    peer_d_buf = cute.make_rmem_tensor(
                        cute.make_layout(K_PAD), cutlass.Float32
                    )
                    peer_i_buf = cute.make_rmem_tensor(
                        cute.make_layout(K_PAD), cutlass.Int32
                    )
                    for kk in cutlass.range_constexpr(K_PAD):
                        peer_d_buf[kk] = cute.arch.shuffle_sync_bfly(
                            heap_d[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        peer_i_buf[kk] = cute.arch.shuffle_sync_bfly(
                            heap_i[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )

                    # 2) Iteratively insert each snapshotted peer entry
                    # into our heap (now safe: snapshot is independent
                    # of heap mutations).
                    hmax = heap_max[i]
                    for kk in cutlass.range_constexpr(K_PAD):
                        peer_d = peer_d_buf[kk]
                        peer_i = peer_i_buf[kk]
                        if peer_d < hmax:
                            argmax_k = cutlass.Int32(0)
                            max_d = heap_d[(i, 0)]
                            for kk2 in cutlass.range_constexpr(1, K_PAD):
                                hv = heap_d[(i, kk2)]
                                if hv > max_d:
                                    max_d = hv
                                    argmax_k = cutlass.Int32(kk2)
                            for kk2 in cutlass.range_constexpr(K_PAD):
                                is_target = (cutlass.Int32(kk2) == argmax_k)
                                if is_target:
                                    heap_d[(i, kk2)] = peer_d
                                    heap_i[(i, kk2)] = peer_i
                            new_max = heap_d[(i, 0)]
                            for kk2 in cutlass.range_constexpr(1, K_PAD):
                                hv = heap_d[(i, kk2)]
                                if hv > new_max:
                                    new_max = hv
                            hmax = new_max
                    heap_max[i] = hmax
                offset = offset // 2

    @cute.jit
    def _sort_topk_rows(self, heap_d, heap_i, M_per_thr, K_PAD: cutlass.Constexpr):
        """In-place selection sort over each owned row's K_PAD entries."""
        for i in cutlass.range_constexpr(M_per_thr):
            for s in cutlass.range_constexpr(K_PAD - 1):
                argmin_k = cutlass.Int32(s)
                min_d = heap_d[(i, s)]
                min_i = heap_i[(i, s)]
                for kk in cutlass.range_constexpr(s + 1, K_PAD):
                    hv = heap_d[(i, kk)]
                    if hv < min_d:
                        min_d = hv
                        min_i = heap_i[(i, kk)]
                        argmin_k = cutlass.Int32(kk)
                # Swap by writing min into slot s and writing the
                # original slot s value into the argmin slot.
                old_s_d = heap_d[(i, s)]
                old_s_i = heap_i[(i, s)]
                heap_d[(i, s)] = min_d
                heap_i[(i, s)] = min_i
                # Write old_s back into argmin_k slot. We loop because
                # argmin_k is dynamic.
                for kk in cutlass.range_constexpr(s + 1, K_PAD):
                    is_t = (cutlass.Int32(kk) == argmin_k)
                    if is_t:
                        heap_d[(i, kk)] = old_s_d
                        heap_i[(i, kk)] = old_s_i

    # ----------------------------------------------------------------------
    # Per-thread top-K (the K_PAD>=3 fast path; the "sortmerge" strategy)
    # ----------------------------------------------------------------------
    #
    # The cooperative insert path above pays O(K_PAD * (N_per_thr + 3*K_PAD))
    # per row per chunk because:
    #   * the K_PAD outer iterations are constexpr-unrolled and ALWAYS
    #     execute (the runtime ``if gval < hmax`` only gates the heap touch);
    #   * each iter does K_PAD shuffles across threads-in-row.
    # The K_PAD shuffles per chunk per row is the dominant constant factor
    # and is what tanks K=10 vs Triton.
    #
    # The fast path here breaks the cooperative pattern entirely:
    #
    #   1. Each thread maintains an UNSORTED per-thread top-K over only
    #      ITS column slice of the database (N_per_thr columns per row).
    #      No cross-thread shuffles in the mainloop.
    #   2. Per chunk per row: iterate the chunk's N_per_thr elements
    #      sequentially. For each, threshold-test against per-thread
    #      heap_max[i]; on insert do a K_PAD-wide find-worst + replace.
    #   3. After the mainloop, butterfly-merge per-thread top-Ks across
    #      threads-in-row using a bitonic merge so every thread holds the
    #      global top-K, then row-leader sorts and writes.
    #
    # Why this beats the cooperative insert at K_PAD>=3:
    #   * No cross-thread shuffle in the inner loop -- the shuffle is the
    #     bottleneck (~5 cycles each, K_PAD * log T per row per chunk in
    #     the cooperative path).
    #   * Threshold prune (per-element ``if c_d < heap_max[i]``) skips the
    #     K_PAD-wide find-worst at steady state; we still pay 1 compare
    #     per element, but that's free vs the shuffle path.
    #   * Post-loop butterfly merge is a one-time cost amortised across
    #     the whole mainloop -- negligible at M >= 10K.
    #
    # Why we ALSO tried (and abandoned) bitonic chunk-sort + bitonic-merge:
    # a constexpr-unrolled bitonic sort of N_per_thr=16 in registers is ~80
    # compare-swaps per row per chunk, which alone exceeds the threshold-
    # pruned per-element insert. Triton's tl.sort is hardware-accelerated;
    # CuteDSL has no equivalent. The bitonic helpers below are kept ONLY
    # for the post-loop butterfly merge, where the input is a sorted
    # K_PAD-wide top-K (smaller, one-shot, hidden in the epilogue).
    #
    # Tie-breaking on equal distances: keep the smaller idx, matching
    # Triton's packed-uint64 sort -- gives bit-exact parity on bf16-tied
    # data after the post-loop sort.

    @staticmethod
    def _next_pow2(n: int) -> int:
        p = 1
        while p < n:
            p *= 2
        return p

    @staticmethod
    def _bitonic_sort_pairs(n: int):
        """Compare-swap pairs to sort length-n (pow-of-2) array ascending.

        Returns a list of (i, j, asc) tuples with i < j. ``asc=True`` means
        the pair should end up with the smaller element at i; ``asc=False``
        means the larger element ends up at i (descending sub-block).
        """
        pairs = []
        k = 2
        while k <= n:
            j = k // 2
            while j > 0:
                for i in range(n):
                    ij = i ^ j
                    if ij > i:
                        asc = ((i & k) == 0)
                        pairs.append((i, ij, asc))
                j //= 2
            k *= 2
        return pairs

    @staticmethod
    def _bitonic_merge_pairs(n: int):
        """Compare-swap pairs to ascending-sort an n-element bitonic array.

        Used after the min-pair step, where the input is already bitonic
        (asc-then-desc), so we only need the merge half of the network.
        Half the cost of a full bitonic sort.
        """
        pairs = []
        j = n // 2
        while j > 0:
            for i in range(n):
                ij = i ^ j
                if ij > i:
                    pairs.append((i, ij))
            j //= 2
        return pairs

    @cute.jit
    def _cmp_swap_asc(self, d_arr, i_arr, a: cutlass.Constexpr, b: cutlass.Constexpr):
        """In-place: ensure d_arr[a] <= d_arr[b].

        Branch-free via select-style writes -- compiles to predicated
        selp PTX, no warp divergence. We deliberately drop the
        ``ia > ib`` tie-break here: the bitonic sort over (d, i) is
        only required to be correct on d, and any consistent placement
        of equal-d entries within a per-thread chunk is fine because
        the cross-thread butterfly merge is also a value-only sort.

        Inline-PTX experiment (kept as ``_cmp_swap_asc_ptx`` for
        reference): hand-rolled ``min.f32 + max.f32 + setp + 2x
        selp`` packed in one ``llvm.inline_asm`` block. Empirically
        2-3% faster than the MLIR baseline at ``K_INTERNAL <= 16``
        (no spills, FMNMX appears in SASS) but 9% SLOWER at
        ``K_INTERNAL = 32`` (the struct-return adds a few bytes of
        spill pressure in an already-spilling regime). Net impact
        on the autotuner is zero -- at K_PAD=14..16 the per-thread
        bubble (``perthread`` strategy) outperforms ``sortmerge`` by
        more than 3%, and at K_PAD>=20 the inline PTX is gated off
        anyway. Code reverted to the MLIR lowering for simplicity;
        ``_cmp_swap_asc_ptx`` retained for future experiments.
        """
        da = d_arr[a]
        ia = i_arr[a]
        db = d_arr[b]
        ib = i_arr[b]
        swap = da > db
        d_arr[a] = db if swap else da
        i_arr[a] = ib if swap else ia
        d_arr[b] = da if swap else db
        i_arr[b] = ia if swap else ib

    @cute.jit
    def _cmp_swap_desc(self, d_arr, i_arr, a: cutlass.Constexpr, b: cutlass.Constexpr):
        """In-place: ensure d_arr[a] >= d_arr[b]. Tie-break dropped (see
        ``_cmp_swap_asc`` for the rationale)."""
        da = d_arr[a]
        ia = i_arr[a]
        db = d_arr[b]
        ib = i_arr[b]
        swap = da < db
        d_arr[a] = db if swap else da
        i_arr[a] = ib if swap else ia
        d_arr[b] = da if swap else db
        i_arr[b] = ia if swap else ib

    @cute.jit
    def _bitonic_sort_local(self, d_arr, i_arr, length: cutlass.Constexpr):
        """Sort length-N (pow-of-2) (d, i) arrays ascending in place.

        Network is generated at trace time; all loops are Python so the
        compare-swap calls are constexpr-unrolled.
        """
        for a, b, asc in self._bitonic_sort_pairs(length):
            if asc:
                self._cmp_swap_asc(d_arr, i_arr, a, b)
            else:
                self._cmp_swap_desc(d_arr, i_arr, a, b)

    @cute.jit
    def _bitonic_merge_finish(self, d_arr, i_arr, length: cutlass.Constexpr):
        """Ascending-sort an n-element bitonic (asc-then-desc) array."""
        for a, b in self._bitonic_merge_pairs(length):
            self._cmp_swap_asc(d_arr, i_arr, a, b)

    @cute.jit
    def _cmp_swap_2d_asc(
        self, d_arr, i_arr,
        row: cutlass.Constexpr,
        a: cutlass.Constexpr, b: cutlass.Constexpr,
    ):
        """Branch-free ascending compare-swap on row ``row`` of a 2D
        tensor. Tie-break dropped (see ``_cmp_swap_asc`` rationale)."""
        da = d_arr[(row, a)]
        ia = i_arr[(row, a)]
        db = d_arr[(row, b)]
        ib = i_arr[(row, b)]
        swap = da > db
        d_arr[(row, a)] = db if swap else da
        i_arr[(row, a)] = ib if swap else ia
        d_arr[(row, b)] = da if swap else db
        i_arr[(row, b)] = ia if swap else ib

    @cute.jit
    def _bitonic_merge_finish_2d(
        self, d_arr, i_arr,
        row: cutlass.Constexpr,
        length: cutlass.Constexpr,
    ):
        """Asc-sort an n-element bitonic row of a 2D (d, i) tensor."""
        for a, b in self._bitonic_merge_pairs(length):
            self._cmp_swap_2d_asc(d_arr, i_arr, row, a, b)

    # ----------------------------------------------------------------------
    # Packed Int64 bitonic helpers (sortmerge_packed strategy)
    # ----------------------------------------------------------------------
    #
    # Triton's stage-1 KNN kernel sorts (distance, index) pairs as packed
    # uint64 (distance bit-pattern in high 32, index in low 32) so a
    # SINGLE unsigned compare gives both the value-ascending order and
    # the smaller-index tie-break for free. We replicate that here.
    #
    # Packing scheme (matches Triton flash_knn stage-1):
    #   high 32 bits = bit-pattern of fp32 distance (>=0 -> sign bit 0)
    #   low  32 bits = int32 db_idx reinterpreted as uint32 (-1 -> 0xFF..)
    # Empty slot (d=+INF, i=-1) sorts as the LARGEST possible value, so
    # any real candidate displaces it.
    #
    # Bit-cast is done via ``cute.recast_tensor``: an Int64 RMEM tensor
    # has aliased Float32 and Int32 views (size doubled) reachable by
    # recast_tensor; reads/writes through any view are visible through
    # the others (verified: probe_recast.py round-trips fp32 3.14 + i32
    # -7 to packed 0x4048f5c3fffffff9).
    #
    # Compute saving per compare-swap vs the (d, i) pair version:
    #   - 4 reads, 3 cmp + 1 and + 1 or, 4 selp, 4 writes  (current ~9 ops)
    #   - 2 reads, 1 cmp,             2 selp, 2 writes     (packed ~3 ops)
    # Plus simpler dependency chain (no bool `or` to chain compares),
    # so the bitonic network stalls less. Cross-lane shfl_xor on 64-bit
    # values is split internally into 2x 32-bit shfls (same as a
    # separate d-shfl + i-shfl pair), so net shfl count is unchanged.

    @cute.jit
    def _cmp_swap_asc_packed(
        self, p_arr,
        a: cutlass.Constexpr, b: cutlass.Constexpr,
    ):
        """Asc compare-swap on a 1D Int64 packed array (smallest at a).

        Uses inline-PTX ``_cmp_swap_asc_packed_ptx`` -- forces ptxas
        to emit ``ISETP.GT.U32 + ISETP.GT.U32.EX + 4 SEL.b32`` (6
        SASS ops, the theoretical minimum for u64 cmp_swap on
        Hopper) instead of the MLIR-lowered ~10-12 ops it produces
        from ``arith.cmpi UGT + arith.select`` on i64 (which has
        extra register copies and predicate fan-out).
        """
        p_min, p_max = _cmp_swap_asc_packed_ptx(p_arr[a], p_arr[b])
        p_arr[a] = p_min
        p_arr[b] = p_max

    @cute.jit
    def _cmp_swap_desc_packed(
        self, p_arr,
        a: cutlass.Constexpr, b: cutlass.Constexpr,
    ):
        """Desc compare-swap on a 1D Int64 packed array (largest at a).

        Inline-PTX path: same as ``_cmp_swap_asc_packed`` but writes
        max to ``a`` and min to ``b``.
        """
        p_min, p_max = _cmp_swap_asc_packed_ptx(p_arr[a], p_arr[b])
        p_arr[a] = p_max
        p_arr[b] = p_min

    @cute.jit
    def _cmp_swap_2d_asc_packed(
        self, p_arr,
        row: cutlass.Constexpr,
        a: cutlass.Constexpr, b: cutlass.Constexpr,
    ):
        """Asc compare-swap on row ``row`` of a 2D Int64 packed array."""
        p_min, p_max = _cmp_swap_asc_packed_ptx(p_arr[(row, a)], p_arr[(row, b)])
        p_arr[(row, a)] = p_min
        p_arr[(row, b)] = p_max

    @cute.jit
    def _bitonic_sort_packed(self, p_arr, length: cutlass.Constexpr):
        """Sort length-N (pow-of-2) packed Int64 array ascending in place."""
        for a, b, asc in self._bitonic_sort_pairs(length):
            if asc:
                self._cmp_swap_asc_packed(p_arr, a, b)
            else:
                self._cmp_swap_desc_packed(p_arr, a, b)

    @cute.jit
    def _bitonic_merge_finish_2d_packed(
        self, p_arr,
        row: cutlass.Constexpr,
        length: cutlass.Constexpr,
    ):
        """Asc-sort an n-element bitonic row of a 2D Int64 packed array."""
        for a, b in self._bitonic_merge_pairs(length):
            self._cmp_swap_2d_asc_packed(p_arr, row, a, b)

    @cute.jit
    def _chunk_topk_sortmerge(
        self, dist_buf, ptPcP_mn, topk_d, topk_i, heap_max,
        chunk_d, chunk_i,
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
        cta_n_offset,
    ):
        """Sort-merge top-K update for one database chunk (K_PAD>=16 path).

        Per row:
          1. Optional early scan: O(N_per_thr) chunk_min vs topk worst.
             Active for K_INTERNAL <= 8 -- at small K threshold prune
             rate is >>90% so the scan saves the entire bitonic sort.
          2. Build chunk into chunk_d/chunk_i, pad with INF, bitonic-sort
             ascending.
          3. Post-sort prune on chunk_d[0]; if it doesn't beat the
             current top-K worst, skip the merge.
          4. Min-pair (in place on topk[i, :] with reversed chunk),
             then bitonic-merge-finish to keep topk sorted asc.

        Despite emitting a software bitonic sort (CuteDSL has no
        equivalent of Triton's tl.sort), at large K_INTERNAL (>=16)
        this path wins over per-element bubble insert because:
          * the bubble's K-wide constexpr unroll x N_per_thr inner
            loop blows up to ~K*N_per_thr predicated PTX ops per
            chunk (~3000 at K=32 N=16) -- spills registers and
            saturates the issue width;
          * bitonic chunk sort is O(N log^2 N) ~80 swaps and the
            merge is O(K log K) ~80 swaps -- 160 swaps total,
            constant in K, vs O(K * insert_rate * N_per_thr)
            for the bubble at high K.

        State (caller-allocated, see ``kernel``):
          topk_d[i, k] : fp32 distances, SORTED asc, size K_INTERNAL
          topk_i[i, k] : int32 indices, paired with topk_d
          heap_max[i]  : redundant cache of topk_d[i, K_INTERNAL-1]

        K_INTERNAL note: when ``K_PAD`` is non-pow2 (e.g. 24) the
        heap is rounded UP to ``K_PAD_POW2`` (e.g. 32) so the
        bitonic merge network has a pow2 length. We tried (and
        reverted) two ways to avoid this rounding:

        * ``K_INTERNAL = K_PAD`` + insertion-merge (sequential
          bubble inserts). Ops: ~K_PAD^2 with K_PAD-deep dep chain
          per insert, K_PAD inserts back-to-back -> O(K^2) critical
          path. At K=24 this measured 3-5x SLOWER than the K_INT=32
          bitonic (~25ms vs ~7.2ms at the standard benchmark shape)
          because the bitonic merge has only O(log^2 K) depth.

        * ``K_INTERNAL = K_PAD`` + virtual K_INT bitonic over a
          K_PAD-wide heap with INF padding for slots [K_PAD, K_INT).
          Reads return INF (constexpr), writes are no-ops. PROBLEM:
          the bitonic-merge-finish input ceases to be a true bitonic
          sequence because the "INF padding" must hold the chunk's
          smaller half during min-pair, which isn't simply INF. A
          correct virtual implementation needs full K_INT register
          backing for those slots -> no register saving.

        Conclusion: the K=24/32 register spill (144 bytes) is a
        structural limit of the per-thread bitonic approach. Real
        gains require either Direction 2 (decouple WGMMA tile from
        topk-chunk size, lane-cooperative bitonic via
        shuffle_sync_bfly) or Direction 3 (WS3 with topk in its own
        warpgroup).
        """
        K_INTERNAL = self.k_pad_pow2
        CHUNK_INTERNAL = self._next_pow2(N_per_thr)
        SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
        INF = cutlass.Float32(3.4e38)

        for i in cutlass.range_constexpr(M_per_thr):
            do_sortmerge = True
            # Pre-sort min-reduce gate: would let us skip the bitonic
            # sort entirely when the chunk has nothing below the
            # heap's worst, but the chunk_min scan is itself
            # ~95 ops/row/chunk and the post-sort prune below already
            # skips the merge phase on the same condition. Empirically
            # the scan only pays back at K_INTERNAL <= 8 (where the
            # sort is cheap and the prune rate >>90%); at K_INTERNAL=
            # 16/32 it net-LOSES even with optimistic prune-rate
            # assumptions because the K=32 chunk frequently has
            # SOMETHING that beats the heap, defeating the gate.
            if cutlass.const_expr(K_INTERNAL <= 8):
                chunk_min = dist_buf[(i, 0)]
                for j in cutlass.range_constexpr(1, N_per_thr):
                    cd_j = dist_buf[(i, j)]
                    chunk_min = cd_j if cd_j < chunk_min else chunk_min
                do_sortmerge = chunk_min < topk_d[(i, K_INTERNAL - 1)]
            if do_sortmerge:
                for j in cutlass.range_constexpr(N_per_thr):
                    chunk_d[j] = dist_buf[(i, j)]
                    n_local = ptPcP_mn[(i, j)][1]
                    chunk_i[j] = cutlass.Int32(n_local + cta_n_offset)
                for j in cutlass.range_constexpr(N_per_thr, SCRATCH_LEN):
                    chunk_d[j] = INF
                    chunk_i[j] = cutlass.Int32(-1)
                self._bitonic_sort_local(chunk_d, chunk_i, SCRATCH_LEN)
                if chunk_d[0] < topk_d[(i, K_INTERNAL - 1)]:
                    for k in cutlass.range_constexpr(K_INTERNAL):
                        partner = K_INTERNAL - 1 - k
                        if cutlass.const_expr(partner < SCRATCH_LEN):
                            cd = chunk_d[partner]
                            ci = chunk_i[partner]
                        else:
                            cd = INF
                            ci = cutlass.Int32(-1)
                        td = topk_d[(i, k)]
                        ti = topk_i[(i, k)]
                        # Strict ``>`` is fine here: the running heap
                        # already holds K distinct (db_idx, d) entries
                        # so equal-d candidates from the new chunk can
                        # only happen across truly different rows in
                        # the database; either ordering preserves the
                        # top-K set. Same rationale as the tie-break
                        # drop in ``_cmp_swap_asc/_cmp_swap_2d_asc``.
                        take_chunk = td > cd
                        topk_d[(i, k)] = cd if take_chunk else td
                        topk_i[(i, k)] = ci if take_chunk else ti
                    self._bitonic_merge_finish_2d(
                        topk_d, topk_i, i, K_INTERNAL,
                    )
                    heap_max[i] = topk_d[(i, K_INTERNAL - 1)]

    @cute.jit
    def _warp_merge_topk_sortmerge(
        self, tiled_mma, topk_d, topk_i,
        M_per_thr: cutlass.Constexpr,
    ):
        """Butterfly merge per-thread sorted top-Ks across threads-in-row.

        Companion to ``_chunk_topk_sortmerge``: shuffle peer's K_INTERNAL
        entries via shfl.bfly (snapshot first, mutate after), then
        min-pair + bitonic-merge into our own sorted top-K in place.
        """
        K_INTERNAL = self.k_pad_pow2
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for r in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[r]
            offset = tig // 2
            while offset > 0:
                peer_d_buf = cute.make_rmem_tensor(
                    cute.make_layout(K_INTERNAL), cutlass.Float32
                )
                peer_i_buf = cute.make_rmem_tensor(
                    cute.make_layout(K_INTERNAL), cutlass.Int32
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for kk in cutlass.range_constexpr(K_INTERNAL):
                        peer_d_buf[kk] = cute.arch.shuffle_sync_bfly(
                            topk_d[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        peer_i_buf[kk] = cute.arch.shuffle_sync_bfly(
                            topk_i[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                    for k in cutlass.range_constexpr(K_INTERNAL):
                        partner = K_INTERNAL - 1 - k
                        cd = peer_d_buf[partner]
                        ci = peer_i_buf[partner]
                        td = topk_d[(i, k)]
                        ti = topk_i[(i, k)]
                        # Strict ``>`` -- see ``_chunk_topk_sortmerge``
                        # for the rationale.
                        take_peer = td > cd
                        topk_d[(i, k)] = cd if take_peer else td
                        topk_i[(i, k)] = ci if take_peer else ti
                    self._bitonic_merge_finish_2d(
                        topk_d, topk_i, i, K_INTERNAL,
                    )
                offset = offset // 2

    @cute.jit
    def _chunk_topk_sortmerge_packed(
        self, dist_buf, ptPcP_mn,
        topk_packed, chunk_packed,
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
        cta_n_offset,
    ):
        """Sort-merge top-K with packed Int64 (d, idx) values.

        Same network as ``_chunk_topk_sortmerge`` (per-thread bitonic
        sort of the chunk + min-pair + bitonic-merge into the running
        sorted heap), but the (d_fp32, db_idx_int32) pair is packed
        into a SINGLE Int64 (high 32 = fp32 bit-pattern, low 32 = idx)
        so each compare-swap is one 64-bit cmp + 2 selp instead of
        three 32-bit cmps + 4 selp. Cross-lane shfl.bfly on a 64-bit
        value is split into 2x 32-bit shfls inside CuTeDSL, so total
        shfl traffic is unchanged vs the unpacked version.

        Empty slot = packed (INF, -1) = 0x7F800000FFFFFFFF, sorts as
        the largest possible value so any real candidate displaces it.
        Threshold prune compares the packed values directly: a chunk's
        smallest packed entry (sorted index 0) vs the heap's largest
        (sorted index K-1). If chunk[0] >= heap[K-1] there is nothing
        to merge.
        """
        K_INTERNAL = self.k_pad_pow2
        CHUNK_INTERNAL = self._next_pow2(N_per_thr)
        SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
        # Same Int64 bit-pattern as INF/-1 packed; used as the sentinel
        # when the chunk is shorter than SCRATCH_LEN.
        INF_PACKED = cutlass.Int64(0x7F800000FFFFFFFF)

        for i in cutlass.range_constexpr(M_per_thr):
            for j in cutlass.range_constexpr(N_per_thr):
                # Direct packed-Int64 construction via Uint32 -> Uint64
                # zero-extend (avoids the AND-with-mask the signed
                # Int32 -> Int64 cast would otherwise need, and avoids
                # the recast_tensor half-store idiom which the compiler
                # turns into 64-bit RMW pairs).
                d_bits = _bitcast_f32_to_i32(dist_buf[(i, j)])
                n_local = ptPcP_mn[(i, j)][1]
                idx = cutlass.Int32(n_local + cta_n_offset)
                d_u64 = cutlass.Uint64(_bitcast_i32_to_u32(d_bits))
                idx_u64 = cutlass.Uint64(_bitcast_i32_to_u32(idx))
                chunk_packed[j] = cutlass.Int64(
                    (d_u64 << cutlass.Uint64(32)) | idx_u64
                )
            for j in cutlass.range_constexpr(N_per_thr, SCRATCH_LEN):
                chunk_packed[j] = INF_PACKED
            self._bitonic_sort_packed(chunk_packed, SCRATCH_LEN)
            if chunk_packed[0] < topk_packed[(i, K_INTERNAL - 1)]:
                for k in cutlass.range_constexpr(K_INTERNAL):
                    partner = K_INTERNAL - 1 - k
                    if cutlass.const_expr(partner < SCRATCH_LEN):
                        cd = chunk_packed[partner]
                    else:
                        cd = INF_PACKED
                    td = topk_packed[(i, k)]
                    take_chunk = td > cd
                    topk_packed[(i, k)] = cd if take_chunk else td
                self._bitonic_merge_finish_2d_packed(
                    topk_packed, i, K_INTERNAL,
                )

    @cute.jit
    def _warp_merge_topk_sortmerge_packed(
        self, tiled_mma, topk_packed,
        M_per_thr: cutlass.Constexpr,
    ):
        """Butterfly merge per-thread sorted Int64-packed top-Ks.

        Companion to ``_chunk_topk_sortmerge_packed``: shfl.bfly the
        peer's K_INTERNAL packed entries (snapshot first, then mutate),
        then min-pair + bitonic-merge into our own sorted top-K in
        place.
        """
        K_INTERNAL = self.k_pad_pow2
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for r in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[r]
            offset = tig // 2
            while offset > 0:
                peer_buf = cute.make_rmem_tensor(
                    cute.make_layout(K_INTERNAL), cutlass.Int64
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for kk in cutlass.range_constexpr(K_INTERNAL):
                        peer_buf[kk] = cute.arch.shuffle_sync_bfly(
                            topk_packed[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                    for k in cutlass.range_constexpr(K_INTERNAL):
                        partner = K_INTERNAL - 1 - k
                        cd = peer_buf[partner]
                        td = topk_packed[(i, k)]
                        take_peer = td > cd
                        topk_packed[(i, k)] = cd if take_peer else td
                    self._bitonic_merge_finish_2d_packed(
                        topk_packed, i, K_INTERNAL,
                    )
                offset = offset // 2

    @cute.jit
    def _stage_dist_to_smem(
        self, dist_buf, ptPcP_mn, sDist,
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
    ):
        """Store the WGMMA-distributed dist_buf into row-major SMEM.

        Each thread writes its (M_per_thr, N_per_thr) acc slots to the
        corresponding (m_local, n_local) positions in sDist[BM, BN].
        sDist is pre-allocated row-major via ``make_layout((BM, BN))``.
        Caller is responsible for the warpgroup-wide sync after this
        call before consumers can safely read sDist.
        """
        for i in cutlass.range_constexpr(M_per_thr):
            for j in cutlass.range_constexpr(N_per_thr):
                m_local = ptPcP_mn[(i, j)][0]
                n_local = ptPcP_mn[(i, j)][1]
                sDist[(m_local, n_local)] = dist_buf[(i, j)]

    # ----------------------------------------------------------------------
    # Warp-cooperative bitonic helpers (warpsort strategy)
    # ----------------------------------------------------------------------
    #
    # Mirrors Triton's tl.sort: sorts a 32-element (d, i) sequence
    # distributed 1-element-per-lane across a warp's 32 lanes via
    # shfl_xor + select. Each compare-swap substage is ~3 cycles
    # (1 shfl + 1 setp + 1 selp) regardless of how many elements the
    # row "holds", because all 32 lanes execute the substage in
    # parallel. Bitonic 32 = 15 substages = ~45 cycles per sort.
    #
    # This is the lane-vectorisation that the per-thread sortmerge path
    # gives up by holding N_per_thr=32 elements per thread and doing a
    # serial intra-thread bitonic (240 sequential compare-swaps × ~9
    # ops each = ~2200 cycles per row per chunk, the actual bottleneck
    # at K_PAD>=14 where Triton wins).
    #
    # Algorithm (standard bitonic, see Knuth TAOCP Vol 3 §5.3.4):
    #   for k in 0..log2(n)-1:        # stage = sub-bitonic size 2^(k+1)
    #     for s in k..0:              # substage = stride 2^s
    #       partner_lane = lane XOR (1<<s)
    #       ascend_pair  = ((lane >> (k+1)) & 1) == 0
    #       i_am_lower   = (lane & (1<<s)) == 0
    #       keep_smaller = ascend_pair == i_am_lower
    #       take_my      = (val < peer) == keep_smaller
    #       val          = val if take_my else peer
    # Validated against torch.sort on random inputs (probe_warpsort.py).

    @cute.jit
    def _warp_bitonic_sort_multi_asc(
        self, d_arr, i_arr,
        peer_d_buf, peer_i_buf,
        ROWS: cutlass.Constexpr,
    ):
        """Multi-row warp-cooperative bitonic sort (ROWS in parallel).

        Each lane holds ROWS independent (d, i) pairs (length-ROWS
        register arrays). Bitonic-32 network with the inner per-row
        work as the *outer* loop within each substage so the ROWS
        shfl_xor calls issue back-to-back. To maximise the number of
        shfls in flight, each substage issues ALL ROWS shfls first
        (writing to scratch ``peer_*_buf``) and only then runs the
        compare/select phase against the cached peer values. This
        decouples the shfl issue stream from the setp/selp result
        latency, so the compiler can schedule 2*ROWS shfls into the
        24-cycle bfly latency window.

        In place: lane l holds the l-th smallest (d, i) of each row
        after this call.
        """
        lane = cute.arch.lane_idx()
        for k in cutlass.range_constexpr(5):
            for s_inv in cutlass.range_constexpr(k + 1):
                s = k - s_inv
                offset = 1 << s
                ascend_pair = ((lane >> (k + 1)) & 1) == 0
                i_am_lower = (lane & offset) == 0
                keep_smaller = ascend_pair == i_am_lower
                # Issue ALL 2*ROWS shfls back-to-back (no compare in
                # between) so they all enter the bfly pipeline at
                # full issue rate.
                for r in cutlass.range_constexpr(ROWS):
                    peer_d_buf[r] = cute.arch.shuffle_sync_bfly(
                        d_arr[r], offset=offset, mask=-1, mask_and_clamp=31,
                    )
                for r in cutlass.range_constexpr(ROWS):
                    peer_i_buf[r] = cute.arch.shuffle_sync_bfly(
                        i_arr[r], offset=offset, mask=-1, mask_and_clamp=31,
                    )
                for r in cutlass.range_constexpr(ROWS):
                    take_my = (d_arr[r] < peer_d_buf[r]) == keep_smaller
                    d_arr[r] = d_arr[r] if take_my else peer_d_buf[r]
                    i_arr[r] = i_arr[r] if take_my else peer_i_buf[r]

    @cute.jit
    def _warp_bitonic_finish_multi_asc(
        self, d_arr, i_arr,
        peer_d_buf, peer_i_buf,
        ROWS: cutlass.Constexpr,
    ):
        """Multi-row bitonic-finish: input bitonic, output sorted ASC.

        5 substages (s = 4..0), each with ROWS shfl_xor calls issued
        back-to-back into ``peer_*_buf`` and then a compare/select
        phase against the cached peer values.
        """
        lane = cute.arch.lane_idx()
        for s_inv in cutlass.range_constexpr(5):
            s = 4 - s_inv
            offset = 1 << s
            i_am_lower = (lane & offset) == 0
            for r in cutlass.range_constexpr(ROWS):
                peer_d_buf[r] = cute.arch.shuffle_sync_bfly(
                    d_arr[r], offset=offset, mask=-1, mask_and_clamp=31,
                )
            for r in cutlass.range_constexpr(ROWS):
                peer_i_buf[r] = cute.arch.shuffle_sync_bfly(
                    i_arr[r], offset=offset, mask=-1, mask_and_clamp=31,
                )
            for r in cutlass.range_constexpr(ROWS):
                take_my = (d_arr[r] < peer_d_buf[r]) == i_am_lower
                d_arr[r] = d_arr[r] if take_my else peer_d_buf[r]
                i_arr[r] = i_arr[r] if take_my else peer_i_buf[r]

    @cute.jit
    def _warp_merge_multi_with_chunk(
        self, top_d_arr, top_i_arr, chunk_d_arr, chunk_i_arr,
        peer_d_buf, peer_i_buf,
        ROWS: cutlass.Constexpr,
    ):
        """Merge sorted-ASC top with sorted-ASC chunk for ROWS rows in
        parallel; result written back into ``top_*_arr`` in place.

        For each row r:
          chunk_rev = lane-reverse(chunk_arr[r])  via shfl(offset=31)
          merged    = elementwise min(top_arr[r], chunk_rev)  -> bitonic
          bitonic_finish(merged) -> sorted ASC top-32

        Phases A/B issue all ROWS shfls into ``peer_*_buf`` first,
        decoupling the shfl issue stream from the compare/select.
        """
        lane = cute.arch.lane_idx()
        # Phase A1: issue ROWS reverse-shfls for d into peer_d_buf.
        for r in cutlass.range_constexpr(ROWS):
            peer_d_buf[r] = cute.arch.shuffle_sync_bfly(
                chunk_d_arr[r], offset=31, mask=-1, mask_and_clamp=31,
            )
        # Phase A2: issue ROWS reverse-shfls for i into peer_i_buf.
        for r in cutlass.range_constexpr(ROWS):
            peer_i_buf[r] = cute.arch.shuffle_sync_bfly(
                chunk_i_arr[r], offset=31, mask=-1, mask_and_clamp=31,
            )
        # Phase B: elementwise min into top_*_arr in place. After
        # this, top_*_arr is bitonic.
        for r in cutlass.range_constexpr(ROWS):
            take_top = top_d_arr[r] < peer_d_buf[r]
            top_d_arr[r] = top_d_arr[r] if take_top else peer_d_buf[r]
            top_i_arr[r] = top_i_arr[r] if take_top else peer_i_buf[r]
        # Phase C: bitonic-finish on top_*_arr (now bitonic) -> ASC.
        self._warp_bitonic_finish_multi_asc(
            top_d_arr, top_i_arr, peer_d_buf, peer_i_buf, ROWS,
        )

    @cute.jit
    def _chunk_topk_warpsort(
        self, sDist, heap_d_arr, heap_i_arr,
        chunk_d_arr, chunk_i_arr,
        peer_d_buf, peer_i_buf,
        BM: cutlass.Constexpr, BN: cutlass.Constexpr,
        ROWS_PER_WARP: cutlass.Constexpr,
        K_PAD: cutlass.Constexpr,
        cta_m_offset, cta_n_offset, N_total, M_total,
        warp_idx, num_consumer_warps: cutlass.Constexpr,
    ):
        """Per-chunk multi-row warp-cooperative top-K update.

        For each sub-chunk of 32 cols (BN/32 sub-chunks per chunk):
          1. All ROWS_PER_WARP rows of this warp load their sub-chunk
             distances into chunk_d_arr/chunk_i_arr (1 element per
             row per lane).
          2. Multi-row warp-cooperative bitonic-sort all rows in
             parallel (15 substages, each issues ROWS_PER_WARP shfls
             back-to-back to hide shfl latency).
          3. Multi-row merge with the running top-K
             (heap_d_arr/heap_i_arr).

        K_PAD < 32: heap_d_arr lanes K_PAD..31 are kept at +INF so
        they sort to the tail and never displace real entries.
        """
        lane = cute.arch.lane_idx()
        NUM_SUBCHUNKS = BN // 32
        INF = cutlass.Float32(3.4e38)
        for sub in cutlass.range_constexpr(NUM_SUBCHUNKS):
            # Phase 1: load all ROWS_PER_WARP rows' chunk values.
            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                my_row = warp_idx + r * num_consumer_warps
                col = sub * 32 + lane
                m_global = my_row + cta_m_offset
                n_global = col + cta_n_offset
                # Default to INF/-1 so masked lanes don't pollute the
                # sort. Both my_row<BM and {m,n}_global<{N,M}_total
                # checks fold into the gate.
                chunk_d_arr[r] = INF
                chunk_i_arr[r] = cutlass.Int32(-1)
                if my_row < BM:
                    if m_global < N_total:
                        if n_global < M_total:
                            chunk_d_arr[r] = sDist[(my_row, col)]
                            chunk_i_arr[r] = cutlass.Int32(n_global)
            # Phase 2: multi-row sort all chunks in parallel.
            self._warp_bitonic_sort_multi_asc(
                chunk_d_arr, chunk_i_arr,
                peer_d_buf, peer_i_buf, ROWS_PER_WARP,
            )
            # Phase 3: multi-row merge with running top-K.
            # NOTE: we elide the threshold prune (broadcast chunk_min
            # & heap_worst then gate). The broadcast costs ~10 shfls
            # per row, and the merge is only 6 substages = 6 shfls
            # per row, so prune does not pay back.
            self._warp_merge_multi_with_chunk(
                heap_d_arr, heap_i_arr,
                chunk_d_arr, chunk_i_arr,
                peer_d_buf, peer_i_buf,
                ROWS_PER_WARP,
            )

    @cute.jit
    def _chunk_topk_smem_perthread(
        self, sDist, topk_d, topk_i,
        BM: cutlass.Constexpr, BN: cutlass.Constexpr,
        rows_per_thr: cutlass.Constexpr,
        cta_m_offset, cta_n_offset, N_total, M_total,
        consumer_tidx,
    ):
        """SMEM-relayout per-thread top-K (1 thread per query row).

        Each thread is assigned ``rows_per_thr`` query rows. For each
        row it streams the BN distances from sDist with no cross-thread
        coordination, runs threshold-prune + branch-free bubble insert.

        This is what enables the WS top-K idea: the layout has zero
        cross-thread dependencies in the inner top-K loop, so the
        inner loop runs at 1 IPC per-element-pruned (or K IPC per
        insert) with no shuffles. The price is that 1 thread per row
        sees the full BN columns instead of BN/(threads-in-row) -- but
        that price gets amortised by pipelining GEMM with top-K when
        WS3 staging is enabled.

        Layout:
          sDist : (BM, BN) row-major fp32, written by GEMM stage above.
          topk_d/i : (rows_per_thr, K_PAD), per-thread sorted-asc heap.

        For BM=128 with a 128-thread warpgroup: rows_per_thr=1, every
        thread owns exactly one query row.
        For BM=256 with a 128-thread warpgroup: rows_per_thr=2.
        For BM=64  with a 128-thread warpgroup: half the threads own a
        row (rows_per_thr=1, indexed via consumer_tidx < BM); the rest
        contribute to GEMM but skip top-K.
        """
        K = self.k_pad
        for r in cutlass.range_constexpr(rows_per_thr):
            my_row = consumer_tidx + r * 128
            if my_row < BM:
                m_global = my_row + cta_m_offset
                if m_global < N_total:
                    worst_d = topk_d[(r, K - 1)]
                    # Partial unroll (factor 4) instead of full
                    # range_constexpr(BN). At BN=128 the fully-unrolled
                    # loop emits ~128 * (1 ld.shared + 1 setp + K-deep
                    # bubble insert) inlined instructions; combined
                    # with the unrolled bubble inserts the compiler
                    # blows past the 240-reg cap and spills heavily.
                    # cutlass.range(unroll=4) caps the inlined chunk at
                    # a manageable size while still letting the
                    # compiler pipeline 4 ld.shared issues at a time.
                    for j in cutlass.range(BN, unroll=4):
                        n_global = j + cta_n_offset
                        if n_global < M_total:
                            c_d = sDist[(my_row, j)]
                            if c_d < worst_d:
                                c_i = cutlass.Int32(n_global)
                                self._bubble_insert_asc(
                                    topk_d, topk_i, r, K, c_d, c_i,
                                )
                                worst_d = topk_d[(r, K - 1)]

    @cute.jit
    def _ws3_modeH_chunk_write(
        self,
        acc_mn,                # rmem (M_per_thr, N_per_thr) fp32 acc slice
        sDist_stage,           # smem (BM, BN) slice for this chunk
        sChunkMin_stage,       # smem (BM,) slice for this chunk
        sWorstD,               # smem (BM,) cross-WG worst-d feedback
        cs,                    # cs only, x_sq dropped
        ptPcP_mn,              # rmem identity tensor for (m_local, n_local) lookup
        tiled_mma,             # const for cross-thread reduce shape
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
    ):
        """Mode-H 2-pass dist write (extracted helper for acc pipelining).

        Identical to the inline body in ``kernel_ws3``'s GEMM WG, but
        takes ``acc_mn`` as an arg so a 2-stage acc ping-pong can call
        it once per stage's acc. Pass 1: per-thread row_min from acc.
        Cross-shfl reduce. Pass 2: read sWorstD per row, conditionally
        STS the row's BN d-values to sDist (skip 99% at steady state).
        ALWAYS writes chunk_min to sChunkMin (consumer-side prune
        gate). Caller is responsible for ``producer_acquire``,
        ``fence_proxy(async.shared)``, and ``producer_commit``.
        """
        # ---- Pass 1: compute per-thread row_min (no STS) ----
        row_min_local = cute.make_rmem_tensor(
            cute.make_layout(M_per_thr), cutlass.Float32
        )
        for i in cutlass.range_constexpr(M_per_thr):
            row_min_local[i] = cutlass.Float32(3.4e38)
        for i in cutlass.range_constexpr(M_per_thr):
            rmin_i = row_min_local[i]
            for j in cutlass.range_constexpr(N_per_thr):
                cross = acc_mn[(i, j)]
                # Signed score: c_sq[m] − 2·cross. No x_sq, no clamp.
                d = cs[j] - cutlass.Float32(2.0) * cross
                if d < rmin_i:
                    rmin_i = d
            row_min_local[i] = rmin_i

        # ---- Cross-shfl reduce row_min over threads-in-row ----
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        row_min_full = cute.make_rmem_tensor(
            cute.make_layout(M_per_thr), cutlass.Float32
        )
        for i in cutlass.range_constexpr(M_per_thr):
            rmin_i = row_min_local[i]
            for r in cutlass.range_constexpr(red_rank):
                tig = red_target.shape[r]
                offset = tig // 2
                while offset > 0:
                    peer = cute.arch.shuffle_sync_bfly(
                        rmin_i, offset=offset,
                        mask=-1, mask_and_clamp=31,
                    )
                    if peer < rmin_i:
                        rmin_i = peer
                    offset = offset // 2
            row_min_full[i] = rmin_i

        # ---- Pass 2: per-row conditional dist STS ----
        for i in cutlass.range_constexpr(M_per_thr):
            m_local0 = ptPcP_mn[(i, 0)][0]
            worst_d_stale = sWorstD[m_local0]
            if row_min_full[i] < worst_d_stale:
                for j in cutlass.range_constexpr(N_per_thr):
                    cross = acc_mn[(i, j)]
                    # Signed score: c_sq[m] − 2·cross. No x_sq, no clamp.
                    d = cs[j] - cutlass.Float32(2.0) * cross
                    m_local = ptPcP_mn[(i, j)][0]
                    n_local = ptPcP_mn[(i, j)][1]
                    sDist_stage[(m_local, n_local)] = d

        # ---- Always write chunk_min (consumer's prune gate) ----
        for i in cutlass.range_constexpr(M_per_thr):
            n_local0 = ptPcP_mn[(i, 0)][1]
            if n_local0 == 0:
                m_local = ptPcP_mn[(i, 0)][0]
                sChunkMin_stage[m_local] = row_min_full[i]

    @cute.jit
    def _chunk_topk_smem_perthread_with_chunkmin(
        self, sDist, sChunkMin, topk_d, topk_i,
        BM: cutlass.Constexpr, BN: cutlass.Constexpr,
        rows_per_thr: cutlass.Constexpr,
        cta_m_offset, cta_n_offset, N_total, M_total,
        consumer_tidx, sWorstD=None,
    ):
        """smem_perthread inner loop with WS3 per-row chunk-min prune.

        Identical to ``_chunk_topk_smem_perthread`` except for the
        outer chunk-level early exit -- mirrors Triton stage-1's
        ``if chunk_best < topk_worst_val: ...`` (knn_triton.py:692).

        Algorithm per chunk per row:
          chunk_min = sChunkMin[my_row]              # 1 ld.shared
          if chunk_min >= topk_d[K-1]:
              return                                 # skip ALL BN reads
          for j in 0..BN-1:
              c_d = sDist[my_row, j]
              if c_d < worst_d: bubble_insert(...)

        At steady state with random query/db data, after the heap
        matures (~K chunks in) >99% of subsequent chunks fail the
        outer prune and skip the entire BN-element loop. Per chunk
        per row: 2 ops vs 192 ops (BN ld + BN setp + K-deep insert).

        Empirical at K=24 BM=128 BN=64 N=16384 M=100K D=128:
          - smem_perthread (no chunk-min prune):  6024 us
          - smem_perthread (chunk-min prune):     <expected ~1500 us>
        """
        K = self.k_pad
        for r in cutlass.range_constexpr(rows_per_thr):
            my_row = consumer_tidx + r * 128
            if my_row < BM:
                m_global = my_row + cta_m_offset
                if m_global < N_total:
                    worst_d = topk_d[(r, K - 1)]
                    chunk_min = sChunkMin[my_row]
                    if chunk_min < worst_d:
                        for j in cutlass.range(BN, unroll=4):
                            n_global = j + cta_n_offset
                            if n_global < M_total:
                                c_d = sDist[(my_row, j)]
                                if c_d < worst_d:
                                    c_i = cutlass.Int32(n_global)
                                    self._bubble_insert_asc(
                                        topk_d, topk_i, r, K, c_d, c_i,
                                    )
                                    worst_d = topk_d[(r, K - 1)]
                        # Mode H feedback: write fresh worst_d back
                        # for the GEMM WG to read on subsequent
                        # chunks. Inside the chunk-min branch so we
                        # only pay the STS when the heap actually
                        # changed (~1% of chunks at steady state);
                        # for pruned chunks, sWorstD is already
                        # equal to the unchanged worst_d.
                        if sWorstD is not None:
                            sWorstD[my_row] = worst_d

    @cute.jit
    def _bubble_insert_asc(
        self, topk_d, topk_i,
        row: cutlass.Constexpr,
        K: cutlass.Constexpr,
        c_d, c_i,
    ):
        """Branch-free CUTLASS-style bubble insert into sorted-asc top-K.

        Mirrors ``add_element_to_desc_sorted_array`` from CUTLASS's
        ``sm90_visitor_topk_softmax.hpp`` but for ascending order (we
        want the K SMALLEST distances, not largest values).

        Algorithm (constexpr-unrolled bubble pass):
          pending = (c_d, c_i)
          for k in 0..K-1:
            cur = topk[k]
            take_pending = (cur > pending) or ((cur == pending) and idx>)
            topk[k]  = pending if take_pending else cur
            pending  = cur     if take_pending else pending

        After the loop, ``pending`` holds the displaced ``topk[K-1]``
        (or ``c`` itself if ``c`` was the largest, in which case
        nothing changed). Each iter compiles to ~4 selp + 1 setp PTX
        — no branches, no warp divergence.

        Caller MUST gate on ``c_d < topk_d[row, K-1]`` first to skip
        the whole bubble pass when ``c`` doesn't make the top-K. This
        is the single biggest win over the bitonic path: at steady
        state >90% of chunk elements get pruned with one compare.
        """
        pending_d = c_d
        pending_i = c_i
        for k in cutlass.range_constexpr(K):
            cur_d = topk_d[(row, k)]
            cur_i = topk_i[(row, k)]
            # Strict ``>`` (no idx tie-break). The tie-break costs one
            # SETP, one AND and one OR per insert iter; dropping it
            # gives a measurable speedup with no impact on the final
            # top-K SET (only on the order of equal-d entries within
            # it). Same rationale as the sortmerge tie-break drop.
            take_pending = cur_d > pending_d
            topk_d[(row, k)] = pending_d if take_pending else cur_d
            topk_i[(row, k)] = pending_i if take_pending else cur_i
            pending_d = cur_d if take_pending else pending_d
            pending_i = cur_i if take_pending else pending_i

    @cute.jit
    def _chunk_topk_perthread(
        self, dist_buf, ptPcP_mn, topk_d, topk_i, heap_max,
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
        cta_n_offset,
    ):
        """Per-thread top-K update via branch-free bubble insert.

        Each thread maintains a SORTED ASCENDING top-K of size K_PAD over
        ITS OWN column slice (N_per_thr columns per row). For each chunk
        element:

          1. Threshold compare: ``c_d < topk_d[K-1]``? If not, skip.
             This is one ``setp.lt`` per element -- the dominant cost
             at steady state where the global top-K has converged.
          2. On pass: branch-free bubble insert (see ``_bubble_insert_asc``).
             O(K) selp PTX, no shuffles.

        No cross-thread shuffles in the mainloop. After the mainloop
        ``_warp_merge_topk_perthread`` reduces across threads-in-row.

        References:
        * CUTLASS sm90_visitor_topk_softmax.hpp (per-thread sorted-K
          + add-element-to-sorted-array pattern)
        * Triton tl.sort (standard.py): operates per-CTA across
          warp lanes via reshape-as-hypercube + xor_sum trick. Same
          asymptotic story (bubble-style compare-swap network) but
          Triton parallelises the bubble across the SIMT lanes that
          collectively own the row -- which is impossible here
          because a chunk element only lives in ONE thread's
          register and crossing lanes would re-introduce the
          shuffles we're trying to avoid.

        Tie-break: equal distances keep the smaller original index
        (matches Triton's packed-uint64 sort).
        """
        K = self.k_pad
        for i in cutlass.range_constexpr(M_per_thr):
            worst_d = topk_d[(i, K - 1)]
            for j in cutlass.range_constexpr(N_per_thr):
                c_d = dist_buf[(i, j)]
                if c_d < worst_d:
                    n_local = ptPcP_mn[(i, j)][1]
                    c_i = cutlass.Int32(n_local + cta_n_offset)
                    self._bubble_insert_asc(topk_d, topk_i, i, K, c_d, c_i)
                    worst_d = topk_d[(i, K - 1)]
            heap_max[i] = worst_d

    @cute.jit
    def _warp_merge_topk_perthread(
        self, tiled_mma, topk_d, topk_i,
        M_per_thr: cutlass.Constexpr,
    ):
        """Butterfly merge sorted-asc top-Ks across threads-in-row.

        Same butterfly tournament as the cooperative path, but each
        peer round inserts the peer's K entries one-at-a-time using
        ``_bubble_insert_asc`` instead of running a full bitonic
        merge. K_PAD bubble inserts per peer = O(K^2) per phase per
        row -- tiny in absolute terms (one-shot, not per chunk).

        Snapshot-before-mutate: shuffle ALL K peer entries first
        into local registers, then bubble-insert. If we shuffled +
        inserted in the same step, our heap mutating mid-loop would
        corrupt subsequent shuffles for the same peer (same bug as
        in ``_warp_merge_topk``).
        """
        K = self.k_pad
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for r in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[r]
            offset = tig // 2
            while offset > 0:
                peer_d_buf = cute.make_rmem_tensor(
                    cute.make_layout(K), cutlass.Float32
                )
                peer_i_buf = cute.make_rmem_tensor(
                    cute.make_layout(K), cutlass.Int32
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for kk in cutlass.range_constexpr(K):
                        peer_d_buf[kk] = cute.arch.shuffle_sync_bfly(
                            topk_d[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        peer_i_buf[kk] = cute.arch.shuffle_sync_bfly(
                            topk_i[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                    worst_d = topk_d[(i, K - 1)]
                    for kk in cutlass.range_constexpr(K):
                        c_d = peer_d_buf[kk]
                        c_i = peer_i_buf[kk]
                        if c_d < worst_d:
                            self._bubble_insert_asc(
                                topk_d, topk_i, i, K, c_d, c_i,
                            )
                            worst_d = topk_d[(i, K - 1)]
                offset = offset // 2

    # ----------------------------------------------------------------------
    # Maxtree top-K (ported from the Blackwell BUILD kernel, sm_100)
    # ----------------------------------------------------------------------
    #
    # An UNSORTED per-thread register top-K with a cached
    # ``(worst_d, worst_pos)`` and two wins over the sorted bubble-insert
    # ``perthread`` path:
    #   * group-min-4 prune: scan candidates 4 at a time; if even the min
    #     of the 4 can't beat ``worst_d``, skip all 4 with ONE compare
    #     (vs 1 compare/element for the sorted path).
    #   * worst-of-K recompute (``_worst_row``, K-adaptive): refresh the
    #     evict slot after each insert. K<=10 uses a balanced max-tree
    #     (O(log K) depth); K>=11 uses a streaming running-max (O(K)
    #     compares but only 2 live scalars) because the max-tree's K live
    #     leaves spill in CuteDSL/MLIR at higher K -- the Blackwell BUILD
    #     learning (see ``blackwell_impl._worst_of``).
    # ``(worst_d, worst_pos)`` persist across chunks in small constexpr-
    # indexed register arrays (``heap_max`` / ``heap_wpos``), so a fully
    # pruned chunk touches the heap zero times. The heap stays unsorted in
    # the mainloop; a one-shot selection sort in the epilogue restores
    # ascending order before the GMEM write. See ``blackwell_impl.py``
    # (``_topk_init`` / ``_worst_of`` / ``_topk_consume_tile``) for the
    # sm_100 original.

    @cute.jit
    def _worst_tree_row(
        self, heap_d, row: cutlass.Constexpr, K: cutlass.Constexpr,
    ):
        """Balanced max-tree over ``heap_d[row, :]`` -> (worst_d, worst_pos).

        O(log K) dependent compares to find the largest (worst) entry and
        its slot, vs O(K) for a linear scan. The tree is unrolled at trace
        time (the ``items`` list is Python-level metaprogramming, same as
        the Blackwell ``_worst_tree``)."""
        items = []
        for j in cutlass.range_constexpr(K):
            items.append((heap_d[(row, j)], cutlass.Int32(j)))
        while cutlass.const_expr(len(items) > 1):
            nxt = []
            m = len(items) // 2
            for a in cutlass.range_constexpr(m):
                va, pa = items[2 * a]
                vb, pb = items[2 * a + 1]
                gt = vb > va
                nxt.append((cutlass.max(va, vb), cutlass.select_(gt, pb, pa)))
            if cutlass.const_expr(len(items) % 2 == 1):
                nxt.append(items[-1])
            items = nxt
        return items[0]

    @cute.jit
    def _worst_of_row(
        self, heap_d, row: cutlass.Constexpr, K: cutlass.Constexpr,
    ):
        """Streaming running-max worst-of-K over ``heap_d[row, :]`` ->
        (worst_d, worst_pos).

        Only two scalars (``worst_d`` / ``worst_pos``) stay live across the
        unrolled scan, vs the balanced ``_worst_tree_row`` which materialises
        all K (value, pos) leaves into SSA at once. Ported from the Blackwell
        BUILD kernel's ``_worst_of`` (``blackwell_impl.py``): there the
        max-tree -- K live leaves stacked on the MMA fragment -- makes MLIR
        spill and runs ~2x slower even at K=5/10, while the streaming scan
        keeps occupancy high. Hopper rides the same MLIR backend, so the same
        swap applies. O(K) compares, but it only runs on the rare insert that
        survives the group-min-4 prune, so the local-memory loads stay
        L1-resident and cheap."""
        worst_d = heap_d[(row, 0)]
        worst_pos = cutlass.Int32(0)
        for jj in cutlass.range_constexpr(K - 1):
            j = jj + 1
            cur = heap_d[(row, j)]
            gt = cur > worst_d
            worst_d = cutlass.select_(gt, cur, worst_d)
            worst_pos = cutlass.select_(gt, cutlass.Int32(j), worst_pos)
        return worst_d, worst_pos

    @cute.jit
    def _worst_row(
        self, heap_d, row: cutlass.Constexpr, K: cutlass.Constexpr,
    ):
        """Recompute ``(worst_d, worst_pos)`` over the unsorted heap row.

        K-adaptive (mode "auto", default): the streaming scan keeps 2 scalars
        live and wins once the balanced max-tree's K leaves start spilling.
        Measured crossover on H100 is K~11 at the router tile (BM256/BN64):
        K<=10 the shallower max-tree dependency chain is faster; K>=12 the
        max-tree spills and streaming is up to ~1.6x faster -- it recovers the
        build win the max-tree gives back to the spill (K=16: 0.78x->1.23x vs
        the sorted ``perthread``). ``FLASHLIB_KNN_WORST_STREAM=0/1`` pins
        tree/stream for A/B."""
        mode = self._worst_stream_mode
        if cutlass.const_expr(mode == "0"):
            stream = False
        elif cutlass.const_expr(mode == "1"):
            stream = True
        else:
            stream = cutlass.const_expr(K >= _MAXTREE_STREAM_KMIN)
        if cutlass.const_expr(stream):
            return self._worst_of_row(heap_d, row, K)
        return self._worst_tree_row(heap_d, row, K)

    @cute.jit
    def _maxtree_insert(
        self, heap_d, heap_i, row: cutlass.Constexpr, K: cutlass.Constexpr,
        worst_pos, c_d, c_i,
    ):
        """Evict the current worst entry with the candidate, then refresh
        ``(worst_d, worst_pos)`` via ``_worst_row`` (K-adaptive streaming
        scan / max-tree). Caller guarantees ``c_d < worst_d``. Returns the
        refreshed ``(worst_d, worst_pos)``.

        The dynamic store ``heap_d[row, worst_pos]`` is intentional: an
        O(K) predicated-write loop that keeps the heap register-resident
        was measured ~2x SLOWER here (the per-insert scan of all K slots
        dominates), while the dynamic store touches the heap only on the
        rare inserts that survive the group-min prune."""
        heap_d[(row, worst_pos)] = c_d
        heap_i[(row, worst_pos)] = c_i
        return self._worst_row(heap_d, row, K)

    @cute.jit
    def _chunk_topk_maxtree(
        self, dist_buf, ptPcP_mn, heap_d, heap_i, heap_max, heap_wpos,
        M_per_thr: cutlass.Constexpr,
        N_per_thr: cutlass.Constexpr,
        cta_n_offset,
    ):
        """Per-thread maxtree top-K update over this thread's column slice.

        ``(worst_d, worst_pos)`` live in ``heap_max`` / ``heap_wpos``
        across chunks, so a fully pruned chunk does only ``N_per_thr // 4``
        group-min compares against the cached ``worst_d`` and never reads
        the heap. Index of column ``j`` is recovered from the identity
        tensor ``ptPcP_mn`` (the WGMMA TV-layout column owner)."""
        K = self.k_pad
        NG = N_per_thr // 4
        REM = N_per_thr - NG * 4
        for i in cutlass.range_constexpr(M_per_thr):
            worst_d = heap_max[i]
            worst_pos = heap_wpos[i]
            for g in cutlass.range_constexpr(NG):
                base = g * 4
                c0 = dist_buf[(i, base + 0)]
                c1 = dist_buf[(i, base + 1)]
                c2 = dist_buf[(i, base + 2)]
                c3 = dist_buf[(i, base + 3)]
                gmin = cutlass.min(cutlass.min(c0, c1), cutlass.min(c2, c3))
                if gmin < worst_d:
                    cands = [c0, c1, c2, c3]
                    for t in cutlass.range_constexpr(4):
                        cv = cands[t]
                        if cv < worst_d:
                            n_local = ptPcP_mn[(i, base + t)][1]
                            c_i = cutlass.Int32(n_local + cta_n_offset)
                            worst_d, worst_pos = self._maxtree_insert(
                                heap_d, heap_i, i, K, worst_pos, cv, c_i,
                            )
            for r in cutlass.range_constexpr(REM):
                j = NG * 4 + r
                cv = dist_buf[(i, j)]
                if cv < worst_d:
                    n_local = ptPcP_mn[(i, j)][1]
                    c_i = cutlass.Int32(n_local + cta_n_offset)
                    worst_d, worst_pos = self._maxtree_insert(
                        heap_d, heap_i, i, K, worst_pos, cv, c_i,
                    )
            heap_max[i] = worst_d
            heap_wpos[i] = worst_pos

    @cute.jit
    def _warp_merge_topk_maxtree(
        self, tiled_mma, heap_d, heap_i,
        M_per_thr: cutlass.Constexpr,
    ):
        """Butterfly-merge unsorted per-thread maxtree top-Ks across the
        threads-in-row reduction group. Mirrors
        ``_warp_merge_topk_perthread`` but inserts peer entries via the
        maxtree (unsorted + max-tree) insert. Snapshot-before-mutate: all K
        peer entries are shuffled into registers before any insert."""
        K = self.k_pad
        red_target = self._reduction_target_n(tiled_mma)
        red_rank = cute.rank(red_target)
        for rr in cutlass.range_constexpr(red_rank):
            tig = red_target.shape[rr]
            offset = tig // 2
            while offset > 0:
                peer_d_buf = cute.make_rmem_tensor(
                    cute.make_layout(K), cutlass.Float32
                )
                peer_i_buf = cute.make_rmem_tensor(
                    cute.make_layout(K), cutlass.Int32
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for kk in cutlass.range_constexpr(K):
                        peer_d_buf[kk] = cute.arch.shuffle_sync_bfly(
                            heap_d[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                        peer_i_buf[kk] = cute.arch.shuffle_sync_bfly(
                            heap_i[(i, kk)],
                            offset=offset, mask=-1, mask_and_clamp=31,
                        )
                    worst_d, worst_pos = self._worst_row(heap_d, i, K)
                    for kk in cutlass.range_constexpr(K):
                        c_d = peer_d_buf[kk]
                        c_i = peer_i_buf[kk]
                        if c_d < worst_d:
                            worst_d, worst_pos = self._maxtree_insert(
                                heap_d, heap_i, i, K, worst_pos, c_d, c_i,
                            )
                offset = offset // 2

    @cute.jit
    def _chunk_topk_smem_maxtree(
        self, sDist, sChunkMin, heap_d, heap_i, heap_max, heap_wpos,
        BM: cutlass.Constexpr, BN: cutlass.Constexpr,
        rows_per_thr: cutlass.Constexpr,
        cta_m_offset, cta_n_offset, N_total, M_total,
        consumer_tidx, sWorstD=None,
    ):
        """smem maxtree inner loop (1 thread per row, WS3 / WS4).

        Keeps the per-row chunk-min prune of
        ``_chunk_topk_smem_perthread_with_chunkmin`` (skip the whole
        BN-element loop when the chunk can't beat ``worst_d``) but
        replaces the per-element sorted bubble with the maxtree group-min-4
        prune + unsorted heap + max-tree. OOB db columns are +inf in
        ``sDist`` (the GEMM WG pads ``c_sq``), so they never insert and
        need no per-element bound check. ``worst_d`` (max of K) is the
        same value the sorted path wrote, so the Mode-H ``sWorstD``
        contract is preserved."""
        K = self.k_pad
        NG = BN // 4
        REM = BN - NG * 4
        for r in cutlass.range_constexpr(rows_per_thr):
            my_row = consumer_tidx + r * 128
            if my_row < BM:
                m_global = my_row + cta_m_offset
                if m_global < N_total:
                    worst_d = heap_max[r]
                    worst_pos = heap_wpos[r]
                    chunk_min = sChunkMin[my_row]
                    if chunk_min < worst_d:
                        for g in cutlass.range(NG, unroll=4):
                            base = g * 4
                            c0 = sDist[(my_row, base + 0)]
                            c1 = sDist[(my_row, base + 1)]
                            c2 = sDist[(my_row, base + 2)]
                            c3 = sDist[(my_row, base + 3)]
                            gmin = cutlass.min(
                                cutlass.min(c0, c1), cutlass.min(c2, c3)
                            )
                            if gmin < worst_d:
                                cands = [c0, c1, c2, c3]
                                for t in cutlass.range_constexpr(4):
                                    cv = cands[t]
                                    if cv < worst_d:
                                        c_i = cutlass.Int32(
                                            base + t + cta_n_offset
                                        )
                                        worst_d, worst_pos = self._maxtree_insert(
                                            heap_d, heap_i, r, K,
                                            worst_pos, cv, c_i,
                                        )
                        for rt in cutlass.range_constexpr(REM):
                            j = NG * 4 + rt
                            cv = sDist[(my_row, j)]
                            if cv < worst_d:
                                c_i = cutlass.Int32(j + cta_n_offset)
                                worst_d, worst_pos = self._maxtree_insert(
                                    heap_d, heap_i, r, K, worst_pos, cv, c_i,
                                )
                        heap_max[r] = worst_d
                        heap_wpos[r] = worst_pos
                        if sWorstD is not None:
                            sWorstD[my_row] = worst_d

    # ----------------------------------------------------------------------
    # Warp-specialised device kernel
    # ----------------------------------------------------------------------

    @cute.kernel
    def kernel_ws(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_md: cute.Tensor,
        mCsq_m: cute.Tensor,
        mOutI_nk: cute.Tensor,    # (N, K_PAD) int32  — indices only
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        BM = self.tile_shape_mnk[0]
        BN = self.tile_shape_mnk[1]
        N_total = mX_nd.shape[0]
        M_total = mC_md.shape[0]
        cta_m_offset = bidx * BM
        K_PAD = self.k_pad
        num_c_tiles = (M_total + BN - 1) // BN

        # --- Common setup (both WG roles) ----------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype, cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=cute.size_in_bytes(
                self.c_dtype, cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        gC_md = cute.local_tile(
            mC_md, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), (None, 0),
        )
        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x, 0, cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(
                cute.local_tile(
                    mX_nd, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),
        )
        tma_cS, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_md, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- Producer WG ---------------------------------------------------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.x_stage
            )
            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_stage
            )
            if warp_idx == 0:
                x_pipeline.producer_acquire(x_producer_state)
                cute.copy(
                    tma_atom_x,
                    tma_xG[(None, bidx)],
                    tma_xS[(None, x_producer_state.index)],
                    tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
                )
                x_pipeline.producer_commit(x_producer_state)
                x_producer_state.advance()

                for c_idx in cutlass.range(num_c_tiles, unroll=1):
                    c_pipeline.producer_acquire(c_producer_state)
                    cute.copy(
                        tma_atom_c,
                        tma_cG[(None, c_producer_state.count)],
                        tma_cS[(None, c_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                    )
                    c_pipeline.producer_commit(c_producer_state)
                    c_producer_state.advance()

        # --- Consumer WG(s) ------------------------------------------------
        if warp_group_idx >= self.num_load_warp_groups:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            consumer_tidx = tidx - self.num_threads_per_warp_group
            thr_mma = tiled_mma.get_slice(consumer_tidx)

            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)

            cP = cute.make_identity_tensor((BM, BN))
            ptPcP = thr_mma.partition_C(cP)
            gC_fake = cute.make_identity_tensor((BM, BN))
            tCgC_fake = thr_mma.partition_C(gC_fake)
            acc_shape = tCgC_fake.shape
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stage
            )
            x_pipeline.consumer_wait(x_consumer_state)

            acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
            acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
            )
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])

            # x_sq dropped; signed score = c_sq[m] − 2·cross.

            K_INTERNAL = self.k_pad_pow2 if cutlass.const_expr(
                self.topk_strategy in ("sortmerge", "sortmerge_packed")
            ) else K_PAD
            if cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                heap_packed = cute.make_rmem_tensor(
                    cute.make_layout((M_per_thr, K_INTERNAL)), cutlass.Int64
                )
                INF_PACKED = cutlass.Int64(0x7F800000FFFFFFFF)
                for i in cutlass.range_constexpr(M_per_thr):
                    for k in cutlass.range_constexpr(K_INTERNAL):
                        heap_packed[(i, k)] = INF_PACKED
            else:
                heap_d = cute.make_rmem_tensor(
                    cute.make_layout((M_per_thr, K_INTERNAL)), cutlass.Float32
                )
                heap_i = cute.make_rmem_tensor(
                    cute.make_layout((M_per_thr, K_INTERNAL)), cutlass.Int32
                )
                heap_max = cute.make_rmem_tensor(
                    cute.make_layout(M_per_thr), cutlass.Float32
                )
                # ``maxtree`` persists the worst slot per row (see the
                # non-WS kernel for the rationale). Unused by the other
                # strategies.
                heap_wpos = cute.make_rmem_tensor(
                    cute.make_layout(M_per_thr), cutlass.Int32
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for k in cutlass.range_constexpr(K_INTERNAL):
                        heap_d[(i, k)] = cutlass.Float32(3.4e38)
                        heap_i[(i, k)] = cutlass.Int32(-1)
                    heap_max[i] = cutlass.Float32(3.4e38)
                    heap_wpos[i] = cutlass.Int32(0)

            if cutlass.const_expr(self.topk_strategy == "sortmerge"):
                CHUNK_INTERNAL = self._next_pow2(N_per_thr)
                SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
                chunk_d = cute.make_rmem_tensor(
                    cute.make_layout(SCRATCH_LEN), cutlass.Float32
                )
                chunk_i = cute.make_rmem_tensor(
                    cute.make_layout(SCRATCH_LEN), cutlass.Int32
                )
            elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                CHUNK_INTERNAL = self._next_pow2(N_per_thr)
                SCRATCH_LEN = max(K_INTERNAL, CHUNK_INTERNAL)
                chunk_packed = cute.make_rmem_tensor(
                    cute.make_layout(SCRATCH_LEN), cutlass.Int64
                )

            c_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            c_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            num_k_blocks = cute.size(tCrX, mode=[2])

            for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
                c_pipeline.consumer_wait(c_consumer_read_state)

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma, acc,
                        tCrX[(None, None, k_block_idx, 0)],
                        tCrC[(None, None, k_block_idx, c_consumer_read_state.index)],
                        acc,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                c_pipeline.consumer_release(c_consumer_release_state)
                c_consumer_read_state.advance()
                c_consumer_release_state.advance()

                cta_n_offset = c_tile_idx * BN

                cs = cute.make_rmem_tensor(cute.make_layout(N_per_thr), cutlass.Float32)
                for j in cutlass.range_constexpr(N_per_thr):
                    n_local = ptPcP_mn[(0, j)][1]
                    n_global = n_local + cta_n_offset
                    if n_global < M_total:
                        cs[j] = mCsq_m[n_global]
                    else:
                        cs[j] = cutlass.Float32(3.4e38)

                dist_buf = cute.make_rmem_tensor(
                    cute.make_layout((M_per_thr, N_per_thr)), cutlass.Float32
                )
                for i in cutlass.range_constexpr(M_per_thr):
                    for j in cutlass.range_constexpr(N_per_thr):
                        cross = acc_mn[(i, j)]
                        # Signed score: c_sq[m] − 2·cross. No x_sq, no clamp.
                        d = cs[j] - cutlass.Float32(2.0) * cross
                        dist_buf[(i, j)] = d

                if cutlass.const_expr(self.topk_strategy == "perthread"):
                    self._chunk_topk_perthread(
                        dist_buf, ptPcP_mn, heap_d, heap_i, heap_max,
                        M_per_thr, N_per_thr, cta_n_offset,
                    )
                elif cutlass.const_expr(self.topk_strategy == "maxtree"):
                    self._chunk_topk_maxtree(
                        dist_buf, ptPcP_mn, heap_d, heap_i, heap_max,
                        heap_wpos, M_per_thr, N_per_thr, cta_n_offset,
                    )
                elif cutlass.const_expr(self.topk_strategy == "sortmerge"):
                    self._chunk_topk_sortmerge(
                        dist_buf, ptPcP_mn, heap_d, heap_i, heap_max,
                        chunk_d, chunk_i,
                        M_per_thr, N_per_thr, cta_n_offset,
                    )
                elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                    self._chunk_topk_sortmerge_packed(
                        dist_buf, ptPcP_mn,
                        heap_packed, chunk_packed,
                        M_per_thr, N_per_thr, cta_n_offset,
                    )
                else:
                    self._chunk_topk_insert(
                        tiled_mma, dist_buf, ptPcP_mn,
                        heap_d, heap_i, heap_max,
                        M_per_thr, N_per_thr, cta_n_offset,
                    )

            # Insert path: cooperative replicated heaps -- no merge needed.
            # Perthread/sortmerge: each thread holds a SORTED top-K of its
            # slice; butterfly-merge across threads-in-row before write.
            if cutlass.const_expr(self.topk_strategy == "perthread"):
                self._warp_merge_topk_perthread(
                    tiled_mma, heap_d, heap_i, M_per_thr,
                )
            elif cutlass.const_expr(self.topk_strategy == "maxtree"):
                self._warp_merge_topk_maxtree(
                    tiled_mma, heap_d, heap_i, M_per_thr,
                )
            elif cutlass.const_expr(self.topk_strategy == "sortmerge"):
                self._warp_merge_topk_sortmerge(
                    tiled_mma, heap_d, heap_i, M_per_thr,
                )
            elif cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                self._warp_merge_topk_sortmerge_packed(
                    tiled_mma, heap_packed, M_per_thr,
                )

            if cutlass.const_expr(self.topk_strategy == "sortmerge_packed"):
                heap_packed_f32 = cute.recast_tensor(
                    heap_packed, cutlass.Float32
                )
                heap_packed_i32 = cute.recast_tensor(
                    heap_packed, cutlass.Int32
                )
                if ptPcP[0][1] == 0:
                    for i in cutlass.range_constexpr(M_per_thr):
                        m_local = ptPcP_mn[(i, 0)][0]
                        m_global = m_local + cta_m_offset
                        if m_global < N_total:
                            for kk in cutlass.range_constexpr(K_PAD):
                                mOutI_nk[(m_global, kk)] = heap_packed_i32[
                                    (2 * i, kk)
                                ]
            else:
                if ptPcP[0][1] == 0:
                    if cutlass.const_expr(
                        self.topk_strategy in ("insert", "maxtree")
                    ):
                        self._sort_topk_rows(heap_d, heap_i, M_per_thr, K_PAD)
                    for i in cutlass.range_constexpr(M_per_thr):
                        m_local = ptPcP_mn[(i, 0)][0]
                        m_global = m_local + cta_m_offset
                        if m_global < N_total:
                            for kk in cutlass.range_constexpr(K_PAD):
                                mOutI_nk[(m_global, kk)] = heap_i[(i, kk)]
        return

    @cute.kernel
    def kernel_ws4(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_md: cute.Tensor,
        mCsq_m: cute.Tensor,
        mOutI_nk: cute.Tensor,    # (N, K_PAD) int32  — indices only
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        """4-warp-group warp specialisation: load + 2 GEMM WGs + topK.

        Optimization B (WS4): doubles the WGMMA throughput by running
        TWO GEMM warpgroups on different SM sub-partitions. Each GEMM
        WG processes alternate chunks via its own
        c_pipeline / dist_pipeline. The single TopK WG consumes from
        BOTH dist pipelines in chunk order (alternating).

          WG 0 (load) : TMA loads X (once) + C; routes C to sC_a or
                        sC_b based on chunk parity.
          WG 1 (gemm_a): handles EVEN chunks (0, 2, 4, ...) using
                         c_pipeline_a / dist_pipeline_a.
          WG 2 (gemm_b): handles ODD chunks (1, 3, 5, ...) using
                         c_pipeline_b / dist_pipeline_b.
          WG 3 (topk) : consumes chunks IN ORDER alternating between
                        dist_pipeline_a (even) and dist_pipeline_b
                        (odd). Same Mode-H sWorstD logic as WS3.

        Why 2 GEMM WGs ~= 2x throughput: Hopper SM has 4 sub-partitions
        each with its own WGMMA pipe. A single warpgroup occupies one
        sub-partition's WGMMA pipe at full rate. Adding a 2nd GEMM WG
        engages a second sub-partition's pipe, doubling the issue
        bandwidth.

        Costs:
          * 2x SMEM for sC and sDist (mitigated by halving c_stage,
            keeping dist_stage=2).
          * Register pressure: load(24) + gemm_a(192) + gemm_b(192) +
            topk(96) = 504 / 512 budget per thread. Tight; may need
            tuning per tile.
          * Tile-size gate: BM*BN <= 128*128 (acc must fit 192 reg).
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        BM = self.tile_shape_mnk[0]
        BN = self.tile_shape_mnk[1]
        N_total = mX_nd.shape[0]
        M_total = mC_md.shape[0]
        cta_m_offset = bidx * BM
        K_PAD = self.k_pad
        num_c_tiles = (M_total + BN - 1) // BN

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # X pipeline: load WG -> BOTH GEMM WGs.
        # NOTE: x_stage=1 and X is never re-acquired, so consumer
        # count is only used by consumer_release accounting; we
        # match WS3 with num_consumer_warps for safety.
        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_consumer_warps,
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype, cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # C pipeline A: load WG -> GEMM-A only.
        c_producer_group_a = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group_a = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline_a = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group_a,
            consumer_group=c_consumer_group_a,
            tx_count=cute.size_in_bytes(
                self.c_dtype, cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # C pipeline B: load WG -> GEMM-B only. Separate barrier slot.
        c_producer_group_b = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group_b = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline_b = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_b_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group_b,
            consumer_group=c_consumer_group_b,
            tx_count=cute.size_in_bytes(
                self.c_dtype, cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # Dist pipelines: GEMM-A -> TopK and GEMM-B -> TopK.
        dist_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads_per_warp_group
        )
        dist_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads_per_warp_group
        )
        dist_pipeline_a = pipeline.PipelineAsync.create(
            barrier_storage=storage.dist_pipeline_array_ptr.data_ptr(),
            num_stages=self.dist_stage,
            producer_group=dist_producer_group,
            consumer_group=dist_consumer_group,
        )
        dist_pipeline_b = pipeline.PipelineAsync.create(
            barrier_storage=storage.dist_pipeline_b_array_ptr.data_ptr(),
            num_stages=self.dist_stage,
            producer_group=dist_producer_group,
            consumer_group=dist_consumer_group,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC_a = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        sC_b = storage.sC_b.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        # sDist (a, b) staged layout same as WS3.
        sDist_all_a = storage.sDist.get_tensor(
            cute.make_layout(
                (BM, BN, self.dist_stage),
                stride=(
                    self.dist_smem_row_stride,
                    1,
                    BM * self.dist_smem_row_stride,
                ),
            )
        )
        sDist_all_b = storage.sDist_b.get_tensor(
            cute.make_layout(
                (BM, BN, self.dist_stage),
                stride=(
                    self.dist_smem_row_stride,
                    1,
                    BM * self.dist_smem_row_stride,
                ),
            )
        )
        sChunkMin_all_a = storage.sChunkMin.get_tensor(
            cute.make_layout((BM, self.dist_stage), stride=(1, BM))
        )
        sChunkMin_all_b = storage.sChunkMin_b.get_tensor(
            cute.make_layout((BM, self.dist_stage), stride=(1, BM))
        )
        sWorstD = storage.sWorstD.get_tensor(cute.make_layout(BM))

        if tidx < BM:
            sWorstD[tidx] = cutlass.Float32(3.4e38)

        gC_md = cute.local_tile(
            mC_md, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), (None, 0),
        )
        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x, 0, cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(
                cute.local_tile(
                    mX_nd, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),
        )
        tma_cS_a, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC_a, 0, 2),
            cute.group_modes(gC_md, 0, 2),
        )
        tma_cS_b, _ = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC_b, 0, 2),
            cute.group_modes(gC_md, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ---------- WG 0: Load (TMA) ----------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.x_stage
            )
            c_producer_state_a = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_stage
            )
            c_producer_state_b = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_stage
            )
            if warp_idx == 0:
                # Load X once.
                x_pipeline.producer_acquire(x_producer_state)
                cute.copy(
                    tma_atom_x,
                    tma_xG[(None, bidx)],
                    tma_xS[(None, x_producer_state.index)],
                    tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
                )
                x_pipeline.producer_commit(x_producer_state)
                x_producer_state.advance()

                # Alternate C tile destinations: even -> sC_a, odd -> sC_b.
                # GEMM-A consumes from sC_a, GEMM-B consumes from sC_b.
                for c_idx in cutlass.range(num_c_tiles, unroll=1):
                    if c_idx % 2 == 0:
                        c_pipeline_a.producer_acquire(c_producer_state_a)
                        cute.copy(
                            tma_atom_c,
                            tma_cG[(None, c_idx)],
                            tma_cS_a[(None, c_producer_state_a.index)],
                            tma_bar_ptr=c_pipeline_a.producer_get_barrier(
                                c_producer_state_a
                            ),
                        )
                        c_pipeline_a.producer_commit(c_producer_state_a)
                        c_producer_state_a.advance()
                    else:
                        c_pipeline_b.producer_acquire(c_producer_state_b)
                        cute.copy(
                            tma_atom_c,
                            tma_cG[(None, c_idx)],
                            tma_cS_b[(None, c_producer_state_b.index)],
                            tma_bar_ptr=c_pipeline_b.producer_get_barrier(
                                c_producer_state_b
                            ),
                        )
                        c_pipeline_b.producer_commit(c_producer_state_b)
                        c_producer_state_b.advance()

        # ---------- WG 1: GEMM-A (even chunks) ----------
        elif warp_group_idx == self.gemm_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            consumer_tidx = (
                tidx
                - self.gemm_warp_group_id * self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(consumer_tidx)
            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC_a)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)

            cP = cute.make_identity_tensor((BM, BN))
            ptPcP = thr_mma.partition_C(cP)
            gC_fake = cute.make_identity_tensor((BM, BN))
            tCgC_fake = thr_mma.partition_C(gC_fake)
            acc_shape = tCgC_fake.shape
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stage
            )
            x_pipeline.consumer_wait(x_consumer_state)

            acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
            acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
            )
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])

            # x_sq dropped; signed score = c_sq[m] − 2·cross.

            c_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            c_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            dist_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dist_stage
            )
            num_k_blocks = cute.size(tCrX, mode=[2])

            # GEMM-A handles even chunks: 0, 2, 4, ...
            my_num_chunks_a = (num_c_tiles + 1) // 2

            for my_idx in cutlass.range(my_num_chunks_a, unroll=1):
                c_tile_idx = 2 * my_idx
                c_pipeline_a.consumer_wait(c_consumer_read_state)

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma, acc,
                        tCrX[(None, None, k_block_idx, 0)],
                        tCrC[(None, None, k_block_idx,
                              c_consumer_read_state.index)],
                        acc,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                c_pipeline_a.consumer_release(c_consumer_release_state)
                c_consumer_read_state.advance()
                c_consumer_release_state.advance()

                cta_n_offset = c_tile_idx * BN
                cs = cute.make_rmem_tensor(
                    cute.make_layout(N_per_thr), cutlass.Float32
                )
                for j in cutlass.range_constexpr(N_per_thr):
                    n_local = ptPcP_mn[(0, j)][1]
                    n_global = n_local + cta_n_offset
                    if n_global < M_total:
                        cs[j] = mCsq_m[n_global]
                    else:
                        cs[j] = cutlass.Float32(3.4e38)

                dist_pipeline_a.producer_acquire(dist_producer_state)
                sDist_stage = sDist_all_a[
                    (None, None, dist_producer_state.index)
                ]
                sChunkMin_stage = sChunkMin_all_a[
                    (None, dist_producer_state.index)
                ]

                self._ws3_modeH_chunk_write(
                    acc_mn, sDist_stage, sChunkMin_stage,
                    sWorstD, cs, ptPcP_mn, tiled_mma,
                    M_per_thr, N_per_thr,
                )

                cute.arch.fence_proxy("async.shared", space="cta")
                dist_pipeline_a.producer_commit(dist_producer_state)
                dist_producer_state.advance()

        # ---------- WG 2: GEMM-B (odd chunks) ----------
        elif warp_group_idx == self.gemm_b_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            consumer_tidx = (
                tidx
                - self.gemm_b_warp_group_id * self.num_threads_per_warp_group
            )
            thr_mma = tiled_mma.get_slice(consumer_tidx)
            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC_b)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)

            cP = cute.make_identity_tensor((BM, BN))
            ptPcP = thr_mma.partition_C(cP)
            gC_fake = cute.make_identity_tensor((BM, BN))
            tCgC_fake = thr_mma.partition_C(gC_fake)
            acc_shape = tCgC_fake.shape
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stage
            )
            x_pipeline.consumer_wait(x_consumer_state)

            acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
            acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
            )
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])

            # x_sq dropped; signed score = c_sq[m] − 2·cross.

            c_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            c_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            dist_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dist_stage
            )
            num_k_blocks = cute.size(tCrX, mode=[2])

            # GEMM-B handles odd chunks: 1, 3, 5, ...
            my_num_chunks_b = num_c_tiles // 2

            for my_idx in cutlass.range(my_num_chunks_b, unroll=1):
                c_tile_idx = 2 * my_idx + 1
                c_pipeline_b.consumer_wait(c_consumer_read_state)

                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                cute.nvgpu.warpgroup.fence()
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    cute.gemm(
                        tiled_mma, acc,
                        tCrX[(None, None, k_block_idx, 0)],
                        tCrC[(None, None, k_block_idx,
                              c_consumer_read_state.index)],
                        acc,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                c_pipeline_b.consumer_release(c_consumer_release_state)
                c_consumer_read_state.advance()
                c_consumer_release_state.advance()

                cta_n_offset = c_tile_idx * BN
                cs = cute.make_rmem_tensor(
                    cute.make_layout(N_per_thr), cutlass.Float32
                )
                for j in cutlass.range_constexpr(N_per_thr):
                    n_local = ptPcP_mn[(0, j)][1]
                    n_global = n_local + cta_n_offset
                    if n_global < M_total:
                        cs[j] = mCsq_m[n_global]
                    else:
                        cs[j] = cutlass.Float32(3.4e38)

                dist_pipeline_b.producer_acquire(dist_producer_state)
                sDist_stage = sDist_all_b[
                    (None, None, dist_producer_state.index)
                ]
                sChunkMin_stage = sChunkMin_all_b[
                    (None, dist_producer_state.index)
                ]

                self._ws3_modeH_chunk_write(
                    acc_mn, sDist_stage, sChunkMin_stage,
                    sWorstD, cs, ptPcP_mn, tiled_mma,
                    M_per_thr, N_per_thr,
                )

                cute.arch.fence_proxy("async.shared", space="cta")
                dist_pipeline_b.producer_commit(dist_producer_state)
                dist_producer_state.advance()

        # ---------- WG 3: TopK ----------
        elif warp_group_idx == self.topk_warp_group_id:
            # WS4 has 4 WGs. Default reg quota is 64K/(4*128) = 128
            # reg/thread. We decrease to 96 so the 2 GEMM WGs can
            # claim 192 each (24+192+192+96 = 504 / 512 budget).
            cute.arch.setmaxregister_decrease(self.num_regs_topk)
            topk_tidx = tidx - self.topk_warp_group_id * self.num_threads_per_warp_group

            heap_d = cute.make_rmem_tensor(
                cute.make_layout((1, K_PAD)), cutlass.Float32
            )
            heap_i = cute.make_rmem_tensor(
                cute.make_layout((1, K_PAD)), cutlass.Int32
            )
            for k in cutlass.range_constexpr(K_PAD):
                heap_d[(0, k)] = cutlass.Float32(3.4e38)
                heap_i[(0, k)] = cutlass.Int32(-1)
            # smem_maxtree: persistent (worst_d, worst_pos) for the owned
            # row (unused by smem_perthread; compiles away).
            heap_max = cute.make_rmem_tensor(
                cute.make_layout(1), cutlass.Float32
            )
            heap_wpos = cute.make_rmem_tensor(
                cute.make_layout(1), cutlass.Int32
            )
            heap_max[0] = cutlass.Float32(3.4e38)
            heap_wpos[0] = cutlass.Int32(0)

            ROWS_OWNED = max(1, BM // self.num_threads_per_warp_group)

            dist_consumer_state_a = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dist_stage
            )
            dist_consumer_state_b = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dist_stage
            )

            # Consume chunks IN ORDER, alternating between
            # dist_pipeline_a (even chunks) and dist_pipeline_b (odd).
            for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
                cta_n_offset = c_tile_idx * BN
                if c_tile_idx % 2 == 0:
                    dist_pipeline_a.consumer_wait(dist_consumer_state_a)
                    sDist_stage = sDist_all_a[
                        (None, None, dist_consumer_state_a.index)
                    ]
                    sChunkMin_stage = sChunkMin_all_a[
                        (None, dist_consumer_state_a.index)
                    ]
                    if cutlass.const_expr(self.topk_strategy == "smem_maxtree"):
                        self._chunk_topk_smem_maxtree(
                            sDist_stage, sChunkMin_stage, heap_d, heap_i,
                            heap_max, heap_wpos, BM, BN, ROWS_OWNED,
                            cta_m_offset, cta_n_offset, N_total, M_total,
                            topk_tidx, sWorstD,
                        )
                    else:
                        self._chunk_topk_smem_perthread_with_chunkmin(
                            sDist_stage, sChunkMin_stage, heap_d, heap_i,
                            BM, BN, ROWS_OWNED,
                            cta_m_offset, cta_n_offset, N_total, M_total,
                            topk_tidx, sWorstD,
                        )
                    dist_pipeline_a.consumer_release(dist_consumer_state_a)
                    dist_consumer_state_a.advance()
                else:
                    dist_pipeline_b.consumer_wait(dist_consumer_state_b)
                    sDist_stage = sDist_all_b[
                        (None, None, dist_consumer_state_b.index)
                    ]
                    sChunkMin_stage = sChunkMin_all_b[
                        (None, dist_consumer_state_b.index)
                    ]
                    if cutlass.const_expr(self.topk_strategy == "smem_maxtree"):
                        self._chunk_topk_smem_maxtree(
                            sDist_stage, sChunkMin_stage, heap_d, heap_i,
                            heap_max, heap_wpos, BM, BN, ROWS_OWNED,
                            cta_m_offset, cta_n_offset, N_total, M_total,
                            topk_tidx, sWorstD,
                        )
                    else:
                        self._chunk_topk_smem_perthread_with_chunkmin(
                            sDist_stage, sChunkMin_stage, heap_d, heap_i,
                            BM, BN, ROWS_OWNED,
                            cta_m_offset, cta_n_offset, N_total, M_total,
                            topk_tidx, sWorstD,
                        )
                    dist_pipeline_b.consumer_release(dist_consumer_state_b)
                    dist_consumer_state_b.advance()

            # smem_maxtree leaves the heap UNSORTED -> selection-sort each
            # owned row before the write (smem_perthread is already
            # sorted by its bubble insert).
            if cutlass.const_expr(self.topk_strategy == "smem_maxtree"):
                self._sort_topk_rows(heap_d, heap_i, ROWS_OWNED, K_PAD)
            # Epilogue: write top-K to global. One thread per row.
            for r in cutlass.range_constexpr(ROWS_OWNED):
                my_row = topk_tidx + r * self.num_threads_per_warp_group
                if my_row < BM:
                    m_global = my_row + cta_m_offset
                    if m_global < N_total:
                        for kk in cutlass.range_constexpr(K_PAD):
                            mOutI_nk[(m_global, kk)] = heap_i[(r, kk)]
        return

    @cute.kernel
    def kernel_ws3(
        self,
        tma_atom_x: cute.CopyAtom,
        mX_nd: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_md: cute.Tensor,
        mCsq_m: cute.Tensor,
        mOutI_nk: cute.Tensor,    # (N, K_PAD) int32  — indices only
        tiled_mma: cute.TiledMma,
        x_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
    ):
        """3-warp-group warp specialisation: load + GEMM + topK pipelined.

        Architecture (mirrors the Hopper FMHA-3 / DeepSeek-V3 attention
        WS3 pattern):

          WG 0 (load) : TMA loads X (once) and C (per chunk) into sX/sC.
          WG 1 (GEMM) : WGMMA-accumulates into a register acc, post-
                        processes into ``dist = ||x||^2 + ||c||^2 - 2*acc``,
                        and stages the BM x BN dist tile into sDist[stage]
                        in row-major. Signals dist_pipeline.producer.
          WG 2 (topK) : Waits on dist_pipeline.consumer, then runs the
                        ``smem_perthread`` bubble-insert top-K (one
                        thread per row reads BN cols from sDist[stage]).
                        Epilogue: writes the K-wide sorted heap to
                        global.

        Why this is a win over WS2 (single consumer WG doing GEMM+topK):

          * GEMM and topK previously serialised inside the consumer WG
            (~7.2ms wall at K=24 = ~4ms GEMM + ~3.5ms topK on H100 with
            the smem_perthread strategy at BM=64 BN=64). Decoupling
            them lets the GEMM WG keep WGMMA pipelines saturated while
            the topK WG runs the bubble inserts in parallel: the
            two-stage dist ring buffer (``dist_stage=2``) lets chunk
            ``c`` GEMM run concurrently with chunk ``c-1`` topK.
            Wall ≈ max(t_gemm, t_topk).

          * The topK WG holds the per-thread heap entirely in its own
            register file (240-reg bump), so the GEMM WG's WGMMA acc
            registers never have to coexist with the K_PAD-wide heap
            -- this is what removes the K_INTERNAL=32 spill from the
            sortmerge path in the WS2 kernel. (Not directly relevant
            here since WS3 mandates ``smem_perthread``, but it's the
            same architectural lever.)

        SMEM layout & sizing:

          sDist is sized BM x (BN+1) x dist_stage fp32. The +1 padding
          on BN breaks the 32-way bank conflict on the topK reads (see
          ``dist_smem_row_stride`` rationale in __init__). For
          BM=64 BN=64 dist_stage=2 that's 2 * 64 * 65 * 4 = 33 KB --
          fits comfortably alongside sX / sC in SMEM.

        Synchronization:

          x_pipeline (TMA): producer = load WG (1 thread), consumer =
            GEMM WG (4 warps, x is consumed by WGMMA only).
          c_pipeline (TMA): producer = load WG (1 thread), consumer =
            GEMM WG (4 warps, c is consumed by WGMMA only).
          dist_pipeline (PipelineAsync, mbarrier-based): producer =
            GEMM WG (128 threads), consumer = topK WG (128 threads).
            num_stages = self.dist_stage.

        Constraints (enforced in __init__):

          * topk_strategy == "smem_perthread" (the only strategy that
            doesn't depend on the WGMMA TV layout for its inputs).
          * mma_warp_groups == 1 (BM=64, or BM=128 with BN=64 -> 1
            atom). This caps total CTA size at 3*128 = 384 threads.
        """
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_x)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        BM = self.tile_shape_mnk[0]
        BN = self.tile_shape_mnk[1]
        N_total = mX_nd.shape[0]
        M_total = mC_md.shape[0]
        cta_m_offset = bidx * BM
        K_PAD = self.k_pad
        num_c_tiles = (M_total + BN - 1) // BN

        # --- Common setup ---------------------------------------------------
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # X pipeline: TMA, load WG -> GEMM WG only.
        x_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        x_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        x_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.x_pipeline_array_ptr.data_ptr(),
            num_stages=self.x_stage,
            producer_group=x_producer_group,
            consumer_group=x_consumer_group,
            tx_count=cute.size_in_bytes(
                self.x_dtype, cute.slice_(x_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        c_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_consumer_warps
        )
        c_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.c_pipeline_array_ptr.data_ptr(),
            num_stages=self.c_stage,
            producer_group=c_producer_group,
            consumer_group=c_consumer_group,
            tx_count=cute.size_in_bytes(
                self.c_dtype, cute.slice_(c_smem_layout_staged, (None, None, 0)),
            ),
            defer_sync=True,
        )

        # Dist pipeline (NEW): mbarrier-based, GEMM WG -> topK WG.
        # Both groups are full warpgroups (128 threads each).
        dist_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads_per_warp_group
        )
        dist_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_threads_per_warp_group
        )
        dist_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.dist_pipeline_array_ptr.data_ptr(),
            num_stages=self.dist_stage,
            producer_group=dist_producer_group,
            consumer_group=dist_consumer_group,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sX = storage.sX.get_tensor(
            x_smem_layout_staged.outer, swizzle=x_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )

        # Multi-stage sDist: K_SW128 swizzled layout, (BM, BN,
        # dist_stage). Replaces the BN+1 padded row-major layout that
        # only fixed the consumer-side conflict; the swizzle eliminates
        # both producer (WGMMA store) and consumer (per-thread row
        # stream) bank conflicts (ncu measured 73.86% of shared store
        # wavefronts conflict with padded layout, est. -22% latency).
        # Multi-stage sDist: layout (BM, BN, dist_stage), row-major per
        # stage with the BN+1 bank-conflict padding. (See attempt at
        # K_SW128 swizzle in __init__ comments -- it reduced bank
        # conflicts but added too many XOR ops.)
        sDist_all = storage.sDist.get_tensor(
            cute.make_layout(
                (BM, BN, self.dist_stage),
                stride=(
                    self.dist_smem_row_stride,
                    1,
                    BM * self.dist_smem_row_stride,
                ),
            )
        )

        # Per-row chunk_min staging buffer (BM, dist_stage) fp32. Filled
        # by the GEMM WG (one fp32 per row per chunk = min over BN cols
        # of the dist tile), consumed by the TopK WG as the pre-loop
        # prune gate.
        sChunkMin_all = storage.sChunkMin.get_tensor(
            cute.make_layout((BM, self.dist_stage), stride=(1, BM))
        )

        # Mode H: cross-WG worst_d feedback. (BM,) fp32. TopK WG
        # writes sWorstD[my_row] = topk_d[K-1] after each consumed
        # chunk; GEMM WG reads sWorstD[m_local] BEFORE writing each
        # row's dist values and SKIPS the 16 STS + d-compute for that
        # row when row_chunk_min >= sWorstD[m_local]. Stale-read
        # safety: heap top is monotonically non-increasing, so any
        # read by GEMM is a CONSERVATIVE upper bound on current
        # worst_d (might process a chunk that could've been pruned,
        # never the reverse). Single fp32 LDS/STS are atomic.
        sWorstD = storage.sWorstD.get_tensor(cute.make_layout(BM))

        # Mode H: pre-init sWorstD to +inf BEFORE the WG branching so
        # the GEMM WG's first-chunk read sees a permissive bound (no
        # cross-WG ordering needed since pipeline_init_wait below is
        # a CTA-wide barrier). All ``threads_per_cta`` threads
        # cooperate; thread t inits sWorstD[t] for t < BM.
        if tidx < BM:
            sWorstD[tidx] = cutlass.Float32(3.4e38)

        gC_md = cute.local_tile(
            mC_md, (self.tile_shape_mnk[1], self.tile_shape_mnk[2]), (None, 0),
        )
        tma_xS, tma_xG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_x, 0, cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(
                cute.local_tile(
                    mX_nd, (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
                    (None, 0),
                ),
                0, 2,
            ),
        )
        tma_cS, tma_cG = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(gC_md, 0, 2),
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # --- WG 0: Load (TMA) -----------------------------------------------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            x_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.x_stage
            )
            c_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.c_stage
            )
            if warp_idx == 0:
                x_pipeline.producer_acquire(x_producer_state)
                cute.copy(
                    tma_atom_x,
                    tma_xG[(None, bidx)],
                    tma_xS[(None, x_producer_state.index)],
                    tma_bar_ptr=x_pipeline.producer_get_barrier(x_producer_state),
                )
                x_pipeline.producer_commit(x_producer_state)
                x_producer_state.advance()

                for c_idx in cutlass.range(num_c_tiles, unroll=1):
                    c_pipeline.producer_acquire(c_producer_state)
                    cute.copy(
                        tma_atom_c,
                        tma_cG[(None, c_producer_state.count)],
                        tma_cS[(None, c_producer_state.index)],
                        tma_bar_ptr=c_pipeline.producer_get_barrier(c_producer_state),
                    )
                    c_pipeline.producer_commit(c_producer_state)
                    c_producer_state.advance()

        # --- WG 1: GEMM (WGMMA + dist staging) ------------------------------
        elif warp_group_idx == self.gemm_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)
            consumer_tidx = tidx - self.num_threads_per_warp_group
            thr_mma = tiled_mma.get_slice(consumer_tidx)

            tCsX = thr_mma.partition_A(sX)
            tCsC = thr_mma.partition_B(sC)
            tCrX = tiled_mma.make_fragment_A(tCsX)
            tCrC = tiled_mma.make_fragment_B(tCsC)

            cP = cute.make_identity_tensor((BM, BN))
            ptPcP = thr_mma.partition_C(cP)
            gC_fake = cute.make_identity_tensor((BM, BN))
            tCgC_fake = thr_mma.partition_C(gC_fake)
            acc_shape = tCgC_fake.shape
            acc = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            x_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.x_stage
            )
            x_pipeline.consumer_wait(x_consumer_state)

            acc_mn_layout = self._layout_acc_mn(tiled_mma, acc.layout)
            acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
            ptPcP_mn = cute.make_tensor(
                ptPcP.iterator, self._layout_acc_mn(tiled_mma, ptPcP.layout)
            )
            M_per_thr = cute.size(acc_mn, mode=[0])
            N_per_thr = cute.size(acc_mn, mode=[1])

            # x_sq dropped; signed score = c_sq[m] − 2·cross.

            c_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            c_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.c_stage
            )
            dist_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.dist_stage
            )
            num_k_blocks = cute.size(tCrX, mode=[2])

            # =====================================================
            # NOTE on Optimization A (WGMMA pipelining):
            # =====================================================
            # We tried 3 variants of single-WG WGMMA pipelining:
            #   A1) 2 acc tensors (acc_0/acc_1) + parity branch on
            #       c_tile_idx%2 selecting the cute.gemm target.
            #       Result: 38-164% SLOWER. The if-else around
            #       cute.gemm() doubled the static WGMMA issue
            #       path and the register allocator spilled.
            #   A2) Snapshot copy: single acc + acc_snap, pre-issue
            #       chunk N+1's WGMMA into acc, copy acc->acc_snap
            #       AFTER wait_group, process acc_snap.
            #       Result: -57 to -103% SLOWER. Same root cause:
            #       acc_snap allocation alone caused catastrophic
            #       register spill (kernel ran 100x slower).
            #   A3) Allocate acc_snap but never use it.
            #       Result: STILL 100x SLOWER. Confirms allocation
            #       alone breaks register layout.
            # CuTeDSL's `cute.make_rmem_tensor` doesn't compose well
            # for a 2nd acc-shaped tensor in the GEMM WG -- the
            # compiler can't DCE unused or alternated WGMMA targets.
            # The CUTLASS-canonical approach is WG-level pingpong
            # (e.g. FA3): two GEMM WGs alternating chunks via
            # mbarriers. That's Optimization B (WS4) below.
            if cutlass.const_expr(False):  # disabled, kept for docs
                pass
            else:
                # =====================================================
                # Single-acc path (default).
                # =====================================================
                for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
                    c_pipeline.consumer_wait(c_consumer_read_state)

                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        cute.gemm(
                            tiled_mma, acc,
                            tCrX[(None, None, k_block_idx, 0)],
                            tCrC[(None, None, k_block_idx, c_consumer_read_state.index)],
                            acc,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(0)

                    c_pipeline.consumer_release(c_consumer_release_state)
                    c_consumer_read_state.advance()
                    c_consumer_release_state.advance()

                    cta_n_offset = c_tile_idx * BN

                    cs = cute.make_rmem_tensor(
                        cute.make_layout(N_per_thr), cutlass.Float32
                    )
                    for j in cutlass.range_constexpr(N_per_thr):
                        n_local = ptPcP_mn[(0, j)][1]
                        n_global = n_local + cta_n_offset
                        if n_global < M_total:
                            cs[j] = mCsq_m[n_global]
                        else:
                            cs[j] = cutlass.Float32(3.4e38)

                    dist_pipeline.producer_acquire(dist_producer_state)
                    sDist_stage = sDist_all[(None, None, dist_producer_state.index)]
                    sChunkMin_stage = sChunkMin_all[
                        (None, dist_producer_state.index)
                    ]

                    self._ws3_modeH_chunk_write(
                        acc_mn, sDist_stage, sChunkMin_stage,
                        sWorstD, cs, ptPcP_mn, tiled_mma,
                        M_per_thr, N_per_thr,
                    )

                    cute.arch.fence_proxy("async.shared", space="cta")
                    dist_pipeline.producer_commit(dist_producer_state)
                    dist_producer_state.advance()

        # --- WG 2: TopK (smem_perthread bubble insert) ----------------------
        elif warp_group_idx == self.topk_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_topk)
            topk_tidx = tidx - 2 * self.num_threads_per_warp_group

            heap_d = cute.make_rmem_tensor(
                cute.make_layout((1, K_PAD)), cutlass.Float32
            )
            heap_i = cute.make_rmem_tensor(
                cute.make_layout((1, K_PAD)), cutlass.Int32
            )
            for k in cutlass.range_constexpr(K_PAD):
                heap_d[(0, k)] = cutlass.Float32(3.4e38)
                heap_i[(0, k)] = cutlass.Int32(-1)
            # smem_maxtree: persistent (worst_d, worst_pos) for the owned
            # row (unused by smem_perthread; compiles away).
            heap_max = cute.make_rmem_tensor(
                cute.make_layout(1), cutlass.Float32
            )
            heap_wpos = cute.make_rmem_tensor(
                cute.make_layout(1), cutlass.Int32
            )
            heap_max[0] = cutlass.Float32(3.4e38)
            heap_wpos[0] = cutlass.Int32(0)

            # ROWS_OWNED: smem_perthread normally derives this from BM /
            # 128. With WS3 the topK WG has its own 128 threads, so for
            # BM <= 128 we have rows_per_thr = 1 (with my_row =
            # topk_tidx, gated by < BM). sWorstD init to +inf was
            # done CTA-wide pre-branch (no extra sync needed).
            ROWS_OWNED = max(1, BM // self.num_threads_per_warp_group)

            dist_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.dist_stage
            )

            for c_tile_idx in cutlass.range(num_c_tiles, unroll=1):
                dist_pipeline.consumer_wait(dist_consumer_state)
                sDist_stage = sDist_all[(None, None, dist_consumer_state.index)]
                sChunkMin_stage = sChunkMin_all[
                    (None, dist_consumer_state.index)
                ]
                cta_n_offset = c_tile_idx * BN
                # Mode H: pass sWorstD into the helper so the worst_d
                # writeback happens INSIDE the chunk-min branch (only
                # when the heap actually changed, ~1% of chunks at
                # steady state). The GEMM WG tolerates stale reads
                # (heap top is monotonically non-increasing) so we
                # avoid the fence overhead -- the next chunk's
                # producer_commit + consumer_wait pair establishes
                # eventual visibility for free.
                if cutlass.const_expr(self.topk_strategy == "smem_maxtree"):
                    self._chunk_topk_smem_maxtree(
                        sDist_stage, sChunkMin_stage, heap_d, heap_i,
                        heap_max, heap_wpos, BM, BN, ROWS_OWNED,
                        cta_m_offset, cta_n_offset, N_total, M_total,
                        topk_tidx, sWorstD,
                    )
                else:
                    self._chunk_topk_smem_perthread_with_chunkmin(
                        sDist_stage, sChunkMin_stage, heap_d, heap_i,
                        BM, BN, ROWS_OWNED,
                        cta_m_offset, cta_n_offset, N_total, M_total,
                        topk_tidx, sWorstD,
                    )
                dist_pipeline.consumer_release(dist_consumer_state)
                dist_consumer_state.advance()

            # smem_maxtree leaves the heap UNSORTED -> selection-sort each
            # owned row before the write (smem_perthread is already
            # sorted by its bubble insert).
            if cutlass.const_expr(self.topk_strategy == "smem_maxtree"):
                self._sort_topk_rows(heap_d, heap_i, ROWS_OWNED, K_PAD)
            # Epilogue: write top-K to global. One thread per row.
            for r in cutlass.range_constexpr(ROWS_OWNED):
                my_row = topk_tidx + r * self.num_threads_per_warp_group
                if my_row < BM:
                    m_global = my_row + cta_m_offset
                    if m_global < N_total:
                        for kk in cutlass.range_constexpr(K_PAD):
                            mOutI_nk[(m_global, kk)] = heap_i[(r, kk)]
        return

    # ----------------------------------------------------------------------
    # Layout helpers (lifted from FMHA, same as kmeans kernel)
    # ----------------------------------------------------------------------

    @staticmethod
    @cute.jit
    def _layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for k, v in enumerate(ref):
            if cutlass.const_expr(v < thr):
                lt = cute.append(lt, src[k])
            else:
                ge = cute.append(ge, src[k])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @cute.jit
    def _layout_acc_mn(self, tiled_mma, acc_layout):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0], acc_layout[0], tiled_mma.tv_layout_C.stride[1]
        )
        V_M = separated[0]
        V_N = separated[1]
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc_layout[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc_layout[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc_layout[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc_layout[2])
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    @cute.jit
    def _reduction_target_n(self, tiled_mma):
        separated = self._layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

