"""CAGRA single-CTA graph-traversal search (CuteDSL, SM90).

The Triton kernel keeps its sorted candidate buffer + O(itopk^2) dedup in
*registers* -- ncu measured 121-255 regs/thread and 12-23% occupancy, with
the large-itopk regime spilling at the 255-reg ceiling. Triton exposes no
explicit shared memory, so that state cannot move off-chip. This CuteDSL
kernel puts the whole search state in SMEM and is the faster path once the
buffer is large (itopk>=128, or itopk>=64 at small out-degree): on SIFT1M
itopk=256 it is ~2.2x the Triton kernel at matched recall.

State (all SMEM):

* ``sBuf{D,I,F}`` -- the ``ITOPK`` candidate buffer (dist / id / expanded
  flag), kept **sorted by distance** at all times.
* ``sHash`` -- an open-addressed visited set that mirrors the buffer ids
  exactly (rebuilt from the buffer after every merge with atomic-CAS inserts,
  so no id is lost to a write race). It turns dedup into an O(1) membership
  test instead of an O(NEW*ITOPK) buffer scan.

Algorithm (per query CTA), mirroring the Triton/torch oracle so recall is
identical:

1. Seed the buffer with ``R`` random entry nodes; warp-team distances; sort
   the buffer once (the merge below keeps it sorted thereafter); build the
   visited set from the seeded buffer.
2. For ``MAX_ITERS`` iterations:
   a. pick the ``SEARCH_WIDTH`` closest *unexpanded* buffer entries (block
      arg-min), mark them expanded -> frontier. If the frontier is empty
      every reachable node has been expanded -> early-exit.
   b. gather each frontier node's ``GD`` neighbours (coalesced).
   c. drop neighbours already in the buffer via the visited set (O(HPROBE)
      probes/candidate, short-circuiting on the first empty/own slot).
   d. compute surviving neighbours' squared-L2 (warp-team: 32 lanes co-load
      one row -> coalesced 128-B transactions, butterfly-reduce the sum).
   e. **incremental merge**: the buffer is already sorted, so only the NEW
      candidates are sorted (cheap) and bitonic-*merged* into it
      (log2(PCOMB) passes), instead of a full bitonic *sort* of the pool
      (~(log2 PCOMB)^2 passes), then the visited set is rebuilt to mirror the
      new buffer.
3. Emit the ``k`` closest.

Two throughput levers, both validated on SIFT1M @ matched recall:
* incremental merge (vs full per-iter sort): itopk=256 L1TEX 69%->42%,
  DRAM 17%->29% of peak.
* atomic-CAS visited set (vs buffer scan): removes the O(NEW*ITOPK) dedup,
  lifting occupancy 65%->76% -> itopk=128 +33% QPS, itopk=256 +25% QPS.

A *forgetful* (lossy) hash was tried first and lost recall (evicted live
buffer ids => duplicates); mirroring the buffer with lossless CAS inserts
keeps recall bit-for-bit identical to the buffer scan.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.runtime as cute_rt
import cutlass.utils as utils
import cuda.bindings.driver as cuda

from flashlib.primitives.cagra.index import default_random_seeds
from flashlib.primitives.knn.triton._common import _next_pow2


_INF = 3.4e38


class HopperCagraSearch:
    """Single-CTA CAGRA graph-walk search kernel (SM90, SIMT + SMEM).

    All sizes are baked at compile time (one compile per distinct config);
    ``N`` / ``nq`` are read from the dynamic tensor shapes so a compiled
    kernel serves every dataset / batch size.

    Config:
        D: vector dim. GD: graph out-degree. ITOPK: buffer width (pow2).
        SW: search width (frontier nodes per iter). MAXIT: iterations.
        R: random seeds (<= ITOPK). K: output neighbours.
        PCOMB: merge pool width (pow2 >= ITOPK + SW*GD). HSIZE: visited-set
        capacity (pow2). threads: CTA width (multiple of 32). INCR: use the
        incremental merge (NEW pow2, <=ITOPK).
    """

    # Visited-set hash: multiplicative (Fibonacci) hash + linear probing with
    # lock-free atomic-CAS inserts (so no buffer id is ever lost to a write
    # race -- the table mirrors the buffer exactly, giving the same dedup as a
    # full buffer scan). HMULT is 0x9E3779B1 as a signed int32 (same low bits
    # under two's-complement wrap, which is all the hash uses).
    _HMULT = -1640531535
    _HPROBE = 12

    def __init__(self, *, D, GD, ITOPK, SW, MAXIT, R, K, PCOMB, HSIZE, threads,
                 TS=32, INCR=True):
        self.D = D
        self.GD = GD
        self.ITOPK = ITOPK
        self.SW = SW
        self.MAXIT = MAXIT
        self.R = R
        self.K = K
        self.PCOMB = PCOMB
        self.HSIZE = HSIZE
        self.threads = threads
        # team_size: threads that cooperate on one distance (cuVS's knob). A
        # team loads D/TS *contiguous* dims/thread as vectorized 128-bit loads
        # and butterfly-reduces; T/TS teams run concurrently, so a smaller TS
        # puts more candidate rows in flight (better DRAM latency hiding) as
        # long as TS*16B >= 128B coalescing (TS>=8 with float4). Requires
        # TS | 32 and D % TS == 0 (host guarantees both).
        assert 32 % TS == 0 and D % TS == 0
        self.TS = TS
        # Incremental merge: the buffer is kept sorted, so each iter only the
        # NEW candidates are sorted (cheap) and bitonic-*merged* into it
        # (log2(PCOMB) passes) instead of a full bitonic *sort* of the pool
        # (~(log2 PCOMB)^2 passes). Requires NEW a power of two and NEW<=ITOPK
        # (=> PCOMB==2*ITOPK); the host falls back to a full sort otherwise.
        self.INCR = bool(INCR)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor, mData: cute.Tensor, mGraph: cute.Tensor,
        mSeeds: cute.Tensor, mOutV: cute.Tensor, mOutI: cute.Tensor,
        stream: cuda.CUstream,
    ):
        D, ITOPK, SW, GD = self.D, self.ITOPK, self.SW, self.GD
        PCOMB, T, HSIZE = self.PCOMB, self.threads, self.HSIZE
        NEW = SW * GD
        NWARP = T // 32

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.MemRange[cutlass.Float32, D]
            sBufD: cute.struct.MemRange[cutlass.Float32, ITOPK]
            sBufI: cute.struct.MemRange[cutlass.Int32, ITOPK]
            sBufF: cute.struct.MemRange[cutlass.Int32, ITOPK]
            sHash: cute.struct.MemRange[cutlass.Int32, HSIZE]
            sFrontier: cute.struct.MemRange[cutlass.Int32, SW]
            sCandI: cute.struct.MemRange[cutlass.Int32, NEW]
            sCandD: cute.struct.MemRange[cutlass.Float32, NEW]
            sPoolD: cute.struct.MemRange[cutlass.Float32, PCOMB]
            sPoolI: cute.struct.MemRange[cutlass.Int32, PCOMB]
            sPoolF: cute.struct.MemRange[cutlass.Int32, PCOMB]
            sWarpV: cute.struct.MemRange[cutlass.Float32, NWARP]
            sWarpP: cute.struct.MemRange[cutlass.Int32, NWARP]
            sDone: cute.struct.MemRange[cutlass.Int32, 1]

        self.shared_storage = SharedStorage

        nq = mQ.shape[0]
        grid = (nq, 1, 1)
        self.kernel(mQ, mData, mGraph, mSeeds, mOutV, mOutI).launch(
            grid=grid, block=[self.threads, 1, 1], stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mData: cute.Tensor, mGraph: cute.Tensor,
        mSeeds: cute.Tensor, mOutV: cute.Tensor, mOutI: cute.Tensor,
    ):
        D, GD, ITOPK = self.D, self.GD, self.ITOPK
        SW, MAXIT, R, K = self.SW, self.MAXIT, self.R, self.K
        PCOMB, T, HSIZE = self.PCOMB, self.threads, self.HSIZE
        TS = self.TS
        NEW = SW * GD
        NWARP = T // 32
        NTEAM = T // TS
        VPT = D // TS
        HMASK = HSIZE - 1
        HMULT = self._HMULT
        HPROBE = self._HPROBE

        qid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        wid = tid // 32
        team = tid // TS
        tlane = tid % TS

        N = mData.shape[0]

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sQ = storage.sQ.get_tensor(cute.make_layout(D))
        sBufD = storage.sBufD.get_tensor(cute.make_layout(ITOPK))
        sBufI = storage.sBufI.get_tensor(cute.make_layout(ITOPK))
        sBufF = storage.sBufF.get_tensor(cute.make_layout(ITOPK))
        sHash = storage.sHash.get_tensor(cute.make_layout(HSIZE))
        sFrontier = storage.sFrontier.get_tensor(cute.make_layout(SW))
        sCandI = storage.sCandI.get_tensor(cute.make_layout(NEW))
        sCandD = storage.sCandD.get_tensor(cute.make_layout(NEW))
        sPoolD = storage.sPoolD.get_tensor(cute.make_layout(PCOMB))
        sPoolI = storage.sPoolI.get_tensor(cute.make_layout(PCOMB))
        sPoolF = storage.sPoolF.get_tensor(cute.make_layout(PCOMB))
        sWarpV = storage.sWarpV.get_tensor(cute.make_layout(NWARP))
        sWarpP = storage.sWarpP.get_tensor(cute.make_layout(NWARP))
        sDone = storage.sDone.get_tensor(cute.make_layout(1))

        # ── seed the buffer with R random entry nodes (parallel) ───────
        i = tid
        while i < ITOPK:
            if i < R:
                sBufI[i] = cutlass.Int32(mSeeds[qid, i])
            else:
                sBufI[i] = cutlass.Int32(-1)
            sBufD[i] = cutlass.Float32(_INF)
            sBufF[i] = cutlass.Int32(0)
            i += T
        cute.arch.sync_threads()

        # ── cache the query row in SMEM ────────────────────────────────
        i = tid
        while i < D:
            sQ[i] = mQ[qid, i]
            i += T
        cute.arch.sync_threads()

        # Each thread keeps its VPT-wide query chunk in registers, reused for
        # every distance (cuVS keeps the query resident too). Reading the query
        # from SMEM instead to save registers was measured ~2x slower at
        # D=960/itopk256 (the per-distance SMEM re-reads dominate), so the query
        # stays register-resident regardless of D. The team-distance body
        # (vectorized 128-bit team-load + butterfly reduce) is inlined at each
        # call site below -- CuteDSL forbids closures in device loops.
        qfrag = cute.make_fragment(VPT, cutlass.Float32)
        dfrag = cute.make_fragment(VPT, cutlass.Float32)
        for j in cutlass.range_constexpr(VPT):
            qfrag[j] = sQ[tlane * VPT + j]

        # ── seed distances (NTEAM teams compute NTEAM rows in parallel) ──
        # The butterfly runs for *all* lanes every iteration (outside the
        # cid>=0 guard): teams sharing a warp may diverge on cid, and a warp
        # shuffle needs all lanes -- gating it would deadlock. ITOPK % NTEAM
        # == 0 (both pow2), so every team takes the same iteration count.
        c = team
        while c < ITOPK:
            cid = sBufI[c]
            acc = cutlass.Float32(0.0)
            if cid >= 0:
                row = mData[(cid, None)]
                row2 = cute.make_tensor(
                    row.iterator, cute.make_layout((TS, VPT), stride=(VPT, 1)))
                cute.autovec_copy(row2[(tlane, None)], dfrag)
                for j in cutlass.range_constexpr(VPT):
                    diff = qfrag[j] - dfrag[j]
                    acc = acc + diff * diff
            off = TS // 2
            while off > 0:
                acc = acc + cute.arch.shuffle_sync_bfly(acc, off)
                off //= 2
            if cid >= 0 and tlane == 0:
                sBufD[c] = acc
            c += NTEAM
        cute.arch.sync_threads()

        # ── one-time sort of the seed buffer (incremental merge keeps it
        #    sorted thereafter, so pay this O((log ITOPK)^2) cost once) ───
        if self.INCR:
            kk = 2
            while kk <= ITOPK:
                jj = kk // 2
                while jj > 0:
                    i = tid
                    while i < ITOPK:
                        l = i ^ jj
                        if l > i:
                            up = (i & kk) == 0
                            di = sBufD[i]
                            dl = sBufD[l]
                            if (di > dl) == up:
                                sBufD[i] = dl
                                sBufD[l] = di
                                ti = sBufI[i]
                                sBufI[i] = sBufI[l]
                                sBufI[l] = ti
                                tf = sBufF[i]
                                sBufF[i] = sBufF[l]
                                sBufF[l] = tf
                        i += T
                    cute.arch.sync_threads()
                    jj //= 2
                kk *= 2

        # ── build the visited set to mirror the (seeded) buffer ────────
        # The hash holds exactly the current buffer ids, so dedup is an O(1)
        # membership test instead of the O(NEW*ITOPK) buffer scan, with the
        # same semantics (a candidate already in the buffer is dropped). It
        # is rebuilt from the buffer after every merge (below).
        i = tid
        while i < HSIZE:
            sHash[i] = cutlass.Int32(0)
            i += T
        cute.arch.sync_threads()
        i = tid
        while i < ITOPK:
            cid = sBufI[i]
            if cid >= 0:
                h = (cid * HMULT) & HMASK
                done = cutlass.Int32(0)
                p = cutlass.Int32(0)
                while p < HPROBE:
                    if done == 0:
                        slot = (h + p) & HMASK
                        old = cute.arch.atomic_cas(
                            sHash.iterator + slot, cmp=cutlass.Int32(0),
                            val=cid + 1, scope="cta")
                        if old == 0 or old == cid + 1:
                            done = cutlass.Int32(1)
                    p += 1
            i += T
        cute.arch.sync_threads()

        # ── main traversal (device loop; constexpr unroll would blow up
        #    code size at MAX_ITERS ~ O(itopk)) ──────────────────────────
        _it = cutlass.Int32(0)
        while _it < MAXIT:
            # (a) frontier: SW closest unexpanded entries (block arg-min) ─
            for f in cutlass.range_constexpr(SW):
                bv = cutlass.Float32(_INF)
                bp = cutlass.Int32(-1)
                i = tid
                while i < ITOPK:
                    v = sBufD[i] if sBufF[i] == 0 else cutlass.Float32(_INF)
                    if v < bv:
                        bv = v
                        bp = cutlass.Int32(i)
                    i += T
                off = 16
                while off > 0:
                    ov = cute.arch.shuffle_sync_bfly(bv, off)
                    op = cute.arch.shuffle_sync_bfly(bp, off)
                    if ov < bv:
                        bv = ov
                        bp = op
                    off //= 2
                if lane == 0:
                    sWarpV[wid] = bv
                    sWarpP[wid] = bp
                cute.arch.sync_threads()
                if tid == 0:
                    fv = sWarpV[0]
                    fp = sWarpP[0]
                    for w in cutlass.range_constexpr(1, NWARP):
                        if sWarpV[w] < fv:
                            fv = sWarpV[w]
                            fp = sWarpP[w]
                    if fp >= 0 and fv < cutlass.Float32(_INF):
                        sFrontier[f] = sBufI[fp]
                        sBufF[fp] = cutlass.Int32(1)
                    else:
                        sFrontier[f] = cutlass.Int32(-1)
                cute.arch.sync_threads()

            # early-exit: once every reachable buffer entry has been expanded
            # the frontier is empty and the remaining iters would just re-sort
            # an unchanging buffer. Jumping _it to MAXIT exits after this (now
            # no-op) iteration, so the bitonic merge runs at most once more.
            if tid == 0:
                anyf = cutlass.Int32(0)
                for f in cutlass.range_constexpr(SW):
                    if sFrontier[f] >= 0:
                        anyf = cutlass.Int32(1)
                sDone[0] = anyf
            cute.arch.sync_threads()
            if sDone[0] == 0:
                _it = cutlass.Int32(MAXIT)

            # (b) gather frontier neighbours into sCandI (parallel) ──────
            i = tid
            while i < NEW:
                f = i // GD
                jj = i % GD
                fid = sFrontier[f]
                if fid >= 0:
                    nb = cutlass.Int32(mGraph[fid, jj])
                    sCandI[i] = nb if (nb >= 0 and nb < N) else cutlass.Int32(-1)
                else:
                    sCandI[i] = cutlass.Int32(-1)
                i += T
            cute.arch.sync_threads()

            # (c) drop candidates already in the buffer via the visited set,
            #     which mirrors the buffer exactly -- same result as the old
            #     O(NEW*ITOPK) buffer scan that became the top L1 cost once the
            #     merge was incremental, but O(HPROBE) per candidate. The probe
            #     stops at the first empty/own slot, so it is ~1-2 reads.
            i = tid
            while i < NEW:
                cid = sCandI[i]
                if cid >= 0:
                    h = (cid * HMULT) & HMASK
                    vis = cutlass.Int32(0)
                    stop = cutlass.Int32(0)
                    p = cutlass.Int32(0)
                    while p < HPROBE:
                        if stop == 0:
                            hv = sHash[(h + p) & HMASK]
                            if hv == cid + 1:
                                vis = cutlass.Int32(1)
                                stop = cutlass.Int32(1)
                            elif hv == 0:
                                stop = cutlass.Int32(1)  # empty => not present
                        p += 1
                    if vis == 1:
                        sCandI[i] = cutlass.Int32(-1)
                i += T
            cute.arch.sync_threads()

            # (d) candidate distances (NTEAM teams, vectorized team-load). As
            #     in the seed loop the butterfly is unconditional so all warp
            #     lanes participate; NEW is a multiple of teams-per-warp (pow2
            #     graph degree), so teams stay in lockstep.
            c = team
            while c < NEW:
                cid = sCandI[c]
                acc = cutlass.Float32(0.0)
                if cid >= 0:
                    row = mData[(cid, None)]
                    row2 = cute.make_tensor(
                        row.iterator, cute.make_layout((TS, VPT), stride=(VPT, 1)))
                    cute.autovec_copy(row2[(tlane, None)], dfrag)
                    for j in cutlass.range_constexpr(VPT):
                        diff = qfrag[j] - dfrag[j]
                        acc = acc + diff * diff
                off = TS // 2
                while off > 0:
                    acc = acc + cute.arch.shuffle_sync_bfly(acc, off)
                    off //= 2
                if tlane == 0:
                    if cid >= 0:
                        sCandD[c] = acc
                    else:
                        sCandD[c] = cutlass.Float32(_INF)
                c += NTEAM
            cute.arch.sync_threads()

            # (e) merge the new candidates into the sorted buffer, keep the
            #     ITOPK closest. Two equivalent results, very different cost:
            if self.INCR:
                # Incremental: buffer is already sorted asc. Sort just the NEW
                # candidates (small), lay the pool out as a bitonic sequence
                # (buffer asc | INF pad | candidates desc) and run a single
                # bitonic *merge* (log2(PCOMB) passes) -- vs the full sort's
                # ~(log2 PCOMB)^2 passes that dominated L1/barrier traffic.
                # Sort the NEW candidates (bitonic), register-resident: thread t
                # holds candidate t in registers and every stride jj<32 is a
                # race-free warp shuffle (no SMEM, no barrier). Only the rare
                # cross-warp stride jj>=32 round-trips SMEM with a block barrier
                # (1 stride for NEW<=64). This removes ~18 of 21 block barriers.
                cdi = cutlass.Float32(_INF)
                cii = cutlass.Int32(-1)
                if tid < NEW:
                    cdi = sCandD[tid]
                    cii = sCandI[tid]
                kk = 2
                while kk <= NEW:
                    jj = kk // 2
                    while jj > 0:
                        up = (tid & kk) == 0
                        if jj >= 32:
                            if tid < NEW:
                                sCandD[tid] = cdi
                                sCandI[tid] = cii
                            cute.arch.sync_threads()
                            if tid < NEW:
                                dp = sCandD[tid ^ jj]
                                ip = sCandI[tid ^ jj]
                                if tid < (tid ^ jj):
                                    if (cdi > dp) == up:
                                        cdi = dp
                                        cii = ip
                                else:
                                    if (dp > cdi) == up:
                                        cdi = dp
                                        cii = ip
                            cute.arch.sync_threads()
                        else:
                            if tid < NEW:
                                dp = cute.arch.shuffle_sync_bfly(cdi, jj)
                                ip = cute.arch.shuffle_sync_bfly(cii, jj)
                                if tid < (tid ^ jj):
                                    if (cdi > dp) == up:
                                        cdi = dp
                                        cii = ip
                                else:
                                    if (dp > cdi) == up:
                                        cdi = dp
                                        cii = ip
                        jj //= 2
                    kk *= 2
                if tid < NEW:
                    sCandD[tid] = cdi
                    sCandI[tid] = cii
                cute.arch.sync_threads()

                # Fuse the first bitonic-merge pass (stride ITOPK) into the pool
                # build: pool[i] = min(buffer[i], reversed-candidate[i]) already
                # holds the ITOPK smallest in [0,ITOPK) -- the upper half is
                # discarded, so the remaining log2(ITOPK) merge passes run over
                # ITOPK instead of PCOMB (~half the merge's L1/barrier traffic).
                i = tid
                while i < ITOPK:
                    bd = sBufD[i]
                    bi = sBufI[i]
                    bf = sBufF[i]
                    cd = cutlass.Float32(_INF)
                    ci = cutlass.Int32(-1)
                    if i >= ITOPK - NEW:
                        src = ITOPK - 1 - i        # buffer[i] (asc) pairs cand desc
                        cd = sCandD[src]
                        ci = sCandI[src]
                    if bd <= cd:
                        sPoolD[i] = bd
                        sPoolI[i] = bi
                        sPoolF[i] = bf
                    else:
                        sPoolD[i] = cd
                        sPoolI[i] = ci
                        sPoolF[i] = cutlass.Int32(0)
                    i += T
                cute.arch.sync_threads()

                # Register-resident bitonic merge of the (bitonic) ITOPK pool.
                # slot-major: thread tid owns elements {slot*T+tid}. A stride
                # jj>=T compares a thread's own slots (pure registers); a stride
                # 32<=jj<T round-trips SMEM (cross-warp); jj<32 is a warp
                # shuffle. Only the few mid strides hit SMEM/barriers, vs all
                # log2(ITOPK) before. The sorted result is written straight to
                # the buffer (the separate pool->buffer copyback is dropped).
                ELS = ITOPK // T if ITOPK >= T else 1
                NACT = T if ITOPK >= T else ITOPK
                mdf = cute.make_fragment(ELS, cutlass.Float32)
                mif = cute.make_fragment(ELS, cutlass.Int32)
                mff = cute.make_fragment(ELS, cutlass.Int32)
                if tid < NACT:
                    for s in cutlass.range_constexpr(ELS):
                        e = s * T + tid
                        mdf[s] = sPoolD[e]
                        mif[s] = sPoolI[e]
                        mff[s] = sPoolF[e]
                jj = ITOPK // 2
                while jj > 0:
                    if jj >= T:
                        sj = jj // T
                        if tid < NACT:
                            for s in cutlass.range_constexpr(ELS):
                                sp = s ^ sj
                                if s < sp:
                                    if mdf[s] > mdf[sp]:
                                        td = mdf[s]
                                        mdf[s] = mdf[sp]
                                        mdf[sp] = td
                                        ti = mif[s]
                                        mif[s] = mif[sp]
                                        mif[sp] = ti
                                        tf = mff[s]
                                        mff[s] = mff[sp]
                                        mff[sp] = tf
                    elif jj >= 32:
                        if tid < NACT:
                            for s in cutlass.range_constexpr(ELS):
                                e = s * T + tid
                                sPoolD[e] = mdf[s]
                                sPoolI[e] = mif[s]
                                sPoolF[e] = mff[s]
                        cute.arch.sync_threads()
                        if tid < NACT:
                            for s in cutlass.range_constexpr(ELS):
                                ep = s * T + (tid ^ jj)
                                dp = sPoolD[ep]
                                ip = sPoolI[ep]
                                fp = sPoolF[ep]
                                if tid < (tid ^ jj):
                                    if mdf[s] > dp:
                                        mdf[s] = dp
                                        mif[s] = ip
                                        mff[s] = fp
                                else:
                                    if dp > mdf[s]:
                                        mdf[s] = dp
                                        mif[s] = ip
                                        mff[s] = fp
                        cute.arch.sync_threads()
                    else:
                        if tid < NACT:
                            for s in cutlass.range_constexpr(ELS):
                                dp = cute.arch.shuffle_sync_bfly(mdf[s], jj)
                                ip = cute.arch.shuffle_sync_bfly(mif[s], jj)
                                fp = cute.arch.shuffle_sync_bfly(mff[s], jj)
                                if tid < (tid ^ jj):
                                    if mdf[s] > dp:
                                        mdf[s] = dp
                                        mif[s] = ip
                                        mff[s] = fp
                                else:
                                    if dp > mdf[s]:
                                        mdf[s] = dp
                                        mif[s] = ip
                                        mff[s] = fp
                    jj //= 2
                if tid < NACT:
                    for s in cutlass.range_constexpr(ELS):
                        e = s * T + tid
                        sBufD[e] = mdf[s]
                        sBufI[e] = mif[s]
                        sBufF[e] = mff[s]
                cute.arch.sync_threads()
            else:
                # Fallback (NEW not pow2 or NEW>ITOPK): full bitonic sort.
                i = tid
                while i < PCOMB:
                    if i < ITOPK:
                        sPoolD[i] = sBufD[i]
                        sPoolI[i] = sBufI[i]
                        sPoolF[i] = sBufF[i]
                    elif i < ITOPK + NEW:
                        j = i - ITOPK
                        sPoolD[i] = sCandD[j]
                        sPoolI[i] = sCandI[j]
                        sPoolF[i] = cutlass.Int32(0)
                    else:
                        sPoolD[i] = cutlass.Float32(_INF)
                        sPoolI[i] = cutlass.Int32(-1)
                        sPoolF[i] = cutlass.Int32(0)
                    i += T
                cute.arch.sync_threads()

                kk = 2
                while kk <= PCOMB:
                    jj = kk // 2
                    while jj > 0:
                        i = tid
                        while i < PCOMB:
                            l = i ^ jj
                            if l > i:
                                up = (i & kk) == 0
                                di = sPoolD[i]
                                dl = sPoolD[l]
                                if (di > dl) == up:
                                    sPoolD[i] = dl
                                    sPoolD[l] = di
                                    ti = sPoolI[i]
                                    sPoolI[i] = sPoolI[l]
                                    sPoolI[l] = ti
                                    tf = sPoolF[i]
                                    sPoolF[i] = sPoolF[l]
                                    sPoolF[l] = tf
                            i += T
                        cute.arch.sync_threads()
                        jj //= 2
                    kk *= 2

                i = tid
                while i < ITOPK:
                    sBufD[i] = sPoolD[i]
                    sBufI[i] = sPoolI[i]
                    sBufF[i] = sPoolF[i]
                    i += T
                cute.arch.sync_threads()

            # (f) rebuild the visited set to mirror the new buffer, so next
            #     iter's dedup (step c) is an O(1) membership test.
            i = tid
            while i < HSIZE:
                sHash[i] = cutlass.Int32(0)
                i += T
            cute.arch.sync_threads()
            i = tid
            while i < ITOPK:
                cid = sBufI[i]
                if cid >= 0:
                    h = (cid * HMULT) & HMASK
                    done = cutlass.Int32(0)
                    p = cutlass.Int32(0)
                    while p < HPROBE:
                        if done == 0:
                            slot = (h + p) & HMASK
                            old = cute.arch.atomic_cas(
                                sHash.iterator + slot, cmp=cutlass.Int32(0),
                                val=cid + 1, scope="cta")
                            if old == 0 or old == cid + 1:
                                done = cutlass.Int32(1)
                        p += 1
                i += T
            cute.arch.sync_threads()
            _it += 1

        # ── emit the k closest (buffer is sorted from the last merge) ──
        if tid < K:
            dv = sBufD[tid]
            if dv < cutlass.Float32(_INF):
                mOutV[qid, tid] = dv
                mOutI[qid, tid] = sBufI[tid]
            else:
                mOutV[qid, tid] = cutlass.Float32(_INF)
                mOutI[qid, tid] = cutlass.Int32(-1)


# =============================================================================
# Host entry point
# =============================================================================

_kernel_cache: dict = {}


def _to_cute(t: torch.Tensor):
    mt = cute_rt.from_dlpack(t, assumed_align=16)
    return mt.mark_layout_dynamic(leading_dim=t.ndim - 1)


def _auto_max_iters(itopk: int, search_width: int) -> int:
    return int(math.ceil(itopk / max(1, search_width)) + 16)


def cagra_search_cutedsl(
    index,
    Q: torch.Tensor,
    k: int,
    *,
    itopk_size: int = 64,
    search_width: int = 1,
    max_iterations: int = 0,
    min_iterations: int = 0,
    num_random_seeds: int = 0,
    seed: int = 0,
    threads: int = 128,
    team_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-CTA CuteDSL CAGRA search. Returns ``(vals, ids)`` ``(nq, k)``.

    Drop-in for :func:`cagra_search_triton`: ``vals`` are squared-L2 fp32,
    ``ids`` int64 original row ids (``-1`` padded).
    """
    assert Q.is_cuda and index.dataset.is_cuda and index.graph.is_cuda
    dataset = index.dataset.to(torch.float32).contiguous()
    graph = index.graph.to(torch.int32).contiguous()
    N, D = dataset.shape
    gd = index.graph_degree

    nq = Q.shape[0]
    Qf = Q.to(torch.float32).contiguous()

    itopk_eff = max(int(itopk_size), int(k), gd)
    ITOPK = _next_pow2(itopk_eff)
    S = max(1, int(search_width))

    R = default_random_seeds(num_random_seeds, getattr(index, "n_components", 1),
                             ITOPK, N)

    max_iters = int(max_iterations) or _auto_max_iters(ITOPK, S)
    max_iters = max(max_iters, int(min_iterations), 1)

    NEW = S * gd
    PCOMB = _next_pow2(ITOPK + NEW)
    # Incremental merge needs PCOMB == 2*ITOPK, i.e. NEW a power of two and
    # NEW <= ITOPK (true for the usual sw=1, pow2 graph_degree). Otherwise the
    # kernel uses the full-sort path.
    INCR = (NEW & (NEW - 1)) == 0 and NEW <= ITOPK
    # Visited-set capacity: 4x the buffer (load factor ~0.25) is occupancy-
    # friendly (4 KB at itopk=256). Inserts/lookups short-circuit on the first
    # empty/own slot, so the HPROBE cap only has to be deep enough that a slot
    # is essentially never lost to a full probe window (no dup => recall intact).
    HSIZE = _next_pow2(4 * ITOPK)

    # team_size: threads per distance (must divide 32 and D). Each thread
    # vector-loads VPT=D/TS contiguous dims. There are two competing costs:
    # a *wider* team shrinks VPT (fewer registers/thread -> higher occupancy)
    # but *deepens* the butterfly reduction (more shuffles per distance). A TS
    # sweep on H100 over D=64..960 puts the sweet spot at VPT~4-8: the best TS
    # is the LARGEST valid one whose VPT stays >= 4 (D=64->TS16, D=96->TS16,
    # D=128/256/960->TS32). VPT>=4 also keeps the row load float4-shaped.
    # An explicit team_size overrides when valid.
    def _valid(ts):
        return ts > 0 and 32 % ts == 0 and D % ts == 0
    TS = int(team_size)
    if not _valid(TS):
        cap = max(1, D // 4)                    # VPT = D/TS >= 4  <=>  TS <= D/4
        TS = 1
        for _cand in (2, 4, 8, 16, 32):        # largest valid divisor <= cap
            if _valid(_cand) and _cand <= cap:
                TS = _cand

    g = torch.Generator(device="cpu").manual_seed(seed)
    seeds = torch.randint(0, N, (nq, R), generator=g).to(torch.int32).to(Q.device).contiguous()

    out_val = torch.empty((nq, k), device=Q.device, dtype=torch.float32)
    out_id = torch.empty((nq, k), device=Q.device, dtype=torch.int32)

    key = (D, gd, ITOPK, S, max_iters, R, int(k), PCOMB, HSIZE, threads, TS, INCR)
    compiled = _kernel_cache.get(key)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if compiled is None:
        kernel = HopperCagraSearch(
            D=D, GD=gd, ITOPK=ITOPK, SW=S, MAXIT=max_iters, R=R, K=int(k),
            PCOMB=PCOMB, HSIZE=HSIZE, threads=threads, TS=TS, INCR=INCR,
        )
        compiled = cute.compile(
            kernel,
            _to_cute(Qf), _to_cute(dataset), _to_cute(graph),
            _to_cute(seeds), _to_cute(out_val), _to_cute(out_id),
            stream,
        )
        _kernel_cache[key] = compiled

    compiled(
        _to_cute(Qf), _to_cute(dataset), _to_cute(graph),
        _to_cute(seeds), _to_cute(out_val), _to_cute(out_id),
        stream,
    )

    vals = out_val
    ids = out_id.to(torch.int64)
    ids = torch.where(torch.isinf(vals), torch.full_like(ids, -1), ids)
    return vals, ids


__all__ = ["cagra_search_cutedsl", "HopperCagraSearch"]
