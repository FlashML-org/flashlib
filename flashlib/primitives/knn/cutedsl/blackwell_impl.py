"""Blackwell (sm_100) CuteDSL flash-KNN kernels (BF16, D=128).

Two exact kernels for the regimes where flashlib's Triton path is weakest
on B200:

* :class:`BlackwellKnnBuild` -- self-kNN / large-Q build. tcgen05 MMA + TMA
  loads + a register top-K fused in the MMA epilogue, split-K over the db +
  a fused merge. Scores with ``s = c_sq[m] - 2<x,c>``: the ``x_sq`` term is
  constant per query row, so dropping it preserves the argmin-K (the merge
  re-adds it for true distances). The next db tile's async MMA is issued
  before the CUDA-core-bound top-K so tensor cores and CUDA cores overlap.

* :class:`BlackwellKnnSearch` -- small-Q search, where Triton 3.4 ``tl.dot``
  asserts ``M >= 16`` on sm_100 and cannot run at all. FMA dot-product,
  per-thread top-K, smem tree merge to the CTA top-K, then a fused merge
  over the S splits.

Both return ``(B, N, k) int32`` indices (ascending by true squared-L2),
matching the :func:`flashlib.primitives.knn.flash_knn` contract.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Optional heavy deps: cutlass-dsl + cuda-python. Guarded so importing
# flashlib never fails on machines without them (kernels simply unavailable).
# ---------------------------------------------------------------------------
_BW_AVAILABLE = False
_BW_IMPORT_ERROR: Optional[Exception] = None

try:
    import cuda.bindings.driver as cuda

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.pipeline as pipeline
    from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
    from cutlass.cute.nvgpu import cpasync, tcgen05
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute.runtime import from_dlpack

    _BW_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - any import problem disables the path
    _BW_IMPORT_ERROR = exc


BLOCK_Q = 128
BLOCK_N = 64
D = 128
TILE_M = 128
THREADS = 128


def blackwell_available() -> bool:
    """True iff cutlass-dsl + cuda-python imported (kernels are usable)."""
    return _BW_AVAILABLE


# ===========================================================================
# Triton helpers: row squared-norm + fused split merge (index-only).
# ===========================================================================
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _rownorm_kernel(x_ptr, o_ptr, N, Dd: tl.constexpr, BLK: tl.constexpr):
        row = tl.program_id(0)
        if row >= N:
            return
        offs = tl.arange(0, BLK)
        mask = offs < Dd
        v = tl.load(x_ptr + row * Dd + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(o_ptr + row, tl.sum(v * v, axis=0))

    @triton.jit
    def _merge_kernel(ps_ptr, pi_ptr, xsq_ptr, od_ptr, oi_ptr, N,
                      SK: tl.constexpr, K: tl.constexpr, BLK: tl.constexpr):
        row = tl.program_id(0)
        if row >= N:
            return
        offs = tl.arange(0, BLK)
        mask = offs < SK
        s = tl.load(ps_ptr + row * SK + offs, mask=mask, other=float("inf"))
        idx = tl.load(pi_ptr + row * SK + offs, mask=mask, other=-1)
        xsq = tl.load(xsq_ptr + row)
        for j in tl.static_range(K):
            m = tl.min(s, axis=0)
            pos = tl.argmin(s, axis=0)
            sel = tl.sum(tl.where(offs == pos, idx, 0))
            d = xsq + m
            d = tl.where(d > 0.0, d, 0.0)
            tl.store(od_ptr + row * K + j, d)
            tl.store(oi_ptr + row * K + j, sel)
            s = tl.where(offs == pos, float("inf"), s)

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _row_sqnorm(x2d: torch.Tensor, out=None) -> torch.Tensor:
    N, Dd = x2d.shape
    if out is None:
        out = torch.empty(N, device=x2d.device, dtype=torch.float32)
    if _HAVE_TRITON:
        _rownorm_kernel[(N,)](x2d, out, N, Dd=Dd, BLK=_next_pow2(Dd))
        return out
    out.copy_((x2d.float() * x2d.float()).sum(-1))
    return out


def _merge(part_s, part_i, x_sq, k):
    """Reduce the S*k unsorted partials per row to the global sorted top-k,
    re-adding the dropped x_sq term (clamp >=0)."""
    N, S, k_keep = part_s.shape
    SK = S * k_keep
    ps = part_s.reshape(N, SK).contiguous()
    pi = part_i.reshape(N, SK).contiguous()
    out_d = torch.empty((N, k), device=part_s.device, dtype=torch.float32)
    out_i = torch.empty((N, k), device=part_s.device, dtype=torch.int32)
    if _HAVE_TRITON:
        _merge_kernel[(N,)](ps, pi, x_sq, out_d, out_i, N,
                            SK=SK, K=k, BLK=_next_pow2(SK))
        return out_d, out_i
    vals, pos = torch.topk(ps, k, dim=-1, largest=False, sorted=True)
    out_i = torch.gather(pi, 1, pos)
    out_d = torch.clamp(x_sq.unsqueeze(-1) + vals, min=0.0)
    return out_d, out_i


# ===========================================================================
# Build kernel (tcgen05 MMA + register top-K + split-K).
# ===========================================================================
if _BW_AVAILABLE:
    INF = cutlass.Float32(3.0e38)

    class BlackwellKnnBuild:
        def __init__(self, k: int, num_splits: int = 1,
                     acc_dtype=cutlass.Float32):
            self.k = k
            self.num_splits = num_splits
            self.acc_dtype = acc_dtype
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_Q, BLOCK_N)
            self.num_ab_stage = 2
            self.threads_per_cta = 128

        # ---- reusable device-side exact top-K (@cute.jit-inlined) ----
        # Preprocessed + inlined at trace time. Conventions that matter:
        # mutable rmem (best_d/best_i) is mutated in place, SSA scalars
        # (worst_d/worst_pos) are returned, and the dynamic store
        # best_d[worst_pos] must stay inside the `if` so it lowers to a REAL
        # branch (otherwise the group-min skip predicates instead of skipping).
        @cute.jit
        def _topk_init(self, K: cutlass.Constexpr):
            """Init the unsorted per-thread register top-K (+ cached worst)."""
            best_d = cute.make_rmem_tensor(cute.make_layout((K,)),
                                           cutlass.Float32)
            best_i = cute.make_rmem_tensor(cute.make_layout((K,)), cutlass.Int32)
            for j in cutlass.range_constexpr(K):
                best_d[j] = INF
                best_i[j] = cutlass.Int32(-1)
            return best_d, best_i, cutlass.Float32(INF), cutlass.Int32(0)

        @cute.jit
        def _worst_of(self, best_d, K: cutlass.Constexpr):
            """Streaming running-max worst-of-K (O(1) live state; cake's k32
            scan). Two CuteDSL-specific choices, each picked over the option
            that looks better on paper:

            * max-tree (cake's small-K choice) materialises all K leaves into
              SSA and spills in CuteDSL -- ~2x SLOWER even at k=5/10. The scan
              keeps two scalars live.
            * keeping best_d/best_i register-resident (predicated writes, no
              dynamic store to local memory) forces 2K values live and blows up
              register pressure -- ~7x SLOWER at k=32. The dynamic store + this
              local scan keeps occupancy high, and the group-min skip makes the
              scan rare enough that the local loads stay L1-resident."""
            worst_d = best_d[0]
            worst_pos = cutlass.Int32(0)
            for jj in cutlass.range_constexpr(K - 1):
                j = jj + 1
                gt = best_d[j] > worst_d
                worst_d = cutlass.select_(gt, best_d[j], worst_d)
                worst_pos = cutlass.select_(gt, cutlass.Int32(j), worst_pos)
            return worst_d, worst_pos

        @cute.jit
        def _topk_consume_tile(self, best_d, best_i, worst_d, worst_pos, frag,
                               sCsq, base, K: cutlass.Constexpr,
                               BN: cutlass.Constexpr):
            """Fold one [BN] distance fragment into the top-K. Group-min skip:
            scan 4 candidates at a time and skip the group when even its min
            can't beat the current worst."""
            for g in cutlass.range_constexpr(BN // 4):
                cands = [sCsq[g * 4 + 0] - 2.0 * frag[g * 4 + 0],
                         sCsq[g * 4 + 1] - 2.0 * frag[g * 4 + 1],
                         sCsq[g * 4 + 2] - 2.0 * frag[g * 4 + 2],
                         sCsq[g * 4 + 3] - 2.0 * frag[g * 4 + 3]]
                gmin = cutlass.min(cutlass.min(cands[0], cands[1]),
                                   cutlass.min(cands[2], cands[3]))
                if gmin < worst_d:
                    for t in cutlass.range_constexpr(4):
                        cv = cands[t]
                        if cv < worst_d:
                            best_d[worst_pos] = cv
                            best_i[worst_pos] = cutlass.Int32(base + g * 4 + t)
                            worst_d, worst_pos = self._worst_of(best_d, K)
            return worst_d, worst_pos

        @cute.jit
        def _topk_write_partials(self, best_d, best_i, mPartS, mPartI, q, split,
                                 K: cutlass.Constexpr):
            """Write the unsorted top-K to split partials (the merge sorts)."""
            for j in cutlass.range_constexpr(K):
                mPartS[q, split, j] = best_d[j]
                mPartI[q, split, j] = best_i[j]

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mPartS: cute.Tensor, mPartI: cute.Tensor,
                     stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], 64)

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,))

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.x_dtype, self.num_ab_stage)
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.c_dtype_in, self.num_ab_stage)

            a_op = sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id)
            a_smem_one = cute.slice_(a_smem_layout, (None, None, None, 0))
            tma_atom_a, tma_x = cute.nvgpu.make_tiled_tma_atom_A(
                a_op, mX, a_smem_one, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape)
            b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id)
            b_smem_one = cute.slice_(b_smem_layout, (None, None, None, 0))
            tma_atom_b, tma_c = cute.nvgpu.make_tiled_tma_atom_B(
                b_op, mC, b_smem_one, self.mma_tiler, tiled_mma,
                self.cluster_layout_vmnk.shape)

            elem_bytes = self.x_dtype.width // 8
            self.num_tma_load_bytes = (
                (self.mma_tiler[0] + self.mma_tiler[1]) * self.mma_tiler[2]
                * elem_bytes)
            self.num_tmem_alloc_cols = 64
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // BLOCK_Q, self.num_splits, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mPartS, mPartI, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mPartS, mPartI, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            K = self.k
            num_splits = self.num_splits
            num_ab_stage = self.num_ab_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, bidy, _ = cute.arch.block_idx()

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            @cute.struct
            class SharedStorage:
                ab_full: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
                acc_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, BLOCK_N]

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.ab_full.data_ptr(),
                num_stages=num_ab_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_load_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            ab_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_ab_stage)
            ab_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_ab_stage)

            acc_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.acc_full.data_ptr(), num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.threads_per_cta),
                cta_layout_vmnk=None, defer_sync=True)
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)

            tmem_alloc_bar = pipeline.NamedBarrier(
                barrier_id=1, num_threads=self.threads_per_cta)
            tmem = utils.TmemAllocator(
                storage.tmem_holding, barrier_for_retrieve=tmem_alloc_bar,
                is_two_cta=False,
                two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc)

            pipeline_init_arrive(is_relaxed=True)

            sX = smem.allocate_tensor(self.x_dtype, a_smem_layout.outer, 128,
                                      swizzle=a_smem_layout.inner)
            sC = smem.allocate_tensor(self.c_dtype_in, b_smem_layout.outer, 128,
                                      swizzle=b_smem_layout.inner)
            sCsq = storage.sCsq.get_tensor(cute.make_layout((BLOCK_N,)))

            gX = cute.local_tile(mX, cute.slice_(self.mma_tiler, (None, 0, None)),
                                 (None, None, None))
            gC = cute.local_tile(mC, cute.slice_(self.mma_tiler, (0, None, None)),
                                 (None, None, None))
            n_db_tiles = cute.size(gC, mode=[2])
            k_tile_cnt = cute.size(gX, mode=[3])

            thr_mma = tiled_mma.get_slice(0)
            tCgX = thr_mma.partition_A(gX)
            tCgC = thr_mma.partition_B(gC)
            aL = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            tXsX, tXgX = cpasync.tma_partition(
                tma_atom_a, 0, aL, cute.group_modes(sX, 0, 3),
                cute.group_modes(tCgX, 0, 3))
            bL = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            tCsC, tCgC2 = cpasync.tma_partition(
                tma_atom_b, 0, bL, cute.group_modes(sC, 0, 3),
                cute.group_modes(tCgC, 0, 3))

            tCrX = tiled_mma.make_fragment_A(sX)
            tCrC = tiled_mma.make_fragment_B(sC)
            acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
            tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

            pipeline_init_wait()
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tXgX = tXgX[(None, bidx, None, 0)]

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, cutlass.Float32,
                self.acc_dtype, self.epi_tile, False)
            tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)],
                                        self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_t2r.partition_S(tAcc_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                cute.make_layout(((BLOCK_N, 1), 1, 1)), self.acc_dtype)

            tmem.relinquish_alloc_permit()

            best_d, best_i, worst_d, worst_pos = self._topk_init(K)

            tiles_per_split = n_db_tiles // num_splits
            db_start = bidy * tiles_per_split

            # prologue: warp 0 issues the first db tile's MMA
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    ab_pipeline.producer_acquire(ab_prod)
                    bar = ab_pipeline.producer_get_barrier(ab_prod)
                    cute.copy(tma_atom_a, tXgX[(None, kk)],
                              tXsX[(None, ab_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    cute.copy(tma_atom_b, tCgC2[(None, db_start, kk, 0)],
                              tCsC[(None, ab_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    ab_prod.advance()
                acc_pipeline.producer_acquire(acc_prod)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kk in cutlass.range(k_tile_cnt):
                    ab_pipeline.consumer_wait(ab_cons)
                    nkb = cute.size(tCrX, mode=[2])
                    for kb in cutlass.range(nkb, unroll_full=True):
                        crd = (None, None, kb, ab_cons.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrX[crd], tCrC[crd], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_cons)
                    ab_cons.advance()
                acc_pipeline.producer_commit(acc_prod)
                acc_prod.advance()

            for dd in cutlass.range(tiles_per_split):
                db = db_start + dd
                if tidx < BLOCK_N:
                    sCsq[tidx] = mCsq[db * BLOCK_N + tidx]

                acc_pipeline.consumer_wait(acc_cons)
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)],
                          tTR_rAcc)
                acc_pipeline.consumer_release(acc_cons)
                acc_cons.advance()

                cute.arch.barrier()   # c_sq visible to all threads

                # issue the next tile's MMA now (warp 0): async tcgen05 overlaps
                # the CUDA-core-bound top-K below. 1 acc stage suffices -- the
                # copy above already drained the accumulator to registers.
                db_next = db + 1
                if dd + 1 < tiles_per_split:
                    if warp_idx == 0:
                        for kk in cutlass.range(k_tile_cnt):
                            ab_pipeline.producer_acquire(ab_prod)
                            bar = ab_pipeline.producer_get_barrier(ab_prod)
                            cute.copy(tma_atom_a, tXgX[(None, kk)],
                                      tXsX[(None, ab_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            cute.copy(tma_atom_b, tCgC2[(None, db_next, kk, 0)],
                                      tCsC[(None, ab_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            ab_prod.advance()
                        acc_pipeline.producer_acquire(acc_prod)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range(k_tile_cnt):
                            ab_pipeline.consumer_wait(ab_cons)
                            nkb = cute.size(tCrX, mode=[2])
                            for kb in cutlass.range(nkb, unroll_full=True):
                                crd = (None, None, kb, ab_cons.index)
                                cute.gemm(tiled_mma, tCtAcc, tCrX[crd],
                                          tCrC[crd], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            ab_pipeline.consumer_release(ab_cons)
                            ab_cons.advance()
                        acc_pipeline.producer_commit(acc_prod)
                        acc_prod.advance()

                base = db * BLOCK_N
                frag = tTR_rAcc.load()
                worst_d, worst_pos = self._topk_consume_tile(
                    best_d, best_i, worst_d, worst_pos, frag, sCsq, base,
                    K, BLOCK_N)
                cute.arch.barrier()

            # Unsorted is fine -- the merge does a top-K over all S*K partials.
            q = bidx * BLOCK_Q + tidx
            self._topk_write_partials(best_d, best_i, mPartS, mPartI, q, bidy, K)

            cute.arch.sync_threads()
            tmem.free(tmem_ptr)
            if warp_idx == 0:
                ab_pipeline.producer_tail(ab_prod)

    class BlackwellKnnSearch:
        def __init__(self, k: int, num_splits: int):
            self.k = k
            self.num_splits = num_splits
            self.threads = THREADS

        @cute.jit
        def __call__(self, mQ: cute.Tensor, mC: cute.Tensor,
                     mPartS: cute.Tensor, mPartI: cute.Tensor,
                     stream: cuda.CUstream):
            Q = mQ.shape[0]
            grid = (Q, self.num_splits, 1)
            self.kernel(mQ, mC, mPartS, mPartI).launch(
                grid=grid, block=[self.threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, mQ: cute.Tensor, mC: cute.Tensor,
                   mPartS: cute.Tensor, mPartI: cute.Tensor):
            K = self.k
            S = self.num_splits
            tidx, _, _ = cute.arch.thread_idx()
            q, split, _ = cute.arch.block_idx()
            M = mC.shape[0]

            PAD = D + 1   # conflict-free sC[row, d] reads

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.MemRange[cutlass.Float32, D]
                sC: cute.struct.MemRange[cutlass.BFloat16, TILE_M * PAD]
                sV: cute.struct.MemRange[cutlass.Float32, THREADS * K]
                sI: cute.struct.MemRange[cutlass.Int32, THREADS * K]

            smem = utils.SmemAllocator()
            st = smem.allocate(SharedStorage)
            sQ = st.sQ.get_tensor(cute.make_layout((D,)))
            sC = st.sC.get_tensor(cute.make_layout((TILE_M, PAD)))
            sV = st.sV.get_tensor(cute.make_layout((THREADS * K,)))
            sI = st.sI.get_tensor(cute.make_layout((THREADS * K,)))

            if tidx < D:
                sQ[tidx] = mQ[q, tidx].to(cutlass.Float32)
            cute.arch.barrier()

            topv = cute.make_rmem_tensor(cute.make_layout((K,)), cutlass.Float32)
            topi = cute.make_rmem_tensor(cute.make_layout((K,)), cutlass.Int32)
            for j in cutlass.range(K, unroll_full=True):
                topv[j] = INF
                topi[j] = cutlass.Int32(-1)

            n_tiles = (M + TILE_M - 1) // TILE_M
            tiles_per_split = (n_tiles + S - 1) // S
            tile_start = split * tiles_per_split

            m_last = M - 1
            for tt in cutlass.range(tiles_per_split):
                base = (tile_start + tt) * TILE_M
                for i in cutlass.range(TILE_M, unroll_full=True):
                    grow = cutlass.min(base + i, m_last)
                    sC[i, tidx] = mC[grow, tidx]
                cute.arch.barrier()

                row = base + tidx
                if row < M:
                    acc = cutlass.Float32(0.0)
                    csq = cutlass.Float32(0.0)
                    for d in cutlass.range(D, unroll_full=True):
                        cv = sC[tidx, d].to(cutlass.Float32)
                        acc += sQ[d] * cv
                        csq += cv * cv
                    cand_v = csq - 2.0 * acc
                    cand_i = cutlass.Int32(row)
                    for j in cutlass.range(K, unroll_full=True):
                        ov = topv[j]
                        oi = topi[j]
                        p = cand_v < ov
                        topv[j] = cutlass.min(cand_v, ov)
                        topi[j] = cutlass.Int32(cutlass.select_(p, cand_i, oi))
                        cand_v = cutlass.max(cand_v, ov)
                        cand_i = cutlass.Int32(cutlass.select_(p, oi, cand_i))
                cute.arch.barrier()

            for j in cutlass.range(K, unroll_full=True):
                sV[tidx * K + j] = topv[j]
                sI[tidx * K + j] = topi[j]
            cute.arch.barrier()

            stride = THREADS // 2
            while stride >= 1:
                if tidx < stride:
                    a = tidx * K
                    b = (tidx + stride) * K
                    i = cutlass.Int32(0)
                    jj = cutlass.Int32(0)
                    ov = cute.make_rmem_tensor(cute.make_layout((K,)),
                                               cutlass.Float32)
                    oi = cute.make_rmem_tensor(cute.make_layout((K,)),
                                               cutlass.Int32)
                    for o in cutlass.range(K, unroll_full=True):
                        av = sV[a + i]
                        ai = sI[a + i]
                        bv = sV[b + jj]
                        bi = sI[b + jj]
                        take_a = av <= bv
                        ov[o] = cutlass.min(av, bv)
                        oi[o] = cutlass.Int32(cutlass.select_(take_a, ai, bi))
                        i = i + cutlass.Int32(cutlass.select_(take_a, 1, 0))
                        jj = jj + cutlass.Int32(cutlass.select_(take_a, 0, 1))
                    for o in cutlass.range(K, unroll_full=True):
                        sV[a + o] = ov[o]
                        sI[a + o] = oi[o]
                cute.arch.barrier()
                stride = stride // 2

            if tidx < K:
                mPartS[q, split, tidx] = sV[tidx]
                mPartI[q, split, tidx] = sI[tidx]


# ===========================================================================
# Host drivers + caches.
# ===========================================================================
_BUILD_CACHE: dict = {}
_SEARCH_CACHE: dict = {}


def _pow2_div(want: int, n_db_tiles: int) -> int:
    want = min(max(1, want), n_db_tiles)
    p = 1
    while p * 2 <= want and (n_db_tiles % (p * 2) == 0):
        p *= 2
    return p


_SM_FILL = 96  # ~B200 SM count; below this many q-tiles we split to fill


def pick_splits_build(N: int, k: int = 0) -> int:
    """Split count for the exact build. Each split keeps the full top-k, so
    fewer splits = fewer merge candidates + one partial write: S=1 is cheapest
    once the query tiles alone fill the SMs (``n_q_tiles >= _SM_FILL``). Only
    smaller N splits the db to reach ~2 waves. (S=1 beats S=4 ~1.8x at
    N=16384, k=10.)"""
    n_db_tiles = N // BLOCK_N
    n_q_tiles = max(1, N // BLOCK_Q)
    if n_q_tiles >= _SM_FILL:
        return 1
    return _pow2_div(round(256 / n_q_tiles), n_db_tiles)


def pick_splits_search(Q: int, M: int, target_ctas: int = 320,
                       tps_max: int = 16) -> int:
    n_tiles = (M + TILE_M - 1) // TILE_M
    s_fill = math.ceil(target_ctas / max(1, Q))
    s_serial = math.ceil(n_tiles / tps_max)
    want = min(max(s_fill, s_serial), n_tiles)
    p = 1
    while p * 2 <= want:
        p *= 2
    return p


def _cur_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def knn_build_cutedsl(x: torch.Tensor, k: int, *, num_splits=None,
                      part_s=None, part_i=None, x_sq=None,
                      return_distances: bool = True):
    """Exact self-kNN build for x:(N,D) bf16, D=128 -> idx (N,k) i32 (+ dist
    f32 when ``return_distances``). Each split keeps the full top-k and the
    merge reduces the S*k partials; the worst-of-K recompute is a streaming
    scan so the per-thread top-K stays small even at k=32."""
    if x.dim() == 3:
        x = x[0]
    N, Dd = x.shape
    assert Dd == D and x.dtype == torch.bfloat16
    if N % BLOCK_Q != 0:
        raise ValueError(f"build requires N % {BLOCK_Q} == 0, got N={N}")
    S = num_splits if num_splits is not None else pick_splits_build(N, k)
    if x_sq is None:
        x_sq = _row_sqnorm(x)
    if part_s is None:
        part_s = torch.empty((N, S, k), device=x.device, dtype=torch.float32)
    if part_i is None:
        part_i = torch.empty((N, S, k), device=x.device, dtype=torch.int32)
    x3 = x.unsqueeze(-1)
    stream = _cur_stream()
    x_dl = from_dlpack(x3)
    sq_dl = from_dlpack(x_sq)
    dls = (x_dl, x_dl, sq_dl, from_dlpack(part_s), from_dlpack(part_i))
    key = (N, k, S)
    comp = _BUILD_CACHE.get(key)
    if comp is None:
        comp = cute.compile(BlackwellKnnBuild(k, S), *dls, stream)
        _BUILD_CACHE[key] = comp
    comp(*dls, stream)
    dist, idx = _merge(part_s, part_i, x_sq, k)
    return (dist, idx) if return_distances else idx


def knn_search_cutedsl(qx: torch.Tensor, db: torch.Tensor, k: int, *,
                       num_splits=None, q_sq=None, part_s=None, part_i=None,
                       return_distances: bool = True):
    """Exact search qx:(Q,D) vs db:(M,D) bf16, D=128 -> idx (Q,k) i32 (+ dist
    f32 when ``return_distances``)."""
    if qx.dim() == 3:
        qx = qx[0]
    if db.dim() == 3:
        db = db[0]
    Q, Dd = qx.shape
    M = db.shape[0]
    assert Dd == D and qx.dtype == torch.bfloat16
    S = num_splits if num_splits is not None else pick_splits_search(Q, M)
    if q_sq is None:
        q_sq = _row_sqnorm(qx)
    if part_s is None:
        part_s = torch.empty((Q, S, k), device=qx.device, dtype=torch.float32)
    if part_i is None:
        part_i = torch.empty((Q, S, k), device=qx.device, dtype=torch.int32)
    stream = _cur_stream()
    dls = (from_dlpack(qx), from_dlpack(db),
           from_dlpack(part_s), from_dlpack(part_i))
    key = (Q, M, k, S)
    comp = _SEARCH_CACHE.get(key)
    if comp is None:
        comp = cute.compile(BlackwellKnnSearch(k, S), *dls, stream)
        _SEARCH_CACHE[key] = comp
    comp(*dls, stream)
    dist, idx = _merge(part_s, part_i, q_sq, k)
    return (dist, idx) if return_distances else idx


def blackwell_supported(x: torch.Tensor, c: torch.Tensor, k: int) -> bool:
    """True iff the Blackwell path can run this (3D) workload: bf16, D=128,
    single batch, CUDA, k<=64. Build needs N % BLOCK_Q == 0; search needs a
    db tile."""
    if not _BW_AVAILABLE:
        return False
    if x.dim() != 3 or c.dim() != 3:
        return False
    B, N, Dd = x.shape
    if B != 1 or Dd != D or k > 64:
        return False
    if x.dtype != torch.bfloat16 or c.dtype != torch.bfloat16:
        return False
    if not x.is_cuda or not c.is_cuda:
        return False
    M = c.shape[1]
    is_build = (x.data_ptr() == c.data_ptr() and N == M)
    if is_build:
        return N % BLOCK_Q == 0 and N >= BLOCK_Q
    # search path: needs at least one db tile
    return M >= 1


def blackwell_flash_knn(x: torch.Tensor, c: torch.Tensor, k: int,
                        **kwargs) -> torch.Tensor:
    """Index-only Blackwell KNN (flash_knn contract). x/c are (B,N,D)/(B,M,D)
    bf16, B==1; picks build vs search by shape. Returns (B, N, k) int32."""
    del kwargs
    xb = x[0]
    cb = c[0]
    N = xb.shape[0]
    M = cb.shape[0]
    is_build = (x.data_ptr() == c.data_ptr() and N == M and N % BLOCK_Q == 0)
    if is_build:
        idx = knn_build_cutedsl(xb, k, return_distances=False)
    else:
        idx = knn_search_cutedsl(xb, cb, k, return_distances=False)
    return idx.unsqueeze(0)
