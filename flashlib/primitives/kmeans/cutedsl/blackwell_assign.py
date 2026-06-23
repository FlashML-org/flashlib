"""Blackwell (sm_100) CuteDSL flash-kmeans *assign* kernel (BF16).

A clean-room CuteDSL port of flashlib-cake's ``flash_kmeans_assign`` v15
Blackwell kernel. Computes, for each point ``x`` and centroid set ``c``,

    argmin_n ‖x − c_n‖²  ==  argmax_n (⟨x, c_n⟩ − ½‖c_n‖²)

i.e. the x²-free rank: ``‖x‖²`` is constant per row and drops out of the
argmin, so it is never loaded (cake does the same trick).

Design (mirrors :class:`flashlib...knn.cutedsl.BlackwellKnnBuild`, whose
tcgen05 MMA + TMA + TMEM scaffolding is reused verbatim — only the epilogue
changes from a register top-K to a per-row running argmin):

* One CTA per ``BLOCK_M=128`` point tile. TMA-loads the X tile once, then
  streams the ``K`` centroids in ``BLOCK_N=64`` tiles through a
  ``PipelineTmaUmma`` SMEM pipeline.
* ``tcgen05`` MMA computes ``cross = X · Cᵀ`` (contraction over D in 64-wide
  k-blocks: D=64→1, 128→2, 256→4) into a TMEM accumulator.
* Epilogue: the ``(BLOCK_M, BLOCK_N)`` accumulator is loaded from TMEM so
  that **thread ``tidx`` owns row ``tidx``'s full ``BLOCK_N`` score vector**
  in registers (the tmem-load layout the build relies on). Each thread keeps
  a running ``(best_score, best_idx)`` argmin across all centroid tiles and
  writes a single int32 id at the end. ``½‖c‖²`` for the tile is staged in
  SMEM (as in cake).
* The next centroid tile's async MMA is issued *before* the CUDA-core argmin
  scan so tensor cores overlap the epilogue (the build's overlap trick).

Supported regime: B=1, bf16, D ∈ {64, 128, 256}, ``N % 128 == 0``,
``K % 64 == 0``. Anything else should fall back to the Triton path.
"""
from __future__ import annotations

from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Optional heavy deps: cutlass-dsl + cuda-python. Guarded so importing
# flashlib never fails on machines without them (kernel simply unavailable).
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


BLOCK_M = 128       # points per CTA / threads per CTA (one row per thread)
BLOCK_N = 64        # default centroids per MMA N-tile (tunable per shape)
MMA_K = 64          # D contraction tile
SUPPORTED_D = (64, 128, 256)
SUPPORTED_BN = (64, 128, 256)   # MMA N-tiles the kernel can be built with


def blackwell_assign_available() -> bool:
    """True iff cutlass-dsl + cuda-python imported (kernel is usable)."""
    return _BW_AVAILABLE


if _BW_AVAILABLE:
    INF = cutlass.Float32(3.0e38)

    class BlackwellFlashKmeansAssign:
        """tcgen05 flash-kmeans assign: argmin ‖x−c‖² via x²-free argmax."""

        def __init__(self, acc_dtype=cutlass.Float32, block_n: int = BLOCK_N):
            self.acc_dtype = acc_dtype
            self.block_n = block_n
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_M, block_n)
            self.num_ab_stage = 2
            self.threads_per_cta = BLOCK_M

        # ---- per-tile running argmin (CUDA-core epilogue) -----------------
        # Branchless select keeps the lowest index on ties (strict <),
        # matching the reference argmax(⟨x,c⟩ − ½‖c‖²) tie-break. ``base`` is
        # the centroid-tile offset; ``sCsq`` holds ½‖c‖² for the tile.
        @cute.jit
        def _consume_tile_argmin(self, best_s, best_i, frag, sCsq, base,
                                 BN: cutlass.Constexpr):
            for n in cutlass.range_constexpr(BN):
                s = sCsq[n] - frag[n]
                lt = s < best_s
                best_s = cutlass.select_(lt, s, best_s)
                best_i = cutlass.select_(lt, cutlass.Int32(base + n), best_i)
            return best_s, best_i

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mOut: cute.Tensor, stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], MMA_K)

            # One AB stage per D-contraction k-tile so warp 0's prologue can
            # issue all k-tiles before draining them (no producer-vs-self
            # deadlock); >= 2 for TMA/MMA overlap headroom. D=64->2, 128->2,
            # 256->4.
            D = mX.shape[1]
            self.num_ab_stage = max(2, D // MMA_K)

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
            self.num_tmem_alloc_cols = self.block_n
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // BLOCK_M, 1, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mOut, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mOut, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            num_ab_stage = self.num_ab_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            block_n = self.block_n

            @cute.struct
            class SharedStorage:
                ab_full: cute.struct.MemRange[cutlass.Int64, num_ab_stage * 2]
                acc_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, block_n]

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
            sCsq = storage.sCsq.get_tensor(cute.make_layout((block_n,)))

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
                cute.make_layout(((block_n, 1), 1, 1)), self.acc_dtype)

            tmem.relinquish_alloc_permit()

            best_s = cutlass.Float32(INF)
            best_i = cutlass.Int32(-1)

            # prologue: warp 0 issues the first centroid tile's MMA
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    ab_pipeline.producer_acquire(ab_prod)
                    bar = ab_pipeline.producer_get_barrier(ab_prod)
                    cute.copy(tma_atom_a, tXgX[(None, kk)],
                              tXsX[(None, ab_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    cute.copy(tma_atom_b, tCgC2[(None, 0, kk, 0)],
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

            for dd in cutlass.range(n_db_tiles):
                if tidx < block_n:
                    sCsq[tidx] = 0.5 * mCsq[dd * block_n + tidx]

                acc_pipeline.consumer_wait(acc_cons)
                cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, 0)],
                          tTR_rAcc)
                acc_pipeline.consumer_release(acc_cons)
                acc_cons.advance()

                cute.arch.barrier()   # ½c_sq visible to all threads

                # issue the next tile's MMA now (warp 0): async tcgen05 overlaps
                # the CUDA-core-bound argmin below.
                db_next = dd + 1
                if dd + 1 < n_db_tiles:
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

                base = dd * block_n
                frag = tTR_rAcc.load()
                best_s, best_i = self._consume_tile_argmin(
                    best_s, best_i, frag, sCsq, base, block_n)
                cute.arch.barrier()

            q = bidx * BLOCK_M + tidx
            mOut[q] = best_i

            cute.arch.sync_threads()
            tmem.free(tmem_ptr)
            if warp_idx == 0:
                ab_pipeline.producer_tail(ab_prod)

    class BlackwellFlashKmeansAssignXResident:
        """X-resident tcgen05 flash-kmeans assign.

        Identical math/epilogue to :class:`BlackwellFlashKmeansAssign`, but the
        X (point) tile is TMA-loaded **once** into SMEM and reused across every
        centroid tile -- only the centroids stream through the pipeline. The
        baseline kernel re-TMA'd the whole ``BLOCK_M x D`` X tile for every
        centroid tile (inherited from the KNN build kernel), so its redundant
        X traffic grew with both ``D`` and ``K`` and dominated the compute-bound
        D=256 / large-K regime where cake pulled ahead. Loading X once removes
        that traffic; SMEM footprint is unchanged (the baseline already kept
        ``k_tile_cnt`` X sub-tiles live in its AB pipeline -- here they simply
        are not overwritten).

        Two TMA pipelines replace the baseline's single combined one:
        ``x_pipeline`` (loaded once, ``k_tile_cnt`` stages = the D sub-blocks)
        and ``c_pipeline`` (streams centroid tiles). The accumulator pipeline
        and the per-row running-argmin epilogue are unchanged.
        """

        def __init__(self, acc_dtype=cutlass.Float32, block_n: int = BLOCK_N):
            self.acc_dtype = acc_dtype
            self.block_n = block_n
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_M, block_n)
            self.threads_per_cta = BLOCK_M

        @cute.jit
        def _consume_tile_argmin(self, best_s, best_i, frag, sCsq, base, soff,
                                 BN: cutlass.Constexpr):
            for n in cutlass.range_constexpr(BN):
                s = sCsq[soff + n] - frag[n]
                lt = s < best_s
                best_s = cutlass.select_(lt, s, best_s)
                best_i = cutlass.select_(lt, cutlass.Int32(base + n), best_i)
            return best_s, best_i

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mOut: cute.Tensor, stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], MMA_K)

            D = mX.shape[1]
            # X stays resident: one stage per D sub-block holds the whole X tile.
            self.num_x_stage = max(1, D // MMA_K)
            # Centroid stream depth: >= 2 for TMA/MMA overlap, >= k_tile_cnt so a
            # full centroid tile is in flight while the next is prefetched.
            self.num_c_stage = max(2, D // MMA_K)

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,))

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.x_dtype, self.num_x_stage)
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.c_dtype_in, self.num_c_stage)

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
            self.num_tma_x_bytes = self.mma_tiler[0] * self.mma_tiler[2] * elem_bytes
            self.num_tma_c_bytes = self.mma_tiler[1] * self.mma_tiler[2] * elem_bytes
            self.num_tmem_alloc_cols = self.block_n
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            # The tmem->reg load can only move <=128 columns at once, so a wide
            # (block_n=256) accumulator is drained in 128-wide epilogue chunks.
            epi_w = min(self.mma_tiler[1], 128)
            self.epi_w = epi_w
            self.epi_n = self.mma_tiler[1] // epi_w
            self.epi_tile = (self.mma_tiler[0], epi_w)
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // BLOCK_M, 1, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mOut, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mOut, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            num_x_stage = self.num_x_stage
            num_c_stage = self.num_c_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            block_n = self.block_n

            @cute.struct
            class SharedStorage:
                x_full: cute.struct.MemRange[cutlass.Int64, num_x_stage * 2]
                c_full: cute.struct.MemRange[cutlass.Int64, num_c_stage * 2]
                acc_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, block_n]

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            x_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.x_full.data_ptr(),
                num_stages=num_x_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_x_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            x_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_x_stage)
            x_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_x_stage)

            c_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.c_full.data_ptr(),
                num_stages=num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_c_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            c_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_c_stage)
            c_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_c_stage)

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
            sCsq = storage.sCsq.get_tensor(cute.make_layout((block_n,)))

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
            epi_w = self.epi_w
            epi_n = self.epi_n
            rfrags = [
                cute.make_rmem_tensor(
                    cute.make_layout(((epi_w, 1), 1, 1)), self.acc_dtype)
                for _ in range(epi_n)
            ]

            tmem.relinquish_alloc_permit()

            best_s = cutlass.Float32(INF)
            best_i = cutlass.Int32(-1)

            # ---- load the X tile ONCE; it stays resident in sX -------------
            # Release the stages right after the load completes: releasing only
            # frees the producer-empty barrier (never reacquired here), the SMEM
            # contents persist, so every centroid tile's MMA reads them.
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.producer_acquire(x_prod)
                    bar = x_pipeline.producer_get_barrier(x_prod)
                    cute.copy(tma_atom_a, tXgX[(None, kk)],
                              tXsX[(None, x_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    x_prod.advance()
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.consumer_wait(x_cons)
                    x_pipeline.consumer_release(x_cons)
                    x_cons.advance()

            # prologue: warp 0 streams centroid tile 0 and issues its MMA
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.producer_acquire(c_prod)
                    bar = c_pipeline.producer_get_barrier(c_prod)
                    cute.copy(tma_atom_b, tCgC2[(None, 0, kk, 0)],
                              tCsC[(None, c_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    c_prod.advance()
                acc_pipeline.producer_acquire(acc_prod)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.consumer_wait(c_cons)
                    nkb = cute.size(tCrX, mode=[2])
                    for kb in cutlass.range(nkb, unroll_full=True):
                        cute.gemm(tiled_mma, tCtAcc,
                                  tCrX[(None, None, kb, kk)],
                                  tCrC[(None, None, kb, c_cons.index)], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    c_pipeline.consumer_release(c_cons)
                    c_cons.advance()
                acc_pipeline.producer_commit(acc_prod)
                acc_prod.advance()

            for dd in cutlass.range(n_db_tiles):
                # block_n may exceed the 128-thread CTA; each thread stages
                # epi_n half-tile entries of ½‖c‖² (guard for block_n<128).
                if tidx < epi_w:
                    for j in cutlass.range_constexpr(epi_n):
                        sCsq[j * epi_w + tidx] = (
                            0.5 * mCsq[dd * block_n + j * epi_w + tidx])

                acc_pipeline.consumer_wait(acc_cons)
                for j in cutlass.range_constexpr(epi_n):
                    cute.copy(tiled_copy_t2r,
                              tTR_tAcc[(None, None, None, 0, j)], rfrags[j])
                acc_pipeline.consumer_release(acc_cons)
                acc_cons.advance()

                cute.arch.barrier()   # ½c_sq visible to all threads

                db_next = dd + 1
                if dd + 1 < n_db_tiles:
                    if warp_idx == 0:
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.producer_acquire(c_prod)
                            bar = c_pipeline.producer_get_barrier(c_prod)
                            cute.copy(tma_atom_b, tCgC2[(None, db_next, kk, 0)],
                                      tCsC[(None, c_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            c_prod.advance()
                        acc_pipeline.producer_acquire(acc_prod)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.consumer_wait(c_cons)
                            nkb = cute.size(tCrX, mode=[2])
                            for kb in cutlass.range(nkb, unroll_full=True):
                                cute.gemm(tiled_mma, tCtAcc,
                                          tCrX[(None, None, kb, kk)],
                                          tCrC[(None, None, kb, c_cons.index)],
                                          tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            c_pipeline.consumer_release(c_cons)
                            c_cons.advance()
                        acc_pipeline.producer_commit(acc_prod)
                        acc_prod.advance()

                base = dd * block_n
                for j in cutlass.range_constexpr(epi_n):
                    frag = rfrags[j].load()
                    best_s, best_i = self._consume_tile_argmin(
                        best_s, best_i, frag, sCsq, base + j * epi_w,
                        j * epi_w, epi_w)
                cute.arch.barrier()

            q = bidx * BLOCK_M + tidx
            mOut[q] = best_i

            cute.arch.sync_threads()
            tmem.free(tmem_ptr)
            if warp_idx == 0:
                c_pipeline.producer_tail(c_prod)
                x_pipeline.producer_tail(x_prod)

    class BlackwellFlashKmeansAssignDual:
        """X-resident tcgen05 assign with a *double-buffered* TMEM accumulator.

        Same math/epilogue as the other variants, but two TMEM accumulators are
        kept live so the MMA warp can run two centroid tiles ahead of the
        CUDA-core argmin instead of one. The baseline/X-resident kernels use a
        single accumulator: ``MMA(dd+1)`` overlaps ``argmin(dd)``, but
        ``MMA(dd+2)`` cannot be issued until ``argmin(dd)`` frees the epilogue
        registers, so the tensor cores idle between tiles -- which is why their
        D=256 throughput plateaus (~560 TFLOP/s) while cake, which interleaves
        two score buffers, scales up to ~875 TFLOP/s.

        The centroid loop is unrolled by two: tile ``dd`` drains/refills
        accumulator A, tile ``dd+1`` drains/refills accumulator B, so at every
        point two tiles' MMAs are queued and the argmin is fully hidden under
        them. Requires an even centroid-tile count (host gates on it). X is
        loaded once (resident); only centroids stream.
        """

        def __init__(self, acc_dtype=cutlass.Float32, block_n: int = BLOCK_N):
            self.acc_dtype = acc_dtype
            self.block_n = block_n
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_M, block_n)
            self.threads_per_cta = BLOCK_M

        @cute.jit
        def _consume_tile_argmin(self, best_s, best_i, frag, sCsq, base,
                                 BN: cutlass.Constexpr):
            for n in cutlass.range_constexpr(BN):
                s = sCsq[n] - frag[n]
                lt = s < best_s
                best_s = cutlass.select_(lt, s, best_s)
                best_i = cutlass.select_(lt, cutlass.Int32(base + n), best_i)
            return best_s, best_i

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mOut: cute.Tensor, stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], MMA_K)

            D = mX.shape[1]
            self.num_x_stage = max(1, D // MMA_K)
            self.num_c_stage = max(2, D // MMA_K)

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,))

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.x_dtype, self.num_x_stage)
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.c_dtype_in, self.num_c_stage)

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
            self.num_tma_x_bytes = self.mma_tiler[0] * self.mma_tiler[2] * elem_bytes
            self.num_tma_c_bytes = self.mma_tiler[1] * self.mma_tiler[2] * elem_bytes
            self.num_tmem_alloc_cols = 2 * self.block_n
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // BLOCK_M, 1, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mOut, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mOut, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            num_x_stage = self.num_x_stage
            num_c_stage = self.num_c_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            block_n = self.block_n

            @cute.struct
            class SharedStorage:
                x_full: cute.struct.MemRange[cutlass.Int64, num_x_stage * 2]
                c_full: cute.struct.MemRange[cutlass.Int64, num_c_stage * 2]
                accA_full: cute.struct.MemRange[cutlass.Int64, 2]
                accB_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, block_n]

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            x_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.x_full.data_ptr(),
                num_stages=num_x_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_x_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            x_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_x_stage)
            x_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_x_stage)

            c_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.c_full.data_ptr(),
                num_stages=num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_c_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            c_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_c_stage)
            c_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_c_stage)

            def _make_acc_pipe(buf):
                return pipeline.PipelineUmmaAsync.create(
                    barrier_storage=buf, num_stages=1,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, self.threads_per_cta),
                    cta_layout_vmnk=None, defer_sync=True)

            accA_pipe = _make_acc_pipe(storage.accA_full.data_ptr())
            accB_pipe = _make_acc_pipe(storage.accB_full.data_ptr())
            accA_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accA_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)
            accB_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accB_cons = pipeline.make_pipeline_state(
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
            sCsq = storage.sCsq.get_tensor(cute.make_layout((block_n,)))

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
            pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAccA = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)
            tCtAccB = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)

            tXgX = tXgX[(None, bidx, None, 0)]

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, cutlass.Float32,
                self.acc_dtype, self.epi_tile, False)
            tAcc_epiA = cute.flat_divide(tCtAccA[((None, None), 0, 0)],
                                         self.epi_tile)
            tAcc_epiB = cute.flat_divide(tCtAccB[((None, None), 0, 0)],
                                         self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epiA[(None, None, 0, 0)])
            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAccA = thr_t2r.partition_S(tAcc_epiA)
            tTR_tAccB = thr_t2r.partition_S(tAcc_epiB)
            tTR_rAcc = cute.make_rmem_tensor(
                cute.make_layout(((block_n, 1), 1, 1)), self.acc_dtype)

            tmem.relinquish_alloc_permit()

            best_s = cutlass.Float32(INF)
            best_i = cutlass.Int32(-1)

            # ---- load the X tile ONCE; it stays resident in sX -------------
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.producer_acquire(x_prod)
                    bar = x_pipeline.producer_get_barrier(x_prod)
                    cute.copy(tma_atom_a, tXgX[(None, kk)],
                              tXsX[(None, x_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    x_prod.advance()
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.consumer_wait(x_cons)
                    x_pipeline.consumer_release(x_cons)
                    x_cons.advance()

            # prologue: issue tile 0 -> accA and tile 1 -> accB
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.producer_acquire(c_prod)
                    bar = c_pipeline.producer_get_barrier(c_prod)
                    cute.copy(tma_atom_b, tCgC2[(None, 0, kk, 0)],
                              tCsC[(None, c_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    c_prod.advance()
                accA_pipe.producer_acquire(accA_prod)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.consumer_wait(c_cons)
                    nkb = cute.size(tCrX, mode=[2])
                    for kb in cutlass.range(nkb, unroll_full=True):
                        cute.gemm(tiled_mma, tCtAccA,
                                  tCrX[(None, None, kb, kk)],
                                  tCrC[(None, None, kb, c_cons.index)], tCtAccA)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    c_pipeline.consumer_release(c_cons)
                    c_cons.advance()
                accA_pipe.producer_commit(accA_prod)
                accA_prod.advance()

                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.producer_acquire(c_prod)
                    bar = c_pipeline.producer_get_barrier(c_prod)
                    cute.copy(tma_atom_b, tCgC2[(None, 1, kk, 0)],
                              tCsC[(None, c_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    c_prod.advance()
                accB_pipe.producer_acquire(accB_prod)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.consumer_wait(c_cons)
                    nkb = cute.size(tCrX, mode=[2])
                    for kb in cutlass.range(nkb, unroll_full=True):
                        cute.gemm(tiled_mma, tCtAccB,
                                  tCrX[(None, None, kb, kk)],
                                  tCrC[(None, None, kb, c_cons.index)], tCtAccB)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    c_pipeline.consumer_release(c_cons)
                    c_cons.advance()
                accB_pipe.producer_commit(accB_prod)
                accB_prod.advance()

            n_pairs = n_db_tiles // 2
            for dd2 in cutlass.range(n_pairs):
                ddA = 2 * dd2
                ddB = ddA + 1

                # ---- accumulator A: centroid tile ddA -----------------------
                if tidx < block_n:
                    sCsq[tidx] = 0.5 * mCsq[ddA * block_n + tidx]
                accA_pipe.consumer_wait(accA_cons)
                cute.copy(tiled_copy_t2r, tTR_tAccA[(None, None, None, 0, 0)],
                          tTR_rAcc)
                accA_pipe.consumer_release(accA_cons)
                accA_cons.advance()
                cute.arch.barrier()
                ddA_next = ddA + 2
                if ddA_next < n_db_tiles:
                    if warp_idx == 0:
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.producer_acquire(c_prod)
                            bar = c_pipeline.producer_get_barrier(c_prod)
                            cute.copy(tma_atom_b, tCgC2[(None, ddA_next, kk, 0)],
                                      tCsC[(None, c_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            c_prod.advance()
                        accA_pipe.producer_acquire(accA_prod)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.consumer_wait(c_cons)
                            nkb = cute.size(tCrX, mode=[2])
                            for kb in cutlass.range(nkb, unroll_full=True):
                                cute.gemm(tiled_mma, tCtAccA,
                                          tCrX[(None, None, kb, kk)],
                                          tCrC[(None, None, kb, c_cons.index)],
                                          tCtAccA)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            c_pipeline.consumer_release(c_cons)
                            c_cons.advance()
                        accA_pipe.producer_commit(accA_prod)
                        accA_prod.advance()
                base = ddA * block_n
                frag = tTR_rAcc.load()
                best_s, best_i = self._consume_tile_argmin(
                    best_s, best_i, frag, sCsq, base, block_n)
                cute.arch.barrier()

                # ---- accumulator B: centroid tile ddB -----------------------
                if tidx < block_n:
                    sCsq[tidx] = 0.5 * mCsq[ddB * block_n + tidx]
                accB_pipe.consumer_wait(accB_cons)
                cute.copy(tiled_copy_t2r, tTR_tAccB[(None, None, None, 0, 0)],
                          tTR_rAcc)
                accB_pipe.consumer_release(accB_cons)
                accB_cons.advance()
                cute.arch.barrier()
                ddB_next = ddB + 2
                if ddB_next < n_db_tiles:
                    if warp_idx == 0:
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.producer_acquire(c_prod)
                            bar = c_pipeline.producer_get_barrier(c_prod)
                            cute.copy(tma_atom_b, tCgC2[(None, ddB_next, kk, 0)],
                                      tCsC[(None, c_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            c_prod.advance()
                        accB_pipe.producer_acquire(accB_prod)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.consumer_wait(c_cons)
                            nkb = cute.size(tCrX, mode=[2])
                            for kb in cutlass.range(nkb, unroll_full=True):
                                cute.gemm(tiled_mma, tCtAccB,
                                          tCrX[(None, None, kb, kk)],
                                          tCrC[(None, None, kb, c_cons.index)],
                                          tCtAccB)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            c_pipeline.consumer_release(c_cons)
                            c_cons.advance()
                        accB_pipe.producer_commit(accB_prod)
                        accB_prod.advance()
                base = ddB * block_n
                frag = tTR_rAcc.load()
                best_s, best_i = self._consume_tile_argmin(
                    best_s, best_i, frag, sCsq, base, block_n)
                cute.arch.barrier()

            q = bidx * BLOCK_M + tidx
            mOut[q] = best_i

            cute.arch.sync_threads()
            tmem.free(pool.base_ptr)
            if warp_idx == 0:
                c_pipeline.producer_tail(c_prod)
                x_pipeline.producer_tail(x_prod)

    class BlackwellFlashKmeansAssignPaired:
        """Paired-point-tile tcgen05 assign -- cake's occupancy trick in CuteDSL.

        At large D the assign kernel is SMEM-bound: one CTA's resident X tile +
        centroid pipeline is ~128 KB at D=256, so only one CTA (128 point rows)
        fits per SM and the tensor cores starve. cake instead runs **two** point
        tiles per CTA that *share* the streamed centroid tile -- the centroid
        SMEM is paid once for 256 rows, so ~256 rows/SM stay resident (matching
        cake) and each centroid byte read from L2 feeds two MMAs.

        Layout per CTA (``grid = N / 256``): rows ``[256·b, 256·b+128)`` -> tile
        A, ``[256·b+128, 256·b+256)`` -> tile B. Both X half-tiles are loaded
        once (resident, ``2·k_tile_cnt`` stages). For each centroid tile the
        centroids are TMA'd once into the shared C pipeline and consumed by both
        ``accA = xA·Cᵀ`` and ``accB = xB·Cᵀ`` (two TMEM accumulators) before the
        stage is released. ``½‖c‖²`` is shared by both tiles' argmins. Requires
        ``N % 256 == 0``.
        """

        def __init__(self, acc_dtype=cutlass.Float32, block_n: int = BLOCK_N):
            self.acc_dtype = acc_dtype
            self.block_n = block_n
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_M, block_n)
            self.threads_per_cta = BLOCK_M

        @cute.jit
        def _consume_tile_argmin(self, best_s, best_i, frag, sCsq, base,
                                 BN: cutlass.Constexpr):
            for n in cutlass.range_constexpr(BN):
                s = sCsq[n] - frag[n]
                lt = s < best_s
                best_s = cutlass.select_(lt, s, best_s)
                best_i = cutlass.select_(lt, cutlass.Int32(base + n), best_i)
            return best_s, best_i

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mOut: cute.Tensor, stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], MMA_K)

            D = mX.shape[1]
            k_tile_cnt = D // MMA_K
            # X for BOTH point tiles stays resident (k stages each).
            self.num_x_stage = 2 * k_tile_cnt
            self.num_c_stage = max(2, k_tile_cnt)

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,))

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.x_dtype, self.num_x_stage)
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.c_dtype_in, self.num_c_stage)

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
            self.num_tma_x_bytes = self.mma_tiler[0] * self.mma_tiler[2] * elem_bytes
            self.num_tma_c_bytes = self.mma_tiler[1] * self.mma_tiler[2] * elem_bytes
            self.num_tmem_alloc_cols = 2 * self.block_n
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // (2 * BLOCK_M), 1, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mOut, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mOut, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            num_x_stage = self.num_x_stage
            num_c_stage = self.num_c_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            if warp_idx == 0:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            block_n = self.block_n

            @cute.struct
            class SharedStorage:
                x_full: cute.struct.MemRange[cutlass.Int64, num_x_stage * 2]
                c_full: cute.struct.MemRange[cutlass.Int64, num_c_stage * 2]
                accA_full: cute.struct.MemRange[cutlass.Int64, 2]
                accB_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, block_n]

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            x_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.x_full.data_ptr(),
                num_stages=num_x_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_x_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            x_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_x_stage)
            x_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_x_stage)

            c_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.c_full.data_ptr(),
                num_stages=num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_c_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            c_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_c_stage)
            c_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_c_stage)

            def _make_acc_pipe(buf):
                return pipeline.PipelineUmmaAsync.create(
                    barrier_storage=buf, num_stages=1,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, self.threads_per_cta),
                    cta_layout_vmnk=None, defer_sync=True)

            accA_pipe = _make_acc_pipe(storage.accA_full.data_ptr())
            accB_pipe = _make_acc_pipe(storage.accB_full.data_ptr())
            accA_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accA_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)
            accB_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accB_cons = pipeline.make_pipeline_state(
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
            sCsq = storage.sCsq.get_tensor(cute.make_layout((block_n,)))

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
            pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAccA = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)
            tCtAccB = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)

            tXgX_A = tXgX[(None, 2 * bidx, None, 0)]
            tXgX_B = tXgX[(None, 2 * bidx + 1, None, 0)]

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, cutlass.Float32,
                self.acc_dtype, self.epi_tile, False)
            tAcc_epiA = cute.flat_divide(tCtAccA[((None, None), 0, 0)],
                                         self.epi_tile)
            tAcc_epiB = cute.flat_divide(tCtAccB[((None, None), 0, 0)],
                                         self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epiA[(None, None, 0, 0)])
            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAccA = thr_t2r.partition_S(tAcc_epiA)
            tTR_tAccB = thr_t2r.partition_S(tAcc_epiB)
            tTR_rAccA = cute.make_rmem_tensor(
                cute.make_layout(((block_n, 1), 1, 1)), self.acc_dtype)
            tTR_rAccB = cute.make_rmem_tensor(
                cute.make_layout(((block_n, 1), 1, 1)), self.acc_dtype)

            tmem.relinquish_alloc_permit()

            best_sA = cutlass.Float32(INF)
            best_iA = cutlass.Int32(-1)
            best_sB = cutlass.Float32(INF)
            best_iB = cutlass.Int32(-1)
            nkb = cute.size(tCrX, mode=[2])

            # ---- load both X half-tiles ONCE (resident) -------------------
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.producer_acquire(x_prod)
                    bar = x_pipeline.producer_get_barrier(x_prod)
                    cute.copy(tma_atom_a, tXgX_A[(None, kk)],
                              tXsX[(None, x_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    x_prod.advance()
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.producer_acquire(x_prod)
                    bar = x_pipeline.producer_get_barrier(x_prod)
                    cute.copy(tma_atom_a, tXgX_B[(None, kk)],
                              tXsX[(None, x_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    x_prod.advance()
                for kk in cutlass.range(2 * k_tile_cnt):
                    x_pipeline.consumer_wait(x_cons)
                    x_pipeline.consumer_release(x_cons)
                    x_cons.advance()

            # prologue: stream centroid tile 0, MMA into accA and accB
            if warp_idx == 0:
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.producer_acquire(c_prod)
                    bar = c_pipeline.producer_get_barrier(c_prod)
                    cute.copy(tma_atom_b, tCgC2[(None, 0, kk, 0)],
                              tCsC[(None, c_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    c_prod.advance()
                accA_pipe.producer_acquire(accA_prod)
                accB_pipe.producer_acquire(accB_prod)
                for kk in cutlass.range(k_tile_cnt):
                    c_pipeline.consumer_wait(c_cons)
                    for kb in cutlass.range(nkb, unroll_full=True):
                        acc_on = (kk != 0) or (kb != 0)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, acc_on)
                        cute.gemm(tiled_mma, tCtAccA,
                                  tCrX[(None, None, kb, kk)],
                                  tCrC[(None, None, kb, c_cons.index)], tCtAccA)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, acc_on)
                        cute.gemm(tiled_mma, tCtAccB,
                                  tCrX[(None, None, kb, k_tile_cnt + kk)],
                                  tCrC[(None, None, kb, c_cons.index)], tCtAccB)
                    c_pipeline.consumer_release(c_cons)
                    c_cons.advance()
                accA_pipe.producer_commit(accA_prod)
                accA_prod.advance()
                accB_pipe.producer_commit(accB_prod)
                accB_prod.advance()

            for dd in cutlass.range(n_db_tiles):
                if tidx < block_n:
                    sCsq[tidx] = 0.5 * mCsq[dd * block_n + tidx]

                base = dd * block_n
                # drain BOTH accumulators to registers first, releasing the
                # TMEM so the next tile's MMA can be issued and overlap the
                # (CUDA-core) argmin scans below.
                accA_pipe.consumer_wait(accA_cons)
                cute.copy(tiled_copy_t2r, tTR_tAccA[(None, None, None, 0, 0)],
                          tTR_rAccA)
                accA_pipe.consumer_release(accA_cons)
                accA_cons.advance()

                accB_pipe.consumer_wait(accB_cons)
                cute.copy(tiled_copy_t2r, tTR_tAccB[(None, None, None, 0, 0)],
                          tTR_rAccB)
                accB_pipe.consumer_release(accB_cons)
                accB_cons.advance()

                cute.arch.barrier()
                db_next = dd + 1
                if dd + 1 < n_db_tiles:
                    if warp_idx == 0:
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.producer_acquire(c_prod)
                            bar = c_pipeline.producer_get_barrier(c_prod)
                            cute.copy(tma_atom_b, tCgC2[(None, db_next, kk, 0)],
                                      tCsC[(None, c_prod.index)],
                                      tma_bar_ptr=bar, mcast_mask=None)
                            c_prod.advance()
                        accA_pipe.producer_acquire(accA_prod)
                        accB_pipe.producer_acquire(accB_prod)
                        for kk in cutlass.range(k_tile_cnt):
                            c_pipeline.consumer_wait(c_cons)
                            for kb in cutlass.range(nkb, unroll_full=True):
                                acc_on = (kk != 0) or (kb != 0)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, acc_on)
                                cute.gemm(tiled_mma, tCtAccA,
                                          tCrX[(None, None, kb, kk)],
                                          tCrC[(None, None, kb, c_cons.index)],
                                          tCtAccA)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, acc_on)
                                cute.gemm(tiled_mma, tCtAccB,
                                          tCrX[(None, None, kb, k_tile_cnt + kk)],
                                          tCrC[(None, None, kb, c_cons.index)],
                                          tCtAccB)
                            c_pipeline.consumer_release(c_cons)
                            c_cons.advance()
                        accA_pipe.producer_commit(accA_prod)
                        accA_prod.advance()
                        accB_pipe.producer_commit(accB_prod)
                        accB_prod.advance()

                # argmin scans overlap the MMA issued just above.
                fragA = tTR_rAccA.load()
                best_sA, best_iA = self._consume_tile_argmin(
                    best_sA, best_iA, fragA, sCsq, base, block_n)
                fragB = tTR_rAccB.load()
                best_sB, best_iB = self._consume_tile_argmin(
                    best_sB, best_iB, fragB, sCsq, base, block_n)
                cute.arch.barrier()

            qA = (2 * bidx) * BLOCK_M + tidx
            qB = (2 * bidx + 1) * BLOCK_M + tidx
            mOut[qA] = best_iA
            mOut[qB] = best_iB

            cute.arch.sync_threads()
            tmem.free(pool.base_ptr)
            if warp_idx == 0:
                c_pipeline.producer_tail(c_prod)
                x_pipeline.producer_tail(x_prod)

    class BlackwellFlashKmeansAssignWS:
        """Warp-specialised assign (cake's 3-way structure in CuteDSL).

        The single-CTA variants run the tcgen05 MMA *and* the per-row argmin on
        the same 128 threads with a CTA-wide barrier between them, so the score
        reduction cannot overlap the MMA and D=256/large-K throughput plateaus
        (~560 TFLOP/s vs cake's ~880). Here the CTA's six warps are split by
        role, exactly like cake's v15 kernel:

        * **warp 5** -- load: TMA-loads the point tile once (resident), then
          streams every centroid tile through a depth-2-tile SMEM pipeline so it
          runs ahead of the MMA (hiding TMA latency).
        * **warp 4** -- MMA: issues the tcgen05 MMA into a *two-accumulator*
          TMEM ring (tiles ``2i`` -> A, ``2i+1`` -> B), double-buffered so the
          tensor cores never stall on the epilogue.
        * **warps 0-3** -- epilogue: 128 threads, one centroid score-row each,
          drain each accumulator from TMEM and run the running argmin.

        The three roles are coupled *only* through the X / C / accumulator
        pipelines' mbarriers (no ``__syncthreads`` between roles), so load, MMA
        and argmin all run concurrently -- the tensor cores stay fed and the
        argmin is fully hidden. This is the only variant that closes the D=256
        gap to cake. Requires an even centroid-tile count (host gates on it).
        """

        def __init__(self, acc_dtype=cutlass.Float32, block_n: int = BLOCK_N):
            self.acc_dtype = acc_dtype
            self.block_n = block_n
            self.cta_group = tcgen05.CtaGroup.ONE
            self.cluster_shape_mn = (1, 1)
            self.mma_tiler_mn = (BLOCK_M, block_n)
            self.epi_threads = BLOCK_M            # warps 0-3: one row per thread
            self.threads_per_cta = BLOCK_M + 64   # + MMA warp (4) + load warp (5)

        @cute.jit
        def _consume_tile_argmin(self, best_s, best_i, frag, sCsq, base,
                                 BN: cutlass.Constexpr):
            for n in cutlass.range_constexpr(BN):
                s = sCsq[n] - frag[n]
                lt = s < best_s
                best_s = cutlass.select_(lt, s, best_s)
                best_i = cutlass.select_(lt, cutlass.Int32(base + n), best_i)
            return best_s, best_i

        @cute.jit
        def __call__(self, mX: cute.Tensor, mC: cute.Tensor, mCsq: cute.Tensor,
                     mOut: cute.Tensor, stream: cuda.CUstream):
            self.x_dtype = mX.element_type
            self.c_dtype_in = mC.element_type
            a_major = utils.LayoutEnum.from_tensor(mX).mma_major_mode()
            b_major = utils.LayoutEnum.from_tensor(mC).mma_major_mode()

            tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.x_dtype, a_major, b_major, self.acc_dtype, self.cta_group,
                self.mma_tiler_mn)
            self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], MMA_K)

            D = mX.shape[1]
            self.num_x_stage = max(1, D // MMA_K)
            # Two centroid tiles in flight so the dedicated load warp prefetches
            # tile dd+1 while the MMA warp is still consuming tile dd.
            self.num_c_stage = 2 * max(1, D // MMA_K)

            self.cluster_layout_vmnk = cute.tiled_divide(
                cute.make_layout((*self.cluster_shape_mn, 1)),
                (tiled_mma.thr_id.shape,))

            a_smem_layout = sm100_utils.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.x_dtype, self.num_x_stage)
            b_smem_layout = sm100_utils.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.c_dtype_in, self.num_c_stage)

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
            self.num_tma_x_bytes = self.mma_tiler[0] * self.mma_tiler[2] * elem_bytes
            self.num_tma_c_bytes = self.mma_tiler[1] * self.mma_tiler[2] * elem_bytes
            self.num_tmem_alloc_cols = 2 * self.block_n
            self.cta_tile_shape_mnk = (self.mma_tiler[0], self.mma_tiler[1],
                                       self.mma_tiler[2])
            self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])
            self.c_layout = utils.LayoutEnum.ROW_MAJOR

            N = mX.shape[0]
            grid = (N // BLOCK_M, 1, 1)
            self.kernel(
                tiled_mma, tma_atom_a, tma_x, tma_atom_b, tma_c,
                mCsq, mOut, self.cluster_layout_vmnk,
                a_smem_layout, b_smem_layout,
            ).launch(grid=grid, block=[self.threads_per_cta, 1, 1],
                     cluster=(*self.cluster_shape_mn, 1), stream=stream)

        @cute.kernel
        def kernel(self, tiled_mma, tma_atom_a, mX, tma_atom_b, mC,
                   mCsq, mOut, cluster_layout_vmnk,
                   a_smem_layout, b_smem_layout):
            num_x_stage = self.num_x_stage
            num_c_stage = self.num_c_stage
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()

            # Roles (cake's 3-way split): warp 5 = TMA load, warp 4 = MMA,
            # warps 0-3 = epilogue. The epilogue stays on the natural warps 0-3
            # so each thread's tcgen05 tmem load reads its own row (TMEM
            # partitions are bound to the physical warp).
            is_load = warp_idx == 5
            is_mma = warp_idx == 4
            if is_load:
                cpasync.prefetch_descriptor(tma_atom_a)
                cpasync.prefetch_descriptor(tma_atom_b)

            block_n = self.block_n

            @cute.struct
            class SharedStorage:
                x_full: cute.struct.MemRange[cutlass.Int64, num_x_stage * 2]
                c_full: cute.struct.MemRange[cutlass.Int64, num_c_stage * 2]
                accA_full: cute.struct.MemRange[cutlass.Int64, 2]
                accB_full: cute.struct.MemRange[cutlass.Int64, 2]
                tmem_dealloc: cutlass.Int64
                tmem_holding: cutlass.Int32
                sCsq: cute.struct.MemRange[cutlass.Float32, block_n]

            smem = utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)

            x_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.x_full.data_ptr(),
                num_stages=num_x_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_x_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            x_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_x_stage)
            x_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_x_stage)

            c_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.c_full.data_ptr(),
                num_stages=num_c_stage,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 1),
                tx_count=self.num_tma_c_bytes,
                cta_layout_vmnk=None, defer_sync=True)
            c_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, num_c_stage)
            c_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, num_c_stage)

            def _make_acc_pipe(buf):
                return pipeline.PipelineUmmaAsync.create(
                    barrier_storage=buf, num_stages=1,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread),
                    consumer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread, self.epi_threads),
                    cta_layout_vmnk=None, defer_sync=True)

            accA_pipe = _make_acc_pipe(storage.accA_full.data_ptr())
            accB_pipe = _make_acc_pipe(storage.accB_full.data_ptr())
            accA_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accA_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)
            accB_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1)
            accB_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1)

            # consumer-only barrier (warps 1-4) for ½‖c‖² staging visibility
            epi_bar = pipeline.NamedBarrier(
                barrier_id=2, num_threads=self.epi_threads)

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
            sCsq = storage.sCsq.get_tensor(cute.make_layout((block_n,)))

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
            pool = tmem.reserve(self.num_tmem_alloc_cols)
            tCtAccA = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)
            tCtAccB = pool.allocate_tensor(tCtAcc_fake.layout, self.acc_dtype)

            tXgX = tXgX[(None, bidx, None, 0)]

            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk, self.c_layout, cutlass.Float32,
                self.acc_dtype, self.epi_tile, False)
            tAcc_epiA = cute.flat_divide(tCtAccA[((None, None), 0, 0)],
                                         self.epi_tile)
            tAcc_epiB = cute.flat_divide(tCtAccB[((None, None), 0, 0)],
                                         self.epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_epiA[(None, None, 0, 0)])
            # epilogue warps 0-3: thread tidx owns row tidx (natural mapping)
            thr_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAccA = thr_t2r.partition_S(tAcc_epiA)
            tTR_tAccB = thr_t2r.partition_S(tAcc_epiB)
            tTR_rAcc = cute.make_rmem_tensor(
                cute.make_layout(((block_n, 1), 1, 1)), self.acc_dtype)

            tmem.relinquish_alloc_permit()

            best_s = cutlass.Float32(INF)
            best_i = cutlass.Int32(-1)
            nkb = cute.size(tCrX, mode=[2])
            n_pairs = n_db_tiles // 2

            if is_load:
                # ---- LOAD warp: TMA the point tile once (resident), then
                # stream every centroid tile (runs ahead of the MMA warp).
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.producer_acquire(x_prod)
                    bar = x_pipeline.producer_get_barrier(x_prod)
                    cute.copy(tma_atom_a, tXgX[(None, kk)],
                              tXsX[(None, x_prod.index)], tma_bar_ptr=bar,
                              mcast_mask=None)
                    x_prod.advance()
                for dd in cutlass.range(n_db_tiles):
                    for kk in cutlass.range(k_tile_cnt):
                        c_pipeline.producer_acquire(c_prod)
                        bar = c_pipeline.producer_get_barrier(c_prod)
                        cute.copy(tma_atom_b, tCgC2[(None, dd, kk, 0)],
                                  tCsC[(None, c_prod.index)], tma_bar_ptr=bar,
                                  mcast_mask=None)
                        c_prod.advance()
                c_pipeline.producer_tail(c_prod)
                x_pipeline.producer_tail(x_prod)
            elif is_mma:
                # ---- MMA warp: free the resident-X barriers, then MMA each
                # centroid tile into the two-accumulator ring (A/B).
                for kk in cutlass.range(k_tile_cnt):
                    x_pipeline.consumer_wait(x_cons)
                    x_pipeline.consumer_release(x_cons)
                    x_cons.advance()
                for dd2 in cutlass.range(n_pairs):
                    accA_pipe.producer_acquire(accA_prod)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range(k_tile_cnt):
                        c_pipeline.consumer_wait(c_cons)
                        for kb in cutlass.range(nkb, unroll_full=True):
                            cute.gemm(tiled_mma, tCtAccA,
                                      tCrX[(None, None, kb, kk)],
                                      tCrC[(None, None, kb, c_cons.index)],
                                      tCtAccA)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        c_pipeline.consumer_release(c_cons)
                        c_cons.advance()
                    accA_pipe.producer_commit(accA_prod)
                    accA_prod.advance()

                    accB_pipe.producer_acquire(accB_prod)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kk in cutlass.range(k_tile_cnt):
                        c_pipeline.consumer_wait(c_cons)
                        for kb in cutlass.range(nkb, unroll_full=True):
                            cute.gemm(tiled_mma, tCtAccB,
                                      tCrX[(None, None, kb, kk)],
                                      tCrC[(None, None, kb, c_cons.index)],
                                      tCtAccB)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        c_pipeline.consumer_release(c_cons)
                        c_cons.advance()
                    accB_pipe.producer_commit(accB_prod)
                    accB_prod.advance()
            else:
                # ---- consumer: warps 0-3, one score-row per thread, argmin
                for dd2 in cutlass.range(n_pairs):
                    baseA = (2 * dd2) * block_n
                    if tidx < block_n:
                        sCsq[tidx] = 0.5 * mCsq[baseA + tidx]
                    accA_pipe.consumer_wait(accA_cons)
                    cute.copy(tiled_copy_t2r,
                              tTR_tAccA[(None, None, None, 0, 0)], tTR_rAcc)
                    accA_pipe.consumer_release(accA_cons)
                    accA_cons.advance()
                    epi_bar.arrive_and_wait()
                    fragA = tTR_rAcc.load()
                    best_s, best_i = self._consume_tile_argmin(
                        best_s, best_i, fragA, sCsq, baseA, block_n)
                    epi_bar.arrive_and_wait()

                    baseB = (2 * dd2 + 1) * block_n
                    if tidx < block_n:
                        sCsq[tidx] = 0.5 * mCsq[baseB + tidx]
                    accB_pipe.consumer_wait(accB_cons)
                    cute.copy(tiled_copy_t2r,
                              tTR_tAccB[(None, None, None, 0, 0)], tTR_rAcc)
                    accB_pipe.consumer_release(accB_cons)
                    accB_cons.advance()
                    epi_bar.arrive_and_wait()
                    fragB = tTR_rAcc.load()
                    best_s, best_i = self._consume_tile_argmin(
                        best_s, best_i, fragB, sCsq, baseB, block_n)
                    epi_bar.arrive_and_wait()

                if tidx < block_n:
                    mOut[bidx * BLOCK_M + tidx] = best_i

            cute.arch.sync_threads()
            tmem.free(pool.base_ptr)


# ===========================================================================
# Host driver + compiled-kernel cache.
# ===========================================================================
_ASSIGN_CACHE: dict = {}


def _cur_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def blackwell_assign_supported(N: int, D: int, K: int) -> bool:
    """True iff the Blackwell assign kernel can run this (B=1) shape."""
    if not _BW_AVAILABLE:
        return False
    return (
        D in SUPPORTED_D
        and N % BLOCK_M == 0
        and N >= BLOCK_M
        and K % BLOCK_N == 0
        and K >= BLOCK_N
    )


def _pick_variant(N: int, D: int, K: int, block_n: int) -> str:
    """Pick the assign kernel variant.

    * ``"dual"`` -- X-resident + double-buffered TMEM accumulator (MMA runs two
      centroid tiles ahead of the argmin). Keeps the tensor cores fed in the
      compute-bound large-D / large-K regime where the single-accumulator
      variants idle between tiles. Needs an even centroid-tile count.
    * ``"xres"`` -- X-resident, single accumulator. Lowest overhead; best on the
      bandwidth-bound shapes (small D, small K) where the tensor cores are not
      the bottleneck and the extra accumulator only adds TMEM pressure.
    * ``"base"`` -- the original re-load-X kernel, kept as a fallback.

    Both X-resident variants load the point tile once and stream only the
    centroids (strictly less traffic than ``"base"``).

    Measured on B200:

    * ``"ws"`` -- warp-specialised (dedicated load warp + dedicated MMA warp + 4
      epilogue warps, a two-accumulator TMEM ring with a depth-2 centroid SMEM
      pipeline). Load, MMA and argmin all run concurrently, so this is the only
      variant that closes the D=256 gap to cake (1.01-1.04x vs 1.25-1.58x for
      ``xres``) and it also *wins* at D=64/128 / large K (0.86-0.98x). Needs an
      even centroid-tile count. It carries two extra warps, so in the
      occupancy-bound regime (huge N, small D) the lighter ``xres`` with one
      more resident CTA per SM is faster.
    * ``"xres"`` -- X-resident single-CTA fallback: smallest footprint, best on
      the bandwidth/occupancy-bound shapes and for odd tile counts.

    The single-buffer wide variants (``dual``, ``paired``, ``block_n=256``)
    regress (extra TMEM/register pressure without overlap) and are kept only
    for reference.
    """
    n_db_tiles = K // block_n
    # ws is validated for the 128-wide MMA tile (its 2x64-col accumulator ring
    # mis-addresses TMEM); narrow (block_n=64, i.e. K<512) stays on xres.
    if block_n >= 128 and n_db_tiles % 2 == 0 and n_db_tiles >= 2:
        if D >= 256:
            return "ws"
        if N <= 262144:
            return "ws"
    return "xres"


def _pick_block_n(N: int, D: int, K: int) -> int:
    """Heuristic MMA N-tile (centroids per tile).

    Wider tiles amortise the per-tile TMEM-load + argmin epilogue and the
    MMA launch over more centroids, which helps the compute-bound shapes
    (large K, large D); narrow tiles keep register pressure low and win on
    the bandwidth-bound large-N / small-K shapes where occupancy already
    saturates. Must divide K. Tuned on B200 against cake.
    """
    # BN=128 wins across the board on B200 once there are enough centroid
    # tiles to amortise the wider TMEM-load/argmin epilogue (K >= 512); for
    # tiny K (256) the 64-wide tile keeps more CTAs busy. Measured vs cake.
    if K % 128 == 0 and K >= 512:
        return 128
    return 64


def blackwell_assign_euclid(
    x: torch.Tensor,                       # (B, N, D) or (N, D), bf16
    centroids: torch.Tensor,               # (B, K, D) or (K, D), bf16
    *,
    out: Optional[torch.Tensor] = None,    # (B, N) or (N,), int32
    c_sq: Optional[torch.Tensor] = None,   # (B, K) or (K,), fp32
    block_n: Optional[int] = None,         # MMA N-tile; None -> heuristic
) -> torch.Tensor:
    """Nearest-centroid ids via the Blackwell tcgen05 assign kernel.

    Returns ``out`` (same shape it was passed, or ``(B, N)`` int32). Raises
    if the shape is unsupported -- callers should gate on
    :func:`blackwell_assign_supported` and fall back to Triton otherwise.
    """
    if not _BW_AVAILABLE:
        raise RuntimeError(f"Blackwell assign unavailable: {_BW_IMPORT_ERROR!r}")

    x3 = x if x.dim() == 3 else x.unsqueeze(0)
    c3 = centroids if centroids.dim() == 3 else centroids.unsqueeze(0)
    B, N, D = x3.shape
    K = c3.shape[1]
    assert B == 1, "Blackwell assign supports B=1 only"
    assert x3.dtype == torch.bfloat16 and c3.dtype == torch.bfloat16
    if not blackwell_assign_supported(N, D, K):
        raise ValueError(f"unsupported shape N={N} D={D} K={K}")

    x2d = x3.reshape(N, D)
    c2d = c3.reshape(K, D)
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not c2d.is_contiguous():
        c2d = c2d.contiguous()

    if out is None:
        out = torch.empty((B, N), device=x3.device, dtype=torch.int32)
    out_1d = out.reshape(N)
    if not out_1d.is_contiguous():
        out_1d = out_1d.contiguous()

    if c_sq is None:
        c_sq = (c2d.float() ** 2).sum(-1).contiguous()
    else:
        c_sq = c_sq.reshape(K)
        if not c_sq.is_contiguous():
            c_sq = c_sq.contiguous()

    if block_n is None:
        block_n = _pick_block_n(N, D, K)
    if K % block_n != 0:
        block_n = 64

    # TMA wants a 3-D (M, K, L=1) view of each operand.
    x_dl = from_dlpack(x2d.unsqueeze(-1))
    c_dl = from_dlpack(c2d.unsqueeze(-1))
    cs_dl = from_dlpack(c_sq)
    out_dl = from_dlpack(out_1d)
    stream = _cur_stream()

    variant = _pick_variant(N, D, K, block_n)
    key = (N, D, K, x2d.dtype, block_n, variant)
    comp = _ASSIGN_CACHE.get(key)
    if comp is None:
        kern_cls = {
            "xres": BlackwellFlashKmeansAssignXResident,
            "dual": BlackwellFlashKmeansAssignDual,
            "paired": BlackwellFlashKmeansAssignPaired,
            "ws": BlackwellFlashKmeansAssignWS,
            "base": BlackwellFlashKmeansAssign,
        }[variant]
        comp = cute.compile(kern_cls(block_n=block_n),
                            x_dl, c_dl, cs_dl, out_dl, stream)
        _ASSIGN_CACHE[key] = comp
    comp(x_dl, c_dl, cs_dl, out_dl, stream)

    if out_1d.data_ptr() != out.data_ptr():
        out.reshape(N).copy_(out_1d)
    return out
