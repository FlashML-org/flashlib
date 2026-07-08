"""Flash-KMeans Euclidean assignment guarded seed dispatcher.

Minimum architecture: sm_100a. Every production route is Weave-only and uses
existing tcgen05/TMEM seed kernels; no external runtime fallback is present.
The tcgen05/TMEM routes are not intended for sm_120a/sm_121a where ptxas
rejects tcgen05 instructions.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any
from . import flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_b23d_v2 as _non128d_d160
from . import flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1 as _non128d_v1
from . import flash_kmeans_assign_cleanroom_tcgen05_v10 as _single
from . import flash_kmeans_assign_cleanroom_tcgen05_v15 as _paired
from . import flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1 as _d64_direct
from . import flash_kmeans_assign_d112_f826_c829_full_bucket_weave_evolve_flash_kmeans_assign_a262_v1 as _d112_a262
from . import flash_kmeans_assign_d128_splitk_priority_575c_v1 as _d128_priority
from . import flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1 as _tail_pad192_f9b2
from . import flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4 as _d224_d17c
from . import flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1 as _d288_hybrid
from . import flash_kmeans_assign_d352_exactd_splitk_c95c_v2 as _d352_exactd
from . import flash_kmeans_assign_d480_splitk_k1024_eac2_v1 as _d480_eac2
from . import flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1 as _d768_priority
from . import flash_kmeans_assign_gap_pad_v1 as _gap_pad
from . import flash_kmeans_assign_highd_paired_packedpartial_gridcap160_0194_v1 as _highd_paired_gridcap160
from . import flash_kmeans_assign_highd_splitd_6fcf_v1 as _highd
from . import flash_kmeans_assign_highd_splitk_8de8_v1 as _highd_splitk
from . import flash_kmeans_assign_lowdim_e50c_v1 as _lowdim
from . import flash_kmeans_assign_microdim_hybrid_9c0d_v1 as _microdim
from . import flash_kmeans_assign_microdim_pipeline4_08f9_v4 as _microdim_pipeline4
from . import flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1 as _highd_no_padding_r63
from .flash_kmeans_assign_stream import bind_current_stream
BLOCK_N = _single.BLOCK_N
BLOCK_K = _single.BLOCK_K
FEAT_D = _single.FEAT_D
D144_D176_F9B2_DIMS = _decode_capture(_json_loads('{"__tuple__": [144, 176]}'))
SUPPORTED_DIMS = _decode_capture(_json_loads('{"__tuple__": [16, 32, 64, 48, 112, 224, 288, 352, 416, 480, 80, 96, 128, 144, 176, 160, 192, 256, 320, 384, 448, 512, 768]}'))
D64_SEED_ID = 'd64-direct-single64-1p2gap-9f2a-v1'
D112_A262_SEED_ID = _d112_a262.SEED_ID
MICRODIM_PAD64_SEED_ID = 'microdim-pad64-d64-direct-v1'
MICRODIM_HYBRID_SEED_ID = _microdim.SEED_ID
MICRODIM_PIPELINE4_SEED_ID = _microdim_pipeline4.SEED_ID
LOWDIM_E50C_SEED_ID = 'lowdim-e50c-v1'
SMALL_SEED_ID = 'small-grid-single-tile-v10'
PAIRED_SEED_ID = 'paired-large-v15'
D144_D160_D176_PAD192_F9B2_SEED_ID = 'd144-d160-d176-pad192-tail-repair-f9b2-v1'
D160_PADDED_SEED_ID = 'd160-padded_single-repeated-mma-b23d-v2'
D192_SINGLE_SEED_ID = 'd192-single-repeated-mma-v1'
D192_PAIRED_SEED_ID = 'd192-paired-repeated-mma-v1'
D256_SINGLE_SEED_ID = 'd256-single-repeated-mma-v1'
D288_PARENT_SPLITK_HYBRID_SEED_ID = _d288_hybrid.SEED_ID
HIGHD_SPLITD_SEED_ID = _highd.SEED_ID
HIGHD_SPLITK_SEED_ID = _highd_splitk.SEED_ID
HIGHD_PAIRED_PACKEDPARTIAL_SEED_ID = _highd_paired_gridcap160.SEED_ID
HIGHD_NO_PADDING_PORTFOLIO_R63_SEED_ID = _highd_no_padding_r63.SEED_ID
D768_PRIORITY_SEED_ID = _d768_priority.SEED_ID
D480_EAC2_SEED_ID = _d480_eac2.SEED_ID
GAP_PAD_SEED_ID = _gap_pad.SEED_ID
D352_EXACTD_SEED_ID = _d352_exactd.SEED_ID
D224_D17C_SEED_ID = 'flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4'
D64_ROUTE_ID = 'd64_direct_single64_1p2gap_9f2a_v1'
D112_A262_ROUTE_ID = _d112_a262.ROUTE_ID
D112_A262_LIVE_REBASE_ARTIFACT = 'artifacts/generalize_auto_tuning/generalize-auto-tuning-flash-kmeans-assign-bf8f/dispatcher_consumption/live_incumbent_rebase.json'
D112_A262_LIVE_REBASE_MEASUREMENT_SESSION_ID = 'd112-a262-live-rebase-3673c9957e9a4fba9b59c8b0a4ee076e'
D112_A262_SELECTED_SHAPE_KEYS = frozenset({(2, 2048, 112, 512), (1, 512, 112, 8192), (1, 256, 112, 256), (2, 512, 112, 8192), (1, 384, 112, 4096), (3, 768, 112, 8192)})
D112_A262_RETAINED_SHAPE_KEYS = frozenset(_d112_a262.SHAPE_KEYS) - D112_A262_SELECTED_SHAPE_KEYS
if len(D112_A262_SELECTED_SHAPE_KEYS) != 6 or len(D112_A262_RETAINED_SHAPE_KEYS) != 8:
    raise RuntimeError('D112 a262 additive ownership must preserve the measured 6/8 partition')
MICRODIM_PAD64_ROUTE_ID = 'microdim_pad64_d64_direct_v1'
MICRODIM_HYBRID_ROUTE_ID = _microdim.ROUTE_ID
MICRODIM_PIPELINE4_ROUTE_ID = _microdim_pipeline4.ROUTE_ID
LOWDIM_E50C_ROUTE_ID = 'lowdim_e50c_v1'
GAP_PAD_ROUTE_ID = _gap_pad.ROUTE_ID
D352_EXACTD_ROUTE_ID = _d352_exactd.ROUTE_ID
D224_D17C_ROUTE_ID = 'd224_tmem_abi_repair_d17c_v4'
SMALL_ROUTE_ID = 'small_grid_single_tile_v10'
PAIRED_ROUTE_ID = 'paired_large_v15'
D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID = 'd128_even_near_floor_v10_repair'
D128_PRIORITY_SPLITK_ROUTE_ID = _d128_priority.ROUTE_ID
D144_D160_D176_PAD192_F9B2_ROUTE_ID = 'd144_d160_d176_pad192_tail_repair_f9b2_v1'
D160_PADDED_ROUTE_ID = 'd160_padded_single_repeated_mma_v2'
D192_SINGLE_ROUTE_ID = 'd192_single_repeated_mma_v1'
D192_PAIRED_ROUTE_ID = 'd192_paired_repeated_mma_v1'
D256_SINGLE_ROUTE_ID = 'd256_single_repeated_mma_v1'
D288_PARENT_SPLITK_HYBRID_ROUTE_ID = _d288_hybrid.ROUTE_ID
HIGHD_SPLITD_ROUTE_ID = _highd.ROUTE_ID
HIGHD_SPLITK_ROUTE_ID = _highd_splitk.ROUTE_ID
HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID = _highd_paired_gridcap160.ROUTE_ID
HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID = _highd_no_padding_r63.ROUTE_ID
D768_PRIORITY_ROUTE_ID = _d768_priority.ROUTE_ID
D480_EAC2_ROUTE_ID = _d480_eac2.ROUTE_ID
GENERIC_FALLBACK_ID = 'aligned_weave_v10_fallback'
UNSUPPORTED_ROUTE_ID = 'unsupported_shape'
SMALL_GRID_N_TILE_CAP = 8
SMALL_GRID_K_TILE_CAP = 2
SLOW_ROUTE_SPEEDUP_THRESHOLD = 0.98
BF16_DTYPE_NAMES = {'bfloat16', 'bf16', 'torch.bfloat16'}
D768_PRIORITY_SHAPES = frozenset({(1, 512, 4096), (1, 1024, 4096), (1, 512, 8192), (1, 1024, 8192), (1, 2048, 4096)})
MICRODIM_PAD64_SHAPES = frozenset({(3, 2432, 512, 16), (4, 1024, 512, 16), (3, 2432, 512, 32), (4, 1024, 512, 32)})
MICRODIM_PIPELINE4_SOURCE_SHAPES = frozenset({(8, 65536, 16, 512), (8, 65536, 32, 512)})
MICRODIM_PIPELINE4_SELECTED_SHAPES = MICRODIM_PIPELINE4_SOURCE_SHAPES
MICRODIM_PIPELINE4_LIVE_REBASE_ARTIFACT = 'artifacts/generalize_auto_tuning/generalize-auto-tuning-flash-kmeans-assign-a083/dispatcher_consumption/live_incumbent_rebase.json'
MICRODIM_PIPELINE4_LIVE_REBASE_MEASUREMENT_SESSION_ID = 'a083-live-rebase-80f484e6feb84482873f1cf45ed292bd'
if len(MICRODIM_PIPELINE4_SELECTED_SHAPES) != 2:
    raise RuntimeError('08f9 additive ownership must preserve the measured two-row partition')
D224_D17C_SHAPES = frozenset({(4, 1536, 256, 224)})
PRIORITY_EXPECTED_SEEDS = {128: _d128_priority.SEED_ID}
_PREPARE_LOCK = RLock()

@dataclass(frozen=True)
class RouteDecision:
    route_id: str
    entrypoint: str
    selected_seed: str
    route_kind: str
    route_source: str
    guard_id: str
    guard_condition: str
    reason: str

    def trace_row(self, *, shape_key: str | None=None, expected_seed: str | None=None, dispatcher_kernel_ms: float | None=None, shape_specific_kernel_ms: float | None=None, relative_speedup_vs_baseline: float | None=None) -> dict[str, Any]:
        return {'shape_key': shape_key, 'selected_route': self.route_id, 'selected_entrypoint': self.entrypoint, 'selected_seed': self.selected_seed, 'expected_seed': expected_seed, 'route_kind': self.route_kind, 'route_source': self.route_source, 'guard_id': self.guard_id, 'guard_condition': self.guard_condition, 'classification': _classify_route(self, expected_seed, relative_speedup_vs_baseline), 'dispatcher_kernel_ms': dispatcher_kernel_ms, 'shape_specific_kernel_ms': shape_specific_kernel_ms, 'relative_speedup_vs_baseline': relative_speedup_vs_baseline, 'reason': self.reason}

@dataclass(frozen=True)
class PreparedKMeansLaunchPlan:
    """Exact KMeans route with fully marshalled launches and owned workspace."""
    inputs: dict[str, Any]
    decision: RouteDecision
    arch: str
    shape: tuple[int, int, int, int, str]
    direct_launcher: Callable[..., Any]
    launch_count: int
    device_index: int
    stream: Any
    stream_handle: int
    timeout_ms: float | None = None

    @property
    def selected_route(self) -> str:
        return self.decision.route_id

    @property
    def launch_entrypoint(self) -> str:
        return self.decision.entrypoint

    def launch(self, *, stream: Any=None, timeout_ms: float | None=None, public_inputs_already_recorded: bool=False) -> dict[str, Any]:
        """Submit the bound launch sequence without dispatch or allocation."""
        import torch
        if not isinstance(public_inputs_already_recorded, bool):
            raise TypeError('public_inputs_already_recorded must be a bool')
        with torch.cuda.device(self.device_index):
            resolved_stream = self.stream if stream is None else stream
            resolved_handle = int(resolved_stream.cuda_stream)
            if resolved_handle != self.stream_handle:
                raise RuntimeError(''.join(['prepared Flash-KMeans plan is stream-bound: prepared on stream 0x', format(self.stream_handle, ''.join(['x'])), ', requested 0x', format(resolved_handle, ''.join(['x'])), '; prepare a separate plan inside the target torch.cuda.stream(...) context']))
            try:
                result = self.direct_launcher(self.inputs, stream=None, timeout_ms=self.timeout_ms if timeout_ms is None else timeout_ms)
            finally:
                self.direct_launcher.record_stream(resolved_stream)
                if not public_inputs_already_recorded:
                    seen: set[int] = set()
                    for value in self.inputs.values():
                        identity = id(value)
                        record_stream = getattr(value, 'record_stream', None)
                        if identity not in seen and callable(record_stream):
                            seen.add(identity)
                            record_stream(resolved_stream)
            return result

    def _launch_stream_bound_recorded(self, *, timeout_ms: float | None=None) -> dict[str, Any]:
        """Submit an internally proved fixed-stream slot without re-traversal.

        Exported stateful runtimes may call this only after validating the
        prepared stream, recording every current public tensor, and recording
        the plan's route-private launch storage at slot publication.
        """
        return self.direct_launcher(self.inputs, stream=None, timeout_ms=self.timeout_ms if timeout_ms is None else timeout_ms)
ROUTE_SMALL_V10 = RouteDecision(route_id=SMALL_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_small_grid_single_tile_v10', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 8 and K_tiles <= 2', reason='small-grid anchor uses the v10 single point-tile seed')
ROUTE_PAIRED_V15 = RouteDecision(route_id=PAIRED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v15:launch_for_eval', selected_seed=PAIRED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_paired_large_v15', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 0 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='even point-tile grids use the v15 paired point-tile seed')
ROUTE_ALIGNED_V10_FALLBACK = RouteDecision(route_id=GENERIC_FALLBACK_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='fallback', route_source='generic-weave-fallback', guard_id='guard_aligned_v10_weave_fallback', guard_condition='dtype == bfloat16 and D == 128 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 1 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='v15 paired kernel requires an even number of point tiles; v10 is the aligned Weave fallback')
ROUTE_D128_EVEN_NEAR_FLOOR_V10_REPAIR = RouteDecision(route_id=D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_v10:launch_for_eval', selected_seed=SMALL_SEED_ID, route_kind='specialized', route_source='generated-variant', guard_id='guard_d128_even_b8_n8192_k256_v10_repair', guard_condition='dtype == bfloat16 and B == 8 and D == 128 and N == 8192 and K == 256', reason='same-session post-D895 replay shows v10 beats paired v15 on the B8/N8192/K256 near-floor row')
ROUTE_D128_PRIORITY_SPLITK = RouteDecision(route_id=D128_PRIORITY_SPLITK_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d128_splitk_priority_575c_v1:launch_for_eval', selected_seed=_d128_priority.SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d128_no_padding_splitk_priority_575c_v1', guard_condition='B == 1 and D == 128 and N in [512,1024,2048] and K in [4096,8192]', reason='exact five-row D128 priority bucket consumes the 64x256 G1/R4 Split-K seed')
ROUTE_D64_DIRECT_9F2A = RouteDecision(route_id=D64_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d64_direct_1p2gap_9f2a_v1:launch_for_eval', selected_seed=D64_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d64_direct_single64_1p2gap_9f2a_v1', guard_condition='dtype == bfloat16 and D == 64 and N % 128 == 0 and K % 256 == 0', reason='D64 lowdim-tail bucket uses the 1d9f direct one-MMA tcgen05 score producer')
ROUTE_D112_A262 = RouteDecision(route_id=D112_A262_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d112_f826_c829_full_bucket_weave_evolve_flash_kmeans_assign_a262_v1:launch_for_eval', selected_seed=D112_A262_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_additive_d112_gap_pad_6_a262_bf8f_v1', guard_condition='dtype == bfloat16 and (B, N, D, K) is one of the six rows selected by the bf8f live-current-main CUPTI rebase', reason='the additive a262 portfolio owns only rows where its direct prepared GPU span cleared 0.98x of live current main; the other eight D112 rows retain the existing gap-pad route')
ROUTE_MICRODIM_PAD64 = RouteDecision(route_id=MICRODIM_PAD64_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_gap_pad_v1:launch_for_eval', selected_seed=MICRODIM_PAD64_SEED_ID, route_kind='specialized', route_source='generated-variant', guard_id='guard_microdim_pad64_d64_direct_v1', guard_condition='dtype == bfloat16 and (B,N,K,D) is one of the four retained D16/D32 K512 low-shape rows', reason='two independent same-session B200 A/B runs retained only the B3/N2432 and B4/N1024 rows for zero-padding to the existing D64 direct seed')
ROUTE_MICRODIM_HYBRID = RouteDecision(route_id=MICRODIM_HYBRID_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_microdim_hybrid_9c0d_v1:launch_for_eval', selected_seed=MICRODIM_HYBRID_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_microdim_hybrid_9c0d_v1', guard_condition='dtype == bfloat16 and D in [16, 32] and N % 128 == 0 and K % 256 == 0', reason="D16/D32 consume c92d's hybrid seed: short K=512 rows use direct staging and large/high-K rows keep 6cd2 pack+TMA")
ROUTE_MICRODIM_PIPELINE4 = RouteDecision(route_id=MICRODIM_PIPELINE4_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_microdim_pipeline4_08f9_v4:launch_for_eval', selected_seed=MICRODIM_PIPELINE4_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_additive_microdim_pipeline4_08f9_v4_a083', guard_condition='dtype == bfloat16 and (B, N, D, K) is one of the two rows selected by the a083 live-incumbent CUPTI rebase', reason='the fresh a083 three-order CUPTI ownership screen selected both exact B8/N65536/K512 D16 and D32 rows over microdim_hybrid_9c0d_v1')
ROUTE_D224_D17C = RouteDecision(route_id=D224_D17C_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d224_tmem_abi_repair_d17c_v4:launch_for_eval', selected_seed=D224_D17C_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d224_d17c_exact_b4_n1536_k256', guard_condition='dtype == bfloat16 and (B,N,K,D) == (4,1536,256,224)', reason='the exact D224 single-launch seed removes the measured two-launch gap on this one validated row')
ROUTE_LOWDIM_E50C = RouteDecision(route_id=LOWDIM_E50C_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_lowdim_e50c_v1:launch_for_eval', selected_seed=LOWDIM_E50C_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_lowdim_e50c_v1_d80_d96', guard_condition='dtype == bfloat16 and D in [80, 96] and N % 128 == 0 and K % 256 == 0', reason='D80/D96 lowdim-tail bucket uses the e50c fused pad-to-128 tcgen05 route after fdac missed the same-session floor')
ROUTE_GAP_PAD = RouteDecision(route_id=GAP_PAD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_gap_pad_v1:launch_for_eval', selected_seed=GAP_PAD_SEED_ID, route_kind='specialized', route_source='generated-variant', guard_id='guard_gap_pad_to_supported_seed_v1', guard_condition='dtype == bfloat16 and D in [48, 112, 224, 352, 416, 480] and N % 128 == 0 and K % 256 == 0', reason='between-bucket D rows zero-pad to the next supported Weave seed bucket without changing the delegated seed schedule')
ROUTE_D352_EXACTD = RouteDecision(route_id=D352_EXACTD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d352_exactd_splitk_c95c_v2:launch_for_eval', selected_seed=D352_EXACTD_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d352_exactd_splitk_c95c_v2', guard_condition='dtype == bfloat16 and D == 352 and N/64 <= 32 and K/256 >= 4', reason='exact-D352 D32-chunk Split-K seed removes the D384 pack on its validated envelope')
ROUTE_D480_EAC2 = RouteDecision(route_id=D480_EAC2_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d480_splitk_k1024_eac2_v1:launch_for_eval', selected_seed=D480_EAC2_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d480_splitk_k1024_eac2_v1', guard_condition='dtype == bfloat16 and D == 480 and N % 64 == 0 and K % 256 == 0 and N/64 <= 32 and K/256 >= 4', reason='D480 bounded point-tile K1024+ rows consume the validated tcgen05 producer/reducer seed; low-K and large-N rows retain the Weave gap-pad route')
ROUTE_D288_PARENT_SPLITK_HYBRID = RouteDecision(route_id=D288_PARENT_SPLITK_HYBRID_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d288_parent_splitk_hybrid_20260629_v1:launch_for_eval', selected_seed=D288_PARENT_SPLITK_HYBRID_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d288_parent_splitk_hybrid_20260629_v1', guard_condition='dtype == bfloat16 and D == 288 and N % 128 == 0 and K % 256 == 0 and (K <= 2048 or K >= 4096)', reason='D288 consumes the measured exact-D parent/Split-K hybrid; the high-K branch includes its CTA Split-K reduction in the dispatcher-timed path')
ROUTE_D144_D176_PAD192_F9B2 = RouteDecision(route_id=D144_D160_D176_PAD192_F9B2_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d160_pad192_tail_repair_f9b2_v1:launch_for_eval', selected_seed=D144_D160_D176_PAD192_F9B2_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d144_d176_pad192_tail_repair_f9b2_v1', guard_condition='dtype == bfloat16 and D in [144, 176] and N % 128 == 0 and K % 256 == 0', reason='D144/D176 consume the f9b2 tail-safe seed: pack D to D=192 scratch, then reuse the D192 tcgen05 path; D160 remains on the faster promoted b23d route')
ROUTE_D160_PADDED = RouteDecision(route_id=D160_PADDED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_b23d_v2:launch_for_eval', selected_seed=D160_PADDED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d160_padded_single_repeated_mma_b23d_v2', guard_condition='dtype == bfloat16 and D == 160 and N % 128 == 0 and K % 256 == 0', reason='D160 consumes the 81d5 padded-tail wrapper: pack D=160 to D=192 scratch, then reuse the D192 tcgen05 path')
ROUTE_D192_SINGLE = RouteDecision(route_id=D192_SINGLE_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D192_SINGLE_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d192_single_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 192 and N % 128 == 0 and K % 256 == 0 and (num_n_tiles % 2 != 0 or (num_n_tiles <= 8 and K_tiles <= 2))', reason='D192 small or odd point-tile grids use the b23d single repeated-MMA path')
ROUTE_D192_PAIRED = RouteDecision(route_id=D192_PAIRED_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D192_PAIRED_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d192_paired_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 192 and N % 128 == 0 and K % 256 == 0 and num_n_tiles % 2 == 0 and not (num_n_tiles <= 8 and K_tiles <= 2)', reason='D192 even point-tile grids use the b23d paired repeated-MMA path')
ROUTE_D256_SINGLE = RouteDecision(route_id=D256_SINGLE_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_cleanroom_tcgen05_non128d_splitd_v1:launch_for_eval', selected_seed=D256_SINGLE_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d256_single_repeated_mma_v1', guard_condition='dtype == bfloat16 and D == 256 and N % 128 == 0 and K % 256 == 0', reason='D256 consumes the existing b23d single repeated-MMA route; paired D256 remains a resource-repair lane')
ROUTE_HIGHD_SPLITD = RouteDecision(route_id=HIGHD_SPLITD_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_splitd_6fcf_v1:launch_for_eval', selected_seed=HIGHD_SPLITD_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_splitd_6fcf_v1', guard_condition='dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0 and not (num_n_tiles <= 16 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448)))', reason='high-D rows use the 6fcf split-D seed except where 8de8 split-K A/B showed wins')
ROUTE_HIGHD_SPLITK = RouteDecision(route_id=HIGHD_SPLITK_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_splitk_8de8_v1:launch_for_eval', selected_seed=HIGHD_SPLITK_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_splitk_8de8_v1', guard_condition='dtype == bfloat16 and D in [320, 384, 448, 512] and N % 128 == 0 and K % 256 == 0 and num_n_tiles <= 16 and (K_tiles >= 32 or (K_tiles >= 16 and D >= 448))', reason='8de8 split-K is selected only for measured low point-tile high-K high-D wins')
ROUTE_HIGHD_PAIRED_PACKEDPARTIAL = RouteDecision(route_id=HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_highd_paired_packedpartial_gridcap160_0194_v1:launch_for_eval', selected_seed=HIGHD_PAIRED_PACKEDPARTIAL_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_highd_paired_packedpartial_gridcap160_0194_v1', guard_condition='dtype == bfloat16 and B == 1 and N == 2048 and K == 4096 and D in [448, 512]', reason='round-35 same-session audit showed the paired packed-partial seed beats the exported highd_splitk route for the no-padding D448/D512 paired rows')
ROUTE_HIGHD_NO_PADDING_PORTFOLIO_R63 = RouteDecision(route_id=HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1:launch_for_eval', selected_seed=HIGHD_NO_PADDING_PORTFOLIO_R63_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_no_padding_highd_portfolio_r63_g1_d512n512_v1', guard_condition='dtype == bfloat16 and (B, N, K, D) is one of the exact no-padding high-D portfolio rows measured in R63', reason='R63 portfolio is the validated no-padding high-D path; it keeps the R52 portfolio for established rows and uses the G1/R4 streamdep child for D512/N512/K4096')
ROUTE_D768_PRIORITY = RouteDecision(route_id=D768_PRIORITY_ROUTE_ID, entrypoint='loom.examples.weave.flash_kmeans_assign_d768_no_padding_splitk_priority_d768_exact_seed_v1:launch_for_eval', selected_seed=D768_PRIORITY_SEED_ID, route_kind='specialized', route_source='shape-specific-seed', guard_id='guard_d768_no_padding_priority_exact5_incumbent', guard_condition='dtype == bfloat16 and D == 768 and (B, N, K) in [(1,512,4096), (1,1024,4096), (1,512,8192), (1,1024,8192), (1,2048,4096)]', reason='exact-D768 priority comparator route reuses the validated no-padding G1/R4 Split-K Weave seed; it closes the contract comparator gap without padding')
_ROUTE_FNS: dict[str, Callable[[dict[str, Any]], Any]] = {D64_ROUTE_ID: _d64_direct.launch_for_eval, D112_A262_ROUTE_ID: _d112_a262.launch_for_eval, MICRODIM_PAD64_ROUTE_ID: _gap_pad.launch_for_eval, MICRODIM_PIPELINE4_ROUTE_ID: _microdim_pipeline4.launch_for_eval, MICRODIM_HYBRID_ROUTE_ID: _microdim.launch_for_eval, LOWDIM_E50C_ROUTE_ID: _lowdim.launch_for_eval, D480_EAC2_ROUTE_ID: _d480_eac2.launch_for_eval, GAP_PAD_ROUTE_ID: _gap_pad.launch_for_eval, D352_EXACTD_ROUTE_ID: _d352_exactd.launch_for_eval, D224_D17C_ROUTE_ID: _d224_d17c.launch_for_eval, D288_PARENT_SPLITK_HYBRID_ROUTE_ID: _d288_hybrid.launch_for_eval, SMALL_ROUTE_ID: _single.launch_for_eval, PAIRED_ROUTE_ID: _paired.launch_for_eval, D128_PRIORITY_SPLITK_ROUTE_ID: _d128_priority.launch_for_eval, D128_EVEN_NEAR_FLOOR_V10_ROUTE_ID: _single.launch_for_eval, GENERIC_FALLBACK_ID: _single.launch_for_eval, D144_D160_D176_PAD192_F9B2_ROUTE_ID: _tail_pad192_f9b2.launch_for_eval, D160_PADDED_ROUTE_ID: _non128d_d160.launch_for_eval, D192_SINGLE_ROUTE_ID: _non128d_v1.launch_for_eval, D192_PAIRED_ROUTE_ID: _non128d_v1.launch_for_eval, D256_SINGLE_ROUTE_ID: _non128d_v1.launch_for_eval, HIGHD_SPLITD_ROUTE_ID: _highd.launch_for_eval, HIGHD_SPLITK_ROUTE_ID: _highd_splitk.launch_for_eval, HIGHD_PAIRED_PACKEDPARTIAL_ROUTE_ID: _highd_paired_gridcap160.launch_for_eval, HIGHD_NO_PADDING_PORTFOLIO_R63_ROUTE_ID: _highd_no_padding_r63.launch_for_eval, D768_PRIORITY_ROUTE_ID: _d768_priority.launch_for_eval}
_LAST_ROUTE_TRACE: list[dict[str, Any]] = []

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    """Contract-harness entrypoint for the explicit guarded portfolio."""
    decision = select_route(inputs)
    return _launch_route(inputs, decision)

def prepare_launch_plan(inputs: dict[str, Any], *, arch: str | None=None, stream: Any=None, timeout_ms: float | None=None) -> PreparedKMeansLaunchPlan:
    """Resolve and marshal a complete route once for allocation-free reuse.

    Preparation executes the selected route's host setup under launch capture.
    That eagerly creates every route-specific scratch tensor, TMA descriptor,
    compiled module, launch argument array, and nested child decision without
    submitting GPU work.  The returned plan is tied to these exact input and
    output tensors.
    """
    import torch
    from .._dispatch_runtime import detect_gpu_arch
    from .._dispatch_runtime import capture_kernel_launches
    input_device = inputs['x'].device
    device_index = input_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_index = int(device_index)
    with torch.cuda.device(device_index):
        resolved_stream = torch.cuda.current_stream(device_index) if stream is None else stream
        stream_device = getattr(resolved_stream, 'device', None)
        stream_device_index = getattr(stream_device, 'index', stream_device)
        if stream_device_index is not None and int(stream_device_index) != device_index:
            raise ValueError(''.join(['Flash-KMeans stream device ', format(stream_device_index, ''), ' does not match input device ', format(device_index, '')]))
        stream_handle = int(resolved_stream.cuda_stream)
        with torch.cuda.stream(resolved_stream):
            detected_arch = detect_gpu_arch()
            resolved_arch = detected_arch if arch is None else str(arch)
            if resolved_arch != detected_arch:
                raise ValueError(''.join(['flash_kmeans_assign launch arch must match the active device: requested ', format(resolved_arch, ''), ', detected ', format(detected_arch, '')]))
            if resolved_arch not in {'sm_100a', 'sm_103a'}:
                raise ValueError(''.join(['flash_kmeans_assign prepared routes require sm_100a or sm_103a, got ', format(resolved_arch, '')]))
            decision = select_route(inputs)
            shape = _shape_fields(inputs)
            trace = decision.trace_row(shape_key=_shape_key(inputs))
            with _PREPARE_LOCK:
                for key in tuple(inputs):
                    if key.startswith('_flash_kmeans_assign_'):
                        inputs.pop(key)
                inputs['_flash_kmeans_assign_prepared_stream_key'] = (device_index, stream_handle)
                inputs['_flash_kmeans_assign_dispatch_route'] = trace
                route_launcher = _ROUTE_FNS[decision.route_id]
                with capture_kernel_launches(stream=resolved_stream, arch=resolved_arch, inputs=inputs) as captured:
                    prepared_outputs = route_launcher(inputs)
                prepared_result = _finish_route(inputs, decision, prepared_outputs)
                direct_launcher = captured.bind(prepared_result)
                _LAST_ROUTE_TRACE[:] = [trace]
    return PreparedKMeansLaunchPlan(inputs=inputs, decision=decision, arch=resolved_arch, shape=shape, direct_launcher=direct_launcher, launch_count=direct_launcher.launch_count, device_index=device_index, stream=resolved_stream, stream_handle=stream_handle, timeout_ms=timeout_ms)

def launch_prepared(plan: PreparedKMeansLaunchPlan, *, stream: Any=None, timeout_ms: float | None=None, public_inputs_already_recorded: bool=False) -> dict[str, Any]:
    """Launch a plan returned by :func:`prepare_launch_plan`."""
    if not isinstance(plan, PreparedKMeansLaunchPlan):
        raise TypeError('plan must be returned by prepare_launch_plan')
    return plan.launch(stream=stream, timeout_ms=timeout_ms, public_inputs_already_recorded=public_inputs_already_recorded)

def launch_for_eval_forced_fallback(inputs: dict[str, Any]) -> dict[str, Any]:
    """Coverage probe entrypoint that forces the aligned Weave fallback route."""
    decision = _select_fallback_route(inputs)
    return _launch_route(inputs, decision)

def launch_for_eval_forced_d352_exactd(inputs: dict[str, Any]) -> dict[str, Any]:
    """ABI preflight entrypoint that forces the validated D352 exact-D route."""
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if not _use_d352_exactd(D=dim, num_n_tiles=n_points // BLOCK_N, k_tiles=n_clusters // BLOCK_K):
        raise ValueError('forced D352 exact-D route requires D=352, N/64 <= 32, and K/256 >= 4')
    return _launch_route(inputs, ROUTE_D352_EXACTD)

def launch_for_eval_forced_d112_a262(inputs: dict[str, Any]) -> dict[str, Any]:
    """ABI preflight entrypoint that forces the exact 14-row D112 portfolio."""
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if not _supports_d112_a262(B=bsz, N=n_points, D=dim, K=n_clusters):
        raise ValueError('forced D112 a262 route requires one of the exact 14 assigned D112 rows')
    return _launch_route(inputs, ROUTE_D112_A262)

def launch_for_eval_forced_microdim_pipeline4(inputs: dict[str, Any]) -> dict[str, Any]:
    """ABI preflight entrypoint for both source-supported 08f9 rows."""
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if not _supports_microdim_pipeline4(B=bsz, N=n_points, D=dim, K=n_clusters):
        raise ValueError('forced microdim pipeline4 requires B=8, N=65536, K=512, and D in [16, 32]')
    return _launch_route(inputs, ROUTE_MICRODIM_PIPELINE4)

def last_route_trace() -> list[dict[str, Any]]:
    return [dict(row) for row in _LAST_ROUTE_TRACE]

def select_route(inputs: dict[str, Any]) -> RouteDecision:
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    return select_route_from_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)

def select_route_from_shape(*, B: int, N: int, D: int, K: int, dtype: Any='bfloat16') -> RouteDecision:
    _validate_supported_shape(B=B, N=N, D=D, K=K, dtype=dtype)
    num_n_tiles = N // BLOCK_N
    k_tiles = K // BLOCK_K
    if D == _d64_direct.FEAT_D_PAD:
        return ROUTE_D64_DIRECT_9F2A
    if _use_microdim_pad64(B=B, N=N, D=D, K=K):
        return ROUTE_MICRODIM_PAD64
    if _use_microdim_pipeline4(B=B, N=N, D=D, K=K):
        return ROUTE_MICRODIM_PIPELINE4
    if D in _microdim.SUPPORTED_D:
        return ROUTE_MICRODIM_HYBRID
    if D == 288 and _use_d288_parent_splitk_hybrid(K=K):
        return ROUTE_D288_PARENT_SPLITK_HYBRID
    if _use_d352_exactd(D=D, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
        return ROUTE_D352_EXACTD
    if _use_d480_eac2(D=D, N=N, k_tiles=k_tiles):
        return ROUTE_D480_EAC2
    if _use_d224_d17c(B=B, N=N, D=D, K=K):
        return ROUTE_D224_D17C
    if _use_d112_a262(B=B, N=N, D=D, K=K):
        return ROUTE_D112_A262
    if D in _gap_pad.SUPPORTED_D:
        return ROUTE_GAP_PAD
    if D in {80, 96}:
        return ROUTE_LOWDIM_E50C
    if D in D144_D176_F9B2_DIMS:
        return ROUTE_D144_D176_PAD192_F9B2
    if D == 160:
        return ROUTE_D160_PADDED
    if D == 192:
        if num_n_tiles % 2 != 0 or (num_n_tiles <= SMALL_GRID_N_TILE_CAP and k_tiles <= SMALL_GRID_K_TILE_CAP):
            return ROUTE_D192_SINGLE
        return ROUTE_D192_PAIRED
    if D == 256:
        return ROUTE_D256_SINGLE
    if D == _d768_priority.TARGET_D:
        return ROUTE_D768_PRIORITY
    if _use_d128_priority_splitk(B=B, N=N, D=D, K=K):
        return ROUTE_D128_PRIORITY_SPLITK
    if _use_highd_no_padding_portfolio_r63(B=B, N=N, D=D, K=K):
        return ROUTE_HIGHD_NO_PADDING_PORTFOLIO_R63
    if _use_highd_paired_packedpartial(B=B, N=N, D=D, K=K):
        return ROUTE_HIGHD_PAIRED_PACKEDPARTIAL
    if D in _highd.SUPPORTED_DIMS:
        if _use_highd_splitk(D=D, num_n_tiles=num_n_tiles, k_tiles=k_tiles):
            return ROUTE_HIGHD_SPLITK
        return ROUTE_HIGHD_SPLITD
    if B == 8 and N == 8192 and (K == 256):
        return ROUTE_D128_EVEN_NEAR_FLOOR_V10_REPAIR
    if num_n_tiles <= SMALL_GRID_N_TILE_CAP and k_tiles <= SMALL_GRID_K_TILE_CAP:
        return ROUTE_SMALL_V10
    if num_n_tiles % 2 == 0:
        return ROUTE_PAIRED_V15
    return ROUTE_ALIGNED_V10_FALLBACK

def route_trace_for_shape(label: str, params: dict[str, Any], *, expected_seed: str | None=None, dispatcher_kernel_ms: float | None=None, shape_specific_kernel_ms: float | None=None, relative_speedup_vs_baseline: float | None=None) -> dict[str, Any]:
    try:
        decision = select_route(params)
    except ValueError as exc:
        return _unsupported_trace_row(shape_key=label, expected_seed=expected_seed, dispatcher_kernel_ms=dispatcher_kernel_ms, shape_specific_kernel_ms=shape_specific_kernel_ms, relative_speedup_vs_baseline=relative_speedup_vs_baseline, reason=str(exc))
    return decision.trace_row(shape_key=label, expected_seed=expected_seed, dispatcher_kernel_ms=dispatcher_kernel_ms, shape_specific_kernel_ms=shape_specific_kernel_ms, relative_speedup_vs_baseline=relative_speedup_vs_baseline)

def route_trace_for_shapes(shapes: list[dict[str, Any]], *, expected_seeds: dict[str, str] | None=None, per_shape_metrics: dict[str, dict[str, Any]] | None=None, baseline_metrics: dict[str, dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for shape in shapes:
        label = str(shape['label'])
        params = dict(shape.get('params', {}))
        metrics = (per_shape_metrics or {}).get(label, {})
        baseline = (baseline_metrics or {}).get(label, {})
        dispatcher_ms = _optional_float(metrics.get('kernel_ms'))
        baseline_ms = _optional_float(baseline.get('kernel_ms'))
        speedup = None
        if dispatcher_ms is not None and baseline_ms is not None and (dispatcher_ms > 0.0):
            speedup = baseline_ms / dispatcher_ms
        rows.append(route_trace_for_shape(label, params, expected_seed=(expected_seeds or {}).get(label), dispatcher_kernel_ms=dispatcher_ms, shape_specific_kernel_ms=baseline_ms, relative_speedup_vs_baseline=speedup))
    return rows

def compile_and_launch_flash_kmeans_assign_dispatcher(B: int=2, N: int=1024, K: int=512, D: int=128, *, benchmark: bool=False) -> dict[str, Any]:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    torch.manual_seed(5101)
    x = torch.randn((B, N, D), dtype=torch.bfloat16, device='cuda').contiguous()
    centroids = torch.randn((B, K, D), dtype=torch.bfloat16, device='cuda').contiguous()
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), dtype=torch.int32, device='cuda')
    inputs = {'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'x': x, 'centroids': centroids, 'x_sq': x_sq, 'c_sq': c_sq, 'out': out}
    launch_for_eval(inputs)
    ref_dist = x_sq.unsqueeze(-1) + c_sq.unsqueeze(1) - 2.0 * torch.einsum('bnd,bkd->bnk', x.float(), centroids.float())
    ref = ref_dist.clamp_min(0.0).argmin(dim=-1).to(torch.int32)
    result: dict[str, Any] = {'passed': bool(torch.equal(out, ref)), 'route_trace': last_route_trace()}
    if benchmark:
        from .._dispatch_runtime import evaluate
        result['contract_eval'] = evaluate(launch_for_eval, shapes=[{'label': ''.join(['manual_b', format(B, ''), '_n', format(N, ''), '_k', format(K, ''), '_d', format(D, '')]), 'params': {'B': B, 'N': N, 'D': D, 'K': K, 'dtype': 'bfloat16', 'seed': 5101}}])
    return result

def _launch_route(inputs: dict[str, Any], decision: RouteDecision) -> dict[str, Any]:
    trace = decision.trace_row(shape_key=_shape_key(inputs))
    inputs['_flash_kmeans_assign_dispatch_route'] = trace
    _LAST_ROUTE_TRACE[:] = [trace]
    with bind_current_stream(inputs):
        outputs = _ROUTE_FNS[decision.route_id](inputs)
    return _finish_route(inputs, decision, outputs)

def _finish_route(inputs: dict[str, Any], decision: RouteDecision, outputs: Any) -> dict[str, Any]:
    trace = decision.trace_row(shape_key=_shape_key(inputs))
    normalized = _normalize_outputs(outputs, inputs)
    if decision.route_id in {D112_A262_ROUTE_ID, MICRODIM_PIPELINE4_ROUTE_ID}:
        child_trace = normalized.get('route_trace')
        if isinstance(child_trace, dict):
            trace['child_route_trace'] = dict(child_trace)
        cluster_ids = normalized['cluster_ids']
        caller_out = inputs['out']
        trace['caller_owned_output'] = cluster_ids is caller_out or int(cluster_ids.data_ptr()) == int(caller_out.data_ptr())
        trace['caller_output_data_ptr'] = int(caller_out.data_ptr())
        trace['returned_output_data_ptr'] = int(cluster_ids.data_ptr())
    _LAST_ROUTE_TRACE[:] = [trace]
    normalized['selected_route'] = decision.route_id
    normalized['route_trace'] = trace
    return normalized

def _select_fallback_route(inputs: dict[str, Any]) -> RouteDecision:
    bsz, n_points, dim, n_clusters, dtype = _shape_fields(inputs)
    _validate_supported_shape(B=bsz, N=n_points, D=dim, K=n_clusters, dtype=dtype)
    if dim != FEAT_D:
        raise ValueError(''.join(['forced fallback is only defined for D=', format(FEAT_D, ''), ', got ', format(dim, '')]))
    return ROUTE_ALIGNED_V10_FALLBACK

def _normalize_outputs(outputs: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    if outputs is None:
        return {'cluster_ids': inputs['out']}
    if hasattr(outputs, 'shape'):
        return {'cluster_ids': outputs}
    if isinstance(outputs, dict):
        normalized = dict(outputs)
        if 'cluster_ids' not in normalized and 'out' in normalized:
            normalized['cluster_ids'] = normalized['out']
        if 'cluster_ids' in normalized:
            return normalized
    raise TypeError("flash_kmeans_assign dispatcher route must return cluster_ids or write inputs['out']")

def _shape_fields(inputs: dict[str, Any]) -> tuple[int, int, int, int, str]:
    bsz = int(inputs['B'])
    n_points = int(inputs['N'])
    dim = int(inputs['D'])
    n_clusters = int(inputs['K'])
    dtype = _dtype_name(inputs)
    return (bsz, n_points, dim, n_clusters, dtype)

def _validate_supported_shape(*, B: int, N: int, D: int, K: int, dtype: Any) -> None:
    dtype_name = str(dtype).replace('torch.', '')
    if dtype_name not in BF16_DTYPE_NAMES:
        raise ValueError(''.join(['flash_kmeans_assign_dispatcher requires bfloat16 input, got ', format(dtype, '')]))
    if D not in SUPPORTED_DIMS:
        raise ValueError(''.join(['flash_kmeans_assign_dispatcher has no Weave route for D=', format(D, ''), '; supported D values are ', format(SUPPORTED_DIMS, '')]))
    if D == _d768_priority.TARGET_D and (not _is_d768_priority_shape(B=B, N=N, K=K)):
        raise ValueError(''.join(['flash_kmeans_assign_dispatcher has a D=768 Weave route only for the validated priority shapes ', format(sorted(D768_PRIORITY_SHAPES), ''), '; got N=', format(N, ''), ', K=', format(K, '')]))
    if N % BLOCK_N != 0:
        raise ValueError(''.join(['N must be divisible by BLOCK_N=', format(BLOCK_N, ''), ', got ', format(N, '')]))
    if K % BLOCK_K != 0:
        raise ValueError(''.join(['K must be divisible by BLOCK_K=', format(BLOCK_K, ''), ', got ', format(K, '')]))

def _use_highd_splitk(*, D: int, num_n_tiles: int, k_tiles: int) -> bool:
    if num_n_tiles > 16:
        return False
    return k_tiles >= 32 or (k_tiles >= _highd_splitk.SPLITK_MIN_K_TILES and D >= 448)

def _use_d128_priority_splitk(*, B: int, N: int, D: int, K: int) -> bool:
    return B == 1 and D == _d128_priority.TARGET_D and (N in (512, 1024, 2048)) and (K in (4096, 8192))

def _use_microdim_pad64(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, K, D) in MICRODIM_PAD64_SHAPES

def _supports_microdim_pipeline4(*, B: int, N: int, D: int, K: int) -> bool:
    return (int(B), int(N), int(D), int(K)) in MICRODIM_PIPELINE4_SOURCE_SHAPES

def _use_microdim_pipeline4(*, B: int, N: int, D: int, K: int) -> bool:
    return (int(B), int(N), int(D), int(K)) in MICRODIM_PIPELINE4_SELECTED_SHAPES

def _use_d224_d17c(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, K, D) in D224_D17C_SHAPES

def _supports_d112_a262(*, B: int, N: int, D: int, K: int) -> bool:
    return _d112_a262.supports_shape(B=B, N=N, D=D, K=K)

def _use_d112_a262(*, B: int, N: int, D: int, K: int) -> bool:
    return (int(B), int(N), int(D), int(K)) in D112_A262_SELECTED_SHAPE_KEYS

def _is_d768_priority_shape(*, B: int, N: int, K: int) -> bool:
    return (B, N, K) in D768_PRIORITY_SHAPES

def _use_d288_parent_splitk_hybrid(*, K: int) -> bool:
    """Keep unsupported middle-K values on the existing gap-pad route."""
    return K <= 2048 or K >= 4096

def _use_d352_exactd(*, D: int, num_n_tiles: int, k_tiles: int) -> bool:
    return D == _d352_exactd.FEAT_D and num_n_tiles <= _d352_exactd.MAX_POINT_TILES and (k_tiles >= 4)

def _use_d480_eac2(*, D: int, N: int, k_tiles: int) -> bool:
    """Apply the seed guard in eac2's 64-row point-tile units."""
    return _d480_eac2._use_k1024_splitk(dim=D, num_n_tiles=N // _d480_eac2.BLOCK_N, k_tiles=k_tiles)

def _use_highd_paired_packedpartial(*, B: int, N: int, D: int, K: int) -> bool:
    return B == 1 and N == 2048 and (K == 4096) and (D in _highd_paired_gridcap160.PAIRED_PACKEDPARTIAL_DIMS)

def _use_highd_no_padding_portfolio_r63(*, B: int, N: int, D: int, K: int) -> bool:
    return (B, N, K, D) in _highd_no_padding_r63.NO_PADDING_HIGHD_SHAPES

def _classify_route(route: RouteDecision, expected_seed: str | None, relative_speedup_vs_baseline: float | None) -> str:
    if expected_seed is not None:
        if route.selected_seed != expected_seed:
            return 'guard-miss'
        if relative_speedup_vs_baseline is not None and relative_speedup_vs_baseline < SLOW_ROUTE_SPEEDUP_THRESHOLD:
            return 'fallback-slow' if route.route_kind == 'fallback' else 'kernel-slow'
        return 'seed-consumed'
    if route.route_kind == 'fallback':
        return 'coverage-only'
    return 'route-ok'

def _unsupported_trace_row(*, shape_key: str, expected_seed: str | None, dispatcher_kernel_ms: float | None, shape_specific_kernel_ms: float | None, relative_speedup_vs_baseline: float | None, reason: str) -> dict[str, Any]:
    return {'shape_key': shape_key, 'selected_route': UNSUPPORTED_ROUTE_ID, 'selected_entrypoint': 'none', 'selected_seed': None, 'expected_seed': expected_seed, 'route_kind': 'none', 'route_source': 'unknown', 'guard_id': 'guard_miss_no_supported_weave_route', 'guard_condition': 'no production Weave guard matched this shape', 'classification': 'guard-miss' if expected_seed is not None else 'unmeasured', 'dispatcher_kernel_ms': dispatcher_kernel_ms, 'shape_specific_kernel_ms': shape_specific_kernel_ms, 'relative_speedup_vs_baseline': relative_speedup_vs_baseline, 'reason': reason}

def _shape_key(inputs: dict[str, Any]) -> str:
    label = inputs.get('label')
    if label:
        return str(label)
    return ''.join(['b', format(int(inputs['B']), ''), '_n', format(int(inputs['N']), ''), '_k', format(int(inputs['K']), ''), '_d', format(int(inputs['D']), '')])

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = inputs.get('dtype')
    if dtype is not None:
        return str(dtype).replace('torch.', '')
    x = inputs.get('x')
    if x is not None and hasattr(x, 'dtype'):
        return str(x.dtype).replace('torch.', '')
    return 'bfloat16'

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
