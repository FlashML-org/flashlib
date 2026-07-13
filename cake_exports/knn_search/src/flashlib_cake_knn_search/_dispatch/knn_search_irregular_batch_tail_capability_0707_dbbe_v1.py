"""Exact three-shape irregular-B/Q/M BF16 kNN capability cohort.

Minimum target architecture: sm_100a.  The contract-visible path is the
canonical tcgen05/TMEM D128 split-M producer followed by an exact top-10
merge into caller-owned FP32 distances and INT32 indices.  It is restricted
to the request-owned rows B2/Q96/M196608, B1/Q768/M98304, and
B1/Q1025/M65537.

The structural delta over the two proven source seeds is a single coherent
exact-shape portfolio plus a bounds-free ``partial_full`` producer for the
divisible B2 row.  Q768 keeps its measured split24/full-M route.  The odd-M
Q1025 row keeps the masked producer so the final database tile contributes
one valid row and 127 +INF sentinels.  Query-tail masking remains active for
Q96 and Q1025, and producer/merge launches stay ordered on the caller stream.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
import os
from typing import Any
from .. import _dispatch_runtime as residual_eval
from . import knn_search_mma_split_v1 as mma
from .._dispatch_runtime import detect_gpu_arch
THREADS = mma.THREADS
MERGE_THREADS = mma.MERGE_THREADS
BLOCK_Q = mma.BLOCK_Q
BLOCK_M = mma.BLOCK_M
D_STATIC = mma.D_STATIC
K_MAX = mma.K_MAX
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_general_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))

def _verify_export_ir() -> Any:
    target = os.environ.get('LOOM_KNN_IRREGULAR_BATCH_TAIL_DBBE_VERIFY_KERNEL', 'partial')
    exports = {'partial': partial_ir, 'merge_general': merge_general_ir, 'merge_stream': merge_stream_ir}
    if target not in exports:
        raise ValueError(''.join(['LOOM_KNN_IRREGULAR_BATCH_TAIL_DBBE_VERIFY_KERNEL must be one of ', format(sorted(exports), ''), ', got ', format(repr(target), '')]))
    return exports[target]
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
TARGET_SHAPE_KEYS: tuple[str, ...] = ('blind_0622_b2_q96_m196608_d128_k10', 'target_d128_q768_m98304_k10', 'blind_0622_tail_q1025_m65537_d128_k10')
_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["ksweep_q4096_m20000_d128_k64", {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["glm5_rag_q128_m131072_d256_k64", {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d33_q128_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d33_q128_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 33], ["K", 10], ["dtype", "bfloat16"], ["seed", 611005], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d80_q256_m65536_k10", {"__dict_items__": [["label", "blind_0622_dyn_d80_q256_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 80], ["K", 10], ["dtype", "bfloat16"], ["seed", 611006], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d160_q64_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d160_q64_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 160], ["K", 10], ["dtype", "bfloat16"], ["seed", 611007], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d640_q32_m32768_k10", {"__dict_items__": [["label", "blind_0622_dyn_d640_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 640], ["K", 10], ["dtype", "bfloat16"], ["seed", 611008], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_b2_q96_m196608_d128_k10", {"__dict_items__": [["label", "blind_0622_b2_q96_m196608_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 96], ["M", 196608], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611009], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_tail_q1025_m65537_d128_k10", {"__dict_items__": [["label", "blind_0622_tail_q1025_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1025], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611012], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d64_q128_m131072_k64", {"__dict_items__": [["label", "target_d64_q128_m131072_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 64], ["K", 64], ["dtype", "bfloat16"], ["seed", 611102], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d128_q768_m98304_k10", {"__dict_items__": [["label", "target_d128_q768_m98304_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 768], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611103], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d768_q64_m65536_k10", {"__dict_items__": [["label", "target_d768_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 611108], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d1024_q32_m32768_k64", {"__dict_items__": [["label", "target_d1024_q32_m32768_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 1024], ["K", 64], ["dtype", "bfloat16"], ["seed", 611110], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d2048_q8_m16384_k10", {"__dict_items__": [["label", "target_d2048_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 2048], ["K", 10], ["dtype", "bfloat16"], ["seed", 611111], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q4_m8192_k10", {"__dict_items__": [["label", "target_d4096_q4_m8192_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 8192], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611112], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q8_m16384_k10", {"__dict_items__": [["label", "target_d4096_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611113], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["residual0701_d256_q64_m262144_k64", {"__dict_items__": [["label", "residual0701_d256_q64_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 613001], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}]]}'))
TARGET_SHAPES: list[dict[str, Any]] = [_SHAPES_BY_LABEL[label] for label in TARGET_SHAPE_KEYS]
ENTRYPOINT = 'loom.examples.weave.knn_search_irregular_batch_tail_capability_0707_dbbe_v1:launch_for_eval'
SEED_ID = 'residual_full198_irregular_batch_main_tail_split_capability_dbbe'
ROUTE_B2_Q96 = 'dbbe_irregular_b2_q96_m196608_split_wave_fullm_tcgen05'
ROUTE_Q768 = 'dbbe_irregular_q768_m98304_split24_fullm_tcgen05'
ROUTE_Q1025_TAIL = 'dbbe_irregular_q1025_m65537_split16_masked_tail_tcgen05'
PARENT_B2_TAIL_ENTRYPOINT = 'loom.examples.weave.knn_search_floor13_b2_self_tail_k10_0622_e472_v1:launch_for_eval'
PARENT_Q768_ENTRYPOINT = 'loom.examples.weave.knn_search_q768_m98304_k10_0623_0e8d_v1:launch_for_eval'
_B2_KEY = (2, 96, 196608, 128, 10, False)
_Q768_KEY = (1, 768, 98304, 128, 10, False)
_Q1025_KEY = (1, 1025, 65537, 128, 10, False)
_POLICIES: dict[tuple[int, int, int, int, int, bool], dict[str, Any]] = {_B2_KEY: {'shape_key': TARGET_SHAPE_KEYS[0], 'guard_id': 'dbbe_b2_q96_m196608_d128_k10_exact', 'guard': 'B == 2 and Q == 96 and M == 196608 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_B2_Q96, 'partial_key': 'partial_full', 'merge_key': 'merge', 'parent_entrypoint': PARENT_B2_TAIL_ENTRYPOINT, 'tail': 'Q96 masks q_abs>=96; M is exactly 1536 full M128 tiles'}, _Q768_KEY: {'shape_key': TARGET_SHAPE_KEYS[1], 'guard_id': 'dbbe_q768_m98304_d128_k10_exact', 'guard': 'B == 1 and Q == 768 and M == 98304 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_Q768, 'partial_key': 'partial_full', 'merge_key': 'merge_stream', 'parent_entrypoint': PARENT_Q768_ENTRYPOINT, 'tail': 'Q and M are exact multiples of Q128/M128'}, _Q1025_KEY: {'shape_key': TARGET_SHAPE_KEYS[2], 'guard_id': 'dbbe_q1025_m65537_d128_k10_exact', 'guard': 'B == 1 and Q == 1025 and M == 65537 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_Q1025_TAIL, 'partial_key': 'partial', 'merge_key': 'merge_stream', 'parent_entrypoint': PARENT_B2_TAIL_ENTRYPOINT, 'tail': 'Q1025 masks 127 query rows; M65537 masks 127 database rows in the final M128 tile'}}
SHAPE_DISPATCH_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "blind_0622_b2_q96_m196608_d128_k10"], ["guard", "B == 2 and Q == 96 and M == 196608 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "dbbe_irregular_b2_q96_m196608_split_wave_fullm_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_irregular_batch_tail_capability_0707_dbbe_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_floor13_b2_self_tail_k10_0622_e472_v1:launch_for_eval"], ["selected_seed", "residual_full198_irregular_batch_main_tail_split_capability_dbbe"], ["route_source", "shape-specific-capability-seed"], ["arch_requirement", "sm_100a"]]}, {"__dict_items__": [["shape_key", "target_d128_q768_m98304_k10"], ["guard", "B == 1 and Q == 768 and M == 98304 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "dbbe_irregular_q768_m98304_split24_fullm_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_irregular_batch_tail_capability_0707_dbbe_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_q768_m98304_k10_0623_0e8d_v1:launch_for_eval"], ["selected_seed", "residual_full198_irregular_batch_main_tail_split_capability_dbbe"], ["route_source", "shape-specific-capability-seed"], ["arch_requirement", "sm_100a"]]}, {"__dict_items__": [["shape_key", "blind_0622_tail_q1025_m65537_d128_k10"], ["guard", "B == 1 and Q == 1025 and M == 65537 and D == 128 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}"], ["route", "dbbe_irregular_q1025_m65537_split16_masked_tail_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_irregular_batch_tail_capability_0707_dbbe_v1:launch_for_eval"], ["source_entrypoint", "loom.examples.weave.knn_search_floor13_b2_self_tail_k10_0622_e472_v1:launch_for_eval"], ["selected_seed", "residual_full198_irregular_batch_main_tail_split_capability_dbbe"], ["route_source", "shape-specific-capability-seed"], ["arch_requirement", "sm_100a"]]}]}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _forced_fallback(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('force_fallback', False))

def _supported_arch() -> bool:
    return detect_gpu_arch() in {'sm_100a', 'sm_103a'}

def _policy(inputs: dict[str, Any]) -> dict[str, Any] | None:
    if _forced_fallback(inputs) or not _supported_arch():
        return None
    return _POLICIES.get(_shape_key(inputs))

def _split_m(inputs: dict[str, Any]) -> int:
    key = _shape_key(inputs)
    if key == _B2_KEY:
        return 74 if detect_gpu_arch() == 'sm_103a' else 72
    if key == _Q768_KEY:
        return 24
    if key == _Q1025_KEY:
        return 16
    raise ValueError(''.join(['unsupported irregular capability shape ', format(repr(key), '')]))

def selected_route(inputs: dict[str, Any]) -> str:
    policy = _policy(inputs)
    return str(policy['route']) if policy is not None else 'dbbe_irregular_exact_guard_miss'

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def _workspace_bytes(inputs: dict[str, Any], split_m: int, num_q_tiles: int) -> int:
    elements = int(inputs['B']) * num_q_tiles * split_m * BLOCK_Q * K_MAX
    return elements * (4 + 4)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    policy = _policy(inputs)
    if policy is None:
        return {'route': 'dbbe_irregular_exact_guard_miss', 'selected_route': 'dbbe_irregular_exact_guard_miss', 'route_kind': 'unsupported', 'route_source': 'shape-specific-capability-seed', 'coverage_class': 'guard_miss', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'forced_fallback': _forced_fallback(inputs), 'missing_weave_route': True, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False}
    split_m = _split_m(inputs)
    num_q_tiles = math.ceil(int(inputs['Q']) / BLOCK_Q)
    total_m_tiles = math.ceil(int(inputs['M']) / BLOCK_M)
    return {'route': policy['route'], 'selected_route': policy['route'], 'selected_entrypoint': ENTRYPOINT, 'route_kind': 'specialized', 'route_source': 'shape-specific-capability-seed', 'coverage_class': 'irregular_batch_tail_capability', 'classification': 'seed-produced', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': [str(entry['shape_key']) for entry in SHAPE_DISPATCH_REGISTRY], 'guard_id': policy['guard_id'], 'selected_guard': policy['guard'], 'forced_fallback': False, 'fallback': None, 'missing_weave_route': False, 'selected_seed': SEED_ID, 'parent_route': policy['parent_entrypoint'], 'replaced_route': 'round34_application_wrapper', 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'process-device-stream keyed producer partial cache', 'split_m': split_m, 'num_q_tiles': num_q_tiles, 'total_m_tiles': total_m_tiles, 'tiles_per_split': math.ceil(total_m_tiles / split_m), 'partial_key': policy['partial_key'], 'merge_key': policy['merge_key'], 'workspace_bytes': _workspace_bytes(inputs, split_m, num_q_tiles), 'tail_policy': policy['tail']}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def _launch_exact(inputs: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if not mma._KNN_SEARCH_KERNELS:
        mma._KNN_SEARCH_KERNELS.update(mma._compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    split_m = _split_m(inputs)
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_dist, partial_idx = mma._scratch(inputs, split_m, num_q_tiles)
    mma._KNN_SEARCH_KERNELS[str(policy['partial_key'])].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=mma.MMA_SMEM_BYTES)
    mma._KNN_SEARCH_KERNELS[str(policy['merge_key'])].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, split_m, num_q_tiles], shared_mem=mma.MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    policy = _policy(inputs)
    if policy is None:
        raise ValueError(''.join(['irregular_batch_tail_capability_dbbe accepts only the exact request-owned shapes on sm_100a/sm_103a; got ', format(repr(_shape_key(inputs)), ''), ' on ', format(detect_gpu_arch(), '')]))
    return _launch_exact(inputs, policy)

def knn_search_compile_and_launch_irregular_batch_tail_capability(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    result = residual_eval.evaluate(launch_for_eval, shapes=TARGET_SHAPES if shapes is None else shapes, correctness=True, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
