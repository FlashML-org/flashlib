"""bbab full133 dispatcher with validated 9286 dynamic-D seed routes.

Minimum target architecture: sm_80 for inherited routes and the D5 tile-reduce
route; sm_100a for tcgen05/TMEM dynamic-D routes. This module is dispatcher
synthesis glue only: it preserves the current bbab dispatcher and overlays
validated Weave seed entrypoints for exact dynamic-D buckets.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_dispatch0618_6bea_plus_31af_bbab_v1 as parent
from . import knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1 as ext_highd
from . import knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1 as d1d5
from . import knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1 as d512_q64
from . import knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1 as d768d1024
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
SPLIT_M = parent.SPLIT_M
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10], ["MERGE_SLOTS_", 5]], "cta_group": 1, "threads": 32}'))
merge_stream_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_stream_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
merge_q128_const148_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_q128_const148_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
c08b_parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_lowq_row16_mma_partial_dispatch0610_r18_d14a_vec16_bguard_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 96000, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 512}'))
PROFILE_BBAB_BASE = parent.PROFILE_ALL
PROFILE_Q128_DYNAMIC = 'bbab_plus_9286_q128_dynamic'
PROFILE_HIGH_DYNAMIC = 'bbab_plus_9286_high_dynamic'
PROFILE_ALL = 'bbab_plus_9286_dynamic'
_VALID_PROFILES = {PROFILE_BBAB_BASE, PROFILE_Q128_DYNAMIC, PROFILE_HIGH_DYNAMIC, PROFILE_ALL}
ROUTE_D5 = d1d5.ROUTE_EXT_TINYD_D1D5_TILE_REDUCE
ROUTE_EXT_HIGHD = ext_highd.ROUTE_EXT_DYNAMIC_HIGHD_9286
ROUTE_D512_Q64 = d512_q64.ROUTE_D512_Q64_TCGEN05
ROUTE_D768D1024 = d768d1024.ROUTE_HIGH_DYNAMIC_D_TCGEN05
CONSUMED_D5_SEED = d1d5.CONSUMED_SEED
CONSUMED_EXT_HIGHD_SEED = ext_highd.CONSUMED_EXT_HIGHD_SEED
CONSUMED_D512_Q64_SEED = d512_q64.CONSUMED_SEED
CONSUMED_D768D1024_SEED = d768d1024.CONSUMED_SEED
CONSUMED_SEEDS = (*parent.CONSUMED_SEEDS, CONSUMED_D5_SEED, CONSUMED_EXT_HIGHD_SEED, CONSUMED_D512_Q64_SEED, CONSUMED_D768D1024_SEED)
D5_LABELS: tuple[str, ...] = ('blind_ext_dyn_d5_q128_m65536_k10',)
DYNAMIC_Q128_LABELS: tuple[str, ...] = (*D5_LABELS, *ext_highd.EXT_DYNAMIC_HIGHD_LABELS)
HIGH_DYNAMIC_LABELS: tuple[str, ...] = (*d512_q64.TARGET_LABELS, *d768d1024.HIGH_DYNAMIC_D_LABELS)
DYNAMIC_LABELS: tuple[str, ...] = (*DYNAMIC_Q128_LABELS, *HIGH_DYNAMIC_LABELS)
TARGET_LABELS = _decode_capture(_json_loads('{"__tuple__": ["blind_ext_dyn_d512_k64_q32_m32768", "blind_k64_q4096_m32768_d128_k64", "blind_ext_ivf_q12_m100_d64_k20", "blind_lowk_q4096_m20000_d128_k3", "blind_ext_dbscan_self_q4096_m4096_d3_k32", "blind_ext_dyn_d130_k64_q64_m65536", "blind_ext_tail_q127_m131071_d128_k10", "blind_ext_q513_m98304_d128_k10", "blind_ext_highq_q3072_m49152_d128_k10", "blind_ext_self_q3072_m3072_d128_k10", "blind_ext_dyn_d1_q128_m65536_k10", "blind_ext_k40_q128_m131072_d128", "blind_ext_k56_q128_m65536_d128", "blind_ext_k64_q4096_m49152_d128", "blind_ext_dyn_d5_q128_m65536_k10", "blind_ext_dyn_d15_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_ext_dyn_d130_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_ext_dyn_d258_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10"]}'))
TARGET_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "blind_ext_dyn_d512_k64_q32_m32768"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 512], ["K", 64], ["dtype", "bfloat16"], ["seed", 610930], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_k64_q4096_m32768_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 32768], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610507], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_ivf_q12_m100_d64_k20"], ["params", {"__dict_items__": [["B", 1], ["Q", 12], ["M", 100], ["D", 64], ["K", 20], ["dtype", "bfloat16"], ["seed", 610931], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "blind_lowk_q4096_m20000_d128_k3"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 3], ["dtype", "bfloat16"], ["seed", 610607], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dbscan_self_q4096_m4096_d3_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 3], ["K", 32], ["dtype", "bfloat16"], ["seed", 610932], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_k64_q64_m65536"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 130], ["K", 64], ["dtype", "bfloat16"], ["seed", 610929], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_tail_q127_m131071_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 127], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610901], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_q513_m98304_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 513], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610905], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_highq_q3072_m49152_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 49152], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610906], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_self_q3072_m3072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610912], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 1], ["K", 10], ["dtype", "bfloat16"], ["seed", 610913], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k40_q128_m131072_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 40], ["dtype", "bfloat16"], ["seed", 610926], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k56_q128_m65536_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 56], ["dtype", "bfloat16"], ["seed", 610927], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_k64_q4096_m49152_d128"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 49152], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610928], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d5_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 5], ["K", 10], ["dtype", "bfloat16"], ["seed", 610914], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d15_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 15], ["K", 10], ["dtype", "bfloat16"], ["seed", 610915], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d31_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 31], ["K", 10], ["dtype", "bfloat16"], ["seed", 610916], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d65_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 65], ["K", 10], ["dtype", "bfloat16"], ["seed", 610917], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d127_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 127], ["K", 10], ["dtype", "bfloat16"], ["seed", 610918], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d130_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 130], ["K", 10], ["dtype", "bfloat16"], ["seed", 610919], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d255_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 255], ["K", 10], ["dtype", "bfloat16"], ["seed", 610920], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d258_q128_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 258], ["K", 10], ["dtype", "bfloat16"], ["seed", 610921], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d512_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 512], ["K", 10], ["dtype", "bfloat16"], ["seed", 610922], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d768_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610923], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "blind_ext_dyn_d1024_q16_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 32768], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 610924], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
_D5_ENTRY: dict[str, Any] = {'overlay': '9286_d5_tile_reduce', 'shape_key': '9286_ext_dynamic_d5_q128_m65536_k10', 'labels': D5_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D == 5 and K == 10 and not self_search and not forced_fallback', 'route': ROUTE_D5, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d1d5_q128_m65536_tile_reduce_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D5_SEED, 'source_task': 'weave-evolve-knn-search-9286-d1d5-tile-reduce', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_112_c2e0.md', 'coverage_class': 'bucket_seed_ext_dynamic_d5_q128_m65536_k10', 'route_source': 'shape-specific-seed'}
_EXT_HIGHD_ENTRY: dict[str, Any] = {'overlay': '9286_ext_dynamic_highd', 'shape_key': '9286_ext_dynamic_d_q128_d15_d31_d65_d127_d130_d255_d258_k10', 'labels': ext_highd.EXT_DYNAMIC_HIGHD_LABELS, 'guard': 'B == 1 and Q == 128 and M == 65536 and D in {15,31,65,127,130,255,258} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_EXT_HIGHD, 'entrypoint': 'loom.examples.weave.knn_search_dispatch0618_c492_ext_dynamic_highd_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_EXT_HIGHD_SEED, 'source_task': 'weave-evolve ccef dynamic-D high bucket direct-stride repair', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_111_9286.md', 'coverage_class': 'generated_variant_ext_dynamic_d_highd_q128_k10', 'route_source': 'generated-variant'}
_D512_Q64_ENTRY: dict[str, Any] = {'overlay': '9286_d512_q64_directstride', 'shape_key': '9286_ext_dynamic_d512_q64_m65536_k10', 'labels': d512_q64.TARGET_LABELS, 'guard': 'B == 1 and Q == 64 and M == 65536 and D == 512 and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D512_Q64, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d512_q64_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D512_Q64_SEED, 'source_task': 'weave-evolve-knn-search-9286-d512-q64-row16-directstride', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_112_c2e0.md', 'coverage_class': 'bucket_seed_dynamic_d512_q64_m65536_k10', 'route_source': 'shape-specific-seed'}
_D768D1024_ENTRY: dict[str, Any] = {'overlay': '9286_d768d1024_directstride', 'shape_key': '9286_ext_dynamic_d768d1024_q32q16_m32768_k10', 'labels': d768d1024.HIGH_DYNAMIC_D_LABELS, 'guard': 'B == 1 and M == 32768 and (Q,D) in {(32,768),(16,1024)} and K == 10 and not self_search and not forced_fallback and arch in {sm_100a,sm_103a}', 'route': ROUTE_D768D1024, 'entrypoint': 'loom.examples.weave.knn_search_dynamic_d768d1024_q32q16_tcgen05_0618_9286_v1:launch_for_eval', 'selected_seed': CONSUMED_D768D1024_SEED, 'source_task': 'weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05', 'source_round_doc': 'design_doc/active/generalize_auto_tuning_knn_search_round_113_c1e2.md', 'coverage_class': 'bucket_seed_ext_dynamic_d768d1024_q32q16_m32768_k10', 'route_source': 'shape-specific-seed'}
_Q128_ENTRIES: tuple[dict[str, Any], ...] = (_D5_ENTRY, _EXT_HIGHD_ENTRY)
_HIGH_DYNAMIC_ENTRIES: tuple[dict[str, Any], ...] = (_D512_Q64_ENTRY, _D768D1024_ENTRY)
_ALL_ENTRIES: tuple[dict[str, Any], ...] = (*_Q128_ENTRIES, *_HIGH_DYNAMIC_ENTRIES)
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, Any], ...] = (*_ALL_ENTRIES, *parent.SHAPE_DISPATCH_REGISTRY)
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "bbab_plus_9286_q128_dynamic"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_v1:launch_q128_dynamic_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride"]], ["guard_plan", ["9286_ext_dynamic_d5_q128_m65536_k10", "9286_ext_dynamic_d_q128_d15_d31_d65_d127_d130_d255_d258_k10"]], ["fallback", "bbab full133 dispatcher"], ["expected_shape_wins", ["blind_ext_dyn_d5_q128_m65536_k10", "blind_ext_dyn_d15_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_ext_dyn_d130_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_ext_dyn_d258_q128_m65536_k10"]], ["rejected_reason", "Partial candidate; selected full-dynamic candidate also consumes high-D Q64/Q32/Q16 rows."]]}, {"__dict_items__": [["id", "bbab_plus_9286_high_dynamic"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_v1:launch_high_dynamic_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05"]], ["guard_plan", ["9286_ext_dynamic_d512_q64_m65536_k10", "9286_ext_dynamic_d768d1024_q32q16_m32768_k10"]], ["fallback", "bbab full133 dispatcher"], ["expected_shape_wins", ["blind_ext_dyn_d512_q64_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10"]], ["rejected_reason", "Partial candidate; selected full-dynamic candidate also closes Q128 high-D rows."]]}, {"__dict_items__": [["id", "bbab_plus_9286_dynamic"], ["entrypoint", "loom.examples.weave.knn_search_dispatch0618_6bea_plus_31af_bbab_9286dynamic_v1:launch_for_eval"], ["consumed_seeds", ["weave-evolve-knn-search-9286-d1d5-tile-reduce", "weave-evolve-knn-search-ccef-highd-directstride", "weave-evolve-knn-search-9286-d512-q64-row16-directstride", "weave-evolve-knn-search-9286-d768d1024-directstride-tcgen05"]], ["guard_plan", ["9286_ext_dynamic_d5_q128_m65536_k10", "9286_ext_dynamic_d_q128_d15_d31_d65_d127_d130_d255_d258_k10", "9286_ext_dynamic_d512_q64_m65536_k10", "9286_ext_dynamic_d768d1024_q32q16_m32768_k10"]], ["fallback", "bbab full133 dispatcher"], ["expected_shape_wins", ["blind_ext_dyn_d5_q128_m65536_k10", "blind_ext_dyn_d15_q128_m65536_k10", "blind_ext_dyn_d31_q128_m65536_k10", "blind_ext_dyn_d65_q128_m65536_k10", "blind_ext_dyn_d127_q128_m65536_k10", "blind_ext_dyn_d130_q128_m65536_k10", "blind_ext_dyn_d255_q128_m65536_k10", "blind_ext_dyn_d258_q128_m65536_k10", "blind_ext_dyn_d512_q64_m65536_k10", "blind_ext_dyn_d768_q32_m32768_k10", "blind_ext_dyn_d1024_q16_m32768_k10"]], ["rejected_reason", null]]}]}'))

def __getattr__(name: str) -> Any:
    return getattr(parent, name)

def _candidate_entries(profile: str) -> tuple[dict[str, Any], ...]:
    if profile == PROFILE_BBAB_BASE:
        return ()
    if profile == PROFILE_Q128_DYNAMIC:
        return _Q128_ENTRIES
    if profile == PROFILE_HIGH_DYNAMIC:
        return _HIGH_DYNAMIC_ENTRIES
    if profile == PROFILE_ALL:
        return _ALL_ENTRIES
    raise ValueError(''.join(['unknown 9286 dynamic dispatcher profile: ', format(profile, '')]))

def _guard_order(profile: str) -> list[str]:
    return [*(str(entry['shape_key']) for entry in _candidate_entries(profile)), *parent._guard_order(parent.PROFILE_ALL)]

def _use_d5(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 65536) and (int(inputs['D']) == 5) and (int(inputs['K']) == 10) and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False)))

def _entry_for_inputs(inputs: dict[str, Any], profile: str) -> dict[str, Any] | None:
    if profile not in _VALID_PROFILES:
        raise ValueError(''.join(['unknown 9286 dynamic dispatcher profile: ', format(profile, '')]))
    for entry in _candidate_entries(profile):
        if entry is _D5_ENTRY and _use_d5(inputs):
            return entry
        if entry is _EXT_HIGHD_ENTRY and ext_highd._use_ext_dynamic_highd(inputs):
            return entry
        if entry is _D512_Q64_ENTRY and d512_q64._use_d512_q64_tcgen05(inputs):
            return entry
        if entry is _D768D1024_ENTRY and d768d1024._use_high_dynamic_d_tcgen05(inputs):
            return entry
    return None

def _parent_info(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    info = dict(parent.route_info(inputs))
    route = str(info.get('selected_route') or info.get('route') or parent.selected_route(inputs))
    info['profile'] = profile
    info['route'] = route
    info['selected_route'] = route
    info['guard_order'] = _guard_order(profile)
    info.setdefault('production_policy', 'weave_only')
    info.setdefault('external_fallback', None)
    info.setdefault('coverage_only', False)
    info.setdefault('missing_weave_route', False)
    return info

def _seed_info(inputs: dict[str, Any], profile: str, entry: dict[str, Any]) -> dict[str, Any]:
    parent_info = dict(parent.route_info(inputs))
    parent_route = str(parent_info.get('selected_route') or parent_info.get('route') or parent.selected_route(inputs))
    return {'profile': profile, 'route': entry['route'], 'selected_route': entry['route'], 'selected_entrypoint': entry['entrypoint'], 'parent_route': parent_route, 'replaced_route': parent_route, 'route_kind': 'specialized', 'route_source': entry['route_source'], 'coverage_class': entry['coverage_class'], 'classification': 'seed-consumed', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'guard_order': _guard_order(profile), 'guard_id': entry['shape_key'], 'forced_fallback': False, 'selected_guard': entry['guard'], 'fallback': parent_route, 'missing_weave_route': False, 'source_task': entry['source_task'], 'source_round_doc': entry['source_round_doc'], 'selected_seed': entry['selected_seed'], 'selected_seed_task': entry['source_task'], 'expected_seed': entry['selected_seed'], 'replaced_seed': parent_info.get('selected_seed')}

def selected_route_for_profile(inputs: dict[str, Any], profile: str) -> str:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return str(entry['route'])
    return parent.selected_route(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    return selected_route_for_profile(inputs, PROFILE_ALL)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is not None:
        return _seed_info(inputs, profile, entry)
    return _parent_info(inputs, profile)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    return route_info_for_profile(inputs, PROFILE_ALL)

def route_trace_entry(label: str, inputs: dict[str, Any], profile: str=PROFILE_ALL) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info_for_profile(inputs, profile)}

def launch_for_profile(inputs: dict[str, Any], profile: str) -> dict[str, Any]:
    entry = _entry_for_inputs(inputs, profile)
    if entry is _D5_ENTRY:
        return d1d5.launch_for_eval(inputs)
    if entry is _EXT_HIGHD_ENTRY:
        return ext_highd.launch_for_eval(inputs)
    if entry is _D512_Q64_ENTRY:
        return d512_q64.launch_for_eval(inputs)
    if entry is _D768D1024_ENTRY:
        return d768d1024.launch_for_eval(inputs)
    return parent.launch_for_eval(inputs)

def launch_bbab_base_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_BBAB_BASE)

def launch_q128_dynamic_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_Q128_DYNAMIC)

def launch_high_dynamic_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_HIGH_DYNAMIC)

def launch_current_portfolio_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_for_profile(inputs, PROFILE_ALL)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return launch_current_portfolio_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return select_named_shapes(DYNAMIC_LABELS)
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_6bea_plus_31af_bbab_9286dynamic(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_LABELS) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=DYNAMIC_LABELS) -> dict[str, Any]:
    return knn_search_compile_and_launch_6bea_plus_31af_bbab_9286dynamic(benchmark=benchmark, shape_labels=shape_labels)
