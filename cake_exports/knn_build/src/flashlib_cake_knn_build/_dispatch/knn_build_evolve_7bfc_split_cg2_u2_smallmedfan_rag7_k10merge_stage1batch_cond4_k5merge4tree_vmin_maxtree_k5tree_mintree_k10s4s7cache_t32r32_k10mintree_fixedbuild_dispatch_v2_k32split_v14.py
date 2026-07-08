"""kNN build/search K32 split-build successor with q512 K8 and K32 chunked merge.

Minimum target architecture: sm_100a. This variant keeps the validated K5/K10
fixed-build dispatch from the parent and keeps the build-mode path for
``10 < K <= 32`` BF16 D=128 shapes. The larger-K path compiles the inherited
CTA-group=2 split stage-1 IR with static top-k capacity buckets of 12, 16, 20,
25, 30, and 32. The small-shape K12/K16/K20/K25/K30 buckets use a 32-thread cached
four-way cursor merge over sorted split-local streams, preserving the
split/tcgen05 contract-visible eval path while reducing scalar merge work for
exact bucket sizes. For the underfilled Q=M=512,K=30 probe, stage-1 uses eight
database splits and a specialized eight-way cached merge to expose more
independent tcgen05 work. Large Q=M=4096 K20/K30/K32 boundaries use exact
unordered split-local producers and unordered four-split merge, avoiding
sorted-stream maintenance on large-Q hot paths while preserving exact top-k
membership and matching distances. Fixed-build K1/K2/K8 shapes use the
inherited generic-K split merge instead of cached K10 merge kernels so partial
buffer row strides match the contract-visible K. Q=M=1024/2048,K=12 uses an
exact K12 producer and eight-way cached merge to increase mid-build producer
parallelism without changing q4096 K20/K30/K32 unordered routing. The v10
lineage applies the same shape-gated eight-split sorted-stream fanout to
Q=M=1024/2048,K=20 and adds an exact K20 eight-way cached merge.
This v11 candidate routes Q=M=2048,K=8 through a K8 static split/tcgen05
bucket and seven-way cached merge, and uses a shape-gated K20 fanout mix:
Q=M=1024,K=20 uses sixteen splits with an exact K20 sixteen-way cached merge,
while Q=M=2048,K=20 keeps the v10 eight-split exact merge and Q=M=4096,K=20
keeps the unordered four-split route.
This v12 candidate keeps the v11 shape-gated K20 fanout mix, routes
Q=M=2048,K=8 through eight database splits and an exact eight-way cached merge,
extends the large-Q unordered exact top-k route to Q=M=4096,K=25, and widens
the non-build K10 RAG dispatch: q4096 x 100000 and q10000 x 50000 now use the
inherited split-7 tcgen05 producer plus cached seven-way merge instead of
falling through to the older generic parent route. The v11 K32 diagnostic route
is preserved.
This v13 candidate keeps the v12 routes and tests the q512 guardrail lane by
routing Q=M=512,K=8 through the existing static K8 eight-split producer and
cached eight-way merge. K1/K2 stay on the stride-safe generic-K four-split
route. It also changes the K32 unordered merge: the first split-local
top-32 vector prefills the output accumulator, then the remaining three
split-local vectors use worst-slot replacement. This avoids the v12 merge's
repeated worst scan during the guaranteed initial fill.
This v14 candidate keeps the same K32 unordered producer and prefill semantics,
but tracks four 8-slot worst caches during the unordered merge. Accepted
candidates rescan only the affected 8-slot bucket and the four bucket maxima
instead of rescanning all 32 slots.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import os
from functools import lru_cache
from typing import Any
from .._dispatch_runtime import dc as dc
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2 as parent
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_v1 as parent_lowk
from . import knn_build_evolve_7bfc_split_cg2_u2_v1 as parent_u2
from . import knn_build_evolve_7bfc_split_v1 as parent_split
from . import knn_build_evolve_7bfc_v1 as base_v1
from .._dispatch_runtime import pack_kernel_args
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
FEAT_D = parent.FEAT_D
TOP_K_MAX = parent.TOP_K_MAX
TOP_K_SPLIT_MAX = base_v1.TOP_K_FALLBACK_MAX
STAGE1_THREADS = parent.STAGE1_THREADS
MERGE_THREADS = parent.MERGE_THREADS
K32_MERGE_THREADS = 32
GRID_DIM_DEFAULT = parent.GRID_DIM_DEFAULT
CTA_GROUP = parent.CTA_GROUP
MEDIUM_SPLITS = parent.MEDIUM_SPLITS
K12_MID_SPLITS = 8
K20_Q1024_SPLITS = 16
K20_Q2048_SPLITS = 8
K8_Q2048_SPLITS = 8
K8_MID_MERGE_THREADS = parent_lowk.parent_cached.RAG_MERGE_THREADS
K30_SMALL_SHAPE_MAX = 512
K30_SMALL_SPLITS = 8
Q512_K8_SPLITS = K8_Q2048_SPLITS
RAG_OFFLINE_Q_MIN = 4096
RAG_OFFLINE_M_MIN = 50000

def _ir_with_top_k_max(ir_obj: Any, *, top_k_max: int, suffix: str) -> Any:
    constants = tuple(((name, top_k_max if name == 'TOP_K_MAX' else value) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)

def _ir_with_constants(ir_obj: Any, *, suffix: str, **updates: int) -> Any:
    constants = tuple(((name, updates.get(name, value)) for name, value in ir_obj.constants))
    return dc.replace(ir_obj, symbol=''.join([format(ir_obj.symbol, ''), '_', format(suffix, '')]), constants=constants)
TOP_K_BUCKETS = (8, 12, 16, 20, 25, 30, TOP_K_SPLIT_MAX)
knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
stage1_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k8split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 8]], "cta_group": 1, "threads": 192}'))
stage1_k12_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k12split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 12]], "cta_group": 1, "threads": 192}'))
stage1_k16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k16split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 16]], "cta_group": 1, "threads": 192}'))
stage1_k20_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k20split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
stage1_k25_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k25split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 25]], "cta_group": 1, "threads": 192}'))
stage1_k30_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k30split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 30]], "cta_group": 1, "threads": 192}'))
stage1_k32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
stage1_k32_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
stage1_k20_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k20unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 20]], "cta_group": 1, "threads": 192}'))
stage1_k25_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k25unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 25]], "cta_group": 1, "threads": 192}'))
stage1_k30_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k30unordered", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 30]], "cta_group": 1, "threads": 192}'))
stage1_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))
knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32]], "cta_group": 1, "threads": 32}'))
merge_k12_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k12split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 12]], "cta_group": 1, "threads": 32}'))
merge_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k8split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8]], "cta_group": 1, "threads": 32}'))
merge_k16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k16split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 16]], "cta_group": 1, "threads": 32}'))
merge_k20_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k20split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20]], "cta_group": 1, "threads": 32}'))
merge_k25_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k25split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 25]], "cta_group": 1, "threads": 32}'))
merge_k30_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k30split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30]], "cta_group": 1, "threads": 32}'))
merge_k32_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k32split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32]], "cta_group": 1, "threads": 32}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_rowbase_cache_k32split", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "K", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32]], "cta_group": 1, "threads": 32}'))
knn_build_evolve_7bfc_k32_merge_s4_unordered = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
knn_build_evolve_7bfc_k32_merge_s4_unordered_prefill = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_prefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
knn_build_evolve_7bfc_k32_merge_s4_unordered_chunked_prefill = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_chunked_prefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k32_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k32_unordered_prefill_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_prefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k32_unordered_chunked_prefill_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_chunked_prefill", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 32], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k20_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k20unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k25_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k25unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 25], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
merge_k30_unordered_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k32_merge_s4_unordered_k30unordered", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 4]], "cta_group": 1, "threads": 32}'))
knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k30_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 30], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k12_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k12s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 12], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k20_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k8_s7_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k10_merge_s7_rowbase_cache_k8s7", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 7]], "cta_group": 1, "threads": 32}'))
merge_k8_s8_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k8s8", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 8], ["SPLIT_COUNT", 8]], "cta_group": 1, "threads": 32}'))
merge_k20_s16_ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_k30_merge_s8_rowbase_cache_k20s16", "arg_keys": ["partial_dists", "partial_indices", "out_dists", "out_indices", "total_queries"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["TOP_K_MAX", 20], ["SPLIT_COUNT", 16]], "cta_group": 1, "threads": 32}'))

def _top_k_bucket(top_k: int) -> int:
    for bucket in TOP_K_BUCKETS:
        if top_k <= bucket:
            return bucket
    return TOP_K_SPLIT_MAX

def _stage1_ir_for_bucket(top_k_bucket: int) -> Any:
    if top_k_bucket <= 8:
        return stage1_k8_ir
    if top_k_bucket <= 12:
        return stage1_k12_ir
    if top_k_bucket <= 16:
        return stage1_k16_ir
    if top_k_bucket <= 20:
        return stage1_k20_ir
    if top_k_bucket <= 25:
        return stage1_k25_ir
    if top_k_bucket <= 30:
        return stage1_k30_ir
    return stage1_k32_ir

def _merge_ir_for_bucket(top_k_bucket: int) -> Any:
    if top_k_bucket <= 8:
        return merge_k8_ir
    if top_k_bucket <= 12:
        return merge_k12_ir
    if top_k_bucket <= 16:
        return merge_k16_ir
    if top_k_bucket <= 20:
        return merge_k20_ir
    if top_k_bucket <= 25:
        return merge_k25_ir
    if top_k_bucket <= 30:
        return merge_k30_ir
    return merge_k32_ir

def _verify_export_ir() -> Any:
    bucket_text = os.environ.get('LOOM_KNN_K32SPLIT_VERIFY_TOP_K_BUCKET')
    top_k_bucket = TOP_K_SPLIT_MAX if bucket_text is None else _top_k_bucket(int(bucket_text))
    verify_kernel = os.environ.get('LOOM_KNN_K32SPLIT_VERIFY_KERNEL')
    if verify_kernel == 'stage1_q512_lowk':
        return parent_lowk.stage1_ir
    if verify_kernel == 'merge_generic':
        return parent_split.merge_ir
    if verify_kernel == 'stage1_k10_rag':
        return parent_lowk.stage1_ir
    if verify_kernel == 'merge_k10_s7_cache':
        return parent_lowk.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'stage1_k20_unordered':
        return stage1_k20_unordered_ir
    if verify_kernel == 'stage1_k25_unordered':
        return stage1_k25_unordered_ir
    if verify_kernel == 'stage1_k30_unordered':
        return stage1_k30_unordered_ir
    if verify_kernel == 'stage1_k32_unordered':
        return stage1_k32_unordered_ir
    if verify_kernel == 'merge_k20_unordered':
        return merge_k20_unordered_ir
    if verify_kernel == 'merge_k25_unordered':
        return merge_k25_unordered_ir
    if verify_kernel == 'merge_k30_unordered':
        return merge_k30_unordered_ir
    if verify_kernel == 'merge_k32_unordered':
        return merge_k32_unordered_ir
    if verify_kernel == 'merge_k32_unordered_prefill':
        return merge_k32_unordered_prefill_ir
    if verify_kernel == 'merge_k32_unordered_chunked_prefill':
        return merge_k32_unordered_chunked_prefill_ir
    if verify_kernel == 'merge_s8':
        return merge_k30_s8_ir
    if verify_kernel == 'merge_k12_s8':
        return merge_k12_s8_ir
    if verify_kernel == 'merge_k20_s8':
        return merge_k20_s8_ir
    if verify_kernel == 'merge_k8_s7':
        return merge_k8_s7_ir
    if verify_kernel == 'merge_k8_s8':
        return merge_k8_s8_ir
    if verify_kernel == 'merge_k20_s16':
        return merge_k20_s16_ir
    if verify_kernel == 'merge':
        return _merge_ir_for_bucket(top_k_bucket)
    return _stage1_ir_for_bucket(top_k_bucket)
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32split", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 32]], "cta_group": 1, "threads": 192}'))

def _compile_ir(ir_obj: Any):
    from .._dispatch_runtime import generate_kernel
    from .._dispatch_runtime import _cuda_include_dirs
    from .._dispatch_runtime import compile_cuda
    from .._dispatch_runtime import CUDAKernel
    source = generate_kernel(ir_obj, validate=False, smem_bytes=ir_obj.computed_smem_bytes)
    cubin = compile_cuda(source, arch=base_v1._select_arch_and_preload(), options=['--use_fast_math'], include_dirs=_cuda_include_dirs())
    return CUDAKernel(cubin, ''.join(['kernel_', format(ir_obj.symbol, '')]))

@lru_cache(maxsize=7)
def _compiled_stage1_for_bucket(top_k_bucket: int):
    return _compile_ir(_stage1_ir_for_bucket(top_k_bucket))

def _compiled_stage1_k32_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0168"}'))

@lru_cache(maxsize=4)
def _compiled_stage1_unordered_for_exact_k(top_k: int):
    return _compile_ir(_stage1_unordered_ir_for_exact_k(top_k))

@lru_cache(maxsize=7)
def _compiled_merge_for_bucket(top_k_bucket: int):
    return _compile_ir(_merge_ir_for_bucket(top_k_bucket))

def _compiled_merge_k32_unordered():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0169"}'))

@lru_cache(maxsize=4)
def _compiled_merge_unordered_for_exact_k(top_k: int):
    return _compile_ir(_merge_unordered_ir_for_exact_k(top_k))

def _compiled_merge_k30_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0170"}'))

def _compiled_merge_k12_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0171"}'))

def _compiled_merge_k20_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0172"}'))

def _compiled_merge_k8_s7():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0173"}'))

def _compiled_merge_k8_s8():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0174"}'))

def _compiled_merge_k20_s16():
    return _decode_capture(_json_loads('{"__kernel__": "dispatch_kernel_0175"}'))

def _eligible_k32_split_build(inputs: dict[str, Any]) -> bool:
    top_k = int(inputs['K'])
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (top_k == 8 or TOP_K_MAX < top_k <= TOP_K_SPLIT_MAX) and (int(inputs['Q']) == int(inputs['M'])) and (512 <= int(inputs['Q']) <= 4096)

def _stage1_unordered_ir_for_exact_k(top_k: int) -> Any:
    if top_k == 20:
        return stage1_k20_unordered_ir
    if top_k == 25:
        return stage1_k25_unordered_ir
    if top_k == 30:
        return stage1_k30_unordered_ir
    if top_k == TOP_K_SPLIT_MAX:
        return stage1_k32_unordered_ir
    raise ValueError(''.join(['no unordered stage-1 specialization for K=', format(top_k, '')]))

def _merge_unordered_ir_for_exact_k(top_k: int) -> Any:
    if top_k == 20:
        return merge_k20_unordered_ir
    if top_k == 25:
        return merge_k25_unordered_ir
    if top_k == 30:
        return merge_k30_unordered_ir
    if top_k == TOP_K_SPLIT_MAX:
        return merge_k32_unordered_chunked_prefill_ir
    raise ValueError(''.join(['no unordered merge specialization for K=', format(top_k, '')]))

def _k32_split_count(inputs: dict[str, Any]) -> int | None:
    if not _eligible_k32_split_build(inputs):
        return None
    if int(inputs['K']) == 8:
        return K8_Q2048_SPLITS if int(inputs['Q']) == 2048 else None
    if int(inputs['K']) == 12 and 1024 <= int(inputs['Q']) <= 2048:
        return K12_MID_SPLITS
    if int(inputs['K']) == 20 and int(inputs['Q']) == 1024:
        return K20_Q1024_SPLITS
    if int(inputs['K']) == 20 and int(inputs['Q']) == 2048:
        return K20_Q2048_SPLITS
    if int(inputs['K']) == 30 and int(inputs['Q']) <= K30_SMALL_SHAPE_MAX:
        return K30_SMALL_SPLITS
    return MEDIUM_SPLITS

def _eligible_fixed_build_generic_lowk(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) in (1, 2, 8)) and (int(inputs['Q']) == int(inputs['M'])) and (int(inputs['Q']) <= parent_lowk.SMALL_SHAPE_MAX)

def _eligible_q512_k8_static(inputs: dict[str, Any]) -> bool:
    return _eligible_k32_split_build(inputs) and int(inputs['K']) == 8 and (int(inputs['Q']) == 512) and (int(inputs['M']) == 512)

def _eligible_rag_k10_cached(inputs: dict[str, Any]) -> bool:
    return not bool(inputs.get('build', False)) and str(inputs['query'].dtype) == 'torch.bfloat16' and (str(inputs['database'].dtype) == 'torch.bfloat16') and (int(inputs['D']) == FEAT_D) and (int(inputs['K']) == TOP_K_MAX) and (int(inputs['Q']) >= RAG_OFFLINE_Q_MIN) and (int(inputs['M']) >= RAG_OFFLINE_M_MIN)

def _launch_rag_k10_cached(inputs: dict[str, Any]) -> None:
    parent_lowk._launch_k10_cached_path(inputs, split_count=parent_lowk.RAG_SPLITS, merge_threads=parent_lowk.parent_cached.RAG_MERGE_THREADS, merge_kernel=parent_lowk.parent_cached._compiled_merge_k10_s7_cache(), merge_ir=parent_lowk.parent_cached.merge_k10_s7_cache_ir)

def _launch_k32_split_path(inputs: dict[str, Any], *, split_count: int) -> None:
    query = inputs['query']
    database = inputs['database']
    bsz = int(inputs['B'])
    n_query = int(inputs['Q'])
    n_database = int(inputs['M'])
    dim = int(inputs['D'])
    top_k = int(inputs['K'])
    num_q_tiles = (n_query + BLOCK_Q - 1) // BLOCK_Q
    num_q_tile_pairs = (num_q_tiles + CTA_GROUP - 1) // CTA_GROUP
    num_db_tiles = (n_database + BLOCK_M - 1) // BLOCK_M
    db_tiles_per_split = (num_db_tiles + split_count - 1) // split_count
    total_work = bsz * num_q_tile_pairs * split_count
    stage1_grid = min(total_work * CTA_GROUP, GRID_DIM_DEFAULT)
    top_k_bucket = _top_k_bucket(top_k)
    use_unordered = top_k in (20, 25, 30, TOP_K_SPLIT_MAX) and split_count == MEDIUM_SPLITS and (n_query >= 4096)
    stage1_ir_obj = _stage1_unordered_ir_for_exact_k(top_k) if use_unordered else _stage1_ir_for_bucket(top_k_bucket)
    use_k8_s7_merge = split_count == parent.RAG_SPLITS and top_k == 8
    use_k8_s8_merge = split_count == K8_Q2048_SPLITS and top_k == 8
    use_k12_s8_merge = split_count == K12_MID_SPLITS and top_k == 12
    use_k20_s16_merge = split_count == K20_Q1024_SPLITS and top_k == 20
    use_k20_s8_merge = split_count == K20_Q2048_SPLITS and top_k == 20
    use_k30_s8_merge = split_count == K30_SMALL_SPLITS and top_k == 30
    merge_threads = K8_MID_MERGE_THREADS if use_k8_s7_merge else K32_MERGE_THREADS
    merge_grid = min((bsz * n_query + merge_threads - 1) // merge_threads, GRID_DIM_DEFAULT)
    merge_ir_obj = _merge_unordered_ir_for_exact_k(top_k) if use_unordered else merge_k8_s7_ir if use_k8_s7_merge else merge_k8_s8_ir if use_k8_s8_merge else merge_k12_s8_ir if use_k12_s8_merge else merge_k20_s16_ir if use_k20_s16_merge else merge_k20_s8_ir if use_k20_s8_merge else merge_k30_s8_ir if use_k30_s8_merge else _merge_ir_for_bucket(top_k_bucket)
    partial_dists, partial_indices = parent_split._partial_buffers(split_count=split_count, bsz=bsz, n_query=n_query, top_k=top_k, device=query.device)
    tmap_query = base_v1._create_tensor_map_3d_oob_zero(query.data_ptr(), bsz * n_query, BLOCK_Q, dim, dim)
    tmap_database = base_v1._create_tensor_map_3d_oob_zero(database.data_ptr(), bsz * n_database, BLOCK_M, dim, dim)
    stage1_kernel = _compiled_stage1_unordered_for_exact_k(top_k) if use_unordered else _compiled_stage1_for_bucket(top_k_bucket)
    stage1_kernel.launch_cluster(grid=(stage1_grid, 1, 1), block=(STAGE1_THREADS, 1, 1), args=pack_kernel_args(stage1_ir_obj, tmap_query=tmap_query, tmap_database=tmap_database, query_sq=inputs['query_sq'], database_sq=inputs['database_sq'], partial_dists=partial_dists, partial_indices=partial_indices, B=bsz, Q=n_query, M=n_database, K=top_k, num_q_tile_pairs=num_q_tile_pairs, db_tiles_per_split=db_tiles_per_split, split_count=split_count, total_work=total_work), cluster_dims=(CTA_GROUP, 1, 1), shared_mem=stage1_ir_obj.computed_smem_bytes)
    if use_unordered:
        merge_kernel = _compiled_merge_unordered_for_exact_k(top_k)
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k8_s7_merge:
        merge_kernel = _compiled_merge_k8_s7()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K8_MID_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k8_s8_merge:
        merge_kernel = _compiled_merge_k8_s8()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K8_MID_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k12_s8_merge:
        merge_kernel = _compiled_merge_k12_s8()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k20_s16_merge:
        merge_kernel = _compiled_merge_k20_s16()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k20_s8_merge:
        merge_kernel = _compiled_merge_k20_s8()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    elif use_k30_s8_merge:
        merge_kernel = _compiled_merge_k30_s8()
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)
    else:
        merge_kernel = _compiled_merge_for_bucket(top_k_bucket)
        merge_kernel.launch(grid=(merge_grid, 1, 1), block=(K32_MERGE_THREADS, 1, 1), args=[partial_dists, partial_indices, inputs['out_dists'], inputs['out_indices'], top_k, bsz * n_query], shared_mem=merge_ir_obj.computed_smem_bytes)

def launch_from_contract_inputs(inputs: dict[str, Any]) -> None:
    if _eligible_q512_k8_static(inputs):
        _launch_k32_split_path(inputs, split_count=Q512_K8_SPLITS)
        return
    if _eligible_fixed_build_generic_lowk(inputs):
        parent_lowk._launch_cg2_split_path(inputs, split_count=parent_lowk.SMALL_SPLITS)
        return
    if _eligible_rag_k10_cached(inputs):
        _launch_rag_k10_cached(inputs)
        return
    split_count = _k32_split_count(inputs)
    if split_count is not None:
        _launch_k32_split_path(inputs, split_count=split_count)
        return
    parent.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]):
    launch_from_contract_inputs(inputs)
    return None

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    return evaluate(candidate, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    from .._dispatch_runtime import CANONICAL_SHAPES
    if shape_labels is None:
        return list(CANONICAL_SHAPES)
    wanted = {str(label) for label in shape_labels}
    selected = [shape for shape in CANONICAL_SHAPES if shape['label'] in wanted]
    missing = wanted - {shape['label'] for shape in selected}
    if missing:
        raise ValueError(''.join(['unknown kNN build contract shape(s): ', format(sorted(missing), '')]))
    return selected

def compile_and_launch_knn_build(*, shape_labels=('flashml_correctness_b1_q256_m256_d128_k5',), benchmark: bool=False) -> dict[str, Any]:
    """e2e-test entrypoint: run real contract correctness for selected shapes."""
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _largek_k25_probe_shape() -> dict[str, Any]:
    return {'label': 'build_qm4096_d128_k25_probe', 'params': {'B': 1, 'Q': 4096, 'M': 4096, 'D': 128, 'K': 25, 'dtype': 'bfloat16', 'seed': 606425, 'build': True, 'check_correctness': True, 'correctness_query_sample': 256, 'recall_min': 0.999, 'benchmark': True, 'time_flashlib': True}}

def benchmark_largek_k25_k32_v12() -> dict[str, Any]:
    """Opt-in bench-regression hook for the custom K25 probe plus K32 diagnostic."""
    shapes = [_largek_k25_probe_shape(), _select_contract_shapes(('build_largek_stress_qm4096_k32',))[0]]
    report = evaluate_contract(shapes=shapes, correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}

def benchmark_largek_k32_v13() -> dict[str, Any]:
    """Opt-in benchmark hook for the canonical K32 diagnostic route."""
    report = evaluate_contract(shapes=_select_contract_shapes(('build_largek_stress_qm4096_k32',)), correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}

def benchmark_largek_k32_v14() -> dict[str, Any]:
    """Opt-in benchmark hook for the canonical K32 diagnostic route."""
    report = evaluate_contract(shapes=_select_contract_shapes(('build_largek_stress_qm4096_k32',)), correctness=True, benchmark=True)
    return {'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'report': report}
