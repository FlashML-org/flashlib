"""Round-34 Q4096/K64 split79 two-tile odd-even-sort fast-merge route.

Minimum target architecture: sm_100a. This additive wrapper preserves the
round-33 odd-even local K64 sort and r32 index-tie merge semantics for the exact
``B=1,Q=4096,M=20000,D=128,K=64`` bucket, but retunes the two-tile producer
from split80 to split79. For M=20000/BLOCK_M=128 this is the smallest split
count that keeps every split at no more than two M tiles, reducing the producer
grid and scratch envelope from 320 to 316 partial lists. The merge consumer is
guarded for the non-multiple-of-32 list count.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from . import knn_search_k32_mma_capacity_0611_r12_v1 as base
from . import knn_search_k64_q4096split80_twotile_distanceonly_fastmerge_0612_r32_11c1_v1 as parent
from . import knn_search_k64_q4096split80_twotile_distanceonly_stagingunrolled_0612_r31_11c1_v1 as producer_parent
from . import knn_search_k64_stable_merge_0612_r23_4e96_v1 as stable
K64_MAX = parent.K64_MAX
Q4096_ROWS = parent.Q4096_ROWS
Q4096_M_ROWS = parent.Q4096_M_ROWS
Q4096_K64_SPLIT_M = 79
Q4096_K64_PARTIAL_LISTS = Q4096_K64_SPLIT_M * parent.MMA_POST_MMA_COL_COHORTS
MERGE10_SPLITS_PER_LANE_MAX = (Q4096_K64_PARTIAL_LISTS + parent.MERGE_THREADS - 1) // parent.MERGE_THREADS
MERGE10_LAST_SLOT_VALID_LANES = Q4096_K64_PARTIAL_LISTS - (MERGE10_SPLITS_PER_LANE_MAX - 1) * parent.MERGE_THREADS
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
MERGE_THREADS = parent.MERGE_THREADS
MMA_POST_MMA_COL_COHORTS = parent.MMA_POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = parent.MMA_SMEM_BYTES
MERGE_SMEM_BYTES = parent.MERGE_SMEM_BYTES
MMA_SMEM_POOL_BYTES = producer_parent.MMA_SMEM_POOL_BYTES
MMA_SMEM_B0_OFFSET = producer_parent.MMA_SMEM_B0_OFFSET
MMA_SMEM_B1_OFFSET = producer_parent.MMA_SMEM_B1_OFFSET
MMA_SMEM_Q_NORM_PART_OFFSET = producer_parent.MMA_SMEM_Q_NORM_PART_OFFSET
MMA_SMEM_DB_NORM_PART0_OFFSET = producer_parent.MMA_SMEM_DB_NORM_PART0_OFFSET
MMA_SMEM_DB_NORM_PART1_OFFSET = producer_parent.MMA_SMEM_DB_NORM_PART1_OFFSET
MMA_SMEM_DB_NORM0_OFFSET = producer_parent.MMA_SMEM_DB_NORM0_OFFSET
MMA_SMEM_DB_NORM1_OFFSET = producer_parent.MMA_SMEM_DB_NORM1_OFFSET
MMA_Q_NORM_PARTS = producer_parent.MMA_Q_NORM_PARTS
Q4096_K64_TOTAL_M_TILES = producer_parent.Q4096_K64_TOTAL_M_TILES
K64_Q4096_SPLIT79_ODDEVENSORT_FASTMERGE_SHAPES: list[dict[str, Any]] = [{'label': 'ksweep_q128_m131072_d128_k64', 'params': {'B': 1, 'Q': 128, 'M': 131072, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610312, 'self_search': False, 'min_recall': 0.999}}, {'label': 'ksweep_q4096_m20000_d128_k64', 'params': {'B': 1, 'Q': 4096, 'M': 20000, 'D': 128, 'K': 64, 'dtype': 'bfloat16', 'seed': 610313, 'self_search': False, 'min_recall': 0.999}}]

def _odd_even_sort_pairs(n: int) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []

    def merge(lo: int, width: int, stride: int) -> None:
        step = stride * 2
        if step < width:
            merge(lo, width, step)
            merge(lo + stride, width, step)
            for left in range(lo + stride, lo + width - stride, step):
                pairs.append((left, left + stride))
        else:
            pairs.append((lo, lo + stride))

    def sort(lo: int, width: int) -> None:
        if width > 1:
            half = width // 2
            sort(lo, half)
            sort(lo + half, half)
            merge(lo, width, 1)
    sort(0, n)
    return tuple(pairs)
_ODDEVEN64_COMPARATORS = _decode_capture(_json_loads('{"__tuple__": [{"__tuple__": [0, 1]}, {"__tuple__": [2, 3]}, {"__tuple__": [0, 2]}, {"__tuple__": [1, 3]}, {"__tuple__": [1, 2]}, {"__tuple__": [4, 5]}, {"__tuple__": [6, 7]}, {"__tuple__": [4, 6]}, {"__tuple__": [5, 7]}, {"__tuple__": [5, 6]}, {"__tuple__": [0, 4]}, {"__tuple__": [2, 6]}, {"__tuple__": [2, 4]}, {"__tuple__": [1, 5]}, {"__tuple__": [3, 7]}, {"__tuple__": [3, 5]}, {"__tuple__": [1, 2]}, {"__tuple__": [3, 4]}, {"__tuple__": [5, 6]}, {"__tuple__": [8, 9]}, {"__tuple__": [10, 11]}, {"__tuple__": [8, 10]}, {"__tuple__": [9, 11]}, {"__tuple__": [9, 10]}, {"__tuple__": [12, 13]}, {"__tuple__": [14, 15]}, {"__tuple__": [12, 14]}, {"__tuple__": [13, 15]}, {"__tuple__": [13, 14]}, {"__tuple__": [8, 12]}, {"__tuple__": [10, 14]}, {"__tuple__": [10, 12]}, {"__tuple__": [9, 13]}, {"__tuple__": [11, 15]}, {"__tuple__": [11, 13]}, {"__tuple__": [9, 10]}, {"__tuple__": [11, 12]}, {"__tuple__": [13, 14]}, {"__tuple__": [0, 8]}, {"__tuple__": [4, 12]}, {"__tuple__": [4, 8]}, {"__tuple__": [2, 10]}, {"__tuple__": [6, 14]}, {"__tuple__": [6, 10]}, {"__tuple__": [2, 4]}, {"__tuple__": [6, 8]}, {"__tuple__": [10, 12]}, {"__tuple__": [1, 9]}, {"__tuple__": [5, 13]}, {"__tuple__": [5, 9]}, {"__tuple__": [3, 11]}, {"__tuple__": [7, 15]}, {"__tuple__": [7, 11]}, {"__tuple__": [3, 5]}, {"__tuple__": [7, 9]}, {"__tuple__": [11, 13]}, {"__tuple__": [1, 2]}, {"__tuple__": [3, 4]}, {"__tuple__": [5, 6]}, {"__tuple__": [7, 8]}, {"__tuple__": [9, 10]}, {"__tuple__": [11, 12]}, {"__tuple__": [13, 14]}, {"__tuple__": [16, 17]}, {"__tuple__": [18, 19]}, {"__tuple__": [16, 18]}, {"__tuple__": [17, 19]}, {"__tuple__": [17, 18]}, {"__tuple__": [20, 21]}, {"__tuple__": [22, 23]}, {"__tuple__": [20, 22]}, {"__tuple__": [21, 23]}, {"__tuple__": [21, 22]}, {"__tuple__": [16, 20]}, {"__tuple__": [18, 22]}, {"__tuple__": [18, 20]}, {"__tuple__": [17, 21]}, {"__tuple__": [19, 23]}, {"__tuple__": [19, 21]}, {"__tuple__": [17, 18]}, {"__tuple__": [19, 20]}, {"__tuple__": [21, 22]}, {"__tuple__": [24, 25]}, {"__tuple__": [26, 27]}, {"__tuple__": [24, 26]}, {"__tuple__": [25, 27]}, {"__tuple__": [25, 26]}, {"__tuple__": [28, 29]}, {"__tuple__": [30, 31]}, {"__tuple__": [28, 30]}, {"__tuple__": [29, 31]}, {"__tuple__": [29, 30]}, {"__tuple__": [24, 28]}, {"__tuple__": [26, 30]}, {"__tuple__": [26, 28]}, {"__tuple__": [25, 29]}, {"__tuple__": [27, 31]}, {"__tuple__": [27, 29]}, {"__tuple__": [25, 26]}, {"__tuple__": [27, 28]}, {"__tuple__": [29, 30]}, {"__tuple__": [16, 24]}, {"__tuple__": [20, 28]}, {"__tuple__": [20, 24]}, {"__tuple__": [18, 26]}, {"__tuple__": [22, 30]}, {"__tuple__": [22, 26]}, {"__tuple__": [18, 20]}, {"__tuple__": [22, 24]}, {"__tuple__": [26, 28]}, {"__tuple__": [17, 25]}, {"__tuple__": [21, 29]}, {"__tuple__": [21, 25]}, {"__tuple__": [19, 27]}, {"__tuple__": [23, 31]}, {"__tuple__": [23, 27]}, {"__tuple__": [19, 21]}, {"__tuple__": [23, 25]}, {"__tuple__": [27, 29]}, {"__tuple__": [17, 18]}, {"__tuple__": [19, 20]}, {"__tuple__": [21, 22]}, {"__tuple__": [23, 24]}, {"__tuple__": [25, 26]}, {"__tuple__": [27, 28]}, {"__tuple__": [29, 30]}, {"__tuple__": [0, 16]}, {"__tuple__": [8, 24]}, {"__tuple__": [8, 16]}, {"__tuple__": [4, 20]}, {"__tuple__": [12, 28]}, {"__tuple__": [12, 20]}, {"__tuple__": [4, 8]}, {"__tuple__": [12, 16]}, {"__tuple__": [20, 24]}, {"__tuple__": [2, 18]}, {"__tuple__": [10, 26]}, {"__tuple__": [10, 18]}, {"__tuple__": [6, 22]}, {"__tuple__": [14, 30]}, {"__tuple__": [14, 22]}, {"__tuple__": [6, 10]}, {"__tuple__": [14, 18]}, {"__tuple__": [22, 26]}, {"__tuple__": [2, 4]}, {"__tuple__": [6, 8]}, {"__tuple__": [10, 12]}, {"__tuple__": [14, 16]}, {"__tuple__": [18, 20]}, {"__tuple__": [22, 24]}, {"__tuple__": [26, 28]}, {"__tuple__": [1, 17]}, {"__tuple__": [9, 25]}, {"__tuple__": [9, 17]}, {"__tuple__": [5, 21]}, {"__tuple__": [13, 29]}, {"__tuple__": [13, 21]}, {"__tuple__": [5, 9]}, {"__tuple__": [13, 17]}, {"__tuple__": [21, 25]}, {"__tuple__": [3, 19]}, {"__tuple__": [11, 27]}, {"__tuple__": [11, 19]}, {"__tuple__": [7, 23]}, {"__tuple__": [15, 31]}, {"__tuple__": [15, 23]}, {"__tuple__": [7, 11]}, {"__tuple__": [15, 19]}, {"__tuple__": [23, 27]}, {"__tuple__": [3, 5]}, {"__tuple__": [7, 9]}, {"__tuple__": [11, 13]}, {"__tuple__": [15, 17]}, {"__tuple__": [19, 21]}, {"__tuple__": [23, 25]}, {"__tuple__": [27, 29]}, {"__tuple__": [1, 2]}, {"__tuple__": [3, 4]}, {"__tuple__": [5, 6]}, {"__tuple__": [7, 8]}, {"__tuple__": [9, 10]}, {"__tuple__": [11, 12]}, {"__tuple__": [13, 14]}, {"__tuple__": [15, 16]}, {"__tuple__": [17, 18]}, {"__tuple__": [19, 20]}, {"__tuple__": [21, 22]}, {"__tuple__": [23, 24]}, {"__tuple__": [25, 26]}, {"__tuple__": [27, 28]}, {"__tuple__": [29, 30]}, {"__tuple__": [32, 33]}, {"__tuple__": [34, 35]}, {"__tuple__": [32, 34]}, {"__tuple__": [33, 35]}, {"__tuple__": [33, 34]}, {"__tuple__": [36, 37]}, {"__tuple__": [38, 39]}, {"__tuple__": [36, 38]}, {"__tuple__": [37, 39]}, {"__tuple__": [37, 38]}, {"__tuple__": [32, 36]}, {"__tuple__": [34, 38]}, {"__tuple__": [34, 36]}, {"__tuple__": [33, 37]}, {"__tuple__": [35, 39]}, {"__tuple__": [35, 37]}, {"__tuple__": [33, 34]}, {"__tuple__": [35, 36]}, {"__tuple__": [37, 38]}, {"__tuple__": [40, 41]}, {"__tuple__": [42, 43]}, {"__tuple__": [40, 42]}, {"__tuple__": [41, 43]}, {"__tuple__": [41, 42]}, {"__tuple__": [44, 45]}, {"__tuple__": [46, 47]}, {"__tuple__": [44, 46]}, {"__tuple__": [45, 47]}, {"__tuple__": [45, 46]}, {"__tuple__": [40, 44]}, {"__tuple__": [42, 46]}, {"__tuple__": [42, 44]}, {"__tuple__": [41, 45]}, {"__tuple__": [43, 47]}, {"__tuple__": [43, 45]}, {"__tuple__": [41, 42]}, {"__tuple__": [43, 44]}, {"__tuple__": [45, 46]}, {"__tuple__": [32, 40]}, {"__tuple__": [36, 44]}, {"__tuple__": [36, 40]}, {"__tuple__": [34, 42]}, {"__tuple__": [38, 46]}, {"__tuple__": [38, 42]}, {"__tuple__": [34, 36]}, {"__tuple__": [38, 40]}, {"__tuple__": [42, 44]}, {"__tuple__": [33, 41]}, {"__tuple__": [37, 45]}, {"__tuple__": [37, 41]}, {"__tuple__": [35, 43]}, {"__tuple__": [39, 47]}, {"__tuple__": [39, 43]}, {"__tuple__": [35, 37]}, {"__tuple__": [39, 41]}, {"__tuple__": [43, 45]}, {"__tuple__": [33, 34]}, {"__tuple__": [35, 36]}, {"__tuple__": [37, 38]}, {"__tuple__": [39, 40]}, {"__tuple__": [41, 42]}, {"__tuple__": [43, 44]}, {"__tuple__": [45, 46]}, {"__tuple__": [48, 49]}, {"__tuple__": [50, 51]}, {"__tuple__": [48, 50]}, {"__tuple__": [49, 51]}, {"__tuple__": [49, 50]}, {"__tuple__": [52, 53]}, {"__tuple__": [54, 55]}, {"__tuple__": [52, 54]}, {"__tuple__": [53, 55]}, {"__tuple__": [53, 54]}, {"__tuple__": [48, 52]}, {"__tuple__": [50, 54]}, {"__tuple__": [50, 52]}, {"__tuple__": [49, 53]}, {"__tuple__": [51, 55]}, {"__tuple__": [51, 53]}, {"__tuple__": [49, 50]}, {"__tuple__": [51, 52]}, {"__tuple__": [53, 54]}, {"__tuple__": [56, 57]}, {"__tuple__": [58, 59]}, {"__tuple__": [56, 58]}, {"__tuple__": [57, 59]}, {"__tuple__": [57, 58]}, {"__tuple__": [60, 61]}, {"__tuple__": [62, 63]}, {"__tuple__": [60, 62]}, {"__tuple__": [61, 63]}, {"__tuple__": [61, 62]}, {"__tuple__": [56, 60]}, {"__tuple__": [58, 62]}, {"__tuple__": [58, 60]}, {"__tuple__": [57, 61]}, {"__tuple__": [59, 63]}, {"__tuple__": [59, 61]}, {"__tuple__": [57, 58]}, {"__tuple__": [59, 60]}, {"__tuple__": [61, 62]}, {"__tuple__": [48, 56]}, {"__tuple__": [52, 60]}, {"__tuple__": [52, 56]}, {"__tuple__": [50, 58]}, {"__tuple__": [54, 62]}, {"__tuple__": [54, 58]}, {"__tuple__": [50, 52]}, {"__tuple__": [54, 56]}, {"__tuple__": [58, 60]}, {"__tuple__": [49, 57]}, {"__tuple__": [53, 61]}, {"__tuple__": [53, 57]}, {"__tuple__": [51, 59]}, {"__tuple__": [55, 63]}, {"__tuple__": [55, 59]}, {"__tuple__": [51, 53]}, {"__tuple__": [55, 57]}, {"__tuple__": [59, 61]}, {"__tuple__": [49, 50]}, {"__tuple__": [51, 52]}, {"__tuple__": [53, 54]}, {"__tuple__": [55, 56]}, {"__tuple__": [57, 58]}, {"__tuple__": [59, 60]}, {"__tuple__": [61, 62]}, {"__tuple__": [32, 48]}, {"__tuple__": [40, 56]}, {"__tuple__": [40, 48]}, {"__tuple__": [36, 52]}, {"__tuple__": [44, 60]}, {"__tuple__": [44, 52]}, {"__tuple__": [36, 40]}, {"__tuple__": [44, 48]}, {"__tuple__": [52, 56]}, {"__tuple__": [34, 50]}, {"__tuple__": [42, 58]}, {"__tuple__": [42, 50]}, {"__tuple__": [38, 54]}, {"__tuple__": [46, 62]}, {"__tuple__": [46, 54]}, {"__tuple__": [38, 42]}, {"__tuple__": [46, 50]}, {"__tuple__": [54, 58]}, {"__tuple__": [34, 36]}, {"__tuple__": [38, 40]}, {"__tuple__": [42, 44]}, {"__tuple__": [46, 48]}, {"__tuple__": [50, 52]}, {"__tuple__": [54, 56]}, {"__tuple__": [58, 60]}, {"__tuple__": [33, 49]}, {"__tuple__": [41, 57]}, {"__tuple__": [41, 49]}, {"__tuple__": [37, 53]}, {"__tuple__": [45, 61]}, {"__tuple__": [45, 53]}, {"__tuple__": [37, 41]}, {"__tuple__": [45, 49]}, {"__tuple__": [53, 57]}, {"__tuple__": [35, 51]}, {"__tuple__": [43, 59]}, {"__tuple__": [43, 51]}, {"__tuple__": [39, 55]}, {"__tuple__": [47, 63]}, {"__tuple__": [47, 55]}, {"__tuple__": [39, 43]}, {"__tuple__": [47, 51]}, {"__tuple__": [55, 59]}, {"__tuple__": [35, 37]}, {"__tuple__": [39, 41]}, {"__tuple__": [43, 45]}, {"__tuple__": [47, 49]}, {"__tuple__": [51, 53]}, {"__tuple__": [55, 57]}, {"__tuple__": [59, 61]}, {"__tuple__": [33, 34]}, {"__tuple__": [35, 36]}, {"__tuple__": [37, 38]}, {"__tuple__": [39, 40]}, {"__tuple__": [41, 42]}, {"__tuple__": [43, 44]}, {"__tuple__": [45, 46]}, {"__tuple__": [47, 48]}, {"__tuple__": [49, 50]}, {"__tuple__": [51, 52]}, {"__tuple__": [53, 54]}, {"__tuple__": [55, 56]}, {"__tuple__": [57, 58]}, {"__tuple__": [59, 60]}, {"__tuple__": [61, 62]}, {"__tuple__": [0, 32]}, {"__tuple__": [16, 48]}, {"__tuple__": [16, 32]}, {"__tuple__": [8, 40]}, {"__tuple__": [24, 56]}, {"__tuple__": [24, 40]}, {"__tuple__": [8, 16]}, {"__tuple__": [24, 32]}, {"__tuple__": [40, 48]}, {"__tuple__": [4, 36]}, {"__tuple__": [20, 52]}, {"__tuple__": [20, 36]}, {"__tuple__": [12, 44]}, {"__tuple__": [28, 60]}, {"__tuple__": [28, 44]}, {"__tuple__": [12, 20]}, {"__tuple__": [28, 36]}, {"__tuple__": [44, 52]}, {"__tuple__": [4, 8]}, {"__tuple__": [12, 16]}, {"__tuple__": [20, 24]}, {"__tuple__": [28, 32]}, {"__tuple__": [36, 40]}, {"__tuple__": [44, 48]}, {"__tuple__": [52, 56]}, {"__tuple__": [2, 34]}, {"__tuple__": [18, 50]}, {"__tuple__": [18, 34]}, {"__tuple__": [10, 42]}, {"__tuple__": [26, 58]}, {"__tuple__": [26, 42]}, {"__tuple__": [10, 18]}, {"__tuple__": [26, 34]}, {"__tuple__": [42, 50]}, {"__tuple__": [6, 38]}, {"__tuple__": [22, 54]}, {"__tuple__": [22, 38]}, {"__tuple__": [14, 46]}, {"__tuple__": [30, 62]}, {"__tuple__": [30, 46]}, {"__tuple__": [14, 22]}, {"__tuple__": [30, 38]}, {"__tuple__": [46, 54]}, {"__tuple__": [6, 10]}, {"__tuple__": [14, 18]}, {"__tuple__": [22, 26]}, {"__tuple__": [30, 34]}, {"__tuple__": [38, 42]}, {"__tuple__": [46, 50]}, {"__tuple__": [54, 58]}, {"__tuple__": [2, 4]}, {"__tuple__": [6, 8]}, {"__tuple__": [10, 12]}, {"__tuple__": [14, 16]}, {"__tuple__": [18, 20]}, {"__tuple__": [22, 24]}, {"__tuple__": [26, 28]}, {"__tuple__": [30, 32]}, {"__tuple__": [34, 36]}, {"__tuple__": [38, 40]}, {"__tuple__": [42, 44]}, {"__tuple__": [46, 48]}, {"__tuple__": [50, 52]}, {"__tuple__": [54, 56]}, {"__tuple__": [58, 60]}, {"__tuple__": [1, 33]}, {"__tuple__": [17, 49]}, {"__tuple__": [17, 33]}, {"__tuple__": [9, 41]}, {"__tuple__": [25, 57]}, {"__tuple__": [25, 41]}, {"__tuple__": [9, 17]}, {"__tuple__": [25, 33]}, {"__tuple__": [41, 49]}, {"__tuple__": [5, 37]}, {"__tuple__": [21, 53]}, {"__tuple__": [21, 37]}, {"__tuple__": [13, 45]}, {"__tuple__": [29, 61]}, {"__tuple__": [29, 45]}, {"__tuple__": [13, 21]}, {"__tuple__": [29, 37]}, {"__tuple__": [45, 53]}, {"__tuple__": [5, 9]}, {"__tuple__": [13, 17]}, {"__tuple__": [21, 25]}, {"__tuple__": [29, 33]}, {"__tuple__": [37, 41]}, {"__tuple__": [45, 49]}, {"__tuple__": [53, 57]}, {"__tuple__": [3, 35]}, {"__tuple__": [19, 51]}, {"__tuple__": [19, 35]}, {"__tuple__": [11, 43]}, {"__tuple__": [27, 59]}, {"__tuple__": [27, 43]}, {"__tuple__": [11, 19]}, {"__tuple__": [27, 35]}, {"__tuple__": [43, 51]}, {"__tuple__": [7, 39]}, {"__tuple__": [23, 55]}, {"__tuple__": [23, 39]}, {"__tuple__": [15, 47]}, {"__tuple__": [31, 63]}, {"__tuple__": [31, 47]}, {"__tuple__": [15, 23]}, {"__tuple__": [31, 39]}, {"__tuple__": [47, 55]}, {"__tuple__": [7, 11]}, {"__tuple__": [15, 19]}, {"__tuple__": [23, 27]}, {"__tuple__": [31, 35]}, {"__tuple__": [39, 43]}, {"__tuple__": [47, 51]}, {"__tuple__": [55, 59]}, {"__tuple__": [3, 5]}, {"__tuple__": [7, 9]}, {"__tuple__": [11, 13]}, {"__tuple__": [15, 17]}, {"__tuple__": [19, 21]}, {"__tuple__": [23, 25]}, {"__tuple__": [27, 29]}, {"__tuple__": [31, 33]}, {"__tuple__": [35, 37]}, {"__tuple__": [39, 41]}, {"__tuple__": [43, 45]}, {"__tuple__": [47, 49]}, {"__tuple__": [51, 53]}, {"__tuple__": [55, 57]}, {"__tuple__": [59, 61]}, {"__tuple__": [1, 2]}, {"__tuple__": [3, 4]}, {"__tuple__": [5, 6]}, {"__tuple__": [7, 8]}, {"__tuple__": [9, 10]}, {"__tuple__": [11, 12]}, {"__tuple__": [13, 14]}, {"__tuple__": [15, 16]}, {"__tuple__": [17, 18]}, {"__tuple__": [19, 20]}, {"__tuple__": [21, 22]}, {"__tuple__": [23, 24]}, {"__tuple__": [25, 26]}, {"__tuple__": [27, 28]}, {"__tuple__": [29, 30]}, {"__tuple__": [31, 32]}, {"__tuple__": [33, 34]}, {"__tuple__": [35, 36]}, {"__tuple__": [37, 38]}, {"__tuple__": [39, 40]}, {"__tuple__": [41, 42]}, {"__tuple__": [43, 44]}, {"__tuple__": [45, 46]}, {"__tuple__": [47, 48]}, {"__tuple__": [49, 50]}, {"__tuple__": [51, 52]}, {"__tuple__": [53, 54]}, {"__tuple__": [55, 56]}, {"__tuple__": [57, 58]}, {"__tuple__": [59, 60]}, {"__tuple__": [61, 62]}]}'))
_KNN_SEARCH_K64_SPLIT79_ODDEVENSORT_FASTMERGE_KERNELS: dict[str, Any] = {}
_knn_sort64_oddeven = _ir_proxy('loom.examples.weave.knn_search_k64_q4096split79_twotile_oddevensort_fastmerge_0612_r34_11c1_v1:_knn_sort64_oddeven', 256)
knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_indexfastmerge10_guarded_0612_r34_11c1_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_twotile_oddevensort_partial_0612_r34_11c1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))

def _compile_k64_split79_oddevensort_fastmerge_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"merge": {"__kernel__": "dispatch_kernel_0028"}, "partial": {"__kernel__": "dispatch_kernel_0027"}}'))

def _use_q4096_k64_split79_oddevensort_fastmerge(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['K']) == K64_MAX and (int(inputs['Q']) == Q4096_ROWS) and (int(inputs['M']) == Q4096_M_ROWS) and (int(inputs['D']) == D_STATIC) and base._tcgen05_capable_arch()

def _launch_q4096_k64_split79_oddevensort_fastmerge(inputs: dict[str, Any]) -> dict[str, Any]:
    import torch
    if not _KNN_SEARCH_K64_SPLIT79_ODDEVENSORT_FASTMERGE_KERNELS:
        _KNN_SEARCH_K64_SPLIT79_ODDEVENSORT_FASTMERGE_KERNELS.update(_compile_k64_split79_oddevensort_fastmerge_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    split_m = min(Q4096_K64_SPLIT_M, total_m_tiles)
    tiles_per_split = math.ceil(total_m_tiles / split_m)
    partial_list_count = split_m * MMA_POST_MMA_COL_COHORTS
    partial_dist, partial_idx = stable._scratch(inputs, split_m, num_q_tiles)
    _KNN_SEARCH_K64_SPLIT79_ODDEVENSORT_FASTMERGE_KERNELS['partial'].launch(grid=(bsz * num_q_tiles * split_m, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_dist, partial_idx, bsz, q_rows, m_rows, split_m, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KNN_SEARCH_K64_SPLIT79_ODDEVENSORT_FASTMERGE_KERNELS['merge'].launch(grid=(bsz * q_rows, 1, 1), block=(MERGE_THREADS, 1, 1), args=[partial_dist, partial_idx, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, partial_list_count, num_q_tiles], shared_mem=MERGE_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k64_split79_oddevensort_fastmerge(inputs):
        return _launch_q4096_k64_split79_oddevensort_fastmerge(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    return parent._select_contract_shapes(shape_labels)

def knn_search_compile_and_launch_mma_split(*, benchmark: bool=True, shape_labels: str | tuple[str, ...] | list[str] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=_select_contract_shapes(shape_labels), benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_k64_q4096_split79_oddevensort_fastmerge(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=K64_Q4096_SPLIT79_ODDEVENSORT_FASTMERGE_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
