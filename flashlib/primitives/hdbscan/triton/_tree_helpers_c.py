"""C versions of the HDBSCAN tree helpers in :mod:`._tree_helpers`.

Line-for-line ports of the pure-Python reference implementations, compiled on
first use by :func:`flashlib._cjit.load_c_library`. Outputs are identical to the
Python versions (same loop order, no FMA contraction).
"""
from __future__ import annotations

import ctypes
import functools
from typing import Optional

import numpy as np

from flashlib._cjit import load_c_library

_SRC = r"""
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static int64_t bfs_into_buffer(const double* h, int64_t root, int64_t num_points,
                               int64_t* queue, int64_t* out) {
    int64_t n_out = 0, qstart = 0, qend = 0;
    queue[qend++] = root;
    while (qstart < qend) {
        int64_t x = queue[qstart++];
        out[n_out++] = x;
        if (x >= num_points) {
            int64_t row = x - num_points;
            queue[qend++] = (int64_t)h[4 * row];
            queue[qend++] = (int64_t)h[4 * row + 1];
        }
    }
    return n_out;
}

/* Single-linkage tree (n_edges, 4) from MST edges (n_edges, 3) sorted by weight. */
int64_t fast_label(const double* mst, int64_t n_edges, double* out) {
    int64_t n = n_edges + 1;
    int64_t* parent = malloc(2 * n * sizeof(int64_t));
    int64_t* sizes = malloc(2 * n * sizeof(int64_t));
    if (!parent || !sizes) { free(parent); free(sizes); return -1; }
    for (int64_t i = 0; i < 2 * n; ++i) { parent[i] = i; sizes[i] = 1; }
    int64_t next_id = n;
    for (int64_t i = 0; i < n_edges; ++i) {
        int64_t ru = (int64_t)mst[3 * i], rv = (int64_t)mst[3 * i + 1];
        while (parent[ru] != ru) { parent[ru] = parent[parent[ru]]; ru = parent[ru]; }
        while (parent[rv] != rv) { parent[rv] = parent[parent[rv]]; rv = parent[rv]; }
        out[4 * i] = (double)ru;
        out[4 * i + 1] = (double)rv;
        out[4 * i + 2] = mst[3 * i + 2];
        out[4 * i + 3] = (double)(sizes[ru] + sizes[rv]);
        parent[ru] = next_id;
        parent[rv] = next_id;
        sizes[next_id] = sizes[ru] + sizes[rv];
        ++next_id;
    }
    free(parent); free(sizes);
    return 0;
}

/* Condensed tree; outputs must hold 2 * (n_merges + 1) entries. Returns the row count. */
int64_t condense_tree(const double* h, int64_t n_merges, int64_t min_cluster_size,
                      int64_t* out_parent, int64_t* out_child, double* out_lambda, int64_t* out_size) {
    int64_t num_points = n_merges + 1, root = 2 * n_merges, buf = 2 * num_points;
    int64_t* queue_main = malloc(buf * sizeof(int64_t));
    int64_t* out_main = malloc(buf * sizeof(int64_t));
    int64_t* queue_sub = malloc(buf * sizeof(int64_t));
    int64_t* out_sub = malloc(buf * sizeof(int64_t));
    int64_t* relabel = malloc((root + 1) * sizeof(int64_t));
    uint8_t* ignore = calloc(buf, 1);
    int64_t n_out = -1;
    if (!queue_main || !out_main || !queue_sub || !out_sub || !relabel || !ignore) goto done;

    int64_t n_node_list = bfs_into_buffer(h, root, num_points, queue_main, out_main);
    relabel[root] = num_points;
    int64_t next_label = num_points + 1;
    n_out = 0;
    for (int64_t ni = 0; ni < n_node_list; ++ni) {
        int64_t node = out_main[ni];
        if (ignore[node] || node < num_points) continue;
        int64_t row = node - num_points;
        int64_t left = (int64_t)h[4 * row], right = (int64_t)h[4 * row + 1];
        double d = h[4 * row + 2];
        double lambda_value = d > 0.0 ? 1.0 / d : INFINITY;
        int64_t left_count = left >= num_points ? (int64_t)h[4 * (left - num_points) + 3] : 1;
        int64_t right_count = right >= num_points ? (int64_t)h[4 * (right - num_points) + 3] : 1;
        int64_t rn = relabel[node];
        int64_t drop[2];
        int n_drop = 0;
        if (left_count >= min_cluster_size && right_count >= min_cluster_size) {
            relabel[left] = next_label++;
            out_parent[n_out] = rn; out_child[n_out] = relabel[left];
            out_lambda[n_out] = lambda_value; out_size[n_out] = left_count; ++n_out;
            relabel[right] = next_label++;
            out_parent[n_out] = rn; out_child[n_out] = relabel[right];
            out_lambda[n_out] = lambda_value; out_size[n_out] = right_count; ++n_out;
        } else if (left_count < min_cluster_size && right_count < min_cluster_size) {
            drop[n_drop++] = left; drop[n_drop++] = right;
        } else if (left_count < min_cluster_size) {
            relabel[right] = rn; drop[n_drop++] = left;
        } else {
            relabel[left] = rn; drop[n_drop++] = right;
        }
        for (int di = 0; di < n_drop; ++di) {
            int64_t ns = bfs_into_buffer(h, drop[di], num_points, queue_sub, out_sub);
            for (int64_t k = 0; k < ns; ++k) {
                int64_t sub = out_sub[k];
                if (sub < num_points) {
                    out_parent[n_out] = rn; out_child[n_out] = sub;
                    out_lambda[n_out] = lambda_value; out_size[n_out] = 1; ++n_out;
                }
                ignore[sub] = 1;
            }
        }
    }
done:
    free(queue_main); free(out_main); free(queue_sub); free(out_sub); free(relabel); free(ignore);
    return n_out;
}

/* births must hold largest_child + 1 entries; result (num_clusters) must be zeroed. */
void compute_stability(const int64_t* parents, const int64_t* children, const double* lambdas,
                       const int64_t* sizes, int64_t n, int64_t smallest_cluster,
                       int64_t largest_child, double* births, double* result) {
    for (int64_t i = 0; i <= largest_child; ++i) births[i] = INFINITY;
    for (int64_t i = 0; i < n; ++i) {
        if (lambdas[i] < births[children[i]]) births[children[i]] = lambdas[i];
    }
    births[smallest_cluster] = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double bp = births[parents[i]];
        if (bp == INFINITY) bp = 0.0;
        result[parents[i] - smallest_cluster] += (lambdas[i] - bp) * (double)sizes[i];
    }
}

/* EOM cluster selection + flat labels. order_desc = argsort(cluster_ids)[::-1]. */
int64_t get_clusters(const int64_t* parents, const int64_t* children, int64_t n,
                     const int64_t* cluster_ids, const int64_t* order_desc,
                     const double* stability, int64_t n_clusters, int64_t num_points, int64_t* labels) {
    int64_t smallest = cluster_ids[0], largest_id = -1, n_cedges = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (children[i] >= smallest) ++n_cedges;
        if (parents[i] > largest_id) largest_id = parents[i];
        if (children[i] > largest_id) largest_id = children[i];
    }
    uint8_t* is_cluster = malloc(n_clusters);
    int64_t* off = calloc(n_clusters + 1, sizeof(int64_t));
    int64_t* fill = malloc((n_clusters + 1) * sizeof(int64_t));
    int64_t* kids = malloc((n_cedges ? n_cedges : 1) * sizeof(int64_t));
    double* subtree = malloc(n_clusters * sizeof(double));
    int64_t* stack = malloc(n_clusters * sizeof(int64_t));
    int64_t* parent_of = malloc((largest_id + 1) * sizeof(int64_t));
    int64_t* label_map = malloc(n_clusters * sizeof(int64_t));
    int64_t rc = -1;
    if (!is_cluster || !off || !fill || !kids || !subtree || !stack || !parent_of || !label_map) goto done;

    for (int64_t i = 0; i < n; ++i)
        if (children[i] >= smallest) ++off[parents[i] - smallest + 1];
    for (int64_t k = 0; k < n_clusters; ++k) off[k + 1] += off[k];
    for (int64_t k = 0; k <= n_clusters; ++k) fill[k] = off[k];
    for (int64_t i = 0; i < n; ++i)
        if (children[i] >= smallest) kids[fill[parents[i] - smallest]++] = children[i];

    for (int64_t k = 0; k < n_clusters; ++k) { is_cluster[k] = 1; subtree[k] = stability[k]; }
    for (int64_t k = 0; k < n_clusters; ++k) {
        int64_t ki = cluster_ids[order_desc[k]] - smallest;
        int64_t lo = off[ki], hi = off[ki + 1];
        if (hi == lo) continue;
        double sub_sum = 0.0;
        for (int64_t j = lo; j < hi; ++j) sub_sum += subtree[kids[j] - smallest];
        if (subtree[ki] < sub_sum) {
            subtree[ki] = sub_sum;
            is_cluster[ki] = 0;
        } else {
            int64_t top = 0;
            for (int64_t j = lo; j < hi; ++j) stack[top++] = kids[j];
            while (top > 0) {
                int64_t nki = stack[--top] - smallest;
                is_cluster[nki] = 0;
                for (int64_t j = off[nki]; j < off[nki + 1]; ++j) stack[top++] = kids[j];
            }
        }
    }

    for (int64_t i = 0; i <= largest_id; ++i) parent_of[i] = -1;
    for (int64_t i = 0; i < n; ++i) parent_of[children[i]] = parents[i];
    int64_t next_lbl = 0;
    for (int64_t k = 0; k < n_clusters; ++k) label_map[k] = is_cluster[k] ? next_lbl++ : -1;
    for (int64_t p = 0; p < num_points; ++p) {
        labels[p] = -1;
        int64_t cur = p <= largest_id ? parent_of[p] : -1;
        while (cur >= 0) {
            if (cur >= smallest && is_cluster[cur - smallest]) { labels[p] = label_map[cur - smallest]; break; }
            cur = cur <= largest_id ? parent_of[cur] : -1;
        }
    }
    rc = 0;
done:
    free(is_cluster); free(off); free(fill); free(kids); free(subtree); free(stack); free(parent_of); free(label_map);
    return rc;
}
"""

_P = ctypes.c_void_p
_I = ctypes.c_int64


@functools.lru_cache(maxsize=None)
def _load():
    lib, problem = load_c_library("hdbscan_tree", _SRC)
    if lib is not None:
        lib.fast_label.argtypes = [_P, _I, _P]
        lib.fast_label.restype = _I
        lib.condense_tree.argtypes = [_P, _I, _I, _P, _P, _P, _P]
        lib.condense_tree.restype = _I
        lib.compute_stability.argtypes = [_P, _P, _P, _P, _I, _I, _I, _P, _P]
        lib.compute_stability.restype = None
        lib.get_clusters.argtypes = [_P, _P, _I, _P, _P, _P, _I, _I, _P]
        lib.get_clusters.restype = _I
    return lib, problem


def library() -> Optional[ctypes.CDLL]:
    return _load()[0]


def build_problem() -> Optional[str]:
    """Why the C helpers could not be built, or None if they are available."""
    return _load()[1]


def _check(rc: int) -> None:
    if rc < 0:
        raise MemoryError("flashlib: HDBSCAN C helper could not allocate its scratch buffers")


def _i64(a):
    return np.ascontiguousarray(a, dtype=np.int64)


def _f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def fast_label(mst_sorted):
    mst = _f64(mst_sorted)
    out = np.empty((mst.shape[0], 4), dtype=np.float64)
    _check(library().fast_label(mst.ctypes.data, mst.shape[0], out.ctypes.data))
    return out


def fast_condense_tree(hierarchy, min_cluster_size):
    h = _f64(hierarchy)
    n_max = 2 * (h.shape[0] + 1)
    parent = np.empty(n_max, dtype=np.int64)
    child = np.empty(n_max, dtype=np.int64)
    lam = np.empty(n_max, dtype=np.float64)
    size = np.empty(n_max, dtype=np.int64)
    n_out = library().condense_tree(h.ctypes.data, h.shape[0], int(min_cluster_size),
                                    parent.ctypes.data, child.ctypes.data, lam.ctypes.data, size.ctypes.data)
    _check(n_out)
    return parent[:n_out], child[:n_out], lam[:n_out], size[:n_out]


def fast_compute_stability(parents, children, lambdas, sizes):
    parents, children, sizes, lambdas = _i64(parents), _i64(children), _i64(sizes), _f64(lambdas)
    n = parents.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    smallest_cluster = int(parents.min())
    largest_parent = int(parents.max())
    largest_child = max(int(children.max()), smallest_cluster)
    births = np.empty(largest_child + 1, dtype=np.float64)
    result = np.zeros(largest_parent - smallest_cluster + 1, dtype=np.float64)
    library().compute_stability(parents.ctypes.data, children.ctypes.data, lambdas.ctypes.data,
                                sizes.ctypes.data, n, smallest_cluster, largest_child,
                                births.ctypes.data, result.ctypes.data)
    return np.arange(smallest_cluster, largest_parent + 1), result


def fast_get_clusters(parents, children, lambdas, sizes, cluster_ids, stability_arr, num_points):
    parents, children, cluster_ids = _i64(parents), _i64(children), _i64(cluster_ids)
    stability = _f64(stability_arr)
    if parents.shape[0] == 0:
        return np.full(num_points, -1, dtype=np.int64)
    order_desc = _i64(np.argsort(cluster_ids)[::-1])
    labels = np.empty(num_points, dtype=np.int64)
    _check(library().get_clusters(parents.ctypes.data, children.ctypes.data, parents.shape[0],
                                  cluster_ids.ctypes.data, order_desc.ctypes.data, stability.ctypes.data,
                                  cluster_ids.shape[0], int(num_points), labels.ctypes.data))
    return labels
