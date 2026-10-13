"""HDBSCAN CPU-side tree-manipulation helpers (numba @njit).

Used by both ``flash_hdbscan`` and ``flash_hdbscan_sparse`` for:
  • iterative BFS over the single-linkage tree
  • condensing the tree (removes nodes with size < min_cluster_size)
  • computing per-cluster stability
  • extracting flat labels from the condensed tree (EOM / leaf)

Without numba the same functions run as plain Python (slower, same results).
"""
import numpy as np

from flashlib._optional import has_numba, warn_no_numba

if has_numba():
    import numba

    _njit = numba.njit
else:
    warn_no_numba("flash_hdbscan")

    def _njit(**_kwargs):
        return lambda f: f


@_njit(cache=True)
def _bfs_into_buffer(hierarchy, root, num_points, queue, out):
    """Iterative BFS over SLT — writes nodes in BFS order to `out`, returns count.
    `queue` and `out` must be pre-allocated (size >= 2*num_points)."""
    n_out = np.int64(0)
    qstart = np.int64(0)
    qend = np.int64(0)
    queue[qend] = root; qend += 1
    while qstart < qend:
        x = queue[qstart]; qstart += 1
        out[n_out] = x; n_out += 1
        if x >= num_points:
            row = x - num_points
            queue[qend] = np.int64(hierarchy[row, 0]); qend += 1
            queue[qend] = np.int64(hierarchy[row, 1]); qend += 1
    return n_out


@_njit(cache=True)
def _bfs_from_hierarchy(hierarchy, root, num_points):
    """Allocating wrapper — kept for backwards compatibility within this module."""
    queue = np.empty(2 * num_points, dtype=np.int64)
    out = np.empty(2 * num_points, dtype=np.int64)
    n = _bfs_into_buffer(hierarchy, root, num_points, queue, out)
    return out[:n]


@_njit(cache=True)
def _fast_condense_tree(hierarchy, min_cluster_size):
    """Port of upstream condense_tree to numba. Returns 4 parallel arrays:
    parent, child, lambda_val, child_size.

    Pre-allocates BFS scratch buffers ONCE outside the main loop. The original
    upstream code allocated new arrays per BFS call, which is the dominant cost
    when many runt branches trigger sub-BFS calls.
    """
    n_merges = hierarchy.shape[0]
    num_points = n_merges + 1
    root = 2 * n_merges

    # Pre-allocated scratch
    bfs_buf = 2 * num_points
    queue_main = np.empty(bfs_buf, dtype=np.int64)
    out_main = np.empty(bfs_buf, dtype=np.int64)
    queue_sub = np.empty(bfs_buf, dtype=np.int64)
    out_sub = np.empty(bfs_buf, dtype=np.int64)

    n_node_list = _bfs_into_buffer(hierarchy, root, num_points, queue_main, out_main)

    relabel = np.empty(root + 1, dtype=np.int64)
    relabel[root] = num_points
    ignore = np.zeros(2 * num_points, dtype=np.uint8)

    out_max = 2 * num_points
    out_parent = np.empty(out_max, dtype=np.int64)
    out_child = np.empty(out_max, dtype=np.int64)
    out_lambda = np.empty(out_max, dtype=np.float64)
    out_size = np.empty(out_max, dtype=np.int64)
    n_out = np.int64(0)
    next_label = num_points + 1

    for ni in range(n_node_list):
        node = out_main[ni]
        if ignore[node] != 0 or node < num_points:
            continue
        row = node - num_points
        left = np.int64(hierarchy[row, 0])
        right = np.int64(hierarchy[row, 1])
        d = hierarchy[row, 2]
        lambda_value = 1.0 / d if d > 0.0 else np.inf

        if left >= num_points:
            left_count = np.int64(hierarchy[left - num_points, 3])
        else:
            left_count = 1
        if right >= num_points:
            right_count = np.int64(hierarchy[right - num_points, 3])
        else:
            right_count = 1

        rn = relabel[node]
        if left_count >= min_cluster_size and right_count >= min_cluster_size:
            relabel[left] = next_label; next_label += 1
            out_parent[n_out] = rn; out_child[n_out] = relabel[left]
            out_lambda[n_out] = lambda_value; out_size[n_out] = left_count
            n_out += 1
            relabel[right] = next_label; next_label += 1
            out_parent[n_out] = rn; out_child[n_out] = relabel[right]
            out_lambda[n_out] = lambda_value; out_size[n_out] = right_count
            n_out += 1
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            ns = _bfs_into_buffer(hierarchy, left, num_points, queue_sub, out_sub)
            for k in range(ns):
                sub = out_sub[k]
                if sub < num_points:
                    out_parent[n_out] = rn; out_child[n_out] = sub
                    out_lambda[n_out] = lambda_value; out_size[n_out] = 1
                    n_out += 1
                ignore[sub] = 1
            ns = _bfs_into_buffer(hierarchy, right, num_points, queue_sub, out_sub)
            for k in range(ns):
                sub = out_sub[k]
                if sub < num_points:
                    out_parent[n_out] = rn; out_child[n_out] = sub
                    out_lambda[n_out] = lambda_value; out_size[n_out] = 1
                    n_out += 1
                ignore[sub] = 1
        elif left_count < min_cluster_size:
            relabel[right] = rn
            ns = _bfs_into_buffer(hierarchy, left, num_points, queue_sub, out_sub)
            for k in range(ns):
                sub = out_sub[k]
                if sub < num_points:
                    out_parent[n_out] = rn; out_child[n_out] = sub
                    out_lambda[n_out] = lambda_value; out_size[n_out] = 1
                    n_out += 1
                ignore[sub] = 1
        else:
            relabel[left] = rn
            ns = _bfs_into_buffer(hierarchy, right, num_points, queue_sub, out_sub)
            for k in range(ns):
                sub = out_sub[k]
                if sub < num_points:
                    out_parent[n_out] = rn; out_child[n_out] = sub
                    out_lambda[n_out] = lambda_value; out_size[n_out] = 1
                    n_out += 1
                ignore[sub] = 1

    return (out_parent[:n_out], out_child[:n_out],
            out_lambda[:n_out], out_size[:n_out])


@_njit(cache=True)
def _fast_compute_stability(parents, children, lambdas, sizes):
    """Port of compute_stability. Returns (cluster_ids, stability) parallel arrays.

    Two-pass O(N) implementation — no sort. births[c] is computed by single
    linear scan tracking the running minimum lambda per child.
    """
    n = parents.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    # Find array bounds in one pass
    smallest_cluster = parents[0]
    largest_parent = parents[0]
    largest_child = children[0]
    for i in range(1, n):
        p = parents[i]
        c = children[i]
        if p < smallest_cluster:
            smallest_cluster = p
        if p > largest_parent:
            largest_parent = p
        if c > largest_child:
            largest_child = c
    if largest_child < smallest_cluster:
        largest_child = smallest_cluster

    # Pass 1: births[c] = min lambda over rows where children[i] == c
    # No sort needed — direct scatter-min.
    births = np.full(largest_child + 1, np.inf, dtype=np.float64)
    for i in range(n):
        c = children[i]
        lam = lambdas[i]
        if lam < births[c]:
            births[c] = lam
    # Replace untouched (inf) with NaN-equivalent (0 sentinel for parent lookup)
    births[smallest_cluster] = 0.0

    # Pass 2: stability[parent] = Σ (lambda - births[parent]) * size
    num_clusters = largest_parent - smallest_cluster + 1
    result = np.zeros(num_clusters, dtype=np.float64)
    for i in range(n):
        p = parents[i]
        bp = births[p]
        if bp == np.inf:
            bp = 0.0
        result[p - smallest_cluster] += (lambdas[i] - bp) * sizes[i]

    cluster_ids = np.arange(smallest_cluster, largest_parent + 1)
    return cluster_ids, result


@_njit(cache=True)
def _fast_get_clusters(parents, children, lambdas, sizes,
                       cluster_ids, stability_arr, num_points):
    """Port of get_clusters (eom strategy, no allow_single_cluster, no epsilon).
    Returns labels[num_points] int64.

    O(N + E) version: builds parent->children CSR adjacency once instead of
    scanning the edge list O(C) times.
    """
    n = parents.shape[0]
    if n == 0:
        return np.full(num_points, -1, dtype=np.int64)

    n_clusters = cluster_ids.shape[0]
    is_cluster = np.ones(n_clusters, dtype=np.uint8)
    smallest_cluster = cluster_ids[0]

    # ── Build CSR: parent_cluster_idx → list of child_cluster_ids ──
    # First, count children per cluster
    n_cedges = 0
    for i in range(n):
        if children[i] >= smallest_cluster:
            n_cedges += 1
    children_count = np.zeros(n_clusters, dtype=np.int64)
    for i in range(n):
        if children[i] >= smallest_cluster:
            ki = parents[i] - smallest_cluster
            children_count[ki] += 1
    children_off = np.zeros(n_clusters + 1, dtype=np.int64)
    for k in range(n_clusters):
        children_off[k + 1] = children_off[k] + children_count[k]
    children_buf = np.empty(n_cedges, dtype=np.int64)
    fill_pos = children_off.copy()
    for i in range(n):
        if children[i] >= smallest_cluster:
            ki = parents[i] - smallest_cluster
            children_buf[fill_pos[ki]] = children[i]
            fill_pos[ki] += 1

    # ── Bottom-up DP for subtree stability + selection ──
    # Process in descending cluster_id order (children before parents — under
    # HDBSCAN's relabel scheme, child cluster ids > parent cluster id).
    subtree = stability_arr.copy()
    order_desc = np.argsort(cluster_ids)[::-1]
    deselect_stack = np.empty(n_clusters, dtype=np.int64)

    for k in range(n_clusters):
        ci = cluster_ids[order_desc[k]]
        ki = ci - smallest_cluster
        kid_lo = children_off[ki]
        kid_hi = children_off[ki + 1]
        if kid_hi == kid_lo:
            continue  # leaf cluster — keep current subtree[ki] = stability
        sub_sum = 0.0
        for j in range(kid_lo, kid_hi):
            sub_sum += subtree[children_buf[j] - smallest_cluster]
        if subtree[ki] < sub_sum:
            subtree[ki] = sub_sum
            is_cluster[ki] = 0
        else:
            # Deselect all descendants via BFS using CSR
            top = 0
            for j in range(kid_lo, kid_hi):
                deselect_stack[top] = children_buf[j]; top += 1
            while top > 0:
                top -= 1
                nc = deselect_stack[top]
                nki = nc - smallest_cluster
                is_cluster[nki] = 0
                ch_lo = children_off[nki]
                ch_hi = children_off[nki + 1]
                for j in range(ch_lo, ch_hi):
                    deselect_stack[top] = children_buf[j]; top += 1

    # Build parent_of[child] -> parent map. Need size = max(children) + 1
    largest_id = np.int64(-1)
    for i in range(n):
        if parents[i] > largest_id:
            largest_id = parents[i]
        if children[i] > largest_id:
            largest_id = children[i]
    parent_of_size = largest_id + 1
    parent_of = np.full(parent_of_size, -1, dtype=np.int64)
    for i in range(n):
        parent_of[children[i]] = parents[i]

    labels = np.full(num_points, -1, dtype=np.int64)
    label_map = np.full(n_clusters, -1, dtype=np.int64)
    next_lbl = np.int64(0)
    for k in range(n_clusters):
        if is_cluster[k] != 0:
            label_map[k] = next_lbl
            next_lbl += 1

    for p in range(num_points):
        cur = parent_of[p]
        while cur >= 0:
            if cur >= smallest_cluster and is_cluster[cur - smallest_cluster] != 0:
                labels[p] = label_map[cur - smallest_cluster]
                break
            if cur < parent_of_size:
                cur = parent_of[cur]
            else:
                cur = np.int64(-1)

    return labels


@_njit(cache=True, fastmath=True)
def _fast_label(mst_sorted):
    """Single-linkage dendrogram from sorted MST edges.

    mst_sorted: (N-1, 3) float64 array of [u, v, weight] sorted by weight.
    Returns: (N-1, 4) float64 single-linkage tree
             [child1, child2, distance, cluster_size]
    Uses iterative path-compressed union-find — ~10x faster than upstream
    hdbscan._hdbscan_linkage.label() which does naive union-find.
    """
    N = mst_sorted.shape[0] + 1
    parent = np.arange(2 * N, dtype=np.int64)
    sizes = np.ones(2 * N, dtype=np.int64)
    result = np.empty((N - 1, 4), dtype=np.float64)

    next_id = N
    for i in range(N - 1):
        u = np.int64(mst_sorted[i, 0])
        v = np.int64(mst_sorted[i, 1])
        w = mst_sorted[i, 2]

        # find root of u with path halving
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        rv = v
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]

        result[i, 0] = float(ru)
        result[i, 1] = float(rv)
        result[i, 2] = w
        result[i, 3] = float(sizes[ru] + sizes[rv])
        parent[ru] = next_id
        parent[rv] = next_id
        sizes[next_id] = sizes[ru] + sizes[rv]
        next_id += 1

    return result
