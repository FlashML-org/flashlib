"""Vendored by tools/export-generated-programs (smoke demo).

Provenance: loom @ unknown. k-NN search (top-K nearest neighbours); builds random inputs, runs, checks against a torch brute force.
Runtime-free: this module imports no ``loom`` package.
"""

import argparse

import torch

from . import interface

SMOKE_SHAPES = [(1, 8, 10, 32, 10, 0), (1, 128, 8192, 128, 10, 0)]  # (B, Q, M, D, K, self_search)
_RECALL_MIN = 0.99
_DIST_ATOL = 1.0e-2


def _build_inputs(shape, device, seed):
    B, Q, M, D, K, self_search = shape
    g = torch.Generator(device=device).manual_seed(seed)
    queries = torch.randn(B, Q, D, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16)
    if self_search:
        if Q != M:
            raise ValueError("self_search requires Q == M")
        database = queries
    else:
        database = torch.randn(B, M, D, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16)
    out_dists = torch.full((B, Q, K), float("inf"), dtype=torch.float32, device=device)
    out_indices = torch.full((B, Q, K), -1, dtype=torch.int32, device=device)
    tensors = [queries, database, out_dists, out_indices]
    return tensors, {"queries": queries, "database": database, "out_dists": out_dists, "out_indices": out_indices}


def _reference(named, shape):
    q = named["queries"].float()
    db = named["database"].float()
    dist = (q * q).sum(-1)[..., None] + (db * db).sum(-1)[:, None, :] - 2.0 * torch.matmul(q, db.transpose(-1, -2))
    return dist.clamp_min(0.0)


def _check(named, shape, dists):
    B, Q, M, D, K, self_search = shape
    return _knn_recall(named["out_dists"], named["out_indices"], dists, M, _RECALL_MIN, _DIST_ATOL)


def _knn_recall(out_dists, out_indices, dists, m_rows, recall_min, dist_atol):
    """Tie-aware recall@k against the fp32 brute-force top-K (matches the loom eval)."""
    k = out_indices.shape[-1]
    ref_vals, _ = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
    kth = ref_vals.amax(-1, keepdim=True)
    idx = out_indices.clamp_min(0).long()
    exact = torch.gather(dists, -1, idx)
    within = (out_indices >= 0) & (out_indices < m_rows) & (exact <= kth + kth.abs() * 1.0e-3 + dist_atol)
    recall = within.float().mean().item()
    valid = out_indices >= 0
    max_err = (out_dists[valid] - exact[valid]).abs().max().item() if bool(valid.any().item()) else float("inf")
    ok = recall >= recall_min and max_err <= dist_atol
    return ok, f"recall={recall:.4f} (>= {recall_min}) max_dist_err={max_err:.3e} (<= {dist_atol})"



def run_shape(shape, *, device="cuda", seed=0):
    scalars = tuple(int(s) for s in shape)
    label = " ".join(f"{n}={v}" for n, v in zip(interface.SCALAR_NAMES, scalars))
    if not interface.covered(*scalars):
        raise SystemExit(f"[knn_search] shape not covered by any route: {label}")
    route = interface.route_id(*scalars)
    tensors, named = _build_inputs(shape, device, seed)
    interface.run(*tensors, *scalars)
    torch.cuda.synchronize()
    reference = _reference(named, shape)
    ok, detail = _check(named, shape, reference)
    print(f"[knn_search] {'OK  ' if ok else 'FAIL'} {label} route={route} {detail}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke demo for the knn_search generated programs")
    parser.add_argument(
        "--assert-no-loom",
        action="store_true",
        help="fail if the 'loom' package is importable (self-containment gate)",
    )
    args = parser.parse_args(argv)
    if args.assert_no_loom:
        import importlib

        try:
            importlib.import_module("loom")
        except ImportError:
            pass
        else:
            raise SystemExit("self-containment violated: 'loom' is importable inside the demo process")
    if not torch.cuda.is_available():
        raise SystemExit("[knn_search] a CUDA device is required")
    all_ok = True
    for shape in SMOKE_SHAPES:
        all_ok = run_shape(shape) and all_ok
    if not all_ok:
        raise SystemExit("[knn_search] DEMO FAILED")
    print("[knn_search] DEMO PASS")


if __name__ == "__main__":
    main()
