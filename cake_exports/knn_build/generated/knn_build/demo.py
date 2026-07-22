"""Vendored by tools/export-generated-programs (smoke demo).

Provenance: loom @ ff502f39df09ffdb317efc57ebdac3a668bb3aa4. k-NN graph build (top-K nearest neighbours); builds random inputs, runs, checks against a torch brute force.
Runtime-free: this module imports no ``loom`` package.
"""

import argparse

import torch

from . import interface

SMOKE_SHAPES = [(1, 64, 4096, 128, 10, 0, 0)]  # (B, Q, M, D, K, dtype_code, build)
_RECALL_MIN = 0.99
_DIST_ATOL = 1.0e-2


def _build_inputs(shape, device, seed):
    B, Q, M, D, K, dtype_code, build = shape
    tdt = torch.bfloat16 if dtype_code == 0 else torch.float16
    g = torch.Generator(device=device).manual_seed(seed)
    query = torch.randn(B, Q, D, generator=g, dtype=torch.float32, device=device).to(tdt)
    if build:
        if Q != M:
            raise ValueError("build requires Q == M")
        database = query
    else:
        database = torch.randn(B, M, D, generator=g, dtype=torch.float32, device=device).to(tdt)
    query_sq = (query.float() ** 2).sum(-1).contiguous()
    database_sq = (database.float() ** 2).sum(-1).contiguous()
    out_dists = torch.full((B, Q, K), float("inf"), dtype=torch.float32, device=device)
    out_indices = torch.full((B, Q, K), -1, dtype=torch.int32, device=device)
    tensors = [query, database, query_sq, database_sq, out_dists, out_indices]
    return tensors, {"query": query, "database": database, "out_dists": out_dists, "out_indices": out_indices}


def _reference(named, shape):
    q = named["query"].float()
    db = named["database"].float()
    dist = (q * q).sum(-1)[..., None] + (db * db).sum(-1)[:, None, :] - 2.0 * torch.matmul(q, db.transpose(-1, -2))
    return dist.clamp_min(0.0)


def _check(named, shape, dists):
    B, Q, M, D, K, dtype_code, build = shape
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
        raise SystemExit(f"[knn_build] shape not covered by any route: {label}")
    route = interface.route_id(*scalars)
    tensors, named = _build_inputs(shape, device, seed)
    interface.run(*tensors, *scalars)
    torch.cuda.synchronize()
    reference = _reference(named, shape)
    ok, detail = _check(named, shape, reference)
    print(f"[knn_build] {'OK  ' if ok else 'FAIL'} {label} route={route} {detail}")
    norm_sources = getattr(interface, "NORM_SOURCES", None)
    if ok and norm_sources:
        # Exercise the internal norm path: pass every declared norm as None so
        # the shipped rowwise-sqnorm kernel fills it, then re-check.
        blanked = [None if key in norm_sources else t for key, t in zip(interface.TENSOR_KEYS, tensors)]
        interface.run(*blanked, *scalars)
        torch.cuda.synchronize()
        ok, detail = _check(named, shape, reference)
        print(f"[knn_build] {'OK  ' if ok else 'FAIL'} {label} route={route} internal-norms {detail}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke demo for the knn_build generated programs")
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
        raise SystemExit("[knn_build] a CUDA device is required")
    all_ok = True
    for shape in SMOKE_SHAPES:
        all_ok = run_shape(shape) and all_ok
    if not all_ok:
        raise SystemExit("[knn_build] DEMO FAILED")
    print("[knn_build] DEMO PASS")


if __name__ == "__main__":
    main()
