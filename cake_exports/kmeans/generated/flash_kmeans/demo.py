"""Vendored by tools/export-generated-programs (smoke demo).

Provenance: loom @ 7a7cebd73ee298188e1ed3b6d6d8d7e43dbf5f04. Flash K-Means assignment; builds random inputs, runs, checks against a torch brute force.
Runtime-free: this module imports no ``loom`` package.
"""

import argparse

import torch

from . import interface

SMOKE_SHAPES = [(1, 256, 128, 256)]  # (B, N, D, K)
_DIST_TIE = 1.0e-3


def _build_inputs(shape, device, seed):
    B, N, D, K = shape
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(B, N, D, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16)
    centroids = torch.randn(B, K, D, generator=g, dtype=torch.float32, device=device).to(torch.bfloat16)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty(B, N, dtype=torch.int32, device=device)
    tensors = [x, centroids, x_sq, c_sq, out]
    return tensors, {"x": x, "centroids": centroids, "out": out}


def _reference(named, shape):
    x = named["x"].float()
    c = named["centroids"].float()
    dist = (x * x).sum(-1)[..., None] + (c * c).sum(-1)[:, None, :] - 2.0 * torch.einsum("bnd,bkd->bnk", x, c)
    return dist.clamp_min(0.0).argmin(-1).to(torch.int32)


def _check(named, shape, ref):
    B, N, D, K = shape
    pred = named["out"]
    xf = named["x"].float()
    cf = named["centroids"].float()

    def _dist_to(idx):
        chosen = torch.gather(cf, 1, idx.long()[..., None].expand(B, N, D))
        return ((xf - chosen) ** 2).sum(-1)

    tie_ok = (pred == ref) | ((_dist_to(pred) - _dist_to(ref)).abs() <= _DIST_TIE)
    frac = tie_ok.float().mean().item()
    return bool(tie_ok.all().item()), f"match={frac:.4f} (squared-L2 tie tol {_DIST_TIE})"



def run_shape(shape, *, device="cuda", seed=0):
    scalars = tuple(int(s) for s in shape)
    label = " ".join(f"{n}={v}" for n, v in zip(interface.SCALAR_NAMES, scalars))
    if not interface.covered(*scalars):
        raise SystemExit(f"[flash_kmeans] shape not covered by any route: {label}")
    route = interface.route_id(*scalars)
    tensors, named = _build_inputs(shape, device, seed)
    interface.run(*tensors, *scalars)
    torch.cuda.synchronize()
    reference = _reference(named, shape)
    ok, detail = _check(named, shape, reference)
    print(f"[flash_kmeans] {'OK  ' if ok else 'FAIL'} {label} route={route} {detail}")
    norm_sources = getattr(interface, "NORM_SOURCES", None)
    if ok and norm_sources:
        # Exercise the internal norm path: pass every declared norm as None so
        # the shipped rowwise-sqnorm kernel fills it, then re-check.
        blanked = [None if key in norm_sources else t for key, t in zip(interface.TENSOR_KEYS, tensors)]
        interface.run(*blanked, *scalars)
        torch.cuda.synchronize()
        ok, detail = _check(named, shape, reference)
        print(f"[flash_kmeans] {'OK  ' if ok else 'FAIL'} {label} route={route} internal-norms {detail}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(description="Smoke demo for the flash_kmeans generated programs")
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
        raise SystemExit("[flash_kmeans] a CUDA device is required")
    all_ok = True
    for shape in SMOKE_SHAPES:
        all_ok = run_shape(shape) and all_ok
    if not all_ok:
        raise SystemExit("[flash_kmeans] DEMO FAILED")
    print("[flash_kmeans] DEMO PASS")


if __name__ == "__main__":
    main()
