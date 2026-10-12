"""flashlib.diagnose() — print environment + backend availability.

Run from CLI: python -c "import flashlib; flashlib.diagnose()"
"""
from __future__ import annotations

import sys


def diagnose() -> None:
    """Print versions, GPU info, and primitive availability."""
    print(f"flashlib {_get_version()}")
    print(f"  python:  {sys.version.split()[0]}")
    print(f"  torch:   {_safe_version('torch')}")
    print(f"  triton:  {_safe_version('triton')}")
    print(f"  numpy:   {_safe_version('numpy')}")
    print()
    _print_cuda_info()
    print()
    _print_optional_status()
    print()
    _print_primitive_status()


def _get_version() -> str:
    try:
        from flashlib import __version__
        return __version__
    except Exception:
        return "?"


def _safe_version(modname: str) -> str:
    try:
        mod = __import__(modname)
        return getattr(mod, "__version__", "?")
    except Exception as e:
        return f"NOT INSTALLED ({type(e).__name__})"


def _print_cuda_info() -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print("  CUDA:    not available")
            return
        n = torch.cuda.device_count()
        print(f"  CUDA:    {torch.version.cuda} | {n} device(s)")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}  SM{p.major}.{p.minor}  "
                  f"{p.total_memory / 1e9:.0f}GB")
    except Exception as e:
        print(f"  CUDA:    error ({e})")


def _print_optional_status() -> None:
    from flashlib import _optional

    print("optional extras:")
    problem = _optional._cutedsl_problem()
    if problem is None:
        print(f"  [OK]   cutedsl (CuTe DSL backends, nvidia-cutlass-dsl {_safe_version('cutlass')})")
    elif problem == "rocm":
        print("  [--]   cutedsl (CuTe DSL backends): NVIDIA GPUs only")
    else:
        print(f"  [--]   cutedsl (CuTe DSL backends): {problem}; pip install 'flashlib[cutedsl]'")
    numba_problem = _optional._numba_problem()
    if numba_problem is None:
        print(f"  [OK]   numba   (fast flash_hdbscan tree steps, numba {_safe_version('numba')})")
    else:
        print(f"  [--]   numba   (flash_hdbscan tree steps run in pure Python): {numba_problem}; "
              f"pip install 'flashlib[numba]'")


def _print_primitive_status() -> None:
    print("primitives:")
    for name in [
        "standard_scaler",
        "kmeans", "knn",
        "pca", "truncated_svd",
        "linear_regression", "ridge", "logistic_regression",
        "dbscan", "hdbscan",
        "umap", "tsne",
        "multinomial_nb", "random_forest",
        "spectral_clustering",
    ]:
        try:
            __import__(f"flashlib.primitives.{name}")
            print(f"  [OK]   {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    print("linalg + kernels:")
    for name, modpath in [
        ("kernels.distance",              "flashlib.kernels.distance"),
        ("kernels.connected_components",  "flashlib.kernels.connected_components"),
        ("linalg.cov_gemm",               "flashlib.linalg.cov_gemm"),
        ("linalg.gram_gemm",              "flashlib.linalg.gram_gemm"),
        ("linalg.ab_gemm",                "flashlib.linalg.ab_gemm"),
        ("linalg.eigh",                   "flashlib.linalg.eigh"),
        ("linalg.gemm",                   "flashlib.linalg.gemm"),
        ("linalg.polar",                  "flashlib.linalg.polar"),
        ("linalg.orthonormalize",         "flashlib.linalg.orthonormalize"),
    ]:
        try:
            __import__(modpath)
            print(f"  [OK]   {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
