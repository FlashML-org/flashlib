"""Every op must run without the optional ``[cutedsl]`` extra (and without numba).

Each check runs in a fresh interpreter that hides ``cutlass`` and ``numba``
from the import system, so it exercises the plain ``pip install flashlib``
setup even on a machine that has both installed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

_HIDE_EXTRAS = '''
import importlib.abc, importlib.metadata as _md, sys

class _HideExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("cutlass", "numba"):
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None

sys.meta_path.insert(0, _HideExtras())
_version = _md.version

def _no_cutlass_dsl(dist):
    if dist == "nvidia-cutlass-dsl":
        raise _md.PackageNotFoundError(dist)
    return _version(dist)

_md.version = _no_cutlass_dsl
'''

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _run_without_extras(body: str, **env_overrides: str) -> str:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _HIDE_EXTRAS + textwrap.dedent(body)],
        capture_output=True, text=True, timeout=900, env=env, cwd=ROOT,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc.stderr


def test_import_all_names_without_extras():
    _run_without_extras("""
        import sys
        import flashlib
        from flashlib._optional import has_cutedsl
        assert not has_cutedsl()
        for name in flashlib._LAZY_ATTRS:
            getattr(flashlib, name)
        assert not [m for m in sys.modules if m.split(".")[0] in ("cutlass", "numba")]
        flashlib.diagnose()
    """)


@needs_cuda
def test_cutedsl_only_linalg_ops_run_without_cutedsl():
    _run_without_extras("""
        import torch
        import flashlib
        from flashlib.linalg.gemm import gemm_ozaki2_cute
        from flashlib.linalg.cholesky import potrf_3xbf16
        from flashlib.linalg.trsm import trsm_3xbf16, cholesky_solve_3xbf16
        from flashlib.linalg.qr import geqrf_3xbf16

        def rel(x, ref):
            return ((x.double() - ref).norm() / ref.norm()).item()

        torch.manual_seed(0)
        A = torch.randn(1000, 333, device="cuda")
        B = torch.randn(333, 777, device="cuda")
        ref = A.double() @ B.double()
        assert rel(flashlib.gemm_3xbf16(A, B), ref) < 1e-4
        assert rel(flashlib.gemm_fp16_x9(A, B), ref) < 1e-5
        assert rel(gemm_ozaki2_cute(A, B, num_moduli=8), ref) < 1e-6

        X = torch.randn(1000, 1000, device="cuda")
        S = X @ X.T / 1000 + torch.eye(1000, device="cuda")
        L = potrf_3xbf16(S, leaf=256)
        assert rel(L @ L.T, S.double()) < 1e-3
        R = torch.randn(1000, 50, device="cuda")
        assert rel(L @ trsm_3xbf16(L, R, leaf=256), R.double()) < 1e-3
        assert rel(S @ cholesky_solve_3xbf16(L, R, leaf=256), R.double()) < 1e-2
        Q, Rq = geqrf_3xbf16(torch.randn(2000, 300, device="cuda"))
        assert (Q.T @ Q - torch.eye(300, device="cuda")).abs().max().item() < 1e-3
    """)


@needs_cuda
def test_hdbscan_runs_without_numba(tmp_path):
    stderr = _run_without_extras("""
        import torch
        import flashlib

        torch.manual_seed(0)
        centers = torch.tensor([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
        X = (centers.repeat_interleave(300, 0) + torch.randn(900, 2)).cuda()
        labels = flashlib.flash_hdbscan(X, min_cluster_size=25)
        assert labels.max() + 1 == 3, labels.max() + 1
    """, FLASHLIB_CACHE_DIR=str(tmp_path))
    assert "pure Python" not in stderr, stderr


@needs_cuda
def test_knn_small_query_on_blackwell_without_cutedsl():
    _run_without_extras("""
        import dataclasses
        import torch
        from flashlib import _hw
        from flashlib.primitives.knn import flash_knn
        from flashlib.primitives.knn.torch_fallback import knn_torch_naive

        blackwell = dataclasses.replace(_hw.current(), sm_arch=100)
        _hw.current = lambda device=None: blackwell
        torch.manual_seed(0)
        x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
        c = torch.randn(5000, 128, device="cuda", dtype=torch.bfloat16)
        vals, idxs = flash_knn(x, c, 10)
        ref_vals, _ = knn_torch_naive(x[None], c[None], 10)
        assert torch.allclose(vals, ref_vals[0], rtol=1e-3, atol=1e-3)
    """)


def test_installed_but_unusable_cutedsl_warns(monkeypatch):
    from flashlib import _optional

    monkeypatch.setattr(_optional, "_cutedsl_problem", lambda: "nvidia-cutlass-dsl 9.9 is outside the supported range")
    _optional._warn_cutedsl_unusable.cache_clear()
    with pytest.warns(RuntimeWarning, match="installed but cannot be used"):
        assert not _optional.has_cutedsl()
