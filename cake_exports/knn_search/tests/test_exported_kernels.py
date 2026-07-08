from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE_NAME = 'flashlib_cake_knn_search'


def _manifest() -> dict:
    return json.loads((SRC / PACKAGE_NAME / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_matches_package_exports():
    pkg = importlib.import_module(PACKAGE_NAME)
    manifest = _manifest()
    names = [entry["name"] for entry in manifest["kernels"]]

    assert names
    assert set(pkg.KERNELS) == set(names)
    assert manifest["package"] == PACKAGE_NAME
    package_exports = manifest.get("export_plan", {}).get("package_exports", {})
    for public_name in package_exports:
        assert hasattr(pkg, public_name), public_name
        assert public_name in pkg.__all__, public_name

    if "export_plan" in manifest:
        entrypoints = manifest["export_plan"].get("entrypoints", {})
        assert set(entrypoints) == {
            "python_interface",
            "correctness_test",
            "performance_benchmark",
        }
        for path in entrypoints.values():
            assert (ROOT / path).is_file(), path


def test_manifest_sources_exist_and_contain_symbols():
    manifest = _manifest()
    package_dir = SRC / PACKAGE_NAME

    for entry in manifest["kernels"]:
        source_path = package_dir / entry["source"]
        assert source_path.is_file(), entry["source"]
        source = source_path.read_text(encoding="utf-8")
        assert entry["symbol"] in source
        assert entry["parameters"]


def test_package_import_and_source_text_do_not_require_cuda_runtime():
    pkg = importlib.import_module(PACKAGE_NAME)

    for name, kernel in pkg.KERNELS.items():
        assert kernel.source_text().startswith("typedef "), name
        assert kernel.parameters


def test_exported_repo_docs_and_benchmarks_exist():
    assert (ROOT / "README.md").is_file()
    assert (ROOT / "RESULTS.md").is_file()
    assert (ROOT / "benchmarks" / "benchmark_exported_kernels.py").is_file()
    assert (ROOT / "benchmarks" / "benchmark_shapes.py").is_file()
    assert (ROOT / "benchmarks" / "workload.py").is_file()
    assert (SRC / PACKAGE_NAME / "tvm_ffi.py").is_file()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'cake-std==0.1.13.dev20260704+g7b8dbc8' in pyproject


def test_tvm_ffi_adapter_registers_low_level_functions_without_import_time_dependency(monkeypatch):
    pkg = importlib.import_module(PACKAGE_NAME)
    registrations = {}

    class FakeTensor:
        pass

    def register_global_func(name, function, *, override=False):
        assert override is False
        registrations[name] = function

    fake_tvm_ffi = types.SimpleNamespace(
        Tensor=FakeTensor,
        register_global_func=register_global_func,
        get_raw_stream=lambda device: 0,
    )
    monkeypatch.setitem(sys.modules, "tvm_ffi", fake_tvm_ffi)

    expected = pkg.tvm_ffi_function_names("export_test")
    registered = pkg.register_tvm_ffi("export_test")
    assert registered == expected
    assert set(registrations) == set(expected)
    assert registered == pkg.register_tvm_ffi("export_test")
    assert len(registrations) == len(expected)


def test_benchmark_runtime_requires_cupti_without_event_or_wall_clock_fallback():
    source = (SRC / PACKAGE_NAME / "_benchmark.py").read_text(encoding="utf-8")
    assert "activity_register_callbacks" in source
    assert "torch.cuda.Event" not in source
    assert "perf_counter" not in source
    assert "active_union_times_ms" in source
    assert "activity_counts" in source
    assert "launch_activity_counts" in source
    assert "kernel_activity_counts" in source
    assert "submission_times_ms" in source
    assert "synchronized_e2e_times_ms" in source
    assert "cold_first_call_host_enqueue_ms" in source
    assert "cold_first_call_synchronized_e2e_ms" in source
    helper = source.split("def _complete_l2_flush_before_bracket", 1)[1].split(
        "def _correlate", 1
    )[0]
    assert helper.index("flusher.flush()") < helper.index("synchronize()")
    measured_loop = source.split("for _ in range(bench_iters):", 1)[1]
    assert measured_loop.index("_complete_l2_flush_before_bracket") < measured_loop.index(
        "start = cupti.get_timestamp()"
    )


def test_benchmark_runtime_preserves_exact_activity_diagnostics():
    benchmark = importlib.import_module(f"{PACKAGE_NAME}._benchmark")
    timing = benchmark._correlate(
        [(0, 100, 10_000)],
        [(10, 11, 1), (20, 21, 2)],
        [(1_000, 4_000, 1), (2_000, 5_000, 2)],
    )
    assert timing.gpu_span_ms == [0.004]
    assert timing.kernel_sum_ms == [0.006]
    assert timing.active_union_ms == [0.004]
    assert timing.inter_kernel_gap_ms == [0.0]
    assert timing.activity_count == [2]
    assert timing.launch_activity_count == [2]
    assert timing.kernel_activity_count == [2]
    with pytest.raises(ValueError, match="must be 'cupti'"):
        benchmark.BenchResult(times_ms=[1.0], backend="cuda_event")


def test_runtime_has_content_cache_and_launch_context():
    source = (SRC / PACKAGE_NAME / "_runtime.py").read_text(encoding="utf-8")
    assert "def launch_context" in source
    assert "def load_cached_kernel" in source
    assert "_CUBIN_CACHE" in source
    assert "_MODULE_CACHE" in source
    assert "_KERNEL_CACHE" in source
    kernels_source = (SRC / PACKAGE_NAME / "kernels.py").read_text(encoding="utf-8")
    assert "self._arg_types = tuple" in kernels_source
    assert "self._default_block" in kernels_source


def test_launch_argument_count_is_checked_before_compilation():
    pkg = importlib.import_module(PACKAGE_NAME)
    kernel = next(iter(pkg.KERNELS.values()))
    bad_args = [object()] * (len(kernel.parameters) + 1)

    with pytest.raises(TypeError, match="expects"):
        kernel.launch(*bad_args, grid=(1, 1, 1))

