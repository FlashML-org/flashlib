from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))


def _benchmark_module():
    spec = importlib.util.spec_from_file_location("knn_build_benchmark", BENCHMARKS / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _benchmark_module()


def test_knn_build_runtime_api_is_exported():
    from flashlib_cake_knn_build import (
        KNNBuildRuntime,
        init,
        knn_build,
        knn_build_prepared,
        prepare_knn_build,
    )

    assert KNNBuildRuntime is not None
    assert callable(init)
    assert callable(knn_build)
    assert callable(prepare_knn_build)
    assert callable(knn_build_prepared)


@pytest.mark.export_validation_shape
@pytest.mark.parametrize("name", list(BENCHMARK.SHAPES))
def test_knn_build_matches_reference(name: str):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_knn_build import init

    runtime = init()
    try:
        result = BENCHMARK._run_shape(
            name,
            BENCHMARK.SHAPES[name],
            runtime=runtime,
            arch=None,
            correctness=True,
            benchmark=False,
        )
    finally:
        runtime.clear()
    assert result["route_matches_expected"], result
    assert result["correct"], result


def test_knn_build_runtime_reuses_shape_a_b_a():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_knn_build import init

    labels = (
        "flashml_correctness_b1_q256_m256_d128_k5",
        "rag_online_b1_q1_m65536_d128_k10",
        "flashml_correctness_b1_q256_m256_d128_k5",
    )
    runtime = init()
    try:
        results = [
            BENCHMARK._run_shape(
                label,
                BENCHMARK.SHAPES[label],
                runtime=runtime,
                arch=None,
                correctness=True,
                benchmark=False,
            )
            for label in labels
        ]
        assert all(result["correct"] and result["route_matches_expected"] for result in results)
        assert results[-1]["first_shape_lookup_cache_hit"] is True
        assert runtime.cache_info()["size"] == 2
    finally:
        runtime.clear()
