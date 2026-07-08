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
    spec = importlib.util.spec_from_file_location("kmeans_benchmark", BENCHMARKS / "benchmark.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BENCHMARK = _benchmark_module()


def test_kmeans_runtime_api_is_exported():
    from flashlib_cake_kmeans import FlashKMeansAssignRuntime, init

    assert FlashKMeansAssignRuntime is not None
    assert callable(init)


@pytest.mark.export_validation_shape
@pytest.mark.parametrize("row_index", range(len(BENCHMARK.FLASH_KMEANS_REGISTRY_SHAPES)))
def test_kmeans_matches_reference(row_index: int):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    row = BENCHMARK.FLASH_KMEANS_REGISTRY_SHAPES[row_index]
    from flashlib_cake_kmeans import init

    runtime = init()
    baseline_adapter = BENCHMARK.TritonH20007cfRawAdapter(
        device_index=runtime.device_index,
        arch=runtime.arch,
    )
    try:
        result = BENCHMARK._run_shape(
            row,
            runtime=runtime,
            baseline_adapter=baseline_adapter,
            arch=None,
            correctness=True,
            benchmark=False,
            reference_chunk_rows=128,
            measurement_session_id="standalone-correctness-test",
            measurement_order_seed=BENCHMARK.DEFAULT_MEASUREMENT_ORDER_SEED,
        )
    finally:
        runtime.clear()
        baseline_adapter.clear()
    assert result["route_matches_expected"], result
    assert result["correct"], result


@pytest.mark.gpu
def test_kmeans_runtime_reuses_shape_after_intervening_shape():
    """Exercise A -> B -> fresh-pointer A on one live multi-shape runtime."""

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_kmeans import init

    rows = {row["label"]: row for row in BENCHMARK.FLASH_KMEANS_SHAPES}
    shape_a = rows["boundary_b1_n128_k256_d128"]
    shape_b = rows["small_b1_n256_k256_d128"]
    runtime = init()

    def run(row, *, variant: int):
        x, centroids = BENCHMARK._make_inputs(row, variant=variant)
        out = torch.empty((int(row["B"]), int(row["N"])), dtype=torch.int32, device=x.device)
        result, info = runtime.compute(
            x,
            centroids,
            out=out,
            return_info=True,
        )
        reference = BENCHMARK._reference_assign(x, centroids, chunk_rows=128)
        torch.cuda.synchronize()
        correctness = BENCHMARK._assignment_correctness(result, reference, x, centroids)
        assert correctness["correct"], correctness
        assert info["norm_mode"] == "fused_bf16_pair_row_norm:x_sq,c_sq"
        assert info["norm_compute_fields"] == ["x_sq", "c_sq"]
        return x, info

    try:
        first_x, first_info = run(shape_a, variant=101)
        _, middle_info = run(shape_b, variant=201)
        fresh_x, revisit_info = run(shape_a, variant=102)
        assert first_x.data_ptr() != fresh_x.data_ptr()
        assert first_info["runtime_cache_hit"] is False
        assert middle_info["runtime_cache_hit"] is False
        assert revisit_info["runtime_cache_hit"] is True
        assert runtime.cache_info() == {
            "size": 2,
            "hits": 1,
            "misses": 2,
            "max_cached_shapes": None,
        }
    finally:
        runtime.clear()


@pytest.mark.gpu
def test_internal_fused_norms_and_explicit_precomputed_norms_are_correct():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_kmeans import init

    rows = {row["label"]: row for row in BENCHMARK.FLASH_KMEANS_SHAPES}
    row = rows["boundary_b1_n128_k256_d128"]
    x, centroids = BENCHMARK._make_inputs(row, variant=250)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    internal_out = torch.empty((1, 128), dtype=torch.int32, device=x.device)
    explicit_out = torch.empty_like(internal_out)
    x_explicit_out = torch.empty_like(internal_out)
    c_explicit_out = torch.empty_like(internal_out)
    runtime = init()
    try:
        internal_result, internal_info = runtime.compute(
            x,
            centroids,
            out=internal_out,
            return_info=True,
        )
        explicit_result, explicit_info = runtime.compute(
            x,
            centroids,
            out=explicit_out,
            x_sq=x_sq,
            c_sq=c_sq,
            return_info=True,
        )
        x_explicit_result, x_explicit_info = runtime.compute(
            x,
            centroids,
            out=x_explicit_out,
            x_sq=x_sq,
            return_info=True,
        )
        c_explicit_result, c_explicit_info = runtime.compute(
            x,
            centroids,
            out=c_explicit_out,
            c_sq=c_sq,
            return_info=True,
        )
        reference = BENCHMARK._reference_assign(x, centroids, chunk_rows=128)
        torch.cuda.synchronize()
        for result in (
            internal_result,
            explicit_result,
            x_explicit_result,
            c_explicit_result,
        ):
            correctness = BENCHMARK._assignment_correctness(
                result,
                reference,
                x,
                centroids,
            )
            assert correctness["correct"], correctness
        assert internal_info["norm_mode"] == "fused_bf16_pair_row_norm:x_sq,c_sq"
        assert internal_info["norm_compute_fields"] == ["x_sq", "c_sq"]
        assert explicit_info["norm_mode"] == "explicit_precomputed"
        assert explicit_info["norm_compute_fields"] == []
        assert x_explicit_info["norm_mode"] == "fused_bf16_pair_row_norm:c_sq"
        assert x_explicit_info["norm_compute_fields"] == ["c_sq"]
        assert c_explicit_info["norm_mode"] == "fused_bf16_pair_row_norm:x_sq"
        assert c_explicit_info["norm_compute_fields"] == ["x_sq"]
        assert explicit_info["runtime_cache_hit"] is False
        assert x_explicit_info["runtime_cache_hit"] is False
        assert c_explicit_info["runtime_cache_hit"] is False
        assert runtime.cache_info()["size"] == 4
    finally:
        runtime.clear()


@pytest.mark.gpu
def test_kmeans_runtime_isolates_tma_and_workspace_across_streams():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for exported-kernel correctness")
    from flashlib_cake_kmeans import init

    rows = {row["label"]: row for row in BENCHMARK.FLASH_KMEANS_SHAPES}
    row = rows["boundary_b1_n128_k256_d128"]
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    runtime = init()

    def inputs(variant: int):
        x, centroids = BENCHMARK._make_inputs(row, variant=variant)
        return (
            x,
            centroids,
            (x.float() ** 2).sum(-1).contiguous(),
            (centroids.float() ** 2).sum(-1).contiguous(),
            torch.empty((int(row["B"]), int(row["N"])), dtype=torch.int32, device=x.device),
        )

    values_a = inputs(301)
    values_b = inputs(302)
    try:
        for stream, values in ((stream_a, values_a), (stream_b, values_b)):
            x, centroids, x_sq, c_sq, out = values
            with torch.cuda.stream(stream):
                runtime.compute(
                    x,
                    centroids,
                    x_sq=x_sq,
                    c_sq=c_sq,
                    out=out,
                    stream=stream,
                )
        stream_a.synchronize()
        stream_b.synchronize()

        outputs = []
        for stream, values in ((stream_a, values_a), (stream_b, values_b)):
            x, centroids, x_sq, c_sq, out = values
            with torch.cuda.stream(stream):
                outputs.append(
                    runtime.compute(
                        x,
                        centroids,
                        x_sq=x_sq,
                        c_sq=c_sq,
                        out=out,
                        stream=stream,
                    )
                )
        stream_a.synchronize()
        stream_b.synchronize()
        for output, values in zip(outputs, (values_a, values_b), strict=True):
            x, centroids, *_ = values
            reference = BENCHMARK._reference_assign(x, centroids, chunk_rows=128)
            correctness = BENCHMARK._assignment_correctness(output, reference, x, centroids)
            assert correctness["correct"], correctness
        assert runtime.cache_info()["size"] == 2
    finally:
        runtime.clear()
