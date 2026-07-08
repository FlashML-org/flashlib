from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE_NAME = 'flashlib_cake_knn_build'


def _shape_name(shape: Mapping[str, Any], index: int) -> str:
    return str(shape.get("name") or f"shape_{index}")


def _selected_shapes(workload: Any, requested: list[str] | None) -> list[dict[str, Any]]:
    shapes = [dict(shape) for shape in getattr(workload, "SHAPES", ())]
    names = [_shape_name(shape, index) for index, shape in enumerate(shapes)]
    if len(names) != len(set(names)):
        raise ValueError("workload.SHAPES contains duplicate names")
    if not requested:
        return shapes
    missing = sorted(set(requested) - set(names))
    if missing:
        raise ValueError(f"unknown shape(s) {missing}. Available: {', '.join(names)}")
    requested_set = set(requested)
    return [shape for index, shape in enumerate(shapes) if _shape_name(shape, index) in requested_set]


def _comparison_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"passed": value}
    if not isinstance(value, Mapping) or "passed" not in value:
        raise TypeError("case['compare'] must return bool or a mapping containing 'passed'")
    return dict(value)


def _run_shape(
    pkg: Any,
    workload: Any,
    shape: dict[str, Any],
    *,
    index: int,
    correctness_only: bool,
    warmup_iters: int,
    bench_iters: int,
) -> dict[str, Any]:
    name = _shape_name(shape, index)
    entry: dict[str, Any] = {"name": name, "shape": shape, "status": "pending"}
    try:
        case = workload.make_case(pkg, shape)
        if not isinstance(case, Mapping):
            raise TypeError("make_case() must return a mapping")
        missing = [key for key in ("run", "reference", "compare") if not callable(case.get(key))]
        if missing:
            raise TypeError(f"benchmark case is missing callable(s): {', '.join(missing)}")

        expected = case["reference"]()
        import torch

        cold_first_call = None
        if correctness_only:
            actual = case["run"]()
            torch.cuda.synchronize()
        else:
            from flashlib_cake_knn_build._benchmark import measure_host_call

            actual, cold_first_call = measure_host_call(case["run"])
        comparison = _comparison_payload(case["compare"](actual, expected))
        entry["correctness"] = comparison
        if not comparison["passed"]:
            entry["status"] = "incorrect"
            return entry
        if correctness_only:
            entry["status"] = "passed"
            return entry

        from flashlib_cake_knn_build._benchmark import bench_gpu_time

        timing = bench_gpu_time(
            case["run"],
            warmup_iters=warmup_iters,
            bench_iters=bench_iters,
            cold_l2=True,
            cold_first_call=cold_first_call,
        )
        entry["timing"] = {
            "backend": timing.backend,
            "official_gpu_metric": "gpu_span_ms",
            "median_ms": timing.median_ms,
            "min_ms": timing.min_ms,
            "mean_ms": timing.mean_ms,
            "iterations": len(timing.times_ms),
            "cold_l2": True,
            "gpu_span_ms": {
                "median": timing.median_gpu_span_ms,
                "iterations": timing.times_ms,
            },
            "kernel_sum_ms": {
                "median": timing.median_kernel_sum_ms,
                "iterations": timing.kernel_sum_times_ms,
            },
            "active_union_ms": {
                "median": timing.median_active_union_ms,
                "iterations": timing.active_union_times_ms,
            },
            "inter_kernel_gap_ms": {
                "median": timing.median_inter_kernel_gap_ms,
                "iterations": timing.inter_kernel_gap_times_ms,
            },
            "activity_count": {
                "median": timing.median_activity_count,
                "iterations": timing.activity_counts,
            },
            "correlated_launch_activity_count": {
                "median": timing.median_launch_activity_count,
                "iterations": timing.launch_activity_counts,
            },
            "correlated_kernel_activity_count": {
                "median": timing.median_kernel_activity_count,
                "iterations": timing.kernel_activity_counts,
            },
            "host_enqueue_ms": {
                "median": timing.median_host_enqueue_ms,
                "iterations": timing.host_enqueue_times_ms,
            },
            "synchronized_e2e_ms": {
                "median": timing.median_synchronized_e2e_ms,
                "iterations": timing.synchronized_e2e_times_ms,
            },
            "cold_first_call": {
                "host_enqueue_ms": timing.cold_first_call_host_enqueue_ms,
                "synchronized_e2e_ms": timing.cold_first_call_synchronized_e2e_ms,
            },
        }
        flops = case.get("flops")
        bytes_moved = case.get("bytes")
        if flops is not None:
            entry["tflops"] = float(flops) / timing.median_ms / 1e9
        if bytes_moved is not None:
            entry["gbps"] = float(bytes_moved) / timing.median_ms / 1e6
        if case.get("metrics") is not None:
            entry["metrics"] = dict(case["metrics"])
        entry["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - preserve per-shape failure in JSON.
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
    return entry


def run_benchmark(
    *,
    shapes: list[str] | None = None,
    metadata_only: bool = False,
    correctness_only: bool = False,
    warmup_iters: int = 5,
    bench_iters: int = 20,
) -> dict[str, Any]:
    pkg = importlib.import_module(PACKAGE_NAME)
    workload = importlib.import_module("workload")
    selected = _selected_shapes(workload, shapes)
    configured = bool(getattr(workload, "CONFIGURED", bool(selected)))
    entries = [
        {"name": _shape_name(shape, index), "shape": shape, "status": "metadata_only"}
        for index, shape in enumerate(selected)
    ]
    if not metadata_only:
        if not configured or not selected:
            raise RuntimeError(
                "benchmarks/workload.py is not configured; provide --benchmark-adapter during export "
                "or implement SHAPES and make_case()"
            )
        if not correctness_only:
            # CUPTI must be imported before workload adapters import
            # torch, which may otherwise load an incompatible system
            # CUPTI soname first.
            benchmark_runtime = importlib.import_module(f"{PACKAGE_NAME}._benchmark")
            benchmark_runtime.require_cupti()
        entries = [
            _run_shape(
                pkg,
                workload,
                shape,
                index=index,
                correctness_only=correctness_only,
                warmup_iters=warmup_iters,
                bench_iters=bench_iters,
            )
            for index, shape in enumerate(selected)
        ]
    passed = sum(1 for entry in entries if entry["status"] in {"passed", "metadata_only"})
    failed = len(entries) - passed
    return {
        "benchmark": "exported_kernel_shapes",
        "package": PACKAGE_NAME,
        "metadata_only": metadata_only,
        "correctness_only": correctness_only,
        "adapter_configured": configured,
        "timing_backend_requested": "cupti",
        "summary": {
            "shape_count": len(entries),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
        },
        "shapes": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and benchmark exported kernels by shape.")
    parser.add_argument("--shape", action="append", dest="shapes", help="Shape name to run.")
    parser.add_argument("--metadata-only", action="store_true", help="Emit adapter metadata without CUDA.")
    parser.add_argument("--correctness-only", action="store_true", help="Validate without timing.")
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--json", type=Path, help="Write benchmark results as JSON.")
    args = parser.parse_args(argv)
    if args.warmup_iters < 0 or args.bench_iters <= 0:
        parser.error("--warmup-iters must be non-negative and --bench-iters must be positive")
    try:
        payload = run_benchmark(
            shapes=args.shapes,
            metadata_only=args.metadata_only,
            correctness_only=args.correctness_only,
            warmup_iters=args.warmup_iters,
            bench_iters=args.bench_iters,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports configuration failures cleanly.
        parser.error(str(exc))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = payload["summary"]
    print(
        f"benchmark={payload['benchmark']} shapes={summary['shape_count']} "
        f"passed={summary['passed']} failed={summary['failed']}"
    )
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

