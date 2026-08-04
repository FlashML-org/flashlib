#!/usr/bin/env python3
"""Full-shape Cake standalone export (MR 415 head) versus requested production baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

CAKE_COMMIT = "ff502f39df09ffdb317efc57ebdac3a668bb3aa4"
FLASHLIB_COMMIT = "711204f32871af4aeb3ef7ed952cb5eb74c57f46"
KMEANS_BASELINE_COMMIT = "07cf2a27928aacf6790c950a265d8b8dc83c87cf"
EXPECTED_SHAPES = {"flash_kmeans": 124, "knn_build": 112, "knn_search": 198}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _bench_payload(result: Any) -> dict[str, Any]:
    return {
        "gpu_span_ms": result.median_ms,
        "kernel_sum_ms": result.median_kernel_sum_ms,
        "active_union_ms": result.median_active_union_ms,
        "inter_kernel_gap_ms": result.median_inter_kernel_gap_ms,
        "host_enqueue_ms": result.median_host_enqueue_ms,
        "synchronized_e2e_ms": result.median_synchronized_e2e_ms,
        "activity_count": result.median_activity_count,
        "sample_count": len(result.times_ms),
        "backend": result.backend,
        "requested_backend": result.requested_backend,
        "backend_fallback_reason": result.backend_fallback_reason,
        "sm_clock_mhz": result.sm_clock_mhz,
        "sm_clock_max_mhz": result.sm_clock_max_mhz,
        "warmup_iters": result.warmup_iters,
        "warmup_extensions": result.warmup_extensions,
    }


def _dtype_code(name: str) -> int:
    normalized = str(name).replace("torch.", "").lower()
    if normalized in {"bf16", "bfloat16"}:
        return 0
    if normalized in {"fp16", "float16", "half"}:
        return 1
    raise ValueError(f"unsupported dtype {name!r}")


def _shape_setup(domain: str, exported: Any):
    if domain == "flash_kmeans":
        from loom.bench.contracts import flash_kmeans_assign_dispatch_full_contract_eval as full_eval
        from loom.bench.contracts import flash_kmeans_assign_eval as eval_mod

        shapes = list(full_eval.ALL_NAMED_SHAPES)

        def scalars(inputs: dict[str, Any]) -> tuple[int, ...]:
            return (int(inputs["B"]), int(inputs["N"]), int(inputs["D"]), int(inputs["K"]))

        def exported_prepared(inputs: dict[str, Any]) -> dict[str, Any]:
            exported.run(
                inputs["x"], inputs["centroids"], inputs["x_sq"], inputs["c_sq"], inputs["out"],
                *scalars(inputs), arch="sm_100a",
            )
            return {"cluster_ids": inputs["out"]}

        def correctness(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
            expected = eval_mod.reference(inputs)
            return eval_mod._cluster_id_correctness(outputs["cluster_ids"], expected["cluster_ids"], inputs)

    elif domain == "knn_build":
        from loom.bench.contracts import knn_build_eval as eval_mod

        shapes = list(eval_mod.CANONICAL_SHAPES)

        def scalars(inputs: dict[str, Any]) -> tuple[int, ...]:
            return (
                int(inputs["B"]), int(inputs["Q"]), int(inputs["M"]), int(inputs["D"]), int(inputs["K"]),
                _dtype_code(inputs["dtype"]), 1 if bool(inputs["build"]) else 0,
            )

        def exported_prepared(inputs: dict[str, Any]) -> dict[str, Any]:
            exported.run(
                inputs["query"], inputs["database"], inputs["query_sq"], inputs["database_sq"],
                inputs["out_dists"], inputs["out_indices"], *scalars(inputs), arch="sm_100a",
            )
            return {"distances": inputs["out_dists"], "indices": inputs["out_indices"]}

        def correctness(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
            return eval_mod._knn_correctness(outputs, eval_mod.reference(inputs), inputs)

    elif domain == "knn_search":
        from loom.bench.contracts import knn_search_eval as eval_mod
        from loom.export.benchmark_registry import resolve_benchmark_registry_shapes

        registry = json.loads(Path("benchmark_data.json").read_text(encoding="utf-8"))["knn_search"]
        selectors = registry["dispatcher_per_shape_performance_gate"]["shape_selectors"]
        shapes = list(resolve_benchmark_registry_shapes("knn_search", selectors)["union"])

        def scalars(inputs: dict[str, Any]) -> tuple[int, ...]:
            return (
                int(inputs["B"]), int(inputs["Q"]), int(inputs["M"]), int(inputs["D"]), int(inputs["K"]),
                1 if bool(inputs["self_search"]) else 0,
            )

        def exported_prepared(inputs: dict[str, Any]) -> dict[str, Any]:
            exported.run(
                inputs["queries"], inputs["database"], inputs["out_distances"], inputs["out_indices"],
                *scalars(inputs), arch="sm_100a",
            )
            return {"distances": inputs["out_distances"], "indices": inputs["out_indices"]}

        def correctness(inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
            return eval_mod._knn_correctness(outputs, eval_mod.reference(inputs), inputs)

    else:
        raise ValueError(domain)

    if len(shapes) != EXPECTED_SHAPES[domain]:
        raise RuntimeError(f"{domain}: expected {EXPECTED_SHAPES[domain]} shapes, got {len(shapes)}")
    return eval_mod, shapes, scalars, exported_prepared, correctness


def _make_raw_calls(
    domain: str,
    exported: Any,
    inputs: dict[str, Any],
    scalars: Callable[[dict[str, Any]], tuple[int, ...]],
    baseline: Callable[..., Any],
):
    import torch

    candidate_holder: list[dict[str, Any] | None] = [None]
    baseline_holder: list[dict[str, Any] | None] = [None]

    if domain == "flash_kmeans":
        x, c = inputs["x"], inputs["centroids"]

        def candidate_call():
            x_sq = (x.float() ** 2).sum(-1).contiguous()
            c_sq = (c.float() ** 2).sum(-1).contiguous()
            out = torch.empty((int(inputs["B"]), int(inputs["N"])), dtype=torch.int32, device=x.device)
            exported.run(x, c, x_sq, c_sq, out, *scalars(inputs), arch="sm_100a")
            candidate_holder[0] = {"cluster_ids": out}
            return candidate_holder[0]

        def baseline_call():
            x_sq = (x.float() ** 2).sum(-1).contiguous()
            c_sq = (c.float() ** 2).sum(-1).contiguous()
            out = torch.empty((int(inputs["B"]), int(inputs["N"])), dtype=torch.int32, device=x.device)
            baseline(x, c, x_sq, c_sq, out=out)
            baseline_holder[0] = {"cluster_ids": out}
            return baseline_holder[0]

        prepared_baseline_out = torch.empty_like(inputs["out"])

        def candidate_prepared_call():
            exported.run(
                x, c, inputs["x_sq"], inputs["c_sq"], inputs["out"], *scalars(inputs), arch="sm_100a"
            )

        def baseline_prepared_call():
            baseline(x, c, inputs["x_sq"], inputs["c_sq"], out=prepared_baseline_out)

        prepared_calls = (candidate_prepared_call, baseline_prepared_call)

    elif domain == "knn_build":
        query, database = inputs["query"], inputs["database"]
        shape_out = (int(inputs["B"]), int(inputs["Q"]), int(inputs["K"]))

        def candidate_call():
            query_sq = (query.float() ** 2).sum(-1).contiguous()
            database_sq = query_sq if query.data_ptr() == database.data_ptr() else (database.float() ** 2).sum(-1).contiguous()
            out_dists = torch.empty(shape_out, dtype=torch.float32, device=query.device)
            out_indices = torch.empty(shape_out, dtype=torch.int32, device=query.device)
            exported.run(
                query, database, query_sq, database_sq, out_dists, out_indices,
                *scalars(inputs), arch="sm_100a",
            )
            candidate_holder[0] = {"distances": out_dists, "indices": out_indices}
            return candidate_holder[0]

        def baseline_call():
            distances, indices = baseline(query, database, k=int(inputs["K"]))
            baseline_holder[0] = {"distances": distances, "indices": indices}
            return baseline_holder[0]

        def candidate_prepared_call():
            exported.run(
                query,
                database,
                inputs["query_sq"],
                inputs["database_sq"],
                inputs["out_dists"],
                inputs["out_indices"],
                *scalars(inputs),
                arch="sm_100a",
            )

        # FlashLib has no precomputed-norm entrypoint. Reuse the same measured
        # raw baseline GPU span so this diagnostic isolates the generated Cake
        # dispatch with its required norms prepared outside timing.
        prepared_calls = (candidate_prepared_call, None)

    else:
        query, database = inputs["queries"], inputs["database"]
        shape_out = (int(inputs["B"]), int(inputs["Q"]), int(inputs["K"]))

        def candidate_call():
            out_distances = torch.empty(shape_out, dtype=torch.float32, device=query.device)
            out_indices = torch.empty(shape_out, dtype=torch.int32, device=query.device)
            exported.run(query, database, out_distances, out_indices, *scalars(inputs), arch="sm_100a")
            candidate_holder[0] = {"distances": out_distances, "indices": out_indices}
            return candidate_holder[0]

        def baseline_call():
            distances, indices = baseline(query, database, k=int(inputs["K"]))
            baseline_holder[0] = {"distances": distances, "indices": indices}
            return baseline_holder[0]

        def candidate_prepared_call():
            exported.run(
                query,
                database,
                inputs["out_distances"],
                inputs["out_indices"],
                *scalars(inputs),
                arch="sm_100a",
            )

        # Search has no norm inputs; this lane removes only default-output
        # allocation and uses the already measured public FlashLib baseline.
        prepared_calls = (candidate_prepared_call, None)

    return candidate_call, baseline_call, candidate_holder, baseline_holder, prepared_calls


def _measure(fn: Callable[[], Any]) -> Any:
    from loom.bench import bench_gpu_time_warm

    result = bench_gpu_time_warm(
        fn,
        warmup_iters=10,
        bench_iters=16,
        max_warmup_extensions=8,
        cold_l2=True,
        use_cupti=True,
    )
    if result.backend != "cupti" or result.backend_fallback_reason is not None:
        raise RuntimeError(f"CUPTI required, got {result.backend}: {result.backend_fallback_reason}")
    return result


def _run_one(
    domain: str,
    shape: dict[str, Any],
    eval_mod: Any,
    exported: Any,
    scalars: Callable[[dict[str, Any]], tuple[int, ...]],
    exported_prepared: Callable[[dict[str, Any]], dict[str, Any]],
    correctness: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    baseline: Callable[..., Any],
) -> dict[str, Any]:
    import torch

    label = str(shape["label"])
    params = dict(shape.get("params", {}))
    inputs = eval_mod._make_inputs({**params, "label": label})
    route = str(exported.route_id(*scalars(inputs), arch="sm_100a"))

    prepared_output = exported_prepared(inputs)
    torch.cuda.synchronize()
    prepared_correctness = correctness(inputs, prepared_output)
    candidate_call, baseline_call, candidate_holder, baseline_holder, prepared_calls = _make_raw_calls(
        domain, exported, inputs, scalars, baseline
    )
    candidate_call()
    baseline_call()
    torch.cuda.synchronize()
    candidate_correctness = correctness(inputs, candidate_holder[0])
    baseline_correctness = correctness(inputs, baseline_holder[0])
    if not all(bool(item.get("passed")) for item in (prepared_correctness, candidate_correctness, baseline_correctness)):
        raise RuntimeError(
            f"correctness failed: prepared={prepared_correctness!r} candidate={candidate_correctness!r} "
            f"baseline={baseline_correctness!r}"
        )

    raw_order = ["candidate", "baseline"]
    if hashlib.sha256(label.encode("utf-8")).digest()[0] & 1:
        raw_order.reverse()
    timers = {"candidate": candidate_call, "baseline": baseline_call}
    timings = {role: _measure(timers[role]) for role in raw_order}
    candidate_timing = timings["candidate"]
    baseline_timing = timings["baseline"]
    candidate_gpu = float(candidate_timing.median_ms)
    baseline_gpu = float(baseline_timing.median_ms)
    candidate_e2e = float(candidate_timing.median_synchronized_e2e_ms)
    baseline_e2e = float(baseline_timing.median_synchronized_e2e_ms)
    row = {
        "label": label,
        "params": params,
        "scalars": list(scalars(inputs)),
        "route": route,
        "correctness": {
            "candidate_prepared": prepared_correctness,
            "candidate_raw": candidate_correctness,
            "baseline_raw": baseline_correctness,
        },
        "measurement_order": raw_order,
        "candidate": _bench_payload(candidate_timing),
        "baseline": _bench_payload(baseline_timing),
        "gpu_span_speedup": baseline_gpu / candidate_gpu,
        "synchronized_e2e_speedup": baseline_e2e / candidate_e2e,
    }

    if prepared_calls is not None:
        if prepared_calls[1] is None:
            prepared_order = ["candidate"]
            prepared_timings = {"candidate": _measure(prepared_calls[0]), "baseline": baseline_timing}
            baseline_boundary = "same-shape raw FlashLib baseline reused from the paired raw lane"
        else:
            prepared_order = ["candidate", "baseline"]
            if hashlib.sha256((label + ":prepared").encode("utf-8")).digest()[0] & 1:
                prepared_order.reverse()
            prepared_timers = {"candidate": prepared_calls[0], "baseline": prepared_calls[1]}
            prepared_timings = {role: _measure(prepared_timers[role]) for role in prepared_order}
            baseline_boundary = "precomputed baseline measured in its own paired lane"
        row["prepared"] = {
            "measurement_order": prepared_order,
            "baseline_boundary": baseline_boundary,
            "candidate": _bench_payload(prepared_timings["candidate"]),
            "baseline": _bench_payload(prepared_timings["baseline"]),
            "gpu_span_speedup": (
                float(prepared_timings["baseline"].median_ms) / float(prepared_timings["candidate"].median_ms)
            ),
            "synchronized_e2e_speedup": (
                float(prepared_timings["baseline"].median_synchronized_e2e_ms)
                / float(prepared_timings["candidate"].median_synchronized_e2e_ms)
            ),
        }
    return row


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "geomean": math.exp(statistics.fmean(math.log(value) for value in values)),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "max": max(values),
        "below_1x_count": sum(value < 1.0 for value in values),
    }


def _summary(domain: str, rows: list[dict[str, Any]], health: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    raw_gpu = [float(row["gpu_span_speedup"]) for row in rows if "gpu_span_speedup" in row]
    raw_e2e = [float(row["synchronized_e2e_speedup"]) for row in rows if "synchronized_e2e_speedup" in row]
    prepared_gpu = [float(row["prepared"]["gpu_span_speedup"]) for row in rows if "prepared" in row]
    failed = [row for row in rows if "exception" in row]
    summary = {
        "domain": domain,
        "expected_shape_count": EXPECTED_SHAPES[domain],
        "completed_shape_count": len(rows),
        "passed_shape_count": len(rows) - len(failed),
        "failed_shape_count": len(failed),
        "raw_operator_gpu_span_speedup": _distribution(raw_gpu) if raw_gpu else None,
        "raw_operator_synchronized_e2e_speedup": _distribution(raw_e2e) if raw_e2e else None,
        "health": health,
        "elapsed_s": elapsed_s,
    }
    prepared_key = (
        "prepared_assignment_gpu_span_speedup"
        if domain == "flash_kmeans"
        else "prepared_candidate_gpu_span_speedup_vs_raw_baseline"
    )
    summary[prepared_key] = _distribution(prepared_gpu) if prepared_gpu else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(EXPECTED_SHAPES), required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reuse-export-root", required=True)
    parser.add_argument("--flashlib-root")
    parser.add_argument("--kmeans-baseline-root")
    args = parser.parse_args()

    import torch

    from loom.bench import pre_warm_gpu
    from loom.bench.gpu_health import healthy_gpu_bench_session
    from loom.runtime.compiler import detect_gpu_arch

    started = time.time()
    torch.cuda.set_device(0)
    arch = str(detect_gpu_arch())
    if arch != "sm_100a":
        raise RuntimeError(f"B200 sm_100a required, detected {arch}")

    sys.path.insert(0, str(Path(args.reuse_export_root).resolve()))
    exported = __import__(f"{args.domain}.interface", fromlist=["interface"])
    if args.domain == "flash_kmeans":
        if not args.kmeans_baseline_root:
            raise RuntimeError("--kmeans-baseline-root is required")
        sys.path.insert(0, str(Path(args.kmeans_baseline_root).resolve()))
        from flash_kmeans_triton_h200 import euclid_assign_triton_h200 as baseline
        baseline_identity = {"name": "flash_kmeans_triton_h200", "commit": KMEANS_BASELINE_COMMIT}
    else:
        if not args.flashlib_root:
            raise RuntimeError("--flashlib-root is required")
        sys.path.insert(0, str(Path(args.flashlib_root).resolve()))
        import flashlib

        baseline = flashlib.flash_knn
        baseline_identity = {
            "name": "flashlib.flash_knn",
            "commit": FLASHLIB_COMMIT,
            "module": str(Path(flashlib.__file__).resolve()),
        }

    eval_mod, shapes, scalars, exported_prepared, correctness = _shape_setup(args.domain, exported)
    output_root = Path(args.output_root).resolve()
    result_path = output_root / f"{args.domain}-baseline.json"
    progress_path = output_root / f"{args.domain}-baseline.progress.json"
    rows: list[dict[str, Any]] = []
    with healthy_gpu_bench_session(require_cupti=True) as health:
        pre_warm_gpu(verbose=True)
        for index, shape in enumerate(shapes):
            try:
                row = _run_one(
                    args.domain, shape, eval_mod, exported, scalars, exported_prepared, correctness, baseline
                )
            except Exception as exc:
                row = {
                    "label": str(shape["label"]),
                    "params": shape.get("params", {}),
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
            rows.append(row)
            partial = {
                "schema": "mr405-flashlib-baseline-v1",
                "candidate_commit": CAKE_COMMIT,
                "baseline": baseline_identity,
                "completed": len(rows),
                "expected": EXPECTED_SHAPES[args.domain],
                "rows": rows,
            }
            _atomic_json(progress_path, partial)
            print(
                f"[{args.domain}] {index + 1}/{len(shapes)} {row['label']} "
                f"gpu_speedup={row.get('gpu_span_speedup')} error={row.get('exception')}",
                flush=True,
            )

    summary = _summary(args.domain, rows, health, time.time() - started)
    payload = {
        "schema": "mr405-flashlib-baseline-v1",
        "candidate": {
            "name": f"Cake standalone {args.domain}",
            "commit": CAKE_COMMIT,
            "timing_boundary": "raw inputs, default output, exported interface.run; torch row norms where required",
        },
        "baseline": baseline_identity,
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "arch": arch,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "summary": summary,
        "rows": rows,
    }
    _atomic_json(result_path, payload)
    print(f"[{args.domain}] SUMMARY {json.dumps(summary, sort_keys=True)}", flush=True)
    if summary["failed_shape_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
