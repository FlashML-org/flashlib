from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import uuid
from itertools import permutations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flashlib_cake_knn_build._benchmark import (  # noqa: E402
    bench_gpu_time,
    compare_runtime_lifecycles,
    measure_host_call,
    require_cupti,
    runtime_lifecycle_metrics,
)

SHAPE_RECORDS = json.loads((Path(__file__).with_name("shape_records.json")).read_text(encoding="utf-8"))
ROUTE_MANIFEST = json.loads((Path(__file__).with_name("expected_routes.json")).read_text(encoding="utf-8"))
EXPECTED_ROUTES = {row["shape"]: row["selected_route"] for row in ROUTE_MANIFEST}
SEMANTIC_ENTRYPOINT = (
    "loom.examples.weave.knn_build_dispatch_q1m524_v10_d320recurrence_consumption_v1:launch_from_contract_inputs"
)
SHAPES: dict[str, dict[str, Any]] = {
    row["label"]: {**row["params"], "recorded": row["recorded"]} for row in SHAPE_RECORDS
}
BASELINE_NAME = "flashlib.flash_knn"
MEASUREMENT_ORDER_SEED = "flashlib-knn-build-runtime-compute-paired-v2"
_COLD_ORDERED_LABELS = tuple(
    sorted(SHAPES, key=lambda label: hashlib.sha256(f"{MEASUREMENT_ORDER_SEED}:cold:{label}".encode()).digest())
)
_CANDIDATE_FIRST_COLD_LABELS = frozenset(_COLD_ORDERED_LABELS[::2])


def _measurement_order(label: str) -> tuple[str, str, str]:
    """Choose one stable per-shape order for baseline/compute/prepared timing."""

    orders = tuple(permutations(("baseline", "compute", "prepared")))
    digest = hashlib.sha256(f"{MEASUREMENT_ORDER_SEED}:{label}".encode()).digest()
    return orders[int.from_bytes(digest[:2], "little") % len(orders)]


def _cold_measurement_order(label: str) -> tuple[str, str]:
    """Counterbalance process-shared cold effects across contract shapes."""

    return ("candidate", "baseline") if label in _CANDIDATE_FIRST_COLD_LABELS else ("baseline", "candidate")


def _measurement_session_fields(measurement_session_id: str) -> dict[str, Any]:
    if not isinstance(measurement_session_id, str) or not measurement_session_id.strip():
        raise ValueError("measurement_session_id must be a non-empty string")
    return {
        "measurement_session_id": measurement_session_id,
        "baseline_measurement_session_id": measurement_session_id,
        "compute_measurement_session_id": measurement_session_id,
        "prepared_measurement_session_id": measurement_session_id,
        "baseline_compute_prepared_same_session": True,
    }


def _recorded_diagnostics(recorded: dict[str, Any]) -> dict[str, Any]:
    return {f"recorded_{key}": value for key, value in recorded.items()}


def _shape_metadata(shape: dict[str, Any]) -> dict[str, Any]:
    return {
        **{key: value for key, value in shape.items() if key != "recorded"},
        **_recorded_diagnostics(shape["recorded"]),
    }


def _load_flashlib_baseline():
    flash_knn = getattr(importlib.import_module("flashlib"), "flash_knn", None)
    if not callable(flash_knn):
        raise RuntimeError("flashlib.flash_knn is required for same-process KNN-build benchmarking")
    return flash_knn


def _require_correct_baseline(name: str, diagnostics: dict[str, Any]) -> None:
    if bool(diagnostics.get("correct")):
        return
    raise RuntimeError(
        f"{BASELINE_NAME} correctness failed for {name}: "
        f"recall={diagnostics.get('recall')!r}, "
        f"max_abs_dist_error={diagnostics.get('max_abs_dist_error')!r}, "
        f"required_recall={diagnostics.get('required_recall')!r}"
    )


def _host_call_diagnostics(timing: Any) -> dict[str, float] | None:
    if timing is None:
        return None
    return {
        "host_enqueue_ms": timing.host_enqueue_ms,
        "synchronized_e2e_ms": timing.synchronized_e2e_ms,
    }


def _timing_diagnostics(timing: Any) -> dict[str, Any]:
    return {
        "official_gpu_metric": "gpu_span_ms",
        "gpu_span_ms": {"median": timing.median_gpu_span_ms, "iterations": timing.times_ms},
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


def _write_json_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_inputs(shape: dict[str, Any], *, seed_offset: int = 0):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(shape["seed"]) + int(seed_offset))
    database = torch.randn(
        (int(shape["B"]), int(shape["M"]), int(shape["D"])),
        dtype=torch.float16 if shape.get("dtype") == "float16" else torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    if bool(shape.get("build", False)):
        query = database
    else:
        query = torch.randn(
            (int(shape["B"]), int(shape["Q"]), int(shape["D"])),
            dtype=database.dtype,
            device="cuda",
            generator=generator,
        ).contiguous()
    return query, database


def _alternating_call(fn: Any, first: tuple[Any, Any], second: tuple[Any, Any]):
    """Return a call that alternates two equal-shape, different-pointer inputs."""

    pairs = (first, second)
    iteration = 0

    def call():
        nonlocal iteration
        query, database = pairs[iteration & 1]
        iteration += 1
        return fn(query, database)

    return call


def _correctness_query_indices(shape: dict[str, Any], query):
    import torch

    n_query = int(query.shape[1])
    sample_count = int(shape.get("correctness_query_sample", n_query))
    sample_count = min(sample_count, n_query)
    if sample_count <= 0:
        raise ValueError("correctness_query_sample must select at least one query")
    if sample_count == n_query:
        return torch.arange(n_query, dtype=torch.int64, device=query.device)
    return torch.linspace(0, n_query - 1, sample_count, device=query.device).round().to(torch.int64).unique()


def _reference_topk(query, database, k: int, query_indices=None):
    import torch

    values: list[Any] = []
    indices: list[Any] = []
    db_f32 = database.float()
    db_sq = (db_f32 * db_f32).sum(-1)
    block = 256
    query_f32 = query.float() if query_indices is None else query[:, query_indices.long(), :].float()
    for start in range(0, int(query_f32.shape[1]), block):
        query_block = query_f32[:, start : start + block, :]
        q_sq = (query_block * query_block).sum(-1)
        dots = torch.matmul(query_block, db_f32.transpose(-1, -2))
        dists = q_sq.unsqueeze(-1) + db_sq.unsqueeze(1) - 2.0 * dots
        vals, idx = torch.topk(dists, k, dim=-1, largest=False, sorted=True)
        values.append(vals)
        indices.append(idx.to(torch.int32))
    return torch.cat(values, dim=1), torch.cat(indices, dim=1)


def _recall(got_indices, expected_indices) -> float:
    matches = (got_indices.unsqueeze(-1) == expected_indices.unsqueeze(-2)).any(dim=-1)
    return float(matches.to(got_indices.device, dtype=got_indices.float().dtype).mean().item())


def _distances_for_indices(query, database, indices):
    import torch

    db_f32 = database.float()
    bsz, n_rows, _ = db_f32.shape
    safe_indices = indices.to(torch.int64).clamp(0, n_rows - 1)
    batch = torch.arange(bsz, device=database.device)[:, None, None]
    neighbors = db_f32[batch, safe_indices]
    return ((query.float().unsqueeze(2) - neighbors) ** 2).sum(-1)


def _knn_correctness_diagnostics(
    query,
    database,
    output,
    expected_indices,
    *,
    query_indices,
    required_recall: float,
) -> dict[str, Any]:
    sampled_query = query[:, query_indices.long(), :]
    sampled_distances = output[0][:, query_indices.long(), :]
    sampled_indices = output[1][:, query_indices.long(), :]
    exact_dists = _distances_for_indices(sampled_query, database, sampled_indices)
    recall = _recall(sampled_indices, expected_indices)
    max_abs_dist_error = float((sampled_distances - exact_dists).abs().max().item())
    return {
        "recall": recall,
        "max_abs_dist_error": max_abs_dist_error,
        "required_recall": required_recall,
        "sampled_queries": int(query_indices.numel()),
        "correct": bool(recall >= required_recall and max_abs_dist_error <= 1.0e-2),
    }


def _run_shape(
    name: str,
    shape: dict[str, Any],
    *,
    runtime: Any,
    runtime_init_timing: Any = None,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    measurement_session_id: str | None = None,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_knn_build import knn_build_prepared, prepare_knn_build

    session_id = measurement_session_id or uuid.uuid4().hex
    session_fields = _measurement_session_fields(session_id)
    query_a, database_a = _make_inputs(shape)
    query_b, database_b = _make_inputs(shape, seed_offset=1_000_003)
    if database_a.data_ptr() == database_b.data_ptr() or query_a.data_ptr() == query_b.data_ptr():
        raise RuntimeError("fresh-pointer KNN-build input pair unexpectedly aliases")
    # Keep the cold public-call brackets honest: fixture generation is
    # asynchronous and must complete before the baseline/slot-miss timestamp.
    torch.cuda.synchronize()
    k = int(shape["K"])
    build = bool(shape.get("build", False))

    def compute(query, database, *, return_info: bool = False):
        return runtime.compute(
            query,
            database,
            k,
            build=build,
            shape_label=name,
            return_info=return_info,
        )

    baseline_out = None
    baseline_cold_first_call = None
    first_shape_lookup_timing = None
    fresh_pointer_hit_timing = None
    prepare_cold_call = None
    prepared_cold_first_call = None
    cold_measurement_order = _cold_measurement_order(name)
    if benchmark:
        flash_knn = _load_flashlib_baseline()
        for lane in cold_measurement_order:
            if lane == "baseline":
                baseline_out, baseline_cold_first_call = measure_host_call(lambda: flash_knn(query_b, database_b, k=k))
            else:
                (first_out, first_route_info), first_shape_lookup_timing = measure_host_call(
                    lambda: compute(query_a, database_a, return_info=True)
                )
        (out, route_info), fresh_pointer_hit_timing = measure_host_call(
            lambda: compute(query_b, database_b, return_info=True)
        )
        if not bool(route_info.get("runtime_cache_hit")):
            raise RuntimeError("fresh-pointer KNN-build call did not hit the runtime shape cache")
        if first_route_info["selected_route"] != route_info["selected_route"]:
            raise RuntimeError(
                "fresh-pointer KNN-build call changed routes: "
                f"{first_route_info['selected_route']!r} != {route_info['selected_route']!r}"
            )
        if first_route_info["norm_compute_fields"] != route_info["norm_compute_fields"]:
            raise RuntimeError(
                "fresh-pointer KNN-build call changed the route-specific norm plan: "
                f"{first_route_info['norm_compute_fields']!r} != "
                f"{route_info['norm_compute_fields']!r}"
            )
        if any(left.data_ptr() == right.data_ptr() for left, right in zip(first_out, out, strict=True)):
            raise RuntimeError("fresh-pointer KNN-build call reused the prior default output allocation")
        if route_info.get("hot_launch_path") != "cuda_graph":
            raise RuntimeError(
                "KNN-build signature did not capture into a CUDA graph: "
                f"hot_launch_path={route_info.get('hot_launch_path')!r}, "
                f"graph_capture_error={route_info.get('graph_capture_error')!r}; "
                "every contract signature freezes to a LaunchPlan today, so a "
                "prepared-path fallback means the graph runtime silently degraded"
            )
        prepared, prepare_cold_call = measure_host_call(
            lambda: prepare_knn_build(
                query_a,
                database_a,
                k,
                build=build,
                shape_label=name,
                arch=arch,
            )
        )
        (_, prepared_route_info), prepared_cold_first_call = measure_host_call(
            lambda: knn_build_prepared(prepared, return_info=True)
        )
        if first_route_info["selected_route"] != prepared_route_info["selected_route"]:
            raise RuntimeError(
                "runtime and prepared KNN-build calls selected different routes: "
                f"{first_route_info['selected_route']!r} != "
                f"{prepared_route_info['selected_route']!r}"
            )
    else:
        first_out, first_route_info = compute(query_a, database_a, return_info=True)
        out, route_info = compute(query_b, database_b, return_info=True)
        if not bool(route_info.get("runtime_cache_hit")):
            raise RuntimeError("fresh-pointer KNN-build call did not hit the runtime shape cache")
        if first_route_info["selected_route"] != route_info["selected_route"]:
            raise RuntimeError("fresh-pointer KNN-build call changed routes")
        if route_info.get("hot_launch_path") != "cuda_graph":
            raise RuntimeError(
                "KNN-build signature did not capture into a CUDA graph: "
                f"hot_launch_path={route_info.get('hot_launch_path')!r}, "
                f"graph_capture_error={route_info.get('graph_capture_error')!r}"
            )
        torch.cuda.synchronize()

    result: dict[str, Any] = {
        "shape": name,
        "B": int(shape["B"]),
        "Q": int(shape["Q"]),
        "M": int(shape["M"]),
        "D": int(shape["D"]),
        "K": k,
        "semantic_entrypoint": route_info["semantic_entrypoint"],
        "selected_route": route_info["selected_route"],
        "expected_route": EXPECTED_ROUTES[name],
        "route_matches_expected": route_info["selected_route"] == EXPECTED_ROUTES[name],
        "baseline_name": BASELINE_NAME,
        "baseline_entrypoint": BASELINE_NAME,
        "first_shape_lookup_cache_hit": bool(first_route_info.get("runtime_cache_hit")),
        "fresh_pointer_cache_hit": bool(route_info.get("runtime_cache_hit")),
        "fresh_pointer_rebind_verified": True,
        "hot_launch_path": route_info.get("hot_launch_path"),
        "graph_kernel_count": route_info.get("graph_kernel_count"),
        "graph_capture_error": route_info.get("graph_capture_error"),
        **session_fields,
        **_recorded_diagnostics(shape["recorded"]),
    }
    if benchmark:
        result.update(
            {
                "cold_measurement_order": list(cold_measurement_order),
                "cold_measurement_order_policy": "deterministic_balanced_per_publication_contract_portfolio",
                "prepared_launch_count": route_info["prepared_launch_count"],
                "compute_launch_count": route_info["runtime_launch_count"],
                "compute_norm_launch_count": route_info["norm_launch_count"],
                "compute_norm_policy": route_info["norm_mode"],
                "compute_norm_fields": route_info["norm_compute_fields"],
                "cold_baseline_call": _host_call_diagnostics(baseline_cold_first_call),
                "cold_compute_first_shape_lookup": _host_call_diagnostics(first_shape_lookup_timing),
                "cold_compute_fresh_pointer_hit": _host_call_diagnostics(fresh_pointer_hit_timing),
                "prepared_setup_after_runtime_hit": _host_call_diagnostics(prepare_cold_call),
                "cold_prepared_call": _host_call_diagnostics(prepared_cold_first_call),
                "runtime_cache_info_after_fresh_pointer_hit": runtime.cache_info(),
            }
        )

    first_ref_indices = None
    ref_indices = None
    query_indices = None
    if correctness or benchmark:
        query_indices = _correctness_query_indices(shape, query_b)
        _, ref_indices = _reference_topk(query_b, database_b, k, query_indices)
        if correctness:
            _, first_ref_indices = _reference_topk(query_a, database_a, k, query_indices)
        torch.cuda.synchronize()

    if correctness:
        required_recall = float(shape.get("recall_min", 0.999))
        first_candidate_correctness = _knn_correctness_diagnostics(
            query_a,
            database_a,
            first_out,
            first_ref_indices,
            query_indices=query_indices,
            required_recall=required_recall,
        )
        candidate_correctness = _knn_correctness_diagnostics(
            query_b,
            database_b,
            out,
            ref_indices,
            query_indices=query_indices,
            required_recall=required_recall,
        )
        result["first_pointer_correctness"] = first_candidate_correctness
        result["fresh_pointer_correctness"] = candidate_correctness
        if not first_candidate_correctness["correct"] or not candidate_correctness["correct"]:
            raise RuntimeError(
                f"runtime.compute fresh-pointer correctness failed for {name}: "
                f"first={first_candidate_correctness!r}, fresh={candidate_correctness!r}"
            )
        rebound_out, rebound_info = compute(query_a, database_a, return_info=True)
        torch.cuda.synchronize()
        rebound_correctness = _knn_correctness_diagnostics(
            query_a,
            database_a,
            rebound_out,
            first_ref_indices,
            query_indices=query_indices,
            required_recall=required_recall,
        )
        if not rebound_info.get("runtime_cache_hit") or not rebound_correctness["correct"]:
            raise RuntimeError(
                f"runtime.compute B-to-A pointer rebound failed for {name}: "
                f"info={rebound_info!r}, correctness={rebound_correctness!r}"
            )
        result["return_pointer_correctness"] = rebound_correctness
        result.update(candidate_correctness)

    if benchmark:
        baseline_correctness = _knn_correctness_diagnostics(
            query_b,
            database_b,
            baseline_out,
            ref_indices,
            query_indices=query_indices,
            required_recall=float(shape.get("recall_min", 0.999)),
        )
        result.update({f"baseline_{key}": value for key, value in baseline_correctness.items()})
        _require_correct_baseline(name, baseline_correctness)

        baseline_output_holder = [baseline_out]
        compute_output_holder = [out]

        def baseline_for(query, database):
            baseline_output_holder[0] = flash_knn(query, database, k=k)
            return baseline_output_holder[0]

        def compute_for(query, database):
            compute_output_holder[0] = compute(query, database)
            return compute_output_holder[0]

        run_baseline = _alternating_call(
            baseline_for,
            (query_a, database_a),
            (query_b, database_b),
        )
        run_compute = _alternating_call(
            compute_for,
            (query_a, database_a),
            (query_b, database_b),
        )

        def time_baseline():
            return bench_gpu_time(
                run_baseline,
                cold_l2=True,
                cold_first_call=baseline_cold_first_call,
            )

        def time_compute():
            return bench_gpu_time(
                run_compute,
                cold_l2=True,
                cold_first_call=fresh_pointer_hit_timing,
            )

        def time_prepared():
            return bench_gpu_time(
                lambda: knn_build_prepared(prepared),
                cold_l2=True,
                cold_first_call=prepared_cold_first_call,
            )

        measurement_order = _measurement_order(name)
        timers = {
            "baseline": time_baseline,
            "compute": time_compute,
            "prepared": time_prepared,
        }
        timings = {timer_name: timers[timer_name]() for timer_name in measurement_order}
        baseline_timing = timings["baseline"]
        compute_timing = timings["compute"]
        prepared_timing = timings["prepared"]
        timing_backends = {
            baseline_timing.backend,
            compute_timing.backend,
            prepared_timing.backend,
        }
        if timing_backends != {"cupti"}:
            raise RuntimeError(f"baseline/compute/prepared must all use CUPTI, got {sorted(timing_backends)!r}")

        baseline_e2e_ms = baseline_timing.median_synchronized_e2e_ms
        compute_e2e_ms = compute_timing.median_synchronized_e2e_ms
        if baseline_e2e_ms is None or compute_e2e_ms is None:
            raise RuntimeError("CUPTI benchmark did not report synchronized E2E timing")

        result.update(
            {
                "measurement_order": list(measurement_order),
                "measurement_order_policy": "deterministic_sha256_per_shape_permutation",
                "baseline_ms": baseline_timing.median_ms,
                "baseline_gpu_span_ms": baseline_timing.median_gpu_span_ms,
                "baseline_kernel_span_ms": baseline_timing.median_gpu_span_ms,
                "baseline_kernel_sum_ms": baseline_timing.median_kernel_sum_ms,
                "baseline_active_union_ms": baseline_timing.median_active_union_ms,
                "baseline_inter_kernel_gap_ms": baseline_timing.median_inter_kernel_gap_ms,
                "baseline_synchronized_e2e_ms": baseline_e2e_ms,
                "baseline_timing_backend": baseline_timing.backend,
                "baseline_bench_iters": len(baseline_timing.times_ms),
                "baseline_timing_diagnostics": _timing_diagnostics(baseline_timing),
                "kernel_ms": compute_timing.median_gpu_span_ms,
                "compute_gpu_span_ms": compute_timing.median_gpu_span_ms,
                "compute_kernel_span_ms": compute_timing.median_gpu_span_ms,
                "compute_gpu_span_scope": "correlated_kernel_activity_only_excludes_memcpy_and_pre_kernel_host_work",
                "compute_kernel_sum_ms": compute_timing.median_kernel_sum_ms,
                "compute_active_union_ms": compute_timing.median_active_union_ms,
                "compute_inter_kernel_gap_ms": compute_timing.median_inter_kernel_gap_ms,
                "compute_host_enqueue_ms": compute_timing.median_host_enqueue_ms,
                "compute_synchronized_e2e_ms": compute_e2e_ms,
                "compute_timing_diagnostics": _timing_diagnostics(compute_timing),
                "prepared_gpu_span_ms": prepared_timing.median_gpu_span_ms,
                "prepared_kernel_sum_ms": prepared_timing.median_kernel_sum_ms,
                "prepared_active_union_ms": prepared_timing.median_active_union_ms,
                "prepared_inter_kernel_gap_ms": prepared_timing.median_inter_kernel_gap_ms,
                "prepared_timing_diagnostics": _timing_diagnostics(prepared_timing),
                "compute_gpu_over_prepared": (compute_timing.median_gpu_span_ms / prepared_timing.median_gpu_span_ms),
                "timing_backend": compute_timing.backend,
                "bench_iters": len(compute_timing.times_ms),
                "compute_gpu_speedup_vs_baseline": (
                    baseline_timing.median_gpu_span_ms / compute_timing.median_gpu_span_ms
                ),
                "compute_speedup_vs_baseline": baseline_e2e_ms / compute_e2e_ms,
                "speedup_vs_baseline": (baseline_timing.median_gpu_span_ms / compute_timing.median_gpu_span_ms),
                "prepared_speedup_vs_baseline": (baseline_timing.median_ms / prepared_timing.median_ms),
            }
        )
        candidate_lifecycle = runtime_lifecycle_metrics(
            api="flashlib_cake_knn_build.init().compute",
            measurement_session_id=session_id,
            timing_boundary="raw_inputs_default_output_route_required_norms_synchronized_e2e",
            output_policy="default_output_allocated_inside_compute",
            init=runtime_init_timing,
            init_sample_id=session_id,
            first_compute=first_shape_lookup_timing,
            first_cache_state=("shape_slot_hit" if first_route_info.get("runtime_cache_hit") else "shape_slot_miss"),
            hot_compute=compute_timing,
            hot_cache_state="fresh_pointer_shape_slot_hit",
            code_cache_state="process_order_dependent",
        )
        baseline_lifecycle = runtime_lifecycle_metrics(
            api=BASELINE_NAME,
            measurement_session_id=session_id,
            timing_boundary="raw_inputs_default_output_synchronized_e2e",
            output_policy="default_output_allocated_inside_flashlib_call",
            init=None,
            init_sample_id=None,
            first_compute=baseline_cold_first_call,
            first_cache_state="first_public_call",
            hot_compute=baseline_timing,
            hot_cache_state="repeated_public_call",
            code_cache_state="process_order_dependent",
        )
        lifecycle_comparison = compare_runtime_lifecycles(candidate_lifecycle, baseline_lifecycle)
        if not math.isclose(
            lifecycle_comparison["hot_synchronized_e2e_speedup"],
            result["compute_speedup_vs_baseline"],
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError("KNN-build lifecycle hot E2E speedup disagrees with publication speedup")
        result["candidate_runtime_lifecycle"] = candidate_lifecycle
        result["baseline_runtime_lifecycle"] = baseline_lifecycle
        result["runtime_lifecycle_comparison"] = lifecycle_comparison
        flops = 2.0 * int(shape["B"]) * int(shape["Q"]) * int(shape["M"]) * int(shape["D"])
        result["gpu_tflops"] = flops / compute_timing.median_gpu_span_ms / 1e9
        result["tflops"] = flops / compute_e2e_ms / 1e9
        result["qps"] = int(shape["B"]) * int(shape["Q"]) / (compute_e2e_ms / 1000.0)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for reusable KNN-build runtime.compute"
    )
    parser.add_argument(
        "--shape",
        action="append",
        choices=sorted(SHAPES),
        help="Shape label to run. Repeatable.",
    )
    parser.add_argument("--arch", default=None, help="NVRTC architecture, e.g. sm_100a.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Emit available benchmark metadata without CUDA.",
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip candidate reference checks. The measured FlashLib baseline remains fail-closed.",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip CUPTI timing.")
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based validation shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of validation shards.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the full JSON payload.")
    args = parser.parse_args()

    selected = args.shape or list(SHAPES)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < count and count must be positive")
    if args.json is not None:
        args.json.unlink(missing_ok=True)
    selected = selected[args.shard_index :: args.shard_count]
    measurement_session_id = uuid.uuid4().hex
    unique_signature_count = len(
        {
            (
                int(SHAPES[name]["B"]),
                int(SHAPES[name]["Q"]),
                int(SHAPES[name]["M"]),
                int(SHAPES[name]["D"]),
                int(SHAPES[name]["K"]),
                str(SHAPES[name].get("dtype", "bfloat16")),
                bool(SHAPES[name].get("build", False)),
            )
            for name in selected
        }
    )
    payload: dict[str, Any] = {
        "api": "flashlib_cake_knn_build.init().compute",
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "baseline_name": BASELINE_NAME,
        "baseline_entrypoint": BASELINE_NAME,
        "publication_speedup_convention": (
            "compute_speedup_vs_baseline = "
            "same_session_flashlib_flash_knn_synchronized_e2e_ms / "
            "exported_runtime_compute_synchronized_e2e_ms"
        ),
        "speedup_convention": (
            "speedup_vs_baseline = same_session_flashlib_flash_knn_gpu_span_ms / "
            "exported_runtime_compute_gpu_span_ms (legacy GPU-span alias)"
        ),
        "gpu_speedup_convention": (
            "compute_gpu_speedup_vs_baseline = "
            "same_session_flashlib_flash_knn_gpu_span_ms / "
            "exported_runtime_compute_gpu_span_ms"
        ),
        "shapes": {name: _shape_metadata(SHAPES[name]) for name in selected},
        "metadata_only": bool(args.metadata_only),
        "validation_shard": {"index": args.shard_index, "count": args.shard_count},
        "measurement_session": {
            "id": measurement_session_id,
            "scope": "shared_runtime_per_shape_deterministic_path_blocks_with_pointer_alternation",
            "path_timing_mode": "separate_cupti_blocks",
            "pointer_timing_mode": "alternating_fresh_pointer_sets_within_each_path",
            "baseline_candidate_same_process": True,
            "baseline_compute_prepared_same_session": True,
            "candidate_norm_policy": "internal_fused_row_norm_with_runtime_owned_scratch",
            "baseline_norm_policy": "x2_free_flashlib_score",
            "runtime_initialized_once": True,
            "runtime_instance_reused_across_shapes": True,
            "resident_multi_shape_cache_benchmarked": False,
            "cache_policy": "synchronize_and_clear_after_each_completed_shape",
            "order_policy": "deterministic_sha256_per_shape_permutation",
            "cold_order_policy": "deterministic_balanced_per_publication_contract_portfolio",
            "init_composition": "runtime_init_only",
            "init_order_policy": "candidate_only_baseline_has_no_explicit_init",
            "baseline_has_explicit_init": False,
            "order_seed": MEASUREMENT_ORDER_SEED,
        },
        "runtime_cache_summary": {
            "selected_shape_count": len(selected),
            "unique_signature_count": unique_signature_count,
        },
        "runtime_lifecycle": {
            "schema": "loom-public-runtime-lifecycle-v1",
            "candidate_api": "flashlib_cake_knn_build.init().compute",
            "baseline_api": BASELINE_NAME,
            "candidate_timing_boundary": "raw_inputs_default_output_route_required_norms_synchronized_e2e",
            "baseline_timing_boundary": "raw_inputs_default_output_synchronized_e2e",
            "init_scope": "once_per_validation_shard_process_device_operator",
            "amortization_call_counts": [1, 10, 100, 1000],
            "cache_policy": "synchronize_and_clear_after_each_completed_shape",
            "cold_order_policy": "deterministic_balanced_per_publication_contract_portfolio",
            "init_composition": "runtime_init_only",
            "init_order_policy": "candidate_only_baseline_has_no_explicit_init",
            "baseline_has_explicit_init": False,
            "resident_multi_shape_cache_benchmarked": False,
            "candidate_output_policy": "default_output_allocated_inside_compute",
            "baseline_output_policy": "default_output_allocated_inside_flashlib_call",
            "session": None,
        },
    }
    if args.metadata_only:
        payload["results"] = []
    else:
        import torch
        from flashlib_cake_knn_build import init

        capability = torch.cuda.get_device_capability()
        detected_arch = f"sm_{capability[0]}{capability[1]}" + ("" if capability[0] == 8 else "a")
        payload["hardware"] = {
            "device": torch.cuda.get_device_name(),
            "arch": detected_arch,
        }
        if not args.no_benchmark:
            require_cupti()
            runtime, runtime_init_timing = measure_host_call(lambda: init(arch=args.arch))
        else:
            runtime = init(arch=args.arch)
            runtime_init_timing = None
        results = []
        for name in selected:
            results.append(
                _run_shape(
                    name,
                    SHAPES[name],
                    runtime=runtime,
                    runtime_init_timing=runtime_init_timing,
                    arch=args.arch,
                    correctness=not args.no_correctness,
                    benchmark=not args.no_benchmark,
                    measurement_session_id=measurement_session_id,
                )
            )
            runtime.clear()
        payload["results"] = results
        payload["cold_runtime_init"] = _host_call_diagnostics(runtime_init_timing)
        first_lookup_miss_count = sum(not bool(row["first_shape_lookup_cache_hit"]) for row in results)
        fresh_pointer_hit_count = sum(bool(row["fresh_pointer_cache_hit"]) for row in results)
        final_cache_info = runtime.cache_info()
        payload["runtime_cache_summary"].update(
            {
                "first_lookup_miss_count": first_lookup_miss_count,
                "fresh_pointer_hit_count": fresh_pointer_hit_count,
                "final_cache_info": final_cache_info,
                "workspace_lifecycle": "synchronize_and_clear_after_each_completed_shape",
            }
        )
        if args.no_benchmark:
            payload.pop("runtime_lifecycle", None)
        else:
            payload["runtime_lifecycle"]["session"] = {
                "id": measurement_session_id,
                "init_measurement_order": ["candidate"],
                "candidate_init": _host_call_diagnostics(runtime_init_timing),
                "baseline_init": None,
                "clear_count": len(results),
                "first_lookup_miss_count": first_lookup_miss_count,
                "fresh_pointer_hit_count": fresh_pointer_hit_count,
                "final_candidate_cache_info": final_cache_info,
            }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.json, text)
    if not args.quiet:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
