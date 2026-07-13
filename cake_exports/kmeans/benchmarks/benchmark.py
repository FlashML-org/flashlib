from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flash_kmeans_shapes import (  # noqa: E402
    FLASH_KMEANS_EVOLUTION_ARTIFACT,
    FLASH_KMEANS_EVOLUTION_SUMMARY,
)
from flash_kmeans_triton_h200 import euclid_assign_triton_h200  # noqa: E402
from flash_kmeans_triton_h200_raw_adapter import (  # noqa: E402
    BASELINE_COMMIT,
    PREPROCESS_IMPL,
    TritonH20007cfRawAdapter,
)
from flash_kmeans_triton_h200_raw_adapter import (  # noqa: E402
    BASELINE_NAME as PUBLIC_RAW_BASELINE_NAME,
)
from flashlib_cake_kmeans._benchmark import (  # noqa: E402
    BenchResult,
    HostCallTiming,
    bench_gpu_time,
    compare_runtime_lifecycles,
    measure_host_call,
    require_cupti,
    runtime_lifecycle_metrics,
)


def _sum_host_timings(*timings: HostCallTiming) -> HostCallTiming:
    """Compose sequential one-time setup costs for standalone-lane modeling."""

    if not timings:
        raise ValueError("at least one host timing is required")
    return HostCallTiming(
        host_enqueue_ms=sum(float(timing.host_enqueue_ms) for timing in timings),
        synchronized_e2e_ms=sum(float(timing.synchronized_e2e_ms) for timing in timings),
    )


SHAPE_RECORDS = json.loads((Path(__file__).with_name("shape_records.json")).read_text(encoding="utf-8"))
ROUTE_MANIFEST = json.loads((Path(__file__).with_name("expected_routes.json")).read_text(encoding="utf-8"))
ALL_ROUTE_MANIFEST = json.loads((Path(__file__).with_name("all_expected_routes.json")).read_text(encoding="utf-8"))
EXPECTED_ROUTES = {row["shape"]: row["selected_route"] for row in ALL_ROUTE_MANIFEST}


def _shape_from_record(record: dict[str, Any]) -> dict[str, Any]:
    params = dict(record["params"])
    recorded = dict(record.get("recorded", {}))
    row = {"label": record["label"], **params}
    row["runtime_coverage"] = bool(record.get("runtime_coverage", False))
    row["route"] = recorded.get("evolution_route", params.get("route"))
    for key in (
        "evolution_kernel_ms",
        "evolution_flashlib_ms",
        "evolution_tflops",
        "evolution_flashlib_equiv_tflops",
        "evolution_speedup",
    ):
        row[key] = recorded.get(key, params.get(key))
    return row


FLASH_KMEANS_SHAPES = [_shape_from_record(record) for record in SHAPE_RECORDS]
_SHAPES_BY_LABEL = {row["label"]: row for row in FLASH_KMEANS_SHAPES}
FLASH_KMEANS_REGISTRY_SHAPES = [_SHAPES_BY_LABEL[route["shape"]] for route in ROUTE_MANIFEST]
SEMANTIC_ENTRYPOINT = "loom.examples.weave.flash_kmeans_assign_dispatcher:launch_for_eval"
PRECOMPUTED_BASELINE_NAME = "triton_h200_07cf_precomputed"
BASELINE_REGISTRY_KEY = "triton_h200_07cf_dual_lane_v1"
# Updated together with benchmark_data.json by the registry sync gate.
BASELINE_REGISTRY_SHA256 = "c93ce3c190e5ac96b22de94cc187e7dd4bcec4cd18461aa2cddecc8cad7842eb"
REGISTRY_CANDIDATE_ENTRYPOINT = SEMANTIC_ENTRYPOINT
MEASURED_CANDIDATE_ENTRYPOINT = "flashlib_cake_kmeans.interface:FlashKMeansAssignRuntime.compute"
BASELINE_ENTRYPOINT = "benchmarks.flash_kmeans_triton_h200_raw_adapter:TritonH20007cfRawAdapter.compute"
PARITY_CANDIDATE_ENTRYPOINT = "flashlib_cake_kmeans.interface:flash_kmeans_assign_prepared"
PARITY_BASELINE_ENTRYPOINT = "benchmarks.flash_kmeans_triton_h200:euclid_assign_triton_h200"
REGISTRY_CANDIDATE_TIMING_BOUNDARY = (
    "raw_inputs_default_output_internal_required_norms_cached_dispatch_and_synchronized_completion"
)
CANDIDATE_TIMING_BOUNDARY = (
    "raw_inputs_default_output_internal_required_norms_cached_dispatch_and_synchronized_completion"
)
BASELINE_TIMING_BOUNDARY = "raw_inputs_default_output_shared_fused_pair_norm_plus_pinned_07cf_assign_synchronized_e2e"
PARITY_CANDIDATE_TIMING_BOUNDARY = "precomputed_norms_preallocated_output_prepared_assignment_gpu_span"
PARITY_BASELINE_TIMING_BOUNDARY = "precomputed_norms_preallocated_output_pinned_07cf_assignment_gpu_span"
BASELINE_REGISTRY_PROFILE = {
    "registry_candidate_entrypoint": REGISTRY_CANDIDATE_ENTRYPOINT,
    "shared_preprocess": {
        "implementation_entrypoint": PREPROCESS_IMPL,
        "source_sha256": "b32a719eecf883459c98479710e4adbdfeb1fc8f9883276ec4bf74ab44eeef68",
        "result_source_sha256_field": "preprocess_source_sha256",
    },
    "official": {
        "role": "publication",
        "candidate_entrypoint": MEASURED_CANDIDATE_ENTRYPOINT,
        "baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": BASELINE_ENTRYPOINT,
        # This digest-bound identity belongs to the frozen benchmark registry.
        # The live raw payload carries the request-owned official boundary.
        "candidate_timing_boundary": REGISTRY_CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": BASELINE_TIMING_BOUNDARY,
        "candidate_timing_backend_field": "candidate_public_raw_timing_backend",
        "baseline_timing_backend_field": "baseline_07cf_adapter_timing_backend",
        "speedup_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "speedup_numerator_metric": "baseline_07cf_adapter_synchronized_e2e_ms",
        "speedup_denominator_metric": "candidate_public_raw_synchronized_e2e_ms",
        "timing_backend": "cupti",
    },
    "parity": {
        "role": "diagnostic_only",
        "candidate_entrypoint": PARITY_CANDIDATE_ENTRYPOINT,
        "baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": PARITY_BASELINE_ENTRYPOINT,
        "candidate_timing_boundary": PARITY_CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": PARITY_BASELINE_TIMING_BOUNDARY,
        "candidate_timing_backend_field": "candidate_precomputed_timing_backend",
        "baseline_timing_backend_field": "baseline_07cf_precomputed_timing_backend",
        "speedup_metric": "precomputed_gpu_speedup_vs_07cf",
        "speedup_numerator_metric": "baseline_07cf_precomputed_gpu_span_ms",
        "speedup_denominator_metric": "candidate_precomputed_gpu_span_ms",
        "timing_backend": "cupti",
    },
}
WARMUP_MS = 20.0
BENCH_MS = 100.0
DEFAULT_MEASUREMENT_ORDER_SEED = "flashlib-kmeans-export-paired-replay-v1"
MEASUREMENT_ROLES = (
    "candidate_public_raw",
    "baseline_public_raw",
    "candidate_precomputed",
    "baseline_precomputed",
)


def _preprocess_source_digest() -> str:
    """Hash the exact Python/CUDA pair-row-norm implementation in this export."""

    source_dir = Path(__file__).resolve().parent
    candidates = (
        (
            ROOT / "src/flashlib_cake_kmeans/_row_norm.py",
            ROOT / "src/flashlib_cake_kmeans/_row_norm.cu",
        ),
        (source_dir / "row_norm.py", source_dir / "row_norm.cu"),
    )
    for paths in candidates:
        if all(path.is_file() for path in paths):
            digest = hashlib.sha256()
            for canonical_name, path in zip(("_row_norm.py", "_row_norm.cu"), paths, strict=True):
                digest.update(canonical_name.encode("utf-8"))
                digest.update(b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            return digest.hexdigest()
    searched = ", ".join(str(path) for pair in candidates for path in pair)
    raise RuntimeError(f"cannot locate exported pair-row-norm sources; searched {searched}")


def _timing_diagnostics(timing: Any, *, primary_metric: str) -> dict[str, Any]:
    return {
        "lane_primary_metric": primary_metric,
        # KMeans uses an adaptive 100 ms window that can contain thousands of
        # iterations. The generated BenchResult retains every exact sample;
        # the publication JSON records medians and sample count to avoid
        # multiplying artifact size across the 228-shape portfolio.
        "sample_count": len(timing.times_ms),
        "gpu_span_ms": {"median": timing.median_gpu_span_ms},
        "kernel_sum_ms": {"median": timing.median_kernel_sum_ms},
        "active_union_ms": {"median": timing.median_active_union_ms},
        "inter_kernel_gap_ms": {"median": timing.median_inter_kernel_gap_ms},
        "activity_count": {"median": timing.median_activity_count},
        "correlated_launch_activity_count": {"median": timing.median_launch_activity_count},
        "correlated_kernel_activity_count": {"median": timing.median_kernel_activity_count},
        "submission_ms": {"median": timing.median_submission_ms},
        "host_enqueue_ms": {"median": timing.median_host_enqueue_ms},
        "synchronized_e2e_ms": {"median": timing.median_synchronized_e2e_ms},
        "cold_first_call": {
            "host_enqueue_ms": timing.cold_first_call_host_enqueue_ms,
            "synchronized_e2e_ms": timing.cold_first_call_synchronized_e2e_ms,
        },
    }


def _pair_order(
    label: str,
    pair_name: str,
    pair_index: int,
    order_seed: str,
) -> tuple[str, str]:
    pairs = {
        "public": ("candidate_public_raw", "baseline_public_raw"),
        "precomputed": ("candidate_precomputed", "baseline_precomputed"),
    }
    pair = pairs[pair_name]
    digest = hashlib.sha256(
        f"{order_seed}:{label}:{pair_name}:{pair_index}".encode()
    ).digest()
    return pair if digest[0] & 1 else tuple(reversed(pair))


def _interleaved_pair_schedule(
    label: str,
    *,
    public_pair_count: int,
    precomputed_pair_count: int,
    order_seed: str,
) -> tuple[str, ...]:
    """Build balanced adjacent A/B pairs for one shared CUPTI session."""

    if not isinstance(order_seed, str) or not order_seed:
        raise ValueError("measurement order seed must be a non-empty string")
    if public_pair_count <= 0 or precomputed_pair_count <= 0:
        raise ValueError("interleaved pair counts must be positive")
    pair_slots = [
        ((index + 0.5) / public_pair_count, "public", index)
        for index in range(public_pair_count)
    ]
    pair_slots.extend(
        ((index + 0.5) / precomputed_pair_count, "precomputed", index)
        for index in range(precomputed_pair_count)
    )
    pair_slots.sort(
        key=lambda item: (
            item[0],
            hashlib.sha256(
                f"{order_seed}:{label}:pair-slot:{item[1]}:{item[2]}".encode()
            ).digest(),
        )
    )
    schedule: list[str] = []
    for _position, pair_name, pair_index in pair_slots:
        schedule.extend(_pair_order(label, pair_name, pair_index, order_seed))
    counts = {role: schedule.count(role) for role in MEASUREMENT_ROLES}
    expected = {
        "candidate_public_raw": public_pair_count,
        "baseline_public_raw": public_pair_count,
        "candidate_precomputed": precomputed_pair_count,
        "baseline_precomputed": precomputed_pair_count,
    }
    if counts != expected:
        raise RuntimeError(f"interleaved role schedule is incomplete: {counts!r} != {expected!r}")
    return tuple(schedule)


def _slice_bench_result(
    result: BenchResult,
    indices: list[int],
    *,
    cold_first_call: HostCallTiming | None,
) -> BenchResult:
    if not indices:
        raise ValueError("each paired role requires at least one reportable sample")

    def select(values):
        if values is None:
            return None
        if len(values) != len(result.times_ms):
            raise RuntimeError("shared CUPTI result fields have inconsistent sample counts")
        return [values[index] for index in indices]

    return BenchResult(
        times_ms=select(result.times_ms),
        backend=result.backend,
        kernel_sum_times_ms=select(result.kernel_sum_times_ms),
        inter_kernel_gap_times_ms=select(result.inter_kernel_gap_times_ms),
        active_union_times_ms=select(result.active_union_times_ms),
        activity_counts=select(result.activity_counts),
        launch_activity_counts=select(result.launch_activity_counts),
        kernel_activity_counts=select(result.kernel_activity_counts),
        submission_times_ms=select(result.submission_times_ms),
        synchronized_e2e_times_ms=select(result.synchronized_e2e_times_ms),
        cold_first_call_host_enqueue_ms=(
            cold_first_call.host_enqueue_ms if cold_first_call is not None else None
        ),
        cold_first_call_synchronized_e2e_ms=(
            cold_first_call.synchronized_e2e_ms if cold_first_call is not None else None
        ),
    )


def _bench_interleaved_roles(
    *,
    label: str,
    order_seed: str,
    role_factories: dict[str, Any],
    cold_first_calls: dict[str, HostCallTiming | None],
) -> tuple[dict[str, BenchResult], dict[str, Any]]:
    """Measure all reportable roles in one interleaved CUPTI activity session."""

    if tuple(role_factories) != MEASUREMENT_ROLES:
        raise ValueError("paired benchmark role factories are incomplete or out of order")
    probes: dict[str, BenchResult] = {}
    for role in MEASUREMENT_ROLES:
        probes[role] = bench_gpu_time(
            role_factories[role](),
            warmup_iters=5,
            bench_iters=20,
            cold_l2=True,
        )
    estimates = {role: float(probes[role].median_ms) for role in MEASUREMENT_ROLES}
    if any(not math.isfinite(value) or value <= 0.0 for value in estimates.values()):
        raise RuntimeError(f"paired benchmark probes produced invalid estimates: {estimates!r}")

    public_bench_pairs = max(
        8,
        math.ceil(BENCH_MS / estimates["candidate_public_raw"]),
        math.ceil(BENCH_MS / estimates["baseline_public_raw"]),
    )
    precomputed_bench_pairs = max(
        8,
        math.ceil(BENCH_MS / estimates["candidate_precomputed"]),
        math.ceil(BENCH_MS / estimates["baseline_precomputed"]),
    )
    public_warmup_pairs = max(
        1,
        math.ceil(WARMUP_MS / estimates["candidate_public_raw"]),
        math.ceil(WARMUP_MS / estimates["baseline_public_raw"]),
    )
    precomputed_warmup_pairs = max(
        1,
        math.ceil(WARMUP_MS / estimates["candidate_precomputed"]),
        math.ceil(WARMUP_MS / estimates["baseline_precomputed"]),
    )
    warmup_schedule = _interleaved_pair_schedule(
        label,
        public_pair_count=public_warmup_pairs,
        precomputed_pair_count=precomputed_warmup_pairs,
        order_seed=f"{order_seed}:warmup",
    )
    measurement_schedule = _interleaved_pair_schedule(
        label,
        public_pair_count=public_bench_pairs,
        precomputed_pair_count=precomputed_bench_pairs,
        order_seed=order_seed,
    )
    combined_schedule = (*warmup_schedule, *measurement_schedule)
    role_callables = {role: role_factories[role]() for role in MEASUREMENT_ROLES}
    next_index = 0

    def invoke_scheduled_role():
        nonlocal next_index
        role = combined_schedule[next_index]
        next_index += 1
        return role_callables[role]()

    shared = bench_gpu_time(
        invoke_scheduled_role,
        warmup_iters=len(warmup_schedule),
        bench_iters=len(measurement_schedule),
        cold_l2=True,
    )
    if next_index != len(combined_schedule):
        raise RuntimeError("paired benchmark did not consume its complete role schedule")
    timings: dict[str, BenchResult] = {}
    sample_counts: dict[str, int] = {}
    for role in MEASUREMENT_ROLES:
        indices = [index for index, scheduled_role in enumerate(measurement_schedule) if scheduled_role == role]
        timings[role] = _slice_bench_result(
            shared,
            indices,
            cold_first_call=cold_first_calls.get(role),
        )
        sample_counts[role] = len(indices)

    schedule_text = "\n".join(measurement_schedule).encode()
    provenance = {
        "schema": "flash-kmeans-interleaved-cupti-session-v1",
        "same_cupti_session": True,
        "interleaved": True,
        "adjacent_candidate_baseline_pairs": True,
        "reportable_cupti_session_count": 1,
        "adaptive_probe_cupti_session_count": len(MEASUREMENT_ROLES),
        "adaptive_probes_reportable": False,
        "order_seed": order_seed,
        "measurement_schedule_sha256": hashlib.sha256(schedule_text).hexdigest(),
        "measurement_schedule_iteration_count": len(measurement_schedule),
        "warmup_schedule_iteration_count": len(warmup_schedule),
        "role_sample_counts": sample_counts,
        "public_pair_count": public_bench_pairs,
        "precomputed_pair_count": precomputed_bench_pairs,
        "first_reportable_roles": list(measurement_schedule[:16]),
    }
    return timings, provenance


def _label_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


_REGISTRY_COLD_LABELS = frozenset(row["label"] for row in FLASH_KMEANS_REGISTRY_SHAPES)
_CANDIDATE_FIRST_COLD_LABELS_BY_SEED: dict[str, frozenset[str]] = {}


def _candidate_first_cold_labels(order_seed: str) -> frozenset[str]:
    cached = _CANDIDATE_FIRST_COLD_LABELS_BY_SEED.get(order_seed)
    if cached is None:
        ordered = sorted(
            (row["label"] for row in FLASH_KMEANS_REGISTRY_SHAPES),
            key=lambda label: hashlib.sha256(f"{order_seed}:cold:{label}".encode()).digest(),
        )
        cached = frozenset(ordered[::2])
        _CANDIDATE_FIRST_COLD_LABELS_BY_SEED[order_seed] = cached
    return cached


def _cold_measurement_order(label: str, order_seed: str) -> tuple[str, str]:
    """Counterbalance process-shared cold effects across contract shapes.

    Rank-alternate over the seeded hash order of the full contract portfolio
    so the split is exactly balanced (|candidate_first - baseline_first| <= 1)
    for every shard decomposition, matching the declared
    ``deterministic_balanced_per_publication_contract_portfolio`` policy.
    """

    if label in _REGISTRY_COLD_LABELS:
        first_candidate = label in _candidate_first_cold_labels(order_seed)
    else:
        # Runtime-coverage diagnostic rows sit outside the publication
        # contract balance gate; keep the per-label seeded coin flip.
        digest = hashlib.sha256(f"{order_seed}:cold:{label}".encode()).digest()
        first_candidate = bool(digest[0] & 1)
    return ("candidate", "baseline") if first_candidate else ("baseline", "candidate")


def _alternating_call(first, second):
    """Return a callable that alternates two equivalent fresh-pointer cases."""

    calls = (first, second)
    next_index = 0

    def invoke():
        nonlocal next_index
        current = next_index
        next_index ^= 1
        return calls[current]()

    return invoke


def _required_median(value: float | None, *, metric: str) -> float:
    if value is None:
        raise RuntimeError(f"CUPTI benchmark did not report required {metric}")
    return float(value)


def _dual_lane_speedups(
    *,
    candidate_public_raw_timing: Any,
    baseline_public_raw_timing: Any,
    candidate_precomputed_timing: Any,
    baseline_precomputed_timing: Any,
) -> dict[str, float]:
    """Return the one official E2E ratio and the precomputed GPU parity ratio."""

    candidate_public_raw_e2e = _required_median(
        candidate_public_raw_timing.median_synchronized_e2e_ms,
        metric="candidate public-raw synchronized_e2e_ms",
    )
    baseline_public_raw_e2e = _required_median(
        baseline_public_raw_timing.median_synchronized_e2e_ms,
        metric="07cf raw-adapter synchronized_e2e_ms",
    )
    return {
        "public_raw_e2e_speedup_vs_07cf_adapter": baseline_public_raw_e2e / candidate_public_raw_e2e,
        "precomputed_gpu_speedup_vs_07cf": (
            float(baseline_precomputed_timing.median_gpu_span_ms)
            / float(candidate_precomputed_timing.median_gpu_span_ms)
        ),
    }


def _public_prepared_ratios(public_timing: Any, prepared_timing: Any) -> dict[str, float]:
    """Return public/prepared ratios from one per-shape measurement session."""

    public_submission = _required_median(
        public_timing.median_submission_ms,
        metric="public submission_ms",
    )
    prepared_submission = _required_median(
        prepared_timing.median_submission_ms,
        metric="prepared submission_ms",
    )
    public_e2e = _required_median(
        public_timing.median_synchronized_e2e_ms,
        metric="public synchronized_e2e_ms",
    )
    prepared_e2e = _required_median(
        prepared_timing.median_synchronized_e2e_ms,
        metric="prepared synchronized_e2e_ms",
    )
    denominators = {
        "gpu_span_ms": float(prepared_timing.median_gpu_span_ms),
        "submission_ms": prepared_submission,
        "synchronized_e2e_ms": prepared_e2e,
    }
    if any(value <= 0.0 for value in denominators.values()):
        raise RuntimeError(f"prepared timing ratios require positive denominators, got {denominators!r}")
    return {
        "gpu_span": float(public_timing.median_gpu_span_ms) / denominators["gpu_span_ms"],
        "submission": public_submission / denominators["submission_ms"],
        "synchronized_e2e": public_e2e / denominators["synchronized_e2e_ms"],
    }


def _cold_call_fields(timing: Any | None) -> dict[str, float | None]:
    return {
        "submission_ms": timing.host_enqueue_ms if timing is not None else None,
        "host_enqueue_ms": timing.host_enqueue_ms if timing is not None else None,
        "synchronized_e2e_ms": timing.synchronized_e2e_ms if timing is not None else None,
    }


def _shape_key(row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(row["B"]),
        int(row["N"]),
        int(row["D"]),
        int(row["K"]),
        str(row.get("dtype", "bfloat16")),
    )


def _selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = list(FLASH_KMEANS_SHAPES if args.include_runtime_coverage else FLASH_KMEANS_REGISTRY_SHAPES)
    if args.source:
        wanted_sources = set(args.source)
        rows = [row for row in rows if row["source"] in wanted_sources]
    if args.shape:
        by_label = {row["label"]: row for row in rows}
        missing = sorted(set(args.shape) - set(by_label))
        if missing:
            available = ", ".join(sorted(by_label))
            raise SystemExit(f"unknown shape label(s) {missing}. Available: {available}")
        rows = [by_label[label] for label in args.shape]
    if args.unique:
        seen: set[tuple[int, int, int, int, str]] = set()
        unique_rows: list[dict[str, Any]] = []
        for row in rows:
            key = _shape_key(row)
            if key in seen:
                continue
            seen.add(key)
            unique_rows.append(row)
        rows = unique_rows
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.shard_count]
    return rows


def _write_json_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text + "\n", encoding="utf-8")
    temporary.replace(path)


def _make_inputs(row: dict[str, Any], *, variant: int = 0):
    import torch

    generator = torch.Generator(device="cuda")
    seed = int(row.get("seed", _label_seed(str(row["label"]))))
    if variant != 0:
        seed = _label_seed(f"{row['label']}:fresh-pointer:{variant}:{seed}")
    generator.manual_seed(seed)
    x = torch.randn(
        (int(row["B"]), int(row["N"]), int(row["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    centroids = torch.randn(
        (int(row["B"]), int(row["K"]), int(row["D"])),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    ).contiguous()
    return x, centroids


def _measure_shared_preprocess_cold_compile(row: dict[str, Any], *, arch: str):
    """Compile the shared row-norm support once, outside either comparison lane."""

    import torch
    from flashlib_cake_kmeans._row_norm import prepare_bf16_pair_row_norm

    x, centroids = _make_inputs(row, variant=0xC01D)
    x_sq = torch.empty((int(row["B"]), int(row["N"])), dtype=torch.float32, device=x.device)
    c_sq = torch.empty((int(row["B"]), int(row["K"])), dtype=torch.float32, device=x.device)
    # Fixture generation is asynchronous. Complete it before the cold bracket
    # so neither lane inherits unrelated random-generation/allocation work.
    torch.cuda.synchronize()

    def prepare_shared_support():
        prepared = prepare_bf16_pair_row_norm(
            x,
            centroids,
            x_sq,
            c_sq,
            compute_x=True,
            compute_c=True,
            arch=arch,
        )
        prepared.release_bound_callers(x_sq)

    _, cold_compile = measure_host_call(prepare_shared_support)
    return cold_compile


def _reference_assign(x, centroids, *, chunk_rows: int):
    import torch

    bsz = int(x.shape[0])
    n_points = int(x.shape[1])
    x_f32 = x.float()
    c_f32 = centroids.float()
    c_sq = (c_f32 * c_f32).sum(-1)
    ref = torch.empty((bsz, n_points), dtype=torch.int32, device=x.device)
    with torch.no_grad():
        for b in range(bsz):
            c_t = c_f32[b].transpose(0, 1).contiguous()
            c_bias = 0.5 * c_sq[b]
            for start in range(0, n_points, chunk_rows):
                q = x_f32[b, start : start + chunk_rows]
                scores = torch.matmul(q, c_t) - c_bias.unsqueeze(0)
                ref[b, start : start + q.shape[0]] = scores.argmax(dim=-1).to(torch.int32)
    return ref


def _assignment_correctness(cluster_ids, ref, x, centroids) -> dict[str, Any]:
    """Validate assignment indices while accepting exact-distance ties."""

    import torch

    matches = cluster_ids == ref
    mismatch_count = int((~matches).sum().item())
    diagnostics: dict[str, Any] = {
        "match_rate": float(matches.float().mean().item()),
        "mismatch_count": mismatch_count,
    }
    if mismatch_count == 0:
        diagnostics.update({"correct": True, "tie_inclusive": False})
        return diagnostics

    dim = int(x.shape[-1])
    pred_idx = cluster_ids.to(torch.int64).unsqueeze(-1).expand(-1, -1, dim)
    ref_idx = ref.to(torch.int64).unsqueeze(-1).expand(-1, -1, dim)
    pred_centroids = torch.gather(centroids, 1, pred_idx).float()
    ref_centroids = torch.gather(centroids, 1, ref_idx).float()
    points = x.float()
    pred_dist = ((points - pred_centroids) ** 2).sum(-1)
    ref_dist = ((points - ref_centroids) ** 2).sum(-1)
    distance_delta = (pred_dist - ref_dist).abs()
    tie_ok = matches | (distance_delta <= 1.0e-3)
    diagnostics.update(
        {
            "correct": bool(tie_ok.all().item()),
            "tie_inclusive": True,
            "tie_inclusive_match_rate": float(tie_ok.float().mean().item()),
            "max_selected_distance_delta": float(distance_delta.max().item()),
        }
    )
    return diagnostics


def _run_shape(
    row: dict[str, Any],
    *,
    runtime: Any,
    baseline_adapter: Any,
    candidate_init_timing: Any = None,
    baseline_init_timing: Any = None,
    arch: str | None,
    correctness: bool,
    benchmark: bool,
    reference_chunk_rows: int,
    measurement_session_id: str,
    measurement_order_seed: str,
) -> dict[str, Any]:
    import torch
    from flashlib_cake_kmeans import (
        flash_kmeans_assign_prepared,
        prepare_flash_kmeans_assign,
    )

    x, centroids = _make_inputs(row, variant=0)
    fresh_x, fresh_centroids = _make_inputs(row, variant=1)
    x_sq = (x.float() ** 2).sum(-1).contiguous()
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    fresh_x_sq = (fresh_x.float() ** 2).sum(-1).contiguous()
    fresh_c_sq = (fresh_centroids.float() ** 2).sum(-1).contiguous()
    out_shape = (int(row["B"]), int(row["N"]))
    candidate_precomputed_out = torch.empty(out_shape, dtype=torch.int32, device=x.device)
    fresh_candidate_precomputed_out = torch.empty_like(candidate_precomputed_out)
    baseline_precomputed_out = torch.empty_like(candidate_precomputed_out)
    fresh_baseline_precomputed_out = torch.empty_like(candidate_precomputed_out)

    pointer_pairs = (
        (x, fresh_x),
        (centroids, fresh_centroids),
        (x_sq, fresh_x_sq),
        (c_sq, fresh_c_sq),
    )
    if any(first.data_ptr() == second.data_ptr() for first, second in pointer_pairs):
        raise RuntimeError("fresh-pointer benchmark inputs unexpectedly alias the first input set")
    # Keep cold diagnostics honest: random input generation, explicit parity
    # norms, and preallocated parity outputs all precede every measured call.
    torch.cuda.synchronize()

    def baseline_public_raw_first_call():
        return baseline_adapter.compute(x, centroids, return_info=True)

    def baseline_public_raw_fresh_pointer_call():
        return baseline_adapter.compute(fresh_x, fresh_centroids, return_info=True)

    def baseline_precomputed_first_call():
        return euclid_assign_triton_h200(
            x,
            centroids,
            x_sq,
            c_sq,
            out=baseline_precomputed_out,
        )

    def baseline_precomputed_fresh_pointer_call():
        return euclid_assign_triton_h200(
            fresh_x,
            fresh_centroids,
            fresh_x_sq,
            fresh_c_sq,
            out=fresh_baseline_precomputed_out,
        )

    def candidate_public_raw_shape_first_call():
        return runtime.compute(
            x,
            centroids,
            return_info=True,
        )

    def candidate_public_raw_fresh_pointer_call():
        return runtime.compute(
            fresh_x,
            fresh_centroids,
            return_info=True,
        )

    baseline_public_raw_shape_cold_call = None
    baseline_public_raw_fresh_cold_call = None
    baseline_precomputed_first_cold_call = None
    baseline_precomputed_fresh_cold_call = None
    candidate_public_raw_shape_cold_call = None
    candidate_public_raw_fresh_cold_call = None
    candidate_precomputed_prepare_cold_call = None
    fresh_candidate_precomputed_prepare_cold_call = None
    candidate_precomputed_first_cold_call = None
    fresh_candidate_precomputed_first_cold_call = None
    cold_measurement_order = _cold_measurement_order(
        str(row["label"]),
        measurement_order_seed,
    )
    if benchmark:
        for lane in cold_measurement_order:
            if lane == "baseline":
                (
                    (baseline_public_raw_out, baseline_public_raw_shape_info),
                    baseline_public_raw_shape_cold_call,
                ) = measure_host_call(baseline_public_raw_first_call)
            else:
                (
                    (candidate_public_raw_out, candidate_public_raw_shape_info),
                    candidate_public_raw_shape_cold_call,
                ) = measure_host_call(candidate_public_raw_shape_first_call)
        for lane in cold_measurement_order:
            if lane == "baseline":
                (
                    (fresh_baseline_public_raw_out, baseline_public_raw_fresh_info),
                    baseline_public_raw_fresh_cold_call,
                ) = measure_host_call(baseline_public_raw_fresh_pointer_call)
            else:
                (
                    (fresh_candidate_public_raw_out, candidate_public_raw_info),
                    candidate_public_raw_fresh_cold_call,
                ) = measure_host_call(candidate_public_raw_fresh_pointer_call)
        candidate_prepared, candidate_precomputed_prepare_cold_call = measure_host_call(
            lambda: prepare_flash_kmeans_assign(
                x,
                centroids,
                out=candidate_precomputed_out,
                x_sq=x_sq,
                c_sq=c_sq,
                arch=arch,
            )
        )
        fresh_candidate_prepared, fresh_candidate_precomputed_prepare_cold_call = measure_host_call(
            lambda: prepare_flash_kmeans_assign(
                fresh_x,
                fresh_centroids,
                out=fresh_candidate_precomputed_out,
                x_sq=fresh_x_sq,
                c_sq=fresh_c_sq,
                arch=arch,
            )
        )
        (
            (
                candidate_precomputed_result,
                candidate_precomputed_info,
            ),
            candidate_precomputed_first_cold_call,
        ) = measure_host_call(lambda: flash_kmeans_assign_prepared(candidate_prepared, return_info=True))
        (
            (
                fresh_candidate_precomputed_result,
                fresh_candidate_precomputed_info,
            ),
            fresh_candidate_precomputed_first_cold_call,
        ) = measure_host_call(lambda: flash_kmeans_assign_prepared(fresh_candidate_prepared, return_info=True))
        (
            (
                baseline_precomputed_result,
                baseline_precomputed_config,
            ),
            baseline_precomputed_first_cold_call,
        ) = measure_host_call(baseline_precomputed_first_call)
        (
            (
                fresh_baseline_precomputed_result,
                fresh_baseline_precomputed_config,
            ),
            baseline_precomputed_fresh_cold_call,
        ) = measure_host_call(baseline_precomputed_fresh_pointer_call)
    else:
        (
            baseline_public_raw_out,
            baseline_public_raw_shape_info,
        ) = baseline_public_raw_first_call()
        (
            fresh_baseline_public_raw_out,
            baseline_public_raw_fresh_info,
        ) = baseline_public_raw_fresh_pointer_call()
        candidate_public_raw_out, candidate_public_raw_shape_info = candidate_public_raw_shape_first_call()
        fresh_candidate_public_raw_out, candidate_public_raw_info = candidate_public_raw_fresh_pointer_call()
        candidate_prepared = prepare_flash_kmeans_assign(
            x,
            centroids,
            out=candidate_precomputed_out,
            x_sq=x_sq,
            c_sq=c_sq,
            arch=arch,
        )
        fresh_candidate_prepared = prepare_flash_kmeans_assign(
            fresh_x,
            fresh_centroids,
            out=fresh_candidate_precomputed_out,
            x_sq=fresh_x_sq,
            c_sq=fresh_c_sq,
            arch=arch,
        )
        candidate_precomputed_result, candidate_precomputed_info = flash_kmeans_assign_prepared(
            candidate_prepared,
            return_info=True,
        )
        fresh_candidate_precomputed_result, fresh_candidate_precomputed_info = flash_kmeans_assign_prepared(
            fresh_candidate_prepared,
            return_info=True,
        )
        baseline_precomputed_result, baseline_precomputed_config = baseline_precomputed_first_call()
        fresh_baseline_precomputed_result, fresh_baseline_precomputed_config = baseline_precomputed_fresh_pointer_call()
        torch.cuda.synchronize()

    baseline_configs = (
        baseline_public_raw_shape_info["triton_h200_07cf_config"],
        baseline_public_raw_fresh_info["triton_h200_07cf_config"],
        baseline_precomputed_config,
        fresh_baseline_precomputed_config,
    )
    if any(config != baseline_configs[0] for config in baseline_configs[1:]):
        raise RuntimeError("07cf Triton selected different configs across equivalent dual-lane calls")
    if candidate_public_raw_info["selected_route"] != candidate_public_raw_shape_info["selected_route"]:
        raise RuntimeError(
            "runtime shape-first and fresh-pointer calls selected different routes: "
            f"{candidate_public_raw_shape_info['selected_route']!r} != "
            f"{candidate_public_raw_info['selected_route']!r}"
        )
    if not candidate_public_raw_info["runtime_cache_hit"]:
        raise RuntimeError("fresh-pointer runtime.compute call did not reuse the shape/stream launch plan")
    if not baseline_public_raw_fresh_info["runtime_cache_hit"]:
        raise RuntimeError("fresh-pointer 07cf raw-adapter call did not reuse the shape/stream norm plan")
    if candidate_public_raw_info.get("hot_launch_path") != "cuda_graph":
        raise RuntimeError(
            "Flash-KMeans signature did not capture into a CUDA graph: "
            f"hot_launch_path={candidate_public_raw_info.get('hot_launch_path')!r}, "
            f"graph_capture_error={candidate_public_raw_info.get('graph_capture_error')!r}; "
            "every contract signature freezes to a LaunchPlan today, so a "
            "prepared-path fallback means the graph runtime silently degraded"
        )
    candidate_shape_norm_fields = tuple(candidate_public_raw_shape_info.get("norm_compute_fields", ()))
    candidate_fresh_norm_fields = tuple(candidate_public_raw_info.get("norm_compute_fields", ()))
    if candidate_shape_norm_fields != candidate_fresh_norm_fields:
        raise RuntimeError(
            "runtime shape-first and fresh-pointer calls computed different norm fields: "
            f"{candidate_shape_norm_fields!r} != {candidate_fresh_norm_fields!r}"
        )
    for prepared_info in (candidate_precomputed_info, fresh_candidate_precomputed_info):
        if prepared_info["selected_route"] != candidate_public_raw_info["selected_route"]:
            raise RuntimeError(
                "public-raw and precomputed Flash-KMeans calls selected different routes: "
                f"{candidate_public_raw_info['selected_route']!r} != {prepared_info['selected_route']!r}"
            )
    if candidate_precomputed_result is not candidate_precomputed_out:
        raise RuntimeError("prepared candidate did not preserve its first caller-owned output")
    if fresh_candidate_precomputed_result is not fresh_candidate_precomputed_out:
        raise RuntimeError("prepared candidate did not preserve its fresh-pointer caller-owned output")

    prepared_direct_launcher = candidate_prepared.launch_plan.direct_launcher
    fresh_prepared_direct_launcher = fresh_candidate_prepared.launch_plan.direct_launcher
    prepared_path_proof = {
        "resolved_direct_launcher": bool(callable(prepared_direct_launcher)),
        "stable_direct_launcher_tokens": [
            f"{type(prepared_direct_launcher).__module__}.{type(prepared_direct_launcher).__qualname__}"
            f"@{id(prepared_direct_launcher):x}",
            f"{type(fresh_prepared_direct_launcher).__module__}."
            f"{type(fresh_prepared_direct_launcher).__qualname__}@{id(fresh_prepared_direct_launcher):x}",
        ],
        "caller_owned_outputs": True,
        "scratch_reused": True,
        "stream_bound": bool(
            candidate_precomputed_info["stream_handle"] == candidate_prepared.launch_plan.stream_handle
            and fresh_candidate_precomputed_info["stream_handle"]
            == fresh_candidate_prepared.launch_plan.stream_handle
        ),
        "root_dispatch_traversal_count": 0,
        "parent_dispatch_traversal_count": 0,
        "internal_device_wide_synchronization_count": 0,
        "proof_scope": "prepared_direct_launcher_identity_and_source_invariants",
    }
    if not all(
        prepared_path_proof[key]
        for key in ("resolved_direct_launcher", "caller_owned_outputs", "scratch_reused", "stream_bound")
    ):
        raise RuntimeError(f"prepared candidate hot-path proof failed: {prepared_path_proof!r}")

    candidate_shape_first_was_cache_hit = bool(candidate_public_raw_shape_info["runtime_cache_hit"])
    baseline_shape_first_was_cache_hit = bool(baseline_public_raw_shape_info["runtime_cache_hit"])

    result: dict[str, Any] = {
        "shape": row["label"],
        "label": row["label"],
        "source": row["source"],
        "seed": row.get("seed"),
        "runtime_coverage": bool(row.get("runtime_coverage", False)),
        "B": int(row["B"]),
        "N": int(row["N"]),
        "D": int(row["D"]),
        "K": int(row["K"]),
        "dtype": row.get("dtype", "bfloat16"),
        "semantic_entrypoint": candidate_public_raw_info["semantic_entrypoint"],
        "expected_route": EXPECTED_ROUTES[row["label"]],
        "selected_route": candidate_public_raw_info["selected_route"],
        "child_route": candidate_public_raw_info.get("child_route"),
        "route_matches_expected": bool(candidate_public_raw_info["selected_route"] == EXPECTED_ROUTES[row["label"]]),
        "evolution_kernel_ms": row["evolution_kernel_ms"],
        "evolution_flashlib_ms": row["evolution_flashlib_ms"],
        "evolution_tflops": row["evolution_tflops"],
        "evolution_speedup": row["evolution_speedup"],
        "public_raw_baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "precomputed_baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "triton_h200_07cf_config": baseline_configs[0],
        "measurement_session_id": measurement_session_id,
        "candidate_public_raw_assignment_launch_count": candidate_public_raw_info["assignment_launch_count"],
        "candidate_public_raw_norm_launch_count": candidate_public_raw_info["norm_launch_count"],
        "candidate_public_raw_norm_compute_fields": list(candidate_fresh_norm_fields),
        "candidate_public_raw_total_launch_count": candidate_public_raw_info["runtime_launch_count"],
        "baseline_public_raw_assignment_launch_count": baseline_public_raw_fresh_info["assignment_launch_count"],
        "baseline_public_raw_norm_launch_count": baseline_public_raw_fresh_info["norm_launch_count"],
        "baseline_public_raw_norm_compute_fields": list(baseline_public_raw_fresh_info["norm_compute_fields"]),
        "baseline_public_raw_total_launch_count": baseline_public_raw_fresh_info["runtime_launch_count"],
        "candidate_precomputed_launch_count": candidate_precomputed_info["prepared_launch_count"],
        "candidate_public_raw_norm_policy": "route_required_internal_fused_bf16_pair_row_norm",
        "baseline_public_raw_norm_policy": "shared_kernel_all_fields_required_by_frozen_07cf",
        "candidate_precomputed_norm_policy": "explicit_precomputed_outside_timing",
        "baseline_precomputed_norm_policy": "explicit_precomputed_outside_timing",
        "candidate_public_raw_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
        "baseline_public_raw_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
        "candidate_precomputed_output_policy": "preallocated_outside_timing",
        "baseline_precomputed_output_policy": "preallocated_outside_timing",
        "candidate_public_raw_shape_first_was_cache_hit": candidate_shape_first_was_cache_hit,
        "baseline_public_raw_shape_first_was_cache_hit": baseline_shape_first_was_cache_hit,
        "candidate_public_raw_fresh_pointer_cache_hit": bool(candidate_public_raw_info["runtime_cache_hit"]),
        "baseline_public_raw_fresh_pointer_cache_hit": bool(baseline_public_raw_fresh_info["runtime_cache_hit"]),
        "fresh_pointer_rebind_verified": True,
        "candidate_hot_launch_path": candidate_public_raw_info.get("hot_launch_path"),
        "candidate_graph_kernel_count": candidate_public_raw_info.get("graph_kernel_count"),
        "candidate_graph_capture_error": candidate_public_raw_info.get("graph_capture_error"),
        "candidate_public_raw_cold_first_call_activity": candidate_public_raw_shape_info[
            "cold_first_call_activity"
        ],
        "candidate_prepared_path_proof": prepared_path_proof,
        "cold_candidate_public_raw_shape_miss": (
            None if candidate_shape_first_was_cache_hit else _cold_call_fields(candidate_public_raw_shape_cold_call)
        ),
        "cold_candidate_public_raw_existing_shape_hit": (
            _cold_call_fields(candidate_public_raw_shape_cold_call) if candidate_shape_first_was_cache_hit else None
        ),
        "cold_candidate_public_raw_fresh_pointer_hit": _cold_call_fields(candidate_public_raw_fresh_cold_call),
        "cold_baseline_public_raw_shape_miss": (
            None if baseline_shape_first_was_cache_hit else _cold_call_fields(baseline_public_raw_shape_cold_call)
        ),
        "cold_baseline_public_raw_existing_shape_hit": (
            _cold_call_fields(baseline_public_raw_shape_cold_call) if baseline_shape_first_was_cache_hit else None
        ),
        "cold_baseline_public_raw_fresh_pointer_hit": _cold_call_fields(baseline_public_raw_fresh_cold_call),
        "cold_candidate_precomputed_prepare_first_pointer": _cold_call_fields(candidate_precomputed_prepare_cold_call),
        "cold_candidate_precomputed_prepare_fresh_pointer": _cold_call_fields(
            fresh_candidate_precomputed_prepare_cold_call
        ),
        "cold_candidate_precomputed_first_pointer": _cold_call_fields(candidate_precomputed_first_cold_call),
        "cold_candidate_precomputed_fresh_pointer": _cold_call_fields(fresh_candidate_precomputed_first_cold_call),
        "cold_baseline_precomputed_first_pointer": _cold_call_fields(baseline_precomputed_first_cold_call),
        "cold_baseline_precomputed_fresh_pointer": _cold_call_fields(baseline_precomputed_fresh_cold_call),
    }

    if correctness:
        ref = _reference_assign(x, centroids, chunk_rows=reference_chunk_rows)
        fresh_ref = _reference_assign(fresh_x, fresh_centroids, chunk_rows=reference_chunk_rows)
        torch.cuda.synchronize()
        baseline_public_raw_correctness = _assignment_correctness(
            baseline_public_raw_out,
            ref,
            x,
            centroids,
        )
        fresh_baseline_public_raw_correctness = _assignment_correctness(
            fresh_baseline_public_raw_out,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        candidate_public_raw_correctness = _assignment_correctness(
            candidate_public_raw_out,
            ref,
            x,
            centroids,
        )
        fresh_candidate_public_raw_correctness = _assignment_correctness(
            fresh_candidate_public_raw_out,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        candidate_precomputed_correctness = _assignment_correctness(
            candidate_precomputed_result,
            ref,
            x,
            centroids,
        )
        fresh_candidate_precomputed_correctness = _assignment_correctness(
            fresh_candidate_precomputed_result,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        baseline_precomputed_correctness = _assignment_correctness(
            baseline_precomputed_result,
            ref,
            x,
            centroids,
        )
        fresh_baseline_precomputed_correctness = _assignment_correctness(
            fresh_baseline_precomputed_result,
            fresh_ref,
            fresh_x,
            fresh_centroids,
        )
        result["candidate_public_raw_correctness"] = candidate_public_raw_correctness
        result["candidate_public_raw_fresh_pointer_correctness"] = fresh_candidate_public_raw_correctness
        result["baseline_public_raw_correctness"] = baseline_public_raw_correctness
        result["baseline_public_raw_fresh_pointer_correctness"] = fresh_baseline_public_raw_correctness
        result["candidate_precomputed_correctness"] = candidate_precomputed_correctness
        result["candidate_precomputed_fresh_pointer_correctness"] = fresh_candidate_precomputed_correctness
        result["baseline_precomputed_correctness"] = baseline_precomputed_correctness
        result["baseline_precomputed_fresh_pointer_correctness"] = fresh_baseline_precomputed_correctness
        correctness_rows = (
            candidate_public_raw_correctness,
            fresh_candidate_public_raw_correctness,
            baseline_public_raw_correctness,
            fresh_baseline_public_raw_correctness,
            candidate_precomputed_correctness,
            fresh_candidate_precomputed_correctness,
            baseline_precomputed_correctness,
            fresh_baseline_precomputed_correctness,
        )
        result["correct"] = all(bool(item["correct"]) for item in correctness_rows)

    if benchmark:
        baseline_public_raw_output_holder = [fresh_baseline_public_raw_out]
        candidate_public_raw_output_holder = [fresh_candidate_public_raw_out]

        def held_baseline_public_raw(x_value, centroids_value):
            baseline_public_raw_output_holder[0] = baseline_adapter.compute(x_value, centroids_value)
            return baseline_public_raw_output_holder[0]

        def held_candidate_public_raw(x_value, centroids_value):
            candidate_public_raw_output_holder[0] = runtime.compute(x_value, centroids_value)
            return candidate_public_raw_output_holder[0]

        role_factories = {
            "candidate_public_raw": lambda: _alternating_call(
                lambda: held_candidate_public_raw(x, centroids),
                lambda: held_candidate_public_raw(fresh_x, fresh_centroids),
            ),
            "baseline_public_raw": lambda: _alternating_call(
                lambda: held_baseline_public_raw(x, centroids),
                lambda: held_baseline_public_raw(fresh_x, fresh_centroids),
            ),
            "candidate_precomputed": lambda: _alternating_call(
                lambda: flash_kmeans_assign_prepared(candidate_prepared),
                lambda: flash_kmeans_assign_prepared(fresh_candidate_prepared),
            ),
            "baseline_precomputed": lambda: _alternating_call(
                baseline_precomputed_first_call,
                baseline_precomputed_fresh_pointer_call,
            ),
        }
        cold_first_calls = {
            "candidate_public_raw": candidate_public_raw_fresh_cold_call,
            "baseline_public_raw": baseline_public_raw_fresh_cold_call,
            "candidate_precomputed": fresh_candidate_precomputed_first_cold_call,
            "baseline_precomputed": baseline_precomputed_fresh_cold_call,
        }
        timings, paired_session = _bench_interleaved_roles(
            label=str(row["label"]),
            order_seed=measurement_order_seed,
            role_factories=role_factories,
            cold_first_calls=cold_first_calls,
        )
        candidate_public_raw_timing = timings["candidate_public_raw"]
        baseline_public_raw_timing = timings["baseline_public_raw"]
        candidate_precomputed_timing = timings["candidate_precomputed"]
        baseline_precomputed_timing = timings["baseline_precomputed"]
        timing_backends = {
            candidate_public_raw_timing.backend,
            baseline_public_raw_timing.backend,
            candidate_precomputed_timing.backend,
            baseline_precomputed_timing.backend,
        }
        if timing_backends != {"cupti"}:
            raise RuntimeError(f"dual-lane benchmark requires CUPTI for every timing block, got {timing_backends}")
        result["measurement_order"] = paired_session["first_reportable_roles"]
        result["measurement_order_policy"] = "deterministic_interleaved_adjacent_ab_pairs"
        result["measurement_order_seed"] = measurement_order_seed
        result["paired_cupti_session_id"] = f"{measurement_session_id}:{row['label']}"
        result["paired_cupti_session"] = paired_session
        result["cold_measurement_order"] = list(cold_measurement_order)
        result["cold_measurement_order_policy"] = "deterministic_balanced_per_publication_contract_portfolio"
        result["timing_backend"] = "cupti"
        result["candidate_public_raw_gpu_span_ms"] = candidate_public_raw_timing.median_gpu_span_ms
        result["candidate_public_raw_kernel_sum_ms"] = candidate_public_raw_timing.median_kernel_sum_ms
        result["candidate_public_raw_inter_kernel_gap_ms"] = candidate_public_raw_timing.median_inter_kernel_gap_ms
        result["candidate_public_raw_submission_ms"] = _required_median(
            candidate_public_raw_timing.median_submission_ms,
            metric="candidate public-raw submission_ms",
        )
        result["candidate_public_raw_host_enqueue_ms"] = _required_median(
            candidate_public_raw_timing.median_host_enqueue_ms,
            metric="candidate public-raw host_enqueue_ms",
        )
        result["candidate_public_raw_synchronized_e2e_ms"] = _required_median(
            candidate_public_raw_timing.median_synchronized_e2e_ms,
            metric="candidate public-raw synchronized_e2e_ms",
        )
        result["candidate_public_raw_timing_backend"] = candidate_public_raw_timing.backend
        result["candidate_public_raw_bench_iters"] = len(candidate_public_raw_timing.times_ms)
        result["candidate_public_raw_timing_diagnostics"] = _timing_diagnostics(
            candidate_public_raw_timing,
            primary_metric="synchronized_e2e_ms",
        )

        result["baseline_07cf_adapter_gpu_span_ms"] = baseline_public_raw_timing.median_gpu_span_ms
        result["baseline_07cf_adapter_kernel_sum_ms"] = baseline_public_raw_timing.median_kernel_sum_ms
        result["baseline_07cf_adapter_inter_kernel_gap_ms"] = baseline_public_raw_timing.median_inter_kernel_gap_ms
        result["baseline_07cf_adapter_submission_ms"] = _required_median(
            baseline_public_raw_timing.median_submission_ms,
            metric="07cf raw-adapter submission_ms",
        )
        result["baseline_07cf_adapter_host_enqueue_ms"] = _required_median(
            baseline_public_raw_timing.median_host_enqueue_ms,
            metric="07cf raw-adapter host_enqueue_ms",
        )
        result["baseline_07cf_adapter_synchronized_e2e_ms"] = _required_median(
            baseline_public_raw_timing.median_synchronized_e2e_ms,
            metric="07cf raw-adapter synchronized_e2e_ms",
        )
        result["baseline_07cf_adapter_timing_backend"] = baseline_public_raw_timing.backend
        result["baseline_07cf_adapter_bench_iters"] = len(baseline_public_raw_timing.times_ms)
        result["baseline_07cf_adapter_timing_diagnostics"] = _timing_diagnostics(
            baseline_public_raw_timing,
            primary_metric="synchronized_e2e_ms",
        )

        result["candidate_precomputed_gpu_span_ms"] = candidate_precomputed_timing.median_gpu_span_ms
        result["candidate_precomputed_kernel_sum_ms"] = candidate_precomputed_timing.median_kernel_sum_ms
        result["candidate_precomputed_inter_kernel_gap_ms"] = candidate_precomputed_timing.median_inter_kernel_gap_ms
        result["candidate_precomputed_submission_ms"] = _required_median(
            candidate_precomputed_timing.median_submission_ms,
            metric="candidate prepared submission_ms",
        )
        result["candidate_precomputed_host_enqueue_ms"] = _required_median(
            candidate_precomputed_timing.median_host_enqueue_ms,
            metric="candidate prepared host_enqueue_ms",
        )
        result["candidate_precomputed_synchronized_e2e_ms"] = _required_median(
            candidate_precomputed_timing.median_synchronized_e2e_ms,
            metric="candidate prepared synchronized_e2e_ms",
        )
        result["candidate_precomputed_timing_backend"] = candidate_precomputed_timing.backend
        result["candidate_precomputed_bench_iters"] = len(candidate_precomputed_timing.times_ms)
        result["candidate_precomputed_timing_diagnostics"] = _timing_diagnostics(
            candidate_precomputed_timing,
            primary_metric="gpu_span_ms",
        )

        result["baseline_07cf_precomputed_gpu_span_ms"] = baseline_precomputed_timing.median_gpu_span_ms
        result["baseline_07cf_precomputed_kernel_sum_ms"] = baseline_precomputed_timing.median_kernel_sum_ms
        result["baseline_07cf_precomputed_inter_kernel_gap_ms"] = baseline_precomputed_timing.median_inter_kernel_gap_ms
        result["baseline_07cf_precomputed_submission_ms"] = _required_median(
            baseline_precomputed_timing.median_submission_ms,
            metric="07cf prepared submission_ms",
        )
        result["baseline_07cf_precomputed_host_enqueue_ms"] = _required_median(
            baseline_precomputed_timing.median_host_enqueue_ms,
            metric="07cf prepared host_enqueue_ms",
        )
        result["baseline_07cf_precomputed_synchronized_e2e_ms"] = _required_median(
            baseline_precomputed_timing.median_synchronized_e2e_ms,
            metric="07cf prepared synchronized_e2e_ms",
        )
        result["baseline_07cf_precomputed_timing_backend"] = baseline_precomputed_timing.backend
        result["baseline_07cf_precomputed_bench_iters"] = len(baseline_precomputed_timing.times_ms)
        result["baseline_07cf_precomputed_timing_diagnostics"] = _timing_diagnostics(
            baseline_precomputed_timing,
            primary_metric="gpu_span_ms",
        )

        flops = 2.0 * int(row["B"]) * int(row["N"]) * int(row["K"]) * int(row["D"])
        result["candidate_public_raw_tflops_from_gpu_span"] = (
            flops / candidate_public_raw_timing.median_gpu_span_ms / 1e9
        )
        result.update(
            _dual_lane_speedups(
                candidate_public_raw_timing=candidate_public_raw_timing,
                baseline_public_raw_timing=baseline_public_raw_timing,
                candidate_precomputed_timing=candidate_precomputed_timing,
                baseline_precomputed_timing=baseline_precomputed_timing,
            )
        )
        result["candidate_public_prepared_ratios"] = _public_prepared_ratios(
            candidate_public_raw_timing,
            candidate_precomputed_timing,
        )
        result["baseline_public_prepared_ratios"] = _public_prepared_ratios(
            baseline_public_raw_timing,
            baseline_precomputed_timing,
        )
        candidate_lifecycle = runtime_lifecycle_metrics(
            api="flashlib_cake_kmeans.init(...).compute",
            measurement_session_id=measurement_session_id,
            timing_boundary=CANDIDATE_TIMING_BOUNDARY,
            output_policy="default_output_allocated_before_preprocessing_inside_timing",
            init=candidate_init_timing,
            init_sample_id=measurement_session_id,
            first_compute=candidate_public_raw_shape_cold_call,
            first_cache_state=("shape_slot_hit" if candidate_shape_first_was_cache_hit else "shape_slot_miss"),
            hot_compute=candidate_public_raw_timing,
            hot_cache_state="fresh_pointer_shape_slot_hit",
            code_cache_state="process_order_dependent",
        )
        baseline_lifecycle = runtime_lifecycle_metrics(
            api="triton_h200_07cf_raw_adapter_v1.compute",
            measurement_session_id=measurement_session_id,
            timing_boundary=BASELINE_TIMING_BOUNDARY,
            output_policy="default_output_allocated_before_preprocessing_inside_timing",
            init=baseline_init_timing,
            init_sample_id=measurement_session_id,
            first_compute=baseline_public_raw_shape_cold_call,
            first_cache_state=("shape_slot_hit" if baseline_shape_first_was_cache_hit else "shape_slot_miss"),
            hot_compute=baseline_public_raw_timing,
            hot_cache_state="fresh_pointer_shape_slot_hit",
            code_cache_state="process_order_dependent",
        )
        lifecycle_comparison = compare_runtime_lifecycles(candidate_lifecycle, baseline_lifecycle)
        if not math.isclose(
            lifecycle_comparison["hot_synchronized_e2e_speedup"],
            result["public_raw_e2e_speedup_vs_07cf_adapter"],
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError("KMeans lifecycle hot E2E speedup disagrees with publication speedup")
        result["candidate_runtime_lifecycle"] = candidate_lifecycle
        result["baseline_runtime_lifecycle"] = baseline_lifecycle
        result["runtime_lifecycle_comparison"] = lifecycle_comparison

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness and CUPTI benchmark for flashlib_cake_kmeans.flash_kmeans_assign"
    )
    parser.add_argument("--shape", action="append", help="Shape label to run. Repeatable.")
    parser.add_argument(
        "--source",
        action="append",
        help="Source set to run, for example full95. Repeatable.",
    )
    parser.add_argument("--unique", action="store_true", help="Run one row per unique (B,N,D,K,dtype).")
    parser.add_argument(
        "--include-runtime-coverage",
        action="store_true",
        help="Include the 104 runtime-only rows; publication defaults to the registry 124.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit selected rows after filtering.")
    parser.add_argument("--arch", default=None, help="NVRTC architecture, e.g. sm_100a.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Emit available benchmark metadata without CUDA.",
    )
    parser.add_argument(
        "--no-correctness",
        action="store_true",
        help="Skip PyTorch reference correctness checks.",
    )
    parser.add_argument("--no-benchmark", action="store_true", help="Skip CUPTI timing.")
    parser.add_argument(
        "--reference-chunk-rows",
        type=int,
        default=128,
        help="Rows per chunk for PyTorch reference.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional path for JSON output.")
    parser.add_argument(
        "--order-seed",
        default=DEFAULT_MEASUREMENT_ORDER_SEED,
        help="Deterministic seed for the interleaved paired CUPTI schedule.",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based validation shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Number of validation shards.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the full JSON payload.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.reference_chunk_rows <= 0:
        parser.error("--reference-chunk-rows must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must satisfy 0 <= index < count and count must be positive")
    if not args.order_seed:
        parser.error("--order-seed must be non-empty")
    if args.json is not None:
        args.json.unlink(missing_ok=True)

    rows = _selected_rows(args)
    measurement_session_id = uuid.uuid4().hex
    preprocess_source_digest = _preprocess_source_digest()
    if preprocess_source_digest != BASELINE_REGISTRY_PROFILE["shared_preprocess"]["source_sha256"]:
        raise RuntimeError("exported preprocessing sources do not match the pinned registry profile")
    profile_digest = hashlib.sha256(
        json.dumps(
            BASELINE_REGISTRY_PROFILE,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    if profile_digest != BASELINE_REGISTRY_SHA256:
        raise RuntimeError("embedded registry baseline profile digest is stale")
    payload: dict[str, Any] = {
        "schema": "flash-kmeans-export-dual-lane-v2",
        "api": "flashlib_cake_kmeans.init(...).compute",
        "semantic_entrypoint": SEMANTIC_ENTRYPOINT,
        "publication_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "parity_metric": "precomputed_gpu_speedup_vs_07cf",
        "baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "public_raw_baseline_name": PUBLIC_RAW_BASELINE_NAME,
        "precomputed_baseline_name": PRECOMPUTED_BASELINE_NAME,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_entrypoint": BASELINE_ENTRYPOINT,
        "benchmark_registry_baseline_key": BASELINE_REGISTRY_KEY,
        "benchmark_registry_baseline_sha256": BASELINE_REGISTRY_SHA256,
        "benchmark_registry_baseline_profile": BASELINE_REGISTRY_PROFILE,
        "registry_candidate_entrypoint": REGISTRY_CANDIDATE_ENTRYPOINT,
        "measured_candidate_entrypoint": MEASURED_CANDIDATE_ENTRYPOINT,
        "official_timing_boundary": CANDIDATE_TIMING_BOUNDARY,
        "candidate_timing_boundary": CANDIDATE_TIMING_BOUNDARY,
        "benchmark_registry_candidate_timing_boundary": REGISTRY_CANDIDATE_TIMING_BOUNDARY,
        "baseline_timing_boundary": BASELINE_TIMING_BOUNDARY,
        "publication_speedup_metric": "public_raw_e2e_speedup_vs_07cf_adapter",
        "publication_timing_backend": "cupti",
        "publication_speedup_convention": (
            "public_raw_e2e_speedup_vs_07cf_adapter = "
            "baseline_07cf_adapter_synchronized_e2e_ms / candidate_public_raw_synchronized_e2e_ms"
        ),
        "precomputed_parity_speedup_convention": (
            "precomputed_gpu_speedup_vs_07cf = "
            "baseline_07cf_precomputed_gpu_span_ms / candidate_precomputed_gpu_span_ms"
        ),
        "preprocess_impl": PREPROCESS_IMPL,
        "preprocess_source_sha256": preprocess_source_digest,
        "timing_window_ms": {"warmup_ms": WARMUP_MS, "bench_ms": BENCH_MS},
        "submission_metric": {
            "field": "submission_ms",
            "host_enqueue_alias_field": "host_enqueue_ms",
            "relationship": "same_cpu_submission_bracket",
        },
        "artifact": FLASH_KMEANS_EVOLUTION_ARTIFACT,
        "evolution_summary": FLASH_KMEANS_EVOLUTION_SUMMARY,
        "selected_row_count": len(rows),
        "selected_unique_shape_count": len({_shape_key(row) for row in rows}),
        "metadata_only": bool(args.metadata_only),
        "validation_shard": {"index": args.shard_index, "count": args.shard_count},
        "shapes": rows,
        "measurement_session": {
            "id": measurement_session_id,
            "scope": "per_shape_single_cupti_activity_session_interleaved_paired_roles",
            "same_process": True,
            "same_cupti_session": (
                True if not args.metadata_only and not args.no_benchmark else None
            ),
            "interleaved": (
                True if not args.metadata_only and not args.no_benchmark else None
            ),
            "reportable_timing_collected": bool(
                not args.metadata_only and not args.no_benchmark
            ),
            "randomized_or_interleaved_order": True,
            "sequential_full_sweeps": False,
            "timing_blocks": "one_interleaved_cupti_activity_session_per_shape",
            "adaptive_probe_scope": "separate_nonreportable_cupti_estimation_only",
            "alternate_two_pointer_sets": True,
            "public_raw_e2e": {
                "candidate_api": "flashlib_cake_kmeans.init(...).compute(raw_inputs)",
                "baseline_api": "triton_h200_07cf_raw_adapter_v1.compute(raw_inputs)",
                "comparison_scope": "complete_raw_input_operators_not_assignment_only",
                "candidate_norm_policy": "route_required_internal_fused_bf16_pair_row_norm",
                "baseline_norm_policy": "shared_kernel_all_fields_required_by_frozen_07cf",
                "candidate_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
                "baseline_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
                "candidate_scratch_policy": "per_shape_per_stream_cached",
                "baseline_scratch_policy": "per_shape_per_stream_cached",
                "candidate_runtime_initialized_once": True,
                "baseline_runtime_initialized_once": True,
                "preprocess_impl": PREPROCESS_IMPL,
                "preprocess_source_sha256": preprocess_source_digest,
                "shared_preprocess_cold_compile_attributed_to_lane": None,
                "fixture_synchronized_before_cold_calls": True,
                "assignment_baseline": "frozen_07cf",
            },
            "precomputed_kernel_parity": {
                "candidate_api": "flash_kmeans_assign_prepared",
                "baseline_api": "euclid_assign_triton_h200_07cf",
                "candidate_norm_policy": "explicit_precomputed_outside_timing",
                "baseline_norm_policy": "explicit_precomputed_outside_timing",
                "candidate_output_policy": "preallocated_outside_timing",
                "baseline_output_policy": "preallocated_outside_timing",
                "both_pointer_sets_prepared_before_timing": True,
            },
            "runtime_instances_reused_across_shapes": True,
            "resident_multi_shape_cache_benchmarked": False,
            "cache_policy": "synchronize_and_clear_after_each_completed_shape",
            "baseline_commit": BASELINE_COMMIT,
            "order_policy": "deterministic_interleaved_adjacent_ab_pairs",
            "cold_order_policy": "deterministic_balanced_per_publication_contract_portfolio",
            "init_composition": "runtime_init_plus_shared_preprocess_support_each_standalone_lane",
            "init_order_policy": "alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support",
            "baseline_has_explicit_init": True,
            "order_seed": args.order_seed,
        },
        "runtime_lifecycle": {
            "schema": "loom-public-runtime-lifecycle-v1",
            "candidate_api": "flashlib_cake_kmeans.init(...).compute",
            "baseline_api": "triton_h200_07cf_raw_adapter_v1.compute",
            "candidate_timing_boundary": CANDIDATE_TIMING_BOUNDARY,
            "baseline_timing_boundary": BASELINE_TIMING_BOUNDARY,
            "init_scope": (
                "runtime_init_plus_standalone_shared_preprocess_support_once_per_"
                "validation_shard_process_device_operator"
            ),
            "amortization_call_counts": [1, 10, 100, 1000],
            "cache_policy": "synchronize_and_clear_after_each_completed_shape",
            "cold_order_policy": "deterministic_balanced_per_publication_contract_portfolio",
            "init_composition": "runtime_init_plus_shared_preprocess_support_each_standalone_lane",
            "init_order_policy": "alternate_candidate_baseline_first_by_validation_shard_parity_then_shared_support",
            "baseline_has_explicit_init": True,
            "resident_multi_shape_cache_benchmarked": False,
            "candidate_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
            "baseline_output_policy": "default_output_allocated_before_preprocessing_inside_timing",
            "session": None,
        },
    }
    if args.metadata_only:
        payload["results"] = []
    else:
        import torch
        from flashlib_cake_kmeans import init

        capability = torch.cuda.get_device_capability()
        detected_arch = f"sm_{capability[0]}{capability[1]}" + ("" if capability[0] == 8 else "a")
        payload["hardware"] = {
            "device": torch.cuda.get_device_name(),
            "arch": detected_arch,
        }
        device_index = torch.cuda.current_device()
        resolved_arch = args.arch or payload["hardware"]["arch"]
        init_measurement_order = ("candidate", "baseline") if args.shard_index % 2 == 0 else ("baseline", "candidate")
        if not args.no_benchmark:
            require_cupti()
            runtime = None
            baseline_adapter = None
            runtime_init_call = None
            baseline_adapter_init_call = None
            for lane in init_measurement_order:
                if lane == "candidate":
                    runtime, runtime_init_call = measure_host_call(
                        lambda: init(
                            device=device_index,
                            arch=resolved_arch,
                            compile="lazy",
                        )
                    )
                else:
                    baseline_adapter, baseline_adapter_init_call = measure_host_call(
                        lambda: TritonH20007cfRawAdapter(
                            device_index=device_index,
                            arch=resolved_arch,
                        )
                    )
            if runtime is None or baseline_adapter is None:
                raise RuntimeError("KMeans init measurement did not construct both standalone lanes")
        else:
            runtime = init(
                device=device_index,
                arch=resolved_arch,
                compile="lazy",
            )
            runtime_init_call = None
            baseline_adapter = TritonH20007cfRawAdapter(
                device_index=device_index,
                arch=resolved_arch,
            )
            baseline_adapter_init_call = None
        shared_preprocess_cold_compile = (
            _measure_shared_preprocess_cold_compile(rows[0], arch=runtime.arch)
            if rows and not args.no_benchmark
            else None
        )
        candidate_lifecycle_init = (
            _sum_host_timings(runtime_init_call, shared_preprocess_cold_compile)
            if runtime_init_call is not None and shared_preprocess_cold_compile is not None
            else runtime_init_call
        )
        baseline_lifecycle_init = (
            _sum_host_timings(baseline_adapter_init_call, shared_preprocess_cold_compile)
            if baseline_adapter_init_call is not None and shared_preprocess_cold_compile is not None
            else baseline_adapter_init_call
        )
        payload["cold_candidate_runtime_init"] = _cold_call_fields(runtime_init_call)
        payload["cold_baseline_07cf_adapter_init"] = _cold_call_fields(baseline_adapter_init_call)
        payload["cold_shared_preprocess_compile"] = _cold_call_fields(shared_preprocess_cold_compile)
        results = []
        for row in rows:
            try:
                results.append(
                    _run_shape(
                        row,
                        runtime=runtime,
                        baseline_adapter=baseline_adapter,
                        candidate_init_timing=candidate_lifecycle_init,
                        baseline_init_timing=baseline_lifecycle_init,
                        arch=args.arch,
                        correctness=not args.no_correctness,
                        benchmark=not args.no_benchmark,
                        reference_chunk_rows=args.reference_chunk_rows,
                        measurement_session_id=measurement_session_id,
                        measurement_order_seed=args.order_seed,
                    )
                )
            finally:
                runtime.clear()
                baseline_adapter.clear()
        payload["results"] = results
        payload["runtime_workspace_lifecycle"] = (
            "both_init_once_runtimes_synchronize_and_clear_after_each_completed_shape"
        )
        candidate_first_miss_count = sum(
            not bool(row["candidate_public_raw_shape_first_was_cache_hit"]) for row in results
        )
        baseline_first_miss_count = sum(
            not bool(row["baseline_public_raw_shape_first_was_cache_hit"]) for row in results
        )
        candidate_fresh_hit_count = sum(bool(row["candidate_public_raw_fresh_pointer_cache_hit"]) for row in results)
        baseline_fresh_hit_count = sum(bool(row["baseline_public_raw_fresh_pointer_cache_hit"]) for row in results)
        if args.no_benchmark:
            payload.pop("runtime_lifecycle", None)
        else:
            payload["runtime_lifecycle"]["session"] = {
                "id": measurement_session_id,
                "init_measurement_order": [*init_measurement_order, "shared_preprocess_compile"],
                "candidate_init": _cold_call_fields(candidate_lifecycle_init),
                "baseline_init": _cold_call_fields(baseline_lifecycle_init),
                "init_components": {
                    "candidate_runtime_init": _cold_call_fields(runtime_init_call),
                    "baseline_runtime_init": _cold_call_fields(baseline_adapter_init_call),
                    "shared_preprocess_compile": _cold_call_fields(shared_preprocess_cold_compile),
                    "attribution": "included_once_in_each_standalone_lane_init_total",
                },
                "clear_count": {"candidate": len(results), "baseline": len(results)},
                "first_lookup_miss_count": {
                    "candidate": candidate_first_miss_count,
                    "baseline": baseline_first_miss_count,
                },
                "fresh_pointer_hit_count": {
                    "candidate": candidate_fresh_hit_count,
                    "baseline": baseline_fresh_hit_count,
                },
                "final_candidate_cache_info": runtime.cache_info(),
                "final_baseline_cache_info": baseline_adapter.cache_info(),
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
