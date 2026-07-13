from __future__ import annotations

import bisect
import contextlib
import ctypes
import importlib
import importlib.metadata
import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchResult:
    """Strict CUPTI timing plus explicitly non-official host diagnostics.

    Cold-L2 flushing completes before host timestamps begin, so synchronized
    E2E isolates semantic call start through candidate completion. ``times_ms``
    remains the official correlated GPU span.
    """

    times_ms: list[float]
    backend: str = "cupti"
    kernel_sum_times_ms: list[float] | None = None
    inter_kernel_gap_times_ms: list[float] | None = None
    active_union_times_ms: list[float] | None = None
    activity_counts: list[int] | None = None
    launch_activity_counts: list[int] | None = None
    kernel_activity_counts: list[int] | None = None
    submission_times_ms: list[float] | None = None
    synchronized_e2e_times_ms: list[float] | None = None
    cold_first_call_host_enqueue_ms: float | None = None
    cold_first_call_synchronized_e2e_ms: float | None = None

    def __post_init__(self) -> None:
        if self.backend != "cupti":
            raise ValueError(f"exported benchmark timing backend must be 'cupti', got {self.backend!r}")

    @property
    def median_ms(self) -> float:
        return float(statistics.median(self.times_ms))

    @property
    def min_ms(self) -> float:
        return float(min(self.times_ms))

    @property
    def mean_ms(self) -> float:
        return float(statistics.fmean(self.times_ms))

    @property
    def median_gpu_span_ms(self) -> float:
        return self.median_ms

    @property
    def median_kernel_sum_ms(self) -> float | None:
        if self.kernel_sum_times_ms is None:
            return None
        return float(statistics.median(self.kernel_sum_times_ms))

    @property
    def median_inter_kernel_gap_ms(self) -> float | None:
        if self.inter_kernel_gap_times_ms is None:
            return None
        return float(statistics.median(self.inter_kernel_gap_times_ms))

    @property
    def median_active_union_ms(self) -> float | None:
        if self.active_union_times_ms is None:
            return None
        return float(statistics.median(self.active_union_times_ms))

    @property
    def median_activity_count(self) -> float | None:
        if self.activity_counts is None:
            return None
        return float(statistics.median(self.activity_counts))

    @property
    def median_launch_activity_count(self) -> float | None:
        if self.launch_activity_counts is None:
            return None
        return float(statistics.median(self.launch_activity_counts))

    @property
    def median_kernel_activity_count(self) -> float | None:
        if self.kernel_activity_counts is None:
            return None
        return float(statistics.median(self.kernel_activity_counts))

    @property
    def median_submission_ms(self) -> float | None:
        if self.submission_times_ms is None:
            return None
        return float(statistics.median(self.submission_times_ms))

    @property
    def median_host_enqueue_ms(self) -> float | None:
        return self.median_submission_ms

    @property
    def host_enqueue_times_ms(self) -> list[float] | None:
        return self.submission_times_ms

    @property
    def median_synchronized_e2e_ms(self) -> float | None:
        if self.synchronized_e2e_times_ms is None:
            return None
        return float(statistics.median(self.synchronized_e2e_times_ms))


@dataclass(frozen=True)
class HostCallTiming:
    """Diagnostic host brackets for one call; never an official GPU timing."""

    host_enqueue_ms: float
    synchronized_e2e_ms: float


@dataclass(frozen=True)
class _CuptiTiming:
    gpu_span_ms: list[float]
    kernel_sum_ms: list[float]
    inter_kernel_gap_ms: list[float]
    active_union_ms: list[float]
    activity_count: list[int]
    launch_activity_count: list[int]
    kernel_activity_count: list[int]


class _L2Flusher:
    def __init__(self) -> None:
        import torch

        l2_size = int(torch.cuda.get_device_properties(0).L2_cache_size)
        if l2_size <= 0:
            raise RuntimeError("CUDA device did not report a positive L2 cache size")
        self._buffer = torch.empty(2 * l2_size, dtype=torch.int8, device="cuda")

    def flush(self) -> None:
        self._buffer.zero_()


_CUPTI: Any | None = None


def _extend_cuda_namespace_for_pathfinder() -> None:
    try:
        import cuda
    except ImportError:
        return
    cuda_paths = getattr(cuda, "__path__", None)
    if cuda_paths is None:
        return
    known = {str(path) for path in cuda_paths}
    for entry in sys.path:
        if not entry:
            continue
        cuda_root = Path(entry) / "cuda"
        if (cuda_root / "pathfinder").is_dir() and str(cuda_root) not in known:
            cuda_paths.append(str(cuda_root))
            known.add(str(cuda_root))


def _preload_cupti_library() -> None:
    # cupti-python imports cuda.pathfinder even when libcupti was found through
    # the nvidia-cuda-cupti distribution.  Extend the split CUDA namespace
    # before either loading path so the later extension import is reliable.
    _extend_cuda_namespace_for_pathfinder()
    try:
        distribution = importlib.metadata.distribution("nvidia-cuda-cupti")
        major = importlib.metadata.version("cupti-python").split(".", 1)[0]
        packaged = next(
            (
                distribution.locate_file(path)
                for path in distribution.files or ()
                if str(path).endswith(f"/libcupti.so.{major}")
            ),
            None,
        )
        if packaged is not None and Path(packaged).is_file():
            ctypes.CDLL(str(packaged), mode=ctypes.RTLD_GLOBAL)
            return
    except (importlib.metadata.PackageNotFoundError, OSError):
        pass
    try:
        pathfinder = importlib.import_module("cuda.pathfinder")
    except ImportError:
        return
    loader = getattr(pathfinder, "load_nvidia_dynamic_lib", None)
    if loader is None:
        return
    loaded = loader("cupti")
    loaded_path = str(getattr(loaded, "abs_path", ""))
    try:
        major = importlib.metadata.version("cupti-python").split(".", 1)[0]
    except importlib.metadata.PackageNotFoundError:
        return
    expected = f"libcupti.so.{major}" if major.isdigit() else None
    if expected and loaded_path:
        name = Path(loaded_path).name
        if name.startswith("libcupti.so.") and name != expected:
            raise ImportError(f"incompatible CUPTI library {loaded_path}; cupti-python expects {expected}")


def require_cupti() -> Any:
    global _CUPTI
    if _CUPTI is not None:
        return _CUPTI
    try:
        _preload_cupti_library()
        from cupti import cupti
    except ImportError as exc:
        raise RuntimeError(
            "CUPTI timing is required; install the exported repository with its benchmark extra: "
            "python -m pip install -e '.[benchmark]'"
        ) from exc
    _CUPTI = cupti
    return _CUPTI


def measure_host_call(fn: Callable[[], Any]) -> tuple[Any, HostCallTiming]:
    """Run one call and return explicitly labeled host diagnostics.

    The enqueue bracket ends when ``fn`` returns.  The synchronized bracket
    ends after ``torch.cuda.synchronize()``.  A function that synchronizes
    internally will therefore have a blocking enqueue bracket; callers should
    not interpret either value as GPU-only execution time.
    """
    import torch

    cupti = require_cupti()
    start = cupti.get_timestamp()
    value = fn()
    submitted = cupti.get_timestamp()
    torch.cuda.synchronize()
    completed = cupti.get_timestamp()
    return value, HostCallTiming(
        host_enqueue_ms=(submitted - start) / 1e6,
        synchronized_e2e_ms=(completed - start) / 1e6,
    )


def _finite_timing(value: Any, *, name: str, positive: bool) -> float:
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        relation = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {relation}, got {value!r}")
    return number


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * float(fraction)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_distribution(values: Any, *, name: str, positive: bool) -> dict[str, Any]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must contain at least one timing sample")
    samples = [_finite_timing(value, name=name, positive=positive) for value in values]
    return {
        "sample_count": len(samples),
        "min": min(samples),
        "median": float(statistics.median(samples)),
        "p90": _percentile(samples, 0.90),
        "max": max(samples),
    }


def _host_call_payload(timing: HostCallTiming, *, name: str) -> dict[str, Any]:
    host_enqueue_ms = _finite_timing(timing.host_enqueue_ms, name=f"{name}.host_enqueue_ms", positive=False)
    synchronized_e2e_ms = _finite_timing(
        timing.synchronized_e2e_ms,
        name=f"{name}.synchronized_e2e_ms",
        positive=True,
    )
    if host_enqueue_ms > synchronized_e2e_ms:
        raise ValueError(f"{name}.host_enqueue_ms cannot exceed synchronized_e2e_ms")
    return {
        "timing_class": "cupti_timestamp_host_diagnostic",
        "host_enqueue_ms": host_enqueue_ms,
        "synchronized_e2e_ms": synchronized_e2e_ms,
    }


def _hot_compute_payload(timing: BenchResult, *, name: str, cache_state: str) -> dict[str, Any]:
    if timing.backend != "cupti":
        raise ValueError(f"{name} must use CUPTI")
    gpu_span = _timing_distribution(timing.times_ms, name=f"{name}.gpu_span_ms", positive=True)
    host_enqueue = _timing_distribution(
        timing.submission_times_ms,
        name=f"{name}.host_enqueue_ms",
        positive=False,
    )
    synchronized_e2e = _timing_distribution(
        timing.synchronized_e2e_times_ms,
        name=f"{name}.synchronized_e2e_ms",
        positive=True,
    )
    sample_counts = {
        gpu_span["sample_count"],
        host_enqueue["sample_count"],
        synchronized_e2e["sample_count"],
    }
    if len(sample_counts) != 1:
        raise ValueError(f"{name} CUPTI/host timing sample counts must match")
    for index, (enqueue_sample, e2e_sample) in enumerate(
        zip(timing.submission_times_ms, timing.synchronized_e2e_times_ms, strict=True)
    ):
        if float(enqueue_sample) > float(e2e_sample):
            raise ValueError(f"{name} host enqueue sample {index} cannot exceed its synchronized E2E sample")
    return {
        "timing_backend": "cupti",
        "cache_state": cache_state,
        "sample_count": gpu_span["sample_count"],
        "gpu_span_ms": gpu_span,
        "host_enqueue_ms": host_enqueue,
        "synchronized_e2e_ms": synchronized_e2e,
    }


def runtime_lifecycle_metrics(
    *,
    api: str,
    measurement_session_id: str,
    timing_boundary: str,
    output_policy: str,
    init: HostCallTiming | None,
    init_sample_id: str | None,
    first_compute: HostCallTiming,
    first_cache_state: str,
    hot_compute: BenchResult,
    hot_cache_state: str,
    amortization_call_counts: tuple[int, ...] = (1, 10, 100, 1000),
    code_cache_state: str = "process_order_dependent",
) -> dict[str, Any]:
    """Normalize init-once, first-signature, and repeated public-call evidence.

    ``first_compute`` is intentionally a CUPTI-timestamp host diagnostic: route
    selection, compilation, allocation, launch, and completion all belong to
    its synchronized E2E bracket, but it is not mislabeled as GPU-only time.
    ``hot_compute`` retains strict correlated CUPTI activity timing.
    """

    for field_name, value in (
        ("api", api),
        ("measurement_session_id", measurement_session_id),
        ("timing_boundary", timing_boundary),
        ("output_policy", output_policy),
        ("first_cache_state", first_cache_state),
        ("hot_cache_state", hot_cache_state),
        ("code_cache_state", code_cache_state),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"runtime lifecycle {field_name} must be a non-empty string")
    if (
        not amortization_call_counts
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in amortization_call_counts)
        or tuple(amortization_call_counts) != tuple(sorted(set(amortization_call_counts)))
    ):
        raise ValueError("amortization_call_counts must be strictly increasing positive integers")

    init_payload = _host_call_payload(init, name="init_once") if init is not None else None
    if init_payload is not None:
        if not isinstance(init_sample_id, str) or not init_sample_id.strip():
            raise ValueError("init_sample_id is required when init timing is present")
        init_payload = {"sample_id": init_sample_id, **init_payload}
    elif init_sample_id is not None:
        raise ValueError("init_sample_id requires init timing")
    first_payload = {
        "cache_state": first_cache_state,
        "code_cache_state": code_cache_state,
        **_host_call_payload(first_compute, name="first_compute"),
        "gpu_span_ms": None,
        "gpu_span_status": "not_collected_for_slot_miss_host_diagnostic",
    }
    hot_payload = _hot_compute_payload(hot_compute, name="hot_compute", cache_state=hot_cache_state)

    first_enqueue = first_payload["host_enqueue_ms"]
    first_e2e = first_payload["synchronized_e2e_ms"]
    hot_enqueue = hot_payload["host_enqueue_ms"]["median"]
    hot_e2e = hot_payload["synchronized_e2e_ms"]["median"]
    init_enqueue = init_payload["host_enqueue_ms"] if init_payload is not None else 0.0
    init_e2e = init_payload["synchronized_e2e_ms"] if init_payload is not None else 0.0
    amortized: list[dict[str, Any]] = []
    for call_count in amortization_call_counts:
        repeated = call_count - 1
        after_init_enqueue = (first_enqueue + repeated * hot_enqueue) / call_count
        after_init_e2e = (first_e2e + repeated * hot_e2e) / call_count
        amortized.append(
            {
                "public_call_count": call_count,
                "after_init_host_enqueue_ms_per_call": after_init_enqueue,
                "after_init_synchronized_e2e_ms_per_call": after_init_e2e,
                "including_init_host_enqueue_ms_per_call": (
                    (init_enqueue + first_enqueue + repeated * hot_enqueue) / call_count
                ),
                "including_init_synchronized_e2e_ms_per_call": (
                    (init_e2e + first_e2e + repeated * hot_e2e) / call_count
                ),
            }
        )

    return {
        "schema": "loom-public-runtime-lifecycle-v1",
        "api": api,
        "measurement_session_id": measurement_session_id,
        "timing_boundary": timing_boundary,
        "output_policy": output_policy,
        "init_once": init_payload,
        "first_compute": first_payload,
        "hot_compute": hot_payload,
        "amortization": {
            "model": "observed_first_call_plus_repeated_hot_median",
            "init_attribution": "one_init_sample_per_validation_shard_runtime",
            "missing_init_policy": "zero_for_api_without_explicit_init",
            "call_counts": amortized,
        },
    }


def compare_runtime_lifecycles(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare two normalized public-API lifecycle records."""

    for name, lifecycle in (("candidate", candidate), ("baseline", baseline)):
        if not isinstance(lifecycle, dict) or lifecycle.get("schema") != "loom-public-runtime-lifecycle-v1":
            raise ValueError(f"{name} runtime lifecycle has an invalid schema")
    candidate_hot = candidate["hot_compute"]
    baseline_hot = baseline["hot_compute"]
    candidate_rows = candidate["amortization"]["call_counts"]
    baseline_rows = baseline["amortization"]["call_counts"]
    candidate_counts = [row["public_call_count"] for row in candidate_rows]
    baseline_counts = [row["public_call_count"] for row in baseline_rows]
    if candidate_counts != baseline_counts:
        raise ValueError("candidate and baseline amortization call counts must match")

    amortized: list[dict[str, Any]] = []
    for candidate_row, baseline_row in zip(candidate_rows, baseline_rows, strict=True):
        candidate_after_init = candidate_row["after_init_synchronized_e2e_ms_per_call"]
        baseline_after_init = baseline_row["after_init_synchronized_e2e_ms_per_call"]
        candidate_including_init = candidate_row["including_init_synchronized_e2e_ms_per_call"]
        baseline_including_init = baseline_row["including_init_synchronized_e2e_ms_per_call"]
        amortized.append(
            {
                "public_call_count": candidate_row["public_call_count"],
                "after_init_synchronized_e2e_speedup": baseline_after_init / candidate_after_init,
                "including_init_synchronized_e2e_speedup": baseline_including_init / candidate_including_init,
            }
        )

    return {
        "schema": "loom-public-runtime-lifecycle-comparison-v1",
        "speedup_convention": "baseline_latency_divided_by_candidate_latency",
        "hot_synchronized_e2e_speedup": (
            baseline_hot["synchronized_e2e_ms"]["median"]
            / candidate_hot["synchronized_e2e_ms"]["median"]
        ),
        "hot_gpu_span_speedup": (
            baseline_hot["gpu_span_ms"]["median"] / candidate_hot["gpu_span_ms"]["median"]
        ),
        "amortized": amortized,
    }


def _complete_l2_flush_before_bracket(flusher: Any, synchronize: Callable[[], None]) -> None:
    """Finish cold-L2 preconditioning before host latency timestamps begin."""
    if flusher is None:
        return
    flusher.flush()
    synchronize()


def _correlate(
    cpu_brackets: list[tuple[int, int, int]],
    launches: list[tuple[int, int, int]],
    kernels: list[tuple[int, int, int]],
) -> _CuptiTiming:
    if not launches or not kernels:
        raise RuntimeError("CUPTI collected no launch/kernel activities")
    launches.sort(key=lambda item: item[0])
    launch_starts = [item[0] for item in launches]
    kernels.sort(key=lambda item: item[0])
    kernel_indices_by_correlation: dict[int, list[int]] = {}
    for index, (_start, _end, correlation_id) in enumerate(kernels):
        kernel_indices_by_correlation.setdefault(correlation_id, []).append(index)

    gpu_span_ms: list[float] = []
    kernel_sum_ms: list[float] = []
    inter_kernel_gap_ms: list[float] = []
    active_union_ms: list[float] = []
    activity_count: list[int] = []
    launch_activity_count: list[int] = []
    kernel_activity_count: list[int] = []
    for bracket_start, bracket_end, _completed in cpu_brackets:
        lo = bisect.bisect_left(launch_starts, bracket_start)
        hi = bisect.bisect_right(launch_starts, bracket_end)
        correlation_ids = {launches[index][2] for index in range(lo, hi)}
        launch_activity_count.append(
            sum(
                1
                for index in range(lo, hi)
                if kernel_indices_by_correlation.get(launches[index][2])
            )
        )
        selected_indices = {
            index
            for correlation_id in correlation_ids
            for index in kernel_indices_by_correlation.get(correlation_id, ())
        }
        if not selected_indices:
            raise RuntimeError("CUPTI could not correlate a benchmark iteration with GPU kernel activity")
        spans = [(kernels[index][0], kernels[index][1]) for index in selected_indices]
        span_ns = max(end for _, end in spans) - min(start for start, _ in spans)
        sum_ns = sum(end - start for start, end in spans)
        covered_ns = 0
        current_start, current_end = sorted(spans)[0]
        for start, end in sorted(spans)[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                covered_ns += current_end - current_start
                current_start, current_end = start, end
        covered_ns += current_end - current_start
        gpu_span_ms.append(span_ns / 1e6)
        kernel_sum_ms.append(sum_ns / 1e6)
        active_union_ms.append(covered_ns / 1e6)
        inter_kernel_gap_ms.append((span_ns - covered_ns) / 1e6)
        activity_count.append(len(selected_indices))
        kernel_activity_count.append(len(selected_indices))
    return _CuptiTiming(
        gpu_span_ms=gpu_span_ms,
        kernel_sum_ms=kernel_sum_ms,
        inter_kernel_gap_ms=inter_kernel_gap_ms,
        active_union_ms=active_union_ms,
        activity_count=activity_count,
        launch_activity_count=launch_activity_count,
        kernel_activity_count=kernel_activity_count,
    )


def bench_gpu_time(
    fn: Callable[[], Any],
    *,
    warmup_iters: int = 5,
    bench_iters: int = 20,
    cold_l2: bool = True,
    cold_first_call: HostCallTiming | None = None,
) -> BenchResult:
    """Measure a zero-argument GPU workload with strict CUPTI activity tracing.

    L2 is flushed before every warmup and measured iteration. This function
    never falls back to wall-clock or CUDA-event timing.
    """
    if warmup_iters < 0 or bench_iters <= 0:
        raise ValueError("warmup_iters must be non-negative and bench_iters must be positive")

    import torch

    cupti = require_cupti()
    flusher = _L2Flusher() if cold_l2 else None
    for _ in range(warmup_iters):
        if flusher is not None:
            flusher.flush()
        fn()
    torch.cuda.synchronize()

    launch_kinds = {int(cupti.ActivityKind.RUNTIME), int(cupti.ActivityKind.DRIVER)}
    kernel_kinds = {int(cupti.ActivityKind.CONCURRENT_KERNEL)}
    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int]] = []

    def _buffer_requested():
        return 8 * 1024 * 1024, 0

    def _buffer_completed(activities: list[Any]) -> None:
        for activity in activities:
            record = (activity.start, activity.end, activity.correlation_id)
            kind = int(activity.kind)
            if kind in launch_kinds:
                launches.append(record)
            elif kind in kernel_kinds:
                kernels.append(record)

    kinds = [
        cupti.ActivityKind.RUNTIME,
        cupti.ActivityKind.DRIVER,
        cupti.ActivityKind.CONCURRENT_KERNEL,
        cupti.ActivityKind.MEMCPY,
        cupti.ActivityKind.MEMSET,
    ]
    enabled: list[Any] = []
    cpu_brackets: list[tuple[int, int, int]] = []
    watchdog = sys.modules.get("loom.runtime.cuda_watchdog")
    suspend_polling = getattr(watchdog, "suspend_polling", None)
    polling_guard = suspend_polling() if callable(suspend_polling) else contextlib.nullcontext()
    with polling_guard:
        cupti.activity_register_callbacks(_buffer_requested, _buffer_completed)
        try:
            for kind in kinds:
                cupti.activity_enable(kind)
                enabled.append(kind)

            for _ in range(bench_iters):
                _complete_l2_flush_before_bracket(flusher, torch.cuda.synchronize)
                start = cupti.get_timestamp()
                fn()
                submitted = cupti.get_timestamp()
                torch.cuda.synchronize()
                completed = cupti.get_timestamp()
                cpu_brackets.append((start, submitted, completed))
            cupti.activity_flush_all(1)
        finally:
            for kind in reversed(enabled):
                cupti.activity_disable(kind)
            cupti.finalize()

    timing = _correlate(cpu_brackets, launches, kernels)
    return BenchResult(
        times_ms=timing.gpu_span_ms,
        kernel_sum_times_ms=timing.kernel_sum_ms,
        inter_kernel_gap_times_ms=timing.inter_kernel_gap_ms,
        active_union_times_ms=timing.active_union_ms,
        activity_counts=timing.activity_count,
        launch_activity_counts=timing.launch_activity_count,
        kernel_activity_counts=timing.kernel_activity_count,
        submission_times_ms=[(submitted - start) / 1e6 for start, submitted, _ in cpu_brackets],
        synchronized_e2e_times_ms=[
            (completed - start) / 1e6 for start, _, completed in cpu_brackets
        ],
        cold_first_call_host_enqueue_ms=(
            cold_first_call.host_enqueue_ms if cold_first_call is not None else None
        ),
        cold_first_call_synchronized_e2e_ms=(
            cold_first_call.synchronized_e2e_ms if cold_first_call is not None else None
        ),
    )
