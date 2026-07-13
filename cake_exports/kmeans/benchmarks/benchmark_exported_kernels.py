from __future__ import annotations

import argparse
import importlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PACKAGE_NAME = 'flashlib_cake_kmeans'


def _selected_names(pkg: Any, requested: list[str] | None) -> list[str]:
    if not requested:
        return list(pkg.KERNELS)
    missing = sorted(set(requested) - set(pkg.KERNELS))
    if missing:
        available = ", ".join(sorted(pkg.KERNELS))
        raise SystemExit(f"unknown kernel(s) {missing}. Available: {available}")
    return requested


def _entry_for_kernel(pkg: Any, name: str, *, metadata_only: bool, arch: str | None, iterations: int):
    kernel = pkg.get_kernel(name)
    entry: dict[str, Any] = {
        "name": name,
        "symbol": kernel.spec.symbol,
        "launch_mode": kernel.spec.launch_mode,
        "threads": kernel.spec.threads,
        "shared_mem_bytes": kernel.spec.shared_mem_bytes,
        "parameter_count": len(kernel.parameters),
        "status": "metadata_only" if metadata_only else "pending",
    }
    if metadata_only:
        return entry

    times_ms: list[float] = []
    try:
        runtime = importlib.import_module(f"{PACKAGE_NAME}._runtime")
        for _ in range(iterations):
            runtime.clear_compilation_cache()
            start = time.perf_counter()
            kernel.compile(arch=arch)
            times_ms.append((time.perf_counter() - start) * 1000.0)
    except Exception as exc:  # noqa: BLE001 - benchmark report should preserve the failure.
        entry["status"] = "failed"
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry

    entry["status"] = "passed"
    entry["compile_ms_median"] = statistics.median(times_ms)
    entry["compile_ms_min"] = min(times_ms)
    entry["compile_ms_max"] = max(times_ms)
    entry["iterations"] = iterations
    return entry


def run_benchmark(
    *,
    kernels: list[str] | None = None,
    arch: str | None = None,
    iterations: int = 1,
    metadata_only: bool = False,
) -> dict[str, Any]:
    pkg = importlib.import_module(PACKAGE_NAME)
    selected = _selected_names(pkg, kernels)
    entries = [
        _entry_for_kernel(pkg, name, metadata_only=metadata_only, arch=arch, iterations=iterations)
        for name in selected
    ]
    passed = sum(1 for entry in entries if entry["status"] in {"passed", "metadata_only"})
    failed = sum(1 for entry in entries if entry["status"] == "failed")
    return {
        "benchmark": "exported_kernel_compile",
        "package": PACKAGE_NAME,
        "arch": arch,
        "iterations": iterations,
        "metadata_only": metadata_only,
        "summary": {
            "kernel_count": len(entries),
            "passed": passed,
            "failed": failed,
            "all_passed": failed == 0,
        },
        "kernels": entries,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print(
        "benchmark={benchmark} package={package} kernels={kernel_count} passed={passed} failed={failed}".format(
            benchmark=payload["benchmark"],
            package=payload["package"],
            kernel_count=summary["kernel_count"],
            passed=summary["passed"],
            failed=summary["failed"],
        )
    )
    for entry in payload["kernels"]:
        if entry["status"] == "passed":
            print(
                "{name}: compile_median={compile_ms_median:.3f} ms status=passed".format(**entry)
            )
        else:
            detail = f" error={entry['error']}" if "error" in entry else ""
            print(f"{entry['name']}: status={entry['status']}{detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark exported CUDA kernel compile latency.")
    parser.add_argument("--kernel", action="append", dest="kernels", help="Kernel name to benchmark.")
    parser.add_argument("--arch", help="NVRTC GPU architecture, for example sm_100a.")
    parser.add_argument("--iterations", type=int, default=1, help="Compile iterations per kernel.")
    parser.add_argument("--metadata-only", action="store_true", help="Do not compile; emit benchmark schema.")
    parser.add_argument("--json", type=Path, help="Write benchmark results as JSON.")
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    payload = run_benchmark(
        kernels=args.kernels,
        arch=args.arch,
        iterations=args.iterations,
        metadata_only=args.metadata_only,
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(payload)
    return 0 if payload["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

