from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_harness_metadata_mode(tmp_path):
    output = tmp_path / "benchmark_metadata.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_exported_kernels.py",
            "--metadata-only",
            "--json",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "exported_kernel_compile"
    assert payload["metadata_only"] is True
    assert payload["kernels"]
    assert payload["summary"]["kernel_count"] == len(payload["kernels"])


def test_shape_benchmark_metadata_mode(tmp_path):
    output = tmp_path / "shape_benchmark_metadata.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_shapes.py",
            "--metadata-only",
            "--json",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "exported_kernel_shapes"
    assert payload["metadata_only"] is True
    assert payload["timing_backend_requested"] == "cupti"

