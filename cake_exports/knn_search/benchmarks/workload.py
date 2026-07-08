"""Workload adapter for semantic correctness and per-shape performance.

Replace this template or pass ``--benchmark-adapter`` to the Cake exporter.
The exported repository never needs Weave IR: this module calls its public
Python API and provides an independent reference implementation.
"""

from __future__ import annotations

from typing import Any

CONFIGURED = False
SHAPES: list[dict[str, Any]] = []


def make_case(package: Any, shape: dict[str, Any]) -> dict[str, Any]:
    """Return run/reference/compare callables and optional work estimates.

    Required keys in the returned mapping:
      run: zero-argument callable invoking the exported semantic API
      reference: zero-argument callable computing independent expected output
      compare: callable(actual, expected) returning bool or {"passed": bool, ...}

    Optional keys:
      flops: useful floating-point operations for TFLOPS reporting
      bytes: useful bytes moved for GB/s reporting
      metrics: static metadata copied into the result
    """
    raise NotImplementedError("configure benchmarks/workload.py before running shape benchmarks")
