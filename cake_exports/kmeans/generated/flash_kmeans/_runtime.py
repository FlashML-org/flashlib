"""Vendored by tools/export-generated-programs (runtime helper).

Provenance: loom @ 7a7cebd73ee298188e1ed3b6d6d8d7e43dbf5f04. detect_gpu_arch, ported from loom.runtime.compiler.detect_gpu_arch.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

import functools
import os


def _arch_flag_for_cc(major: int, minor: int) -> str:
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


@functools.cache
def get_device_capability(device_id: int) -> tuple[int, int]:
    """Compute capability for one device, queried once per process."""
    import torch

    return tuple(torch.cuda.get_device_capability(device_id))


def detect_gpu_arch() -> str:
    """Loom arch name (sm_100a, sm_103a, ...) for the current CUDA device.

    Honours ``LOOM_EXPORTED_FORCE_ARCH`` / ``LOOM_FORCE_ARCH``. The
    capability probe is cached per device id — a hot call costs one
    ``current_device()`` plus a dict hit, never a driver round-trip.
    Falls back to ``sm_100a`` (the family min arch) when no device is
    reachable, matching the loom runtime's detection contract.
    """
    forced = os.environ.get("LOOM_EXPORTED_FORCE_ARCH") or os.environ.get("LOOM_FORCE_ARCH")
    if forced:
        return forced
    try:
        import torch

        major, minor = get_device_capability(int(torch.cuda.current_device()))
        return _arch_flag_for_cc(major, minor)
    except Exception:
        return "sm_100a"
