"""Vendored by tools/export-generated-programs (runtime helper).

Provenance: loom @ unknown. detect_gpu_arch, ported from loom.runtime.compiler.detect_gpu_arch.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

import os


def _arch_flag_for_cc(major: int, minor: int) -> str:
    sm = int(major) * 10 + int(minor)
    return f"sm_{sm}a" if sm >= 90 else f"sm_{sm}"


def detect_gpu_arch() -> str:
    """Detect the attached CUDA device's loom arch name (sm_100a, sm_103a, ...).

    Honours ``LOOM_EXPORTED_FORCE_ARCH`` / ``LOOM_FORCE_ARCH`` for headless
    builds.  Falls back to ``sm_100a`` (the family min arch) when no device is
    reachable, matching the loom runtime's detection contract.
    """
    forced = os.environ.get("LOOM_EXPORTED_FORCE_ARCH") or os.environ.get("LOOM_FORCE_ARCH")
    if forced:
        return forced
    try:
        from cuda.bindings import driver
    except Exception:
        return "sm_100a"
    try:
        (err,) = driver.cuInit(0)
        if err != 0:
            return "sm_100a"
        err, dev = driver.cuDeviceGet(0)
        if err != 0:
            return "sm_100a"
        err, major = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev
        )
        err2, minor = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev
        )
        if err != 0 or err2 != 0:
            return "sm_100a"
        return _arch_flag_for_cc(int(major), int(minor))
    except Exception:
        return "sm_100a"
