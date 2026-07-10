"""Hardware fingerprint shared by all `route.py` files and the tuner.

This is the *single* place that turns a CUDA device into a small,
hashable bundle of properties (`HwProps`) that hand-written routing
rules in ``flashlib/.../<op>/route.py`` can branch on.

Routing rules look like::

    from flashlib import _hw

    def _route(*, B, N, M, D, k, hw=None):
        hw = hw or _hw.current()
        if hw.sm_arch >= 90 and N >= 4096 and D >= 256:
            return "cutedsl", "build"
        ...

The same ``hw.device_tag`` value names the per-device subdirectory under
``benchmarks/tune/results/`` so that grid sweeps from different machines
never collide.

Detection is best-effort and fully ``cuda``-optional: on CPU-only boxes
``current()`` returns a sentinel ``HwProps(device_tag="cpu", ...)`` that
all rules can still pattern-match on.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import torch


__all__ = ["HwProps", "current", "device_tag"]


@dataclass(frozen=True)
class HwProps:
    """Lightweight, hashable hardware fingerprint.

    Attributes
    ----------
    device_tag : str
        Stable short string used for filesystem paths and rule branches:
        ``"H100"``, ``"H200"``, ``"A100"``, ``"GB200"``, ``"L40S"``,
        ``"sm89"`` (fallback), ``"cpu"``.
    sm_arch : int
        Compute capability major*10 + minor (e.g. 90 for Hopper, 100
        for Blackwell sm100). 0 on CPU.
    sm_count : int
        Number of streaming multiprocessors.
    l2_bytes : int
        L2 cache size in bytes.
    smem_per_sm_bytes : int
        Maximum opt-in shared memory per SM, in bytes (matters for big
        Triton/CuteDSL tiles).
    total_mem_bytes : int
        Total device memory in bytes.
    name : str
        Raw ``torch.cuda.get_device_name(...)`` for diagnostics. Never
        used for routing — use ``device_tag`` instead.
    """

    device_tag: str
    sm_arch: int
    sm_count: int
    l2_bytes: int
    smem_per_sm_bytes: int
    total_mem_bytes: int
    name: str

    # ------------------------------------------------------------------
    # Convenience predicates so rules read like prose.
    # ------------------------------------------------------------------
    @property
    def is_hopper(self) -> bool:
        """True for sm_arch in [90, 99] (H100/H200/H800/...)."""
        return 90 <= self.sm_arch < 100

    @property
    def is_blackwell(self) -> bool:
        """True for sm_arch >= 100 (B100/B200/GB200/...)."""
        return self.sm_arch >= 100

    @property
    def is_ampere(self) -> bool:
        """True for sm_arch in [80, 89] (A100/A40/A10/...)."""
        return 80 <= self.sm_arch < 90

    @property
    def is_cuda(self) -> bool:
        return self.sm_arch > 0

    @property
    def is_neuron(self) -> bool:
        """True on an AWS Trainium/Inferentia (NeuronDevice) host.

        Used by routing rules to select the ``nki`` backend.
        ``sm_arch`` stays 0 (no CUDA), so ``is_cuda`` is False.
        """
        return self.device_tag.startswith("trn") or self.device_tag.startswith("inf")


# H200 has 60 MB L2; H100 has 50 MB; A100 has 40 MB. We use the device
# *name* prefix to disambiguate inside the same compute capability.
_NAME_TO_TAG: dict[str, str] = {
    # exact prefixes — order matters (longest wins)
    "NVIDIA H200":  "H200",
    "NVIDIA H100":  "H100",
    "NVIDIA H800":  "H800",
    "NVIDIA A100":  "A100",
    "NVIDIA A40":   "A40",
    "NVIDIA A10":   "A10",
    "NVIDIA L40S":  "L40S",
    "NVIDIA L40":   "L40",
    "NVIDIA L4":    "L4",
    "NVIDIA B200":  "B200",
    "NVIDIA B100":  "B100",
    "NVIDIA GB200": "GB200",
    "NVIDIA RTX 4090": "RTX4090",
    "NVIDIA RTX 5090": "RTX5090",
}


def _classify(name: str, sm_arch: int) -> str:
    """Map a raw device name to a stable short tag.

    Falls back to ``smXY`` (e.g. ``sm90``) when the device is not in the
    lookup table — that's still good enough to keep tuner results from
    different chips separated.
    """
    # Longest-prefix match so "NVIDIA H200 NVL" still maps to "H200".
    for prefix, tag in sorted(_NAME_TO_TAG.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return tag
    if sm_arch > 0:
        return f"sm{sm_arch}"
    return "cpu"


def _detect_neuron() -> Optional[HwProps]:
    """Best-effort AWS NeuronDevice fingerprint (no torch/subprocess).

    Detection is a cheap ``/dev/neuron*`` probe. The instance family
    (``trn2`` vs ``trn1`` / ``inf2``) is read from the DMI product name
    when available, else defaults to ``"trn2"`` (the primary target of the
    NKI backends). ``sm_count`` carries the NeuronCore count.
    """
    devs = glob.glob("/dev/neuron*")
    if not devs:
        return None
    tag = "trn2"
    try:
        with open("/sys/devices/virtual/dmi/id/product_name") as f:
            product = f.read().strip().lower()
        for family in ("trn2", "trn1", "inf2", "inf1"):
            if family in product:
                tag = family
                break
    except Exception:
        pass
    return HwProps(
        device_tag=tag, sm_arch=0, sm_count=len(devs),
        l2_bytes=0, smem_per_sm_bytes=0,
        total_mem_bytes=0, name=f"neuron:{tag}",
    )


@lru_cache(maxsize=4)
def current(device: Optional[int] = None) -> HwProps:
    """Return the cached :class:`HwProps` for the given (or default) device.

    The result is cached per ``device`` so repeated calls inside a hot
    routing path are free.
    """
    if not torch.cuda.is_available():
        neuron = _detect_neuron()
        if neuron is not None:
            return neuron
        return HwProps(
            device_tag="cpu", sm_arch=0, sm_count=0,
            l2_bytes=0, smem_per_sm_bytes=0,
            total_mem_bytes=0, name="cpu",
        )

    idx = device if device is not None else torch.cuda.current_device()
    p = torch.cuda.get_device_properties(idx)
    sm_arch = p.major * 10 + p.minor
    name = p.name

    # The exact attribute name has flip-flopped across torch versions
    # (``L2_cache_size`` on current builds, ``l2_cache_size`` on some
    # nightlies). Try every known spelling and fall back to 0.
    l2 = (
        getattr(p, "L2_cache_size", None)
        or getattr(p, "l2_cache_size", None)
        or getattr(p, "L2CacheSize", None)
        or 0
    )
    smem = (
        getattr(p, "shared_memory_per_multiprocessor", None)
        or getattr(p, "shared_memory_per_block_optin", None)
        or 0
    )
    return HwProps(
        device_tag=_classify(name, sm_arch),
        sm_arch=sm_arch,
        sm_count=p.multi_processor_count,
        l2_bytes=int(l2),
        smem_per_sm_bytes=int(smem),
        total_mem_bytes=int(p.total_memory),
        name=name,
    )


def device_tag(device: Optional[int] = None) -> str:
    """Stable short tag for the given device. Used by the tuner to name
    per-device result directories under ``benchmarks/tune/results/``.
    """
    return current(device).device_tag
