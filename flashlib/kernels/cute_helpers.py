"""Small CuTeDSL utilities shared across primitives, kernels, and linalg.

These are *general* CuTeDSL Python helpers (dlpack conversion, JIT cache,
stream wrapping) — used by primitives' ``cutedsl/`` backends and by the
linalg CuTeDSL kernels. They live at the top of ``kernels`` because they
are cross-cutting and not tied to a specific op or DSL surface.

All public functions are pure-Python / lazy in CUTLASS imports so the module
is safe to import in environments where ``cutlass`` is not installed (e.g.
CPU-only CI). The functions raise informative errors at call time if CUTLASS
is missing.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any


def is_cutedsl_available() -> bool:
    """Return True iff the NVIDIA CUTLASS Python DSL is importable."""
    from flashlib._optional import has_cutedsl

    return has_cutedsl()


def torch_to_cute(t, dtype, leading_dim: int):
    """Wrap a torch tensor as a CuTe dynamic-layout tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor (any rank ≥ 2). Caller is responsible for the trailing
        ``L=1`` dimension if the kernel expects a 3-D layout.
    dtype : cutlass.Numeric subclass (e.g. ``cutlass.BFloat16``).
    leading_dim : int
        Which axis is the leading (contiguous) dimension. Matches the
        ``mark_layout_dynamic`` semantics in the CUTLASS DSL.
    """
    import cutlass  # noqa: F401  (errors out clearly if missing)
    from cutlass.cute.runtime import from_dlpack

    mt = from_dlpack(t, assumed_align=16)
    mt.element_type = dtype
    return mt.mark_layout_dynamic(leading_dim=leading_dim)


@lru_cache(maxsize=None)
def cute_compile_cached(kernel_factory, key, *args):
    """LRU-cached ``cute.compile`` wrapper.

    ``kernel_factory`` is a no-arg callable returning a fresh CuTe kernel
    object; ``key`` is a hashable shape/config tuple used as the cache key.
    The remaining ``*args`` are positional stub tensors the JIT needs.
    """
    import cutlass.cute as cute  # noqa: F401

    kernel = kernel_factory()
    return cute.compile(kernel, *args)


def make_torch_stream() -> Any:
    """Return a ``cuda.bindings.driver.CUstream`` for the current torch stream."""
    import cuda.bindings.driver as cuda
    import torch

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
