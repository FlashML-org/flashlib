"""slot_cache triton backend (lru_ensure).

Re-exports the single public wrapper; the ``@triton.jit`` kernels and the three
selection strategies behind them stay private to ``lru_ensure.py``.
"""
from flashlib.kernels.slot_cache.triton.lru_ensure import lru_ensure

__all__ = ["lru_ensure"]
