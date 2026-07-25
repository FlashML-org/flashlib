"""GPU slot cache -- the admission oracle for an out-of-core table.

    CPU table  [num_total,  num_feature]      backing store
    GPU table  [num_cached, num_feature]      the cache

    from flashlib.kernels.slot_cache import lru_ensure

``lru_ensure`` answers one question per call: given the ids this step needs, where does
each of them live in the GPU table, and which rows must be copied in? It never touches
the payload -- ``num_feature`` does not appear anywhere -- so one call serves any row
width, dtype, or number of parallel arrays sharing a slot index.

Admission runs on device: no host sync and fixed shapes, so the look-up -> evict -> fill
-> compute chain is CUDA-graph capturable.

The caller owns the state tensors and passes them in, so a consumer with an existing
cache aliases its own arrays rather than allocating a second set.

Two selection strategies sit behind the one entry point and produce bit-identical
results, so the routing is purely a cost decision; see
:mod:`flashlib.kernels.slot_cache.cost` for the crossover.
"""
from flashlib.kernels.slot_cache import cost
from flashlib.kernels.slot_cache.triton import lru_ensure
from flashlib.kernels.slot_cache.triton.lru_ensure import N_STATS, Stat

__all__ = ["lru_ensure", "Stat", "N_STATS", "cost"]
