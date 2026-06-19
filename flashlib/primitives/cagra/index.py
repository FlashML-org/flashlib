"""``CagraIndex`` -- the in-memory container for a built CAGRA graph index.

Kept in its own module (no Triton import) so both the Triton builder
(:mod:`flashlib.primitives.cagra.triton`) and the pure-torch reference
(:mod:`flashlib.primitives.cagra.torch_fallback`) can construct/consume
it without an import cycle through
:mod:`flashlib.primitives.cagra.impl`.

Layout
------
CAGRA is a *graph* ANN method (loosely NSG-style, GPU-native). Unlike
IVF it has no inverted lists: the index is just

* ``dataset`` -- the ``(N, D)`` full-precision vectors (kept on device;
  the greedy traversal recomputes true squared-L2 distances against the
  neighbours it visits, so the raw vectors must be retained), and
* ``graph`` -- the optimized fixed-out-degree adjacency
  ``(N, graph_degree)`` int32, where ``graph[i]`` lists node ``i``'s
  out-neighbours (already in the dataset's own row-id space, so search
  results need no id remap).

The graph is produced by building an initial kNN graph of degree
``intermediate_degree`` then pruning it (rank-based detour removal +
reverse-edge addition) down to ``graph_degree`` -- see
:mod:`flashlib.primitives.cagra.triton.optimize`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# A connected graph is reachable from a handful of entry nodes, so the search
# only needs a few random seeds (the rest of the buffer fills via traversal);
# seeding the whole itopk buffer just reads ITOPK far-away rows up front (pure
# DRAM cost) that get evicted almost immediately. cuVS likewise starts from a
# small random sampling. Measured +6-7% QPS at equal recall on SIFT1M.
_CONNECTED_SEEDS = 32


def default_random_seeds(num_random_seeds: int, n_components: int,
                         buf: int, N: int) -> int:
    """Resolve the auto (``<= 0``) random-seed count from graph connectivity.

    * explicit ``num_random_seeds > 0`` -> honoured (clamped to ``buf``/``N``).
    * connected graph (``n_components <= 1``) -> a small fixed seed set.
    * disconnected graph (``n_components > 1``) -> seed the whole buffer
      (``min(buf, N)``): the traversal cannot cross between components, so
      recall needs ~one seed per component and *scales* with the buffer.

    ``buf`` is the effective buffer width (CuteDSL/Triton ``ITOPK``) or the
    candidate budget ``M`` (torch). Shared by all backends so seeding -- and
    therefore the result set -- stays identical for the parity tests.
    """
    if int(num_random_seeds) > 0:
        return max(1, min(int(num_random_seeds), buf, N))
    if int(n_components) > 1:
        return max(1, min(buf, N))
    return max(1, min(_CONNECTED_SEEDS, buf, N))


@dataclass
class CagraIndex:
    """A built CAGRA graph index.

    Attributes:
        dataset: ``(N, D)`` full-precision database vectors (fp32/fp16),
            cell order = original caller row ids.
        graph: ``(N, graph_degree)`` int32 adjacency -- ``graph[i, j]`` is
            the ``j``-th out-neighbour (row id) of node ``i``.
        graph_degree: fixed out-degree of the optimized graph.
        intermediate_degree: degree of the initial kNN graph before
            pruning (``>= graph_degree``; recorded for diagnostics).
        metric: distance metric (only ``"l2"`` supported).
        D: feature dimension of ``dataset``.
        Dp: working dimension used by the kernels (``== D``; the search
            kernel masks the trailing dims, so no zero-padding is needed).
        build_algo: how the initial kNN graph was built
            (``"bruteforce"`` | ``"ivf_pq"``), recorded for diagnostics.
        n_components: number of weakly-connected components in the graph
            (``1`` == fully connected). Governs the search's automatic
            random-seed count: a connected graph is reachable from a few
            entry nodes, but a disconnected graph (e.g. well-separated
            clusters) needs ~one seed per component or the traversal can
            never cross into the query's component. ``0`` == not computed.
    """

    dataset: torch.Tensor
    graph: torch.Tensor
    graph_degree: int
    intermediate_degree: int
    metric: str
    D: int
    Dp: int
    build_algo: str = "auto"
    n_components: int = 1

    @property
    def N(self) -> int:
        return int(self.dataset.shape[0])

    @property
    def device(self) -> torch.device:
        return self.dataset.device

    @property
    def dtype(self) -> torch.dtype:
        """Working dtype of the stored vectors."""
        return self.dataset.dtype

    def graph_size_bytes(self) -> int:
        """Adjacency storage in bytes (``N * graph_degree * 4``)."""
        return int(self.graph.numel() * self.graph.element_size())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"CagraIndex(N={self.N}, D={self.D}, graph_degree={self.graph_degree}, "
            f"intermediate_degree={self.intermediate_degree}, "
            f"build_algo={self.build_algo!r}, metric={self.metric!r}, "
            f"dtype={self.dtype}, device={self.device})"
        )


__all__ = ["CagraIndex", "default_random_seeds"]
