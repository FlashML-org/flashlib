"""Cake backend for the K-Means primitive.

This package contains source-level CUDA kernels generated from Cake MR149,
plus Python launch wrappers. It intentionally does not depend on Cake, Loom,
or Weave IR at runtime.
"""

from .kernels import KERNELS, ExportedKernel, KernelSpec, get_kernel
from .kmeans import RouteDecision, flash_kmeans_assign, select_flash_kmeans_route
from .lloyd import batch_kmeans_Euclid, flash_kmeans_cake

flash_kmeans_assign_cake = flash_kmeans_assign

__all__ = [
    "KERNELS",
    "KernelSpec",
    "ExportedKernel",
    "get_kernel",
    "RouteDecision",
    "flash_kmeans_assign",
    "flash_kmeans_assign_cake",
    "select_flash_kmeans_route",
    "batch_kmeans_Euclid",
    "flash_kmeans_cake",
]
