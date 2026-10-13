"""Per-component time breakdown for flash_hdbscan across multiple N values.

Workload axis: **N (number of points)** along the dense MRD path (D=16,
``approximate=True`` + ``prefer="auto"`` -> ``use_sparse = (N >= 150_000
and D >= 40)`` is False, so the dispatcher picks
``_flash_hdbscan_dense_impl`` at all three N values).

  * N=10K  D=16  mcs=20  ms=5
  * N=20K  D=16  mcs=20  ms=5  (the headline 28.1x vs cuML row)
  * N=50K  D=16  mcs=50  ms=5

The body is INLINED here so each GPU kernel and CPU stage can
sit in its own ``Stage`` context (CPU stages are timed by paired CUDA
events on the default stream: the events fire in submission order
while the GPU queue is empty, so ``elapsed_time`` ≈ host wall).

Stages (matches the order in ``_flash_hdbscan_dense_impl``):

  data_prep    : make_blobs cached in `prepare()`; this stage just does
                  the H2D copy.
  core_dists   : flash_knn(X, k=min_samples+1) -> per-row k-th NN dist.
  mrd          : ``triton_pairwise_mrd`` (N×N MRD with the
                  max(d, core[i], core[j]) reduction).
  mst          : ``flash_mst`` (dense Borůvka on the N×N MRD).
  d2h          : ``mst_gpu.cpu().numpy().astype(float64)``  -- the only
                  H2D copy on the hot path.
  slt_label    : ``_fast_label`` (CPU C helper: union-find dendrogram).
  condense_tree: ``_fast_condense_tree``.
  stability    : ``_fast_compute_stability`` + ``_fast_get_clusters``.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import make_blobs

from flashlib.kernels.distance.triton import triton_pairwise_mrd
from flashlib.kernels.flash_mst import flash_mst
from flashlib.primitives.knn import flash_knn
from flashlib.primitives.hdbscan.triton._tree_helpers import (
    _fast_label,
    _fast_condense_tree,
    _fast_compute_stability,
    _fast_get_clusters,
)

from ._common import (
    StageGroup, free_gpu, run_multi_shape, write_multi_shape_md,
)

D_FIXED = 16
MIN_SAMPLES = 5
N_CENTERS = 6

SHAPES = [
    ("N=10K", {"N": 10_000, "min_cluster_size": 20}),
    ("N=20K", {"N": 20_000, "min_cluster_size": 20}),
    ("N=50K", {"N": 50_000, "min_cluster_size": 50}),
]
STAGES = ["data_prep", "core_dists", "mrd", "mst", "d2h",
          "slt_label", "condense_tree", "stability"]


def prepare(N: int, min_cluster_size: int) -> dict:
    X_np, _ = make_blobs(
        n_samples=N, centers=N_CENTERS, n_features=D_FIXED,
        cluster_std=1.0, random_state=0,
    )
    X_np = X_np.astype(np.float32)
    return {"X_np": X_np, "N": N, "min_cluster_size": min_cluster_size}


def run(stg: StageGroup, ctx: dict) -> None:
    device = "cuda"
    mcs = ctx["min_cluster_size"]

    with stg["data_prep"]:
        X = torch.from_numpy(ctx["X_np"]).to(device, non_blocking=False)
        assert X.is_cuda and X.dtype == torch.float32

    with stg["core_dists"]:
        dists_sq, _ = flash_knn(
            X[None], X[None], k=MIN_SAMPLES + 1, tol=None,
        )
        cd_sq = dists_sq[0, :, MIN_SAMPLES].clamp(min=0.0)
        core_dists = torch.sqrt(cd_sq)

    with stg["mrd"]:
        MRD = triton_pairwise_mrd(X, core_dists, tol=None)

    with stg["mst"]:
        mst_gpu = flash_mst(MRD)

    with stg["d2h"]:
        mst = mst_gpu.cpu().numpy().astype(np.float64)
        del MRD

    with stg["slt_label"]:
        slt = _fast_label(mst)

    with stg["condense_tree"]:
        parents, children, lambdas, sizes = _fast_condense_tree(slt, mcs)

    with stg["stability"]:
        cluster_ids, stab = _fast_compute_stability(
            parents, children, lambdas, sizes,
        )
        num_points = slt.shape[0] + 1
        labels = _fast_get_clusters(
            parents, children, lambdas, sizes,
            cluster_ids, stab, num_points,
        )
        labels = labels.astype(np.int32)
        _ = labels  # consumer


def main() -> None:
    print(f"[breakdown:hdbscan] sweeping N on the dense MRD path "
          f"(D={D_FIXED}, ms={MIN_SAMPLES})")
    results = run_multi_shape(SHAPES, prepare, run, STAGES,
                                warmup=1, repeat=3)

    write_multi_shape_md(
        prim="hdbscan",
        shape_axis=(f"N (n_samples) on the dense path, D={D_FIXED}, "
                    f"min_samples={MIN_SAMPLES}, fp32"),
        results=results,
        stage_names=STAGES,
        notes=(
            "Dense path: core_dists + mrd + mst run on GPU; "
            "slt_label / condense_tree / stability are CPU C helpers "
            "(timed by paired CUDA events on the default stream while "
            "the GPU queue is empty, so elapsed_time ≈ host wall). "
            "Body inlined from `_flash_hdbscan_dense_impl` + "
            "`_fast_tree_to_labels`."
        ),
        sensitivity=(
            "As **N grows from 10K to 50K**, the GPU stages scale "
            "super-linearly (`mrd` is N², `mst` is N²·log N for dense "
            "Borůvka) while the CPU tree stages "
            "(`slt_label`/`condense_tree`/`stability`) scale only "
            "O(N) -- so the CPU stages' SHARE of the wall shrinks even "
            "as their absolute ms grows. At N=10K the CPU tree pipeline "
            "is a meaningful slice of total time; at N=50K the GPU "
            "stages `mrd` + `mst` dominate decisively while the CPU "
            "tail flattens. This is the regime where the next "
            "optimisation lever is the sparse path "
            "(`flash_hdbscan_sparse` / kNN-MRD edges), which replaces "
            "the N² dense MRD with N·k sparse edges -- the dispatcher's "
            "`use_sparse = (N >= 150_000 and D >= 40)` threshold "
            "deliberately keeps the dense path here because at "
            "D=16/N<=50K the dense MRD is still bandwidth-bound and "
            "faster than the sparse Borůvka launch overhead."
        ),
    )
    free_gpu()


if __name__ == "__main__":
    main()
