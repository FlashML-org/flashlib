"""Sweep team_size for the vectorized cutedsl CAGRA search on SIFT1M."""
import numpy as np
import torch

from benchmarks.vs_cuml._common import load_ann_dataset, recall_at_k, time_gpu
from flashlib import flash_cagra_build
from flashlib.primitives.cagra.cutedsl.search_kernel import cagra_search_cutedsl

Xc_np, Xq_np, gt = load_ann_dataset("sift-128-euclidean")
Xc = torch.as_tensor(Xc_np, device="cuda")
Xq = torch.as_tensor(Xq_np, device="cuda")
nq = Xq.shape[0]
k = 10
gt = gt[:, :k]

index = flash_cagra_build(Xc, graph_degree=64, intermediate_degree=128, seed=0)

for itopk in (64, 128, 256):
    for R in (8, 16, 32):
        def _search():
            return cagra_search_cutedsl(index, Xq, k, itopk_size=itopk,
                                        search_width=1, num_random_seeds=R)
        ids = _search()[1].cpu().numpy()
        t = time_gpu(_search, repeat=10, warmup=3)
        qps = nq / (t / 1000.0)
        rec = recall_at_k(ids, gt, k)
        lbl = "full" if R == 0 else str(R)
        print(f"itopk={itopk:4d} R={lbl:>4}  QPS={qps:11,.0f}  recall@{k}={rec:.4f}  ({t:.3f} ms)")
    print()
