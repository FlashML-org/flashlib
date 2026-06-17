# FlashLib

[![Slack](https://img.shields.io/badge/Slack-join%20the%20community-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/flashml/shared_invite/zt-3zpdh5j10-9dwTXrgLiqpVxizhA9KVbA)
[![Discord](https://img.shields.io/badge/Discord-join%20the%20community-5865F2?logo=discord&logoColor=white)](https://discord.gg/ce5Xa5pf)

A GPU library for classical machine-learning operators — `kmeans`, `knn`,
`ivf-flat`, `pca`, `svd`, `dbscan`, `hdbscan`, `umap`, `t-sne`, regression,
GEMM, and more — built on Triton and CuteDSL.

See [the blog post](https://flashml-org.github.io/) for motivation, design,
and benchmarks.

## Installation

Install with `pip`:

```bash
pip install flashlib
```

From source:

```bash
git clone https://github.com/FlashML-org/flashlib.git
cd flashlib
pip install -e .
```

## Usage

```python
import torch
from flashlib import flash_kmeans

x = torch.randn(1_000_000, 128, device="cuda", dtype=torch.float32)
labels, centroids, n_iter = flash_kmeans(x, n_clusters=1024, max_iters=20)
```

Every primitive is exposed as a top-level `flash_*` function and as a
sklearn-style class (`KMeans`, `PCA`, `HDBSCAN`, …).

Index-based primitives like IVF-Flat and IVF-PQ (GPU approximate nearest
neighbours) build an index once and query it many times:

```python
import torch
from flashlib import IVFFlat

db = torch.randn(1_000_000, 128, device="cuda")
queries = torch.randn(10_000, 128, device="cuda")

index = IVFFlat(nlist=1024, nprobe=16).fit(db)
distances, indices = index.kneighbors(queries, n_neighbors=10)  # squared L2
```

`nprobe` is the recall knob: at a fixed `(nlist, nprobe)` the probed
candidate set — and thus recall — matches a reference IVF-Flat (FAISS /
cuVS), so raising it trades speed for recall without changing the kernel.

For billion-scale corpora, `IVFPQ` adds product-quantization compression:
each vector is stored as `m` 1-byte codes (8–32× smaller than fp32):

```python
from flashlib import IVFPQ

# 128-dim fp32 (512 B/vec) -> m=16 PQ codes (16 B/vec) = 32x compression
index = IVFPQ(nlist=1024, m=16, nprobe=16).fit(db)
distances, indices = index.kneighbors(queries, n_neighbors=10)  # ADC squared L2
print(index.compression_ratio)  # 32.0
```

`m` (sub-quantizers) is the accuracy/size knob; returned distances are the
asymmetric (ADC) squared-L2 to each candidate's PQ reconstruction.

For batched search the fine-scan groups queries by the list they probe,
**decodes** each list's PQ codes back to sub-vectors once per query tile,
and scores them with a tensor-core cross term — ADC as a GEMM, with **no
lookup table** to build or blow up with `nprobe` (an oversampled exact
re-rank keeps distances ADC-exact). On SIFT1M (H100) this runs **1.2–2.4×
faster than cuVS** IVF-PQ; tiny query batches fall back to a fused on-chip
ADC lookup table. Same no-materialization spirit as the rest of flashlib —
no `(queries × candidates)` matrix ever exists.

### Informative API

The `flashlib.info` submodule predicts runtime, FLOPs, and HBM bytes for any
primitive in ~5&nbsp;µs on pure CPU — useful for budgeting a pipeline before
launching it, and small enough for an LLM agent to call in a GPU-less
environment. It does not import torch, triton, or cutlass.

```python
import flashlib.info as info

est = info.estimate("kmeans",
                    shape=(100_000, 64),
                    params={"K": 256, "max_iters": 20},
                    device="H200")
print(est.summary_line())
```

See the blog post for the full API, the tolerance-driven dispatch, and
per-primitive benchmarks.

## Coverage

The current release ships **17 high-level primitives** across the following families:

| family         | primitives                                                                       |
| -------------- | -------------------------------------------------------------------------------- |
| Clustering     | `flash_kmeans`, `flash_dbscan`, `flash_hdbscan`, `flash_spectral_clustering`     |
| Nearest nbrs   | `flash_knn`, `flash_ivf_flat` (IVF-Flat ANN), `flash_ivf_pq` (IVF-PQ ANN)        |
| Decomposition  | `flash_pca`, `flash_truncated_svd`                                               |
| Manifold       | `flash_umap`, `flash_tsne`                                                       |
| Regression     | `flash_linear_regression`, `flash_ridge`, `flash_logistic_regression`            |
| Classification | `flash_multinomial_nb`, `flash_random_forest`                                    |
| Preprocessing  | `flash_standard_scaler`                                                          |

Plus low-level linear-algebra primitives (`cov_gemm`, `gram_gemm`, `ab_gemm`,
`eigh`, `polar`, `msign`, `cholqr2`, `split_basis`) and a Pareto-frontier set
of multi-precision GEMM variants (`gemm`, `gemm_tf32`, `gemm_3xtf32`,
`gemm_bf16`, `gemm_fp16`, `gemm_fp16_x9`, `gemm_fp16_x3_kahan`,
`gemm_ozaki2_int8`, …).

## Citation

```bibtex
@misc{yang2026flashlib,
  title  = {FlashLib: Bringing Flash Magic to Classical Machine Learning Operators},
  author = {Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Mang, Qiuyang and
            Wang, Zhe and Sun, Shanlin and Keutzer, Kurt and Gonzalez, Joseph E. and
            Han, Song and Xu, Chenfeng and Stoica, Ion},
  year   = {2026},
  url    = {https://flashml-org.github.io/},
}
```

## License

[Apache License 2.0](LICENSE).
