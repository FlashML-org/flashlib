# End-to-end Lloyd k-means: Blackwell (B200)

GPU: **NVIDIA B200** sm100, torch 2.8.0+cu128. Full Lloyd loop, 10 iters, identical seeded init + Triton centroid update; only the **assign** kernel differs. Median wall-time over the loop (warm).

| N | D | K | triton ms | cutedsl_bw ms | cake ms | bw vs triton | bw vs cake | agree(bw/cake) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2048 | 128 | 4096 | 12.243 | 10.785 | 10.934 | **1.14x** | 1.01x | 1.000/1.000 |
| 4096 | 256 | 4096 | 13.980 | 11.116 | 11.036 | **1.26x** | 0.99x | 1.000/1.000 |
| 16384 | 128 | 256 | 3.653 | 4.124 | 4.217 | **0.89x** | 1.02x | 1.000/1.000 |
| 65536 | 128 | 1024 | 4.080 | 4.184 | 4.209 | **0.98x** | 1.01x | 1.000/1.000 |
| 65536 | 128 | 4096 | 11.416 | 4.219 | 4.118 | **2.71x** | 0.98x | 1.000/1.000 |
| 262144 | 128 | 1024 | 10.394 | 4.232 | 4.077 | **2.46x** | 0.96x | 1.000/1.000 |
| 1048576 | 128 | 1024 | 36.339 | 6.634 | 7.569 | **5.48x** | 1.14x | 0.976/0.951 |
| 65536 | 64 | 1024 | 3.747 | 4.193 | 4.067 | **0.89x** | 0.97x | 1.000/1.000 |
| 262144 | 64 | 4096 | 22.304 | 4.639 | 5.098 | **4.81x** | 1.10x | 1.000/1.000 |
| 65536 | 256 | 1024 | 6.139 | 4.273 | 4.049 | **1.44x** | 0.95x | 1.000/1.000 |
| 65536 | 256 | 4096 | 19.768 | 5.795 | 5.730 | **3.41x** | 0.99x | 1.000/1.000 |

`bw vs triton` > 1 means the new Blackwell assign makes the full clustering faster than the Triton baseline; `bw vs cake` > 1 means the new kernel's loop beats the cake-assign loop. `agree` is the label match-rate vs the Triton loop (1.000 = identical clustering).

Source: `benchmarks/blackwell/bench_kmeans_e2e.py`.
