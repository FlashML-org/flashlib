# B200 KMeans random-shape validation

Seed `20260713`; shapes were sampled after the heuristic was fixed.

| region | B | N | D | K | heuristic | oracle | regret | correctness |
|---|---:|---:|---:|---:|---|---|---:|---:|
| small | 1 | 146176 | 256 | 8448 | BN128_BK64_W8_S1 | BN128_BK64_W8_S1 | 1.000x | 1.000 |
| small | 1 | 78976 | 256 | 7168 | BN128_BK64_W8_S1 | BN128_BK128_W4_S2 | 1.061x | 1.000 |
| small | 1 | 162432 | 64 | 4096 | BN128_BK32_W4_S2 | BN128_BK32_W4_S2 | 1.000x | 1.000 |
| small | 1 | 432128 | 256 | 35072 | BN128_BK64_W8_S1 | BN128_BK64_W8_S1 | 1.000x | 1.000 |
| small | 1 | 355328 | 128 | 4864 | BN128_BK64_W4_S2 | BN128_BK128_W4_S1 | 1.055x | 1.000 |
| small | 1 | 294016 | 128 | 1792 | BN128_BK128_W4_S1 | BN128_BK128_W4_S1 | 1.000x | 1.000 |
| small | 2 | 147456 | 128 | 6144 | BN128_BK64_W4_S2 | BN128_BK128_W4_S1 | 1.035x | 1.000 |
| small | 4 | 40960 | 256 | 2048 | BN128_BK64_W8_S1 | BN128_BK64_W4_S1 | 1.037x | 1.000 |
| small | 8 | 12288 | 64 | 3072 | BN128_BK32_W4_S4 | BN128_BK64_W4_S3 | 1.018x | 1.000 |
| wide | 1 | 311168 | 1024 | 56320 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x | 1.000 |
| wide | 1 | 74624 | 1024 | 46848 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x | 1.000 |
| wide | 1 | 529280 | 768 | 15616 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD32_W4_S4 | 1.013x | 1.000 |
| wide | 1 | 440192 | 768 | 7680 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x | 1.000 |
| wide | 1 | 262528 | 1024 | 2304 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x | 1.000 |
| wide | 1 | 393984 | 1024 | 12288 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x | 1.000 |
| wide | 2 | 147456 | 768 | 16384 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD32_W4_S4 | 1.048x | 1.000 |
| wide | 4 | 40960 | 1024 | 10240 | BN128_BK128_BD64_W4_S4 | BN64_BK128_BD64_W4_S4 | 1.011x | 1.000 |

geomean regret **1.016x**, p95 **1.056x**, max **1.061x**.
Pass: **True**.
