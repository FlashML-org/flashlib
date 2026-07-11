# B200 KMeans FP16/FP32 random validation

Seed `20260721`.

| dtype | region | B | N | D | K | heuristic | oracle | regret |
|---|---|---:|---:|---:|---:|---|---|---:|
| float16 | small | 4 | 22784 | 128 | 9728 | BN128_BK64_W4_S2 | BN128_BK128_W4_S1 | 1.043x |
| float16 | small | 2 | 67584 | 128 | 512 | BN128_BK64_W4_S2 | BN128_BK128_W4_S1 | 1.019x |
| float16 | small | 4 | 36352 | 256 | 768 | BN128_BK64_W8_S1 | BN128_BK64_W8_S1 | 1.000x |
| float16 | small | 1 | 281600 | 128 | 6400 | BN128_BK64_W4_S2 | BN128_BK64_W4_S2 | 1.000x |
| float16 | wide | 4 | 35584 | 768 | 4096 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x |
| float16 | wide | 4 | 42752 | 1536 | 11776 | BN64_BK128_BD64_W4_S4 | BN64_BK128_BD64_W4_S4 | 1.000x |
| float16 | wide | 1 | 73472 | 1024 | 3328 | BN128_BK128_BD64_W4_S4 | BN128_BK128_BD64_W4_S4 | 1.000x |
| float16 | wide | 4 | 21248 | 1536 | 16128 | BN64_BK128_BD64_W4_S4 | BN64_BK128_BD64_W4_S4 | 1.000x |
| float32 | small | 2 | 84352 | 128 | 4096 | BN128_BK64_W8_S1 | BN128_BK64_W8_S1 | 1.000x |
| float32 | wide | 4 | 26368 | 256 | 2304 | BN128_BK128_BD32_W4_S4 | BN128_BK128_BD32_W4_S2 | 1.025x |
| float32 | small | 2 | 85504 | 64 | 1024 | BN128_BK64_W4_S2 | BN128_BK32_W4_S1 | 1.033x |
| float32 | wide | 4 | 23168 | 256 | 12544 | BN128_BK128_BD32_W4_S4 | BN64_BK128_BD32_W4_S2 | 1.003x |
| float32 | wide | 1 | 223360 | 1024 | 10240 | BN64_BK128_BD32_W4_S4 | BN64_BK128_BD32_W4_S4 | 1.000x |
| float32 | wide | 1 | 175872 | 512 | 1536 | BN128_BK128_BD32_W4_S4 | BN128_BK128_BD32_W4_S4 | 1.000x |
| float32 | wide | 1 | 87296 | 384 | 2048 | BN128_BK128_BD32_W4_S4 | BN128_BK128_BD32_W4_S4 | 1.000x |
| float32 | wide | 4 | 63488 | 2048 | 1280 | BN64_BK128_BD64_W4_S4 | BN64_BK128_BD64_W4_S4 | 1.000x |

geomean **1.008x**, p95 **1.035x**, max **1.043x**, correctness **1.000**.
