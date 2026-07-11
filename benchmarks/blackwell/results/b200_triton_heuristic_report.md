# B200 Triton KMeans heuristic

## Scope and implementation

- Reference tuner: `svg-project/flash-kmeans-tune@6671a162`
- Hardware profile: `flashlib._hw.HwProps.device_tag == "B200"`
- Tuned dtypes: BF16, FP16, FP32
- Batch is not restricted; config selection receives per-batch `N`, matching
  every existing per-GPU heuristic
- No product-name parsing or `allow_b200` API was added

The implementation keeps the file's existing form:

- `_heuristic_euclid_config_{h100,h200,a100,b200}_*` functions
- `_arch_smallD_picker` and `_arch_largeD_picker`
- one `_arch_small_d_max` helper for kernel choice

No dataclass, profile registry, or B200-only policy object is introduced.
Architecture selection never filters on dtype. FP32 and wider dtypes use the
same per-GPU non-half branch followed by the existing byte-aware SMEM fitter.

The selected GPU function defines:

- two-byte dtypes use small-D through D256
- four-byte and wider dtypes use small-D through D128
- larger or non-power-of-two D uses split-D
- other hardware → pre-existing conservative behavior

## Config evidence

Small-D:

- fit/holdout workload grid: 30 shapes
- expanded oracle: 58 configs
- final fit-grid heuristic regret: geomean 1.006x, p95 1.037x,
  maximum 1.046x

Wide-D:

- D768/D1024 fit grid: 16 large-N/K shapes
- fixed holdouts: 6 shapes
- initial focused oracle: 24 configs
- the config-space audit and random validation use 92 configs

The inferred continuous split-D rule is:

- `256 < D < 768`: `BN128/BK128/BD32/W4/S4`
- `D >= 768`: `BN128/BK128/BD64/W4/S4`

It is not a D768/D1024 whitelist.

Direct boundary audits explain the byte-aware limit. At D512, split-D was
1.19–1.43x faster for BF16, 1.18–1.47x for FP16, and 2.96–4.15x for FP32.
At D256, small-D remained 1.04–1.12x faster for BF16 and within 3% for
FP16, while split-D was already 1.22–1.31x faster for FP32. D128 checks
kept small-D 1.12–1.31x ahead for FP32. These are measured per-GPU table
boundaries, not a B200-only dispatch mechanism or dtype eligibility gate.

## Fresh random validation

After the heuristic was fixed, seed `20260713` generated 17 unseen shapes:

- 9 small-D shapes
- 8 wide-D shapes
- B in {1,2,4,8}
- random aligned N/K
- 58-config small-D and 92-config wide-D oracles

Result:

- geomean regret: **1.016x**
- p95 regret: **1.056x**
- maximum regret: **1.061x**
- minimum correctness: **1.0**

## D-axis extrapolation

After the continuous D rule was fixed, seed `20260716` sampled:

- D={320,416,512,704,1152,2560}
- B={1,2,4}
- 92-config oracle per shape

Result:

- geomean regret: **1.009x**
- p95 regret: **1.043x**
- maximum regret: **1.052x**
- minimum correctness: **1.0**

## FP16 / FP32 generalization

Six representative FP16 and six FP32 cells were profiled first to infer
dtype-specific branches. A fresh seed `20260721` then generated 16 unseen
small/wide shapes with B={1,2,4}, each compared with the 58/92-config oracle.

After the final FP16 transition and FP32 D256 route audit:

- geomean regret: **1.008x**
- p95 regret: **1.035x**
- maximum regret: **1.043x**
- tie-aware correctness: **1.0**

A separate four-shape D1152/1280/1408/1792 boundary check for the FP16
transition produced:

- geomean regret: **1.006x**
- maximum regret: **1.025x**
- correctness: **1.0**

FP32 correctness is tie-aware because Triton uses TF32 dot products; exact
index equality can differ for numerically indistinguishable centroids.

## Corrected full Lloyd before/after

The “before” lane launches the former split-D
`BN32/BK32/BD32/W4/S1` kernel directly, without monkeypatching dispatch.

| N | D | K | before ms | after ms | speedup | label match |
|---:|---:|---:|---:|---:|---:|---:|
| 65K | 64 | 256 | 2.113 | 2.416 | 0.87x | 1.0000 |
| 65K | 128 | 4096 | 5.609 | 3.225 | 1.74x | 1.0000 |
| 262K | 256 | 1024 | 8.678 | 2.405 | 3.61x | 1.0000 |
| 1M | 64 | 1024 | 11.124 | 2.930 | 3.80x | 1.0000 |
| 1M | 128 | 4096 | 67.321 | 9.785 | 6.88x | 1.0000 |
| 1M | 256 | 4096 | 120.805 | 17.603 | 6.86x | 1.0000 |
| 262K | 768 | 4096 | 87.662 | 15.138 | 5.79x | 1.0000 |
| 262K | 1024 | 4096 | 115.922 | 17.320 | 6.69x | 1.0000 |
| 1M | 768 | 16384 | 1309.909 | 176.254 | 7.43x | 1.0000 |
| 1M | 1024 | 16384 | 1765.955 | 235.181 | 7.51x | 1.0000 |

The N65K/D64/K256 cell is deliberately retained as a negative boundary.

## Automatic backend routing

The final Triton table was compared with Blackwell CuTeDSL on 18 low-K
shapes. Triton won 11 and CuTeDSL won 7; Triton/CuTeDSL geomean was 0.937x.
Because no simple N/D/K threshold wins every cell, this patch does not alter
the public `flash_kmeans` backend route.

## Reproducible artifacts

- `tune_triton_b200_subset.py`
- `tune_triton_b200_wide.py`
- `validate_triton_b200_random.py`
- `validate_triton_b200_d_extrapolation.py`
- `tune_triton_b200_dtypes.py`
- `validate_triton_b200_dtypes_random.py`
- `validate_triton_b200_fp16_wide_boundary.py`
- `bench_triton_heuristic_lloyd.py`
- `b200_triton_subset_expanded.json`
- `b200_triton_wide_subset.json`
- `b200_random_validation.json`
- `b200_d_extrapolation_validation.json`
- `b200_dtype_subset.json`
- `b200_dtype_random.json`
- `b200_fp16_wide_boundary.json`
- `b200_triton_lloyd.json`

