# Cake standalone Flash K-Means export

This directory vendors the source-only standalone export of Cake's
`flash_kmeans` dispatcher from Cake main (MR 417 merge) at commit
`7a7cebd73ee298188e1ed3b6d6d8d7e43dbf5f04`. The generated package contains
the CUDA device sources, one generated C++ dispatch translation unit, a
runtime-free `tvm-ffi` loader, and a thin Python interface. It does not import
or require Loom at build or run time.

New in this refresh: **materialized padding is fully eliminated** — every
gap/off-menu D (including the D144/176, D160, D80/96, and tiny-D bands that
v10 still packed into scratch buffers) loads through fused TMA out-of-bounds
zero-fill, so no shape runs a padding pass and the raw compiled `run` ABI
carries three workspace slots instead of five (the public `interface.run`
signature is unchanged). The package bundles an auxiliary rowwise
squared-norm dispatch family — passing `x_sq` or `c_sq` as `None` to
`interface.run` computes the norms in-package on the same stream (k-means
iterations recompute `c_sq` every round), and `interface.compute_norms(t)`
exposes the kernel directly. The compiled dispatch also carries the merged
hot-path host rework (workspace arena, process-invariant caches, explicit
per-call CUDA-stream ABI).

The export targets Blackwell datacenter GPUs (`sm_100a` and `sm_103a`). Build
the B200 binary from source with:

```bash
cd cake_exports/kmeans/generated
python build.py --arch sm_100a --domains flash_kmeans
python -m flash_kmeans.demo
```

Build requirements are Python 3.10+, a CUDA-enabled PyTorch, `tvm_ffi`, CUDA
12.8 or newer, and `nvcc` on `PATH`. The public generated API is
`flash_kmeans.interface`; its tensor and scalar ABI is documented by
`TENSOR_KEYS` and `SCALAR_NAMES` in that module.

## Benchmark contract

The recorded denominator is the complete 124-shape Cake Flash K-Means
contract. The baseline is the exact Triton source from flashlib-cake commit
`07cf2a27928aacf6790c950a265d8b8dc83c87cf`; its vendored source has SHA-256
`b451fbba737458dc235bff2af026c666e2782e05e84b530dbcf3d2e54d436596`.

The reportable comparison uses `loom.bench.bench_gpu_time_warm` solely as the
CUPTI measurement/reference harness. Candidate execution itself remains
standalone. Run the harness from a checkout of the Cake commit above:

```bash
cd /path/to/cake
python /path/to/flashlib/cake_exports/kmeans/benchmarks/benchmark.py \
  --domain flash_kmeans \
  --output-root /tmp/kmeans-results \
  --reuse-export-root /path/to/flashlib/cake_exports/kmeans/generated \
  --kmeans-baseline-root /path/to/flashlib/cake_exports/kmeans/benchmarks
```

Both raw-operator lanes include the same Torch row-norm expression inside the
timed call. The report also includes an assignment-only lane with precomputed
norms. GPU span is the primary metric; synchronized host E2E is retained as a
diagnostic. See `RESULTS.md` and `BENCHMARK_RESULTS.json` for the measured B200
results.
