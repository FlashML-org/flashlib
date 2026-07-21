# Cake standalone Flash K-Means export

This directory vendors the source-only standalone export of Cake's
`flash_kmeans` dispatcher from Cake MR 405 at commit
`9f1df8d0def3d2995a81a1279761e52f32427120`. The generated package contains
the CUDA device sources, one generated C++ dispatch translation unit, a
runtime-free `tvm-ffi` loader, and a thin Python interface. It does not import
or require Loom at build or run time.

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
