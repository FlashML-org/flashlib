# Cake standalone KNN-search export

This directory vendors the source-only standalone export of Cake's
`knn_search` dispatcher from Cake MR 405 at commit
`9f1df8d0def3d2995a81a1279761e52f32427120`. The generated package contains
the CUDA device sources, five generated C++ dispatch families, runtime-free
`tvm-ffi` loaders, and a thin Python interface. It does not import or require
Loom at build or run time.

The export targets Blackwell datacenter GPUs (`sm_100a` and `sm_103a`). Build
the B200 binaries from source with:

```bash
cd cake_exports/knn_search/generated
python build.py --arch sm_100a --domains knn_search
python -m knn_search.demo
```

Build requirements are Python 3.10+, a CUDA-enabled PyTorch, `tvm_ffi`, CUDA
12.8 or newer, and `nvcc` on `PATH`. The public generated API is
`knn_search.interface`; its tensor and scalar ABI is documented by
`TENSOR_KEYS` and `SCALAR_NAMES` in that module.

## Benchmark contract

The recorded denominator is the full 198-shape union from Cake's KNN-search
benchmark registry. The baseline is `flashlib.flash_knn` from this
repository's main commit `711204f32871af4aeb3ef7ed952cb5eb74c57f46`.

The reportable comparison uses `loom.bench.bench_gpu_time_warm` solely as the
CUPTI measurement/reference harness. Candidate execution itself remains
standalone. Run the harness from a checkout of the Cake commit above:

```bash
cd /path/to/cake
python /path/to/flashlib/cake_exports/knn_search/benchmarks/benchmark.py \
  --domain knn_search \
  --output-root /tmp/knn-search-results \
  --reuse-export-root /path/to/flashlib/cake_exports/knn_search/generated \
  --flashlib-root /path/to/flashlib
```

Both lanes accept raw query/database inputs and allocate default outputs inside
the timed call. GPU span is the primary metric; synchronized host E2E is
retained as a diagnostic. See `RESULTS.md` and `BENCHMARK_RESULTS.json` for the
measured B200 results.
