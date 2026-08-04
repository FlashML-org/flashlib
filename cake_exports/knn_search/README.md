# Cake standalone KNN-search export

This directory vendors the source-only standalone export of Cake's
`knn_search` dispatcher from the Cake MR 415 head at commit
`ff502f39df09ffdb317efc57ebdac3a668bb3aa4`. The generated package contains
the CUDA device sources, five generated C++ dispatch families, runtime-free
`tvm-ffi` loaders, and a thin Python interface. It does not import or require
Loom at build or run time.

New in this refresh: the compiled dispatch carries the merged hot-path host
rework — per-(device, stream) workspace arena with cached exact-shape views,
process-cached device/family resolution, `-O3` host builds, and an explicit
per-call CUDA-stream ABI with a once-per-process stream-handle lookup. The
kernels and dispatch tables are unchanged; Python host overhead per call
dropped from 141-235us to ~30-90us.

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
retained as a diagnostic. A second, explicitly labeled diagnostic removes only
the exported interface's output allocation; FlashLib has no matching output-
buffer API, so that lane reuses the same-shape raw FlashLib timing and is not an
API-boundary-equivalent comparison. See `RESULTS.md` and
`BENCHMARK_RESULTS.json` for the measured B200 results.
