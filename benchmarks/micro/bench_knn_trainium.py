"""Micro-benchmark: Trainium (NKI) flash-knn kernel.

Measures the standalone NKI top-k kernel (the fused ``nc_matmul`` cross +
``max8``/``nc_match_replace8`` reduction, no HBM ``(Nq, M)`` matrix) via
``nki.benchmark`` and reports achieved bf16 PE throughput
(``2 * Nq * M * D / latency``) for the *build* regime (Q and corpus both
large -- compute bound) and effective HBM GB/s for the *search* regime (small
Q, large corpus -- bandwidth bound).

Run on a Trainium instance inside the Neuron venv::

    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    python benchmarks/micro/bench_knn_trainium.py

v1 kernel is single-shot (corpus resident in SBUF), so ``M <= 16384``; the
shapes below stay within that. The kernel compiles once per shape (disk
cached); ``nki.benchmark`` reports warm kernel time.
"""
from __future__ import annotations

import numpy as np

# Single-NeuronCore bf16 matmul ceiling (see bench_kmeans_trainium).
_TRN2_CORE_BF16_TFLOPS = 75.0
_TRN2_CORE_BW_TBS = 1.5

_PMAX, _BK = 128, 512


def _ceil(a, b):
    return ((a + b - 1) // b) * b


# (Nq, M, D, k, regime)
_SHAPES = [
    (8192, 16384, 64, 10, "build"),
    (8192, 16384, 128, 32, "build"),
    (4096, 16384, 128, 10, "build"),
    (128, 16384, 64, 10, "search"),
    (256, 8192, 128, 32, "search"),
]


def _bench_one(Nq, M, D, k, warmup=5, iters=20):
    import ml_dtypes  # noqa: F401
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.trainium.knn_kernel import _knn_kernel

    Qp, Mp, Dp, kpad = (_ceil(Nq, _PMAX), _ceil(M, _BK),
                        _ceil(D, _PMAX), _ceil(k, 8))
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Nq, D)).astype(np.float32)
    c = rng.standard_normal((M, D)).astype(np.float32)
    qT = np.zeros((Dp, Qp), np.float32); qT[:D, :Nq] = q.T
    cT = np.zeros((Dp, Mp), np.float32); cT[:D, :M] = c.T
    csq = np.full((1, Mp), 3.0e38, np.float32); csq[0, :M] = (c * c).sum(1)

    kernel = nki.benchmark(warmup=warmup, iters=iters)(_knn_kernel)
    kernel(qT.astype(ml_dtypes.bfloat16), cT.astype(ml_dtypes.bfloat16), csq, kpad)
    p50_us = float(kernel.benchmark_result.nc_latency.get_latency_percentile(50))

    flops = 2.0 * Nq * M * D
    tflops = flops / (p50_us * 1e-6) / 1e12
    # Search regime: bytes = corpus stream (M*D) + query load + k idx out.
    gbps = ((Nq + M) * D * 2 + Nq * k * 4) / (p50_us * 1e-6) / 1e9
    return p50_us, tflops, gbps


def main():
    print(f"{'Nq':>6} {'M':>7} {'D':>5} {'k':>4} {'regime':>7} "
          f"{'p50(us)':>9} {'TFLOPs':>8} {'%pk':>5} {'GB/s':>7} {'%bw':>5}")
    for Nq, M, D, k, regime in _SHAPES:
        try:
            p50, tflops, gbps = _bench_one(Nq, M, D, k)
            pct = 100.0 * tflops / _TRN2_CORE_BF16_TFLOPS
            pbw = 100.0 * gbps / (_TRN2_CORE_BW_TBS * 1e3)
            print(f"{Nq:>6} {M:>7} {D:>5} {k:>4} {regime:>7} "
                  f"{p50:>9.1f} {tflops:>8.1f} {pct:>4.0f}% {gbps:>7.1f} {pbw:>4.0f}%")
        except Exception as e:  # noqa: BLE001
            print(f"{Nq:>6} {M:>7} {D:>5} {k:>4} {regime:>7}  FAILED: "
                  f"{str(e).splitlines()[0][:50]}")


if __name__ == "__main__":
    main()
