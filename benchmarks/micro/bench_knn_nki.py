"""Micro-benchmark: Trainium (NKI) flash-knn kernel.

Measures the standalone NKI top-k kernel (the fused ``nc_matmul`` cross +
``max8``/``nc_match_replace8`` reduction, no HBM ``(Nq, M)`` matrix) via
``nki.benchmark`` and reports achieved TF32 PE throughput
(``2 * Nq * M * D / latency``) for the *build* regime (Q and corpus both
large -- compute bound) and effective HBM GB/s for the *search* regime (small
Q, large corpus -- bandwidth bound).

Run on a Trainium instance inside the Neuron venv::

    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    python benchmarks/micro/bench_knn_nki.py

v1 kernel is single-shot (corpus resident in SBUF), so ``M <= 16384``; the
shapes below stay within that. The kernel compiles once per shape (disk
cached); ``nki.benchmark`` reports warm kernel time.

The ``--xla`` mode measures the full device-resident operator. On this
trn2.3xlarge at ``Q=4096,M=8192,D=128`` it measures about 3.7 ms for ``k=10``
on one LNC2 versus 13.7 ms for the materialising XLA baseline. Four
``NEURON_RT_VISIBLE_CORES=0..3`` workers retain 1.75--2.06 ms/call queued
throughput each (near-linear aggregate scaling); forcing ``.cpu()`` after every
call instead measures host-synchronisation contention, not device throughput.
The fused device epilogue also cuts the former NumPy-round-trip path from
30.7 ms to 4.0 ms p50 for CPU callers at this shape.
"""
from __future__ import annotations

import numpy as np

# Physical NeuronCore-v3 peak denominators.
_TRN2_CORE_TF32_TFLOPS = 47.0
_TRN2_CORE_BW_TBS = 0.3625

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
    import neuronxcc.nki as nki
    from flashlib.primitives.knn.nki.knn_kernel import _knn_kernel

    # Match the driver: fp32 inputs select the higher-precision TF32 matmul
    # path and one exact fp32 row folds -0.5||c||^2.
    Qp, Mp, kpad = _ceil(Nq, _PMAX), _ceil(M, _BK), _ceil(k, 8)
    Dp = _ceil(D + 1, _PMAX)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Nq, D)).astype(np.float32)
    c = rng.standard_normal((M, D)).astype(np.float32)
    qT = np.zeros((Dp, Qp), np.float32)
    qT[:D, :Nq] = q.T
    qT[D, :Nq] = 1.0
    cT = np.zeros((Dp, Mp), np.float32)
    cT[:D, :M] = c.T
    bias = -0.5 * (c * c).sum(1)
    cT[D, :M] = bias
    cT[D, M:] = -1.0e4
    csq = np.zeros((1, Mp), np.float32)   # unused (fold_csq=True)

    kernel = nki.benchmark(warmup=warmup, iters=iters)(_knn_kernel)
    kernel(qT, cT, csq, kpad)
    lat = kernel.benchmark_result.nc_latency
    p50_us = float(lat.get_latency_percentile(50))
    p90_us = float(lat.get_latency_percentile(90))

    flops = 2.0 * Nq * M * D
    tflops = flops / (p50_us * 1e-6) / 1e12
    # Search regime: bytes = corpus stream (M*D) + query load + k idx out.
    gbps = ((Nq + M) * D * 4 + Nq * k * 4) / (p50_us * 1e-6) / 1e9
    return p50_us, p90_us, tflops, gbps


def torch_xla_flash_knn_ms(
    Nq=4096, M=8192, D=128, k=10, n_shards=2, warmup=3, reps=10,
):
    """Warm, device-resident full operator latency (rank + fp32 epilogue)."""
    import time
    import torch
    import torch_xla.core.xla_model as xm
    from flashlib.primitives.knn.nki.knn import _nki_knn_xla

    dev = xm.xla_device()
    q = torch.randn(Nq, D, device=dev)
    c = torch.randn(M, D, device=dev)

    def step():
        return _nki_knn_xla(
            q, c, k, return_distances=True, n_shards=n_shards)

    for _ in range(warmup):
        vals, idx = step()
        vals.cpu()
        idx.cpu()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        vals, idx = step()
        vals.cpu()
        idx.cpu()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def torch_xla_materialize_knn_ms(Nq, M, D, k, warmup=3, reps=10):
    """Same-hardware NAIVE k-NN baseline: materialise the full ``(Nq, M)``
    distance matrix on the NeuronCore (torch-neuronx / XLA) and peel the top-k.

    This is the honest "flash vs materialise" comparison (same device). Two
    things to note about the baseline itself:

    * ``torch.topk`` is **rejected** by the Neuron compiler on Trainium2 (sort
      unsupported, ``NCC_EVRF029``), so the naive path cannot even use the
      obvious top-k -- it must peel with ``k`` rounds of ``argmin`` + an
      iota/equal one-hot mask (sort-free), each round re-reading the full
      matrix from HBM.
    * It materialises an ``Nq*M`` fp32 matrix the flash kernel never allocates.

    Kept entirely on-device (one host sync at the end). Measured (median warm
    iter, trn2, D=128, Nq=4096, M=8192) vs the flash NKI kernel on **all usable
    NeuronCores** (cached ``nki.jit`` + ``nl.nc``, the driver's real path) --
    same process, both use the device's cores, so this is apples-to-apples:

        k     naive (materialise)   full flash operator   speedup
        10        13.7 ms               ~3.7 ms            ~3.7x
        32        36.0 ms               ~6.5 ms            ~5.5x

    The gap grows with k because the naive path re-reads the whole matrix once
    per argmin round while flash peels its score row resident in SBUF. Flash
    also never allocates the ``Nq*M`` matrix. Requires the AL2023 torch-xla fix
    (``tools/neuron/patch_torch_xla_glibc.py``).
    """
    import time
    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    q = torch.randn(Nq, D, device=dev)
    c = torch.randn(M, D, device=dev)
    arange = torch.arange(M, device=dev)[None, :]

    def step():
        d = (q * q).sum(1, keepdim=True) + (c * c).sum(1)[None, :] - 2.0 * (q @ c.t())
        idxs = []
        for _ in range(k):
            idx = d.argmin(1)                              # sort-free (no topk)
            idxs.append(idx)
            d = d + (arange == idx[:, None]).to(d.dtype) * 1e30
        return torch.stack(idxs, 1)

    for _ in range(warmup):
        o = step()
        xm.mark_step()
        o.cpu()
    t0 = time.perf_counter()
    for _ in range(reps):
        o = step()
        xm.mark_step()
        o.cpu()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xla", action="store_true",
        help="benchmark device-resident full operator at LNC1/LNC2")
    args = parser.parse_args()
    if args.xla:
        for ns in (1, 2):
            p50, p90 = torch_xla_flash_knn_ms(n_shards=ns)
            print(f"LNC width {ns}: p50={p50:.3f} ms p90={p90:.3f} ms")
        return

    print(f"{'Nq':>6} {'M':>7} {'D':>5} {'k':>4} {'regime':>7} "
          f"{'p50(us)':>9} {'p90(us)':>9} {'TFLOPs':>8} "
          f"{'%pk':>5} {'GB/s':>7} {'%bw':>5}")
    for Nq, M, D, k, regime in _SHAPES:
        try:
            p50, p90, tflops, gbps = _bench_one(Nq, M, D, k)
            pct = 100.0 * tflops / _TRN2_CORE_TF32_TFLOPS
            pbw = 100.0 * gbps / (_TRN2_CORE_BW_TBS * 1e3)
            print(f"{Nq:>6} {M:>7} {D:>5} {k:>4} {regime:>7} "
                  f"{p50:>9.1f} {p90:>9.1f} {tflops:>8.1f} "
                  f"{pct:>4.0f}% {gbps:>7.1f} {pbw:>4.0f}%")
        except Exception as e:  # noqa: BLE001
            print(f"{Nq:>6} {M:>7} {D:>5} {k:>4} {regime:>7}  FAILED: "
                  f"{str(e).splitlines()[0][:50]}")


if __name__ == "__main__":
    main()
