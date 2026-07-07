"""Micro-benchmark: Trainium (NKI) flash-kmeans assign bf16 utilization.

Measures the sustained bf16 PE-array throughput of the standalone flash
assign kernel (the FLOP-dominant Lloyd step) via ``nki.benchmark`` and
reports achieved TFLOPs = ``2 * N * K * D / latency`` against the
single-NeuronCore bf16 peak.

Run on a Trainium instance inside the Neuron venv, e.g.::

    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    python benchmarks/micro/bench_kmeans_trainium.py

The assign kernel is compiled once per shape by ``nki.benchmark`` (warmup
+ timed iters), so this reports kernel time without the per-call
baremetal recompile that the end-to-end Lloyd loop pays.

Baseline vs torch-on-Trainium
-----------------------------
``torch_xla_materialize_assign_ms`` times the *naive* assign that
materialises the full ``(N, K)`` distance matrix in device HBM, run on the
same NeuronCore via torch-neuronx / XLA. This is the true apples-to-apples
"flash vs materialise" comparison (same hardware). Measured (assign step,
median warm iter):

    Shape (N, D, K)      | torch-xla (materialise) | NKI (flash) | speedup
    ---------------------+-------------------------+-------------+--------
    262144 x 128 x 4096  |        59.9 ms          |   31.3 ms   |  1.9x
    262144 x 256 x 4096  |        43.4 ms          |   31.4 ms   |  1.4x
    131072 x 512 x 4096  |        32.5 ms          |   15.8 ms   |  2.1x
    262144 x 256 x 16384 |       206.5 ms          |  125.1 ms   |  1.7x

Beyond the ~1.4-2.1x speedup, the flash kernel is ``O(N + K)`` in device
memory vs the materialise path's ``O(N * K)`` -- e.g. N=1M, K=16384 is a
64 GB distance matrix that the naive path cannot hold but flash streams.

Multi-core (more compute)
-------------------------
Sharding the point (M) axis across NeuronCores (``make_assign_kernel(nc)``
launched with ``[nl.nc(nc)]``) is embarrassingly parallel -- no cross-core
reduction. Measured device latency (NCL(50), neuron-bench) at
N=262144, D=256, K=4096:

    single core (nc=1)  : 31.4 ms
    2 cores    (nc=2)   : 15.7 ms   -> 2.0x

trn2.3xlarge exposes 2 logical NeuronCores (LNC=2), so 2x is the ceiling
here; the same code scales further on larger trn2 instances. Note:
intra-kernel tweaks (swapping the argmax op to Vector-only or manual
PSUM double-buffering) did *not* help -- the epilogue is element-throughput
bound across the Vector+GpSimd engines and the compiler already overlaps
them, so "more cores" is the effective lever, not "different ops".

NOTE: ``nki.benchmark`` spawns ``neuron-bench --fixed-nc-count 1`` and
cannot benchmark a multi-core NEFF (or share the device with an in-process
torch-xla runtime). To time multi-core, save the NEFF
(``nki.baremetal(save_neff_name=...)(make_assign_kernel(nc))[nl.nc(nc)]``)
and run ``neuron-bench exec --fixed-nc-count <nc> ... file.neff``. Run the
torch-xla baseline in a *separate* process from the NKI sweep.

On Amazon Linux 2023, torch-xla needs the one-line ELF fix in
``tools/neuron/patch_torch_xla_glibc.py`` first (glibc 2.34 vs the wheel's
``hypot@GLIBC_2.35``).
"""
from __future__ import annotations

import numpy as np

# Achievable single-NeuronCore bf16 matmul throughput measured on trn2 with
# both operands resident (128x128x512 tile, 2048 reuses): ~75 TFLOPs. Used
# only to report a "% of achievable peak"; re-measure per instance with the
# `compute-bound bf16 matmul peak` probe if needed.
_TRN2_CORE_BF16_TFLOPS = 75.0

_SHAPES = [
    # (N, D, K)
    (262144, 64, 1024),
    (262144, 128, 4096),
    (262144, 256, 4096),
    (1048576, 128, 4096),
    (262144, 256, 16384),
    (131072, 512, 4096),
]


def _bench_one(N, D, K, warmup=5, iters=20):
    import ml_dtypes  # noqa: F401
    import neuronxcc.nki as nki

    from flashlib.primitives.kmeans.trainium.assign_kernel import _flash_assign_kernel
    from flashlib.primitives.kmeans.trainium.assign import prep_assign

    rng = np.random.default_rng(0)
    x = rng.standard_normal((N, D), dtype=np.float32)
    c = rng.standard_normal((K, D), dtype=np.float32)
    xT, cT, csq = prep_assign(x, c)

    kernel = nki.benchmark(warmup=warmup, iters=iters)(_flash_assign_kernel)
    kernel(xT, cT, csq)
    lat = kernel.benchmark_result.nc_latency
    p50_us = float(lat.get_latency_percentile(50))

    flops = 2.0 * N * K * D
    tflops = flops / (p50_us * 1e-6) / 1e12
    pct = 100.0 * tflops / _TRN2_CORE_BF16_TFLOPS
    return p50_us, tflops, pct


def torch_xla_materialize_assign_ms(N, D, K, warmup=3, reps=10):
    """Naive torch assign (materialise the (N,K) distance matrix) on Trainium.

    Runs on the NeuronCore via torch-neuronx / XLA -- the same-hardware
    baseline for the flash kernel. ``.item()`` forces full execution so XLA
    cannot DCE the unused output. Run in a separate process from the NKI
    sweep (device sharing conflict). Requires the AL2023 torch-xla fix
    (``tools/neuron/patch_torch_xla_glibc.py``).
    """
    import time
    import torch
    import torch_xla.core.xla_model as xm

    dev = xm.xla_device()
    x = torch.randn(N, D, device=dev)
    c = torch.randn(K, D, device=dev)

    def step():
        d = (x * x).sum(1, keepdim=True) + (c * c).sum(1)[None, :] - 2.0 * (x @ c.t())
        return int(d.argmin(1).sum().item())

    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    return (time.perf_counter() - t0) / reps * 1e3


def bench_lloyd_update_ab(N=131072, D=128, K=4096, iters=15):
    """End-to-end A/B of the full Lloyd loop: the transposed one-hot-matmul
    NKI update (``use_nki_update=True``, default, D<=128) vs the ``index_add``
    scatter. Reports total wall (incl. one-time, disk-cached compile) per mode.

    Measured on trn2 (warm Neuron compile cache), N=131072, D=128, K=4096,
    15 iters::

        index_add scatter   : ~11.8 s
        transposed NKI matmul: ~1.1 s   -> ~10x

    The win is the M-step: the scatter is data-dependent (Trainium has no
    hardware sort; ``torch.sort`` is rejected, ``NCC_EVRF029``), while the
    transposed one-hot matmul (x stationary, one-hot the 512-wide moving
    operand -> assign-like matmul-tile count and ~seconds compile) runs on the
    PE array. Requires torch-xla.
    """
    import time
    import numpy as np
    import torch

    from flashlib.primitives.kmeans.trainium.kmeans import trainium_kmeans_Euclid

    x = torch.from_numpy(np.random.default_rng(0).standard_normal((N, D), dtype=np.float32))
    init = x[:K].clone()
    out = {}
    for use_nki in (False, True):
        t0 = time.perf_counter()
        trainium_kmeans_Euclid(x, K, max_iters=iters, init_centroids=init.clone(),
                               use_nki_update=use_nki)
        out[use_nki] = time.perf_counter() - t0
    label = {False: "index_add scatter", True: "transposed NKI matmul"}
    for use_nki in (False, True):
        print(f"  {label[use_nki]:>22}: {out[use_nki]:6.1f} s "
              f"({out[use_nki]/iters*1e3:5.0f} ms/iter incl compile)")
    print(f"  speedup (index_add / NKI): {out[False]/out[True]:.1f}x")
    return out


def main():
    print(f"{'N':>8} {'D':>5} {'K':>7} {'p50(us)':>10} {'TFLOPs':>8} "
          f"{'%peak':>7}")
    for N, D, K in _SHAPES:
        try:
            p50, tflops, pct = _bench_one(N, D, K)
            print(f"{N:>8} {D:>5} {K:>7} {p50:>10.1f} {tflops:>8.1f} {pct:>6.1f}%")
        except Exception as e:  # noqa: BLE001
            print(f"{N:>8} {D:>5} {K:>7}  FAILED: {str(e).splitlines()[0][:60]}")
    print("\nLloyd update A/B (full loop, transposed NKI matmul vs index_add):")
    try:
        bench_lloyd_update_ab()
    except Exception as e:  # noqa: BLE001
        print(f"  update A/B skipped: {str(e).splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
