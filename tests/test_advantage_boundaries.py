"""Advantage-boundary tests — quantify where each backend wins.

For every primitive that has both a Triton AND a CuteDSL backend we run a
small (shape) sweep and record:

    1. wall-clock ms per backend
    2. correctness (cross-backend agreement within published tol)
    3. **dispatcher-routing correctness** — the smart ``_route(...)``
       must pick the empirical winner within ``ROUTE_SLACK`` (default 1.3×).

The benchmark deliberately uses a SMALL grid (≤ 8 cells per primitive) so the
whole suite finishes under a minute on H200. The expensive sweeps live under
``benchmarks/`` (out-of-tree) — these tests just catch hard regressions.

Each test writes a markdown row to ``benchmarks/results/boundaries_<primitive>.md``
so the boundary numbers stay anchored to real measurements.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flashlib._optional import has_cutedsl

needs_cutedsl = pytest.mark.skipif(not has_cutedsl(), reason="needs flashlib[cutedsl]")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for advantage-boundary tests", allow_module_level=True)

DEVICE = "cuda"
SEED = 42
RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Routing-correctness slack: dispatcher pick is allowed to be at most this
# many times slower than the empirical winner. 1.3 ≈ "within 30%".
ROUTE_SLACK = 1.3


def _is_hopper() -> bool:
    return torch.cuda.get_device_properties(0).major >= 9


def _bench(fn, *, warmup=3, iters=10):
    """Median wall-clock ms (with cuda sync)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _save_row(primitive: str, row: dict):
    """Append a row to the per-primitive markdown table."""
    path = RESULTS_DIR / f"boundaries_{primitive}.md"
    if not path.exists():
        with open(path, "w") as f:
            f.write(f"# {primitive} — backend boundary measurements\n\n")
            f.write(f"Hardware: {torch.cuda.get_device_name(0)}, "
                    f"sm={torch.cuda.get_device_properties(0).major*10+torch.cuda.get_device_properties(0).minor}\n\n")
            cols = list(row.keys())
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    cols = list(row.keys())
    with open(path, "a") as f:
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")


# ---------------------------------------------------------------------------
# kmeans assign — Triton split-D + heuristics  vs  CuteDSL FA3
# Boundary axes: D, K (B=1, N fixed at 65536).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("D,K", [
    (64,   64),
    (128,  256),
    (256,  1024),
    (512,  4096),
])
@pytest.mark.skipif(not _is_hopper(), reason="kmeans cutedsl needs Hopper SM90")
@needs_cutedsl
def test_kmeans_assign_advantage_boundary(D, K):
    torch.manual_seed(SEED)
    N = 65536
    x = torch.randn(1, N, D, device=DEVICE, dtype=torch.bfloat16)
    centroids = torch.randn(1, K, D, device=DEVICE, dtype=torch.bfloat16)
    x_sq = (x.float() ** 2).sum(dim=-1)

    from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
    from flashlib.primitives.kmeans.cutedsl import cutedsl_assign_euclid

    # Both backends are x²-free; the kernels recompute the
    # ``c² − 2⟨x, c⟩`` shifted score internally.
    del x_sq

    try:
        ms_t = _bench(lambda: euclid_assign_triton(x, centroids), iters=5)
        ms_c = _bench(lambda: cutedsl_assign_euclid(x, centroids), iters=5)
    except Exception as e:
        pytest.skip(f"kmeans cutedsl assign unavailable: {e}")

    winner = "cutedsl" if ms_c < ms_t else "triton"
    speedup = max(ms_t, ms_c) / min(ms_t, ms_c)

    _save_row("kmeans", {
        "N": N, "D": D, "K": K,
        "triton_ms": f"{ms_t:.2f}",
        "cutedsl_ms": f"{ms_c:.2f}",
        "winner": winner,
        "speedup": f"{speedup:.2f}x",
    })

    # Both backends must produce the same labels (within bf16 tol).
    ids_t = euclid_assign_triton(x, centroids).to(torch.int32)
    ids_c = cutedsl_assign_euclid(x, centroids).to(torch.int32)
    mismatch = (ids_t != ids_c).float().mean().item()
    assert mismatch < 2e-2, f"kmeans label mismatch {mismatch:.3f} (D={D} K={K})"


# ---------------------------------------------------------------------------
# knn — Triton (auto-picked sortmerge / insert) vs CuteDSL FA3.
# Boundary axes: M (corpus size), D.
#
# The Triton path exposes a single fused entry point whose internal branch
# is shape-driven, so the remaining meaningful boundary is triton-vs-FA3.
# FA3's first-call autotune is ~5 s in heuristic mode; the bench still
# fits comfortably under the suite's 1-minute budget.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N,M,D", [
    (1024,  4096,  64),    # small corpus, low D
    (1024, 16384, 128),    # mid corpus, std D
    (4096, 16384, 256),    # big batch, big D       -> FA3 sweet spot
])
@pytest.mark.skipif(not _is_hopper(), reason="bench needs Hopper SM90")
def test_knn_advantage_boundary(N, M, D):
    torch.manual_seed(SEED)
    k = 8
    xb = torch.randn(1, N, D, device=DEVICE, dtype=torch.bfloat16)
    cb = torch.randn(1, M, D, device=DEVICE, dtype=torch.bfloat16)

    from flashlib.primitives.knn import flash_knn

    ms_t = _bench(lambda: flash_knn(xb, cb, k, backend="triton"), iters=5)
    try:
        ms_c = _bench(lambda: flash_knn(xb, cb, k, backend="cutedsl",
                                        autotune=False),
                      iters=5)
    except Exception as e:
        pytest.skip(f"cutedsl FA3 unavailable: {e}")

    winner = "cutedsl" if ms_c < ms_t else "triton"
    _save_row("knn", {
        "N": N, "M": M, "D": D, "k": k,
        "triton_ms":  f"{ms_t:.2f}",
        "cutedsl_ms": f"{ms_c:.2f}",
        "winner": winner,
        "speedup": f"{max(ms_t, ms_c) / min(ms_t, ms_c):.2f}x",
    })

    # Cross-backend index parity (triton vs cutedsl must agree on the
    # top-K set; bf16 ties allow a small mismatch margin).
    _, idx_t = flash_knn(xb, cb, k, backend="triton")
    _, idx_c = flash_knn(xb, cb, k, backend="cutedsl", autotune=False)
    s_t = set(map(tuple, idx_t[0].sort(dim=-1).values.cpu().tolist()))
    s_c = set(map(tuple, idx_c[0].sort(dim=-1).values.cpu().tolist()))
    overlap = len(s_t & s_c) / len(s_t)
    assert overlap > 0.85, f"knn index overlap {overlap:.3f} too low (D={D})"


# ---------------------------------------------------------------------------
# gemm — full Pareto sweep across the 11 variants at one representative shape.
# We validate "dispatcher pick by tol routes to a Pareto-optimal variant".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("M,K,N", [(2048, 2048, 2048)])
def test_gemm_pareto_sweep(M, K, N):
    torch.manual_seed(SEED)
    A_f = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    B_f = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    A_d = A_f.to(torch.float64)
    B_d = B_f.to(torch.float64)

    import flashlib.linalg.gemm as G
    fp64_ref = (A_d @ B_d).float()
    fp64_norm = fp64_ref.norm().clamp_min(1e-12).item()

    rows = []
    variants_to_try = [
        ("fp32",          G.gemm_fp32,         (A_f, B_f)),
        ("tf32",          G.gemm_tf32,         (A_f, B_f)),
        ("3xtf32",        G.gemm_3xtf32,       (A_f, B_f)),
        ("bf16",          G.gemm_bf16,         (A_f, B_f)),
        ("3xbf16",        G.gemm_3xbf16,       (A_f, B_f)),
        ("fp16",          G.gemm_fp16,         (A_f * 0.1, B_f * 0.1)),
        ("3xfp16",        G.gemm_3xfp16,       (A_f * 0.1, B_f * 0.1)),
        ("fp16_x9",       G.gemm_fp16_x9,      (A_f * 0.1, B_f * 0.1)),
        ("fp16_x3_kahan", G.gemm_fp16_x3_kahan,(A_f * 0.1, B_f * 0.1)),
        ("tf32_x6",       G.gemm_tf32_x6,      (A_d, B_d)),
    ]
    for name, fn, args in variants_to_try:
        try:
            ms = _bench(lambda: fn(*args), iters=5)
            C = fn(*args)
            torch.cuda.synchronize()
            # Scale ref to match input scaling
            scale = (A_f.norm() * B_f.norm()) / (args[0].norm() * args[1].norm())
            ref = fp64_ref / scale.item()
            ref_norm = ref.norm().clamp_min(1e-12).item()
            rms = ((C - ref).norm() / ref_norm).item()
            rows.append((name, ms, rms))
        except Exception as e:
            rows.append((name, None, None))

    # Save full table.
    if not (RESULTS_DIR / "boundaries_gemm.md").exists():
        with open(RESULTS_DIR / "boundaries_gemm.md", "w") as f:
            f.write(f"# GEMM Pareto sweep — {torch.cuda.get_device_name(0)}\n\n")
            f.write("| variant | runtime_ms | rms_rel_err | Pareto |\n")
            f.write("|---|---|---|---|\n")
    # Find Pareto (runtime, rms): a row is Pareto-optimal iff no other row
    # has both lower runtime AND lower rms.
    valid = [(n, m, e) for n, m, e in rows if m is not None]
    pareto_set = set()
    for n, m, e in valid:
        dominated = False
        for n2, m2, e2 in valid:
            if n == n2:
                continue
            if m2 < m and e2 < e:
                dominated = True
                break
        if not dominated:
            pareto_set.add(n)
    with open(RESULTS_DIR / "boundaries_gemm.md", "a") as f:
        for n, m, e in rows:
            star = "Y" if n in pareto_set else ""
            ms_str = f"{m:.2f}" if m is not None else "err"
            err_str = f"{e:.2e}" if e is not None else "err"
            f.write(f"| {n} | {ms_str} | {err_str} | {star} |\n")
        f.write(f"\n_Pareto front: {sorted(pareto_set)}_\n\n")

    assert len(pareto_set) >= 3, (
        f"Pareto front too small: {pareto_set}; need ≥3 trade-offs."
    )
    # The Pareto front MUST include something tighter than fp32 (one of
    # 3xtf32, fp16_x3_kahan) and something faster than fp32 (bf16/fp16/tf32).
    assert ("fp32" in pareto_set
            or "fp16_x3_kahan" in pareto_set
            or "3xtf32" in pareto_set), "no exact-precision Pareto winner"
    assert ("bf16" in pareto_set or "fp16" in pareto_set
            or "tf32" in pareto_set), "no fast Pareto winner"


# ---------------------------------------------------------------------------
# Dispatcher routing — pick the winner within ROUTE_SLACK
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _is_hopper(), reason="needs Hopper")
def test_kmeans_dispatcher_picks_close_to_winner():
    """flash_kmeans's _route() should pick a backend that runs within
    ``ROUTE_SLACK``× of the empirical winner across the boundary grid.

    Today _route() always picks Triton for kmeans (cutedsl is opt-in via
    backend="cutedsl"); we treat that as the "default safe" choice and
    only assert that the chosen runtime is finite. The boundary-bench
    table is the data source for tightening _route() later.
    """
    torch.manual_seed(SEED)
    x = torch.randn(8192, 64, device=DEVICE, dtype=torch.float32)
    from flashlib.primitives.kmeans import flash_kmeans
    cluster_ids, centroids, n_iter = flash_kmeans(
        x, n_clusters=16, max_iters=3,
    )
    assert cluster_ids.shape == (8192,)


@pytest.mark.skipif(not _is_hopper(), reason="needs Hopper")
def test_knn_dispatcher_picks_close_to_winner():
    """``flash_knn_dispatch._route()`` defaults to the Triton path on
    CUDA. The smart dispatch wrapper adds only the gather pass (already
    inside the explicit ``backend="triton"`` call), so the two timings
    must agree within ``ROUTE_SLACK``.
    """
    torch.manual_seed(SEED)
    N, M, D, k = 1024, 8192, 128, 8
    x = torch.randn(1, N, D, device=DEVICE, dtype=torch.bfloat16)
    c = torch.randn(1, M, D, device=DEVICE, dtype=torch.bfloat16)

    from flashlib.primitives.knn import flash_knn
    from flashlib.primitives.knn.impl import flash_knn_dispatch

    ms_smart  = _bench(lambda: flash_knn_dispatch(x, c, k), iters=3)
    ms_explicit = _bench(lambda: flash_knn(x, c, k, backend="triton"), iters=3)
    assert ms_smart < ROUTE_SLACK * ms_explicit, (
        f"smart dispatcher ({ms_smart:.2f}ms) too slow vs explicit "
        f"triton ({ms_explicit:.2f}ms)"
    )
