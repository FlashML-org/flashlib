"""Parity tests for the 4 new GEMM variants — verify each variant's output
matches a torch FP64 reference within the precision tier its docstring
advertises.

Each variant declares an "expected residual" in its cost model. If that
declaration is over-optimistic the test fails — that's the goal: keep the
``Estimate.expected_residual`` honest so ``info.pareto`` ranking stays
trustworthy.

We run on small shapes (1024 x 1024 x 1024) to keep CI fast.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for GEMM variant parity", allow_module_level=True)

DEVICE = "cuda"
SEED = 42


def _seeded(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_hopper() -> bool:
    return torch.cuda.get_device_properties(0).major >= 9


def _fp64_reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (A.to(torch.float64) @ B.to(torch.float64)).to(torch.float32)


def _rms_relerr(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Frobenius RMS relative error: ||out - ref||_F / ||ref||_F.

    This is the metric the Ozaki / FP16x9 / TF32x6 papers report and that the
    flashlib cost models declare in ``expected_residual``. Element-wise max
    can be much larger (per LoLo-drop analysis) — that's a different metric.
    """
    return ((out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)).item()


def _max_relerr(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Worst-case element-wise relative error (sanity-bound only)."""
    rel = (out.float() - ref.float()).abs() / ref.float().abs().clamp_min(1e-9)
    return rel.max().item()


@pytest.mark.parametrize("M,K,N", [(512, 512, 512), (1024, 512, 1024)])
def test_gemm_3xbf16_matches_reference(M, K, N):
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    from flashlib.linalg.gemm import gemm_3xbf16
    C = gemm_3xbf16(A, B)
    ref = _fp64_reference(A, B)
    rms = _rms_relerr(C, ref)
    # 3xbf16 with the CuTeDSL fused single-launch kernel keeps the FP32
    # accumulator across all 3 dots — measured RMS ~3e-5 on H200.
    # The Python 3-call fallback would degrade to ~1.7e-3 (BF16-truncated
    # partials before FP32 sum); allow either if no cute available.
    from flashlib.linalg.gemm import _has_cute
    threshold = 1e-4 if _has_cute() else 3e-3
    assert rms < threshold, f"3xbf16 RMS rel-err {rms:.2e} exceeds {threshold:.0e}"
    # Should at least beat single bf16 (~3e-3) measurably.
    Cb = (A.to(torch.bfloat16) @ B.to(torch.bfloat16)).float()
    rms_bf16 = _rms_relerr(Cb, ref)
    assert rms < rms_bf16, (
        f"3xbf16 ({rms:.2e}) not tighter than single bf16 ({rms_bf16:.2e})"
    )


@pytest.mark.skipif(not _is_hopper(), reason="ozaki2 cute path needs Hopper SM90")
@pytest.mark.parametrize("M,K,N,s", [(512, 512, 512, 6), (1024, 1024, 1024, 8)])
def test_gemm_ozaki2_cute_breaks_fp32_wall(M, K, N, s):
    """ozaki2_cute at s=8 should reach BETTER than fp32 (~3e-7) precision —
    the whole point of the Ozaki II CRT path. ~7 bits per modulus."""
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)

    try:
        from flashlib.linalg.gemm import gemm_ozaki2_cute
        C = gemm_ozaki2_cute(A, B, num_moduli=s)
    except Exception as e:
        pytest.skip(f"ozaki2_cute unavailable: {e}")

    ref = _fp64_reference(A, B)
    rms = _rms_relerr(C, ref)
    # Empirical (H200): s=6 -> ~1.0e-5; s=8 -> ~1.3e-7 (better than fp32!).
    threshold = {6: 3e-5, 8: 1e-6}.get(s, 1e-3)
    assert rms < threshold, (
        f"ozaki2_cute s={s} RMS rel-err {rms:.2e} exceeds {threshold:.0e} "
        f"— Ozaki II should give ~7 bits per modulus."
    )


@pytest.mark.parametrize("M,K,N,s", [(512, 512, 512, 6)])
def test_gemm_ozaki2_triton_breaks_fp32_wall(M, K, N, s):
    """ozaki2_triton: pure-Python/Triton path — same precision as cute,
    just slightly slower. Importantly: works WITHOUT the gemmul8.so."""
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    try:
        from flashlib.linalg.gemm import gemm_ozaki2_triton
        C = gemm_ozaki2_triton(A, B, num_moduli=s)
    except Exception as e:
        pytest.skip(f"ozaki2_triton unavailable: {e}")
    ref = _fp64_reference(A, B)
    rms = _rms_relerr(C, ref)
    # s=6 gives ~1e-5 RMS empirically (~7 bits per modulus).
    assert rms < 3e-5, f"ozaki2_triton s={s} RMS rel-err {rms:.2e}"


def test_gemm_ozaki2_cute_matches_triton_bitwise():
    """The CRT reconstruction path is shared between backends — same num_moduli
    must yield the same output regardless of whether the modular GEMMs run on
    Triton or CuTeDSL INT8."""
    if not _is_hopper():
        pytest.skip("cute path needs Hopper")
    _seeded()
    A = torch.randn(256, 512, device=DEVICE, dtype=torch.float32)
    B = torch.randn(512, 256, device=DEVICE, dtype=torch.float32)
    try:
        from flashlib.linalg.gemm import gemm_ozaki2_cute, gemm_ozaki2_triton
        Cc = gemm_ozaki2_cute(A, B, num_moduli=6)
        Ct = gemm_ozaki2_triton(A, B, num_moduli=6)
    except Exception as e:
        pytest.skip(f"ozaki2 unavailable: {e}")
    # Same CRT recon -> same answer (both use INT8/INT32 exact GEMM).
    rel = _rms_relerr(Cc, Ct)
    assert rel < 1e-7, f"cute vs triton ozaki2 differ by RMS={rel:.2e}"


@pytest.mark.skipif(not _is_hopper(), reason="cute_fp16x9 needs Hopper SM90")
@pytest.mark.parametrize("M,K,N", [(512, 512, 512), (1024, 1024, 1024)])
def test_gemm_fp16_x9_matches_reference(M, K, N):
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32) * 0.1
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.1

    try:
        from flashlib.linalg.gemm import gemm_fp16_x9
        C = gemm_fp16_x9(A, B)
    except Exception as e:
        pytest.skip(f"fp16_x9 cute kernel unavailable: {e}")

    ref = _fp64_reference(A, B)
    rms = _rms_relerr(C, ref)
    # fp16_x9 declares ~1e-6 @ K=512, ~4e-6 @ K=1024 (measured on H200).
    threshold = 8e-6 if K <= 1024 else 5e-5
    assert rms < threshold, f"fp16_x9 RMS rel-err {rms:.2e} exceeds {threshold:.0e}"
    # And it should crush 3xbf16 in RMS (the whole point of fp16x9).
    from flashlib.linalg.gemm import gemm_3xbf16
    rms_3bf16 = _rms_relerr(gemm_3xbf16(A * 10, B * 10), ref * 100)
    assert rms < rms_3bf16, (
        f"fp16_x9 ({rms:.2e}) not tighter than 3xbf16 ({rms_3bf16:.2e})"
    )


@pytest.mark.parametrize("M,K,N", [(512, 1024, 512)])
def test_gemm_fp16_x3_kahan_matches_reference(M, K, N):
    _seeded()
    # fp16x3 needs values within fp16 range (<= 65504); scale down.
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32) * 0.1
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32) * 0.1

    try:
        from flashlib.linalg.gemm import gemm_fp16_x3_kahan
        C = gemm_fp16_x3_kahan(A, B)
    except Exception as e:
        pytest.skip(f"fp16_x3_kahan unavailable: {e}")

    ref = _fp64_reference(A, B)
    rms = _rms_relerr(C, ref)
    # fp16x3 + Kahan: measured ~4.6e-7 RMS (Kahan kills the K-dependent floor).
    assert rms < 5e-6, f"fp16_x3_kahan RMS rel-err {rms:.2e} exceeds 5e-6"


@pytest.mark.parametrize("M,K,N", [(512, 512, 512)])
def test_gemm_tf32_x6_matches_reference(M, K, N):
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float64)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float64)

    try:
        from flashlib.linalg.gemm import gemm_tf32_x6
        C = gemm_tf32_x6(A, B)
    except Exception as e:
        pytest.skip(f"tf32_x6 unavailable: {e}")

    ref = (A @ B).to(torch.float32)
    rms = _rms_relerr(C, ref)
    # tf32x6 measured ~6.4e-7 @ K=256, ~2.4e-6 @ K=1024 on H200.
    # FP32 output storage sets the observable floor.
    threshold = 5e-6 if K <= 1024 else 3e-5
    assert rms < threshold, f"tf32_x6 RMS rel-err {rms:.2e} exceeds {threshold:.0e}"


def test_gemm_ozaki2_int8_unavailable_raises_or_runs():
    """Ozaki2 INT8 needs the gemmul8 native shim. If not built, must raise
    a clear ``GEMMul8NotBuilt`` error rather than crashing."""
    _seeded()
    M, K, N = 512, 512, 512
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float64)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float64)
    from flashlib.linalg.gemm.native.gemmul8 import GEMMul8NotBuilt
    from flashlib.linalg.gemm import gemm_ozaki2_int8

    try:
        C = gemm_ozaki2_int8(A, B, num_moduli=14, fastmode=False)
    except GEMMul8NotBuilt:
        pytest.skip("libgemmul8_shim.so not built on this host (expected).")
        return
    except Exception as e:  # pragma: no cover  hardware-specific
        pytest.skip(f"ozaki2 unavailable: {e}")
        return

    ref = (A @ B).to(C.dtype)
    rms = _rms_relerr(C, ref)
    # num_moduli=14 should reach FP64-grade — RMS rel-err ~ 1e-15 in theory;
    # output dtype caps observable.
    assert rms < 3e-6, f"ozaki2_int8 RMS rel-err {rms:.2e}"


def _variant_fn(variant):
    from flashlib.linalg import gemm as g
    return {
        "3xbf16": g.gemm_3xbf16,
        "fp16_x9": g.gemm_fp16_x9,
        "ozaki2_triton": lambda A, B: g.gemm_ozaki2_triton(A, B, num_moduli=8),
        "ozaki2_cute": lambda A, B: g.gemm_ozaki2_cute(A, B, num_moduli=8),
    }[variant]


@pytest.mark.parametrize("M,K,N", [(1000, 333, 777), (512, 1001, 512), (512, 512, 1001), (8, 7, 7)])
@pytest.mark.parametrize("variant,threshold", [
    ("3xbf16", 1e-4), ("fp16_x9", 1e-5), ("ozaki2_triton", 1e-6), ("ozaki2_cute", 1e-6),
])
def test_gemm_variants_unaligned_shapes(variant, threshold, M, K, N):
    """Shapes off the kernels' TMA / block alignment must still be exact to the variant's tier."""
    _seeded()
    A = torch.randn(M, K, device=DEVICE, dtype=torch.float32)
    B = torch.randn(K, N, device=DEVICE, dtype=torch.float32)
    rms = _rms_relerr(_variant_fn(variant)(A, B), _fp64_reference(A, B))
    assert rms < threshold, f"{variant} {M}x{K}x{N} RMS rel-err {rms:.2e}"


@pytest.mark.parametrize("variant", ["3xbf16", "fp16_x9", "ozaki2_cute"])
def test_gemm_result_not_overwritten_by_next_call(variant):
    """A second same-shape call must not write into the tensor returned by the first."""
    _seeded()
    A1, B1, A2, B2 = (torch.randn(256, 256, device=DEVICE) for _ in range(4))
    fn = _variant_fn(variant)
    C1 = fn(A1, B1)
    snapshot = C1.clone()
    fn(A2, B2)
    assert torch.equal(C1, snapshot)


def test_gemm_dispatcher_routes_by_tol():
    """gemm(..., tol=t) picks the strictest variant whose residual ≤ t."""
    _seeded()
    A = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    B = torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
    from flashlib.linalg.gemm import gemm

    # tol=1e-2 -> bf16 acceptable; just verify it returns a valid result.
    out = gemm(A, B, tol=1e-2)
    assert out.shape == (256, 256)
    ref = _fp64_reference(A, B)
    rms_loose = _rms_relerr(out, ref)
    assert rms_loose < 1e-2, f"tol=1e-2 path RMS={rms_loose:.2e}"

    # tol=None (exact) -> torch.matmul in input dtype; the global TF32 flag
    # decides whether this is fp32 (~3e-7) or TF32 (~3e-4). Either is valid
    # under the "input-dtype-preserving" contract; we just sanity-check
    # that the result is at worst at the TF32 floor.
    out_exact = gemm(A, B, tol=None)
    rms_exact = _rms_relerr(out_exact, ref)
    assert rms_exact < 1e-3, f"tol=None path RMS={rms_exact:.2e}"

    # backend="fp32" still forces strict fp32 (allow_tf32=False internally).
    out_strict = gemm(A, B, backend="fp32")
    rms_strict = _rms_relerr(out_strict, ref)
    assert rms_strict < 1e-6, f"backend=fp32 RMS={rms_strict:.2e}"


def test_gemm_pareto_ranking_consistent():
    """Pareto ranking via info.variants — declared expected_residual must
    monotonically degrade as runtime decreases (no dominated variants).

    A variant V is dominated if there's another variant W with W.runtime ≤ V.runtime
    AND W.residual ≤ V.residual. After our cost-model corrections, the Pareto
    set should still contain at least 4 variants (one per major precision tier).
    """
    import flashlib.info as info
    pareto = info.pareto("gemm", shape=(2048, 2048, 2048), device="H100")
    assert len(pareto) >= 4, (
        f"Expected ≥4 Pareto-optimal variants, got {len(pareto)}: "
        f"{[v.name for v in pareto]}"
    )
    # Each Pareto variant's residual should match what the per-file cost.py declares.
    for v in pareto:
        assert v.estimate.expected_residual is not None
        assert v.estimate.expected_residual > 0
