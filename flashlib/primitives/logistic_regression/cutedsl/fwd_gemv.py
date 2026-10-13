"""CuteDSL forward GEMV for flash logistic regression.

Strategy: re-use flashlib's HopperWgmmaGemmKernel for the *forward* pass by
casting the GEMV (X @ w, w shape (D,)) into a thin GEMM (X @ W) where W is
(D, 64) — w replicated 64 times. The replicated GEMM:
  - reads X exactly once (BW-bound term — same as GEMV)
  - over-computes 64x more output bytes (64MB vs 1MB) — negligible vs 4GB X read
  - benefits from WGMMA + TMA pipeline with multi-stage SMEM
  - has compute-bound headroom that GEMV lacks

The backward GEMV is GEMM-shaped (D, K=N=1M, output D) so we can also use
flashlib gemm_bf16 there — but cuBLAS already hits 93% peak BW, so the win
is marginal. We keep cuBLAS for backward.

For the forward pass, we observe:
  cuBLAS GEMV (1M, 2K)@(2K) bf16: ~955 us (theoretical 833 us, 87%)
  CuteDSL GEMM (1M, 64)@(2K, 64) bf16: targets <900 us
  Triton fused fwd (GEMV + sigmoid + resid + loss): 912 us, but does extra work

Public:
  cutedsl_logistic_regression(X, y, n_iter, ...) → (w, b)
    Same signature as triton_logistic_regression. Uses the CuteDSL fused
    forward inside the L-BFGS inner loop.
  cutedsl_fwd_gemv(X_bf, w_bf) → logits_fp32
    Standalone forward GEMV for benchmarking.

Toolchain blockers:
  - Cannot beat cuBLAS for backward GEMV without writing a custom split-K
    reduction kernel (cuBLAS at 93% peak is hard to top).
  - Forward GEMV has ~13% headroom; we target getting close to peak BW.
"""

import torch
import numpy as np

# Lazy import — CuteDSL only available when invoked.
_CUTEDSL_OK = None
_GEMM_FWD = None


def _ensure_cutedsl():
    global _CUTEDSL_OK, _GEMM_FWD
    if _CUTEDSL_OK is not None:
        return _CUTEDSL_OK
    try:
        from flashlib.linalg.gemm.cutedsl.bf16_chained import gemm_bf16_kn
        _GEMM_FWD = gemm_bf16_kn
        _CUTEDSL_OK = True
    except Exception as e:
        import warnings
        warnings.warn(
            f"flashlib gemm_bf16 unavailable for cutedsl logistic forward: {e}",
            ImportWarning,
            stacklevel=2,
        )
        _CUTEDSL_OK = False
    return _CUTEDSL_OK


# =============================================================================
# CuteDSL forward GEMV via tiled GEMM
# =============================================================================

# Shape-keyed cache of (W_replicated_buffer, w_padded, gemm_out_buffer).
# We replicate `w` of shape (D,) into (64, D) k-major, then call
# gemm_bf16_kn(X, W_rep) -> C of shape (N, 64) fp32.
# Logits = C[:, 0].

_REPLICATE_N = 64  # Use (128, 64) tile so W_rep doesn't bloat too much
_TILE_MN = (128, 64)


def _cutedsl_buffers(N, D, device):
    key = (N, D, str(device))
    cache = getattr(_cutedsl_buffers, "_c", {})
    if key in cache:
        return cache[key]
    W_rep = torch.zeros(_REPLICATE_N, D, device=device, dtype=torch.bfloat16)
    out = torch.empty(N, _REPLICATE_N, device=device, dtype=torch.float32)
    cache[key] = (W_rep, out)
    _cutedsl_buffers._c = cache
    return W_rep, out


def cutedsl_fwd_gemv(X_bf, w_bf, out_fp32=None):
    """Forward GEMV: logits = X @ w via CuteDSL GEMM with w replicated 64x.

    Uses tile_mn=(128, 64) — minimum N tile that satisfies WGMMA constraints.
    Output write is (N, 64) fp32 = 64x oversized but on H200's 4.8 TB/s HBM,
    that's only ~50us at N=1M. The win comes from CuteDSL's better scheduling
    on the X read pipeline (TMA + multi-stage SMEM).
    """
    if not _ensure_cutedsl():
        return torch.matmul(X_bf, w_bf).float()
    N, D = X_bf.shape
    W_rep, gemm_out = _cutedsl_buffers(N, D, X_bf.device)
    # Replicate w
    W_rep.copy_(w_bf.unsqueeze(0).expand(_REPLICATE_N, -1))
    _GEMM_FWD(X_bf, W_rep, out=gemm_out, tile_mn=_TILE_MN)
    if out_fp32 is None:
        out_fp32 = gemm_out[:, 0].clone()
    else:
        out_fp32.copy_(gemm_out[:, 0])
    return out_fp32


# =============================================================================
# Public driver: cutedsl_logistic_regression
# Re-uses the Triton infrastructure but swaps the forward GEMV for CuteDSL.
# =============================================================================


def cutedsl_logistic_regression(X: torch.Tensor, y: torch.Tensor,
                                 n_iter: int = 100,
                                 lr: float = None,
                                 C: float = 1.0,
                                 gtol: float = 1e-4,
                                 m_lbfgs: int = 10):
    """L-BFGS Logistic Regression with CuteDSL forward GEMV.

    Falls back to Triton implementation if CuteDSL/flashlib unavailable.
    The bf16 GEMV cast is hardwired in this CuteDSL path because the
    fused fwd kernel was specialised for bf16; pass ``backend=None`` /
    ``"triton"`` plus ``tol=None`` if you need an exact-dtype GEMV.
    """
    # The CuTe GEMV needs D % 8 == 0 (TMA row alignment); without it there is
    # nothing to gain from the bf16 cast, so run the exact Triton solver.
    if not _ensure_cutedsl() or X.shape[1] % 8:
        from flashlib.primitives.logistic_regression.triton import (
            triton_logistic_regression,
        )
        return triton_logistic_regression(X, y, n_iter=n_iter, lr=lr,
                                           C=C, gtol=gtol, m_lbfgs=m_lbfgs)

    # Use Triton infra for everything except the forward GEMV.
    from flashlib.primitives.logistic_regression.triton.logistic_regression import (
        _initial_step_analytical, _lbfgs_two_loop,
        _matvec, _matvec_t, _fused_logits_to_loss_resid,
    )

    N, D = X.shape
    if not X.is_contiguous():
        X = X.contiguous()
    y_f = y if y.dtype == torch.float32 else y.float()

    inv_N = 1.0 / N
    C_inv = (1.0 / (C * N)) if C > 0 else 0.0

    # bf16 conversion (cached).
    if X.dtype == torch.float32:
        cache_key = (X.data_ptr(), X.numel(), X.stride(0))
        cache = getattr(X, "_flash_lr_bf16_cache", None)
        if cache is not None and cache[0] == cache_key:
            X_bf = cache[1]
        else:
            X_bf = X.bfloat16()
            try:
                X._flash_lr_bf16_cache = (cache_key, X_bf)
            except (AttributeError, RuntimeError):
                pass
    else:
        X_bf = X

    # Iter 0: analytical Newton step
    w_aug, grad0, loss, grad = _initial_step_analytical(
        X_bf, y_f, C_inv, inv_N, N, D)

    grad_inf = grad.abs().max().item()
    if grad_inf < gtol:
        return w_aug[:D], w_aug[D:D+1]

    # CuteDSL fwd GEMV requires bf16 X; other input dtypes use the cuBLAS matvec.
    use_cute_fwd = X_bf.dtype == torch.bfloat16

    def eval_loss_grad_cutedsl(w_aug):
        w_w = w_aug[:D]
        w_b = w_aug[D:D+1]
        if use_cute_fwd:
            w_w_bf = w_w.to(X_bf.dtype)
            logits = cutedsl_fwd_gemv(X_bf, w_w_bf) + w_b
        else:
            logits = _matvec(X_bf, w_w) + w_b
        loss_t, resid = _fused_logits_to_loss_resid(logits, y_f)
        loss_data = loss_t * inv_N
        # cuBLAS backward (already 93% peak)
        grad_w_part = _matvec_t(X_bf, resid) * inv_N
        grad_b_part = (resid.sum() * inv_N).reshape(1)
        if C_inv != 0.0:
            grad_w_part = grad_w_part + C_inv * w_w
        grad_out = torch.cat([grad_w_part, grad_b_part])
        if C_inv != 0.0:
            loss_total = loss_data + 0.5 * C_inv * (w_w * w_w).sum()
        else:
            loss_total = loss_data
        return loss_total, grad_out

    s_list = [w_aug.clone()]
    y_diff_init = grad - grad0
    y_list = [y_diff_init]
    sy0 = s_list[0].dot(y_diff_init)
    rho_list = [1.0 / sy0.clamp(min=1e-10)]

    for it in range(1, n_iter):
        d = _lbfgs_two_loop(grad, s_list, y_list, rho_list)
        w_new = w_aug + d
        loss_new, grad_new = eval_loss_grad_cutedsl(w_new)

        s = w_new - w_aug
        y_diff = grad_new - grad
        sy = s.dot(y_diff)
        rho = 1.0 / sy.clamp(min=1e-10)
        if len(s_list) >= m_lbfgs:
            s_list.pop(0); y_list.pop(0); rho_list.pop(0)
        s_list.append(s); y_list.append(y_diff); rho_list.append(rho)

        w_aug = w_new
        grad = grad_new
        loss = loss_new

        grad_inf = grad.abs().max().item()
        if grad_inf < gtol:
            break

    return w_aug[:D], w_aug[D:D+1]


# Public alias
flash_cutedsl_logistic_regression = cutedsl_logistic_regression
