"""Focused FP16/FP32 oracle sweep for B200 KMeans heuristic extrapolation."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import triton

import flashlib.primitives.kmeans.triton.assign as assign
from benchmarks.blackwell import tune_triton_b200_subset as small_tune
from benchmarks.blackwell import tune_triton_b200_wide as wide_tune
from benchmarks.blackwell.validate_triton_b200_random import WIDE_CONFIGS


SMALL_SHAPES = (
    (65_536, 64, 1_024),
    (262_144, 128, 4_096),
    (262_144, 256, 16_384),
)
WIDE_SHAPES = (
    (65_536, 512, 4_096),
    (262_144, 1_024, 4_096),
    (65_536, 2_048, 16_384),
)


def _inputs(N, D, K, dtype, B=1):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(N + D + K)
    x = torch.randn((B, N, D), device="cuda", dtype=dtype,
                    generator=generator)
    c = torch.randn((B, K, D), device="cuda", dtype=dtype,
                    generator=generator)
    c_sq = (c.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    return x, c, c_sq, out


def _time(calls, repetitions=4):
    valid = {}
    rejected = {}
    for name, fn in calls.items():
        try:
            fn(); torch.cuda.synchronize()
            valid[name] = fn
        except Exception as exc:
            rejected[name] = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
    samples = {name: [] for name in valid}
    for repetition in range(repetitions):
        order = list(valid)
        if repetition & 1:
            order.reverse()
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); valid[name](); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return {
        name: statistics.median(values) for name, values in samples.items()
    }, rejected


def _small_call(x, c, c_sq, out, config):
    B, N, D = x.shape
    K = c.shape[1]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)
    assign._euclid_assign_kernel[grid](
        x, c, c_sq, out,
        B, N, K, D,
        *x.stride(),
        *c.stride(),
        *c_sq.stride(),
        *out.stride(),
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return out


def _run_shape(N, D, K, dtype, wide, B=1):
    x, c, c_sq, out = _inputs(N, D, K, dtype, B=B)
    configs = WIDE_CONFIGS if wide else small_tune.EXPANDED_CONFIGS
    key_fn = wide_tune._key if wide else small_tune._config_key
    calls = {}
    config_by_key = {}
    for config in configs:
        key = key_fn(config)
        config_by_key[key] = config
        if wide:
            calls[key] = lambda config=config: assign.euclid_assign_triton(
                x, c, out=out, c_sq=c_sq, config=config,
                use_heuristic=False)
        else:
            calls[key] = lambda config=config: _small_call(
                x, c, c_sq, out, config)
    timings, rejected = _time(calls)
    best = min(timings, key=timings.get)
    calls[best](); torch.cuda.synchronize()
    rows = min(N, 32)
    points = x[:, :rows].float()
    cents = c.float()
    reference = (
        points @ cents.transpose(-1, -2)
        - 0.5 * (cents * cents).sum(-1).unsqueeze(1)
    ).argmax(-1).int()
    predicted = out[:, :rows]
    exact = predicted == reference
    batch = torch.arange(B, device=x.device)[:, None]
    predicted_centroids = cents[batch, predicted.long()]
    reference_centroids = cents[batch, reference.long()]
    predicted_distance = ((points - predicted_centroids) ** 2).sum(-1)
    reference_distance = ((points - reference_centroids) ** 2).sum(-1)
    distance_delta = (predicted_distance - reference_distance).abs()
    tie_ok = exact | (
        distance_delta
        <= 1e-3 + 1e-4 * reference_distance.abs()
    )
    return {
        "B": B, "N": N, "D": D, "K": K,
        "region": "wide" if wide else "small",
        "best_key": best,
        "best_config": config_by_key[best],
        "best_ms": timings[best],
        "timings": timings,
        "rejected_count": len(rejected),
        "exact_match": float(exact.float().mean()),
        "correctness": float(tie_ok.float().mean()),
    }


def main():
    rows = []
    for dtype_name, dtype in (
        ("float16", torch.float16),
        ("float32", torch.float32),
    ):
        for wide, shapes in ((False, SMALL_SHAPES), (True, WIDE_SHAPES)):
            for N, D, K in shapes:
                print(f"[{dtype_name}] N={N} D={D} K={K}", flush=True)
                row = _run_shape(N, D, K, dtype, wide)
                row["dtype"] = dtype_name
                rows.append(row)
                torch.cuda.empty_cache()
    output = (
        Path(__file__).parent / "results" / "b200_dtype_subset.json"
    )
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n",
                      encoding="utf-8")
    lines = [
        "# B200 KMeans FP16/FP32 subset oracle", "",
        "| dtype | region | N | D | K | best config | ms | correctness |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dtype']} | {row['region']} | {row['N']} | "
            f"{row['D']} | {row['K']} | {row['best_key']} | "
            f"{row['best_ms']:.3f} | {row['correctness']:.3f} |")
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

