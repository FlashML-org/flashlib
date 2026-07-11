"""Focused B200/BF16 split-D sweep for D=768 and D=1024.

The grid emphasizes large N and large K while retaining K=1024 as the lower
boundary.  Twenty-four curated configs cover the useful BN/BK/BD and
pipeline-depth choices without running the full 162-config Cartesian product.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

import flashlib.primitives.kmeans.triton.assign as TA
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton


def _cfg(BN, BK, BD, W, S):
    return {
        "BLOCK_N": BN,
        "BLOCK_K": BK,
        "BLOCK_D": BD,
        "num_warps": W,
        "num_stages": S,
    }


CONFIGS = (
    _cfg(128, 128, 32, 8, 4),
    _cfg(128, 128, 64, 8, 4),
    _cfg(128, 128, 128, 8, 4),
    _cfg(128, 128, 32, 8, 2),
    _cfg(128, 128, 64, 8, 2),
    _cfg(128, 128, 128, 8, 2),
    _cfg(128, 128, 64, 4, 4),
    _cfg(128, 128, 128, 4, 4),
    _cfg(128, 128, 64, 4, 2),
    _cfg(128, 128, 128, 4, 2),
    _cfg(64, 128, 32, 4, 4),
    _cfg(64, 128, 64, 4, 4),
    _cfg(64, 128, 128, 4, 4),
    _cfg(64, 128, 64, 4, 2),
    _cfg(64, 128, 128, 4, 2),
    _cfg(64, 128, 64, 8, 4),
    _cfg(64, 128, 128, 8, 4),
    _cfg(128, 64, 64, 4, 4),
    _cfg(128, 64, 128, 4, 4),
    _cfg(128, 64, 64, 8, 2),
    _cfg(128, 64, 128, 8, 2),
    _cfg(64, 64, 64, 4, 4),
    _cfg(64, 64, 128, 4, 4),
    _cfg(64, 64, 128, 4, 2),
)

FIT_SHAPES = tuple(
    (N, D, K)
    for N in (262_144, 1_048_576)
    for D in (768, 1_024)
    for K in (1_024, 4_096, 16_384, 65_536)
)
HOLDOUT_SHAPES = (
    (65_536, 768, 4_096),
    (65_536, 768, 65_536),
    (65_536, 1_024, 4_096),
    (65_536, 1_024, 65_536),
    (524_288, 768, 16_384),
    (524_288, 1_024, 16_384),
)


def _key(config):
    return (
        f"BN{config['BLOCK_N']}_BK{config['BLOCK_K']}_"
        f"BD{config['BLOCK_D']}_W{config['num_warps']}_"
        f"S{config['num_stages']}"
    )


def _inputs(N, D, K, B=1):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(N + D + K)
    x = torch.randn((B, N, D), device="cuda", dtype=torch.bfloat16,
                    generator=generator)
    centroids = torch.randn((B, K, D), device="cuda",
                            dtype=torch.bfloat16, generator=generator)
    c_sq = (centroids.float() ** 2).sum(-1).contiguous()
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    return x, centroids, c_sq, out


def _flusher():
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "l2_cache_size", 0) or 0)
    return torch.empty(max(256 << 20, 2 * l2) // 4,
                       device="cuda", dtype=torch.int32)


def _fits(config):
    return TA._smem_bytes_split_d(
        config["BLOCK_D"],
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["num_stages"],
        2,
    ) <= TA._smem_limit(torch.device("cuda"))


def _repetitions(N, D, K):
    work = N * D * K
    if work >= 30_000_000_000_000:
        return 3
    if work >= 5_000_000_000_000:
        return 5
    if work >= 500_000_000_000:
        return 8
    return 12


def _time(calls: dict[str, Callable[[], Any]], flusher, repetitions, blocks):
    for fn in calls.values():
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    block_medians = {name: [] for name in calls}
    raw = {name: [] for name in calls}
    names = list(calls)
    for block in range(blocks):
        order = names[block % len(names):] + names[:block % len(names)]
        samples = {name: [] for name in names}
        for _ in range(repetitions):
            for name in order:
                flusher.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record(); calls[name](); end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end))
        for name in names:
            block_medians[name].append(statistics.median(samples[name]))
            raw[name].extend(samples[name])
    return {
        name: {
            "median_ms": statistics.median(block_medians[name]),
            "block_medians_ms": block_medians[name],
            "samples_ms": raw[name],
        }
        for name in names
    }


def _sample_correct(x, centroids, out, rows=64):
    rows = min(rows, x.shape[1])
    points = x[:, :rows].float()
    cents = centroids.float()
    reference = (
        torch.matmul(points, cents.transpose(-1, -2))
        - 0.5 * (cents * cents).sum(-1).unsqueeze(1)
    ).argmax(-1).int()
    return float((out[:, :rows] == reference).float().mean())


def _run_shape(N, D, K, flusher, blocks, *, B=1, configs=None):
    x, centroids, c_sq, out = _inputs(N, D, K, B=B)
    calls = {}
    configs_by_key = {}
    rejected = {}
    for config in (configs or CONFIGS):
        if not _fits(config):
            continue
        name = _key(config)
        configs_by_key[name] = config
        calls[name] = lambda config=config: euclid_assign_triton(
            x, centroids, out=out, c_sq=c_sq, config=config,
            use_heuristic=False,
        )
    valid_calls = {}
    for name, fn in calls.items():
        try:
            fn()
            torch.cuda.synchronize()
            valid_calls[name] = fn
        except Exception as exc:  # resource limits are config-specific
            rejected[name] = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
    if not valid_calls:
        raise RuntimeError(f"no valid split-D configs: {rejected}")
    timing = _time(
        valid_calls, flusher, _repetitions(N, D, K), blocks)
    correctness = {}
    for name, fn in valid_calls.items():
        fn(); torch.cuda.synchronize()
        correctness[name] = _sample_correct(x, centroids, out)
    best = min(timing, key=lambda name: timing[name]["median_ms"])
    default_call = lambda: euclid_assign_triton(
        x, centroids, out=out, c_sq=c_sq)
    default = _time(
        {"default": default_call},
        flusher,
        _repetitions(N, D, K),
        blocks,
    )["default"]
    return {
        "N": N, "D": D, "K": K,
        "B": B,
        "best_key": best,
        "best_config": configs_by_key[best],
        "best_ms": timing[best]["median_ms"],
        "default_ms": default["median_ms"],
        "default_over_oracle": default["median_ms"] / timing[best]["median_ms"],
        "timings": timing,
        "correctness": correctness,
        "rejected": rejected,
    }


def _percentile(values, q):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "results"
        / "b200_triton_wide_subset.json",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flusher = _flusher()
    shapes = tuple(("fit", shape) for shape in FIT_SHAPES) + tuple(
        ("holdout", shape) for shape in HOLDOUT_SHAPES)
    rows = []
    for index, (split, (N, D, K)) in enumerate(shapes, 1):
        print(f"[{index}/{len(shapes)}] {split} N={N} D={D} K={K}",
              flush=True)
        row = _run_shape(N, D, K, flusher, args.blocks)
        row["split"] = split
        rows.append(row)
        torch.cuda.empty_cache()
    penalties = [row["default_over_oracle"] for row in rows]
    payload = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "B": 1,
            "dtype": "bfloat16",
            "configs": list(CONFIGS),
        },
        "rows": rows,
        "summary": {
            "shapes": len(rows),
            "minimum_correctness": min(
                min(row["correctness"].values()) for row in rows),
            "default_over_oracle_geomean": math.exp(
                statistics.fmean(math.log(value) for value in penalties)),
            "default_over_oracle_p95": _percentile(penalties, 0.95),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    lines = [
        "# B200 Triton KMeans D768/D1024 focused sweep", "",
        "| split | N | D | K | best config | oracle ms | default ms | penalty |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['N']} | {row['D']} | {row['K']} | "
            f"{row['best_key']} | {row['best_ms']:.3f} | "
            f"{row['default_ms']:.3f} | {row['default_over_oracle']:.2f}x |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

