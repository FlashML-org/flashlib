"""Expanded-config validation outside the D=768/1024 fit points."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import torch

import flashlib.primitives.kmeans.triton.assign as assign
from benchmarks.blackwell import tune_triton_b200_wide as wide_tune
from benchmarks.blackwell.validate_triton_b200_random import WIDE_CONFIGS


SEED = 20260716


def _multiple_log_uniform(rng, low, high, multiple):
    value = math.exp(rng.uniform(math.log(low), math.log(high)))
    return max(multiple, int(round(value / multiple)) * multiple)


def _shapes():
    rng = random.Random(SEED)
    dimensions = (320, 416, 512, 704, 1_152, 2_560)
    rows = []
    for D in dimensions:
        rows.append((
            1,
            _multiple_log_uniform(rng, 65_536, 524_288, 128),
            D,
            _multiple_log_uniform(rng, 1_024, 32_768, 256),
        ))
    rows += [
        (2, 65_536, 416, 16_384),
        (2, 98_304, 1_152, 8_192),
        (4, 32_768, 2_560, 4_096),
    ]
    return rows


def _percentile(values, q):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def main():
    flusher = wide_tune._flusher()
    rows = []
    shapes = _shapes()
    for index, (B, N, D, K) in enumerate(shapes, 1):
        print(f"[{index}/{len(shapes)}] B={B} N={N} D={D} K={K}",
              flush=True)
        oracle = wide_tune._run_shape(
            N, D, K, flusher, blocks=1, B=B, configs=WIDE_CONFIGS)
        config = assign._heuristic_euclid_config_split_d(
            N, K, D, device=torch.device("cuda"), dtype=torch.bfloat16)
        key = wide_tune._key(config)
        rows.append({
            "B": B, "N": N, "D": D, "K": K,
            "heuristic_key": key,
            "oracle_key": oracle["best_key"],
            "regret": oracle["timings"][key]["median_ms"] / oracle["best_ms"],
            "correctness": oracle["correctness"][key],
        })
        torch.cuda.empty_cache()
    regrets = [row["regret"] for row in rows]
    payload = {
        "seed": SEED,
        "config_count": len(WIDE_CONFIGS),
        "rows": rows,
        "summary": {
            "shapes": len(rows),
            "regret_geomean": math.exp(
                statistics.fmean(math.log(value) for value in regrets)),
            "regret_p95": _percentile(regrets, 0.95),
            "regret_max": max(regrets),
            "minimum_correctness": min(row["correctness"] for row in rows),
        },
    }
    output = (
        Path(__file__).parent / "results"
        / "b200_d_extrapolation_validation.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B200 KMeans D-axis extrapolation validation", "",
        f"Seed `{SEED}`, oracle over {len(WIDE_CONFIGS)} configs.", "",
        "| B | N | D | K | heuristic | oracle | regret | correctness |",
        "|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['B']} | {row['N']} | {row['D']} | {row['K']} | "
            f"{row['heuristic_key']} | {row['oracle_key']} | "
            f"{row['regret']:.3f}x | {row['correctness']:.3f} |")
    summary = payload["summary"]
    lines += [
        "",
        f"geomean **{summary['regret_geomean']:.3f}x**, "
        f"p95 **{summary['regret_p95']:.3f}x**, "
        f"max **{summary['regret_max']:.3f}x**.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

