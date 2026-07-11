"""Fresh random-shape validation for inferred B200 FP16/FP32 tables."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import torch

import flashlib.primitives.kmeans.triton.assign as assign
from benchmarks.blackwell import tune_triton_b200_dtypes as dtype_tune
from benchmarks.blackwell import tune_triton_b200_subset as small_tune
from benchmarks.blackwell import tune_triton_b200_wide as wide_tune


SEED = 20260721


def _multiple_log_uniform(rng, low, high, multiple):
    value = math.exp(rng.uniform(math.log(low), math.log(high)))
    return max(multiple, int(round(value / multiple)) * multiple)


def _shapes():
    rng = random.Random(SEED)
    rows = []
    for dtype_name in ("float16", "float32"):
        for _ in range(4):
            B = rng.choice((1, 2, 4))
            total_rows = _multiple_log_uniform(
                rng, 65_536, 393_216, 128 * B)
            rows.append((
                dtype_name,
                False,
                B,
                total_rows // B,
                rng.choice((64, 128, 256)),
                _multiple_log_uniform(rng, 512, 32_768, 256),
            ))
        for _ in range(4):
            B = rng.choice((1, 2, 4))
            total_rows = _multiple_log_uniform(
                rng, 65_536, 262_144, 128 * B)
            rows.append((
                dtype_name,
                True,
                B,
                total_rows // B,
                rng.choice((384, 512, 768, 1_024, 1_536, 2_048)),
                _multiple_log_uniform(rng, 1_024, 16_384, 256),
            ))
    return rows


def _percentile(values, q):
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    weight = rank - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def main():
    rows = []
    shapes = _shapes()
    for index, (dtype_name, _wide_sample, B, N, D, K) in enumerate(shapes, 1):
        dtype = getattr(torch, dtype_name)
        wide = assign._need_split_d(D, dtype, torch.device("cuda"))
        print(
            f"[{index}/{len(shapes)}] {dtype_name} B={B} "
            f"N={N} D={D} K={K}",
            flush=True,
        )
        oracle = dtype_tune._run_shape(N, D, K, dtype, wide, B=B)
        if wide:
            config = assign._heuristic_euclid_config_split_d(
                N, K, D, device=torch.device("cuda"), dtype=dtype)
            key = wide_tune._key(config)
        else:
            config = assign._heuristic_euclid_config(
                N, K, D, device=torch.device("cuda"), dtype=dtype)
            key = small_tune._config_key(config)
        rows.append({
            "dtype": dtype_name,
            "region": "wide" if wide else "small",
            "B": B, "N": N, "D": D, "K": K,
            "heuristic_key": key,
            "oracle_key": oracle["best_key"],
            "regret": oracle["timings"][key] / oracle["best_ms"],
            "correctness": oracle["correctness"],
        })
        torch.cuda.empty_cache()
    regrets = [row["regret"] for row in rows]
    payload = {
        "seed": SEED,
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
        Path(__file__).parent / "results" / "b200_dtype_random.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B200 KMeans FP16/FP32 random validation", "",
        f"Seed `{SEED}`.", "",
        "| dtype | region | B | N | D | K | heuristic | oracle | regret |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dtype']} | {row['region']} | {row['B']} | "
            f"{row['N']} | {row['D']} | {row['K']} | "
            f"{row['heuristic_key']} | {row['oracle_key']} | "
            f"{row['regret']:.3f}x |")
    summary = payload["summary"]
    lines += [
        "",
        f"geomean **{summary['regret_geomean']:.3f}x**, "
        f"p95 **{summary['regret_p95']:.3f}x**, "
        f"max **{summary['regret_max']:.3f}x**, "
        f"correctness **{summary['minimum_correctness']:.3f}**.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

