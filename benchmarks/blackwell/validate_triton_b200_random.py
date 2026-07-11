"""Deterministic random-shape oracle validation for the B200 heuristics."""
from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

import torch

import flashlib.primitives.kmeans.triton.assign as assign
from benchmarks.blackwell import tune_triton_b200_subset as small_tune
from benchmarks.blackwell import tune_triton_b200_wide as wide_tune


SEED = 20260713
SAMPLES_PER_REGION = 6


def _multiple_log_uniform(rng, low, high, multiple):
    value = math.exp(rng.uniform(math.log(low), math.log(high)))
    return max(multiple, int(round(value / multiple)) * multiple)


def _sample_shapes():
    rng = random.Random(SEED)
    known_small = set(small_tune.FIT_SHAPES) | set(small_tune.HOLDOUT_SHAPES)
    known_wide = set(wide_tune.FIT_SHAPES) | set(wide_tune.HOLDOUT_SHAPES)
    small = []
    while len(small) < SAMPLES_PER_REGION:
        shape = (
            1,
            _multiple_log_uniform(rng, 65_536, 1_048_576, 128),
            rng.choice((64, 128, 256)),
            _multiple_log_uniform(rng, 256, 65_536, 256),
        )
        if shape[1:] not in known_small and shape not in small:
            small.append(shape)
    wide = []
    while len(wide) < SAMPLES_PER_REGION:
        shape = (
            1,
            _multiple_log_uniform(rng, 65_536, 1_048_576, 128),
            rng.choice((768, 1_024)),
            _multiple_log_uniform(rng, 1_024, 65_536, 256),
        )
        if shape[1:] not in known_wide and shape not in wide:
            wide.append(shape)
    # Batch is a grid axis, not a different kernel. These shapes validate that
    # the same heuristic remains near-oracle when the first grid dimension is
    # replicated, without fitting batch-specific thresholds.
    small += [
        (2, 147_456, 128, 6_144),
        (4, 40_960, 256, 2_048),
        (8, 12_288, 64, 3_072),
    ]
    wide += [
        (2, 147_456, 768, 16_384),
        (4, 40_960, 1_024, 10_240),
    ]
    return small, wide


def _dict_config(config):
    return {
        **{key: int(value) for key, value in config.kwargs.items()},
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
    }


def _deduplicate(configs, key_fn):
    result = {}
    for config in configs:
        result[key_fn(config)] = config
    return tuple(result.values())


SMALL_CONFIGS = _deduplicate(
    tuple(small_tune.CONFIGS)
    + tuple(_dict_config(config) for config in assign._TUNE_CONFIGS),
    small_tune._config_key,
)

_wide_configs = []
for _config in assign._TUNE_CONFIGS_SPLIT_D:
    _row = _dict_config(_config)
    _BN, _BK, _BD = (
        _row["BLOCK_N"], _row["BLOCK_K"], _row["BLOCK_D"])
    if _BN >= 64 and _BK >= 64:
        _wide_configs.append(_row)
    elif (
        _row["num_warps"] == 4
        and _row["num_stages"] <= 2
        and _BD in (64, 128)
    ):
        _wide_configs.append(_row)
WIDE_CONFIGS = _deduplicate(
    tuple(wide_tune.CONFIGS) + tuple(_wide_configs),
    wide_tune._key,
)


def _percentile(values, q):
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def main():
    small_shapes, wide_shapes = _sample_shapes()
    rows = []
    small_flusher = small_tune._l2_flusher()
    for index, (B, N, D, K) in enumerate(small_shapes, 1):
        print(f"[random small {index}/{len(small_shapes)}] "
              f"B={B} N={N} D={D} K={K}", flush=True)
        oracle = small_tune._run_shape(
            N, D, K, small_flusher, blocks=1,
            B=B, configs=SMALL_CONFIGS)
        config = assign._heuristic_euclid_config(
            N, K, D, device=torch.device("cuda"), dtype=torch.bfloat16,
        )
        key = small_tune._config_key(config)
        rows.append({
            "region": "small",
            "B": B, "N": N, "D": D, "K": K,
            "heuristic_key": key,
            "oracle_key": oracle["best_key"],
            "regret": (
                oracle["timings"][key]["median_ms"] / oracle["best_ms"]),
            "correctness": oracle["correctness"][key],
        })
        torch.cuda.empty_cache()

    wide_flusher = wide_tune._flusher()
    for index, (B, N, D, K) in enumerate(wide_shapes, 1):
        print(f"[random wide {index}/{len(wide_shapes)}] "
              f"B={B} N={N} D={D} K={K}", flush=True)
        oracle = wide_tune._run_shape(
            N, D, K, wide_flusher, blocks=1,
            B=B, configs=WIDE_CONFIGS)
        config = assign._heuristic_euclid_config_split_d(
            N, K, D, device=torch.device("cuda"), dtype=torch.bfloat16)
        key = wide_tune._key(config)
        rows.append({
            "region": "wide",
            "B": B, "N": N, "D": D, "K": K,
            "heuristic_key": key,
            "oracle_key": oracle["best_key"],
            "regret": (
                oracle["timings"][key]["median_ms"] / oracle["best_ms"]),
            "correctness": oracle["correctness"][key],
        })
        torch.cuda.empty_cache()

    regrets = [row["regret"] for row in rows]
    payload = {
        "seed": SEED,
        "samples_per_region": SAMPLES_PER_REGION,
        "small_config_count": len(SMALL_CONFIGS),
        "wide_config_count": len(WIDE_CONFIGS),
        "rows": rows,
        "summary": {
            "shapes": len(rows),
            "regret_geomean": math.exp(
                statistics.fmean(math.log(value) for value in regrets)),
            "regret_p95": _percentile(regrets, 0.95),
            "regret_max": max(regrets),
            "minimum_correctness": min(
                row["correctness"] for row in rows),
            "passes": (
                _percentile(regrets, 0.95) <= 1.10
                and max(regrets) <= 1.15
                and min(row["correctness"] for row in rows) == 1.0
            ),
        },
    }
    output = Path(__file__).parent / "results" / "b200_random_validation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B200 KMeans random-shape validation", "",
        f"Seed `{SEED}`; shapes were sampled after the heuristic was fixed.",
        "",
        "| region | B | N | D | K | heuristic | oracle | regret | correctness |",
        "|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['region']} | {row['B']} | {row['N']} | "
            f"{row['D']} | {row['K']} | "
            f"{row['heuristic_key']} | {row['oracle_key']} | "
            f"{row['regret']:.3f}x | {row['correctness']:.3f} |")
    summary = payload["summary"]
    lines += [
        "",
        f"geomean regret **{summary['regret_geomean']:.3f}x**, "
        f"p95 **{summary['regret_p95']:.3f}x**, "
        f"max **{summary['regret_max']:.3f}x**.",
        f"Pass: **{summary['passes']}**.",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

