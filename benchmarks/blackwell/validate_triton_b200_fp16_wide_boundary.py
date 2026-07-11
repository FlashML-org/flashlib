"""Fresh FP16 validation around the BN128 -> BN64 wide-D transition."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import torch

import flashlib.primitives.kmeans.triton.assign as assign
from benchmarks.blackwell import tune_triton_b200_dtypes as dtype_tune
from benchmarks.blackwell import tune_triton_b200_wide as wide_tune


SHAPES = (
    (1, 147_456, 1_152, 6_144),
    (2, 73_728, 1_280, 10_240),
    (4, 32_768, 1_408, 4_096),
    (1, 196_608, 1_792, 2_048),
)


def main():
    rows = []
    for B, N, D, K in SHAPES:
        print(f"B={B} N={N} D={D} K={K}", flush=True)
        oracle = dtype_tune._run_shape(
            N, D, K, torch.float16, True, B=B)
        config = assign._heuristic_euclid_config_split_d(
            N, K, D, device=torch.device("cuda"), dtype=torch.float16)
        key = wide_tune._key(config)
        rows.append({
            "B": B, "N": N, "D": D, "K": K,
            "heuristic_key": key,
            "oracle_key": oracle["best_key"],
            "regret": oracle["timings"][key] / oracle["best_ms"],
            "correctness": oracle["correctness"],
        })
        torch.cuda.empty_cache()
    regrets = [row["regret"] for row in rows]
    payload = {
        "rows": rows,
        "summary": {
            "regret_geomean": math.exp(
                statistics.fmean(math.log(value) for value in regrets)),
            "regret_max": max(regrets),
            "minimum_correctness": min(row["correctness"] for row in rows),
        },
    }
    output = (
        Path(__file__).parent / "results"
        / "b200_fp16_wide_boundary.json"
    )
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(payload["summary"], flush=True)


if __name__ == "__main__":
    main()

