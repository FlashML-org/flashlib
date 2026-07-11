"""Focused B200/BF16 Triton KMeans assignment sweep.

This deliberately avoids the full flash-kmeans-tune Cartesian product. It
tests eight tcgen05-eligible BLOCK_N=128 configurations on a 24-shape fitting
grid plus six N=262K holdouts. The fit grid is B=1/BF16; the helpers also
support the separate B>1 validation scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from flashlib import _hw
import flashlib.primitives.kmeans.triton.assign as TA
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton


CONFIGS = (
    {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 8, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4},
    {"BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_N": 128, "BLOCK_K": 128, "num_warps": 8, "num_stages": 3},
    {"BLOCK_N": 128, "BLOCK_K": 256, "num_warps": 8, "num_stages": 2},
)

FIT_SHAPES = tuple(
    (N, D, K)
    for N in (65_536, 1_048_576)
    for D in (64, 128, 256)
    for K in (256, 1_024, 4_096, 65_536)
)
HOLDOUT_SHAPES = tuple(
    (262_144, D, K)
    for D in (64, 128, 256)
    for K in (1_024, 16_384)
)


def _config_key(config: dict[str, int]) -> str:
    return (
        f"BN{config['BLOCK_N']}_BK{config['BLOCK_K']}_"
        f"W{config['num_warps']}_S{config['num_stages']}"
    )


def _dict_config(config):
    return {
        **{key: int(value) for key, value in config.kwargs.items()},
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
    }


_expanded = {_config_key(config): config for config in CONFIGS}
for _triton_config in TA._TUNE_CONFIGS:
    _row = _dict_config(_triton_config)
    _expanded[_config_key(_row)] = _row
EXPANDED_CONFIGS = tuple(_expanded.values())


def _inputs(N: int, D: int, K: int, B: int = 1):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0)
    x = torch.randn(
        (B, N, D), device="cuda", dtype=torch.bfloat16, generator=generator
    ).contiguous()
    centroids = torch.randn(
        (B, K, D), device="cuda", dtype=torch.bfloat16, generator=generator
    ).contiguous()
    c_sq = (centroids.float() * centroids.float()).sum(-1).contiguous()
    out = torch.empty((B, N), device="cuda", dtype=torch.int32)
    return x, centroids, c_sq, out


def _sample_reference(x, centroids, rows: int = 128):
    points = x[:, : min(rows, x.shape[1])].float()
    cents = centroids.float()
    c_sq = (cents * cents).sum(-1)
    return (
        torch.matmul(points, cents.transpose(-1, -2))
        - 0.5 * c_sq.unsqueeze(1)
    ).argmax(-1)


def _l2_flusher():
    props = torch.cuda.get_device_properties(0)
    l2 = int(getattr(props, "l2_cache_size", 0) or 0)
    nbytes = max(256 << 20, 2 * l2)
    return torch.empty(nbytes // 4, device="cuda", dtype=torch.int32)


def _adaptive_repetitions(N: int, D: int, K: int) -> int:
    work = N * D * K
    if work >= 20_000_000_000_000:
        return 4
    if work >= 3_000_000_000_000:
        return 6
    if work >= 300_000_000_000:
        return 10
    return 16


def _interleaved_blocks(
    calls: dict[str, Callable[[], Any]],
    flusher,
    repetitions: int,
    blocks: int,
):
    block_medians = {name: [] for name in calls}
    raw = {name: [] for name in calls}
    names = list(calls)
    for fn in calls.values():
        for _ in range(2):
            fn()
    torch.cuda.synchronize()
    for block in range(blocks):
        order = names[block % len(names) :] + names[: block % len(names)]
        samples = {name: [] for name in names}
        for _ in range(repetitions):
            for name in order:
                flusher.zero_()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                calls[name]()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end))
        for name in names:
            raw[name].extend(samples[name])
            block_medians[name].append(statistics.median(samples[name]))
    return {
        name: {
            "median_ms": statistics.median(block_medians[name]),
            "block_medians_ms": block_medians[name],
            "samples_ms": raw[name],
        }
        for name in names
    }


def _fit_config(config: dict[str, int], D: int) -> bool:
    return (
        TA._smem_bytes(
            D,
            config["BLOCK_N"],
            config["BLOCK_K"],
            config["num_stages"],
            2,
        )
        <= TA._smem_limit(torch.device("cuda"))
    )


def _run_shape(
    N: int,
    D: int,
    K: int,
    flusher,
    blocks: int,
    *,
    B: int = 1,
    configs=None,
):
    x, centroids, c_sq, out = _inputs(N, D, K, B=B)
    reference = _sample_reference(x, centroids)
    calls: dict[str, Callable[[], Any]] = {}
    configs_by_key = {}
    rejected = {}
    for config in (configs or CONFIGS):
        if not _fit_config(config, D):
            continue
        key = _config_key(config)
        configs_by_key[key] = config
        calls[key] = lambda config=config: euclid_assign_triton(
            x,
            centroids,
            out=out,
            c_sq=c_sq,
            config=config,
            use_heuristic=False,
        )
    valid_calls = {}
    for key, fn in calls.items():
        try:
            fn(); torch.cuda.synchronize()
            valid_calls[key] = fn
        except Exception as exc:
            rejected[key] = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
    if not valid_calls:
        raise RuntimeError(f"no valid small-D configs: {rejected}")
    timings = _interleaved_blocks(
        valid_calls,
        flusher,
        repetitions=_adaptive_repetitions(N, D, K),
        blocks=blocks,
    )
    correctness = {}
    for key, fn in valid_calls.items():
        fn()
        torch.cuda.synchronize()
        correctness[key] = float(
            (out[:, : reference.shape[1]].long() == reference).float().mean()
        )
    best_key = min(timings, key=lambda key: timings[key]["median_ms"])
    default_out = torch.empty_like(out)
    default_call = lambda: euclid_assign_triton(
        x, centroids, out=default_out, c_sq=c_sq
    )
    default = _interleaved_blocks(
        {"default": default_call},
        flusher,
        repetitions=_adaptive_repetitions(N, D, K),
        blocks=blocks,
    )["default"]
    return {
        "N": N,
        "B": B,
        "D": D,
        "K": K,
        "best_key": best_key,
        "best_config": configs_by_key[best_key],
        "best_ms": timings[best_key]["median_ms"],
        "default_ms": default["median_ms"],
        "default_over_oracle": (
            default["median_ms"] / timings[best_key]["median_ms"]
        ),
        "timings": timings,
        "correctness": correctness,
        "rejected": rejected,
    }


def _percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    weight = rank - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def _write_markdown(path: Path, rows: list[dict[str, Any]]):
    lines = [
        "# B200 Triton KMeans focused subset",
        "",
        "B=1, BF16, cold-L2 interleaved CUDA-event medians. `default/oracle` "
        "shows the current production dispatch penalty.",
        "",
        "| split | N | D | K | best config | oracle ms | default ms | default/oracle |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        config = row["best_config"]
        lines.append(
            f"| {row['split']} | {row['N']} | {row['D']} | {row['K']} | "
            f"BN{config['BLOCK_N']}/BK{config['BLOCK_K']}/"
            f"W{config['num_warps']}/S{config['num_stages']} | "
            f"{row['best_ms']:.4f} | {row['default_ms']:.4f} | "
            f"{row['default_over_oracle']:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        filename = (
            "b200_triton_subset_expanded.json"
            if args.expanded
            else "b200_triton_subset.json"
        )
        args.output = Path(__file__).parent / "results" / filename
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if _hw.device_tag() != "B200":
        raise RuntimeError("this focused table is valid only on NVIDIA B200")

    flusher = _l2_flusher()
    rows = []
    all_shapes = (("fit", shape) for shape in FIT_SHAPES)
    all_shapes = tuple(all_shapes) + tuple(
        ("holdout", shape) for shape in HOLDOUT_SHAPES
    )
    for index, (split, (N, D, K)) in enumerate(all_shapes, 1):
        print(
            f"[{index}/{len(all_shapes)}] {split} N={N} D={D} K={K}",
            flush=True,
        )
        row = _run_shape(
            N, D, K, flusher, blocks=args.blocks,
            configs=EXPANDED_CONFIGS if args.expanded else CONFIGS,
        )
        row["split"] = split
        rows.append(row)
        torch.cuda.empty_cache()

    regret = [row["default_over_oracle"] for row in rows]
    payload = {
        "meta": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "dtype": "bfloat16",
            "B": 1,
            "expanded": args.expanded,
            "configs": list(
                EXPANDED_CONFIGS if args.expanded else CONFIGS),
        },
        "rows": rows,
        "summary": {
            "shapes": len(rows),
            "default_over_oracle_geomean": math.exp(
                statistics.fmean(math.log(value) for value in regret)
            ),
            "default_over_oracle_p95": _percentile(regret, 0.95),
            "minimum_correctness": min(
                min(row["correctness"].values()) for row in rows
            ),
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(args.output.with_suffix(".md"), rows)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

