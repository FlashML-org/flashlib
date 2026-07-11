"""Five-iteration Lloyd before/after for the B200 Triton heuristic."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
import triton

import flashlib.primitives.kmeans.triton.assign as assign
from flashlib.primitives.kmeans.triton.update import (
    triton_lloyd_centroid_step_euclid,
)


SHAPES = (
    (65_536, 64, 256),
    (65_536, 128, 4_096),
    (262_144, 256, 1_024),
    (1_048_576, 64, 1_024),
    (1_048_576, 128, 4_096),
    (1_048_576, 256, 4_096),
    (262_144, 768, 4_096),
    (262_144, 1_024, 4_096),
    (1_048_576, 768, 16_384),
    (1_048_576, 1_024, 16_384),
)
ITERS = 5


def _historical_split_assign(x, centroids, c_sq, out):
    """Launch the pre-heuristic split-D config without global monkeypatching."""
    B, N, D = x.shape
    K = centroids.shape[1]
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), B)
    assign._euclid_assign_kernel_split_d[grid](
        x, centroids, c_sq, out,
        B, N, K, D,
        *x.stride(),
        *centroids.stride(),
        *c_sq.stride(),
        *out.stride(),
        BLOCK_N=32,
        BLOCK_K=32,
        BLOCK_D=32,
        num_warps=4,
        num_stages=1,
    )
    return out


class Loop:
    def __init__(self, x, initial, old_dispatch):
        B, N, D = x.shape
        K = initial.shape[1]
        self.x = x
        self.initial = initial
        self.old_dispatch = old_dispatch
        self.cur = torch.empty_like(initial)
        self.nxt = torch.empty_like(initial)
        self.ids = torch.empty((B, N), device="cuda", dtype=torch.int32)
        self.sums = torch.empty((B, K, D), device="cuda", dtype=torch.float32)
        self.counts = torch.empty((B, K), device="cuda", dtype=torch.int32)
        self.shifts = torch.empty((B, K), device="cuda", dtype=torch.float32)

    def __call__(self):
        self.cur.copy_(self.initial)
        cur, nxt = self.cur, self.nxt
        for _ in range(ITERS):
            c_sq = (cur.float() ** 2).sum(-1).contiguous()
            if self.old_dispatch:
                _historical_split_assign(
                    self.x, cur, c_sq, self.ids
                )
            else:
                assign.euclid_assign_triton(
                    self.x, cur, out=self.ids, c_sq=c_sq
                )
            triton_lloyd_centroid_step_euclid(
                self.x,
                self.ids,
                cur,
                sums_buf=self.sums,
                cnts_buf=self.counts,
                new_buf=nxt,
                shift_buf=self.shifts,
            )
            cur, nxt = nxt, cur
        return self.ids, cur


def _time(calls):
    for fn in calls.values():
        fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in calls}
    for repetition in range(7):
        order = list(calls)
        if repetition & 1:
            order.reverse()
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); calls[name](); end.record(); end.synchronize()
            samples[name].append(start.elapsed_time(end))
    return {
        name: statistics.median(values) for name, values in samples.items()
    }


def main():
    rows = []
    for N, D, K in SHAPES:
        print(f"[lloyd] N={N} D={D} K={K}", flush=True)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(11)
        x = torch.randn((1, N, D), device="cuda", dtype=torch.bfloat16,
                        generator=generator)
        selected = torch.randint(0, N, (1, K), device="cuda",
                                 generator=generator)
        initial = torch.gather(
            x, 1, selected[..., None].expand(-1, -1, D)
        ).contiguous()
        calls = {
            "before": Loop(x, initial, True),
            "after": Loop(x, initial, False),
        }
        outputs = {name: fn() for name, fn in calls.items()}
        torch.cuda.synchronize()
        label_match = float(
            (outputs["before"][0] == outputs["after"][0]).float().mean()
        )
        timing = _time(calls)
        rows.append({
            "N": N, "D": D, "K": K, "iterations": ITERS,
            "before_ms": timing["before"],
            "after_ms": timing["after"],
            "speedup": timing["before"] / timing["after"],
            "label_match": label_match,
        })
        torch.cuda.empty_cache()
    out = Path(__file__).parent / "results" / "b200_triton_lloyd.json"
    out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# B200 Triton KMeans Lloyd before/after", "",
        "| N | D | K | before ms | after ms | speedup | label match |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['N']} | {row['D']} | {row['K']} | "
            f"{row['before_ms']:.3f} | {row['after_ms']:.3f} | "
            f"{row['speedup']:.2f}x | {row['label_match']:.4f} |")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

