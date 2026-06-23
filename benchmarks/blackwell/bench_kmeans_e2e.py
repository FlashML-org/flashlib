"""End-to-end Lloyd k-means benchmark on Blackwell (B200).

Times the *full* clustering loop (assign + centroid update, ``max_iters``
iterations) for three variants that share the **same** Triton centroid
update and differ only in the assignment kernel:

    triton      -- flashlib.batch_kmeans_Euclid (Triton assign + update).
    cutedsl_bw  -- flashlib.cutedsl_kmeans_Euclid: the new Blackwell tcgen05
                   assign (this port) + Triton update.
    cake        -- flashlib_cake.flash_kmeans_assign (the reference Blackwell
                   CUDA assign) + the identical Triton update.

cake ships only the assign step, so the ``cake`` row reuses flashlib's Triton
update to make the loops comparable; the only moving part between the three is
the assign kernel. All variants use the same seeded ``init_centroids`` and the
same ``max_iters``, so they produce the same clustering (verified) -- the
comparison is purely speed.

Usage:
    python -m benchmarks.blackwell.bench_kmeans_e2e [--iters N] [--lloyd M]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from blackwell._common import (  # noqa: E402
    SHAPES,
    bench_gpu_ms_multi,
    device_info,
    make_inputs,
    tflops,
)


def _seed_centroids(x: torch.Tensor, K: int, seed: int = 0) -> torch.Tensor:
    """Deterministic (B, K, D) initial centroids: K seeded random points."""
    B, N, D = x.shape
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    idx = torch.randint(0, N, (B, K), generator=g, device=x.device)
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, D)).contiguous()


def _make_cake_lloyd():
    """Return a Lloyd loop using cake assign + flashlib's Triton update.

    Mirrors ``cutedsl_kmeans_Euclid`` exactly, swapping only the assign call.
    """
    from flashlib_cake import flash_kmeans_assign
    from flashlib.primitives.kmeans.triton.update import (
        triton_lloyd_centroid_step_euclid,
    )

    def cake_lloyd(x, n_clusters, *, max_iters, init_centroids):
        B, N, D = x.shape
        K = n_clusters
        centroids = init_centroids.view(B, K, D).contiguous()
        cur = centroids
        nxt = torch.empty_like(cur)
        sums_buf = torch.zeros((B, K, D), device=x.device, dtype=torch.float32)
        cnts_buf = torch.zeros((B, K), device=x.device, dtype=torch.int32)
        shift_buf = torch.empty((B, K), device=x.device, dtype=torch.float32)
        ids_buf = torch.empty((B, N), device=x.device, dtype=torch.int32)
        x_sq = (x.float() ** 2).sum(-1).contiguous()   # constant across iters
        cluster_ids = ids_buf
        for _ in range(max_iters):
            c_sq = (cur.float() ** 2).sum(-1).contiguous()
            cluster_ids = flash_kmeans_assign(
                x, cur, out=ids_buf, x_sq=x_sq, c_sq=c_sq, arch="sm_100a"
            )
            new_cents, _, _ = triton_lloyd_centroid_step_euclid(
                x, cluster_ids, cur,
                sums_buf=sums_buf, cnts_buf=cnts_buf,
                new_buf=nxt, shift_buf=shift_buf,
            )
            cur, nxt = nxt, cur
        return cluster_ids, cur

    return cake_lloyd


def run(iters: int, lloyd_iters: int, json_path: Path | None) -> int:
    name, cap = device_info()
    print(f"GPU: {name}  sm{cap[0]}{cap[1]}  torch {torch.__version__}  "
          f"Lloyd iters={lloyd_iters}, outer reps={iters}")

    from flashlib.primitives.kmeans import batch_kmeans_Euclid, cutedsl_kmeans_Euclid
    cake_lloyd = _make_cake_lloyd()

    rows = []
    for (N, D, K) in SHAPES:
        x, _ = make_inputs(N, D, K, seed=0)
        init = _seed_centroids(x, K, seed=0)

        # correctness: all three Lloyd loops should agree on final labels
        ids_t, *_ = batch_kmeans_Euclid(x, K, max_iters=lloyd_iters, init_centroids=init.clone())
        ids_b, *_ = cutedsl_kmeans_Euclid(x, K, max_iters=lloyd_iters, init_centroids=init.clone())
        ids_c, *_ = cake_lloyd(x, K, max_iters=lloyd_iters, init_centroids=init.clone())
        torch.cuda.synchronize()
        agree_bt = (ids_b.view(-1) == ids_t.view(-1)).float().mean().item()
        agree_ct = (ids_c.view(-1) == ids_t.view(-1)).float().mean().item()

        fns = {
            "triton": lambda: batch_kmeans_Euclid(x, K, max_iters=lloyd_iters, init_centroids=init.clone()),
            "cutedsl_bw": lambda: cutedsl_kmeans_Euclid(x, K, max_iters=lloyd_iters, init_centroids=init.clone()),
            "cake": lambda: cake_lloyd(x, K, max_iters=lloyd_iters, init_centroids=init.clone()),
        }
        order = ["triton", "cutedsl_bw", "cake"]
        timings = bench_gpu_ms_multi(
            fns, warmup=2, iters=iters, cold_l2=False, order=order,
        )

        row = {"N": N, "D": D, "K": K,
               "agree_bw_triton": agree_bt, "agree_cake_triton": agree_ct}
        for e in order:
            med, _ = timings[e]
            row[f"{e}_ms"] = med
            row[f"{e}_per_iter_ms"] = med / lloyd_iters
            # assign-GEMM TFLOPs over the whole loop (incl. update overhead)
            row[f"{e}_tflops"] = tflops(N, D, K, med / lloyd_iters)
        rows.append(row)

        t, b, c = row["triton_ms"], row["cutedsl_bw_ms"], row["cake_ms"]
        print(f"N{N:>8} D{D:>4} K{K:>6} | "
              f"triton={t:8.3f}ms  bw={b:8.3f}ms  cake={c:8.3f}ms | "
              f"bw vs triton {t / b:4.2f}x  bw vs cake {c / b:4.2f}x | "
              f"agree(bw,cake)={agree_bt:.3f},{agree_ct:.3f}")

    _write_markdown(rows, name, cap, lloyd_iters)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(
            {"device": name, "cap": cap, "lloyd_iters": lloyd_iters, "rows": rows},
            indent=2))
        print(f"\nWrote {json_path}")
    return 0


def _write_markdown(rows, name, cap, lloyd_iters):
    out = _BENCH_ROOT / "blackwell" / "results" / "e2e_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# End-to-end Lloyd k-means: Blackwell (B200)",
        "",
        f"GPU: **{name}** sm{cap[0]}{cap[1]}, torch {torch.__version__}. "
        f"Full Lloyd loop, {lloyd_iters} iters, identical seeded init + Triton "
        "centroid update; only the **assign** kernel differs. Median wall-time "
        "over the loop (warm).",
        "",
        "| N | D | K | triton ms | cutedsl_bw ms | cake ms | bw vs triton | bw vs cake | agree(bw/cake) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        t, b, c = r["triton_ms"], r["cutedsl_bw_ms"], r["cake_ms"]
        lines.append(
            f"| {r['N']} | {r['D']} | {r['K']} | {t:.3f} | {b:.3f} | {c:.3f} | "
            f"**{t / b:.2f}x** | {c / b:.2f}x | "
            f"{r['agree_bw_triton']:.3f}/{r['agree_cake_triton']:.3f} |"
        )
    lines += [
        "",
        "`bw vs triton` > 1 means the new Blackwell assign makes the full "
        "clustering faster than the Triton baseline; `bw vs cake` > 1 means "
        "the new kernel's loop beats the cake-assign loop. `agree` is the "
        "label match-rate vs the Triton loop (1.000 = identical clustering).",
        "",
        "Source: `benchmarks/blackwell/bench_kmeans_e2e.py`.",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=5, help="outer timing repeats")
    p.add_argument("--lloyd", type=int, default=10, help="Lloyd iterations per run")
    p.add_argument("--json", type=Path,
                   default=_BENCH_ROOT / "blackwell" / "results" / "e2e_benchmark.json")
    args = p.parse_args()
    assert torch.cuda.is_available(), "need CUDA"
    return run(args.iters, args.lloyd, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
