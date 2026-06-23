"""Wide-grid assign benchmark on B200, modeled on svg-project/flash-kmeans-tune.

Grid (the repo's cube, restricted to the regime cake/ours both support):
    N in {65536, 262144, 1048576}
    K in {256, 1024, 4096, 16384, 65536}     (K % 256 == 0; drops 200000)
    D in {64, 128, 256}                        (drops 512+; ours is D<=256)
    B = 1, bf16

Engines (all solve argmin ||x - c||^2):
    tuned  - Triton with the best tcgen05 config (BLOCK_N=128) from a per-shape
             sweep, matching the repo's "pick best config" methodology. The
             stock B200 default path routes to split-D + tiny tiles (BLOCK_N=32)
             which is below the tcgen05 M=128 atom and falls back to mma.sync;
             forcing BLOCK_N=128 is what flips Triton onto tcgen05.
    cake   - flashlib_cake.flash_kmeans_assign(arch="sm_100a")
    ours   - flashlib blackwell CuteDSL tcgen05 kernel

Fair interleaved cold-L2 CUDA-events timing; iters adapt to problem size.
Writes results/wide_assign_benchmark.{json,md}.

Usage:
    python benchmarks/blackwell/bench_kmeans_wide.py
"""
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1], Path("/root/flashlib/benchmarks")):
    if (_p / "blackwell" / "_common.py").exists():
        sys.path.insert(0, str(_p))
        break
from blackwell._common import (  # noqa: E402
    bench_gpu_ms, bench_gpu_ms_multi, device_info, make_l2_flush_buffer, tflops,
)

import flashlib.primitives.kmeans.triton.assign as TA  # noqa: E402
from flashlib.primitives.kmeans import euclid_assign_triton  # noqa: E402
from flashlib.primitives.kmeans.cutedsl.blackwell_assign import (  # noqa: E402
    blackwell_assign_euclid,
)
from flashlib_cake import flash_kmeans_assign  # noqa: E402

_ORIG_SPLIT = TA._need_split_d
_NO_SPLIT = lambda *a, **k: False

NS = (65536, 262144, 1048576)
KS = (256, 1024, 4096, 16384, 65536)
DS = (64, 128, 256)


def _smem_ok(D, BN, BK, S):
    return TA._smem_bytes(D, BN, BK, S, 2) <= TA._smem_limit(torch.device("cuda"))


def _cands(D):
    """tcgen05-eligible (BLOCK_N=128) candidate configs, SMEM-filtered."""
    base = [
        (64, 4, 3), (64, 8, 3), (64, 4, 4), (64, 4, 2),
        (128, 4, 2), (128, 8, 2), (128, 8, 3), (256, 8, 2),
    ]
    out = []
    for BK, W, S in base:
        if _smem_ok(D, 128, BK, S):
            out.append({"BLOCK_N": 128, "BLOCK_K": BK, "num_warps": W, "num_stages": S})
    return out


def _adaptive(N, D, K):
    flop = 2.0 * N * D * K
    if flop > 2e13:
        return 10, 3, 4
    if flop > 3e12:
        return 16, 3, 5
    if flop > 3e11:
        return 25, 4, 6
    return 40, 5, 6


def pick_tuned(x, c, c_sq, D, sweep_iters):
    TA._need_split_d = _NO_SPLIT
    out = torch.empty((1, x.shape[1]), dtype=torch.int32, device=x.device)
    best, best_ms = None, float("inf")
    for cfg in _cands(D):
        try:
            fn = lambda cfg=cfg: euclid_assign_triton(
                x, c, out=out, c_sq=c_sq, config=cfg, use_heuristic=False)
            ms, _ = bench_gpu_ms(fn, warmup=3, iters=sweep_iters, cold_l2=False)
            if ms < best_ms:
                best, best_ms = cfg, ms
        except Exception:
            continue
    TA._need_split_d = _ORIG_SPLIT
    return best


def sample_match(ids, x, c, s=8192):
    s = min(s, x.shape[1])
    xf = x[0, :s].float()
    cf = c[0].float()
    csq = (cf * cf).sum(-1)
    ref = (xf @ cf.t() - 0.5 * csq[None, :]).argmax(-1).int()
    return (ids[0, :s] == ref).float().mean().item()


def geo(v):
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")


def main():
    name, cap = device_info()
    print(f"GPU: {name} sm{cap[0]}{cap[1]} torch {torch.__version__}")
    print(f"grid: N{list(NS)} x K{list(KS)} x D{list(DS)} = {len(NS)*len(KS)*len(DS)} shapes, bf16\n")
    flush = make_l2_flush_buffer()

    # warm clocks
    xw = torch.randn(1, 1_048_576, 128, device="cuda", dtype=torch.bfloat16)
    cw = torch.randn(1, 1024, 128, device="cuda", dtype=torch.bfloat16)
    cwsq = (cw.float() ** 2).sum(-1).contiguous()
    wo = torch.empty((1, 1_048_576), dtype=torch.int32, device="cuda")
    for _ in range(15):
        flash_kmeans_assign(xw, cw, out=wo, c_sq=cwsq, arch="sm_100a")
    torch.cuda.synchronize()
    del xw, cw, cwsq, wo
    torch.cuda.empty_cache()

    rows = []
    print(f"{'N':>9} {'D':>4} {'K':>6} | {'tuned':>8} {'cake':>8} {'ours':>8} ms | "
          f"{'t/cake':>7} {'o/cake':>7} | match")
    for N in NS:
        for D in DS:
            for K in KS:
                g = torch.Generator(device="cuda"); g.manual_seed(0)
                x = torch.randn((1, N, D), dtype=torch.bfloat16, device="cuda", generator=g).contiguous()
                c = torch.randn((1, K, D), dtype=torch.bfloat16, device="cuda", generator=g).contiguous()
                c_sq = (c.float() ** 2).sum(-1).contiguous()
                x_sq = (x.float() ** 2).sum(-1).contiguous()
                iters, warm, sweep_it = _adaptive(N, D, K)

                cfg = pick_tuned(x, c, c_sq, D, sweep_it)
                o = {k: torch.empty((1, N), dtype=torch.int32, device="cuda")
                     for k in ("tuned", "cake", "ours")}

                def f_tuned():
                    TA._need_split_d = _NO_SPLIT
                    r = euclid_assign_triton(x, c, out=o["tuned"], c_sq=c_sq,
                                             config=cfg, use_heuristic=False)
                    TA._need_split_d = _ORIG_SPLIT
                    return r

                def f_cake():
                    return flash_kmeans_assign(x, c, out=o["cake"], x_sq=x_sq,
                                               c_sq=c_sq, arch="sm_100a")

                def f_ours():
                    return blackwell_assign_euclid(x, c, out=o["ours"], c_sq=c_sq)

                fns, mt = {}, {}
                for nm, f in (("tuned", f_tuned), ("cake", f_cake), ("ours", f_ours)):
                    try:
                        ids = f(); torch.cuda.synchronize()
                        mt[nm] = sample_match(ids, x, c)
                        fns[nm] = f
                    except Exception as exc:
                        mt[nm] = None
                        print(f"  [err] {nm} N{N} D{D} K{K}: {type(exc).__name__}: {str(exc)[:60]}")

                order = [k for k in ("tuned", "cake", "ours") if k in fns]
                t = bench_gpu_ms_multi(fns, warmup=warm, iters=iters, flush_buf=flush, order=order)
                row = {"N": N, "D": D, "K": K, "cfg": cfg}
                for nm in order:
                    row[nm] = t[nm][0]
                    row[nm + "_tf"] = tflops(N, D, K, t[nm][0])
                row["match_min"] = min(v for v in mt.values() if v is not None)
                rows.append(row)

                tn, ck, ou = row.get("tuned"), row.get("cake"), row.get("ours")
                tc = tn / ck if (tn and ck) else float("nan")
                oc = ou / ck if (ou and ck) else float("nan")
                print(f"{N:>9} {D:>4} {K:>6} | {tn:8.3f} {ck:8.3f} {ou:8.3f} ms | "
                      f"{tc:6.2f}x {oc:6.2f}x | {row['match_min']:.3f}")

                del x, c, c_sq, x_sq, o
                torch.cuda.empty_cache()

    # ---- summary ----
    print("\n\n## Wide grid: tuned Triton vs cake vs ours (B200, cold-L2 median ms)\n")
    print("| N | D | K | tuned ms | cake ms | ours ms | tuned TF | cake TF | ours TF | tuned/cake | ours/cake |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        tn, ck, ou = r.get("tuned"), r.get("cake"), r.get("ours")
        print(f"| {r['N']} | {r['D']} | {r['K']} | {tn:.3f} | {ck:.3f} | {ou:.3f} | "
              f"{r['tuned_tf']:.0f} | {r['cake_tf']:.0f} | {r['ours_tf']:.0f} | "
              f"{tn/ck:.2f}x | {ou/ck:.2f}x |")

    tc_all = [r['tuned']/r['cake'] for r in rows if r.get('tuned') and r.get('cake')]
    oc_all = [r['ours']/r['cake'] for r in rows if r.get('ours') and r.get('cake')]
    print(f"\ngeomean ALL: tuned/cake {geo(tc_all):.2f}x   ours/cake {geo(oc_all):.2f}x   (n={len(tc_all)})")
    print(f"correctness sample-match min over all shapes: {min(r['match_min'] for r in rows):.3f}")

    print("\nby D:")
    for D in DS:
        tc = [r['tuned']/r['cake'] for r in rows if r['D'] == D and r.get('tuned') and r.get('cake')]
        oc = [r['ours']/r['cake'] for r in rows if r['D'] == D and r.get('ours') and r.get('cake')]
        print(f"  D={D:>3}: tuned/cake {geo(tc):.2f}x   ours/cake {geo(oc):.2f}x")
    print("\nby K:")
    for K in KS:
        tc = [r['tuned']/r['cake'] for r in rows if r['K'] == K and r.get('tuned') and r.get('cake')]
        oc = [r['ours']/r['cake'] for r in rows if r['K'] == K and r.get('ours') and r.get('cake')]
        print(f"  K={K:>6}: tuned/cake {geo(tc):.2f}x   ours/cake {geo(oc):.2f}x")
    print("\nby N:")
    for N in NS:
        tc = [r['tuned']/r['cake'] for r in rows if r['N'] == N and r.get('tuned') and r.get('cake')]
        oc = [r['ours']/r['cake'] for r in rows if r['N'] == N and r.get('ours') and r.get('cake')]
        print(f"  N={N:>9}: tuned/cake {geo(tc):.2f}x   ours/cake {geo(oc):.2f}x")

    win = sum(1 for v in tc_all if v < 0.98); tie = sum(1 for v in tc_all if 0.98 <= v <= 1.02)
    loss = sum(1 for v in tc_all if v > 1.02)
    print(f"\ntuned vs cake: win {win}, tie {tie}, loss {loss} (of {len(tc_all)})")

    # ---- persist ----
    out_dir = _HERE.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    jrows = [{"N": r["N"], "D": r["D"], "K": r["K"],
              "tuned_ms": r.get("tuned"), "cake_ms": r.get("cake"), "ours_ms": r.get("ours"),
              "tuned_tflops": r.get("tuned_tf"), "cake_tflops": r.get("cake_tf"),
              "ours_tflops": r.get("ours_tf"),
              "tuned_over_cake": (r["tuned"]/r["cake"]) if r.get("tuned") and r.get("cake") else None,
              "ours_over_cake": (r["ours"]/r["cake"]) if r.get("ours") and r.get("cake") else None,
              "match_min": r.get("match_min"),
              "cfg": r.get("cfg")} for r in rows]
    meta = {"gpu": name, "cap": f"sm{cap[0]}{cap[1]}", "torch": torch.__version__,
            "dtype": "bfloat16", "B": 1,
            "grid": {"N": list(NS), "K": list(KS), "D": list(DS)},
            "reference_grid": "https://github.com/svg-project/flash-kmeans-tune"}
    summary = {"geomean_all": {"tuned_over_cake": geo(tc_all), "ours_over_cake": geo(oc_all)},
               "tuned_vs_cake": {"win": win, "tie": tie, "loss": loss, "n": len(tc_all)}}
    (out_dir / "wide_assign_benchmark.json").write_text(
        json.dumps({"meta": meta, "rows": jrows, "summary": summary}, indent=2))
    lines = ["# Flash-KMeans assign: wide grid (B200)", "",
             f"GPU {name} sm{cap[0]}{cap[1]}, torch {torch.__version__}, bf16. "
             f"Cold-L2 median ms, interleaved. tuned=Triton best tcgen05 cfg, "
             f"cake=flashlib_cake, ours=CuteDSL. Grid modeled on flash-kmeans-tune.", "",
             "| N | D | K | tuned ms | cake ms | ours ms | tuned/cake | ours/cake |",
             "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in jrows:
        tc = f"{r['tuned_over_cake']:.2f}x" if r['tuned_over_cake'] else "--"
        oc = f"{r['ours_over_cake']:.2f}x" if r['ours_over_cake'] else "--"
        lines.append(f"| {r['N']} | {r['D']} | {r['K']} | {r['tuned_ms']:.3f} | {r['cake_ms']:.3f} | "
                     f"{r['ours_ms']:.3f} | {tc} | {oc} |")
    lines += ["", f"geomean tuned/cake {geo(tc_all):.2f}x, ours/cake {geo(oc_all):.2f}x; "
              f"tuned vs cake win/tie/loss {win}/{tie}/{loss} of {len(tc_all)}."]
    (out_dir / "wide_assign_benchmark.md").write_text("\n".join(lines))
    print(f"\nWrote {out_dir/'wide_assign_benchmark.json'} and .md")


if __name__ == "__main__":
    main()
