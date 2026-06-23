"""Head-to-head benchmark: flash-kmeans *assign* on Blackwell (B200).

Engines (each solves the same argmin ‖x - c‖² assignment):

    triton       -- flashlib.euclid_assign_triton (current default on B200).
    cutedsl_api  -- flashlib.cutedsl_assign_euclid (public CuteDSL entry).
                    Before the Blackwell kernel is wired in it falls back to
                    Triton on sm_100; after wiring it dispatches to the new
                    Blackwell tcgen05 kernel.
    cutedsl_bw   -- the new Blackwell tcgen05 CuteDSL kernel called directly
                    (skipped until blackwell_assign.py exists).
    cake         -- flashlib_cake.flash_kmeans_assign (the reference Blackwell
                    CUDA kernel we are porting).

Norm tensors (x_sq / c_sq) are precomputed outside the timed region for every
engine so the comparison is kernel-vs-kernel. Timing is cold-L2 CUDA events.

Usage:
    python -m benchmarks.blackwell.bench_kmeans_assign [--iters N] [--json PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Allow `python benchmarks/blackwell/bench_kmeans_assign.py` and `-m` both.
_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from blackwell._common import (  # noqa: E402
    SHAPES,
    bench_gpu_ms_multi,
    correctness,
    device_info,
    make_inputs,
    make_l2_flush_buffer,
    reference_assign,
    tflops,
)


def _build_engines():
    """Return ``{name: callable(x, c, c_sq, x_sq, out) -> ids}``."""
    from flashlib.primitives.kmeans import euclid_assign_triton, cutedsl_assign_euclid

    engines = {}

    def _triton(x, c, c_sq, x_sq, out):
        return euclid_assign_triton(x, c, out=out, c_sq=c_sq)

    def _cutedsl_api(x, c, c_sq, x_sq, out):
        return cutedsl_assign_euclid(x, c, out=out, c_sq=c_sq, autotune=False)

    engines["triton"] = _triton
    engines["cutedsl_api"] = _cutedsl_api

    # New Blackwell kernel called directly (present after task 2).
    try:
        from flashlib.primitives.kmeans.cutedsl.blackwell_assign import (
            blackwell_assign_supported,
            blackwell_assign_euclid,
        )

        def _cutedsl_bw(x, c, c_sq, x_sq, out):
            return blackwell_assign_euclid(x, c, out=out, c_sq=c_sq)

        engines["cutedsl_bw"] = _cutedsl_bw
        engines["_bw_supported"] = blackwell_assign_supported  # type: ignore
    except Exception:
        pass

    try:
        from flashlib_cake import flash_kmeans_assign

        def _cake(x, c, c_sq, x_sq, out):
            return flash_kmeans_assign(
                x, c, out=out, x_sq=x_sq, c_sq=c_sq, arch="sm_100a"
            )

        engines["cake"] = _cake
    except Exception as exc:  # pragma: no cover
        print(f"[warn] flashlib_cake unavailable: {exc}")

    return engines


def run(iters: int, warmup: int, json_path: Path | None) -> int:
    name, cap = device_info()
    print(f"GPU: {name}  sm{cap[0]}{cap[1]}  torch {torch.__version__}")
    engines = _build_engines()
    bw_supported = engines.pop("_bw_supported", None)
    order = [k for k in ("triton", "cutedsl_api", "cutedsl_bw", "cake") if k in engines]
    print("engines:", ", ".join(order))
    flush = make_l2_flush_buffer()

    # global warmup to bring clocks out of idle before the first measurement
    warm_fn = engines.get("cake") or engines.get("triton") or next(iter(engines.values()))
    xw, cw = make_inputs(1_048_576, 128, 1024, seed=9)
    cw_sq = (cw.float() ** 2).sum(-1).contiguous()
    wout = torch.empty((1, 1_048_576), dtype=torch.int32, device=xw.device)
    for _ in range(15):
        warm_fn(xw, cw, cw_sq, None, wout)
    torch.cuda.synchronize()
    del xw, cw, cw_sq, wout

    rows = []
    for (N, D, K) in SHAPES:
        x, c = make_inputs(N, D, K, seed=0)
        c_sq = (c.float() ** 2).sum(-1).contiguous()       # (1, K)
        x_sq = (x.float() ** 2).sum(-1).contiguous()       # (1, N)
        ref = reference_assign(x, c)
        torch.cuda.synchronize()

        row: dict = {"N": N, "D": D, "K": K}
        # build the per-shape callables + correctness, skipping unsupported
        fns = {}
        run_order = []
        outs = {}
        for ename in order:
            fn = engines[ename]
            if ename == "cutedsl_bw" and bw_supported is not None and not bw_supported(N, D, K):
                row[f"{ename}_ms"] = None
                row[f"{ename}_skip"] = "unsupported shape"
                continue
            out = torch.empty((1, N), dtype=torch.int32, device=x.device)
            outs[ename] = out
            try:
                ids = fn(x, c, c_sq, x_sq, out)
                torch.cuda.synchronize()
                stats = correctness(ids, ref, x, c)
                row[f"{ename}_exact"] = stats["exact_match"]
                row[f"{ename}_tie_ok"] = stats["tie_ok"]
                fns[ename] = (lambda fn=fn, out=out: fn(x, c, c_sq, x_sq, out))
                run_order.append(ename)
            except Exception as exc:
                row[f"{ename}_ms"] = None
                row[f"{ename}_err"] = f"{type(exc).__name__}: {str(exc)[:80]}"
                print(f"  [err] {ename} N{N} D{D} K{K}: {row[f'{ename}_err']}")

        # interleaved timing -> fair across engines under clock drift
        timings = bench_gpu_ms_multi(
            fns, warmup=warmup, iters=iters, flush_buf=flush, order=run_order,
        )
        for ename in run_order:
            med, _ = timings[ename]
            row[f"{ename}_ms"] = med
            row[f"{ename}_tflops"] = tflops(N, D, K, med)
        rows.append(row)
        _print_row(row, order)

    _print_summary(rows, order)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"device": name, "cap": cap, "rows": rows}, indent=2))
        print(f"\nWrote {json_path}")
    _write_markdown(rows, order, name, cap)
    return 0


def _fmt(v, spec="8.4f"):
    return ("{:" + spec + "}").format(v) if isinstance(v, (int, float)) else f"{'--':>8}"


def _print_row(row, order):
    head = f"N{row['N']:>8} D{row['D']:>4} K{row['K']:>6} |"
    parts = []
    for e in order:
        ms = row.get(f"{e}_ms")
        parts.append(f"{e}={_fmt(ms)}ms")
    cake = row.get("cake_ms")
    extra = ""
    if cake:
        for e in order:
            if e == "cake":
                continue
            ms = row.get(f"{e}_ms")
            if ms:
                extra += f"  {e}->cake {ms / cake:.2f}x"
    print(head, "  ".join(parts), extra)


def _print_summary(rows, order):
    print("\n=== speedup vs cake (median ms; >1 means slower than cake) ===")
    for row in rows:
        cake = row.get("cake_ms")
        if not cake:
            continue
        s = f"N{row['N']:>8} D{row['D']:>4} K{row['K']:>6} | cake {cake:7.4f}ms"
        for e in order:
            if e == "cake":
                continue
            ms = row.get(f"{e}_ms")
            if ms:
                s += f" | {e} {ms / cake:5.2f}x"
        print(s)


def _write_markdown(rows, order, name, cap):
    out = _BENCH_ROOT / "blackwell" / "results" / "assign_benchmark.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Flash-KMeans assign: Blackwell head-to-head",
        "",
        f"GPU: **{name}** sm{cap[0]}{cap[1]}, torch {torch.__version__}. "
        "Cold-L2 CUDA-events median ms; `c_sq`/`x_sq` precomputed outside the "
        "timed region. All engines solve the same argmin ‖x-c‖² assignment.",
        "",
    ]
    hdr = "| N | D | K |"
    sep = "|---:|---:|---:|"
    for e in order:
        hdr += f" {e} ms | {e} TFLOPs |"
        sep += "---:|---:|"
    if "cake" in order:
        for e in order:
            if e != "cake":
                hdr += f" {e}/cake |"
                sep += "---:|"
    lines += [hdr, sep]
    for row in rows:
        cells = f"| {row['N']} | {row['D']} | {row['K']} |"
        for e in order:
            ms = row.get(f"{e}_ms")
            tf = row.get(f"{e}_tflops")
            cells += f" {ms:.4f} |" if isinstance(ms, float) else " -- |"
            cells += f" {tf:.1f} |" if isinstance(tf, float) else " -- |"
        if "cake" in order:
            cake = row.get("cake_ms")
            for e in order:
                if e == "cake":
                    continue
                ms = row.get(f"{e}_ms")
                if isinstance(ms, float) and cake:
                    cells += f" {ms / cake:.2f}x |"
                else:
                    cells += " -- |"
        lines.append(cells)
    lines += [
        "",
        "`X/cake` < 1.0 means X is faster than cake; > 1.0 means slower. "
        "Source: `benchmarks/blackwell/bench_kmeans_assign.py`.",
        "",
    ]
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--json", type=Path, default=_BENCH_ROOT / "blackwell" / "results" / "assign_benchmark.json")
    args = p.parse_args()
    assert torch.cuda.is_available(), "need CUDA"
    return run(args.iters, args.warmup, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
