"""bench_throughput.py — routing-cost benchmark for online MoE routing.

Times each router's dispatch on synthetic gate-score matrices of growing token
count n (fixed experts m, top-k), reporting per-token latency and throughput.
Also records language-independent properties: online vs batch, whether the full
batch is required before deciding, and whether the top-k compute budget is
respected. Produces results/throughput.csv and figures/throughput.pdf.

This isolates the *algorithmic* routing work in a common NumPy/Python
implementation. The qualitative columns (online?, full-batch?, respects-k?) are
language-independent; the timings contrast WFPD's single-pass O(m + k log C)
cost with LPR's T-iteration Sinkhorn and Expert-choice's full-batch selection.

Run from the project root:
    python src/moe_routing/bench_throughput.py
    python src/moe_routing/bench_throughput.py --experts 64 --trials 7
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import WaterFillingRouter                          # noqa: E402
from baselines import (TokenChoiceRouter, ExpertChoiceRouter,      # noqa: E402
                       LPRRouter)

ROUTERS = {
    "WFPD": WaterFillingRouter,
    "Token-choice": TokenChoiceRouter,
    "Expert-choice": ExpertChoiceRouter,
    "LPR (OT)": LPRRouter,
}

# Language-independent properties (the real differentiators).
PROPS = {
    "WFPD":          dict(online="yes", full_batch="no",  respects_k="yes"),
    "Token-choice":  dict(online="yes", full_batch="no",  respects_k="yes"),
    "Expert-choice": dict(online="no",  full_batch="yes", respects_k="no"),
    "LPR (OT)":      dict(online="no",  full_batch="yes", respects_k="yes"),
}


def time_router(R, G: np.ndarray, k: int, cap_factor: float, trials: int):
    n, m = G.shape
    C = int(np.ceil(cap_factor * n * k / m))
    R(m, C, k).route_batch(G)                       # warmup (allocations, caches)
    best, total = float("inf"), 0.0
    for _ in range(trials):
        t0 = time.perf_counter()
        R(m, C, k).route_batch(G)
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
    return total / trials, best                     # (mean_s, best_s)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--cap-factor", type=float, default=1.25)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--figures", default=os.path.join(root, "figures"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m, k = args.experts, args.top_k

    rows = []                                        # (name, n, mean_ms, best_ms, us/tok, tok/s)
    timings = {name: [] for name in ROUTERS}         # name -> [mean_ms per size]
    for n in args.sizes:
        G = rng.gamma(shape=2.0, scale=1.0, size=(n, m))  # skewed gate scores
        for name, R in ROUTERS.items():
            mean_s, best_s = time_router(R, G, k, args.cap_factor, args.trials)
            rows.append((name, n, mean_s * 1e3, best_s * 1e3,
                         mean_s / n * 1e6, n / mean_s))
            timings[name].append(mean_s * 1e3)

    # ---- console: scaling table ----
    print(f"\nexperts m={m}  top-k={k}  cap-factor={args.cap_factor}  "
          f"trials={args.trials}")
    print(f"\n{'router':14s} {'n':>7s} {'mean(ms)':>10s} {'best(ms)':>10s} "
          f"{'us/token':>10s} {'tok/s':>12s}")
    print("-" * 68)
    for name, n, mean_ms, best_ms, us_tok, tok_s in rows:
        print(f"{name:14s} {n:7d} {mean_ms:10.3f} {best_ms:10.3f} "
              f"{us_tok:10.3f} {tok_s:12.0f}")

    # ---- console: property + speed summary at the largest n ----
    ref_n = args.sizes[-1]
    print(f"\n=== summary at n={ref_n} ===")
    print(f"{'router':14s} {'online':>7s} {'full-batch':>11s} "
          f"{'respects-k':>11s} {'us/token':>10s} {'tok/s':>12s}")
    print("-" * 70)
    for name in ROUTERS:
        r = next(x for x in rows if x[0] == name and x[1] == ref_n)
        p = PROPS[name]
        print(f"{name:14s} {p['online']:>7s} {p['full_batch']:>11s} "
              f"{p['respects_k']:>11s} {r[4]:10.3f} {r[5]:12.0f}")

    # ---- CSV ----
    os.makedirs(args.results, exist_ok=True)
    csv_path = os.path.join(args.results, "throughput.csv")
    with open(csv_path, "w") as f:
        f.write("router,n,mean_ms,best_ms,us_per_token,tokens_per_s,"
                "online,full_batch,respects_k\n")
        for name, n, mean_ms, best_ms, us_tok, tok_s in rows:
            p = PROPS[name]
            f.write(f"{name},{n},{mean_ms:.6f},{best_ms:.6f},{us_tok:.6f},"
                    f"{tok_s:.3f},{p['online']},{p['full_batch']},"
                    f"{p['respects_k']}\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"WFPD": "#2563eb", "Token-choice": "#9ca3af",
              "Expert-choice": "#f59e0b", "LPR (OT)": "#10b981"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in ROUTERS:
        c = colors[name]
        lw = 2.5 if name == "WFPD" else 1.5
        ax1.plot(args.sizes, timings[name], marker="o", color=c, lw=lw, label=name)
        per_tok = [t / n * 1e3 for t, n in zip(timings[name], args.sizes)]  # us/token
        ax2.plot(args.sizes, per_tok, marker="o", color=c, lw=lw, label=name)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("tokens  n"); ax1.set_ylabel("routing time (ms)")
    ax1.set_title("Total routing time vs n  (log-log)")
    ax1.legend(fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("tokens  n"); ax2.set_ylabel("us / token")
    ax2.set_title("Per-token routing cost")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(args.figures, exist_ok=True)
    pdf = os.path.join(args.figures, "throughput.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(args.figures, "throughput.png"), dpi=150)
    plt.close(fig)

    print(f"\nsaved table  -> {csv_path}")
    print(f"saved figure -> {pdf}")


if __name__ == "__main__":
    main()
