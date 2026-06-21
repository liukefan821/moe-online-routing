"""evaluate.py — offline evaluation harness for online MoE routing.

Loads cached gate-logit traces (data/traces/*.npz), runs WFPD and the baselines
on each token block, and reports paper-grade metrics against the true offline
optimum (OPT, via a HiGHS LP): retained-score ratio (ALG/OPT), max-load, Gini,
served fraction, and mean experts/token. Aggregates over MoE layers, writes a
CSV summary, and renders comparison figures (PDF+PNG) for the paper.

Why ratio-to-OPT instead of raw objective: routers that violate the "<= k experts
per token" budget (e.g. Expert-choice) inflate raw objective and look artificially
strong. Measuring ALG / OPT under the SAME (<=k, <=C) constraints is the rigorous
competitive ratio of Theorem 1 (guaranteed >= 1 - 1/e ~ 0.632; typically near 1).

Run from the project root:
    python src/moe_routing/trace_extract.py        # produce data/traces/*.npz first
    python src/moe_routing/evaluate.py
    python src/moe_routing/evaluate.py --cap-factor 1.25 --score softmax
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

# sibling modules (script dir is on sys.path when run as a file)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import WaterFillingRouter                          # noqa: E402
from baselines import (TokenChoiceRouter, ExpertChoiceRouter,      # noqa: E402
                       LPRRouter, _gini)

ROUTERS = {
    "WFPD": WaterFillingRouter,
    "Token-choice": TokenChoiceRouter,
    "Expert-choice": ExpertChoiceRouter,
    "LPR (OT)": LPRRouter,
}


def to_scores(logits: np.ndarray, mode: str) -> np.ndarray:
    """Map raw gate logits to nonnegative routing scores g(t, e) >= 0."""
    if mode == "softmax":
        z = logits - logits.max(axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(axis=1, keepdims=True)
    if mode == "relu":
        return np.maximum(logits, 0.0)
    return logits - logits.min(axis=1, keepdims=True)  # shift to nonneg


def optimal_value(G: np.ndarray, C: int, k: int) -> float:
    """True offline optimum of  max sum(G*x)  s.t. row<=k, col<=C, 0<=x<=1.
    The constraint matrix is totally unimodular, so the LP optimum is integral."""
    n, m = G.shape
    N = n * m
    var = np.arange(N)
    rows = np.concatenate([var // m, n + (var % m)])   # token rows, then expert rows
    cols = np.concatenate([var, var])
    data = np.ones(2 * N)
    A_ub = sp.coo_matrix((data, (rows, cols)), shape=(n + m, N)).tocsr()
    b_ub = np.concatenate([np.full(n, k), np.full(m, C)]).astype(float)
    res = linprog(c=(-G).ravel(), A_ub=A_ub, b_ub=b_ub,
                  bounds=(0.0, 1.0), method="highs")
    return float(-res.fun)


def metrics(G: np.ndarray, A: np.ndarray, C: int, opt: float) -> dict:
    load = A.sum(0)
    score = float((G * A).sum())
    return {
        "retained": score / opt if opt > 0 else float("nan"),  # ALG / OPT
        "gini": _gini(load),                                    # 0 = perfect balance
        "Lmax/C": float(load.max() / C),                       # tail vs capacity
        "minmax": float(load.min() / max(load.max(), 1)),      # 1 = perfect balance
        "served": float((A.sum(1) > 0).mean()),
        "experts/tok": float(A.sum(1).mean()),                 # exposes >k violations
    }


def evaluate_trace(npz_path: str, manifest: dict, cap_factor: float,
                   block_size: int, max_tokens: int, score_mode: str):
    data = np.load(npz_path)
    k = int(manifest.get("top_k") or 2)
    per_router = {name: [] for name in ROUTERS}

    for key in data.files:                          # one array per router layer
        logits = np.asarray(data[key], dtype=np.float64)
        N = min(logits.shape[0], max_tokens)
        for start in range(0, N, block_size):
            G = to_scores(logits[start:start + block_size], score_mode)
            n, m = G.shape
            if n < m:                               # block too small to be meaningful
                continue
            C = int(np.ceil(cap_factor * n * k / m))
            opt = optimal_value(G, C, k)
            for name, R in ROUTERS.items():
                A = R(m, C, k).route_batch(G)
                per_router[name].append(metrics(G, A, C, opt))
    return per_router, k


def aggregate(per_router: dict) -> dict:
    summary = {}
    for name, recs in per_router.items():
        if recs:
            summary[name] = {key: float(np.nanmean([r[key] for r in recs]))
                             for key in recs[0]}
    return summary


def make_figures(summary: dict, out_dir: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summary.keys())
    panels = [
        ("Retained score (ALG / OPT)", [summary[n]["retained"] for n in names],
         1 - 1 / np.e),
        ("Load Gini (lower = better)", [summary[n]["gini"] for n in names], None),
        ("Max-load / capacity", [summary[n]["Lmax/C"] for n in names], None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, vals, ref) in zip(axes, panels):
        bars = ax.bar(names, vals, color="#9ca3af")
        bars[0].set_color("#2563eb")                # highlight WFPD
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        if ref is not None:
            ax.axhline(ref, ls="--", c="#dc2626", lw=1)
            ax.text(0, ref, f"  1-1/e={ref:.3f}", color="#dc2626",
                    va="bottom", fontsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    pdf = os.path.join(out_dir, "comparison.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(out_dir, "comparison.png"), dpi=150)
    plt.close(fig)
    return pdf


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(root, "data", "traces"))
    ap.add_argument("--cap-factor", type=float, default=1.25)
    ap.add_argument("--score", default="softmax", choices=["softmax", "relu", "shift"])
    ap.add_argument("--block-size", type=int, default=2048)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--results", default=os.path.join(root, "results"))
    ap.add_argument("--figures", default=os.path.join(root, "figures"))
    args = ap.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.traces, "*_gate_logits.npz")))
    if not npz_files:
        sys.exit(f"No traces in {args.traces}; run trace_extract.py first.")

    all_records = {name: [] for name in ROUTERS}
    k_used = None
    for npz in npz_files:
        man_path = npz.replace("_gate_logits.npz", "_manifest.json")
        manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
        per_router, k_used = evaluate_trace(
            npz, manifest, args.cap_factor, args.block_size,
            args.max_tokens, args.score)
        for name, recs in per_router.items():
            all_records[name].extend(recs)

    summary = aggregate(all_records)
    if not summary:
        sys.exit("No evaluable blocks (traces too small?).")

    cols = ["retained", "gini", "Lmax/C", "minmax", "served", "experts/tok"]
    header = f"{'router':14s} " + " ".join(f"{c:>12s}" for c in cols)
    print(f"\nscore={args.score}   top-k={k_used}   cap-factor={args.cap_factor}")
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(f"{name:14s} " + " ".join(f"{s[c]:12.4f}" for c in cols))

    os.makedirs(args.results, exist_ok=True)
    csv_path = os.path.join(args.results, "eval_summary.csv")
    with open(csv_path, "w") as f:
        f.write("router," + ",".join(cols) + "\n")
        for name, s in summary.items():
            f.write(name + "," + ",".join(f"{s[c]:.6f}" for c in cols) + "\n")

    os.makedirs(args.figures, exist_ok=True)
    pdf = make_figures(summary, args.figures)
    print(f"\nsaved table  -> {csv_path}")
    print(f"saved figure -> {pdf}")


if __name__ == "__main__":
    main()
