# moe-online-routing

**Online Capacitated Routing for Mixture-of-Experts: A Single-Pass
Water-Filling Router for Load Balancing without Auxiliary Losses**

Reference implementation and reproducibility package for **WFPD**, a
lightweight single-pass water-filling router for Mixture-of-Experts (MoE)
expert load balancing. Target venue: JPDC (Elsevier).

## What this is

We formulate inference-time MoE routing as an **online capacitated packing**
problem and use its offline linear-programming optimum purely as an
*evaluation benchmark*. WFPD attaches a price (water level) to each expert
that rises as it fills, and routes each token to its top-$k$ positive-surplus
experts, so load balancing emerges from the routing rule — no auxiliary loss,
no token-dropping heuristic, no tuning — at $O(m + k\log C)$ time and $O(m)$
space per token.

### Scope (please read)

This is an **empirical / systems** contribution. WFPD is a heuristic with
constant per-token cost that performs strongly on real routing traces. It
does **not** claim a worst-case competitive guarantee: under adversarially
ordered inputs with continuous gate scores, no fixed online rule can
guarantee a constant fraction of the offline optimum, and WFPD's monotone
price update does not reach near-optimality under random order either. A
variant with provable guarantees (via a dual mirror-descent price update) is
left to future work. See the Scope section of the paper for details.

## Structure

- `src/moe_routing/algorithms.py` — WFPD router (core algorithm)
- `src/moe_routing/baselines.py` — Token-choice, Expert-choice, LPR/OT baselines
- `src/moe_routing/evaluate.py` — offline LP optimum (HiGHS) + retained / Gini / balance metrics
- `src/moe_routing/trace_extract.py` — gate-logit extraction from HF MoE models (tiny model by default)
- `notebooks/extract_gate_traces.ipynb` — Colab GPU pipeline that produced the OLMoE-1B-7B traces (4-bit, WikiText-2)
- `data/` — cached gate-logit traces (not versioned)
- `results/` — evaluation summary CSV and figures
- `figures/` — paper figures
- `paper/main.tex` — manuscript (Elsevier `elsarticle`)
- `tests/` — unit / property tests

## Install

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```
python src/moe_routing/algorithms.py     # WFPD smoke test
python src/moe_routing/baselines.py      # baseline comparison on synthetic G
python src/moe_routing/evaluate.py       # offline OPT + metrics on cached traces
python src/moe_routing/trace_extract.py  # extract gate logits (tiny model by default)
```

To regenerate the real traces used in the paper, run
`notebooks/extract_gate_traces.ipynb` on a GPU (e.g. Colab), then place the
resulting `*.npz` under `data/traces/`.

## Results (OLMoE-1B-7B, m=64, k=8, aggregated over all MoE layers)

| Router        | Retained | Gini  | Min/Max | Exp./tok |
| ------------- | -------- | ----- | ------- | -------- |
| WFPD (ours)   | 0.953    | 0.106 | 0.380   | 7.78     |
| Token-choice  | 0.877    | 0.245 | 0.069   | 6.75     |
| LPR (OT)      | 0.969    | 0.141 | 0.267   | 8.00     |

WFPD improves on the deployed token-choice gate on both retained gate score
and load balance, and matches the optimal-transport solver to within ~1.6
points of gate score while balancing better — in a single online pass at
constant per-token cost.

## Citation

```bibtex
@unpublished{liu_wfpd,
  author = {Kefan Liu},
  title  = {Online Capacitated Routing for Mixture-of-Experts: A Single-Pass
            Water-Filling Router for Load Balancing without Auxiliary Losses},
  note   = {Manuscript under submission to JPDC},
  year   = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
