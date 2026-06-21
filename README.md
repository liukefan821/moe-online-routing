# moe-online-routing

**Online Capacitated Routing for Mixture-of-Experts: Max-Load Bounds and
Competitive Gate-Score Retention without Auxiliary Losses**

Reference implementation + reproducibility package for the WFPD
(Water-Filling Primal-Dual) router. Target venue: JPDC.

## Structure
- `src/moe_routing/algorithms.py`     — WFPD router (core algorithm)
- `src/moe_routing/baselines.py`      — Token-choice, Expert-choice, LPR/OT baselines
- `src/moe_routing/trace_extract.py`  — gate-logit extraction from HF MoE models
- `data/`                             — cached gate-logit traces (not versioned)
- `notebooks/`                        — analysis & paper-figure generation
- `tests/`                            — unit / property tests

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start
```bash
python src/moe_routing/algorithms.py     # WFPD smoke test
python src/moe_routing/baselines.py      # baseline comparison on synthetic G
python src/moe_routing/trace_extract.py  # extract gate logits (tiny model by default)
```
