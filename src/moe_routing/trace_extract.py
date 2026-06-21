"""Extract and cache gating (router) logits from open HuggingFace MoE models.

Registers forward hooks on every router gate (the nn.Linear mapping hidden ->
per-expert logits) inside each sparse-MoE block, runs a handful of prompts, and
writes the concatenated per-layer logits to data/traces/ as a compressed .npz
plus a JSON manifest. Each cached matrix is exactly the G in (n, m) form
consumed by baselines.py / algorithms.py.

Auto-detects gate modules, so it works for Mixtral and Qwen-MoE families.
DEFAULT_MODEL is a tiny MoE for a laptop smoke-run; swap --model for the full
checkpoint on a GPU node to obtain real traces.
"""
from __future__ import annotations
import argparse
import json
import os
import re

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Loads on a laptop for a pipeline smoke-run. Real traces:
#   "mistralai/Mixtral-8x7B-v0.1"   or   "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_MODEL = "hf-internal-testing/tiny-random-MixtralForCausalLM"

SAMPLE_PROMPTS = [
    "The theory of online algorithms studies decision making under uncertainty.",
    "Mixture-of-experts models route each token to a small subset of experts.",
    "Primal-dual analysis yields competitive ratios for matching problems.",
    "Load balancing across experts is critical for inference throughput.",
    "A water-filling price rises as an expert approaches its capacity.",
    "Sublinear data structures power large-scale streaming systems.",
    "Optimal transport gives balanced assignments without auxiliary losses.",
    "Reproducibility requires caching the exact gate logits used in evaluation.",
]


def _num_experts_from_config(cfg) -> int | None:
    for attr in ("num_local_experts", "num_experts", "n_routed_experts",
                 "moe_num_experts"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 1:
            return v
    return None


def find_router_modules(model: nn.Module, num_experts: int | None):
    """Return [(qualified_name, gate_linear)] for every MoE router gate."""
    routers = []
    for name, module in model.named_modules():
        gate = getattr(module, "gate", None)
        if isinstance(gate, nn.Linear):
            if num_experts is None or gate.out_features == num_experts:
                routers.append((f"{name}.gate", gate))
    return routers


@torch.no_grad()
def extract(model_id: str, device: str, dtype: torch.dtype,
            max_tokens: int, num_prompts: int):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    cfg = model.config
    num_experts = _num_experts_from_config(cfg)
    routers = find_router_modules(model, num_experts)
    if not routers:
        raise RuntimeError("No MoE router gate found; is this a MoE checkpoint?")

    buffers: dict[str, list[np.ndarray]] = {name: [] for name, _ in routers}

    def make_hook(name: str):
        def hook(_m, _inp, out):
            logits = out[0] if isinstance(out, tuple) else out
            arr = logits.detach().to(torch.float32).reshape(-1, logits.shape[-1])
            buffers[name].append(arr.cpu().numpy().astype(np.float16))
        return hook

    handles = [g.register_forward_hook(make_hook(n)) for n, g in routers]
    try:
        for prompt in SAMPLE_PROMPTS[:num_prompts]:
            ids = tok(prompt, return_tensors="pt",
                      truncation=True, max_length=max_tokens).to(device)
            model(**ids)
    finally:
        for h in handles:
            h.remove()

    traces = {re.sub(r"[^0-9a-zA-Z]+", "_", n): np.concatenate(v, axis=0)
              for n, v in buffers.items() if v}
    inferred = next(iter(traces.values())).shape[1]
    n_layers = max(len(traces), 1)
    manifest = {
        "model_id": model_id,
        "num_experts": int(num_experts or inferred),
        "top_k": int(getattr(cfg, "num_experts_per_tok", 0)
                     or getattr(cfg, "moe_top_k", 0) or 0),
        "num_router_layers": len(traces),
        "hidden_size": int(getattr(cfg, "hidden_size", 0)),
        "tokens_per_layer": int(sum(a.shape[0] for a in traces.values()) // n_layers),
        "dtype": "float16",
        "layer_keys": sorted(traces.keys()),
    }
    return traces, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=len(SAMPLE_PROMPTS))
    ap.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="data/traces")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    traces, manifest = extract(args.model, args.device, dtype,
                               args.max_tokens, args.num_prompts)

    os.makedirs(args.out, exist_ok=True)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", args.model)
    npz_path = os.path.join(args.out, f"{tag}_gate_logits.npz")
    man_path = os.path.join(args.out, f"{tag}_manifest.json")
    np.savez_compressed(npz_path, **traces)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"saved {len(traces)} router layers -> {npz_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
