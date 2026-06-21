"""Extract and cache gating (router) logits from open HuggingFace MoE models.

Uses the official, version-stable API: model(..., output_router_logits=True)
returns outputs.router_logits, a tuple of per-layer logits tensors of shape
(num_tokens, num_experts). This avoids fragile module hooking and works across
transformers v4/v5 and the Mixtral / Qwen-MoE families.

Writes concatenated per-layer logits to data/traces/ as a compressed .npz plus a
JSON manifest. Each cached matrix is exactly the G in (n, m) form consumed by
baselines.py / evaluate.py.

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


def _load_model(model_id: str, dtype: torch.dtype):
    """Load a causal-LM checkpoint, tolerating the v4 (torch_dtype) /
    v5 (dtype) keyword rename."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)


@torch.no_grad()
def extract(model_id: str, device: str, dtype: torch.dtype,
            max_tokens: int, num_prompts: int):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = _load_model(model_id, dtype).to(device).eval()
    model.config.output_router_logits = True
    cfg = model.config

    buffers: dict[int, list[np.ndarray]] = {}
    for prompt in SAMPLE_PROMPTS[:num_prompts]:
        ids = tok(prompt, return_tensors="pt",
                  truncation=True, max_length=max_tokens).to(device)
        out = model(**ids, output_router_logits=True)
        rl = getattr(out, "router_logits", None)
        if not rl:
            raise RuntimeError(
                "Model returned no router_logits. Use a MoE checkpoint that "
                "supports output_router_logits (e.g. Mixtral or Qwen-MoE).")
        for i, logit in enumerate(rl):
            if logit is None:
                continue
            arr = (logit.detach().to(torch.float32)
                   .reshape(-1, logit.shape[-1]).cpu().numpy().astype(np.float16))
            buffers.setdefault(i, []).append(arr)

    traces = {f"layer_{i:02d}": np.concatenate(v, axis=0)
              for i, v in sorted(buffers.items()) if v}
    if not traces:
        raise RuntimeError("No router logits captured.")

    inferred = next(iter(traces.values())).shape[1]
    n_layers = len(traces)
    manifest = {
        "model_id": model_id,
        "num_experts": int(_num_experts_from_config(cfg) or inferred),
        "top_k": int(getattr(cfg, "num_experts_per_tok", 0)
                     or getattr(cfg, "moe_top_k", 0) or 0),
        "num_router_layers": n_layers,
        "hidden_size": int(getattr(cfg, "hidden_size", 0)),
        "tokens_per_layer": int(sum(a.shape[0] for a in traces.values())
                                // max(n_layers, 1)),
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
