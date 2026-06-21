"""Extract and cache gating (router) logits from open HuggingFace MoE models.

Uses the official, version-stable API: model(..., output_router_logits=True)
returns outputs.router_logits, a tuple of per-layer logits of shape
(num_tokens, num_experts). Works across transformers v4/v5 and the
Mixtral / Qwen-MoE / OLMoE families.

Text source:
  * default  -> 8 built-in prompts (quick pipeline check; too few tokens for
               trustworthy load statistics)
  * --dataset -> stream real text from a HuggingFace dataset (e.g. wikitext) to
               get thousands of tokens per layer, so Gini / min-max / retained
               estimates are stable. Recommended for any reported numbers.

Writes per-layer logits to data/traces/<tag>_gate_logits.npz + a JSON manifest.

Examples:
  python src/moe_routing/trace_extract.py                       # tiny smoke test
  python src/moe_routing/trace_extract.py \\
      --model allenai/OLMoE-1B-7B-0924 --device cpu --dtype bfloat16 \\
      --dataset wikitext --dataset-config wikitext-2-raw-v1 \\
      --num-docs 32 --max-tokens 64                             # robust real run
"""
from __future__ import annotations
import argparse
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    """Load a causal-LM, tolerating the v4 (torch_dtype) / v5 (dtype) rename."""
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=True)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True)


def get_texts(args) -> list[str]:
    if not args.dataset:
        return SAMPLE_PROMPTS[: args.num_prompts]
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets first:  pip install datasets")
    ds = load_dataset(args.dataset, args.dataset_config,
                      split=args.dataset_split, streaming=True)
    texts: list[str] = []
    for ex in ds:
        t = (ex.get(args.text_field) or "").strip()
        if len(t) >= 32:                       # skip empty lines / headers
            texts.append(t)
        if len(texts) >= args.num_docs:
            break
    if not texts:
        raise SystemExit("No usable texts pulled from the dataset.")
    return texts


@torch.no_grad()
def extract(model, tok, texts: list[str], device: str, max_tokens: int):
    model.config.output_router_logits = True
    try:
        from tqdm import tqdm
        iterator = tqdm(texts, desc="extracting")
    except ImportError:
        iterator = texts

    buffers: dict[int, list[np.ndarray]] = {}
    for text in iterator:
        ids = tok(text, return_tensors="pt",
                  truncation=True, max_length=max_tokens).to(device)
        out = model(**ids, output_router_logits=True)
        rl = getattr(out, "router_logits", None)
        if not rl:
            raise SystemExit(
                "Model returned no router_logits. Use a MoE checkpoint that "
                "supports output_router_logits (e.g. Mixtral / Qwen-MoE / OLMoE).")
        for i, logit in enumerate(rl):
            if logit is None:
                continue
            arr = (logit.detach().to(torch.float32)
                   .reshape(-1, logit.shape[-1]).cpu().numpy().astype(np.float16))
            buffers.setdefault(i, []).append(arr)

    traces = {f"layer_{i:02d}": np.concatenate(v, axis=0)
              for i, v in sorted(buffers.items()) if v}
    if not traces:
        raise SystemExit("No router logits captured.")
    return traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=len(SAMPLE_PROMPTS),
                    help="how many built-in prompts to use (no --dataset)")
    # dataset source (recommended for reported numbers)
    ap.add_argument("--dataset", default=None,
                    help="HF dataset name, e.g. wikitext")
    ap.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    ap.add_argument("--dataset-split", default="train")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--num-docs", type=int, default=64,
                    help="how many dataset documents to stream")
    ap.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="data/traces")
    args = ap.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = _load_model(args.model, dtype).to(args.device).eval()

    texts = get_texts(args)
    traces = extract(model, tok, texts, args.device, args.max_tokens)
    cfg = model.config

    n_layers = len(traces)
    manifest = {
        "model_id": args.model,
        "num_experts": int(_num_experts_from_config(cfg)
                           or next(iter(traces.values())).shape[1]),
        "top_k": int(getattr(cfg, "num_experts_per_tok", 0)
                     or getattr(cfg, "moe_top_k", 0) or 0),
        "num_router_layers": n_layers,
        "hidden_size": int(getattr(cfg, "hidden_size", 0)),
        "tokens_per_layer": int(sum(a.shape[0] for a in traces.values())
                                // max(n_layers, 1)),
        "num_docs": len(texts),
        "source": args.dataset or "builtin_prompts",
        "dtype": "float16",
        "layer_keys": sorted(traces.keys()),
    }

    os.makedirs(args.out, exist_ok=True)
    tag = re.sub(r"[^0-9a-zA-Z]+", "_", args.model)
    npz_path = os.path.join(args.out, f"{tag}_gate_logits.npz")
    man_path = os.path.join(args.out, f"{tag}_manifest.json")
    np.savez_compressed(npz_path, **traces)
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nsaved {n_layers} router layers -> {npz_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
