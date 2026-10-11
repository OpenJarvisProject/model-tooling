"""Compression-combination experiment: one condition per invocation.

Stages run in memory, in the order given, e.g. `P:1.25 F:1.25`
(prune layers with a 1.25x val-perplexity budget relative to that
stage's input, then factorize with the same). Every stage's accept/
reject decisions use the same wikitext-2 validation metric; the
held-out wikitext-2 test set is measured after each stage and is the
only thing used to compare conditions.

Quantization is always applied last (pruning's calibration and
factorization's SVD both need full-precision weights), simulated here
as symmetric round-to-nearest with one scale per group of 32 weights
(roughly llama.cpp's Q4_0). Simulated rather than real llama-quantize
because GGUF/llama.cpp has no representation for factorized layers,
so real quantization can't be applied to F conditions at all; the
simulation is checked against real Q4_K_M separately on the non-F
conditions.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from evalharness import param_count, test_ppl, val_ppl_fn
from factorize import factorize_iteratively
from prune_layers import prune_iteratively

KEEP_FRACTION = 0.4  # below the square-matrix compression breakeven (0.5)
GROUP = 32


def fake_quantize_(model, bits: int):
    qmax = 2 ** (bits - 1) - 1
    seen = set()
    for mod in model.modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        w = mod.weight
        if id(w) in seen:  # tied lm_head/embedding: quantize once
            continue
        seen.add(id(w))
        rows, cols = w.shape
        pad = (-cols) % GROUP
        g = nn.functional.pad(w.data, (0, pad)).view(rows, -1, GROUP)
        scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
        q = (g / scale).round().clamp(-qmax, qmax) * scale
        w.data = q.view(rows, -1)[:, :cols].contiguous()


def est_mb(params: int, bits: float) -> float:
    # +0.5 bits/weight for one fp16 scale per group of 32 (quantized cases).
    effective = bits if bits >= 16 else bits + 16 / GROUP
    return params * effective / 8 / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("stages", nargs="*", help="e.g. P:1.25 F:1.25 (empty = original)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--quant-bits", default="4,3")
    args = parser.parse_args()

    out_dir = Path("outputs/combo")
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    val = val_ppl_fn(args.model, n_chunks=4)

    def snapshot(label):
        params = param_count(model)
        row = {
            "stage": label,
            "params": params,
            "fp16_mb": round(est_mb(params, 16), 1),
            "val_ppl": round(val(model), 3),
            "test_ppl": round(test_ppl(model, args.model), 3),
        }
        print("RESULT", json.dumps(row), flush=True)
        return row

    results = {"name": args.name, "stages": args.stages, "rows": [snapshot("start")]}

    for spec in args.stages:
        kind, ratio = spec.split(":")
        t0 = time.time()
        if kind == "P":
            model = prune_iteratively(
                model, tok, "cpu", num_to_remove=len(model.model.layers) - 2,
                max_ppl_ratio=float(ratio), ppl_fn=val,
            )
        elif kind == "F":
            factorize_iteratively(
                model, tok, "cpu", KEEP_FRACTION, float(ratio), ppl_fn=val,
            )
        else:
            raise ValueError(f"unknown stage {spec}")
        row = snapshot(spec)
        row["stage_seconds"] = round(time.time() - t0)
        results["rows"].append(row)

    if not any(s.startswith("F") for s in args.stages):
        # Plain HF checkpoint so real llama-quantize can validate the simulation.
        model.save_pretrained(out_dir / f"{args.name}-hf")
        tok.save_pretrained(out_dir / f"{args.name}-hf")

    for bits in (int(b) for b in args.quant_bits.split(",")):
        q = copy.deepcopy(model)
        fake_quantize_(q, bits)
        params = param_count(q)
        row = {
            "stage": f"Q{bits}(sim)",
            "params": params,
            "est_mb": round(est_mb(params, bits), 1),
            "test_ppl": round(test_ppl(q, args.model), 3),
        }
        print("RESULT", json.dumps(row), flush=True)
        results["rows"].append(row)
        del q

    (out_dir / f"{args.name}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
