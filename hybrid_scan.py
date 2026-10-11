"""Per-tensor-group sensitivity scan for hybrid quantization.

Start from one base quant type (+imatrix) and, one variant at a time,
raise exactly ONE tensor group to a higher-precision type via
llama-quantize --tensor-type. Each variant is scored with the same
wikitext-2 test perplexity as quant_study.py, so the payoff of each
group is directly comparable as perplexity recovered per extra MB.

  python hybrid_scan.py qwen2.5-3b gguf-compare/original-3b-f16.gguf \
      outputs/quant-study/qwen2.5-3b/imatrix.gguf IQ2_M \
      ffn_up=iq3_s ffn_gate=iq3_s ...

Each positional override after the base type is its own variant; join
several with '+' to apply them together (e.g. for a final hybrid mix).
"""

import argparse
import json
from pathlib import Path

from quant_study import BIN, HERE, perplexity, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("f16_gguf", type=Path)
    parser.add_argument("imatrix", type=Path)
    parser.add_argument("base_type")
    parser.add_argument("variants", nargs="*", help="pattern=type[+pattern=type...]")
    args = parser.parse_args()

    out_dir = HERE / "outputs" / "hybrid-scan" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    gguf = out_dir / "tmp.gguf"
    results = []

    for variant in ["base", *args.variants]:
        cmd = [BIN / "llama-quantize", "--imatrix", args.imatrix]
        if variant != "base":
            for override in variant.split("+"):
                cmd += ["--tensor-type", override]
        run(cmd + [args.f16_gguf, gguf, args.base_type])
        ppl, err = perplexity(gguf)
        row = {"model": args.label, "base": args.base_type, "variant": variant,
               "mb": round(gguf.stat().st_size / 1e6, 1), "ppl": ppl, "ppl_err": err}
        if results:
            base = results[0]
            extra_mb = row["mb"] - base["mb"]
            row["ppl_gain_per_100mb"] = round(100 * (base["ppl"] - ppl) / extra_mb, 3) if extra_mb > 0 else None
        print("RESULT", json.dumps(row), flush=True)
        results.append(row)
        gguf.unlink()

    (out_dir / f"{args.base_type}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
