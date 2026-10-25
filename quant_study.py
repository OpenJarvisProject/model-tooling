"""Training-free quantization study, one variable at a time, all GGUF.

  step 1  imatrix on/off at fixed quant type   (Q4_K_M, Q3_K_M, Q2_K)
  step 2  I-quant types, all with imatrix       (IQ4_XS, IQ3_M, IQ3_XXS, IQ2_M)
  step 3  token-embedding precision override    (on Q4_K_M+imat, IQ3_M+imat)

Reference is Q8_0. The imatrix is calibrated on wikitext-2 *train*;
every reported perplexity is llama-perplexity on wikitext-2 *test*
(64 chunks x 512 tokens), fully offloaded to the GPU via Vulkan. This
harness's absolute numbers are NOT comparable to evalharness.py's
PyTorch numbers (different scoring) - compare within this study only.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

BIN = Path("/media/openjarvisproject-model-tooling/llama.cpp/build-vk/bin")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json"}
HERE = Path(__file__).parent


def run(cmd):
    proc = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    return proc.stdout + proc.stderr


def perplexity(gguf: Path) -> tuple[float, float]:
    out = run([BIN / "llama-perplexity", "-m", gguf, "-f", HERE / "wikitext2-test.txt",
               "-c", "512", "--chunks", "64", "-ngl", "99"])
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"no PPL in output:\n{out[-1500:]}")
    return float(m.group(1)), float(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("f16_gguf", type=Path)
    parser.add_argument("--embed-types", required=True, help="comma list, must fit the model's row length")
    args = parser.parse_args()

    out_dir = HERE / "outputs" / "quant-study" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    def quantize_and_score(name, qtype, imatrix=None, embed=None, keep=False):
        gguf = out_dir / f"{name}.gguf"
        cmd = [BIN / "llama-quantize"]
        if imatrix:
            cmd += ["--imatrix", imatrix]
        if embed:
            cmd += ["--token-embedding-type", embed]
        run(cmd + [args.f16_gguf, gguf, qtype])
        ppl, err = perplexity(gguf)
        row = {"model": args.label, "variant": name, "qtype": qtype, "imatrix": bool(imatrix),
               "token_embd": embed or "default", "mb": round(gguf.stat().st_size / 1e6, 1),
               "ppl": ppl, "ppl_err": err}
        print("RESULT", json.dumps(row), flush=True)
        results.append(row)
        if not keep:
            gguf.unlink()
        return gguf

    ref = quantize_and_score("Q8_0", "Q8_0", keep=True)

    imat = out_dir / "imatrix.gguf"
    run([BIN / "llama-imatrix", "-m", ref, "-f", HERE / "wikitext2-train.txt",
         "-o", imat, "--chunks", "100", "-c", "512", "-ngl", "99"])
    ref.unlink()

    for qtype in ("Q4_K_M", "Q3_K_M", "Q2_K"):
        quantize_and_score(qtype, qtype)
        quantize_and_score(f"{qtype}+imat", qtype, imatrix=imat)

    for qtype in ("IQ4_XS", "IQ3_M", "IQ3_XXS", "IQ2_M"):
        quantize_and_score(f"{qtype}+imat", qtype, imatrix=imat)

    for base in ("Q4_K_M", "IQ3_M"):
        for embed in args.embed_types.split(","):
            quantize_and_score(f"{base}+imat+emb_{embed}", base, imatrix=imat, embed=embed)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
