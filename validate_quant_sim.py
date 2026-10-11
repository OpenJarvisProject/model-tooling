"""Checks combine.py's simulated quantization against real llama-quantize,
on the conditions that *can* go through GGUF (no factorized layers).
Real GGUF files are loaded back via transformers' gguf_file support so
they're scored by the exact same evalharness test metric.
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from evalharness import test_ppl

CONVERT = "/tmp/llama-cpp-convert/convert_hf_to_gguf.py"
QUANTIZE = "/tmp/llama-cpp-convert/build/bin/llama-quantize"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = Path("outputs/combo")

results = []
for name in sys.argv[1:]:
    hf_dir = OUT / f"{name}-hf"
    f16 = OUT / f"{name}-f16.gguf"
    subprocess.run([sys.executable, CONVERT, "--outtype", "f16", "--outfile", str(f16), str(hf_dir)],
                   check=True, capture_output=True)
    for qtype in ("Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K"):
        gguf = OUT / f"{name}-{qtype}.gguf"
        subprocess.run([QUANTIZE, str(f16), str(gguf), qtype], check=True, capture_output=True)
        model = AutoModelForCausalLM.from_pretrained(str(OUT), gguf_file=gguf.name, dtype=torch.float32)
        row = {
            "condition": name,
            "real_quant": qtype,
            "gguf_mb": round(gguf.stat().st_size / 1e6, 1),
            "test_ppl": round(test_ppl(model, TOKENIZER), 3),
        }
        print("RESULT", json.dumps(row), flush=True)
        results.append(row)
        del model
        gguf.unlink()
    f16.unlink()

(OUT / "quant_sim_validation.json").write_text(json.dumps(results, indent=2))
