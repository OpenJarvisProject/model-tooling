"""Deep comparison of a few finalist GGUFs against the unquantized original.

Per model, all on the GPU via llama.cpp's own tools:
  - KL divergence + same-top-token % vs the f16 original (wikitext-2 test)
  - HellaSwag, Winogrande, ARC, MMLU, TruthfulQA accuracy
  - agent_tests.py (arithmetic, instructions, tool-call JSON, router labels)
    through llama-server, temperature 0
Speed (llama-bench) is a separate `--speed` pass, run only when the CPU
is otherwise idle so CPU timings aren't polluted by other jobs.
"""

import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

import agent_tests

BIN = Path("/tmp/llama-cpp-convert/build-vk/bin")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json"}
HERE = Path(__file__).parent
DATA = HERE / "bench-data"
OUT = HERE / "outputs" / "close-compare"


def run(cmd):
    proc = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{proc.stderr[-2000:]}")
    return proc.stdout + proc.stderr


def ppl_tool(model, *args):
    return run([BIN / "llama-perplexity", "-m", model, "-ngl", "99", *args])


def kl_divergence(model, base_logits):
    out = ppl_tool(model, "--kl-divergence-base", base_logits, "--kl-divergence")
    kld = re.search(r"Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)", out)
    top = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)", out)
    return {"mean_kld": float(kld.group(1)), "same_top_pct": float(top.group(1)),
            "same_top_err": float(top.group(2))}


def hellaswag(model):
    out = ppl_tool(model, "--hellaswag", "-f", DATA / "hellaswag_val_full.txt", "--hellaswag-tasks", "1000")
    n, acc, lo, hi = re.findall(r"^(\d+)\t([0-9.]+)%\t\[([0-9.]+)%, ([0-9.]+)%\]", out, re.M)[-1]
    return {"acc": float(acc), "ci95": [float(lo), float(hi)], "n": int(n)}


def winogrande(model):
    out = ppl_tool(model, "--winogrande", "-f", DATA / "winogrande-debiased-eval.csv")
    n, acc = re.findall(r"^(\d+)\t([0-9.]+)\t", out, re.M)[-1]
    n, p = int(n), float(acc) / 100
    return {"acc": float(acc), "err": round(100 * math.sqrt(p * (1 - p) / n), 2), "n": n}


def multiple_choice(model, binfile):
    out = ppl_tool(model, "--multiple-choice", "-bf", DATA / binfile)
    acc, err = re.search(r"Final result: ([0-9.]+) \+/- ([0-9.]+)", out).groups()
    return {"acc": float(acc), "err": float(err)}


def agent(model):
    port = 8099
    server = subprocess.Popen([str(BIN / "llama-server"), "-m", str(model), "-ngl", "99", "--port", str(port),
                               "-c", "4096", "--jinja"], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server never became healthy")
        return agent_tests.run(f"http://127.0.0.1:{port}")
    finally:
        server.terminate()
        server.wait()


def speed(model):
    out = run([BIN / "llama-bench", "-m", model, "-ngl", "0,99", "-p", "256", "-n", "64", "-o", "json"])
    rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    return {f"{'gpu' if r['n_gpu_layers'] else 'cpu'}_{'prompt' if r['n_prompt'] else 'gen'}_tok_s": round(r["avg_ts"], 1)
            for r in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="label=path.gguf; the FIRST is the reference original")
    parser.add_argument("--speed", action="store_true", help="only run llama-bench")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = [m.split("=", 1) for m in args.models]

    if args.speed:
        for label, path in models:
            row = {"model": label, "speed": speed(path)}
            print("RESULT", json.dumps(row), flush=True)
        return

    base_logits = OUT / "base.kld"
    ppl_tool(models[0][1], "-f", HERE / "wikitext2-test.txt", "-c", "512", "--chunks", "20",
             "--kl-divergence-base", base_logits)

    for label, path in models:
        row = {"model": label, "mb": round(Path(path).stat().st_size / 1e6, 1)}
        if label != models[0][0]:
            row["vs_original"] = kl_divergence(path, base_logits)
        row["hellaswag"] = hellaswag(path)
        row["winogrande"] = winogrande(path)
        row["arc"] = multiple_choice(path, "arc-combined-validation.bin")
        row["mmlu"] = multiple_choice(path, "mmlu-validation.bin")
        row["truthfulqa"] = multiple_choice(path, "truthful-qa-validation.bin")
        row["agent"] = agent(path)
        print("RESULT", json.dumps(row), flush=True)
        (OUT / f"{label}.json").write_text(json.dumps(row, indent=2))

    base_logits.unlink()


if __name__ == "__main__":
    main()
