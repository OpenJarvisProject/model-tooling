"""One-variable calibration experiment on Qwen2.5-1.5B-Instruct (Q4_K_M fixed).

Only the imatrix calibration text changes:
  wiki   wikitext-2 train (the earlier studies' choice)
  tasks  OJP's own prompts: tool calls, routing, chat (calib-ojp-tasks.txt)
  mixed  half and half (calib-ojp-mixed.txt)
Scored two ways: wikitext-2 test perplexity (general text) and KL divergence
against the f16 model on a held-out text built from the assistant's test
requests (eval-ojp-tasks.txt; none of it is in the calibration text).
"""
import json, os, re, subprocess, sys
from pathlib import Path

BIN = Path("/media/openjarvisproject-model-tooling/llama.cpp/build-vk/bin")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json"}
HERE = Path(__file__).parent
F16 = HERE / "gguf-compare/original-1.5b-f16.gguf"
OUT = HERE / "outputs/calib-1.5b"; OUT.mkdir(parents=True, exist_ok=True)

def run(cmd):
    p = subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f"{cmd[0]} failed:\n{p.stderr[-1500:]}")
    return p.stdout + p.stderr

def ppl(m):
    out = run([BIN/"llama-perplexity","-m",m,"-f",HERE/"wikitext2-test.txt","-c","512","--chunks","64","-ngl","99"])
    r = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", out); return float(r.group(1)), float(r.group(2))

base = OUT/"eval.kld"
if not base.exists():
    run([BIN/"llama-perplexity","-m",F16,"-f",HERE/"eval-ojp-tasks.txt","-c","2048","-ngl","99","--kl-divergence-base",base])

def kl(m):
    out = run([BIN/"llama-perplexity","-m",m,"-f",HERE/"eval-ojp-tasks.txt","-c","2048","-ngl","99","--kl-divergence-base",base,"--kl-divergence"])
    k = re.search(r"Mean\s+KLD:\s+([0-9.]+)\s+±\s+([0-9.]+)", out); t = re.search(r"Same top p:\s+([0-9.]+)\s+±\s+([0-9.]+)", out)
    return float(k.group(1)), float(k.group(2)), float(t.group(1))

rows = []
sets = {"wiki": HERE/"outputs/quant-study/qwen2.5-1.5b/imatrix.gguf"}
for name, corpus in (("tasks","calib-ojp-tasks.txt"),("mixed","calib-ojp-mixed.txt")):
    im = OUT/f"imat-{name}.gguf"
    if not im.exists():
        run([BIN/"llama-imatrix","-m",F16,"-f",HERE/corpus,"-o",im,"--chunks","100","-c","512","-ngl","99"])
    sets[name] = im
for name, im in sets.items():
    q = OUT/f"q4km-{name}.gguf"
    run([BIN/"llama-quantize","--imatrix",im,F16,q,"Q4_K_M"])
    p, pe = ppl(q); k, ke, top = kl(q)
    row = {"calibration": name, "mb": round(q.stat().st_size/1e6,1), "wikitext_ppl": p, "ppl_err": pe, "task_kld": k, "task_kld_err": ke, "task_same_top": top}
    print("RESULT", json.dumps(row), flush=True); rows.append(row)
    if name != "tasks" and name != "mixed": q.unlink()  # keep only the new ones
(OUT/"results.json").write_text(json.dumps(rows, indent=2))
base.unlink()
