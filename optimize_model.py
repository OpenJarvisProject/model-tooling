#!/usr/bin/env python3
"""One model through the whole OJP optimisation recipe, unoptimised vs optimised.

  optimize_model.py <name> <source.gguf (BF16/F16)> <template> <role: router|main> [--emb q5_K]

Variants built from the source (all Q4_K_M, one change at a time):
  stock      no imatrix, full vocabulary             (the unoptimised baseline)
  imat       imatrix from wikitext-2 train, full vocabulary
  trimmed    imatrix + vocabulary trimmed to the corpus tokens (+ optional --emb type)
Per variant: file size, bits per byte on English / European / other-script text,
the tame-creative-prompt refusal count, and (router) probe accuracy, (main) the
tool test. Results go to /media/ojp-eval/<name>/summary.jsonl.
"""
import argparse, json, math, os, re, subprocess, sys
from pathlib import Path

R = Path("/media/openjarvisproject-model-tooling")
B = R / "llama.cpp/build-vk/bin"
OJP = Path("/media/ojp")
# A frozen copy of ojp-cli, so rebuilding it during a long run cannot change the measurements.
CLI = os.environ.get("OJP_CLI", "/media/ojp-eval/bin/ojp-cli")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json", "OJP_CLI": os.environ.get("OJP_CLI", "/media/ojp-eval/bin/ojp-cli")}
FILES = {"english": R / "wikitext2-test.txt", "latin": R / "vocab/multi/latin.txt", "other": R / "vocab/multi/other.txt"}

def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], env=ENV, capture_output=True, text=True, **kw)

def bpb(model, path):
    toks = json.loads(run([B / "llama-tokenize", "-m", model, "-f", path, "--ids", "--no-bos", "--log-disable"]).stdout)
    out = run([B / "llama-perplexity", "-m", model, "-f", path, "-c", "512", "-ngl", "99"])
    ppl = float(re.findall(r"PPL = ([0-9.]+)", out.stdout + out.stderr)[-1])
    return round(math.log(ppl) * len(toks) / os.path.getsize(path) / math.log(2), 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("source", type=Path); ap.add_argument("template"); ap.add_argument("role")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--keep-ids", type=Path, default=R / "vocab/keep-ids.txt")
    ap.add_argument("--keep-variants", action="store_true", help="keep the stock and imatrix files afterwards")
    a = ap.parse_args()
    out = R / "qwen35" / "opt" / a.name; out.mkdir(parents=True, exist_ok=True)
    ev = Path("/media/ojp-eval") / a.name; ev.mkdir(parents=True, exist_ok=True)
    summary = ev / "summary.jsonl"
    log = lambda d: (summary.open("a").write(json.dumps(d) + "\n"), print(json.dumps(d), flush=True))

    imat = out / "imatrix.gguf"
    if not imat.exists():
        q8 = out / "ref-q8.gguf"
        run([B / "llama-quantize", a.source, q8, "Q8_0"])
        run([B / "llama-imatrix", "-m", q8, "-f", R / "wikitext2-train.txt", "-o", imat, "--chunks", "100", "-c", "512", "-ngl", "99"])
        q8.unlink()
    embargs = ["--token-embedding-type", a.emb] if a.emb else []
    variants = {}
    stock = out / "stock.gguf"
    if not stock.exists(): run([B / "llama-quantize", a.source, stock, "Q4_K_M"])
    variants["stock"] = stock
    full_imat = out / "imat.gguf"
    if not full_imat.exists(): run([B / "llama-quantize", "--imatrix", imat, a.source, full_imat, "Q4_K_M"])
    variants["imat"] = full_imat
    trimmed = out / "trimmed.gguf"
    if not trimmed.exists():
        bf = out / "trimmed-BF16.gguf"
        r = subprocess.run(["python3", R / "prune_vocab.py", a.source, bf, a.keep_ids], capture_output=True, text=True)
        log({"stage": "prune", "note": r.stdout.splitlines()[0] if r.stdout else r.stderr[-300:]})
        run([B / "llama-quantize", "--imatrix", imat, *embargs, bf, trimmed, "Q4_K_M"]); bf.unlink()
    variants["trimmed"] = trimmed

    env = {**os.environ, "OJP_CLI": CLI, "OJP_TEMPLATE": a.template, "OJP_N_CTX": "2048", "OJP_LOW_MEMORY": "1"}
    for name, f in variants.items():
        row = {"model": a.name, "variant": name, "mb": round(f.stat().st_size / 1e6)}
        for k, p in FILES.items(): row[f"bpb_{k}"] = bpb(f, p)
        ref = subprocess.run([CLI, "refusal-eval", f], env=env, capture_output=True, text=True).stdout
        m = re.search(r"refusals: (\d+)/(\d+)", ref); row["refusals"] = m and f"{m.group(1)}/{m.group(2)}"
        if a.role == "router":
            pr = subprocess.run([OJP / "scripts/make-router-probe.sh", f, a.template, f"{f}.probe.json"], capture_output=True, text=True, cwd=OJP, env=ENV).stdout
            m = re.search(r"CV ([0-9.]+).*XL \(n=80\) ([0-9.]+)", pr); row["probe_cv"] = m and float(m.group(1)); row["probe_xl_heldout"] = m and float(m.group(2))
            g = subprocess.run([CLI, "route-eval", f, "--xl"], env=env, capture_output=True, text=True).stdout
            m = re.search(r"router: (\d+)/80", g); row["generated_xl"] = m and int(m.group(1))
        elif name in ("stock", "trimmed"):
            t = subprocess.run([CLI, "tool-eval", f], env=env, capture_output=True, text=True).stdout
            row["tool"] = re.findall(r"full pipeline \(routed AND right tool/args\): (\d+/\d+), mean (\d+) ms", t)
        log(row)
    if not a.keep_variants:
        for n in ("stock", "imat"):
            variants[n].unlink(missing_ok=True)

if __name__ == "__main__":
    main()
