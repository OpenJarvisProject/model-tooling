#!/usr/bin/env python3
"""Small gains for a probe router: training data, depth, embedding precision, vocabulary
size, scratch-buffer size. One change per row against the 12-block IQ4_XS baseline.

  router_tune.py <name> <f16 source> <template> <keep-ids>
Rows go to /media/ojp-eval/<name>/tune.jsonl."""
import json, os, re, subprocess, sys
from pathlib import Path

R = Path("/media/openjarvisproject-model-tooling"); B = R / "llama.cpp/build-vk/bin"; OJP = Path("/media/ojp")
CLI = os.environ.get("OJP_CLI", "/media/ojp-eval/bin/ojp-cli-v2")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json", "OJP_CLI": CLI}
name, src, template, keep = sys.argv[1:5]
work = Path("/media/ojp-eval") / name / "work"; work.mkdir(parents=True, exist_ok=True)
log = Path("/media/ojp-eval") / name / "tune.jsonl"
def run(*cmd, **kw):
    kw.setdefault("env", ENV)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
def acc(f, probe, tsv):
    e = {**ENV, "OJP_TEMPLATE": template, "OJP_N_CTX": "1024", "OJP_LOW_MEMORY": "1", "OJP_LIGHT": "1", "OJP_ROUTER_PROBE": probe}
    out = run(CLI, "route-eval", f, tsv, env=e).stdout
    m = re.search(r"router: (\d+)/(\d+) correct, mean (\d+) ms", out)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else (None, None, None)
def ram(f, probe, extra=None):
    e = {**ENV, "OJP_TEMPLATE": template, "OJP_N_CTX": "1024", "OJP_LOW_MEMORY": "1", "OJP_LIGHT": "1", "OJP_ROUTER_PROBE": probe, **(extra or {})}
    code = ("import resource,subprocess,sys;subprocess.run([sys.argv[1],'route-eval',sys.argv[2],'--xl'],env=eval(sys.argv[3]),capture_output=True);"
            "print(round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024))")
    return int(run("python3", "-c", code, CLI, f, repr(e)).stdout.strip() or 0)
def measure(label, f, train="router-train", ubatch=None):
    f = Path(f)
    r = run(OJP / "scripts/make-router-probe.sh", f, template, f"{f}.probe.json", cwd=OJP, env={**ENV, "TRAIN_TSV": train})
    m = re.search(r"CV ([0-9.]+).*XL \(n=80\) ([0-9.]+)", r.stdout)
    probe = f"{f}.probe.json"
    w1 = acc(f, probe, OJP / "ojp-core/data/router-wild.tsv"); w2 = acc(f, probe, OJP / "ojp-core/data/router-wild2.tsv")
    row = {"change": label, "mb": round(f.stat().st_size / 1e6), "cv": float(m.group(1)) if m else None, "xl_heldout": float(m.group(2)) if m else None,
           "wild42": w1[0], "wild2_44": w2[0], "ms": w2[2], "ram_mb": ram(f, probe, {"OJP_N_UBATCH": str(ubatch)} if ubatch else None)}
    log.open("a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)

T = work / "trim.gguf"
if not T.exists(): run("python3", R / "prune_vocab.py", src, T, keep)
def cutq(n, qt="IQ4_XS", emb=None, base=T, tag=""):
    c = work / f"cut{n}{tag}.gguf"
    if not c.exists(): run("python3", R / "cut_layers.py", base, c, n)
    q = work / f"cut{n}{tag}-{qt}{'-emb_' + emb if emb else ''}.gguf"
    if not q.exists(): run(B / "llama-quantize", *(["--token-embedding-type", emb] if emb else []), c, q, qt)
    return q
V15 = work / "trim15.gguf"
measure("COMBINED: 12 blocks, vocabulary 15%, noisy copies, scratch buffer 32", cutq(12, base=V15, tag="v15"), train="router-train-aug", ubatch=32)
V10 = work / "trim10.gguf"
if not V10.exists(): run("python3", R / "prune_vocab.py", src, V10, keep, "--top-pct", "10")
measure("12 blocks, vocabulary 10% (15k tokens), noisy copies, scratch buffer 32", cutq(12, base=V10, tag="v10"), train="router-train-aug", ubatch=32)
for emb in ("q4_0", "iq4_nl"):
    measure(f"12 blocks, vocabulary 15%, embedding {emb}, noisy copies, scratch buffer 32", cutq(12, emb=emb, base=V15, tag="v15"), train="router-train-aug", ubatch=32)
