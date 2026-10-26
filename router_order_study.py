#!/usr/bin/env python3
"""Order-of-operations and depth study for a probe-only router (Qwen2.5-0.5B family).

  router_order_study.py <name> <f16 source> <imatrix of the full model> <template> <keep-ids> [out dir]

Builds the same final router in different orders and records, per file: size,
probe accuracy on the 80 held-out requests (probe fitted without them), 5-fold CV,
accuracy on the 42 wild requests (probe never fitted on them), light-mode RAM and ms.
Rows go to /media/ojp-eval/<name>/order-study.jsonl.
"""
import json, os, re, resource, subprocess, sys
from pathlib import Path

R = Path("/media/openjarvisproject-model-tooling"); B = R / "llama.cpp/build-vk/bin"; OJP = Path("/media/ojp")
CLI = os.environ.get("OJP_CLI", "/media/ojp-eval/bin/ojp-cli")
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json", "OJP_CLI": CLI}
name, src, imat, template, keep, *rest = sys.argv[1:]
work = Path(rest[0] if rest else R / "qwen25unc/work" / name); work.mkdir(parents=True, exist_ok=True)
out = Path("/media/ojp-eval") / name; out.mkdir(parents=True, exist_ok=True)
log = out / "order-study.jsonl"

def run(*cmd, **kw):
    kw.setdefault("env", ENV)
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)
def trim(i, o): run("python3", R / "prune_vocab.py", i, o, keep)
def cut(i, o, n): run("python3", R / "cut_layers.py", i, o, n)
def quant(i, o, q, im=imat): run(B / "llama-quantize", *(["--imatrix", im] if im else []), i, o, q)

def measure(label, f, probe_from=None):
    """Fit (or reuse) the probe, then measure the file."""
    f = Path(f)
    if probe_from is None:
        r = run(OJP / "scripts/make-router-probe.sh", f, template, f"{f}.probe.json", cwd=OJP)
        m = re.search(r"CV ([0-9.]+).*XL \(n=80\) ([0-9.]+)", r.stdout)
        cv, xl = (float(m.group(1)), float(m.group(2))) if m else (None, None)
        probe = f"{f}.probe.json"
    else:
        probe, cv, xl = probe_from, None, None
    env = {**ENV, "OJP_TEMPLATE": template, "OJP_N_CTX": "1024", "OJP_LOW_MEMORY": "1", "OJP_LIGHT": "1", "OJP_ROUTER_PROBE": probe}
    wild = run(CLI, "route-eval", f, OJP / "ojp-core/data/router-wild.tsv", env=env).stdout
    m = re.search(r"router: (\d+)/(\d+) correct, mean (\d+) ms", wild)
    xlr = run(CLI, "route-eval", f, "--xl", env=env).stdout
    mx = re.search(r"router: (\d+)/80", xlr)
    row = {"study": label, "mb": round(f.stat().st_size / 1e6), "cv": cv, "xl_heldout_fit": xl,
           "xl_with_given_probe": int(mx.group(1)) if mx else None, "wild_of_42": int(m.group(1)) if m else None, "ms": int(m.group(3)) if m else None}
    # RAM in its own process.
    code = ("import resource,subprocess,sys,os;"
            "subprocess.run([sys.argv[1],'route-eval',sys.argv[2],'--xl'],env=eval(sys.argv[3]),capture_output=True);"
            "print(round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024))")
    row["ram_mb"] = int(run("python3", "-c", code, CLI, f, repr(env)).stdout.strip() or 0)
    log.open("a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
    return probe

T = work / "trim-f16.gguf"
if not T.exists(): trim(src, T)
# 1. depth sweep, standard order (trim -> cut -> quantize with the full model's imatrix), IQ4_XS
for n in (8, 12, 16, 20):
    c = work / f"cut{n}.gguf"
    if not c.exists(): cut(T, c, n)
    q = work / f"std-cut{n}-IQ4_XS.gguf"
    if not q.exists(): quant(c, q, "IQ4_XS")
    measure(f"depth {n} of 24, trim>cut>quant, IQ4_XS", q)
    c.unlink() if n != 12 else None
# 2. quantization level at 12 blocks
c12 = work / "cut12.gguf"
for qt in ("Q4_K_M", "Q3_K_M", "IQ3_M", "Q2_K"):
    q = work / f"std-cut12-{qt}.gguf"
    if not q.exists(): quant(c12, q, qt)
    measure(f"12 blocks, {qt}", q)
q = work / "std-cut12-noimat.gguf"
if not q.exists(): quant(c12, q, "IQ4_XS", im=None)
measure("12 blocks, IQ4_XS, no imatrix", q)
# 3. order: cut first, then trim (should be identical)
c = work / "cutfirst12.gguf"
if not c.exists(): cut(src, c, 12)
t2 = work / "cutfirst-trim12.gguf"
if not t2.exists(): trim(c, t2)
q = work / "cutfirst-trim12-IQ4_XS.gguf"
if not q.exists(): quant(t2, q, "IQ4_XS")
measure("order: cut > trim > quant (12, IQ4_XS)", q)
# 4. order: quantize the full model first, then trim and cut the quantized file
qf = work / "full-IQ4_XS.gguf"
if not qf.exists(): quant(src, qf, "IQ4_XS")
qt = work / "quantfirst-trim.gguf"
if not qt.exists(): trim(qf, qt)
qc = work / "quantfirst-trim-cut12.gguf"
if not qc.exists(): cut(qt, qc, 12)
measure("order: quant > trim > cut (12, IQ4_XS from the full model)", qc)
# 5. order: imatrix measured on the cut model itself
q8 = work / "cut12-q8.gguf"
if not q8.exists(): quant(c12, q8, "Q8_0", im=None)
imc = work / "imatrix-cut12.gguf"
if not imc.exists(): run(B / "llama-imatrix", "-m", q8, "-f", R / "wikitext2-train.txt", "-o", imc, "--chunks", "100", "-c", "512", "-ngl", "99")
q = work / "imatcut-cut12-IQ4_XS.gguf"
if not q.exists(): quant(c12, q, "IQ4_XS", im=imc)
measure("order: imatrix from the cut model (12, IQ4_XS)", q)
# 6. order: probe fitted on the unquantized cut model, used on the quantized file
f16c = work / "cut12-f16probe.gguf"
if not f16c.exists(): run("cp", c12, f16c)
p16 = measure("probe fitted on the unquantized 12-block model (f16), for reference", f16c)
measure("order: probe fitted before quantizing, used on IQ4_XS", work / "std-cut12-IQ4_XS.gguf", probe_from=p16) if (work / "std-cut12-IQ4_XS.gguf").exists() else None
