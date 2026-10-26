#!/usr/bin/env python3
"""Custom mixed quantization search by KL divergence: start from Q4_K_M + imatrix, upgrade one tensor
group (or layer band) at a time, record MB and KLD (wikitext-2 test, 64 chunks) against the Q8_0.

  hybrid_kld.py <work dir> <f16.gguf> <imatrix> <base Q8_0.gguf> <spec file>
spec file lines: <label>|<tensor-type overrides, e.g. blk\\.(\\d+)\\.ffn_down\\.weight=q6_K ; ...>
Rows go to <work dir>/hybrid.jsonl."""
import json, os, re, subprocess, sys
from pathlib import Path
R = Path("/media/openjarvisproject-model-tooling"); B = R / "llama.cpp/build-vk/bin"
ENV = {**os.environ, "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/radeon_icd.json"}
work, src, imat, ref, specs = map(Path, sys.argv[1:6]); work.mkdir(exist_ok=True)
text = R / "wikitext2-test.txt"; base = work / "base.kld"
def run(*c): return subprocess.run([str(x) for x in c], env=ENV, capture_output=True, text=True)
if not base.exists(): run(B / "llama-perplexity", "-m", ref, "-f", text, "-c", "512", "--chunks", "64", "-ngl", "99", "--kl-divergence-base", base)
done = {json.loads(l)["label"] for l in (work / "hybrid.jsonl").open()} if (work / "hybrid.jsonl").exists() else set()
for line in specs.read_text().splitlines():
    if not line.strip() or line.startswith("#"): continue
    label, over = line.split("|", 1)
    if label in done: continue
    out = work / "t.gguf"
    cmd = [B / "llama-quantize", "--imatrix", imat]
    for o in filter(None, [x.strip() for x in over.split(";")]): cmd += ["--tensor-type", o]
    run(*cmd, src, out, "Q4_K_M")
    r = run(B / "llama-perplexity", "-m", out, "-f", text, "-c", "512", "--chunks", "64", "-ngl", "99", "--kl-divergence-base", base, "--kl-divergence")
    k = re.search(r"Mean\s+KLD:\s+([0-9.]+)", r.stdout + r.stderr); t = re.search(r"Same top p:\s+([0-9.]+)", r.stdout + r.stderr)
    row = {"label": label, "mb": round(out.stat().st_size / 1e6, 1), "kld": float(k.group(1)) if k else None, "top_p": float(t.group(1)) if t else None}
    (work / "hybrid.jsonl").open("a").write(json.dumps(row) + "\n"); print(json.dumps(row), flush=True)
    out.unlink()
