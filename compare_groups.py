#!/usr/bin/env python3
"""Tables from /media/ojp-eval/<model>/summary.jsonl (written by optimize_model.py):
every model, unoptimised (stock) vs optimised (imatrix, then trimmed vocabulary)."""
import json
from pathlib import Path

root = Path("/media/ojp-eval")
order = ["q25-0.5b", "q25-1.5b", "q35-0.8b", "q35-2b", "q35-4b", "unc-0.8b", "unc-2b", "unc-4b"]
rows = {}
for name in order:
    f = root / name / "summary.jsonl"
    if not f.exists():
        continue
    for line in f.read_text().splitlines():
        d = json.loads(line)
        if "variant" in d:
            rows[(name, d["variant"])] = d
print("| Model | Variant | MB | bpb English | bpb European | bpb other scripts | Refusals (of 18) | Router probe (held-out) | Router generated | Tool test |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (name, var), d in rows.items():
    tool = "; ".join(f"{a} ({b} ms)" for a, b in d.get("tool", [])[:1]) if d.get("tool") else ""
    print(f"| {name} | {var} | {d['mb']} | {d.get('bpb_english','')} | {d.get('bpb_latin','')} | {d.get('bpb_other','')} | {d.get('refusals','')} | "
          f"{d.get('probe_xl_heldout','')} | {d.get('generated_xl','')} | {tool} |")
