#!/usr/bin/env python3
"""Keep only the first N transformer blocks of a GGUF (for probe-only routers).

  cut_layers.py in.gguf out.gguf N

Drops every tensor of blocks >= N and sets <arch>.block_count = N. The result is no
longer a usable language model (its last hidden state skips the later layers), but
a probe fitted on its hidden state classifies requests in a fraction of the time
and memory. N should keep the attention/recurrent layer pattern intact (a multiple of 4
for Qwen3.5)."""
import re, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf

src, dst, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
r = gguf.GGUFReader(src)
arch = r.get_field("general.architecture").contents()
w = gguf.GGUFWriter(dst, arch)
for f in r.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = n if f.name == f"{arch}.block_count" else f.contents()
    w.add_key_value(f.name, val, vt, sub_type=sub)
keep = [t for t in r.tensors if not (m := re.match(r"blk\.(\d+)\.", t.name)) or int(m.group(1)) < n]
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data)
w.close()
print(f"kept {len(keep)} of {len(r.tensors)} tensors, {n} blocks -> {dst}")
