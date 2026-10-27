#!/usr/bin/env python3
"""Weight-space interpolation of two GGUFs with the same architecture:
   out = base + alpha * (tuned - base)      (alpha 0 = base, 1 = tuned)

  merge_models.py base.gguf tuned.gguf alpha out.gguf

Done tensor by tensor (low memory). Matrices are written as F16, small tensors as F32. Metadata comes
from the tuned model. Used to give back some of what an uncensoring pass cost, without training."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf

base_p, tuned_p, alpha, out_p = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
rb, rt = gguf.GGUFReader(base_p), gguf.GGUFReader(tuned_p)
base = {t.name: t for t in rb.tensors}
arch = rt.get_field("general.architecture").contents()
w = gguf.GGUFWriter(out_p, arch)
for f in rt.fields.values():
    if f.name == "general.architecture" or f.name.startswith("GGUF."):
        continue
    vt = f.types[0]; sub = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    w.add_key_value(f.name, f.contents(), vt, sub_type=sub)

def logical(t):
    return tuple(int(x) for x in reversed(t.shape))

def as_f32(t):
    d = np.asarray(t.data)
    if t.tensor_type == gguf.GGMLQuantizationType.BF16:
        d16 = d.view(np.uint16)
        return (d16.astype(np.uint32) << 16).view(np.float32).reshape(logical(t))
    if t.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        return d.astype(np.float32).reshape(logical(t))
    raise SystemExit(f"{t.name}: unsupported source type {t.tensor_type.name}")

plan = []
for t in rt.tensors:
    b = base[t.name]
    if tuple(b.shape) != tuple(t.shape):
        raise SystemExit(f"shape mismatch for {t.name}: {b.shape} vs {t.shape}")
    small = len(t.shape) < 2 or t.name.endswith(("ssm_conv1d.weight",)) or "norm" in t.name or t.name.endswith(("ssm_a", "ssm_dt.bias"))
    dtype = np.float32 if small else np.float16
    shape = logical(t)
    plan.append((t, b, dtype))
    qt = gguf.GGMLQuantizationType.F32 if dtype == np.float32 else gguf.GGMLQuantizationType.F16
    w.add_tensor_info(t.name, shape, dtype, int(np.prod(shape)) * np.dtype(dtype).itemsize, qt)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t, b, dtype in plan:
    x = as_f32(b) + alpha * (as_f32(t) - as_f32(b))
    w.write_tensor_data(np.ascontiguousarray(x.astype(dtype)))
w.close()
print("wrote", out_p, "alpha", alpha)
