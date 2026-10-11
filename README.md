# model-tooling

Maintainer-side tooling for OpenJarvisProject's model compression
pipeline (pruning, quantization, factorization) — see the
[`docs`](https://github.com/OpenJarvisProject/docs) repo's
`architecture/model-compression.md`. This never ships to end users; it
produces the model artifacts that [`core`](https://github.com/OpenJarvisProject/core)
(the end-user runtime) consumes.

## `gpu-check` (Rust, `burn` + wgpu/Vulkan)

Confirms `burn`'s WGPU backend can actually run compute on this
machine's GPU — a separate concern from `core`'s already-validated
`llama.cpp` Vulkan inference path, since pruning/fine-tuning is a
training-style workload (`burn`), not inference.

**Why this exists**: this project's RX 580 is confirmed dead under
ROCm 7.2.4 for PyTorch-style GPU compute — `rocminfo` fails at the HSA
runtime level even with the `HSA_OVERRIDE_GFX_VERSION` workaround, so
any ROCm-based PyTorch call would hit the identical failure. This
smoke test proves the Vulkan fallback (identified for exactly this
scenario) actually works before any real pruning work depends on it.

```bash
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json cargo run --release
```

(See `core`'s README for why `VK_ICD_FILENAMES` is needed on this
system — a broken default proprietary Vulkan driver, worked around
rather than fixed system-wide.)

Confirmed output: `wgpu` enumerates the RX 580 twice — once via Vulkan
(`RADV POLARIS10`, `DiscreteGpu`, preferred) and once via a GL
fallback — and a real matrix multiply through `burn::Tensor` on the
backend-resolved-as-"Auto" device produces the correct result,
confirming it runs on real GPU hardware, not just compiling.

## Pruning pipeline (Python/PyTorch, CPU-only for now)

Not yet implemented. Per the runtime-architecture decision (Python for
this one-time, maintainer-side step, reusing published pruning
research code directly rather than reimplementing it in Rust), running
on CPU since ROCm is dead on this card — acceptable since pruning is
infrequent, not a latency-sensitive path.
