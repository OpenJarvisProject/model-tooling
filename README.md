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

## `prune_layers.py` — whole-layer removal pruning (Python/PyTorch, CPU)

Structural depth pruning: removes entire transformer decoder blocks
(not individual weights, unlike Wanda-style unstructured sparsity,
which doesn't actually shrink tensor shapes or help on `llama.cpp`,
which has no sparse-tensor-core support). Running on CPU since ROCm is
confirmed dead on this card (see the GPU-check section above) —
acceptable since pruning is a one-time, infrequent job, not a
latency-sensitive path.

```bash
source .venv-cpu/bin/activate
pip install torch transformers
python prune_layers.py Qwen/Qwen2.5-0.5B-Instruct ./pruned-model --remove-fraction 0.25
```

**Removal is iterative and perplexity-validated, not a single batch.**
A first version measured layer importance once (cosine similarity
between each layer's input and output — the "Block Influence" idea
from ShortGPT-style depth pruning) on the *intact* model, then removed
6 layers at once based on that one snapshot. It broke the model
outright (fluent-looking but complete gibberish output) once tested
with a fair dtype comparison — removing one layer changes what the
next layer actually receives as input, which a one-shot ranking never
accounts for, and that compounding effect is sharper on a small
24-layer model than on the larger models this technique is usually
demonstrated on.

The fix: re-measure importance after every single removal, and
validate each candidate against a held-out perplexity check (reject if
it pushes perplexity past `--max-ppl-ratio`, default 1.5x baseline)
before committing to it — falling back to the next-ranked candidate if
one fails, and **stopping early rather than forcing a bad removal** if
none of the remaining layers pass. Confirmed working on
`Qwen/Qwen2.5-0.5B-Instruct`: targeted removing 6/24 layers, the
checker correctly stopped at 5/24 once every remaining candidate
failed validation (one rejected candidate — layer 0 — would have
pushed perplexity to over 12,000x baseline, correctly confirming early
layers are load-bearing). Result: 24→19 layers, 954MB→812MB on disk,
and the pruned model still generates fluent, on-topic (if sometimes
factually imperfect) text — a genuinely smaller/weaker model, not a
broken one.
