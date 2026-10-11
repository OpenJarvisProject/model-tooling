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

### Threshold sensitivity (Qwen2.5-0.5B, found 2026-10-11)

Confirmed deterministic first (exact-repeat diff of two identical runs
produced zero differences) — ruled out CPU multi-threaded BLAS
non-determinism as a confound before trusting any threshold
comparison. Looser `--max-ppl-ratio` doesn't just gate the final
cutoff decision — it reshapes the entire greedy trajectory, since an
earlier marginal candidate's accept/reject decision changes the model
state for every subsequent step. On this model specifically:

| `--max-ppl-ratio` | Layers removed (of 24) |
|---|---|
| 1.5x | 5 |
| 3.0x | 5 (same count, different removal path) |
| 10.0x | 7 |

There's a real structural cliff somewhere between 3x and 10x acceptable
perplexity degradation on this model — not a smooth gradient. Pushing
the nominal removal target higher (6 vs 22) doesn't change the ceiling
at a fixed threshold; confirmed identical stopping point either way.

### Larger base model comparison (Qwen2.5-1.5B, found 2026-10-11)

Same pipeline, same 1.5x threshold, on `Qwen/Qwen2.5-1.5B-Instruct`
(28 layers vs the 0.5B model's 24): removed 6/28 (21.4%), landing
within half a percentage point of the 0.5B model's ceiling (5/24,
20.8%). The "bigger models tolerate proportionally more removal"
expectation from depth-pruning literature doesn't show up clearly at
this scale difference (0.5B→1.5B is only a 3x jump; the effect is
usually demonstrated at much larger scales, e.g. 7B vs 70B) — worth
knowing before assuming this method scales favorably just because a
model is bigger. Output stays fluent at this size too (not broken),
2.4GB on disk unquantized (22 layers, ~1.26B params).

## `factorize.py` — low-rank factorization (SVD, W ≈ A×B)

Targets the large attention (q/k/v/o_proj) and MLP (gate/up/down_proj)
linear layers. Same lesson as pruning, rediscovered independently:

**First attempt (batch, uniform 75% keep-fraction across all 168
target matrices) broke the model catastrophically** — 9509x baseline
perplexity, worse than any pruning failure. Isolating a single matrix
first proved the implementation itself was correct (1.03% reconstruction
error, 1.01x perplexity for one matrix alone) — the failure was a
compounding effect across 168 simultaneous changes, the exact same
shape as pruning's batch-removal bug. Fixed the same way: factorize
one matrix at a time, safest (lowest reconstruction error) first,
validate perplexity after each, revert on failure.

**A second, separate gotcha, not a bug**: compression from `W ≈ A×B`
only happens when `k < (in×out)/(in+out)`. For roughly square matrices
(q_proj/o_proj here, ~896×896) that breakeven is `k < 0.5×d` — a 75%
keep-fraction is *above* that, making those matrices larger, not
smaller. It only helps on rectangular matrices (gate/up/down_proj,
896×4864, breakeven ≈84%). Applying one uniform fraction everywhere
without accounting for this produced a net -1.8% "reduction" (i.e. the
model got *bigger*) even though the iterative safety-validation itself
worked correctly (109/168 matrices accepted, 1.45x baseline
perplexity). Testing lower fractions, safely under the square-matrix
breakeven, next.
