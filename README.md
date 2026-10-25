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
perplexity).

**At keep_fraction=0.4 (below the square-matrix breakeven), real
compression is achieved**: 77/168 matrices accepted, 70.9M→45.8M
params in the targeted layers (**35.4% reduction**), 1.48x baseline
perplexity — a very similar safety margin to the 0.75 run, just with
an actually-shrunk model this time.

**A third problem, caught by self-verification, not assumed away**:
the saved checkpoint generated complete gibberish on reload despite
the in-script perplexity looking fine. Root cause: `save_pretrained`/
plain `from_pretrained` has no idea `FactorizedLinear` exists -
reloading reconstructs a *standard* Qwen2 architecture, finds the
checkpoint's `*.first.weight`/`*.second.weight` keys don't match the
expected plain `*.weight` key, and **silently randomly reinitializes**
every factorized matrix instead of loading the real trained weights.
The in-script perplexity numbers were always trustworthy (they ran on
the correctly-structured live model) - only the saved artifact was
broken. Fixed by adding `factorization_metadata.json` (which matrices
were replaced, at what rank) alongside the normal save, and a proper
`load_factorized_model()` that rebuilds the exact same module
structure *before* loading weights, rather than trusting
`from_pretrained` to guess it. `main()` now calls this reload-and-
verify step automatically on every run, specifically so this bug
class can't silently reappear and go unnoticed again.

Two more real bugs surfaced immediately by that same self-check
(not separately caught by luck - this is exactly why the check was
added): a `strict=True` `load_state_dict` failing on `lm_head.weight`
(tied to the input embedding, so legitimately absent from the saved
file - fixed with `strict=False` plus an explicit check that no
*other*, unexpected keys are missing, so a real future bug can't hide
behind this fix), and a dtype mismatch (`from_config` doesn't respect
`config.torch_dtype` the way `from_pretrained(dtype="auto")` does,
so the rebuilt skeleton defaulted to float32 against bf16 checkpoint
data - fixed by resolving and passing the dtype explicitly). Final
state, verified both by perplexity match and actual coherent
generation on the reloaded model: correct, working, 35.4% smaller in
the targeted layers.

### 7B model tested (2026-10-11): less prunable than smaller models, not more

`Qwen/Qwen2.5-7B-Instruct` (28 layers), same pipeline, same 1.5x
perplexity threshold: removed only **2/28 layers (7.1%)** before
stopping — much *less* than the 0.5B model's 20.8% or the 1.5B model's
21.4%. This complicates the earlier "model size doesn't change
removability" finding rather than confirming it — three data points
(0.5B: 20.8%, 1.5B: 21.4%, 7B: 7.1%) don't form a clean trend either
way. Plausible explanation, not yet confirmed: 7B's baseline
perplexity (6.879) is already lower/better than the smaller models',
so a *fixed ratio* threshold like 1.5x represents a smaller absolute
tolerance when the baseline is already low — a real methodological
question about whether a ratio-based threshold is actually comparable
across models with different baseline quality, not just a straight
"bigger model, more/less prunable" result.

Had to resume the pipeline mid-run after it hit a background time
limit right as the 7B pruning step started - cost nothing since
steps 1-2 (download, convert, quantize the original) had already
completed and were preserved.

**Both the original and pruned 7B models were tested on GPU (Vulkan),
as specifically requested - both fit entirely in the RX 580's 8GB
VRAM**, no partial CPU offload needed:

| | Original (28L) | Pruned (26L) | Δ |
|---|---|---|---|
| Model weights (VRAM) | 4168.09 MiB | 3900.82 MiB | -6.4% |
| KV-cache (VRAM) | 112.00 MiB | 104.00 MiB | -7.1% (exact 26/28 match) |
| Compute buffer (VRAM) | 304.00 MiB | 304.00 MiB | 0% (confirms the n_ubatch finding holds at 7B too) |
| Host RSS | ~4572 MiB | ~4305 MiB | -5.9% |
| Wall time (5 reps) | 3.684s | 2.927s | -20.6% (disproportionately bigger than the size reduction - noted, not over-interpreted from one measurement) |

## Combination study: what to apply, in what order (2026-10-11)

`combine.py` + `evalharness.py` + `validate_quant_sim.py`. All numbers
below are on **Qwen2.5-0.5B-Instruct**, measured on held-out
wikitext-2 *test* (16K tokens); in-loop accept/reject decisions use
wikitext-2 *validation* (2K tokens). Full-precision baseline: test
perplexity **17.59**.

### Methodology correction first

Every earlier pruning/factorization quality number in this README was
overstated. The in-loop check used 4 hand-written sentences and the
greedy search overfit to them. Re-measured on held-out text:

| Earlier artifact | Claimed | Held-out reality | Params saved |
|---|---|---|---|
| Pruned, 5 layers | 1.50x | **2.42x** | 15.1% |
| Factorized, keep 0.4 | 1.48x | **3.52x** | 5.1% |

With the larger validation set the search no longer overfits: P's
validation ratio (1.52x) and test ratio (1.45x) agree.

### Equal-budget comparison (1.5625x total val budget each)

| Condition | fp16 size | Test ppl |
|---|---|---|
| Original | 988 MB | 17.59 |
| Prune only (4 layers removed) | 869 MB (-12.1%) | 25.48 (1.45x) |
| Factorize only (58/168 matrices) | 971 MB (-1.7%) | 27.19 (1.55x) |

Size saved per unit of quality lost: quantization ≈ 10x pruning ≈ 10x
factorization. Factorization is dominated at this scale.

### Real quantization, same held-out metric

| llama.cpp quant | Original (24L) | Pruned (20L) |
|---|---|---|
| Q5_K_M | 420 MB @ 18.13 | 375 MB @ 26.20 |
| Q4_K_M | 398 MB @ 18.21 | 357 MB @ 26.56 |
| Q3_K_M | 356 MB @ 19.53 | 321 MB @ 28.60 |
| Q4_0 | 352 MB @ 20.16 | 319 MB @ 29.77 |
| Q2_K | 339 MB @ 22.33 | 307 MB @ 36.35 |

At equal file size, quantizing harder beats pruning by a wide margin
(Q3_K_M original 356 MB @ 19.5 vs Q4_K_M pruned 357 MB @ 26.6).
Simulated round-to-nearest 4-bit (used for factorized conditions,
which GGUF can't represent) lands at 21.97 vs real Q4_0's 20.16: same
ballpark, slightly pessimistic, fine for ranking. Simulated 3-bit is
far worse than real Q3_K_M - don't trust the 3-bit simulation.

### The real size floor: the embedding table

Quantization bottoms out near ~338 MB because `token_embd` (152K vocab
x 896 = 136M params, 28% of the model, tied to the output projection)
stays at **Q8_0 at every quant level** - 37% of the Q4_K_M file, 43%
of Q2_K. Pruning and factorization never touch it.

Gotcha: K-quants need rows divisible by 256 and this model's hidden
size is 896, so `--token-embedding-type q6_K/q5_K/q4_K` fails outright
- and it's also why llama-quantize logged "144 of 290 tensors required
fallback quantization": on 0.5B, "Q4_K_M" isn't really Q4_K_M for half
its tensors. Block-32 types fit:

| Layers | Embedding | Size | Test ppl |
|---|---|---|---|
| Q4_K_M | Q8_0 (default) | 398 MB | 18.21 |
| Q4_K_M | Q5_0 | **347 MB** | **18.39** |
| Q4_K_M | Q4_0 | 330 MB | 19.04 |
| Q3_K_M | Q8_0 (default) | 356 MB | 19.53 |
| Q3_K_M | Q5_0 | **304 MB** | **19.72** |
| Q3_K_M | Q4_0 | 287 MB | 20.48 |

**304 MB @ 19.7 is smaller than every pruned model produced (best: 307
MB @ 36.3).** Every point on the size/quality frontier at this scale is
"original weights, mixed-precision quantization". Recommended default
for small Qwen models: `llama-quantize --token-embedding-type q5_0
<f16> <out> Q4_K_M` (or Q3_K_M for a tighter budget).

### Why the structural methods lose, and what would change that

Both pruning and factorization here are training-free. Published depth/
width-pruning work (ShortGPT, LLM-Pruner, NVIDIA Minitron) consistently
pairs them with a recovery step - LoRA fine-tuning or distillation
from the unpruned model - and that step is where most of the lost
quality comes back. The likely-correct full order is **prune ->
heal (distill/fine-tune) -> quantize (layers + embeddings)**. Untested
here: it needs a training loop, which on this machine means CPU or a
burn/Vulkan path (ROCm is dead on the RX 580).

## `prune_vocab.py` — vocabulary trimming for byte-level BPE models

Drops tokens a typed English assistant never uses from a GGUF's vocabulary
and embedding matrix (Qwen3.5: 248,320 -> 48,736 tokens, 28% smaller at 2B,
33% at 0.8B, under 1% worse). Needs `llama-tokenize --ids` output over a
keep-corpus. Results: `RESULTS.md` section 8.

```bash
llama-tokenize -m model-BF16.gguf -f vocab/keep-corpus.txt --ids --no-bos --log-disable > vocab/keep-ids.txt
python prune_vocab.py model-BF16.gguf pruned-BF16.gguf vocab/keep-ids.txt   # then llama-quantize
```
