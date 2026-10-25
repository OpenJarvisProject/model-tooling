# Model compression results (consolidated)

Every number here was measured on one machine: RX 580 8GB (Vulkan/RADV),
Ryzen 5 4500, 23GB RAM. Raw data for every row is in `results/`. The
scripts that produced it are in this repo, and each section names its
script.

Lower perplexity (ppl) is better. Read changes as relative: 10% higher
ppl means 10% more "surprised" by real text. Two harnesses were used, and
their absolute numbers **must not be compared with each other**:

- **PyTorch harness** (`evalharness.py`): wikitext-2 test, ctx 512, fp32.
  Used for pruning/factorization, which needs HF checkpoints.
- **llama.cpp harness** (`llama-perplexity`): wikitext-2 test, 64×512
  tokens, Vulkan. Used for everything GGUF.

## TL;DR

1. **Quantization dominates.** Training-free pruning and SVD factorization
   both lose to "just quantize harder" at every size tested.
2. **The 0.5B can shrink ~3x with no measurable quality loss.** Q4_K_M +
   imatrix + q5_0 token embedding is 347MB vs 994MB, and every benchmark
   stays within error of the original.
3. **Low-bit 3B problems are specific tensors, not "too few bits":**
   - q3_K breaks the FFN tensors.
   - IQ2_M starves the last ~8 layers and attn_k.
   - Fixing just those (a custom hybrid mix) cut IQ2_M's loss from 39% to
     26% for +5% size.
4. **Pruning first beats factorizing first,** but both stacks lose to
   plain quantization. The untested next step is heal (LoRA) after
   pruning.

## 1. Qwen2.5-0.5B: original vs the two finalists (deep comparison)

`close_compare.py`, `resources.py`. "Original" is the unquantized f16 GGUF.

### Quality

| | Original (f16) | Q4_K_M + imatrix + emb q5_0 | IQ3_M + imatrix + emb q5_0 |
|---|---|---|---|
| File size | 994 MB | **347 MB** (-65%) | **292 MB** (-71%) |
| KL divergence vs original (lower = closer) | 0 | 0.031 | 0.091 |
| Same top token as original | 100% | 90.0% | 83.7% |
| HellaSwag (1000 tasks) | 50.8% | 51.1% | 50.8% |
| Winogrande (1267) | 57.4 ± 1.4% | 56.2 ± 1.4% | 56.4 ± 1.4% |
| ARC (val) | 49.8 ± 1.7% | 47.4 ± 1.7% | **43.3 ± 1.7%** |
| MMLU (val) | 30.4 ± 1.2% | 30.2 ± 1.2% | 30.4 ± 1.2% |
| TruthfulQA mc | 25.0 ± 1.5% | 25.5 ± 1.5% | 24.5 ± 1.5% |
| Agent tests (34, `agent_tests.py`) | 19/34 | 21/34 | **16/34** |
| wikitext-2 ppl (llama.cpp harness, Q8_0 ref = 14.69) | n/a | 15.09 | 15.89 |

**Verdict.** Q4_K_M+imat+emb q5_0 is indistinguishable from the original
and is the default. IQ3_M+imat+emb q5_0 visibly loses reasoning (ARC)
and agent ability, so use it only when every MB counts.

### Resource use

Load plus a 32-token completion in the real runtime (`ojp-core`). Mean of
3 runs; run-to-run SD is under 2% unless noted.

| | Original (f16) | Q4_K_M+imat+emb q5_0 | IQ3_M+imat+emb q5_0 |
|---|---|---|---|
| **CPU-only** (GPU hidden) | | | |
| Peak RAM (RSS) | 1054 MB | 462 MB | 513 MB ¹ |
| Avg CPU cores busy (of 6) | 3.2 | 2.7 | 2.7 |
| Wall time | 1.56 s | 0.83 s | 0.88 s |
| Generation speed (llama-bench) | 30 tok/s | **80 tok/s** | 75 tok/s |
| Prompt processing (llama-bench) | 889 tok/s | 996 tok/s | 1059 tok/s |
| **GPU (all layers on Vulkan)** | | | |
| VRAM used ² | 1333 MB | 684 MB | 629 MB |
| Peak RAM (RSS) ³ | 1055 MB | 437 MB | 385 MB |
| Avg CPU cores busy | 0.8 | 0.8 | 0.8 |
| Wall time | 0.96 s | 0.68 s | 0.68 s (SD 0.8 s, one slow run) |
| Generation speed (llama-bench) | 157 tok/s | 200 tok/s | 212 tok/s |
| Prompt processing (llama-bench) | 1588 tok/s | 1741 tok/s | 1748 tok/s |

1. IQ3_M uses more RAM than Q4_K_M on the CPU despite the smaller file.
   The likely cause is llama.cpp's CPU repack buffers for I-quants; this
   is unconfirmed.
2. About 300MB of each VRAM figure is the fixed compute buffer, which
   scales with `n_ubatch` and not model size (`n_ubatch=128` cuts it to
   ~75MB). See the core repo.
3. RSS on GPU includes the mmapped model file's page cache, which is
   reclaimable, so real RAM pressure is lower.
- Quantized models are faster on the CPU because generation is
  memory-bandwidth bound, so fewer bytes per token means more tokens/s.
- **"CPU-only" gotcha:** with a Vulkan device present, llama.cpp still
  offloads big matmuls to the GPU at `n_gpu_layers=0` (measured 400–600MB
  VRAM). True CPU-only needs the GPU hidden.

## 2. GGUF quantization study, one variable at a time (`quant_study.py`)

imatrix is calibrated on wikitext-2 train and scored on test, llama.cpp harness.

| Variant | 0.5B MB | 0.5B ppl | 3B MB | 3B ppl |
|---|---|---|---|---|
| Q8_0 (reference) | 531 | 14.69 | 3286 | 8.64 |
| Q4_K_M | 398 | 15.07 | 1930 | 9.11 |
| Q4_K_M + imatrix | 398 | 14.94 | 1930 | 8.93 |
| Q3_K_M | 356 | 16.01 | 1591 | 15.67 |
| Q3_K_M + imatrix | 356 | 15.55 | 1591 | 13.48 |
| Q2_K | ~340 | 18.48 | 1275 | 22,245 (broken) |
| Q2_K + imatrix | ~340 | 17.72 | 1275 | 11.36 |
| IQ4_XS + imatrix | 349 | 15.23 | 1739 | **8.88** |
| IQ3_M + imatrix | 343 | 15.72 | 1489 | 9.51 |
| IQ3_XXS + imatrix | 334 | 16.14 | 1283 | 10.03 |
| IQ2_M + imatrix | 329 | 17.36 | 1141 | 12.05 |
| Q4_K_M + imat + emb q5_0 / q4_K ⁴ | 347 | 15.09 | 1850 | 9.06 |
| Q4_K_M + imat + emb q4_0 | 330 | 15.57 | – | – |
| IQ3_M + imat + emb q5_0 / q4_K ⁴ | 292 | 15.89 | 1409 | 9.68 |
| IQ3_M + imat + emb q4_0 | 275 | 16.38 | – | – |

4. The 0.5B uses q5_0 because its hidden size of 896 isn't divisible by
   256, so K-quant embeddings fail. The 3B uses q4_K. On the 3B, the
   default embedding at both Q4_K_M and IQ3_M is already q6_K: overriding
   to q6_K gives byte-identical results.

Findings:
- **imatrix always helps,** and most at low bits. It turns 3B Q2_K from
  broken (22,245) into usable (11.4).
- **I-quants beat K-quants at the same size.**
- **Token embedding** is 28% of the 0.5B's parameters and is kept at
  Q8_0 by default, so it sets the size floor. Lowering it is the cheapest
  size win on small models. On the 3B it matters much less.

## 3. Why 3B low-bit quants misbehave, and the fix (`hybrid_scan.py`)

Per-tensor types for each mix are in `results/tensor-mix/`.

**Q3_K_M anomaly (13.48, worse than IQ3_M's 9.51 despite more bits).**
Raising one group at a time to q4_K:

| Raised group | MB | ppl |
|---|---|---|
| none | 1591 | 13.48 |
| attn_q | +20 | 13.56 |
| attn_k | +2.5 | 13.33 |
| **ffn_gate** | +108 | **9.19** |
| ffn_up | +108 | 10.21 |

Cause: the q3_K format itself on this model's FFN gate/up tensors. iq3_s
on the same tensors is fine. Rule: **avoid q3_K for Qwen2.5-3B FFN.**

**IQ2_M sensitivity (base 1140.5 MB @ 12.05).** Each row raises one
thing; gain is ppl recovered per extra 100MB.

| Raised | +MB | ppl | gain / 100MB |
|---|---|---|---|
| attn_k → iq3_s | 2 | 11.92 | **6.31** |
| attn_v → q5_K | 2 | 12.02 | 1.08 |
| attn_q → iq3_s | 17 | 11.91 | 0.84 |
| attn_output → q4_K | 20 | 12.19 | worse |
| ffn_gate → iq3_s | 89 | 11.49 | 0.63 |
| ffn_up → iq3_s | 89 | 10.92 | 1.28 |
| ffn_down → iq3_s | 79 | 11.25 | 1.01 |
| FFN of layers 0–3 | 30 | 11.90 | 0.52 |
| 4–7 | 30 | 11.82 | 0.79 |
| 8–11 | 30 | 11.89 | 0.55 |
| 12–15 | 30 | 11.73 | 1.08 |
| 16–19 | 30 | 11.76 | 0.97 |
| 20–23 | 30 | 11.79 | 0.89 |
| 24–27 | 30 | 11.91 | 0.48 |
| 28–31 | 30 | 11.59 | 1.55 |
| **32–35** | 30 | 11.40 | **2.19** |

**Built hybrids.** Each step adds one change to the previous step:

| Mix | MB | ppl | loss vs Q8_0 |
|---|---|---|---|
| IQ2_M (stock) | 1140.5 | 12.05 | 39% |
| + attn_k=iq3_s | 1142.6 | 11.92 | 38% |
| **+ layers 28–35 FFN=iq3_s** | **1201.8** | **10.86** | **26%** |
| + all ffn_up=iq3_s | 1270.8 | 10.20 | 18% |
| layers 28–35 FFN=iq4_xs instead | 1325.8 | 10.01 | 16% |
| *IQ3_XXS (stock) for comparison* | 1283 | 10.03 | 16% |

The 1202MB hybrid sits clearly below the stock IQ2_M→IQ3_XXS line (which
predicts ~11.2 at that size). At ≥1283MB the stock IQ3_XXS/IQ3_M mixes
were as good as anything tried.

```
llama-quantize --imatrix imatrix.gguf \
  --tensor-type 'attn_k\.weight=iq3_s' \
  --tensor-type 'blk\.(2[89]|3[0-5])\.ffn_.*=iq3_s' \
  qwen2.5-3b-f16.gguf out.gguf IQ2_M
```

## 4. Pruning and factorization (PyTorch harness, 0.5B, fp baseline ppl 17.59)

`prune_layers.py`, `factorize.py`, `combine.py`.

| Pipeline | Size (fp16 / est.) | Test ppl |
|---|---|---|
| Original | 988 MB | 17.59 |
| Prune 4 layers (1.25x budget) | 869 MB | 25.48 |
| Factorize (SVD, keep 40%, 1.25x budget) | 971 MB | 27.19 |
| Prune → Factorize | 917 MB | 25.56 |
| Factorize → Prune | 943 MB | 26.51 |
| Prune → Factorize → sim-Q4 | 258 MB | 35.9 |
| Factorize → Prune → sim-Q4 | 265 MB | 39.9 |
| Prune → Factorize → sim-Q3 | 201 MB | 617 |
| Factorize → Prune → sim-Q3 | 206 MB | 2156 |
| *Original → real Q4_K_M* | 398 MB | 18.21 |
| *Original → real Q3_K_M + emb q5_0* | 304 MB | 19.72 |
| *Pruned → real Q4_K_M* | 357 MB | 26.56 |

- Pruning first beats factorizing first on both size and quality.
- Factorized layers can't be represented in GGUF (llama.cpp has no
  factored-linear op), so factorization has no deployable benefit today.
- Training-free pruning costs about 10x more quality per MB saved than
  quantization does.
- **Superseded:** an early 4-sentence validation set overfit badly. It
  claimed pruning cost ~1.5x ppl when the held-out cost was 2.4x.
  `results/quant-sweep.csv` comes from that era: its absolute ppl values
  are on a different, tiny text and must not be compared with anything
  above.

## 5. Other measurements

- **7B (GPU only):** pruned 4 of 28 layers. 7B proved less prunable than
  0.5B/1.5B. See README "7B model tested".
- **The compute buffer doesn't shrink with pruning.** It scales exactly
  with `n_ubatch` (512→298.5MiB, 256→149, 128→75, 64→37).
- **Router (core repo):** with a grammar constraint and 8 few-shot
  examples, the 0.5B scores 16/24 and the 1.5B 19/24 on the dev set.
  Zero-shot, the 0.5B scores 6/24.

## 6. Qwen2.5-1.5B-Instruct: quantization study and task tests

`quant_study.py` (same method as section 2; imatrix from wikitext-2 train,
scored on wikitext-2 test, llama.cpp/Vulkan). Reference is Q8_0 at 9.708 ppl
(the stock official Q4_K_M, 1066 MB, was not scored this way).

| Variant | MB | ppl | vs Q8_0 |
|---|---|---|---|
| Q8_0 | 1647 | 9.708 | |
| Q4_K_M | 986 | 10.213 | +5.2% |
| **Q4_K_M + imatrix** | **986** | **9.954** | **+2.5%** |
| Q4_K_M + imat + emb q5_K | 955 | 10.041 | +3.4% |
| Q4_K_M + imat + emb q4_K | 926 | 10.236 | +5.4% |
| IQ4_XS + imatrix | 896 | 10.081 | +3.8% |
| IQ3_M + imatrix | 777 | 10.919 | +12.5% |
| IQ3_XXS + imatrix | 669 | 11.944 | +23% |
| Q3_K_M + imatrix | 824 | 10.924 | +12.5% |
| Q2_K + imatrix | 676 | 13.472 | +39% |
| IQ2_M + imatrix | 601 | 14.474 | +49% |

The perplexity error bars are about ±0.2, so differences under that are noise.

**Task tests** (OJP `ojp-cli route-eval`, `tool-eval`, `agent-eval`; CPU, the
model both routes and picks tools; 4 candidates, one change each):

| Model | MB | route test / dev | tool seen / unseen / multi | agent tasks | s per request |
|---|---|---|---|---|---|
| stock official Q4_K_M | 1066 | 10/10, 23/24 | 36/39, 19/20, 14/14 | 26/28 | 2.1 |
| **Q4_K_M + imatrix** | **941** | 10/10, **24/24** | **37/39, 20/20**, 14/14 | 26/28 | 2.1 |
| IQ4_XS + imatrix | 855 | 10/10, 23/24 | 36/39, 20/20, 14/14 | 25/28 | 2.7 |
| IQ3_M + imatrix | 741 | 10/10, 23/24 | 32/39, 20/20, 14/14 | 25/28 | 6.1 |

Findings:
- **Q4_K_M + imatrix is the pick:** 12% smaller than the stock file, equal or
  better on every task test, same speed. sha256
  `c98e1a583924c7158e7b56d161ea121a864755de6d60c148228ea4e4abb669d6`.
- **Smaller isn't faster on a CPU:** the I-quants (IQ4_XS, IQ3_M) unpack slower,
  so they cost time (up to 3x) and a little accuracy for 10-30% less disk.
  Perplexity alone would not have shown this.
- Below 4 bits quality falls quickly (IQ3_M +12.5%, 2-bit +49%).

## 7. Qwen3.5 (official Apache-2.0 weights): quantization study and task tests

Source: unsloth's BF16 GGUF conversions of the official Qwen3.5-0.8B/2B/4B
(`qwen35/`). Same method as section 6 (wikitext-2 imatrix, wikitext-2 test
ppl, 64 chunks, Vulkan). llama.cpp cbe28bb. Reference is Q8_0.

| Variant | 0.8B MB / ppl | 2B MB / ppl | 4B MB / ppl |
|---|---|---|---|
| Q8_0 | 812 / 17.61 | 2012 / 12.10 | 4482 / 9.60 |
| Q4_K_M | 529 / 18.19 | 1274 / 12.77 | 2709 / 10.01 |
| Q4_K_M + imatrix | 529 / 18.60 | 1274 / 12.35 | 2709 / 9.77 |
| Q4_K_M + imat + emb q5_K | 496 / 18.74 | 1207 / 12.34 | 2624 / 9.76 |
| IQ4_XS + imatrix | 504 / 18.78 | 1196 / 12.51 | 2514 / 9.83 |
| IQ3_M + imatrix | 454 / 19.16 | 1059 / 13.21 | 2163 / 10.52 |
| Q2_K + imatrix | 422 / 30.1 | 969 / 17.1 | 1916 / 11.65 |

Error bars are about ±0.4 (0.8B), ±0.27 (2B), ±0.2 (4B).
- imatrix helps at 2B and 4B, as it did for Qwen2.5; at 0.8B it is within noise.
- Qwen3.5's 248K-token vocabulary makes the embedding table large, so
  `--token-embedding-type q5_K` saves 70-90 MB for no measurable loss at 2B and 4B.
- Below 4 bits quality falls as before; Q2_K without imatrix collapses.

Task tests (`ojp-cli route-eval`, `tool-eval`, `agent-eval`; CPU; thinking off
via `OJP_TEMPLATE=chatml-nothink`; prompts are the ones tuned for Qwen2.5;
machine under ~1 core of other load, so compare times only within a run of
this table):

| Model (as main, tool test) | route quick / dev | tool seen / unseen / multi | s per request seen |
|---|---|---|---|
| Qwen2.5-1.5B Q4_K_M+imat (baseline) | 10/10 / 22/24 | 38/39, 20/20, 14/14 | 2.3 |
| Qwen3.5-2B stock Q4_K_M | 9/10 / 20/24 | 33/39, 17/20, 14/14 | 6.8 |
| Qwen3.5-2B Q4_K_M+imat | 9/10 / 21/24 | 33/39, 19/20, 14/14 | 6.8 |
| Qwen3.5-2B IQ4_XS+imat | 10/10 / 23/24 | 32/39, 18/20, 14/14 | 7.1 |
| Qwen3.5-4B stock Q4_K_M | 10/10 / 24/24 | 36/39, 20/20, 14/14 | 16.1 |
| Qwen3.5-4B Q4_K_M+imat | 10/10 / 24/24 | 35/39, 20/20, 14/14 | 16.0 |
| Qwen3.5-4B IQ4_XS+imat | 10/10 / 24/24 | 36/39, 20/20, 14/14 | 16.5 |

Router only (0.8B vs the 0.5B router build): Qwen2.5-0.5B 8/10 quick, 19/24 dev;
Qwen3.5-0.8B Q4_K_M(+imat) 9/10, 18-19/24; Q8_0 10/10, 21/24; IQ4_XS+imat 9/10, 21/24.

Whole assistant (`agent-eval`, 36 tasks, router + main from the table):

| Stack | tasks | memory | wall time |
|---|---|---|---|
| Qwen2.5 0.5B + 1.5B | 32/36 | 8/8 | 8m37 |
| Qwen3.5 0.8B + 2B (Q4_K_M+imat) | 29/36 (28 on the first run) | 8/8 | 10m41 |
| Qwen3.5 0.8B + 4B (Q4_K_M+imat) | 33/36 (34 once a "$1,200" vs "1200" matching quirk is ignored) | 8/8 | n/a |

- Qwen3.5-2B fails mostly by saying "I can't access your files" or by
  summing numbers instead of reading them, i.e. it follows the Qwen2.5-tuned
  tool prompts less well. The 4B follows them as well as the 1.5B did.
- On CPU the hybrid architecture costs: cold prompt processing is roughly
  half the speed of Qwen2.5 at the same size, and the app has to snapshot
  recurrent state instead of trimming it (see `ojp-core/src/model.rs`).
  Steady-state per-request time for the 2B matches Qwen2.5-1.5B (0.44 s route
  + 1.8 s tool call); the 4B is about 2x that. Vulkan is faster for both.
- Not tested: prompts tuned for Qwen3.5, GPU speed in the app, uncensored builds.

## 8. Shrinking the Qwen3.5 router: vocabulary trimming, embedding precision, RAM

Qwen3.5 has a 248,320-token vocabulary and a tied embedding/output matrix, so
at 0.8B the embedding alone is 208 MB of the 518 MB Q4_K_M file (a third of
the model). `prune_vocab.py` keeps the special tokens, the 256 byte tokens,
every token seen in a keep-corpus (wikitext-2 *train*, the OJP calibration
texts, the OJP sources and README) and the BPE merge ancestors of those:
248,320 -> 48,736 tokens. Anything else still tokenizes, just into smaller
pieces (Chinese and emoji become bytes). Held-out wikitext-2 *test* needs
+0.7% more tokens than with the full vocabulary.

**Quantization within the trimmed vocabulary** (KL divergence from the Q8_0 of
the same trimmed model, wikitext-2 test, 100 chunks; lower is better):

| 0.8B variant (trimmed vocab) | MB | KLD | same top token |
|---|---|---|---|
| Q5_K_M + imatrix | 401 | 0.0116 | 93.9% |
| **Q4_K_M + imatrix** | **352** | **0.0378** | **89.5%** |
| Q4_K_M + imat + emb q5_0 | 346 | 0.0423 | 88.7% |
| Q4_K_M + imat + emb q4_K | 340 | 0.0501 | 87.5% |
| IQ4_XS + imatrix | 327 | 0.0509 | 87.9% |
| Q4_K_M, no imatrix | 352 | 0.0602 | 87.0% |
| Q4_K_M + imat + emb q3_K | 333 | 0.1020 | 80.5% |
| IQ3_M + imatrix | 277 | 0.1422 | 80.5% |

- The embedding doubles as the output layer, so it hurts more to squeeze
  than in Qwen2.5: keep it at the default (Q6_K); q3_K or lower is bad.
- imatrix cuts the error by 37% at 0.8B (the earlier perplexity runs hid this
  in their ±0.4 noise; KLD is far less noisy than perplexity or the 24-case tests).

**Trimmed vs full vocabulary, same quantization** (bits per byte, bpb, on the whole
wikitext-2 test; Q4_K_M + imatrix except where noted):

| Model | full vocab MB / bpb | trimmed MB / bpb | smaller | worse |
|---|---|---|---|---|
| 0.8B (Q8_0) | 812 / 0.963 | 585 / 0.970 | 28% | +0.7% |
| 2B (+emb q5_K) | 1274 / 0.8517 | 917 / 0.8592 | 28% | +0.9% |
| 4B (+emb q5_K) | 2709 / 0.7719 | 2264 / 0.7780 | 16% | +0.8% |

**The 0.8B router against the 0.5B router it would replace** (`route-eval --dev`,
24 cases; this machine also runs other work, compare within the table):

| Router | file MB | peak RAM MB | RAM, low-memory mode | ms/request | dev score |
|---|---|---|---|---|---|
| Qwen2.5 0.5B (Q4_K_M + imat + q5_0 emb) | 347 | 553 | 494 | 306 | 19/24 |
| Qwen3.5 0.8B, full vocab, Q4_K_M + imat | 529 | 1039 | 782 | 513 | 19/24 |
| **Qwen3.5 0.8B, trimmed vocab, Q4_K_M + imat** | **352** | **786** | **529** | 439 (482 low-memory) | 19/24 |

- "Low-memory mode" (`models.low_memory`, `OJP_LOW_MEMORY`): llama.cpp keeps no
  repacked copy of the weights (-205 MB) and uses 128-token scratch buffers
  (-50 MB); about 10% slower. The same switch also saves 59 MB on the 0.5B.
- Trimmed 0.8B router: same disk as the 0.5B (352 vs 347 MB), -33% against the
  untrimmed 0.8B, RAM within 35 MB of the 0.5B in low-memory mode, same
  accuracy on the router tests, about 1.4-1.6x the time per request.
- Whole assistant (36 tasks, trimmed models): 0.8B + 2B 32/36, 0.8B + 4B 34/36
  (untrimmed: 29/36 and 33-34/36); differences are within run-to-run noise.

## 9. Qwen3.5 0.8B router: probe, vocabulary size, language packs, calibration order

### The router becomes a probe

The small model's own answer to "which of four labels" is weak and varied (80-case
held-out set, never used for tuning): Qwen2.5-0.5B 66/80, Qwen3.5-0.8B 61/80 (Q8_0
58/80, so it is not the quantization), Qwen2.5-1.5B 74/80. A linear classifier on the
model's hidden state at the answer position (a "probe", fitted on 592 requests, a few
KB of weights, `scripts/make-router-probe.sh`) scores far higher on the same held-out
requests, and also on an out-of-distribution set of 42 requests with typos, slang
and some French (`ojp-core/data/router-wild.tsv`, never fitted):

| Router | generated, 80 held-out | probe, 80 held-out | generated, 42 wild | probe, 42 wild | ms: generated -> probe |
|---|---|---|---|---|---|
| Qwen2.5-0.5B (current) | 66 (82%) | 78 (97.5%) | 33 | 40 | 207 -> 133 |
| Qwen3.5-0.8B trimmed Q4_K_M | 61 (76%) | 79 (98.8%) | - | - | 282 -> 174 |
| Qwen3.5-0.8B trimmed IQ4_XS | - | 80 (100%) | 30 | 41 | 379 -> 207 |
| Qwen2.5-1.5B (main model routing) | 74 (92.5%) | - | 35 | - | 456 |

(The probe columns for the held-out set come from a probe fitted without that set;
the shipped probe is refitted on train + held-out.) The probe needs no worked
examples in the prompt (700 tokens -> about 60), so each request is faster, and it
barely depends on quantization (probe accuracy, held-out 80, per 0.8B variant):

| 0.8B trimmed variant | MB | probe held-out | 5-fold CV | peak RAM MB (low-memory) |
|---|---|---|---|---|
| Q8_0 | 585 | 100% | 99.2% | 798 |
| Q4_K_M + imat | 352 | 98.8% | 99.3% | 578 |
| IQ4_XS + imat | 327 | 100% | 99.0% | 554 (456 light) |
| IQ3_M + imat | 277 | 100% | 99.5% | 507 |
| IQ3_XXS + imat | 248 | 98.8% | 98.8% | - |
| Q2_K + imat | 245 | 97.5% | 98.5% | 476 |

For comparison the 0.5B router with a probe peaks at 614 MB. "Light" = one sequence and
a 1024-token context, used when a local main model also exists (it answers and runs tools).

### Vocabulary: how much to keep (0.8B, Q4_K_M + imatrix; bits per byte, lower is better)

| Kept tokens | MB | English (held-out wikitext) | European (fr/de/es/pt/it) | Russian | Japanese | Arabic |
|---|---|---|---|---|---|---|
| all 248,320 | 529 | 0.982 | 1.002 | 0.613 | 0.971 | 0.882 |
| 15% of them by corpus frequency | 346 | 0.999 (+1.7%) | 1.74 | - | - | - |
| 10% | 335 | 1.028 (+4.6%) | 1.94 | - | - | - |
| 5% | 324 | 1.129 (+15%) | 2.20 | - | - | - |
| 2.5% | 318 | 1.286 (+31%) | 2.48 | - | - | - |
| **every token seen in the English corpus (19.6%)** | **352** | **0.989 (+0.7%)** | 1.63 | 2.62 | 3.14 | 2.73 |

The 20/80 rule holds for English: the 20% of the vocabulary that real English text
uses costs 0.7%, and going below it quickly hurts (10% is +4.6% for 17 MB). Text in
other scripts degrades a lot because it falls back to bytes.

**Optional language packs** (extra tokens kept on top of the English 19.6%, built with
`prune_vocab.py --scripts`; English is unchanged in every row):

| Pack | MB | tokens | what it fixes |
|---|---|---|---|
| none | 352 | 48,736 | |
| Cyrillic | 369 | 67,184 | Russian 2.62 -> 0.62 (full vocabulary 0.61) |
| Arabic | 360 | 57,457 | Arabic 2.73 -> 0.90 (full 0.88) |
| CJK + kana | 405 | 107,256 | Japanese 3.14 -> 1.08 (full 0.97) |
| Latin accents | 364 | 61,503 | European 1.63 -> 1.52 (only part of the gap) |
| Cyrillic + Greek + Latin accents | 382 | 81,375 | Russian 0.62, European 1.52 |
| every script | 454 | 162,737 | Russian 0.63, Japanese 1.08, Arabic 0.91, European 1.51 |

So a language pack costs 8-53 MB and restores that language almost completely; the
cheap way to offer this is a second GGUF per pack that the user picks (the model path
is already configurable), not the default. European languages written in Latin
letters need their common words, not just accented tokens; keeping by the original
BPE id order (a rough all-language frequency rank) in addition to the English corpus
tokens helps gradually (European 1.63 -> 1.55 / 1.47 / 1.33 for ids below 20k / 40k / 80k
for +5 / +14 / +36 MB) but is a worse use of bytes than the corpus for English alone
(ids below 49k with no corpus: English 1.088).

### Order of operations and calibration text (0.8B trimmed, Q4_K_M; KL divergence from the
Q8_0 of the same trimmed model, lower is better)

| imatrix | wikitext-2 test KLD | OJP task text KLD |
|---|---|---|
| none | 0.0602 | 0.0378 |
| from the full-vocabulary model, wikitext (trim after) | 0.0378 | 0.0235 |
| from the trimmed model, wikitext (trim before) | 0.0378 | 0.0235 |
| from the trimmed model, OJP calibration mix | 0.0386 | 0.0247 |
| from the trimmed model, OJP task text only | 0.0393 | 0.0251 |

- Trim-then-imatrix and imatrix-then-trim give identical results; the order does not matter.
- Generic text calibrates as well as OJP's own text, even when scored on OJP-style text.
- The imatrix is worth about 38% of the quantization error in both cases.

## Not yet tested

- Prune → **heal** (LoRA/distillation) → quantize.
- The hybrid scan on 1.5B (the plain study is in section 6).
- Hybrid mixes built on an IQ3_XXS base.
- aarch64 (phone/laptop) speed and memory numbers.
