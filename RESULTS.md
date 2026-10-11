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

## Not yet tested

- Prune → **heal** (LoRA/distillation) → quantize.
- The full quant study and hybrid scan on 1.5B.
- Hybrid mixes built on an IQ3_XXS base.
- aarch64 (phone/laptop) speed and memory numbers.
