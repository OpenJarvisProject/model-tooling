---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: quantized
language:
- en
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- imatrix
- openjarvisproject
---

# Qwen2.5-0.5B-Instruct, OJP GGUF (Q4_K_M + imatrix + q5_0 embedding)

A 347 MB quantization of [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
(Apache-2.0, © Alibaba Cloud / the Qwen team), made for
[OpenJarvis](https://github.com/OpenJarvisProject/ojp). There it is the
always-loaded **router** model: it classifies each request, and a bigger
model writes the answers. It is the same model with no fine-tuning, only
quantized differently.

| File | SHA-256 |
|---|---|
| `qwen2.5-0.5b-instruct-q4_k_m-imat-embq5_0.gguf` (346.8 MB) | `225dd294b01dab1cd24b5d404e12bc591dc9335d99f8a1f584e7b4c5bca86186` |
| `imatrix.gguf` (calibration data, for reproducing) | `33fbfea080d60424d6d69428703810ee38cbb90abe2b0ed4444432f0df2d21bf` |

## Why it is smaller at the same quality

Stock Q4_K_M keeps the 151,936 × 896 token embedding at Q8_0: 28% of the
parameters and the largest single block of the file. Qwen's hidden size
of 896 isn't a multiple of 256, so K-quant types can't be used for it.
Q5_0 (block size 32) can. Combined with an importance matrix, the result
is 13% smaller than the stock Q4_K_M (398 MB) with a lower perplexity.

## Measured vs the unquantized original (f16, 994 MB)

All runs used llama.cpp on an RX 580 (Vulkan). The method and raw data
are in
[model-tooling/RESULTS.md](https://github.com/OpenJarvisProject/model-tooling/blob/main/RESULTS.md).

| | f16 original | **this file** |
|---|---|---|
| Size | 994 MB | **347 MB** |
| KL divergence vs original | 0 | 0.031 |
| Same top token as original | 100% | 90.0% |
| HellaSwag (1000) | 50.8% | 51.1% |
| Winogrande | 57.4 ± 1.4% | 56.2 ± 1.4% |
| ARC (val) | 49.8 ± 1.7% | 47.4 ± 1.7% |
| MMLU (val) | 30.4 ± 1.2% | 30.2 ± 1.2% |
| TruthfulQA (mc) | 25.0 ± 1.5% | 25.5 ± 1.5% |
| Generation speed, CPU (Ryzen 5 4500) | 30 tok/s | 80 tok/s |
| Peak RAM, CPU-only | 1054 MB | 462 MB |

Every benchmark is within error of the original. For reference, wikitext-2
test perplexity (64×512) is: Q8_0 14.69, stock Q4_K_M 15.07, this file 15.09,
and Q4_K_M + imatrix with the default Q8_0 embedding 14.94.

Like any 0.5B model, it is weak at facts. Use it for classification and
routing, not as your only answering model.

## How it was made

llama.cpp `23b0202` (2026-10-10):

```bash
python convert_hf_to_gguf.py Qwen2.5-0.5B-Instruct --outtype f16 --outfile f16.gguf
llama-quantize f16.gguf q8_0.gguf Q8_0
llama-imatrix -m q8_0.gguf -f wikitext2-train.txt -o imatrix.gguf --chunks 100 -c 512
llama-quantize --imatrix imatrix.gguf --token-embedding-type q5_0 \
    f16.gguf qwen2.5-0.5b-instruct-q4_k_m-imat-embq5_0.gguf Q4_K_M
```

The imatrix is calibrated on wikitext-2 *train*; every number above is on
held-out data.
