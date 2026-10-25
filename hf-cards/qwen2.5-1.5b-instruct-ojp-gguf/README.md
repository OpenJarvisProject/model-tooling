---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE
base_model: Qwen/Qwen2.5-1.5B-Instruct
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

# Qwen2.5-1.5B-Instruct, OJP GGUF (Q4_K_M and IQ4_XS, imatrix)

DRAFT, not uploaded. Quantizations of
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache-2.0, © Alibaba Cloud / the Qwen team), made for
[OpenJarvis](https://github.com/OpenJarvisProject/ojp), where it is the main
model that routes requests, picks tools and writes answers. Same model, no
fine-tuning, only quantized differently with an importance matrix.

| File | Size | SHA-256 |
|---|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m-imat.gguf` | 986 MB | `c98e1a583924c7158e7b56d161ea121a864755de6d60c148228ea4e4abb669d6` |
| `qwen2.5-1.5b-instruct-iq4_xs-imat.gguf` | 896 MB | `0ac1b372e3d9786e8af763a41f9a5fc6353d16d57b47a69c4794c9768cee364c` |
| `imatrix.gguf` (calibration data, for reproducing) | 4.1 MB | see model-tooling |

## Which one

- **Q4_K_M + imatrix**: best agreement with the original; faster in OpenJarvis's
  bundled llama.cpp (it is repacked into a quicker layout) but uses about 1.8 GB RAM.
- **IQ4_XS + imatrix**: 9% smaller and about 1.2 GB RAM, but about 0.5 s slower per
  request in OpenJarvis's bundled llama.cpp (newer llama.cpp runs it faster).

## Measured vs the unquantized original (f16, 3094 MB)

llama.cpp, RX 580 (Vulkan) for quality; CPU (Ryzen 5 4500, 6 threads) for speed.
Method and raw data: model-tooling `RESULTS.md` section 6.

| | f16 | Q4_K_M+imat | IQ4_XS+imat |
|---|---|---|---|
| Size | 3094 MB | 986 MB | 896 MB |
| wikitext-2 perplexity (Q8_0 = 9.71) | n/a | 9.95 | 10.08 |
| KL divergence vs original | 0 | 0.029 | 0.039 |
| Same top token as original | 100% | 90.8% | 89.2% |
| HellaSwag (1000) | 66.4% | 64.3% | 64.7% |
| Winogrande | 63.7 ± 1.4% | 63.3 ± 1.4% | 63.9 ± 1.4% |
| ARC (val) | 57.8 ± 1.7% | 55.8 ± 1.7% | 54.9 ± 1.7% |
| MMLU (val) | 36.2 ± 1.2% | 35.1 ± 1.2% | 35.2 ± 1.2% |
| TruthfulQA (mc) | 26.2 ± 1.5% | 25.5 ± 1.5% | 26.7 ± 1.5% |
| Generation speed, CPU (llama-bench) | n/a | 28.0 tok/s | 30.4 tok/s |
| Peak RAM in OpenJarvis (CPU) | n/a | 1844 MB | 1172 MB |

Every benchmark is within error of the original. On OpenJarvis's own tests
(39 + 20 tool requests, 14 multi-step, 28 scripted tasks) both files are
within one test of the official Q4_K_M and of each other.
