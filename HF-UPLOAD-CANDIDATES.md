# Files recommended for Hugging Face (nothing is uploaded; decide against these stats)

Updated 2026-10-24. Uploads would go to the `OpenJarvisProject` org like the 0.5B router
(`OpenJarvisProject/Qwen2.5-0.5B-Instruct-OJP-GGUF`, already public).

## Recommended: Qwen2.5-1.5B-Instruct (Apache-2.0)
Draft card: `hf-cards/qwen2.5-1.5b-instruct-ojp-gguf/README.md`. Both are quantizations of the
official model, no fine-tuning. Local files: `outputs/cand-1.5b/`.

| | Q4_K_M + imatrix | IQ4_XS + imatrix | Official Qwen Q4_K_M (for reference) |
|---|---|---|---|
| Size | 986 MB | 896 MB | 1117 MB |
| SHA-256 | c98e1a58...b669d6 | 0ac1b372...e364c | 6a1a2eb6...09e1 |
| Perplexity (Q8_0 9.71) | 9.95 | 10.08 | not measured the same way |
| Same top token as f16 | 90.8% | 89.2% | n/a |
| HellaSwag / Wino / ARC / MMLU / TQA | 64.3 / 63.3 / 55.8 / 35.1 / 25.5 | 64.7 / 63.9 / 54.9 / 35.2 / 26.7 | n/a |
| CPU generation (llama-bench) | 28.0 tok/s | 30.4 tok/s | n/a |
| Peak RAM in OpenJarvis | 1844 MB | 1172 MB | n/a |
| Load + first request in OpenJarvis | 7.1 s | 11.3 s | n/a |
| OJP route test / dev | 10/10, 24/24 | 10/10, 23/24 | 10/10, 23/24 |
| OJP tools seen / unseen / multi | 37/39, 20/20, 14/14 | 36/39, 20/20, 14/14 | 36/39, 19/20, 14/14 |
| OJP agent tasks | 26/28 | 25/28 | 26/28 |

Quality gap between the two is at noise level (about 1.6 points of agreement, 1.3% perplexity,
at most one test). The real tradeoff is memory (0.67 GB less) against speed in the bundled engine.

## Not recommended
- **Qwen2.5-3B-Instruct**: its license is the Qwen Research License (`other`), NOT Apache-2.0,
  so it fails the open-source requirement. Also no measurable gain on OJP's tests
  (tools 37/39 and 20/20 and 14/14, agent 25/28 or 26/28, same as the 1.5B) at about 2x the time
  per request and 2.0 to 3.4 GB RAM. Quantization study numbers for it are in RESULTS.md section 2.
- IQ3_M and below for the 1.5B: clearly worse (KL 0.12) and slower. 0.5B: already published.

## If a bigger model is wanted later
Qwen2.5-7B-Instruct is Apache-2.0 (see `7B-NOTES.md`; run only by hand because of the GPU crashes).
