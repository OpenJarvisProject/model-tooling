# Overnight study, 2026-10-25/26: Qwen2.5 vs Qwen3.5 vs uncensored builds

Everything needed to repeat it and to read the raw numbers. The model files themselves
were deleted afterwards to free disk; `artifact-manifest.csv` lists the size and sha256
of the main ones as built.

## Sources (all Apache-2.0 tagged on Hugging Face, same tokenizer within a family)

| Role | File used | Where from |
|---|---|---|
| Qwen2.5-0.5B / 1.5B (regular) | `original-f16.gguf`, `original-1.5b-f16.gguf` | converted earlier from Qwen/Qwen2.5-*-Instruct |
| Qwen2.5-0.5B uncensored | `Qwen2.5-0.5B-Instruct-abliterated.f16.gguf` | mradermacher/Qwen2.5-0.5B-Instruct-abliterated-GGUF (huihui-ai's abliteration) |
| Qwen2.5-1.5B uncensored | `Qwen2.5-1.5B-Instruct-abliterated.f16.gguf` | mradermacher/Qwen2.5-1.5B-Instruct-abliterated-GGUF (huihui-ai) |
| Qwen3.5 0.8B / 2B / 4B (regular) | `Qwen3.5-*-BF16.gguf` | unsloth/Qwen3.5-{0.8B,2B,4B}-GGUF (conversions of the official weights) |
| Qwen3.5 0.8B uncensored | `Huihui-Qwen3.5-0.8B-abliterated.f16.gguf` | mradermacher/Huihui-Qwen3.5-0.8B-abliterated-GGUF |
| Qwen3.5 2B / 4B uncensored | `...-Uncensored-HauhauCS-Aggressive-BF16.gguf` | HauhauCS/Qwen3.5-{2B,4B}-Uncensored-HauhauCS-Aggressive |

llama.cpp build cbe28bb (2026-10-13) for quantize/imatrix/perplexity/tokenize; the app's
ojp-cli from the `ojp` repo (tool-eval, route-eval, agent-eval, refusal-eval, route-feats,
tool-feats). Tokenizer hashes: Qwen2.5 family `b1671b30`/`b0a02c3b`, Qwen3.5 family
`86b06fa6`/`f43208bf` (tokens/merges md5 prefix), so one keep-ids file per family works for
every size and for the uncensored builds.

## Recipe (what `optimize_model.py` does)

1. imatrix from the Q8_0 of the source, 100 chunks of wikitext-2 train.
2. stock = Q4_K_M without imatrix; imat = Q4_K_M with it; trimmed = `prune_vocab.py`
   (keep-ids from tokenizing `vocab/keep-corpus.txt` with the family's tokenizer), then
   quantize with the imatrix (embedding q5_K for 2B/4B, default for smaller).
3. Router models only: `cut_layers.py` to N blocks, quantize, `scripts/make-router-probe.sh`.
4. Measure with `compare_groups.py` (reads `/media/ojp-eval/<model>/summary.jsonl`; copies here
   as `summary-*.jsonl`).

## Pitfalls hit

- `tool-eval` needs `OJP_N_CTX=4096` (its prompt is about 2,250 tokens); at 2048 every case fails.
- Run long chains from script files with literal paths; never `pkill -f <pattern>` when the
  pattern also appears in your own command line (it kills your shell).
- A step that fails halfway can leave a truncated GGUF: delete it before rerunning, and do
  not delete source files unless the step exited with status 0.
- Disk filled up three times; the 4B BF16 conversions need 7 GB temporary space each.
