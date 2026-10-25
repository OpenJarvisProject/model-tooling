# Qwen2.5-7B-Instruct on the RX 580 (8 GB): what was learned before the files were deleted

Written 2026-10-24 when the 7B files were removed to free disk. The 7B work was
stopped on 2026-10-12 after the GPU crashed twice (both needed a hard reboot) and the
user ruled out automated 7B tests. Scripts are kept but refuse to run unless
`OJP_USER_APPROVED_7B=yes` is set; the systemd units are in `disabled-7b/`.

## The crashes
- 2026-10-12 08:31, during the 7B study: `ring gfx` timeout (seen in kwin), "device lost
  from bus", recovery failed, reboot needed. A second crash followed. Both happened under
  long sustained Vulkan load from the 7B runs (HellaSwag over 1000 tasks etc.).
- Lesson: the 7B fits in 8 GB, but running it flat out for long stretches on this card
  is not safe. Do any future 7B run by hand, short, with the desktop otherwise idle.

## Perplexity sweep (wikitext-2 test, 64x512, all layers on the GPU)
| Variant | MB | ppl |
|---|---|---|
| IQ2_M+imat | 2780 | 9.01 |
| IQ3_XXS+imat | 3115 | 8.22 |
| IQ3_M+imat | 3574 | 7.89 |
| Q3_K_M+imat | 3808 | 7.83 |
| IQ4_XS+imat | 4218 | 7.51 |
| Q4_K_M | 4683 | 7.53 |
| Q4_K_M+imat | 4683 | 7.50 |
| Q5_K_M+imat | 5445 | 7.38 |
| Q6_K | 6254 | 7.35 |
Q4_K_M+imat and IQ4_XS+imat are the best size/quality points; Q5_K_M and up gain little.
VRAM readings for Q4_K_M+imat (1372) and IQ3_M were bogus (measurement glitch); the other
rows are whole-process VRAM at 4K context: roughly file size + 100 to 150 MB.

## Deep comparison (only Q5_K_M+imat finished)
HellaSwag 79.3, Winogrande 71.2, ARC 53.9, MMLU 39.5, TruthfulQA 37.1, agent quiz 29/34,
about 28 tokens/s generation on the GPU. Q4_K_M+imat agreement with the reference: mean KLD
0.0375 (higher than the 1.5B's 0.029, same order). IQ4_XS and IQ3_M never ran.

## Pruning (earlier, README "7B model tested")
7B was less prunable than 0.5B/1.5B: only 2 of 28 layers removed at the 1.5x perplexity
threshold (7.1% vs 20.8% and 21.4%), likely because a fixed ratio is a tighter absolute
limit when the baseline is already good. Original vs pruned (26 layers): weights 4168 vs
3901 MiB VRAM, KV cache 112 vs 104 MiB, compute buffer 304 MiB unchanged (it scales with
n_ubatch), host RSS about 4572 vs 4305 MiB.

## What was deleted
f16 conversion and the five quantized GGUFs (31 GB), the Q5_K_M KL reference (1.5 GB),
`outputs/7b/` (two Q4_K_M files, 8.5 GB) and the 7B download in `.hf-cache`. Kept: this
file, `outputs/7b-study/STATUS.md`, logs and step JSON. Everything can be recreated from
`Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) with `run_7b_pipeline.sh` if the user approves.
