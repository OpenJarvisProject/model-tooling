#!/bin/bash
# DISABLED 2026-10-12: 7B runs crashed the GPU/system twice (hard reboots).
# Not to be run automatically; only Claude, on the user's explicit request.
[ "${OJP_USER_APPROVED_7B:-}" = "yes" ] || { echo "7B tests are disabled (see RESULTS.md); refusing to run." >&2; exit 1; }
set -euo pipefail
cd /media/openjarvisproject-model-tooling
source .venv-cpu/bin/activate

# Root partition (/) is at 99% full, 5.5GB free - nowhere near enough
# for a 7B model's ~14GB of shards. Redirect HF's cache to /media
# (123GB free) instead of the default ~/.cache/huggingface on root.
export HF_HOME=/media/openjarvisproject-model-tooling/.hf-cache

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUT=outputs/7b
mkdir -p "$OUT"

echo "=== [1/6] Downloading + converting original to GGUF f16 ==="
SNAPSHOT=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID'))")
python /media/openjarvisproject-model-tooling/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile "$OUT/original-f16.gguf" "$SNAPSHOT"

echo "=== [2/6] Quantizing original to Q4_K_M ==="
/tmp/llama-cpp-convert/build/bin/llama-quantize "$OUT/original-f16.gguf" "$OUT/original-q4_k_m.gguf" Q4_K_M
rm -f "$OUT/original-f16.gguf"

echo "=== [3/6] Pruning (iterative, perplexity-validated, in-place remove/restore) ==="
python prune_layers.py "$MODEL_ID" "$OUT/pruned-hf" --remove-fraction 0.9 --max-ppl-ratio 1.5

echo "=== [4/6] Converting pruned model to GGUF f16 ==="
python /media/openjarvisproject-model-tooling/llama.cpp/convert_hf_to_gguf.py --outtype f16 --outfile "$OUT/pruned-f16.gguf" "$OUT/pruned-hf"

echo "=== [5/6] Quantizing pruned to Q4_K_M ==="
/tmp/llama-cpp-convert/build/bin/llama-quantize "$OUT/pruned-f16.gguf" "$OUT/pruned-q4_k_m.gguf" Q4_K_M
rm -f "$OUT/pruned-f16.gguf"

echo "=== [6/6] Sizes ==="
ls -lh "$OUT"/*.gguf

echo "=== DONE ==="
