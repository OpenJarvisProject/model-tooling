#!/bin/bash
set -euo pipefail
cd /media/openjarvisproject-model-tooling
source .venv-cpu/bin/activate
export HF_HOME=/media/openjarvisproject-model-tooling/.hf-cache

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
OUT=outputs/7b

echo "=== [3/6] Pruning (resumed - steps 1-2 already done) ==="
python prune_layers.py "$MODEL_ID" "$OUT/pruned-hf" --remove-fraction 0.9 --max-ppl-ratio 1.5

echo "=== [4/6] Converting pruned model to GGUF f16 ==="
python /tmp/llama-cpp-convert/convert_hf_to_gguf.py --outtype f16 --outfile "$OUT/pruned-f16.gguf" "$OUT/pruned-hf"

echo "=== [5/6] Quantizing pruned to Q4_K_M ==="
/tmp/llama-cpp-convert/build/bin/llama-quantize "$OUT/pruned-f16.gguf" "$OUT/pruned-q4_k_m.gguf" Q4_K_M
rm -f "$OUT/pruned-f16.gguf"

echo "=== [6/6] Sizes ==="
ls -lh "$OUT"/*.gguf

echo "=== DONE ==="
