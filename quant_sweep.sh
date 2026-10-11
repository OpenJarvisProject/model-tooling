#!/bin/bash
set -euo pipefail
cd /media/openjarvisproject-model-tooling
QUANTIZE=/tmp/llama-cpp-convert/build/bin/llama-quantize
PERPLEXITY=/tmp/llama-cpp-convert/build/bin/llama-perplexity
CORPUS=eval_corpus.txt
OUT=outputs/quant-sweep
mkdir -p "$OUT"

LEVELS="Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K"

echo "model,quant,size_mb,perplexity" > "$OUT/results.csv"

for model in original pruned original-1.5b pruned-1.5b; do
  F16="gguf-compare/${model}-f16.gguf"
  for level in $LEVELS; do
    OUTFILE="$OUT/${model}-${level}.gguf"
    echo "=== quantizing $model @ $level ==="
    $QUANTIZE "$F16" "$OUTFILE" "$level" > /dev/null 2>&1
    SIZE_MB=$(du -m "$OUTFILE" | cut -f1)
    echo "=== measuring perplexity: $model @ $level ==="
    PPL=$($PERPLEXITY -m "$OUTFILE" -f "$CORPUS" --chunks 4 2>&1 | grep -oP "Final estimate: PPL = \K[0-9.]+" || echo "FAILED")
    echo "$model,$level,$SIZE_MB,$PPL" | tee -a "$OUT/results.csv"
    rm -f "$OUTFILE"
  done
done

echo "=== DONE, results in $OUT/results.csv ==="
cat "$OUT/results.csv"
