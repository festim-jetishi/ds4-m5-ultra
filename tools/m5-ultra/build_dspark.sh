#!/usr/bin/env bash
set -euo pipefail

# One-command builder for DeepSeek-V4.1-Flash-1017-DSpark.gguf (8.0 GB).
# Downloads ONLY the 5 required files (~7.4 GiB, instead of the ~600 GiB full
# checkpoint) from Hugging Face and extracts the lossless DSpark drafter GGUF.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REV="1bae0e1088993b9b82de08f59de5be5a34e42170"
HF_BASE="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/${REV}"
SRC_DIR="${DS4_DSPARK_SRC_DIR:-$REPO_ROOT/models/DeepSeek-V4.1-Flash-1017}"
OUT_GGUF="${1:-$REPO_ROOT/gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf}"

FILES=(
  "config.json"
  "model.safetensors.index.json"
  "model-00044-of-00048.safetensors"
  "model-00045-of-00048.safetensors"
  "model-00046-of-00048.safetensors"
)

mkdir -p "$SRC_DIR" "$(dirname "$OUT_GGUF")"

echo "==> Downloading only the 5 DSpark source files (~7.4 GiB total)..."
for f in "${FILES[@]}"; do
  dest="$SRC_DIR/$f"
  echo "  -> $f"
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl -fL --progress-bar -C - -H "Authorization: Bearer $HF_TOKEN" -o "$dest" "$HF_BASE/$f"
  else
    curl -fL --progress-bar -C - -o "$dest" "$HF_BASE/$f"
  fi
done

echo "==> Extracting lossless DSpark GGUF -> $OUT_GGUF"
python3 "$REPO_ROOT/gguf-tools/deepseek41_dspark.py" \
  --hf "$SRC_DIR" \
  --source-revision "$REV" \
  --out "$OUT_GGUF"

echo "==> Verifying output GGUF..."
python3 "$REPO_ROOT/gguf-tools/deepseek41_dspark.py" \
  --hf "$SRC_DIR" \
  --source-revision "$REV" \
  --out "$OUT_GGUF" \
  --verify

echo "Done: $OUT_GGUF ($(du -h "$OUT_GGUF" | awk '{print $1}'))"
echo "(You may delete $SRC_DIR after verification to reclaim ~7.4 GiB.)"
