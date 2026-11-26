#!/usr/bin/env bash
set -euo pipefail

# One-command builder for DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf (372.2 GiB).
# Streams ONLY layers 30-39 (~67.2 GiB via HTTP Range requests, instead of 500+ GiB)
# from antirez/deepseek-v4.1-flash-gguf, verifies cosine similarity against the Q2
# base weights, and writes the spliced GGUF preserving ds4's resident-first layout.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LAYERS="${LAYERS:-30-39}"
BASE_GGUF="${1:-$REPO_ROOT/gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf}"
OUT_GGUF="${2:-$REPO_ROOT/gguf/DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf}"
WORK_DIR="${DS4_MIXQ_WORK_DIR:-$REPO_ROOT/gguf/.mixq-work}"
HF_BASE="https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/main"

if [[ ! -f "$BASE_GGUF" ]]; then
  echo "ERROR: Base Q2 GGUF not found: $BASE_GGUF" >&2
  echo "Run ./download_model.sh ds41f-q2 first." >&2
  exit 1
fi

if [[ -e "$OUT_GGUF" ]]; then
  echo "Output already exists: $OUT_GGUF" >&2
  exit 0
fi

mkdir -p "$WORK_DIR"

echo "==> Building mixcheck cosine verifier..."
cc -O2 -o "$WORK_DIR/mixcheck" "$SCRIPT_DIR/mixcheck.c" -lm

echo "==> Reading local Q2 header..."
if [[ ! -f "$WORK_DIR/q2.json" ]]; then
  python3 "$SCRIPT_DIR/gguf_inspect.py" "$BASE_GGUF" "$WORK_DIR/q2.json"
fi

echo "==> Reading remote MXFP4 shard headers (~10 MiB total via HTTP Range)..."
for i in $(seq 1 9); do
  if [[ ! -f "$WORK_DIR/mx${i}.json" ]]; then
    shard=$(printf "DeepSeek-V4.1-Flash-1017-MXFP4-experts-%05d-of-00009.gguf" "$i")
    python3 "$SCRIPT_DIR/gguf_inspect.py" "$HF_BASE/$shard" "$WORK_DIR/mx${i}.json"
  fi
done

echo "==> Streaming layers $LAYERS (~67.2 GiB, resumable + cosine-verified)..."
python3 "$SCRIPT_DIR/mixfetch.py" "$LAYERS" "$WORK_DIR/l${LAYERS}.pack" \
  --base "$BASE_GGUF" \
  --meta-dir "$WORK_DIR" \
  --mixcheck "$WORK_DIR/mixcheck"

echo "==> Writing spliced GGUF -> $OUT_GGUF (~4-5 min on NVMe)..."
python3 "$SCRIPT_DIR/mixwrite.py" "$BASE_GGUF" "$OUT_GGUF" "$LAYERS" "$WORK_DIR/l${LAYERS}.pack"

echo "Done: $OUT_GGUF ($(du -h "$OUT_GGUF" | awk '{print $1}'))"
echo "(You may remove $WORK_DIR to reclaim ~67 GiB of temporary pack space.)"
