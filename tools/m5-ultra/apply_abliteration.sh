#!/usr/bin/env bash
set -euo pipefail

# One-command in-place abliteration patcher for DeepSeek-V4.1-Flash GGUFs.
# Clones BASE -> OUT using APFS copy-on-write (cp -c, instant & minimal extra
# disk space on macOS) and patches only the Q8_0 attn_output_b and
# ffn_down_shexp weights using abliteration-edit.npz (~43 KB, MIT license,
# from mrfakename/DeepSeek-V4.1-Flash-1017-abliterated).
#
# Works on either DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf or the pure Q2
# base DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BASE_GGUF="${1:-$REPO_ROOT/gguf/DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf}"
OUT_GGUF="${2:-${BASE_GGUF%.gguf}-abliterated.gguf}"
WORK_DIR="${DS4_ABL_WORK_DIR:-$REPO_ROOT/gguf/.abl-work}"

if [[ ! -f "$BASE_GGUF" ]]; then
  echo "ERROR: Base GGUF not found: $BASE_GGUF" >&2
  echo "Pass the path to your Q2 or Q2-L30-39-MXFP4 GGUF as the first argument." >&2
  exit 1
fi

if [[ -e "$OUT_GGUF" ]]; then
  echo "Output already exists: $OUT_GGUF" >&2
  exit 0
fi

mkdir -p "$WORK_DIR"

echo "==> Inspecting GGUF header..."
python3 "$SCRIPT_DIR/gguf_inspect.py" "$BASE_GGUF" "$WORK_DIR/base.json"

echo "==> Cloning $BASE_GGUF -> $OUT_GGUF..."
if [[ "$(uname -s)" == "Darwin" ]]; then
  cp -c "$BASE_GGUF" "$OUT_GGUF"
else
  cp --reflink=auto "$BASE_GGUF" "$OUT_GGUF"
fi

echo "==> Applying directional ablation patch (W' = W - s * Q * Q^T * W)..."
python3 "$SCRIPT_DIR/ablconv.py" \
  --base "$BASE_GGUF" \
  --base-json "$WORK_DIR/base.json" \
  --edit "$SCRIPT_DIR/abliteration-edit.npz" \
  --write "$OUT_GGUF"

echo "Done: $OUT_GGUF"
