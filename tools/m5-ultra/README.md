# Model Builders (`tools/m5-ultra/`)

One-command, minimal-download scripts to reproduce the exact GGUF files benchmarked on Mac Studio M5 Ultra (256 GiB) without downloading full ~500–600 GiB checkpoints.

## 1. Build DSpark Drafter (`8.0 GB` GGUF from only `7.4 GiB` download)

Downloads only the 5 required files (`config.json`, `model.safetensors.index.json`, and shards `00044–00046`) from [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) and runs `gguf-tools/deepseek41_dspark.py`:

```bash
./tools/m5-ultra/build_dspark.sh
```

Output: `gguf/DeepSeek-V4.1-Flash-1017-DSpark.gguf` (required for `--dspark` / `--mtp-exact-sampling`).

## 2. Build Mixed `Q2 + L30–39 MXFP4` Model (Optional — `67.2 GiB` HTTP-Range fetch)

> Note: Stock `./download_model.sh ds41f-q2` (pure Q2, `151.8 GiB` resident) works out of the box, uses **31.6 GiB less RAM**, and decodes slightly faster. Run this only if you want the exact higher-precision `L30–39 MXFP4` splice (`183.4 GiB` resident) used in the benchmarks.

First download the pure Q2 base (`./download_model.sh ds41f-q2`), then run:

```bash
./tools/m5-ultra/build_mixed_l30_39_mxfp4.sh
```

How it works:
1. Reads local Q2 + all 9 remote MXFP4 shard headers via HTTP Range (`gguf_inspect.py`, ~10 MiB total).
2. Streams **only layers 30–39** (`67.2 GiB` instead of `500+ GiB`, resumable) and verifies every tensor's cosine similarity against Q2 (`mixfetch.py` + `mixcheck.c`).
3. Writes `gguf/DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf` (`372.2 GiB`) preserving ds4's resident-first, Engram-last disk layout (`mixwrite.py`).

## 3. Optional Directional Ablation / Refusal Removal (`apply_abliteration.sh`)

Applies the MIT-licensed rank-2 directional refusal ablation (`W' = W - s·Q·Qᵀ·W`) from [`mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`](https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated) directly to the Q8_0 `attn_output_b` and `ffn_down_shexp` tensors of either the mixed `Q2-L30-39-MXFP4` GGUF or pure Q2 GGUF — using the included `43 KB` `abliteration-edit.npz` and fixed-seed stochastic Q8_0 rounding (pure NumPy, ~15 seconds, zero large downloads, APFS copy-on-write `cp -c` on macOS):

```bash
# Patches gguf/DeepSeek-V4.1-Flash-1017-Q2-L30-39-MXFP4.gguf -> ...-abliterated.gguf
./tools/m5-ultra/apply_abliteration.sh

# Or pass any compatible DeepSeek-V4.1-Flash GGUF (e.g. pure Q2):
./tools/m5-ultra/apply_abliteration.sh \
  gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf \
  gguf/DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix-abliterated.gguf
```

- Routed experts, embeddings, Engram tables, and DSpark layers remain bit-identical (`0%` speed change).
- Optional FP8 reference cross-checker: `ablfetch.py` downloads only the edited FP8 tensors via HTTP Range requests for `--verify-dir` comparison.
- *Disclaimer (from upstream `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`):* an abliterated model will comply with harmful requests; you are responsible for how you use it.
