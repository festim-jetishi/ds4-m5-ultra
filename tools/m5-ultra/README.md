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
