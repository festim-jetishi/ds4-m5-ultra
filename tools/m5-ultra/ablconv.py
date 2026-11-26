#!/usr/bin/env python3
"""Apply directional refusal ablation (W' = W - s * Q * Q^T * W) directly to the
Q8_0 attn_output_b and ffn_down_shexp tensors of a DeepSeek-V4.1-Flash GGUF.

Uses abliteration-edit.npz (from mrfakename/DeepSeek-V4.1-Flash-1017-abliterated,
MIT license) and requantizes edited Q8_0 tensors with fixed-seed stochastic
rounding so the directional edit is not undone by quantization rounding.

Works on either the pure Q2 GGUF or the mixed Q2-L30-39-MXFP4 GGUF in seconds
using pure NumPy (no large checkpoint download needed). Optionally cross-checks
against FP8 tensors fetched by ablfetch.py via --verify-dir.
"""

import argparse
import json
import os
import sys
import numpy as np

PARTS = (("attn.wo_b", "attn_output_b"), ("ffn.shared_experts.w2", "ffn_down_shexp"))
Q8 = np.dtype([("d", "<f2"), ("q", "i1", 32)])
SEED = 20261022


def deq(raw, rows, cols):
    b = np.frombuffer(raw, dtype=Q8)
    return (b["d"].astype(np.float32)[:, None] * b["q"].astype(np.float32)).reshape(rows, cols)


def q8_stochastic(x, rng):
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(axis=1)
    d = amax / 127.0
    inv = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0)
    v = blocks * inv[:, None]
    q = np.clip(np.floor(v + rng.random(v.shape, dtype=np.float32)), -127, 127).astype(np.int8)
    out = np.empty(len(blocks), dtype=Q8)
    out["d"] = d.astype(np.float16)
    out["q"] = q
    return out.tobytes()


class VerifyFiles:
    def __init__(self, work_dir):
        self.work_dir = work_dir
        with open(os.path.join(work_dir, "tensors.json"), "r", encoding="utf-8") as fp:
            self.meta = json.load(fp)

    def info(self, name):
        return self.meta[name]

    def read(self, name):
        with open(os.path.join(self.work_dir, "t", name), "rb") as fp:
            return fp.read()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Source GGUF (read-only)")
    parser.add_argument("--base-json", required=True, help="Header JSON from gguf_inspect.py")
    parser.add_argument("--edit", default=os.path.join(script_dir, "abliteration-edit.npz"),
                        help="Path to abliteration-edit.npz")
    parser.add_argument("--write", default=None,
                        help="Target GGUF clone to patch in place (e.g. cp -c BASE OUT)")
    parser.add_argument("--verify-dir", default=None,
                        help="Optional ablfetch.py work dir to cross-check against FP8 reference")
    args = parser.parse_args()

    with open(args.base_json, "r", encoding="utf-8") as fp:
        hdr = json.load(fp)
    tens = {t[0]: t for t in hdr["tensors"]}

    z = np.load(args.edit)
    Q, S = z["Q"].astype(np.float32), z["scales"]

    quant = files = qtype_q8 = None
    if args.verify_dir:
        gguf_tools = os.path.join(repo_root, "gguf-tools")
        sys.path.insert(0, gguf_tools)
        from deepseek41_quantize import NativeQuantizer  # noqa: E402
        from glm53_quantize import QTYPE_Q8_0  # noqa: E402
        quant = NativeQuantizer(os.path.join(gguf_tools, "libds4quants.dylib"))
        files = VerifyFiles(args.verify_dir)
        qtype_q8 = QTYPE_Q8_0

    rng = np.random.default_rng(SEED)
    base = open(args.base, "rb")
    out = open(args.write, "r+b") if args.write else None
    edited = untouched = exact = mismatch = 0

    for il in range(40):
        for k, (src, dst) in enumerate(PARTS):
            name = "blk.%d.%s.weight" % (il, dst)
            _, dims, ty, off = tens[name]
            assert ty == 8, (name, ty)  # Q8_0
            cols, rows = dims[0], dims[1]
            nbytes = rows * cols // 32 * 34
            base.seek(hdr["data_start"] + off)
            ours = base.read(nbytes)

            theirs = None
            if quant is not None:
                theirs = quant.encode(quant.to_f32(files, "layers.%d.%s.weight" % (il, src)), qtype_q8)
                assert len(theirs) == nbytes, (name, len(theirs), nbytes)

            s = float(S[il, k])
            if s == 0.0:
                untouched += 1
                if theirs is not None:
                    if theirs == ours:
                        exact += 1
                    else:
                        mismatch += 1
                        print("%-28s s=0 but FP8 reference differs!" % name, flush=True)
                a = ours
            else:
                edited += 1
                w0 = deq(ours, rows, cols)
                delta = -s * (Q @ (Q.T @ w0))
                a = q8_stochastic(w0 + delta, rng)
                want = Q.T @ delta
                checks = [("A", a)]
                if theirs is not None:
                    checks.append(("B", theirs))
                line = []
                for label, raw in checks:
                    diff = deq(raw, rows, cols) - w0
                    got = Q.T @ diff
                    keep = float((got * want).sum() / (want * want).sum())
                    off_q = diff - Q @ got
                    noise = float(np.linalg.norm(off_q) / np.linalg.norm(w0))
                    line.append("%s keep %.3f noise %.4f" % (label, keep, noise))
                print("%-28s s=%.3f  %s" % (name, s, "  |  ".join(line)), flush=True)

            if out:
                out.seek(hdr["data_start"] + off)
                out.write(a)

    base.close()
    print("tensors: %d edited, %d untouched" % (edited, untouched), flush=True)
    if quant is not None:
        print("FP8 verification on s=0: %d byte-identical, %d different" % (exact, mismatch), flush=True)
    if out:
        out.flush()
        os.fsync(out.fileno())
        out.close()
        print("patched", args.write, flush=True)


if __name__ == "__main__":
    main()
