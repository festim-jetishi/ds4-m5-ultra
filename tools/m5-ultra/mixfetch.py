#!/usr/bin/env python3
"""Stream only selected routed-expert layers from Hugging Face MXFP4 shards.

Downloads only the routed-expert tensors of LAYERS (e.g. "30-39", ~67.2 GiB
instead of downloading all 500+ GiB of shards) via resumable HTTP Range
requests into PACK, verifying every finished tensor's cosine similarity
against the matching Q2 tensor with mixcheck.
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

REPO = "https://huggingface.co/antirez/deepseek-v4.1-flash-gguf/resolve/main/"
SHARD = "DeepSeek-V4.1-Flash-1017-MXFP4-experts-%05d-of-00009.gguf"
TBYTES = 2406481920
WORKERS = 4
CHUNK = 16 << 20


def layers_arg(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return sorted(set(out))


def make_request(url, rng, token):
    headers = {"Range": rng}
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.Request(url, headers=headers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("layers", help='Layer range, e.g. "30-39"')
    parser.add_argument("pack", help="Output temporary pack file path")
    parser.add_argument("--base", required=True, help="Path to pure Q2 base GGUF")
    parser.add_argument("--meta-dir", required=True, help="Directory with q2.json and mx1..9.json")
    parser.add_argument("--mixcheck", required=True, help="Path to compiled mixcheck binary")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    layers = layers_arg(args.layers)
    pack = args.pack
    token = os.environ.get("HF_TOKEN", "").strip()

    with open(os.path.join(args.meta_dir, "q2.json"), "r", encoding="utf-8") as f:
        q2 = json.load(f)
    q2t = {t[0]: t for t in q2["tensors"]}

    src = {}
    for i in range(1, 10):
        with open(os.path.join(args.meta_dir, "mx%d.json" % i), "r", encoding="utf-8") as f:
            d = json.load(f)
        for t in d["tensors"]:
            src[t[0]] = (i, d["data_start"] + t[3], t[1], t[2])

    idx = json.load(open(pack + ".json", "r", encoding="utf-8")) if os.path.exists(pack + ".json") else {}
    for il in layers:
        for p in ("gate", "up", "down"):
            name = "blk.%d.ffn_%s_exps.weight" % (il, p)
            shard, off, dims, ty = src[name]
            assert ty == 39 and dims == q2t[name][1], name
            if name not in idx:
                idx[name] = {
                    "shard": shard,
                    "src_off": off,
                    "pack_off": len(idx) * TBYTES,
                    "bytes": TBYTES,
                    "done": 0,
                    "cos": None,
                }

    todo = [n for n, e in idx.items() if e["cos"] is None]
    left = sum(e["bytes"] - e["done"] for e in idx.values() if e["cos"] is None)
    print("%d tensors in pack, %d to fetch, %.2f GiB to download" % (len(idx), len(todo), left / 2**30))
    if args.dry_run:
        for n in todo:
            e = idx[n]
            print("  %-30s shard %d bytes %d-%d" % (n, e["shard"], e["src_off"] + e["done"], e["src_off"] + e["bytes"] - 1))
        return

    fd = os.open(pack, os.O_RDWR | os.O_CREAT, 0o644)
    lock = threading.Lock()
    t0, got = time.time(), [0]

    def save():
        with open(pack + ".json.tmp", "w", encoding="utf-8") as fp:
            json.dump(idx, fp, indent=1)
        os.replace(pack + ".json.tmp", pack + ".json")

    def fetch(name):
        e = idx[name]
        for attempt in range(30):
            try:
                rng = "bytes=%d-%d" % (e["src_off"] + e["done"], e["src_off"] + e["bytes"] - 1)
                req = make_request(REPO + SHARD % e["shard"], rng, token)
                with urllib.request.urlopen(req, timeout=60) as r:
                    if r.status != 206:
                        raise RuntimeError("no range support (status %d)" % r.status)
                    while e["done"] < e["bytes"]:
                        buf = r.read(min(CHUNK, e["bytes"] - e["done"]))
                        if not buf:
                            raise RuntimeError("connection closed")
                        os.pwrite(fd, buf, e["pack_off"] + e["done"])
                        with lock:
                            e["done"] += len(buf)
                            got[0] += len(buf)
                            save()
                break
            except Exception as ex:
                print("  %s: %s, retry %d" % (name, ex, attempt + 1), flush=True)
                time.sleep(min(60, 2 ** attempt))
        else:
            sys.exit("giving up on " + name)

        os.fsync(fd)
        t = q2t[name]
        out = subprocess.check_output([
            args.mixcheck, args.base, str(t[2]), str(t[1][0]),
            str(t[1][1] * t[1][2]), str(q2["data_start"] + t[3]),
            pack, str(e["pack_off"]),
        ]).decode().strip()
        cos = float(out.split("cos mean ")[1].split()[0])
        with lock:
            e["cos"] = cos if cos > 0.8 else None
            if e["cos"] is None:
                e["done"] = 0
            save()
        elapsed = max(time.time() - t0, 1e-6)
        print("%s %s  [%.1f GiB, %.0f MiB/s]" % (
            name, out if cos > 0.8 else "CHECK FAILED: " + out,
            got[0] / 2**30, got[0] / 2**20 / elapsed,
        ), flush=True)

    queue = list(todo)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            fetch(name)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    os.close(fd)

    bad = [n for n, e in idx.items() if e["cos"] is None]
    print("done; failed checks:", bad or "none")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
