#!/usr/bin/env python3
"""Optional helper: download only attn.wo_b and ffn.shared_experts.w2 (FP8 + E8M0
scales) for layers 0-39 from mrfakename/DeepSeek-V4.1-Flash-1017-abliterated via
resumable HTTP Range requests for cross-checking against ablconv.py.
"""

import argparse
import json
import os
import struct
import sys
import threading
import time
import urllib.request

REPO = (
    "https://huggingface.co/mrfakename/DeepSeek-V4.1-Flash-1017-abliterated/resolve/"
    "517efab9ccfef7f714dbd7d679f181a86f43ccf9/"
)
WORKERS = 6
CHUNK = 8 << 20


def make_req(url, token, rng=None):
    headers = {}
    if rng:
        headers["Range"] = rng
    if token:
        headers["Authorization"] = "Bearer " + token
    return urllib.request.Request(url, headers=headers)


def get_range(url, start, end, token):
    r = urllib.request.urlopen(make_req(url, token, "bytes=%d-%d" % (start, end)), timeout=60)
    if r.status != 206:
        raise RuntimeError("no range support (status %d)" % r.status)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, help="Directory for index, headers, and tensors")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    here = args.work_dir
    token = os.environ.get("HF_TOKEN", "").strip()
    os.makedirs(os.path.join(here, "hdr"), exist_ok=True)
    os.makedirs(os.path.join(here, "t"), exist_ok=True)

    idx_path = os.path.join(here, "index.json")
    if not os.path.exists(idx_path):
        print("Downloading model.safetensors.index.json...", flush=True)
        with urllib.request.urlopen(make_req(REPO + "model.safetensors.index.json", token), timeout=60) as r:
            with open(idx_path, "wb") as f:
                f.write(r.read())

    def header(shard):
        path = os.path.join(here, "hdr", shard + ".json")
        if not os.path.exists(path):
            with get_range(REPO + shard, 0, 7, token) as r:
                (n,) = struct.unpack("<Q", r.read())
            with get_range(REPO + shard, 8, 8 + n - 1, token) as r:
                h = json.loads(r.read())
            h["__data_start__"] = 8 + n
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(h, fp)
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    with open(idx_path, "r", encoding="utf-8") as fp:
        wmap = json.load(fp)["weight_map"]

    jobs = []
    for il in range(40):
        for part in ("attn.wo_b", "ffn.shared_experts.w2"):
            for kind in ("weight", "scale"):
                name = "layers.%d.%s.%s" % (il, part, kind)
                shard = wmap[name]
                h = header(shard)
                t = h[name]
                b, e = t["data_offsets"]
                jobs.append((name, shard, h["__data_start__"] + b, e - b, t["dtype"], t["shape"]))

    meta = {j[0]: {"dtype": j[4], "shape": j[5], "bytes": j[3]} for j in jobs}
    with open(os.path.join(here, "tensors.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=1)

    todo = [
        j for j in jobs
        if not os.path.exists(os.path.join(here, "t", j[0]))
        or os.path.getsize(os.path.join(here, "t", j[0])) != j[3]
    ]
    total = sum(j[3] for j in todo)
    print("%d tensors, %d to fetch, %.2f GiB" % (len(jobs), len(todo), total / 2**30), flush=True)
    lock = threading.Lock()
    t0, got = time.time(), [0]

    def fetch(job):
        name, shard, off, nbytes = job[:4]
        path = os.path.join(here, "t", name)
        for attempt in range(20):
            try:
                done = os.path.getsize(path) if os.path.exists(path) else 0
                if done > nbytes:
                    os.remove(path)
                    done = 0
                if done < nbytes:
                    with get_range(REPO + shard, off + done, off + nbytes - 1, token) as r:
                        with open(path, "ab") as f:
                            while done < nbytes:
                                buf = r.read(min(CHUNK, nbytes - done))
                                if not buf:
                                    raise RuntimeError("connection closed")
                                f.write(buf)
                                done += len(buf)
                                with lock:
                                    got[0] += len(buf)
                break
            except Exception as ex:
                print("  %s: %s, retry %d" % (name, ex, attempt + 1), flush=True)
                time.sleep(min(60, 2 ** attempt))
        else:
            sys.exit("giving up on " + name)

    queue = list(todo)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
            fetch(job)

    def progress():
        while any(t.is_alive() for t in th):
            time.sleep(15)
            elapsed = max(time.time() - t0, 1e-6)
            print("  %.2f GiB, %.0f MiB/s" % (got[0] / 2**30, got[0] / 2**20 / elapsed), flush=True)

    th = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in th:
        t.start()
    threading.Thread(target=progress, daemon=True).start()
    for t in th:
        t.join()

    bad = [j[0] for j in jobs if os.path.getsize(os.path.join(here, "t", j[0])) != j[3]]
    print("done in %.0f s; incomplete: %s" % (time.time() - t0, bad or "none"), flush=True)
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
