#!/usr/bin/env python3
"""Rewrite a V4.1 Q2 GGUF with selected routed-expert layers replaced by MXFP4.

Preserves ds4's exact disk layout: resident tensors first in base order, the two
SSD-streamed Engram tables last. Reads real MXFP4 payloads from PACK (produced
by mixfetch.py); without PACK, synthesizes MXFP4 from Q2 via mixsyn for
loader/speed testing.
"""

import fcntl
import json
import os
import subprocess
import sys
import time

MXFP4 = 39
F_NOCACHE = 48
ENGRAM = ("blk.1.engram_embd.weight", "blk.14.engram_embd.weight")
BLOCK = {
    0: (1, 4),
    1: (1, 2),
    8: (32, 34),
    10: (256, 84),
    16: (256, 66),
    24: (1, 1),
    26: (1, 4),
    30: (1, 2),
    MXFP4: (32, 17),
}


def nbytes(dims, ty):
    n = 1
    for d in dims:
        n *= d
    be, bb = BLOCK[ty]
    return n // be * bb


def parse(path):
    with open(path, "rb") as f:
        def rd(fmt):
            import struct
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))

        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def skip(t):
            if t == 8:
                (n,) = rd("Q")
                f.seek(n, 1)
            elif t == 9:
                et, n = rd("IQ")
                if et in (8, 9):
                    for _ in range(n):
                        skip(et)
                else:
                    f.seek(sizes[et] * n, 1)
            else:
                f.seek(sizes[t], 1)

        assert f.read(4) == b"GGUF"
        ver, nt, nkv = rd("IQQ")
        align = 32
        for _ in range(nkv):
            (n,) = rd("Q")
            key = f.read(n).decode()
            (t,) = rd("I")
            if key == "general.alignment":
                (align,) = rd("I")
            else:
                skip(t)
        kv_end = f.tell()
        f.seek(24)
        kv_blob = f.read(kv_end - 24)
        tensors = []
        for _ in range(nt):
            (n,) = rd("Q")
            name = f.read(n).decode()
            (nd,) = rd("I")
            dims = list(rd("Q" * nd))
            ty, off = rd("IQ")
            tensors.append({"name": name, "dims": dims, "type": ty, "off": off})
        data_start = (f.tell() + align - 1) // align * align
        return ver, nkv, kv_blob, align, data_start, tensors


def layers_arg(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return set(out)


def main():
    import struct
    if len(sys.argv) < 4:
        sys.exit("usage: mixwrite.py BASE OUT LAYERS [PACK]")
    base, out, layers = sys.argv[1], sys.argv[2], layers_arg(sys.argv[3])
    ver, nkv, kv_blob, align, data_start, tensors = parse(base)
    for t in tensors:
        t["src"] = data_start + t["off"]
        t["bytes"] = nbytes(t["dims"], t["type"])
        parts = t["name"].split(".")
        t["mx"] = "_exps." in t["name"] and int(parts[1]) in layers
        if t["mx"]:
            t["new_type"] = MXFP4
            t["new_bytes"] = nbytes(t["dims"], MXFP4)
        else:
            t["new_type"], t["new_bytes"] = t["type"], t["bytes"]

    # Resident tensors in base order first, SSD-streamed Engram tables last
    order = sorted((t for t in tensors if t["name"] not in ENGRAM), key=lambda t: t["off"])
    order += [next(t for t in tensors if t["name"] == n) for n in ENGRAM]
    rel = 0
    for t in order:
        t["new_off"] = rel
        rel += (t["new_bytes"] + align - 1) // align * align

    head = bytearray(b"GGUF" + struct.pack("<IQQ", ver, len(tensors), nkv) + kv_blob)
    for t in tensors:
        name = t["name"].encode()
        head += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t["dims"]))
        head += struct.pack("<%dQ" % len(t["dims"]), *t["dims"])
        head += struct.pack("<IQ", t["new_type"], t["new_off"])
    new_start = (len(head) + align - 1) // align * align
    assert new_start == data_start, (new_start, data_start)
    head += b"\0" * (new_start - len(head))
    total = new_start + rel
    print("layers %s: %d MXFP4 tensors, output %.2f GiB (base %.2f GiB)" % (
        ",".join(map(str, sorted(layers))), sum(t["mx"] for t in tensors), total / 2**30,
        os.path.getsize(base) / 2**30,
    ))
    if os.path.exists(out):
        sys.exit("exists: " + out)

    src = os.open(base, os.O_RDONLY)
    pack = idx = None
    if len(sys.argv) > 4:
        pack = os.open(sys.argv[4], os.O_RDONLY)
        with open(sys.argv[4] + ".json", "r", encoding="utf-8") as fp:
            idx = json.load(fp)
        missing = [t["name"] for t in tensors if t["mx"] and (idx.get(t["name"]) or {}).get("cos") is None]
        if missing:
            sys.exit("not in pack (or unverified): " + ", ".join(missing[:6]))

    dst = os.open(out, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
    if sys.platform == "darwin":
        fcntl.fcntl(src, F_NOCACHE, 1)
        fcntl.fcntl(dst, F_NOCACHE, 1)
    os.ftruncate(dst, total)
    os.pwrite(dst, bytes(head), 0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    t0, done, chunk = time.time(), 0, 64 << 20
    for t in order:
        dpos = new_start + t["new_off"]
        if t["mx"] and pack is None:
            subprocess.check_call([
                os.path.join(script_dir, "mixsyn"), base, str(t["type"]),
                str(t["dims"][0]), str(t["dims"][1] * t["dims"][2]), str(t["src"]),
                out, str(dpos), str(os.cpu_count() or 8),
            ], stdout=subprocess.DEVNULL)
            done += t["new_bytes"]
            continue
        fd, left, spos = src, t["bytes"], t["src"]
        if t["mx"]:
            fd, left, spos = pack, t["new_bytes"], idx[t["name"]]["pack_off"]
        while left:
            buf = os.pread(fd, min(chunk, left), spos)
            assert buf, "short read"
            os.pwrite(dst, buf, dpos)
            spos += len(buf)
            dpos += len(buf)
            left -= len(buf)
            done += len(buf)
    os.fsync(dst)
    os.close(src)
    os.close(dst)
    if pack is not None:
        os.close(pack)
    print("wrote %.2f GiB in %.0f s" % (done / 2**30, time.time() - t0))


if __name__ == "__main__":
    main()
