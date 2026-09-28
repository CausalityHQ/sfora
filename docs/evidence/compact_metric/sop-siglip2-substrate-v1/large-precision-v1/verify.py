"""Dependency-free saved-vector replay: python3 verify.py."""
import ast
import hashlib
import json
import math
import struct
import zipfile
from pathlib import Path

p = Path(__file__).resolve().parent
r = json.loads((p / "large-precision-diagnostic.json").read_text())
assert hashlib.sha256((p / "large-precision-diagnostic.npz").read_bytes()).hexdigest() == r["features_sha256"]
xs = {}
with zipfile.ZipFile(p / "large-precision-diagnostic.npz") as z:
    for name in z.namelist():
        b = z.read(name)
        assert b[:8] == b"\x93NUMPY\x01\x00"
        n = int.from_bytes(b[8:10], "little")
        h = ast.literal_eval(b[10:10+n].decode())
        assert h["descr"] == "<f4" and h["shape"] == (4, 1024) and not h["fortran_order"]
        v = struct.unpack("<4096f", b[10+n:])
        xs[name[:-4]] = [v[i:i+1024] for i in range(0, 4096, 1024)]
for pair, actual in r["cosines"].items():
    a, b = pair.split("_vs_")
    out = [
        math.fsum(x*y for x, y in zip(u, v)) /
        (math.sqrt(math.fsum(x*x for x in u)) * math.sqrt(math.fsum(y*y for y in v)))
        for u, v in zip(xs[a], xs[b])
    ]
    assert max(abs(x-y) for x, y in zip(out, actual)) < 5e-7
    print(pair, "PASS min", min(out))
assert xs["bf16_eval"] == xs["bf16_train"]
assert min(r["cosines"]["fp32_eval_vs_fp16_eval"]) >= .999
assert min(r["cosines"]["fp32_eval_vs_bf16_eval"]) < .999
assert r["optimizer_updates"] == 0 and not r["quality_read"]
print("PASS scalar cosine replay, byte-identical BF16 train/eval, fixed decision")
