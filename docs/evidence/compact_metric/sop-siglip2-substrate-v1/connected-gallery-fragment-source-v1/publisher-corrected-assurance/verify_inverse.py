"""Prove: repaired files = exact baseline + a finite, literal, invertible edit list."""
import ast, hashlib, subprocess, sys
from pathlib import Path

BASE = "fb0e1661416e77cc320e497da7bb8f4721f00237"
def git_show(path): return subprocess.run(["git", "show", f"{BASE}:{path}"], capture_output=True, check=True).stdout.decode()
sha = lambda text: hashlib.sha256(text.encode()).hexdigest()

# ---------------- production ----------------
PROD = "src/sfora/atomic_publication.py"
base, cur = git_show(PROD), Path(PROD).read_text()
assert sha(base) == "ee0870e4e34a3bb08e82b6aa6d222a1f29dce7f17edfb331973d34dadac8ed30"
bt, ct = ast.parse(base), ast.parse(cur)
TWO = {"publish_writer_noreplace", "publish_large_writer_noreplace"}

def keyed(tree):
    return [(type(n).__name__, getattr(n, "name", None) or ast.dump(n), n) for n in tree.body]

b_nodes, c_nodes = keyed(bt), keyed(ct)
b_keys = {(k, name) for k, name, _ in b_nodes}
added = [n for n in c_nodes if (n[0], n[1]) not in b_keys]
assert [ast.unparse(n[2]) for n in added] == ["import sys"], [ast.unparse(n[2])[:40] for n in added]
c_rest = [n for n in c_nodes if not (n[0] == "Import" and ast.unparse(n[2]) == "import sys")]
assert [(k, name) for k, name, _ in c_rest] == [(k, name) for k, name, _ in b_nodes], "top-level names/order changed"
changed = []
for (k, name, bn), (_, _, cn) in zip(b_nodes, c_rest):
    same_ast = ast.dump(bn) == ast.dump(cn)
    same_src = ast.get_source_segment(base, bn) == ast.get_source_segment(cur, cn)
    if name in TWO:
        assert not same_ast; changed.append(name)
    else:
        assert same_ast and same_src, f"unrelated node differs: {name}"
print(f"production: {len(b_nodes)} top-level nodes, {len(b_nodes) - 2} byte+AST identical, changed exactly: {changed}; added only: import sys")

def fn(tree, name): return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
for name in sorted(TWO):
    bf, cf = fn(bt, name), fn(ct, name)
    assert ast.dump(bf.args) == ast.dump(cf.args) and ast.dump(bf.returns) == ast.dump(cf.returns)
    assert [ast.dump(x) for x in bf.body[:-1]] == [ast.dump(x) for x in cf.body[:-4]], "pre-try statements differ"
    btry, ctry = bf.body[-1], next(n for n in cf.body if isinstance(n, ast.Try))
    assert [ast.dump(x) for x in btry.body[:-2]] == [ast.dump(x) for x in ctry.body[:-2]], "try body (publication protocol) differs"
    print(f"  {name}: signature, setup, and the {len(btry.body) - 2} protocol statements before the return are AST-identical")

# literal inverse: 1 import edit + per function (return-shape edit, finally edit, tail edit)
def inverse_prod(text):
    out = text
    def rep(old, new, count):
        nonlocal out
        assert out.count(old) == count, (out.count(old), old[:60]); out = out.replace(old, new)
    rep("import os\nimport sys\nfrom", "import os\nfrom", 1)
    for cls, fields in (("PublishedFile", "            payload=reopened_payload,\n            identity=owned,\n            size=len(reopened_payload),\n"),
                        ("PublishedLargeFile", "            identity=owned,\n            size=info.st_size,\n")):
        rep(f"        result = {cls}(\n{fields}            descriptor=-1,\n        )\n        completed = True\n",
            f"        completed = True\n        return {cls}(\n{fields}            descriptor=retained_descriptor,\n        )\n", 1)
    for var in ("info", "final"):
        pass
    start = "    finally:\n        failures: list[BaseException] = []\n"
    tail = "    assert retained_descriptor is not None\n    result.descriptor = retained_descriptor\n    return result\n"
    for var in ("info", "final"):
        s0 = out.index(start); s1 = out.index(tail, s0) + len(tail)
        block = out[s0:s1]
        assert f"{var} = path.lstat()" in block
        original = (f"    finally:\n        if published and not completed and owned is not None and os.path.lexists(path):\n"
                    f"            {var} = path.lstat()\n            if ({var}.st_dev, {var}.st_ino) == owned:\n"
                    f"                path.unlink()\n                os.fsync(directory)\n"
                    f"        if descriptor is not None:\n            os.close(descriptor)\n"
                    f"        if not completed and retained_descriptor is not None:\n            os.close(retained_descriptor)\n"
                    f"        os.close(directory)\n")
        out = out[:s0] + original + out[s1:]
    return out
restored = inverse_prod(cur)
assert restored == base and sha(restored) == "ee0870e4e34a3bb08e82b6aa6d222a1f29dce7f17edfb331973d34dadac8ed30"
print("production inverse (1 import edit + 2x[return-shape, finally, tail]) restores sha ee0870e4... exactly")
print("production sha256 now:", sha(cur))

# ---------------- tests ----------------
TST = "tests/test_atomic_publication.py"
tb, tc = git_show(TST), Path(TST).read_text()
btt, ctt = ast.parse(tb), ast.parse(tc)
b_fns = [n for n in btt.body if isinstance(n, ast.FunctionDef)]
c_fns = {n.name: n for n in ctt.body if isinstance(n, ast.FunctionDef)}
for n in b_fns:
    assert ast.dump(n) == ast.dump(c_fns[n.name]) and ast.get_source_segment(tb, n) == ast.get_source_segment(tc, c_fns[n.name]), n.name
print(f"tests: all {len(b_fns)} existing test bodies byte+AST identical; new top-level defs: {len(c_fns) - len(b_fns)} functions/fixtures + helpers")
marker = "\n\n# --- cleanup hygiene"
assert tc.count(marker) == 1
head = tc.split(marker)[0]
head = head.replace("import os\nimport traceback\nfrom collections.abc import Callable\nfrom pathlib import Path, PosixPath\nfrom typing import Any\n", "import os\nfrom pathlib import Path\n")
assert head == tb and sha(head) == sha(tb)
print("tests inverse (drop appended block + 4 import lines -> 'from pathlib import Path') restores sha", sha(tb)[:16] + "... exactly")
print("tests sha256 now:", sha(tc))
