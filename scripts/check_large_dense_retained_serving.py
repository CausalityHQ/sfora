#!/usr/bin/env python3
"""Stdlib-only source/authority regression check; never imports model helpers."""
import ast
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import qualify_large_dense_retained_serving as serving


def rejects(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("changed authority accepted")


with TemporaryDirectory() as temporary:
    root = Path(temporary)
    old_export, old_score, current = (root / name for name in ("export", "score", "current"))
    for directory in (old_export, old_score, current):
        directory.mkdir()
    exported = {f"helper{i}.py": serving.sha(Path(serving.__file__)) for i in range(100)}
    exported["export_large_dense_pilot.py"] = next(iter(exported.values()))
    previous = {**exported, "score_large_dense_pilot.py": next(iter(exported.values()))}
    code = {**previous, serving.DRIVER: next(iter(exported.values()))}
    for directory, sources in ((old_export, exported), (old_score, previous), (current, code)):
        for name in sources:
            (directory / name).write_bytes(Path(serving.__file__).read_bytes())
    def manifest(directory, name, values):
        path = directory / name
        path.write_text(json.dumps(values))
        return serving.sha(path)
    export_sha = manifest(old_export, "dense-pilot-export-execution.json", exported)
    score_sha = manifest(old_score, "dense-pilot-score-execution.json", previous)
    manifest(current, "dense-pilot-export-execution.json", exported)
    manifest(current, "dense-pilot-score-execution.json", previous)
    execution = manifest(current, serving.MANIFEST, code)
    with patch.multiple(serving, EXPORT_ROOT=old_export, SCORE_ROOT=old_score,
                        EXPORT_SHA=export_sha, SCORE_SHA=score_sha):
        assert serving.source_authority(current, execution) == (code, previous, exported)
        # No authority read modifies the original manifest or original source.
        assert serving.sha(old_export / "dense-pilot-export-execution.json") == export_sha
        assert serving.sha(old_score / "dense-pilot-score-execution.json") == score_sha
        for path in (current / serving.DRIVER, old_score / "score_large_dense_pilot.py",
                     old_export / "export_large_dense_pilot.py", current / "helper0.py"):
            original = path.read_bytes()
            path.write_bytes(original + b"\n# changed\n")
            rejects(lambda: serving.source_authority(current, execution))
            rejects(lambda: serving.startup(current, execution))
            path.write_bytes(original)
        for invalid in (previous, {**code, "unexpected.py": "0" * 64},
                        {**code, "helper0.py": "0" * 64}):
            altered = manifest(current, serving.MANIFEST, invalid)
            rejects(lambda: serving.source_authority(current, altered))
        execution = manifest(current, serving.MANIFEST, code)
        assert serving.source_authority(current, execution) == (code, previous, exported)

# Verify deferred imports and the frozen constants without loading Torch.
tree = ast.parse(Path(serving.__file__).read_text())
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
        assert all(n.split(".")[0] in sys.stdlib_module_names for n in names)
assert "torch" not in sys.modules and "numpy" not in sys.modules
assert str(serving.CHECKPOINT) == "/home/riomus/runs/sfora-dense-pilot-179032-v1/native.pt"
assert serving.DECISION_SHA == "83781b7988874e1374dafbca8eb71c31f90dcb43a06d8cef585ee4218b1458a3"
assert serving.CHECKPOINT_SHA == "163b02268c44062dbdde2a1b07696c4d0365214ffbabfac76e575281957d362f"
print("PASS stdlib 101/102 original-root authority, 103 extension and changed-driver rejection; no Torch/model/native")
