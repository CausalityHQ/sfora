#!/usr/bin/env python3
"""Stdlib-only closure/tamper checks; no Torch, models, images or native calls."""
import ast
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

import export_large_dense_retained_official as export
import score_large_dense_retained_official as score


def rejects(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError('changed authority accepted')


with TemporaryDirectory() as temporary:
    base = Path(temporary)
    retained, exported, scored = (base / n for n in ('retained', 'export', 'score'))
    for root in (retained, exported, scored):
        root.mkdir()
    text = b'# fixture\n'
    fixture = base / 'fixture.py'
    fixture.write_bytes(text)
    digest = export.sha(fixture)
    old = {f'helper{i}.py': digest for i in range(102)}
    old['qualify_large_dense_retained_serving.py'] = digest
    current = {**old, export.DRIVER: digest}
    final = {**current, export.SCORE_DRIVER: digest}
    def manifest(root, name, code):
        path = root / name
        path.write_text(json.dumps(code))
        return export.sha(path)
    for root, code in ((retained, old), (exported, current), (scored, final)):
        for name in code:
            (root / name).write_bytes(text)
        old_sha = manifest(root, 'dense-retained-serving-execution.json', old)
        if root != retained:
            export_sha = manifest(root, export.MANIFEST, current)
    score_sha = manifest(scored, export.SCORE_MANIFEST, final)
    with patch.multiple(export, RETAINED_ROOT=retained, RETAINED_SHA=old_sha, SOURCE=exported):
        assert export.source_authority(exported, export_sha) == (current, current)
        assert export.source_authority(scored, score_sha, True) == (final, current)
        for path in (retained / 'helper0.py', exported / export.DRIVER, scored / export.SCORE_DRIVER, scored / 'helper1.py'):
            path.write_bytes(text + b'# tamper\n')
            rejects(lambda: export.source_authority(scored, score_sha, True))
            if path.parent != scored:
                rejects(lambda: export.source_authority(exported, export_sha))
            path.write_bytes(text)
        for invalid in (old, {**current, 'rogue.py': digest}, {**current, 'helper0.py': '0' * 64}):
            bad_sha = manifest(exported, export.MANIFEST, invalid)
            rejects(lambda: export.source_authority(exported, bad_sha))
        export_sha = manifest(exported, export.MANIFEST, current)
        for invalid in (current, {**final, 'rogue.py': digest}, {**final, 'helper0.py': '0' * 64}):
            bad_sha = manifest(scored, export.SCORE_MANIFEST, invalid)
            rejects(lambda: export.source_authority(scored, bad_sha, True))
        score_sha = manifest(scored, export.SCORE_MANIFEST, final)
        assert export.source_authority(scored, score_sha, True) == (final, current)
        module = ModuleType('late_fixture')
        module.__file__ = str(scored / 'rogue.py')
        Path(module.__file__).write_bytes(text)
        with patch.dict(sys.modules, late_fixture=module):
            rejects(lambda: export.loaded_authority(scored, final))
            export.loaded_authority(scored, {**final, 'rogue.py': digest})
            Path(module.__file__).write_bytes(text + b'# tamper\n')
            rejects(lambda: export.loaded_authority(scored, {**final, 'rogue.py': digest}))

for driver in (export, score):
    tree = ast.parse(Path(driver.__file__).read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(n.split('.')[0] in sys.stdlib_module_names or n == export.DRIVER[:-3] for n in names)
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
assert len(export.ORDER) == 4 and len(set(export.ORDER)) == 4
assert export.PROTOCOL_SHA == 'be7b27c68b5e2895063d4f24b2b9d4e396a77ed79b2426596893c293df2c20f2'
print('PASS unchanged103 ->104 ->105, tamper/extra-source/late-import rejection; stdlib only')
