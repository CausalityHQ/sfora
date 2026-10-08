"""Stdlib-only reproduction against the immutable rejected candidate, no native work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

COMMIT = '57887f36463a739e60176f03147561f55713f996'
SHA256 = '9cf68f903f297ff7c1ae79b27e82658a799037291db4d9b6dfa2ec08dfa2188b'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


root = Path(__file__).resolve().parents[5]
raw = subprocess.check_output(['git', 'show', COMMIT + ':scripts/evaluate_siglip2_connected_mlp.py'], cwd=root)
assert hashlib.sha256(raw).hexdigest() == SHA256
with tempfile.TemporaryDirectory(prefix='sfora-namespace-falsifier-') as directory:
    path = Path(directory)
    (path / 'candidate.py').write_bytes(raw)
    candidate = load('_root_endpoint_guard_probe', path / 'candidate.py')
    source = path / 'genuine.py'
    source.write_text('def valid(value):\n    return len(value) == 1\n')
    module = load('_root_genuine_probe', source)
    assert module.valid([]) is False
    module.len = lambda value: 1
    guard = candidate.source_live_guard(module, hashlib.sha256(source.read_bytes()).hexdigest(), {})
    guard()
    assert module.valid([]) is True
    assert not {'torch', 'numpy', 'transformers', 'PIL'} & sys.modules.keys()
    print(json.dumps({'candidate': COMMIT, 'unexpected_global_accepted': True,
                     'invalid_value_accepted': True, 'native_qualification': False}))
