"""Execute the genuine probe consumer against an additive source-only API."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

root = Path('/home/rb/worktrees/sfora-positive-causality')
owned = []
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    owned.append((name, module))
    spec.loader.exec_module(module)
    return module

try:
    probe = load('_reader_probe_consumer_check', root/'scripts/test_connected_probe_inference_extraction.py')
    oracle = load('_reader_mlp_oracle_check', root/'scripts/test_connected_inference_extraction.py')
    guard = oracle.NoNative()
    sys.meta_path.insert(0, guard)
    try:
        raw = oracle.RUNTIME.read_text()
        source = probe.RUNTIME.read_text()
        probe.correspondence(oracle, source)
        additive = raw + '\n\ndef load_serving_inference():\n    pass\n'
        with patch.object(oracle, 'RUNTIME', SimpleNamespace(read_text=lambda: additive)):
            try:
                probe.correspondence(oracle, source)
            except KeyError as error:
                assert error.args == ('load_serving_inference',)
            else:
                raise AssertionError('new public definition was silently ignored')
        print(json.dumps({'baseline_consumer_pass':True,'additive_api_rejected':'KeyError(load_serving_inference)','native_executed':False,'scope':'Actual correspondence oracle; appended inert definition, no production edit'},indent=2))
    finally:
        sys.meta_path.remove(guard)
finally:
    for name,module in owned:
        if sys.modules.get(name) is module:
            del sys.modules[name]
