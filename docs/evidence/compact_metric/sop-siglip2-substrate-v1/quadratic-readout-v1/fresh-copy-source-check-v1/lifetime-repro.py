"""Root source-only check: returned hash calls must release gathered tensor views."""
import gc
import importlib.util
import json
from pathlib import Path
import types

root = Path(__file__).resolve().parents[6]
path = root / 'scripts/test_siglip2_quadratic_readout.py'
spec = importlib.util.spec_from_file_location('lifetime_check', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ContractTests.setUpClass()
gc.collect()
enabled = gc.isenabled()
gc.disable()
try:
    test = module.ContractTests('test_fingerprint_batches_fresh_cuda_bytes')
    test.test_fingerprint_batches_fresh_cuda_bytes()
    test.doCleanups()
    rows = []
    for fn in gc.get_objects():
        if (isinstance(fn, types.FunctionType) and fn.__code__.co_name == 'gather'
                and fn.__globals__.get('__name__') == '_fresh_quadratic_encoder_frames'):
            cells = dict(zip(fn.__code__.co_freevars, fn.__closure__ or ()))
            parts = cells['parts'].cell_contents
            if parts:
                rows.append(sum(len(value) for value in parts.values()))
    print(json.dumps({'gc_disabled': True, 'retaining_calls': len(rows),
                      'retained_tensor_views': sum(rows), 'native_imported': False}))
    assert not rows, 'returned fingerprint calls retain gathered tensor views'
finally:
    if enabled:
        gc.enable()
    gc.collect()
