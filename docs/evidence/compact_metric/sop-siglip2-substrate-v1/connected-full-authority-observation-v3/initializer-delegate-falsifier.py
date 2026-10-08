"""Stdlib-only reproducers for two held repairs; require their preserved Git objects."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[5]
cases = (('6841013ea0e20cde27cfa2ca8787b66ff1eadc9f', False),
         ('15eb6e0c6ff450d95e1d4c07b01bb360f8d42dd7', True))
for commit, mask_dictionary in cases:
 with tempfile.TemporaryDirectory() as directory:
    root = Path(directory) / 'scripts'; root.mkdir()
    for path in (repo / 'scripts').glob('*.py'):
        shutil.copyfile(path, root / path.name)
    for name in ('evaluate_siglip2_connected_mlp.py', 'test_connected_mlp_evaluation.py'):
        (root / name).write_bytes(subprocess.check_output(['git', '-C', str(repo), 'show', commit + ':scripts/' + name]))
    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, root / filename)
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module); return module
    evaluator = load('_delegate_held', 'evaluate_siglip2_connected_mlp.py')
    tests = load('_delegate_held_tests', 'test_connected_mlp_evaluation.py')
    saved = evaluator.source_live_guard.initializer
    saved_dictionary = evaluator.source_live_guard.__dict__
    if mask_dictionary:
        class Mask(dict):
            def get(self, key, default=None):
                return saved if key == 'initializer' else super().get(key, default)
        evaluator.source_live_guard.__dict__ = Mask(saved_dictionary)
    evaluator.source_live_guard.initializer = lambda context: lambda: None
    try:
        tests.initializer_constructor_contract(evaluator)
    except AssertionError as error:
        assert str(error) == 'accepted mutant: authenticated', str(error)
        print('REPRODUCED:', commit, 'masked dictionary' if mask_dictionary else 'mutable delegate')
    else:
        raise AssertionError('held mutant was not reproduced')
    finally:
        evaluator.source_live_guard.__dict__ = saved_dictionary
        evaluator.source_live_guard.initializer = saved
assert not {'torch', 'numpy', 'PIL', 'transformers', 'sfora'} & sys.modules.keys()
