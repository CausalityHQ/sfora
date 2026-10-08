"""Stdlib-only reproducer for held 6841013; requires its preserved Git object."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

repo = Path(__file__).resolve().parents[5]
commit = '6841013ea0e20cde27cfa2ca8787b66ff1eadc9f'
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
    evaluator.source_live_guard.initializer = lambda context: lambda: None
    try:
        tests.initializer_constructor_contract(evaluator)
    except AssertionError as error:
        assert str(error) == 'accepted mutant: authenticated', str(error)
        print('REPRODUCED: mutable delegate skips initializer authentication')
    else:
        raise AssertionError('held mutant was not reproduced')
    finally:
        evaluator.source_live_guard.initializer = saved
assert not {'torch', 'numpy', 'PIL', 'transformers', 'sfora'} & sys.modules.keys()
