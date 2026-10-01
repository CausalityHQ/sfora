#!/usr/bin/env python3
"""One stdlib falsifier. Native arithmetic/resource qualification is parent-owned.

Run: python3 -B -S scripts/test_siglip2_cached_readout.py
No Torch/NumPy/native imports, image decoding, SSH, GPU, scorer or quality reads.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def rejects(call, message):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError) as error:
        assert message in str(error), (message, str(error))
        return
    raise AssertionError('invalid contract accepted: ' + message)


class Tensor:
    """Only metadata; never masquerades as native numerical evidence."""
    def __init__(self, shape, dtype='torch.float32'):
        self.shape, self.dtype = shape, dtype


class Step(Tensor):
    def __init__(self, value):
        super().__init__(())
        self.value = value

    def __float__(self):
        return float(self.value)


def fixture(driver):
    groups = [{'lr': .0001, 'params': [0, 1, 2, 3]}, {'lr': .0001, 'params': [4]}]
    ident = {'seed': 179032, 'source': {'checkpoint': {'sha256': driver.SOURCE_SHA}},
             'parameter_names': driver.PARAMETERS.copy(), 'numerical_flags': {'deterministic': True},
             'optimizer_defaults': {'lr': .001}, 'optimizer_serial_groups': groups, 'positive_shape': (13283, 8)}
    saved = {k: None for k in driver.PAYLOAD_KEYS}
    saved.update(schema=driver.SCHEMA, identity=ident, counter=8, seed=ident['seed'], source=ident['source'],
                 numerical_flags=ident['numerical_flags'], optimizer_defaults=ident['optimizer_defaults'],
                 optimizer={'param_groups': copy.deepcopy(groups), 'state': {
                     i: {'step': Step(8), 'exp_avg': Tensor(shape), 'exp_avg_sq': Tensor(shape)}
                     for i, shape in enumerate(driver.SHAPES)}},
                 scaler={'scale': 128, 'growth_factor': 2., 'backoff_factor': .5, 'growth_interval': 2000, '_growth_tracker': 8},
                 cpu_rng=Tensor((5056,), 'torch.uint8'), cuda_rng=[Tensor((16,), 'torch.uint8')],
                 classifier=Tensor((2004, 128)), bank=Tensor((13283, 128)), target=Tensor((13283,), 'torch.int64'),
                 positive=Tensor((13283, 8), 'torch.int64'), pca={'mean': Tensor((1152,)), 'components': Tensor((128, 1152))},
                 schedules={str(s): Tensor((1000, 64), 'torch.int64') for s in driver.SEEDS},
                 head={n: Tensor(shape) for n, shape in zip(
                     ('primary.weight', 'primary.bias', 'down.weight', 'up.weight', 'center', 'preactivation_std'),
                     (*driver.SHAPES[:4], (1152,), ()), strict=True)})
    return saved, ident


def payload_checks(driver):
    saved, ident = fixture(driver)
    driver.check_payload(saved, ident, 8)
    initial = copy.deepcopy(saved)
    initial['counter'] = 0
    initial['optimizer']['state'] = {}
    initial['scaler']['_growth_tracker'] = 0
    driver.check_payload(initial, ident, 0)
    for key in driver.PAYLOAD_KEYS:
        mutant = copy.deepcopy(saved)
        del mutant[key]
        rejects(lambda: driver.check_payload(mutant, ident, 8), 'complete resume identity/state')
    for mutation, message in (
        (lambda m: m.update(vision={}), 'complete resume identity/state'),
        (lambda m: m.update(counter=9), 'complete resume identity/state'),
        (lambda m: m['optimizer']['state'].pop(4), 'EXACT five optimizer'),
        (lambda m: m['optimizer']['state'].update({5: m['optimizer']['state'][4]}), 'EXACT five optimizer'),
        (lambda m: m['optimizer']['param_groups'][1].update(params=[3]), 'EXACT five optimizer'),
        (lambda m: m['optimizer']['state'][0].update(step=Step(7)), 'optimizer counters/scaler/RNG'),
        (lambda m: m['optimizer']['state'][0].pop('exp_avg'), 'optimizer counters/scaler/RNG'),
        (lambda m: m['scaler'].update(_growth_tracker=7), 'optimizer counters/scaler/RNG'),
        (lambda m: m.update(cuda_rng=[]), 'optimizer counters/scaler/RNG'),
        (lambda m: m['scaler'].update(scale=float('inf')), 'optimizer counters/scaler/RNG'),
        (lambda m: m['head'].pop('center'), 'complete head/PCA/schedules'),
        (lambda m: m['schedules'].pop('179041'), 'complete head/PCA/schedules'),
        (lambda m: m['head'].update({'primary.weight': Tensor((128, 1024))}), 'complete tensor layout'),
        (lambda m: m.update(bank=Tensor((13282, 128))), 'complete tensor layout'),
        (lambda m: m.update(target=Tensor((13283,))), 'complete tensor layout'),
        (lambda m: m['schedules'].update({'179032': Tensor((100, 64), 'torch.int64')}), 'complete tensor layout'),
        (lambda m: m['optimizer']['state'][0].update(exp_avg=Tensor((128, 1024))), 'complete tensor layout'),
        (lambda m: m.update(cpu_rng=Tensor((0,), 'torch.uint8')), 'complete RNG layout'),
    ):
        mutant = copy.deepcopy(saved)
        mutation(mutant)
        rejects(lambda: driver.check_payload(mutant, ident, 8), message)
    wrong = copy.deepcopy(ident)
    wrong['parameter_names'] = wrong['parameter_names'][:-1]
    rejects(lambda: driver.check_payload(saved, wrong, 8), 'complete resume identity/state')


def launch_fixture(driver, phase='train', arm='candidate', seed=179032):
    return {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'a' * 64,
            'phase': phase, 'arm': arm, 'seed': seed, 'reference': {'root': '/frozen/original', 'execution_sha256': 'b' * 64},
            'fit_inventory': {'path': '/frozen/fit.json', 'sha256': driver.FIT_INVENTORY_SHA},
            'initialized_inventory': {'path': '/frozen/initialized.json', 'sha256': driver.INITIALIZED_INVENTORY_SHA},
            'selected_cpu': None if phase == 'cpu' else {'receipt': 'placeholder'},
            'selected_mechanics': {a: {'receipt': 'placeholder'} for a in driver.ARMS} if phase == 'train' else None,
            'recipe': copy.deepcopy(driver.RECIPE), 'resource_policy': driver.policy(phase), 'both_locks_held': True}


def authority_checks(driver):
    for phase, arm, seed in (('cpu', 'control', 179032), ('mechanics', 'candidate', 179032),
                             ('train', 'control', 179032), ('train', 'candidate', 179041)):
        launch = launch_fixture(driver, phase, arm, seed)
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256='a' * 64)
        driver.check_launch(launch, args)
        for mutation, message in (
            (lambda m: m.update(extra=True), 'launch profile'),
            (lambda m: m.update(both_locks_held=False), 'launch profile'),
            (lambda m: m['recipe'].update(steps=100), 'launch profile'),
            (lambda m: m['recipe'].update(augmentation=True), 'launch profile'),
            (lambda m: m['recipe'].update(width=1024), 'launch profile'),
            (lambda m: m['resource_policy'].update(seconds=301), 'launch profile'),
            (lambda m: m['fit_inventory'].update(sha256='0' * 64), 'original collected inventory pins'),
            (lambda m: m.update(selected_cpu={} if phase == 'cpu' else None), 'phase prerequisites'),
        ):
            mutant = copy.deepcopy(launch)
            mutation(mutant)
            rejects(lambda: driver.check_launch(mutant, args), message)
    launch = launch_fixture(driver)
    launch['selected_mechanics'].pop('control')
    rejects(lambda: driver.check_launch(launch, SimpleNamespace(phase='train', arm='candidate', seed=179032,
                                                             execution_sha256='a' * 64)), 'BOTH mechanics')
    source = {'proof': {'checkpoint': {'sha256': driver.SOURCE_SHA}, 'runtime': {'vision': dict.fromkeys(range(448))},
                        'sample': {}}, 'entry': {'revision': driver.REVISION}, 'args': SimpleNamespace(fit_manifest_sha256=driver.MANIFEST_SHA)}
    initialized = {'source_context': source, 'pca': {'export': {'cache': {
        'sha256': driver.FIT_SHA, 'shape': [13283, 1152], 'dtype': 'float32'}}},
        'record': {'artifact': {'sha256': driver.PCA_SHA}, 'arrays': {}}}
    driver.source_binding(initialized)
    for mutation in (
        lambda m: m['source_context']['entry'].update(revision='0' * 40),
        lambda m: m['source_context']['proof']['checkpoint'].update(sha256='0' * 64),
        lambda m: m['record']['artifact'].update(sha256='0' * 64),
        lambda m: m['pca']['export']['cache'].update(shape=[13283, 1024]),
        lambda m: m['pca']['export']['cache'].update(shape=[13282, 1152]),
        lambda m: m['pca']['export']['cache'].update(dtype='float16'),
    ):
        mutant = copy.deepcopy(initialized)
        mutation(mutant)
        rejects(lambda: driver.source_binding(mutant), 'authenticated So400')


def terminal_checks(driver):
    launch = launch_fixture(driver)
    record = {'schema': driver.SCHEMA, 'phase': 'mechanics', 'arm': 'candidate', 'seed': 179032,
              'pass': True, 'quality_read': False, 'exit_rehash_pass': True, 'strict_reload_exact': True,
              'optimizer_members': 5, 'source': {'checkpoint': {'sha256': driver.SOURCE_SHA}},
              'launch': launch_fixture(driver, 'mechanics'), 'resource_policy': driver.policy('mechanics'),
              'completed_step': 17, 'checkpoint': None, 'training_state_discarded': True, 'replay_exact': True,
              'peak_cuda_allocated_bytes': 1000000}
    record['steps'] = [{'step': i, 'batch': list(range(64)), 'ce': 1., 'rank': .1, 'loss': 1.8,
                        'scale': 128., 'preclip_norm': 1., 'seconds': .01,
                        'schedule_sha256': 'a' * 64, 'feature_rows_sha256': 'b' * 64, 'state_sha256': 'c' * 64,
                        'gradient_norms': dict.fromkeys(driver.PARAMETERS, 0.)} for i in range(1, 18)]
    record['resumed_steps'] = copy.deepcopy(record['steps'][8:])
    driver.check_terminal_record(record, launch, 'mechanics', 'candidate')
    for mutation, message in (
        (lambda m: m.update(optimizer_members=208), 'new method terminal binding'),
        (lambda m: m.update(completed_step=16), 'discarded17'),
        (lambda m: m.update(replay_exact=False), 'discarded17'),
        (lambda m: m.update(checkpoint={'path': 'partial'}), 'discarded17'),
        (lambda m: m['resumed_steps'][0].update(loss=99), 'discarded17'),
        (lambda m: m.update(peak_cuda_allocated_bytes=10_000_000_000), 'discarded17'),
        (lambda m: [r.pop('feature_rows_sha256') for r in m['steps'] + m['resumed_steps']], 'complete finite update'),
    ):
        mutant = copy.deepcopy(record)
        mutation(mutant)
        rejects(lambda: driver.check_terminal_record(mutant, launch, 'mechanics', 'candidate'), message)


def file_checks(driver, root):
    path = root / 'dependency'
    path.write_bytes(b'original')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    guards = {}
    driver.bound_file(guards, path, digest)
    path.write_bytes(b'changed')
    rejects(lambda: driver.bound_file(guards, path, digest), 'file SHA256 differs')
    link = root / 'link'
    link.symlink_to(path)
    rejects(lambda: driver.bound_file({}, link, digest), 'canonical file')
    rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate JSON')
    rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite JSON')
    for name in driver.FILES:
        (root / name).write_text('# frozen\n')
    code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in driver.FILES}
    manifest = root / 'execution.json'
    manifest.write_text(json.dumps(code))
    sha = lambda: hashlib.sha256(manifest.read_bytes()).hexdigest()
    driver.closure(root, sha(), driver.FILES, {})
    for keys in (dict(code, extra='0' * 64), {next(iter(code)): next(iter(code.values()))}):
        manifest.write_text(json.dumps(keys))
        rejects(lambda: driver.closure(root, sha(), driver.FILES, {}), 'exact execution closure')


def source_checks(driver, path):
    tree = ast.parse(path.read_bytes())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    original = path.parent / 'train_siglip2_substrate_adaptation.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == driver.TRAINER_SHA
    # No serializer rewrite, source optimizer fake or image augmentation path.
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert 'fingerprint' in names and 'valid_rank' in names and 'exit_rehash' in names
    assert 'reset_peak_memory_stats' not in names and 'augmented_pixels' not in names
    assert 'fresh' not in {n.attr for n in ast.walk(functions['gpu_run']) if isinstance(n, ast.Attribute)}
    assert driver.PARAMETERS == ['compact_head.primary.weight', 'compact_head.primary.bias',
                                  'compact_head.down.weight', 'compact_head.up.weight', 'classifier']
    assert sum(math.prod(s) for s in driver.SHAPES[:4]) == 188544
    assert sum(math.prod(s) for s in driver.SHAPES) == 445056
    # The original prefix draws keep precisely the same RNG calls and inputs.
    old_tree = ast.parse((path.parent / 'qualify_siglip2_initialized_cpu.py').read_bytes())
    old_schedule = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule')
    old_prefix = ast.Module(body=old_schedule.body[1:9], type_ignores=[])
    new_prefix = ast.Module(body=functions['schedule'].body[1:9], type_ignores=[])
    assert ast.dump(old_prefix, include_attributes=False) == ast.dump(new_prefix, include_attributes=False)
    old_loop = old_schedule.body[10]
    new_outer = functions['schedule'].body[9].value.args[0]
    assert isinstance(old_loop, ast.For) and isinstance(new_outer, ast.ListComp)
    assert ast.unparse(new_outer.generators[0].iter) == 'range(1000)'
    new_inner = copy.deepcopy(new_outer.elt)
    assert ast.dump(new_inner.generators[0].iter, include_attributes=False) == ast.dump(old_loop.body[0].value, include_attributes=False)
    new_inner.generators[0].iter = ast.Name(id='classes', ctx=ast.Load())
    assert ast.dump(new_inner, include_attributes=False) == ast.dump(old_loop.body[1].value.args[0], include_attributes=False)
    head_code = ast.unparse(functions['head_from'])
    assert "approximate='none'" in head_code and 'manual_seed(179034)' in head_code
    assert 'std(unbiased=False)' in head_code and 'weight.zero_()' in head_code


def main():
    path = Path(__file__).absolute().with_name('train_siglip2_cached_readout.py')
    native = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
    old_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in native:
            raise AssertionError('native import attempted by stdlib falsifier: ' + name)
        return old_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    try:
        spec = importlib.util.spec_from_file_location('cached_readout', path)
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        payload_checks(driver)
        authority_checks(driver)
        terminal_checks(driver)
        source_checks(driver, path)
        with TemporaryDirectory() as directory:
            file_checks(driver, Path(directory))
        assert not native.intersection(n.split('.')[0] for n in sys.modules)
    finally:
        builtins.__import__ = old_import
    for flags in ([], ['-O'], ['-OO']):
        result = subprocess.run([sys.executable, '-B', '-S', *flags, str(path), '--help'], capture_output=True, text=True)
        assert result.returncode == (1 if flags else 0), result.stderr
        assert ('optimized mode is forbidden' in result.stderr) if flags else ('--phase {cpu,mechanics,train}' in result.stdout)
    print('PASS stdlib authority/dependency/five-state/incomplete-resume/mechanics/source contract negatives; help/-O/-OO')


if __name__ == '__main__':
    main()
