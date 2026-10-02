#!/usr/bin/env python3
"""Focused stdlib falsifier; native CPU/GPU/math/resource qualification is UNRUN.

python3 -B -S scripts/test_siglip2_identity_mix.py
No Torch/NumPy/native imports, feature reads, SSH, images or quality evaluation.
"""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def rejects(call, message=None):
    try:
        call()
    except (ValueError, KeyError, TypeError, OSError) as error:
        if message:
            assert message in str(error), (message, str(error))
        return
    raise AssertionError('invalid contract accepted: ' + str(message))


class Array:
    """Tiny list arithmetic to exercise the SAME dependency-free mixing path.

    This is never a native numerical/precision/qualification witness.
    """
    def __init__(self, rows):
        self.rows = rows

    def clone(self):
        return Array(copy.deepcopy(self.rows))

    def __getitem__(self, index):
        indexes = index.rows if isinstance(index, Array) else index
        if indexes and type(indexes[0]) is bool:
            indexes = [i for i, yes in enumerate(indexes) if yes]
        return Array([self.rows[i] for i in indexes])

    def __setitem__(self, mask, values):
        for i, row in zip([i for i, yes in enumerate(mask.rows) if yes], values.rows, strict=True):
            self.rows[i] = row

    def any(self):
        return any(self.rows)

    def unsqueeze(self, dimension):
        assert dimension == 1
        return Array([[v] for v in self.rows])

    def __rsub__(self, scalar):
        return Array([[scalar - v for v in row] for row in self.rows])

    def __mul__(self, other):
        return Array([[a * b for a, b in zip(left * len(right) if len(left) == 1 else left, right, strict=True)]
                      for left, right in zip(self.rows, other.rows, strict=True)])

    def __add__(self, other):
        return Array([[a + b for a, b in zip(left, right, strict=True)]
                      for left, right in zip(self.rows, other.rows, strict=True)])


def mixing_checks(driver):
    target = [0, 1, 0, 2, 1, 1]
    members = driver.donor_members(target)
    for anchor, label in enumerate(target):
        if len(members[label]) == 1:
            assert driver.donor_at(members, target, anchor, 0) == anchor
        else:
            support = [driver.donor_at(members, target, anchor, i) for i in range(len(members[label]) - 1)]
            assert sorted(support) == [r for r in members[label] if r != anchor]
            rejects(lambda: driver.donor_at(members, target, anchor, len(members[label]) - 1), 'outside support')
    anchors, donors, coefficients, masks = [0, 3, 4], [2, 3, 1], [.25, 0., .75], [True, False, True]
    driver.check_mixing(target, anchors, donors, coefficients, masks)
    for a, d, c, m in ((anchors, [1, 3, 1], coefficients, masks),
                       (anchors, [0, 3, 1], coefficients, masks),
                       (anchors, donors, [.25, .5, .75], masks),
                       (anchors, donors, coefficients, [True, True, True]),
                       (anchors, donors, [float('nan'), 0., .75], masks),
                       (anchors, donors, [.1, 0., .75], masks),
                       (anchors, donors, coefficients, [1, False, True])):
        rejects(lambda: driver.check_mixing(target, a, d, c, m), 'product-safe FP32')
    features = Array([[1., 0.], [0., 1.], [0., 1.], [-1., 0.]])
    anchor, donor, lam, mask = Array([0, 1, 2]), Array([2, 0, 0]), Array([.5, .5, .5]), Array([True, False, True])
    def normalize(rows):
        assert all(all(math.isfinite(v) for v in row) and math.hypot(*row) > 0 for row in rows.rows)
        return Array([[v / math.hypot(*row) for v in row] for row in rows.rows])
    control = driver.mix_inputs(features, anchor, donor, lam, mask, 'control', normalize)
    candidate = driver.mix_inputs(features, anchor, donor, lam, mask, 'candidate', normalize)
    assert control.rows == [features.rows[i] for i in anchor.rows]
    assert candidate.rows[1] == features.rows[1]  # untreated identity
    assert candidate.rows[0] == candidate.rows[2]
    assert all(abs(math.hypot(*row) - 1) < 1e-15 for row in candidate.rows)
    assert features.rows == [[1., 0.], [0., 1.], [0., 1.], [-1., 0.]]
    try:
        driver.mix_inputs(features, Array([0]), Array([3]), Array([.5]), Array([True]), 'candidate', normalize)
    except AssertionError:
        pass
    else:
        raise AssertionError('zero mixed row accepted')


class Tensor:
    """Metadata ONLY; no native numerical claim."""
    def __init__(self, shape, dtype='torch.float32', value=None):
        self.shape, self.dtype, self.value = shape, dtype, value

    def __float__(self):
        return float(self.value)


def resume_fixture(driver):
    groups = [{'lr': .0001, 'params': [0, 1, 2, 3]}, {'lr': .0001, 'params': [4]}]
    ident = {'seed': driver.SEEDS[0], 'source': {}, 'parameter_names': driver.PARAMETERS.copy(),
             'numerical_flags': {}, 'optimizer_defaults': {}, 'optimizer_serial_groups': groups,
             'positive_shape': (driver.ROWS, 8), 'device': 'cuda'}
    initial = {'head': {n: Tensor(shape) for n, shape in zip(
        ('primary.weight', 'primary.bias', 'down.weight', 'up.weight', 'center', 'preactivation_std'),
        (*driver.SHAPES[:4], (driver.WIDTH,), ()), strict=True)},
        'classifier': Tensor(driver.SHAPES[4]), 'bank': Tensor((driver.ROWS, driver.DIM)),
        'target': Tensor((driver.ROWS,), 'torch.int64'), 'positive': Tensor(ident['positive_shape'], 'torch.int64'),
        'original_rows': Tensor((driver.ROWS,), 'torch.int64'),
        'pca': {'mean': Tensor((driver.WIDTH,)), 'components': Tensor((driver.DIM, driver.WIDTH))},
        'schedules': {str(s): Tensor((1000, 64), 'torch.int64') for s in driver.SEEDS},
        'mixing': {str(s): {'donors': Tensor((1000, 64), 'torch.int64'),
                          'lambda': Tensor((1000, 64)), 'mask': Tensor((1000, 64), 'torch.bool')} for s in driver.SEEDS}}
    saved = {'schema': driver.SCHEMA, 'identity': ident, **copy.deepcopy(initial), 'initializer': initial,
             'partition': {}, 'counter': 8, 'seed': ident['seed'], 'source': {}, 'numerical_flags': {},
             'optimizer_defaults': {}, 'optimizer': {'param_groups': copy.deepcopy(groups), 'state': {
                 i: {'step': Tensor((), value=8), 'exp_avg': Tensor(shape), 'exp_avg_sq': Tensor(shape)}
                 for i, shape in enumerate(driver.SHAPES)}},
             'scaler': {'scale': 128, 'growth_factor': 2., 'backoff_factor': .5, 'growth_interval': 2000, '_growth_tracker': 8},
             'cpu_rng': Tensor((5056,), 'torch.uint8'), 'cuda_rng': [Tensor((16,), 'torch.uint8')]}
    return saved, ident


def resume_checks(driver):
    saved, ident = resume_fixture(driver)
    driver.check_payload(saved, ident, 8)
    for key in driver.PAYLOAD_KEYS:
        mutant = copy.deepcopy(saved); del mutant[key]
        rejects(lambda: driver.check_payload(mutant, ident, 8), 'complete resume identity/state')
    for mutation in (
        lambda m: m.update(schema='siglip2-cached-readout-v1'),
        lambda m: m['mixing'][str(driver.SEEDS[0])].pop('mask'),
        lambda m: m['mixing'][str(driver.SEEDS[0])].update(donors=Tensor((1000, 64))),
        lambda m: m['initializer'].pop('bank'),
        lambda m: m['initializer']['head'].pop('center'),
        lambda m: m.update(classifier=Tensor((2004, 128))),
        lambda m: m.update(original_rows=Tensor((13283,), 'torch.int64')),
        lambda m: m['optimizer']['state'].pop(4),
        lambda m: m['optimizer']['state'][0].update(step=Tensor((), value=7)),
        lambda m: m['scaler'].update(_growth_tracker=7),
        lambda m: m.update(cuda_rng=[]),
    ):
        mutant = copy.deepcopy(saved); mutation(mutant)
        rejects(lambda: driver.check_payload(mutant, ident, 8))
    cpu = copy.deepcopy(saved)
    cpu_ident = copy.deepcopy(ident); cpu_ident['device'] = 'cpu'
    cpu.update(identity=cpu_ident, cuda_rng=[])
    driver.check_payload(cpu, cpu_ident, 8)
    initial = copy.deepcopy(cpu)
    initial.update(counter=0); initial['optimizer']['state'] = {}; initial['scaler']['_growth_tracker'] = 0
    driver.check_payload(initial, cpu_ident, 0)


def launch_fixture(driver, phase='train', arm='candidate', seed=179061):
    return {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': 'a' * 64,
            'phase': phase, 'arm': arm, 'seed': seed,
            'cached_reference': {'root': '/frozen/original', 'execution_sha256': driver.CACHED_EXECUTION_SHA},
            'original_authority': {'path': '/frozen/cpu.json', 'sha256': driver.ORIGINAL_AUTHORITY_SHA},
            'partition': {'path': '/frozen/partition.json', 'sha256': driver.PARTITION_SHA},
            'selected_cpu': None if phase == 'cpu' else {'receipt': 'placeholder'},
            'selected_mechanics': {a: {'receipt': 'placeholder'} for a in driver.ARMS} if phase == 'train' else None,
            'recipe': copy.deepcopy(driver.RECIPE), 'resource_policy': driver.policy(phase), 'both_locks_held': True}


def authority_checks(driver):
    for phase, arm, seed in (('cpu', 'control', 179061), ('mechanics', 'candidate', 179061),
                             ('train', 'control', 179061), ('train', 'candidate', 179069)):
        launch = launch_fixture(driver, phase, arm, seed)
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256='a' * 64)
        driver.check_launch(launch, args)
        for mutation in (
            lambda m: m.update(extra=True), lambda m: m.update(both_locks_held=False),
            lambda m: m['recipe'].update(candidate='gelu-exact'),
            lambda m: m['recipe'].update(rows=13283), lambda m: m['recipe'].update(initialization_seed=179034),
            lambda m: m['resource_policy'].update(seconds=301),
            lambda m: m['cached_reference'].update(execution_sha256='0' * 64),
            lambda m: m['partition'].update(sha256='0' * 64),
            lambda m: m['original_authority'].update(sha256='0' * 64),
            lambda m: m.update(selected_cpu={} if phase == 'cpu' else None),
        ):
            mutant = copy.deepcopy(launch); mutation(mutant)
            rejects(lambda: driver.check_launch(mutant, args))
    launch = launch_fixture(driver); launch['selected_mechanics'].pop('control')
    rejects(lambda: driver.check_launch(launch, SimpleNamespace(phase='train', arm='candidate', seed=179061,
                                                             execution_sha256='a' * 64)), 'BOTH mechanics')


def terminal_checks(driver):
    launch = launch_fixture(driver)
    record = {'schema': driver.SCHEMA, 'phase': 'mechanics', 'arm': 'candidate', 'seed': 179061,
              'pass': True, 'quality_read': False, 'exit_rehash_pass': True, 'strict_reload_exact': True,
              'optimizer_members': 5, 'launch': launch_fixture(driver, 'mechanics'),
              'resource_policy': driver.policy('mechanics'), 'trained_state_reused': False,
              'completed_step': 17, 'checkpoint': None, 'training_state_discarded': True,
              'replay_exact': True, 'peak_cuda_allocated_bytes': 1000000}
    record['steps'] = [{'step': i, 'batch': list(range(64)), 'ce': 1., 'rank': .1, 'loss': 1.8,
                        'scale': 128., 'preclip_norm': 1., 'seconds': .01,
                        **{k: 'b' * 64 for k in ('schedule_sha256', 'feature_rows_sha256', 'mixing_sha256',
                                                'clean_bank_sha256', 'state_sha256')},
                        'gradient_norms': dict.fromkeys(driver.PARAMETERS, 0.)} for i in range(1, 18)]
    record['resumed_steps'] = copy.deepcopy(record['steps'][8:])
    driver.check_terminal_record(record, launch, 'mechanics', 'candidate')
    for mutation in (lambda m: m.update(optimizer_members=208), lambda m: m.update(completed_step=16),
                     lambda m: m['steps'][0]['batch'].__setitem__(0, 6355),
                     lambda m: m['resumed_steps'][0].update(clean_bank_sha256='c' * 64),
                     lambda m: m.update(checkpoint={'path': 'partial'}),
                     lambda m: m.update(peak_cuda_allocated_bytes=10_000_000_000)):
        mutant = copy.deepcopy(record); mutation(mutant)
        rejects(lambda: driver.check_terminal_record(mutant, launch, 'mechanics', 'candidate'))


def partition_checks(driver, path):
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == driver.PARTITION_SHA
    partition = json.loads(raw)
    targets = [None] * 13283
    # Metadata-only artificial dense labels, consistent with the real panel map.
    for panel in partition['panels'].values():
        ids = panel['original_class_ids']
        if 'query' in panel:
            for role in ('query', 'gallery'):
                for position, row in enumerate(panel[role]):
                    targets[panel['original_rows'][row]] = ids[position % len(ids)]
        else:
            for position, row in enumerate(panel['original_rows']):
                targets[row] = ids[position % len(ids)]
    fit = {'targets': targets, 'class_names': partition['global_class_names']}
    labels = driver.check_partition(partition, fit)
    assert len(labels) == driver.ROWS and sorted(set(labels)) == list(range(driver.CLASSES))
    for mutation in (
        lambda m: m['panels']['train']['original_rows'].__setitem__(0, m['panels']['selection']['original_rows'][0]),
        lambda m: m['panels']['train']['original_class_ids'].__setitem__(0, m['panels']['selection']['original_class_ids'][0]),
        lambda m: m['panels']['selection']['query'].__setitem__(0, m['panels']['selection']['gallery'][0]),
        lambda m: m['panels']['selection']['query'].__setitem__(0, 3449),
        lambda m: m['original_cache'].update(sha256='0' * 64),
    ):
        mutant = copy.deepcopy(partition); mutation(mutant)
        rejects(lambda: driver.check_partition(mutant, fit))


def file_checks(driver, root):
    path = root / 'input'; path.write_bytes(b'original')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    guards = {}; driver.bound_file(guards, path, digest)
    path.write_bytes(b'changed')
    rejects(lambda: driver.bound_file(guards, path, digest), 'file SHA256 differs')
    link = root / 'link'; link.symlink_to(path)
    rejects(lambda: driver.bound_file({}, link, digest), 'canonical file')
    rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate JSON')
    rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite JSON')
    for name in driver.FILES:
        (root / name).write_text('# frozen\n')
    code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in driver.FILES}
    manifest = root / 'execution.json'; manifest.write_text(json.dumps(code))
    sha = lambda: hashlib.sha256(manifest.read_bytes()).hexdigest()
    driver.closure(root, sha(), driver.FILES, {})
    manifest.write_text(json.dumps(dict(code, extra='0' * 64)))
    rejects(lambda: driver.closure(root, sha(), driver.FILES, {}), 'exact execution closure')


def initializer_lifetime_checks(driver, root):
    """Real run/initializer/loader boundaries, scalar stand-ins; native math UNRUN.

    Loading inside initializer again must fail on its independent reconstruction.
    """
    path = Path(driver.__file__).with_name('initialize_siglip2_substrate_fit.py')
    spec = importlib.util.spec_from_file_location('_identity_mix_lifetime_init', path)
    init = importlib.util.module_from_spec(spec); spec.loader.exec_module(init)
    helper_path = root / 'representation_ceiling.py'
    helper_path.write_text('''fit_calls = 0
transforms = []
class CenteredPcaTransform:
    def __init__(self, mean, components):
        self.mean, self.components = mean, components
        transforms.append(self)
    def apply(self, value):
        return value.clone()
def fit_centered_pca(value, dimensions):
    global fit_calls
    fit_calls += 1
    assert dimensions == 128
    return CenteredPcaTransform(value.mean(0), value.clone())
''')
    digest = hashlib.sha256(helper_path.read_bytes()).hexdigest()

    class Scalar:
        shape, device = (driver.ROWS, driver.WIDTH), SimpleNamespace(type='cpu')
        def __init__(self, value=1): self.value = value
        def clone(self): return Scalar(self.value)
        def __matmul__(self, other): return Scalar(self.value * other.value)
        def __neg__(self): return Scalar(-self.value)
        def __sub__(self, other): return Scalar(self.value - other.value)
        def __add__(self, other): return Scalar(self.value + other.value)
        def __getitem__(self, index): return self
        def __setitem__(self, index, value): self.value = value.value
        def __gt__(self, other): return Scalar(self.value > other)
        def __float__(self): return float(self.value)
        def all(self): return self
        def item(self): return self.value
        def mean(self, dimension): return self.clone()
        def std(self, unbiased): return Scalar()
        def norm(self, dim): return Scalar()
        def detach(self): return self
        def numpy(self): return [0]
        def copy_(self, other): self.value = other.value
        def div_(self, other): self.value /= other.value

    heads = []
    def head_from(arm, tensors):
        assert arm == 'control' and float(tensors['primary.bias']) == -1
        class Head:
            def __call__(self, value): return value.clone()
            def state_dict(self): return tensors
        head = Head()
        head.down = lambda value: value.clone()
        head.down.weight = tensors['down.weight']
        head.center, head.preactivation_std = tensors['center'], tensors['preactivation_std']
        heads.append(head)
        return head

    normalize = lambda value, dim: value.clone()
    nn = SimpleNamespace(init=SimpleNamespace(kaiming_uniform_=lambda *a, **kw: None),
                         functional=SimpleNamespace(normalize=normalize))
    torch = SimpleNamespace(nn=nn, zeros=lambda *a: Scalar(0), ones=lambda *a: Scalar(),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed), no_grad=nullcontext,
        tensor=lambda *a, **kw: Scalar(), from_numpy=lambda value: Scalar(), int64='int64',
        isfinite=lambda value: Scalar(True), cuda=SimpleNamespace(is_initialized=lambda: False),
        set_num_threads=lambda n: None, get_num_interop_threads=lambda: 1,
        random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: None),
                               get_rng_state=lambda: Scalar()))
    before = {'path': '/test/lifetime.service', 'values': {'memory.max': str(8 * 1024**3),
        'memory.current': '1', 'memory.peak': '1', 'memory.swap.current': '0',
        'memory.swap.peak': '0', 'memory.swap.max': '0', 'memory.events': 'max 0\noom 0\noom_kill 0'}}
    flags = {'threads': 1, 'interop_threads': 1}
    parent = {'root': root, 'launch': {'pca_helper_sha256': digest}}
    prior = {'python': str(Path(sys.executable).resolve()), 'python_sha256': init.sha(Path(sys.executable).resolve()),
             'python_version': sys.version, 'numerical_flags': flags, 'origins': {'files': {}}}
    context = {'initialized': {'init': init, 'pca': parent, 'source': SimpleNamespace(
        cgroup_memory=lambda: before, numerical_flags=lambda: flags, imported_origins=lambda *a: {'files': {}}),
        'source_context': {'extract': None}, 'packages': {}}, 'old_cpu': {'invocation': prior, **prior},
        'original': SimpleNamespace(reference_math=lambda old: SimpleNamespace(
            member_bank_positive_ordinals=lambda *a, **kw: Scalar())), 'old': {},
        'cached': SimpleNamespace(head_from=head_from), 'target': [0],
        'partition': {'panels': {'train': {'original_rows': [0]}}}}
    args = SimpleNamespace(phase='cpu', arm='control', seed=driver.SEEDS[0], output=root / 'output',
                           execution_sha256='a' * 64, authority=root / 'authority', authority_sha256='b' * 64)
    argv = [str(Path(driver.__file__).absolute()), '--execution-sha256', args.execution_sha256,
        '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
        '--phase', args.phase, '--arm', args.arm, '--seed', str(args.seed), '--output', str(args.output)]
    imports, results = [], []
    real_import = __import__
    def scalar_import(name, *a, **kw):
        if name in ('torch', 'torch.nn'):
            assert context['pca_helper'] is sys.modules['_identity_mix_pinned_pca']
            imports.append(name)
            return torch if name == 'torch' else nn
        return real_import(name, *a, **kw)
    class BoundaryComplete(Exception): pass
    def witnesses(ctx, ref, output, numerical_flags):
        results.append(driver.initializer(ctx, ref, Scalar()))
        results.append(driver.initializer(ctx, ref, Scalar(), pca=results[0]['pca']))
        raise BoundaryComplete
    with patch.object(driver, 'authority', return_value=context), patch.object(driver, 'cpu_witnesses', witnesses), \
         patch.object(driver, 'schedule_and_mixing', return_value=(None, {})), patch.object(sys, 'argv', argv), \
         patch.dict(driver.os.environ, CUDA_VISIBLE_DEVICES='', INVOCATION_ID='c' * 32), \
         patch('builtins.__import__', scalar_import):
        before['values']['memory.max'] = '1'
        rejects(lambda: driver.run(args), 'whole-cgroup memory/swap caps differ')
        before['values']['memory.max'] = str(8 * 1024**3)
        assert not imports and '_identity_mix_pinned_pca' not in sys.modules
        try:
            driver.run(args)
        except BoundaryComplete:
            pass
        else:
            raise AssertionError('initializer boundary did not execute')
        helper = sys.modules['_identity_mix_pinned_pca']
        assert context['pca_helper'] is helper
        assert helper.__file__ == helper.__spec__.origin == str(helper_path)
        assert helper.fit_calls == 1 and len(helper.transforms) == len(heads) == len(results) == 2
        assert helper.transforms[0] is not helper.transforms[1] and heads[0] is not heads[1]
        assert results[0]['head'] is not results[1]['head']
        assert results[0]['classifier'] is not results[1]['classifier']
        assert results[0]['bank'] is not results[1]['bank']
        output_imports = imports.copy()
        parent['root'] = root / 'missing'
        rejects(lambda: driver.run(args), 'canonical regular file')
        parent['root'] = root
        parent['launch']['pca_helper_sha256'] = '0' * 64
        rejects(lambda: driver.run(args), 'bound file SHA256 differs')
        parent['launch']['pca_helper_sha256'] = digest
        rejects(lambda: driver.run(args), 'helper already loaded; origin is not admissible')
        assert imports == output_imports and sys.modules['_identity_mix_pinned_pca'] is helper


def source_checks(driver, path):
    tree = ast.parse(path.read_bytes())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    # These are observable contracts, not assertions of native arithmetic success.
    clean = ast.unparse(functions['clean_bank_descriptors'])
    update = ast.unparse(functions['update'])
    assert 'torch.no_grad()' in clean and "state['features'][index]" in clean and '.detach()' in clean
    assert update.index('clean_bank_descriptors(') < update.index('scaler.step(')
    assert 'member_bank_refresh_values(clean, clean,' in update and 'member_bank_refresh_rows(batch)' in update
    assert 'rows.append(raw.detach())' not in update
    features = ast.unparse(functions['training_features'])
    assert "mmap_mode='r'" in features and "cache[context['partition']['panels']['train']['original_rows']]" in features
    init = ast.unparse(functions['initializer'])
    assert 'fit_centered_pca(normalized, dimensions=DIM)' in init and 'load_initializers' not in init
    mixing = ast.unparse(functions['schedule_and_mixing'])
    assert 'PCG64(seed + offset)' in mixing and 'coefficient_rng.random((1000, 64)).astype(np.float32)' in mixing
    restore = ast.unparse(functions['restore'])
    for text in ('weights_only=True', 'load_state_dict', 'set_rng_state', 'set_rng_state_all', 'check_payload'):
        assert text in restore
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assert not any('__globals__' in ast.unparse(t) for t in targets)
        if isinstance(node, ast.Call):
            assert not ast.unparse(node.func).endswith(('reset_peak_memory_stats', 'fresh_source', 'load_initializers'))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any('torch' in ast.unparse(n) or 'numpy' in ast.unparse(n) for n in imports)
    original = path.parent / 'train_siglip2_cached_readout.py'
    assert hashlib.sha256(original.read_bytes()).hexdigest() == driver.CACHED_TRAINER_SHA


def main():
    path = Path(__file__).resolve().with_name('train_siglip2_identity_mix.py')
    spec = importlib.util.spec_from_file_location('_identity_mix_test', path)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    mixing_checks(driver); resume_checks(driver); authority_checks(driver); terminal_checks(driver)
    partition_checks(driver, path.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/identity-mix-v1/partition.json')
    with TemporaryDirectory() as directory:
        file_checks(driver, Path(directory))
    with TemporaryDirectory() as directory:
        initializer_lifetime_checks(driver, Path(directory))
    source_checks(driver, path)
    for file in (path, Path(__file__).resolve()):
        compile(file.read_bytes(), str(file), 'exec')
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, '-B', '-S', flag, str(file), '--help'], capture_output=True, text=True)
            assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
    result = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and all('--' + n in result.stdout for n in
        ('execution-sha256', 'authority-sha256', 'phase', 'arm', 'seed', 'output'))
    assert not any(n.split('.')[0] in {'torch', 'numpy', 'PIL', 'transformers', 'sfora'} for n in sys.modules)
    print('PASS: donors/singletons/mixing/norm/clean-bank/resume/partition/authority/tamper/initializer-lifetime/syntax/help/-O/-OO; native UNRUN')


if __name__ == '__main__':
    main()
