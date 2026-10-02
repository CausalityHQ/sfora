#!/usr/bin/env python3
"""Five-second stdlib falsifier; native comparison is parent-owned."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch


PATH = Path(__file__).resolve().with_name('compare_quadratic_fresh_copy.py')


def rejected(action, text):
    try:
        action()
    except (ValueError, OSError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError('accepted: ' + text)


def check():
    spec = importlib.util.spec_from_file_location('_fresh_copy_test', PATH)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)
    assert d.POLICY == dict(seconds=300, host_bytes=8589934592, swap_bytes=0,
                           cuda_allocated_bytes_exclusive=10000000000)
    assert len(d.FILES) == 2 and d.BASELINE_CODE.keys() == {
        'train_siglip2_quadratic_readout.py', 'test_siglip2_quadratic_readout.py',
        'quadratic_readout.py', 'quadratic_encoder_frames.py'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in d.FILES:
            (root / name).write_bytes(name.encode())
        code = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in d.FILES}
        raw = json.dumps(code).encode()
        (root / 'execution.json').write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        guards = {}
        assert d.closure(root, digest, d.FILES, guards) == code
        assert len(guards) == 3
        victim = root / next(iter(d.FILES))
        prior = victim.stat()
        victim.write_bytes(b'x' * prior.st_size)
        os.utime(victim, ns=(prior.st_atime_ns, prior.st_mtime_ns))
        rejected(lambda: d.closure(root, digest, d.FILES, {}), 'SHA256')
        (root / 'execution.json').write_text(json.dumps({**code, 'extra.py': 'a' * 64}))
        sha = hashlib.sha256((root / 'execution.json').read_bytes()).hexdigest()
        rejected(lambda: d.closure(root, sha, d.FILES, {}), 'closure')
        rejected(lambda: d.strict_json(b'{"a":1,"a":2}'), 'duplicate')
        rejected(lambda: d.strict_json(b'{"a":NaN}'), 'nonfinite')
        rejected(lambda: d.source_bytes({'path': str(victim), 'sha256': 'A' * 64}, {}), 'FILE')
        authority = dict(schema=d.AUTHORITY_SCHEMA, execution_sha256='d' * 64,
            python=d.PYTHON.copy(), baseline=d.BASELINE.copy(),
            original_authority={'path': str(root / 'original.json'), 'sha256': d.ORIGINAL_AUTHORITY_SHA},
            proposed_helper={'path': str(root / 'quadratic_encoder_frames.py'), 'sha256': 'b' * 64},
            output=str(root / 'new'), unit='fresh-native-test', both_locks_held=True,
            resource_policy=d.POLICY.copy(), qualification_eligible=False,
            state_reuse_eligible=False, quality_read=False)
        args = SimpleNamespace(execution_sha256='d' * 64, output=root / 'new')
        d.check_authority(authority, args)
        mutations = [('schema', 'bad'), ('extra', 0), ('qualification_eligible', True),
            ('state_reuse_eligible', True), ('quality_read', True), ('both_locks_held', False),
            ('execution_sha256', 'e' * 64), ('unit', 'bad/service'), ('output', str(root / 'wrong')),
            ('resource_policy', dict(d.POLICY, seconds=301)),
            ('baseline', dict(d.BASELINE, execution_sha256='e' * 64)),
            ('python', dict(d.PYTHON, sha256='e' * 64)),
            ('original_authority', dict(authority['original_authority'], sha256='e' * 64))]
        for key, value in mutations:
            rejected(lambda key=key, value=value: d.check_authority({**authority, key: value}, args), 'authority')
        d.exclusive_output(root / 'new', [root / 'sources'])
        (root / 'new').mkdir()
        rejected(lambda: d.exclusive_output(root / 'new', []), 'exclusive')
        (root / 'link').symlink_to(root / 'new', target_is_directory=True)
        rejected(lambda: d.exclusive_output(root / 'link', []), 'canonical')
        rejected(lambda: d.exclusive_output(root / 'sources' / 'out', [root / 'sources']), 'exclusive')
        receipt = root / 'receipt.json'
        d.write_receipt(receipt, {'pass': False})
        assert json.loads(receipt.read_bytes()) == {'pass': False}
        rejected(lambda: d.write_receipt(receipt, {'pass': True}), 'exists')

        # Authenticate both complete closures and the copied authority before
        # the original admission seam; this source stand-in imports no native code.
        driver, original, proposed_root = (root / n for n in ('driver', 'original', 'proposed'))
        for path in (driver, original, proposed_root): path.mkdir()
        for name in d.FILES: (driver / name).write_text('# stdlib source fixture\n')
        for name in d.BASELINE_CODE: (original / name).write_text('# baseline source fixture\n')
        (original / 'train_siglip2_quadratic_readout.py').write_text('''from types import SimpleNamespace
def authority(args):
    assert (args.phase, args.arm, args.seed) == ('mechanics', 'control', 179061)
    return dict(args=args, guards={}, frames=SimpleNamespace(ORIGINAL_SHA256='pin'),
                genuine=SimpleNamespace(load_helper=load_helper))
def load_helper(name, path, digest, guards):
    return SimpleNamespace(ORIGINAL_SHA256='pin', seal=lambda *args: None)
''')
        def manifest(path, names):
            mapping = {n: hashlib.sha256((path / n).read_bytes()).hexdigest() for n in names}
            raw = json.dumps(mapping).encode()
            (path / 'execution.json').write_bytes(raw)
            return mapping, hashlib.sha256(raw).hexdigest()
        own_code, own_sha = manifest(driver, d.FILES)
        original_code, original_sha = manifest(original, d.BASELINE_CODE)
        copied = driver / 'original-authority.json'
        copied.write_bytes(b'{}\n')
        copied_sha = hashlib.sha256(copied.read_bytes()).hexdigest()
        helper = proposed_root / 'quadratic_encoder_frames.py'
        helper.write_text('# proposed source fixture\n')
        helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
        python = Path(sys.executable).resolve()
        with python.open('rb') as stream:
            python_binding = dict(path=str(python), sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
        baseline_binding = dict(root=str(original), execution_sha256=original_sha)
        config = {**authority, 'execution_sha256': own_sha, 'python': python_binding,
            'baseline': baseline_binding, 'original_authority': dict(path=str(copied), sha256=copied_sha),
            'proposed_helper': dict(path=str(helper), sha256=helper_sha), 'output': str(root / 'fresh')}
        launch_path = driver / 'authority.json'
        launch_path.write_text(json.dumps(config))
        runtime_args = SimpleNamespace(execution_sha256=own_sha, authority=launch_path,
            authority_sha256=hashlib.sha256(launch_path.read_bytes()).hexdigest(), output=root / 'fresh')
        with patch.multiple(d, __file__=str(driver / PATH.name), BASELINE=baseline_binding,
                BASELINE_CODE=original_code, PYTHON=python_binding, PYTHON_VERSION=sys.version,
                ORIGINAL_AUTHORITY_SHA=copied_sha), patch.object(sys, 'argv', d.cli(driver, runtime_args)), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                           INVOCATION_ID='1' * 32):
            try:
                old, context, loaded, admitted, own = d.prepare(runtime_args)
                assert own == own_code and admitted == config and loaded.ORIGINAL_SHA256 == 'pin'
                assert context['args'].authority_sha256 == copied_sha and context['args'].output == runtime_args.output
                assert context['guards'][str(helper)] == helper_sha
                assert all(context['guards'][str(original / n)] == h for n, h in original_code.items())
            finally:
                sys.modules.pop('_quadratic_fresh_copy_baseline', None)
            assert not runtime_args.output.exists()
            helper.write_text('# tampered proposed fixture\n')
            rejected(lambda: d.prepare(runtime_args), 'SHA256')
            assert '_quadratic_fresh_copy_baseline' not in sys.modules

    expected = ('a' * 64, 'a' * 64, 'b' * 64)
    calls, syncs, rows = [], [], []
    clock = SimpleNamespace(value=0.)
    def now(): return clock.value
    def sync(): syncs.append(len(calls))
    def arm(name, seconds, digest=expected):
        def run():
            calls.append(name)
            clock.value += seconds
            return digest
        return run
    result = d.paired_samples(arm('baseline', .04), arm('proposed', .02), sync, rows, now)
    assert calls == [n for i in range(8) for n in
                     (('baseline', 'proposed') if i % 2 == 0 else ('proposed', 'baseline'))]
    assert len(syncs) == 32 and len(rows) == 8
    assert result['cost_improved'] and result['paired_median_delta_seconds'] < 0
    assert all(tuple(row['baseline']['digests']) == tuple(row['proposed']['digests']) == expected for row in rows)
    rows.clear(); clock.value = 0.
    result = d.paired_samples(arm('baseline', .02), arm('proposed', .04), sync, rows, now)
    assert not result['cost_improved'] and len(rows) == 8
    rejected(lambda: d.require(result['cost_improved'], 'no median improvement'), 'median')
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01), arm('proposed', .01,
        ('c' * 64, *expected[1:])), sync, rows, now), 'digests')
    assert len(rows) == 1  # Failure evidence survives.
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', 6.), arm('proposed', 6.), sync, rows, now), '10-second')
    assert len(rows) == 1
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01, ('x',) * 3),
        arm('proposed', .01, ('x',) * 3), sync, rows, now), 'digests')
    rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01), arm('proposed', .01),
        sync, rows, now, expected=('c' * 64, 'c' * 64, 'd' * 64)), 'digests')
    def broken_sync():
        if calls[-1:] == ['baseline']:
            raise ValueError('sync failed')
    calls.clear(); rows.clear(); clock.value = 0.
    rejected(lambda: d.paired_samples(arm('baseline', .01), arm('proposed', .01),
        broken_sync, rows, now), 'sync failed')
    calls.clear(); rows.clear(); clock.value = 0.
    def broken_arm():
        calls.append('baseline')
        raise ValueError('arm failed')
    rejected(lambda: d.paired_samples(broken_arm, arm('proposed', .01),
        broken_sync, rows, now), 'arm failed')

    # Replace precisely the two admitted capsules; tensor objects remain identical.
    encoder, partition, other = object(), object(), object()
    proposed = (object(), object())
    tree = dict(encoder=encoder, partition=partition, tensor=other)
    assert d.replace_capsules(tree, (encoder, partition), proposed) == dict(
        encoder=proposed[0], partition=proposed[1], tensor=other)
    rejected(lambda: d.replace_capsules(dict(tree, encoder=object()),
        (encoder, partition), proposed), 'capsule')
    order = []
    primary = ValueError('primary')
    def fail():
        order.append('release')
        raise RuntimeError('cleanup')
    retained = d.cleanup_steps(primary, [('release', fail), ('exit', lambda: order.append('exit'))], {})
    assert retained is primary and order == ['release', 'exit'] and 'cleanup' in primary.__notes__[0]
    rejected(lambda: (_ for _ in ()).throw(d.cleanup_steps(None,
        [('exit', lambda: (_ for _ in ()).throw(ValueError('fresh exit failed')))], {})), 'fresh exit')

    tree = ast.parse(PATH.read_bytes())
    banned = {'run', 'update', 'gpu_run', 'cpu_witnesses', 'save', 'restore'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
        isinstance(n.func.value, ast.Name) and n.func.value.id == 'old' and n.func.attr in banned
        for n in ast.walk(tree))
    probe = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native_probes')
    assignments = {n.targets[0].id: n.value for n in probe.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name)}
    for name in ('cpu', 'cuda'):
        view = assignments[name]
        assert isinstance(view, ast.Subscript) and ast.unparse(view.value) == name + '_base.T'
        slices = view.slice.elts
        assert [s.lower.value for s in slices] == [1, 1] and slices[0].step.value == 2
    guard = next(n for n in probe.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and 'nonzero-offset' in ast.unparse(n))
    assert all(text in ast.unparse(guard) for text in (
        'view.storage_offset() > 0', 'not view.is_contiguous()',
        'view.untyped_storage().data_ptr() == base.untyped_storage().data_ptr()'))
    assert ast.unparse(assignments['saved']) == 'cuda[0, 0].item()'
    mutation = next(n for n in probe.body if isinstance(n, ast.Try))
    assert ast.unparse(mutation.body[0]) == 'cuda.data[0, 0] += 7'
    assert len(mutation.finalbody) == 1 and ast.unparse(mutation.finalbody[0]) == 'cuda.data[0, 0] = saved'
    after = probe.body[probe.body.index(mutation) + 1:]
    assert ast.unparse(after[0]) == 'restored = original(ordinary)'
    assert 'cuda._version == version and restored == baseline(old_value) == proposed(new_value) == expected' \
        in ast.unparse(after[1])
    for flags in (['-B', '--help'], ['-O', '-B', '--help']):
        run = subprocess.run([sys.executable, *flags[:-1], str(PATH), flags[-1]],
                             capture_output=True, text=True, timeout=2)
        assert (run.returncode == 0) == ('-O' not in flags), run.stderr
        assert ('optimized mode' in run.stderr) if '-O' in flags else '--authority-sha256' in run.stdout
    assert not any(n.split('.')[0] in d.NATIVE for n in sys.modules)
    print('PASS: stdlib authority/closure/output, eight paired samples, negatives, cleanup; native UNRUN')


if __name__ == '__main__':
    signal.signal(signal.SIGALRM, lambda *unused: (_ for _ in ()).throw(TimeoutError('5-second test cap')))
    signal.alarm(5)
    try:
        check()
    finally:
        signal.alarm(0)
