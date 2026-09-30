#!/usr/bin/env python3
"""Stdlib admission/math/layout checks; all real tensor work belongs to parent."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import builtins
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import io
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return sha(path)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError, TypeError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid input accepted: ' + message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Flags(list):
    def any(self):
        return any(self)

    def sum(self):
        return sum(self)


class Positives(list):
    def __ge__(self, other):
        return SimpleNamespace(any=lambda dim: Flags(any(n >= other for n in row) for row in self))

    def __getitem__(self, key):
        return Positives(v for v, flag in zip(self, key, strict=True) if flag)


class Rows(list):
    def __getitem__(self, key):
        return Rows(v for v, flag in zip(self, key, strict=True) if flag)

    def sum(self):
        return sum(self)


def checkpoint_write_checks(driver, root):
    # Removing the writer from save must fail on oversized writes, before final SHA.
    events, lifecycle = [], []
    page_size = driver.os.sysconf('SC_PAGESIZE')
    real_io = True
    raw = None
    class Stream:
        def write(self, data):
            events.append(('write', raw.tell(), len(data)))
            assert len(data) <= 1024**2, 'checkpoint write exceeds 1MiB'
            return raw.write(data)
        def tell(self):
            return raw.tell()
        def fileno(self):
            return raw.fileno()
        def flush(self):
            events.append(('flush', raw.tell()))
            raw.flush()
    def exclusive(path):
        nonlocal raw
        raw = path.open('x+b')
        class Exclusive:
            def __enter__(self):
                return Stream()
            def __exit__(self, *args):
                raw.close()
        return Exclusive()
    real_fsync, real_advice = driver.os.fsync, driver.os.posix_fadvise
    def fsync(fd):
        assert fd == raw.fileno()
        events.append(('fsync', raw.tell()))
        if real_io:
            real_fsync(fd)
    def advice(fd, offset, count, hint):
        assert fd == raw.fileno() and hint == driver.os.POSIX_FADV_DONTNEED
        assert offset % page_size == count % page_size == 0 and 0 < count <= 64 * 1024**2 + page_size
        assert offset + count <= raw.tell() < offset + count + page_size
        assert events[-2:] == [('flush', raw.tell()), ('fsync', raw.tell())], events[-2:]
        events.append(('advice', offset, count))
        if real_io:
            real_advice(fd, offset, count, hint)
    saved, identity, flags = {'complete': ['original', 8]}, {'seed': 179032}, {'threads': 1}
    state = {'optimizer': SimpleNamespace(zero_grad=lambda **kw: lifecycle.append(('zero_grad', kw))),
             'frozen_cache': {}}
    pieces = ()
    def serialize(value, stream):
        assert value is saved
        lifecycle.append('serialize')
        for piece in pieces:
            assert stream.write(piece) == len(piece)
        stream.flush()  # Deployed zip writer's exit flush.
    def imported(name, *args):
        assert name == 'torch'
        return SimpleNamespace(save=serialize)
    def integrity(actual, binding):
        assert actual is state and binding is identity
        lifecycle.append('integrity')
    def payload(actual, binding, numerical):
        assert actual is state and binding is identity and numerical is flags
        lifecycle.append('payload')
        return saved
    def fingerprint(value, frozen):
        assert value is saved and frozen is state['frozen_cache']
        lifecycle.append('fingerprint')
        return 'complete-state-fingerprint'
    def full_sha(path):
        assert raw.closed  # Preserve the original full-file hash after exclusive close.
        lifecycle.append('sha')
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    # Run the real save code with a serializer standin; no Torch import/patch or source-helper mutation.
    save = FunctionType(driver.save.__code__, {**vars(driver), 'integrity': integrity, 'payload': payload,
        'fingerprint': fingerprint, '__builtins__': {**vars(builtins), '__import__': imported}})
    context = {'initialized': {'source_context': {'extract': SimpleNamespace(exclusive=exclusive)},
                               'init': SimpleNamespace(sha=full_sha)}}
    block = bytes(range(256)) * 8192 + b'boundary'
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zipped:
        for name, data in (('state', block), ('empty', b''), ('tail', b'tail!')):
            zipped.writestr(zipfile.ZipInfo(name), data)
    zip_bytes = archive.getvalue()
    # Deterministic irregular serializer calls, including >1MiB, empty and a partial page.
    zip_pieces = (zip_bytes[:17], zip_bytes[17:1048610], b'', zip_bytes[1048610:1048683],
                  zip_bytes[1048683:-3], zip_bytes[-3:])
    for index, pieces in enumerate(((), (b'',), (b'header', block, b'', b'tail!'),
                                    zip_pieces)):
        events.clear(); lifecycle.clear()
        checkpoint = root / f'checkpoint-{index}.bin'
        expected = hashlib.sha256()
        for piece in pieces:
            expected.update(piece)
        with patch.object(driver.os, 'fsync', fsync), patch.object(driver.os, 'posix_fadvise', advice):
            assert save(context, state, identity, flags, checkpoint) == (
                expected.hexdigest(), 'complete-state-fingerprint')
        with checkpoint.open('rb') as stream:
            for piece in pieces:
                assert stream.read(len(piece)) == piece
            assert stream.read(1) == b''
        assert lifecycle == [('zero_grad', {'set_to_none': True}), 'integrity', 'payload',
                             'serialize', 'sha', 'fingerprint']
        if index == 3:
            with zipfile.ZipFile(checkpoint) as zipped:
                assert zipped.namelist() == ['state', 'empty', 'tail']
                assert [zipped.read(name) for name in zipped.namelist()] == [block, b'', b'tail!']
        total = sum(map(len, pieces))
        expected_ranges = [('advice', 0, total - total % page_size)] if total >= page_size else []
        assert [event for event in events if event[0] == 'advice'] == expected_ranges
        assert events[-2:] == [('flush', checkpoint.stat().st_size), ('fsync', checkpoint.stat().st_size)]
        rejects(lambda: save(context, state, identity, flags, checkpoint), 'File exists')
    # Exercise both real 64MiB boundaries without writing a large regression artifact.
    class CountingRaw:
        position = 0
        def write(self, data):
            self.position += len(data)
            return len(data)
        def tell(self):
            return self.position
        def fileno(self):
            return 123
        def flush(self):
            pass
    raw, real_io = CountingRaw(), False
    events.clear()
    with patch.object(driver.os, 'fsync', fsync), patch.object(driver.os, 'posix_fadvise', advice):
        writer = driver.CheckpointWriter(Stream())
        for _ in range(64):
            assert writer.write(block) == len(block)
            assert writer.write(b'') == 0
        assert writer.write(b'tail' * page_size + b'!') == 4 * page_size + 1
        writer.flush()
    assert [event for event in events if event[0] == 'advice'] == [
        ('advice', 0, 67108864), ('advice', 67108864, 67108864), ('advice', 134217728, 4 * page_size)]
    # Every low-level failure and short write is fatal, including final-tail flush/advice.
    with (root / 'writer-errors.bin').open('x+b') as raw:
        stream = Stream()
        writer = driver.CheckpointWriter(stream)
        for operation in ('write', 'flush'):
            with patch.object(stream, operation, side_effect=OSError('injected ' + operation)):
                rejects(lambda: writer.write(b'x') if operation == 'write' else writer.flush(), 'injected')
        for result in (0, None):
            with patch.object(stream, 'write', return_value=result):
                rejects(lambda: writer.write(b'x'), 'short checkpoint write')
        assert writer.write(b'tail' * page_size + b'!') == 4 * page_size + 1
        for operation in ('fsync', 'posix_fadvise'):
            with patch.object(driver.os, operation, side_effect=OSError('injected ' + operation)):
                rejects(writer.flush, 'injected')
        writer.flush()
        raw.seek(0)
        rejects(lambda: writer.write(b'wrong offset'), 'checkpoint offset disorder')
        rejects(writer.flush, 'checkpoint offset disorder')
        raw.seek(1)
        rejects(lambda: driver.CheckpointWriter(stream), 'checkpoint offset disorder')
    with (root / 'short-write.bin').open('x+b') as raw:
        stream = Stream()
        writer = driver.CheckpointWriter(stream)
        with patch.object(stream, 'write', side_effect=lambda data: raw.write(data[:-1])):
            rejects(lambda: writer.write(b'bytes'), 'short checkpoint write')
        assert raw.tell() == 4
        rejects(writer.flush, 'checkpoint offset disorder')


def main():
    path = Path(__file__).with_name('train_siglip2_substrate_adaptation.py').resolve()
    driver = module(path, 'adaptation_under_test')
    assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
    tree = ast.parse(path.read_bytes())
    # No inherited startup/package initialization, peak reset, quality API or rescue.
    banned = {'reset_peak_memory_stats', 'evaluate', 'from_pretrained_model', 'set_default_dtype'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in banned
    for flags, arguments, code, marker in (([], ['--help'], 0, '--authority-sha256'),
                                          ([], [], 2, 'required'),
                                          (['-O'], ['--help'], 1, 'optimized'),
                                          (['-OO'], ['--help'], 1, 'optimized')):
        result = subprocess.run([sys.executable, '-B', '-S', *flags, str(path), *arguments],
                                capture_output=True, text=True)
        assert result.returncode == code and marker in result.stdout + result.stderr
    assert driver.policy('mechanics')['seconds'] == 120 and driver.policy('train')['seconds'] == 300
    rejects(lambda: driver.policy('quality'), 'fixed phase')
    native_facts = {'seed': 179041, 'counter': 0, 'parameter_names': ['original'],
                    'optimizer_defaults': {'betas': (.9, .999), 'eps': 1e-8},
                    'optimizer_groups': [{'betas': (.9, .999), 'lr': 1e-5}],
                    'optimizer_state': {'param_groups': [{'betas': (.9, .999), 'params': [0, 1]}], 'state': {}},
                    'runtime': {'config': {'id2label': {0: 'zero', 1: 'one'}}}}
    persisted_facts = json.loads(json.dumps({**native_facts, 'seed': 179032}))
    assert {**native_facts, 'seed': 179032} != persisted_facts  # Reproduces the actual JSON boundary defect.
    driver.check_cpu_state(native_facts, persisted_facts)
    for key in native_facts:
        if key != 'seed':
            rejects(lambda key=key: driver.check_cpu_state({**native_facts, key: 'tampered'}, persisted_facts), 'CPU state')
    targets = [n % 2004 for n in range(13283)]
    batches = [[(step * 64 + offset) % 2004 for offset in range(64)] for step in range(100)]
    for seed in driver.SEEDS:
        driver.check_schedule(batches, targets, seed)
    for changed, altered_targets, seed in ((batches[:-1], targets, 179032),
                                           ([[0]*64] + batches[1:], targets, 179032),
                                           ([[-1] + batches[0][1:]] + batches[1:], targets, 179032),
                                           (batches, targets[:-1], 179032),
                                           (batches, targets, 179019)):
        rejects(lambda: driver.check_schedule(changed, altered_targets, seed), 'schedule')
    calls = []
    def rank(raw, bank, head, positives, ordinals, *, live_head):
        calls.append((list(raw), list(positives), list(ordinals), live_head))
        assert live_head is False and all(any(n >= 0 for n in row) for row in positives)
        return 3.
    ref = SimpleNamespace(member_bank_rank_loss=rank)
    # Valid anchors retain the ORIGINAL microbatch denominator, including zero-valid.
    assert driver.valid_rank(ref, Rows([1, 2, 3, 4]), None, None,
                             Positives([[2, -1], [-1, -1], [0, 1], [-1, -1]]), Rows([0, 1, 2, 3])) == 1.5
    assert calls == [([1, 3], [[2, -1], [0, 1]], [0, 2], False)]
    assert driver.valid_rank(ref, Rows([1, 2]), None, None, Positives([[-1], [-1]]), Rows([0, 1])) == 0
    assert len(calls) == 1
    identity, defaults, groups, names, flags = {'seed': 179032}, {'lr': .001}, [{'params': list(range(208))}], list(range(208)), {'threads': 1}
    saved = {key: None for key in driver.PAYLOAD_KEYS}
    saved.update(schema=driver.SCHEMA, identity=identity, counter=8, seed=179032, numerical_flags=flags,
                 optimizer_defaults=defaults, optimizer={'param_groups': groups,
                    'state': {n: {'step': 8, 'exp_avg': 0., 'exp_avg_sq': 0.} for n in range(208)}},
                 cuda_rng=['CUDA'], scaler={'scale': 128., '_growth_tracker': 8})
    driver.check_payload(saved, identity, 8, defaults, groups, names, flags)
    for key in driver.PAYLOAD_KEYS:
        rejects(lambda key=key: driver.check_payload({k: v for k, v in saved.items() if k != key},
            identity, 8, defaults, groups, names, flags), 'complete state')
    for key, value in (('counter', 7), ('seed', 179041), ('cuda_rng', []), ('optimizer_defaults', {'lr': .1}),
                       ('numerical_flags', {}), ('scaler', {'scale': 64, '_growth_tracker': 8})):
        rejects(lambda: driver.check_payload({**saved, key: value}, identity, 8, defaults, groups, names, flags), 'complete state')
    bad = copy.deepcopy(saved)
    bad['optimizer']['state'][0]['step'] = 7
    rejects(lambda: driver.check_payload(bad, identity, 8, defaults, groups, names, flags), 'complete state')
    bad = copy.deepcopy(saved)
    bad['optimizer']['param_groups'][0]['params'].reverse()
    rejects(lambda: driver.check_payload(bad, identity, 8, defaults, groups, names, flags), 'complete state')
    with patch.dict(sys.modules, {'torch': SimpleNamespace(Tensor=type('NoRealTensor', (), {}))}):
        assert driver.fingerprint({'a': [1, 2], 'b': 'x'}) == driver.fingerprint({'b': 'x', 'a': [1, 2]})
        assert driver.fingerprint({'a': [1, 2]}) != driver.fingerprint({'a': (1, 2)})
        assert driver.fingerprint({'a': 1}) != driver.fingerprint({'a': True})
    with TemporaryDirectory() as directory, patch.dict(sys.modules):
        root = Path(directory).resolve()
        checkpoint_write_checks(driver, root)
        # Phase logs must flush live original-unit facts without entering replay state.
        assert hasattr(driver, 'phase_diagnostic'), 'missing flushed phase diagnostics'
        proc = root / 'cgroup'
        proc.write_text('0::/original.service\n')
        memory = root / 'memory'
        memory.mkdir()
        values = {'memory.current': '128', 'memory.peak': '256', 'memory.max': '8589934592',
                  'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                  'memory.events': 'max 0\noom 0\noom_kill 0',
                  'memory.stat': 'anon 64\nfile 32\nfile_dirty 16\nfile_writeback 0'}
        for name, value in values.items():
            (memory / name).write_text(value + '\n')
        def diagnostic_path(value):
            if value == '/proc/self/cgroup':
                return proc
            path = Path(value)
            assert path == Path('/sys/fs/cgroup')
            return root
        (root / 'original.service').symlink_to(memory, target_is_directory=True)
        flushed = []
        class PhaseLog(io.StringIO):
            def flush(self):
                flushed.append(self.getvalue())
        output = PhaseLog()
        rng, replay = random.getstate(), {'step': 8, 'state_sha256': 'fixed', 'seconds': 1.}
        original_replay = driver.diagnostic(replay)
        with patch.object(driver, 'Path', side_effect=diagnostic_path), \
                patch.object(driver.time, 'monotonic', side_effect=(102.5, 104., 105.)), \
                patch.dict(driver.os.environ, {'INVOCATION_ID': 'a'*32}), patch.object(sys, 'stdout', output):
            assert driver.phase_diagnostic('save8.begin', 100.) is None
            (memory / 'memory.current').write_text('512\n')
            (memory / 'memory.peak').write_text('8589934592\n')
            (memory / 'memory.events').write_text('max 1\noom 0\noom_kill 0\n')
            (memory / 'memory.stat').write_text('anon 64\nfile 448\nfile_dirty 400\nfile_writeback 16\n')
            expected_bytes = {p.name: p.read_bytes() for p in memory.iterdir()}
            assert driver.phase_diagnostic('save8.end', 100.) is None
            assert {p.name: p.read_bytes() for p in memory.iterdir()} == expected_bytes
            (memory / 'memory.peak').unlink()
            expected_bytes.pop('memory.peak')
            assert driver.phase_diagnostic('unavailable', 100.) is None
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [r['elapsed_seconds'] for r in rows] == [2.5, 4., 5.]
        assert [r['phase'] for r in rows] == ['save8.begin', 'save8.end', 'unavailable']
        assert all(r['diagnostic'] == 'phase' and r['invocation_id'] == 'a'*32 for r in rows)
        assert rows[0]['cgroup']['values'] == values
        assert rows[1]['cgroup']['values']['memory.current'] == '512'
        assert rows[1]['cgroup']['values']['memory.peak'] == '8589934592'
        assert rows[1]['cgroup']['values']['memory.events'].startswith('max 1\n')
        assert rows[1]['cgroup']['values']['memory.stat'] == 'anon 64\nfile 448\nfile_dirty 400\nfile_writeback 16'
        assert 'memory.peak' in rows[2]['cgroup']['error'] and 'values' not in rows[2]['cgroup']
        assert len(flushed) == 3 and flushed[0] == output.getvalue().splitlines(keepends=True)[0]
        assert {p.name: p.read_bytes() for p in memory.iterdir()} == expected_bytes
        assert random.getstate() == rng and driver.diagnostic(replay) == original_replay
        assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
        # Exact SHA including empty EOF/tail, advising ONLY consumed 1MiB ranges.
        hashed = root / 'hashed.bin'
        multichunk = bytes(range(256)) * 8192
        for data in (b'', multichunk, multichunk + b'tail' * 9 + b'!'):
            hashed.write_bytes(data)
            with patch.object(driver.os, 'posix_fadvise', wraps=driver.os.posix_fadvise) as advice:
                assert driver.bound_file({}, hashed, hashlib.sha256(data).hexdigest()) == hashed
            ranges = [(offset, count, hint) for fd, offset, count, hint in
                      (call.args for call in advice.call_args_list)]
            assert ranges == [(offset, min(1024**2, len(data) - offset), driver.os.POSIX_FADV_DONTNEED)
                              for offset in range(0, len(data), 1024**2)], ranges
        dataset_root = root / 'images'
        dataset_root.mkdir()
        paths, image_rows = [], []
        for ordinal in range(64):
            image_path = dataset_root / f'{ordinal}.bin'
            image_path.write_bytes(bytes([ordinal]))
            paths.append(image_path)
            image_rows.append({'relative_path': image_path.name, 'image_sha256': sha(image_path)})
        verified, closed, seeds = [], [], []
        class FakeRGB:
            size = (256, 256)
            def __init__(self, ordinal):
                self.ordinal = ordinal
            def tobytes(self):
                return bytes([self.ordinal]) * 3
            def close(self):
                closed.append(self.ordinal)
        class ImageRows:
            def __init__(self, source_paths, labels, *, augment):
                assert source_paths == tuple(paths) and labels == tuple(range(64)) and augment is True
            def __getitem__(self, ordinal):
                assert verified[-1] == paths[ordinal]  # Actual SHA guard must precede decoding.
                return FakeRGB(ordinal), ordinal
        rng = SimpleNamespace(clone=lambda: 'RNG')
        fake_torch = SimpleNamespace(float32='FP32', equal=lambda a, b: (a == 'RNG' and b is rng),
            random=SimpleNamespace(get_rng_state=lambda: rng, fork_rng=lambda devices: nullcontext(),
                                   default_generator=SimpleNamespace(manual_seed=seeds.append)))
        augmentation_context = {'guards': {}, 'initialized': {'source_context': {'fit': {
            'dataset_root': str(dataset_root), 'targets': list(range(64)), 'rows': image_rows}},
            'pca': {'source_context': {'all_images': paths}}}}
        augmentation_state = {'target': list(range(64)), 'seed': 179041,
            'processor': lambda *, images, return_tensors: {'pixel_values': SimpleNamespace(
                dtype='FP32', shape=(len(images), 3, 256, 256))}}
        real_bound = driver.bound_file
        def verified_bound(guards, source_path, expected):
            result = real_bound(guards, source_path, expected)
            verified.append(result)
            return result
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(driver, 'bound_file', verified_bound):
            _, rgb_sha = driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                                                augmentation_state, tuple(range(64)), 17)
            original_pair_digest = hashlib.sha256()
            for ordinal in range(64):
                original_pair_digest.update(str((256, 256)).encode())
                original_pair_digest.update(bytes([ordinal]) * 3)
            assert rgb_sha == original_pair_digest.hexdigest() and closed == list(range(64))
            assert seeds == [179032 * 100000 + 17]  # Independent of schedule seed179041.
            paths[-1].write_bytes(b'changed')
            rejects(lambda: driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                augmentation_state, tuple(range(64)), 18), 'SHA256')
            image_rows[0]['relative_path'] = '../escape.bin'
            rejects(lambda: driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                augmentation_state, tuple(range(64)), 18), 'resolution/containment')
        own, qualified = root / 'own', root / 'qualified'
        own.mkdir(); qualified.mkdir()
        originals = {'deployed_code_rank.py': path.parent.parent / 'src/sfora/deployed_code_rank.py',
                     'reference_train_sop_siglip2_compact.py': path.with_name('train_sop_siglip2_compact.py'),
                     'reference_unicom_training.py': path.parent.parent / 'src/sfora/unicom_training.py'}
        for name in driver.FILES:
            shutil.copyfile(originals.get(name, path.parent / name), own / name)
        for name in driver.QUALIFIER_FILES:
            source = path.parent / name if name != 'joint_relational_compaction.py' else path.parent.parent / 'src/sfora' / name
            shutil.copyfile(source, qualified / name)
        code = {name: sha(own / name) for name in driver.FILES}
        qualifier_code = {name: sha(qualified / name) for name in driver.QUALIFIER_FILES}
        execution = write(own / 'execution.json', code)
        qualifier_execution = write(qualified / 'execution.json', qualifier_code)
        args = SimpleNamespace(execution_sha256=execution, authority=root / 'launch.json', authority_sha256=None,
                               arm='large', seed=179032, phase='mechanics', output=root / 'NEW')
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': execution, 'phase': 'mechanics',
                  'arm': 'large', 'seed': 179032, 'qualifier_root': str(qualified),
                  'qualifier_execution_sha256': qualifier_execution,
                  'qualifier_authority': {'path': str(root / 'qualified-launch.json'), 'sha256': 'a'*64},
                  'selected_cpu': {}, 'selected_mechanics': None, 'resource_policy': driver.policy('mechanics'),
                  'both_locks_held': True}
        args.authority_sha256 = write(args.authority, launch)
        with patch.object(driver, '__file__', str(own / path.name)):
            _, actual_code, _, actual_launch, q = driver.bootstrap(args)
            assert actual_code == code and actual_launch == launch and q.FILES == driver.QUALIFIER_FILES
            sys.modules.pop('_siglip2_pinned_adaptation_qualifier')
            for key, value in (('schema', 'wrong'), ('execution_sha256', '0'*64), ('arm', 'so400'),
                               ('seed', 179041), ('phase', 'train'), ('both_locks_held', False),
                               ('selected_mechanics', {}), ('resource_policy', driver.policy('train'))):
                args.authority_sha256 = write(args.authority, {**launch, key: value})
                rejects(lambda: driver.bootstrap(args), 'launch authority')
            args.authority_sha256 = write(args.authority, launch)
            for broken in ({**code, 'extra.py': '0'*64}, {n: h for n, h in code.items() if n != path.name}):
                args.execution_sha256 = write(own / 'execution.json', broken)
                rejects(lambda: driver.bootstrap(args), 'exactly declared')
            args.execution_sha256 = write(own / 'execution.json', code)
            raw = (own / path.name).read_bytes()
            (own / path.name).write_bytes(raw + b'\n# tamper\n')
            rejects(lambda: driver.bootstrap(args), 'SHA256')
            (own / path.name).write_bytes(raw)
            with patch.dict(sys.modules, {'torch.fake': SimpleNamespace()}):
                rejects(lambda: driver.authority(args), 'preceded admission')
            rejects(lambda: driver.descriptor_json({'path': str(root / 'missing'), 'sha256': 'a'*64}, {}), 'canonical file')
        for name, pin in driver.REFERENCES.items():
            namespace = {'Dataset': object}
            exec(driver.selected_ast(own / name, pin), namespace)
            assert set(namespace) == {'Dataset', '__builtins__', *pin['names']}
            for symbol in pin['names']:
                obj = namespace[symbol]
                if isinstance(obj, type):
                    obj = obj.__init__
                assert obj.__code__.co_filename == str(own / name)
            if 'member_bank_refresh_rows' in namespace:
                assert namespace['member_bank_refresh_rows']((3, 1, 3, 1, 2)) == ((1, 2, 3), (3, 4, 2))
                rejects(lambda: namespace['member_bank_refresh_rows'](()), 'refresh batch')
            else:
                assert [(s.start, s.stop) for s in namespace['_class_slices'](7, 3)] == [(0, 3), (3, 5), (5, 7)]
            rejects(lambda: driver.selected_ast(own / name, {**pin, 'ast': '0'*64}), 'AST differs')
            rejects(lambda: driver.selected_ast(own / name, {**pin, 'names': (*pin['names'], 'missing')}), 'selected reference')
        selected = root / 'selected.py'
        selected.write_text('raise RuntimeError("top-level executed")\ndef wanted():\n    return 7\n')
        node = ast.parse(selected.read_bytes()).body[1]
        pin = {'source': sha(selected), 'names': ('wanted',), 'ast': hashlib.sha256(
            ast.dump(ast.Module(body=[node], type_ignores=[]), include_attributes=False).encode()).hexdigest()}
        namespace = {}
        exec(driver.selected_ast(selected, pin), namespace)
        assert namespace['wanted']() == 7
        selected.write_text(selected.read_text() + '\ndef wanted():\n    return 8\n')
        rejects(lambda: driver.selected_ast(selected, {**pin, 'source': sha(selected)}), 'selected reference')
        rejects(lambda: driver.load_bare('unpinned', selected, pin['source']), 'SHA256')
        rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate')
        rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite')
        # Reuse the actual original terminal parser, not a permissive replacement.
        init = module(path.with_name('initialize_siglip2_substrate_fit.py'), 'terminal_under_test')
        unit, invocation = 'fixture-initialized', 'a'*32
        memory = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(8 * 1024**3), 'memory.current': '128', 'memory.peak': '256',
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'max 0\noom 0\noom_kill 0'}}
        record = {'wall_seconds': 1, 'process_peak_rss_kib': 1, 'cgroup_before': memory, 'cgroup_after': memory,
                  'invocation': {'invocation_id': invocation, 'optimize': 0}}
        log = root / 'original.log'
        logtext = (f'Running as unit: {unit}.service; invocation ID: {invocation}\n'
                   '\tExit status: 0\nFinished with result: success\n'
                   'Main processes terminated with: code=exited/status=0\n'
                   '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 111.519s\n'
                   '\tMaximum resident set size (kbytes): 2\nFINAL_CGROUP ' +
                   json.dumps({**memory, 'invocation_id': invocation}) + '\n')
        log.write_text(logtext)
        descriptor = {'receipt': {'path': str(root / 'proof.json'), 'sha256': 'b'*64},
                      'log': {'path': str(log), 'sha256': sha(log)}, 'unit': unit, 'invocation_id': invocation,
                      'service_seconds': 111.519, 'native_peak_rss_kib': 2, 'both_locks_held': True}
        assert init.admit_terminal(record, descriptor, 120, {})['invocation_id'] == invocation
        for badtext, marker in ((logtext.replace('status=0', 'status=1'), 'normal-exit'),
                                (logtext.replace('max 0', 'max 1'), 'memory failure'),
                                (logtext.replace('swap.peak": "0', 'swap.peak": "1'), 'memory/swap')):
            log.write_text(badtext)
            descriptor['log']['sha256'] = sha(log)
            rejects(lambda: init.admit_terminal(record, descriptor, 120, {}), marker)
        log.write_text(logtext)
        descriptor['log']['sha256'] = sha(log)
        rejects(lambda: init.admit_terminal(record, {**descriptor, 'service_seconds': 120.001}, 120, {}), 'duration/RSS')
        initialized = {'code': {'qualifier': 'fixed'}, 'launch': {
            'initializer_authority': {'path': '/original/pca.json', 'sha256': 'c'*64},
            'selected_initializer': {'original': True}}, 'pca_final_cgroup': memory,
            'record': {'source_binding': {'original': True}, 'ordered_input_sha256': 'd'*64,
                       'ordered_rgb_sha256': 'e'*64, 'artifact': {'path': '/original/initializers.npz', 'sha256': 'f'*64},
                       'arrays': {'original': True}},
            'source_context': {'proof': {'sample': {'original': True}, 'runtime': {'original': True},
                                       'numerical_flags': {'threads': 1}}},
            'packages': {}, 'pca': {'startup': {'invocation': {
                'python': '/original/python', 'python_sha256': '1'*64, 'python_version': 'original'}}}, 'init': init}
        launch['selected_cpu'] = descriptor
        cpu = {**record, 'schema': q.SCHEMA, 'phase': 'initialized-cpu', 'arm': 'large', 'width': 1024,
               'output_dim': 128, 'authority_sha256': launch['qualifier_authority']['sha256'],
               'execution_sha256': launch['qualifier_execution_sha256'], 'code': initialized['code'],
               'initializer_authority': initialized['launch']['initializer_authority'],
               'selected_initializer': initialized['launch']['selected_initializer'], 'pca_final_cgroup': memory,
               'source_binding': initialized['record']['source_binding'],
               'source_sample': initialized['source_context']['proof']['sample'],
               'ordered_input_sha256': 'd'*64, 'ordered_rgb_sha256': 'e'*64,
               'initializers': initialized['record']['artifact'], 'resource_policy': q.POLICY,
               'numerical_flags': {'threads': 1}, 'augmentation': q.AUGMENTATION,
               'state': {'counter': 0, 'seed': 179032, 'arrays': initialized['record']['arrays'],
                         'runtime': initialized['source_context']['proof']['runtime'],
                         'parameter_names': list(range(208))},
               'updates': 0, 'head_updates': 0, 'optimizer_state_entries': 0,
               'checkpoint': {'path': str(root / 'initialized.pt'), 'sha256': 'a'*64},
               'input_guards': {'/original/input': '2'*64}, 'origins': {'packages': {}, 'files': {}}}
        for key in ('pass', 'initializer_qualified', 'source_qualified', 'reload_exact', 'fresh_source',
                    'first_model_released_before_independent_clone', 'source_cpu_runtime_and_first2_exact',
                    'constructor_rng_preserved', 'exit_rehash_pass', 'optimizer_created',
                    'both_locks_held_in_parent_authority', 'terminal_exit_and_both_locks_require_parent_receipt'):
            cpu[key] = True
        for key in ('training_qualified', 'quality_qualified', 'quality_read', 'cuda_initialized', 'gradients_created',
                    'pca_rerun', 'teacher_state_reused', 'trained_state_reused'):
            cpu[key] = False
        cpu['invocation'] = {**cpu['invocation'], **initialized['pca']['startup']['invocation'],
            'cuda_visible_devices': '', 'argv': [str(qualified / 'qualify_siglip2_initialized_cpu.py'),
                '--execution-sha256', qualifier_execution, '--authority', launch['qualifier_authority']['path'],
                '--authority-sha256', launch['qualifier_authority']['sha256'], '--arm', 'large', '--output', str(root)]}
        context = {'qualifier': q, 'launch': launch, 'initialized': initialized, 'args': args, 'guards': {},
                   'initialized_prereq_guards': {'/original/input': '2'*64}}
        descriptor['receipt']['sha256'] = write(root / 'proof.json', cpu)
        mechanics_root = root / 'mechanics'
        mechanics_root.mkdir()
        mechanics_descriptor = {**descriptor, 'receipt': {'path': str(mechanics_root / 'receipt.json'), 'sha256': None}}
        train_launch = {**launch, 'phase': 'train', 'seed': 179041, 'selected_mechanics': mechanics_descriptor,
                        'resource_policy': driver.policy('train')}
        original_launch_path = root / 'mechanics-launch.json'
        original_launch_sha = write(original_launch_path, launch)
        mechanics = {**record, 'schema': driver.SCHEMA, 'phase': 'mechanics', 'arm': 'large', 'seed': 179032,
            'execution_sha256': execution, 'code': code, 'selected_cpu': descriptor,
            'reference_pins': json.loads(json.dumps(driver.REFERENCES)), 'rank_helper_sha256': driver.RANK_SHA256,
            'qualifier_authority': launch['qualifier_authority'], 'resource_policy': driver.policy('mechanics'),
            'augmentation': driver.AUGMENTATION, 'pass': True, 'quality_read': False, 'completed_step': 17,
            'training_state_discarded': True, 'native17_equals_serialized8_plus9_exact': True,
            'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
            'first_references_released_before_reload': True, 'exit_rehash_pass': True,
            'peak_cuda_allocated_bytes': 1024,
            'steps': [{'step': step, 'seconds': 1, 'loss': float(step)} for step in range(1, 18)],
            'resumed_steps': [{'step': step, 'seconds': 2, 'loss': float(step)} for step in range(9, 18)],
            'authority_sha256': original_launch_sha,
            'invocation': {**cpu['invocation'], 'cuda_visible_devices': '0', 'argv': [str(own / path.name),
                '--execution-sha256', execution, '--authority', str(original_launch_path),
                '--authority-sha256', original_launch_sha, '--phase', 'mechanics', '--arm', 'large',
                '--seed', '179032', '--output', str(mechanics_root)]}}
        mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', mechanics)
        context.update(launch=train_launch, root=own, code=code, cpu=cpu, guards={})
        # The same seed179032 mechanics proof admits BOTH separately CPU-pinned schedules.
        for seed in driver.SEEDS:
            args.seed = seed
            assert driver.admit_mechanics(context)[0] == mechanics
        for key, value in (('arm', 'so400'), ('seed', 179041), ('training_state_discarded', False),
                           ('native17_equals_serialized8_plus9_exact', False),
                           ('strict_independent_whole_head_buffers_raw_packed_reload_exact', False),
                           ('peak_cuda_allocated_bytes', 10_000_000_000), ('resumed_steps', [])):
            context['guards'] = {}
            mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', {**mechanics, key: value})
            rejects(lambda: driver.admit_mechanics(context), 'mechanics proof')
        context['guards'] = {}
        broken = copy.deepcopy(mechanics)
        broken['resumed_steps'][0]['loss'] += 1
        mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', broken)
        rejects(lambda: driver.admit_mechanics(context), 'mechanics proof')
        assert driver.admit_cpu(context)[0] == cpu
        for key, value, marker in (('arm', 'so400', 'CPU binding'), ('pass', False, 'CPU qualification'),
                                   ('updates', 1, 'CPU qualification'), ('training_qualified', True, 'CPU qualification'),
                                   ('quality_read', True, 'CPU qualification'), ('input_guards', {}, 'input guards'),
                                   ('state', {**cpu['state'], 'parameter_names': []}, 'CPU qualification')):
            descriptor['receipt']['sha256'] = write(root / 'proof.json', {**cpu, key: value})
            context['guards'] = {}
            rejects(lambda: driver.admit_cpu(context), marker)
        descriptor['receipt']['sha256'] = write(root / 'proof.json', cpu)
    assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
    print('PASS stdlib authority/reference/math/schedule/full-state/JSON/RGB/source-byte/streamed-SHA/checkpoint-write/ZIP/phase-log checks; native unrun')


if __name__ == '__main__':
    main()
