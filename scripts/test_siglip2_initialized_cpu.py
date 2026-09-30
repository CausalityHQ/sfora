#!/usr/bin/env python3
"""Stdlib admission regressions only; tensor/reload parity is the parent native gate."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return sha(path)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid input accepted: ' + message)


def main():
    path = Path(__file__).with_name('qualify_siglip2_initialized_cpu.py').resolve()
    assert path.is_file(), 'initialized qualifier implementation is missing'
    spec = importlib.util.spec_from_file_location('initialized_under_test', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # Compare the actual frozen sampler operations to original coverage.schedule,
    # without importing its trained-model dependencies or executing NumPy.
    def sampler_core(source):
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'schedule')
        return [ast.dump(node, include_attributes=False) for node in function.body
                if isinstance(node, (ast.Assign, ast.For))]
    assert sampler_core(path) == sampler_core(path.with_name('pe_large_coverage.py'))
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in ('step', 'backward', 'reset_peak_memory_stats')
    for flags, args, code, marker in (([], ['--help'], 0, '--authority-sha256'),
                                      ([], [], 2, 'required'),
                                      (['-O'], ['--help'], 1, 'optimized'),
                                      (['-OO'], ['--help'], 1, 'optimized')):
        result = subprocess.run([sys.executable, '-B', '-S', *flags, str(path), *args],
                                capture_output=True, text=True)
        assert result.returncode == code and marker in result.stdout + result.stderr
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        own, pca = root / 'own', root / 'pca'
        own.mkdir()
        pca.mkdir()
        for name in driver.FILES:
            source = path.parent / name if name != 'joint_relational_compaction.py' else path.parent.parent / 'src/sfora' / name
            shutil.copyfile(source, own / name)
        for name in driver.INITIALIZER_FILES:
            source = path.parent / name if name != 'representation_ceiling.py' else path.parent.parent / 'src/sfora' / name
            shutil.copyfile(source, pca / name)
        code = {name: sha(own / name) for name in driver.FILES}
        pca_code = {name: sha(pca / name) for name in driver.INITIALIZER_FILES}
        execution = write(own / 'execution.json', code)
        pca_execution = write(pca / 'execution.json', pca_code)
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': execution,
                  'packing_helper_sha256': code['joint_relational_compaction.py'],
                  'initializer_root': str(pca), 'initializer_execution_sha256': pca_execution,
                  'initializer_authority': {'path': str(root / 'pca-launch.json'), 'sha256': 'a'*64},
                  'selected_initializer': {}, 'resource_policy': driver.POLICY, 'both_locks_held': True}
        args = SimpleNamespace(execution_sha256=execution, authority=root / 'launch.json',
                               authority_sha256=write(root / 'launch.json', launch),
                               arm='large', output=root / 'NEW')
        with patch.object(driver, '__file__', str(own / path.name)), patch.dict(sys.modules):
            init, actual_code, guards, actual_launch = driver.bootstrap(args)
            assert actual_code == code and actual_launch == launch and 'torch' not in sys.modules
            assert init.POLICY == driver.POLICY
            # An NPZ without a successful original PCA receipt is inadmissible.
            missing = {'selected_initializer': {'receipt': {'path': str(root / 'missing-receipt.json'),
                                                            'sha256': 'a'*64}}}
            rejects(lambda: driver.admit_pca(init, {'guards': {}}, missing, args), 'canonical regular file')
            # Keep the original Decimal duration parser and whole-unit gates.
            unit, invocation = 'fixture-pca', 'a'*32
            values = {'memory.max': str(driver.POLICY['host_bytes']), 'memory.current': '128',
                      'memory.peak': '256', 'memory.swap.current': '0', 'memory.swap.peak': '0',
                      'memory.swap.max': '0', 'memory.events': 'max 0\noom 0\noom_kill 0'}
            memory = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': values}
            record = {'wall_seconds': 1, 'process_peak_rss_kib': 1,
                      'invocation': {'invocation_id': invocation, 'optimize': 0},
                      'cgroup_before': memory, 'cgroup_after': memory}
            log = root / 'original.log'
            text = (f'Running as unit: {unit}.service; invocation ID: {invocation}\n'
                    '\tExit status: 0\nFinished with result: success\n'
                    'Main processes terminated with: code=exited/status=0\n'
                    '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 3min 15.231s\n'
                    '\tMaximum resident set size (kbytes): 2\nFINAL_CGROUP ' +
                    json.dumps({**memory, 'invocation_id': invocation}) + '\n')
            log.write_text(text)
            descriptor = {'receipt': {'path': str(root / 'receipt.json'), 'sha256': 'b'*64},
                          'log': {'path': str(log), 'sha256': sha(log)}, 'unit': unit,
                          'invocation_id': invocation, 'service_seconds': 195.231,
                          'native_peak_rss_kib': 2, 'both_locks_held': True}
            assert init.admit_terminal(record, descriptor, 300, {}) == {**memory, 'invocation_id': invocation}
            rejects(lambda: init.admit_terminal(record, descriptor, 120, {}), 'duration/RSS')
            log.write_text(text.replace('status=0', 'status=1'))
            descriptor['log']['sha256'] = sha(log)
            rejects(lambda: init.admit_terminal(record, descriptor, 300, {}), 'normal-exit')
            log.write_text(text.replace('max 0', 'max 1'))
            descriptor['log']['sha256'] = sha(log)
            rejects(lambda: init.admit_terminal(record, descriptor, 300, {}), 'memory failure')
            for field, value in (('schema', 'bad'), ('execution_sha256', '0'*64),
                                 ('packing_helper_sha256', '0'*64), ('both_locks_held', False),
                                 ('resource_policy', {**driver.POLICY, 'seconds': 121})):
                args.authority_sha256 = write(args.authority, {**launch, field: value})
                rejects(lambda: driver.bootstrap(args), 'launch authority')
            args.authority_sha256 = write(args.authority, launch)
            for malformed in ({**code, 'extra.py': '0'*64},
                              {k: v for k, v in code.items() if k != path.name}):
                args.execution_sha256 = write(own / 'execution.json', malformed)
                rejects(lambda: driver.bootstrap(args), 'exactly declared')
            args.execution_sha256 = write(own / 'execution.json', code)
            content = (own / path.name).read_bytes()
            (own / path.name).write_bytes(content + b'\n# tamper\n')
            rejects(lambda: driver.bootstrap(args), 'SHA256')
            (own / path.name).write_bytes(content)
            with patch.dict(sys.modules, {'torch.fake': SimpleNamespace()}):
                rejects(lambda: driver.authority(args), 'precede authority')
            helper = root / 'helper.py'
            helper.write_text('value = 1\n')
            helper_sha = sha(helper)
            helper.write_text('raise RuntimeError("unpinned execution")\n')
            rejects(lambda: driver.load_bare('untrusted_helper', helper, helper_sha), 'SHA256')
            helper.write_text('value = 1\n')
            with patch.dict(sys.modules, {'shadow_helper': SimpleNamespace(__file__=str(helper))}):
                rejects(lambda: driver.load_bare('shadow_helper', helper, helper_sha), 'already loaded')
            rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate')
            rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite')
            # Complete synthetic initializer arrays, independent of NumPy/Torch.
            shapes = driver.array_shapes(1024)
            raw_arrays = {}
            facts = {}
            for name, shape in shapes.items():
                count = 1
                for n in shape:
                    count *= n
                values = struct.pack('<q', 0) * count if name == 'target' else struct.pack('<f', 0) * count
                raw_arrays[name] = values
                facts[name] = {'dtype': 'torch.int64' if name == 'target' else 'torch.float32',
                               'shape': shape, 'sha256': hashlib.sha256(values).hexdigest()}
            def artifact(overrides=None, extra=False):
                arrays = {**raw_arrays, **(overrides or {})}
                with zipfile.ZipFile(root / 'initializers.npz', 'w') as output:
                    for name, values in arrays.items():
                        header = repr({'descr': '<i8' if name == 'target' else '<f4',
                                       'fortran_order': False, 'shape': tuple(shapes[name])}).encode()
                        prefix = b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header
                        output.writestr(name + '.npy', prefix + values)
                    if extra:
                        output.writestr('extra.npy', b'bad')
                return root / 'initializers.npz'
            archive = artifact()
            driver.validate_arrays(init, archive, sha(archive), 1024, facts, [0]*13283, {})
            bad = copy.deepcopy(facts)
            bad['bank']['sha256'] = '0'*64
            rejects(lambda: driver.validate_arrays(init, archive, sha(archive), 1024, bad, [0]*13283, {}), 'array byte')
            rejects(lambda: driver.validate_arrays(init, archive, sha(archive), 1152, facts, [0]*13283, {}), 'array shape')
            rejects(lambda: driver.validate_arrays(init, archive, sha(archive), 1024, facts, [1]*13283, {}), 'FIT target')
            archive = artifact(extra=True)
            rejects(lambda: driver.validate_arrays(init, archive, sha(archive), 1024, facts, [0]*13283, {}), 'exact array')
            nan = struct.pack('<f', float('nan')) + raw_arrays['mean'][4:]
            archive = artifact({'mean': nan})
            bad = copy.deepcopy(facts)
            bad['mean']['sha256'] = hashlib.sha256(nan).hexdigest()
            rejects(lambda: driver.validate_arrays(init, archive, sha(archive), 1024, bad, [0]*13283, {}), 'nonfinite')
    assert not any(name.split('.')[0] in driver.NATIVE_PACKAGES or name == 'sfora' for name in sys.modules)
    print('PASS: stdlib help/-O/-OO, exact closures, launch/tamper/origin, missing/failed PCA/cgroup rejection, original seeded sampler and complete initializer shape/dtype/hash/target/nonfinite negatives; no native qualification')


if __name__ == '__main__':
    main()
