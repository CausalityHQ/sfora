#!/usr/bin/env python3
"""One runnable stdlib authority/tamper/origin check; no native qualification."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return sha(path)


def rejects(call, message):
    try:
        call()
    except (ValueError, FileExistsError, KeyError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid authority accepted: ' + message)


def main():
    path = Path(__file__).with_name('initialize_siglip2_substrate_fit.py')
    assert path.is_file(), 'initializer authority implementation is missing'
    spec = importlib.util.spec_from_file_location('initializer_under_test', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with TemporaryDirectory() as directory:
        base = Path(directory).resolve()
        for name in driver.FILES:
            if name == 'representation_ceiling.py':
                (base / name).write_text('pinned_helper = True\n')
            else:
                shutil.copyfile(path.parent / name, base / name)
        code = {name: sha(base / name) for name in driver.FILES}
        execution_sha = write(base / 'execution.json', code)
        assert driver.closure(base, execution_sha, driver.FILES) == code
        rejects(lambda: driver.closure(base, '0'*64, driver.FILES), 'SHA256')
        for changed in ({**code, 'hidden.py': '0'*64},
                        {k: v for k, v in code.items() if k != 'representation_ceiling.py'}):
            changed_sha = write(base / 'execution.json', changed)
            rejects(lambda: driver.closure(base, changed_sha, driver.FILES), 'exactly')
        write(base / 'execution.json', code)
        launch_path = base / 'launch.json'
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': execution_sha,
                  'pca_helper_sha256': code['representation_ceiling.py'], 'export_root': str(base / 'export'),
                  'export_execution_sha256': 'c'*64, 'export_authority': {}, 'selected_export': {},
                  'startup': {}, 'resource_policy': driver.POLICY, 'both_locks_held': True}
        args = SimpleNamespace(execution_sha256=execution_sha, authority=launch_path,
                               authority_sha256=write(launch_path, launch), arm='large', output=base / 'NEW')
        with patch.object(driver, '__file__', str(base / path.name)):
            for field, value in (('schema', 'wrong'), ('execution_sha256', '0'*64),
                                 ('pca_helper_sha256', '0'*64), ('both_locks_held', False),
                                 ('resource_policy', {**driver.POLICY, 'seconds': 121})):
                args.authority_sha256 = write(launch_path, {**launch, field: value})
                rejects(lambda: driver.authority(args), 'launch authority')
            args.authority_sha256 = write(launch_path, launch)
            launch_path.write_bytes(launch_path.read_bytes() + b'\n')
            rejects(lambda: driver.authority(args), 'SHA256')
        # Real full-size synthetic NPY, no NumPy/Torch or source features.
        cache = base / 'fit.npy'
        header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (13283, 1024)}).encode()
        header += b' ' * ((-10 - len(header) - 1) % 64) + b'\n'
        prefix = b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header
        row = struct.pack('<1024f', 1.0, *([0.0]*1023))
        with cache.open('wb') as stream:
            stream.write(prefix)
            for _ in range(13283):
                stream.write(row)
        assert driver.cache_facts(cache, sha(cache), 1024, {})['maximum_unit_norm_error'] == 0.0
        rejects(lambda: driver.cache_facts(cache, sha(cache), 1152, {}), 'shape/dtype')
        with cache.open('r+b') as stream:
            stream.seek(len(prefix))
            stream.write(struct.pack('<f', float('nan')))
        rejects(lambda: driver.cache_facts(cache, sha(cache), 1024, {}), 'nonfinite')
        with cache.open('r+b') as stream:
            stream.seek(len(prefix))
            stream.write(struct.pack('<f', 0.0))
        rejects(lambda: driver.cache_facts(cache, sha(cache), 1024, {}), 'unit norm')
        helper = base / 'representation_ceiling.py'
        saved = helper.read_bytes()
        helper.write_bytes(saved + b'raise RuntimeError("must not execute tampered helper")\n')
        rejects(lambda: driver.closure(base, execution_sha, driver.FILES), 'file SHA256')
        rejects(lambda: driver.load_bare('pinned_test_helper', helper, code[helper.name]), 'SHA256')
        helper.write_bytes(saved)
        with patch.dict(sys.modules):
            spoof = ModuleType('pinned_test_helper')
            spoof.__file__ = str(base / 'shadow.py')
            sys.modules['pinned_test_helper'] = spoof
            rejects(lambda: driver.load_bare('pinned_test_helper', helper, code[helper.name]), 'origin')
            del sys.modules['pinned_test_helper']
            loaded = driver.load_bare('pinned_test_helper', helper, code[helper.name])
            assert loaded.pinned_helper and Path(loaded.__spec__.origin) == helper
        rejects(lambda: driver.strict_json('{"x":1,"x":2}'), 'duplicate')
        rejects(lambda: driver.strict_json('{"x":NaN}'), 'nonfinite')
        # Real log/footer admission, without a model, package or Linux unit.
        unit, invocation = 'fixture-export', 'a'*32
        values = {'memory.max': str(8*1024**3), 'memory.current': '128', 'memory.peak': '256',
                  'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                  'memory.events': 'max 0\noom 0\noom_kill 0'}
        memory = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': values}
        record = {'wall_seconds': 1, 'process_peak_rss_kib': 1,
                  'invocation': {'invocation_id': invocation, 'optimize': 0},
                  'cgroup_before': memory, 'cgroup_after': memory}
        log = base / 'original.log'
        text = (f'Running as unit: {unit}.service; invocation ID: {invocation}\n'
                '\tExit status: 0\nFinished with result: success\n'
                'Main processes terminated with: code=exited/status=0\n'
                '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 2.0s\n'
                '\tMaximum resident set size (kbytes): 2\nFINAL_CGROUP ' +
                json.dumps({**memory, 'invocation_id': invocation}) + '\n')
        log.write_text(text)
        descriptor = {'receipt': {'path': str(base / 'receipt.json'), 'sha256': 'b'*64},
                      'log': {'path': str(log), 'sha256': sha(log)}, 'unit': unit,
                      'invocation_id': invocation, 'service_seconds': 2.0,
                      'native_peak_rss_kib': 2, 'both_locks_held': True}
        guards = {}
        assert driver.admit_terminal(record, descriptor, 300, guards)['values']['memory.peak'] == '256'
        native_duration = {**descriptor, 'service_seconds': 195.231}
        log.write_text(text.replace('Service runtime: 2.0s', 'Service runtime: 3min 15.231s'))
        native_duration['log'] = {'path': str(log), 'sha256': sha(log)}
        driver.admit_terminal(record, native_duration, 300, {})
        rejects(lambda: driver.admit_terminal(record, {**native_duration, 'service_seconds': 195.232}, 300, {}),
                'runtime')
        log.write_text(text)
        for old, new in (('status=0', 'status=1'), ('FINAL_CGROUP ', 'MISSING '),
                         ('"memory.swap.peak": "0"', '"memory.swap.peak": "1"')):
            log.write_text(text.replace(old, new))
            altered = copy.deepcopy(descriptor)
            altered['log']['sha256'] = sha(log)
            rejects(lambda: driver.admit_terminal(record, altered, 300, {}),
                    'normal-exit' if old == 'status=0' else 'footer' if old == 'FINAL_CGROUP ' else 'caps')
        log.write_text(text)
        altered = {**descriptor, 'both_locks_held': False}
        rejects(lambda: driver.admit_terminal(record, altered, 300, {}), 'locks')
    result = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--authority-sha256' in result.stdout, result.stderr
    for mode in ('-O', '-OO'):
        result = subprocess.run([sys.executable, '-B', '-S', mode, str(path), '--help'], capture_output=True, text=True)
        assert result.returncode != 0 and 'optimized mode' in result.stderr
    assert not any(name in sys.modules for name in ('torch', 'numpy', 'sfora', 'transformers'))
    print('PASS: stdlib authority/exact closure/SHA/tamper/bare helper origin/full-size synthetic NPY/original log/footer/caps/locks/help/-O/-OO; no native qualification')


if __name__ == '__main__':
    main()
