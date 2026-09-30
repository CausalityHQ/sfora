#!/usr/bin/env python3
"""Stdlib admission checks only; these fixtures never qualify a native model."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import hashlib
import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
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
    except (ValueError, FileExistsError, KeyError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid authority accepted: ' + message)


def authority_fixture(base, driver):
    """Reuse frozen stdlib source fixture; no tensor/processor/quality execution."""
    own, source_root = base / 'export', base / 'source'
    own.mkdir()
    source_root.mkdir()
    repo = Path(__file__).parent
    for root, names in ((own, driver.FILES), (source_root, driver.SOURCE_FILES)):
        for name in names:
            shutil.copyfile(repo / name, root / name)
        write(root / 'execution.json', {name: sha(root / name) for name in names})
    source, source_code = driver.load_source(source_root, sha(source_root / 'execution.json'))
    extract, _ = source.bootstrap(source_root, sha(source_root / 'execution.json'))
    spec = importlib.util.spec_from_file_location('fit_source_test_fixture', source_root / 'test_siglip2_substrate_cpu.py')
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    source_args, sources, fit, inventory, provenance = fixture.fixture(source_root, extract)
    # The original fixture has two witness files. All exporter rows must exist.
    for index in range(2, 13283):
        path = source_root / f'dataset/Img/img/{index}.jpg'
        path.write_bytes(b'FIT stdlib bytes ' + str(index).encode())
        fit['rows'][index]['image_sha256'] = sha(path)
    receipt = Path(fit['original_receipt']['path'])
    fit['original_receipt']['sha256'] = write(receipt, {'fit_manifest': fit['rows'], 'target_products': fit['targets']})
    source_args.fit_manifest_sha256 = write(source_root / 'fit.json', fit)
    original = {'schema': 'native256-source-cpu-launch-v1', 'code': source_code,
                'execution_sha256': source_args.execution_sha256, 'sources_sha256': source_args.sources_sha256,
                'fit_manifest_sha256': source_args.fit_manifest_sha256, 'resource_policy': source.POLICY}
    original_path = base / 'original-authority.json'
    original_sha = write(original_path, original)
    # Authenticity fixture uses the committed actual inventory, not a fake model.
    evidence = repo.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1'
    proof = json.loads((evidence / 'native256-source-cpu-large-v4.json').read_text())
    with patch.object(extract, 'read_header', return_value=(inventory, {}, b'')):
        context = source.authority(source_args)
    checkpoint_root = base / 'source-proof'
    checkpoint_root.mkdir()
    checkpoint = checkpoint_root / 'fresh_vision.pt'
    checkpoint.write_bytes(b'authenticated checkpoint METADATA; NEVER loaded')
    proof.update(selected_source=context['entry'], source_model=context['entry']['source_model'],
                 revision=context['entry']['revision'], code=source_code,
                 execution_sha256=source_args.execution_sha256, sources_sha256=source_args.sources_sha256,
                 fit_manifest_sha256=source_args.fit_manifest_sha256,
                 checkpoint={'path': str(checkpoint), 'sha256': sha(checkpoint)},
                 origins={'packages': {}, 'files': {}, 'modules': {}, 'native_files': []})
    proof['input_guards'] = {**context['guards'], str(checkpoint): sha(checkpoint)}
    for name in proof['runtime']['vision']:
        proof['runtime']['vision'][name]['sha256'] = context['mapping'][name]['sha256']
    for sample, path in zip(proof['sample']['images'], context['images']):
        sample.update(path=str(path), image_sha256=sha(path))
    proof['invocation']['python'] = str(Path(sys.executable).resolve())
    proof['invocation']['python_sha256'] = sha(Path(sys.executable).resolve())
    proof['invocation']['python_version'] = sys.version
    proof['invocation']['argv'] = [str(source_root / 'qualify_siglip2_substrate_cpu.py'),
        '--execution-sha256', original['execution_sha256'], '--sources', str(source_root / 'sources.json'),
        '--sources-sha256', original['sources_sha256'], '--fit-manifest', str(source_root / 'fit.json'),
        '--fit-manifest-sha256', original['fit_manifest_sha256'], '--arm', 'large', '--output', str(checkpoint_root)]
    proof_path = checkpoint_root / 'proof.json'
    proof_sha = write(proof_path, proof)
    unit = Path(proof['cgroup_after']['path']).name.removesuffix('.service')
    final = {**proof['cgroup_after'], 'invocation_id': proof['invocation']['invocation_id']}
    log_path = base / 'cpu.log'
    log_path.write_text(f'Running as unit: {unit}.service; invocation ID: {final["invocation_id"]}\n'
        '\tExit status: 0\nFinished with result: success\nMain processes terminated with: code=exited/status=0\n'
        '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 35.984s\n'
        '\tMaximum resident set size (kbytes): 3648632\nFINAL_CGROUP ' + json.dumps(final) + '\n')
    descriptor = {'proof': {'path': str(proof_path), 'sha256': proof_sha},
                  'log': {'path': str(log_path), 'sha256': sha(log_path)}, 'unit': unit,
                  'invocation_id': final['invocation_id'], 'service_seconds': 35.984,
                  'native_peak_rss_kib': 3648632, 'both_locks_held': True}
    launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': sha(own / 'execution.json'),
              'source_root': str(source_root), 'source_execution_sha256': source_args.execution_sha256,
              'source_cpu_authority': {'path': str(original_path), 'sha256': original_sha},
              'sources': {'path': str(source_root / 'sources.json'), 'sha256': source_args.sources_sha256},
              'fit_manifest': {'path': str(source_root / 'fit.json'), 'sha256': source_args.fit_manifest_sha256},
              'startup_policy': driver.STARTUP_POLICY, 'export_policy': driver.EXPORT_POLICY,
              'source_cpu': {'large': descriptor, 'so400': copy.deepcopy(descriptor)}}
    authority_path = base / 'launch.json'
    args = SimpleNamespace(execution_sha256=launch['execution_sha256'], authority=authority_path,
        authority_sha256=write(authority_path, launch), arm='large', output=base / 'NEW',
        startup=None, startup_sha256=None, check_startup_only=False)
    return args, launch, proof, source, extract, inventory, own, source_root


def authority_checks(base, driver):
    args, launch, proof, source, extract, inventory, own, source_root = authority_fixture(base, driver)
    def freeze(value):
        args.authority_sha256 = write(args.authority, value)
    def freeze_proof(value):
        altered = copy.deepcopy(launch)
        altered['source_cpu']['large']['proof']['sha256'] = write(Path(altered['source_cpu']['large']['proof']['path']), value)
        freeze(altered)
    with patch.object(driver, '__file__', str(own / 'export_siglip2_substrate_fit.py')), \
            patch.object(driver, 'SOURCE_ROOT', source_root), \
            patch.object(extract, 'read_header', return_value=(inventory, {}, b'')):
        context = driver.authority(args)
        assert len(context['all_images']) == 13283
        driver.rehash(context)
        startup = {'schema': driver.SCHEMA, 'phase': 'startup', 'pass': True,
                   'binding': driver.binding(context), 'native_imported': False, 'model_constructed': False,
                   'exported': False, 'exit_rehash_pass': True, 'packages': {},
                   'input_guards': context['guards'].copy(), 'resource_policy': driver.STARTUP_POLICY,
                   'wall_seconds': 2, 'invocation': proof['invocation'],
                   'cgroup_before': proof['cgroup_before'], 'cgroup_after': proof['cgroup_after']}
        assert hasattr(driver, 'admit_startup'), 'missing pinned startup admission'
        driver.admit_startup(context, startup)
        for field, value in (('phase', 'export'), ('pass', False), ('native_imported', True),
                             ('model_constructed', True), ('exported', True), ('input_guards', {}),
                             ('wall_seconds', 121), ('binding', {})):
            bad = copy.deepcopy(startup)
            bad[field] = value
            rejects(lambda: driver.admit_startup(context, bad), 'startup')
        # Real exclusive JSON writer and full stdlib startup path; only native
        # package discovery and Linux cgroup reading are replaced for this host.
        args.output = base / 'startup.json'
        with patch.object(source, 'package_origins', return_value={}), \
                patch.object(source, 'cgroup_memory', return_value=proof['cgroup_after']), \
                patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', INVOCATION_ID='a'*32):
            record = driver.startup(args)
        assert record['phase'] == 'startup' and record['native_imported'] is False
        assert json.loads(args.output.read_text()) == record
        rejects(lambda: driver.write_json(extract, args.output, {}), 'output already exists')
        startup_path = args.output
        args.output = base / 'NEW'
        args.startup, args.startup_sha256 = startup_path, sha(startup_path)
        bad = copy.deepcopy(record)
        bad['phase'] = 'export'
        args.startup_sha256 = write(startup_path, bad)
        rejects(lambda: driver.export(args), 'startup')
        assert not args.output.exists() and 'torch' not in sys.modules
        args.startup_sha256 = write(startup_path, record)
        data = startup_path.read_bytes()
        startup_path.write_bytes(data + b'\n')
        rejects(lambda: driver.export(args), 'authority SHA256')
        startup_path.write_bytes(data)
        for field, value, message in (
                ('source_root', str(own), 'original immutable'),
                ('execution_sha256', '0'*64, 'launch authority'),
                ('source_execution_sha256', '0'*64, 'SHA256'),
                ('export_policy', {**driver.EXPORT_POLICY, 'seconds': 301}, 'launch authority')):
            bad = copy.deepcopy(launch)
            bad[field] = value
            freeze(bad)
            rejects(lambda: driver.authority(args), message)
        freeze(launch)
        args.output = source_root / 'NEVER-APPEND'
        rejects(lambda: driver.authority(args), 'original immutable')
        args.output = base / 'NEW'
        for role, path in (('sources', own / 'execution.json'), ('fit_manifest', own / 'execution.json')):
            bad = copy.deepcopy(launch)
            bad[role]['path'] = str(path)
            freeze(bad)
            rejects(lambda: driver.authority(args), 'path role')
        freeze(launch)
        args.output.symlink_to(base / 'missing')
        rejects(lambda: driver.authority(args), 'output already exists')
        args.output.unlink()
        saved = Path(launch['source_cpu']['large']['proof']['path']).read_bytes()
        for mutate, message in (
                (lambda p: p.update(arm='so400'), 'source-only'),
                (lambda p: p.update(source_qualified=False), 'source-only'),
                (lambda p: p.update(updates=1), 'source-only'),
                (lambda p: p.update(selected_source={}), 'source binding'),
                (lambda p: p['sample']['raw'].update(shape=[2, 1152]), 'buffer/sample'),
                (lambda p: p['runtime']['buffers']['embeddings.position_ids'].update(persistent=True), 'buffer/sample'),
                (lambda p: p['runtime']['roles'][0].update(role='trainable'), 'roles'),
                (lambda p: p['invocation']['argv'].append('--changed'), 'argv'),
                (lambda p: p['cgroup_after']['values'].update({'memory.swap.peak': '1'}), 'caps')):
            bad = copy.deepcopy(proof)
            mutate(bad)
            freeze_proof(bad)
            rejects(lambda: driver.authority(args), message)
        Path(launch['source_cpu']['large']['proof']['path']).write_bytes(saved)
        freeze(launch)
        log_path = Path(launch['source_cpu']['large']['log']['path'])
        log = log_path.read_text()
        for old, new, message in (('status=0', 'status=1', 'normal-exit'),
                                  ('FINAL_CGROUP ', 'NO_FINAL ', 'footer'),
                                  ('"memory.peak": "5466943488"', '"memory.peak": "1"', 'caps')):
            assert old in log
            log_path.write_text(log.replace(old, new))
            bad = copy.deepcopy(launch)
            bad['source_cpu']['large']['log']['sha256'] = sha(log_path)
            freeze(bad)
            rejects(lambda: driver.authority(args), message)
        log_path.write_text(log)
        freeze(launch)
        # ALL image paths, not just the old first-two source witnesses.
        late = context['all_images'][-1]
        data = late.read_bytes()
        late.write_bytes(b'changed late row')
        rejects(lambda: driver.authority(args), 'SHA256')
        rejects(lambda: driver.rehash(context), 'SHA256')
        late.write_bytes(data)
        logical = source_root / 'dataset/Img'
        resolved = source_root / 'dataset/img'
        logical.rename(resolved)
        logical.symlink_to(resolved, target_is_directory=True)
        # Source proof canonical first2 paths are immutable; exercise resolution
        # separately to avoid inventing a newly qualified native CPU proof.
        linked = {**context, 'guards': {}, 'all_images': []}
        images = driver.all_fit_images(linked)
        assert images[-1] == resolved / 'img/13282.jpg'
        logical.unlink()
        logical.symlink_to(base, target_is_directory=True)
        rejects(lambda: driver.all_fit_images(linked), 'escaped dataset root')
        logical.unlink()
        resolved.rename(logical)
        driver.rehash(context)
        (source_root / 'qualify_siglip2_substrate_cpu.py').write_bytes(b'changed original source closure')
        rejects(lambda: driver.authority(args), 'file SHA256')
        rejects(lambda: driver.rehash(context), 'exit authority')


def main():
    path = Path(__file__).with_name('export_siglip2_substrate_fit.py')
    result = subprocess.run([sys.executable, '-B', '-S', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--authority-sha256' in result.stdout, result.stderr
    import export_siglip2_substrate_fit as driver
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        for name in driver.FILES:
            shutil.copyfile(path.parent / name, root / name)
        code = {name: sha(root / name) for name in driver.FILES}
        digest = write(root / 'execution.json', code)
        assert driver.bootstrap(root, digest) == code
        rejects(lambda: driver.bootstrap(root, '0'*64), 'execution SHA256')
        digest = write(root / 'execution.json', {**code, 'extra.py': '0'*64})
        rejects(lambda: driver.bootstrap(root, digest), 'exactly two')
        digest = write(root / 'execution.json', code)
        (root / path.name).write_bytes(b'tampered')
        rejects(lambda: driver.bootstrap(root, digest), 'file SHA256')
    with TemporaryDirectory() as directory:
        with patch.dict(sys.modules), patch.object(sys, 'path', list(sys.path)):
            authority_checks(Path(directory).resolve(), driver)
    for mode in ('-O', '-OO'):
        result = subprocess.run([sys.executable, '-B', '-S', mode, str(path), '--help'], capture_output=True, text=True)
        assert result.returncode != 0 and 'optimized mode' in result.stderr
    rejects(lambda: driver.strict_json('{"x":1,"x":2}'), 'duplicate')
    rejects(lambda: driver.strict_json('{"x":NaN}'), 'nonfinite')
    assert not any(package in sys.modules for package in ('torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision'))
    print('PASS: stdlib authority/closure/sourceCPU/log/cgroup/all-row/tamper/path/roles and -O/-OO/help; no native qualification')


if __name__ == '__main__':
    main()
