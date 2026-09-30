#!/usr/bin/env python3
"""Stdlib-only full2000 source/receipt/proof negatives; never load a model."""
if not __debug__:
    raise SystemExit('Checks require Python assertions; optimized mode is forbidden')

import argparse
import ast
import copy
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import qualify_full_valid_anchor_serving as serving
import benchmark_full_valid_anchor_serving as benchmark

DEFAULT_EVIDENCE = Path('/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/full-valid-anchor-2000-v1')


def rejects(call):
    try:
        call()
    except (AssertionError, FileNotFoundError, KeyError, ValueError):
        return
    raise AssertionError('altered authority accepted')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return serving.sha(path)


def source_checks(root, old):
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        original, current = base / 'original', base / 'current'
        original.mkdir(); current.mkdir()
        fixture = b'# source fixture\n'
        sources = dict.fromkeys(old)
        for directory in (original, current):
            for name in sources:
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(fixture)
        sources = {name: serving.sha(current / name) for name in sources}
        manifest = 'full-valid-anchor-checkpoint-execution.json'
        previous = write(original / manifest, sources)
        write(current / manifest, sources)
        for name in serving.ADDITIONS:
            (current / name).write_bytes(fixture)
        code = {**sources, **{name: serving.sha(current / name) for name in serving.ADDITIONS}}
        execution = write(current / serving.MANIFEST, code)
        authority = {'original_root': str(original), 'original_execution_sha256': previous, 'execution_sha256': execution}
        with patch.multiple(serving, ORIGINAL_ROOT=original, ORIGINAL_EXECUTION=previous):
            assert serving.source_authority(current, authority) == (code, sources)
            for directory, name in ((original, next(iter(sources))), (current, next(iter(sources))),
                                    *((current, name) for name in sorted(serving.ADDITIONS))):
                path = directory / name
                path.write_bytes(fixture + b'# changed\n')
                rejects(lambda: serving.source_authority(current, authority))
                path.write_bytes(fixture)
            for invalid in (sources, {**code, 'extra.py': '0' * 64},
                            {**code, next(iter(sources)): '0' * 64}, {**code, '../escape.py': '0' * 64}):
                authority['execution_sha256'] = write(current / serving.MANIFEST, invalid)
                rejects(lambda: serving.source_authority(current, authority))
            authority['execution_sha256'] = write(current / serving.MANIFEST, code)
            assert serving.source_authority(current, authority) == (code, sources)
            assert serving.sha(original / manifest) == previous
            for field in ('original_root', 'original_execution_sha256', 'execution_sha256'):
                changed = {**authority, field: str(base / 'wrong') if field.endswith('root') else '0' * 64}
                rejects(lambda: serving.source_authority(current, changed))


def receipt_checks(evidence, old):
    receipts = {}
    for key, name in {'export_receipt': 'qualification/export-receipt.json', 'training_receipt': '2000/receipt.json',
                      'accepted_cpu_proof': 'qualification/updated-cpu-proof.json', 'official_receipt': 'official/receipt.json'}.items():
        path = evidence / name
        assert serving.sha(path) == serving.PINNED[key], name
        receipts[key] = json.loads(path.read_text())
    assert receipts['accepted_cpu_proof']['code'] == old
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        paths = {}
        for name, value in receipts.items():
            paths[name] = base / (name + '.json')
            write(paths[name], value)
        paths['checkpoint'] = base / 'native.pt'
        paths['export_receipt'] = base / 'receipt.json'
        write(paths['export_receipt'], receipts['export_receipt'])
        protocol = {'pass': True, 'train_query_gallery_ids_disjoint': True, 'prior_official_benchmark_exposure': True,
                    'partition_sha256': 'c' * 64,
                    'protocol': {role: [{'relative_path': role + '/image.jpg', 'image_sha256': 'd' * 64, 'product': 'x'}] for role in serving.COUNTS}}
        paths['protocol'] = base / 'protocol.json'
        write(paths['protocol'], protocol)
        for name in ('accepted_cpu_log', 'export_log', 'official_log'):
            paths[name] = base / (name + '.log')
            paths[name].write_text('Finished with result: success\ncode=exited/status=0\nMemory swap peak: 0B\n')
        expected_sha = {}
        arrays = {}
        for role in serving.COUNTS:
            for field in ('codes', 'inverse'):
                name = role + '_' + field
                path = base / (role + '.' + field + '.npy')
                digest = receipts['official_receipt'][name + '_sha256']
                arrays[name] = {'path': str(path), 'sha256': digest}
                expected_sha[str(path)] = digest
        cpu = receipts['accepted_cpu_proof']
        expected_sha.update(cpu['environment']['native_files'])
        authority = {'environment': cpu['environment'], 'partition_sha256': 'c' * 64,
                     'ordered_protocol_sha256': {role: serving.object_sha(protocol['protocol'][role]) for role in serving.COUNTS},
                     'arrays': arrays, 'artifacts': {name: {'original_path': str(paths[name]), 'sha256': serving.sha(paths[name])}
                                                   for name in ('accepted_cpu_log', 'export_log', 'official_log')}}
        real_sha = serving.sha
        def fake_sha(path):
            return expected_sha[str(path)] if str(path) in expected_sha else real_sha(path)
        # The real pinned receipts drive semantic negatives; only large remote
        # artifact reads are stubbed, and artifact byte checks are tested below.
        with patch.multiple(serving, CHECKPOINT=paths['checkpoint'], OFFICIAL=paths['official_receipt'], COUNTS={'query': 1, 'gallery': 1}), \
             patch.object(serving, 'artifact', lambda _, name: paths.get(name, base / name)), patch.object(serving, 'sha', fake_sha):
            official = copy.deepcopy(receipts['official_receipt'])
            official['query_images'] = official['gallery_images'] = 1
            write(paths['official_receipt'], official)
            serving.receipt_authority(authority, old)
            for key, field, altered in (
                ('export_receipt', 'completed_step', 1000), ('training_receipt', 'completed_step', 1000),
                ('training_receipt', 'arm', 'half'), ('training_receipt', 'source_checkpoint_sha256', '0' * 64),
                ('accepted_cpu_proof', 'updated_head_sha256', '0' * 64),
                ('accepted_cpu_proof', 'updated_f16_whole_sha256', '0' * 64),
                ('accepted_cpu_proof', 'frozen_prefix_matches_original', False),
                ('accepted_cpu_proof', 'environment', {}), ('official_receipt', 'batch', 1),
                ('official_receipt', 'checkpoint_sha256', '0' * 64), ('official_receipt', 'code', {}),
                ('official_receipt', 'query_codes_sha256', '0' * 64)):
                before = paths[key].read_bytes()
                value = json.loads(before)
                value[field] = altered
                write(paths[key], value)
                rejects(lambda: serving.receipt_authority(authority, old))
                paths[key].write_bytes(before)
            wrong = copy.deepcopy(authority)
            wrong['arrays']['query_codes'] = authority['arrays']['gallery_codes']
            rejects(lambda: serving.receipt_authority(wrong, old))
            wrong = copy.deepcopy(authority)
            wrong['ordered_protocol_sha256']['query'] = '0' * 64
            rejects(lambda: serving.receipt_authority(wrong, old))
            serving.receipt_authority(authority, old)


def proof_checks():
    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        proof, log = base / 'proof.json', base / 'proof.log'
        log.write_text('Finished with result: success\ncode=exited/status=0\nMemory swap peak: 0B\n')
        log_sha = serving.sha(log)
        binding = {'source_code': {'qualify_full_valid_anchor_serving.py': 'a' * 64}, 'environment': {}}
        model = {'whole_sha256': serving.F32, 'head_sha256': serving.HEAD, 'buffers_sha256': 'b' * 64, 'environment': {},
                 'runtime': {'config': {'hidden_size': 1024, 'num_hidden_layers': 24, 'image_size': 256},
                             'checkpointing': False, 'attention_implementation': 'sdpa'}}
        value = {'schema': 'full-valid-anchor-serving-qualification-v1', 'phase': 'cpu', 'bindings': binding,
                 'pass': True, 'python_assertions_enabled': True, 'source_head_buffers_processor_runtime_rng_preserved': True,
                 'loaded_entrypoint_code_verified': True,
                 'official_quality_scored': False, 'decoded_previously_observed_official_sentinels': False,
                 'optimizer_updates': 0, 'quality_read': False, 'p99_certified': False, 'claim_eligible': False,
                 'paired_public_speed_win': False, 'strict400_f32_f16_head_fit_raw_normalized_packed_exact': True,
                 'cuda_initialized': False, 'resources': {'wall_cap_seconds': 120, 'wall_seconds': 1,
                    'max_rss_bytes': 1, 'cgroup_memory_peak_bytes': 1, 'memory_max_bytes': serving.HOST_CAP,
                    'memory_swap_max_bytes': 0, 'swap_bytes': 0, 'swaps': 0, 'admission_locks': []},
                 'models': {'native_f32': model, 'independent_f32': model, 'cast_f16': {**model, 'whole_sha256': serving.F16}},
                 'invocation': {'optimize': 0, 'driver_sha256': 'a' * 64}}
        digest = write(proof, value)
        assert serving.proof_authority(proof, digest, log, log_sha, 'cpu', binding) == value
        rejects(lambda: serving.proof_authority(base / 'absent', digest, log, log_sha, 'cpu', binding))
        rejects(lambda: serving.proof_authority(proof, '0' * 64, log, log_sha, 'cpu', binding))
        rejects(lambda: serving.proof_authority(proof, digest, log, '0' * 64, 'cpu', binding))
        rejects(lambda: serving.proof_authority(proof, digest, log, log_sha, 'gpu', binding))
        for field, altered in (('bindings', {}), ('pass', False), ('python_assertions_enabled', False),
                               ('strict400_f32_f16_head_fit_raw_normalized_packed_exact', False), ('cuda_initialized', True),
                               ('source_head_buffers_processor_runtime_rng_preserved', False), ('quality_read', True),
                               ('official_quality_scored', True), ('p99_certified', True), ('models', {})):
            changed = {**value, field: altered}
            digest = write(proof, changed)
            rejects(lambda: serving.proof_authority(proof, digest, log, log_sha, 'cpu', binding))
        for field, altered in (('wall_seconds', 121), ('swap_bytes', 1), ('memory_max_bytes', serving.HOST_CAP + 1),
                               ('max_rss_bytes', serving.HOST_CAP), ('cgroup_memory_peak_bytes', serving.HOST_CAP)):
            changed = {**value, 'resources': {**value['resources'], field: altered}}
            digest = write(proof, changed)
            rejects(lambda: serving.proof_authority(proof, digest, log, log_sha, 'cpu', binding))
        for field, altered in (('head_sha256', '0' * 64), ('whole_sha256', '0' * 64), ('environment', {'changed': True})):
            changed = copy.deepcopy(value)
            changed['models']['native_f32'][field] = altered
            digest = write(proof, changed)
            rejects(lambda: serving.proof_authority(proof, digest, log, log_sha, 'cpu', binding))
        for name in ('native_library', 'compiler'):
            file = base / name
            file.write_bytes(b'native fixture')
            authority = {'artifacts': {name: {'path': str(file), 'sha256': serving.sha(file)}}}
            with patch.dict(serving.PINNED, {name: serving.sha(file)}):
                assert serving.artifact(authority, name) == file
                file.write_bytes(b'changed native')
                rejects(lambda: serving.artifact(authority, name))
        lock = base / 'admission.lock'
        with lock.open('w') as stream:
            rejects(lambda: serving.lock_snapshot([str(lock)]))
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert serving.lock_snapshot([str(lock)]) == [{'path': str(lock), 'owner_pid': os.getpid()}]
            fcntl.flock(stream, fcntl.LOCK_UN)
            rejects(lambda: serving.lock_snapshot([str(lock)]))
        serving.new_output(base / 'new.json')
        rejects(lambda: serving.new_output(proof))
        link = base / 'dangling.json'
        link.symlink_to(base / 'absent')
        rejects(lambda: serving.new_output(link))


def source_contracts(root):
    code = {name: serving.sha(root / name) for name in serving.ADDITIONS}
    serving.loaded_code_guard(root, code)
    original = benchmark.sequence
    try:
        benchmark.sequence = lambda size, calls=100: []
        rejects(lambda: serving.loaded_code_guard(root, code))
    finally:
        benchmark.sequence = original
    for name in serving.ADDITIONS:
        tree = ast.parse((root / name).read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module]
                assert all(n.split('.')[0] in sys.stdlib_module_names or n in {Path(p).stem for p in serving.ADDITIONS} for n in names)
        assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                       and node.func.attr in ('register_forward_hook', 'register_forward_pre_hook') for node in ast.walk(tree))
    assert benchmark.sequence(1)[-1] == [99] and benchmark.sequence(32)[-1] == list(range(3168, 3200))
    assert len(benchmark.sequence(32, 10)) == 10 and benchmark.CALLS == 100 and benchmark.WARMUPS == 10
    rejects(lambda: benchmark.sequence(2))
    assert serving.sentinel_groups()[-1] == {'role': 'gallery', 'indices': list(range(12608, 12612))}
    assert serving.sentinel_groups()[1] == {'role': 'query', 'indices': list(range(14208, 14218))}
    for name in serving.ADDITIONS:
        command = [sys.executable, '-B', '-S', str(root / name), '--help']
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        for flag in ('-O', '-OO'):
            result = subprocess.run(command[:3] + [flag] + command[3:], capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr and 'ModuleNotFoundError' not in result.stderr
        result = subprocess.run(command, env={**os.environ, 'PYTHONOPTIMIZE': '1'}, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'PIL' not in sys.modules


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--evidence-root', type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    old = json.loads((args.evidence_root / 'qualification/full-valid-anchor-checkpoint-execution.json').read_text())
    assert len(old) == 102
    assert serving.sha(args.evidence_root / 'qualification/full-valid-anchor-checkpoint-execution.json') == serving.ORIGINAL_EXECUTION
    source_checks(root, old)
    receipt_checks(args.evidence_root, old)
    proof_checks()
    source_contracts(root)
    print('PASS stdlib full2000 real receipts,102+3 source, mode/head/prefix/role/order/native/proof/log/output negatives; help/optimized rejection; no Torch')


if __name__ == '__main__':
    main()
