#!/usr/bin/env python3
"""Qualify the preserved full2000 public encoder; no quality or speed claim."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import resource
import signal
import subprocess
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

SCHEMA = 'full-valid-anchor-serving-authority-v1'
MANIFEST = 'full-valid-anchor-serving-execution.json'
ADDITIONS = {'qualify_full_valid_anchor_serving.py', 'benchmark_full_valid_anchor_serving.py',
             'test_full_valid_anchor_serving.py'}
ORIGINAL_ROOT = Path('/home/riomus/runs/sfora-full-valid-anchor-checkpoint-v1')
ORIGINAL_EXECUTION = '4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375'
CHECKPOINT = Path('/home/riomus/runs/sfora-full-valid-anchor-native2000-v1/native.pt')
OFFICIAL = Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1/receipt.json')
PINNED = {
    'checkpoint': 'cb1ae064e53d118472856cb418458effb9a322e344af57ab5a087ea66f33a9a0',
    'export_receipt': 'f0f65232bbc8e9e35c6400490dd4d477be0a70baec1db234d2a1d7b002487ad6',
    'training_receipt': 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e',
    'source_resume': 'e32dd813c456a0ed319d933b74ffbc9b42886cf0641cb673e4c36945b90c43fa',
    'accepted_cpu_proof': 'd420f6e6476715a23cef04741ff74135584bb266cb7addaeaf35a824e76e2174',
    'official_receipt': 'fde8508c057057bec0edc515e7c25c28b53686b34ec31c157dad227cdebbed84',
    'protocol': 'be7b27c68b5e2895063d4f24b2b9d4e396a77ed79b2426596893c293df2c20f2',
    'source_checkpoint': 'f5fbf0e8eb3492274b6febeb6fd06a4b0d8b0fce4c7170069f3fa0d7b0a8df4d',
    'native_library': '39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c',
    'compiler': 'df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae',
}
F32 = '80d33d75a28575e3637b2b0545cf8a456f776e70064a073d079c1dfc406b8965'
F16 = '03a01b279542d6817959e3fb5327873981bc2d98fa72a66bcdaca8366207a2e0'
HEAD = 'ec167f7cbcbc7d66edff9ef0a5fc3de84ddc809aa134d6a6257ccab54d7a42c0'
COUNTS = {'query': 14218, 'gallery': 12612}
HOST_CAP = 8 * 1024**3
CUDA_CAP = 10_000_000_000


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path, expected):
    assert isinstance(expected, str) and len(expected) == 64 and set(expected) <= set('0123456789abcdef')
    assert sha(path) == expected, 'artifact authority differs: ' + str(path)
    return json.loads(Path(path).read_text())


def new_output(path):
    assert path.is_absolute() and path.parent.resolve() == path.parent
    assert not path.exists() and not path.is_symlink(), 'output already exists'


def save(path, value):
    new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def source_authority(root, authority):
    original = Path(authority['original_root'])
    assert original == ORIGINAL_ROOT and original.resolve() == original
    assert root != original and root.resolve() == root
    assert authority['original_execution_sha256'] == ORIGINAL_EXECUTION
    old = read(original / 'full-valid-anchor-checkpoint-execution.json', ORIGINAL_EXECUTION)
    assert len(old) == 102 and not ADDITIONS & old.keys()
    assert sha(root / 'full-valid-anchor-checkpoint-execution.json') == ORIGINAL_EXECUTION
    code = read(root / MANIFEST, authority['execution_sha256'])
    assert len(code) == 105 and code.keys() == old.keys() | ADDITIONS
    assert all(code[n] == h for n, h in old.items()), 'original102 changed'
    for directory, values in ((original, old), (root, code)):
        for name, digest in values.items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            path = directory / relative
            assert path.resolve().is_relative_to(directory) and sha(path) == digest, 'source differs: ' + name
    return code, old


def artifact(authority, name):
    entry = authority['artifacts'][name]
    path = Path(entry['path'])
    assert path.is_absolute() and path.resolve() == path and path.is_file(), name
    assert sha(path) == entry['sha256'], 'artifact differs: ' + name
    if name in PINNED:
        assert entry['sha256'] == PINNED[name], 'wrong candidate artifact: ' + name
    return path


def log_authority(path, expected):
    assert sha(path) == expected
    value = Path(path).read_text()
    assert all(s in value for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    return value


def protocol_authority(protocol, authority):
    assert protocol['pass'] and protocol['train_query_gallery_ids_disjoint']
    assert protocol['prior_official_benchmark_exposure']
    assert protocol['partition_sha256'] == authority['partition_sha256']
    paths = {}
    for role, count in COUNTS.items():
        rows = protocol['protocol'][role]
        assert len(rows) == count and object_sha(rows) == authority['ordered_protocol_sha256'][role]
        paths[role] = []
        for row in rows:
            name = Path(row['relative_path'])
            assert not name.is_absolute() and '..' not in name.parts
            assert row['product'] and len(row['image_sha256']) == 64
            paths[role].append(str(name))
        assert len(set(paths[role])) == count
    assert not set(paths['query']) & set(paths['gallery'])


def receipt_authority(authority, old):
    paths = {name: artifact(authority, name) for name in (*PINNED, 'accepted_cpu_log', 'export_log', 'official_log')}
    assert paths['checkpoint'] == CHECKPOINT and paths['official_receipt'] == OFFICIAL
    assert paths['export_receipt'] == CHECKPOINT.parent / 'receipt.json'
    exported, terminal, cpu, official, protocol = (
        json.loads(paths[n].read_text()) for n in
        ('export_receipt', 'training_receipt', 'accepted_cpu_proof', 'official_receipt', 'protocol'))
    assert exported['pass'] and terminal['pass'] and cpu['pass']
    assert terminal['arm'] == 'full'
    for value in (exported, cpu, official):
        assert value['arm'] == 'full' and value['optimizer_updates'] == 0
    for value in (exported, cpu, official):
        assert value['execution_sha256'] == ORIGINAL_EXECUTION and value['checkpoint_sha256'] == PINNED['checkpoint']
        assert value['training_receipt_sha256'] == PINNED['training_receipt']
    assert exported['completed_step'] == terminal['completed_step'] == terminal['total_updates'] == 2000
    assert terminal['intervention'] == exported['intervention'] == 'full-valid-anchor-v1'
    assert terminal['seed'] == 179032 and terminal['execution_sha256'] == '0da63376f73c9bc55c4c5f6e8c0f3f80fab7efe211ef6234a72af2ce4ed3ac36'
    assert terminal['checkpoint_sha256'] == exported['source_resume_sha256'] == PINNED['source_resume']
    assert terminal['source_checkpoint_sha256'] == exported['source_checkpoint_sha256'] == PINNED['source_checkpoint']
    assert terminal['frozen_source_code_environment_rng_preserved'] and terminal['all_input_hashes_equal_archived_control']
    assert exported['strict400_terminal_vision_head_export'] and exported['recovered_rank_updates'] == 375
    assert not exported['quality_read'] and not terminal['quality_read'] and not cpu['quality_read']
    assert cpu['code'] == official['code'] == old and cpu['strict_updated_native_CPU_B2_whole_head_packed_exact']
    assert cpu['frozen_prefix_matches_original'] and cpu['changed_driver_rejected']
    assert cpu['updated_f32_whole_sha256'] == exported['updated_whole_sha256'] == F32
    assert cpu['updated_f16_whole_sha256'] == F16 and cpu['updated_head_sha256'] == exported['updated_head_sha256'] == HEAD
    assert cpu['environment'] == authority['environment'] and cpu['protocol_sha256'] == PINNED['protocol']
    assert official['cpu_authority_sha256'] == PINNED['accepted_cpu_proof']
    assert official['batch'] == 32 and official['query_images'] == COUNTS['query'] and official['gallery_images'] == COUNTS['gallery']
    assert official['native_all_query_top10_ordinal_score_bits_exact'] and official['source_state_rng_environment_code_library_preserved']
    assert official['independent_updated_native_original_preprocessor_first32_each_role_packed_exact']
    assert official['peak_cuda_allocated_bytes'] < CUDA_CAP and not official['claim_eligible']
    for role in COUNTS:
        for field in ('codes', 'inverse'):
            name = role + '_' + field
            entry = authority['arrays'][name]
            assert Path(entry['path']) == OFFICIAL.parent / (role + '.' + field + '.npy')
            assert sha(entry['path']) == entry['sha256'] == official[name + '_sha256']
    for file, digest in cpu['environment']['native_files'].items():
        assert sha(file) == digest
    for name in ('accepted_cpu_log', 'export_log', 'official_log'):
        entry = authority['artifacts'][name]
        log_authority(paths[name], entry['sha256'])
        assert sha(entry['original_path']) == entry['sha256']
    protocol_authority(protocol, authority)
    return paths, exported, cpu, protocol


def startup(root, path, expected):
    """Authority extends the parent's exact collected inventory with future105."""
    authority = read(path, expected)
    assert authority['schema'] == SCHEMA and root == Path(authority['prepared_root'])
    inputs = authority['inputs']
    names = {'accepted_cpu', 'accepted_cpu_log', 'checkpoint', 'export', 'export_log',
             'native_library', 'official', 'official_log', 'protocol', 'source_resume', 'training',
             'query.codes.npy', 'query.inverse.npy', 'gallery.codes.npy', 'gallery.inverse.npy'}
    assert inputs.keys() == names
    aliases = {'accepted_cpu_proof': 'accepted_cpu', 'export_receipt': 'export',
               'official_receipt': 'official', 'training_receipt': 'training'}
    authority['artifacts'] = {name: inputs[aliases.get(name, name)] for name in
        ('accepted_cpu_proof', 'accepted_cpu_log', 'checkpoint', 'export_receipt', 'export_log',
         'native_library', 'official_receipt', 'official_log', 'protocol', 'source_resume', 'training_receipt')}
    authority['artifacts'].update(compiler=authority['compiler'], source_checkpoint=authority['source_checkpoint'])
    authority['arrays'] = {role + '_' + field: inputs[role + '.' + field + '.npy']
                           for role in COUNTS for field in ('codes', 'inverse')}
    for name, digest in {'accepted_cpu_log': '0ff7ee03aa962cf5aa199735815860fa0bac1003ba856dc85283956413e835de',
                         'export_log': 'e7e3b003657b194d3f0ecb6271f6acf1f023f1734b0df4ff1f4102884924955f',
                         'official_log': 'b444b42ff2f71e82648d942d0b7518c722c842abd1b3dc6a99814f0783085b52'}.items():
        assert inputs[name]['sha256'] == digest
    code, old = source_authority(root, authority)
    cpu = read(inputs['accepted_cpu']['path'], PINNED['accepted_cpu_proof'])
    protocol = read(inputs['protocol']['path'], PINNED['protocol'])
    authority['environment'] = cpu['environment']
    authority['partition_sha256'] = protocol['partition_sha256']
    authority['ordered_protocol_sha256'] = {role: object_sha(protocol['protocol'][role]) for role in COUNTS}
    paths, exported, cpu, protocol = receipt_authority(authority, old)
    return authority, code, old, paths, exported, cpu, protocol


def loaded_code_guard(root, code):
    """Authenticate actual own function bytecode as well as loaded module paths."""
    for name in ADDITIONS:
        current = sys.modules.get('__main__')
        module = current if getattr(current, '__file__', None) and Path(current.__file__).resolve() == root / name else importlib.import_module(Path(name).stem)
        path = Path(module.__file__).resolve()
        assert path == root / name and sha(path) == code[name]
        compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True, optimize=0)
        expected = {c.co_name: c for c in compiled.co_consts if isinstance(c, types.CodeType)}
        for function_name, expected_code in expected.items():
            function = getattr(module, function_name)
            assert isinstance(function, types.FunctionType) and function.__module__ == module.__name__
            assert function.__code__ == expected_code and function.__code__.co_filename == str(path), 'loaded code differs: ' + name + ':' + function_name
    scripts = {Path(n).stem for n in code if '/' not in n and n.endswith('.py')}
    for name, module in list(sys.modules.items()):
        if name in scripts or name.split('.')[0] in {'sfora', 'core', 'einops', 'ftfy'}:
            file = getattr(module, '__file__', None)
            if file:
                path = Path(file).resolve()
                assert path.is_relative_to(root), 'loaded source outside closure: ' + name
                assert code.get(str(path.relative_to(root))) == sha(path), name


def runtime_startup(root, data):
    authority, code, old, paths, exported, accepted, protocol = data
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(0, str(root / 'isolated-deps'))
    import qualify_pe_full_valid_anchor_checkpoint as full
    loaded_code_guard(root, code)
    helpers = full.qualified.selected.helpers
    executing = full.pair.executing_authority

    def extended(r, historical):
        assert all(code[n] == h for n, h in historical.items())
        return executing(r, code)

    # Historical manifests keep their exact cardinalities; loaded modules bind105.
    with patch.object(full.qualified.selected, 'helpers', lambda r, _: helpers(r, code)), patch.object(full.pair, 'executing_authority', extended):
        control, source, prior, proof, run, terminal, actual_protocol, _, actual_old, _ = full.authority(
            root, ORIGINAL_EXECUTION, PINNED['training_receipt'])
    assert actual_old == old and actual_protocol == protocol
    assert run / 'resume.pt' == paths['source_resume'] and sha(run / 'receipt.json') == PINNED['training_receipt']
    assert full.driver.coverage.teacher.TEACHER == paths['source_checkpoint']
    assert source['environment'] == accepted['environment']
    assert sha(control.dataset_root / 'Eval/list_eval_partition.txt') == authority['partition_sha256']
    assert all(sha(control.large_snapshot / n) == h for n, h in full.pair.smoke.MODEL_HASHES.items())
    assert len(proof['arms']['full']['rows']) == 25882 and len(proof['arms']['full']['classes']) == 3997
    loaded_code_guard(root, code)
    return full, control, source, prior, proof


def lock_snapshot(paths):
    # The parent launcher may own admission locks outside this systemd cgroup.
    lines = [line.split() for line in Path('/proc/locks').read_text().splitlines()]
    result = []
    for file in paths:
        stat = Path(file).stat()
        inode = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
        owners = [int(line[4]) for line in lines if line[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and line[5] == inode]
        assert len(owners) == 1 and owners[0] > 0 and Path(f'/proc/{owners[0]}').is_dir(), 'admission lock not held'
        result.append({'path': file, 'owner_pid': owners[0]})
    return result


def resource_start(phase, authority):
    cap = 120 if phase == 'cpu' else 300
    started = time.perf_counter()

    def timeout(signum, frame):
        raise TimeoutError('serving wall cap exceeded')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cap)
    relative = next(line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    group = Path('/sys/fs/cgroup') / relative.lstrip('/')
    assert int((group / 'memory.max').read_text()) == HOST_CAP
    assert int((group / 'memory.swap.max').read_text()) == 0
    locks = authority['lock_paths'][phase]
    assert len(locks) == (0 if phase == 'cpu' else 2) and len(set(locks)) == len(locks)
    return started, cap, group, lock_snapshot(locks)


def usage(resource_state):
    started, cap, group, locks = resource_state
    assert lock_snapshot([item['path'] for item in locks]) == locks
    stats = resource.getrusage(resource.RUSAGE_SELF)
    swap = int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmSwap:'))) * 1024
    seconds = time.perf_counter() - started
    peak = int((group / 'memory.peak').read_text())
    assert int((group / 'memory.max').read_text()) == HOST_CAP
    assert int((group / 'memory.swap.max').read_text()) == 0
    assert seconds < cap and stats.ru_maxrss * 1024 < HOST_CAP and peak < HOST_CAP
    assert swap == stats.ru_nswap == 0 and int((group / 'memory.swap.current').read_text()) == 0
    swap_peak = group / 'memory.swap.peak'
    if swap_peak.exists():
        assert int(swap_peak.read_text()) == 0
    return {'wall_seconds': seconds, 'wall_cap_seconds': cap, 'max_rss_bytes': stats.ru_maxrss * 1024,
            'cgroup': str(group), 'cgroup_memory_peak_bytes': peak, 'memory_max_bytes': HOST_CAP,
            'memory_swap_max_bytes': 0, 'swap_bytes': swap, 'swaps': stats.ru_nswap, 'admission_locks': locks}


def proof_authority(path, digest, log, log_digest, phase, binding):
    value = read(path, digest)
    log_authority(log, log_digest)
    assert value['schema'] == 'full-valid-anchor-serving-qualification-v1' and value['phase'] == phase
    assert value['bindings'] == binding and value['pass'] is True and value['python_assertions_enabled'] is True
    assert value['source_head_buffers_processor_runtime_rng_preserved'] is True
    assert value['optimizer_updates'] == 0 and value['quality_read'] is False and value['p99_certified'] is False
    assert value['official_quality_scored'] is False
    assert value['decoded_previously_observed_official_sentinels'] == (phase == 'gpu')
    assert value['claim_eligible'] is False and value['paired_public_speed_win'] is False
    stats = value['resources']
    assert stats['wall_cap_seconds'] == (120 if phase == 'cpu' else 300)
    assert 0 < stats['wall_seconds'] < stats['wall_cap_seconds']
    assert stats['max_rss_bytes'] < HOST_CAP and stats['cgroup_memory_peak_bytes'] < HOST_CAP
    assert stats['memory_max_bytes'] == HOST_CAP and stats['memory_swap_max_bytes'] == stats['swap_bytes'] == stats['swaps'] == 0
    assert len(stats['admission_locks']) == (0 if phase == 'cpu' else 2)
    if phase == 'cpu':
        assert value['strict400_f32_f16_head_fit_raw_normalized_packed_exact'] is True
        assert value['cuda_initialized'] is False
        wholes = {'native_f32': F32, 'independent_f32': F32, 'cast_f16': F16}
    else:
        assert value['precision'] == 'fp16_native' and value['gallery_images'] == COUNTS['gallery']
        assert 0 < value['peak_cuda_allocated_bytes'] < CUDA_CAP
        assert value['public_native_packed_top10_ordinal_score_bits_exact'] is True
        assert value['B32_sentinels'] == sentinel_groups()
        assert value['B1_query_indices'] == list(range(32)) + [COUNTS['query'] - 1]
        assert len(value['B1_matches_saved_B32']) == 33 and all(type(x) is bool for x in value['B1_matches_saved_B32'])
        assert value['cpu_proof_sha256'] and value['cpu_log_sha256']
        proof_authority(Path(value['cpu_proof']), value['cpu_proof_sha256'],
                        Path(value['cpu_log']), value['cpu_log_sha256'], 'cpu', binding)
        assert isinstance(value['compiler_version'], str) and value['compiler_version']
        wholes = {'public_f16': F16, 'independent_f16': F16}
    assert value['models'].keys() == wholes.keys()
    for name, whole in wholes.items():
        model = value['models'][name]
        assert model['whole_sha256'] == whole and model['head_sha256'] == HEAD
        assert len(model['buffers_sha256']) == 64 and model['environment'] == binding['environment']
        config = model['runtime']['config']
        assert config['hidden_size'] == 1024 and config['num_hidden_layers'] == 24 and config['image_size'] == 256
        assert not model['runtime']['checkpointing'] and model['runtime']['attention_implementation'] == 'sdpa'
    assert value['invocation']['optimize'] == 0 and value['invocation']['driver_sha256'] == binding['source_code']['qualify_full_valid_anchor_serving.py']
    assert value['loaded_entrypoint_code_verified'] is True
    return value


def bindings(authority, digest, code):
    return {'authority_sha256': digest, 'execution_sha256': authority['execution_sha256'],
            'original_root': str(ORIGINAL_ROOT), 'original_execution_sha256': ORIGINAL_EXECUTION,
            'source_code': code, 'artifacts': authority['artifacts'], 'arrays': authority['arrays'],
            'ordered_protocol_sha256': authority['ordered_protocol_sha256'],
            'partition_sha256': authority['partition_sha256'], 'environment': authority['environment'],
            'f32_whole_sha256': F32, 'f16_whole_sha256': F16, 'head_sha256': HEAD,
            'precision': 'fp16_native', 'width': 128, 'query_images': COUNTS['query'], 'gallery_images': COUNTS['gallery']}


def invocation():
    executable = Path(sys.executable).resolve()
    return {'argv': sys.argv, 'python': str(executable), 'python_sha256': sha(executable),
            'version': sys.version, 'optimize': sys.flags.optimize, 'driver': str(Path(__file__).resolve()),
            'driver_sha256': sha(__file__), 'pid': os.getpid()}


def native_reload(full, control, independent=False):
    """Construct before mmap; copy strict400/head, then release the entire mapping."""
    import torch
    from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel
    if independent:
        model = SiglipVisionModel(AutoConfig.from_pretrained(control.large_snapshot, local_files_only=True).vision_config).float()
        processor = AutoImageProcessor.from_pretrained(control.large_snapshot / 'preprocessor_config.json', local_files_only=True, backend='torchvision')
    else:
        model, processor = full.pair.smoke.load_arm(control, 'large')
    head = torch.nn.Linear(1024, 128).float()
    disk = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
    assert set(disk) == {'vision', 'head'} and len(disk['vision']) == 400
    assert set(disk['head']) == {'weight', 'bias'} and disk['head']['weight'].shape == (128, 1024) and disk['head']['bias'].shape == (128,)
    assert all(torch.isfinite(v).all() for group in disk.values() for v in group.values())
    model.load_state_dict(disk['vision'], strict=True)
    head.load_state_dict(disk['head'], strict=True)
    del disk
    gc.collect()
    return model.requires_grad_(False).eval(), head.requires_grad_(False).eval(), processor


def model_facts(full, model, head, processor, whole, prefix=None):
    import torch
    qualified, pair = full.qualified, full.pair
    assert len(model.state_dict()) == 400 and head.weight.shape == (128, 1024)
    assert pair.smoke.digest(qualified.trained.base.whole_state(model)) == whole
    assert pair.smoke.digest(head.state_dict()) == HEAD
    assert all(not m.training and not m._forward_hooks and not m._forward_pre_hooks for top in (model, head) for m in top.modules())
    assert all(not p.requires_grad and p.grad is None and torch.isfinite(p).all() for top in (model, head) for p in top.parameters())
    assert all(p.dtype == (torch.float32 if top is head or whole == F32 else torch.float16) for top in (model, head) for p in top.parameters())
    assert all(b.device == next(top.parameters()).device for top in (model, head) for b in top.buffers())
    if prefix is not None:
        roots = ('embeddings.',) + tuple(f'encoder.layers.{i}.' for i in range(12))
        inventory = {'frozen': tuple(n for n, _ in model.named_parameters() if n.startswith(roots))}
        assert qualified.coverage.frozen_digest(model, inventory) == prefix
    return {'whole_sha256': whole, 'head_sha256': HEAD,
            'buffers_sha256': pair.smoke.digest({f'{i}.{n}': b for i, top in enumerate((model, head)) for n, b in top.named_buffers()}),
            'runtime': qualified.trained.base.runtime_identity(model),
            'environment': json.loads(json.dumps(qualified.trained.native.environment(model, processor)))}


def rng_fingerprint(full, gpu):
    import numpy as np
    import torch
    state = np.random.get_state()
    return full.driver.fingerprint({'cpu': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if gpu else [], 'python': random.getstate(),
        'numpy': (state[0], state[1].tolist(), *state[2:])})


def configure(full, authority, gpu, prior):
    import torch
    torch.set_num_threads(8)
    assert torch.cuda.is_available() == gpu
    if gpu:
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        assert os.environ.get('CUTILE_TILEIRAS_PATH') == authority['artifacts']['compiler']['path']
        compiler = artifact(authority, 'compiler')
        version = subprocess.run([str(compiler), '--version'], check=True, capture_output=True, text=True, timeout=10)
        compiler_version = version.stdout + version.stderr
    else:
        assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1') and not torch.cuda.is_initialized()
        compiler_version = None
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert full.qualified.teacher.qualified.numerical_flags() == prior['numerical_flags']
    return compiler_version


def public_encoder(full, control, reset_peak=True):
    import torch
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder
    # This is the sole peak reset, before any public or independent constructor.
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot,
            checkpoint=CHECKPOINT, expected_checkpoint_sha256=PINNED['checkpoint'],
            model_file_sha256=full.pair.smoke.MODEL_HASHES, precision='fp16_native', device=torch.device('cuda'))
    encoder.vision.requires_grad_(False)
    encoder.head.requires_grad_(False)
    assert encoder.precision == 'fp16_native' and encoder._batch1_graph is None
    return encoder


def packed_slice(packed, indices):
    from sfora.joint_relational_compaction import PackedInt8Embeddings
    return PackedInt8Embeddings(packed.codes[indices].contiguous(), packed.inverse_norms[indices].contiguous())


def load_wires(authority):
    import numpy as np
    import torch
    from sfora.joint_relational_compaction import PackedInt8Embeddings
    result = {}
    for role, count in COUNTS.items():
        arrays = [np.load(authority['arrays'][role + '_' + field]['path'], allow_pickle=False) for field in ('codes', 'inverse')]
        assert arrays[0].dtype == np.int8 and arrays[0].shape == (count, 128)
        assert arrays[1].dtype == np.float16 and arrays[1].shape == (count,)
        packed = PackedInt8Embeddings(*(torch.from_numpy(a) for a in arrays))
        norms = torch.linalg.vector_norm(packed.codes.float(), dim=1)
        assert packed.codes.min() >= -127 and (norms > 0).all()
        assert torch.equal(norms.reciprocal().half(), packed.inverse_norms)
        result[role] = packed
    return result


def cpu_reference(query, gallery):
    import torch
    assert query.codes.device.type == gallery.codes.device.type == 'cpu'
    scores = (query.codes.float() @ gallery.codes.float().T) * query.inverse_norms.float()[:, None] * gallery.inverse_norms.float()[None, :]
    order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
    return order.numpy(), scores.gather(1, order).numpy()


def same_results(actual, expected):
    assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
               for a, b in zip(actual, expected, strict=True)), 'top10 ordinals or float32 score bits differ'


def decode(paths):
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    return images


def image_paths(control, protocol, role, indices):
    rows = [protocol['protocol'][role][i] for i in indices]
    paths = [control.dataset_root / row['relative_path'] for row in rows]
    assert all(sha(p) == row['image_sha256'] for p, row in zip(paths, rows, strict=True))
    return paths


def native_reference(full, model, head, processor, images):
    import torch
    from sfora.joint_relational_compaction import pack_int8_unit_embeddings
    pixels = full.pair.pixels(processor, images, 'large').cuda().half()
    with torch.inference_mode(), torch.amp.autocast('cuda', enabled=False):
        pooled = model(pixel_values=pixels).pooler_output
        raw = full.pair.smoke.compact_head_features(pooled, head)
        normalized = torch.nn.functional.normalize(raw, dim=1).cpu()
    return pack_int8_unit_embeddings(normalized)


def sentinel_groups():
    return [{'role': role, 'indices': list(range(start, min(start + 32, count)))}
            for role, count in COUNTS.items() for start in (0, count // 32 * 32)]


def cpu_phase(full, control, proof, authority):
    import torch
    from sfora.joint_relational_compaction import pack_int8_unit_embeddings
    before = rng_fingerprint(full, False)
    with torch.random.fork_rng(devices=[]):
        model, head, processor = native_reload(full, control)
        other, other_head, other_processor = native_reload(full, control, independent=True)
    a = model_facts(full, model, head, processor, F32, proof['frozen_prefix_sha256'])
    b = model_facts(full, other, other_head, other_processor, F32, proof['frozen_prefix_sha256'])
    full.qualified.same_runtime(model, other)
    assert a['environment'] == b['environment'] == authority['environment'] and a['buffers_sha256'] == b['buffers_sha256']
    images, rgb = full.pair.augmented_images(control.dataset_root, proof['arms']['full']['rows'], (0, 1), None)
    pixels = full.pair.pixels(processor, images, 'large')
    assert torch.equal(pixels, full.pair.pixels(other_processor, images, 'large'))
    with torch.inference_mode():
        pooled = [m(pixel_values=pixels).pooler_output for m in (model, other)]
        raw = [full.pair.smoke.compact_head_features(p, h) for p, h in zip(pooled, (head, other_head), strict=True)]
        normalized = [torch.nn.functional.normalize(v, dim=1) for v in raw]
        assert torch.equal(*pooled) and torch.equal(*raw) and torch.equal(*normalized)
        full.qualified.selected.fp16.same(*(pack_int8_unit_embeddings(v) for v in normalized))
    assert model_facts(full, model, head, processor, F32, proof['frozen_prefix_sha256']) == a
    assert model_facts(full, other, other_head, other_processor, F32, proof['frozen_prefix_sha256']) == b
    other.half()
    cast = model_facts(full, other, other_head, other_processor, F16)
    assert rng_fingerprint(full, False) == before and not torch.cuda.is_initialized()
    return {'strict400_f32_f16_head_fit_raw_normalized_packed_exact': True, 'cuda_initialized': False,
            'FIT_indices': [0, 1], 'FIT_rgb_sha256': rgb, 'FIT_pixels_sha256': full.pair.smoke.digest({'pixels': pixels}),
            'FIT_raw_sha256': full.pair.smoke.digest({'raw': raw[0]}), 'models': {'native_f32': a, 'independent_f32': b, 'cast_f16': cast},
            'rng_sha256': before}


def gpu_phase(full, control, authority, protocol, state):
    import torch
    from sfora.siglip2_compact_serving import Siglip2CompactIndex
    from sfora.cutile_int8 import CutilePackedInt8Gallery
    before = rng_fingerprint(full, True)
    # Qualify the independent model first; retain only small CPU packed references.
    # Keeping both full models live exceeded the frozen 8GiB cgroup peak gate.
    torch.cuda.reset_peak_memory_stats()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        model, head, processor = native_reload(full, control)
        model.half().cuda(); head.cuda()
    b = model_facts(full, model, head, processor, F16)
    wires = load_wires(authority)
    groups = sentinel_groups()
    b1_indices = list(range(32)) + [COUNTS['query'] - 1]
    references = []
    for group in groups + [{'role': 'query', 'indices': [i]} for i in b1_indices]:
        images = decode(image_paths(control, protocol, group['role'], group['indices']))
        reference = native_reference(full, model, head, processor, images)
        references.append(reference)
        assert torch.cuda.max_memory_allocated() < CUDA_CAP
        usage(state)
    assert model_facts(full, model, head, processor, F16) == b
    del model, head, processor, images, reference
    gc.collect()
    torch.cuda.empty_cache()
    # Preserve the complete-unit peak from the independent constructor.
    encoder = public_encoder(full, control, reset_peak=False)
    a = model_facts(full, encoder.vision, encoder.head, encoder.processor, F16)
    native_runtime, public_runtime = (json.loads(json.dumps(f['runtime'])) for f in (b, a))
    assert native_runtime['config'].pop('dtype') == 'float32'
    assert public_runtime['config'].pop('dtype') is None
    assert native_runtime == public_runtime
    assert a['environment'] == b['environment'] == authority['environment'] and a['buffers_sha256'] == b['buffers_sha256']
    equal_b32, observations = [], []
    with Siglip2CompactIndex(encoder, CutilePackedInt8Gallery.open_packed(artifact(authority, 'native_library'), wires['gallery'])) as index:
        for group, reference in zip(groups, references[:len(groups)], strict=True):
            role, indices = group['role'], group['indices']
            images = decode(image_paths(control, protocol, role, indices))
            actual = encoder.encode_images(images)
            full.qualified.selected.fp16.same(actual, reference)
            full.qualified.selected.fp16.same(actual, packed_slice(wires[role], indices))
            same_results(index.search_images(images), cpu_reference(reference, wires['gallery']))
            observations.append({**group, 'packed_sha256': full.pair.smoke.digest({'codes': actual.codes, 'inverse': actual.inverse_norms})})
            assert torch.cuda.max_memory_allocated() < CUDA_CAP
            usage(state)
        for i, reference in zip(b1_indices, references[len(groups):], strict=True):
            images = decode(image_paths(control, protocol, 'query', [i]))
            actual = encoder.encode_images(images)
            full.qualified.selected.fp16.same(actual, reference)
            same_results(index.search_images(images), cpu_reference(reference, wires['gallery']))
            saved = packed_slice(wires['query'], [i])
            equal_b32.append(bool(torch.equal(actual.codes, saved.codes) and torch.equal(actual.inverse_norms, saved.inverse_norms)))
            assert torch.cuda.max_memory_allocated() < CUDA_CAP
            usage(state)
    assert model_facts(full, encoder.vision, encoder.head, encoder.processor, F16) == a
    assert rng_fingerprint(full, True) == before
    return {'precision': 'fp16_native', 'gallery_images': COUNTS['gallery'], 'B32_sentinels': sentinel_groups(),
            'B32_observations': observations, 'B1_query_indices': b1_indices, 'B1_matches_saved_B32': equal_b32,
            'public_native_packed_top10_ordinal_score_bits_exact': True, 'B1_scope': '33 sentinels; no full B1 official quality',
            'models': {'public_f16': a, 'independent_f16': b}, 'rng_sha256': before,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--phase', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--authority', type=Path)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-proof', type=Path)
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--cpu-log', type=Path)
    parser.add_argument('--cpu-log-sha256')
    args = parser.parse_args()
    new_output(args.output)
    root = Path(__file__).resolve().parent
    path = args.authority or root / 'full-valid-anchor-serving-authority.json'
    raw = read(path, args.authority_sha256)
    state = resource_start(args.phase, raw)
    data = startup(root, path, args.authority_sha256)
    authority, code = data[:2]
    binding = bindings(authority, args.authority_sha256, code)
    pins = (args.cpu_proof, args.cpu_sha256, args.cpu_log, args.cpu_log_sha256)
    if args.phase == 'gpu':
        assert all(pins), 'GPU requires pinned CPU proof and original log'
        proof_authority(*pins, 'cpu', binding)
    else:
        assert not any(pins)
    full, control, _, prior, proof = runtime_startup(root, data)
    compiler_version = configure(full, authority, args.phase == 'gpu', prior)
    facts = cpu_phase(full, control, proof, authority) if args.phase == 'cpu' else gpu_phase(full, control, authority, data[-1], state)
    assert full.qualified.teacher.qualified.numerical_flags() == prior['numerical_flags']
    assert startup(root, path, args.authority_sha256) == data
    loaded_code_guard(root, code)
    if args.phase == 'gpu':
        proof_authority(*pins, 'cpu', binding)
        facts.update(cpu_proof_sha256=args.cpu_sha256, cpu_log_sha256=args.cpu_log_sha256,
                     cpu_proof=str(args.cpu_proof), cpu_log=str(args.cpu_log))
    save(args.output, {'schema': 'full-valid-anchor-serving-qualification-v1', 'pass': True, 'phase': args.phase,
        'bindings': binding, 'invocation': invocation(), 'compiler_version': compiler_version,
        'model_snapshot': str(control.large_snapshot), 'model_file_sha256': full.pair.smoke.MODEL_HASHES,
        'dataset_root': str(control.dataset_root), 'frozen_prefix_sha256': proof['frozen_prefix_sha256'],
        'numerical_flags': prior['numerical_flags'],
        'source_head_buffers_processor_runtime_rng_preserved': True, 'python_assertions_enabled': __debug__,
        'loaded_entrypoint_code_verified': True,
        'rng_components': ['torch_CPU', 'all_torch_CUDA', 'Python_random', 'NumPy_global'],
        'optimizer_updates': 0, 'quality_read': False, 'official_quality_scored': False,
        'decoded_previously_observed_official_sentinels': args.phase == 'gpu',
        'claim_eligible': False, 'paired_public_speed_win': False,
        'p99_certified': False, **facts, 'resources': usage(state)})
    print('PASS full2000 ' + args.phase + ' serving qualification; no quality/speed/p99 claim', flush=True)


if __name__ == '__main__':
    main()
