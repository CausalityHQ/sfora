#!/usr/bin/env python3
"""Contingent native256 full-TRAIN2000, twenty exclusive 100-update chunks.

Parent alone admits the pilot GO and successful original unit terminals. Supply
the arm's seed-179032 TRAIN100 resume.pt, its SHA and receipt SHA on EVERY chunk.
The receipt locates the immutable original pilot closure/authority; never stage
this file inside that closure. --previous is the previous receipt.json, bound by
--previous-sha256. Retain every output directory. No quality is read here.

The full stage resets AdamW/scaler/counter, preserves learned FIT proxies, adds
new proxies from the normalized class means of one unaugmented B32 source pass,
and trains all 25,882 TRAIN rows in original partition order. Native execution,
mechanics qualification, terminal admission and launch remain parent-owned.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import resource
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
    raise SystemExit('Qualification requires unoptimized Python')

SCHEMA = 'siglip2-substrate-full-v1'
WIDTHS = {'large': 1024, 'so400': 1152}
SEED = 179032
TOTAL_UPDATES = 2000
HANG_BACKSTOP_SECONDS = 3600
POLICY = {'seconds': HANG_BACKSTOP_SECONDS, 'host_bytes': 8 * 1024**3,
          'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}
PILOT_SHA = '1b73b2a6bdc1ee0cc468109e15746c8c5cde32569ac3b5de05d07f9e2f0b573f'
PARTITION_SHA = 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'
PARSER_SHA = '635d300c66b2bebe7753707c9196d4735a97f07c7dd999f186f9b9b1251b7f1f'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        digest, buffer = hashlib.sha256(), bytearray(1024**2)
        while size := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:size])
            os.posix_fadvise(stream.fileno(), stream.tell() - size, size, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def descriptor(path, digest=None):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical regular file required')
    actual = sha(path)
    require(digest is None or actual == digest, 'file SHA256 differs: ' + str(path))
    return {'path': str(path), 'sha256': actual}


def read_json(ref):
    descriptor(ref['path'], ref['sha256'])
    raw = Path(ref['path']).read_bytes()
    require(len(raw) < 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'JSON size/SHA differs')
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_module(name, path, digest):
    descriptor(path, digest)
    require(name not in sys.modules, 'module already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'module changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))
    return module


def validate_layout(arm, width, head_shape, vision_count):
    require(arm in WIDTHS and width == WIDTHS[arm] and tuple(head_shape) == (128, WIDTHS[arm]) and
            vision_count == {'large': 400, 'so400': 448}[arm], 'arm/width/head/vision mismatch')


def validate_source(record, arm, checkpoint):
    require(record['schema'] == 'siglip2-substrate-adaptation-v1' and record['phase'] == 'train' and
            record['arm'] == arm and record['seed'] == SEED and record['width'] == WIDTHS[arm] and
            record['completed_step'] == 100 and record['checkpoint'] == checkpoint and
            record['identity']['arm'] == arm and record['identity']['seed'] == SEED and
            record['identity']['config']['hidden_size'] == WIDTHS[arm] and
            record['pass'] is True and record['training_qualified'] is True and
            record['quality_read'] is False and record['training_state_discarded'] is False and
            record['strict_independent_whole_head_buffers_raw_packed_reload_exact'] is True and
            record['exit_rehash_pass'] is True and record['resource_policy'] == POLICY and
            record['code']['train_siglip2_substrate_adaptation.py'] == PILOT_SHA,
            'accepted v9 arm seed-179032 TRAIN100 source required')


def validate_receipt(row, end, arm, source, code):
    require(row['schema'] == SCHEMA and row['pass'] is True and row['quality_read'] is False and
            row['arm'] == arm and row['width'] == WIDTHS[arm] and row['seed'] == SEED and
            row['completed_step'] == end and row['chunk_start'] == end - 100 and
            row['source'] == source and row['code'] == code and row['resource_policy'] == POLICY and
            row['reload_exact'] is True and row['exit_rehash_pass'] is True and
            row['identity']['arm'] == arm and row['identity']['source'] == source and
            row['identity']['code'] == code and row['identity']['schema'] == SCHEMA,
            'chunk receipt lineage/step/arm differs')
    require([r['step'] for r in row['steps']] == list(range(end - 99, end + 1)), 'chunk update rows differ')
    require(row['previous_checkpoint_sha256'] is None if end == 100 else
            isinstance(row['previous_checkpoint_sha256'], str), 'previous checkpoint link differs')


def check_chain(previous, end, arm, source, code):
    require(type(end) is int and 0 <= end <= TOTAL_UPDATES and end % 100 == 0, 'chunk endpoint differs')
    require(bool(previous) == bool(end), 'previous receipt required exactly after chunk one')
    latest = child = None
    while end:
        row = read_json(previous)
        validate_receipt(row, end, arm, source, code)
        require(Path(row['checkpoint']['path']) == Path(previous['path']).parent / 'resume.pt', 'checkpoint path differs')
        if child is not None:
            require(child['previous_checkpoint_sha256'] == row['checkpoint']['sha256'] and
                    child['identity'] == row['identity'], 'chunk SHA/state chain differs')
        else:
            descriptor(row['checkpoint']['path'], row['checkpoint']['sha256'])
            latest = row
        child, previous, end = row, row['previous'], end - 100
        require(bool(previous) == bool(end), 'truncated or extended chunk chain')
    return latest


def admit_source(source, arm, output):
    record = read_json(source['receipt'])
    descriptor(source['checkpoint']['path'], source['checkpoint']['sha256'])
    validate_source(record, arm, source['checkpoint'])
    pilot_path = Path(record['invocation']['argv'][0])
    pilot = load_module('_siglip2_full_original_pilot', pilot_path, PILOT_SHA)
    argv = record['invocation']['argv']
    require(len(argv) == 15 and argv[1::2] == ['--execution-sha256', '--authority', '--authority-sha256',
            '--phase', '--arm', '--seed', '--output'] and argv[8:14:2] == ['train', arm, str(SEED)] and
            Path(argv[14]) == Path(source['checkpoint']['path']).parent, 'original pilot argv differs')
    context = pilot.authority(SimpleNamespace(execution_sha256=argv[2], authority=Path(argv[4]),
        authority_sha256=argv[6], phase='train', arm=arm, seed=SEED, output=output))
    require(record['execution_sha256'] == argv[2] and record['authority_sha256'] == argv[6] and
            record['selected_cpu'] == context['launch']['selected_cpu'] and
            record['selected_mechanics'] == context['launch']['selected_mechanics'] and
            record['identity']['qualifier_authority'] == context['launch']['qualifier_authority'],
            'source receipt original authority differs')
    for ref in source.values():
        pilot.bound_file(context['guards'], ref['path'], ref['sha256'])
    context.update(pilot=pilot, source_record=record, full_source=source)
    return context


def native_start(context):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'explicit CUDA/CUBLAS configuration required')
    before = context['source'].cgroup_memory()
    prior = context['cpu']['invocation']
    require(str(Path(sys.executable).resolve()) == prior['python'] and sha(prior['python']) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'qualified interpreter differs')
    import torch
    require(not torch.cuda.is_initialized(), 'admission must precede CUDA')
    flags = context['cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(context['source'].numerical_flags() == flags and torch.cuda.is_available() and
            torch.cuda.device_count() == 1, 'qualified numerical flags/one CUDA device required')
    context['ref'] = context['pilot'].reference_math(context)
    return before, flags


def resources(context, before, started):
    import torch
    after = context['source'].cgroup_memory()
    wall, rss, peak = time.monotonic() - started, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, torch.cuda.max_memory_allocated()
    require(after['path'] == before['path'] and wall < HANG_BACKSTOP_SECONDS and
            rss * 1024 <= POLICY['host_bytes'] and peak < POLICY['cuda_allocated_bytes_exclusive'],
            'whole-job resource predicate failed')
    return {'cgroup_before': before, 'cgroup_after': after, 'wall_seconds': wall,
            'process_peak_rss_kib': rss, 'peak_cuda_allocated_bytes': peak}


def records(context, parser_path):
    parser = load_module('_siglip2_full_partition', parser_path, PARSER_SHA)
    context['pilot'].bound_file(context['guards'], parser_path, PARSER_SHA)
    root = Path(context['initialized']['source_context']['fit']['dataset_root'])
    context['pilot'].bound_file(context['guards'], root / 'Eval/list_eval_partition.txt', PARTITION_SHA)
    result = parser.parse_inshop_partition(root)
    require(all(r.image_path.resolve().is_relative_to(root) for r in result), 'dataset image escaped root')
    return result


def full_rows(context, all_records):
    rows = []
    for row in all_records:
        if row.split == 'train':
            path = row.image_path.resolve()
            value = descriptor(path)
            require(context['guards'].setdefault(str(path), value['sha256']) == value['sha256'], 'TRAIN image authority conflict')
            rows.append({'path': str(path), 'product': row.label, 'image_sha256': value['sha256']})
    require(len(rows) == 25882 and len({r['path'] for r in rows}) == 25882 and
            len({r['product'] for r in rows}) == 3997, 'full TRAIN rows/classes differ')
    return rows


def schedule(target):
    # Copied from pe_large_optimization.schedule; no incumbent globals patched.
    import numpy as np
    target=np.asarray(target,dtype=np.int64)
    names=sorted(set(target.tolist()))
    assert names==list(range(len(names))) and len(names)>=64
    rng=np.random.default_rng(SEED)
    order=rng.permutation(names)
    members={c:np.flatnonzero(target==c) for c in names}
    batches=np.asarray([[int(rng.choice(members[int(c)])) for c in order[(step*64+np.arange(64))%len(order)]] for step in range(TOTAL_UPDATES)],dtype=np.int64)
    assert batches.shape==(TOTAL_UPDATES,64) and all(len(set(target[b]))==64 for b in batches)
    assert set(target[batches.ravel()].tolist())==set(names)
    return batches


def classifier_from_source(manifest, old_names, old_proxy, bank):
    # Copied full-arm rules from pe_large_coverage.inputs, preserving proxy scale.
    import numpy as np
    import torch
    from torch.nn import functional as F
    names=sorted(set(r['product'] for r in manifest))
    classes={name:i for i,name in enumerate(names)}
    target=np.asarray([classes[r['product']] for r in manifest],dtype=np.int64)
    classifier=torch.empty((len(names),128),dtype=torch.float32)
    for i,name in enumerate(names):
        if name in old_names:
            classifier[i]=old_proxy[old_names.index(name)]
        else:
            classifier[i]=F.normalize(bank[torch.from_numpy(target==i)].mean(dim=0),dim=0)
    assert torch.equal(classifier[torch.tensor([classes[n] for n in old_names])],old_proxy)
    assert torch.isfinite(classifier).all() and (classifier.norm(dim=1)>0).all()
    return target, classifier


def pixels(context, processor, rows, batch, step=None):
    import torch
    paths = tuple(Path(rows[i]['path']) for i in batch)
    for path, ordinal in zip(paths, batch, strict=True):
        context['pilot'].bound_file(context['guards'], path, rows[ordinal]['image_sha256'])
    dataset = context['ref'].ImageRows(paths, tuple(batch), augment=step is not None)
    images, rgb = [], hashlib.sha256()
    try:
        with torch.random.fork_rng(devices=[]):
            if step is not None:
                torch.random.default_generator.manual_seed(SEED * 100000 + step)
            for i in range(len(batch)):
                image = dataset[i][0]
                images.append(image)
                rgb.update(str(image.size).encode())
                rgb.update(image.tobytes())
        result = processor(images=images, return_tensors='pt')['pixel_values']
    finally:
        for image in images:
            image.close()
    require(result.dtype == torch.float32 and tuple(result.shape) == (len(batch), 3, 256, 256), 'native256 pixels differ')
    return result, rgb.hexdigest()


def source_state(context, flags):
    import torch
    pilot, record = context['pilot'], context['source_record']
    path = Path(record['checkpoint']['path'])
    disk = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    identity = disk['identity']
    require(canonical(identity) == record['identity'], 'source typed identity differs')
    validate_layout(identity['arm'], disk['config']['hidden_size'], disk['head']['weight'].shape, len(disk['vision']))
    require(tuple(disk['classifier'].shape) == (2004, 128) and tuple(disk['bank'].shape) == (13283, 128),
            'source FIT geometry differs')
    del disk
    gc.collect()
    state = pilot.restore_independent(context, context['ref'], path, record['checkpoint']['sha256'],
                                     record['terminal_state_sha256'], identity, flags, 100)
    require(canonical(state['inventory']) == record['identity']['roles'], 'source trainable roles differ')
    return state, identity


def initialize_full(context, state, identity, rows, code):
    import torch
    from torch.nn import functional as F
    pilot = context['pilot']
    old_proxy = state['classifier'].detach().cpu().clone()
    fit = context['initialized']['source_context']['fit']
    old_names = sorted({r['product'] for r in fit['rows']})
    require(len(old_names) == 2004 and [old_names[i] for i in fit['targets']] ==
            [r['product'] for r in fit['rows']], 'source FIT class mapping differs')
    # Release pilot moments and FIT bank before the one full source export.
    del state['optimizer'], state['bank'], state['classifier'], state['params']
    state['model'].eval(); state['head'].eval()
    bank = torch.empty((25882, 128), dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch = tuple(range(start, min(start + 32, len(rows))))
            x, _ = pixels(context, state['processor'], rows, batch)
            with torch.autocast('cuda', dtype=torch.float16):
                pooled = state['model'](pixel_values=x.cuda()).pooler_output
            bank[start:start + len(batch)] = F.normalize(state['head'](F.normalize(pooled.float(), dim=1)), dim=1).cpu()
            print(json.dumps({'phase': 'source_bank', 'rows': start + len(batch)}), flush=True)
    require(torch.isfinite(bank).all().item() and torch.allclose(bank.norm(dim=1), torch.ones(len(bank)), atol=1e-5, rtol=0),
            'full source bank must be finite unit vectors')
    target, classifier = classifier_from_source(rows, old_names, old_proxy, bank)
    state.update(bank=bank, target=torch.from_numpy(target), classifier=torch.nn.Parameter(classifier),
                 schedules={str(SEED): torch.from_numpy(schedule(target))}, seed=SEED, counter=0)
    state = pilot.move_cuda(context, state, context['ref'])
    identity = {**identity, 'schema': SCHEMA, 'source': context['full_source'], 'code': code,
                'full_rows': rows, 'full_rows_sha256': json_sha(rows), 'total_updates': TOTAL_UPDATES,
                'augmentation': {'seed': SEED, 'formula': '179032*100000+step', 'steps': [1, TOTAL_UPDATES]},
                'schedule_sha256': pilot.fingerprint(state['schedules'][str(SEED)]),
                'schedule_facts': {str(SEED): context['source'].tensor_fact(state['schedules'][str(SEED)])},
                'optimizer_defaults': state['optimizer'].defaults.copy(),
                'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
                'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
                'initial_bank_sha256': pilot.fingerprint(state['bank']),
                'initial_classifier_sha256': pilot.fingerprint(state['classifier']),
                'frozen_prefix_sha256': context['source_record']['frozen_prefix_sha256']}
    pilot.integrity(state, identity)
    return state, identity


def validate_scaler(value, step):
    require(type(step) is int and 0 <= step <= TOTAL_UPDATES and value['growth_interval'] == 2000 and
            value['growth_factor'] == 2. and value['backoff_factor'] == .5 and
            value['_growth_tracker'] == step % 2000 and value['scale'] == 128 * 2 ** (step // 2000),
            'full-stage successful scaler counter differs')


def payload(context, state, identity, flags):
    return {**context['pilot'].payload(state, identity, flags), 'schema': SCHEMA}


def check_payload(context, disk, record, flags):
    pilot, identity, arm, step = context['pilot'], disk['identity'], record['arm'], record['completed_step']
    require(disk.keys() == pilot.PAYLOAD_KEYS and disk['schema'] == SCHEMA and
            canonical(identity) == record['identity'] and disk['counter'] == step and disk['seed'] == SEED and
            disk['numerical_flags'] == flags and disk['config'] == identity['config'] and
            disk['optimizer_defaults'] == identity['optimizer_defaults'] and
            disk['optimizer']['param_groups'] == identity['optimizer_serial_groups'] and
            set(disk['optimizer']['state']) == set(range(208)) and len(identity['parameter_names']) == 208 and
            all(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and int(v['step']) == step
                for v in disk['optimizer']['state'].values()) and len(disk['cuda_rng']) == 1,
            'complete full-stage state/optimizer/RNG differs')
    validate_layout(arm, disk['config']['hidden_size'], disk['head']['weight'].shape, len(disk['vision']))
    validate_scaler(disk['scaler'], step)
    require(tuple(disk['classifier'].shape) == (3997, 128) and tuple(disk['bank'].shape) == (25882, 128) and
            tuple(disk['target'].shape) == (25882,) and disk['schedules'].keys() == {str(SEED)} and
            tuple(disk['schedules'][str(SEED)].shape) == (TOTAL_UPDATES, 64) and
            json_sha(identity['full_rows']) == identity['full_rows_sha256'], 'full-stage data geometry differs')


def restore_full(context, record, flags, *, inference=False):
    """Pilot strict constructor/page-release rules, with full-stage payload checks."""
    import torch
    from transformers import AutoImageProcessor
    pilot, source, initialized = context['pilot'], context['source'], context['initialized']
    checkpoint = Path(record['checkpoint']['path'])
    descriptor(checkpoint, record['checkpoint']['sha256'])
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    with checkpoint.open('rb') as stream:
        pages = pilot.CheckpointPages(stream)
        check_payload(context, disk, record, flags)
        identity = disk['identity']
        require(pilot.fingerprint(disk, consumed=pages.consume) == record['terminal_state_sha256'], 'serialized full state differs')
        model = source.construct(disk['config'], initialized['source_context'])
        pilot.load_vision(model, disk['vision'], pages)
        roles = source.configure_roles(model, initialized['source_context']['expected'], model.config.num_hidden_layers)
        require(canonical(roles) == canonical(identity['roles']), 'complete trainable roles differ')
        buffers = dict(model.named_buffers())
        require(buffers.keys() == disk['buffers'].keys(), 'complete buffer inventory differs')
        with torch.no_grad():
            for name, value in buffers.items():
                require(value.shape == disk['buffers'][name].shape and value.dtype == disk['buffers'][name].dtype, 'buffer layout differs')
                value.copy_(disk['buffers'][name]); pages.consume(disk['buffers'][name])
        head = context['qualifier'].head_from({'head.weight': disk['head']['weight'], 'head.bias': disk['head']['bias']}, WIDTHS[record['arm']])
        for tensor in disk['head'].values():
            pages.consume(tensor)
        processor = AutoImageProcessor.from_pretrained(initialized['source_context']['entry']['input']['preprocessor']['path'],
                                                       local_files_only=True, backend='torchvision')
        state = {'model': model, 'head': head, 'processor': processor, 'inventory': roles}
        if not inference:
            state.update(classifier=torch.nn.Parameter(pages.copy(disk['classifier'])), bank=pages.copy(disk['bank']).detach(),
                         target=pages.copy(disk['target']), pca={n: pages.copy(v) for n, v in disk['pca'].items()},
                         schedules={n: pages.copy(v) for n, v in disk['schedules'].items()}, seed=SEED, counter=disk['counter'])
            state = pilot.move_cuda(context, state, context['ref'])
            require(torch.equal(state['positive'].cpu(), disk['positive']), 'singleton/positive table differs')
            pages.consume(disk['positive'])
            for moments in disk['optimizer']['state'].values():
                for name in moments:
                    moments[name] = pages.copy(moments[name], 'cpu' if name == 'step' else 'cuda')
            state['optimizer'].load_state_dict(disk['optimizer'])
            state['scaler'].load_state_dict(disk['scaler'])
        else:
            model.train(); head.train()
        require(pilot.runtime(context, state) == identity['runtime'] and
                pilot.fingerprint(dict(model.named_buffers())) == identity['buffers_sha256'] and
                pilot.fingerprint({n: p for n, p in model.named_parameters() if not p.requires_grad}) == identity['frozen_prefix_sha256'],
                'runtime/buffer/frozen source differs')
        torch.random.set_rng_state(pages.copy(disk['cpu_rng']))
        torch.cuda.set_rng_state_all([pages.copy(v) for v in disk['cuda_rng']])
    del disk, buffers, tensor, value, model, head
    gc.collect()
    if not inference:
        pilot.integrity(state, identity)
        require(pilot.fingerprint(payload(context, state, identity, flags), state['frozen_cache']) == record['terminal_state_sha256'],
                'independent complete full-stage restore differs')
    return state, identity


def save(context, state, identity, flags, path):
    import torch
    pilot = context['pilot']
    state['optimizer'].zero_grad(set_to_none=True)
    pilot.integrity(state, identity)
    saved = payload(context, state, identity, flags)
    with path.open('xb') as stream:
        writer = pilot.CheckpointWriter(stream)
        torch.save(saved, writer)
        writer.flush()
    return descriptor(path), pilot.fingerprint(saved, state['frozen_cache'])


def update(context, ref, state, identity, flags, step):
    import torch
    pilot = context['pilot']
    from torch.nn import functional as F
    torch.cuda.synchronize(); started = time.perf_counter()
    require(state['counter'] == step - 1, 'uninterrupted update counter differs')
    batch = tuple(state['schedules'][str(state['seed'])][step - 1].tolist())
    batch_pixels, rgb = pixels(context, state['processor'], identity['full_rows'], batch, step)
    pixel_sha = pilot.fingerprint(batch_pixels)
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version = state['bank']._version
    ce_sum = rank_sum = 0.
    raw_rows = []
    for offset in range(0, 64, 16):
        index = torch.tensor(batch[offset:offset + 16], device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pooled = state['model'](pixel_values=batch_pixels[offset:offset + 16].cuda()).pooler_output
        with torch.autocast(device_type='cuda', enabled=False):
            raw = state['head'](F.normalize(pooled.float(), dim=1))
            ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                              torch.arange(128, device='cuda').unsqueeze(0), margin=.3, scale=64)
            rank = pilot.valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
            loss = (ce + 8 * rank) * .25
        require(raw.dtype == torch.float32 and torch.isfinite(pooled).all().item() and
                torch.isfinite(raw).all().item() and torch.isfinite(loss).item(), 'nonfinite update')
        scaler.scale(loss).backward()
        ce_sum += float(ce.detach()) * .25; rank_sum += float(rank.detach()) * .25
        raw_rows.append(raw.detach())
        del pooled, raw, ce, rank, loss, index
    require(state['bank']._version == version, 'bank changed during backward')
    scaler.unscale_(optimizer)
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for _, p in state['params']), 'all208 finite FP32 gradients required')
    boundary = state['model'].config.num_hidden_layers - 12
    gradients = {str(i): sum(float(p.grad.double().norm()) for p in state['model'].encoder.layers[i].parameters())
                 for i in range(boundary, boundary + 12)}
    require(all(math.isfinite(v) and v > 0 for v in gradients.values()), 'final12 active gradients differ')
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in state['params']], 1., error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    require(scaler.get_scale() >= scale, 'optimizer update skipped')
    state['counter'] += 1
    validate_scaler(scaler.state_dict(), state['counter'])
    rows, positions = ref.member_bank_refresh_rows(batch)
    raw = torch.cat(raw_rows)
    state['bank'][torch.tensor(rows, device='cuda')] = ref.member_bank_refresh_values(
        raw, raw, torch.tensor(positions, device='cuda'), live_head=False)
    require(state['bank']._version == version + 1, 'last duplicate detached bank refresh differs')
    optimizer.zero_grad(set_to_none=True)
    pilot.integrity(state, identity)
    require(context['source'].numerical_flags() == flags and torch.cuda.max_memory_allocated() < 10_000_000_000,
            'numerical flags/CUDA whole-job peak differs')
    row = {'step': step, 'batch': list(batch), 'schedule_sha256': identity['schedule_sha256'],
           'augmentation_seed': 179032 * 100000 + step, 'rgb_sha256': rgb, 'pixels_sha256': pixel_sha,
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients,
           'state_sha256': pilot.fingerprint(payload(context, state, identity, flags), state['frozen_cache'])}
    torch.cuda.synchronize(); row['seconds'] = time.perf_counter() - started
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def install_backstop():
    def expired(signum, frame):
        raise TimeoutError('3600-second hang backstop; invalid run, no quality conclusion')
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(HANG_BACKSTOP_SECONDS)


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--arm', choices=sorted(WIDTHS), required=True)
    result.add_argument('--source-resume', type=Path, required=True)
    result.add_argument('--source-sha256', required=True)
    result.add_argument('--source-receipt-sha256', required=True)
    result.add_argument('--chunk-end', type=int, choices=range(100, TOTAL_UPDATES + 1, 100), required=True)
    result.add_argument('--previous', type=Path)
    result.add_argument('--previous-sha256')
    result.add_argument('--partition-parser', type=Path, default=Path(__file__).resolve().parents[1] / 'src/sfora/unicom_inshop.py')
    result.add_argument('--output', type=Path, required=True)
    return result


def run(args):
    started = time.monotonic()
    install_backstop()
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
            not args.output.exists() and not args.output.is_symlink(), 'exclusive canonical output required')
    source = {'checkpoint': descriptor(args.source_resume, args.source_sha256),
              'receipt': descriptor(args.source_resume.parent / 'receipt.json', args.source_receipt_sha256)}
    code = {'trainer': sha(Path(__file__)), 'partition_parser': PARSER_SHA}
    require(bool(args.previous) == bool(args.previous_sha256), 'previous path/SHA pair required')
    previous = descriptor(args.previous, args.previous_sha256) if args.previous else None
    prior = check_chain(previous, args.chunk_end - 100, args.arm, source, code)
    context = admit_source(source, args.arm, args.output)
    context['pilot'].bound_file(context['guards'], Path(__file__).resolve(), code['trainer'])
    before, flags = native_start(context)
    import torch
    all_records = records(context, args.partition_parser)
    rows = full_rows(context, all_records)
    if prior:
        state, identity = restore_full(context, prior, flags)
        require(identity['full_rows'] == rows, 'full TRAIN image/order drift')
    else:
        state, identity = source_state(context, flags)
        state, identity = initialize_full(context, state, identity, rows, code)
    require(torch.equal(state['schedules'][str(SEED)], torch.from_numpy(schedule(state['target'].cpu().numpy()))), 'full schedule differs')
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    args.output.mkdir()
    tick = time.monotonic()
    steps = [update(context, context['ref'], state, identity, flags, step)
             for step in range(args.chunk_end - 99, args.chunk_end + 1)]
    training_seconds = time.monotonic() - tick
    checkpoint, terminal = save(context, state, identity, flags, args.output / 'resume.pt')
    record = {'schema': SCHEMA, 'pass': True, 'arm': args.arm, 'width': WIDTHS[args.arm], 'seed': SEED,
              'source': source, 'code': code, 'identity': canonical(identity), 'chunk_start': args.chunk_end - 100,
              'completed_step': args.chunk_end, 'steps': steps, 'checkpoint': checkpoint, 'terminal_state_sha256': terminal,
              'previous': previous, 'previous_checkpoint_sha256': prior['checkpoint']['sha256'] if prior else None,
              'training_wall_seconds': training_seconds, 'quality_read': False, 'resource_policy': POLICY}
    del state
    gc.collect(); torch.cuda.empty_cache()
    state, _ = restore_full(context, record, flags)
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and
            all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)), 'training RNG drift')
    del state
    gc.collect(); torch.cuda.empty_cache()
    context['pilot'].exit_rehash(context)
    if previous:
        check_chain(previous, args.chunk_end - 100, args.arm, source, code)
    require(context['source'].numerical_flags() == flags, 'exit numerical flags drift')
    record.update(resources(context, before, started), reload_exact=True, exit_rehash_pass=True,
                  terminal_exit_and_both_locks_require_parent_receipt=True)
    publish(args.output / 'receipt.json', record)
    return record


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({'pass': True, 'arm': args.arm, 'completed_step': result['completed_step'], 'quality_read': False}), flush=True)


if __name__ == '__main__':
    main()
