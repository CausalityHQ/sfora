#!/usr/bin/env python3
"""Discarded native256 mechanics17 and one fresh TRAIN100; no quality reads.

Freeze execution.json with exactly FILES (copy reference sources unchanged to
the declared reference names). External initialized qualifier3/PCA3/export2/
source3 closures remain separate. All admission is stdlib, before Torch.

Launch native256-substrate-adaptation-launch-v1 has exactly: schema,
execution_sha256, phase, arm, seed, qualifier_root, qualifier_execution_sha256,
qualifier_authority:{path,sha256}, selected_cpu:TERMINAL,
selected_mechanics:null|TERMINAL, resource_policy:policy(phase), both_locks_held.
TERMINAL is the original initializer.admit_terminal descriptor: receipt/log
{path,sha256}, unit, invocation_id, service_seconds, native_peak_rss_kib,
both_locks_held:true. CPU receipt is proof.json; mechanics receipt.json.
Mechanics uses seed179032 once per arm, admitting both TRAIN schedule seeds.
Only TRAIN179032 replays mechanics diagnostics; both use common augmentation.

CLI: --execution-sha256 SHA --authority PATH --authority-sha256 SHA
--phase {mechanics,train} --arm {large,so400} --seed {179032,179041}
--output NEWDIR. Output receipt.json; TRAIN also complete resume.pt.
Mechanics deletes all checkpoints and trained state. Native resource/terminal
qualification and whole-service costs require the parent's original log/footer.

API: authority(args) before native imports; selected_ast(path,pin) compiles
only the fixed named definitions, never reference imports or startup;
reference_math(context) binds native dependencies after admission;
payload/fingerprint/check_payload describe the complete process-boundary state.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import __future__
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import statistics
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

SCHEMA = 'siglip2-substrate-adaptation-v1'
AUTHORITY_SCHEMA = 'native256-substrate-adaptation-launch-v1'
FILES = {'train_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_adaptation.py',
         'deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py'}
QUALIFIER_FILES = {'qualify_siglip2_initialized_cpu.py', 'test_siglip2_initialized_cpu.py',
                   'joint_relational_compaction.py'}
RANK_SHA256 = '435bd73d9a3dde04c9daa819ec3694e228b4cadcc2523c35e4c877ae5d034870'
REFERENCES = {
    'reference_train_sop_siglip2_compact.py': {
        'source': '1946bafee086e14f83290631a05f926ba46480d2fc905eef0016b2211641b610',
        'ast': '9bfbaeb039a36a38a2fc3aef73f29b74bc677a366c58fa1d4ea74ce8a4ba9ebf',
        'names': ('ImageRows', 'member_bank_positive_ordinals', 'member_bank_refresh_rows',
                  'member_bank_refresh_values', 'member_bank_rank_loss')},
    'reference_unicom_training.py': {
        'source': 'a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272',
        'ast': '405dd26ed39efd34ec0673febdb48fa52f0b67db5d503c343646fc06ebe443ba',
        'names': ('_class_slices', 'sharded_mask_arcface_logits', 'sharded_mask_arcface_loss')}}
WIDTHS = {'large': 1024, 'so400': 1152}
SEEDS = (179032, 179041)
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
AUGMENTATION = {'seed': 179032, 'formula': '179032*100000+step', 'steps': [1, 100]}
PAYLOAD_KEYS = {'schema', 'identity', 'config', 'vision', 'buffers', 'head', 'classifier', 'bank', 'target',
                'pca', 'schedules', 'positive', 'optimizer', 'optimizer_defaults', 'scaler', 'cpu_rng',
                'cuda_rng', 'counter', 'seed', 'numerical_flags'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(phase):
    require(phase in ('mechanics', 'train'), 'fixed phase required')
    return {'seconds': 120 if phase == 'mechanics' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) is not None, 'SHA256 required')
    with path.open('rb') as stream:
        digest, buffer = hashlib.sha256(), bytearray(1024**2)
        while read := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:read])
            # Same consumed-range advice as the frozen initializer, not a cgroup guarantee.
            os.posix_fadvise(stream.fileno(), stream.tell() - read, read, os.POSIX_FADV_DONTNEED)
        require(digest.hexdigest() == expected, 'file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path


def read_json(path, expected, guards):
    with bound_file(guards, path, expected).open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == expected, 'JSON size/SHA256 differs')
    return strict_json(raw)


def descriptor_json(value, guards):
    require(value.keys() == {'path', 'sha256'}, 'exact JSON descriptor required')
    return read_json(value['path'], value['sha256'], guards)


def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json(root / 'execution.json', expected, guards)
    require(code.keys() == names, 'execution requires exactly declared files')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def load_bare(name, path, expected):
    path = bound_file({}, path, expected)
    require(name not in sys.modules, 'helper already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'bare origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'helper changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))
    bound_file({}, path, expected)
    require(Path(module.__file__) == Path(module.__spec__.origin) == path, 'loaded origin differs')
    return module


def selected_ast(path, pin):
    """Fixed extraction: original filenames/line numbers, future annotations."""
    raw = bound_file({}, path, pin['source']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['source'], 'reference source changed')
    tree = ast.parse(raw, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and n.name in pin['names']]
    require(tuple(n.name for n in nodes) == pin['names'] and
            all(type(n) is (ast.ClassDef if n.name == 'ImageRows' else ast.FunctionDef) and
                not n.decorator_list for n in nodes), 'extra/missing selected reference definitions')
    selected = ast.Module(body=nodes, type_ignores=[])
    digest = hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest()
    require(digest == pin['ast'], 'selected reference AST differs')
    return compile(selected, str(path), 'exec', flags=__future__.annotations.compiler_flag, dont_inherit=True)


def bootstrap(args):
    require(args.seed in SEEDS and args.arm in WIDTHS and
            (args.phase != 'mechanics' or args.seed == SEEDS[0]), 'fixed phase/arm/seed required')
    require(not any(name.split('.')[0] in NATIVE for name in sys.modules), 'native packages preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    launch = read_json(args.authority, args.authority_sha256, guards)
    require(launch.keys() == {'schema', 'execution_sha256', 'phase', 'arm', 'seed', 'qualifier_root',
            'qualifier_execution_sha256', 'qualifier_authority', 'selected_cpu', 'selected_mechanics',
            'resource_policy', 'both_locks_held'} and launch['schema'] == AUTHORITY_SCHEMA and
            launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
            launch['arm'] == args.arm and launch['seed'] == args.seed and
            launch['resource_policy'] == policy(args.phase) and launch['both_locks_held'] is True and
            (launch['selected_mechanics'] is None) == (args.phase == 'mechanics'), 'launch authority/profile differs')
    require(code['deployed_code_rank.py'] == RANK_SHA256, 'rank helper pin differs')
    for name, pin in REFERENCES.items():
        require(code[name] == pin['source'], 'reference full-file pin differs')
        selected_ast(root / name, pin)
    qualifier_root = Path(launch['qualifier_root'])
    qualifier_code = closure(qualifier_root, launch['qualifier_execution_sha256'], QUALIFIER_FILES, guards)
    require(not root.is_relative_to(qualifier_root) and not qualifier_root.is_relative_to(root) and
            args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
            not args.output.exists() and not args.output.is_symlink() and
            not any(args.output.is_relative_to(p) for p in (root, qualifier_root)), 'separate exclusive output required')
    qualifier = load_bare('_siglip2_pinned_adaptation_qualifier',
                          qualifier_root / 'qualify_siglip2_initialized_cpu.py',
                          qualifier_code['qualify_siglip2_initialized_cpu.py'])
    require(qualifier.FILES == QUALIFIER_FILES and qualifier.WIDTHS == WIDTHS and qualifier.SEEDS == SEEDS,
            'qualifier profile differs')
    return root, code, guards, launch, qualifier


def admit_cpu(context):
    q, launch, initialized = context['qualifier'], context['launch'], context['initialized']
    record = descriptor_json(launch['selected_cpu']['receipt'], context['guards'])
    expected = {'schema': q.SCHEMA, 'phase': 'initialized-cpu', 'arm': context['args'].arm,
                'width': WIDTHS[context['args'].arm], 'output_dim': 128,
                'authority_sha256': launch['qualifier_authority']['sha256'],
                'execution_sha256': launch['qualifier_execution_sha256'], 'code': initialized['code'],
                'initializer_authority': initialized['launch']['initializer_authority'],
                'selected_initializer': initialized['launch']['selected_initializer'],
                'pca_final_cgroup': initialized['pca_final_cgroup'],
                'source_binding': initialized['record']['source_binding'],
                'source_sample': initialized['source_context']['proof']['sample'],
                'ordered_input_sha256': initialized['record']['ordered_input_sha256'],
                'ordered_rgb_sha256': initialized['record']['ordered_rgb_sha256'],
                'initializers': initialized['record']['artifact'], 'resource_policy': q.POLICY,
                'numerical_flags': initialized['source_context']['proof']['numerical_flags'],
                'augmentation': q.AUGMENTATION}
    require(all(record[k] == v for k, v in expected.items()), 'original initialized CPU binding differs')
    yes = ('pass', 'initializer_qualified', 'source_qualified', 'reload_exact', 'fresh_source',
           'first_model_released_before_independent_clone', 'source_cpu_runtime_and_first2_exact',
           'constructor_rng_preserved', 'exit_rehash_pass', 'optimizer_created',
           'both_locks_held_in_parent_authority', 'terminal_exit_and_both_locks_require_parent_receipt')
    no = ('training_qualified', 'quality_qualified', 'quality_read', 'cuda_initialized', 'gradients_created',
          'pca_rerun', 'teacher_state_reused', 'trained_state_reused')
    require(all(record[k] is True for k in yes) and all(record[k] is False for k in no) and
            record['updates'] == record['head_updates'] == record['optimizer_state_entries'] == 0 and
            record['state']['counter'] == 0 and record['state']['seed'] == SEEDS[0] and
            record['state']['arrays'] == initialized['record']['arrays'] and
            record['state']['runtime'] == initialized['source_context']['proof']['runtime'] and
            len(record['state']['parameter_names']) == 208, 'initialized CPU qualification/state differs')
    receipt = Path(launch['selected_cpu']['receipt']['path'])
    require(receipt.name == 'proof.json' and record['checkpoint']['path'] == str(receipt.parent / 'initialized.pt'),
            'CPU proof/checkpoint roles differ')
    prior, invocation = initialized['pca']['startup']['invocation'], record['invocation']
    require(invocation['argv'] == [str(Path(launch['qualifier_root']) / 'qualify_siglip2_initialized_cpu.py'),
            '--execution-sha256', launch['qualifier_execution_sha256'], '--authority', launch['qualifier_authority']['path'],
            '--authority-sha256', launch['qualifier_authority']['sha256'], '--arm', context['args'].arm,
            '--output', str(receipt.parent)] and invocation['cuda_visible_devices'] == '' and
            all(invocation[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')),
            'original initialized CPU argv/interpreter differs')
    require(all(record['input_guards'].get(p) == h for p, h in context['initialized_prereq_guards'].items()),
            'original CPU input guards differ')
    require(record['origins']['packages'] == initialized['packages'], 'CPU origin packages differ')
    for path, digest in record['origins']['files'].items():
        require(record['input_guards'].get(path) == digest, 'CPU origin file guard differs')
        bound_file(context['guards'], path, digest)
    final = initialized['init'].admit_terminal(record, launch['selected_cpu'], 120, context['guards'])
    return record, final


def diagnostic(row):
    return {k: v for k, v in row.items() if k != 'seconds'}


def phase_diagnostic(phase, started):
    """Log-only snapshots, including failing facts; qualification checks stay separate."""
    cgroup = {}
    try:
        unified = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                   if line.startswith('0::')]
        require(len(unified) == 1 and unified[0] != '/', 'enclosing cgroup v2 unit unavailable')
        root = Path('/sys/fs/cgroup') / unified[0].lstrip('/')
        cgroup['path'] = str(root)
        cgroup['values'] = {name: (root / name).read_text().strip() for name in
                           ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current',
                            'memory.swap.peak', 'memory.swap.max', 'memory.events')}
    except (OSError, ValueError) as error:
        cgroup['error'] = str(error)
    print(json.dumps({'diagnostic': 'phase', 'phase': phase, 'elapsed_seconds': time.monotonic() - started,
                      'invocation_id': os.environ.get('INVOCATION_ID'), 'cgroup': cgroup},
                     sort_keys=True, allow_nan=False), flush=True)


def check_cpu_state(facts, original):
    # Receipts canonicalize tuples and config integer keys; torch checkpoints retain them.
    require(strict_json(json.dumps({**facts, 'seed': SEEDS[0]}, allow_nan=False)) == original,
            'fresh complete initialized CPU state differs')


def check_schedule(batches, targets, seed):
    require(seed in SEEDS and len(targets) == 13283 and set(targets) == set(range(2004)) and
            len(batches) == 100 and all(len(b) == 64 and all(type(n) is int and 0 <= n < 13283 for n in b) and
                len({targets[n] for n in b}) == 64 for b in batches) and
            {targets[n] for b in batches for n in b} == set(range(2004)), 'dense FIT B64x100 schedule differs')


def admit_mechanics(context):
    descriptor = context['launch']['selected_mechanics']
    if descriptor is None:
        return None, None
    record = descriptor_json(descriptor['receipt'], context['guards'])
    require(record['schema'] == SCHEMA and record['phase'] == 'mechanics' and record['arm'] == context['args'].arm and
            record['seed'] == SEEDS[0] and record['execution_sha256'] == context['args'].execution_sha256 and
            record['code'] == context['code'] and record['selected_cpu'] == context['launch']['selected_cpu'] and
            record['reference_pins'] == strict_json(json.dumps(REFERENCES)) and
            record['rank_helper_sha256'] == RANK_SHA256 and
            record['qualifier_authority'] == context['launch']['qualifier_authority'] and
            record['resource_policy'] == policy('mechanics') and record['augmentation'] == AUGMENTATION and
            record['pass'] is True and record['quality_read'] is False and record['completed_step'] == 17 and
            record['training_state_discarded'] is True and record['native17_equals_serialized8_plus9_exact'] is True and
            record['strict_independent_whole_head_buffers_raw_packed_reload_exact'] is True and
            record['first_references_released_before_reload'] is True and record['exit_rehash_pass'] is True and
            record['peak_cuda_allocated_bytes'] < 10_000_000_000 and
            len(record['steps']) == 17 and len(record['resumed_steps']) == 9 and
            [r['step'] for r in record['steps']] == list(range(1, 18)) and
            all(diagnostic(a) == diagnostic(b) for a, b in zip(record['steps'][8:], record['resumed_steps'], strict=True)),
            'same-arm original mechanics proof differs')
    require(Path(descriptor['receipt']['path']).name == 'receipt.json', 'mechanics receipt role differs')
    invocation = record['invocation']
    argv = invocation['argv']
    require(len(argv) == 15 and argv == [str(context['root'] / 'train_siglip2_substrate_adaptation.py'),
            '--execution-sha256', context['args'].execution_sha256, '--authority', argv[4],
            '--authority-sha256', record['authority_sha256'], '--phase', 'mechanics', '--arm', context['args'].arm,
            '--seed', str(SEEDS[0]), '--output', str(Path(descriptor['receipt']['path']).parent)] and
            invocation['cuda_visible_devices'] not in (None, '') and
            all(invocation[k] == context['cpu']['invocation'][k] for k in ('python', 'python_sha256', 'python_version')),
            'original mechanics argv/interpreter differs')
    original_launch = read_json(argv[4], record['authority_sha256'], context['guards'])
    require(original_launch == {**context['launch'], 'phase': 'mechanics', 'seed': SEEDS[0],
                                'selected_mechanics': None, 'resource_policy': policy('mechanics')},
            'original mechanics launch authority differs')
    return record, context['initialized']['init'].admit_terminal(record, descriptor, 120, context['guards'])


def authority(args):
    root, code, guards, launch, qualifier = bootstrap(args)
    descriptor_json(launch['qualifier_authority'], guards)
    initialized = qualifier.authority(SimpleNamespace(
        execution_sha256=launch['qualifier_execution_sha256'], authority=Path(launch['qualifier_authority']['path']),
        authority_sha256=launch['qualifier_authority']['sha256'], arm=args.arm, output=args.output))
    initialized_prereq_guards = initialized['guards'].copy()
    for path, digest in guards.items():
        require(initialized['guards'].setdefault(path, digest) == digest, 'conflicting admission guard')
    context = {'args': args, 'root': root, 'code': code, 'guards': initialized['guards'], 'launch': launch,
               'qualifier': qualifier, 'initialized': initialized, 'source': initialized['source'],
               'initialized_prereq_guards': initialized_prereq_guards}
    external = (initialized['root'], initialized['pca']['root'], initialized['source_context']['root'],
                initialized['source_context']['own_root'])
    require(all(not root.is_relative_to(p) and not p.is_relative_to(root) and
                not args.output.is_relative_to(p) for p in external), 'original closures must remain separate')
    context['cpu'], context['cpu_final_cgroup'] = admit_cpu(context)
    context['mechanics'], context['mechanics_final_cgroup'] = admit_mechanics(context)
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports occurred during admission')
    return context


def reference_math(context):
    import numpy as np
    import torch
    from PIL import Image
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset
    from torchvision import transforms
    rank = load_bare('_siglip2_pinned_adaptation_rank', context['root'] / 'deployed_code_rank.py', RANK_SHA256)
    namespace = {'__name__': '_siglip2_pinned_adaptation_reference_math',
                 'torch': torch, 'np': np, 'F': F, 'nn': nn, 'math': math, 'Dataset': Dataset,
                 'transforms': transforms, 'Image': Image, 'Path': Path,
                 'smooth_ap_bank_loss': rank.smooth_ap_bank_loss}
    for name, pin in REFERENCES.items():
        exec(selected_ast(context['root'] / name, pin), namespace)
    return SimpleNamespace(**namespace)


def valid_rank(ref, raw, bank, head, positives, ordinals):
    valid = (positives >= 0).any(dim=1)
    if not bool(valid.any()):
        return raw.sum() * 0
    return ref.member_bank_rank_loss(raw[valid], bank, head, positives[valid], ordinals[valid],
                                    live_head=False) * (valid.sum() / len(valid))


def fingerprint(value, frozen=None):
    """Typed, length-framed complete tree hash; tensor device is not identity."""
    import torch
    digest = hashlib.sha256()
    def frame(raw):
        raw = raw.encode() if isinstance(raw, str) else raw
        digest.update(str(len(raw)).encode() + b':' + raw)
    def visit(item):
        if isinstance(item, torch.Tensor):
            frame('Tensor')
            key = (item.data_ptr(), item._version, str(item.dtype), tuple(item.shape))
            fact = frozen.get(key) if frozen is not None else None
            if fact is None:
                raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                fact = (str(item.dtype), tuple(item.shape), hashlib.sha256(memoryview(raw)).hexdigest())
            visit(fact)
        elif isinstance(item, dict):
            frame('dict'); frame(str(len(item)))
            for key in sorted(item, key=repr):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            frame(type(item).__name__); frame(str(len(item)))
            for child in item:
                visit(child)
        else:
            frame(type(item).__name__); frame(repr(item))
    visit(value)
    return digest.hexdigest()


def payload(state, identity, flags):
    import torch
    return {'schema': SCHEMA, 'identity': identity, 'config': state['model'].config.to_dict(),
            'vision': dict(state['model'].state_dict()), 'buffers': dict(state['model'].named_buffers()),
            'head': dict(state['head'].state_dict()), 'classifier': state['classifier'].detach(),
            'bank': state['bank'], 'target': state['target'], 'pca': state['pca'],
            'schedules': state['schedules'], 'positive': state['positive'],
            'optimizer': state['optimizer'].state_dict(), 'optimizer_defaults': state['optimizer'].defaults.copy(),
            'scaler': state['scaler'].state_dict(), 'cpu_rng': torch.random.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(), 'counter': state['counter'], 'seed': state['seed'],
            'numerical_flags': flags}


def check_payload(saved, identity, expected_step, defaults, groups, names, flags):
    require(saved.keys() == PAYLOAD_KEYS and saved['schema'] == SCHEMA and saved['identity'] == identity and
            saved['counter'] == expected_step and saved['seed'] == identity['seed'] and
            saved['numerical_flags'] == flags and saved['optimizer_defaults'] == defaults and
            saved['optimizer']['param_groups'] == groups and len(names) == 208 and
            set(saved['optimizer']['state']) == set(range(208)) and
            all(set(v) == {'step', 'exp_avg', 'exp_avg_sq'} and int(v['step']) == expected_step
                for v in saved['optimizer']['state'].values()) and
            len(saved['cuda_rng']) == 1 and saved['scaler']['scale'] >= 128 and
            saved['scaler']['_growth_tracker'] == expected_step,
            'complete state/optimizer/counter/scaler/RNG identity differs')


def frozen_cache(model):
    """Only immutable prefix tensors are cached; versions are audited each update."""
    import torch
    result = {}
    for _, value in model.named_parameters():
        if not value.requires_grad:
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            result[(value.data_ptr(), value._version, str(value.dtype), tuple(value.shape))] = (
                str(value.dtype), tuple(value.shape), hashlib.sha256(memoryview(raw)).hexdigest())
    return result


def runtime(context, state):
    """Original module/processor facts without hashing updated weights twice."""
    source, packages = context['source'], context['initialized']['packages']
    modules = []
    for name, module in state['model'].named_modules():
        row = {'name': name, **source.module_origin(type(module), packages), 'training': module.training}
        row['attributes'] = {key: value for key, value in vars(module).items() if not key.startswith('_') and
                             (value is None or isinstance(value, (bool, int, float, str)) or
                              isinstance(value, (tuple, list)) and
                              all(isinstance(item, (bool, int, float, str)) for item in value))}
        if hasattr(module, 'config'):
            row['attn_implementation'] = module.config._attn_implementation
        modules.append(row)
    return json.loads(json.dumps({'modules': modules,
        'processor': {'origin': source.module_origin(type(state['processor']), packages),
                      'config': json.loads(state['processor'].to_json_string()),
                      'backend': state['processor'].backend}}, allow_nan=False))


def integrity(state, identity):
    import torch
    require(state['model'].training and state['head'].training and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks and
                not getattr(m, 'gradient_checkpointing', False) for m in state['model'].modules()) and
            state['model'].config.to_dict() == identity['config'] and
            state['model'].config._attn_implementation == 'sdpa', 'training config/modes/hooks differ')
    actual = [(n, p) for n, p in state['model'].named_parameters() if p.requires_grad]
    actual += [('compact_head.' + n, p) for n, p in state['head'].named_parameters()] + [('classifier', state['classifier'])]
    require([n for n, _ in actual] == identity['parameter_names'] and
            [id(p) for _, p in actual] == [id(p) for _, p in state['params']] ==
            [id(p) for g in state['optimizer'].param_groups for p in g['params']] and
            len(actual) == 208 and len({id(p) for _, p in actual}) == 208 and
            state['optimizer'].defaults == identity['optimizer_defaults'] and
            [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups] ==
            identity['optimizer_groups'], 'actual208 optimizer roles/order/defaults differ')
    require(all(p.dtype == torch.float32 and p.device.type == 'cuda' and torch.isfinite(p).all().item()
                for _, p in actual) and state['bank'].dtype == torch.float32 and not state['bank'].requires_grad and
            state['bank'].grad is None and torch.isfinite(state['bank']).all().item(), 'finite FP32 members/bank differ')
    require(all(p.grad is None for p in state['model'].parameters() if not p.requires_grad) and
            [(n, p._version) for n, p in state['model'].named_parameters() if not p.requires_grad] ==
            state['frozen_versions'] and fingerprint(dict(state['model'].named_buffers())) == identity['buffers_sha256'],
            'frozen prefix/buffers changed')
    require(len(state['optimizer'].state) == (208 if state['counter'] else 0), 'optimizer state count differs')
    for _, param in actual:
        if state['counter']:
            moments = state['optimizer'].state[param]
            require(set(moments) == {'step', 'exp_avg', 'exp_avg_sq'} and int(moments['step']) == state['counter'] and
                    all(moments[n].dtype == torch.float32 and moments[n].shape == param.shape and
                        torch.isfinite(moments[n]).all().item() for n in ('exp_avg', 'exp_avg_sq')),
                    'FP32 moments/steps differ')


def save(context, state, identity, flags, path):
    state['optimizer'].zero_grad(set_to_none=True)
    integrity(state, identity)
    saved = payload(state, identity, flags)
    with context['initialized']['source_context']['extract'].exclusive(path) as stream:
        import torch
        torch.save(saved, stream)
        stream.flush(); os.fsync(stream.fileno())
    return context['initialized']['init'].sha(path), fingerprint(saved, state['frozen_cache'])


def move_cuda(context, state, ref):
    import torch
    model, head = state['model'], state['head']
    model.cuda().train(); head.cuda().train()
    state['classifier'] = torch.nn.Parameter(state['classifier'].detach().cuda())
    state['bank'] = state['bank'].cuda().detach()
    state['target'] = state['target'].cuda()
    state['params'], state['optimizer'] = context['qualifier'].optimizer_state(model, head, state['classifier'])
    state['scaler'] = torch.amp.GradScaler('cuda', init_scale=128)
    state['positive'] = ref.member_bank_positive_ordinals(state['target'].cpu().numpy(), allow_singletons=True).cuda()
    state['frozen_versions'] = [(n, p._version) for n, p in model.named_parameters() if not p.requires_grad]
    state['frozen_cache'] = frozen_cache(model)
    state.pop('witness', None)
    return state


def restore_independent(context, ref, checkpoint, expected_sha, expected_fingerprint, identity, flags, step):
    """Caller has released every reference to the previous model and optimizer."""
    import torch
    from transformers import AutoImageProcessor
    source, initialized = context['source'], context['initialized']
    bound_file({}, checkpoint, expected_sha)
    disk = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    check_payload(disk, identity, step, identity['optimizer_defaults'], identity['optimizer_serial_groups'],
                  identity['parameter_names'], flags)
    require(fingerprint(disk) == expected_fingerprint and disk['config'] == identity['config'], 'serialized complete state differs')
    model = source.construct(disk['config'], initialized['source_context'])
    model.load_state_dict(disk['vision'], strict=True)
    roles = source.configure_roles(model, initialized['source_context']['expected'], model.config.num_hidden_layers)
    buffers = dict(model.named_buffers())
    require(buffers.keys() == disk['buffers'].keys(), 'independent complete buffer inventory differs')
    with torch.no_grad():
        for name, value in buffers.items():
            require(value.shape == disk['buffers'][name].shape and value.dtype == disk['buffers'][name].dtype,
                    'independent buffer shape/dtype differs')
            value.copy_(disk['buffers'][name])
    head = context['qualifier'].head_from({'head.weight': disk['head']['weight'], 'head.bias': disk['head']['bias']},
                                        WIDTHS[context['args'].arm])
    processor = AutoImageProcessor.from_pretrained(initialized['source_context']['entry']['input']['preprocessor']['path'],
                                                   local_files_only=True, backend='torchvision')
    state = {'model': model, 'head': head, 'processor': processor, 'inventory': roles,
             'classifier': torch.nn.Parameter(disk['classifier'].clone()), 'bank': disk['bank'].clone().detach(),
             'target': disk['target'].clone(), 'pca': {n: v.clone() for n, v in disk['pca'].items()},
             'schedules': {n: v.clone() for n, v in disk['schedules'].items()},
             'seed': disk['seed'], 'counter': disk['counter']}
    state = move_cuda(context, state, ref)
    require(runtime(context, state) == identity['runtime'], 'independent complete runtime/processor differs')
    require(torch.equal(state['positive'].cpu(), disk['positive']), 'independent singleton/positive table differs')
    state['optimizer'].load_state_dict(disk['optimizer'])
    state['scaler'].load_state_dict(disk['scaler'])
    torch.random.set_rng_state(disk['cpu_rng'])
    torch.cuda.set_rng_state_all(disk['cuda_rng'])
    del disk, buffers, value, model, head
    gc.collect()
    integrity(state, identity)
    require(fingerprint(payload(state, identity, flags), state['frozen_cache']) == expected_fingerprint,
            'independent whole/head/buffers/proxy/bank/optimizer/scaler/RNG reload differs')
    return state


def augmented_pixels(context, ref, state, batch, step):
    import torch
    require(type(step) is int and 1 <= step <= 100 and len(batch) == 64 and
            all(type(n) is int and 0 <= n < 13283 for n in batch), 'augmentation step/FIT ordinals differ')
    cpu_rng = torch.random.get_rng_state().clone()
    images, rgb = [], hashlib.sha256()
    fit = context['initialized']['source_context']['fit']
    dataset_root = Path(fit['dataset_root'])
    paths = tuple(context['initialized']['pca']['source_context']['all_images'])
    dataset = ref.ImageRows(paths, tuple(fit['targets']), augment=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(179032 * 100000 + step)
        for ordinal in batch:
            row = fit['rows'][ordinal]
            resolved = (dataset_root / row['relative_path']).resolve()
            require(resolved.is_relative_to(dataset_root) and resolved == paths[ordinal],
                    'augmented FIT image resolution/containment differs')
            bound_file(context['guards'], resolved, row['image_sha256'])
            image, target = dataset[ordinal]
            require(target == int(state['target'][ordinal]), 'augmented FIT target differs')
            rgb.update(str(image.size).encode())
            rgb.update(image.tobytes())
            images.append(image)
    pixels = state['processor'](images=images, return_tensors='pt')['pixel_values']
    for image in images:
        image.close()
    require(torch.equal(cpu_rng, torch.random.get_rng_state()) and pixels.dtype == torch.float32 and
            pixels.shape == (64, 3, 256, 256), 'common augmentation RNG/native256 pixels differ')
    return pixels, rgb.hexdigest()


def update(context, ref, state, identity, flags, step):
    import torch
    from torch.nn import functional as F
    torch.cuda.synchronize(); started = time.perf_counter()
    require(state['counter'] == step - 1, 'uninterrupted update counter differs')
    batch = tuple(state['schedules'][str(state['seed'])][step - 1].tolist())
    pixels, rgb = augmented_pixels(context, ref, state, batch, step)
    pixel_sha = fingerprint(pixels)
    optimizer, scaler = state['optimizer'], state['scaler']
    optimizer.zero_grad(set_to_none=True)
    version = state['bank']._version
    ce_sum = rank_sum = 0.
    raw_rows = []
    for offset in range(0, 64, 16):
        index = torch.tensor(batch[offset:offset + 16], device='cuda')
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            pooled = state['model'](pixel_values=pixels[offset:offset + 16].cuda()).pooler_output
        with torch.autocast(device_type='cuda', enabled=False):
            raw = state['head'](F.normalize(pooled.float(), dim=1))
            ce = ref.sharded_mask_arcface_loss(raw, state['classifier'], state['target'][index],
                                              torch.arange(128, device='cuda').unsqueeze(0), margin=.3, scale=64)
            rank = valid_rank(ref, raw, state['bank'], state['head'], state['positive'][index], index)
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
    require(scaler.state_dict()['_growth_tracker'] == state['counter'], 'scaler successful-update counter differs')
    rows, positions = ref.member_bank_refresh_rows(batch)
    raw = torch.cat(raw_rows)
    state['bank'][torch.tensor(rows, device='cuda')] = ref.member_bank_refresh_values(
        raw, raw, torch.tensor(positions, device='cuda'), live_head=False)
    require(state['bank']._version == version + 1, 'last duplicate detached bank refresh differs')
    optimizer.zero_grad(set_to_none=True)
    integrity(state, identity)
    require(context['source'].numerical_flags() == flags and torch.cuda.max_memory_allocated() < 10_000_000_000,
            'numerical flags/CUDA whole-job peak differs')
    row = {'step': step, 'batch': list(batch), 'schedule_sha256': identity['schedule_sha256'],
           'augmentation_seed': 179032 * 100000 + step, 'rgb_sha256': rgb, 'pixels_sha256': pixel_sha,
           'ce': ce_sum, 'rank': rank_sum, 'loss': ce_sum + 8 * rank_sum, 'scale': scaler.get_scale(),
           'preclip_norm': float(norm), 'gradient_norms': gradients,
           'state_sha256': fingerprint(payload(state, identity, flags), state['frozen_cache'])}
    torch.cuda.synchronize(); row['seconds'] = time.perf_counter() - started
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
    return row


def calibration(state):
    import torch
    from torch.nn import functional as F
    packing = sys.modules['_siglip2_pinned_initialized_packing']
    pixels = state['calibration_pixels'].cuda()
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        pooled = state['model'](pixel_values=pixels).pooler_output
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
        raw = state['head'](F.normalize(pooled.float(), dim=1))
        packed = packing.pack_int8_unit_embeddings(F.normalize(raw, dim=1).cpu())
    return {'raw': raw.cpu(), 'codes': packed.codes.cpu(), 'inverse_norms': packed.inverse_norms.cpu(),
            'wire_sha256': hashlib.sha256(packed.to_bytes()).hexdigest()}


def run(args):
    started = time.perf_counter()
    diagnostic_started = time.monotonic()
    require(sys.flags.optimize == 0 and os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '') and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')) is not None,
            'original unoptimized CUDA/CUBLAS/systemd launch required')
    require(sys.argv == [str(Path(__file__).absolute()), '--execution-sha256', args.execution_sha256,
            '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
            '--phase', args.phase, '--arm', args.arm, '--seed', str(args.seed), '--output', str(args.output)],
            'fixed canonical launch argv order required')
    phase_diagnostic('authority.begin', diagnostic_started)
    context = authority(args)
    phase_diagnostic('authority.end', diagnostic_started)
    source, q, initialized = context['source'], context['qualifier'], context['initialized']
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    initialized['init'].admit_cgroup(before, unit)
    prior = context['cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and initialized['init'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'original qualified interpreter differs')
    phase_diagnostic('native_import.begin', diagnostic_started)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU admission must precede CUDA construction')
    flags = context['cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original CPU defaults differ')
    torch.random.default_generator.manual_seed(SEEDS[0])
    phase_diagnostic('native_import.end', diagnostic_started)
    phase_diagnostic('fresh_cpu.begin', diagnostic_started)
    state = q.fresh(initialized, args.seed, 'cpu')
    facts = q.state_facts(initialized, state)
    check_cpu_state(facts, context['cpu']['state'])
    for seed in SEEDS:
        check_schedule(state['schedules'][str(seed)].tolist(), state['target'].tolist(), seed)
    witness = state['witness']
    require(source.tensor_fact(witness['raw']) == context['cpu']['raw_first2'] and
            {n: source.tensor_fact(witness[n]) for n in ('codes', 'inverse_norms')} ==
            {n: context['cpu']['packed_first2'][n] for n in ('codes', 'inverse_norms')} and
            witness['wire_sha256'] == context['cpu']['packed_first2']['wire_sha256'],
            'fresh CPU raw/packed first2 differs')
    calibration_pixels = state['witness']['pixels'].clone()
    del witness
    phase_diagnostic('fresh_cpu.end', diagnostic_started)
    phase_diagnostic('cpu_origins.begin', diagnostic_started)
    cpu_origins = source.imported_origins(initialized['source_context']['extract'], initialized['packages'])
    require(all(context['cpu']['origins']['files'].get(p) == h for p, h in cpu_origins['files'].items()),
            'actual CPU imports differ from qualified origins')
    ref = reference_math(context)
    phase_diagnostic('cpu_origins.end', diagnostic_started)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            not torch.cuda.is_initialized(), 'one visible uninitialized CUDA device required')
    phase_diagnostic('cuda.begin', diagnostic_started)
    torch.cuda.manual_seed_all(SEEDS[0])
    state = move_cuda(context, state, ref)
    phase_diagnostic('cuda.end', diagnostic_started)
    phase_diagnostic('initial_state.begin', diagnostic_started)
    expected_runtime = {'modules': [{**row, 'training': True, 'attributes': {**row['attributes'], 'training': True}}
                                    for row in facts['runtime']['modules']],
                        'processor': facts['runtime']['processor']}
    require(runtime(context, state) == expected_runtime, 'fresh actual training runtime/processor differs')
    identity = {'arm': args.arm, 'seed': args.seed, 'execution_sha256': args.execution_sha256,
                'qualifier_authority': context['launch']['qualifier_authority'],
                'selected_cpu': context['launch']['selected_cpu'], 'config': state['model'].config.to_dict(),
                'roles': state['inventory'], 'parameter_names': facts['parameter_names'],
                'optimizer_defaults': state['optimizer'].defaults.copy(),
                'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in state['optimizer'].param_groups],
                'optimizer_serial_groups': state['optimizer'].state_dict()['param_groups'],
                'runtime': expected_runtime,
                'buffers_sha256': fingerprint(dict(state['model'].named_buffers())),
                'schedule_sha256': fingerprint(state['schedules'][str(args.seed)]),
                'schedule_facts': state['schedule_facts'], 'augmentation': AUGMENTATION}
    integrity(state, identity)
    initial_sha = fingerprint(payload(state, identity, flags), state['frozen_cache'])
    frozen_sha = fingerprint({n: p for n, p in state['model'].named_parameters() if not p.requires_grad})
    phase_diagnostic('initial_state.end', diagnostic_started)
    rows, resumed = [], []
    state['calibration_pixels'] = calibration_pixels
    args.output.mkdir()
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    with TemporaryDirectory(prefix='discard-mechanics-', dir=args.output) as temporary:
        temporary = Path(temporary)
        checkpoint8, sha8, fingerprint8 = temporary / 'step8.pt', None, None
        total = 17 if args.phase == 'mechanics' else 100
        phase_diagnostic('updates.begin', diagnostic_started)
        tick = time.perf_counter()
        for step in range(1, total + 1):
            row = update(context, ref, state, identity, flags, step)
            if context['mechanics'] is not None and args.seed == SEEDS[0] and step <= 17:
                require(diagnostic(row) == diagnostic(context['mechanics']['steps'][step - 1]),
                        'fresh TRAIN first17 mechanics diagnostic replay differs')
            rows.append(row)
            if args.phase == 'mechanics' and step == 8:
                phase_diagnostic('save8.begin', diagnostic_started)
                sha8, fingerprint8 = save(context, state, identity, flags, checkpoint8)
                phase_diagnostic('save8.end', diagnostic_started)
        training_seconds = time.perf_counter() - tick
        phase_diagnostic('updates.end', diagnostic_started)
        checkpoint = temporary / 'step17.pt' if args.phase == 'mechanics' else args.output / 'resume.pt'
        phase_diagnostic(f'save{total}.begin', diagnostic_started)
        checkpoint_sha, terminal_sha = save(context, state, identity, flags, checkpoint)
        phase_diagnostic(f'save{total}.end', diagnostic_started)
        phase_diagnostic('calibration.begin', diagnostic_started)
        expected_calibration = calibration(state)
        phase_diagnostic('calibration.end', diagnostic_started)
        phase_diagnostic('updated_state.begin', diagnostic_started)
        require(torch.equal(cpu_rng, torch.random.get_rng_state()) and
                all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
                'training RNG changed outside common augmentation fork')
        require(fingerprint({n: p for n, p in state['model'].named_parameters() if not p.requires_grad}) == frozen_sha,
                'whole frozen prefix bytes changed')
        require(runtime(context, state) == identity['runtime'], 'updated complete runtime/processor differs')
        phase_diagnostic('updated_state.end', diagnostic_started)
        phase_diagnostic('droprefs.begin', diagnostic_started)
        del state
        gc.collect(); torch.cuda.empty_cache()
        phase_diagnostic('droprefs.end', diagnostic_started)
        if args.phase == 'mechanics':
            phase_diagnostic('restore8.begin', diagnostic_started)
            state = restore_independent(context, ref, checkpoint8, sha8, fingerprint8, identity, flags, 8)
            phase_diagnostic('restore8.end', diagnostic_started)
            state['calibration_pixels'] = calibration_pixels
            phase_diagnostic('resume.begin', diagnostic_started)
            resumed = [update(context, ref, state, identity, flags, step) for step in range(9, 18)]
            require(all(diagnostic(a) == diagnostic(b) for a, b in zip(rows[8:], resumed, strict=True)) and
                    fingerprint(payload(state, identity, flags), state['frozen_cache']) == terminal_sha,
                    'native17 versus serialized8 plus independent9 differs')
            phase_diagnostic('resume.end', diagnostic_started)
            phase_diagnostic('resume_droprefs.begin', diagnostic_started)
            del state
            gc.collect(); torch.cuda.empty_cache()
            phase_diagnostic('resume_droprefs.end', diagnostic_started)
        phase_diagnostic('finalreload.begin', diagnostic_started)
        state = restore_independent(context, ref, checkpoint, checkpoint_sha, terminal_sha, identity, flags, total)
        state['calibration_pixels'] = calibration_pixels
        require(fingerprint(calibration(state)) == fingerprint(expected_calibration), 'independent raw/packed reload differs')
        require(fingerprint({n: p for n, p in state['model'].named_parameters() if not p.requires_grad}) == frozen_sha,
                'independent frozen prefix differs')
        phase_diagnostic('finalreload.end', diagnostic_started)
        phase_diagnostic('final_droprefs.begin', diagnostic_started)
        del state, expected_calibration, calibration_pixels
        gc.collect(); torch.cuda.empty_cache()
        phase_diagnostic('final_droprefs.end', diagnostic_started)
    phase_diagnostic('exit_rehash.begin', diagnostic_started)
    origins = source.imported_origins(initialized['source_context']['extract'], initialized['packages'])
    for path, digest in origins['files'].items():
        bound_file(context['guards'], path, digest)
    q.rehash(initialized)
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'], 'exit own closure differs')
    phase_diagnostic('exit_rehash.end', diagnostic_started)
    after = source.cgroup_memory()
    initialized['init'].admit_cgroup(after, unit)
    wall = time.perf_counter() - started
    rss, peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, torch.cuda.max_memory_allocated()
    require(before['path'] == after['path'] and 0 < wall < policy(args.phase)['seconds'] and
            0 < rss <= 8 * 1024**2 and 0 < peak < 10_000_000_000 and source.numerical_flags() == flags and
            torch.equal(cpu_rng, torch.random.get_rng_state()) and
            all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            'whole job duration/RSS/CUDA/RNG/flags caps differ')
    receipt = {'schema': SCHEMA, 'phase': args.phase, 'arm': args.arm, 'seed': args.seed, 'width': WIDTHS[args.arm],
               'output_dim': 128, 'pass': True, 'quality_read': False, 'quality_qualified': False,
               'training_qualified': args.phase == 'train', 'fresh_source': True, 'trained_state_reused': False,
               'execution_sha256': args.execution_sha256, 'authority_sha256': args.authority_sha256,
               'code': context['code'], 'reference_pins': REFERENCES, 'rank_helper_sha256': RANK_SHA256,
               'qualifier_authority': context['launch']['qualifier_authority'],
               'selected_cpu': context['launch']['selected_cpu'], 'cpu_final_cgroup': context['cpu_final_cgroup'],
               'selected_mechanics': context['launch']['selected_mechanics'],
               'mechanics_final_cgroup': context['mechanics_final_cgroup'],
               'initial_state_sha256': initial_sha, 'terminal_state_sha256': terminal_sha,
               'identity': identity, 'steps': rows, 'resumed_steps': resumed, 'completed_step': total,
               'augmentation': AUGMENTATION, 'training_state_discarded': args.phase == 'mechanics',
               'native17_equals_serialized8_plus9_exact': args.phase == 'mechanics',
               'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
               'first_references_released_before_reload': True, 'optimizer_members': 208,
               'checkpoint': None if args.phase == 'mechanics' else {'path': str(checkpoint), 'sha256': checkpoint_sha},
               'frozen_prefix_sha256': frozen_sha, 'constructor_rng_preserved': True,
               'first17_mechanics_replay_exact': args.phase == 'train' and args.seed == SEEDS[0],
               'training_wall_seconds': training_seconds, 'median_update_seconds': statistics.median(r['seconds'] for r in rows[2:]),
               'mean_update_seconds': statistics.mean(r['seconds'] for r in rows),
               'wall_seconds': wall, 'process_peak_rss_kib': rss, 'peak_cuda_allocated_bytes': peak,
               'resource_policy': policy(args.phase), 'cgroup_before': before, 'cgroup_after': after,
               'origins': origins, 'input_guards': context['guards'], 'exit_rehash_pass': True,
               'both_locks_held_in_parent_authority': True, 'terminal_exit_and_both_locks_require_parent_receipt': True,
               'invocation': {'argv': sys.argv, 'python': str(python), 'python_sha256': prior['python_sha256'],
                              'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
                              'invocation_id': os.environ['INVOCATION_ID'],
                              'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}}
    phase_diagnostic('receipt.begin', diagnostic_started)
    initialized['pca']['exporter'].write_json(initialized['source_context']['extract'], args.output / 'receipt.json', receipt)
    phase_diagnostic('exit', diagnostic_started)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('mechanics', 'train'), required=True)
    result.add_argument('--arm', choices=sorted(WIDTHS), required=True)
    result.add_argument('--seed', type=int, choices=SEEDS, required=True)
    result.add_argument('--output', type=Path, required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Substrate adaptation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
