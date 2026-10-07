#!/usr/bin/env python3
"""One discarded genuine micro16 actual-objective connectivity witness; zero updates.

CLI: python -B qualify_actual_objective_encoder_gradients.py --authority FILE
     --authority-sha256 SHA --output NEWDIR
Authority has exactly schema=actual-objective-encoder-gradients-authority-v1,
files (FILES), python={path,sha256}, output, resource_policy=POLICY, and
both_locks_held=true. Parent freezes actual prospective hashes and collects
the enclosing normal exit and BOTH lifetime locks. No peak reset, state reuse,
optimizer update, quality read, TRAIN admission, or cost qualification.
"""
import argparse
import ast
import copy
from functools import lru_cache
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import resource
import runpy
import sys
import time
import traceback
from types import CodeType, FunctionType, SimpleNamespace
import weakref

HERE = Path(__file__).absolute().parent
LEGACY_SHA = '86cfd83947218cf5a10011efb21297811bb5f13188486faf314ae6f52e8b3dba'
if hashlib.sha256((HERE / 'qualify_connected_encoder_gradients.py').read_bytes()).hexdigest() != LEGACY_SHA:
    raise ValueError('original qualifier source SHA256 differs')
legacy_probe = SimpleNamespace(**runpy.run_path(str(HERE / 'qualify_connected_encoder_gradients.py')))
require, sha, pairs = legacy_probe.require, legacy_probe.sha, legacy_probe.pairs
MLP, POLICY = legacy_probe.MLP, legacy_probe.POLICY
HELPERS = {**legacy_probe.HELPERS, 'qualify_connected_encoder_gradients.py': LEGACY_SHA}
FILES = {*HELPERS, 'qualify_actual_objective_encoder_gradients.py', 'test_actual_objective_encoder_gradients.py'}
AUTHORITY_SCHEMA = 'actual-objective-encoder-gradients-authority-v1'
TRAIN_ROOT = Path('/home/riomus/runs/sfora-so400-identity-diversity-train-source-v5')
TRAIN_EXECUTION = 'bd081a02a49f1f0dd8305b78bb3f8aa91f5bb7331f3045325a185971fb7c04a8'
TRAIN_CODE = {'train_siglip2_identity_diversity.py': '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8',
              'test_siglip2_identity_diversity.py': '96a704f357eafcf07d07381f0bc2ce6537712e9d6b84829ba860cd1f0a436102'}
CPU_AUTHORITY_SHA = '8fcfb9ae55c33ed842c68ab01b59fea378f1258b4b784d5874e40e9f49801eac'
CPU_UNIT = {'both_locks_held': True, 'invocation_id': '8b1805166f8b4aa0940e944bd987e8a3',
    'log': {'path': str(TRAIN_ROOT / 'cpu-v5.log'), 'sha256': '70773d48671d83c41fbeafabe0895aca8a0ff70818fe77e7d6074363507dd6df'},
    'native_peak_rss_kib': 6085564,
    'receipt': {'path': '/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/receipt.json',
                'sha256': 'd2239da896465ee004ebb95c9316713d634fa457c91c708c6c410769a8c17b03'},
    'service_seconds': 424.51, 'unit': 'sfora-so400-identity-diversity-cpu-v5'}
INITIALIZER = {'path': '/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179061.pt',
               'sha256': '8b89897fdaaad94770e40bd2e69422734709e6d2d73710aa496610bdb2882454'}
PAYLOAD_SHA = '4a09ad011dfc4bc1897482b96ed69f882efe0c0962833a280f58a3cbd5adeaee'
ANCHORS = (185,419,3802,1536,5675,3633,5030,3737,5622,1070,6090,343,1312,4889,5235,1321)
RTOL, ATOL = 1e-5, 1e-6  # Original trainer objective/gradient correspondence tolerances.
BACKEND_SHA = '250394884a9f90845cf87b6fc0cf3341337b193428556c6ff61ed2b04bedb692'


def authenticate(path, expected, output):
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected) and
            sha(path) == expected, 'authority SHA256 differs')
    with path.open('rb') as stream:
        raw = stream.read(1024*1024 + 1)
    require(len(raw) <= 1024*1024 and hashlib.sha256(raw).hexdigest() == expected,
            'authority changed or oversized')
    a = json.loads(raw, object_pairs_hook=pairs)
    require(isinstance(a, dict) and a.keys() ==
            {'schema', 'files', 'python', 'output', 'resource_policy', 'both_locks_held'} and
            a['schema'] == AUTHORITY_SCHEMA and a['both_locks_held'] is True and
            a['resource_policy'] == POLICY and all(type(v) is int for v in a['resource_policy'].values()) and
            a['output'] == str(output) and isinstance(a['files'], dict) and a['files'].keys() == FILES,
            'authority schema/policy/files/output differ')
    require(output.is_absolute() and output.resolve() == output and output.parent.is_dir() and
            not output.exists() and not output.is_symlink(), 'canonical exclusive new output required')
    require(HERE.resolve() == HERE, 'canonical source directory required')
    guards = {str(path): expected}
    for name, digest in a['files'].items():
        require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) and
                (name not in HELPERS or digest == HELPERS[name]) and sha(HERE / name) == digest,
                'source SHA256 differs: ' + name)
        guards[str(HERE / name)] = digest
    python = a['python']
    require(isinstance(python, dict) and python.keys() == {'path', 'sha256'} and
            python['path'] == str(Path(sys.executable).resolve()) and
            sha(python['path']) == python['sha256'], 'interpreter authority differs')
    guards[python['path']] = python['sha256']
    return guards


def select_initializer(record):
    choices = [q for q in record['qualifications'] if q.get('arm') == 'control' and
               q.get('seed') == 179061 and q.get('identity', {}).get('device') == 'cpu']
    require(len(choices) == 1 and choices[0]['checkpoint'] == INITIALIZER and
            choices[0]['terminal_state_sha256'] == PAYLOAD_SHA, 'accepted CPU control061 initializer differs')
    return choices[0]


def first_microbatch(trainer, state):
    batch = state['schedules']['179061'][0].tolist()
    anchors = batch[:16]
    valid = trainer.ranking_membership(state['ranking_bank'], batch)['valid']
    require(len(batch) == 64 and anchors == list(ANCHORS) and valid == 63 and
            trainer.loss_denominators(valid) == (128, 126) and
            trainer.ranking_membership(state['ranking_bank'], anchors)['valid'] == 16,
            'exact first B64/micro16/global denominator differs')
    return batch, anchors, valid


def gradient_fact(torch, gradient, label, *, nonzero=True):
    if nonzero:
        return legacy_probe.gradient_fact(torch, SimpleNamespace(grad=gradient), label)
    require(gradient is not None and torch.isfinite(gradient).all().item(), 'missing/nonfinite gradient: '+label)
    norm = float(gradient.norm())
    require(math.isfinite(norm), 'nonfinite gradient norm: '+label)
    return {'norm':norm, 'nonzero':int(torch.count_nonzero(gradient).item())}


def _load_initializer(trainer, context, qualification):
    """Original model-free typed validator/restore; retain no mapped cache aliases."""
    import torch
    path = trainer.bound_file(context['guards'], INITIALIZER['path'], INITIALIZER['sha256'])
    disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    try:
        require(disk.keys() == trainer.PAYLOAD_KEYS and disk['counter'] == 0 and
                type(disk['counter']) is int and disk['optimizer']['state'] == {} and
                json.loads(json.dumps(disk['identity'], allow_nan=False)) == qualification['identity'],
                'typed disk initializer identity differs')
        ident = copy.deepcopy(disk['identity'])  # Preserve tuple-valued optimizer identity.
        context['flags'] = ident['numerical_flags']
        context['initial_static_sha256'] = ident['static_sha256']
        for key in ('common_initial_sha256', 'common_statistics_sha256', 'initial_A_sha256',
                    'initial_C_sha256', 'mu_train_sha256', 'mu_train_provenance_sha256'):
            context[key] = ident[key]
        context['initial'] = {k: trainer.clone(context, disk[k]) for k in ('provenance', 'scope')}
        with path.open('rb') as stream:
            pages = context['legacy']['original'].CheckpointPages(stream)
            trainer.check_payload(context, disk, ident, 0)
            require(trainer.fingerprint(context, disk, consumed=pages.consume) == PAYLOAD_SHA,
                    'complete typed initializer fingerprint differs')
        torch.random.set_rng_state(disk['cpu_rng'].clone())
    finally:
        del disk
        gc.collect()
    return trainer.restore(context, path, INITIALIZER['sha256'], PAYLOAD_SHA, ident, 0), ident


def pixels_for(trainer, context, state, processor, anchors, view):
    import torch
    from PIL import Image
    from torchvision import transforms
    selected = context['legacy']['selected']
    exporter, genuine = selected['exporter'], selected['genuine']
    manifest = genuine['selected']
    namespace = {'Dataset': torch.utils.data.Dataset, 'Path': Path, 'Image': Image, 'transforms': transforms}
    path = genuine['launch']['image_rows']['path']
    node = exporter.image_rows_node(path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), namespace)
    dataset = namespace['ImageRows'](tuple(Path(p) for p in manifest['resolved_paths']),
                                     tuple(manifest['targets']), augment=True)
    images, facts = [], []
    rng = torch.random.get_rng_state().clone()
    def rgb_fact(image):
        return {'mode': image.mode, 'size': list(image.size), 'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
    try:
        for ordinal in anchors:
            scoped, resolved, _ = trainer.canonical_row(context, state, ordinal)
            row, path = manifest['rows'][ordinal], Path(manifest['resolved_paths'][ordinal])
            require(path == resolved and row['train_row'] == scoped['original_train_row'] and
                    row['image_sha256'] == scoped['image_sha256'] and
                    manifest['targets'][ordinal] == int(state['target'][ordinal]) and
                    context['legacy']['extract'].sha(path) == row['image_sha256'],
                    'selected genuine TRAIN pixels differ')
            with Image.open(path) as opened:
                original = opened.convert('RGB')
            original_rgb = rgb_fact(original)
            try:
                image = original.copy() if view == 'canonical' else exporter.isolated_view(
                    torch, original, dataset.augment, row['train_row'])
            finally:
                original.close()
            images.append(image)
            facts.append({'view': view, 'ordinal': ordinal, 'original_row': manifest['original_rows'][ordinal],
                'train_row': row['train_row'], 'target': manifest['targets'][ordinal], 'path': str(path),
                'relative_path': row['relative_path'], 'image_sha256': row['image_sha256'],
                'rng_seed': None if view == 'canonical' else 179081 + row['train_row'],
                'original_rgb': original_rgb, 'rgb': rgb_fact(image)})
        pixels = processor(images=images, return_tensors='pt')['pixel_values']
        require(pixels.dtype == torch.float32 and pixels.shape == (16,3,256,256) and
                pixels.device.type == 'cpu' and torch.isfinite(pixels).all().item() and
                torch.equal(rng, torch.random.get_rng_state()), 'genuine FP32 pixels/global CPU RNG differ')
        for i, fact in enumerate(facts):
            fact['pixels'] = context['legacy']['source_driver'].tensor_fact(pixels[i])
        return pixels, facts
    finally:
        for image in images:
            image.close()


def _gradients(torch, mse, rank, members):
    return {label: torch.autograd.grad(term, members, retain_graph=True, allow_unused=True)
            for label, term in (('regression', mse), ('ranking', rank), ('total', mse+rank))}


def _match(torch, left, right, label, *, exact=False):
    require(left.shape == right.shape and left.dtype == right.dtype and
            torch.isfinite(left).all().item() and torch.isfinite(right).all().item() and
            (torch.equal(left, right) if exact else torch.allclose(left, right, rtol=RTOL, atol=ATOL)),
            'live correspondence differs: ' + label)
    return {'max_abs': float((left.detach()-right.detach()).abs().max()), 'rtol': RTOL, 'atol': ATOL,
            'bitwise': bool(torch.equal(left, right))}


def _diagnose_release(released, label, names=None):
    """Report surviving identities without retaining objects or changing the gate."""
    surviving = []
    for i, ref in enumerate(released):
        value = ref()
        if value is not None:
            cls = type(value)
            surviving.append({'name': (names or {}).get(id(value), f'released[{i}]'),
                              'type': cls.__module__ + '.' + cls.__qualname__})
    if surviving:
        print(json.dumps({'event': 'release_diagnostic', 'scope': label,
                          'surviving': surviving}), file=sys.stderr, flush=True)


def _processor_cache(processor, guards, *, empty=False):
    """Authenticate the one original self-keyed cache; never adopt prior entries."""
    backend = sys.modules.get('transformers.image_processing_backends')
    siglip = sys.modules.get('transformers.models.siglip.image_processing_siglip')
    require(backend is not None and siglip is not None and
            type(processor) is getattr(siglip, 'SiglipImageProcessor', None) and
            type(processor).__bases__ == (getattr(backend, 'TorchvisionBackend', None),),
            'processor cache owner differs')
    path = Path(backend.__file__)
    require(path.is_absolute() and path.resolve() == path and
            guards.get(str(path)) == BACKEND_SHA, 'processor cache source guard differs')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BACKEND_SHA, 'processor cache source differs')
    name = '_fuse_mean_std_and_rescale_factor'
    wrapper = vars(backend.TorchvisionBackend).get(name)
    require(type(wrapper) is type(lru_cache(maxsize=10)(lambda: None)) and
            inspect.getattr_static(processor, name) is wrapper and
            wrapper.cache_parameters() == {'maxsize': 10, 'typed': False} and
            all(getattr(wrapper, key) == getattr(type(wrapper), key).__get__(wrapper)
                for key in ('cache_info', 'cache_clear')) and wrapper.cache_info().maxsize == 10,
            'processor cache wrapper differs')
    function = getattr(wrapper, '__wrapped__', None)
    # Read-only GC edges bind __wrapped__ to the LRU's actual callable, not a decoy.
    require(type(function) is FunctionType and
            [obj for obj in gc.get_referents(wrapper) if type(obj) is FunctionType] == [function],
            'processor cache wrapper callable differs')
    # Compile only: no re-execution of the backend or replacement of its globals/math.
    module_code = compile(raw, str(path), 'exec', dont_inherit=True)
    class_code = next(c for c in module_code.co_consts if isinstance(c, CodeType) and c.co_name == 'TorchvisionBackend')
    original = next(c for c in class_code.co_consts if isinstance(c, CodeType) and c.co_name == name)
    require(function.__code__ == original and function.__globals__ is vars(backend) and
            function.__defaults__ == (None,) * 6 and function.__kwdefaults__ is None and
            function.__closure__ is None, 'processor cache live code differs')
    require(not empty or wrapper.cache_info().currsize == 0, 'processor cache must initially be empty')
    return wrapper  # Unbound: retaining this handle does not itself retain the processor.


def view_probe(trainer, context, cpu_state, model, processor, connected, anchors, valid, view):
    import torch
    from torch.nn import functional as F
    source = context['legacy']['source_driver']
    temporary, released, failure = {}, [], None
    result = None
    try:
        temporary['cpu_pixels'], image_facts = pixels_for(trainer, context, cpu_state, processor, anchors, view)
        temporary['pixels'] = temporary['cpu_pixels'].to('cuda')
        state = {'arm': 'control', 'device': 'cuda', 'views': cpu_state['views'],
            'ranking_bank': cpu_state['ranking_bank'],
            **{k: trainer.clone(context, cpu_state[k], 'cuda') for k in ('teachers','target','means','mu_train')}}
        temporary['state'] = state
        state['head_object'] = copy.deepcopy(cpu_state['head_object']).to('cuda').requires_grad_(False).train()
        for name in ('A', 'C'):
            state[name] = torch.nn.Parameter(trainer.clone(context, cpu_state[name], 'cuda'))
        released.extend(trainer.tensor_weakrefs(context, {k: state[k] for k in ('A','C','teachers','target','means','mu_train')}))
        released.extend(weakref.ref(p) for p in state['head_object'].parameters())
        released.extend(weakref.ref(p) for p in state['head_object'].buffers())
        frozen_before = trainer.fingerprint(context, {k: state[k] for k in ('A','C','teachers','target','means','mu_train')})
        head_before = trainer.fingerprint(context, dict(state['head_object'].state_dict()))
        temporary['cpu_rng'] = torch.random.get_rng_state().clone()
        temporary['cuda_rng'] = torch.cuda.get_rng_state_all()
        model.requires_grad_(False)
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.autocast('cuda', enabled=False):
            temporary['control_features'] = F.normalize(model(pixel_values=temporary['pixels']).pooler_output.float(), dim=1)
            temporary['control_raw'] = trainer.raw_features(context, state, temporary['control_features'])
            temporary['control_unit'] = F.normalize(temporary['control_raw'], dim=1).detach()
            temporary['control_scores'] = (temporary['control_unit'] @ trainer.ranking_gallery(context, state).T).detach()
            temporary['control_mse'], temporary['control_rank'], control_facts = trainer.loss_terms(
                context, state, temporary['control_raw'], anchors, valid)
            temporary['control_grads'] = _gradients(torch, temporary['control_mse'], temporary['control_rank'],
                                                    tuple(state[n] for n in ('A','C')))
        torch.cuda.synchronize()
        control_seconds = time.perf_counter()-tick
        for name, p in model.named_parameters():
            p.requires_grad_(name in MLP)
        del p
        torch.random.set_rng_state(temporary['cpu_rng'])
        torch.cuda.set_rng_state_all(temporary['cuda_rng'])
        temporary['members'] = tuple(dict(model.named_parameters())[n] for n in MLP) + tuple(state[n] for n in ('A','C'))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.autocast('cuda', enabled=False):
            features = F.normalize(model(pixel_values=temporary['pixels']).pooler_output.float(), dim=1)
            temporary['features'] = features
            require(features.shape == (16,1152) and features.dtype == torch.float32 and features.requires_grad,
                    'connected original FP32 features required')
            temporary['raw'] = connected.raw_features(features, state['head_object'], state['A'], state['means'],
                state['C'], state['mu_train'], context['legacy']['quadratic'], trainer.helper_guard(context))
            temporary['unit'] = F.normalize(temporary['raw'], dim=1)
            temporary['scores'] = (temporary['unit'] @ trainer.ranking_gallery(context, state).T).detach()
            temporary['mse'], temporary['rank'], facts = trainer.loss_terms(context, state, temporary['raw'], anchors, valid)
            temporary['grads'] = _gradients(torch, temporary['mse'], temporary['rank'], temporary['members'])
        torch.cuda.synchronize()
        connected_seconds = time.perf_counter()-tick
        require(control_facts == facts, 'original live-control membership/loss facts differ')
        pairing = {key: _match(torch, temporary[key], temporary['control_'+key], key)
                   for key in ('raw','unit','scores','mse','rank')}
        gradients = {label: {name: gradient_fact(torch, gradient, label+':'+name, nonzero=label != 'regression')
                            for name, gradient in zip((*MLP,'A','C'), grads, strict=True)}
                     for label, grads in temporary['grads'].items()}
        for name in ('A','C'):
            for label in ('regression','ranking','total'):
                _match(torch, temporary['grads'][label][4+('A','C').index(name)],
                       temporary['control_grads'][label][('A','C').index(name)], 'control '+label+name)
        # Same connected features; independent equal A/C leaves separate both routes.
        query, gallery = dict(state), dict(state)
        temporary.update(query=query, gallery=gallery)
        for name in ('A','C'):
            query[name] = torch.nn.Parameter(state[name].detach().clone())
            gallery[name] = torch.nn.Parameter(state[name].detach().clone())
            released.extend((weakref.ref(query[name]), weakref.ref(gallery[name])))
        temporary['split_raw'] = connected.raw_features(features, query['head_object'], query['A'], query['means'],
            query['C'], query['mu_train'], context['legacy']['quadratic'], trainer.helper_guard(context))
        temporary['split_mse'], temporary['split_rank'], split_facts = trainer.loss_terms(
            context, gallery, temporary['split_raw'], anchors, valid)
        _match(torch, temporary['rank'], temporary['split_rank'], 'split ranking', exact=True)
        require(split_facts == facts, 'split membership differs')
        temporary['split_gradients'] = torch.autograd.grad(temporary['split_rank'],
            tuple(query[n] for n in ('A','C')) + tuple(gallery[n] for n in ('A','C')), retain_graph=True)
        decomposition = {}
        for i, name in enumerate(('A','C')):
            temporary['tied'] = temporary['grads']['ranking'][4+i]
            temporary['query_part'] = temporary['split_gradients'][i]
            temporary['gallery_part'] = temporary['split_gradients'][2+i]
            _match(torch, temporary['tied'], temporary['query_part']+temporary['gallery_part'], 'tied=query+gallery '+name)
            decomposition[name] = {key: gradient_fact(torch, temporary[key+'_part'] if key != 'tied' else temporary[key], name+key)
                                   for key in ('query','gallery','tied')}
        model.zero_grad(set_to_none=True)
        state['A'].grad = state['C'].grad = None
        temporary['mutant'] = trainer.raw_features(context, state, features.detach())
        _match(torch, temporary['raw'].detach(), temporary['mutant'].detach(), 'detached original raw', exact=True)
        temporary['mutant_mse'], temporary['mutant_rank'], mutant_facts = trainer.loss_terms(
            context, state, temporary['mutant'], anchors, valid)
        require(mutant_facts == facts, 'detached original membership differs')
        for label in ('mse','rank'):
            _match(torch, temporary[label], temporary['mutant_'+label], 'detached '+label, exact=True)
        temporary['mutant_gradients'] = _gradients(torch, temporary['mutant_mse'], temporary['mutant_rank'], temporary['members'])
        for label, grads in temporary['mutant_gradients'].items():
            require(all(g is None or torch.count_nonzero(g).item() == 0 for g in grads[:4]),
                    'detached original query reached encoder')
            for i, name in enumerate(('A','C')):
                _match(torch, temporary['grads'][label][4+i], grads[4+i], 'detached '+label+name)
        require(all(p.grad is None and p.requires_grad is (n in MLP) for n,p in model.named_parameters()) and
                all(p.grad is None and not p.requires_grad for p in state['head_object'].parameters()),
                'four/444 encoder or frozen head roles/gradients differ')
        require(source.numerical_flags() == context['flags'] and
                torch.equal(temporary['cpu_rng'], torch.random.get_rng_state()) and
                all(torch.equal(a,b) for a,b in zip(temporary['cuda_rng'],torch.cuda.get_rng_state_all(),strict=True)),
                'paired original numerical flags/RNG changed')
        require(frozen_before == trainer.fingerprint(context, {k: state[k] for k in ('A','C','teachers','target','means','mu_train')}) and
                head_before == trainer.fingerprint(context, dict(state['head_object'].state_dict())), 'GPU fixed readout bytes changed')
        result = {'view':view, 'images':image_facts, 'pairing':pairing, 'gradients':gradients,
            'A_C_decomposition':decomposition, 'membership':facts, 'mse':float(temporary['mse'].detach()),
            'rank':float(temporary['rank'].detach()), 'loss':float((temporary['mse']+temporary['rank']).detach()),
            'live_control_core_seconds':control_seconds, 'connected_core_seconds':connected_seconds,
            'detached_forward_identical':True, 'detached_encoder_gradients_absent':True,
            'detached_A_C_gradients_match':True, 'frozen_encoder_gradients_absent':444,
            'GPU_readout_unchanged':True, 'global_RNG_unchanged':True}
        released.extend(trainer.tensor_weakrefs(context, {k:v for k,v in temporary.items()
                        if k not in ('state','query','gallery','members')}))
        del features, state, query, gallery, grads
    except Exception as error:
        traceback.print_exception(error, file=sys.stderr)  # Preserve the caught failure before cleanup can mask it.
        failure = error.with_traceback(None)
    finally:
        # Clear loop aliases on rejection too, before the owned dictionaries.
        features = state = query = gallery = grads = p = None
        temporary.clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _diagnose_release(released, 'view '+view)
        require(all(ref() is None for ref in released), 'view graph/GPU readout/pixel lifetime survived release')
    if failure is not None:
        raise failure
    result['temporary_references_released'] = True
    return result


def qualify(args):
    started = time.perf_counter()
    require(not sys.flags.optimize and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID','')), 'original enclosing single-GPU unit required')
    prospective = authenticate(args.authority, args.authority_sha256, args.output)
    require(sha(TRAIN_ROOT / 'execution.json') == TRAIN_EXECUTION and
            json.loads((TRAIN_ROOT / 'execution.json').read_text(), object_pairs_hook=pairs) == TRAIN_CODE,
            'original source-v5 execution differs')
    for name, digest in TRAIN_CODE.items():
        require(sha(TRAIN_ROOT / name) == digest, 'original trainer closure differs')
    # Use the original authenticated loader, preserving module __file__/spec.
    import importlib.util
    spec = importlib.util.spec_from_file_location('_actual_original_trainer', TRAIN_ROOT / 'train_siglip2_identity_diversity.py')
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    historical_args = SimpleNamespace(execution_sha256=TRAIN_EXECUTION, authority=TRAIN_ROOT / 'authority-cpu-v5.json',
        authority_sha256=CPU_AUTHORITY_SHA, output=args.output, phase='cpu', arm='control', seed=179061)
    context = trainer.authority(historical_args)
    record = trainer.admit_terminal(context, CPU_UNIT, 'cpu', 'control', 179061)
    qualification = select_initializer(record)
    require(all(context['guards'].get(p,h) == h for p,h in prospective.items()), 'prospective guard conflict')
    context['guards'].update(prospective)
    context['started'] = started
    context['fit_context']['unit_started'] = started
    source, legacy = context['legacy']['source_driver'], context['legacy']
    before = source.cgroup_memory()
    for value in (before,): context['old'].zero_events(value)
    unit = Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before, unit)
    packages = source.package_origins(legacy['prior'])
    require(packages == legacy['selected']['packages'], 'original package preflight differs')
    legacy['prior']['packages'] = packages
    import torch
    require(not torch.cuda.is_initialized() and torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            'historical admission must precede CUDA')
    flags = record['numerical_flags']
    require(str(Path(sys.executable).resolve()) == record['invocation']['python'] and
            sha(Path(sys.executable).resolve()) == record['invocation']['python_sha256'] and
            sys.version == record['invocation']['python_version'], 'qualified original interpreter differs')
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original numerical flags differ')
    legacy['flags'] = flags
    pack = legacy['selected']['launch']['helpers']['packing']
    legacy['packing'] = legacy['genuine'].load_helper('_quadratic_packing', pack['path'], pack['sha256'], context['guards'])
    connected = SimpleNamespace(**runpy.run_path(str(HERE / 'connected_residual_readout.py')))
    owned, released, failure, results = {}, [], None, []
    release_names = {}
    processor_cache = None
    try:
        owned['cpu_state'], ident = _load_initializer(trainer, context, qualification)
        trainer.integrity(context, owned['cpu_state'], ident)
        require(trainer.canonical_initial_witness(context, owned['cpu_state'], ident) == qualification['canonical_initial'],
                'original CPU canonical initializer witness differs')
        batch, anchors, valid = first_microbatch(trainer, owned['cpu_state'])
        owned['cpu_rng'] = torch.random.get_rng_state().clone()
        owned['model'], owned['processor'], roles = source.fresh_source(legacy['prior'])
        processor_cache = _processor_cache(owned['processor'], context['guards'], empty=True)
        require(source.model_facts(owned['model'], owned['processor'], roles, packages) ==
                owned['cpu_state']['provenance']['encoder']['source_proof']['runtime'], 'original full source runtime differs')
        legacy_probe.select_mlp(owned['model'], legacy['prior']['expected'])
        owned['model'].to('cuda').eval()
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one genuine CUDA encoder required')
        context['live_model'] = weakref.ref(owned['model'])
        release_names.update({id(owned[n]): n for n in ('model', 'processor')})
        release_names.update({id(p): 'model.'+n for n,p in owned['model'].named_parameters()})
        release_names.update({id(p): 'model.'+n for n,p in owned['model'].named_buffers()})
        released.extend((weakref.ref(owned['model']), weakref.ref(owned['processor'])))
        released.extend(weakref.ref(p) for p in owned['model'].parameters())
        released.extend(weakref.ref(p) for p in owned['model'].buffers())
        for view in trainer.VIEWS:
            results.append(view_probe(trainer, context, owned['cpu_state'], owned['model'], owned['processor'],
                                      connected, anchors, valid, view))
        # Restore original factory roles only for its exact complete runtime check.
        for role, (_, parameter) in zip(roles, owned['model'].named_parameters(), strict=True):
            parameter.requires_grad_(role['role'] == 'trainable')
        del parameter
        # The original position_ids validator constructs its reference on CPU.
        owned['model'].to('cpu')
        release_names.update({id(p): 'model.'+n for n,p in owned['model'].named_parameters()})
        release_names.update({id(p): 'model.'+n for n,p in owned['model'].named_buffers()})
        released.extend(weakref.ref(p) for p in owned['model'].parameters())
        released.extend(weakref.ref(p) for p in owned['model'].buffers())
        require(source.model_facts(owned['model'], owned['processor'], roles, packages) ==
                owned['cpu_state']['provenance']['encoder']['source_proof']['runtime'],
                'all448 source bytes/config/processor/nonpersistent buffers changed')
    except Exception as error:
        traceback.print_exception(error, file=sys.stderr)  # Preserve the caught failure before cleanup can mask it.
        failure = error.with_traceback(None)
    finally:
        parameter = None
        try:
            if processor_cache is not None:
                require(_processor_cache(owned['processor'], context['guards']) is processor_cache,
                        'processor cache wrapper changed before release')
                processor_cache.cache_clear()
                require(processor_cache.cache_info().currsize == 0, 'processor cache cleanup failed')
        except Exception as error:
            traceback.print_exception(error, file=sys.stderr)
            if failure is None:
                failure = error.with_traceback(None)
        owned.pop('model', None)
        owned.pop('processor', None)
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _diagnose_release(released, 'encoder/processor', release_names)
        require(all(ref() is None for ref in released), 'encoder/processor tensor lifetime survived release')
        if 'cpu_state' in owned:
            if 'cpu_rng' in owned: torch.random.set_rng_state(owned.pop('cpu_rng'))
            require(source.numerical_flags() == flags, 'exit original numerical flags changed')
            trainer.integrity(context, owned['cpu_state'], ident)
            require(trainer.fingerprint(context, trainer.payload(context, owned['cpu_state'], ident)) == PAYLOAD_SHA,
                    'complete CPU initializer changed after probe')
            cpu_refs = trainer.tensor_weakrefs(context, {
                'static': {k:owned['cpu_state'][k] for k in trainer.STATIC_KEYS},
                'A': owned['cpu_state']['A'], 'C': owned['cpu_state']['C'],
                'head_parameters':dict(owned['cpu_state']['head_object'].named_parameters()),
                'head_buffers':dict(owned['cpu_state']['head_object'].named_buffers())})
            cpu_refs.extend(weakref.ref(owned['cpu_state'][n]) for n in
                            ('head_object','optimizer_object','scaler_object'))
            trainer.release(context, owned.pop('cpu_state'))
            context['initial'].clear()
            gc.collect()
            require(all(ref() is None for ref in cpu_refs), 'restored CPU state lifetime survived release')
        trainer.exit_rehash(context)  # Original complete fresh uncached reader, including prospective guards.
    if failure is not None:
        raise failure
    torch.cuda.synchronize()
    after = source.cgroup_memory()
    legacy['selected']['genuine']['reference'].admit_cgroup(after, unit)
    for value in (before, after): context['old'].zero_events(value)
    peak = torch.cuda.max_memory_allocated()
    elapsed = time.perf_counter()-started
    require(before['path'] == after['path'] and elapsed < POLICY['seconds'] and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8*1024**2 and
            peak < POLICY['cuda_allocated_bytes_exclusive'], 'whole discarded-witness resource envelope exceeded')
    args.output.mkdir()
    receipt = {'schema':'actual-objective-encoder-gradients-v1', 'pass':True, 'engineering_only':True,
        'model_fit_qualified':False, 'cost_qualified':False, 'state_reuse':False, 'state_reuse_eligible':False,
        'quality_read':False, 'training_updates':0, 'batch':batch, 'anchors':anchors, 'full_valid':valid,
        'regression_denominator_rows':128, 'ranking_denominator':126, 'views':results,
        'gradient_parameters':list(MLP), 'frozen_encoder_tensors':444, 'unchanged_encoder_tensors':448,
        'initializer':INITIALIZER, 'initializer_payload_sha256':PAYLOAD_SHA, 'CPU_unit':CPU_UNIT,
        'trainer_execution_sha256':TRAIN_EXECUTION, 'trainer_code':TRAIN_CODE,
        'authority_sha256':args.authority_sha256, 'input_guards':context['guards'], 'resource_policy':POLICY,
        'numerical_flags':flags, 'peak_cuda_allocated_bytes':peak, 'whole_seconds':elapsed,
        'cgroup_before':before, 'cgroup_after':after, 'invocation_id':os.environ['INVOCATION_ID'],
        'exit_rehash_pass':True, 'all_temporary_references_released':True,
        'terminal_exit_and_both_locks_require_parent_receipt':True}
    with (args.output / 'receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    require(time.perf_counter()-started < POLICY['seconds'], 'receipt included deadline exceeded')
    print(json.dumps({'pass':True, 'engineering_only':True, 'whole_seconds':elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    qualify(parser.parse_args())


if __name__ == '__main__':
    main()
