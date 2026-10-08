#!/usr/bin/env python3
"""One fixed selection-only B32/B6/B6 observation; native gate UNRUN.

CLI: --execution-sha256 SHA --authority FILE --authority-sha256 SHA --output NEWDIR.
execution.json contains exactly FILES. FILE={path:canonical absolute file,sha256}.
Authority is exactly LAUNCH_KEYS: historical={source:FILE,execution:FILE,authority:FILE}
pins the unchanged v3 diagnostic and its original launch. Its prepare, Sources.admit,
terminal_admission and origin_audit authenticate all four old endpoint/export byte
artifacts and the six original UNITs for integrity ONLY. No candidate payload is
loaded, no candidate arrays are interpreted, and no quality function is called.
Historical guards finish before prospective guards are promoted. control={bundle:FILE,
export:FILE} must equal accepted control061; images is the ordered32 FILE list from
query batch53. output binds NEWDIR. Root supplies actual own2/authority/output hashes.

Only these selection images are decoded, never original cache groups containing VAL.
The genuine public loader owns full448, processor LRU and readout state. The copied
finite inference body has exact AST correspondence, with only pixel checking and
CPU capture added. No original functions/globals/math are replaced. B32 then B6 twice
is exactly44 image-forwards; accepted RGB/pixels/outputs and original cache must match
before the B6 comparison. Same B6 readout for cache, B32 gather and both B6 results.
All state and images are released before fresh whole-exit hashes/origins/maps/caps.
Only a non-reusable JSON observation is emitted after success. Parent must bind the
normal enclosing terminal exit and both locks; no receipt alone qualifies execution.
700s/8GiB/zero swap/CUDA<10GB includes admission, construction, all hashes and exit.
KILL and original v1-v3 FAIL remain immutable. Drift is a fixed-witness observation,
not the cause of the original tail, quality evidence or a serving correction.
"""
if not __debug__:
    raise SystemExit('optimized mode forbidden')

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

FILES = {'observe_connected_control_batch_execution.py', 'test_connected_control_batch_execution.py'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'historical', 'control', 'images', 'output',
               'resource_policy', 'both_locks_held', 'candidate_status',
               'qualification_eligible', 'state_reuse_eligible'}
HISTORICAL = {'source': '3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434',
              'execution': '6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4',
              'authority': '179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d'}
CONNECTED_SHA = '79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b'
EXPORT_SHA = 'db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407'
SUBSET_FIT = [13220, 13221, 13222, 13233, 13234, 13235]
LIMITS = {'seconds': 700, 'host_bytes': 8*1024**3, 'swap_bytes': 0,
          'cuda_visible_devices': '0', 'cuda_allocated_bytes_exclusive': 10_000_000_000}
KEY = 'control-179061'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def authenticated(fact, guards, *, keep=False):
    """Bootstrap and prospective FILEs; image admission remains a separate allowlist."""
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and
            type(fact['path']) is str and type(fact['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'exact FILE required')
    path = Path(fact['path'])
    require(path.is_absolute() and str(path) == fact['path'] and path.resolve() == path and
            path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    before = path.stat()
    digest = hashlib.sha256()
    raw = None
    with path.open('rb') as stream:
        if keep:
            raw = stream.read(64*1024**2+1)
            require(len(raw) <= 64*1024**2, 'metadata/source exceeds64MiB')
            digest.update(raw)
        else:
            while block := stream.read(1024**2):
                digest.update(block)
                os.posix_fadvise(stream.fileno(), stream.tell() - len(block), len(block), os.POSIX_FADV_DONTNEED)
        after = os.fstat(stream.fileno())
    require(before == after == path.stat(), 'FILE changed during authentication')
    require(digest.hexdigest() == fact['sha256'], 'FILE SHA differs: '+str(path))
    require(guards.setdefault(str(path), fact['sha256']) == fact['sha256'], 'conflicting FILE')
    return raw if keep else path


def read_json(fact, guards):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(authenticated(fact, guards, keep=True), object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def load_source(name, fact, guards):
    raw = authenticated(fact, guards, keep=True)
    require(name not in sys.modules, 'source namespace already owned')
    spec = importlib.util.spec_from_file_location(name, fact['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, fact['path'], 'exec', dont_inherit=True), vars(module))
    except BaseException:
        del sys.modules[name]
        raise
    return module


def remove_source(module):
    require(sys.modules.get(module.__name__) is module, 'source registry ownership changed')
    del sys.modules[module.__name__]


def prepare(args):
    guards = {}
    root = Path(__file__).absolute().parent
    code = read_json({'path': str(root/'execution.json'), 'sha256': args.execution_sha256}, guards)
    require(type(code) is dict and code.keys() == FILES, 'exact2 observer closure required')
    for name, digest in code.items():
        authenticated({'path': str(root/name), 'sha256': digest}, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    require(type(launch) is dict and launch.keys() == LAUNCH_KEYS and
            launch['schema'] == 'connected-control-batch-execution-launch-v1' and
            launch['execution_sha256'] == args.execution_sha256 and launch['resource_policy'] == LIMITS and
            launch['both_locks_held'] is True and launch['candidate_status'] == 'KILL' and
            launch['qualification_eligible'] is False and launch['state_reuse_eligible'] is False,
            'exact control observer authority/caps required')
    output = Path(args.output)
    require(str(output) == launch['output'] and output.is_absolute() and
            output.parent.resolve() == output.parent and not output.exists(), 'exclusive bound NEWDIR required')
    history = launch['historical']
    require(type(history) is dict and history.keys() == HISTORICAL.keys() and
            all(history[k]['sha256'] == h for k, h in HISTORICAL.items()), 'pinned original v3 admission required')
    require(Path(history['source']['path']).name == 'diagnose_connected_gallery_freshness.py' and
            Path(history['execution']['path']) == Path(history['source']['path']).parent/'execution.json',
            'separate original execution/source required')
    for fact in history.values():
        authenticated(fact, guards)
    return {'launch': launch, 'guards': guards, 'code': code, 'output': output}


def fixed_batch(context):
    panel = context['partition']['panels']['selection']
    batch = context['exports'][KEY]['images'][53]
    indices = panel['query'][1696:1728]
    rows = batch['rows']
    require(batch['role'] == 'query' and len(rows) == len(indices) == 32 and
            [r['panel_ordinal'] for r in rows] == indices and
            [r['original_row'] for r in rows] == [panel['original_rows'][i] for i in indices] and
            [r['original_row'] for r in rows[26:]] == SUBSET_FIT,
            'fixed query batch53/subset roles differ')
    forbidden = set(context['partition']['panels']['train']['original_rows'])
    forbidden.update(context['partition']['panels']['validation']['original_rows'])
    require(not forbidden.intersection(r['original_row'] for r in rows), 'TRAIN/VAL image denied')
    for row in rows:
        original = context['fit']['rows'][row['original_row']]
        require(row['role'] == 'query' and all(row[k] == original[k] for k in
                ('train_row', 'relative_path', 'image_sha256')), 'canonical FIT image binding differs')
    return batch


def bind_control(prospective, context):
    launch = prospective['launch']
    endpoint = next(e for e in context['score']['launch']['endpoints'] if (e['seed'], e['arm']) == (179061, 'control'))
    require(launch['control'] == {'bundle': endpoint['bundle'],
            'export': context['score']['launch']['exports'][KEY]['receipt']} and
            launch['control']['export']['sha256'] == EXPORT_SHA and
            context['launch']['sources']['connected']['sha256'] == CONNECTED_SHA,
            'accepted control061 only')
    batch = fixed_batch(context)
    require(launch['images'] == [{'path': r['path'], 'sha256': r['image_sha256']} for r in batch['rows']],
            'exact ordered32 selection image FILEs required')
    for fact in launch['images']:
        authenticated(fact, prospective['guards'])
    for path, digest in prospective['guards'].items():
        require(context['guards'].setdefault(path, digest) == digest, 'prospective guard conflict')
    return endpoint, batch


def check_capture_ast(original_raw, observer_raw=None):
    """Invert only named binding/capture seams, then compare the complete finite body."""
    original = next(n for n in ast.parse(original_raw).body if isinstance(n, ast.FunctionDef) and n.name == 'inference_outputs')
    own = Path(__file__).read_bytes() if observer_raw is None else observer_raw
    node = next(n for n in ast.parse(own).body if isinstance(n, ast.FunctionDef) and n.name == 'capture_inference_outputs')
    signature = ast.parse('def f(connected, endpoint, images, expected_pixels=None): pass').body[0].args
    require(ast.dump(node.args) == ast.dump(signature), 'capture signature changed')
    checks = [i for i, n in enumerate(node.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
              isinstance(n.value.func, ast.Name) and n.value.func.id == 'require_pixels']
    require(len(checks) == 1 and ast.unparse(node.body[checks[0]]) == 'require_pixels(pixels, expected_pixels)' and
            checks[0]+1 == next(i for i, n in enumerate(node.body) if isinstance(n, ast.With)),
            'exact pre-forward pixel seam required')
    del node.body[checks[0]]
    returned = node.body[-1].value
    require(isinstance(returned, ast.Tuple) and len(returned.elts) == 4 and
            [ast.unparse(n) for n in returned.elts[1:]] == ['pixels', 'pooled.cpu()', 'features.cpu()'],
            'exact CPU-only capture seam required')
    node.body[-1].value = returned.elts[0]
    class Restore(ast.NodeTransformer):
        def visit_Attribute(self, value):
            if isinstance(value.value, ast.Name) and value.value.id == 'connected' and value.attr in (
                    'bound_file', 'encoder_facts', 'inference_readout_tree', 'fullfeature_raw_features'):
                return ast.copy_location(ast.Name(id=value.attr, ctx=value.ctx), value)
            return self.generic_visit(value)
    node = Restore().visit(node)
    node.name = original.name
    node.args = original.args
    require(ast.dump(node, include_attributes=False) == ast.dump(original, include_attributes=False),
            'complete original inference math/guards AST differs')


def tensor_bytes(value):
    return memoryview(value.detach().cpu().contiguous().reshape(-1).view(sys.modules['torch'].uint8).numpy()).tobytes()


def require_pixels(actual, expected):
    if expected is not None:
        require(actual.shape == expected.shape and actual.dtype == expected.dtype and
                tensor_bytes(actual) == tensor_bytes(expected), 'B6 pixels differ from B32slice26:32')


def capture_inference_outputs(connected, endpoint, images, expected_pixels=None):
    import torch
    from torch.nn import functional as F
    modules,device = endpoint['modules'],endpoint['device']
    original,source = modules['train_siglip2_substrate_adaptation.py'],modules['qualify_siglip2_substrate_cpu.py']
    for module in modules.values():
        connected.bound_file({},module.__file__,endpoint['guards'][module.__file__])
    require(0 < len(images) <= 32 and source.numerical_flags() == endpoint['flags'], 'serving batch/numerics differ')
    require(endpoint['encoder_identity'] == endpoint['manifest']['encoder_identity'] and
            endpoint['vision_sha256'] == endpoint['manifest']['vision_sha256'] and
            all(p.grad is None and not p.requires_grad and p.dtype == torch.float32 and p.device.type == device
                for p in endpoint['head_object'].parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in endpoint['head_object'].modules()), 'serving authenticated encoder/frozen head roles/hooks differ')
    require(connected.encoder_facts(endpoint,original,source,endpoint['manifest']['environment']['packages'],serving=True)['vision_sha256'] ==
            endpoint['vision_sha256'] and original.fingerprint(connected.inference_readout_tree(endpoint)) == endpoint['readout_sha256'],
            'current .data updated encoder/readout/role substitution rejected')
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else []
    pixels = endpoint['processor_object'](images=images,return_tensors='pt')['pixel_values']
    require(pixels.shape == (len(images),3,256,256) and pixels.dtype == torch.float32 and torch.isfinite(pixels).all().item(),
            'owned processor pixels differ')
    require_pixels(pixels, expected_pixels)
    with torch.no_grad():
        with torch.autocast(device,dtype=torch.float16,enabled=device == 'cuda'):
            pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device,enabled=False):
            features = F.normalize(pooled.float(),dim=1)
            raw = connected.fullfeature_raw_features(features,endpoint['head_object'],endpoint['A'],endpoint['means'],endpoint['C'],
                endpoint['mu_train'],endpoint['arm'],modules['quadratic_readout.py'],modules['prototype_residual_readout.py'])
            require((raw.norm(dim=1) > 0).all().item(), 'nonzero portable raw required')
            unit = F.normalize(raw,dim=1)
            packed = modules['joint_relational_compaction.py'].pack_int8_unit_embeddings(unit.cpu())
    require(torch.equal(cpu_rng,torch.random.get_rng_state()) and
            all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if device == 'cuda' else [],strict=True)),
            'serving complete RNG changed')
    return ({'raw':raw.cpu(),'unit':unit.cpu(),'codes':packed.codes.cpu(),'inverse_norms':packed.inverse_norms.cpu(),'wire':packed.to_bytes()}, pixels, pooled.cpu(), features.cpu())


def check_endpoint(connected, state):
    import torch
    original = state['modules']['train_siglip2_substrate_adaptation.py']
    source = state['modules']['qualify_siglip2_substrate_cpu.py']
    require(state['arm'] == 'control' and state['device'] == 'cuda' and
            source.numerical_flags() == state['flags'] and
            connected.encoder_facts(state, original, source, state['manifest']['environment']['packages'], serving=True)['vision_sha256'] ==
            state['manifest']['vision_sha256'] == state['vision_sha256'] and
            state['encoder_identity'] == state['manifest']['encoder_identity'] and
            original.fingerprint(connected.inference_readout_tree(state)) == state['readout_sha256'],
            'current full448/readout/processor/flags differ')
    require(all(p.grad is None and not p.requires_grad and p.dtype == torch.float32 and p.device.type == 'cuda'
                for p in state['head_object'].parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in state['head_object'].modules()), 'frozen head roles/hooks differ')


def readout(connected, owner, state, features):
    import torch
    check_endpoint(connected, state)
    modules = state['modules']
    require(features.shape in ((32, 1152), (6, 1152)) and features.dtype == torch.float32 and
            torch.isfinite(features).all().item(), 'finite fixed readout inputs required')
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        raw = connected.fullfeature_raw_features(features.to('cuda'), state['head_object'], state['A'], state['means'],
            state['C'], state['mu_train'], state['arm'], modules['quadratic_readout.py'], modules['prototype_residual_readout.py'])
        output = owner.packed_outputs({'packing': modules['joint_relational_compaction.py']}, raw)
    del raw
    check_endpoint(connected, state)
    return output


def compare_bytes(actual, expected):
    require(actual.keys() == expected.keys(), 'comparison member inventory differs')
    return {k: {'exact': actual[k] == expected[k],
                'different_bytes': sum(a != b for a, b in zip(actual[k], expected[k])) + abs(len(actual[k])-len(expected[k]))}
            for k in actual}


def exact(actual, expected, message):
    require(all(v['exact'] for v in compare_bytes(actual, expected).values()), message)


def interpretation(pooled, features, outputs):
    if pooled:
        return 'FIXED_WITNESS_ENCODER_BATCH_DRIFT'
    if features:
        return 'FIXED_WITNESS_NORMALIZATION_BATCH_DRIFT'
    require(not outputs, 'same features but changed same-B6 readout')
    return 'FIXED_WITNESS_FALSIFIED'


def observe(diagnostic, context, sources, state, batch, boundary):
    """Small real execution seam: fixed schedule, admission stops and byte decisions."""
    from PIL import Image
    import torch
    connected = sources.modules['connected']
    original = sources.modules['original']
    images = []
    cache = fresh = values = pixels = pooled = features = gathered = None
    first = None
    try:
        rgb = hashlib.sha256()
        for row in batch['rows']:
            boundary()
            authenticated({'path': row['path'], 'sha256': row['image_sha256']}, context['guards'])
            with Image.open(row['path']) as opened:
                image = opened.convert('RGB')
            images.append(image)
            require(image.size == (256, 256), 'fixed image dimensions differ')
            rgb.update(str(image.size).encode())
            rgb.update(image.tobytes())
        require(rgb.hexdigest() == batch['rgb_sha256'], 'accepted B32 RGB proof differs')
        boundary()
        values, pixels, pooled, features = capture_inference_outputs(connected, state, images)
        require(original.fingerprint(pixels) == batch['pixels_sha256'] and
                original.fingerprint(values) == batch['outputs_sha256'], 'accepted B32 pixel/output proof differs')
        require(torch.isfinite(pooled).all().item() and torch.isfinite(features).all().item(), 'finite captures required')
        boundary()
        endpoint = next(e for e in context['score']['launch']['endpoints'] if diagnostic.label(e) == KEY)
        fresh = diagnostic.fresh_values(context, sources, endpoint)
        indices = [r['panel_ordinal'] for r in batch['rows']]
        expected = {k: fresh[k][indices] for k in ('raw', 'unit', 'codes', 'inverse_norms')}
        expected['wire'] = b''.join(fresh['wire'][i*130:(i+1)*130] for i in indices)
        exact(diagnostic.output_bytes(values), diagnostic.output_bytes(expected), 'accepted B32 descriptor bytes differ')
        del expected
        fresh = None
        cache = sources.modules['baseline'].cache_rows(context, [r['original_row'] for r in batch['rows']])
        connected.mapping_absent(context['launch']['original_cache']['path'])
        exact({'features': tensor_bytes(features)}, {'features': tensor_bytes(cache)}, 'B32 original cache features differ')
        cached32 = readout(connected, sources.modules['quadratic_owner'], state, cache)
        exact(diagnostic.output_bytes(values), diagnostic.output_bytes(cached32), 'B32 original cache readout differs')
        del cached32
        baseline = {'pooled': tensor_bytes(pooled[26:32]), 'features': tensor_bytes(features[26:32])}
        gathered = diagnostic.output_bytes(readout(connected, sources.modules['quadratic_owner'], state, features[26:32]))
        cached6 = diagnostic.output_bytes(readout(connected, sources.modules['quadratic_owner'], state, cache[26:32]))
        exact(gathered, cached6, 'same B6 cache/gather readout differs')
        expected_pixels = pixels[26:32].clone()
        b32_facts = {'rgb_sha256': rgb.hexdigest(), 'pixels_sha256': original.fingerprint(pixels),
                     'outputs_sha256': original.fingerprint(values), 'cache_features_exact': True,
                     'cache_readout_exact': True}
        values = pixels = pooled = features = cache = None
        for repeat in range(2):
            boundary()
            values, pixels, pooled, features = capture_inference_outputs(connected, state, images[26:32], expected_pixels)
            require(torch.isfinite(pooled).all().item() and torch.isfinite(features).all().item(), 'finite B6 captures required')
            same6 = diagnostic.output_bytes(readout(connected, sources.modules['quadratic_owner'], state, features))
            exact(diagnostic.output_bytes(values), same6, 'public vs original same B6 packing differs')
            current = {'pooled': tensor_bytes(pooled), 'features': tensor_bytes(features), **same6}
            if repeat == 0:
                first = current
            else:
                exact(current, first, 'B6 repeatability differs; stop')
            values = pixels = pooled = features = None
            boundary()
        delta = compare_bytes(first, {**baseline, **gathered})
        decision = interpretation(not delta['pooled']['exact'], not delta['features']['exact'],
                                  any(not delta[k]['exact'] for k in gathered))
        del expected_pixels
        return {'decision': decision, 'b32': b32_facts, 'b6_repeat_exact': True,
                'same_b6_cache_gather_exact': True, 'differences': delta,
                'capture_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in first.items()},
                'encoder_forward_sizes': [32, 6, 6], 'image_forwards': 44,
                'original_tail_cause_established': False, 'serving_correction_authorized': False}
    finally:
        values = pixels = pooled = features = gathered = fresh = cache = None
        for image in images:
            image.close()
        images.clear()
        gc.collect()


def exact_four(context, sources, origins):
    m = sources.modules
    authority = read_json(context['launch']['runtime']['native_authority'], context['guards'])
    require(authority['proof']['sha256'] == m['nearest'].NATIVE_PROOF_PINS['proof'], 'original native proof differs')
    proof = read_json(authority['proof'], context['guards'])
    site = Path(proof['authority']['installed_site_root'])
    expected = {str(site/name): fact['sha256'] for name, fact in proof['comparison']['selected_members'].items()}
    known = set(context['source_cpu']['origins']['files']) | set(context['warm']['origins']['files'])
    require(len(expected) == 4 and set(origins['files'])-known == set(expected) and
            set(expected) <= set(origins['native_files']) and
            all(origins['files'][p] == h for p, h in expected.items()), 'native supplement must be exact original four')


def rehash(guards):
    for path, digest in tuple(guards.items()):
        authenticated({'path': path, 'sha256': digest}, {})


def memory_snapshot(path, phase):
    """Bounded scalar diagnostics from the already admitted cgroup; no cap decision."""
    values = {}
    for name, keys in (
        ('memory.current', ()), ('memory.peak', ()),
        ('memory.stat', ('anon', 'file', 'kernel')),
        ('memory.events', ('low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill')),
        ('memory.swap.current', ()), ('memory.swap.peak', ())):
        with (Path(path)/name).open('rb') as stream:
            raw = stream.read(16*1024+1)
        require(len(raw) <= 16*1024, 'memory diagnostic exceeds16KiB')
        if keys:
            pairs = [line.split() for line in raw.decode('ascii').splitlines()]
            fields = dict(pairs)
            require(len(fields) == len(pairs), 'duplicate memory diagnostic key')
            for key in keys:
                values[name+'.'+key] = int(fields[key])
        else:
            values[name] = int(raw)
    require(all(v >= 0 for v in values.values()), 'nonnegative memory diagnostic required')
    print(json.dumps({'diagnostic': 'memory_snapshot', 'phase': phase, 'path': str(path), **values},
                     sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)


def elapsed_cap(started):
    require(time.perf_counter()-started < 700, 'whole700-second observer cap reached')


def cuda_ownership_snapshot():
    """Partial Python tensor metadata only; native allocator owners may be invisible."""
    torch = sys.modules.get('torch')
    if torch is None or not torch.cuda.is_initialized():
        return
    allocated = torch.cuda.memory_allocated()
    objects = gc.get_objects() if allocated else []
    samples = []
    seen = scanned = 0
    for value in objects:
        if scanned == 100000:
            break
        scanned += 1
        if not issubclass(type(value), torch.Tensor) or value.device.type != 'cuda':
            continue
        seen += 1
        if len(samples) < 16:
            samples.append({'type': (type(value).__module__+'.'+type(value).__qualname__)[:128],
                'shape': list(value.shape[:8]), 'rank': len(value.shape),
                'dtype': str(value.dtype)[:128], 'device': str(value.device)[:128],
                'owner_types': sorted({type(owner).__name__[:128] for owner in gc.get_referrers(value)[:8]
                                       if owner is not objects})})
    print(json.dumps({'diagnostic': 'cuda_ownership_snapshot', 'status': 'UNACCEPTED',
        'phase': 'before_final_state', 'allocated_bytes': allocated,
        'reserved_bytes': torch.cuda.memory_reserved(), 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'python_gc_only': True, 'gc_objects': len(objects), 'gc_objects_scanned': scanned,
        'gc_scan_complete': bool(allocated) and scanned == len(objects),
        'cuda_tensors_seen': seen, 'tensor_samples': samples},
        sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)


def run(args):
    started = time.perf_counter()
    prospective = context = diagnostic = sources = state = audit = receipt = None
    admitted = False
    error = None
    callbacks = []
    checks = []
    serving_checks = []
    require(not {'torch', 'numpy', 'PIL', 'transformers', 'torchvision', 'sfora'}.intersection(sys.modules),
            'native imports must follow explicit historical admission')
    try:
        prospective = prepare(args)
        historical = prospective['launch']['historical']
        diagnostic = load_source('_control_batch_historical', historical['source'], prospective['guards'])
        original_args = SimpleNamespace(execution_sha256=historical['execution']['sha256'],
            authority=Path(historical['authority']['path']), authority_sha256=historical['authority']['sha256'], output=args.output)
        context = diagnostic.prepare(original_args)
        sources = diagnostic.Sources(context)
        sources.admit()
        invocations = diagnostic.terminal_admission(context, sources)
        require(len(invocations) == 6, 'six distinct original UNITs required')
        audit = diagnostic.origin_audit(context, sources)
        audit()
        m = sources.modules
        guard = m['evaluator'].source_live_guard
        checks.append(guard(diagnostic, historical['source']['sha256'], context['guards'], class_name='Sources'))
        endpoint, batch = bind_control(prospective, context)
        checks.append(guard(sys.modules[__name__], prospective['code'][Path(__file__).name], context['guards']))
        check_capture_ast(authenticated(context['launch']['sources']['connected'], context['guards'], keep=True))
        elapsed_cap(started)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
                re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')) and
                os.environ['INVOCATION_ID'] not in invocations, 'fresh enclosing CUDA0 UNIT required')
        prior = context['score']['invocation']
        python = Path(sys.executable).resolve()
        require(str(python) == prior['python'] and m['extract'].sha(python) == prior['python_sha256'] and
                sys.version == prior['python_version'], 'original interpreter required')
        source = m['source_driver']
        before = source.cgroup_memory()
        unit = Path(before['path']).name.removesuffix('.service')
        initializer = m['initializer']
        initializer.admit_cgroup(before, unit)
        admitted = True
        memory_snapshot(before['path'], 'after_admission')
        import torch
        require(not torch.cuda.is_initialized(), 'CUDA initialized before admission')
        flags = copy.deepcopy(context['exports'][KEY]['numerical_flags'])
        require(flags == context['cpu']['numerical_flags'] and flags['cudnn_allow_tf32'] is True and
                flags['matmul_allow_tf32'] is False, 'accepted live flags required')
        torch.set_num_threads(flags['threads'])
        if torch.get_num_interop_threads() != flags['interop_threads']:
            torch.set_num_interop_threads(flags['interop_threads'])
        require(source.numerical_flags() == flags, 'live numerical flags differ')
        rng = (torch.random.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])
        require(torch.cuda.device_count() == 1, 'one CUDA device required')
        packing = sources.load('packing')
        sources.checks.append(guard(packing, context['launch']['sources']['packing']['sha256'], context['guards']))
        budget = SimpleNamespace(check=lambda: elapsed_cap(started))
        def final_resources():
            try:
                cuda_ownership_snapshot()
            finally:
                diagnostic.final_state(budget, source, initializer, before, rng, flags, receipt)
        callbacks.append(final_resources)
        connected = m['connected']
        def boundary():
            elapsed_cap(started)
            sources.guard()
            for check in checks + serving_checks:
                check()
            require(source.numerical_flags() == flags and torch.equal(torch.random.get_rng_state(), rng[0]) and
                    len(torch.cuda.get_rng_state_all()) == len(rng[1]) and
                    all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), rng[1], strict=True)),
                    'complete RNG/flags changed')
            require(torch.cuda.max_memory_allocated() < LIMITS['cuda_allocated_bytes_exclusive'], 'CUDA cap reached')
            if state is not None:
                check_endpoint(connected, state)
        boundary()
        memory_snapshot(before['path'], 'before_load_inference')
        state = connected.load_inference(Path(endpoint['bundle']['path']).parent, endpoint['bundle']['sha256'], 'cuda')
        memory_snapshot(before['path'], 'after_load_inference')
        for module in state['modules'].values():
            serving_checks.append(guard(module, state['guards'][module.__file__], context['guards'],
                class_name='FlatAdmission' if Path(module.__file__).name == 'train_siglip2_substrate_adaptation.py' else None))
        for path, digest in state['guards'].items():
            require(context['guards'].setdefault(path, digest) == digest, 'serving guard conflict')
        boundary()
        audit()
        memory_snapshot(before['path'], 'before_observe')
        observation = observe(diagnostic, context, sources, state, batch, boundary)
        print(json.dumps({'diagnostic': 'observation_summary', 'status': 'UNACCEPTED',
            'pending_final_checks': True,
            **{k: observation[k] for k in ('decision', 'b32', 'b6_repeat_exact',
                'same_b6_cache_gather_exact', 'differences', 'capture_sha256',
                'image_forwards', 'original_tail_cause_established',
                'serving_correction_authorized')}},
            sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)
        memory_snapshot(before['path'], 'after_observe')
        boundary()
        exact_four(context, sources, audit())
        receipt = {'schema': 'connected-control-batch-execution-observation-v1', **observation,
            'authority': {'path': str(args.authority), 'sha256': args.authority_sha256},
            'launch': prospective['launch'], 'candidate_status': 'KILL unchanged',
            'qualification_eligible': False, 'state_reuse_eligible': False, 'quality_read': False,
            'validation_read': False, 'official_read': False, 'candidate_execution': False,
            'historical_artifact_reads': 'all four accepted exports/endpoints: integrity hashes only',
            'original_six_units_authenticated': True, 'numerical_flags': flags,
            'rng_flags_preserved': True, 'cgroup_before': before, 'resource_policy': LIMITS,
            'invocation': {'invocation_id': os.environ['INVOCATION_ID'], 'python': str(python),
                           'python_sha256': prior['python_sha256'], 'python_version': sys.version, 'optimize': 0},
            'terminal_exit_and_both_locks_require_parent_receipt': True}
    except BaseException as failure:
        error = failure
        failure.__traceback__ = None
    finally:
        def cleanup_state():
            nonlocal state
            if state is not None:
                def release():
                    nonlocal state
                    serving_checks.clear()
                    sources.modules['connected'].release_inference(state)
                    state = None
                    memory_snapshot(before['path'], 'after_release_inference')
                diagnostic.cleanup_error(None, [*checks, *serving_checks, release])
        def exit_origins():
            for check in checks:
                check()
            origins = audit()
            exact_four(context, sources, origins)
            if receipt is not None:
                receipt['origins'] = origins
        def maps():
            connected = sources.modules['connected']
            connected.mapping_absent(context['launch']['original_cache']['path'])
            for endpoint in context['score']['launch']['endpoints']:
                connected.mapping_absent(Path(endpoint['bundle']['path']).parent/'endpoint.pt')
            control = prospective['launch']['control']['bundle']
            connected.mapping_absent(Path(control['path']).parent/'vision.pt')
        if diagnostic is not None:
            tail = callbacks[:]
            actions = [cleanup_state]
            if audit is not None:
                actions.append(exit_origins)
            if sources is not None and 'connected' in sources.modules:
                actions.append(maps)
            if context is not None:
                if admitted:
                    actions.append(lambda: memory_snapshot(before['path'], 'before_rehash_context'))
                actions.append(lambda: rehash(context['guards']))
                if admitted:
                    actions.append(lambda: memory_snapshot(before['path'], 'after_rehash_context'))
            if prospective is not None:
                if admitted:
                    actions.append(lambda: memory_snapshot(before['path'], 'before_rehash_prospective'))
                actions.append(lambda: rehash(prospective['guards']))
                if admitted:
                    actions.append(lambda: memory_snapshot(before['path'], 'after_rehash_prospective'))
            actions.append(lambda: elapsed_cap(started))
            if sources is not None:
                actions.append(sources.close)
            actions.extend(tail)
            actions.append(lambda: remove_source(diagnostic))
            diagnostic.cleanup_error(error, actions)
        elif error is not None:
            raise error
    receipt.update(exit_rehash_pass=True, integrity_pass=True, cleanup_pass=True,
                   wall_seconds=time.perf_counter()-started)
    prospective['output'].mkdir()
    diagnostic.write_json(prospective['output']/'receipt.json', receipt)
    elapsed_cap(started)
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--authority', required=True, type=Path)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    module = load_source('_control_batch_entry', {'path': str(Path(__file__).absolute()),
                        'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, {})
    try:
        module.run(args)
    finally:
        remove_source(module)
