#!/usr/bin/env python3
"""One stdlib falsifier; no Torch/native/images/quality execution.

Reject changed row/view provenance, held leakage, RNG leakage, immutable pins,
cache/closure tampering, startup terminal omissions and whole-lifecycle limits.
Synthetic metadata proves predicates only, never native or quality admission.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError, TypeError):
        return
    raise AssertionError('accepted ' + message)


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CPUState:
    """Tiny deterministic protocol double; intentionally no native RNG imported."""
    def __init__(self, value=17):
        self.value = value
        self.default_generator = self

    def clone(self):
        return CPUState(self.value)

    def get_rng_state(self):
        return self.clone()

    def manual_seed(self, value):
        self.value = value

    @contextmanager
    def fork_rng(self, *, devices):
        check(devices == [], 'CPU-only RNG fork required')
        saved = self.value
        try:
            yield
        finally:
            self.value = saved


def cpu_witness_checks(d, path):
    """Execute the real nested calls; missing CPU transfer or altered bytes fails."""
    class Tensor:
        def __init__(self, shape, raw, device='cuda', detached=False):
            self.shape, self.raw = shape, raw
            self.dtype, self.device = 'torch.float32', SimpleNamespace(type=device)
            self.detached = detached

        def detach(self):
            return Tensor(self.shape, self.raw, self.device.type, True)

        def cpu(self):
            return Tensor(self.shape, self.raw, 'cpu', self.detached)

        def float(self):
            return self

        def norm(self, *, dim):
            check(dim == 1, 'calibration norm dimension')
            return self

        def __gt__(self, other):
            return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))

    def fact(value):
        d.require(value.device.type == 'cpu', 'CPU finite tensor required')
        check(value.detached, 'witness must detach before CPU fact')
        return {'dtype': value.dtype, 'shape': list(value.shape),
                'sha256': hashlib.sha256(value.raw).hexdigest()}

    tree = ast.parse(path.read_text())
    export_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'export')
    functions = [n for n in export_node.body if isinstance(n, ast.FunctionDef) and
                 n.name not in ('pixels_for', 'infer')]
    namespace = {'source': SimpleNamespace(tensor_fact=fact), 'require': d.require,
                 'torch': SimpleNamespace(isfinite=lambda _: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True))),
                 'F': SimpleNamespace(cosine_similarity=lambda *args, **kwargs:
                     SimpleNamespace(cpu=lambda: SimpleNamespace(tolist=lambda: [1.0] * 4)))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    for view in d.VIEWS:
        for count in (4, d.BATCH, d.ROWS % d.BATCH):
            # Tiny shapes, exact F32 bits including signed zero and a subnormal.
            pixels = Tensor((count, 3, 2, 2), bytes.fromhex('0000008001000000') * (count * 6))
            fp32 = Tensor((count, 2), bytes.fromhex('0000803f000000c0') * count)
            raw = Tensor((count, 2), bytes.fromhex('0000803f0000803f') * count)
            unit = Tensor((count, 2), bytes.fromhex('f304353ff304353f') * count)
            images = [{'view': view, 'ordinal': i} for i in range(count)]
            namespace.update(pixels_for=lambda *args: (pixels, images),
                             model=lambda **kwargs: SimpleNamespace(pooler_output=fp32),
                             infer=lambda _: (raw, unit))
            if count == 4:
                calibration = namespace['calibrate'](view)
                check(calibration == {'images': images,
                    'fp32': {'dtype': 'torch.float32', 'shape': [4, 2], 'sha256': hashlib.sha256(fp32.raw).hexdigest()},
                    'fp16': {'dtype': 'torch.float32', 'shape': [4, 2], 'sha256': hashlib.sha256(raw.raw).hexdigest()},
                    'cosines': [1.0] * 4}, 'calibration CPU facts preserve exact dtype/shape/bytes')
            expected = {'images': images, **{name: {'dtype': 'torch.float32', 'shape': list(value.shape),
                        'sha256': hashlib.sha256(value.raw).hexdigest()}
                        for name, value in (('pixels', pixels), ('raw', raw), ('unit', unit))}}
            check(namespace['witness'](pixels, images, raw, unit) == expected,
                  'batch/tail CPU witnesses preserve exact dtype/shape/bytes')
            check(all(value.device.type == 'cuda' and not value.detached for value in (pixels, fp32, raw, unit)),
                  'witness copies leave inference tensors unchanged')
    rejects(lambda: fact(Tensor((1,), b'\0' * 4)), 'CUDA fact without transfer')


def main():
    scripts = Path(__file__).absolute().parent
    path = scripts / 'export_siglip2_genuine_views.py'
    check(path.is_file(), 'TRAIN-only paired exporter is missing')
    spec = importlib.util.spec_from_file_location('genuine_views', path)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    cpu_witness_checks(d, path)
    check(d.FILES == {'export_siglip2_genuine_views.py', 'test_siglip2_genuine_views.py'}, 'exact two-file closure')
    evidence = scripts.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    fit_path = evidence / 'late-dense-v1/native256-fit-manifest-v1.json'
    part_path = evidence / 'identity-mix-v1/partition.json'
    check(d.sha(fit_path) == d.MANIFEST_SHA and d.sha(part_path) == d.PARTITION_SHA, 'committed metadata pins')
    fit, partition = json.loads(fit_path.read_text()), json.loads(part_path.read_text())
    manifest = d.selected_manifest(partition, fit)
    manifest['resolved_paths'] = [str(Path(fit['dataset_root']) / r['relative_path']) for r in manifest['rows']]
    check(len(manifest['rows']) == 6355 and len(manifest['class_names']) == 1008 and
          manifest['rows'] == [fit['rows'][i] for i in partition['panels']['train']['original_rows']], 'TRAIN mapping')
    for mutation in ('swapped', 'leaked', 'classes', 'coverage', 'query', 'source'):
        changed = copy.deepcopy(partition)
        if mutation == 'swapped':
            changed['panels']['train']['original_rows'][:2] = reversed(changed['panels']['train']['original_rows'][:2])
        elif mutation == 'leaked':
            changed['panels']['train']['original_rows'][0] = changed['panels']['selection']['original_rows'][0]
        elif mutation == 'classes':
            changed['panels']['train']['original_class_ids'][0] = changed['panels']['validation']['original_class_ids'][0]
        elif mutation == 'coverage':
            changed['panels']['train']['original_rows'][-1] = changed['panels']['train']['original_rows'][-2]
        elif mutation == 'query':
            changed['panels']['selection']['query'][0] = changed['panels']['selection']['gallery'][0]
        else:
            changed['original_fit']['sha256'] = '0' * 64
        rejects(lambda: d.selected_manifest(changed, fit), mutation)
    pixel = {'shape': [3, 256, 256], 'dtype': 'torch.float32', 'sha256': 'a' * 64}
    rgb = {'mode': 'RGB', 'size': [300, 400], 'sha256': 'b' * 64}
    mapping = {}
    for view in d.VIEWS:
        mapping[view] = [{'view': view, 'ordinal': i, 'original_row': original,
                         'train_row': row['train_row'], 'target': manifest['targets'][i],
                         'path': manifest['resolved_paths'][i], 'relative_path': row['relative_path'],
                         'image_sha256': row['image_sha256'], 'original_rgb': copy.deepcopy(rgb),
                         'rgb': copy.deepcopy(rgb) if view == 'canonical' else {**rgb, 'size': [256, 256]},
                         'rng_seed': None if view == 'canonical' else 179081 + row['train_row'],
                         'pixels': copy.deepcopy(pixel)} for i, (original, row) in enumerate(zip(manifest['original_rows'], manifest['rows']))]
    d.validate_views(manifest, mapping)
    for mutation in ('row', 'view', 'rng', 'rgb', 'pixels', 'path', 'partial'):
        changed = copy.deepcopy(mapping)
        fact = changed['augmented'][0]
        if mutation == 'row':
            changed['augmented'][0], changed['augmented'][1] = changed['augmented'][1], changed['augmented'][0]
        elif mutation == 'view':
            fact['view'] = 'canonical'
        elif mutation == 'rng':
            fact['rng_seed'] += 1
        elif mutation == 'rgb':
            fact['original_rgb']['sha256'] = 'c' * 64
        elif mutation == 'pixels':
            fact['pixels']['shape'] = [3, 224, 224]
        elif mutation == 'path':
            fact['path'] = '/held/image.jpg'
        else:
            changed['augmented'].pop()
        rejects(lambda: d.validate_views(manifest, changed), mutation)
    state = CPUState()
    fake = SimpleNamespace(random=state, equal=lambda a, b: a.value == b.value)
    def transform(image):
        result = state.value
        state.value += 99
        return image, result
    output = d.isolated_view(fake, 'pixel', transform, 73)
    check(output == ('pixel', 179154) and state.value == 17 and
          d.isolated_view(fake, 'pixel', transform, 73) == output, 'repeat view seed/global RNG restore')
    def fails(image):
        state.value += 100
        raise OSError('failed transform')
    rejects(lambda: d.isolated_view(fake, 'pixel', fails, 73), 'exception view')
    check(state.value == 17, 'exception RNG restore')
    rejects(lambda: d.isolated_view(fake, 'pixel', transform, True), 'bool original row')
    node = d.image_rows_node(scripts / 'train_sop_siglip2_compact.py')
    check(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == d.IMAGE_ROWS_AST_SHA, 'ImageRows AST pin')
    old_code = json.loads((evidence / 'late-dense-v1/native256-fit-export-source-v1-execution.json').read_text())
    check(old_code.keys() == d.REF_FILES and all(d.sha(scripts / name) == digest for name, digest in old_code.items()), 'unchanged original exporter')
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in d.FILES:
            (root / name).write_bytes((scripts / name).read_bytes())
        code = {name: d.sha(root / name) for name in d.FILES}
        execution_sha = write(root / 'execution.json', code)
        check(d.closure(root, execution_sha, d.FILES, {}) == code, 'valid closure')
        changed = dict(code, extra='0' * 64)
        digest = write(root / 'execution.json', changed)
        rejects(lambda: d.closure(root, digest, d.FILES, {}), 'extra closure member')
        digest = write(root / 'execution.json', {next(iter(code)): next(iter(code.values()))})
        rejects(lambda: d.closure(root, digest, d.FILES, {}), 'missing closure member')
        write(root / 'execution.json', code)
        (root / 'export_siglip2_genuine_views.py').write_bytes(b'tampered')
        rejects(lambda: d.closure(root, execution_sha, d.FILES, {}), 'changed closure bytes')
        cache = root / 'canonical.npy'
        cache.write_bytes(b'cache bytes, stdlib metadata fixture only')
        guards = {str(cache): d.sha(cache)}
        # Same exit function checks cached bytes before any helper/native access.
        context = {'reference': SimpleNamespace(rehash=lambda _: None), 'prior': {}, 'guards': guards}
        cache.write_bytes(b'changed cache bytes')
        rejects(lambda: d.rehash(context), 'cache tampering')
        altered = root / 'changed_ImageRows.py'
        altered.write_text((scripts / 'train_sop_siglip2_compact.py').read_text().replace('scale=(0.8, 1.0)', 'scale=(0.7, 1.0)'))
        rejects(lambda: d.image_rows_node(altered), 'crop defaults')
        launch = {'schema': d.AUTHORITY_SCHEMA, 'execution_sha256': 'd' * 64,
                  'reference': {'root': str(d.ORIGINAL_REF_ROOT), 'execution_sha256': d.REF_EXECUTION_SHA,
                                'authority': {'path': str(d.ORIGINAL_REF_ROOT / 'authority.json'), 'sha256': d.ORIGINAL_AUTHORITY_SHA}},
                  'partition': {'path': str(part_path), 'sha256': d.PARTITION_SHA},
                  'image_rows': {'path': str(scripts / 'train_sop_siglip2_compact.py'), 'sha256': d.sha(scripts / 'train_sop_siglip2_compact.py')},
                  'startup_policy': d.STARTUP_POLICY, 'export_policy': d.EXPORT_POLICY}
        d.check_launch(launch, 'd' * 64)
        for field in ('reference', 'partition', 'startup_policy', 'export_policy'):
            changed = copy.deepcopy(launch)
            if field == 'reference':
                changed[field]['execution_sha256'] = '0' * 64
            elif field == 'partition':
                changed[field]['sha256'] = '0' * 64
            else:
                changed[field]['seconds'] += 1
            rejects(lambda: d.check_launch(changed, 'd' * 64), field + ' pin')
        terminal_checks(d, root)
    class CUDA:
        @staticmethod
        def max_memory_allocated():
            return 10_000_000_000
    rejects(lambda: d.limits(d.time.perf_counter(), SimpleNamespace(cuda=CUDA)), 'CUDA exclusive cap')
    rejects(lambda: d.limits(d.time.perf_counter() - 301, SimpleNamespace(cuda=CUDA)), 'whole lifecycle300')
    for optimize in ('-O', '-OO'):
        run = subprocess.run([sys.executable, '-B', '-S', optimize, str(path), '--help'], capture_output=True, text=True)
        check(run.returncode != 0 and 'optimized mode is forbidden' in run.stderr, 'optimized rejection')
    check(not any(n in sys.modules for n in ('torch', 'numpy', 'PIL', 'transformers', 'torchvision', 'safetensors')), 'no native imports')
    print('genuine paired-view stdlib falsifier PASS; native/resource/quality qualification unrun')


def terminal_checks(d, root):
    spec = importlib.util.spec_from_file_location('old_fit_metadata', Path(__file__).with_name('export_siglip2_substrate_fit.py'))
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = SimpleNamespace(execution_sha256='d' * 64, authority=root / 'authority.json', authority_sha256='e' * 64)
    prior_identity = {'python': '/immutable/python', 'python_sha256': 'f' * 64, 'python_version': 'fixture'}
    context = {'args': args, 'root': root, 'code': {}, 'launch': {'reference': {}, 'partition': {}, 'image_rows': {}},
               'selected': {}, 'guards': {}, 'reference': old,
               'prior': {'proof': {'invocation': prior_identity}, 'guards': {}, 'export_args': SimpleNamespace(arm='so400', execution_sha256='a' * 64),
                         'authority_sha256': 'a' * 64, 'own_code': {}, 'root': root,
                         'args': SimpleNamespace(execution_sha256='a' * 64, sources_sha256='a' * 64, fit_manifest_sha256='a' * 64),
                         'launch': {'source_cpu': {'so400': {}}}}}
    unit, invocation = 'genuine-startup-fixture', '1' * 32
    memory = {'path': '/sys/fs/cgroup/' + unit + '.service',
              'values': {'memory.max': str(8 * 1024**3), 'memory.current': '100', 'memory.peak': '200',
                         'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                         'memory.events': 'max 0\noom 0\noom_kill 0'}}
    proof_path, log_path = root / 'proof.json', root / 'startup.log'
    proof = {'schema': d.SCHEMA, 'phase': 'startup', 'pass': True, 'binding': d.binding(context),
             'exit_rehash_pass': True, 'resource_policy': d.STARTUP_POLICY, 'wall_seconds': 10,
             'process_peak_rss_kib': 10, 'native_imported': False, 'model_constructed': False, 'exported': False,
             'input_guards': {}, 'original_input_guards': {}, 'cgroup_before': memory, 'cgroup_after': memory,
             'invocation': {**prior_identity, 'invocation_id': invocation, 'optimize': 0, 'cuda_visible_devices': '',
                            'argv': [str(root / 'export_siglip2_genuine_views.py'), '--execution-sha256', args.execution_sha256,
                                     '--authority', str(args.authority), '--authority-sha256', args.authority_sha256,
                                     '--phase', 'startup', '--output', str(root)]}}
    original_log = f'Running as unit: {unit}.service; invocation ID: {invocation}\n' + '\n'.join([
        '\tExit status: 0', 'Finished with result: success', 'Main processes terminated with: code=exited/status=0',
        '\tSwaps: 0', 'Memory swap peak: 0B', 'Service runtime: 11s', '\tMaximum resident set size (kbytes): 20',
        'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation})]) + '\n'
    log_path.write_text(original_log)
    descriptor = {'proof': {'path': str(proof_path), 'sha256': write(proof_path, proof)},
                  'log': {'path': str(log_path), 'sha256': d.sha(log_path)}, 'unit': unit,
                  'invocation_id': invocation, 'service_seconds': 11, 'native_peak_rss_kib': 20, 'both_locks_held': True}
    d.admit_terminal(context, descriptor, proof, 'startup')
    wrong_proof = copy.deepcopy(descriptor)
    wrong_proof['proof']['sha256'] = '0' * 64
    rejects(lambda: d.admit_terminal(context, wrong_proof, proof, 'startup'), 'terminal proof bytes')
    extra_log = copy.deepcopy(descriptor)
    extra_log['log']['extra'] = True
    rejects(lambda: d.admit_terminal(context, extra_log, proof, 'startup'), 'extra log descriptor field')
    minute_proof, minute_descriptor = copy.deepcopy(proof), copy.deepcopy(descriptor)
    minute_proof['wall_seconds'] = 60
    minute_descriptor['service_seconds'] = 61.125
    minute_descriptor['proof']['sha256'] = write(proof_path, minute_proof)
    log_path.write_text(original_log.replace('Service runtime: 11s', 'Service runtime: 1min 1.125s'))
    minute_descriptor['log']['sha256'] = d.sha(log_path)
    d.admit_terminal(context, minute_descriptor, minute_proof, 'startup')
    for duration in ('1min 61.125s', '1min 1.126s', '61.125', 'NaNs', '1min -1s'):
        log_path.write_text(original_log.replace('Service runtime: 11s', 'Service runtime: ' + duration))
        minute_descriptor['log']['sha256'] = d.sha(log_path)
        rejects(lambda: d.admit_terminal(context, minute_descriptor, minute_proof, 'startup'), 'service runtime ' + duration)
    log_path.write_text(original_log)
    write(proof_path, proof)
    for change in ('locks', 'cap', 'argv', 'guards', 'binding', 'hidden', 'native', 'swap', 'events'):
        altered, terminal = copy.deepcopy(proof), copy.deepcopy(descriptor)
        if change == 'locks':
            terminal['both_locks_held'] = False
        elif change == 'cap':
            terminal['service_seconds'] = 121
        elif change == 'argv':
            altered['invocation']['argv'][-1] += '-other'
        elif change == 'guards':
            altered['original_input_guards']['foreign'] = '0' * 64
        elif change == 'binding':
            altered['binding']['ordered_input_sha256'] = '0' * 64
        elif change == 'hidden':
            altered['invocation']['cuda_visible_devices'] = '0'
        elif change == 'native':
            altered['native_imported'] = True
        elif change == 'swap':
            altered['cgroup_after']['values']['memory.swap.peak'] = '1'
        else:
            altered['cgroup_after']['values']['memory.events'] = 'max 1\noom 0\noom_kill 0'
        terminal['proof']['sha256'] = write(proof_path, altered)
        rejects(lambda: d.admit_terminal(context, terminal, altered, 'startup'), 'startup ' + change)
    write(proof_path, proof)
    log_path.write_text(original_log.replace('\tExit status: 0', '\tExit status: 1'))
    changed = copy.deepcopy(descriptor)
    changed['log']['sha256'] = d.sha(log_path)
    rejects(lambda: d.admit_terminal(context, changed, proof, 'startup'), 'failed original exit')
    log_path.write_text(original_log + 'FINAL_CGROUP ' + json.dumps({**memory, 'invocation_id': invocation}) + '\n')
    changed['log']['sha256'] = d.sha(log_path)
    rejects(lambda: d.admit_terminal(context, changed, proof, 'startup'), 'duplicate footer')


if __name__ == '__main__':
    main()
