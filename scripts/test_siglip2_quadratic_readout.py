#!/usr/bin/env python3
"""Stdlib contracts only. Native qualification is deliberately parent-owned."""
if not __debug__:
    raise SystemExit('checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import io
import os
import signal
import time
from collections import Counter
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import unittest

PATH = Path(__file__).resolve().with_name('train_siglip2_quadratic_readout.py')


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.is_file():
            raise AssertionError('quadratic trainer contract is missing')
        spec = importlib.util.spec_from_file_location('quadratic_trainer', PATH)
        cls.driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.driver)

    def test_exact_own_three_file_closure_and_fixed_resources(self):
        d = self.driver
        self.assertEqual(d.FILES, {PATH.name, Path(__file__).name, 'quadratic_readout.py'})
        self.assertEqual(d.RECIPE['steps'], 1000)
        self.assertEqual(d.parameter_names('candidate'), ['A'])
        self.assertEqual(d.parameter_names('control'), ['A'])
        for phase in ('cpu', 'mechanics', 'train'):
            self.assertEqual(d.policy(phase), dict(seconds=120 if phase == 'cpu' else 300,
                host_bytes=8 * 1024**3, swap_bytes=0, cuda_allocated_bytes_exclusive=10_000_000_000))
        self.assertNotIn('torch', sys.modules)

    def unit(self):
        return dict(receipt=dict(path='/actual/receipt.json', sha256='a' * 64),
            log=dict(path='/actual/unit.log', sha256='b' * 64), unit='actual-unit',
            invocation_id='c' * 32, service_seconds=1., native_peak_rss_kib=100., both_locks_held=True)

    def launch(self, phase='cpu', arm='control', seed=179061):
        d = self.driver
        warm = dict(launch=d.WARM_LAUNCH.copy(), checkpoint=d.WARM_CHECKPOINT.copy(),
                    terminal_state_sha256=d.WARM_STATE_SHA, terminal=self.unit())
        warm['terminal']['receipt'] = d.WARM_RECEIPT.copy()
        launch = dict(schema=d.AUTHORITY_SCHEMA, execution_sha256='d' * 64, phase=phase, arm=arm, seed=seed,
            genuine_reference=d.GENUINE_REFERENCE.copy(), warm_start=warm,
            partition=dict(path='/actual/partition.json', sha256=d.PARTITION_SHA), recipe=copy.deepcopy(d.RECIPE),
            resource_policy=d.policy(phase), both_locks_held=True,
            selected_cpu=None if phase == 'cpu' else self.unit(),
            selected_mechanics={a: self.unit() for a in d.ARMS} if phase == 'train' else None)
        args = SimpleNamespace(phase=phase, arm=arm, seed=seed, execution_sha256='d' * 64)
        return launch, args

    def encoder(self):
        # Structural fixtures exercise admission predicates, never certify native source bytes.
        roles = [dict(name='tensor.' + str(i), shape=[1], dtype='torch.float32',
                      role='trainable' if i >= 243 else 'frozen') for i in range(448)]
        runtime = dict(attn_implementation='sdpa', config=dict(hidden_size=1152), modules=[],
            processor=dict(backend='torchvision'), roles=roles,
            vision={r['name']: dict(shape=r['shape'], dtype='torch.float32', sha256='a' * 64) for r in roles},
            buffers={'embeddings.position_ids': dict(shape=[1, 256], dtype='torch.int64', persistent=False,
                sha256='bbd330b12e8159e117376ef24fa106413bc9fc18032a0d43e95c5dae5e47953f')})
        checkpoint = dict(path='/actual/fresh_vision.pt',
            sha256='5dade5510a57637019adcf3c37a2ef66af0828ba072d5c847e768c8de2d48189')
        proof = dict(checkpoint=checkpoint, runtime=runtime, pass_=True, source_qualified=True,
            reload_exact=True, exit_rehash_pass=True, quality_read=False, training_qualified=False,
            gradients_created=False, optimizer_created=False, updates=0)
        proof['pass'] = proof.pop('pass_')
        exported = copy.deepcopy(runtime)
        exported['roles'] = [{**r, 'role': 'frozen'} for r in roles]
        binding = dict(source=dict(source_cpu=dict(proof=dict(
            sha256='e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf'))))
        return dict(checkpoint=checkpoint, source_binding=binding['source'], source_proof=proof,
                    export_binding=binding, export_terminal=self.unit(), export_runtime=exported,
                    inventory=exported['roles'], warm_payload_sha256=self.driver.WARM_CHECKPOINT['sha256'],
                    warm_state_sha256=self.driver.WARM_STATE_SHA)

    def test_authority_rejects_source_recipe_phase_prerequisite_and_cap_mutations(self):
        d = self.driver
        for phase in ('cpu', 'mechanics', 'train'):
            launch, args = self.launch(phase)
            d.check_launch(launch, args)
            mutations = [lambda x: x.update(extra=True), lambda x: x.update(both_locks_held=False),
                lambda x: x['recipe'].update(steps=100), lambda x: x['recipe'].update(learning_rate=1e-3),
                lambda x: x['resource_policy'].update(seconds=301),
                lambda x: x['resource_policy'].update(host_bytes=9 * 1024**3),
                lambda x: x['warm_start']['checkpoint'].update(sha256='0' * 64),
                lambda x: x['warm_start']['terminal'].pop('log'),
                lambda x: x['genuine_reference'].update(execution_sha256='0' * 64),
                lambda x: x['partition'].update(sha256='0' * 64)]
            if phase == 'train':
                mutations.append(lambda x: x['selected_mechanics'].pop('candidate'))
            for mutate in mutations:
                bad = copy.deepcopy(launch); mutate(bad)
                with self.assertRaises((ValueError, KeyError, TypeError)):
                    d.check_launch(bad, args)
        for phase, arm, seed in (('cpu', 'candidate', 179061), ('cpu', 'control', 179069),
                                 ('mechanics', 'control', 179069)):
            launch, args = self.launch(phase, arm, seed)
            with self.assertRaises(ValueError):
                d.check_launch(launch, args)
        for phase in ('mechanics', 'train'):
            launch, args = self.launch(phase)
            launch['selected_cpu'] = None
            with self.assertRaises(ValueError):
                d.check_launch(launch, args)

    def test_encoder_composition_requires_all448_source205_export_frozen_config_and_buffers(self):
        d, encoder = self.driver, self.encoder()
        d.check_encoder(encoder)
        for key in d.ENCODER_KEYS:
            bad = copy.deepcopy(encoder); del bad[key]
            with self.assertRaises((ValueError, KeyError)):
                d.check_encoder(bad)
        for mutate in (lambda x: x['inventory'].pop(),
            lambda x: x['inventory'][0].update(role='trainable'),
            lambda x: x['source_proof'].update(reload_exact=False),
            lambda x: x['source_proof']['runtime']['vision'].pop(next(iter(x['source_proof']['runtime']['vision']))),
            lambda x: x['export_runtime']['config'].update(hidden_size=1024),
            lambda x: x['export_runtime']['processor'].update(backend='pil'),
            lambda x: x['export_runtime']['buffers']['embeddings.position_ids'].update(persistent=True),
            lambda x: x['export_binding']['source']['source_cpu']['proof'].update(sha256='0' * 64),
            lambda x: x['checkpoint'].update(sha256='0' * 64)):
            bad = copy.deepcopy(encoder); mutate(bad)
            with self.assertRaises((ValueError, KeyError)):
                d.check_encoder(bad)

    @staticmethod
    def tensor(shape, dtype='torch.float32', **changes):
        return SimpleNamespace(shape=shape, dtype=dtype, requires_grad=False, is_leaf=True,
                               grad_fn=None, grad=None, **changes)

    def test_one_member_optimizer_rejects_extra_duplicate_reordering_and_any_frozen_gradient(self):
        d = self.driver
        for arm in d.ARMS:
            A = self.tensor((128, 32)); A.requires_grad = True
            frozen = [self.tensor(s) for s in d.HEAD_SHAPES]
            d.check_membership([('A', A)], [[A]], arm, frozen)
            cases = [([('A', A)], [[A, A]], frozen),
                     ([('A', A), ('classifier', frozen[-1])], [[A, frozen[-1]]], frozen),
                     ([('A', A)], [[], [A]], frozen), ([('wrong', A)], [[A]], frozen),
                     ([('A', A)], [[frozen[0]]], frozen), ([('A', A)], [[A]], [A])]
            for pairs, groups, complement in cases:
                with self.assertRaises(ValueError):
                    d.check_membership(pairs, groups, arm, complement)
            for field, bad_value in (('requires_grad', False), ('dtype', 'torch.float16'),
                                     ('shape', (32, 128)), ('is_leaf', False), ('grad_fn', object())):
                previous = getattr(A, field); setattr(A, field, bad_value)
                with self.assertRaises(ValueError):
                    d.check_membership([('A', A)], [[A]], arm, frozen)
                setattr(A, field, previous)
            for field, value in (('requires_grad', True), ('grad', object()), ('grad_fn', object())):
                setattr(frozen[0], field, value)
                with self.assertRaises(ValueError):
                    d.check_membership([('A', A)], [[A]], arm, frozen)
                setattr(frozen[0], field, False if field == 'requires_grad' else None)

    def test_integrity_head_uses_actual_A_device_and_retains_exact_tensor_checks(self):
        primitive = PATH.with_name('quadratic_readout.py')
        tree = ast.parse(primitive.read_bytes())
        namespace = dict(Path=Path)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('_require', '_check_tensor', '_check_base')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(primitive), 'exec'), namespace)
        # Only factory provenance is synthetic; tensor validation is the genuine primitive.
        factory = '''def head_from(arm):
    class Residual:
        def forward(self): pass
        def residual(self): return arm
        def named_parameters(self): return self.params.items()
        def named_buffers(self): return self.buffers.items()
    return Residual()
'''
        exec(compile(factory, 'train_siglip2_cached_readout.py', 'exec'), namespace)
        integrity = next(n for n in ast.parse(PATH.read_bytes()).body
                         if isinstance(n, ast.FunctionDef) and n.name == 'integrity')
        calls = [n for n in integrity.body if isinstance(n, ast.Expr) and
                 isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                 n.value.func.attr == '_check_base']
        self.assertEqual(len(calls), 1)
        boundary = compile(ast.Module(body=calls, type_ignores=[]), str(PATH), 'exec')
        for device in ('cpu', 'cuda:0'):
            head = namespace['head_from']('control')
            tensors = {name: self.tensor(shape, device=device, layout='torch.strided')
                       for name, shape in self.driver.HEAD_LAYOUT.items()}
            head.params = {name: tensors[name] for name in self.driver.HEAD_LAYOUT if name.endswith(('weight', 'bias'))}
            head.buffers = {name: tensors[name] for name in ('center', 'preactivation_std')}
            state = dict(head=head, A=self.tensor((128, 32), device=device), device=device.split(':')[0])
            environment = dict(context={'quadratic': SimpleNamespace(**namespace)}, state=state)
            exec(boundary, environment)
            for tensor in tensors.values():
                for field, value in (('device', 'cuda:1'), ('dtype', 'torch.float16'),
                                     ('layout', 'torch.sparse_coo'), ('shape', (1,)),
                                     ('requires_grad', True), ('grad_fn', object())):
                    previous = getattr(tensor, field)
                    setattr(tensor, field, value)
                    with self.subTest(device=device, field=field), self.assertRaises(ValueError):
                        exec(boundary, environment)
                    setattr(tensor, field, previous)
        self.assertNotIn('torch', sys.modules)

    def fake_payload(self, arm='control', step=0):
        d, t = self.driver, self.tensor
        launch, _ = self.launch()
        encoder = self.encoder()
        defaults = dict(lr=.001, betas=[.9, .999], eps=1e-8, weight_decay=.05, amsgrad=False,
                        maximize=False, foreach=None, capturable=False, differentiable=False, fused=None)
        group = dict(defaults, lr=1e-4)
        views = dict(caches={}, ordered_input_sha256='a' * 64, ordered_view_sha256={})
        ident = dict(method=d.method(launch), source=dict(warm_source=views), arm=arm, seed=179061, device='cpu',
            parameter_names=['A'], config=copy.deepcopy(encoder['export_runtime']['config']),
            native_inventory=encoder['inventory'], optimizer_defaults=defaults, optimizer_groups=[group],
            optimizer_serial_groups=[dict(group, params=[0])], positive_shape=[6355, 110], numerical_flags={})
        for key in ('encoder_sha256', 'complement_sha256', 'buffers_sha256', 'head_buffers_sha256', 'means_sha256',
                    'features_sha256', 'frozen_sha256', 'static_sha256', 'warm_members_sha256',
                    'schedule_sha256', 'full_schedule_sha256'):
            ident[key] = 'a' * 64
        saved = dict(schema=d.SCHEMA, identity=ident, source=ident['source'], encoder=encoder, config=copy.deepcopy(ident['config']),
            buffers={'embeddings.position_ids': t((1, 256), 'torch.int64')},
            head={n: t(shape) for n, shape in d.HEAD_LAYOUT.items()}, classifier=t((1008, 128)),
            A=t((128, 32)), means={k: t((32,)) for k in ('linear', 'quadratic')}, bank=t((6355, 128)),
            pca=dict(mean=t((1152,)), components=t((128, 1152))), target=t((6355,), 'torch.int64'),
            positive=t((6355, 110), 'torch.int64'), original_rows=t((6355,), 'torch.int64'),
            schedules={str(s): t((1000, 64), 'torch.int64') for s in d.SEEDS},
            masks={str(s): t((1000, 64), 'torch.bool') for s in d.SEEDS},
            partition=dict(schema='siglip2-identity-mix-partition-v1'), views=views,
            warm_start=dict(endpoint=launch['warm_start'], warm_source_updates=1000,
                local_initial_counter=0, bank_context='original canonical B32'),
            optimizer=dict(state={}, param_groups=copy.deepcopy(ident['optimizer_serial_groups'])),
            optimizer_defaults=dict(defaults, betas=(.9, .999)),
            scaler=dict(scale=128., growth_factor=2., backoff_factor=.5, growth_interval=2000, _growth_tracker=step),
            cpu_rng=t((5000,), 'torch.uint8'), cuda_rng=[], counter=step, seed=179061, numerical_flags={})
        if step:
            class Step:
                shape, dtype = (), 'torch.float32'
                def __init__(self, value): self.value = value
                def __float__(self): return float(self.value)
            saved['optimizer']['state'][0] = dict(step=Step(step), exp_avg=t((128, 32)), exp_avg_sq=t((128, 32)))
        return saved, ident

    def test_every_payload_omission_and_updated_optimizer_scaler_counter_tamper_rejected(self):
        d = self.driver
        for arm in d.ARMS:
            for step in (0, 1, 2, 8, 17, 1000):
                saved, ident = self.fake_payload(arm, step)
                d.check_payload(saved, ident, step)
                for key in d.PAYLOAD_KEYS:
                    bad = saved.copy(); del bad[key]
                    with self.assertRaises((ValueError, KeyError), msg=key):
                        d.check_payload(bad, ident, step)
                for member in ('center', 'preactivation_std'):
                    bad = copy.deepcopy(saved); del bad['head'][member]
                    with self.assertRaises(ValueError):
                        d.check_payload(bad, ident, step)
                for mutate in (lambda x: x['means'].pop('linear'), lambda x: x['means'].pop('quadratic'),
                    lambda x: x['A'].__dict__.update(shape=(32, 128)),
                    lambda x: x['buffers'].clear(), lambda x: x['scaler'].update(scale=64),
                    lambda x: x['optimizer']['param_groups'][0].update(params=[0, 0]),
                    lambda x: x['schedules'].pop('179069'),
                    lambda x: x['warm_start'].update(local_initial_counter=1000),
                    lambda x: x.update(counter=step + 1)):
                    bad = copy.deepcopy(saved); mutate(bad)
                    with self.assertRaises((ValueError, KeyError)):
                        d.check_payload(bad, ident, step)
                if step:
                    bad = copy.deepcopy(saved); bad['optimizer']['state'][0]['step'].value = step - 1
                    with self.assertRaises(ValueError):
                        d.check_payload(bad, ident, step)
                self.assertIsInstance(saved['optimizer_defaults']['betas'], tuple)

    def test_json_presentation_does_not_convert_or_replace_typed_config_defaults(self):
        d = self.driver
        saved, ident = self.fake_payload(step=1)
        config = {**saved['config'], 'typed_label': {0: 'class'}, 'typed_shape': (256, 1152)}
        saved['config'] = config
        # Presentation equality is used only at the config identity boundary.
        ident['config'] = d.json_form(config)
        saved['encoder']['source_proof']['runtime']['config'] = d.json_form(config)
        saved['encoder']['export_runtime']['config'] = d.json_form(config)
        d.check_payload(saved, ident, 1)
        self.assertEqual(config['typed_label'], {0: 'class'})
        self.assertIsInstance(config['typed_shape'], tuple)
        self.assertIsInstance(saved['optimizer_defaults']['betas'], tuple)
        config['typed_label'][0] = 'changed'
        with self.assertRaises(ValueError):
            d.check_payload(saved, ident, 1)
        tree = ast.parse(PATH.read_text())
        for name in ('payload', 'save', 'restore', 'update', 'cpu_witnesses'):
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'json_form'
                                 for n in ast.walk(function)), name)

    def test_original_full1000_schedules_and_sequential_owner(self):
        d = self.driver
        full = [[(i + j) % 6355 for j in range(64)] for i in range(1000)]
        d.check_continuation(full, copy.deepcopy(full))
        for selected in (full[:100], full[1:], full[1:] + full[:1]):
            with self.assertRaises(ValueError):
                d.check_continuation(full, selected)
        bad = copy.deepcopy(full); bad[999][0] = 6355
        with self.assertRaises(ValueError):
            d.check_continuation(bad, bad)
        context = {}
        class Readout: pass
        owner = Readout(); d.claim_model(context, owner)
        with self.assertRaises(ValueError): d.require_no_model(context)
        del owner
        d.require_no_model(context)

    def test_complete_frozen_byte_guard_catches_mutation_without_version_change(self):
        d = self.driver
        value = SimpleNamespace(data='original', _version=0)
        state = dict(head=SimpleNamespace(state_dict=lambda: {'primary.weight': value}), classifier='classifier',
                     encoder='encoder', config={}, buffers={}, means={'linear': 'mean', 'quadratic': 'mean2'},
                     features='canonical', **{k: {} for k in d.STATIC_KEYS})
        def digest(tree): return hashlib.sha256(repr(tree).encode()).hexdigest()
        context = dict(original=SimpleNamespace(fingerprint=digest))
        ident = dict(frozen_sha256=digest(d.frozen_tree(state)))
        d.check_complement(context, state, ident)
        value.data = 'same-version-tamper'
        with self.assertRaises(ValueError): d.check_complement(context, state, ident)
        value.data = 'original'; d.check_complement(context, state, ident)
        state['means']['quadratic'] = 'tampered-mean'
        with self.assertRaises(ValueError): d.check_complement(context, state, ident)

    def test_closure_and_fresh_exit_detect_same_size_same_mtime_tamper(self):
        d = self.driver
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = {}
            for name in d.FILES:
                (root / name).write_bytes(b'original')
                code[name] = hashlib.sha256(b'original').hexdigest()
            raw = json.dumps(code).encode(); (root / 'execution.json').write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            guards = {}
            self.assertEqual(d.closure(root, sha, d.FILES, guards), code)
            self.assertEqual(len(guards), 4)
            target = root / 'quadratic_readout.py'
            stat = target.stat(); target.write_bytes(b'tampered')
            import os
            os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            with self.assertRaises(ValueError): d.closure(root, sha, d.FILES, {})
            with self.assertRaises(ValueError): d.bound_file({}, target, code[target.name])
            target.write_bytes(b'original')
            del code['quadratic_readout.py']; raw = json.dumps(code).encode()
            (root / 'execution.json').write_bytes(raw)
            with self.assertRaises(ValueError):
                d.closure(root, hashlib.sha256(raw).hexdigest(), d.FILES, {})
        with self.assertRaises(ValueError): d.strict_json('{"key":1,"key":2}')
        with self.assertRaises(ValueError): d.strict_json('{"key":NaN}')

    def test_current_native_origin_union_rejects_unknown_or_conflicting_files_and_modules(self):
        d = self.driver
        for initial in (True, False):
            for case in ('cpu', 'warm', 'unknown_file', 'unknown_module', 'changed_sha', 'changed_module_path',
                         'file_union_conflict', 'module_union_conflict'):
                with self.subTest(initial=initial, case=case), tempfile.TemporaryDirectory() as temporary:
                    paths, hashes = {}, {}
                    for name in ('cpu', 'warm', 'unknown'):
                        path = Path(temporary) / (name + '.py'); path.write_bytes(name.encode())
                        paths[name] = str(path); hashes[name] = hashlib.sha256(name.encode()).hexdigest()
                    cpu = dict(packages={}, modules={'torch': paths['cpu']}, native_files=[], files={paths['cpu']: hashes['cpu']})
                    warm = copy.deepcopy(cpu); warm['modules']['torch.warm'] = paths['warm']; warm['files'][paths['warm']] = hashes['warm']
                    actual = copy.deepcopy(cpu)
                    if case == 'warm': actual = copy.deepcopy(warm)
                    elif case == 'unknown_file': actual['files'][paths['unknown']] = hashes['unknown']
                    elif case == 'unknown_module': actual['modules']['torch.alias'] = paths['cpu']
                    elif case == 'changed_sha': actual['files'][paths['cpu']] = '0' * 64
                    elif case == 'changed_module_path': actual['modules']['torch'] = paths['warm']
                    elif case == 'file_union_conflict': warm['files'][paths['cpu']] = '0' * 64
                    elif case == 'module_union_conflict': warm['modules']['torch'] = paths['warm']
                    context = dict(source_driver=SimpleNamespace(imported_origins=lambda *args: actual), extract=None,
                        selected=dict(packages={}, source_cpu=dict(origins=cpu)), warm_record=dict(origins=warm),
                        guards=copy.deepcopy(cpu['files']), prior=dict(guards=copy.deepcopy(cpu['files'])))
                    if case == 'cpu' or case == 'warm' and not initial:
                        d.audit_origins(context, initial=initial)
                        self.assertEqual(context['origins'], actual)
                    else:
                        with self.assertRaises(ValueError): d.audit_origins(context, initial=initial)
                        self.assertNotIn('origins', context)

    def test_flat_admission_reuses_only_successful_same_call_ownership_exit_stays_uncached(self):
        d = self.driver
        original = PATH.with_name('train_siglip2_substrate_adaptation.py')
        if not original.is_file():
            # The frozen own3 test closure does not contain the external genuine math closure.
            return
        tree = ast.parse(original.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlatAdmission')
        reads = []
        def fresh_hash(guards, path, sha):
            reads.append(str(path))
            return d.bound_file(guards, path, sha)
        namespace = dict(Path=Path, re=d.re, require=d.require, bound_file=fresh_hash, strict_json=d.strict_json)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source'; path.write_bytes(b'original')
            sha = hashlib.sha256(b'original').hexdigest()
            admission = namespace['FlatAdmission']()
            context = dict(admission=admission, guards={})
            d.admitted_file(context, path, sha)
            d.admitted_file(context, path, sha)
            self.assertEqual(reads, [str(path)])
            self.assertIn(str(path), admission.verified)
            with self.assertRaises(ValueError): d.admitted_file(context, path, '0' * 64)
            bad = namespace['FlatAdmission']()
            with self.assertRaises(ValueError): bad.bound_file({}, path, '0' * 64)
            self.assertNotIn(str(path), bad.verified)
            # In-call ownership may reuse admitted bytes; the enclosing exit always rereads.
            path.write_bytes(b'tampered')
            with self.assertRaises(ValueError): d.bound_file({}, path, sha)

    def test_exit_only_differential_falsifier(self):
        """Catch lost nested predicates, expected-only seeding and repeated bulk reads."""
        d = self.driver
        started = time.monotonic()
        def deadline(*unused):
            raise AssertionError('exit falsifier exceeded 30 seconds')
        old_handler = signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, 30)
        self.addCleanup(signal.signal, signal.SIGALRM, old_handler)
        self.addCleanup(signal.setitimer, signal.ITIMER_REAL, 0)
        pins = {
            'train_siglip2_substrate_adaptation.py': 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543',
            'qualify_siglip2_substrate_cpu.py': 'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38',
            'export_siglip2_substrate_fit.py': '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8',
            'export_siglip2_genuine_views.py': 'e5e98f9bc85680cab013d53752e7fa140da9d5e7f7ecd4537413b29b1047f65e',
            'extract_siglip2_vision_source.py': 'a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d'}
        if not all((PATH.parent / name).is_file() for name in pins):
            self.skipTest('external pinned stdlib source closures required for differential falsifier')
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        for name, pin in pins.items():
            self.assertEqual(hashlib.sha256((PATH.parent / name).read_bytes()).hexdigest(), pin, name)
        original = load('_exit_original', PATH.with_name('train_siglip2_substrate_adaptation.py'))
        source = load('_exit_source', PATH.with_name('qualify_siglip2_substrate_cpu.py'))
        reference = load('_exit_reference', PATH.with_name('export_siglip2_substrate_fit.py'))
        exporter = load('_exit_exporter', PATH.with_name('export_siglip2_genuine_views.py'))
        # Pin every other original module node, including math, typed state, use
        # boundaries, reload/ownership/RNG, source448/config and resource admission.
        tree = ast.parse(PATH.read_bytes())
        tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or
                     n.name not in ('audit_origins', 'exit_rehash')]
        self.assertEqual(hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
                         '5a46018cd5c2e25d8662ad8e0a02b1a4de1e7b4b4b2e5c67f62f8d979fbf6f14')
        # Revert only the reviewed caller argument to prove every other node is unchanged.
        integrity = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'integrity')
        boundary = next(n.value for n in integrity.body if isinstance(n, ast.Expr) and
                        isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
                        n.value.func.attr == '_check_base')
        self.assertEqual(ast.dump(boundary.args[1]), ast.dump(ast.parse("state['A'].device", mode='eval').body))
        boundary.args[1] = ast.parse("state['device']", mode='eval').body
        self.assertEqual(hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
                         'f5ca80d12a65e66f947a98b70959b9ae17d93effce821b74043444ffaf2ad7c7')
        origin_node = next(n for n in ast.parse(PATH.read_bytes()).body
                           if isinstance(n, ast.FunctionDef) and n.name == 'audit_origins')
        origin_node.args.args.pop(); origin_node.args.defaults.pop()
        loop = next(n for n in origin_node.body if isinstance(n, ast.For) and
                    ast.unparse(n.iter) == "origins['files'].items()")
        loop.body[:1] = loop.body[0].body
        self.assertEqual(hashlib.sha256(ast.dump(origin_node, include_attributes=False).encode()).hexdigest(),
                         'e7ab66d946b7d92bf53fc6b4fddc045681f41d519f999e3dfe0461cf7fb7ac1b')
        flat_node = next(n for n in ast.parse(PATH.with_name('train_siglip2_substrate_adaptation.py').read_bytes()).body
                         if isinstance(n, ast.ClassDef) and n.name == 'FlatAdmission')
        flat_images = next(n for n in flat_node.body if isinstance(n, ast.FunctionDef) and n.name == 'all_fit_images')
        original_images = next(n for n in ast.parse(PATH.with_name('export_siglip2_substrate_fit.py').read_bytes()).body
                             if isinstance(n, ast.FunctionDef) and n.name == 'all_fit_images')
        self.assertEqual([ast.dump(n) for n in flat_images.body[:6]],
                         [ast.dump(n) for n in original_images.body[:6]])
        # Differential rejections use the exact original nested stage validator,
        # avoiding repeated irrelevant prefixes of the full 13,283-image oracle.
        evidence = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
        fit = json.loads((evidence / 'late-dense-v1/native256-fit-manifest-v1.json').read_bytes())
        partition = json.loads((evidence / 'identity-mix-v1/partition.json').read_bytes())
        with tempfile.TemporaryDirectory(dir='/dev/shm' if Path('/dev/shm').is_dir() else None) as temporary:
            root = Path(temporary)
            metadata = set()
            def sha(file):
                return hashlib.sha256(file.read_bytes()).hexdigest()
            def json_file(file, value):
                file.write_text(json.dumps(value, separators=(',', ':')))
                metadata.add(str(file))
                return dict(path=str(file), sha256=sha(file))
            def frozen(name, names):
                directory = root / name; directory.mkdir()
                for file in names:
                    (directory / file).write_bytes((PATH.parent / file).read_bytes()
                        if file == 'extract_siglip2_vision_source.py' else b'# synthetic closure\n')
                code = {file: sha(directory / file) for file in names}
                descriptor = json_file(directory / 'execution.json', code)
                metadata.update(str(directory / file) for file in names)
                return directory, code, descriptor['sha256']
            source_root, code, source_sha = frozen('source', source.FILES)
            fit_root, fit_code, fit_sha = frozen('fit', reference.FILES)
            genuine_root, genuine_code, genuine_sha = frozen('genuine', exporter.FILES)
            own_root, own_code, own_sha = frozen('own', d.FILES)
            extract = load('extract_siglip2_vision_source', source_root / 'extract_siglip2_vision_source.py')
            dataset = root / 'images'; (dataset / 'Img/img').mkdir(parents=True)
            fit['dataset_root'] = str(dataset)
            images = []
            for index, row in enumerate(fit['rows']):
                row.update(relative_path=f'Img/img/{index}.jpg', image_sha256=hashlib.sha256(b'image').hexdigest())
                image = dataset / row['relative_path']; image.write_bytes(b'image'); images.append(image)
            part = json_file(root / 'partition.json', partition)
            ast_path = root / 'ImageRows.py'
            ast_path.write_text(ast.unparse(exporter.image_rows_node(PATH.with_name('train_sop_siglip2_compact.py'))))
            metadata.add(str(ast_path))
            payload = root / 'payload.pt'; payload.write_bytes(bytes(range(256)) * 4096 + b'tail')
            package = root / 'package'; package.mkdir()
            origin = package / 'origin.py'; origin.write_bytes(b'# actual loaded origin\n')
            warm_origin = package / 'warm.py'; warm_origin.write_bytes(b'# warm loaded origin\n')
            unknown = package / 'unknown.py'; unknown.write_bytes(b'# unknown origin\n')
            library = root / 'mapped.so'; library.write_bytes(b'mapped library')
            packages = {'_exit_fixture': {'root': str(package)}}
            def module(name, file):
                value = ModuleType(name); value.__file__ = str(file); return value
            loaded = {'extract_siglip2_vision_source': extract,
                      '_exit_fixture.origin': module('_exit_fixture.origin', origin)}
            maps = f'0-1 r-xp 00000000 00:00 1 {library}\n'
            real_text, real_open = Path.read_text, Path.open
            def maps_text(file, *args, **kwargs):
                return maps if str(file) == '/proc/self/maps' else real_text(file, *args, **kwargs)
            streamed, opened = Counter(), Counter()
            failed = None
            class Stream:
                def __init__(self, stream, file): self.stream, self.file = stream, str(file)
                def __enter__(self): return self
                def __exit__(self, *args): return self.stream.__exit__(*args)
                def __getattr__(self, name): return getattr(self.stream, name)
                def read(self, size=-1):
                    if self.file == failed and self.stream.tell(): raise OSError('synthetic mid-read failure')
                    raw = self.stream.read(size); streamed[self.file] += len(raw); return raw
                def readinto(self, buffer):
                    if self.file == failed: raise OSError('synthetic read failure')
                    count = self.stream.readinto(buffer); streamed[self.file] += count; return count
            def recording_open(file, mode='r', *args, **kwargs):
                stream = real_open(file, mode, *args, **kwargs)
                if mode == 'rb':
                    opened[str(file)] += 1
                    return Stream(stream, file)
                return stream
            with patch.object(Path, 'read_text', maps_text), patch.dict(sys.modules, loaded), redirect_stdout(io.StringIO()):
                expected = source.imported_origins(extract, packages)
                warm = copy.deepcopy(expected)
                warm['modules']['_exit_fixture.warm'] = str(warm_origin)
                warm['files'][str(warm_origin)] = sha(warm_origin)
                prior = dict(source_driver=source, extract=extract, fit=fit, images=images[:2], all_images=images,
                    root=source_root, code=code, args=SimpleNamespace(execution_sha256=source_sha),
                    own_root=fit_root, own_code=fit_code, export_args=SimpleNamespace(execution_sha256=fit_sha))
                manifest = exporter.selected_manifest(partition, fit)
                manifest['resolved_paths'] = [str(images[r]) for r in manifest['original_rows']]
                genuine = dict(prior=prior, reference=reference, root=genuine_root, code=genuine_code,
                    args=SimpleNamespace(execution_sha256=genuine_sha), selected=manifest,
                    launch=dict(partition=part, image_rows=dict(path=str(ast_path), sha256=sha(ast_path))))
                prior['guards'] = {str(file): sha(file) for file in (*images, payload, origin, library, warm_origin)}
                genuine['guards'] = {file: sha(Path(file)) for file in metadata}
                guards = {**prior['guards'], **genuine['guards']}
                selected = dict(source_driver=source, exporter=exporter, genuine=genuine, packages=packages,
                                source_cpu=dict(origins=copy.deepcopy(expected)))
                context = dict(original=original, source_driver=source, extract=extract, prior=prior, selected=selected,
                    warm_record=dict(origins=warm), guards=guards, root=own_root, code=own_code,
                    args=SimpleNamespace(execution_sha256=own_sha, phase='cpu'))
                # A deliberately warm startup cache is never exit authority.
                context['admission'] = original.FlatAdmission()
                for file, digest in guards.items():
                    context['admission'].verified.add(str(context['admission'].register({}, file, digest)))
                inventories = [prior['guards'], genuine['guards'], guards]
                before = [dict(value) for value in inventories]
                def old_exit():
                    d.require_no_model(context)
                    d.audit_origins(context)
                    exporter.rehash(genuine)
                    for file, digest in guards.items(): d.bound_file({}, file, digest)
                    d.require(d.closure(own_root, own_sha, d.FILES, {}) == own_code, 'exit code changed')
                def new_exit(): d.exit_rehash(context)
                bulk = guards.keys() - metadata
                with patch.object(Path, 'open', recording_open):
                    old_exit()
                self.assertEqual(opened[str(images[-1])], 3)
                self.assertEqual(opened[str(origin)], 4)
                self.assertEqual(context['origins'], expected)
                self.assertEqual(inventories, before)
                for _ in range(2):
                    opened.clear(); streamed.clear()
                    with patch.object(Path, 'open', recording_open): new_exit()
                    self.assertEqual(context['origins'], expected)
                    self.assertEqual(inventories, before)
                    for file in bulk:
                        self.assertEqual(opened[file], 1, file)
                        self.assertEqual(streamed[file], Path(file).stat().st_size, file)
                def reject(label, old_check=None, composed_check=None):
                    if old_check is None: old_check = lambda: d.audit_origins(context)
                    for traversal in (old_check, composed_check or new_exit):
                        with self.assertRaises((ValueError, OSError), msg=label): traversal()
                def change(mapping, key, value, label, old_check=None, composed_check=None):
                    with patch.dict(mapping, {key: value}): reject(label, old_check, composed_check)
                def old_paths():
                    d.require(reference.all_fit_images(prior) == prior['all_images'], 'FIT image resolution changed')
                # Extract only whole genuine metadata statements from the new
                # exit for late mutants; full entrypoints were exercised above.
                # No validator/cardinality/hash implementation is mocked.
                function = next(n for n in ast.parse(PATH.read_bytes()).body
                                if isinstance(n, ast.FunctionDef) and n.name == 'exit_rehash')
                selected_nodes = []
                for node in function.body:
                    if (any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
                           n.func.attr in ('bootstrap', 'closure', 'selected_manifest', 'image_rows_node')
                           for n in ast.walk(node)) or any(isinstance(n, ast.Call) and
                           isinstance(n.func, ast.Name) and n.func.id == 'closure' for n in ast.walk(node)) or
                       isinstance(node, ast.Assign) and any(isinstance(n, ast.Subscript) and
                           ast.unparse(n) == "selected['resolved_paths']" for n in node.targets) or
                       isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                       node.value.func.id == 'require' and 'exit TRAIN mapping changed' in ast.unparse(node)):
                        selected_nodes.append(node)
                metadata_code = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(PATH), 'exec')
                def new_metadata():
                    namespace = dict(vars(d), context=context, prior=prior, source=source,
                                     genuine=genuine, exporter=exporter)
                    exec(metadata_code, namespace)
                def old_first_two():
                    d.require(source.fit_rows(extract, fit) == prior['images'], 'FIT image resolution changed')
                def old_mapping():
                    value = exporter.selected_manifest(exporter.file_json(part, {}), fit)
                    value['resolved_paths'] = [str(prior['all_images'][r]) for r in value['original_rows']]
                    d.require(value == genuine['selected'], 'exit TRAIN mapping changed')
                def flip(file):
                    stat = file.stat()
                    with real_open(file, 'r+b') as stream:
                        stream.seek(stat.st_size - 1); byte = stream.read(1)
                        stream.seek(stat.st_size - 1); stream.write(bytes([byte[0] ^ 1]))
                    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    self.assertEqual((file.stat().st_size, file.stat().st_mtime_ns), (stat.st_size, stat.st_mtime_ns))
                    def restore():
                        with real_open(file, 'r+b') as stream:
                            stream.seek(stat.st_size - 1); stream.write(byte)
                        os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                    return restore
                for file in (payload, images[-1], origin):
                    restore = flip(file)
                    try: reject('same-size restored-mtime tail: ' + file.name, lambda: d.bound_file({}, file, guards[str(file)]))
                    finally: restore()
                change(genuine['guards'], str(payload), '0' * 64, 'cross-stage guard conflict',
                       lambda: d.bound_file({}, payload, genuine['guards'][str(payload)]))
                change(warm['files'], str(origin), '0' * 64, 'conflicting file origin authority')
                change(warm['modules'], '_exit_fixture.origin', str(warm_origin), 'conflicting module origin authority')
                with patch.dict(sys.modules, {'_exit_fixture.alias': module('_exit_fixture.alias', origin)}):
                    reject('unknown module with known file')
                with patch.dict(sys.modules, {'_exit_fixture.origin': module('_exit_fixture.origin', unknown)}):
                    reject('unknown file and changed module path')
                with patch.dict(sys.modules, {'_exit_fixture.warm': module('_exit_fixture.warm', warm_origin)}):
                    d.audit_origins(context, admission=original.FlatAdmission())
                    self.assertEqual(context['origins']['files'][str(warm_origin)], warm['files'][str(warm_origin)])
                old_maps = maps; maps += f'0-1 r-xp 00000000 00:00 1 {unknown}.so\n'
                unknown_so = Path(str(unknown) + '.so'); unknown_so.write_bytes(b'unknown library')
                try: reject('unknown mapped file with unchanged modules')
                finally: maps = old_maps
                for target in (unknown, images[-2]):
                    image = images[-1]; image.unlink(); image.symlink_to(target)
                    try: reject('escaped image' if target == unknown else 'aliased image', old_paths)
                    finally: image.unlink(); image.write_bytes(b'image')
                change(prior, 'images', list(reversed(images[:2])), 'first-two order', old_first_two)
                change(prior, 'all_images', images[:-2] + list(reversed(images[-2:])), 'full ordered paths', old_paths)
                changed = fit['rows'].copy(); changed[-2:] = reversed(changed[-2:])
                targets = fit['targets'].copy(); targets[-2:] = reversed(targets[-2:])
                with patch.dict(fit, rows=changed, targets=targets): reject('reordered FIT rows', old_paths)
                bad_manifest = copy.deepcopy(manifest); bad_manifest['resolved_paths'][-2:] = reversed(bad_manifest['resolved_paths'][-2:])
                change(genuine, 'selected', bad_manifest, 'resolved TRAIN mapping', old_mapping, new_metadata)
                change(fit, 'quality_read', True, 'FIT metadata', lambda: source.fit_rows(extract, fit))
                # Authenticated changes still have to pass the fresh semantic predicates.
                for file, value, label in ((Path(part['path']), {**partition, 'partition_seeds': [0, 1]}, 'partition'),
                                          (ast_path, 'class ImageRows: pass\n', 'ImageRows AST')):
                    raw = file.read_bytes()
                    if isinstance(value, dict): file.write_text(json.dumps(value, separators=(',', ':')))
                    else: file.write_text(value)
                    digest = sha(file)
                    descriptor = part if isinstance(value, dict) else genuine['launch']['image_rows']
                    try:
                        with patch.dict(descriptor, sha256=digest), patch.dict(genuine['guards'], {str(file): digest}), patch.dict(guards, {str(file): digest}):
                            reject(label, old_mapping if isinstance(value, dict) else lambda: exporter.image_rows_node(ast_path), new_metadata)
                    finally: file.write_bytes(raw)
                for mapping, key, read_code in (
                    (prior, 'code', lambda: source.bootstrap(source_root, source_sha)[1]),
                    (prior, 'own_code', lambda: reference.bootstrap(fit_root, fit_sha)),
                    (genuine, 'code', lambda: exporter.closure(genuine_root, genuine_sha, exporter.FILES, {})),
                    (context, 'code', lambda: d.closure(own_root, own_sha, d.FILES, {}))):
                    change(mapping, key, {}, 'closure equality: ' + key,
                           lambda: d.require(read_code() == mapping[key], 'closure differs'), new_metadata)
                with patch.object(extract, '__file__', str(unknown)):
                    reject('actual extractor loaded origin', lambda: source.bootstrap(source_root, source_sha), new_metadata)
                with patch.object(extract.__spec__, 'origin', str(unknown)):
                    reject('actual extractor spec origin', lambda: source.bootstrap(source_root, source_sha), new_metadata)
                failed = str(payload)
                with patch.object(Path, 'open', recording_open):
                    reject('bulk read failure', lambda: d.bound_file({}, payload, guards[str(payload)]),
                           lambda: original.FlatAdmission().bound_file({}, payload, guards[str(payload)]))
                failed = str(origin)
                with patch.object(Path, 'open', recording_open): reject('genuine imported origin read failure')
                failed = None
            self.assertLessEqual(sum(file.stat().st_size for file in root.rglob('*') if file.is_file()), 4 * 1024**2)
        self.assertLess(time.monotonic() - started, 30)
        self.assertNotIn('torch', sys.modules)

    def test_original_fingerprint_keeps_tuple_and_integer_key_types(self):
        original = PATH.with_name('train_siglip2_substrate_adaptation.py')
        if not original.is_file():
            return
        tree = ast.parse(original.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')
        node.body = [n for n in node.body if not isinstance(n, ast.Import)]
        class UnusedTensor: pass
        namespace = dict(hashlib=hashlib, torch=SimpleNamespace(Tensor=UnusedTensor))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(original), 'exec'), namespace)
        fingerprint = namespace['fingerprint']
        self.assertNotEqual(fingerprint({'betas': (.9, .999)}), fingerprint({'betas': [.9, .999]}))
        self.assertNotEqual(fingerprint({0: 'class'}), fingerprint({'0': 'class'}))
        self.assertNotIn('torch', sys.modules)

    @staticmethod
    def scope_checks(tree):
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        def has(name, source):
            expression = ast.dump(ast.parse(source, mode='eval').body)
            return any(ast.dump(n) == expression for n in ast.walk(functions[name]))
        assert has('fresh', "context['selected']['cached'].head_from('control', tensors=initial['head']).requires_grad_(False).train()")
        assert has('fresh', "context['quadratic'].fit_means(raw, head)")
        assert has('canonical_features', "context['genuine'].normalize_nonzero(raw)")
        assert has('integrity', "context['quadratic']._check_base(state['head'], state['A'].device)")
        assert has('update', "context['quadratic'].raw_features(state['features'][index], state['head'], state['A'], state['means'], state['arm'])")
        assert has('update', '(ce + 8 * rank) * .25')
        assert has('update', 'range(0, 64, 16)')
        assert has('update', 'scaler.unscale_(optimizer)')
        assert has('update', 'gradient_norms(state)')
        assert has('update', "torch.nn.utils.clip_grad_norm_([state['A']], 1., error_if_nonfinite=True)")
        assert has('update', 'torch.cat(raw_rows).detach()')
        assert has('update', "context['ref'].member_bank_refresh_rows(batch)")
        assert has('update', "context['ref'].member_bank_refresh_values(raw, raw, torch.tensor(positions, device=device), live_head=False)")
        assert has('cpu_witnesses', 'update(context, state, ident, 1)')
        assert has('cpu_witnesses', 'restore(context, path, sha1, digest1, ident, 1)')
        assert has('cpu_witnesses', 'update(context, state, ident, 2)')
        assert has('cpu_witnesses', 'diagnostic(resumed2) == diagnostic(step2)')
        assert has('cpu_witnesses', 'original.fingerprint(payload(state, ident)) == expected_digest')
        assert has('cpu_witnesses', 'original.fingerprint(calibration(context, state)) == expected_witness')
        assert has('prepare_native', "genuine.check_payload(disk, disk['identity'], 1000)")
        assert has('prepare_native', 'context[\'original\'].fingerprint(disk, consumed=pages.consume) == WARM_STATE_SHA')
        assert has('restore', 'context[\'original\'].fingerprint(disk, consumed=pages.consume) == digest')
        assert has('gpu_run', "restore(context, temporary / 'step8.pt', sha8, digest8, ident, 8)")
        assert has('gpu_run', 'range(9, 18)')
        assert has('gpu_run', '17 if args.phase == \'mechanics\' else 1000')
        assert has('gpu_run', "diagnostic(row) == diagnostic(mechanics['steps'][step - 1])")
        assert has('exit_rehash', 'admission.bound_file({}, path, digest)')
        assert has('exit_rehash', "context['original'].FlatAdmission()")
        assert has('exit_rehash', 'audit_origins(context, admission=admission)')
        assert has('exit_rehash', 'admission.all_fit_images(prior)')
        for name in ('update', 'cpu_witnesses', 'gpu_run'):
            body = functions[name].body
            # Each independent restore follows explicit destruction of the preceding owner.
            if name == 'cpu_witnesses':
                calls = [n for n in ast.walk(functions[name]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
                assert sum(n.func.id == 'release' for n in calls) >= 2
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module]
                assert not {n.split('.')[0] for n in names} & {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {'fresh_source', 'construct', 'from_pretrained', 'native_raw', 'canonical_pixels',
                    'initializer', 'training_features', 'view_inputs', 'pixels_and_raw', 'augmented_pixels',
                    'half', 'reset_peak_memory_stats', 'fit_centered_pca', 'set_float32_matmul_precision'}
                if node.func.attr == 'head_from': assert ast.literal_eval(node.args[0]) == 'control'
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assert not isinstance(target, ast.Attribute) or ast.unparse(target) == 'admission.init', \
                        'helper/global/module rebinding is forbidden'

    def test_cached_math_updated_reload_bank_and_native_scope(self):
        text = PATH.read_text()
        self.scope_checks(ast.parse(text))
        for old, new in (("head_from('control',", "head_from('candidate',"),
                         ('range(0, 64, 16)', 'range(0, 2, 2)'),
                         ('scaler.unscale_(optimizer)', 'scaler.get_scale()'),
                         ('restore(context, path, sha1, digest1, ident, 1)', 'restore(context, path, sha1, digest1, ident, 0)'),
                         ('raw, raw, torch.tensor(positions', 'raw, state[\'A\'], torch.tensor(positions'),
                         ("_check_base(state['head'], state['A'].device)", "_check_base(state['head'], state['device'])"),
                         ('admission.bound_file({}, path, digest)', 'admitted_file(context, path, digest)')):
            with self.assertRaises(AssertionError, msg=old): self.scope_checks(ast.parse(text.replace(old, new)))
        self.assertNotIn('torch', sys.modules)

    def test_fixed_cli_help_lazy_imports_and_optimized_mode_rejection(self):
        d = self.driver
        parser = d.parser()
        argv = d.cli('/actual/root', '/actual/authority.json', 'a' * 64, 'b' * 64,
                     'cpu', 'control', 179061, '/actual/output')
        parsed = parser.parse_args(argv[1:])
        self.assertEqual((parsed.phase, parsed.arm, parsed.seed), ('cpu', 'control', 179061))
        self.assertEqual(argv[1::2], ['--execution-sha256', '--authority', '--authority-sha256', '--phase', '--arm', '--seed', '--output'])
        result = subprocess.run([sys.executable, '-B', str(PATH), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('composed with', result.stdout)
        for word in ('--phase', '--arm', '--seed', '--authority-sha256'):
            self.assertIn(word, result.stdout)
        for optimize in ('-O', '-OO'):
            for path in (PATH, Path(__file__).resolve()):
                result = subprocess.run([sys.executable, '-B', optimize, str(path), '--help'], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('optimized mode is forbidden', result.stderr)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
