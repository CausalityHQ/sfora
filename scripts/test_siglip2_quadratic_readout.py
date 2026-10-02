#!/usr/bin/env python3
"""Stdlib contracts only. Native qualification is deliberately parent-owned."""
if not __debug__:
    raise SystemExit('checks require assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
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
        assert has('exit_rehash', 'bound_file({}, path, digest)')
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
                         ('bound_file({}, path, digest)', 'admitted_file(context, path, digest)')):
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
