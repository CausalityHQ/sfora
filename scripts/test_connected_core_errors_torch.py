#!/usr/bin/env python3
"""Bounded stdlib falsifiers only; Torch, native execution and DGX UNRUN.

These prove structure (pins, exact scorer AST + adapter inverse, helper APIs,
gating, geometry, exit behaviour). They cannot prove Torch numerical equality:
only the root's single native job can falsify that.
"""
import ast
import builtins
import contextlib
import copy
import dis
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'census_connected_core_errors_torch.py'
EVIDENCE = HERE.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
CPU = EVIDENCE / 'late-dense-v1'
INPUTS = Path('/tmp/sfora-connected-core-census-inputs-v1/fetch-receipt.json')
INVOCATION = '0123456789abcdef0123456789abcdef'


def load_driver(path=DRIVER):
    spec = importlib.util.spec_from_file_location('_torch_census_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fact(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def f32_neighbor(value, step):
    bits = struct.unpack('<I', struct.pack('<f', value))[0] + step
    return struct.unpack('<f', struct.pack('<I', bits))[0]


class Row:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class Scores:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, index):
        return Row(self.rows[index])


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_driver()
        cls.scorer = (HERE / 'compare_inshop_sop_warmstart_100.py').read_bytes()

    def tmp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name)

    def authority(self, mutate=None):
        m, tmp = self.m, self.tmp()
        command, inputs = tmp / 'command.sh', tmp / 'inputs.json'
        command.write_text('#!/bin/bash\n')
        inputs.write_text('{}')
        authority = {
            'schema': m.LAUNCH_SCHEMA, 'inputs': fact(inputs),
            'sources': {r: fact(command) if n is None else fact(HERE / n) for r, (n, _) in m.SOURCE_ROLES.items()},
            'native_sources': fact(CPU / 'paired-native256-vision-source-cpu-inputs-v1.json'),
            'source_cpu': {'proof': fact(CPU / 'native256-source-cpu-so400-v4.json'),
                           'log': fact(CPU / 'native256-source-cpu-so400-v4.log'),
                           'unit': 'sfora-native256-source-cpu-so400-v4',
                           'invocation_id': '1e6981dda5fc44989dd59a81ec975687', 'service_seconds': 43.803,
                           'native_peak_rss_kib': 4502620, 'both_locks_held': True},
            'locks': [{'path': str(tmp / 'a.lock'), 'fd': 7}, {'path': str(tmp / 'b.lock'), 'fd': 8}],
            'resource_policy': dict(m.LIMITS), 'both_locks_held': True, 'candidate_status': 'KILL',
            'qualification_eligible': False, 'state_reuse_eligible': False}
        if mutate:
            mutate(authority)
        path = tmp / 'authority.json'
        path.write_text(json.dumps(authority))
        return authority, str(path), hashlib.sha256(path.read_bytes()).hexdigest()

    def helpers(self, authority=None):
        authority = authority or self.authority()[0]
        guards = {}
        serving, owned, modules = self.m.load_helpers(authority, guards)
        self.addCleanup(lambda: owned and self.m.close_sources(owned))
        return authority, guards, serving, owned, modules


class AuthorityTests(Base):
    def test_exact_authority_is_admitted_with_current_bytes(self):
        authority, path, digest = self.authority()
        loaded, guards = self.m.read_authority(path, digest)
        self.assertEqual(loaded, authority)
        self.assertEqual(guards[str(DRIVER)], hashlib.sha256(DRIVER.read_bytes()).hexdigest())
        self.assertTrue(set(authority['sources']) == self.m.SOURCE_ROLES.keys())

    def test_rejects_shape_policy_and_status(self):
        cases = {
            'extra key': lambda a: a.update(extra=1),
            'missing key': lambda a: a.pop('locks'),
            'schema': lambda a: a.update(schema='connected-gallery-freshness-launch-v1'),
            'seconds': lambda a: a['resource_policy'].update(whole_process_seconds=700),
            'reserve': lambda a: a['resource_policy'].update(exit_reserve_seconds=0),
            'memory': lambda a: a['resource_policy'].update(host_bytes=16 * 1024**3),
            'swap': lambda a: a['resource_policy'].update(swap_bytes=1),
            'swap bool alias': lambda a: a['resource_policy'].update(swap_bytes=False),
            'seconds float alias': lambda a: a['resource_policy'].update(whole_process_seconds=900.0),
            'cuda': lambda a: a['resource_policy'].update(cuda_visible_devices='0'),
            'locks held': lambda a: a.update(both_locks_held=False),
            'status': lambda a: a.update(candidate_status='GO'),
            'qualification': lambda a: a.update(qualification_eligible=True),
            'reuse': lambda a: a.update(state_reuse_eligible=True)}
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.read_authority(*self.authority(mutate)[1:])

    def test_rejects_source_roles_names_pins_and_driver(self):
        census = self.m.SOURCE_ROLES['census'][0]
        cases = {
            'missing role': lambda a: a['sources'].pop('scorer'),
            'extra role': lambda a: a['sources'].update(extra=a['sources']['command']),
            'wrong basename': lambda a: a['sources'].update(census=a['sources']['scorer']),
            'wrong driver': lambda a: a['sources'].update(driver=a['sources']['test']),
            'stale hash': lambda a: a['sources']['scorer'].update(sha256='0' * 64),
            'nested key': lambda a: a['sources']['scorer'].update(extra=1),
            'uppercase hash': lambda a: a['sources']['scorer'].update(sha256=a['sources']['scorer']['sha256'].upper()),
            'native sources pin': lambda a: a.update(native_sources=a['inputs'])}
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.read_authority(*self.authority(mutate)[1:])
        authority, path, digest = self.authority()
        with patch.dict(self.m.SOURCE_ROLES, {'census': (census, '0' * 64)}), self.assertRaises(ValueError):
            self.m.read_authority(path, digest)

    def test_file_descriptor_rejects_unsafe_files(self):
        tmp = self.tmp()
        target, link = tmp / 'target.bin', tmp / 'link.bin'
        target.write_bytes(b'x')
        link.symlink_to(target)
        digest = hashlib.sha256(b'x').hexdigest()
        for name, value in {'relative': {'path': 'target.bin', 'sha256': digest},
                            'symlink': {'path': str(link), 'sha256': digest},
                            'directory': {'path': str(tmp), 'sha256': digest},
                            'missing': {'path': str(tmp / 'none'), 'sha256': digest},
                            'extra': {'path': str(target), 'sha256': digest, 'bytes': 1},
                            'short hash': {'path': str(target), 'sha256': digest[:63]}}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.file_fact(value)
        self.assertEqual(self.m.file_fact({'path': str(target), 'sha256': digest}), target)
        with self.assertRaises(ValueError):
            self.m.read_file({'path': str(target), 'sha256': '0' * 64}, {})

    def test_duplicate_and_nonfinite_json_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(raw), self.assertRaises(ValueError):
                self.m.strict_json(raw)

    def test_contract_constants(self):
        m = self.m
        self.assertEqual((m.LIMITS['whole_process_seconds'], m.LIMITS['exit_reserve_seconds'],
                          m.LIMITS['host_bytes'], m.LIMITS['swap_bytes'], m.LIMITS['cuda_visible_devices']),
                         (900, 120, 8 * 1024**3, 0, ''))
        self.assertEqual((sum(m.BATCHES), len(m.BATCHES), m.BATCHES[-1], m.WIDTH), (1734, 14, 70, 81))
        self.assertEqual(m.FALSIFIER, {'endpoint': 'candidate-179061', 'query_index': 747,
                                       'expected_ap': 0.290910005569458})
        for role, (name, pin) in m.SOURCE_ROLES.items():
            if pin:
                self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(), pin, role)


class PreservedOriginalTests(Base):
    def test_original_failure_source_and_receipts_are_unchanged(self):
        census = HERE / 'census_connected_core_errors.py'
        self.assertEqual(hashlib.sha256(census.read_bytes()).hexdigest(), self.m.SOURCE_ROLES['census'][1])
        self.assertEqual(hashlib.sha256((HERE / 'test_connected_core_errors.py').read_bytes()).hexdigest(),
                         '22ec30708b26c6025026672b70dc78db0e2e401767840c968fbbd9c71246e7da')
        terminal = json.loads((EVIDENCE / 'connected-core-error-census-v1/terminal.json').read_text())
        self.assertEqual((terminal['decision'], terminal['failure_query_index'], terminal['census_published'],
                          terminal['original_candidate_decision']), ('FAIL_EXACT_REPLAY', 747, False, 'KILL'))
        self.assertEqual(terminal['expected_original_ap'], self.m.FALSIFIER['expected_ap'])
        accepted = EVIDENCE / 'connected-mlp-evaluation-full-selection-score-v1/receipt.json'
        self.assertEqual(hashlib.sha256(accepted.read_bytes()).hexdigest(),
                         '01ae023cb89b828c029817582cdb48204ff8177e76047ccec0773e5d90aafbfa')


class HelperTests(Base):
    def test_helpers_load_guard_and_close_without_native_imports(self):
        _, _, serving, owned, modules = self.helpers()
        self.assertFalse({n.split('.')[0] for n in sys.modules} & self.m.NATIVE)
        for name in ('Source', 'Locks'):
            self.assertTrue(hasattr(serving, name))
        api = {'census': ('admit_inputs', 'map_rows', 'core_queries', 'decode_wire', 'publish', 'rehash',
                          'ENDPOINTS', 'POPULATION'),
               'source_driver': ('package_origins', 'imported_origins', 'numerical_flags', 'cgroup_memory',
                                 'POLICY', 'SCHEMA'),
               'extract': ('sha',), 'quadratic_owner': ('audit_origins',), 'original': ('FlatAdmission',),
               'initializer': ('admit_cgroup',)}
        for role, names in api.items():
            for name in names:
                self.assertTrue(hasattr(modules[role], name), f'{role}.{name}')
        for source in owned:
            source.check()
        modules['source_driver'].SCHEMA = 'tampered'
        with self.assertRaises(ValueError):
            owned[2].check()
        modules['source_driver'].SCHEMA = 'siglip2-substrate-source-cpu-v1'
        names = [s.module.__name__ for s in owned]
        self.m.close_sources(owned)
        self.assertEqual(owned, [])
        self.assertFalse(any(n in sys.modules for n in names))

    def test_genuine_audit_origins_with_proof_as_sole_authority(self):
        _, _, _, _, modules = self.helpers()
        packages = {'torch': {'root': '/nonexistent/torch', 'origin': '/nonexistent/torch/__init__.py', 'version': '0'}}
        origins = modules['source_driver'].imported_origins(modules['extract'], packages)
        self.assertTrue(origins['files'])
        proof = {'origins': copy.deepcopy(origins)}
        guards = {}
        counts = self.m.make_audit(modules, proof, guards)()
        self.assertEqual(counts['files'], len(origins['files']))
        self.assertEqual(set(guards), set(origins['files']))
        first = next(iter(origins['files']))
        changed, unknown, moved = (copy.deepcopy(proof) for _ in range(3))
        changed['origins']['files'][first] = '0' * 64
        del unknown['origins']['files'][first]
        moved['origins']['modules'] = {'torch.fake': first}
        for name, bad in {'changed': changed, 'unknown': unknown}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.make_audit(modules, bad, {})()
        fake = copy.deepcopy(origins)
        fake['modules'] = {'torch': first}
        with patch.object(modules['source_driver'], 'imported_origins', return_value=fake), \
                self.assertRaises(ValueError):
            self.m.make_audit(modules, proof, {})()

    def test_package_origins_call_shape_and_proof_binding(self):
        authority, guards, _, _, modules = self.helpers()
        tmp = self.tmp()
        site = tmp / 'site'
        distributions = {'torch': 'torch', 'transformers': 'transformers', 'torchvision': 'torchvision',
                         'safetensors': 'safetensors', 'numpy': 'numpy', 'PIL': 'Pillow'}
        inits, versions = {}, {}
        for package, distribution in distributions.items():
            (site / package).mkdir(parents=True)
            (site / package / '__init__.py').write_text(f'# {package}\n')
            inits[package], versions[distribution.lower()] = site / package / '__init__.py', '1.0'
        constructor = site / 'constructor.py'
        constructor.write_text('x = 1\n')
        record = {'schema': 'paired-native256-vision-sources-v1', 'native_environment': {
            'schema': 'native256-installed-source-observation-v1', 'native_imported': False, 'model_executed': False,
            'quality_read': False, 'site_packages': str(site), 'versions': versions,
            'files': {str(constructor): {'sha256': fact(constructor)['sha256'], 'bytes': constructor.stat().st_size}},
            'vision_constructor': {'direct_bare_state_keys_source_observed': True, 'path': str(constructor),
                                   'assigned_self_attributes': ['config', 'embeddings', 'encoder', 'head',
                                                                'post_layernorm', 'use_head']}}}
        path = tmp / 'sources.json'
        path.write_text(json.dumps(record))
        packages = {p: {'root': str(site / p), 'origin': str(inits[p]), 'version': '1.0'} for p in distributions}
        proof = {'input_guards': {str(path): fact(path)['sha256']},
                 'origins': {'packages': packages, 'files': {str(i): fact(i)['sha256'] for i in inits.values()}}}
        authority = {'native_sources': fact(path)}
        stub_spec = lambda name: SimpleNamespace(origin=str(inits[name]))
        stub_dist = lambda name: SimpleNamespace(version='1.0', locate_file=lambda rel: site / rel)
        with patch.object(self.m, 'NATIVE_SOURCES_SHA', fact(path)['sha256']), \
                patch('importlib.util.find_spec', stub_spec), patch('importlib.metadata.distribution', stub_dist):
            self.m.check_packages(modules, authority, proof, {})
            bad_version = lambda name: SimpleNamespace(version='2.0', locate_file=lambda rel: site / rel)
            with patch('importlib.metadata.distribution', bad_version), self.assertRaises(ValueError):
                self.m.check_packages(modules, authority, proof, {})
            wrong = copy.deepcopy(proof)
            wrong['origins']['files'][str(inits['torch'])] = '0' * 64
            with self.assertRaises(ValueError):
                self.m.check_packages(modules, authority, wrong, {})
            wrong = copy.deepcopy(proof)
            wrong['origins']['packages']['torch']['version'] = '9'
            with self.assertRaises(ValueError):
                self.m.check_packages(modules, authority, wrong, {})
        with self.assertRaises(ValueError):
            self.m.check_packages(modules, authority, proof, {})

    def test_interpreter_must_be_the_original_proof_interpreter(self):
        _, _, _, _, modules = self.helpers()
        python = Path(sys.executable).resolve()
        good = {'invocation': {'python': str(python), 'python_sha256': modules['extract'].sha(python),
                               'python_version': sys.version}}
        self.m.check_interpreter(good, modules['extract'])
        for key in good['invocation']:
            bad = copy.deepcopy(good)
            bad['invocation'][key] = bad['invocation'][key] + 'x'
            with self.subTest(key), self.assertRaises(ValueError):
                self.m.check_interpreter(bad, modules['extract'])


class EvidenceTests(Base):
    def unit(self, mutate=None, log=None):
        authority = self.authority()[0]['source_cpu']
        if log is not None:
            path = self.tmp() / 'terminal.log'
            path.write_text(log)
            authority['log'] = fact(path)
        if mutate:
            mutate(authority)
        return authority

    def test_original_source_cpu_unit_is_admitted_without_input_rehash(self):
        modules, guards = self.helpers()[4], {}
        proof = self.m.admit_source_cpu(self.unit(), guards, modules)
        self.assertEqual(proof['invocation']['invocation_id'], '1e6981dda5fc44989dd59a81ec975687')
        self.assertFalse(any('/datasets/' in path or path.endswith('fresh_vision.pt') for path in guards))
        self.assertEqual(len(guards), 2)

    def test_source_cpu_unit_rejects_descriptor_log_and_cgroup_drift(self):
        _, guards, _, _, modules = self.helpers()
        log = (CPU / 'native256-source-cpu-so400-v4.log').read_text()
        cases = {
            'extra key': (lambda u: u.update(extra=1), None),
            'locks': (lambda u: u.update(both_locks_held=False), None),
            'proof pin': (lambda u: u.update(proof=fact(CPU / 'native256-source-cpu-large-v4.json')), None),
            'unit': (lambda u: u.update(unit='other-unit'), None),
            'invocation': (lambda u: u.update(invocation_id='0' * 32), None),
            'seconds': (lambda u: u.update(service_seconds=43.8), None),
            'seconds bool': (lambda u: u.update(service_seconds=True), None),
            'rss': (lambda u: u.update(native_peak_rss_kib=4502619), None),
            'exit status': (None, log.replace('\tExit status: 0\n', '\tExit status: 1\n')),
            'swaps': (None, log.replace('\tSwaps: 0\n', '\tSwaps: 1\n')),
            'duplicate line': (None, log + 'Finished with result: success\n'),
            'footer id': (None, log.replace('FINAL_CGROUP {"invocation_id": "1e69', 'FINAL_CGROUP {"invocation_id": "0e69')),
            'footer cap': (None, log.replace('"memory.max": "8589934592"', '"memory.max": "9589934592"')),
            'footer peak': (None, log.replace('"memory.peak": "6855409664"', '"memory.peak": "1"'))}
        for name, (mutate, text) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.admit_source_cpu(self.unit(mutate, text), guards, modules)

    def test_accepted_score_environment_is_inside_the_cpu_proof(self):
        proof = json.loads((CPU / 'native256-source-cpu-so400-v4.json').read_text())
        receipt = json.loads((EVIDENCE / 'connected-mlp-evaluation-full-selection-score-v1/receipt.json').read_text())
        self.m.check_accepted_environment(receipt, proof)
        self.assertEqual((len(receipt['origins']['files']), len(receipt['origins']['modules'])), (1361, 1291))
        cases = {
            'python': lambda r, p: r['invocation'].update(python_sha256='0' * 64),
            'flags': lambda r, p: r['numerical_flags'].update(threads=7),
            'cuda': lambda r, p: r['invocation'].update(cuda_visible_devices='0'),
            'packages': lambda r, p: r['origins']['packages']['torch'].update(version='9'),
            'new file': lambda r, p: r['origins']['files'].update({'/new/native.so': '0' * 64}),
            'changed file': lambda r, p: r['origins']['files'].update({next(iter(r['origins']['files'])): '0' * 64}),
            'new module': lambda r, p: r['origins']['modules'].update({'torch.new': '/new.py'}),
            'new native': lambda r, p: r['origins']['native_files'].append('/new/lib.so')}
        for name, mutate in cases.items():
            r, p = copy.deepcopy(receipt), copy.deepcopy(proof)
            mutate(r, p)
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.check_accepted_environment(r, p)


class ScorerTests(Base):
    MUTATIONS = {
        'reduction': ('.sum(dim=1) / counts', '.mean(dim=1) / counts'),
        'sum dim': ('.sum(dim=1) / counts', '.sum(dim=0) / counts'),
        'reassociated multiply': ('* inverse[rows, None] * inverse[None, gallery]',
                                  '* (inverse[rows, None] * inverse[None, gallery])'),
        'swapped multiply': ('* inverse[rows, None] * inverse[None, gallery]',
                             '* inverse[None, gallery] * inverse[rows, None]'),
        'unstable sort': ('stable=True', 'stable=False'),
        'ascending sort': ('descending=True', 'descending=False'),
        'batch size': ('range(0, len(query), 128)', 'range(0, len(query), 64)'),
        'width slice': ('[:, :width]', '[:, :80]'),
        'width max': ('width = int(relevant.max())', 'width = int(relevant.min())'),
        'code promotion': ('packed.codes.float()', 'packed.codes.double()'),
        'inverse promotion': ('packed.inverse_norms.float()', 'packed.inverse_norms'),
        'matmul spelling': ('(code[rows] @ code[gallery].T)', 'torch.matmul(code[rows], code[gallery].T)'),
        'precision dtype': ('/ ranks[None, :]', '/ ranks[None, :].float()'),
        'rank mask': ('ranks[None, :] <= counts[:, None]', 'ranks[None, :] < counts[:, None]'),
        'int32 scorer': ('(code[rows] @ code[gallery].T)', '(code[rows].int() @ code[gallery].T.int()).float()')}

    def mutate(self, text, old, new):
        self.assertEqual(text.count(old), 1, old)
        return text.replace(old, new)

    def test_pinned_scorer_bytes_and_function_ast(self):
        function = self.m.scorer_function(self.scorer)
        self.assertEqual(function.name, 'packed_quality')
        self.assertEqual(hashlib.sha256(self.scorer).hexdigest(), self.m.SOURCE_ROLES['scorer'][1])

    def test_any_computational_change_to_the_original_is_rejected(self):
        source = self.scorer.decode()
        for name, (old, new) in self.MUTATIONS.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.scorer_function(self.mutate(source, old, new).encode())

    def test_adapter_is_the_exact_inverse_and_only_touches_five_things(self):
        original = self.m.scorer_function(self.scorer)
        adapted = self.m.adapt(original)
        self.m.verify_adapter(original, adapted)
        self.assertEqual(self.m.dump(self.m.unadapt(adapted)), self.m.dump(original))
        text = ast.unparse(adapted)
        for gone in ('pack_int8_unit_embeddings', 'np.', 'recall_at_1', 'map_at_r', 'values'):
            self.assertNotIn(gone, text)
        loops = [n for n in adapted.body if isinstance(n, ast.For)]
        self.assertEqual(len(loops), 1)
        self.assertEqual(ast.unparse(loops[0].iter), 'range(0, len(query), 128)')
        self.assertEqual([ast.unparse(n) for n in loops[0].body if 'census_capture' in ast.unparse(n)],
                         ['census_capture(start, rows, scores)'])
        self.assertEqual(ast.unparse(loops[0].body[-1]), 'census_capture(start, rows, scores)')
        self.assertEqual([ast.unparse(n) for n in loops[0].body if 'census_batch' in ast.unparse(n)],
                         ['census_batch(start)'])
        self.assertEqual(ast.unparse(loops[0].body[0]), 'census_batch(start)')
        self.assertLess(0, next(i for i, n in enumerate(loops[0].body) if ' @ ' in ast.unparse(n)))
        self.assertEqual([a.arg for a in adapted.args.args], ['packed', 'labels', 'query', 'gallery'])
        for statement in ('code = packed.codes.float().to(device)', 'inverse = packed.inverse_norms.float().to(device)'):
            self.assertIn(statement, text)
        before = [ast.unparse(n) for n in original.body if not isinstance(n, (ast.For, ast.Return))]
        after = [ast.unparse(n) for n in adapted.body if not isinstance(n, (ast.For, ast.Return))]
        self.assertEqual([s for s in before if 'pack_int8' not in s], after)

    def test_tampered_adaptation_is_not_the_inverse(self):
        original = self.m.scorer_function(self.scorer)
        text = ast.unparse(self.m.adapt(original))
        extra = {'capture mutates scores': ('census_capture(start, rows, scores)',
                                            'scores = scores.double()\n        census_capture(start, rows, scores)'),
                 'capture twice': ('census_capture(start, rows, scores)',
                                   'census_capture(start, rows, scores)\n        census_capture(start, rows, scores)'),
                 'capture moved': ('        census_capture(start, rows, scores)\n', ''),
                 'batch hook removed': ('        census_batch(start)\n', ''),
                 'batch hook twice': ('census_batch(start)', 'census_batch(start)\n        census_batch(start)'),
                 'batch hook argument': ('census_batch(start)', 'census_batch(0)'),
                 'aggregate kept': ("return {'per_query_r1'", "return {'recall_at_1': 1.0, 'per_query_r1'"),
                 'extra statement': ('    hits: list[int] = []', '    packed = packed\n    hits: list[int] = []')}
        cases = {n: m for n, m in self.MUTATIONS.items() if n not in ('code promotion',)}
        cases['code promotion'] = ('packed.codes.float()', 'packed.codes.double()')
        for name, (old, new) in {**cases, **extra}.items():
            if old not in text:
                old = old.replace('(code[rows] @ code[gallery].T)', 'code[rows] @ code[gallery].T')
            with self.subTest(name):
                tampered = ast.parse(text.replace(old, new, 1)).body[0] if old in text else None
                self.assertIsNotNone(tampered, old)
                with self.assertRaises(ValueError):
                    self.m.verify_adapter(original, tampered)
        wrong_original = ast.parse(self.mutate(self.scorer.decode(), 'stable=True', 'stable=False')).body
        function = next(n for n in wrong_original if isinstance(n, ast.FunctionDef) and n.name == 'packed_quality')
        with self.assertRaises(ValueError):
            self.m.verify_adapter(function, self.m.adapt(original))

    def test_adapted_scorer_resolves_only_torch_and_the_two_hooks(self):
        adapted = self.m.adapt(self.m.scorer_function(self.scorer))
        module = compile(ast.Module(body=[adapted], type_ignores=[]), 'scorer', 'exec', dont_inherit=True)
        names, pending = set(), [module]
        while pending:
            code = pending.pop()
            names |= {i.argval for i in dis.get_instructions(code) if i.opname == 'LOAD_GLOBAL'}
            pending += [c for c in code.co_consts if hasattr(c, 'co_code')]
        self.assertEqual({n for n in names if not hasattr(builtins, n)}, {'torch', 'census_capture', 'census_batch'})

    def test_compile_scorer_defines_the_adapted_function_signature(self):
        torch = SimpleNamespace(inference_mode=lambda: (lambda fn: fn), device=type('device', (), {}))
        fn = self.m.compile_scorer(torch, self.scorer, str(HERE / 'compare_inshop_sop_warmstart_100.py'), print, print)
        parameters = inspect.signature(fn).parameters
        self.assertEqual(list(parameters), ['packed', 'labels', 'query', 'gallery', 'device'])
        self.assertEqual(parameters['device'].kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(ValueError):
            broken = self.mutate(self.scorer.decode(), 'stable=True', 'stable=False').encode()
            self.m.compile_scorer(torch, broken, 'x', print, print)

    def test_in_place_adapter_declaration_mutation_is_rejected_before_execution(self):
        declared = {'VALUES_ARG': 'values: np.ndarray', 'CAPTURE_STATEMENT': 'census_capture(start, rows, scores)',
                    'BATCH_STATEMENT': 'census_batch(start)',
                    'PACK_STATEMENT': 'packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))',
                    'AGGREGATES': "{'recall_at_1': float(np.mean(hits)), 'map_at_r': float(np.mean(aps))}"}
        for name, text in declared.items():
            self.assertEqual(ast.unparse(getattr(self.m, name)), text)
        forged = ast.parse('census_capture(start, rows, forged_scores)').body[0]
        cases = {'capture argument forged': lambda m: setattr(m.CAPTURE_STATEMENT.value.args[2], 'id', 'forged_scores'),
                 'capture rebound': lambda m: setattr(m, 'CAPTURE_STATEMENT', forged),
                 'batch argument forged': lambda m: setattr(m.BATCH_STATEMENT.value.args[0], 'id', 'forged_start'),
                 'batch rebound': lambda m: setattr(m, 'BATCH_STATEMENT', ast.parse('census_batch(0)').body[0]),
                 'values argument': lambda m: setattr(m.VALUES_ARG, 'arg', 'forged'),
                 'pack target': lambda m: setattr(m.PACK_STATEMENT.targets[0], 'id', 'forged'),
                 'aggregate key': lambda m: setattr(m.AGGREGATES.keys[0], 'value', 'forged')}
        control = load_driver()
        with patch.object(builtins, 'exec') as run, self.assertRaises(KeyError):
            control.compile_scorer(SimpleNamespace(), self.scorer, 'x', print, print)
        run.assert_called_once()
        for name, mutate in cases.items():
            with self.subTest(name):
                m = load_driver()
                mutate(m)
                with patch.object(builtins, 'exec') as run, self.assertRaises(ValueError):
                    m.compile_scorer(SimpleNamespace(), self.scorer, 'x', print, print)
                run.assert_not_called()

    def test_packed_input_keeps_original_dtypes_and_rejects_drift(self):
        class Tensor:
            def __init__(self, data, dtype):
                self.data, self.dtype = data, dtype
            def tolist(self):
                return [list(x) if isinstance(x, tuple) else x for x in self.data]
        rows = [((1, -2) + (0,) * 126, 0.5), ((127,) + (-128,) * 127, 0.000514984130859375)]
        torch = SimpleNamespace(int8='int8', float16='float16', tensor=Tensor)
        packed = self.m.packed_input(torch, rows)
        self.assertEqual((packed.codes.dtype, packed.inverse_norms.dtype), ('int8', 'float16'))
        drift = SimpleNamespace(int8='int8', float16='float16', tensor=lambda data, dtype: Tensor(
            [x + 1 if dtype == 'float16' else x for x in data], dtype))
        with self.assertRaises(ValueError):
            self.m.packed_input(drift, rows)


class ReplayTests(Base):
    def expected(self, n=6):
        return {'a': {'per_query_r1': [1, 0] * (n // 2), 'per_query_ap': [i / 7 for i in range(n)]},
                'b': {'per_query_r1': [0] * n, 'per_query_ap': [0.5] * n}}

    def test_exact_equality_is_required_everywhere(self):
        expected = self.expected()
        self.m.replay_exact(copy.deepcopy(expected), expected)
        cases = {
            'one ULP up': lambda r: r['a']['per_query_ap'].__setitem__(2, f32_neighbor(r['a']['per_query_ap'][2], 1)),
            'one ULP down': lambda r: r['b']['per_query_ap'].__setitem__(5, f32_neighbor(r['b']['per_query_ap'][5], -1)),
            'r1 flip': lambda r: r['a']['per_query_r1'].__setitem__(0, 0),
            'nan': lambda r: r['a']['per_query_ap'].__setitem__(1, math.nan),
            'short': lambda r: r['b']['per_query_ap'].pop(),
            'extra array': lambda r: r['b'].update(recall_at_1=0.),
            'missing array': lambda r: r['b'].pop('per_query_r1')}
        for name, mutate in cases.items():
            actual = copy.deepcopy(expected)
            mutate(actual)
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.replay_exact(actual, expected)

    def test_failure_lists_endpoint_query_and_both_values(self):
        expected, actual = self.expected(), self.expected()
        actual['a']['per_query_ap'][4] = f32_neighbor(expected['a']['per_query_ap'][4], 1)
        actual['b']['per_query_r1'][1] = 1
        with self.assertRaises(ValueError) as caught:
            self.m.replay_exact(actual, expected)
        message = str(caught.exception)
        self.assertIn('a: 1 differ', message)
        self.assertIn('b: 1 differ', message)
        self.assertIn(repr(expected['a']['per_query_ap'][4]), message)
        self.assertIn("('per_query_r1', 1, 1, 0)", message)

    def test_budget_reserve_and_whole_process_boundaries(self):
        clock = Clock()
        budget = self.m.Budget(started=0., clock=clock)
        clock.now = 779.99
        budget.check()
        clock.now = 780.
        with self.assertRaises(ValueError):
            budget.check()
        budget.check(reserve=False)
        clock.now = 899.99
        budget.check(reserve=False)
        clock.now = 900.
        with self.assertRaises(ValueError):
            budget.check(reserve=False)

    def test_capture_keeps_only_core_rows_across_original_batches(self):
        clock, retained = Clock(), {}
        core = {0, 127, 128, 1663, 1664, 1733}
        capture = self.m.make_capture(core, retained, self.m.Budget(started=0., clock=clock))
        row = lambda index: [float(index), -1.5]
        for start in range(0, 1734, 128):
            size = min(128, 1734 - start)
            capture(start, list(range(start, start + size)), Scores({i: row(start + i) for i in range(size)}))
        self.assertEqual(retained, {i: row(i) for i in core})
        self.assertEqual([min(128, 1734 - s) for s in range(0, 1734, 128)], list(self.m.BATCHES))
        with self.assertRaises(ValueError):
            capture(0, [0], Scores({0: row(0)}))
        clock.now = 780.
        with self.assertRaises(ValueError):
            capture(128, [128], Scores({0: row(128)}))

    def test_geometry_uses_stable_full_gallery_order_and_ties(self):
        gallery = list(range(1, 101))

        def describe(positives, ties, r1):
            labels = ['Q'] + ['Q' if i in positives else 'N' for i in range(100)]
            scores = [1.0 - i * 0.001 for i in range(100)]
            for i in ties:
                scores[i] = 2.0
            return scores, self.m.describe_scores(scores, labels, 0, gallery, {'per_query_r1': r1, 'per_query_ap': 0.25})
        # impostor 5 ties positive 9 at the top: lower gallery index wins the tie
        scores, values = describe({9, 60}, (5, 9), 0)
        self.assertEqual((values['top_impostor_gallery_index'], values['top_impostor_rank']), (5, 1))
        self.assertEqual((values['best_positive_gallery_index'], values['best_positive_rank']), (9, 2))
        self.assertEqual((values['positive_count'], values['positive_minus_impostor_margin']), (2, 0.0))
        # positive 5 ties impostor 9: the positive now comes first
        scores, values = describe({5, 60}, (5, 9), 1)
        self.assertEqual((values['best_positive_gallery_index'], values['best_positive_rank']), (5, 1))
        self.assertEqual((values['top_impostor_gallery_index'], values['top_impostor_rank']), (9, 2))
        # the first positive is below the original AP width of 81 yet still ranked in the full gallery
        scores, values = describe({90, 95}, (), 0)
        self.assertEqual((values['best_positive_rank'], values['top_impostor_rank']), (91, 1))
        self.assertEqual((values['best_positive_score'], values['top_impostor_score']), (scores[90], scores[0]))
        self.assertEqual((values['per_query_r1'], values['per_query_ap']), (0, 0.25))

    def test_geometry_rejects_disagreement_and_malformed_rows(self):
        gallery = list(range(1, 121))
        labels = ['Q'] + ['Q'] * 3 + ['N'] * 117
        scores = [1.0 - i * 0.001 for i in range(120)]
        self.m.describe_scores(scores, labels, 0, gallery, {'per_query_r1': 1, 'per_query_ap': 1.})
        cases = {'derived r1 differs': (scores, {'per_query_r1': 0, 'per_query_ap': 1.}),
                 'nan': ([math.nan] + scores[1:], {'per_query_r1': 1, 'per_query_ap': 1.}),
                 'short': (scores[:-1], {'per_query_r1': 1, 'per_query_ap': 1.})}
        for name, (row, replayed) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.describe_scores(row, labels, 0, gallery, replayed)
        with self.assertRaises(ValueError):
            self.m.describe_scores(scores, ['Q'] * 121, 0, gallery, {'per_query_r1': 1, 'per_query_ap': 1.})
        with self.assertRaises(ValueError):
            self.m.describe_scores(scores, ['Q'] + ['N'] * 120, 0, gallery, {'per_query_r1': 1, 'per_query_ap': 1.})


class GatingTests(Base):
    """replay() may reach geometry only after every pair is exact; no aggregate is ever emitted."""
    QUERIES, GALLERY, POSITIVES = 800, 90, 81

    def context(self, tmp=None):
        query, gallery = list(range(self.QUERIES)), list(range(self.QUERIES, self.QUERIES + self.GALLERY))
        rows = [{'product': 'P' if i < self.QUERIES + self.POSITIVES else 'N', 'panel_ordinal': i}
                for i in query + gallery]
        census = SimpleNamespace(ENDPOINTS=(('179061', 'control'), ('179061', 'candidate'),
                                            ('179069', 'control'), ('179069', 'candidate')))
        quality = {s: {a: {'per_query_r1': [i % 2 for i in query],
                           'per_query_ap': [0.25 + i / 4096 for i in query]} for a in ('control', 'candidate')}
                   for s in ('179061', '179069')}
        quality['179061']['candidate']['per_query_ap'][747] = self.m.FALSIFIER['expected_ap']
        state = {'census': census, 'rows': rows, 'core': [3, 747, 799], 'wires': {
                     f'{a}-{s}': f'wire-{a}-{s}' for s, a in census.ENDPOINTS},
                 'partition': {'panels': {'selection': {'query': query, 'gallery': gallery}}},
                 'receipt': {'quality': quality, 'decision': 'KILL'}}
        scorer = fact(HERE / 'compare_inshop_sop_warmstart_100.py')
        return SimpleNamespace(
            torch=SimpleNamespace(device=lambda name: name, __version__='fixture'), state=state, flags={},
            budget=self.m.Budget(started=0., clock=Clock()), guards={}, guard=lambda: None,
            authority={'sources': {'scorer': scorer}}, payload=None)

    def scripted(self, ctx, tamper=None):
        calls = []
        self.matmuls = matmuls = []
        expected = {f'{a}-{s}': ctx.state['receipt']['quality'][s][a] for s, a in ctx.state['census'].ENDPOINTS}
        order = iter(expected)
        def compile_scorer(torch, raw, path, capture, batch):
            key = next(order)
            result = copy.deepcopy({k: expected[key][k] for k in ('per_query_r1', 'per_query_ap')})
            if tamper:
                tamper(key, result)
            def fn(packed, labels, query, gallery, *, device):
                calls.append((key, packed, device, len(query), len(gallery)))
                for start in range(0, len(query), 128):
                    batch(start)
                    matmuls.append((key, start))
                    size = min(128, len(query) - start)
                    capture(start, query[start:start + size], Scores({i: [0.5] * len(gallery) for i in range(size)}))
                return result
            return fn
        return compile_scorer, calls

    def test_exact_replay_precedes_geometry_and_uses_original_batches(self):
        ctx = self.context()
        compile_scorer, calls = self.scripted(ctx)
        with patch.object(self.m, 'compile_scorer', compile_scorer), patch.object(self.m, 'packed_input', lambda t, w: w), \
                patch.object(self.m, 'build_census', return_value={'built': True}) as build:
            self.m.replay(ctx)
        self.assertEqual(ctx.payload, {'built': True})
        self.assertEqual([c[0] for c in calls], ['control-179061', 'candidate-179061', 'control-179069', 'candidate-179069'])
        self.assertTrue(all(c[2] == 'cpu' and c[3:] == (self.QUERIES, self.GALLERY) for c in calls))
        retained = build.call_args.args[1]
        self.assertTrue(all(set(rows) == {3, 747, 799} for rows in retained.values()))
        self.assertEqual(ctx.state['wires']['control-179061'], calls[0][1])

    def test_any_mismatch_or_missing_core_row_stops_before_geometry(self):
        mutations = {
            'last pair one ULP': lambda key, r: key == 'candidate-179069' and r['per_query_ap'].__setitem__(
                799, f32_neighbor(r['per_query_ap'][799], 1)),
            'falsifier ULP': lambda key, r: key == 'candidate-179061' and r['per_query_ap'].__setitem__(
                747, 0.2909099757671356),
            'r1': lambda key, r: key == 'control-179061' and r['per_query_r1'].__setitem__(10, 1)}
        for name, tamper in mutations.items():
            ctx = self.context()
            compile_scorer, _ = self.scripted(ctx, tamper)
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', lambda t, w: w), \
                    patch.object(self.m, 'build_census') as build, self.assertRaises(ValueError):
                self.m.replay(ctx)
            build.assert_not_called()
            self.assertIsNone(ctx.payload)
        ctx = self.context()
        ctx.state['receipt']['quality']['179061']['candidate']['per_query_ap'][747] = 0.5
        compile_scorer, _ = self.scripted(ctx)
        with patch.object(self.m, 'compile_scorer', compile_scorer), patch.object(self.m, 'packed_input', lambda t, w: w), \
                patch.object(self.m, 'build_census') as build, self.assertRaises(ValueError):
            self.m.replay(ctx)
        build.assert_not_called()

    def test_reserve_crossing_skips_the_next_matmul_at_prelude_capture_or_between_batches(self):
        crossing = {'prelude': None, 'before capture of batch 1': ('metric', 1), 'between batches 1 and 2': ('gap', 1)}
        expected = {'prelude': [], 'before capture of batch 1': [0, 128], 'between batches 1 and 2': [0, 128]}
        for name, trigger in crossing.items():
            ctx, matmuls = self.context(), []

            def compile_scorer(torch, raw, path, capture, batch):
                def fn(packed, labels, query, gallery, *, device):
                    if trigger is None:
                        ctx.budget.clock.now = 780.
                    for index, start in enumerate(range(0, len(query), 128)):
                        batch(start)
                        matmuls.append(start)
                        if trigger == ('metric', index):
                            ctx.budget.clock.now = 780.
                        capture(start, query[start:start + 128], Scores({i: [0.5] * len(gallery) for i in range(128)}))
                        if trigger == ('gap', index):
                            ctx.budget.clock.now = 780.
                    return {}
                return fn
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', lambda t, w: w), \
                    patch.object(self.m, 'build_census') as build, self.assertRaises(ValueError):
                self.m.replay(ctx)
            self.assertEqual(matmuls, expected[name])
            build.assert_not_called()

    def test_final_check_follows_preparation_and_precedes_the_original_call(self):
        for name in ('compile', 'packed input'):
            ctx, called = self.context(), []
            def fn(*args, **kwargs):
                called.append(1)
            def compile_scorer(torch, raw, path, capture, batch):
                if name == 'compile':
                    ctx.budget.clock.now = 780.
                return fn
            def packed_input(torch, wire):
                if name == 'packed input':
                    ctx.budget.clock.now = 780.
                return wire
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', packed_input), self.assertRaises(ValueError):
                self.m.replay(ctx)
            self.assertEqual(called, [])

    def test_reserve_blocks_metric_work_and_width_is_checked(self):
        ctx = self.context()
        ctx.budget = self.m.Budget(started=0., clock=lambda: 780.)
        with patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError):
            self.m.replay(ctx)
        compile_scorer.assert_not_called()
        ctx = self.context()
        ctx.state['rows'][self.QUERIES + self.POSITIVES - 1]['product'] = 'other'
        with patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError):
            self.m.replay(ctx)
        compile_scorer.assert_not_called()

    def test_census_has_no_aggregate_or_raw_rows_and_is_finite_json(self):
        ctx = self.context()
        retained = {k: {i: [1.0 - j * 0.001 for j in range(self.GALLERY)] for i in ctx.state['core']}
                    for k in ('control-179061', 'candidate-179061', 'control-179069', 'candidate-179069')}
        results = {}
        for s, a in ctx.state['census'].ENDPOINTS:
            quality = ctx.state['receipt']['quality'][s][a]
            results[f'{a}-{s}'] = {k: list(quality[k]) for k in ('per_query_r1', 'per_query_ap')}
            for index in ctx.state['core']:
                results[f'{a}-{s}']['per_query_r1'][index] = 0
        # query 0 is positive for every gallery row; make top-1 positive to match R1=1 where needed
        for rows in results.values():
            for index in ctx.state['core']:
                rows['per_query_r1'][index] = 1
        payload = self.m.build_census(ctx.state, retained, results, {'fixture': True})
        text = json.dumps(payload, allow_nan=False)
        self.assertNotIn('recall_at_1', text)
        self.assertNotIn('map_at_r', text)
        self.assertNotIn('bootstrap', text)
        self.assertEqual((payload['scored_queries'], payload['gallery_rows'], payload['endpoints'], payload['scored_pairs']),
                         (3, self.GALLERY, 4, 3 * self.GALLERY * 4))
        self.assertEqual(payload['replay']['per_query_pairs'], self.QUERIES * 4)
        self.assertIsNone(payload['replay']['tolerance'])
        self.assertEqual(payload['replay']['falsifier']['actual_ap'], self.m.FALSIFIER['expected_ap'])
        self.assertEqual(set(payload['core_score_rows_sha256']), set(retained))
        self.assertEqual(payload['candidate_status'], 'KILL unchanged')
        self.assertFalse(payload['qualification_eligible'])

    def test_core_census_bytes_are_unchanged_by_the_all_query_mode(self):
        ctx = self.context()
        retained = {k: {i: [1.0 - j * 0.001 for j in range(self.GALLERY)] for i in ctx.state['core']}
                    for k in ('control-179061', 'candidate-179061', 'control-179069', 'candidate-179069')}
        results = {f'{a}-{s}': {k: list(ctx.state['receipt']['quality'][s][a][k]) for k in ('per_query_r1', 'per_query_ap')}
                   for s, a in ctx.state['census'].ENDPOINTS}
        for rows in results.values():
            for index in ctx.state['core']:
                rows['per_query_r1'][index] = 1
        raw = json.dumps(self.m.build_census(ctx.state, retained, results, {'fixture': True}), sort_keys=True,
                         indent=2, allow_nan=False)
        # digest of the same fixture rendered by the frozen census-v1 driver (ccbae231...) before the all-query mode existed
        self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                         '3f31de6c10c2f99f6d81077eaa137e295654ff906829d5c8aca7110bfaed1f16')


class ExitTests(Base):
    def test_cleanup_runs_everything_and_keeps_the_first_error(self):
        seen = []
        def failing(name):
            def run():
                seen.append(name)
                raise RuntimeError(name)
            return run
        first = ValueError('body')
        with self.assertRaises(ValueError) as caught:
            self.m.cleanup_error(first, [failing('a'), lambda: seen.append('b'), failing('c')])
        self.assertIs(caught.exception, first)
        self.assertEqual(seen, ['a', 'b', 'c'])
        self.assertEqual(len(first.__notes__), 2)
        with self.assertRaises(RuntimeError) as caught:
            self.m.cleanup_error(None, [failing('x'), failing('y')])
        self.assertEqual(str(caught.exception), 'x')
        self.m.cleanup_error(None, [])

    def test_exit_runs_origins_every_guard_sources_and_disposal(self):
        authority, guards, _, owned, modules = self.helpers()
        path = self.tmp() / 'guarded.txt'
        path.write_text('x')
        guards[str(path)] = fact(path)['sha256']
        audits, locks, rehashed = [], [], []
        ctx = SimpleNamespace(
            audit=lambda: audits.append(1) or {'files': 1}, origins={}, guards=guards, state=None, torch=None,
            locks=SimpleNamespace(check=lambda: locks.append(1)), owned=owned, modules=modules, own=None, proof=None)
        self.m.cleanup_error(None, self.m.exit_checks(ctx))
        self.assertEqual((audits, locks, owned, ctx.origins), ([1], [1], [], {'final': {'files': 1}}))
        path.write_text('changed')
        _, guards2, _, owned2, _ = self.helpers()
        guards2[str(path)] = fact(path)['sha256']
        path.write_text('again')
        ctx = SimpleNamespace(audit=None, origins={}, guards=guards2, state=None, torch=None, locks=None, owned=owned2,
                              own=None, proof=None)
        with self.assertRaises(ValueError):
            self.m.cleanup_error(None, self.m.exit_checks(ctx))
        self.assertEqual(owned2, [], 'sources are disposed even when an exit proof fails')

    def test_exit_repeats_the_admission_interpreter_predicate_uncached(self):
        extract = self.helpers()[4]['extract']
        tmp = self.tmp()
        real, other, launch = tmp / 'python-real', tmp / 'python-other', tmp / 'python'
        real.write_bytes(b'interpreter-a')
        other.write_bytes(b'interpreter-a')
        launch.symlink_to(real)
        invocation = {'python': str(real.resolve()), 'python_sha256': extract.sha(real), 'python_version': sys.version}
        ctx = SimpleNamespace(audit=None, origins={}, guards={}, state=None, torch=None, locks=None, owned=[], own=None,
                              proof={'invocation': invocation}, modules={'extract': extract})
        with patch.object(sys, 'executable', str(launch)):
            self.m.cleanup_error(None, self.m.exit_checks(ctx))
            real.write_bytes(b'interpreter-b')
            with self.assertRaises(ValueError):
                self.m.cleanup_error(None, self.m.exit_checks(ctx))
            real.write_bytes(b'interpreter-a')
            self.m.cleanup_error(None, self.m.exit_checks(ctx))
            launch.unlink()
            launch.symlink_to(other)
            with self.assertRaises(ValueError):
                self.m.cleanup_error(None, self.m.exit_checks(ctx))

    def test_exit_state_requires_hidden_cuda_flags_cgroup_and_cap(self):
        modules = {'source_driver': SimpleNamespace(numerical_flags=lambda: {'threads': 8},
                                                    cgroup_memory=lambda: {'path': '/cg/unit.service'}),
                   'initializer': SimpleNamespace(admit_cgroup=lambda value, unit: None)}
        def ctx(**override):
            base = dict(audit=None, origins={}, guards={}, state=None, locks=None, owned=[], modules=modules, own=None,
                        proof=None,
                        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)), flags={'threads': 8},
                        before={'path': '/cg/unit.service'}, unit='unit', budget=self.m.Budget(started=0., clock=lambda: 10.))
            return SimpleNamespace(**{**base, **override})
        good = ctx()
        self.m.cleanup_error(None, self.m.exit_checks(good))
        self.assertEqual(good.after['path'], '/cg/unit.service')
        self.assertGreater(good.peak, 0)
        bad = {'cuda': dict(torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))),
               'flags': dict(flags={'threads': 7}),
               'cgroup moved': dict(before={'path': '/cg/other.service'}),
               'cap': dict(budget=self.m.Budget(started=0., clock=lambda: 900.))}
        for name, override in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.cleanup_error(None, self.m.exit_checks(ctx(**override)))


class PublishTests(Base):
    """Stage with the original publish, promote by exclusive link under the cap; only our staged inode is removed."""
    PAYLOAD = {'schema': 'x', 'full_uncached_exit_pass': True}
    RAW = (json.dumps(PAYLOAD, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

    def attempt(self, before_link=None, after_link=None, cross_before=False, staged=None, cleanup_error=None):
        _, _, _, owned, modules = self.helpers()
        census = modules['census']
        self.m.close_sources(owned)
        if staged is not None:
            real = census
            census = SimpleNamespace(
                publish=lambda out, payload, guards: (real.publish(out, payload, guards), staged(out)))
        directory = self.tmp()
        output = directory / 'census.json'

        def clock():
            if not os.path.lexists(output):
                if before_link is not None:
                    before_link(directory, output)
                return 901. if cross_before else 0.
            if after_link is None:
                return 0.
            after_link(directory, output)
            return 901.
        real_cleanup = tempfile.TemporaryDirectory.cleanup

        def failing_cleanup(staging):
            real_cleanup(staging)
            raise cleanup_error
        error = None
        with patch.object(tempfile.TemporaryDirectory, 'cleanup', failing_cleanup) if cleanup_error is not None \
                else contextlib.nullcontext():
            try:
                self.m.publish_census(census, str(output), self.PAYLOAD, {}, self.m.Budget(started=0., clock=clock))
            except Exception as failure:
                error = failure
        return directory, output, error

    def test_success_leaves_exactly_the_original_publication_and_no_staging(self):
        directory, output, error = self.attempt()
        self.assertIsNone(error)
        self.assertEqual((output.read_bytes(), [p.name for p in directory.iterdir()]), (self.RAW, ['census.json']))

    def test_cap_crossing_before_promotion_publishes_nothing(self):
        directory, output, error = self.attempt(cross_before=True)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(list(directory.iterdir()), [])

    def test_cap_crossing_after_promotion_removes_the_owned_output_and_leaves_no_staging(self):
        directory, output, error = self.attempt(after_link=lambda d, o: None)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(list(directory.iterdir()), [])

    def test_staged_file_failure_before_promotion_publishes_nothing(self):
        def missing(out):
            os.unlink(out)

        def symlink(out):
            os.unlink(out)
            os.symlink('/nonexistent-target', out)

        def directory_instead(out):
            os.unlink(out)
            os.mkdir(out)
        for name, staged in {'lstat fails': missing, 'symlink': symlink, 'not regular': directory_instead}.items():
            with self.subTest(name):
                directory, output, error = self.attempt(staged=staged)
                self.assertIsInstance(error, (OSError, ValueError))
                self.assertEqual(list(directory.iterdir()), [])

    def test_foreign_output_before_or_after_promotion_is_never_removed(self):
        def replace(data):
            def act(directory, output):
                foreign = directory / 'foreign.tmp'
                foreign.write_bytes(data)
                os.replace(foreign, output)
            return act

        def symlink(directory, output):
            output.unlink()
            (directory / 'target').write_bytes(self.RAW)
            output.symlink_to(directory / 'target')

        def edit(directory, output):
            with output.open('ab') as stream:
                stream.write(b'x')
        cases = {'different bytes': (replace(b'foreign'), b'foreign'),
                 'same bytes new inode': (replace(self.RAW), self.RAW),
                 'symlink': (symlink, self.RAW), 'edited in place': (edit, self.RAW + b'x')}
        for name, (act, content) in cases.items():
            with self.subTest(name):
                directory, output, error = self.attempt(after_link=act)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(output.read_bytes(), content)
                self.assertTrue(any('foreign' in note for note in error.__notes__), error.__notes__)
                self.assertEqual(sorted(p.name for p in directory.iterdir()),
                                 sorted(['census.json'] + (['target'] if name == 'symlink' else [])))
        with self.subTest('foreign file appears before promotion'):
            directory, output, error = self.attempt(before_link=lambda d, o: o.write_bytes(b'foreign'))
            self.assertIsInstance(error, FileExistsError)
            self.assertEqual(output.read_bytes(), b'foreign')
            self.assertEqual([p.name for p in directory.iterdir()], ['census.json'])
            self.assertEqual(getattr(error, '__notes__', []), [], 'nothing was promoted, so nothing is removed')
        with self.subTest('owned output vanished'):
            directory, output, error = self.attempt(after_link=lambda d, o: o.unlink())
            self.assertIsInstance(error, ValueError)
            self.assertEqual(list(directory.iterdir()), [])

    def test_staging_cleanup_failure_after_promotion_still_removes_only_the_owned_output(self):
        clean = OSError('staging cleanup failed')
        directory, output, error = self.attempt(cleanup_error=clean)
        self.assertIs(error, clean)
        self.assertEqual(list(directory.iterdir()), [])
        crossed = OSError('cleanup after cap crossing')
        directory, output, error = self.attempt(after_link=lambda d, o: None, cleanup_error=crossed)
        self.assertIs(error, crossed)
        self.assertIsInstance(error.__context__, ValueError)
        self.assertEqual(list(directory.iterdir()), [])

        def replace(directory, output):
            foreign = directory / 'foreign.tmp'
            foreign.write_bytes(b'foreign')
            os.replace(foreign, output)
        foreign_error = OSError('cleanup with foreign output')
        directory, output, error = self.attempt(after_link=replace, cleanup_error=foreign_error)
        self.assertIs(error, foreign_error)
        self.assertEqual((output.read_bytes(), [p.name for p in directory.iterdir()]), (b'foreign', ['census.json']))
        self.assertTrue(any('foreign' in note for note in error.__notes__), error.__notes__)
        before = OSError('cleanup after a failure before promotion')
        directory, output, error = self.attempt(cross_before=True, cleanup_error=before)
        self.assertIs(error, before)
        self.assertEqual(list(directory.iterdir()), [])

    def test_run_publishes_only_through_the_owned_output_guard(self):
        run = next(n for n in ast.parse(DRIVER.read_text()).body if getattr(n, 'name', None) == 'run')
        text = ast.unparse(run)
        self.assertIn("publish_census(ctx.state['census'], args.output, ctx.payload, ctx.state['guards'], "
                      "ctx.budget)", text)
        self.assertNotIn('.publish(', text)
        self.assertNotIn('check(reserve=False)', text)


class OwnSourceTests(Base):
    """The running driver is live code too: a real serving.Source over this module, kept out of the owned set."""
    def own(self, path=DRIVER):
        m = load_driver(path)
        sys.modules[m.__name__] = m
        self.addCleanup(sys.modules.pop, m.__name__, None)
        authority = self.authority()[0]
        authority['sources']['driver'] = fact(path)
        _, _, serving, owned, _ = self.helpers(authority)
        try:
            source = m.own_source(serving, authority)
            m.check_own(source)
        finally:
            self.m.close_sources(owned)
        return m, source

    def test_live_mutation_file_registry_and_policy_types_are_caught(self):
        def code(m):
            m.require.__code__ = (lambda condition, message: None).__code__

        def registry(m):
            sys.modules[m.__name__] = SimpleNamespace()
        cases = {
            'function code': code,
            'function default': lambda m: setattr(m.Budget.check, '__defaults__', (False,)),
            'global': lambda m: setattr(m, 'WIDTH', 82),
            'dict literal': lambda m: m.FALSIFIER.update(query_index=0),
            'registry': registry,
            'policy bool alias': lambda m: m.LIMITS.update(swap_bytes=False),
            'policy float alias': lambda m: m.LIMITS.update(whole_process_seconds=900.0)}
        for name, mutate in cases.items():
            with self.subTest(name):
                m, source = self.own()
                mutate(m)
                with self.assertRaises(ValueError):
                    m.check_own(source)
        copy = self.tmp() / DRIVER.name
        copy.write_bytes(DRIVER.read_bytes())
        m, source = self.own(copy)
        copy.write_bytes(copy.read_bytes() + b'\n')
        with self.assertRaises(ValueError):
            m.check_own(source)

    def test_guard_and_exit_check_the_own_source_and_never_remove_it(self):
        m, source = self.own()
        owned = self.helpers()[3]
        ctx = SimpleNamespace(locks=SimpleNamespace(check=lambda: None), owned=owned, own=source, audit=None,
                              origins={}, guards={}, state=None, torch=None, proof=None)
        m.guard(ctx)
        m.cleanup_error(None, m.exit_checks(ctx))
        self.assertEqual(owned, [])
        self.assertIs(sys.modules[m.__name__], m)
        m.LIMITS['swap_bytes'] = False
        checks = {'guard': lambda: m.guard(ctx), 'exit': lambda: m.cleanup_error(None, m.exit_checks(ctx))}
        for name, check in checks.items():
            with self.subTest(name), self.assertRaises(ValueError):
                check()

    def test_admit_authenticates_the_own_source_after_helpers_and_before_native_import(self):
        admit = next(n for n in ast.parse(DRIVER.read_text()).body if getattr(n, 'name', None) == 'admit')
        lines = [ast.unparse(n) for n in admit.body]
        at = lambda text: next(i for i, line in enumerate(lines) if text in line)
        self.assertLess(at('load_helpers('), at('own_source('))
        self.assertLess(at('own_source('), at('check_own(ctx.own)'))
        self.assertLess(at('check_own(ctx.own)'), at('import torch'))


class ProcessTests(Base):
    def run_driver(self, *args, env=None, flags=('-B',)):
        base = {k: v for k, v in os.environ.items() if k not in ('PYTHONDONTWRITEBYTECODE', 'CUDA_VISIBLE_DEVICES', 'INVOCATION_ID')}
        return subprocess.run([sys.executable, *flags, str(DRIVER), *args], capture_output=True, text=True,
                              env={**base, **(env or {})}, timeout=60)

    def test_import_and_help_never_touch_native_modules(self):
        code = ("import importlib.util,sys;s=importlib.util.spec_from_file_location('d',%r);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "print(sorted(n for n in sys.modules if n.split('.')[0] in m.NATIVE))" % str(DRIVER))
        done = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True, timeout=60)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, '[]'), done.stderr)
        helped = self.run_driver('--help')
        self.assertEqual(helped.returncode, 0)
        self.assertIn('--authority-sha256', helped.stdout)

    def test_cli_rejects_without_output_and_before_any_authority_read(self):
        output = self.tmp() / 'census.json'
        authority, path, digest = self.authority()
        args = ('--authority', path, '--authority-sha256', digest, '--output', str(output))
        cases = {'cuda unset': (self.run_driver(*args, env={'INVOCATION_ID': INVOCATION}), 'CUDA must be explicitly hidden'),
                 'cuda visible': (self.run_driver(*args, env={'CUDA_VISIBLE_DEVICES': '0', 'INVOCATION_ID': INVOCATION}),
                                  'CUDA must be explicitly hidden'),
                 'no unit': (self.run_driver(*args, env={'CUDA_VISIBLE_DEVICES': ''}), 'systemd invocation'),
                 'bytecode': (self.run_driver(*args, env={'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': INVOCATION},
                                              flags=()), 'unoptimized -B'),
                 'bad authority hash': (self.run_driver('--authority', path, '--authority-sha256', '0' * 64,
                                                        '--output', str(output),
                                                        env={'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': INVOCATION}),
                                        'exact Torch core census rejected')}
        for name, (done, message) in cases.items():
            with self.subTest(name):
                self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
                self.assertIn(message, done.stderr)
                self.assertFalse(os.path.lexists(output))
        output.write_text('existing')
        done = self.run_driver(*args, env={'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': INVOCATION})
        self.assertEqual((done.returncode, output.read_text()), (1, 'existing'))
        self.assertIn('exclusive canonical NEWFILE', done.stderr)


@unittest.skipUnless(INPUTS.is_file(), 'root local census inputs absent')
class RealInputTests(Base):
    def test_real_inputs_through_the_authenticated_loader_and_original_evidence(self):
        m = self.m
        authority = self.authority(lambda a: a.update(inputs=fact(INPUTS)))[0]
        _, guards, serving, owned, modules = self.helpers(authority)
        state = modules['census'].admit_inputs(authority['inputs']['path'], authority['inputs']['sha256'])
        panel = state['partition']['panels']['selection']
        labels = [r['product'] for r in state['rows']]
        gallery_counts = m.Counter(labels[i] for i in panel['gallery'])
        self.assertEqual((len(state['core']), len(panel['query']), len(panel['gallery']),
                          max(gallery_counts[labels[i]] for i in panel['query'])), (44, 1734, 1715, m.WIDTH))
        quality = state['receipt']['quality'][m.FALSIFIER['endpoint'].split('-')[1]]['candidate']
        self.assertEqual(quality['per_query_ap'][m.FALSIFIER['query_index']], m.FALSIFIER['expected_ap'])
        self.assertIn(m.FALSIFIER['query_index'], state['core'])
        self.assertEqual(set(state['wires']), {f'{a}-{s}' for s, a in modules['census'].ENDPOINTS})
        proof = m.admit_source_cpu(authority['source_cpu'], guards, modules)
        m.check_accepted_environment(state['receipt'], proof)
        for source in owned:
            source.check()


# ======================================================================================================
# All-query margin mode (--mode all-query). Additive falsifiers; everything above keeps guarding the core
# mode, and PreservationTests below proves the core mode itself is unchanged.
# ======================================================================================================
PRIOR_RECEIPT = EVIDENCE / 'connected-exact-torch-census-v1/receipt.json'
SEEDS = ('179061', '179069')
KEYS = tuple(f'{arm}-{seed}' for seed in SEEDS for arm in ('control', 'candidate'))
Q, G = 1734, 90
IMPOSTORS = [j for j in range(G) if j % 10 == 0]  # nine impostors interleaved with 81 positives
POSITIVES = [j for j in range(G) if j % 10 != 0]
EDITED = ('read_authority', 'parser', 'main')
ADDED = ('ALL_QUERY_LAUNCH_SCHEMA', 'ALL_QUERY_RESULT_SCHEMA', 'ALL_QUERY_KEYS', 'PRIOR_CENSUS_SHA', 'PRIOR_TRUE',
         'QUERY_COUNT', 'SEEDS', 'ENDPOINT_KEYS', 'FROZEN_TRANSITIONS', 'TRANSITION', 'strictly', 'admit_prior_census',
         'read_all_query_authority', 'selection_mapping_sha256', 'fetch_identity', 'bind_prior', 'margin_r1', 'spread',
         'movement', 'cohort_summary', 'build_all_query_census', 'replay_all_query', 'run_all_query')
# Digest of every top-level node of the frozen census-v1 driver (sha256 ccbae231...), taken before this mode existed.
HISTORICAL_NODES = (
    ('<docstring>', '12bcef74ce690119cb729f3bc7eea11789da1e7ec641065b75d8e1ddc318970d'),
    ('if not __debug__', 'f7046952d6409432c8eca77a30c47e11c427adb178a2d0569c11a707d07068dc'),
    ('STARTED', '23deef8c2ea3589b3d75859a3a092f9520ef8030c30ac55c26dc160d17ddc3fc'),
    ('LAUNCH_SCHEMA', '5bdf8a0040b63bdd14245cb6b97f971a1abe7c40541e325775316c59ae200ed8'),
    ('RESULT_SCHEMA', '0e2332860c1dfd010273c1738e0f38b7263d4e8f4e359770eb09aec747cae563'),
    ('LAUNCH_KEYS', '6c0a33941904463e81dc01fbb4a2d98db515e0186d5ce6c28bf17675e3e0a17a'),
    ('LIMITS', 'ff150c66cd0f687ee0ff676db2236c3b5d3b9c51ce48a63a698a0844155bd672'),
    ('LIMIT_TYPES', 'a23b45e7a232bcbacac2ec2bc98f2087fd1b96324f803f48fde8cbf71fd5d7d3'),
    ('SOURCE_ROLES', '3a598ef12e8ac3c09d5c0b52a1e9f809c9ad13e05d6091dc545e9b3b58224610'),
    ('HELPERS', '4bb30ea16af8b9820a96d60233e47c158f95d9f6cb17fb878289581f385ce90c'),
    ('UNIT_KEYS', 'b70ea89dc0fc19a343e41444e12f2cb2c248370a3ad740768725608da891fff8'),
    ('CPU_PROOF_SHA', 'a14783b39a2a6a81f7b52cfccc809a443b714d53e154122f7deb398903b4bae5'),
    ('CPU_EXECUTION_SHA', 'cb10109a44587dd07eecf06e6047ad6aad936622ff55e557830d2419edf55e20'),
    ('NATIVE_SOURCES_SHA', 'd69fac04890ac88296691a7e7bd6466f5d84ea74b020d71d4db4bc01cabc018a'),
    ('SCORER_AST', '234da46d7028e2bfb790575f7e264c18581624b8effba715806df88d7fa1d9ea'),
    ('NATIVE', 'e8bd6a3240541c7f17d5102b071e025d3bc4236097f6c80eab9c99e8ee43fa51'),
    ('WIDTH', 'e17f9dd1419a98915ab4b6c38ccd66dd5558c6801edfc80b2ff2b1d62b261502'),
    ('BATCHES', 'dde2b739d14fe62279ef93d1d136295a23a14853725786b94ee222231bf883e5'),
    ('FALSIFIER', 'a4ffd3d438511807c62a027983f5607b64dc68a89c2a2274b750c74a086d3b2e'),
    ('PROOF_TRUE', '336433e2eab0f43c9ddd141c1f7c99814c635ac1cba5e2d999c71d1fb8ec3823'),
    ('PROOF_FALSE', 'a575c4dd9108e8e10858465cbd4c5c4a75534678c1d2894fc49b4718eb2c5dfd'),
    ('VALUES_ARG', 'a673d0c10f6d3cc3fa8a2578222461d30fec4684c7211362e7ab332d446f986b'),
    ('PACK_STATEMENT', '008730bfee60e5b5d3eade14ccd7f10db2058243b73176836b533e776fb9b3e2'),
    ('CAPTURE_STATEMENT', '61dbadac2a798420112047fd246f2b6fd6a254b2eb36d5365c4d75a9f93471d7'),
    ('BATCH_STATEMENT', '7a1afafe9a15569a5e31f36564032a872341e50cb32067eb4025c9c60122a423'),
    ('AGGREGATES', '64ad4323eabed6021412b9c055f9478234f83ffb8810299c3eeebe2f1cb4036b'),
    ('ADAPTER_DUMPS', '5ec5e46ca40d3653bd9a1fd56e210ca151b96af39a8e6bbdea2817279a50df32'),
    ('require', '3b218d97634fd91fd6595535193a2eb525911ab34ffd45793deaba58332ca99e'),
    ('exact_policy', '55dec98b4352942ce1ec06d82d2cf5e66d174ce070955b5178c60985b11b9371'),
    ('file_fact', '9d3fef2c8f59cb9461b0c22a8c4e04f7751ec5f31ff45166c7ae6d755cdf04de'),
    ('file_sha', '71d7fa9c2e5fcbf399e861d360fcbb40601c436888cb5cf7b19f7832489fee66'),
    ('read_file', 'cdc17df38a1686c9e6fb13dbcb91d5d1aae5d4b96425a94e8819e99e7b0e4248'),
    ('strict_json', 'ab493a1275d06998447102c4fa8c70976cadb5cf8b69c655bf8da23f17a03179'),
    ('read_json', 'f2246740e1e015d584a4d6c6e83d4f066a4ccdfa02ec20c150fc0926ec3b48bc'),
    ('rehash_files', '34ad75dcb4299f343bd3406efaa35c761f0ad1bc4340c52bed042ff6f328ea60'),
    ('Budget', '2d80429b8fc6b6992fe1955d6c416014e3907c872fa6445fc2e058b5c497b2f7'),
    ('cleanup_error', '888f7a27f50c5f02b92e06f49bf2da0851acb4c2f652a6548bae46a085beb5d4'),
    ('read_authority', '1be8868c871639429b097535612485231ccfe4817bc4587ace93f5a9b45e47d8'),
    ('bootstrap_serving', 'c0a2bba3ff83a2e5aa25ed3ec4ec6a159813a566ece634c8168e4d7d6335e973'),
    ('load_helpers', '6f242224fbac490fe8dd6a7dcf8b04591e4f4aa070f713e5a03f29ad73e5f505'),
    ('own_source', '11722254b18a52a20a4a27d0a610a654fcfde59ea81007bbacc3df225cebc217'),
    ('check_own', '3e0af4f5bec8704963cba00f98238346b0b753899355d9f5bdcbd133bdab289f'),
    ('guard', 'bb8759de990c31cbc2be5055ae6b1db19c8a3557db7ae2f07d7c8230d6c28be1'),
    ('close_sources', 'ac1432ac36b6ba1a4421af98c7e48af92e08ef9c49edd8003212969af7502aef'),
    ('check_interpreter', 'f2a4ca5e75d00c66a7499599798eba8038eaa9b1da050dd79492ae2b68ca601f'),
    ('check_accepted_environment', '023f66b23d3d3a4a5748a856b2f01ca65b74a1af715108d1ea65535a31c0da3a'),
    ('admit_source_cpu', '1b7acb9e60cf5a926955bded756cac620ffa0320180ee71d05f3087ddf8d41b2'),
    ('check_packages', '58310cf31108811efb80b040f5b9454930e3f936e8d1f586fb94ac35bb272f55'),
    ('make_audit', '4934397f205c77dc2943cd60b8e1f911eb3e76471fab274ee0fa79ffb79f4dc1'),
    ('dump', '260a73c275f79438c69ee354c112d44018fab4ba46dcdc1028de9fe457060e3b'),
    ('scorer_function', 'd86045dfce72fe60bf0e7ec7a88b88aeee52029d1a5a1e818ec8ac9af26f2cb0'),
    ('batch_loop', 'cbc3bb54447d1551322ee36ce1f049d94137b824411a34aa1e31074e05a63082'),
    ('check_declarations', 'c3ce6eab7890964d1617758caf6d419f835cdf499c67f12ad8efad5d311d7d3a'),
    ('adapt', '6b49f56fbdb6033323f9f28b21751f93ee8cb33ceb1b2f76d343cce34e6efff9'),
    ('unadapt', '3e203d86a8bf287ee337cc54c78e1b807dca4e82dd4be98ab87218a2e7d4e1f0'),
    ('verify_adapter', 'f15ddd271a77d90364a4c625ca26080f1244041dc01120680445f6222bc4f03f'),
    ('compile_scorer', '6664939173540d0883b7bcb279a64cd97b07a7668941aca8a7b2932f77f6c761'),
    ('packed_input', '6355bb9843b9a5bf3d2211e3c60e4b062d25909bd2c6a588728adb576a29424c'),
    ('make_capture', '9ba0a06a8228bd7bf32ae7bac3fec1c8ebd8d53478541f7a9c01ceca2437a37b'),
    ('replay_exact', '51ea52994a7b481224033bca4ea9cd7df769e5aaf5de27d5f05158dd9986c579'),
    ('describe_scores', 'f28d0dd22c07169f690f5f2c0a0a27642f686a421abaa833b9a6c7176a8354b0'),
    ('score_digest', '8f0f05f84fcb2c5981379f4d3225b95aa2f8f16e5f8ee6a8f26f72d6d14ecd7c'),
    ('build_census', 'a0695b8b8c7f6f349dc11cd81a228215f489454c864ecd79c89f7f6abfa34cab'),
    ('admit', 'c03438792391b972e5a1f6f259e418335ff262c6594886e9845375886fb59924'),
    ('replay', 'fa9aba7b2773e1e144d74985477e13d60b8b56114f06cc52ba38624ba5c95ee7'),
    ('exit_checks', '834ba3ac693e6450381b9985b83868fe5e96b41a8b74ad1aa175e6ecb2e5198b'),
    ('owned_output', '8ccac9a37fbe1f134e1e466cba305bb2ec21cf51e583edf5790a8c1139ce443a'),
    ('publish_census', '1546d3e12f601ecfa78dc84d60dbc610c3555070f731c35de48b5e925fa11db3'),
    ('finalize', '436333beba3ad906331a01b379d15b998d7a388ad883e4b1e252aa1c8971618c'),
    ('run', 'd64a2b05060cb7ef58ec8ce2ad8e9a0688915968735e53c0ad8f6a38d97e5a37'),
    ('parser', 'c8ef05f6c0c42c94ef5a217a1f1064b37a426476829739d093eb885256a4fe1e'),
    ('main', '1ab55bc6b9ec113b28d5c46558c70296306ad85cc598e7c9d87177ffc9cba109'),
    ("if __name__ == '__main__'", 'ccadb48a0e9e8f52be639870abfcc7790c9aedf12a9a68fb216b33610df6910e'),
)
HISTORICAL_IMPORTS = ('import argparse', 'import ast', 'from collections import Counter', 'import copy', 'import gc', 'import hashlib', 'import importlib.util', 'import json', 'import math', 'import os', 'from pathlib import Path', 'import re', 'import resource', 'import stat', 'import struct', 'import sys', 'import tempfile', 'import time', 'from types import SimpleNamespace')


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def frozen_r1():
    """R@1 arrays with the accepted actual transitions (11/5, 12/6, shared 9/5, control-correct 1678/1677), core 0..43."""
    wrong = {'179061': set(range(55)) | {59}, '179069': set(range(53)) | {55, 56, 57, 58}}
    gains = {'179061': set(range(44, 55)), '179069': set(range(44, 53)) | {55, 56, 57}}
    losses = {'179061': set(range(100, 105)), '179069': set(range(100, 106))}
    r1 = {}
    for seed in SEEDS:
        r1[f'control-{seed}'] = [0 if i in wrong[seed] else 1 for i in range(Q)]
        r1[f'candidate-{seed}'] = [int(i in gains[seed] or (i not in wrong[seed] and i not in losses[seed]))
                                   for i in range(Q)]
    return r1


def score_row(index, key, wins):
    """One query's 90 scores whose stable-order top-1 is a positive exactly when wins; margins and indices vary."""
    salt = sum(key.encode()) % 11 + 1
    positive, impostor = POSITIVES[(index * 7 + salt) % len(POSITIVES)], IMPOSTORS[(index * 3 + salt) % len(IMPOSTORS)]
    other, margin = f32(0.5 + 0.001 * ((index * 5 + salt) % 17)), f32(0.01 * (1 + index % 13) + 0.001 * salt)
    row = [f32(0.2 - 0.0007 * j) for j in range(G)]
    row[impostor], row[positive] = other, f32(other + margin) if wins else f32(other - margin)
    return row


def flat_row(scores):
    """A low background with explicit {gallery index: score} overrides."""
    row = [f32(0.2 - 0.0007 * j) for j in range(G)]
    for index, value in scores.items():
        row[index] = value
    return row


def make_prior(m, state, retained, results):
    rows, panel = state['rows'], state['partition']['panels']['selection']
    labels, query, gallery, core = [r['product'] for r in rows], panel['query'], panel['gallery'], state['core']
    queries = []
    for index in core:
        endpoints = {}
        for key in KEYS:
            values = m.describe_scores(retained[key][index], labels, query[index], gallery,
                                       {k: results[key][k][index] for k in ('per_query_r1', 'per_query_ap')})
            values['best_positive'] = rows[values['best_positive_panel_ordinal']]
            values['top_impostor'] = rows[values['top_impostor_panel_ordinal']]
            endpoints[key] = values
        queries.append({'query_index': index, 'query': rows[query[index]], 'endpoints': endpoints})
    return {'schema': m.RESULT_SCHEMA, 'candidate_status': 'KILL unchanged', 'original_decision': 'KILL',
            'scientific_gate_changed': False, 'qualification_eligible': False, 'state_reuse_eligible': False,
            'full_uncached_exit_pass': True, 'exit_rehash_pass': True, 'cleanup_pass': True,
            'terminal_exit_and_both_locks_require_parent_receipt': True, 'resource_policy': dict(m.LIMITS),
            'core_query_indices': list(core), 'scored_queries': len(core), 'queries': queries,
            'replay': {'queries': Q, 'endpoints': 4, 'per_query_pairs': Q * 4, 'exact': True, 'tolerance': None,
                       'batch_sizes': list(m.BATCHES), 'width': m.WIDTH,
                       'falsifier': {**m.FALSIFIER, 'actual_ap': m.FALSIFIER['expected_ap'], 'exact': True}},
            'scorer': {'ast_sha256': m.SCORER_AST, 'device': 'cpu'},
            'core_score_rows_sha256': {key: m.score_digest(retained[key], core) for key in KEYS},
            'provenance': {'ordered_selection_mapping_sha256': m.selection_mapping_sha256(rows),
                           'fetch_record': copy.deepcopy(state['record'])}}


def make_world(m, r1=None):
    r1 = r1 or frozen_r1()
    query, gallery = list(range(Q)), list(range(Q, Q + G))
    rows = [{'product': 'P', 'panel_ordinal': i} for i in query] + \
           [{'product': 'N' if j % 10 == 0 else 'P', 'panel_ordinal': Q + j} for j in range(G)]
    aps = {key: [0.25 + (i + n) / 8192 for i in range(Q)] for n, key in enumerate(KEYS)}
    aps['candidate-179061'][747] = m.FALSIFIER['expected_ap']
    quality = {s: {a: {'per_query_r1': r1[f'{a}-{s}'], 'per_query_ap': aps[f'{a}-{s}']} for a in ('control', 'candidate')}
               for s in SEEDS}
    record = {'accepted_receipt': {'path': '/local/accepted.json', 'sha256': 'a' * 64},
              'partition': {'path': '/local/partition.json', 'original_path': '/orig/partition.json', 'sha256': 'b' * 64},
              'files': {f'{k}.packed.bin': {'path': f'/local/{k}', 'original_path': f'/orig/{k}', 'sha256': 'c' * 64,
                                            'bytes': 448370} for k in KEYS}}
    state = {'census': SimpleNamespace(ENDPOINTS=tuple((s, a) for s in SEEDS for a in ('control', 'candidate'))),
             'rows': rows, 'core': [i for i in range(Q) if all(r1[k][i] == 0 for k in KEYS)],
             'wires': {k: f'wire-{k}' for k in KEYS},
             'partition': {'panels': {'selection': {'query': query, 'gallery': gallery}}},
             'receipt': {'quality': quality, 'decision': 'KILL'}, 'record': record}
    retained = {k: {i: score_row(i, k, r1[k][i]) for i in range(Q)} for k in KEYS}
    results = {k: {'per_query_r1': list(r1[k]), 'per_query_ap': list(aps[k])} for k in KEYS}
    return SimpleNamespace(state=state, retained=retained, results=results, quality=quality,
                           prior=make_prior(m, state, retained, results))


_WORLDS = {}


def world(m, variant='frozen', r1=None):
    if variant not in _WORLDS:
        _WORLDS[variant] = make_world(m, r1)
    return _WORLDS[variant]


def mutated_r1(*changes):
    r1 = frozen_r1()
    for key, index, value in changes:
        r1[key][index] = value
    return r1


@contextlib.contextmanager
def replaced(container, key, value):
    old = container[key]
    container[key] = value
    try:
        yield
    finally:
        container[key] = old


def node_label(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        return ','.join(ast.unparse(t) for t in node.targets)
    if isinstance(node, ast.Expr):
        return '<docstring>'
    if isinstance(node, ast.If):
        return 'if ' + ast.unparse(node.test)
    return None


def node_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


class Rename(ast.NodeTransformer):
    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        return ast.Name(id=self.names.get(node.id, node.id), ctx=node.ctx)


class Unbranch(ast.NodeTransformer):
    """main(): exactly the mode dispatch call goes back to the one historical callee."""
    fired = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.IfExp):
            assert ast.unparse(node.func) == "run_all_query if args.mode == 'all-query' else run"
            self.fired += 1
            return ast.Call(func=ast.Name(id='run', ctx=ast.Load()), args=node.args, keywords=node.keywords)
        return node


class UnReplay(Rename):
    """replay_all_query() back to replay(): the named deltas undone, each exactly once."""
    fired = {'selection': 0, 'builder arguments': 0, 'message': 0}

    def __init__(self, names):
        super().__init__(names)
        self.fired = dict.fromkeys(self.fired, 0)

    def visit_Call(self, node):
        self.generic_visit(node)
        if ast.unparse(node) == 'set(range(len(query)))':
            self.fired['selection'] += 1
            return ast.parse("set(state['core'])", mode='eval').body
        if isinstance(node.func, ast.Name) and node.func.id == 'build_census':
            assert [ast.unparse(a) for a in node.args[-2:]] == ['ctx.budget', 'prior']
            self.fired['builder arguments'] += 1
            node.args = node.args[:-2]
        return node

    def visit_Constant(self, node):
        if node.value == 'query score rows were not all captured':
            self.fired['message'] += 1
            return ast.Constant('core score rows were not all captured')
        return node


def inverse(node):
    """The historical form of an edited or cloned node, rebuilt from the current one."""
    node = copy.deepcopy(node)
    if node.name == 'read_authority':
        assert [a.arg for a in node.args.args] == ['path', 'digest', 'schema', 'keys']
        assert [ast.unparse(d) for d in node.args.defaults] == ['LAUNCH_SCHEMA', 'LAUNCH_KEYS']
        node.args.args, node.args.defaults = node.args.args[:2], []
        return Rename({'schema': 'LAUNCH_SCHEMA', 'keys': 'LAUNCH_KEYS'}).visit(node)
    if node.name == 'parser':
        kept = [s for s in node.body if "'--mode'" not in ast.unparse(s)]
        assert len(kept) == len(node.body) - 1
        node.body = kept
        return node
    if node.name == 'main':
        unbranch = Unbranch()
        node = unbranch.visit(node)
        assert unbranch.fired == 1
        return node
    if node.name == 'run_all_query':
        node.name = 'run'
        return Rename({'read_all_query_authority': 'read_authority', 'replay_all_query': 'replay'}).visit(node)
    assert node.name == 'replay_all_query'
    node.name = 'replay'
    kept = [s for s in node.body if not ast.unparse(s).startswith(
        ('require(len(query) == QUERY_COUNT', 'prior = read_json(', 'strictly(bind_prior'))]
    assert len(kept) == len(node.body) - 3
    node.body = kept
    node.body[0] = ast.Expr(ast.Constant('Original scorer arithmetic on all four stored wires; '
                                         'complete exact replay before geometry.'))
    unreplay = UnReplay({'everything': 'core', 'build_all_query_census': 'build_census'})
    node = unreplay.visit(node)
    assert unreplay.fired == {'selection': 1, 'builder arguments': 1, 'message': 1}, unreplay.fired
    return node


class AllQueryBase(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.w = world(cls.m)

    def build(self, w=None, budget=None, prior=None):
        w = w or self.w
        return self.m.build_all_query_census(w.state, w.retained, w.results, {'fixture': True},
                                             budget or self.m.Budget(started=0., clock=lambda: 0.), prior or w.prior)

    def aq_authority(self, mutate=None, prior=PRIOR_RECEIPT):
        def make(authority):
            authority['schema'] = self.m.ALL_QUERY_LAUNCH_SCHEMA
            authority['prior_census'] = fact(prior)
            if mutate:
                mutate(authority)
        return self.authority(make)


class PreservationTests(Base):
    """The core mode is unchanged: pinned per-node AST digests, exact inverses of every edit and every clone."""
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tree = ast.parse(DRIVER.read_text())
        cls.nodes = {node_label(n): n for n in cls.tree.body if node_label(n) is not None}

    def test_every_historical_node_is_unchanged_in_value_and_order(self):
        labels = [node_label(n) for n in self.tree.body if node_label(n) is not None]
        self.assertEqual(len(labels), len(set(labels)))
        historical = [label for label, _ in HISTORICAL_NODES]
        self.assertEqual([label for label in labels if label in historical], historical)
        for label, digest in HISTORICAL_NODES:
            if label not in EDITED:
                with self.subTest(label):
                    self.assertEqual(node_digest(self.nodes[label]), digest)
        self.assertEqual(sorted(set(labels) - set(historical)), sorted(ADDED))
        imports = [ast.unparse(n) for n in self.tree.body if node_label(n) is None]
        self.assertEqual([i for i in imports if i in HISTORICAL_IMPORTS], list(HISTORICAL_IMPORTS))
        self.assertEqual(sorted(set(imports) - set(HISTORICAL_IMPORTS)), ['import statistics'])

    def test_edited_and_cloned_nodes_invert_exactly_to_their_historical_digests(self):
        historical = dict(HISTORICAL_NODES)
        for edited, clone in (('read_authority', None), ('parser', None), ('main', None),
                              ('run', 'run_all_query'), ('replay', 'replay_all_query')):
            with self.subTest(edited, clone=clone):
                restored = inverse(self.nodes[clone or edited])
                self.assertEqual(node_digest(restored), historical[edited])
        for name in EDITED:
            self.assertNotEqual(node_digest(self.nodes[name]), historical[name])

    def test_historical_hash_pins_and_limits_are_unchanged(self):
        m = self.m
        self.assertEqual((m.SCORER_AST, m.CPU_PROOF_SHA, m.NATIVE_SOURCES_SHA),
                         ('717489188a008ceba1b930d9e7dff32b90cd5347302835538eb171fe23f4b33e',
                          'e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf',
                          '8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b'))
        self.assertEqual(m.CPU_EXECUTION_SHA, '3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b')
        self.assertEqual(m.LAUNCH_SCHEMA, 'connected-exact-torch-census-launch-v1')
        self.assertEqual(m.RESULT_SCHEMA, 'sfora-connected-core-error-census-torch-v1')
        self.assertEqual(m.ALL_QUERY_KEYS, m.LAUNCH_KEYS | {'prior_census'})
        self.assertEqual(hashlib.sha256(PRIOR_RECEIPT.read_bytes()).hexdigest(), m.PRIOR_CENSUS_SHA)
        self.assertEqual(self.m.ALL_QUERY_LAUNCH_SCHEMA, 'connected-all-query-margin-launch-v1')
        self.assertEqual(self.m.ALL_QUERY_RESULT_SCHEMA, 'sfora-connected-all-query-margin-census-torch-v1')
        self.assertEqual((m.QUERY_COUNT, m.SEEDS, m.ENDPOINT_KEYS), (1734, SEEDS, KEYS))
        self.assertEqual(m.FROZEN_TRANSITIONS, {'179061': {'gains': 11, 'losses': 5, 'control_correct': 1678},
                                                '179069': {'gains': 12, 'losses': 6, 'control_correct': 1677},
                                                'shared': {'gains': 9, 'losses': 5}})

    def test_frozen_transitions_are_the_committed_accepted_receipt_counts(self):
        receipt = json.loads((EVIDENCE / 'connected-mlp-evaluation-full-selection-score-v1/receipt.json').read_text())
        gains, losses = {}, {}
        for seed in SEEDS:
            control, candidate = (receipt['quality'][seed][arm]['per_query_r1'] for arm in ('control', 'candidate'))
            gains[seed] = {i for i in range(Q) if (control[i], candidate[i]) == (0, 1)}
            losses[seed] = {i for i in range(Q) if (control[i], candidate[i]) == (1, 0)}
            self.assertEqual((len(gains[seed]), len(losses[seed]), sum(control)),
                             tuple(self.m.FROZEN_TRANSITIONS[seed][k] for k in ('gains', 'losses', 'control_correct')))
        self.assertEqual((len(gains['179061'] & gains['179069']), len(losses['179061'] & losses['179069'])),
                         (self.m.FROZEN_TRANSITIONS['shared']['gains'], self.m.FROZEN_TRANSITIONS['shared']['losses']))
        # the synthetic fixture world reproduces the same actual counts
        r1 = frozen_r1()
        for seed in SEEDS:
            control, candidate = r1[f'control-{seed}'], r1[f'candidate-{seed}']
            self.assertEqual((sum((c, k) == (0, 1) for c, k in zip(control, candidate)),
                              sum((c, k) == (1, 0) for c, k in zip(control, candidate)), sum(control)),
                             tuple(self.m.FROZEN_TRANSITIONS[seed][k] for k in ('gains', 'losses', 'control_correct')))


class AllQueryAuthorityTests(AllQueryBase):
    def test_exact_all_query_authority_pins_the_prior_core44_census_file(self):
        authority, path, digest = self.aq_authority()
        loaded, guards = self.m.read_all_query_authority(path, digest)
        self.assertEqual(loaded, authority)
        self.assertEqual(loaded.keys(), self.m.ALL_QUERY_KEYS)
        self.assertEqual(guards[str(PRIOR_RECEIPT)], self.m.PRIOR_CENSUS_SHA)
        self.assertEqual(guards[str(DRIVER)], hashlib.sha256(DRIVER.read_bytes()).hexdigest())

    def test_rejects_shape_policy_status_and_prior_file_drift(self):
        tmp = self.tmp()
        edited = tmp / 'receipt.json'
        edited.write_bytes(PRIOR_RECEIPT.read_bytes() + b'\n')
        link = tmp / 'link.json'
        link.symlink_to(PRIOR_RECEIPT)
        cases = {
            'extra key': lambda a: a.update(extra=1),
            'missing key': lambda a: a.pop('locks'),
            'missing prior': lambda a: a.pop('prior_census'),
            'core schema': lambda a: a.update(schema=self.m.LAUNCH_SCHEMA),
            'seconds': lambda a: a['resource_policy'].update(whole_process_seconds=700),
            'reserve': lambda a: a['resource_policy'].update(exit_reserve_seconds=0),
            'memory': lambda a: a['resource_policy'].update(host_bytes=16 * 1024**3),
            'swap bool alias': lambda a: a['resource_policy'].update(swap_bytes=False),
            'cuda': lambda a: a['resource_policy'].update(cuda_visible_devices='0'),
            'locks held': lambda a: a.update(both_locks_held=False),
            'status': lambda a: a.update(candidate_status='GO'),
            'qualification': lambda a: a.update(qualification_eligible=True),
            'reuse': lambda a: a.update(state_reuse_eligible=True),
            'native sources pin': lambda a: a.update(native_sources=a['inputs']),
            'stale source': lambda a: a['sources']['scorer'].update(sha256='0' * 64),
            'prior stale hash': lambda a: a['prior_census'].update(sha256='0' * 64),
            'prior extra key': lambda a: a['prior_census'].update(bytes=1),
            'prior upper-case hash': lambda a: a['prior_census'].update(sha256=a['prior_census']['sha256'].upper()),
            'prior is another actual file': lambda a: a.update(prior_census=fact(CPU / 'native256-source-cpu-so400-v4.json')),
            'prior is edited census bytes': lambda a: a.update(prior_census=fact(edited)),
            'prior symlink': lambda a: a.update(prior_census={'path': str(link), 'sha256': a['prior_census']['sha256']}),
            'prior relative': lambda a: a.update(prior_census={'path': 'receipt.json', 'sha256': a['prior_census']['sha256']})}
        for name, mutate in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.read_all_query_authority(*self.aq_authority(mutate)[1:])

    def test_each_reader_rejects_the_other_modes_authority(self):
        _, core_path, core_digest = self.authority()
        _, aq_path, aq_digest = self.aq_authority()
        with self.assertRaises(ValueError):
            self.m.read_all_query_authority(core_path, core_digest)
        with self.assertRaises(ValueError):
            self.m.read_authority(aq_path, aq_digest)
        self.m.read_authority(core_path, core_digest)

    def test_malformed_records_are_value_errors(self):
        for name, function in {'key': lambda: {}['x'], 'type': lambda: len(5), 'index': lambda: [][1],
                               'attribute': lambda: None.x}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.strictly(function)
        self.assertEqual(self.m.strictly(lambda a, b: a + b, 1, 2), 3)


class PriorCensusTests(AllQueryBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.real = json.loads(PRIOR_RECEIPT.read_text())

    def test_the_committed_exact_census_is_admitted(self):
        self.m.strictly(self.m.admit_prior_census, self.real)
        self.assertEqual((len(self.real['core_query_indices']), self.real['scored_queries'], self.real['gallery_rows']),
                         (44, 44, 1715))
        self.m.strictly(self.m.admit_prior_census, self.w.prior)

    def test_any_status_selection_replay_or_inventory_drift_is_rejected(self):
        cases = {
            'schema': lambda p: p.update(schema='sfora-connected-core-error-census-v1'),
            'status': lambda p: p.update(candidate_status='GO'),
            'decision': lambda p: p.update(original_decision='GO'),
            'gate': lambda p: p.update(scientific_gate_changed=True),
            'qualification': lambda p: p.update(qualification_eligible=True),
            'reuse': lambda p: p.update(state_reuse_eligible=True),
            'exit': lambda p: p.update(full_uncached_exit_pass=False),
            'rehash': lambda p: p.update(exit_rehash_pass=False),
            'cleanup': lambda p: p.update(cleanup_pass=False),
            'parent receipt': lambda p: p.update(terminal_exit_and_both_locks_require_parent_receipt=False),
            'policy': lambda p: p['resource_policy'].update(host_bytes=1),
            'core unsorted': lambda p: p['core_query_indices'].reverse(),
            'core duplicate': lambda p: p['core_query_indices'].__setitem__(1, p['core_query_indices'][0]),
            'core range': lambda p: p['core_query_indices'].__setitem__(-1, 1734),
            'core count': lambda p: p.update(scored_queries=43),
            'inexact': lambda p: p['replay'].update(exact=False),
            'tolerance': lambda p: p['replay'].update(tolerance=0.0),
            'pairs': lambda p: p['replay'].update(per_query_pairs=6935),
            'batches': lambda p: p['replay'].update(batch_sizes=[64] * 28),
            'width': lambda p: p['replay'].update(width=80),
            'falsifier ap': lambda p: p['replay']['falsifier'].update(actual_ap=0.2909099757671356),
            'falsifier query': lambda p: p['replay']['falsifier'].update(query_index=0),
            'scorer ast': lambda p: p['scorer'].update(ast_sha256='0' * 64),
            'scorer device': lambda p: p['scorer'].update(device='cuda'),
            'digest endpoint': lambda p: p['core_score_rows_sha256'].pop('control-179061'),
            'digest short': lambda p: p['core_score_rows_sha256'].update({'control-179061': 'ab'}),
            'queries order': lambda p: p['queries'].reverse(),
            'endpoint missing': lambda p: p['queries'][0]['endpoints'].pop('candidate-179069'),
            'replay missing': lambda p: p.pop('replay')}
        for name, mutate in cases.items():
            prior = copy.deepcopy(self.real)
            mutate(prior)
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.strictly(self.m.admit_prior_census, prior)

    def test_binding_to_this_runs_selection_mapping_and_inputs(self):
        state = self.w.state
        self.m.bind_prior(self.w.prior, state)
        cases = {
            'core': lambda s, p: s.update(core=s['core'][:-1]),
            'mapping': lambda s, p: s['rows'][3].update(product='other'),
            'accepted receipt': lambda s, p: s['record']['accepted_receipt'].update(sha256='d' * 64),
            'partition hash': lambda s, p: s['record']['partition'].update(sha256='d' * 64),
            'partition origin': lambda s, p: s['record']['partition'].update(original_path='/elsewhere'),
            'wire hash': lambda s, p: s['record']['files']['control-179061.packed.bin'].update(sha256='d' * 64),
            'wire size': lambda s, p: s['record']['files']['candidate-179069.packed.bin'].update(bytes=1),
            'wire origin': lambda s, p: s['record']['files']['candidate-179061.packed.bin'].update(original_path='/x'),
            'extra input': lambda s, p: s['record']['files'].update({'extra.bin': dict(
                s['record']['files']['control-179061.packed.bin'])}),
            'prior mapping': lambda s, p: p['provenance'].update(ordered_selection_mapping_sha256='0' * 64)}
        for name, mutate in cases.items():
            local_state, prior = copy.deepcopy({k: v for k, v in state.items() if k != 'census'}), copy.deepcopy(self.w.prior)
            mutate(local_state, prior)
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.strictly(self.m.bind_prior, prior, local_state)
        local_state = copy.deepcopy({k: v for k, v in state.items() if k != 'census'})
        local_state['record']['files']['control-179061.packed.bin']['path'] = '/different/local/staging/path'
        self.m.bind_prior(self.w.prior, local_state)  # staging paths are not identity

    @unittest.skipUnless(INPUTS.is_file(), 'root local census inputs absent')
    def test_the_committed_census_binds_to_the_real_authenticated_inputs(self):
        authority = self.authority(lambda a: a.update(inputs=fact(INPUTS)))[0]
        _, _, _, _, modules = self.helpers(authority)
        state = modules['census'].admit_inputs(authority['inputs']['path'], authority['inputs']['sha256'])
        self.m.bind_prior(self.real, state)
        self.assertEqual(self.real['core_query_indices'], state['core'])


class MarginRuleTests(Base):
    def test_margin_rule_truth_table_includes_both_tie_directions(self):
        rule = lambda margin, positive, impostor: self.m.margin_r1(
            {'positive_minus_impostor_margin': margin, 'best_positive_gallery_index': positive,
             'top_impostor_gallery_index': impostor})
        self.assertEqual([rule(0.25, 9, 2), rule(-0.25, 2, 9), rule(f32_neighbor(0.0, 1), 9, 2),
                          rule(-f32_neighbor(0.0, 1), 2, 9)], [1, 0, 1, 0])
        self.assertEqual([rule(0.0, 2, 9), rule(0.0, 9, 2), rule(-0.0, 2, 9), rule(-0.0, 9, 2)], [1, 0, 1, 0])

    def test_one_fp32_ulp_decides_margin_sign_exactly(self):
        base = f32(0.6)
        up, down = f32_neighbor(base, 1), f32_neighbor(base, -1)
        for positive, impostor, delta in ((9, 2, up), (9, 2, base), (9, 2, down), (2, 9, up), (2, 9, base), (2, 9, down)):
            values = {'positive_minus_impostor_margin': delta - base, 'best_positive_gallery_index': positive,
                      'top_impostor_gallery_index': impostor}
            order = sorted([(-delta, positive), (-base, impostor)])
            self.assertEqual(self.m.margin_r1(values), int(order[0][1] == positive), (positive, impostor, delta))

    def test_rank_beyond_the_ap_width_is_ranked_in_the_full_gallery_and_82_positives_are_rejected(self):
        gallery = list(range(1, 211))
        labels = ['Q'] + ['N'] * 120 + ['Q'] * 81 + ['N'] * 9
        scores = [1.0 - i * 0.001 for i in range(210)]
        values = self.m.describe_scores(scores, labels, 0, gallery, {'per_query_r1': 0, 'per_query_ap': 0.01})
        self.assertEqual((values['best_positive_rank'], values['top_impostor_rank'], values['positive_count']), (121, 1, 81))
        self.assertEqual((self.m.margin_r1(values), values['best_positive_gallery_index']), (0, 120))
        with self.assertRaises(ValueError):
            self.m.describe_scores(scores, ['Q'] + ['N'] * 119 + ['Q'] * 82 + ['N'] * 9, 0, gallery,
                                   {'per_query_r1': 0, 'per_query_ap': 0.01})


class AllQueryCaptureTests(AllQueryBase):
    def test_capture_retains_every_query_ordinal_across_the_original_batches(self):
        clock, retained = Clock(), {}
        capture = self.m.make_capture(set(range(Q)), retained, self.m.Budget(started=0., clock=clock))
        row = lambda index: [float(index), -1.5]
        for start in range(0, Q, 128):
            size = min(128, Q - start)
            capture(start, list(range(start, start + size)), Scores({i: row(start + i) for i in range(size)}))
        self.assertEqual(list(retained), list(range(Q)))
        self.assertEqual(retained, {i: row(i) for i in range(Q)})
        with self.assertRaises(ValueError):
            capture(1664, [1664], Scores({0: row(1664)}))
        clock.now = 780.
        with self.assertRaises(ValueError):
            capture(0, [0], Scores({0: row(0)}))


class AllQueryReplayTests(AllQueryBase):
    """replay_all_query() reaches geometry only after every pair is exact, and never before the prior is bound."""
    def context(self, w=None, prior=None):
        w, tmp = w or self.w, self.tmp()
        prior_path = tmp / 'prior.json'
        prior_path.write_text(json.dumps(prior or w.prior))
        return SimpleNamespace(
            torch=SimpleNamespace(device=lambda name: name, __version__='fixture'), state=w.state, flags={},
            budget=self.m.Budget(started=0., clock=Clock()), guards={}, guard=lambda: None,
            authority={'sources': {'scorer': fact(HERE / 'compare_inshop_sop_warmstart_100.py')},
                       'prior_census': fact(prior_path)}, payload=None)

    def scripted(self, ctx, tamper=None, short=None, twice=None):
        calls, order = [], iter(KEYS)
        expected = {f'{a}-{s}': ctx.state['receipt']['quality'][s][a] for s, a in ctx.state['census'].ENDPOINTS}
        self.starts = starts = []

        def compile_scorer(torch, raw, path, capture, batch):
            key = next(order)
            result = copy.deepcopy({k: expected[key][k] for k in ('per_query_r1', 'per_query_ap')})
            if tamper:
                tamper(key, result)

            def fn(packed, labels, query, gallery, *, device):
                calls.append((key, packed, device, len(query), len(gallery)))
                for start in range(0, len(query), 128):
                    batch(start)
                    starts.append((key, start))
                    size = min(128, len(query) - start)
                    scores = Scores({i: self.w.retained[key][start + i] for i in range(size)})
                    last = size - 1 if short == (key, start) else size
                    capture(start, query[start:start + last], scores)
                    if twice == (key, start):
                        capture(start, query[start:start + size], scores)
                return result
            return fn
        return compile_scorer, calls

    def run_replay(self, ctx, compile_scorer):
        with patch.object(self.m, 'compile_scorer', compile_scorer), patch.object(self.m, 'packed_input', lambda t, w: w), \
                patch.object(self.m, 'build_all_query_census', return_value={'built': True}) as build:
            self.m.replay_all_query(ctx)
        return build

    def test_exact_replay_precedes_geometry_and_every_query_row_is_retained(self):
        ctx = self.context()
        compile_scorer, calls = self.scripted(ctx)
        build = self.run_replay(ctx, compile_scorer)
        self.assertEqual(ctx.payload, {'built': True})
        self.assertEqual([c[0] for c in calls], list(KEYS))
        self.assertTrue(all(c[2] == 'cpu' and c[3:] == (Q, G) for c in calls))
        self.assertEqual([s for k, s in self.starts if k == KEYS[0]], list(range(0, Q, 128)))
        self.assertEqual(len(self.starts), 4 * 14)
        state, retained, results, facts, budget, prior = build.call_args.args
        self.assertTrue(all(rows.keys() == set(range(Q)) and all(len(r) == G for r in rows.values())
                            for rows in retained.values()))
        self.assertEqual((state is ctx.state, budget is ctx.budget, prior), (True, True, self.w.prior))
        self.assertEqual((facts['ast_sha256'], facts['device'], facts['width']), (self.m.SCORER_AST, 'cpu', 81))

    def test_any_mismatch_missing_or_duplicate_row_stops_before_geometry(self):
        ulp = lambda key, index: (lambda k, r: k == key and r['per_query_ap'].__setitem__(
            index, f32_neighbor(r['per_query_ap'][index], 1)))
        mutations = {
            'last pair one ULP': ulp('candidate-179069', Q - 1), 'first pair one ULP': ulp('control-179061', 0),
            'falsifier ULP': lambda k, r: k == 'candidate-179061' and r['per_query_ap'].__setitem__(747, 0.2909099757671356),
            'r1 flip last': lambda k, r: k == 'candidate-179069' and r['per_query_r1'].__setitem__(Q - 1, 0),
            'r1 flip first': lambda k, r: k == 'control-179061' and r['per_query_r1'].__setitem__(0, 1),
            'short r1': lambda k, r: k == 'control-179069' and r['per_query_r1'].pop()}
        for name, tamper in mutations.items():
            ctx = self.context()
            compile_scorer, _ = self.scripted(ctx, tamper)
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', lambda t, w: w), \
                    patch.object(self.m, 'build_all_query_census') as build, self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            build.assert_not_called()
            self.assertIsNone(ctx.payload)
        for name, kwargs in {'missing last row of the last batch': {'short': ('candidate-179069', 1664)},
                             'missing first batch row': {'short': ('control-179061', 0)},
                             'duplicate batch': {'twice': ('control-179069', 256)}}.items():
            ctx = self.context()
            compile_scorer, _ = self.scripted(ctx, **kwargs)
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', lambda t, w: w), \
                    patch.object(self.m, 'build_all_query_census') as build, self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            build.assert_not_called()

    def test_the_range_must_be_exactly_1734_queries(self):
        for name, query in {'short': list(range(Q - 1)), 'long': list(range(Q + 1))}.items():
            state = dict(self.w.state)
            state['partition'] = {'panels': {'selection': {'query': query, 'gallery': list(range(Q, Q + G))}}}
            ctx = self.context()
            ctx.state = state
            with self.subTest(name), patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            compile_scorer.assert_not_called()

    def test_prior_binding_precedes_any_scoring(self):
        for name, mutate in {'core': lambda s, p: s.update(core=s['core'][:-1]),
                             'mapping': lambda s, p: p['provenance'].update(ordered_selection_mapping_sha256='0' * 64),
                             'input identity': lambda s, p: s['record']['partition'].update(sha256='d' * 64),
                             'malformed prior': lambda s, p: p.pop('provenance')}.items():
            ctx = self.context()
            state, prior = dict(self.w.state), copy.deepcopy(self.w.prior)
            state['record'] = copy.deepcopy(state['record'])
            mutate(state, prior)
            ctx.state = state
            Path(ctx.authority['prior_census']['path']).write_text(json.dumps(prior))
            ctx.authority['prior_census'] = fact(ctx.authority['prior_census']['path'])
            with self.subTest(name), patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            compile_scorer.assert_not_called()

    def test_reserve_crossing_skips_the_next_matmul_at_prelude_capture_or_between_batches(self):
        crossing = {'prelude': None, 'before capture of batch 1': ('metric', 1), 'between batches 1 and 2': ('gap', 1)}
        expected = {'prelude': [], 'before capture of batch 1': [0, 128], 'between batches 1 and 2': [0, 128]}
        for name, trigger in crossing.items():
            ctx, matmuls = self.context(), []

            def compile_scorer(torch, raw, path, capture, batch):
                def fn(packed, labels, query, gallery, *, device):
                    if trigger is None:
                        ctx.budget.clock.now = 780.
                    for index, start in enumerate(range(0, len(query), 128)):
                        batch(start)
                        matmuls.append(start)
                        if trigger == ('metric', index):
                            ctx.budget.clock.now = 780.
                        capture(start, query[start:start + 128], Scores({i: [0.5] * len(gallery) for i in range(128)}))
                        if trigger == ('gap', index):
                            ctx.budget.clock.now = 780.
                    return {}
                return fn
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', lambda t, w: w), \
                    patch.object(self.m, 'build_all_query_census') as build, self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            self.assertEqual(matmuls, expected[name])
            build.assert_not_called()

    def test_final_check_follows_preparation_and_the_width_inventory_is_checked(self):
        for name in ('compile', 'packed input'):
            ctx, called = self.context(), []

            def compile_scorer(torch, raw, path, capture, batch):
                if name == 'compile':
                    ctx.budget.clock.now = 780.
                return lambda *args, **kwargs: called.append(1)

            def packed_input(torch, wire):
                if name == 'packed input':
                    ctx.budget.clock.now = 780.
                return wire
            with self.subTest(name), patch.object(self.m, 'compile_scorer', compile_scorer), \
                    patch.object(self.m, 'packed_input', packed_input), self.assertRaises(ValueError):
                self.m.replay_all_query(ctx)
            self.assertEqual(called, [])
        with replaced(self.w.state['rows'][Q + 0], 'product', 'P'):   # 82 positives: the prior is re-bound to this mapping
            prior = copy.deepcopy(self.w.prior)
            prior['provenance']['ordered_selection_mapping_sha256'] = self.m.selection_mapping_sha256(self.w.state['rows'])
            ctx = self.context(prior=prior)
            with patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError) as caught:
                self.m.replay_all_query(ctx)
        self.assertIn('panel-wide AP width', str(caught.exception))
        compile_scorer.assert_not_called()
        ctx = self.context()
        ctx.budget = self.m.Budget(started=0., clock=lambda: 780.)
        with patch.object(self.m, 'compile_scorer') as compile_scorer, self.assertRaises(ValueError):
            self.m.replay_all_query(ctx)
        compile_scorer.assert_not_called()


class AllQueryBuildTests(AllQueryBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.payload = cls.build(cls)

    def test_payload_is_complete_ordered_finite_and_descriptive_only(self):
        payload = self.payload
        text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
        self.assertLess(len(text), 16 * 1024**2)
        self.assertEqual(set(payload), {
            'schema', 'scope', 'original_decision', 'candidate_status', 'close_fs_status', 'scientific_gate_changed',
            'qualification_eligible', 'state_reuse_eligible', 'training_release', 'query_range', 'core_query_indices',
            'queries', 'transitions', 'cohorts', 'margin_ties', 'scored_queries', 'gallery_rows', 'endpoints',
            'scored_pairs', 'replay', 'core_score_rows_sha256', 'score_rows_sha256', 'prior_census_binding', 'scorer'})
        self.assertEqual((payload['schema'], payload['candidate_status'], payload['original_decision']),
                         (self.m.ALL_QUERY_RESULT_SCHEMA, 'KILL unchanged', 'KILL'))
        self.assertEqual([payload[k] for k in ('scientific_gate_changed', 'qualification_eligible', 'state_reuse_eligible',
                                               'training_release')], [False] * 4)
        self.assertEqual(payload['query_range'], {'start': 0, 'stop': Q, 'count': Q})
        self.assertEqual((payload['scored_queries'], payload['gallery_rows'], payload['endpoints'], payload['scored_pairs']),
                         (Q, G, 4, Q * G * 4))
        self.assertEqual(payload['replay']['per_query_pairs'], Q * 4)
        self.assertIsNone(payload['replay']['tolerance'])
        self.assertEqual(payload['replay']['falsifier']['actual_ap'], self.m.FALSIFIER['expected_ap'])
        self.assertTrue(payload['replay']['falsifier']['exact'])

        def keys(value):
            if isinstance(value, dict):
                for k, v in value.items():
                    yield k
                    yield from keys(v)
            elif isinstance(value, list):
                for v in value:
                    yield from keys(v)
        names = ' '.join(keys(payload)).lower()
        for forbidden in ('project', 'interval', 'bootstrap', 'threshold', 'extrapol', 'lower95', 'recall_at_1', 'map_at_r'):
            self.assertNotIn(forbidden, names)

    def test_every_query_record_has_four_endpoints_margins_and_two_exact_deltas(self):
        queries, w = self.payload['queries'], self.w
        self.assertEqual([q['query_index'] for q in queries], list(range(Q)))
        self.assertEqual([q['query_panel_ordinal'] for q in queries], list(range(Q)))
        core = set(w.state['core'])
        base = {'best_positive_rank', 'best_positive_gallery_index', 'best_positive_panel_ordinal', 'top_impostor_rank',
                'top_impostor_gallery_index', 'top_impostor_panel_ordinal', 'best_positive_score', 'top_impostor_score',
                'positive_minus_impostor_margin', 'positive_count', 'per_query_r1', 'per_query_ap', 'margin_rule_r1'}
        for q in queries:
            self.assertEqual(set(q), {'query_index', 'query_panel_ordinal', 'endpoints', 'margin_deltas', 'r1_transitions'})
            self.assertEqual(list(q['endpoints']), list(KEYS))
            index = q['query_index']
            for key, values in q['endpoints'].items():
                self.assertEqual(set(values), base | ({'best_positive', 'top_impostor'} if index in core else set()))
                self.assertEqual((values['per_query_r1'], values['margin_rule_r1']), (w.results[key]['per_query_r1'][index],) * 2)
                self.assertEqual(values['positive_minus_impostor_margin'], values['best_positive_score'] - values['top_impostor_score'])
                self.assertEqual(values['per_query_ap'], w.results[key]['per_query_ap'][index])
                self.assertEqual(values['positive_count'], 81)
            for seed in SEEDS:
                self.assertEqual(q['margin_deltas'][seed], q['endpoints'][f'candidate-{seed}']['positive_minus_impostor_margin'] -
                                 q['endpoints'][f'control-{seed}']['positive_minus_impostor_margin'])

    def test_actual_transitions_and_cohorts_are_recomputed_independently(self):
        payload, w = self.payload, self.w
        r1 = {k: w.results[k]['per_query_r1'] for k in KEYS}
        delta = lambda i, s: payload['queries'][i]['margin_deltas'][s]
        for seed in SEEDS:
            control, candidate = r1[f'control-{seed}'], r1[f'candidate-{seed}']
            gains = [i for i in range(Q) if (control[i], candidate[i]) == (0, 1)]
            losses = [i for i in range(Q) if (control[i], candidate[i]) == (1, 0)]
            t = payload['transitions'][seed]
            self.assertEqual((t['gain_queries'], t['loss_queries'], t['gains'], t['losses']), (gains, losses, len(gains), len(losses)))
            self.assertEqual((t['control_correct'], t['candidate_correct'], t['both_wrong']),
                             (sum(control), sum(candidate), sum((c, k) == (0, 0) for c, k in zip(control, candidate))))
            cohorts = {'control_correct': [i for i in range(Q) if control[i]], 'control_wrong': [i for i in range(Q) if not control[i]],
                       'core44': w.state['core']}
            for name, indices in cohorts.items():
                got = payload['cohorts']['core44'][seed] if name == 'core44' else payload['cohorts'][seed][name]
                deltas = [delta(i, seed) for i in indices]
                margins = lambda key: [payload['queries'][i]['endpoints'][key]['positive_minus_impostor_margin'] for i in indices]
                self.assertEqual(got['queries'], len(indices))
                self.assertEqual((got['gains'], got['losses']), (sum(i in gains for i in indices), sum(i in losses for i in indices)))
                self.assertEqual(got['margin_delta'], {
                    'improved': sum(d > 0 for d in deltas), 'worsened': sum(d < 0 for d in deltas), 'zero': sum(d == 0 for d in deltas),
                    'min': min(deltas), 'median': statistics.median(deltas), 'max': max(deltas),
                    'mean': math.fsum(deltas) / len(deltas)})
                for field, key in (('control_margin', f'control-{seed}'), ('candidate_margin', f'candidate-{seed}')):
                    self.assertEqual(got[field], {'min': min(margins(key)), 'median': statistics.median(margins(key)),
                                                  'max': max(margins(key))})
            self.assertEqual((len(cohorts['control_correct']) + len(cohorts['control_wrong'])), Q)
        self.assertEqual({s: {k: payload['transitions'][s][k] for k in ('gains', 'losses')} for s in SEEDS},
                         {'179061': {'gains': 11, 'losses': 5}, '179069': {'gains': 12, 'losses': 6}})
        self.assertEqual({k: payload['transitions']['shared'][k] for k in ('gains', 'losses')}, {'gains': 9, 'losses': 5})
        self.assertEqual(payload['transitions']['shared']['gain_queries'], list(range(44, 53)))
        self.assertEqual(payload['cohorts']['core44']['179061']['gains'] + payload['cohorts']['core44']['179061']['losses'], 0)
        self.assertEqual(payload['core_query_indices'], list(range(44)))

    def test_core44_geometry_digests_and_all_row_digests(self):
        payload, w = self.payload, self.w
        self.assertEqual(payload['core_score_rows_sha256'], w.prior['core_score_rows_sha256'])
        for key in KEYS:
            digest = hashlib.sha256()
            for i in range(Q):
                digest.update(struct.pack('<q', i) + struct.pack(f'<{G}f', *w.retained[key][i]))
            self.assertEqual(payload['score_rows_sha256'][key], digest.hexdigest())
        for q, prior in zip(payload['queries'][:44], w.prior['queries']):
            self.assertEqual(q['query_index'], prior['query_index'])
            for key, values in q['endpoints'].items():
                self.assertEqual({f: v for f, v in values.items() if f != 'margin_rule_r1'}, prior['endpoints'][key])
        self.assertEqual(self.payload['prior_census_binding'], {
            'schema': self.m.RESULT_SCHEMA, 'core_queries': 44, 'core_geometry_equal': True,
            'core_score_rows_sha256_equal': True})

    def test_tie_winner_r1_in_both_directions_and_inconsistent_replay_is_rejected(self):
        w = self.w
        positive_wins = flat_row({21: f32(0.7), 30: f32(0.7)})   # tie; positive has the lower gallery index
        impostor_wins = flat_row({31: f32(0.7), 20: f32(0.7)})   # tie; impostor has the lower gallery index
        with replaced(w.retained['control-179061'], 200, positive_wins), \
                replaced(w.retained['control-179061'], 300, positive_wins), \
                replaced(w.retained['control-179061'], 59, impostor_wins):
            payload = self.build()
        self.assertEqual(payload['margin_ties']['control-179061'], {'zero_margin': 3, 'positive_wins': 2, 'impostor_wins': 1})
        wins, loses = payload['queries'][200]['endpoints']['control-179061'], payload['queries'][59]['endpoints']['control-179061']
        self.assertEqual((wins['positive_minus_impostor_margin'], wins['per_query_r1'], wins['margin_rule_r1']), (0.0, 1, 1))
        self.assertEqual((loses['positive_minus_impostor_margin'], loses['per_query_r1'], loses['margin_rule_r1']), (0.0, 0, 0))
        self.assertEqual((wins['best_positive_gallery_index'], wins['top_impostor_gallery_index']), (21, 30))
        self.assertEqual((loses['best_positive_gallery_index'], loses['top_impostor_gallery_index']), (31, 20))
        self.assertEqual(sum(payload['margin_ties'][k]['zero_margin'] for k in KEYS[1:]), 0)
        for index, row, wrong in ((200, positive_wins, 0), (59, impostor_wins, 1)):
            results = copy.deepcopy(w.results)
            results['control-179061']['per_query_r1'][index] = wrong
            with self.subTest(index), replaced(w.retained['control-179061'], index, row), self.assertRaises(ValueError):
                self.m.build_all_query_census(w.state, w.retained, results, {}, self.m.Budget(started=0., clock=lambda: 0.), w.prior)

    def test_one_ulp_score_mutations_flip_the_margin_rule_and_are_rejected(self):
        w, key, index = self.w, 'control-179069', 400   # replayed R1 is 1
        base = f32(0.6)
        up, down = f32_neighbor(base, 1), f32_neighbor(base, -1)
        # impostor index 20 < positive index 31: only a strictly positive margin wins
        for positive, accepted in ((up, True), (base, False), (down, False)):
            with self.subTest('impostor first', positive=positive), replaced(w.retained[key], index, flat_row({31: positive, 20: base})):
                if accepted:
                    self.assertEqual(self.build()['queries'][index]['endpoints'][key]['margin_rule_r1'], 1)
                else:
                    with self.assertRaises(ValueError):
                        self.build()
        # positive index 21 < impostor index 30: a tie is still won, one ULP below is not
        for positive, accepted in ((up, True), (base, True), (down, False)):
            with self.subTest('positive first', positive=positive), replaced(w.retained[key], index, flat_row({21: positive, 30: base})):
                if accepted:
                    self.assertEqual(self.build()['queries'][index]['endpoints'][key]['margin_rule_r1'], 1)
                else:
                    with self.assertRaises(ValueError):
                        self.build()

    def test_the_margin_rule_independently_checks_the_replayed_r1(self):
        real, w = self.m.describe_scores, self.w

        def drifting(scores, labels, query, gallery, replayed):
            values = real(scores, labels, query, gallery, replayed)
            if query == 300:   # a describe_scores drift: the margin no longer agrees with the exactly replayed R1
                values['positive_minus_impostor_margin'] = -values['positive_minus_impostor_margin']
            return values
        with patch.object(self.m, 'describe_scores', drifting), self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn('margin rule differs from replayed R1', str(caught.exception))
        self.assertEqual(w.state['partition']['panels']['selection']['query'][300], 300)

    def test_missing_extra_or_short_rows_and_82_positives_are_rejected(self):
        w = self.w
        row = w.retained['candidate-179061']
        short_r1, short_ap = copy.deepcopy(w.results), copy.deepcopy(w.results)
        short_r1['control-179061']['per_query_r1'].pop()
        short_ap['control-179061']['per_query_ap'].append(0.5)
        extra = copy.deepcopy(w.results)
        extra['control-179061']['recall_at_1'] = 1.0
        cases = {'missing last': lambda: row.pop(Q - 1), 'extra ordinal': lambda: row.update({Q: row[0]})}
        for name, mutate in cases.items():
            saved = dict(row)
            mutate()
            with self.subTest(name), self.assertRaises(ValueError):
                self.build()
            row.clear()
            row.update(saved)
        for name, bad in {'short r1': short_r1, 'long ap': short_ap, 'extra array': extra}.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.m.build_all_query_census(w.state, w.retained, bad, {}, self.m.Budget(started=0., clock=lambda: 0.), w.prior)
        missing_endpoint = {k: v for k, v in w.retained.items() if k != 'control-179069'}
        with self.assertRaises(ValueError):
            self.m.build_all_query_census(w.state, missing_endpoint, w.results, {}, self.m.Budget(started=0., clock=lambda: 0.), w.prior)
        with replaced(w.state['rows'][Q + 0], 'product', 'P'), self.assertRaises(ValueError):
            self.build()

    def test_core_mismatch_is_rejected_by_indices_geometry_and_digests(self):
        w = self.w
        prior = copy.deepcopy(w.prior)
        e = prior['queries'][0]['endpoints']['control-179061']
        e['best_positive_score'] = f32_neighbor(e['best_positive_score'], 1)
        with self.assertRaises(ValueError) as caught:
            self.build(prior=prior)
        self.assertIn('core44 geometry differs', str(caught.exception))
        prior = copy.deepcopy(w.prior)
        prior['core_score_rows_sha256']['candidate-179069'] = '0' * 64
        with self.assertRaises(ValueError) as caught:
            self.build(prior=prior)
        self.assertIn('core44 score digests differ', str(caught.exception))
        prior = copy.deepcopy(w.prior)
        prior['core_query_indices'] = prior['core_query_indices'][:-1]
        with self.assertRaises(ValueError):
            self.build(prior=prior)
        row = list(w.retained['candidate-179061'][5])
        row[G - 1] = f32_neighbor(row[G - 1], 1)    # a non-geometry element of a core row changes only the digest
        with replaced(w.retained['candidate-179061'], 5, row), self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn('core44 score digests differ', str(caught.exception))
        row = list(w.retained['candidate-179061'][5])
        top = max(range(G), key=lambda j: row[j])
        row[top] = f32_neighbor(row[top], 1)        # a geometry element changes the score and the geometry
        with replaced(w.retained['candidate-179061'], 5, row), self.assertRaises(ValueError) as caught:
            self.build()
        self.assertIn('core44 geometry differs', str(caught.exception))
        results = copy.deepcopy(w.results)
        results['control-179069']['per_query_r1'][5] = 1   # core44 is no longer the all-endpoint-wrong set
        with self.assertRaises(ValueError) as caught:
            self.m.build_all_query_census(w.state, w.retained, results, {}, self.m.Budget(started=0., clock=lambda: 0.), w.prior)
        self.assertIn('all-endpoint-wrong', str(caught.exception))

    def test_actual_transition_counts_must_reproduce_the_accepted_receipt(self):
        dropped_gain = world(self.m, 'dropped gain', mutated_r1(('candidate-179061', 44, 0)))
        shared_eight = world(self.m, 'shared eight', mutated_r1(('candidate-179069', 52, 0), ('candidate-179069', 58, 1)))
        lost_control = world(self.m, 'control correct', mutated_r1(('control-179069', 59, 0), ('candidate-179069', 59, 0)))
        for name, w in {'dropped gain': dropped_gain, 'shared gains 8': shared_eight, 'control-correct 1676': lost_control}.items():
            with self.subTest(name), self.assertRaises(ValueError) as caught:
                self.build(w)
            self.assertIn('actual transitions differ', str(caught.exception))

    def test_budget_is_checked_before_every_query_and_through_the_expanded_geometry(self):
        counted = []

        class Counting(self.m.Budget):
            def check(self, reserve=True):
                counted.append(reserve)
                super().check(reserve)
        self.build(budget=Counting(started=0., clock=lambda: 0.))
        self.assertGreaterEqual(len(counted), Q + 2)
        self.assertTrue(all(counted))
        clock, calls = Clock(), []
        real = self.m.describe_scores

        def counting(*args):
            calls.append(1)
            if len(calls) == 4 * 700 + 1:
                clock.now = 780.
            return real(*args)
        with patch.object(self.m, 'describe_scores', counting), self.assertRaises(ValueError):
            self.build(budget=self.m.Budget(started=0., clock=clock))
        self.assertEqual(len(calls), 4 * 701)
        clock, digests = Clock(), []
        real_digest = self.m.score_digest

        def late(*args):
            digests.append(1)
            clock.now = 780.
            return real_digest(*args)
        with patch.object(self.m, 'score_digest', late), self.assertRaises(ValueError):
            self.build(budget=self.m.Budget(started=0., clock=clock))
        self.assertEqual(len(digests), 8)

    def test_finalize_adds_only_exit_authority_and_provenance_to_an_all_query_payload(self):
        w, tmp = self.w, self.tmp()
        payload = dict(self.payload)
        authority = self.aq_authority()[0]
        ctx = SimpleNamespace(
            state={'guards': {}, 'record': {}, 'rows': w.state['rows']}, payload=payload, authority=authority,
            guards={'g': 'h'}, before={'path': 'x'}, after={'path': 'x'}, origins={'initial': {}, 'final': {}},
            proof={'invocation': {'python_sha256': 'p'}}, budget=self.m.Budget(started=0., clock=lambda: 3.), peak=7)
        args = SimpleNamespace(authority=str(tmp / 'a.json'), authority_sha256='0' * 64)
        with patch.dict(os.environ, {'INVOCATION_ID': INVOCATION, 'CUDA_VISIBLE_DEVICES': ''}):
            self.m.finalize(args, ctx)
        added = set(payload) - set(self.payload)
        self.assertTrue(all(payload[k] is self.payload[k] for k in self.payload))
        self.assertEqual(added, {'full_uncached_exit_pass', 'exit_rehash_pass', 'cleanup_pass', 'authority', 'launch',
                                 'resource_policy', 'cgroup_before', 'cgroup_after', 'origins', 'invocation', 'resources',
                                 'provenance', 'terminal_exit_and_both_locks_require_parent_receipt'})
        self.assertEqual(payload['launch']['prior_census'], authority['prior_census'])
        self.assertEqual(payload['provenance']['ordered_selection_mapping_sha256'],
                         self.m.selection_mapping_sha256(w.state['rows']))
        json.dumps(payload, allow_nan=False)


class AllQueryRunTests(AllQueryBase):
    def test_dispatch_flag_and_authority_schema_select_the_mode(self):
        parser = self.m.parser()
        base = ['--authority', '/a', '--authority-sha256', '0' * 64, '--output', '/o']
        self.assertEqual(parser.parse_args(base).mode, 'core')
        self.assertEqual(parser.parse_args(base + ['--mode', 'all-query']).mode, 'all-query')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(base + ['--mode', 'both'])
        tmp = self.tmp()
        output = tmp / 'out.json'
        output.write_text('x')
        payload = {'replay': {'per_query_pairs': 6936}}
        for mode in ('core', 'all-query'):
            with patch.object(self.m, 'run', return_value=payload) as core, \
                    patch.object(self.m, 'run_all_query', return_value=payload) as aq, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.m.main(['--authority', '/a', '--authority-sha256', '0' * 64, '--output', str(output),
                                              '--mode', mode]), 0)
            self.assertEqual((core.call_count, aq.call_count), (1, 0) if mode == 'core' else (0, 1))
        with patch.object(self.m, 'run_all_query', side_effect=ValueError('boom')), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(self.m.main(['--authority', '/a', '--authority-sha256', '0' * 64, '--output', str(output),
                                          '--mode', 'all-query']), 1)
        self.assertIn('rejected: ValueError: boom', err.getvalue())

    def test_run_all_query_publishes_only_through_the_owned_output_guard_in_the_core_order(self):
        tree = ast.parse(DRIVER.read_text())
        run = next(n for n in tree.body if getattr(n, 'name', None) == 'run_all_query')
        text = ast.unparse(run)
        self.assertIn("publish_census(ctx.state['census'], args.output, ctx.payload, ctx.state['guards'], ctx.budget)", text)
        self.assertNotIn('.publish(', text)
        self.assertNotIn('check(reserve=False)', text)
        flat = [leaf for stmt in run.body for leaf in (stmt.body if isinstance(stmt, ast.Try) else [stmt])]
        lines = [ast.unparse(s) for s in flat]
        at = lambda needle: next(i for i, line in enumerate(lines) if needle in line)
        self.assertLess(at('read_all_query_authority('), at('admit(ctx)'))
        self.assertLess(at('admit(ctx)'), at('replay_all_query(ctx)'))
        self.assertLess(at('replay_all_query(ctx)'), at('cleanup_error(error, exit_checks(ctx))'))
        self.assertLess(at('cleanup_error(error, exit_checks(ctx))'), at('finalize(args, ctx)'))
        self.assertLess(at('finalize(args, ctx)'), at('publish_census('))

    def test_own_source_check_covers_every_new_global_and_function(self):
        def code(m):
            m.margin_r1.__code__ = (lambda values: 1).__code__
        cases = {
            'query count': lambda m: setattr(m, 'QUERY_COUNT', 1735),
            'prior pin': lambda m: setattr(m, 'PRIOR_CENSUS_SHA', '0' * 64),
            'frozen literal': lambda m: m.FROZEN_TRANSITIONS['179061'].update(gains=0),
            'keys literal': lambda m: m.ALL_QUERY_KEYS.add('extra'),
            'transition literal': lambda m: m.TRANSITION.update({(0, 1): 'loss'}),
            'new function code': code,
            'edited default': lambda m: setattr(m.read_authority, '__defaults__', ('x', set())),
            'new function default': lambda m: setattr(m.cohort_summary, '__defaults__', (1,))}
        for name, mutate in cases.items():
            with self.subTest(name):
                m, source = OwnSourceTests.own(self)
                mutate(m)
                with self.assertRaises(ValueError):
                    m.check_own(source)

    def test_cli_rejects_a_mode_authority_mismatch_before_reading_inputs(self):
        output = self.tmp() / 'census.json'
        env = {'CUDA_VISIBLE_DEVICES': '', 'INVOCATION_ID': INVOCATION}
        _, core_path, core_digest = self.authority()
        _, aq_path, aq_digest = self.aq_authority()
        run = lambda path, digest, mode: ProcessTests.run_driver(self, '--authority', path, '--authority-sha256', digest,
                                                                '--output', str(output), '--mode', mode, env=env)
        for name, done in {'all-query mode, core authority': run(core_path, core_digest, 'all-query'),
                           'core mode, all-query authority': run(aq_path, aq_digest, 'core')}.items():
            with self.subTest(name):
                self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
                self.assertIn('exact Torch core census rejected', done.stderr)
                self.assertIn('exact diagnostic-only KILL authority', done.stderr)
                self.assertFalse(os.path.lexists(output))
        done = run(aq_path, aq_digest, 'all-query')   # a valid all-query authority is admitted, then stops at the fake locks
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn('exact Torch core census rejected', done.stderr)
        self.assertNotIn('exact diagnostic-only KILL authority', done.stderr)
        self.assertFalse(os.path.lexists(output))
        done = ProcessTests.run_driver(self, '--authority', aq_path, '--authority-sha256', aq_digest, '--output', str(output),
                                       '--mode', 'bogus', env=env)
        self.assertEqual(done.returncode, 2)
        self.assertIn('--mode', ProcessTests.run_driver(self, '--help').stdout)


if __name__ == '__main__':
    unittest.main()
