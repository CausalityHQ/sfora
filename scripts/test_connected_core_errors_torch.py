#!/usr/bin/env python3
"""Bounded stdlib falsifiers only; Torch, native execution and DGX UNRUN.

These prove structure (pins, exact scorer AST + adapter inverse, helper APIs,
gating, geometry, exit behaviour). They cannot prove Torch numerical equality:
only the root's single native job can falsify that.
"""
import ast
import builtins
import copy
import dis
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
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

    def test_adapter_is_the_exact_inverse_and_only_touches_four_things(self):
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

    def test_adapted_scorer_resolves_only_torch_and_the_capture_hook(self):
        adapted = self.m.adapt(self.m.scorer_function(self.scorer))
        module = compile(ast.Module(body=[adapted], type_ignores=[]), 'scorer', 'exec', dont_inherit=True)
        names, pending = set(), [module]
        while pending:
            code = pending.pop()
            names |= {i.argval for i in dis.get_instructions(code) if i.opname == 'LOAD_GLOBAL'}
            pending += [c for c in code.co_consts if hasattr(c, 'co_code')]
        self.assertEqual({n for n in names if not hasattr(builtins, n)}, {'torch', 'census_capture'})

    def test_compile_scorer_defines_the_adapted_function_signature(self):
        torch = SimpleNamespace(inference_mode=lambda: (lambda fn: fn), device=type('device', (), {}))
        fn = self.m.compile_scorer(torch, self.scorer, str(HERE / 'compare_inshop_sop_warmstart_100.py'), print)
        parameters = inspect.signature(fn).parameters
        self.assertEqual(list(parameters), ['packed', 'labels', 'query', 'gallery', 'device'])
        self.assertEqual(parameters['device'].kind, inspect.Parameter.KEYWORD_ONLY)
        with self.assertRaises(ValueError):
            self.m.compile_scorer(torch, self.mutate(self.scorer.decode(), 'stable=True', 'stable=False').encode(), 'x', print)

    def test_in_place_adapter_declaration_mutation_is_rejected_before_execution(self):
        declared = {'VALUES_ARG': 'values: np.ndarray', 'CAPTURE_STATEMENT': 'census_capture(start, rows, scores)',
                    'PACK_STATEMENT': 'packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))',
                    'AGGREGATES': "{'recall_at_1': float(np.mean(hits)), 'map_at_r': float(np.mean(aps))}"}
        for name, text in declared.items():
            self.assertEqual(ast.unparse(getattr(self.m, name)), text)
        forged = ast.parse('census_capture(start, rows, forged_scores)').body[0]
        cases = {'capture argument forged': lambda m: setattr(m.CAPTURE_STATEMENT.value.args[2], 'id', 'forged_scores'),
                 'capture rebound': lambda m: setattr(m, 'CAPTURE_STATEMENT', forged),
                 'values argument': lambda m: setattr(m.VALUES_ARG, 'arg', 'forged'),
                 'pack target': lambda m: setattr(m.PACK_STATEMENT.targets[0], 'id', 'forged'),
                 'aggregate key': lambda m: setattr(m.AGGREGATES.keys[0], 'value', 'forged')}
        control = load_driver()
        with patch.object(builtins, 'exec') as run, self.assertRaises(KeyError):
            control.compile_scorer(SimpleNamespace(), self.scorer, 'x', print)
        run.assert_called_once()
        for name, mutate in cases.items():
            with self.subTest(name):
                m = load_driver()
                mutate(m)
                with patch.object(builtins, 'exec') as run, self.assertRaises(ValueError):
                    m.compile_scorer(SimpleNamespace(), self.scorer, 'x', print)
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
        expected = {f'{a}-{s}': ctx.state['receipt']['quality'][s][a] for s, a in ctx.state['census'].ENDPOINTS}
        order = iter(expected)
        def compile_scorer(torch, raw, path, capture):
            key = next(order)
            result = copy.deepcopy({k: expected[key][k] for k in ('per_query_r1', 'per_query_ap')})
            if tamper:
                tamper(key, result)
            def fn(packed, labels, query, gallery, *, device):
                calls.append((key, packed, device, len(query), len(gallery)))
                for start in range(0, len(query), 128):
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
            locks=SimpleNamespace(check=lambda: locks.append(1)), owned=owned, modules=modules, own=None)
        self.m.cleanup_error(None, self.m.exit_checks(ctx))
        self.assertEqual((audits, locks, owned, ctx.origins), ([1], [1], [], {'final': {'files': 1}}))
        path.write_text('changed')
        _, guards2, _, owned2, _ = self.helpers()
        guards2[str(path)] = fact(path)['sha256']
        path.write_text('again')
        ctx = SimpleNamespace(audit=None, origins={}, guards=guards2, state=None, torch=None, locks=None, owned=owned2,
                              own=None)
        with self.assertRaises(ValueError):
            self.m.cleanup_error(None, self.m.exit_checks(ctx))
        self.assertEqual(owned2, [], 'sources are disposed even when an exit proof fails')

    def test_exit_state_requires_hidden_cuda_flags_cgroup_and_cap(self):
        modules = {'source_driver': SimpleNamespace(numerical_flags=lambda: {'threads': 8},
                                                    cgroup_memory=lambda: {'path': '/cg/unit.service'}),
                   'initializer': SimpleNamespace(admit_cgroup=lambda value, unit: None)}
        def ctx(**override):
            base = dict(audit=None, origins={}, guards={}, state=None, locks=None, owned=[], modules=modules, own=None,
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
                              origins={}, guards={}, state=None, torch=None)
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


if __name__ == '__main__':
    unittest.main()
