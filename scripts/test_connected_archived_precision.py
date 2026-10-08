#!/usr/bin/env python3
"""Pure-stdlib toy/adversarial checks; never score the real archived panel."""

from fractions import Fraction
import copy
import hashlib
import importlib.util
import json
import math
import marshal
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().with_name("diagnose_connected_archived_precision.py")


def load_script():
    spec = importlib.util.spec_from_file_location("archived_precision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def vector(values):
    return [fp32(v) for v in values] + [0.0] * (128 - len(values))


def descriptor(path):
    data = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def npy_bytes(header=None, value=0.0):
    if header is None:
        header = "{'descr': '<f4', 'fortran_order': False, 'shape': (3449, 128), }"
    raw = header.encode("ascii")
    return (b"\x93NUMPY\x01\x00" + struct.pack("<H", 118) + raw.ljust(117, b" ") + b"\n"
            + struct.pack("<f", value) * (3449 * 128))


class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_script()

    def test_outward_bound_covers_exact_rational_and_fp32_reduction(self):
        tiny = 2.0 ** -149
        cases = [([1, 1, 1], [1, 2 ** -24, -1]),
                 ([2 ** 30, 1, 2 ** 30], [1, 1, -1]),
                 ([tiny, tiny, tiny], [0.5, 0.5, -0.5]),
                 ([tiny] * 128, [0.5] * 128),
                 ([1.0] * 128, [2 ** -24] * 128),
                 ([1, -1], [1, 1]), ([0], [0])]
        u32 = Fraction(1, 2 ** 24)
        u64 = Fraction(1, 2 ** 53)
        for left, right in cases:
            q, g = vector(left), vector(right)
            products = [Fraction(a) * Fraction(b) for a, b in zip(q, g)]
            exact, magnitude = sum(products), sum(map(abs, products))
            result = self.d.dot_interval(q, g)
            with self.subTest(left=left[:3]):
                self.assertLessEqual(Fraction(result["lower"]), exact)
                self.assertGreaterEqual(Fraction(result["upper"]), exact)
                self.assertGreaterEqual(Fraction(result["sum_abs_upper"]), magnitude)
                theoretical = 256 * u32 / (1 - 256 * u32) * magnitude
                theoretical += Fraction(256, 2 ** 150) / (1 - 256 * u32)
                self.assertGreaterEqual(Fraction(result["fp32_error_bound"]), theoretical)
                self.assertGreaterEqual(Fraction(result["binary64_sum_error_bound"]),
                                        256 * u64 / (1 - 256 * u64) * magnitude)
                self.assertGreaterEqual(Fraction(result["binary64_sum_error_bound"]),
                                        abs(exact - Fraction(result["score_binary64_fsum"])))
                rounded_products = [fp32(a * b) for a, b in zip(q, g)]
                forward = reverse = 0.0
                for p in rounded_products:
                    forward = fp32(forward + p)
                for p in reversed(rounded_products):
                    reverse = fp32(reverse + p)
                pairwise = rounded_products
                while len(pairwise) > 1:
                    pairwise = [fp32(a + b) for a, b in zip(pairwise[::2], pairwise[1::2])]
                for score in (forward, reverse, pairwise[0]):
                    self.assertLessEqual(result["lower"], score)
                    self.assertGreaterEqual(result["upper"], score)

    def test_numeric_failure_rejects_instead_of_certifying(self):
        for q, g in [(vector([math.nan]), vector([1])),
                     (vector([math.inf]), vector([1])),
                     (vector([2 ** 127]), vector([2 ** 127])),
                     ([1.0], vector([1]))]:
            with self.subTest(q=q[:1]), self.assertRaises(ValueError):
                self.d.dot_interval(q, g)

    def test_positive_separation_and_exact_ties(self):
        rows = [vector([1]), vector([0.25]), vector([0.5]), vector([0.5])]
        labels = ["A", "B", "A", "A"]
        result = self.d.compare_units(rows, labels, 0, [1, 2, 3])
        self.assertEqual(result["certificate"], "CERTIFIED_SEPARATION")
        self.assertEqual(result["best_positive"]["lower_witness"]["gallery_index"], 1)
        self.assertEqual(result["best_positive"]["upper_witness"]["panel_ordinal"], 2)
        rows[1] = vector([0.5])
        result = self.d.compare_units(rows, labels, 0, [1, 2, 3])
        self.assertEqual(result["certificate"], "INDETERMINATE")
        rows[1] = vector([1])
        self.assertEqual(self.d.compare_units(rows, labels, 0, [1, 2, 3])["certificate"],
                         "INDETERMINATE")

    def test_max_interval_bounds_can_have_different_witnesses(self):
        rows = [vector([1, 1]), vector([0.5]), vector([0.49]), vector([1024, -1023.52])]
        result = self.d.compare_units(rows, ['A', 'A', 'B', 'B'], 0, [1, 2, 3])
        self.assertEqual(result['best_negative']['lower_witness']['panel_ordinal'], 2)
        self.assertEqual(result['best_negative']['upper_witness']['panel_ordinal'], 3)
        self.assertEqual(result['certificate'], 'INDETERMINATE')


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_script()

    def test_exact_npy_shape_header_and_finites(self):
        rows = self.d.decode_unit(npy_bytes(value=2 ** -149))
        self.assertEqual((len(rows), len(rows[0])), (3449, 128))
        self.assertEqual(rows[-1][-1], 2 ** -149)
        headers = ["{'descr': '>f4', 'fortran_order': False, 'shape': (3449, 128)}",
                   "{'descr': '<f8', 'fortran_order': False, 'shape': (3449, 128)}",
                   "{'descr': '<f4', 'fortran_order': True, 'shape': (3449, 128)}",
                   "{'descr': '<f4', 'fortran_order': 0, 'shape': (3449, 128)}",
                   "{'descr': '<f4', 'fortran_order': False, 'shape': (128, 3449)}",
                   "{'descr': '<f4', 'fortran_order': False, 'shape': [3449, 128]}",
                   "{'descr': '<f4', 'fortran_order': False, 'shape': (3449.0, 128)}",
                   "{'descr': '<f4', 'fortran_order': False, 'shape': (3449, 128), 'x': 1}",
                   "{'descr': '<f4', 'descr': '<f4', 'fortran_order': False, 'shape': (3449, 128)}",
                   "__import__('os').getcwd()"]
        valid = npy_bytes()
        bad = [npy_bytes(h) for h in headers] + [valid + b'x', valid[:-1],
               b'x' + valid[1:], valid[:6] + b'\x02\x00' + valid[8:],
               valid[:8] + struct.pack('<H', 117) + valid[10:],
               valid[:127] + b' ' + valid[128:], npy_bytes(value=math.nan),
               npy_bytes(value=math.inf)]
        for data in bad:
            with self.subTest(header=data[:128]), self.assertRaises(ValueError):
                self.d.decode_unit(data)

    def test_authentication_rejects_changed_bytes_symlinks_and_import_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root / 'driver.py'
            marker = root / 'executed'
            path.write_text(f"open({str(marker)!r}, 'w').close()\n")
            descriptor = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                          'bytes': path.stat().st_size}
            with self.assertRaises(ValueError):
                self.d.load_census(descriptor, {})
            self.assertFalse(marker.exists())
            self.assertEqual(self.d.read_file(descriptor, {}), path.read_bytes())
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): self.d.read_file(descriptor, {})
            path.unlink(); path.symlink_to(SCRIPT)
            with self.assertRaises(ValueError): self.d.read_file(descriptor, {})

    def test_changed_original_source_is_rejected_before_private_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            original = SCRIPT.with_name('census_connected_core_errors.py')
            path = root / original.name
            path.write_bytes(original.read_bytes())
            pin = descriptor(path)
            guards = {}
            census = self.d.load_census(pin, guards)
            self.assertEqual(census.CORE_COUNT, 44)
            marker = root / 'executed'
            path.write_bytes(path.read_bytes() + f"\nopen({str(marker)!r}, 'w').close()\n".encode())
            for changed in (pin, descriptor(path)):
                with self.assertRaises(ValueError): self.d.load_census(changed, {})
            self.assertFalse(marker.exists())

    def test_poisoned_valid_bytecode_cache_is_never_executed_or_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            path = root / 'census_connected_core_errors.py'
            path.write_bytes(SCRIPT.with_name(path.name).read_bytes())
            marker = root / 'cache_executed'
            cache = Path(importlib.util.cache_from_source(str(path)))
            cache.parent.mkdir()
            header = importlib.util.MAGIC_NUMBER + struct.pack('<III', 0, int(path.stat().st_mtime),
                                                              path.stat().st_size)
            poison = compile(f"open({str(marker)!r}, 'w').close()", str(path), 'exec')
            cache.write_bytes(header + marshal.dumps(poison))
            before = {k for k in sys.modules if 'authenticated_archived_precision' in k}
            census = self.d.load_census(descriptor(path), {})
            self.assertEqual(census.CORE_COUNT, 44)
            self.assertFalse(marker.exists())
            self.assertEqual({k for k in sys.modules if 'authenticated_archived_precision' in k}, before)

    def test_json_and_file_descriptors_fail_closed(self):
        for raw in [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}']:
            with self.subTest(raw=raw), self.assertRaises(ValueError): self.d.json_value(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            file = root / 'data'; file.write_bytes(b'x')
            link = root / 'link'; link.symlink_to(root, target_is_directory=True)
            fifo = root / 'fifo'; os.mkfifo(fifo)
            for change in [{'path': str(link / 'data')}, {'path': str(fifo)},
                           {'path': str(root)}, {'path': 'data'}, {'path': str(root) + '/./data'},
                           {'bytes': True}, {'bytes': 2}, {'sha256': '0' * 63}, {'extra': 1}]:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.d.read_file(dict(descriptor(file), **change), {})


def toy_state(census):
    rows = [{'panel_ordinal': i, 'original_row': i, 'role': 'query' if i == 0 else 'gallery',
             'product': label} for i, label in enumerate(['A', 'B', 'A', 'A'])]
    quality = {s: {a: {'per_query_r1': [0], 'per_query_ap': [0.25]}
                   for a in ('control', 'candidate')} for s in ('179061', '179069')}
    state = {'partition': {'panels': {'selection': {'query': [0], 'gallery': [1, 2, 3],
                                                   'original_rows': [0, 1, 2, 3]}}},
             'core': [0], 'core_count': 1, 'rows': rows, 'guards': {}, 'record': {},
             'receipt': {'quality': quality, 'decision': 'KILL'},
             'wires': {f'{a}-{s}': [((1,) * 128, 0.125)] * 4 for s, a in census.ENDPOINTS}}
    units = {name: [vector([1]), vector([0.25]), vector([0.5]), vector([0.5])]
             for name in state['wires']}
    return state, units


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_script()
        cls.c = cls.d.load_census(descriptor(SCRIPT.with_name('census_connected_core_errors.py')), {})

    def test_toy_packed_replay_uses_stable_order_and_never_invokes_ap_gate(self):
        state, units = toy_state(self.c)
        with patch.object(self.c, 'build_census', side_effect=AssertionError('AP census forbidden')), \
             patch.object(self.c, 'replay_query', side_effect=AssertionError('AP gate forbidden')):
            value = self.d.build_diagnostic(self.c, state, units)
        row, = value['queries']
        self.assertEqual(value['precision_mechanism'], 'ARCHIVED_QUANTIZATION_SENSITIVITY_ONLY')
        self.assertEqual(value['original_decision'], 'KILL')
        self.assertEqual(value['original_core_census']['status'], 'FAIL')
        self.assertEqual(value['archived_quality_metadata']['quality'], state['receipt']['quality'])
        for endpoint in row['endpoints'].values():
            self.assertEqual(endpoint['packed_replay']['top1_gallery_index'], 0)
            self.assertEqual(endpoint['packed_replay']['top1_panel_ordinal'], 1)
            self.assertEqual(endpoint['packed_replay']['original_per_query_r1'], 0)
            self.assertEqual(endpoint['archived_ap_metadata'], {'value': 0.25, 'status': 'NOT REPLAYED'})
        for rows in units.values(): rows[1] = vector([0.75])
        value = self.d.build_diagnostic(self.c, state, units)
        self.assertEqual(value['precision_mechanism'],
                         'CLOSED_FOR_THIS_WITNESS_NO_CERTIFIED_POSITIVE_SEPARATION')

    def test_mapping_core_and_endpoint_mutations_reject_whole_artifact(self):
        for mutation in ('core', 'gallery_order', 'overlap', 'missing', 'panel_ordinal',
                         'original_row', 'role', 'endpoint', 'unit_rows', 'r1', 'nonfinite'):
            state, units = toy_state(self.c)
            panel = state['partition']['panels']['selection']
            if mutation == 'core': state['core'] = []
            if mutation == 'gallery_order': panel['gallery'].reverse()
            if mutation == 'overlap': panel['gallery'][0] = 0
            if mutation == 'missing': panel['gallery'].pop()
            if mutation == 'panel_ordinal': state['rows'][2]['panel_ordinal'] = 3
            if mutation == 'original_row': state['rows'][2]['original_row'] = 3
            if mutation == 'role': state['rows'][2]['role'] = 'query'
            if mutation == 'endpoint': del units['candidate-179069']
            if mutation == 'unit_rows': units['candidate-179069'].pop()
            if mutation == 'r1': state['wires']['candidate-179069'][2] = ((2,) * 128, 0.125)
            if mutation == 'nonfinite': units['candidate-179069'][2] = vector([math.nan])
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp).resolve() / 'never.json'
                with self.assertRaises(ValueError):
                    value = self.d.build_diagnostic(self.c, state, units)
                    self.c.publish(str(output), value, state['guards'])
                self.assertFalse(output.exists())

    def test_all_publication_guards_are_rechecked_and_existing_output_preserved(self):
        state, units = toy_state(self.c)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            input_file = root / 'unit.npy'; input_file.write_bytes(b'original')
            self.d.read_file(descriptor(input_file), state['guards'])
            value = self.d.build_diagnostic(self.c, state, units)
            output = root / 'result.json'
            real_fsync = self.c.os.fsync
            def mutate_after_write(fd):
                real_fsync(fd)
                input_file.write_bytes(b'changed')
            with patch.object(self.c.os, 'fsync', side_effect=mutate_after_write):
                with self.assertRaises(ValueError): self.c.publish(str(output), value, state['guards'])
            self.assertFalse(output.exists())
            self.assertEqual(list(root.iterdir()), [input_file])
            input_file.write_bytes(b'original')
            self.c.publish(str(output), value, state['guards'])
            first = output.read_bytes()
            with self.assertRaises(FileExistsError): self.c.publish(str(output), {}, state['guards'])
            self.assertEqual(output.read_bytes(), first)

    def test_unit_endpoint_provenance_and_current_bytes_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            descriptors = []
            originals = {}
            for seed, arm in self.d.ENDPOINTS:
                path = root / f'{arm}-{seed}.unit.npy'; path.write_bytes(npy_bytes())
                source = f'/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-{arm}-{seed}-v2/{path.name}'
                entry = dict(descriptor(path), seed=seed, arm=arm, source_path=source)
                descriptors.append(entry); originals[source] = entry['sha256']
            state = {'guards': {}, 'receipt': {'input_guards': originals}}
            units = self.d.admit_units(self.c, state, descriptors)
            self.assertEqual(list(units), ['control-179061', 'candidate-179061',
                                           'control-179069', 'candidate-179069'])
            self.assertEqual(len(state['guards']), 4)
            for mutation in ('order', 'missing', 'extra', 'seed', 'arm', 'source', 'pin', 'length', 'collision'):
                bad = copy.deepcopy(descriptors)
                if mutation == 'order': bad.reverse()
                if mutation == 'missing': bad.pop()
                if mutation == 'extra': bad.append(bad[0])
                if mutation == 'seed': bad[0]['seed'] = '179061'
                if mutation == 'arm': bad[0]['arm'] = 'candidate'
                if mutation == 'source': bad[0]['source_path'] = '/unbound.unit.npy'
                if mutation == 'pin': bad[0]['sha256'] = '0' * 64
                if mutation == 'length': bad[0]['bytes'] -= 1
                if mutation == 'collision': bad[1]['path'] = bad[0]['path']
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    self.d.admit_units(self.c, {'guards': {}, 'receipt': state['receipt']}, bad)
            path.write_bytes(npy_bytes(value=1))
            with self.assertRaises(ValueError): self.c.rehash(state['guards'])
            with self.assertRaises(ValueError):
                self.d.admit_units(self.c, {'guards': {}, 'receipt': state['receipt']}, descriptors)


class LaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.d = load_script()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.output = self.root / 'result.json'
        self.launch_path = self.root / 'launch.json'

    def test_cli_wrong_launch_hash_never_publishes(self):
        self.launch_path.write_text(json.dumps({'output': str(self.output)}))
        run = subprocess.run([sys.executable, '-B', '-I', '-S', str(SCRIPT), '--launch',
                              str(self.launch_path), '--launch-sha256', '0' * 64],
                             capture_output=True, text=True, timeout=5)
        self.assertEqual(run.returncode, 1)
        self.assertIn('sha256', run.stderr)
        self.assertFalse(self.output.exists())

    def test_cli_requires_bytecode_writes_disabled(self):
        self.launch_path.write_text(json.dumps({'output': str(self.output)}))
        env = dict(os.environ)
        env.pop('PYTHONDONTWRITEBYTECODE', None)
        run = subprocess.run([sys.executable, '-I', '-S', str(SCRIPT), '--launch',
                              str(self.launch_path), '--launch-sha256', descriptor(self.launch_path)['sha256']],
                             capture_output=True, text=True, timeout=5, env=env)
        self.assertEqual(run.returncode, 1)
        self.assertIn('bytecode', run.stderr)
        self.assertFalse(self.output.exists())

    def test_original_complete_admission_without_real_scoring(self):
        inputs = Path('/tmp/sfora-connected-core-census-inputs-v1/fetch-receipt.json')
        original_driver = Path('/tmp/sfora-connected-core-census-run-v1/census_connected_core_errors.py')
        unit_fetch = Path('/tmp/sfora-connected-archived-precision-unit-inputs-v1/fetch-receipt.json')
        if not all(p.exists() for p in (inputs, original_driver, unit_fetch)):
            self.skipTest('parent staged archived files absent; never fetch')
        units = json.loads(unit_fetch.read_bytes())['files']
        units = [{k: v for k, v in entry.items() if k != 'npy'} for entry in units]
        launch = {'schema': 'sfora-connected-archived-precision-launch-v1',
                  'census_inputs': descriptor(inputs),
                  'census_driver': descriptor(original_driver),
                  'sources': {name: descriptor(SCRIPT.with_name(name)) for name in self.d.SOURCES},
                  'units': units, 'output': str(self.output)}
        self.launch_path.write_text(json.dumps(launch))
        census, state, values, _ = self.d.admit_launch(str(self.launch_path), descriptor(self.launch_path)['sha256'])
        self.assertEqual(len(state['core']), 44)
        self.assertEqual(len(state['rows']), 3449)
        self.assertEqual(len(state['guards']), 18)
        self.assertEqual([len(v) for v in values.values()], [3449] * 4)
        census.rehash(state['guards'])
        self.assertFalse(self.output.exists())
        with patch.object(self.d, '__file__', str(self.root / 'renamed_driver.py')):
            with self.assertRaises(ValueError):
                self.d.admit_launch(str(self.launch_path), descriptor(self.launch_path)['sha256'])
        for mutation in ('schema', 'extra', 'own_source', 'missing_source', 'census_input', 'driver', 'existing_output'):
            bad = copy.deepcopy(launch)
            if mutation == 'schema': bad['schema'] = 'other'
            if mutation == 'extra': bad['fallback'] = True
            if mutation == 'own_source': bad['sources'][SCRIPT.name]['sha256'] = '0' * 64
            if mutation == 'missing_source': del bad['sources']['test_connected_archived_precision.py']
            if mutation == 'census_input': bad['census_inputs']['sha256'] = '0' * 64
            if mutation == 'driver': bad['census_driver']['sha256'] = '0' * 64
            if mutation == 'existing_output': self.output.write_text('preserved')
            self.launch_path.write_text(json.dumps(bad))
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, FileExistsError)):
                self.d.admit_launch(str(self.launch_path), descriptor(self.launch_path)['sha256'])
        self.assertEqual(self.output.read_text(), 'preserved')


if __name__ == "__main__":
    unittest.main()
