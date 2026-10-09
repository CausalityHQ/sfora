"""Source-only tests for scripts/observe_asymmetric_exit_residency.py: simulated clocks, no sleep, no target I/O.

AST inverse contract. The production observer is the committed original sampler (SHA-256 pinned below) with
exactly these edits, and undoing exactly them on the production AST must reproduce the original AST:
  1. the module docstring text;
  2. int constants 550 -> 750 (x3: loop bound, final sleep clamp, terminal reason) and 1200 -> 1500 (x1: sample cap);
  3. a new module-level `def exit_marker` (the new-wire predicate) and Phases.read's phase test
     `A` -> `A or exit_marker(value)`, where A is the original COMPACT_PHASE test.
"""
import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE/'observe_asymmetric_exit_residency.py'
ORIGINAL = (HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/'
            'asymmetric-exit-phase-source-v1/original_residency_sampler.py')
ORIGINAL_SHA256 = '53409094fbbffd51eb8d7f38b47e6248d60b9e6346ed1ec184cf5a1574639922'
OUTPUT_CAP = 16*1024**2
BASE = {'phase': 'final_guard.resources', 'boundary': 'entry', 'monotonic_seconds': 123.5}


def load():
    assert SCRIPT.is_file(), 'observer artifact missing'
    spec = importlib.util.spec_from_file_location('observer', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wire(**changes):
    return json.dumps({**BASE, **changes})


def without(key):
    return json.dumps({name: value for name, value in BASE.items() if name != key})


def append(path, data):
    with path.open('ab') as stream:
        stream.write(data if isinstance(data, bytes) else data.encode())


class Inverse(ast.NodeTransformer):
    """Undo edits 2 and 3 on a production AST; the counts prove each edit happened exactly where expected."""
    CALL = ast.dump(ast.parse('exit_marker(value)', mode='eval').body)

    def __init__(self):
        self.constants, self.predicates = {750: 0, 1500: 0}, 0

    def visit_Constant(self, node):
        if type(node.value) is int and node.value in self.constants:
            self.constants[node.value] += 1
            return ast.Constant({750: 550, 1500: 1200}[node.value])
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Or) and len(node.values) == 2 and ast.dump(node.values[1]) == self.CALL:
            self.predicates += 1
            return node.values[0]
        return node


class SamplerFixtures(unittest.TestCase):
    def module(self):
        return load()

    def test_raw_resource_event_and_pid_status_are_observed_without_admission(self):
        sampler = self.module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); cg = root/'unit.service'; cg.mkdir()
            proc = root/'proc'; (proc/'123').mkdir(parents=True)
            for name in sampler.MEMORY_FILES:
                (cg/name).write_text('max 68\noom 0\n' if name == 'memory.events' else '1711945083\n')
            (cg/'cgroup.procs').write_text('123\n456\n')
            (proc/'123/status').write_text('Name:\tpython\nVmRSS:\t5939132 kB\nVmData:\t123 kB\nRssAnon:\t42 kB\nRssFile:\t52 kB\n')
            before = {p: p.read_bytes() for p in cg.iterdir()}
            record = sampler.snapshot(cg, proc)
            self.assertEqual(record['values']['memory.events'], 'max 68\noom 0')
            self.assertEqual(record['pids']['123']['VmRSS'], '5939132 kB')
            self.assertEqual(record['pids']['123']['VmData'], '123 kB')
            self.assertIn('error', record['pids']['456'])
            self.assertEqual(before, {p: p.read_bytes() for p in cg.iterdir()})

    def test_partial_log_and_last_phase(self):
        sampler = self.module()
        with TemporaryDirectory() as tmp:
            log = Path(tmp)/'original.log'
            reader = sampler.Phases(log)
            self.assertIsNone(reader.read())
            line = json.dumps({'event': 'COMPACT_PHASE', 'phase': 'shared_bundle_preparation', 'boundary': 'begin', 'seconds': 191.826})
            log.write_text('startup\n'+line[:20])
            self.assertIsNone(reader.read())
            with log.open('a') as stream: stream.write(line[20:]+'\n')
            self.assertEqual(reader.read(), line)
            self.assertEqual(reader.read(), line)
            with log.open('a') as stream: stream.write('Compact-ranking rejected: cgroup memory failure event\n')
            self.assertEqual(reader.read(), line)

    def test_bounded_exclusive_output_and_target_lifecycle(self):
        sampler = self.module()
        stream = io.StringIO()
        used = sampler.emit(stream, {'event': 'sample'}, 0, 64)
        self.assertGreater(used, 0)
        with self.assertRaisesRegex(ValueError, 'output byte cap'):
            sampler.emit(stream, {'event': 'sample'}, used, used)
        self.assertEqual(len(stream.getvalue().encode()), used)
        with TemporaryDirectory() as tmp:
            root = Path(tmp); cg = root/'unique.service'; output = root/'samples.jsonl'
            clock = [0.0]
            def sleep(seconds):
                clock[0] += seconds
                if clock[0] == .5: cg.mkdir()
                elif clock[0] == 1.: cg.rmdir()
            sampler.observe(cg, root/'original.log', output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]['event'], 'waiting')
            self.assertEqual(rows[1]['event'], 'sample')
            self.assertEqual(rows[-1]['reason'], 'cgroup_disappeared')
            self.assertEqual(clock[0], 1.)
            with self.assertRaises(FileExistsError): sampler.observe(cg, root/'original.log', output)

    def test_watchdog_without_unit_creation(self):
        sampler = self.module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); clock = [0.0]
            def sleep(seconds): clock[0] += seconds
            sampler.observe(root/'absent.service', root/'log', root/'samples', clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in (root/'samples').read_text().splitlines()]
            self.assertEqual(rows[-1]['reason'], 'watchdog')
            self.assertLessEqual(len(rows), 1501)
            self.assertEqual(clock[0], 750.)


class ExitMarkerObserver(unittest.TestCase):
    def test_ast_inverse_is_original(self):
        self.assertEqual(hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(), ORIGINAL_SHA256)
        original, production = ast.parse(ORIGINAL.read_text()), ast.parse(SCRIPT.read_text())
        doc = ast.get_docstring(production, clean=False)
        self.assertIn('750s watchdog', doc); self.assertIn('1500 sample cap', doc); self.assertNotIn('550', doc)
        production.body[0].value.value = ast.get_docstring(original, clean=False)
        helpers = [n for n in production.body if isinstance(n, ast.FunctionDef) and n.name == 'exit_marker']
        self.assertEqual(len(helpers), 1)
        production.body = [n for n in production.body if n is not helpers[0]]
        inverse = Inverse()
        restored = ast.fix_missing_locations(inverse.visit(production))
        self.assertEqual(inverse.constants, {750: 3, 1500: 1})
        self.assertEqual(inverse.predicates, 1)
        self.assertEqual(ast.dump(restored), ast.dump(original))

    def test_exit_marker_wire_accepted_raw_and_partial(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            log = Path(tmp)/'original.log'; reader = sampler.Phases(log)
            for line in [wire(boundary=b) for b in ('entry', 'exit', 'error')] + [
                    wire(monotonic_seconds=0), wire(monotonic_seconds=0.0), wire(monotonic_seconds=10**400)]:
                append(log, 'startup noise\n'+line+'\n')
                self.assertEqual(reader.read(), line)
            line = wire(phase='cleanup.publish_receipt', boundary='exit')
            append(log, line[:25])
            self.assertEqual(reader.read(), wire(monotonic_seconds=10**400))
            append(log, line[25:]+'\n')
            self.assertEqual(reader.read(), line)
            self.assertEqual(reader.read(), line)

    def test_malformed_markers_ignored(self):
        sampler = load()
        bad = [without('phase'), without('boundary'), without('monotonic_seconds'),
               wire(event='other'), wire(extra=1), wire(seconds=1.0),
               wire(boundary='begin'), wire(boundary='ENTRY'), wire(boundary=None), wire(boundary=['entry']),
               wire(boundary=1), wire(phase=None), wire(phase=1), wire(phase=['a']),
               wire(monotonic_seconds=True), wire(monotonic_seconds=False), wire(monotonic_seconds='1.0'),
               wire(monotonic_seconds=None), wire(monotonic_seconds=[1.0]),
               wire(monotonic_seconds=float('nan')), wire(monotonic_seconds=float('inf')),
               wire(monotonic_seconds=-float('inf')), wire(monotonic_seconds=0)[:-2]+'1e999}',
               '[]', '"phase"', '123', 'null', 'true', json.dumps([BASE]), '{"phase": "p", "boundary": "entry"',
               '', '   ']
        with TemporaryDirectory() as tmp:
            log = Path(tmp)/'original.log'; reader = sampler.Phases(log)
            for line in bad:
                append(log, line+'\n')
            append(log, b'\xff\xfe{"phase"\n')
            self.assertIsNone(reader.read())
            good = wire(boundary='exit')
            append(log, good+'\n')
            self.assertEqual(reader.read(), good)
            for line in bad:
                append(log, line+'\n')
            append(log, b'\xff\xfe{"phase"\n')
            self.assertEqual(reader.read(), good)

    def test_last_phase_field_keeps_raw_line_for_both_wires(self):
        sampler = load()
        old = json.dumps({'event': 'COMPACT_PHASE', 'phase': 'x', 'boundary': 'begin', 'seconds': 1.0})
        with TemporaryDirectory() as tmp:
            root = Path(tmp); log = root/'original.log'; output = root/'samples.jsonl'
            clock = [0.0]; log.write_text(old+'\n')
            def sleep(seconds):
                clock[0] += 250.
                append(log, wire(boundary='error')+'\n')
            sampler.observe(root/'absent.service', log, output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row.get('last_COMPACT_PHASE') for row in rows[:3]], [old, wire(boundary='error'), wire(boundary='error')])
            self.assertNotIn('phase_log_error', rows[0])

    def test_clock_700_through_750_is_sampled_then_watchdog(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'samples.jsonl'
            clock, at, requested = [0.0], [], []
            def sleep(seconds):
                at.append(clock[0]); requested.append(seconds); clock[0] += seconds
            reason = sampler.observe(root/'absent.service', root/'log', output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(reason, 'watchdog')
            self.assertEqual(set(requested), {.5})
            self.assertEqual((len(at), at[1400], at[-1]), (1500, 700., 749.5))
            self.assertEqual(rows[1400]['sample'], 1400)
            self.assertEqual(rows[1499]['sample'], 1499)
            self.assertEqual(len(rows), 1501)
            self.assertEqual({k: rows[-1][k] for k in ('reason', 'samples', 'elapsed_seconds')},
                             {'reason': 'watchdog', 'samples': 1500, 'elapsed_seconds': 750.})

    def test_sample_cap_1500_before_watchdog(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'samples.jsonl'; clock = [0.0]
            def sleep(seconds): clock[0] += .1
            reason = sampler.observe(root/'absent.service', root/'log', output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual((reason, len(rows), rows[-1]['samples']), ('sample_cap', 1501, 1500))
            self.assertLess(clock[0], 750.)

    def test_final_sleep_is_clamped_to_750(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); output = root/'samples.jsonl'; clock, requested = [0.0], []
            def sleep(seconds):
                requested.append(seconds); clock[0] += 749.75 if len(requested) == 1 else seconds
            reason = sampler.observe(root/'absent.service', root/'log', output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual((reason, requested, rows[-1]['samples']), ('watchdog', [.5, .25], 2))

    def test_unreadable_log_is_recorded_and_not_fatal(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); log = root/'original.log'; output = root/'samples.jsonl'
            clock = [0.0]; log.write_text(wire()+'\n')
            def sleep(seconds):
                clock[0] += 250.; log.write_text('x\n')
            sampler.observe(root/'absent.service', log, output, clock=lambda: clock[0], sleep=sleep)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]['last_COMPACT_PHASE'], wire())
            self.assertIn('truncated', rows[1]['phase_log_error'])
            self.assertEqual(rows[1]['last_COMPACT_PHASE'], wire())
            self.assertEqual(rows[-1]['reason'], 'watchdog')
            long_line = root/'long.log'; long_line.write_text('x'*65537)
            reader = sampler.Phases(long_line)
            self.assertIsNone(reader.read())
            with self.assertRaisesRegex(ValueError, 'exceeds64KiB'):
                reader.read()

    def test_output_cap_16mib_is_unchanged(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); log = root/'original.log'; output = root/'samples.jsonl'; clock = [0.0]
            log.write_text(json.dumps({'event': 'COMPACT_PHASE', 'pad': 'x'*60000})+'\n')
            def sleep(seconds): clock[0] += seconds
            with self.assertRaisesRegex(ValueError, 'output byte cap'):
                sampler.observe(root/'absent.service', log, output, clock=lambda: clock[0], sleep=sleep)
            lines = output.read_bytes().splitlines()
            self.assertTrue(all(json.loads(line) for line in lines))
            size = output.stat().st_size
            self.assertLessEqual(size, OUTPUT_CAP-4096)
            self.assertGreater(size, OUTPUT_CAP-4096-70000)

    def test_target_and_log_are_only_read(self):
        sampler = load()
        with TemporaryDirectory() as tmp:
            root = Path(tmp); cg = root/'unit.service'; cg.mkdir(); log = root/'original.log'; output = root/'samples.jsonl'
            for name in sampler.MEMORY_FILES:
                (cg/name).write_text('\n' if name == 'cgroup.procs' else '1711945083\n')
            log.write_text(wire()+'\n')
            def state():
                return ({p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in [*cg.iterdir(), log]}, sorted(p.name for p in root.iterdir()))
            before, clock = state(), [0.0]
            def sleep(seconds): clock[0] += 250.
            sampler.observe(cg, log, output, clock=lambda: clock[0], sleep=sleep)
            files, names = state()
            self.assertEqual(files, before[0])
            self.assertEqual(names, sorted([*before[1], 'samples.jsonl']))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['sample']*3+['observer_terminal'])
            self.assertTrue(rows[-1]['cgroup_seen'] and rows[-1]['qualification_eligible'] is False)

    def test_source_surface_is_stdlib_read_only(self):
        tree = ast.parse(SCRIPT.read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imported, {'argparse', 'json', 'pathlib', 'signal', 'time'})
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        modes = sorted(c.args[0].value if c.args else 'r' for c in calls if c.func.attr == 'open')
        self.assertEqual(modes, ['r', 'rb', 'x'])
        forbidden = {'write_text', 'write_bytes', 'unlink', 'rmdir', 'mkdir', 'rename', 'replace', 'touch', 'chmod',
                     'symlink_to', 'truncate', 'kill', 'system', 'Popen', 'start'}
        self.assertEqual({c.func.attr for c in calls} & forbidden, set())
        self.assertEqual(sum(c.func.attr == 'write' for c in calls), 1)


if __name__ == '__main__': unittest.main()
