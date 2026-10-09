import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SCRIPT = Path('/tmp/sfora_identity_diversity_residency_sampler.py')

class SamplerFixtures(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.is_file(), 'sampler artifact missing')
        spec = importlib.util.spec_from_file_location('sampler', SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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
            self.assertLessEqual(len(rows), 1201)
            self.assertEqual(clock[0], 550.)

if __name__ == '__main__': unittest.main()
