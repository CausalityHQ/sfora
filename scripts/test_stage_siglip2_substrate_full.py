#!/usr/bin/env python3
"""Hermetic parent-tool checks: fake tiny checkpoints/logs, no systemd/native/DGX execution."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import stage_siglip2_substrate_full as stage
import test_siglip2_substrate_full as reviewed
from test_siglip2_substrate_full import stdlib_import

HERE = Path(__file__).resolve().parent


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        runs = Path(self.tmp.name)
        self.src = runs / 'src'
        self.src.mkdir()
        for name, value in {
            'RUNS': runs, 'ROOT': runs / 'sfora-native256-full-v1', 'OUTPUT': runs / 'sfora-native256-full-run-v1',
            'PILOT': runs / 'sfora-native256-adaptation-source-v9', 'OLD': runs / 'sfora-native256-adaptation-source-v6',
            'FOOTERS': tuple(runs / name for name in ('sfora-native256-initialized-cpu-final-footer-v1.py', 'sfora-native256-stop-cgroup-footer-v1.py')),
            'PYTHON': sys.executable,
        }.items():
            self.enterContext(patch.object(stage, name, value))
        self.enterContext(patch.dict(os.environ, {'SFORA_NATIVE256_TEST_BYTES': '1'}))
        self.enterContext(patch.dict(sys.modules))
        self.enterContext(patch('builtins.__import__', stdlib_import))
        stage.PILOT.mkdir()
        stage.OLD.mkdir()
        for f in stage.LOCAL:
            source = HERE.parent / 'src/sfora/unicom_inshop.py' if f == 'unicom_inshop.py' else HERE / f
            (self.src / f).write_bytes(source.read_bytes())
        for f in stage.PILOT_FILES:
            (stage.PILOT / f).write_bytes(b'# fake immutable pilot reference: ' + f.encode())
        (stage.PILOT / 'execution.json').write_text(json.dumps({f: stage.sha(stage.PILOT / f) for f in stage.PILOT_FILES}))
        for path in stage.FOOTERS:
            path.write_text('# fake footer\n')
        self.header = (f'#!/bin/bash\nset -euo pipefail\ncd {stage.OLD}\n'
                       f"trap 'status=$?; SFORA_COMMAND_STATUS=$status /usr/bin/python3 {stage.FOOTERS[0]}; exit $status' EXIT\n")
        (stage.OLD / 'large-mechanics-command.sh').write_text(self.header + "sha256sum -c <<'HASHES'\nunused\nHASHES\n")
        self.capture(stage.stage, str(self.src))
        self.full, self.score, self.code = stage.modules()
        self.sources = {}
        for arm in ('large', 'so400'):
            folder = runs / ('pilot-' + arm)
            folder.mkdir()
            (folder / 'resume.pt').write_bytes(b'fake pilot checkpoint')
            checkpoint = self.full.descriptor(folder / 'resume.pt')
            record = {'schema': 'siglip2-substrate-adaptation-v1', 'phase': 'train', 'arm': arm,
                      'seed': 179032, 'width': self.full.WIDTHS[arm], 'completed_step': 100, 'checkpoint': checkpoint,
                      'identity': {'arm': arm, 'seed': 179032, 'config': {'hidden_size': self.full.WIDTHS[arm]}},
                      'pass': True, 'training_qualified': True, 'quality_read': False, 'training_state_discarded': False,
                      'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
                      'exit_rehash_pass': True, 'resource_policy': self.full.POLICY,
                      'code': {'train_siglip2_substrate_adaptation.py': self.full.PILOT_SHA},
                      'execution_sha256': self.score.PILOT_EXECUTION_SHA}
            self.full.publish(folder / 'receipt.json', record)
            receipt = self.full.descriptor(folder / 'receipt.json')
            self.sources[arm] = {'receipt': receipt, 'checkpoint': checkpoint}
        real_modules = stage.modules
        def fake_sources_only():
            full, score, code = real_modules()
            score.PILOT_RECEIPTS = {a: s['receipt'] for a, s in self.sources.items()}
            return full, score, code
        self.enterContext(patch.object(stage, 'modules', side_effect=fake_sources_only))
        self.full, self.score, self.code = stage.modules()

    def capture(self, function, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            function(*args)
        return out.getvalue()

    def command(self, phase, arm, end=None):
        name, _ = stage.target(phase, arm, end)
        return stage.ROOT / f'{name}-command.sh'

    def command_argv(self, phase, arm, end=None):
        text = self.command(phase, arm, end).read_text()
        return shlex.split(text.split('HASHES\n')[1].splitlines()[0])

    def receipt(self, phase, arm, end=None, *, admitted=True, record=None):
        name, folder = stage.target(phase, arm, end)
        folder.mkdir(parents=True)
        if record is None:
            code = {'trainer': self.code[stage.TRAINER], 'partition_parser': self.full.PARSER_SHA}
            source = self.sources[arm]
            prior = folder.parent / f'chunk-{end - 100:04d}'
            previous = self.full.descriptor(prior / 'receipt.json') if end > 100 else None
            previous_terminal = self.full.descriptor(prior / 'terminal.json') if end > 100 else None
            checkpoint = self.full.read_json(previous)['checkpoint']['sha256'] if previous else None
            (folder / 'resume.pt').write_bytes(f'fake {arm} {end}'.encode())
            record = {'schema': self.full.SCHEMA, 'pass': True, 'quality_read': False, 'arm': arm, 'width': self.full.WIDTHS[arm],
                      'seed': self.full.SEED, 'completed_step': end, 'chunk_start': end - 100, 'source': source, 'code': code,
                      'resource_policy': self.full.POLICY, 'reload_exact': True, 'exit_rehash_pass': True,
                      'identity': {'schema': self.full.SCHEMA, 'arm': arm, 'source': source, 'code': code,
                                   'full_rows_sha256': 'rows', 'schedule_sha256': 'schedule'},
                      'steps': [{'step': i} for i in range(end - 99, end + 1)], 'checkpoint': self.full.descriptor(folder / 'resume.pt'),
                      'previous': previous, 'previous_terminal': previous_terminal, 'previous_checkpoint_sha256': checkpoint}
        inv = hashlib.md5(name.encode()).hexdigest()
        # Reuse the reviewed real terminal fixture and move its log to the staged code root.
        record, terminal, log = reviewed.Contract().terminal_fixture(folder, record, name, inv)
        destination = stage.ROOT / f'{name}.log'
        log.rename(destination)
        terminal['log'] = self.full.descriptor(destination)
        (folder / 'terminal.json').write_text(json.dumps(terminal))
        if not admitted:
            (folder / 'terminal.json').unlink()
        return record

    def complete(self):
        for arm in ('large', 'so400'):
            for end in range(100, 2001, 100):
                self.receipt('train', arm, end)

    def score_receipt(self, arm):
        training = self.full.read_json(self.full.descriptor(stage.OUTPUT / arm / 'chunk-2000/receipt.json'))
        terminals = {a: self.full.descriptor(stage.OUTPUT / a / 'chunk-2000/terminal.json') for a in ('large', 'so400')}
        record = {'schema': self.score.SCHEMA, 'pass': True, 'arm': arm, 'resource_policy': self.full.POLICY,
                  'exit_rehash_pass': True, 'training': training, 'paired_training_terminals': terminals,
                  'code': {'trainer': self.code[stage.TRAINER], 'scorer': self.code[stage.SCORER]}}
        if arm == 'so400':
            record.update(control=self.full.descriptor(stage.OUTPUT / 'official/large/receipt.json'),
                          control_terminal=self.full.descriptor(stage.OUTPUT / 'official/large/terminal.json'),
                          paired={'candidate_minus_control': {'per_query_r1': {'mean_delta': .01, 'product_lower95': .001}},
                                  'decision': 'GO', 'bootstrap_draws': 5000, 'bootstrap_seed': 179019})
        self.receipt('score', arm, record=record)
        for name in ('query.pt', 'gallery.pt'):
            (stage.OUTPUT / 'official' / arm / name).write_bytes(b'fake wire')

    def test_staged_closure_and_create_only(self):
        self.assertEqual(set(self.code), set(stage.FILES))
        for f in stage.PILOT_FILES:
            self.assertEqual((stage.ROOT / f).read_bytes(), (stage.PILOT / f).read_bytes())
        self.assertEqual(self.full.inspect_layout(stage.OUTPUT), {'large': 0, 'so400': 0})
        with self.assertRaisesRegex(AssertionError, 'existing path'):
            stage.stage(str(self.src))
        with patch.object(stage, 'ROOT', stage.RUNS / 'unused-code'):
            with self.assertRaisesRegex(AssertionError, 'existing path'):
                stage.stage(str(self.src))
            self.assertFalse(stage.ROOT.exists())
        for f in (stage.TRAINER, stage.SCORER, 'test_siglip2_substrate_full.py'):
            self.assertEqual(self.code[f], stage.PINS[f])
        with patch.dict(os.environ, {'SFORA_NATIVE256_TEST_BYTES': '0'}):
            with self.assertRaisesRegex(AssertionError, 'pinned source'):
                stage.pins({**stage.PINS, stage.TRAINER: 'bad'})

    def test_launch_exact_argv_and_hash_blocks(self):
        line = self.capture(stage.launch, 'train', 'large', '100').strip()
        name, _ = stage.target('train', 'large', 100)
        command = self.command('train', 'large', 100)
        log = stage.ROOT / f'{name}.log'
        argv = shlex.split(line.split(' -- ', 1)[1])
        self.assertEqual(argv, ['systemd-run', '--user', '--wait', '--collect', f'--unit={name}',
            '-p', f'StandardOutput=append:{log}', '-p', f'StandardError=append:{log}',
            '-p', f'WorkingDirectory={stage.ROOT}', '-p', 'RuntimeMaxSec=3600', '-p', 'MemoryMax=8G',
            '-p', 'MemorySwapMax=0', '-p', 'KillMode=control-group', '-p', f'ExecStopPost=/usr/bin/python3 {stage.FOOTERS[1]}',
            '--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8',
            '/usr/bin/flock', '-n', '/home/riomus/runs/.sfora-siglip2-gpu.lock',
            '/usr/bin/flock', '-n', '/home/riomus/.sfora-siglip2-gpu.lock', '/usr/bin/time', '-v', '/bin/bash', str(command)])
        env = shlex.split(line.split(' -- ', 1)[0])
        self.assertIn('BACKSTOP_SECONDS=3780', env)
        self.assertIn(f'STOP_CMD=systemctl --user stop {name}.service', env)
        self.assertIn(f'LIVENESS_CMD=cg=$(systemctl --user show -P ControlGroup {name}.service); [ -n "$cg" ] && cat "/sys/fs/cgroup$cg/cpu.stat"', env)
        self.assertEqual(env[-2:], [str(stage.ROOT / 'run_with_progress_watchdog.sh'), str(log)])
        text = command.read_text()
        self.assertTrue(text.startswith(self.header.replace(f'cd {stage.OLD}', f'cd {stage.ROOT}')))
        blocks = [part.split('HASHES\n')[0] for part in text.split("<<'HASHES'\n")[1:]]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], blocks[1])
        expected = {str(stage.ROOT / f) for f in (*stage.FILES, 'execution.json')}
        expected.update(str(stage.PILOT / f) for f in (*stage.PILOT_FILES, 'execution.json'))
        expected.update(map(str, stage.FOOTERS))
        self.assertEqual({row.split('  ', 1)[1] for row in blocks[0].splitlines()}, expected)
        subprocess.run(['sha256sum', '-c', '--quiet'], input=blocks[0], text=True, check=True, timeout=10)
        args = self.command_argv('train', 'large', 100)
        self.assertEqual(args[:3], [sys.executable, '-B', str(stage.ROOT / stage.TRAINER)])
        self.assertNotIn('--previous', args)
        self.assertEqual(args[args.index('--source-sha256') + 1], self.sources['large']['checkpoint']['sha256'])
        subprocess.run(['bash', '-n', str(command)], check=True, timeout=10)

    def test_existing_log_command_and_tampered_code_refused(self):
        name, _ = stage.target('train', 'large', 100)
        log = stage.ROOT / f'{name}.log'
        log.write_text('prior evidence')
        with self.assertRaisesRegex(AssertionError, 'existing path'):
            stage.launch('train', 'large', '100')
        log.unlink()
        self.capture(stage.launch, 'train', 'large', '100')
        with self.assertRaisesRegex(AssertionError, 'existing path'):
            stage.launch('train', 'large', '100')
        path = stage.ROOT / stage.TRAINER
        path.write_bytes(path.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(AssertionError, 'frozen code SHA'):
            stage.modules()

    def test_real_parent_admission_runtime_and_verified_chaining(self):
        self.capture(stage.launch, 'train', 'large', '100')
        self.receipt('train', 'large', 100, admitted=False)
        with self.assertRaises(ValueError):
            stage.launch('train', 'large', '200')
        real_run = subprocess.run
        def quiet_run(*args, **kwargs):
            return real_run(*args, **kwargs, capture_output=True, text=True, timeout=10)
        with patch.object(stage.subprocess, 'run', side_effect=quiet_run) as process:
            text = self.capture(stage.admit, 'train', 'large', '100')
        self.assertIn('ADMIT ', text)
        self.assertIn('--both-locks-held', process.call_args.args[0])
        term, receipt, _ = stage.terminal(self.full, 'train', 'large', 100)
        self.assertEqual(self.full.read_json(term)['service_seconds'], 180.081)
        self.capture(stage.launch, 'train', 'large', '200')
        argv = self.command_argv('train', 'large', 200)
        for key, value in (('--previous', receipt['path']), ('--previous-sha256', receipt['sha256']),
                           ('--previous-terminal-sha256', term['sha256'])):
            self.assertEqual(argv[argv.index(key) + 1], value)
        with patch.object(stage.subprocess, 'run', side_effect=quiet_run), self.assertRaises(subprocess.CalledProcessError):
            self.capture(stage.admit, 'train', 'large', '100')

    def test_corrupt_previous_receipt_and_log_stop_chaining(self):
        self.receipt('train', 'large', 100)
        receipt = stage.OUTPUT / 'large/chunk-0100/receipt.json'
        original = receipt.read_bytes()
        receipt.write_bytes(original + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            stage.launch('train', 'large', '200')
        receipt.write_bytes(original)
        name, _ = stage.target('train', 'large', 100)
        log = stage.ROOT / f'{name}.log'
        log.write_text(log.read_text() + 'watchdog: stopped\n')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            stage.launch('train', 'large', '200')
        self.assertFalse(self.command('train', 'large', 200).exists())

    def test_fixed_order_and_no_scoring_before_both_chains(self):
        with self.assertRaisesRegex(AssertionError, 'order'):
            stage.launch('train', 'so400', '100')
        with self.assertRaises(ValueError):
            stage.launch('score', 'large')
        self.assertFalse(self.command('score', 'large').exists())

    def test_complete_pair_score_order_control_and_decision(self):
        self.complete()
        with self.assertRaisesRegex(ValueError, 'official attempt'):
            stage.launch('score', 'so400')
        line = self.capture(stage.launch, 'score', 'large')
        self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=0', shlex.split(line))
        self.assertIn('--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8', shlex.split(line))
        args = self.command_argv('score', 'large')
        self.assertNotIn('--control-terminal-sha256', args)
        for arm in ('large', 'so400'):
            value = self.full.descriptor(stage.OUTPUT / arm / 'chunk-2000/terminal.json')['sha256']
            self.assertEqual(args[args.index('--' + arm + '-terminal-sha256') + 1], value)
        self.score_receipt('large')
        self.capture(stage.launch, 'score', 'so400')
        args = self.command_argv('score', 'so400')
        self.assertEqual(args[args.index('--control-terminal-sha256') + 1],
                         self.full.descriptor(stage.OUTPUT / 'official/large/terminal.json')['sha256'])
        self.score_receipt('so400')
        self.assertIn('"decision": "GO"', self.capture(stage.decision))
        path = stage.OUTPUT / 'official/large/receipt.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            stage.decision()


class Driver(unittest.TestCase):
    def test_fixed_order_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            driver = root / 'run_siglip2_substrate_full.sh'
            driver.write_bytes((HERE / driver.name).read_bytes())
            tool = root / 'stage_siglip2_substrate_full.py'
            tool.write_text("""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with (Path(__file__).parent / 'calls').open('a') as stream:
    stream.write(json.dumps(args) + '\\n')
fail = os.environ.get('FAIL_AT') == ' '.join(args)
if args[0] == 'launch':
    print('exit 7' if fail else 'exit 0')
elif fail:
    raise SystemExit(9)
elif args[0] == 'admit':
    print('ADMIT fake')
else:
    print('DECISION fake')
""")
            units = [('train', arm, str(end)) for arm in ('large', 'so400') for end in range(100, 2001, 100)]
            units += [('score', 'large'), ('score', 'so400')]
            for failure in ('', 'launch train large 200', 'admit train large 100'):
                calls = root / 'calls'
                calls.unlink(missing_ok=True)
                result = subprocess.run(['bash', str(driver)], env={**os.environ, 'FAIL_AT': failure},
                                        capture_output=True, text=True, timeout=30)
                actual = [json.loads(line) for line in calls.read_text().splitlines()]
                expected = [list((operation, *unit)) for unit in units for operation in ('launch', 'admit')] + [['decision']]
                if failure:
                    index = expected.index(failure.split())
                    self.assertEqual(actual, expected[:index + 1])
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn('DECISION', result.stdout)
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(actual, expected)
                    self.assertEqual(result.stdout.count('START '), 42)
                    self.assertEqual(result.stdout.count('UNIT_RC '), 42)
                    self.assertEqual(result.stdout.count('ADMIT '), 42)
                    self.assertRegex(result.stdout, r'FULL \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ START train large 100')
                    self.assertIn('DECISION', result.stdout)


if __name__ == '__main__':
    unittest.main()
