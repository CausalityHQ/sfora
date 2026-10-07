#!/usr/bin/env python3
"""One stdlib-only falsifier of source admission, timing and context restoration."""
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref  # Preload the synthetic target's stdlib before its denied-open boundary.

PATH = Path(__file__).resolve().with_name('observe_connected_mlp_phases.py')
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora', '_lsprof'}


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class ObserverTest(unittest.TestCase):
    def test_source_execution_profile_lifecycle_and_admission(self):
        spec = importlib.util.spec_from_file_location('connected_phase_observer_test', PATH)
        d = importlib.util.module_from_spec(spec)
        exec(compile(PATH.read_bytes(), str(PATH), 'exec'), vars(d))
        saved = sys.argv, sys.path, sys.modules['__main__'], sys.getprofile()
        self.assertEqual(d.PHASES, frozenset(('authority', 'fresh', 'load_initializer', 'construct_encoder',
            'update', 'integrity', 'encoder_facts', 'fingerprint', 'save', 'restore', 'export_bundle',
            'qualify_bundle', 'load_inference', 'inference_outputs', 'portable_mutants', 'release',
            'serving_environment', 'run')))
        try:
            with tempfile.TemporaryDirectory() as directory:
                base = Path(directory)
                original = base / 'original'
                original.mkdir()
                target = original / 'trainer.py'
                authority = original / 'authority.json'
                execution = original / 'execution.json'
                authority.write_bytes(b'{"original":true}')
                execution.write_bytes(b'{"source":"unchanged"}')
                argv = [str(target), '--execution-sha256', binding(execution)['sha256'], '--authority',
                    str(authority), '--authority-sha256', binding(authority)['sha256'], '--phase', 'cpu',
                    '--arm', 'control', '--seed', '179061', '--output', str(base / 'output')]
                raw = ("import sys, time, weakref\n"
                    f"assert sys.argv == {argv!r}\n"
                    f"assert __file__ == {str(target)!r} and sys.path[0] == {str(original)!r}\n"
                    "assert __name__ == '__main__' and __spec__ is None and __package__ is None\n"
                    "assert __cached__ is None and __loader__.path == __file__\n"
                    "assert sys.modules['__main__'].__dict__ is globals()\n"
                    "assert 'observer' not in globals() and 'manifest' not in globals()\n"
                    "class Secret: pass\n"
                    "refs = []\n"
                    "def fingerprint(secret):\n"
                    "    refs.append(weakref.ref(secret))\n"
                    "    local_secret = 'private-locals-marker'\n"
                    "    time.sleep(0.001)\n"
                    "    return None\n"
                    "def ignored(): return fingerprint(Secret())\n"
                    "def update(): return ignored()\n"
                    "def run():\n"
                    "    for _ in range(4): update()\n"
                    "run()\n"
                    "assert all(ref() is None for ref in refs)\n").encode()
                # No reads from the tracer, including a denied public boundary.
                def denied_open(*args, **kwargs):
                    raise AssertionError('observer opened a file during target execution')
                import builtins
                terminals = ('', 'raise SystemExit(23)', "raise ValueError('original-error')")
                for terminal in terminals:
                    with self.subTest(terminal=terminal):
                        source = raw + terminal.encode()
                        target.write_bytes(source)
                        output = io.StringIO()
                        tracer = d.PhaseObserver(str(target), output)
                        def previous(frame, event, arg): pass
                        sys.setprofile(previous)
                        with patch.object(builtins, 'open', denied_open), patch.object(Path, 'open', denied_open):
                            if terminal:
                                kind = SystemExit if 'SystemExit' in terminal else ValueError
                                with self.assertRaises(kind) as caught:
                                    d.observe_target(tracer, source, argv, {'manifest_sha256': 'a' * 64})
                                if kind is SystemExit: self.assertEqual(caught.exception.code, 23)
                                else: self.assertEqual(str(caught.exception), 'original-error')
                            else:
                                d.observe_target(tracer, source, argv, {'manifest_sha256': 'a' * 64})
                        self.assertIs(sys.getprofile(), previous)
                        sys.setprofile(saved[3])
                        self.assertIs(sys.argv, saved[0])
                        self.assertIs(sys.path, saved[1])
                        self.assertIs(sys.modules['__main__'], saved[2])
                        self.assertEqual(target.read_bytes(), source)
                        events = [json.loads(line) for line in output.getvalue().splitlines()]
                        self.assertEqual(events[0]['event'], 'SOURCE_BEGIN')
                        self.assertEqual(any(e['event'] == 'SOURCE_END' for e in events), not terminal)
                        phases = [e for e in events if e['event'] == 'PHASE']
                        self.assertEqual(len(phases), 18)
                        self.assertEqual({e['function'] for e in phases}, {'run', 'update', 'fingerprint'})
                        begun = {}
                        for event in phases:
                            self.assertEqual(event['filename'], str(target))
                            self.assertIs(type(event['frame_id']), int)
                            self.assertIs(type(event['line']), int)
                            if event['boundary'] == 'BEGIN': begun[event['frame_id']] = event['seconds']
                            else:
                                self.assertGreaterEqual(event['inclusive_seconds'], 0)
                                self.assertEqual(event['end_semantics'], 'return_or_unwind')
                                self.assertAlmostEqual(event['inclusive_seconds'],
                                    event['seconds'] - begun.pop(event['frame_id']))
                        self.assertFalse(begun)
                        self.assertFalse(tracer.active)
                        self.assertTrue(all(e['qualification_eligible'] is False and
                            e['state_reuse_eligible'] is False for e in events))
                        self.assertNotIn('private-locals-marker', output.getvalue())
                        self.assertNotIn('Secret', output.getvalue())
                        self.assertTrue(all(isinstance(k, int) and isinstance(v, float)
                            for k, v in tracer.active.items()))

                # A different filename with the same whitelist names stays untraced.
                outsider = compile('def update(): return None\nupdate()', str(base / 'copied.py'), 'exec')
                copied_raw = b"def run():\n    exec(__import__('sys').modules['phase_test_outsider'].code, {})\nrun()"
                with patch.dict(sys.modules, phase_test_outsider=SimpleNamespace(code=outsider)):
                    output = io.StringIO()
                    d.observe_target(d.PhaseObserver(str(target), output), copied_raw, argv, {})
                self.assertEqual({e['function'] for e in map(json.loads, output.getvalue().splitlines())
                    if e['event'] == 'PHASE'}, {'run'})

                # The same exception object crosses a nested traced unwind unchanged.
                for error in (SystemExit(23), ValueError('original-error')):
                    with patch.dict(sys.modules, phase_test_error=SimpleNamespace(error=error)):
                        output = io.StringIO()
                        with self.assertRaises(type(error)) as caught:
                            d.observe_target(d.PhaseObserver(str(target), output),
                                b'def fingerprint():\n raise __import__("sys").modules["phase_test_error"].error\n'
                                b'def run(): fingerprint()\nrun()', argv, {})
                    self.assertIs(caught.exception, error)
                    events = [json.loads(line) for line in output.getvalue().splitlines()]
                    self.assertEqual(events[0]['event'], 'SOURCE_BEGIN')
                    self.assertFalse(any(e['event'] == 'SOURCE_END' for e in events))

                # Exercise the real event limit and byte limit with an unchanged target.
                for event_limit, byte_limit in ((2000, 1024**2), (2000, 1500)):
                    output = io.StringIO()
                    tracer = d.PhaseObserver(str(target), output)
                    tracer.max_bytes = byte_limit
                    looping = b'def update(): pass\nfor _ in range(2200): update()\nprint("TARGET_FINISHED")'
                    with redirect_stdout(output): d.observe_target(tracer, looping, argv, {})
                    lines = output.getvalue().splitlines()
                    self.assertEqual(lines[-1], 'TARGET_FINISHED')
                    events = [json.loads(line) for line in lines[:-1]]
                    self.assertEqual(events[-1]['event'], 'TRUNCATED')
                    self.assertEqual(sum(e['event'] == 'TRUNCATED' for e in events), 1)
                    self.assertLessEqual(len(events), event_limit)
                    self.assertLessEqual(len(('\n'.join(lines[:-1]) + '\n').encode()), byte_limit)
                    self.assertFalse(tracer.active)

                class BrokenOutput:
                    def write(self, value): raise OSError('broken diagnostic stdout')
                    def flush(self): raise OSError('broken diagnostic stdout')
                for terminal in terminals:
                    tracer = d.PhaseObserver(str(target), BrokenOutput())
                    if terminal:
                        with self.assertRaises(SystemExit if 'SystemExit' in terminal else ValueError):
                            d.observe_target(tracer, raw + terminal.encode(), argv, {})
                    else: d.observe_target(tracer, raw, argv, {})
                    self.assertIs(sys.getprofile(), saved[3])

                target.write_bytes(raw)
                pins = dict(TRAINER=binding(target), EXECUTION=binding(execution), AUTHORITY=binding(authority),
                    PYTHON=binding(Path(sys.executable).resolve()), PYTHON_VERSION=sys.version)
                manifest = dict(schema=d.SCHEMA, wrapper=binding(PATH), python=pins['PYTHON'], trainer=pins['TRAINER'],
                    execution=pins['EXECUTION'], authority=pins['AUTHORITY'], argv=argv, output=argv[14],
                    unit='standin-observer-unit', stdio=dict(stdout=str(base / 'unit.log'), stderr=str(base / 'unit.log')),
                    qualification_eligible=False, state_reuse_eligible=False)
                manifest_path = base / 'manifest.json'
                real_read_text = Path.read_text
                def metadata(path, *args, **kwargs):
                    if str(path) == '/proc/self/cgroup': return '0::/standin-observer-unit.service\n'
                    return real_read_text(path, *args, **kwargs)
                def invoke(value, bad_hash=False, duplicate=False):
                    text = json.dumps(value)
                    if duplicate: text = '{"schema":"duplicate",' + text[1:]
                    manifest_path.write_text(text)
                    sha = '0' * 64 if bad_hash else binding(manifest_path)['sha256']
                    with patch.multiple(d, **pins), patch.object(Path, 'read_text', metadata), \
                         patch.dict(os.environ, INVOCATION_ID='b' * 32), \
                         patch.object(sys, 'argv', [str(PATH), '--manifest', str(manifest_path), '--manifest-sha256', sha]):
                        output = io.StringIO()
                        with redirect_stdout(output): d.main()
                    return [json.loads(line) for line in output.getvalue().splitlines()]
                events = invoke(manifest)
                self.assertEqual(events[0]['unit'], manifest['unit'])
                self.assertEqual(events[0]['invocation_id'], 'b' * 32)
                self.assertEqual(events[-1]['event'], 'SOURCE_END')
                cases = ('manifest-hash', 'wrapper-hash', 'current-file', 'module-origin', 'builtin-origin',
                    'builtin-function', 'trainer-hash', 'authority-hash', 'execution-hash', 'python-hash', 'version',
                    'argv-order', 'argv-output', 'output-existing', 'unit', 'qualification', 'reuse', 'duplicate',
                    'stdio-collision', 'profile-existing', 'native-import', 'unknown-key', 'symlink',
                    'trainer-bytes', 'authority-bytes', 'execution-bytes')
                for case in cases:
                    with self.subTest(case=case):
                        value = copy.deepcopy(manifest)
                        options, patches, changed = {}, [], None
                        if case == 'manifest-hash': options['bad_hash'] = True
                        elif case.endswith('-hash'): value[case.removesuffix('-hash')]['sha256'] = '0' * 64
                        elif case == 'current-file': patches.append(patch.object(d, '__file__', str(base / 'wrong.py')))
                        elif case == 'module-origin': patches.append(patch.object(d.__spec__, 'origin', str(base / 'wrong.py')))
                        elif case == 'builtin-origin': patches.append(patch.object(time.__spec__, 'origin', 'untrusted-time.so'))
                        elif case == 'builtin-function': patches.append(patch.object(sys, 'setprofile', lambda callback: None))
                        elif case == 'version': patches.append(patch.dict(pins, PYTHON_VERSION='wrong-version'))
                        elif case == 'argv-order': value['argv'][7:11] = ['--arm', 'control', '--phase', 'cpu']
                        elif case == 'argv-output': value['output'] = str(base / 'different')
                        elif case == 'output-existing': Path(value['output']).mkdir()
                        elif case == 'unit': value['unit'] = 'wrong-unit'
                        elif case == 'qualification': value['qualification_eligible'] = True
                        elif case == 'reuse': value['state_reuse_eligible'] = True
                        elif case == 'duplicate': options['duplicate'] = True
                        elif case == 'stdio-collision': value['stdio']['stdout'] = str(target)
                        elif case == 'profile-existing':
                            patches.append(patch.object(sys, 'getprofile', lambda: object()))
                        elif case == 'native-import': patches.append(patch.dict(sys.modules, torch=SimpleNamespace()))
                        elif case == 'unknown-key': value['invocation_id'] = 'b' * 32
                        elif case == 'symlink':
                            link = base / 'wrapper-link.py'
                            link.symlink_to(PATH)
                            value['wrapper']['path'] = str(link)
                        elif case.endswith('-bytes'):
                            changed = Path(value[case.removesuffix('-bytes')]['path'])
                            before = changed.read_bytes()
                            changed.write_bytes(before + b'changed')
                        with ExitStack() as stack:
                            for item in patches: stack.enter_context(item)
                            with self.assertRaises(ValueError): invoke(value, **options)
                        if case == 'output-existing': Path(value['output']).rmdir()
                        if changed is not None: changed.write_bytes(before)
                self.assertIs(sys.getprofile(), saved[3])
                self.assertIs(sys.argv, saved[0])
                self.assertIs(sys.path, saved[1])
                self.assertIs(sys.modules['__main__'], saved[2])
        finally:
            sys.setprofile(saved[3])
            sys.argv, sys.path, sys.modules['__main__'] = saved[:3]


if __name__ == '__main__':
    if not __debug__ or not sys.dont_write_bytecode:
        raise SystemExit('source-only check requires python -B without -O')
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ObserverTest))
    seconds = time.perf_counter() - started
    native = any(name.split('.')[0] in NATIVE for name in sys.modules)
    passed = result.wasSuccessful() and seconds <= 120 and not native
    print(json.dumps({'schema': 'connected-mlp-phase-observer-source-check-v1', 'tests': result.testsRun,
        'seconds': seconds, 'pass': passed, 'native_imported': native}), flush=True)
    raise SystemExit(0 if passed else 1)
