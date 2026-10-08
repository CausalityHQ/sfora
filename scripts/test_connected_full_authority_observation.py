#!/usr/bin/env python3
"""Stdlib falsifier; never execute the frozen evaluator or its native dependencies."""
import ast
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import weakref

PATH = Path(__file__).resolve().with_name('observe_connected_full_authority.py')
FREEZE = PATH.parent.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v2-freeze'


def binding(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def driver():
    spec = importlib.util.spec_from_file_location('_authority_observer_test', PATH)
    module = importlib.util.module_from_spec(spec)
    # Compile the real production control path, not a rewritten authority adapter.
    exec(compile(PATH.read_bytes(), str(PATH), 'exec'), vars(module))
    return module


class ObservationTest(unittest.TestCase):
    def test_one_call_original_identity_and_cleanup(self):
        self.assertTrue(PATH.exists(), 'authority-only observer is missing')
        d = driver()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'original.py'
            output = Path(folder) / 'never-created'
            argv = [str(target), '--output', str(output)]
            raw = ("import argparse, sys, time, weakref\n"
                   "from pathlib import Path\n"
                   "calls = []\n"
                   "class Secret: pass\n"
                   "refs = []\n"
                   "def parser():\n"
                   "    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path); return p\n"
                   "def cli(args): return [__file__, '--output', str(args.output)]\n"
                   "def native_start(*args): raise AssertionError('native_start reached')\n"
                   "def run(*args): raise AssertionError('run reached')\n"
                   "def authority(args):\n"
                   "    calls.append(1); assert len(calls) == 1\n"
                   "    assert sys.argv == cli(args)\n"
                   "    assert __file__ == authority.__code__.co_filename == __spec__.origin\n"
                   "    assert __loader__.path == __file__\n"
                   "    assert sys.modules[__name__].__dict__ is globals()\n"
                   "    assert 'observer' not in globals() and 'manifest' not in globals()\n"
                   "    before = dict(globals())\n"
                   "    secret = Secret(); refs.append(weakref.ref(secret))\n"
                   "    time.sleep(0.30)\n"
                   "    assert globals().keys() == before.keys()\n"
                   "    assert all(globals()[k] is v for k, v in before.items())\n"
                   "    assert not args.output.exists()\n"
                   "    return {'guards': {'a': 'b'}, 'training_context': {'legacy': {'invocations': {1,2}}}, 'secret': secret}, None\n"
                   "if __name__ == '__main__': run()\n").encode()
            target.write_bytes(raw)
            saved = sys.argv, sys.path, list(sys.meta_path), set(sys.modules)
            stream = io.BytesIO()
            observer = d.Sampler(str(target), stream)
            # Keep only weak references to the returned context's secret and module.
            refs = []
            original_module = importlib.util.module_from_spec
            def capture(spec):
                module = original_module(spec)
                refs.append(weakref.ref(module))
                return module
            with patch.object(importlib.util, 'module_from_spec', capture):
                self.assertIsNone(d.observe(raw, argv, observer))
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            returned = next(row for row in rows if row['event'] == 'AUTHORITY_RETURN')
            self.assertEqual((returned['guard_count'], returned['invocation_count']), (1, 2))
            samples = [row for row in rows if row['event'] == 'SAMPLE']
            self.assertTrue(samples)
            self.assertTrue(any(frame['function'] == 'authority' for row in samples for frame in row['frames']))
            self.assertEqual(rows[-1]['event'], 'END')
            self.assertFalse(output.exists())
            self.assertIs(sys.argv, saved[0]); self.assertIs(sys.path, saved[1])
            self.assertEqual(sys.meta_path, saved[2])
            self.assertFalse(any(name.split('.')[0] in d.NATIVE for name in set(sys.modules) - saved[3]))
            self.assertNotIn(d.MODULE, sys.modules)
            self.assertFalse(observer.thread.is_alive())
            self.assertFalse(any('secret' in row for row in rows))
            import gc
            gc.collect()
            self.assertTrue(all(ref() is None for ref in refs))

    def test_pins_match_original_command_and_source_without_executing_it(self):
        d = driver()
        freeze = json.loads((FREEZE / 'full-cpu-v2-freeze.json').read_bytes())
        for fact in (d.EVALUATOR, d.EVALUATOR_TEST, d.EXECUTION, d.AUTHORITY, d.COMMAND):
            path = FREEZE / Path(fact['path']).name
            self.assertEqual(binding(path)['sha256'], fact['sha256'])
            self.assertEqual(freeze['files'][path.name], fact['sha256'])
        import shlex
        command = (FREEZE / 'full-cpu-v2-command.sh').read_text()
        line = next(line for line in command.splitlines() if line.startswith('/home/riomus/group-learning/.venv/bin/python -B '))
        self.assertEqual(shlex.split(line)[2:], d.original_argv())
        self.assertIn(d.PYTHON['sha256'] + '  ' + d.PYTHON['path'], command)
        original = ast.parse((FREEZE / 'evaluate_siglip2_connected_mlp.py').read_bytes())
        native = next(node for node in original.body if isinstance(node, ast.Assign) and
                      any(isinstance(target, ast.Name) and target.id == 'NATIVE' for target in node.targets))
        self.assertEqual(ast.literal_eval(native.value), d.NATIVE)
        # Compile actual frozen authority bytecode without executing even its imports.
        compile(original, str(FREEZE / 'evaluate_siglip2_connected_mlp.py'), 'exec')

    def test_v6_pin_update_has_exact_v2_observer_inverse(self):
        original = (FREEZE.parent / 'connected-full-authority-observation-v2-freeze' / PATH.name).read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),
                         '97c653e45519549fd023909a41a88c61d7f80684ca2cd4ad73c0dd4b9813b9a0')
        inverse = PATH.read_bytes()
        for new, old in (
            (b'ONE genuine frozen v6 full-CPU', b'ONE genuine frozen v5 full-CPU'),
            (b'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/',
             b'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5/'),
            (b'2390c60fe5e87e82ab122c5c0101476b378792541bc454470c37d6c7f0410d40',
             b'919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69'),
            (b'28d0297cebd6a7bab395a44bc60a4aa1bcaacb9160a8093044a54fa270b104d9',
             b'df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb'),
            (b'a0c1f27db4da404e7777d89518dfc83e2e20b607fcc1031776978e9ca8ec9f3c',
             b'a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a'),
            (b'authority-full-cpu-v2.json', b'authority-full-cpu-v1.json'),
            (b'3444df504430a2ee92f33fac04cfed609c512de542d8993f08069dd4bd24e19d',
             b'd8f3b0f92a02937f63539782f2653a20c4d7fbc320022c1554ee261f718fe71f'),
            (b'full-cpu-v2-command.sh', b'full-cpu-v1-command.sh'),
            (b'8998b7abb9ad2cbbb119aff15b722578845b39d42ae179c8a183c3000fa0e1f7',
             b'911428b835ed6323b2c58c927db996121e10b4845a0c306f3c815b2a012998f0'),
            (b'/home/riomus/runs/sfora-connected-mlp-evaluation-full-cpu-v2',
             b'/home/riomus/runs/sfora-connected-mlp-evaluation-full-cpu-v1'),
            (b'_connected_full_authority_v6', b'_connected_full_authority_v5'),
        ):
            self.assertEqual(inverse.count(new), 1, new)
            inverse = inverse.replace(new, old, 1)
        self.assertEqual(inverse, original)
        self.assertEqual(ast.dump(ast.parse(inverse)), ast.dump(ast.parse(original)))

    def test_current_evaluator_compile_only_metadata_cli_and_exact_two_pins(self):
        d = driver()
        raw = (FREEZE / 'evaluate_siglip2_connected_mlp.py').read_bytes()
        self.assertEqual(raw, PATH.with_name('evaluate_siglip2_connected_mlp.py').read_bytes())
        self.assertEqual(json.loads((FREEZE / 'execution.json').read_bytes()),
                         {Path(p['path']).name: p['sha256'] for p in (d.EVALUATOR, d.EVALUATOR_TEST)})
        freeze = json.loads((FREEZE / 'full-cpu-v2-freeze.json').read_bytes())
        self.assertEqual(d.ROOT, freeze['source_root'] + '/')
        self.assertEqual(d.OLD_OUTPUT, freeze['output'])
        self.assertEqual(d.MODULE, '_connected_full_authority_v6')
        self.assertEqual(json.loads((FREEZE / 'authority-full-cpu-v2.json').read_bytes())['execution_sha256'],
                         d.EXECUTION['sha256'])
        spec = importlib.util.spec_from_file_location(d.MODULE, d.EVALUATOR['path'])
        self.assertEqual(spec.name, d.MODULE)
        self.assertTrue(spec.has_location)
        self.assertEqual(spec.origin, d.EVALUATOR['path'])
        self.assertEqual(spec.loader.path, d.EVALUATOR['path'])
        # Inspect genuine code objects; never execute evaluator imports or authority.
        compiled = compile(raw, spec.origin, 'exec', dont_inherit=True)
        functions = {code.co_name: code for code in compiled.co_consts if isinstance(code, type(compiled))}
        for name in ('parser', 'cli', 'authority'):
            self.assertEqual(functions[name].co_filename, spec.origin)
        previous = (FREEZE.parent / 'connected-mlp-evaluation-full-cpu-v1-freeze' /
                    'evaluate_siglip2_connected_mlp.py').read_bytes()
        for name in ('parser', 'cli'):
            current_node = next(node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name == name)
            previous_node = next(node for node in ast.parse(previous).body if isinstance(node, ast.FunctionDef) and node.name == name)
            self.assertEqual(ast.dump(current_node), ast.dump(previous_node))
        self.assertNotIn(d.MODULE, sys.modules)
        d.no_native()

    def test_original_exception_survives_sampler_and_cleanup_failures(self):
        d = driver()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'target.py'
            argv = [str(target), '--output', str(Path(folder) / 'absent')]
            raw = ("import argparse, time\n"
                   "def parser():\n"
                   "    p=argparse.ArgumentParser(); p.add_argument('--output'); return p\n"
                   "def cli(args): return [__file__, '--output', args.output]\n"
                   "def authority(args):\n"
                   "    time.sleep(.03)\n"
                   "    try: raise KeyError('root-original')\n"
                   "    except KeyError as root: raise ValueError('outer-original') from root\n").encode()
            target.write_bytes(raw)
            saved = sys.argv, sys.path, list(sys.meta_path)
            for failure in ('none', 'frames', 'write', 'join'):
                with self.subTest(failure=failure), ExitStack() as stack:
                    stream = io.BytesIO()
                    sampler = d.Sampler(str(target), stream)
                    stack.enter_context(patch.object(d, 'INTERVAL', .001))
                    if failure == 'frames':
                        stack.enter_context(patch.object(sys, '_current_frames', side_effect=OSError('frame-failure')))
                    if failure == 'write':
                        stack.enter_context(patch.object(stream, 'write', side_effect=[100000, OSError('write-failure')]))
                        # START must succeed; subsequent writes fail after the real authority error.
                        real_emit = sampler.emit
                        def emit(event, **fields):
                            if event == 'START': return
                            return real_emit(event, **fields)
                        stack.enter_context(patch.object(sampler, 'emit', emit))
                    if failure == 'join':
                        original_finish = sampler.finish
                        def finish():
                            original_finish()
                            raise RuntimeError('join-check-failure')
                        stack.enter_context(patch.object(sampler, 'finish', finish))
                    with self.assertRaisesRegex(ValueError, 'outer-original') as caught:
                        d.observe(raw, argv, sampler)
                    error = caught.exception
                    self.assertIsInstance(error.__cause__, KeyError)
                    self.assertEqual(error.__cause__.args, ('root-original',))
                    if failure != 'none':
                        self.assertTrue(error.__notes__)
                    if failure != 'write':
                        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
                        chain = next(row for row in rows if row['event'] == 'AUTHORITY_ERROR')['exception']['chain']
                        self.assertEqual([item['type'] for item in chain], ['builtins.ValueError', 'builtins.KeyError'])
                        self.assertNotIn('END', [row['event'] for row in rows])
                    self.assertFalse(sampler.thread.is_alive())
                    self.assertIs(sys.argv, saved[0]); self.assertIs(sys.path, saved[1])
                    self.assertEqual(sys.meta_path, saved[2])
                    self.assertNotIn(d.MODULE, sys.modules)

    def test_native_block_sample_limits_and_fail_closed(self):
        d = driver()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'target.py'
            output = Path(folder) / 'absent'
            argv = [str(target), '--output', str(output)]
            prefix = ("import argparse, time\n"
                      "def parser():\n"
                      "    p=argparse.ArgumentParser(); p.add_argument('--output'); return p\n"
                      "def cli(args): return [__file__, '--output', args.output]\n"
                      "def native_start(*args): raise AssertionError('native_start reached')\n"
                      "def run(*args): raise AssertionError('run reached')\n")
            for root in sorted(d.NATIVE):
                (Path(folder)/(root+'.py')).write_text("raise AssertionError('native code executed')\n")
                raw = (prefix + f'def authority(args):\n    import {root}\n    native_start()\n').encode()
                with self.subTest(root=root), self.assertRaisesRegex(ImportError, 'native import forbidden'):
                    d.observe(raw, argv, d.Sampler(str(target), io.BytesIO()))
                self.assertNotIn(root, sys.modules)
            raw = (prefix + "def authority(args):\n    time.sleep(.03)\n    return {'guards':{}, 'training_context':{'legacy':{'invocations':set()}}}, None\n").encode()
            for failure in ('samples', 'bytes', 'frame-error'):
                with self.subTest(failure=failure), ExitStack() as stack:
                    stream = io.BytesIO()
                    sampler = d.Sampler(str(target), stream)
                    stack.enter_context(patch.object(d, 'INTERVAL', .001))
                    if failure == 'samples': stack.enter_context(patch.object(d, 'MAX_SAMPLES', 2))
                    if failure == 'bytes': stack.enter_context(patch.object(d, 'MAX_BYTES', 66000))
                    if failure == 'frame-error':
                        stack.enter_context(patch.object(sys, '_current_frames', side_effect=OSError('frame-failure')))
                    with self.assertRaisesRegex(RuntimeError, 'Observation failed'):
                        d.observe(raw, argv, sampler)
                    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
                    if failure == 'samples': self.assertEqual(sampler.count, 2)
                    self.assertLessEqual(len(stream.getvalue()), d.MAX_BYTES)
                    self.assertNotIn('END', [row['event'] for row in rows])
                    self.assertFalse(sampler.thread.is_alive())
            self.assertFalse(output.exists())

    def test_native_metadata_lookup_preserves_origin_without_execution(self):
        d = driver()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'target.py'
            native = Path(folder)/'torch.py'
            native.write_text("raise AssertionError('native code executed')\n")
            argv = [str(target),'--output',str(Path(folder)/'absent')]
            raw = ("import argparse, importlib.util, sys\n"
                   "def parser():\n"
                   "    p=argparse.ArgumentParser(); p.add_argument('--output'); return p\n"
                   "def cli(args): return [__file__, '--output', args.output]\n"
                   "def authority(args):\n"
                   "    spec=importlib.util.find_spec('torch')\n"
                   f"    assert spec.origin == {str(native)!r}\n"
                   "    assert 'torch' not in sys.modules\n"
                   "    try: import torch\n"
                   "    except ImportError as e: assert 'native import forbidden' in str(e)\n"
                   "    else: raise AssertionError('native execution allowed')\n"
                   "    assert 'torch' not in sys.modules\n"
                   "    return {'guards':{},'training_context':{'legacy':{'invocations':set()}}},None\n").encode()
            target.write_bytes(raw)
            stream=io.BytesIO()
            d.observe(raw,argv,d.Sampler(str(target),stream))
            self.assertEqual(json.loads(stream.getvalue().splitlines()[-1])['event'],'END')
            self.assertNotIn('torch',sys.modules)

    def test_authenticated_main_and_rejections(self):
        d = driver()
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            root = base / 'source'
            root.mkdir()
            target = root / 'evaluator.py'
            original_test = root / 'test.py'
            original_test.write_text('# synthetic test\n')
            execution = root / 'execution.json'
            authority = root / 'authority.json'
            command = root / 'command.sh'
            command.write_text('# never executed\n')
            output = base / 'absent'
            diagnostic = base / 'diag.jsonl'
            old_output = base / 'historical-output'
            manifest_path = base / 'manifest.json'
            # Actual main/prepare/observe run. Only external pins and cgroup identity
            # are replaced with temporary stdlib evidence, not acceptance functions.
            target.write_text("import argparse\n"
                "from pathlib import Path\n"
                "def parser():\n"
                "    p=argparse.ArgumentParser()\n"
                "    for k in ('execution-sha256','authority','authority-sha256','phase','output'): p.add_argument('--'+k)\n"
                "    return p\n"
                "def cli(a): return [__file__,'--execution-sha256',a.execution_sha256,'--authority',a.authority,'--authority-sha256',a.authority_sha256,'--phase',a.phase,'--output',a.output]\n"
                "def authority(a):\n"
                "    assert a.phase == 'cpu' and not Path(a.output).exists()\n"
                "    return {'guards': {'a':1}, 'training_context': {'legacy': {'invocations':set()}}}, None\n"
                "def run(*a): raise AssertionError('run called')\n"
                "def native_start(*a): raise AssertionError('native_start called')\n"
                "if __name__ == '__main__': run()\n")
            execution.write_text(json.dumps({target.name: binding(target)['sha256'], original_test.name: binding(original_test)['sha256']}))
            authority.write_text(json.dumps({'phase':'cpu', 'stage':'full', 'panel':'selection', **{k:{'root':str(base / k)} for k in
                ('training', 'evaluator_reference', 'nearest_evaluator', 'genuine_evaluator', 'reference')}}))
            pins = dict(EVALUATOR=binding(target), EVALUATOR_TEST=binding(original_test), EXECUTION=binding(execution),
                        AUTHORITY=binding(authority), COMMAND=binding(command), PYTHON=binding(Path(sys.executable).resolve()),
                        PYTHON_VERSION=sys.version, OLD_OUTPUT=str(old_output))
            argv = [str(target), '--execution-sha256', pins['EXECUTION']['sha256'], '--authority', str(authority),
                    '--authority-sha256', pins['AUTHORITY']['sha256'], '--phase', 'cpu', '--output', str(output)]
            manifest = dict(schema='connected-full-authority-observation-v1', observer=binding(PATH),
                test=binding(Path(__file__).resolve()), python=pins['PYTHON'], evaluator=pins['EVALUATOR'],
                execution=pins['EXECUTION'], authority=pins['AUTHORITY'], command=pins['COMMAND'],
                original_argv=argv[:-1]+[str(old_output)], argv=argv, output=str(output), diagnostic=str(diagnostic),
                unit='synthetic-observation', resource_policy={'seconds':900,'host_bytes':8589934592,'swap_bytes':0,'cuda_visible_devices':''},
                both_locks_held=True, qualification_eligible=False, state_reuse_eligible=False)
            real_read_text = Path.read_text
            def metadata(path, *args, **kwargs):
                if str(path) == '/proc/self/cgroup': return '0::/synthetic-observation.service\n'
                return real_read_text(path, *args, **kwargs)
            def invoke(value, *, bad_hash=False, duplicate=False):
                text = json.dumps(value)
                if duplicate: text = '{"schema":"duplicate",' + text[1:]
                manifest_path.write_text(text)
                digest = '0'*64 if bad_hash else binding(manifest_path)['sha256']
                with patch.multiple(d, **pins), patch.object(Path, 'read_text', metadata), \
                     patch.dict(os.environ, CUDA_VISIBLE_DEVICES='', INVOCATION_ID='b'*32), \
                     patch.object(sys, 'argv', [str(PATH), '--manifest', str(manifest_path), '--manifest-sha256', digest]):
                    d.main()
            invoke(manifest)
            rows = [json.loads(line) for line in diagnostic.read_bytes().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['START', 'AUTHORITY_RETURN', 'END'])
            for row in rows:
                self.assertIs(row['qualification_eligible'], False)
                self.assertIs(row['state_reuse_eligible'], False)
            preserved = diagnostic.read_bytes()
            with self.assertRaises(ValueError): invoke(manifest)
            self.assertEqual(diagnostic.read_bytes(), preserved)
            diagnostic.unlink()
            cases = ('manifest-hash', 'observer-hash', 'test-hash', 'python-hash', 'evaluator-hash', 'execution-hash',
                     'authority-hash', 'command-hash', 'version', 'interpreter', 'duplicate', 'unknown', 'argv-order',
                     'argv-output', 'argv-phase', 'original-argv', 'output-existing', 'output-symlink', 'output-source',
                     'diagnostic-overlap', 'diagnostic-symlink', 'diagnostic-source', 'unit', 'native-preloaded',
                     'qualify', 'reuse', 'locks', 'policy', 'trace', 'profile', 'origin', 'evaluator-bytes',
                     'evaluator-test-bytes', 'execution-bytes', 'authority-bytes', 'command-bytes')
            for case in cases:
                with self.subTest(case=case), ExitStack() as stack:
                    value, options = copy.deepcopy(manifest), {}
                    changed = None
                    if case == 'manifest-hash': options['bad_hash'] = True
                    elif case.endswith('-hash'): value[case.removesuffix('-hash')]['sha256'] = '0'*64
                    elif case == 'version': stack.enter_context(patch.dict(pins, PYTHON_VERSION='wrong'))
                    elif case == 'interpreter': stack.enter_context(patch.object(sys, 'executable', str(target)))
                    elif case == 'duplicate': options['duplicate'] = True
                    elif case == 'unknown': value['extra'] = True
                    elif case == 'argv-order': value['argv'][1:3], value['argv'][3:5] = value['argv'][3:5], value['argv'][1:3]
                    elif case == 'argv-output': value['argv'][-1] = str(base / 'other')
                    elif case == 'argv-phase': value['argv'][8] = 'export'
                    elif case == 'original-argv': value['original_argv'][-1] = str(output)
                    elif case == 'output-existing': output.mkdir(); stack.callback(output.rmdir)
                    elif case == 'output-symlink': output.symlink_to(base/'missing'); stack.callback(output.unlink)
                    elif case == 'output-source': value['output'] = value['argv'][-1] = str(root/'absent')
                    elif case == 'diagnostic-overlap': value['diagnostic'] = str(output/'log')
                    elif case == 'diagnostic-symlink': diagnostic.symlink_to(base/'missing'); stack.callback(diagnostic.unlink)
                    elif case == 'diagnostic-source': value['diagnostic'] = str(root/'log')
                    elif case == 'unit': value['unit'] = 'wrong-unit'
                    elif case == 'native-preloaded': stack.enter_context(patch.dict(sys.modules, numpy=object()))
                    elif case == 'qualify': value['qualification_eligible'] = True
                    elif case == 'reuse': value['state_reuse_eligible'] = True
                    elif case == 'locks': value['both_locks_held'] = False
                    elif case == 'policy': value['resource_policy']['seconds'] = 901
                    elif case == 'trace': stack.enter_context(patch.object(sys, 'gettrace', lambda: object()))
                    elif case == 'profile': stack.enter_context(patch.object(sys, 'getprofile', lambda: object()))
                    elif case == 'origin': stack.enter_context(patch.object(d.__spec__, 'origin', str(target)))
                    elif case.endswith('-bytes'):
                        changed = original_test if case == 'evaluator-test-bytes' else Path(value[case.removesuffix('-bytes')]['path'])
                        previous = changed.read_bytes()
                        changed.write_bytes(previous + b'changed')
                        stack.callback(changed.write_bytes, previous)
                    with self.assertRaises((ValueError, FileExistsError)): invoke(value, **options)
                    self.assertFalse(diagnostic.exists())
            self.assertFalse(output.exists())


if __name__ == '__main__':
    if not __debug__ or not sys.dont_write_bytecode:
        raise SystemExit('source-only check requires unoptimized python -B')
    unittest.main(verbosity=2)
