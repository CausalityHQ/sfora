#!/usr/bin/env python3
"""Fast source-only profiler falsifier; never import cProfile/_lsprof/native packages."""
import ast
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import marshal
import os
from pathlib import Path
import struct
import sys
import sysconfig
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().with_name('profile_once.py')
STDLIB_PROFILE = Path(sysconfig.get_path('stdlib')) / 'cProfile.py'
TREE = ast.parse(STDLIB_PROFILE.read_bytes())
PROFILE_CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'Profile')
RUNCTX = next(n for n in PROFILE_CLASS.body if isinstance(n, ast.FunctionDef) and n.name == 'runctx')
SCOPE = {}
exec(compile(ast.Module(body=[RUNCTX], type_ignores=[]), str(STDLIB_PROFILE), 'exec'), SCOPE)


class StandinProfile:
    # This is the actual stdlib runctx body, with enable/disable/dump stand-ins.
    runctx = SCOPE['runctx']

    def __init__(self, fail_dump=False):
        self.enabled = self.disabled = False
        self.dumps = 0
        self.fail_dump = fail_dump

    def enable(self): self.enabled = True
    def disable(self): self.disabled = True

    def dump_stats(self, path):
        self.dumps += 1
        if self.fail_dump:
            raise OSError('standin dump failure')
        Path(path).write_bytes(b'standin stats')


class ProfileOnceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.is_file():
            raise AssertionError('profile_once runner is missing')
        spec = importlib.util.spec_from_file_location('profile_once_contract', PATH)
        cls.runner = importlib.util.module_from_spec(spec)
        exec(compile(PATH.read_bytes(), str(PATH), 'exec'), vars(cls.runner))

    def test_canonical_main_argv_file_path_and_exception_cleanup(self):
        d = self.runner
        saved = (sys.argv, sys.path, sys.modules['__main__'])
        for terminal in ('', 'raise SystemExit(23)', "raise ValueError('target failure')"):
            with self.subTest(terminal=terminal), tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / 'trainer.py'
                argv = [str(target), '--phase', 'mechanics', '--seed', '179061']
                raw = (f"import sys\nassert sys.argv == {argv!r}\n"
                       "assert __name__ == '__main__'\n"
                       f"assert __file__ == {str(target)!r}\n"
                       f"assert sys.path[0] == {directory!r}\n"
                       "assert __spec__ is None and __package__ is None\n"
                       "assert sys.modules['__main__'].__dict__ is globals()\n"
                       "assert __loader__.path == __file__\n" + terminal).encode()
                target.write_bytes(raw)
                stats = Path(directory) / 'new.pstats'
                profiler = StandinProfile()
                if terminal:
                    kind = SystemExit if 'SystemExit' in terminal else ValueError
                    with self.assertRaises(kind) as caught:
                        d.profile_target(profiler, raw, argv, stats)
                    if kind is SystemExit:
                        self.assertEqual(caught.exception.code, 23)
                    else:
                        self.assertEqual(str(caught.exception), 'target failure')
                else:
                    d.profile_target(profiler, raw, argv, stats)
                self.assertEqual(stats.read_bytes(), b'standin stats')
                self.assertTrue(profiler.enabled and profiler.disabled)
                self.assertEqual(profiler.dumps, 1)
                self.assertEqual(target.read_bytes(), raw)
                self.assertIs(sys.argv, saved[0])
                self.assertIs(sys.path, saved[1])
                self.assertIs(sys.modules['__main__'], saved[2])

    def test_dump_failure_preserves_target_error_and_never_overwrites(self):
        d = self.runner
        with tempfile.TemporaryDirectory() as directory:
            target, stats = Path(directory) / 'trainer.py', Path(directory) / 'stats'
            for raw, kind in ((b'raise SystemExit(23)', SystemExit),
                              (b"raise ValueError('original')", ValueError), (b'pass', OSError)):
                with self.subTest(raw=raw), self.assertRaises(kind) as caught:
                    d.profile_target(StandinProfile(fail_dump=True), raw, [str(target)], stats)
                if kind is SystemExit:
                    self.assertEqual(caught.exception.code, 23)
                if kind is not OSError:
                    self.assertIn('standin dump failure', caught.exception.__notes__[0])
                self.assertFalse(stats.exists())
            stats.write_bytes(b'existing artifact')
            with self.assertRaises(ValueError):
                d.profile_target(StandinProfile(), b"raise AssertionError('target ran')", [str(target)], stats)
            self.assertEqual(stats.read_bytes(), b'existing artifact')
            stats.unlink()

            class RacingProfile(StandinProfile):
                def dump_stats(self, path):
                    super().dump_stats(path)
                    stats.write_bytes(b'concurrent artifact')

            with self.assertRaises(SystemExit) as caught:
                d.profile_target(RacingProfile(), b'raise SystemExit(23)', [str(target)], stats)
            self.assertEqual(caught.exception.code, 23)
            self.assertEqual(stats.read_bytes(), b'concurrent artifact')
            self.assertTrue(caught.exception.__notes__)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()), ['stats'])

    def test_cleanup_annotation_cannot_replace_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = b"class Original(ValueError):\n def add_note(self, note): raise RuntimeError('annotation failed')\nraise Original('original')"
            with self.assertRaises(ValueError) as caught:
                self.runner.profile_target(StandinProfile(fail_dump=True), raw,
                    [str(Path(directory) / 'trainer.py')], Path(directory) / 'stats')
            self.assertEqual(str(caught.exception), 'original')

    def fixture(self, directory):
        base = Path(directory)
        original, diagnostic = base / 'original', base / 'diagnostic'
        original.mkdir(); diagnostic.mkdir()
        trainer, authority = original / 'trainer.py', diagnostic / 'authority.json'
        authority.write_bytes(b'{"selected_mechanics":null}')
        bind = lambda path: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        authority_sha = bind(authority)['sha256']
        argv = [str(trainer), '--execution-sha256', 'e' * 64, '--authority', str(authority),
                '--authority-sha256', authority_sha, '--phase', 'mechanics', '--arm', 'control',
                '--seed', '179061', '--output', str(base / 'new-output')]
        marker = base / 'target-entered'
        trainer.write_text(f"import sys\nfrom pathlib import Path\nassert sys.argv == {argv!r}\n"
                           f"assert __file__ == {str(trainer)!r}\nassert sys.path[0] == {str(original)!r}\n"
                           "assert __name__ == '__main__' and sys.modules['__main__'].__dict__ is globals()\n"
                           f"Path({str(marker)!r}).write_bytes(b'entered')\n")
        pure = diagnostic / 'profile.py'
        pure.write_bytes(b"token = 'authenticated-source'\n")
        profiler = diagnostic / 'cProfile.py'
        runctx = '\n'.join('    ' + line for line in ast.unparse(RUNCTX).splitlines())
        profiler.write_text("import profile as _pyprofile\nfrom pathlib import Path\n"
                            "assert _pyprofile.token == 'authenticated-source'\nclass Profile:\n"
                            "    def enable(self): self.enabled = True\n"
                            "    def disable(self): self.disabled = True\n" + runctx + '\n'
                            "    def dump_stats(self, path):\n"
                            "        assert self.enabled and self.disabled\n"
                            "        Path(path).write_bytes(b'standin stats')\n")
        # Valid timestamp pycs deliberately disagree with authenticated source.
        for source in (trainer, pure, profiler):
            cache = Path(importlib.util.cache_from_source(str(source)))
            cache.parent.mkdir(exist_ok=True)
            poison = compile("raise AssertionError('cached bytecode executed')", str(source), 'exec')
            cache.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack('<III', 0,
                              int(source.stat().st_mtime), source.stat().st_size) + marshal.dumps(poison))
        pins = dict(PYTHON=bind(Path(sys.executable).resolve()), PYTHON_VERSION=sys.version,
                    TRAINER=bind(trainer), STDLIB={'profile.py': bind(pure), 'cProfile.py': bind(profiler)},
                    AUTHORITY_SHA=authority_sha, EXECUTION_SHA='e' * 64)
        manifest = dict(schema='quadratic-profile-once-v1', wrapper=bind(PATH), python=pins['PYTHON'],
                        stdlib=pins['STDLIB'], trainer=pins['TRAINER'], argv=argv,
                        profile=str(diagnostic / 'new.pstats'), unit='standin-profile-unit',
                        stdio=dict(stdout=str(diagnostic / 'unit.log'), stderr=str(diagnostic / 'unit.log')),
                        qualification_eligible=False, state_reuse_eligible=False)
        return pins, manifest, marker

    def invoke(self, directory, pins, manifest, mappings='', builtin=True, origin='built-in', duplicate=False,
               bad_manifest_hash=False):
        d = self.runner
        path = Path(directory) / 'manifest.json'
        raw = json.dumps(manifest)
        if duplicate:
            raw = '{"schema":"quadratic-profile-once-v1",' + raw[1:]
        path.write_text(raw)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if bad_manifest_hash: sha = '0' * 64
        original_read = Path.read_text

        def metadata(path, *args, **kwargs):
            if str(path) == '/proc/self/maps': return mappings
            if str(path) == '/proc/self/cgroup': return '0::/standin-profile-unit.service\n'
            return original_read(path, *args, **kwargs)

        names = tuple(n for n in sys.builtin_module_names if n != '_lsprof') + (('_lsprof',) if builtin else ())
        spec = SimpleNamespace(origin=origin, loader=importlib.machinery.BuiltinImporter)
        output = io.StringIO()
        with patch.multiple(d, **pins), patch.dict(sys.modules), patch.object(sys, 'builtin_module_names', names), \
             patch.object(Path, 'read_text', metadata), patch('importlib.util.find_spec', return_value=spec), \
             patch.dict(os.environ, INVOCATION_ID='d' * 32), \
             patch.object(sys, 'argv', [str(PATH), '--manifest', str(path), '--manifest-sha256', sha]), \
             redirect_stdout(output):
            d.main()
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_full_manifest_cli_source_loader_ignores_valid_poison_pyc(self):
        with tempfile.TemporaryDirectory() as directory:
            pins, manifest, marker = self.fixture(directory)
            sources = [Path(b['path']) for b in (pins['TRAINER'], *pins['STDLIB'].values())]
            caches = [Path(importlib.util.cache_from_source(str(p))) for p in sources]
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sources + caches}
            events = self.invoke(directory, pins, manifest)
            self.assertEqual(marker.read_bytes(), b'entered')
            self.assertEqual(Path(manifest['profile']).read_bytes(), b'standin stats')
            self.assertEqual([e['event'] for e in events], ['start', 'dumped'])
            self.assertTrue(all(e['invocation_id'] == 'd' * 32 and e['qualification_eligible'] is False and
                                e['state_reuse_eligible'] is False for e in events))
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sources + caches})

    def test_bad_hash_provenance_manifest_and_mapped_extension_stop_before_target(self):
        cases = ('manifest-hash', 'wrapper-hash', 'trainer-bytes', 'profile-bytes', 'cProfile-bytes',
                 'authority-bytes', 'python-hash', 'stdlib-provenance',
                 'argv-order', 'qualification', 'reuse', 'frozen-invocation', 'duplicate', 'mapped-extension',
                 'not-builtin', 'extension-origin', 'existing-profile', 'existing-output', 'stdio-collision')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                pins, manifest, marker = self.fixture(directory)
                options = {}
                if case == 'manifest-hash': options['bad_manifest_hash'] = True
                elif case == 'wrapper-hash': manifest['wrapper']['sha256'] = '0' * 64
                elif case in ('trainer-bytes', 'profile-bytes', 'cProfile-bytes', 'authority-bytes'):
                    path = pins['TRAINER']['path'] if case == 'trainer-bytes' else (
                        pins['STDLIB']['profile.py']['path'] if case == 'profile-bytes' else (
                        pins['STDLIB']['cProfile.py']['path'] if case == 'cProfile-bytes' else manifest['argv'][4]))
                    with Path(path).open('ab') as stream: stream.write(b'changed')
                elif case == 'python-hash':
                    pins['PYTHON']['sha256'] = '0' * 64
                elif case == 'stdlib-provenance':
                    manifest['stdlib'] = copy.deepcopy(manifest['stdlib'])
                    manifest['stdlib']['cProfile.py']['path'] += '.untrusted'
                elif case == 'argv-order':
                    manifest['argv'][7:11] = ['--arm', 'control', '--phase', 'mechanics']
                elif case == 'qualification': manifest['qualification_eligible'] = True
                elif case == 'reuse': manifest['state_reuse_eligible'] = True
                elif case == 'frozen-invocation': manifest['invocation_id'] = 'c' * 32
                elif case == 'duplicate': options['duplicate'] = True
                elif case == 'mapped-extension': options['mappings'] = '1-2 r-xp 0 00:00 1 /unknown/_lsprof.so\n'
                elif case == 'not-builtin': options['builtin'] = False
                elif case == 'extension-origin': options['origin'] = '/unknown/_lsprof.so'
                elif case == 'existing-profile': Path(manifest['profile']).write_bytes(b'existing')
                elif case == 'existing-output': Path(manifest['argv'][14]).mkdir()
                elif case == 'stdio-collision': manifest['stdio']['stderr'] = manifest['profile']
                with self.assertRaises(ValueError):
                    self.invoke(directory, pins, manifest, **options)
                self.assertFalse(marker.exists())
                if case == 'existing-profile':
                    self.assertEqual(Path(manifest['profile']).read_bytes(), b'existing')
                else:
                    self.assertFalse(Path(manifest['profile']).exists())

    def test_unchanged_native_mapping_union_rejects_unknown_lsprof_so(self):
        root = PATH.parent
        trainer = ast.parse((root / 'train_siglip2_quadratic_readout.py').read_bytes())
        qualification = ast.parse((root / 'qualify_siglip2_substrate_cpu.py').read_bytes())
        audit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'audit_origins')
        discover = next(n for n in qualification.body if isinstance(n, ast.FunctionDef) and n.name == 'imported_origins')
        with tempfile.TemporaryDirectory() as directory:
            extension = Path(directory) / '_lsprof.so'
            extension.write_bytes(b'unknown mapped library, never imported')
            scope = dict(sys=SimpleNamespace(modules={}), Path=Path, canonical=lambda p: p,
                         loaded_module_origin=lambda *args: None)
            exec(compile(ast.Module(body=[discover], type_ignores=[]), 'original-origin-discovery', 'exec'), scope)
            extract = SimpleNamespace(sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
            with patch.object(Path, 'read_text', return_value=f'1-2 r-xp 0 00:00 1 {extension}\n'):
                origins = scope['imported_origins'](extract, {})
            self.assertEqual(origins['native_files'], [str(extension)])
            self.assertIn(str(extension), origins['files'])
            scope = dict(require=self.runner.require)
            exec(compile(ast.Module(body=[audit], type_ignores=[]), 'original-origin-admission', 'exec'), scope)
            context = dict(source_driver=SimpleNamespace(imported_origins=lambda *args: origins), prior={},
                           extract=extract, selected=dict(packages={}, source_cpu=dict(origins=dict(files={}, modules={}))),
                           warm_record=dict(origins=dict(files={}, modules={})))
            with self.assertRaisesRegex(ValueError, 'unknown or changed native origin'):
                scope['audit_origins'](context)


if __name__ == '__main__':
    if not __debug__ or not sys.dont_write_bytecode:
        raise SystemExit('source-only falsifier requires python -B without -O')
    started = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProfileOnceTests))
    seconds = time.perf_counter() - started
    native = any(n == '_lsprof' or n.split('.')[0] in
                 {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'} for n in sys.modules)
    passed = result.wasSuccessful() and seconds <= 5 and not native
    print(json.dumps({'schema': 'quadratic-profile-once-source-check-v1', 'tests': result.testsRun,
                      'seconds': seconds, 'pass': passed, 'native_imported': native}))
    raise SystemExit(0 if passed else 1)
