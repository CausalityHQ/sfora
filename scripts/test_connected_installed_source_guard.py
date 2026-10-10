"""Actual stdlib loaders and opaque fixture bytes only; no model imports."""
import hashlib
import ast
import importlib
import importlib.machinery
import importlib.util
import pathlib
import sys
import sysconfig
import tempfile
import types
import unittest
import json
from unittest import mock

from connected_installed_source_guard import ImportSources


class SourceImports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()
        self.files = {}
        self.name = '_installed_guard_fixture'
        self.addCleanup(self.clear)
        self.write(self.name + '/__init__.py', 'from .child import value\n')
        self.write(self.name + '/child.py', 'value = "fresh"\n')
        self.write(self.name + '/resource.txt', 'exact resource')

    def clear(self):
        for name in list(sys.modules):
            if name == self.name or name.startswith(self.name + '.'):
                del sys.modules[name]

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        raw = path.read_bytes()
        self.files[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
        return path

    def gate(self, namespaces=None):
        return ImportSources(self.files, roots=(str(self.root),), stdlib_roots=(), namespaces=namespaces or {})

    def load(self, gate):
        with mock.patch.object(sys, 'path', [str(self.root), *sys.path]):
            return importlib.import_module(self.name)

    def test_real_relative_import_ignores_valid_poisoned_cache_and_checks_resource(self):
        source = self.root / self.name / 'child.py'
        cache = pathlib.Path(importlib.util.cache_from_source(str(source)))
        cache.parent.mkdir()
        from importlib._bootstrap_external import _code_to_timestamp_pyc
        poison = _code_to_timestamp_pyc(compile('value = "poison"', str(source), 'exec'), int(source.stat().st_mtime), source.stat().st_size)
        cache.write_bytes(poison)
        gate = self.gate()
        with gate, mock.patch.object(importlib.machinery.SourceFileLoader, 'get_data', side_effect=AssertionError('cache/filesystem loader bypass')):
            package = self.load(gate)
            self.assertEqual(package.value, 'fresh')
            self.assertEqual(package.__loader__.get_data(str(self.root / self.name / 'resource.txt')), b'exact resource')
            with self.assertRaises(ValueError):
                package.__loader__.get_data(str(cache))
            gate.check()
        self.assertEqual(cache.read_bytes(), poison)
        self.assertNotIn(gate, sys.meta_path)

    def test_source_mutation_and_symlink_reject_before_execution(self):
        for symlink in [False, True]:
            with self.subTest(symlink=symlink):
                self.clear()
                source = self.root / self.name / 'child.py'
                if source.is_symlink():
                    source.unlink()
                self.write(self.name + '/child.py', 'value = "fresh"\n')
                gate = self.gate()
                if symlink:
                    saved = self.root / 'saved.py'
                    source.rename(saved)
                    source.symlink_to(saved)
                else:
                    source.write_text('raise AssertionError("must not execute")\n')
                with self.assertRaises(ValueError), gate:
                    self.load(gate)

    def test_old_root_unapproved_source_bytecode_and_native_origins_reject(self):
        gate = self.gate()
        foreign = self.root / 'foreign'; foreign.mkdir()
        (foreign / 'unapproved.py').write_text('raise AssertionError("not execute")')
        for filename, loader in [('unapproved.py', importlib.machinery.SourceFileLoader),
                                 ('only.pyc', importlib.machinery.SourcelessFileLoader),
                                 ('native.so', importlib.machinery.ExtensionFileLoader)]:
            spec = importlib.util.spec_from_file_location('unapproved', foreign / filename, loader=loader('unapproved', str(foreign / filename)))
            with self.subTest(filename=filename), mock.patch.object(importlib.machinery.PathFinder, 'find_spec', return_value=spec), self.assertRaises(ImportError):
                gate.find_spec('unapproved')

    def test_loaded_identity_spec_source_and_foreign_replacement_reject(self):
        for mode in ['file', 'spec', 'loader', 'path', 'registry', 'bytes', 'name', 'package']:
            with self.subTest(mode=mode):
                self.clear(); self.write(self.name + '/child.py', 'value = "fresh"\n')
                gate = self.gate()
                gate.__enter__()
                module = self.load(gate)
                if mode == 'file': module.__file__ = '/foreign.py'
                elif mode == 'spec': module.__spec__.origin = '/foreign.py'
                elif mode == 'loader': module.__loader__ = object()
                elif mode == 'path': module.__path__.append('/foreign')
                elif mode == 'registry': sys.modules[self.name] = types.ModuleType(self.name)
                elif mode == 'name': module.__name__ = 'foreign'
                elif mode == 'package': module.__package__ = 'foreign'
                else: (self.root / self.name / 'child.py').write_text('value = "changed"')
                with self.assertRaises(ValueError): gate.check()
                foreign = sys.modules[self.name]
                with self.assertRaises(ValueError): gate.close()
                self.assertIs(sys.modules[self.name], foreign)
                self.assertNotIn(gate, sys.meta_path)

    def test_namespace_requires_exact_declared_locations(self):
        (self.root / 'declared_namespace').mkdir()
        gate = self.gate({'declared_namespace': (str(self.root / 'declared_namespace'),)})
        with mock.patch.object(sys, 'path', [str(self.root)]):
            self.assertIsNotNone(gate.find_spec('declared_namespace'))
            with self.assertRaises(ImportError): self.gate().find_spec('declared_namespace')

    def test_real_namespace_identity_is_retained_and_foreign_replacement_rejects(self):
        name = self.name + '_namespace'
        directory = self.root / name
        directory.mkdir()
        self.addCleanup(lambda: sys.modules.pop(name, None))
        gate = self.gate({name: (str(directory),)})
        gate.__enter__()
        with mock.patch.object(sys, 'path', [str(self.root)]):
            module = importlib.import_module(name)
            gate.check()
            foreign = types.ModuleType(name)
            sys.modules[name] = foreign
            with self.assertRaises(ValueError): gate.close()
            self.assertIs(sys.modules[name], foreign)
        self.assertNotIn(gate, sys.meta_path)

    def test_authority_rows_and_preexisting_modules_reject(self):
        for row in [{'sha256':'x','bytes':1}, {'sha256':'0'*64,'bytes':True}, {'sha256':'0'*64,'bytes':-1}]:
            with self.subTest(row=row), self.assertRaises(ValueError):
                ImportSources({str(self.root / 'bad.py'):row}, roots=(str(self.root),), stdlib_roots=(), namespaces={})
        gate = self.gate()
        existing = types.ModuleType(self.name); existing.__file__ = str(self.root / self.name / '__init__.py')
        sys.modules[self.name] = existing
        with self.assertRaises(ValueError): gate.__enter__()
        self.assertIs(sys.modules[self.name], existing)
        self.assertNotIn(gate, sys.meta_path)

    def test_input_aliases_do_not_change_approved_rows_and_tables_are_read_only(self):
        gate = self.gate()
        path = str(self.root / self.name / 'child.py')
        self.files[path]['sha256'] = '0' * 64
        self.assertEqual(gate.read(path), b'value = "fresh"\n')
        with self.assertRaises(TypeError): gate.files[path] = ('0' * 64, 16)
        with self.assertRaises(TypeError): gate.namespaces['foreign'] = ('/foreign',)

    def test_primary_error_survives_exit_rejection_and_finder_is_removed(self):
        gate = self.gate()
        primary = RuntimeError('original body rejection')
        try:
            with gate:
                self.load(gate)
                (self.root / self.name / 'child.py').write_text('changed bytes')
                raise primary
        except RuntimeError as observed:
            self.assertIs(observed, primary)
            self.assertTrue(any('source guard exit also failed' in note for note in observed.__notes__))
        else:
            self.fail('primary error was suppressed')
        self.assertNotIn(gate, sys.meta_path)

    def lazy_fixture(self):
        observation = pathlib.Path(__file__).resolve().parents[1] / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-installed-source-guard-v1/lazy-source-observation.json'
        rows = json.loads(observation.read_text())
        row = next(v for k, v in rows.items() if k.endswith('/utils/import_utils.py'))
        self.assertEqual(row['sha256'], '854f6a998a194c12f2e813f745b602a9e7aa7816b0e3f4153da19cfb708b6046')
        self.assertEqual(hashlib.sha256(row['source'].encode()).hexdigest(), row['sha256'])
        cls = next(n for n in ast.parse(row['source']).body if isinstance(n, ast.ClassDef) and n.name == '_LazyModule')
        constructor = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        method = ast.get_source_segment(row['source'], constructor)
        provider = self.name + '.provider'
        code = 'from __future__ import annotations\nimport os\nfrom types import ModuleType\nfrom itertools import chain\nclass _LazyModule(ModuleType):\n' + '\n'.join('    ' + line for line in method.splitlines()) + '\n'
        self.write(self.name + '/provider.py', code)
        self.write(self.name + '/__init__.py', 'import sys\nfrom .provider import _LazyModule\nsys.modules[__name__] = _LazyModule(__name__, __file__, {"child": ["value"]}, module_spec=__spec__)\n')
        return provider, ImportSources(self.files, roots=(str(self.root),),
                                       stdlib_roots=(str(pathlib.Path(sysconfig.get_path('stdlib')).resolve()),), namespaces={},
                                       replacements={self.name: (provider, '_LazyModule')})

    def test_actual_finite_lazy_constructor_replacement_and_same_class_foreign_reject(self):
        provider, gate = self.lazy_fixture()
        previous = sys.getprofile()
        gate.__enter__()
        module = self.load(gate)
        self.assertEqual(module._name, self.name)
        self.assertIs(sys.getprofile(), previous)
        gate.check()
        cls = vars(sys.modules[provider])['_LazyModule']
        foreign = cls(self.name, module.__file__, {'child': ['value']}, module_spec=module.__spec__)
        sys.modules[self.name] = foreign
        with self.assertRaises(ValueError): gate.close()
        self.assertIs(sys.modules[self.name], foreign)

    def test_lazy_constructor_code_tampering_rejects_and_profile_is_restored(self):
        provider, gate = self.lazy_fixture()
        previous = sys.getprofile()
        gate.__enter__()
        self.load(gate)
        cls = vars(sys.modules[provider])['_LazyModule']
        saved = cls.__init__.__code__
        cls.__init__.__code__ = saved.replace(co_filename='/foreign.py')
        try:
            with self.assertRaises(ValueError): gate.check()
        finally:
            cls.__init__.__code__ = saved
            gate.close()
        self.assertIs(sys.getprofile(), previous)


if __name__ == '__main__':
    unittest.main()
