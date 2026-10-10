"""Stdlib-only falsifiers for the installed-native relocation correspondence.

Synthetic files exercise the relocation; the committed original bundle, installed
authority and ownership bytes check the actual evidence; the genuine
imported_origins, CombinedAuthority.mappings and CombinedAuthority.collect run
UNCHANGED. No Torch/NumPy/PIL, payload read or native import.
"""
import ast
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVIDENCE = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


reloc = load('_relocation_under_test', HERE / 'connected_installed_native_relocation.py')
qualifier = load('_relocation_genuine_qualifier', HERE / 'qualify_siglip2_substrate_cpu.py')
native = load('_relocation_genuine_native', HERE / 'connected_control_native_authority.py')
extract = load('_relocation_genuine_extract', HERE / 'extract_siglip2_vision_source.py')
helper = load('_relocation_installed_environment', ROOT / 'src/sfora/connected_installed_environment.py')

OLD_SITE = '/home/riomus/group-learning/.venv/lib/python3.13/site-packages'
NEW_SITE = '/home/riomus/runs/sfora-connected-installed-site-v2/site-packages'
BUNDLE_SHA = 'e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153'
AUDIT_SHA = '8e38d2e503a44cdc1f2fe3cf4bf08c2a86dfb1f86603363aef5dcb262812f36f'
INSTALLED_SHA = '383d7200392b1531ccdda52f13081f42fe4c3a27f589922d14025efdee1cea5f'
# Source-text digests of the genuine seams at the exact master base 5b5d922f.
SEAMS = {
    'qualify_siglip2_substrate_cpu.py': {
        'imported_origins': 'd7e7181a05cf16a0029df4c29081bf2afe9a4add7a280c3e0069b2551ab1a5bb',
        'loaded_module_origin': '5f60fdd677d6d1011a907e45dc71144e1d0690e8ee7e4e8ca0bae61dfd7052d1',
        'canonical': 'e7a6de3792c08556abb346f09d4d622fc6b1730fabd15f1ddb39e0196550f6e5'},
    'connected_control_native_authority.py': {
        'CombinedAuthority.identity': 'b5435618fbe254702bb412b474c0cb649da3d31e10cb07a29f86e21440c8818e',
        'CombinedAuthority.mappings': 'b8045559e204ff57001f6a4373550ca449479df24b8d215c4d688314ff87dafa',
        'CombinedAuthority.collect': 'c1e924fd5a11f61ce495bcdd2400486364aae7c7deb6035545bb0724e2017d9c'}}

MEMBERS = {
    'PIL/__init__.py': b'# PIL\n', 'numpy/__init__.py': b'# numpy\n', 'numpy/core.so': b'numpy native bytes',
    'safetensors/__init__.py': b'# safetensors\n', 'torch/__init__.py': b'# torch\n', 'torch/mod.py': b'# torch.mod\n',
    'torch/_C.so': b'torch native bytes', 'torch/twin_a.py': b'identical twin\n', 'torch/twin_b.py': b'identical twin\n',
    'torchvision/__init__.py': b'# torchvision\n', 'transformers/__init__.py': b'# transformers\n'}
NATIVE_MEMBERS = {'torch/_C.so', 'numpy/core.so'}
ANCHORS = {'libanchor.so.1': b'system anchor one', 'libother.so.2': b'system anchor two'}
NOISE = ['55a0-55a1 rw-p 00000000 00:00 0 [heap]', '7f00-7f01 r--p 00000000 00:00 0',
         '7f10-7f11 r--p 00000000 08:01 12 /usr/share/data.txt', '7f20-7f21 r--p 00000000 08:01 13 relative/lib.so',
         '7f30-7f31 r--p 00000000 00:00 0']


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def maps_line(path):
    value = os.stat(path)
    return (f'ffff00000000-ffff00001000 r-xp 00000000 {os.major(value.st_dev):x}:{os.minor(value.st_dev):x} '
            f'{value.st_ino} {path}')


@contextmanager
def fake_maps(lines):
    real, text = Path.read_text, '\n'.join(lines) + '\n'
    def read_text(self, *args, **kwargs):
        return text if str(self) == '/proc/self/maps' else real(self, *args, **kwargs)
    with patch.object(Path, 'read_text', read_text):
        yield


def fake_modules(site):
    modules = {}
    for name, relative in (('torch', 'torch/__init__.py'), ('torch.mod', 'torch/mod.py'), ('numpy', 'numpy/__init__.py')):
        module = ModuleType(name)
        module.__file__ = f'{site}/{relative}'
        modules[name] = module
    return modules


def mappings_reader():
    return native.CombinedAuthority.__new__(native.CombinedAuthority)


class World:
    """Synthetic original and installed trees under one temporary directory."""
    def __init__(self, base, with_old=False, old_name='original', new_name='installed'):
        self.base = Path(base).resolve()
        self.old, self.new = str(self.base / old_name / 'site-packages'), str(self.base / new_name / 'site-packages')
        self.system = self.base / 'system'
        for relative, data in MEMBERS.items():
            self.write(Path(self.new) / relative, data)
            if with_old:
                self.write(Path(self.old) / relative, data)
        for name, data in ANCHORS.items():
            self.write(self.system / name, data)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def environment(self):
        files = {f'{self.old}/{r}': digest(d) for r, d in MEMBERS.items()}
        files.update({str(self.system / n): digest(d) for n, d in ANCHORS.items()})
        natives = {p: h for p, h in files.items() if p.removeprefix(self.old + '/') in NATIVE_MEMBERS or p.startswith(str(self.system))}
        packages = {n: {'root': f'{self.old}/{n}', 'origin': f'{self.old}/{n}/__init__.py', 'version': '1.0'}
                    for n in reloc.PACKAGES}
        return {'files': files, 'native_files': natives, 'packages': packages,
                'vision_constructor': f'{self.old}/transformers/__init__.py'}

    def moved(self, path):
        return self.new + path[len(self.old):] if path.startswith(self.old + '/') else path

    def image(self, env):
        return {'files': {self.moved(p): h for p, h in env['files'].items()},
                'native_files': {self.moved(p): h for p, h in env['native_files'].items()},
                'packages': {n: {k: self.moved(v) if k in ('root', 'origin') else v for k, v in p.items()}
                             for n, p in env['packages'].items()},
                'vision_constructor': self.moved(env['vision_constructor'])}

    def arguments(self, bundle_edit=None, installed_edit=None, raw_bundle=None, **overrides):
        env = self.environment()
        bundle = {'schema': reloc.BUNDLE_SCHEMA, 'environment': copy.deepcopy(env), 'scope': 'synthetic'}
        if bundle_edit:
            bundle_edit(bundle)
        raw_bundle = raw_bundle or json.dumps(bundle).encode()
        try:
            image = self.image(bundle['environment'])
        except (KeyError, TypeError, AttributeError):
            image = self.image(env)
        installed = {'schema': reloc.INSTALLED_SCHEMA, 'original_bundle_sha256': digest(raw_bundle),
                     'original_ownership_audit_sha256': AUDIT_SHA, 'site_packages': self.new,
                     'distributions': {}, 'expected_environment': image}
        if installed_edit:
            installed_edit(installed)
        raw_installed = json.dumps(installed).encode()
        arguments = dict(original_bundle=raw_bundle, installed_authority=raw_installed,
                         trusted_bundle_sha256=digest(raw_bundle), trusted_installed_sha256=digest(raw_installed),
                         trusted_ownership_audit_sha256=AUDIT_SHA, old_site=self.old, new_site=self.new)
        arguments.update(overrides)
        return arguments

    def relocation(self, **kwargs):
        return reloc.relocation(**self.arguments(**kwargs))

    def native_lines(self, site, extra=()):
        paths = [f'{site}/{r}' for r in sorted(NATIVE_MEMBERS)] + [str(self.system / n) for n in sorted(ANCHORS)]
        return [maps_line(p) for p in paths] + list(extra) + NOISE

    def observe(self, rel, site=None, extra=(), packages=None):
        """Genuine imported_origins and CombinedAuthority.mappings over the same patched maps."""
        site = site or self.new
        with patch.dict(sys.modules, fake_modules(site)), fake_maps(self.native_lines(site, extra)):
            origins = qualifier.imported_origins(extract, packages or rel.installed_packages)
            live = mappings_reader().mappings()
        return origins, live

    def expected_origins(self, rel):
        old = lambda r: f'{self.old}/{r}'
        system = {str(self.system / n): digest(d) for n, d in ANCHORS.items()}
        files = {old(r): digest(MEMBERS[r]) for r in ('torch/__init__.py', 'torch/mod.py', 'numpy/__init__.py', *NATIVE_MEMBERS)}
        files.update(system)
        return {'packages': rel.original_packages,
                'modules': {'torch': old('torch/__init__.py'), 'torch.mod': old('torch/mod.py'), 'numpy': old('numpy/__init__.py')},
                'native_files': sorted([old(r) for r in NATIVE_MEMBERS] + list(system)), 'files': files}


class Base(unittest.TestCase):
    def setUp(self):
        self.assertFalse(reloc.PACKAGES & {n.split('.')[0] for n in sys.modules}, 'a real package is imported')
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def rejects(self, pattern, call, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, pattern):
            call(*args, **kwargs)

    def world(self, with_old=False, name='w'):
        return World(self.tmp / name, with_old)


class Construction(Base):
    def test_exact_correspondence_is_accepted_and_immutable(self):
        w = self.world()
        rel = w.relocation()
        self.assertEqual(rel.to_original(w.new + '/torch/_C.so'), w.old + '/torch/_C.so')
        self.assertEqual(rel.to_installed(w.old + '/torch/_C.so'), w.new + '/torch/_C.so')
        anchor = str(w.system / 'libanchor.so.1')
        self.assertEqual((rel.to_original(anchor), rel.to_installed(anchor)), (anchor, anchor))
        self.assertEqual(rel.installed_packages['torch']['root'], w.new + '/torch')
        self.assertEqual(rel.original_packages['torch']['root'], w.old + '/torch')
        with self.assertRaises(AttributeError):
            rel.new_site = '/elsewhere'
        with self.assertRaises(AttributeError):
            rel._to_original = {}
        self.assertFalse(hasattr(rel, '__dict__'))
        rel.installed_packages['torch']['root'] = '/mutated'
        self.assertEqual(rel.installed_packages['torch']['root'], w.new + '/torch')
        with self.assertRaises(TypeError):
            rel._to_original['/x'] = '/y'

    def test_private_package_tables_are_frozen_at_both_levels(self):
        w = self.world()
        rel = w.relocation()
        origins, _ = w.observe(rel)
        for table in (rel._original_packages, rel._installed_packages):
            with self.assertRaises(TypeError):
                table['torch'] = {}
            with self.assertRaises(TypeError):
                del table['torch']
            for field in ('root', 'origin', 'version'):
                with self.assertRaises(TypeError):
                    table['torch'][field] = '/mutated'
                with self.assertRaises(TypeError):
                    del table['torch'][field]
            self.assertFalse(hasattr(table['torch'], 'update'))
        for packages in (rel.original_packages, rel.installed_packages):
            self.assertIs(type(packages), dict)
            self.assertTrue(all(type(record) is dict for record in packages.values()))
            packages['torch']['root'] = '/mutated'
        inverted = rel.invert_origins(origins, packages=rel.original_packages)
        self.assertIs(type(inverted['packages']), dict)
        self.assertTrue(all(type(record) is dict for record in inverted['packages'].values()))
        inverted['packages']['torch']['origin'] = '/mutated'
        self.assertEqual(rel.original_packages['torch']['origin'], w.old + '/torch/__init__.py')
        self.assertEqual(rel.invert_origins(origins, packages=rel.original_packages), w.expected_origins(rel))

    def test_pins_and_types_are_checked_before_parsing(self):
        w = self.world()
        good = w.arguments()
        for key in ('trusted_bundle_sha256', 'trusted_installed_sha256', 'trusted_ownership_audit_sha256'):
            for bad in ('ABC', None, 'A' * 64, 7):
                self.rejects('SHA', reloc.relocation, **{**good, key: bad})
        for key in ('trusted_bundle_sha256', 'trusted_installed_sha256'):
            self.rejects('SHA differs', reloc.relocation, **{**good, key: '0' * 64})
        self.rejects('immutable bytes', reloc.relocation, **{**good, 'original_bundle': good['original_bundle'].decode()})
        self.rejects('immutable bytes', reloc.relocation, **{**good, 'installed_authority': bytearray(good['installed_authority'])})
        self.rejects('immutable bytes', reloc.relocation, **{**good, 'original_bundle': None})
        # A mismatched ownership pin is a different installed authority binding.
        self.rejects('binding differs', reloc.relocation, **{**good, 'trusted_ownership_audit_sha256': '1' * 64})

    def test_strict_json(self):
        w = self.world()
        env = json.dumps(w.environment())
        for raw in (b'{"schema":"x","schema":"y"}', b'{"schema": NaN}', b'{"schema": 1e999}', b'\xff', b'[]', b'{}'):
            self.rejects('strict JSON|schema|duplicate|nonfinite|bundle', w.relocation, raw_bundle=raw)
        raw = ('{"schema":"%s","environment":%s,"environment":%s}' % (reloc.BUNDLE_SCHEMA, env, env)).encode()
        self.rejects('duplicate', w.relocation, raw_bundle=raw)

    def test_original_environment_is_exact(self):
        w = self.world()
        def edit(function):
            return lambda bundle: function(bundle['environment'])
        cases = {
            'exact environment': edit(lambda e: e.update(extra={})),
            'exact environment#2': edit(lambda e: e.pop('vision_constructor')),
            'canonical absolute path': edit(lambda e: e['files'].update({'relative/x.py': '0' * 64})),
            'canonical absolute path#2': edit(lambda e: e['files'].update({w.old + '/../x.py': '0' * 64})),
            'canonical absolute path#3': edit(lambda e: e['files'].update({w.old + '//x.py': '0' * 64})),
            'SHA256': edit(lambda e: e['files'].update({w.old + '/x.py': 'ABC'})),
            'native membership': edit(lambda e: e['native_files'].update({w.old + '/x.so': '0' * 64})),
            'native membership/hash': edit(lambda e: e['native_files'].update({w.old + '/torch/_C.so': '2' * 64})),
            'exact package set': edit(lambda e: e['packages'].pop('PIL')),
            'exact package set#2': edit(lambda e: e['packages'].update(pandas=e['packages']['PIL'])),
            'exact package record': edit(lambda e: e['packages']['PIL'].update(extra=1)),
            'package root/origin': edit(lambda e: e['packages']['PIL'].update(origin=w.old + '/PIL/other.py')),
            'one original site root': edit(lambda e: (e['packages']['PIL'].update(
                root='/elsewhere/PIL', origin='/elsewhere/PIL/__init__.py'),
                e['files'].update({'/elsewhere/PIL/__init__.py': '0' * 64}))),
            'constructor': edit(lambda e: e.update(vision_constructor=w.old + '/transformers/missing.py')),
            'external non-native': edit(lambda e: e['files'].update({'/usr/lib/libfoo.py': '3' * 64})),
        }
        for pattern, function in cases.items():
            with self.subTest(pattern):
                self.rejects(pattern.partition('#')[0], w.relocation, bundle_edit=function)

    def test_roots_are_declared_disjoint_and_exact(self):
        w = self.world()
        self.rejects('disjoint', w.relocation, new_site=w.old)
        self.rejects('disjoint', w.relocation, new_site=w.old + '/inner')
        self.rejects('disjoint', w.relocation, old_site=w.new + '/inner')
        self.rejects('canonical', w.relocation, new_site=w.new + '/')
        self.rejects('canonical', w.relocation, new_site='relative')
        self.rejects('declared old root', w.relocation, old_site=w.old + '-other')
        self.rejects('declared old root', w.relocation, old_site=str(Path(w.old).parent))

    def test_installed_authority_grants_nothing(self):
        w = self.world()
        extra = w.new + '/torch/_extra.so'
        cases = {
            'installed JSON adds a DSO': lambda i: (i['expected_environment']['files'].update({extra: '4' * 64}),
                                                    i['expected_environment']['native_files'].update({extra: '4' * 64})),
            'installed JSON drops a file': lambda i: i['expected_environment']['files'].pop(w.new + '/torch/mod.py'),
            'installed JSON changes a hash': lambda i: i['expected_environment']['files'].update({w.new + '/torch/mod.py': '5' * 64}),
            'installed JSON keeps original wheel path': lambda i: i['expected_environment']['files'].update(
                {w.old + '/torch/mod.py': digest(MEMBERS['torch/mod.py'])}),
            'installed JSON relocates an anchor': lambda i: i['expected_environment']['files'].update(
                {w.new + '/libanchor.so.1': digest(ANCHORS['libanchor.so.1'])}),
            'package root changes': lambda i: i['expected_environment']['packages']['torch'].update(root=w.old + '/torch'),
            'constructor changes': lambda i: i['expected_environment'].update(vision_constructor=w.old + '/transformers/__init__.py'),
            'schema': lambda i: i.update(schema='other'),
            'site name': lambda i: i.update(site_packages=w.new + '-other'),
            'bundle pin': lambda i: i.update(original_bundle_sha256='6' * 64),
            'extra key': lambda i: i.update(extra=1),
            'distributions type': lambda i: i.update(distributions=[]),
        }
        for label, edit in cases.items():
            with self.subTest(label):
                self.rejects('installed', w.relocation, installed_edit=edit)

    def test_anchors_stay_exact_original_paths(self):
        w = self.world()
        anchor = str(w.system / 'libanchor.so.1')
        rel = w.relocation()
        for foreign in (str(w.system / 'libanchor.so.2'), '/tmp/libanchor.so.1', w.new + '/libanchor.so.1',
                        anchor + '/', anchor + '/../libanchor.so.1', '/usr/lib/aarch64-linux-gnu/libc.so.6'):
            with self.subTest(foreign):
                self.rejects('unknown or foreign', rel.to_original, foreign)
        self.rejects('system anchor inside installed root', w.relocation, bundle_edit=lambda b: (
            b['environment']['files'].update({w.new + '/libinside.so': '7' * 64}),
            b['environment']['native_files'].update({w.new + '/libinside.so': '7' * 64})))


class Correspondence(Base):
    def test_every_path_round_trips_and_rejects_foreign_spellings(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        for old in w.environment()['files']:
            self.assertEqual(rel.to_original(rel.to_installed(old)), old)
        for installed in (w.new + '/PIL/_imagingcms.cpython-313-aarch64-linux-gnu.so', w.new + '/torch/other.py',
                          w.new + '-evil/torch/mod.py', w.new + '/torch/../torch/mod.py', w.new + '//torch/mod.py',
                          w.new + '/torch/mod.py/', 'torch/mod.py', '', w.new, None, b'bytes'):
            with self.subTest(installed):
                self.rejects('unknown|foreign|path string', rel.to_original, installed)
        for original in (w.new + '/torch/mod.py', w.old + '/torch/missing.py', w.old, None):
            with self.subTest(original):
                self.rejects('unknown or foreign', rel.to_installed, original)

    def test_original_selected_wheel_path_is_rejected_even_with_equal_bytes(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        old = w.old + '/torch/mod.py'
        self.assertEqual(Path(old).read_bytes(), Path(w.new + '/torch/mod.py').read_bytes())
        self.rejects('original selected-wheel', rel.to_original, old)
        self.rejects('original selected-wheel', rel.fresh, old)

    def test_candidate_extensions_absent_from_the_original_table_are_not_granted(self):
        w = self.world()
        rel = w.relocation()
        for name in ('PIL/_imagingcms.cpython-313-aarch64-linux-gnu.so', 'hf_xet/hf_xet.abi3.so',
                     'charset_normalizer/md.cpython-313-aarch64-linux-gnu.so'):
            path = Path(w.new) / name
            World.write(path, b'candidate extension bytes')
            origins, _ = w.observe(rel)
            origins['files'][str(path)] = digest(b'candidate extension bytes')
            origins['native_files'] = sorted([*origins['native_files'], str(path)])
            self.rejects('unknown or foreign', rel.invert_origins, origins, packages=rel.original_packages)
            self.rejects('unknown or foreign', rel.invert_mappings, {str(path): (os.stat(path).st_dev, os.stat(path).st_ino)})


class Inverse(Base):
    def test_genuine_collector_output_inverts_exactly(self):
        w = self.world()
        rel = w.relocation()
        origins, live = w.observe(rel)
        inverted = rel.invert_origins(origins, packages=rel.original_packages)
        self.assertEqual(inverted, w.expected_origins(rel))
        self.assertEqual(list(inverted), ['packages', 'modules', 'native_files', 'files'])
        self.assertEqual(inverted['native_files'], sorted(inverted['native_files']))
        self.assertEqual(set(rel.invert_mappings(live)), set(inverted['native_files']))
        self.assertEqual(set(live), set(origins['native_files']))
        for path, identity in live.items():
            self.assertEqual(rel.invert_mappings(live)[rel.to_original(path)], identity)
        self.assertEqual(inverted['packages'], rel.original_packages)
        self.assertIsNot(inverted['packages'], origins['packages'])

    def test_inverse_does_not_depend_on_how_the_roots_sort_against_anchors(self):
        # old sorts AFTER the system anchors, new BEFORE them: the two sorted orders differ.
        w = World(self.tmp / 'order', with_old=True, old_name='zold', new_name='anew')
        rel = w.relocation()
        origins, live = w.observe(rel)
        self.assertNotEqual([rel.to_installed(p) for p in sorted(w.expected_origins(rel)['native_files'])], origins['native_files'])
        self.assertEqual(rel.invert_origins(origins, packages=rel.original_packages), w.expected_origins(rel))
        self.assertEqual(set(rel.invert_mappings(live)), set(w.expected_origins(rel)['native_files']))

    def test_canonical_map_scanner_agrees_between_genuine_loops(self):
        w = self.world()
        rel = w.relocation()
        origins, live = w.observe(rel)
        self.assertEqual(set(live), set(origins['native_files']))
        self.assertEqual(len(live), 4)
        collector = (HERE / 'qualify_siglip2_substrate_cpu.py').read_text()
        authority = (HERE / 'connected_control_native_authority.py').read_text()
        scanner = "fields = line.split(maxsplit=5)"
        self.assertIn(scanner + "\n        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5]:", collector)
        self.assertIn(scanner + "\n            if len(fields) != 6 or not fields[5].startswith('/') or '.so' not in fields[5]: continue", authority)
        for line in ('/x/gone.so (deleted)',):
            bad = maps_line(w.new + '/torch/_C.so').rsplit(' ', 1)[0] + ' ' + line
            with fake_maps([bad]):
                self.rejects('deleted', mappings_reader().mappings)
                self.rejects('canonical regular file', qualifier.imported_origins, extract, {})

    def test_fresh_hash_bytes_must_equal_original_accepted_digest(self):
        w = self.world()
        rel = w.relocation()
        for relative in ('torch/mod.py', 'torch/_C.so', 'numpy/__init__.py'):
            with self.subTest(relative):
                path = Path(w.new) / relative
                original = path.read_bytes()
                path.write_bytes(original + b'!')
                try:
                    origins, _ = w.observe(rel)
                    self.rejects('fresh installed bytes differ', rel.invert_origins, origins, packages=rel.original_packages)
                finally:
                    path.write_bytes(original)
        origins, _ = w.observe(rel)
        origins['files'][w.new + '/torch/mod.py'] = '8' * 64
        self.rejects('fresh installed bytes differ', rel.invert_origins, origins, packages=rel.original_packages)
        rejected = w.relocation(bundle_edit=lambda b: b['environment']['files'].update(
            {w.old + '/torch/mod.py': digest(b'changed accepted bytes')}))
        origins, _ = w.observe(rejected)
        self.rejects('fresh installed bytes differ', rejected.invert_origins, origins, packages=rejected.original_packages)

    def test_unknown_foreign_and_original_mappings_are_rejected(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        foreign = w.base / 'foreign'
        World.write(foreign / 'libanchor.so.1', ANCHORS['libanchor.so.1'])
        extra = Path(w.new) / 'torch/_extra.so'
        World.write(extra, b'extra')
        cases = {'unknown or foreign': [str(foreign / 'libanchor.so.1'), str(extra)],
                 'original selected-wheel': [w.old + '/torch/_C.so']}
        for pattern, paths in cases.items():
            for path in paths:
                with self.subTest(path):
                    origins, live = w.observe(rel, extra=[maps_line(path)])
                    self.rejects(pattern, rel.invert_origins, origins, packages=rel.original_packages)
                    self.rejects(pattern, rel.invert_mappings, live)

    def test_supplemental_files_must_be_split_out_before_inversion(self):
        w = self.world()
        rel = w.relocation()
        runtime = w.base / 'runtime/libcandidate.so'
        World.write(runtime, b'separately authenticated supplemental native')
        origins, live = w.observe(rel, extra=[maps_line(runtime)])
        self.assertIn(str(runtime), origins['native_files'])
        self.assertIn(str(runtime), live)
        self.rejects('unknown or foreign', rel.invert_origins, origins, packages=rel.original_packages)
        self.rejects('unknown or foreign', rel.invert_mappings, live)
        supplemental = {str(runtime)}
        split = {**origins, 'files': {p: h for p, h in origins['files'].items() if p not in supplemental},
                 'native_files': [p for p in origins['native_files'] if p not in supplemental]}
        self.assertEqual(rel.invert_origins(split, packages=rel.original_packages), w.expected_origins(rel))
        self.assertEqual(set(rel.invert_mappings({p: v for p, v in live.items() if p not in supplemental})),
                         set(w.expected_origins(rel)['native_files']))

    def test_malformed_observations_are_rejected(self):
        w = self.world()
        rel = w.relocation()
        packages = rel.original_packages
        def run(edit):
            origins, _ = w.observe(rel)
            edit(origins)
            return origins
        cases = {
            'exact origins keys': lambda o: o.pop('modules'),
            'exact origins keys#2': lambda o: o.update(extra={}),
            'origins types': lambda o: o.update(files=list(o['files'])),
            'origins types#2': lambda o: o.update(native_files=tuple(o['native_files'])),
            'package authority': lambda o: o['packages']['torch'].update(root=w.old + '/torch'),
            'native inventory': lambda o: o['native_files'].append(o['native_files'][0]),
            'native inventory#2': lambda o: o['native_files'].append(w.new + '/torch/mod.py'),
            'native inventory#3': lambda o: o['files'].pop(o['native_files'][0]),
            'module origin outside': lambda o: o['modules'].update({'torch.x': w.new + '/numpy/__init__.py'}),
            'module origin outside#2': lambda o: o['modules'].update({'evil.x': w.new + '/torch/mod.py'}),
            'module origin outside#3': lambda o: o['modules'].update({'torch.x': w.new + '/torch/twin_a.py'}),
            'unknown or foreign': lambda o: o['files'].update({w.new + '/torch/other.py': '9' * 64}),
            'SHA256': lambda o: o['files'].update({w.new + '/torch/mod.py': 'XYZ'}),
        }
        for pattern, edit in cases.items():
            with self.subTest(pattern):
                self.rejects(pattern.partition('#')[0], rel.invert_origins, run(edit), packages=packages)
        origins, _ = w.observe(rel)
        self.rejects('package authority', rel.invert_origins, origins, packages={**packages, 'torch': {**packages['torch'], 'version': '9'}})
        other = World(self.tmp / 'other').relocation()
        self.rejects('package authority', other.invert_origins, origins, packages=other.original_packages)

    def test_symlink_missing_and_replaced_files_are_rejected(self):
        w = self.world()
        rel = w.relocation()
        origins, live = w.observe(rel)
        for relative in ('torch/mod.py', 'torch/_C.so'):
            path = Path(w.new) / relative
            original = path.read_bytes()
            copy_path = path.with_name(path.name + '.copy')
            copy_path.write_bytes(original)
            path.unlink()
            path.symlink_to(copy_path)
            try:
                self.rejects('nonsymlink', rel.invert_origins, origins, packages=rel.original_packages)
                if relative in NATIVE_MEMBERS:
                    self.rejects('nonsymlink', rel.invert_mappings, live)
                self.rejects('nonsymlink', rel.fresh, str(path))
            finally:
                path.unlink()
                path.write_bytes(original)
                copy_path.unlink()
        path = Path(w.new) / 'torch/mod.py'
        original = path.read_bytes()
        path.unlink()
        try:
            self.rejects('missing', rel.invert_origins, origins, packages=rel.original_packages)
        finally:
            path.write_bytes(original)
        directory = Path(w.new) / 'torch/mod.py'
        directory.unlink()
        directory.mkdir()
        try:
            self.rejects('nonsymlink', rel.fresh, str(directory))
        finally:
            directory.rmdir()
            directory.write_bytes(original)

    def test_mapped_files_must_be_declared_natives_and_directory_aliases_are_rejected(self):
        w = self.world()
        rel = w.relocation()
        python_file = w.new + '/torch/mod.py'
        value = os.stat(python_file)
        self.assertIn(python_file, rel._to_original)
        self.assertNotIn(python_file, rel.installed_natives)
        self.rejects('outside declared natives', rel.invert_mappings, {python_file: (value.st_dev, value.st_ino)})
        torch = Path(w.new) / 'torch'
        real = Path(w.new) / 'torch_real'
        torch.rename(real)
        torch.symlink_to(real)
        try:
            self.rejects('nonsymlink', rel.fresh, python_file)
            self.rejects('nonsymlink', rel.fresh, w.new + '/torch/_C.so')
        finally:
            torch.unlink()
            real.rename(torch)
        rel.fresh(python_file)

    def test_inode_identity_is_fresh_and_uncached(self):
        w = self.world()
        rel = w.relocation()
        origins, live = w.observe(rel)
        self.assertEqual(rel.invert_mappings(live), rel.invert_mappings(live))
        path = w.new + '/torch/_C.so'
        device, inode = live[path]
        for wrong in ((device, inode + 1), (device + 1, inode), (device, 0), [device, inode], (device, str(inode)), (True, inode)):
            with self.subTest(wrong):
                self.rejects('inode|identity|malformed', rel.invert_mappings, {**live, path: wrong})
        replacement = Path(path + '.new')
        replacement.write_bytes(MEMBERS['torch/_C.so'])
        os.replace(replacement, path)
        self.rejects('inode/dev differs', rel.invert_mappings, live)
        fresh_origins, fresh_live = w.observe(rel)
        rel.invert_mappings(fresh_live)
        rel.invert_origins(fresh_origins, packages=rel.original_packages)
        self.rejects('live mapping dict', rel.invert_mappings, list(fresh_live.items()))

    def test_duplicate_inode_and_hardlink_aliases_are_rejected(self):
        w = self.world()
        rel = w.relocation()
        a, b = Path(w.new) / 'torch/twin_a.py', Path(w.new) / 'torch/twin_b.py'
        b.unlink()
        os.link(a, b)
        origins, _ = w.observe(rel)
        origins['modules'].update({'torch.a': str(a), 'torch.b': str(b)})
        origins['files'].update({str(a): digest(MEMBERS['torch/twin_a.py']), str(b): digest(MEMBERS['torch/twin_b.py'])})
        self.rejects('duplicate installed inode', rel.invert_origins, origins, packages=rel.original_packages)
        native_a, native_b = Path(w.new) / 'torch/_C.so', Path(w.new) / 'numpy/core.so'
        self.assertNotEqual(MEMBERS['torch/_C.so'], MEMBERS['numpy/core.so'])
        live = {str(native_a): (os.stat(native_a).st_dev, os.stat(native_a).st_ino),
                str(native_b): (os.stat(native_a).st_dev, os.stat(native_a).st_ino)}
        self.rejects('inode/dev differs', rel.invert_mappings, live)

    def test_installed_file_that_is_a_hardlink_of_the_original_wheel_file_is_rejected(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        origins, live = w.observe(rel)
        rel.invert_origins(origins, packages=rel.original_packages)
        for relative in ('torch/mod.py', 'torch/_C.so'):
            installed, original = Path(w.new) / relative, Path(w.old) / relative
            installed.unlink()
            os.link(original, installed)
        origins, live = w.observe(rel)
        self.rejects('aliases original wheel file', rel.invert_origins, origins, packages=rel.original_packages)
        self.rejects('aliases original wheel file', rel.invert_mappings, live)

    def test_inverse_never_opens_or_reads_original_wheel_bytes(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        origins, live = w.observe(rel)
        opened = []
        real_open, real_os_open = builtins.open, os.open
        def spy_open(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)
        def spy_os_open(path, *args, **kwargs):
            opened.append(str(path))
            return real_os_open(path, *args, **kwargs)
        with patch.object(builtins, 'open', spy_open), patch.object(os, 'open', spy_os_open), \
                patch.object(Path, 'open', lambda self, *a, **k: opened.append(str(self))), \
                patch.object(Path, 'read_bytes', lambda self, *a, **k: opened.append(str(self))):
            rel.invert_origins(origins, packages=rel.original_packages)
            rel.invert_mappings(live)
            rel.fresh(w.new + '/torch/mod.py')
            self.rejects('original selected-wheel', rel.to_original, w.old + '/torch/mod.py')
        self.assertEqual(opened, [])

    def test_module_source_has_no_reads_execution_or_foreign_imports(self):
        tree = ast.parse((HERE / 'connected_installed_native_relocation.py').read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(names & {'exec', 'eval', 'compile', 'open', '__import__', 'setattr', 'globals', 'vars'})
        self.assertFalse(attributes & {'read_bytes', 'read_text', 'open', 'import_module', 'sha256_file', 'file_digest'})
        imports = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imports |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertEqual(imports, {'ast', 'hashlib', 'json', 'math', 'os', 're', 'stat', 'types'})
        self.assertEqual({n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                          and isinstance(n.func.value, ast.Name) and n.func.value.id == 'os'}, {'lstat'})


class Seams(Base):
    def test_genuine_seam_sources_are_pinned_and_mutations_are_rejected(self):
        for name, expected in SEAMS.items():
            raw = (HERE / name).read_bytes()
            reloc.require_seams(raw, expected)
            for qualified, digest_ in expected.items():
                with self.subTest(qualified):
                    self.assertEqual(reloc.seam_sha256(raw, qualified), digest_)
                    text = raw.decode().splitlines()
                    function = next(n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.FunctionDef) and n.name == qualified.rpartition('.')[2])
                    text[function.end_lineno - 1] += '  # mutated'
                    self.rejects('seam source differs', reloc.require_seams, '\n'.join(text).encode(), {qualified: digest_})
        raw = (HERE / 'qualify_siglip2_substrate_cpu.py').read_bytes()
        self.rejects('exactly one genuine seam', reloc.seam_sha256, raw, 'missing')
        self.rejects('exactly one genuine class', reloc.seam_sha256, raw, 'Missing.collect')
        self.rejects('exactly one genuine seam', reloc.seam_sha256, b'def f():\n    pass\ndef f():\n    pass\n', 'f')
        self.rejects('expected seam digests', reloc.require_seams, raw, {})
        self.rejects('SHA', reloc.require_seams, raw, {'canonical': 'abc'})

    def test_unconsumed_seams_are_where_the_contract_says(self):
        authority = ast.parse((HERE / 'connected_control_native_authority.py').read_text())
        klass = next(n for n in authority.body if isinstance(n, ast.ClassDef) and n.name == 'CombinedAuthority')
        mappings = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'mappings')
        self.assertEqual([a.arg for a in mappings.args.args], ['self'])
        self.assertIn('/proc/self/maps', {n.value for n in ast.walk(mappings) if isinstance(n, ast.Constant)})
        nearest = (HERE / 'train_siglip2_nearest_ranking.py').read_text()
        self.assertIn("for module in (old, fitter, legacy['original'], legacy['source_driver'], legacy['extract']):\n"
                      "        path = Path(module.__file__)\n"
                      "        require(module.__spec__ is not None and Path(module.__spec__.origin) == path and\n"
                      "                path.is_absolute() and str(path) in guards, 'native dependency source origin differs')", nearest)
        for name, text in (('train_siglip2_quadratic_readout.py',
                            "    for path, digest in context['guards'].items():\n        admission.bound_file({}, path, digest)"),
                           ('evaluate_siglip2_connected_mlp.py',
                            "    merge_guards(context['guards'],t['legacy']['origins']['files'])"),
                           ('train_siglip2_connected_mlp.py',
                            "        for path,digest in context['guards'].items():\n            bound_file({},path,digest)")):
            self.assertIn(text, (HERE / name).read_text(), name)
        quadratic = (HERE / 'train_siglip2_quadratic_readout.py').read_text()
        self.assertIn("    for path, digest in origins['files'].items():\n"
                      "        if admission is None:\n"
                      "            bound_file(context['guards'], path, digest)", quadratic)


class UnchangedCollect(Base):
    """The genuine collect() predicates, unchanged, against the exact inverse."""
    def combined(self, packages, historical, driver, cls=None):
        authority = (cls or native.CombinedAuthority).__new__(cls or native.CombinedAuthority)
        authority.check = lambda: None  # frozen-state check is unrelated to relocation
        authority.context = {'legacy': {'extract': extract, 'selected': {'packages': packages}, 'source_driver': driver}}
        authority.historical, authority.files, authority.identities = historical, {}, {}
        authority.admitted, authority.inventory, authority.record = False, None, {'library': {'path': '/none'}}
        return authority

    def historical(self, w, rel):
        return {'files': dict(w.environment()['files']), 'modules': dict(w.expected_origins(rel)['modules'])}

    def test_original_space_control_passes_and_installed_space_is_the_causal_blocker(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        historical = self.historical(w, rel)
        packages = rel.original_packages
        # Control: genuine driver, original packages, original tree, historical tables.
        with patch.dict(sys.modules, fake_modules(w.old)), fake_maps(w.native_lines(w.old)):
            result = self.combined(packages, historical, qualifier).collect(extract, packages)
        self.assertEqual(result, w.expected_origins(rel))
        # The adapter's exact inverse is unusable with the unchanged live-space mappings().
        adapter = SimpleNamespace(imported_origins=lambda e, p: rel.invert_origins(
            qualifier.imported_origins(e, rel.installed_packages), packages=p))
        with patch.dict(sys.modules, fake_modules(w.new)), fake_maps(w.native_lines(w.new)):
            self.rejects('complete genuine supplemental native mapping inventory',
                         self.combined(packages, historical, adapter).collect, extract, packages)
            # Without the inverse, the genuine historical predicates reject installed-space origins.
            self.rejects('loaded native module origin differs',
                         self.combined(packages, historical, qualifier).collect, extract, packages)

    def test_exact_inverse_passes_every_unchanged_genuine_predicate_with_one_mappings_seam(self):
        w = self.world(with_old=True)
        rel = w.relocation()
        historical = self.historical(w, rel)
        packages = rel.original_packages

        class SeamDemonstration(native.CombinedAuthority):  # test-only: genuine mappings() runs first, keys inverted after
            def mappings(self):
                return rel.invert_mappings(super().mappings())
        adapter = SimpleNamespace(imported_origins=lambda e, p: rel.invert_origins(
            qualifier.imported_origins(e, rel.installed_packages), packages=p))
        with patch.dict(sys.modules, fake_modules(w.new)), fake_maps(w.native_lines(w.new)):
            result = self.combined(packages, historical, adapter, SeamDemonstration).collect(extract, packages)
        self.assertEqual(result, w.expected_origins(rel))
        mutants = {'unknown or changed combined native origin': lambda h: h['files'].pop(w.old + '/torch/mod.py'),
                   'unknown or changed combined module origin':
                       lambda h: h['modules'].update({'torch.mod': w.old + '/torch/other.py'})}
        for pattern, mutate in mutants.items():
            mutated = copy.deepcopy(historical)
            mutate(mutated)
            with self.subTest(pattern), patch.dict(sys.modules, fake_modules(w.new)), fake_maps(w.native_lines(w.new)), \
                    contextlib.redirect_stderr(io.StringIO()):  # collect prints its rejected-origin diagnostic
                self.rejects(pattern, self.combined(packages, mutated, adapter, SeamDemonstration).collect, extract, packages)
        path = Path(w.new) / 'torch/mod.py'
        original = path.read_bytes()
        path.write_bytes(original + b'!')
        try:
            with patch.dict(sys.modules, fake_modules(w.new)), fake_maps(w.native_lines(w.new)):
                self.rejects('fresh installed bytes differ',
                             self.combined(packages, historical, adapter, SeamDemonstration).collect, extract, packages)
        finally:
            path.write_bytes(original)


class ActualEvidence(Base):
    @classmethod
    def setUpClass(cls):
        cls.bundle = (EVIDENCE / 'connected-original-environment-audit-v1/original-bundle.json').read_bytes()
        cls.audit = (EVIDENCE / 'connected-original-environment-audit-v1/complete-record-owners.json').read_bytes()
        cls.installed = (EVIDENCE / 'connected-installed-site-filesystem-v1/installed-environment.json').read_bytes()
        cls.rel = reloc.relocation(cls.bundle, cls.installed, trusted_bundle_sha256=BUNDLE_SHA,
                                   trusted_installed_sha256=INSTALLED_SHA, trusted_ownership_audit_sha256=AUDIT_SHA,
                                   old_site=OLD_SITE, new_site=NEW_SITE)

    def test_committed_bytes_match_the_accepted_digests(self):
        self.assertEqual((digest(self.bundle), digest(self.audit), digest(self.installed)), (BUNDLE_SHA, AUDIT_SHA, INSTALLED_SHA))

    def test_installed_authority_is_the_genuine_helper_derivation(self):
        result = helper._prepare(self.bundle, self.audit, BUNDLE_SHA, AUDIT_SHA, NEW_SITE)[0]
        self.assertEqual(json.loads(self.installed), result)

    def test_finite_inventory_counts_and_mapping(self):
        original = json.loads(self.bundle)['environment']
        installed = json.loads(self.installed)['expected_environment']
        self.assertEqual((len(original['files']), len(original['native_files'])), (1767, 262))
        selected = [p for p in original['files'] if p.startswith(OLD_SITE + '/')]
        self.assertEqual((len(selected), len(original['files']) - len(selected)), (1754, 13))
        self.assertEqual(len(self.rel._anchors), 13)
        for old, sha in original['files'].items():
            new = self.rel.to_installed(old)
            self.assertEqual(self.rel.to_original(new), old)
            self.assertEqual(installed['files'][new], sha)
            self.assertEqual(new.startswith(NEW_SITE + '/'), old.startswith(OLD_SITE + '/'))
            if not old.startswith(OLD_SITE + '/'):
                self.assertEqual(new, old)
        self.assertEqual(set(self.rel._to_original), set(installed['files']))
        self.assertEqual(self.rel.installed_natives, frozenset(installed['native_files']))
        self.assertEqual(self.rel.installed_packages, installed['packages'])
        self.assertEqual(self.rel.original_packages, original['packages'])

    def test_thirteen_external_anchors_are_exactly_the_audited_system_members(self):
        audit = json.loads((EVIDENCE / 'connected-original-environment-audit-v1/verification.json').read_bytes())
        system = audit['out_of_six_package_roots']['system_members']
        self.assertEqual(len(system), 13)
        self.assertEqual({p: self.rel._hash[p] for p in self.rel._anchors}, system)
        for anchor in system:
            self.assertEqual((self.rel.to_original(anchor), self.rel.to_installed(anchor)), (anchor, anchor))
        self.rejects('unknown or foreign', self.rel.to_original, '/usr/lib/aarch64-linux-gnu/libm.so.7')
        self.rejects('unknown or foreign', self.rel.to_original, NEW_SITE + '/libc.so.6')

    def test_exact_four_cudnn_members_are_relocated_with_their_accepted_digests(self):
        proof = json.loads((EVIDENCE / 'prototype-residual-ridge-v1/cudnn-wheel-provenance-v1/proof.json').read_bytes())
        self.assertEqual(proof['authority']['installed_site_root'], OLD_SITE)
        members = proof['comparison']['selected_members']
        self.assertEqual(len(members), 4)
        for name, value in members.items():
            old, new = f'{OLD_SITE}/{name}', f'{NEW_SITE}/{name}'
            self.assertEqual((self.rel.to_installed(old), self.rel.to_original(new)), (new, old))
            self.assertEqual(self.rel._hash[old], value['sha256'])
            self.assertIn(new, self.rel.installed_natives)
            self.rejects('original selected-wheel', self.rel.to_original, old)

    def test_candidate_graph_extensions_are_not_in_the_accepted_inventory(self):
        files = json.loads(self.bundle)['environment']['files']
        for needle in ('_imagingcms', 'hf_xet', 'charset_normalizer/md.'):
            self.assertFalse([p for p in files if needle in p], needle)
        for name in ('PIL/_imagingcms.cpython-313-aarch64-linux-gnu.so', 'hf_xet/hf_xet.abi3.so',
                     'charset_normalizer/md.cpython-313-aarch64-linux-gnu.so'):
            self.rejects('unknown or foreign', self.rel.to_original, f'{NEW_SITE}/{name}')
            self.rejects('unknown or foreign', self.rel.to_installed, f'{OLD_SITE}/{name}')

    def test_actual_authority_binding_mutations_are_rejected(self):
        arguments = dict(original_bundle=self.bundle, installed_authority=self.installed, trusted_bundle_sha256=BUNDLE_SHA,
                         trusted_installed_sha256=INSTALLED_SHA, trusted_ownership_audit_sha256=AUDIT_SHA,
                         old_site=OLD_SITE, new_site=NEW_SITE)
        self.rejects('trusted bundle', reloc.relocation, **{**arguments, 'original_bundle': self.bundle + b' '})
        self.rejects('trusted installed', reloc.relocation, **{**arguments, 'installed_authority': self.installed + b' '})
        self.rejects('installed authority binding', reloc.relocation, **{**arguments, 'trusted_ownership_audit_sha256': '0' * 64})
        self.rejects('installed authority binding', reloc.relocation, **{**arguments, 'new_site': NEW_SITE.replace('site-v2', 'site-v1')})
        mutated = json.loads(self.installed)
        mutated['expected_environment']['files'][NEW_SITE + '/torch/__init__.py'] = '0' * 64
        raw = json.dumps(mutated).encode()
        self.rejects('exact image', reloc.relocation, **{**arguments, 'installed_authority': raw, 'trusted_installed_sha256': digest(raw)})


if __name__ == '__main__':
    unittest.main()
