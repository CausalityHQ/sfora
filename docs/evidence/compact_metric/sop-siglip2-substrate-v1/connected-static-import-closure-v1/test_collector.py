import base64
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('source_collector', Path(__file__).with_name('collector.py'))
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class Closure(unittest.TestCase):
    def fixture(self, root, name, sources):
        rows = []
        for relative, text in sources.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = text.encode()
            path.write_bytes(raw)
            digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
            rows.append(f'{relative},sha256={digest},{len(raw)}\n')
        record = root / (name + '-1.dist-info/RECORD')
        record.parent.mkdir()
        record.write_text(''.join(rows))
        return str(record), {'name': name, 'sha256': hashlib.sha256(record.read_bytes()).hexdigest()}

    def settings(self, root):
        record, profile = self.fixture(root, 'demo', {
            'demo/__init__.py': 'from .sub import child\n',
            'demo/sub/__init__.py': 'from . import child\n',
            'demo/sub/child.py': 'import other\nimport native\nimport importlib\nimportlib.import_module("dynamic")\n',
            'native.cpython-313-aarch64-linux-gnu.so': 'not executable',
        })
        foreign_record, foreign_profile = self.fixture(root, 'other', {'other.py': 'raise RuntimeError("never execute")\n'})
        path = str(root / 'demo/__init__.py')
        return {'site': str(root), 'records': {record: profile, foreign_record: foreign_profile},
                'seeds': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()},
                'candidate_distributions': ['demo']}

    def test_relative_child_foreign_native_dynamic_and_no_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            settings = self.settings(root)
            result = collector.collect(settings)
            self.assertEqual(len(result['sources']), 3)
            self.assertEqual(result['outside_candidate_distributions'], [('other', 'other')])
            self.assertIn('native', result['native_import_candidates'])
            self.assertEqual(len(result['dynamic_import_sites']), 1)
            self.assertFalse(result['native_qualified'])
            settings['candidate_distributions'].append('other')
            self.assertEqual(len(collector.collect(settings)['sources']), 4)

    def test_source_mutation_record_mutation_symlink_and_duplicate_owner(self):
        for mode in ['source', 'record', 'symlink', 'owner']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                settings = self.settings(root)
                source = root / 'demo/sub/child.py'
                if mode == 'source':
                    source.write_text('changed bytes')
                elif mode == 'record':
                    Path(next(iter(settings['records']))).write_text('changed bytes')
                elif mode == 'symlink':
                    moved = root / 'saved.py'
                    source.rename(moved)
                    source.symlink_to(moved)
                else:
                    record, profile = self.fixture(root, 'duplicate', {'demo/sub/child.py': source.read_text()})
                    settings['records'][record] = profile
                with self.assertRaises(ValueError):
                    collector.collect(settings)

    def test_final_fresh_read_rejects_changed_consumed_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            settings = self.settings(root)
            original, counts = collector.read_exact, {}
            def read(path, digest, size=None):
                path = str(path)
                counts[path] = counts.get(path, 0) + 1
                if path.endswith('child.py') and counts[path] == 2:
                    Path(path).write_text('changed after parse')
                return original(path, digest, size)
            collector.read_exact = read
            try:
                with self.assertRaises(ValueError):
                    collector.collect(settings)
            finally:
                collector.read_exact = original


if __name__ == '__main__':
    unittest.main()
