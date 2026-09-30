"""Stdlib authority negatives; native constructor proof runs on DGX only."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import qualify_image_queue_adaptation_cpu as cpu


class AuthorityTests(unittest.TestCase):
    def test_closure_rejects_changed_code_prefix_and_members(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = {f"old{i}.py": "" for i in range(114)}
            for name in original:
                (root / name).write_text(name)
                original[name] = cpu.sha(root / name)
            prior = root / "late-dense-execution.json"
            prior.write_text(json.dumps(original))
            code = dict(original)
            for name in cpu.ADDED:
                (root / name).write_text(name)
                code[name] = cpu.sha(root / name)
            manifest = root / "image-queue-cpu-execution.json"
            manifest.write_text(json.dumps(code))
            with patch.object(cpu, "ORIGINAL", cpu.sha(prior)):
                expected = cpu.sha(manifest)
                self.assertEqual(cpu.closure(root, expected), code)
                with self.assertRaises(AssertionError): cpu.closure(root, "foreign")
                (root / "old0.py").write_text("mutation")
                with self.assertRaises(AssertionError): cpu.closure(root, expected)
                (root / "old0.py").write_text("old0.py")
                changed = dict(code); changed["old0.py"] = "foreign"
                manifest.write_text(json.dumps(changed))
                with self.assertRaises(AssertionError): cpu.closure(root, cpu.sha(manifest))
                changed = dict(code); changed.pop(next(iter(cpu.ADDED)))
                manifest.write_text(json.dumps(changed))
                with self.assertRaises(AssertionError): cpu.closure(root, cpu.sha(manifest))
                prior.write_text("foreign")
                with self.assertRaises(AssertionError): cpu.closure(root, cpu.sha(manifest))

    def test_class_sequence_hash_is_order_sensitive(self):
        self.assertEqual(cpu.sha_bytes([[1, 2]]), cpu.sha_bytes([[1, 2]]))
        self.assertNotEqual(cpu.sha_bytes([[1, 2]]), cpu.sha_bytes([[2, 1]]))


if __name__ == "__main__":
    unittest.main()
