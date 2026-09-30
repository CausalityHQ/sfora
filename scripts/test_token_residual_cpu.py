"""Stdlib authority negatives; actual tensor checks belong to the DGX qualifier."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import qualify_token_residual_cpu as cpu


class Authority(unittest.TestCase):
    def test_closure_rejects_mutation_missing_member_and_foreign_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            names = {f"original-{i}.py" for i in range(114)} | cpu.ADDED
            for name in names: (root / name).write_text(name)
            old = {n: cpu.sha(root / n) for n in names - cpu.ADDED}
            original = root / "late-dense-execution.json"
            original.write_text(json.dumps(old))
            code = {n: cpu.sha(root / n) for n in names}
            manifest = root / "token-residual-cpu-execution.json"
            manifest.write_text(json.dumps(code))
            previous = cpu.ORIGINAL
            cpu.ORIGINAL = cpu.sha(original)
            try:
                expected = cpu.sha(manifest)
                self.assertEqual(cpu.closure(root, expected), code)
                with self.assertRaises(AssertionError): cpu.closure(root, "foreign")
                changed = root / "token_residual_readout.py"
                changed.write_text("mutation")
                with self.assertRaises(AssertionError): cpu.closure(root, expected)
                changed.write_text(changed.name)
                reduced = dict(code); reduced.pop("test_token_residual_cpu.py")
                manifest.write_text(json.dumps(reduced))
                with self.assertRaises(AssertionError): cpu.closure(root, cpu.sha(manifest))
                code["original-0.py"] = "foreign"
                manifest.write_text(json.dumps(code))
                with self.assertRaises(AssertionError): cpu.closure(root, cpu.sha(manifest))
            finally:
                cpu.ORIGINAL = previous

    def test_optimized_execution_fails_before_native_import(self):
        result = subprocess.run([sys.executable, "-B", "-S", "-O", str(Path(cpu.__file__))],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Qualification requires assertions", result.stderr)


if __name__ == "__main__":
    unittest.main()
