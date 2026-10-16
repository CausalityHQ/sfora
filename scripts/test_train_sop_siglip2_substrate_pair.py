"""Light source checks for the SOP paired-substrate trainer (stdlib only; run on the DGX)."""
import ast
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW = ROOT / 'scripts/train_sop_siglip2_substrate_pair.py'


def arms():
    tree = ast.parse(NEW.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ARMS')
    return ast.literal_eval(node.value)


class Pair(unittest.TestCase):
    def test_large_arm_is_incumbent(self):
        large = arms()['large']
        old = subprocess.run(['git', '-C', str(ROOT), 'show', '29a45d38:scripts/train_sop_siglip2_compact.py'],
                             capture_output=True, text=True, check=True).stdout
        for name, value in large['model_hashes'].items():
            self.assertIn(f'"{name}": "{value}"', old)
        self.assertEqual((large['width'], large['crop'], large['frozen_blocks'], large['blocks']), (1024, 256, 12, 24))
        self.assertIn('layers[:12]', old)

    def test_so400_arm(self):
        so = arms()['so400']
        self.assertEqual((so['width'], so['crop'], so['frozen_blocks'], so['blocks']), (1152, 384, 13, 27))
        self.assertEqual(so['revision'], 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f')

    def test_only_substrate_wiring_differs(self):
        old = subprocess.run(['git', '-C', str(ROOT), 'show', '29a45d38:scripts/train_sop_siglip2_compact.py'],
                             capture_output=True, text=True, check=True).stdout.splitlines()
        new = NEW.read_text().splitlines()
        import difflib
        changed = [l[1:] for l in difflib.unified_diff(old, new, lineterm='', n=0)
                   if l[:1] in '+-' and not l.startswith(('+++', '---'))]
        allowed = ('ARM', 'MODEL_HASHES', 'substrate', 'crop', 'frozen_blocks', 'blocks', 'width', 'revision',
                   'model_hashes', 'config.json', 'preprocessor_config.json', 'model.safetensors', 'train_sop_siglip2',
                   'schema', '"so400"', '"large"', '}', '},', '# ', 'raise ValueError("SOP SigLIP2 vision depth',
                   'global ARM', 'or (args.substrate', 'layers[', 'MODEL_HASHES = {', 'RandomResizedCrop', 'or processor.size')
        for line in changed:
            if line.strip():
                self.assertTrue(any(tok in line for tok in allowed), line)


if __name__ == '__main__':
    unittest.main()
