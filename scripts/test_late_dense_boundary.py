"""Source admission negatives; actual native qualification runs only on DGX."""
import copy
import subprocess
import sys
import unittest
from pathlib import Path

import late_dense_boundary as late


class SourceAdmission(unittest.TestCase):
    def test_frozen_prefix_and_optimized_entry(self):
        class Vision:
            def state_dict(self):
                return {"encoder.layers.9.weight": 9, "encoder.layers.10.weight": 10}

            def named_buffers(self):
                return [("embeddings.position_ids", 0)]

        self.assertEqual(set(late.frozen_state(Vision(), 10)),
                         {"encoder.layers.9.weight", "embeddings.position_ids"})
        self.assertIn("encoder.layers.10.weight", late.frozen_state(Vision(), 12))
        entry = Path(__file__).with_name("qualify_late_dense_adaptation_cpu.py")
        result = subprocess.run([sys.executable, "-O", str(entry), "--help"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Native qualification requires assertions", result.stderr)

    def test_incomplete_or_foreign_resume_rejected(self):
        identity = {"arm": "half", "width": 128, "total_updates": 1000}
        saved = {name: {} for name in late.RESUME_KEYS}
        saved["identity"] = {**identity, "global_step": 1000}
        saved["vision"] = {str(i): None for i in range(400)}
        saved["optimizer"] = {"state": {i: {"step": 1000} for i in range(208)}}
        late.validate_source(saved, identity)
        for mutation in ("missing_bank", "wrong_step", "weights_only", "wrong_identity"):
            bad = copy.deepcopy(saved)
            if mutation == "missing_bank":
                del bad["bank"]
            elif mutation == "wrong_step":
                bad["optimizer"]["state"][0]["step"] = 999
            elif mutation == "weights_only":
                bad["optimizer"]["state"] = {}
            else:
                bad["identity"]["arm"] = "full"
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                late.validate_source(bad, identity)


if __name__ == "__main__":
    unittest.main()
