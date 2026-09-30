"""Source admission negatives; actual native qualification runs only on DGX."""
import copy
import unittest

import late_dense_boundary as late


class SourceAdmission(unittest.TestCase):
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
