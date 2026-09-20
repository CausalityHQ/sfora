from __future__ import annotations

import unittest

import numpy as np
from _scratch_flip_view_matched_gate import _decision, _validate_matched_features


class FlipViewMatchedGateTests(unittest.TestCase):
    def test_decision_requires_information_effect_and_positive_interval(self) -> None:
        datasets = {
            "food101": {"rows": 4_000, "mean_view_cosine": 0.91, "recall_at_1_delta": 0.004},
            "pet": {"rows": 3_000, "mean_view_cosine": 0.93, "recall_at_1_delta": 0.003},
        }

        passing = _decision(datasets, bootstrap_lower_95=1e-6)
        self.assertTrue(passing["passed"])
        self.assertGreaterEqual(passing["pooled_recall_at_1_delta"], 0.003)

        self.assertFalse(
            _decision(datasets, bootstrap_lower_95=0.0)["passed"],
        )
        too_similar = {**datasets, "pet": {**datasets["pet"], "mean_view_cosine": 0.98}}
        self.assertFalse(_decision(too_similar, bootstrap_lower_95=0.01)["passed"])
        too_small = {
            name: {**row, "recall_at_1_delta": 0.0029}
            for name, row in datasets.items()
        }
        self.assertFalse(_decision(too_small, bootstrap_lower_95=0.01)["passed"])

    def test_matched_features_require_same_finite_unit_shape(self) -> None:
        identity = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        flipped = identity.copy()

        evidence = _validate_matched_features(identity, flipped, expected_rows=2)
        self.assertEqual(evidence["dimensions"], 2)
        self.assertEqual(evidence["identity_max_unit_norm_error"], 0.0)

        with self.assertRaisesRegex(ValueError, "shape"):
            _validate_matched_features(identity, flipped[:1], expected_rows=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            nonfinite = np.asarray([[np.nan, 0.0], [0.0, 1.0]])
            _validate_matched_features(identity, nonfinite, expected_rows=2)
        with self.assertRaisesRegex(ValueError, "unit norm"):
            _validate_matched_features(identity * 0.5, flipped, expected_rows=2)


if __name__ == "__main__":
    unittest.main()
