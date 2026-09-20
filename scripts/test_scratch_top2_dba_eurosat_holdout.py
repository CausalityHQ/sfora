from __future__ import annotations

import unittest
from pathlib import Path

from _scratch_top2_dba_eurosat_holdout import _partition_rows, _quality_decision


class Top2DbaEuroSatHoldoutTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_class_stratified(self) -> None:
        samples = [
            (Path(f"class-{label}/image-{row}.jpg"), label)
            for label in range(2)
            for row in range(10)
        ]

        first = _partition_rows(samples)
        second = _partition_rows(list(reversed(samples)))

        self.assertEqual(first, second)
        self.assertEqual([len(first[name]) for name in ("fit", "query", "gallery")], [8, 6, 6])
        flattened = first["fit"] + first["query"] + first["gallery"]
        self.assertEqual(len(flattened), len(set(flattened)))
        for name, expected in (("fit", 4), ("query", 3), ("gallery", 3)):
            self.assertEqual(sum(row[1] == 0 for row in first[name]), expected)
            self.assertEqual(sum(row[1] == 1 for row in first[name]), expected)

    def test_quality_decision_requires_map_gate_and_nonnegative_recall(self) -> None:
        passing = {"survives": True, "delta": 0.006, "ci95": [0.002, 0.01]}

        self.assertTrue(_quality_decision(passing, recall_delta=0.0)["passed"])
        self.assertFalse(_quality_decision(passing, recall_delta=-1e-6)["passed"])
        self.assertFalse(
            _quality_decision({**passing, "survives": False}, recall_delta=0.01)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
