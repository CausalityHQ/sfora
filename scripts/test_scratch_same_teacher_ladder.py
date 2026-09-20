"""Narrow contracts for the throwaway same-teacher representation ladder."""

from __future__ import annotations

import unittest

import _scratch_same_teacher_ladder as subject
import numpy as np
import torch
from probe_sop_relational_linear import score_symmetric

from sfora.joint_relational_compaction import pack_int8_unit_embeddings


class SameTeacherLadderTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_effect_aware(self) -> None:
        candidate = np.asarray([0.4, 0.6, 0.8, 1.0], dtype=np.float64)
        baseline = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        first = subject.paired_query_bootstrap(
            candidate, baseline, seed=17, replicates=2_000
        )
        second = subject.paired_query_bootstrap(
            candidate, baseline, seed=17, replicates=2_000
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta"], 0.45)
        self.assertGreater(first["ci95"][0], 0.0)
        self.assertTrue(first["ci_excludes_zero"])
        self.assertTrue(first["meets_minimum_effect"])

    def test_paired_bootstrap_rejects_unpaired_or_nonfinite_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "paired query authority"):
            subject.paired_query_bootstrap(
                np.asarray([1.0]), np.asarray([1.0, 2.0]), seed=1, replicates=10
            )
        with self.assertRaisesRegex(ValueError, "paired query authority"):
            subject.paired_query_bootstrap(
                np.asarray([np.nan]), np.asarray([0.0]), seed=1, replicates=10
            )

    def test_score_uses_exact_packed_int8_geometry(self) -> None:
        rows = torch.nn.functional.normalize(
            torch.tensor(
                [[1.0, 0.1], [0.9, 0.2], [-1.0, 0.0], [-0.9, -0.2]],
                dtype=torch.float32,
            ),
            dim=1,
        )
        labels = (1, 1, 2, 2)
        packed = pack_int8_unit_embeddings(rows)
        observed = subject._score(
            packed,
            packed,
            labels,
            labels,
            same_rows=True,
            device=torch.device("cpu"),
        )
        expected = score_symmetric(
            packed, labels, candidate_width=1, device=torch.device("cpu")
        )
        self.assertEqual(observed["per_query_ap"], list(expected["per_query_ap"]))
        self.assertEqual(observed["per_query_r1"], list(expected["per_query_r1"]))


if __name__ == "__main__":
    unittest.main()
