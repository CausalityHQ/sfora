from __future__ import annotations

import unittest

import numpy as np
import torch
from _scratch_top2_dba_gate import _top2_dba_gallery, paired_query_bootstrap
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings


class Top2DbaGateTests(unittest.TestCase):
    def test_top2_dba_excludes_self_and_uses_lexicographic_ties(self) -> None:
        values = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            dim=1,
        )
        packed = pack_int8_unit_embeddings(values)

        augmented, neighbours = _top2_dba_gallery(packed, device=torch.device("cpu"))

        self.assertEqual(neighbours.tolist()[0], [1, 2])
        for row, indexes in enumerate(neighbours.tolist()):
            self.assertNotIn(row, indexes)
        decoded = augmented.codes.float() * augmented.inverse_norms.float()[:, None]
        expected = F.normalize(
            values + values[neighbours].mean(dim=1),
            dim=1,
        )
        self.assertGreater(float(F.cosine_similarity(decoded, expected).min()), 0.9999)

    def test_bootstrap_requires_effect_and_strictly_positive_lower_bound(self) -> None:
        baseline = np.zeros(8, dtype=np.float64)
        clear = np.full(8, 0.006, dtype=np.float64)
        too_small = np.full(8, 0.0049, dtype=np.float64)

        passing = paired_query_bootstrap(clear, baseline, seed=7, replicates=128)
        failing = paired_query_bootstrap(too_small, baseline, seed=7, replicates=128)

        self.assertTrue(passing["survives"])
        self.assertFalse(failing["survives"])
        self.assertEqual(passing["minimum_map_effect"], 0.005)


if __name__ == "__main__":
    unittest.main()
