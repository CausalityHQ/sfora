from __future__ import annotations

import torch

from sfora.cesd_oracle import score_cesd_query_shrinkage_oracle


def _burned_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    descriptors: list[torch.Tensor] = []
    labels: list[int] = []
    confused = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.995, 0.1], [-0.8, 0.6]],
        dtype=torch.float32,
    )
    confused = torch.nn.functional.normalize(confused, dim=1)
    for row, label in zip(confused, (82, 82, 83, 83), strict=True):
        vector = torch.zeros(16, dtype=torch.float32)
        vector[:2] = row
        descriptors.append(vector)
        labels.append(label)
    for dimension, label in enumerate(range(84, 98), start=2):
        for _ in range(2):
            vector = torch.zeros(16, dtype=torch.float32)
            vector[dimension] = 1.0
            descriptors.append(vector)
            labels.append(label)
    return torch.stack(descriptors), torch.tensor(labels, dtype=torch.int64)


def test_cesd_oracle_counts_only_literal_query_shrinkage_rescues() -> None:
    descriptors, labels = _burned_fixture()

    evidence = score_cesd_query_shrinkage_oracle(
        descriptors,
        labels,
        query_block=7,
    )

    assert evidence.query_count == 32
    assert evidence.baseline_hits == 28
    assert evidence.shrinkage_hits == 14
    assert evidence.oracle_hits == 30
    assert evidence.rescued_query_rows == (0, 3)
    assert evidence.alpha_zero_query_rows == (0, 3)
    assert evidence.baseline_recall_ppm == 875_000
    assert evidence.oracle_recall_ppm == 937_500
    assert evidence.gain_ppm == 62_500
    assert evidence.passed
