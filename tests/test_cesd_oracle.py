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
    singleton_confusion = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.995, 0.1], [-0.8, 0.6]],
        dtype=torch.float32,
    )
    singleton_confusion = torch.nn.functional.normalize(singleton_confusion, dim=1)
    for row, label in zip(singleton_confusion, (84, 84, 87, 87), strict=True):
        vector = torch.zeros(16, dtype=torch.float32)
        vector[2:4] = row
        descriptors.append(vector)
        labels.append(label)
    remaining_labels = (85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97)
    for dimension, label in enumerate(remaining_labels, start=4):
        for _ in range(2):
            vector = torch.zeros(16, dtype=torch.float32)
            vector[dimension] = 1.0
            descriptors.append(vector)
            labels.append(label)
    return torch.stack(descriptors), torch.tensor(labels, dtype=torch.int64)


def test_cesd_oracle_counts_only_literal_query_shrinkage_rescues() -> None:
    descriptors, labels = _burned_fixture()

    observed = tuple(
        score_cesd_query_shrinkage_oracle(
            descriptors,
            labels,
            query_block=query_block,
        )
        for query_block in (1, 7, 64)
    )
    assert observed[1:] == observed[:1] * 2
    evidence = observed[0]

    assert evidence.query_count == 32
    assert evidence.baseline_hits == 24
    assert evidence.shrinkage_hits == 10
    assert evidence.oracle_hits == 26
    assert evidence.rescued_query_rows == (0, 3)
    assert evidence.oracle_selected_shrinkage_rows == (0, 3)
    assert evidence.broken_query_rows == (
        8,
        9,
        10,
        11,
        14,
        15,
        16,
        17,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    )
    assert evidence.baseline_recall_ppm == 750_000
    assert evidence.oracle_recall_ppm == 812_500
    assert evidence.gain_ppm == 62_500
    assert evidence.gain_gate_met


def test_cesd_oracle_gain_gate_stays_closed_without_a_rescue() -> None:
    descriptors = torch.eye(16, dtype=torch.float32).repeat_interleave(2, dim=0)
    labels = torch.arange(82, 98, dtype=torch.int64).repeat_interleave(2)

    evidence = score_cesd_query_shrinkage_oracle(descriptors, labels, query_block=32)

    assert evidence.baseline_hits == 32
    assert evidence.rescued_query_rows == ()
    assert evidence.gain_ppm == 0
    assert not evidence.gain_gate_met
