"""Tests for optimization-only intermediate SigLIP readout selection."""

from __future__ import annotations

import json

import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_intermediate_readout import (
    score_intermediate_readout_depths,
    validate_intermediate_readout_result_bytes,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    labels = torch.tensor([label for label in range(8) for _ in range(4)], dtype=torch.int64)
    rows = F.one_hot(labels, num_classes=8).float()
    source = rows.clone()
    final = rows.clone()
    for label in (0, 2, 4, 6):
        row = int(torch.nonzero(labels == label, as_tuple=False)[0])
        final[row] = (
            0.6 * rows[row] + 0.8 * F.one_hot(torch.tensor(label + 1), num_classes=8).float()
        )
    best = rows.clone()
    weak = final.clone()
    return source, labels, tuple(F.normalize(value, dim=1) for value in (weak, best, final))


def _authority(features: torch.Tensor):
    return build_feature_split_authority(
        source_manifest_sha256="1" * 64,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=tuple(f"row-{row:03d}" for row in range(features.shape[0])),
        features=features,
    )


def test_depth_selection_uses_integer_folds_lowest_ties_and_registered_gates() -> None:
    source, labels, planes = _fixture()
    raw = score_intermediate_readout_depths(
        source,
        labels,
        planes,
        split_authority=_authority(source),
        checkpoint_sha256="2" * 64,
        feature_manifest_sha256="3" * 64,
        expected_depth_count=3,
        output_dimensions=8,
        fold_count=4,
    )
    result = validate_intermediate_readout_result_bytes(raw)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert result.schema == "sfora-siglip-intermediate-readout-v1"
    assert result.claim_eligible is False
    assert result.official_test_access is False
    assert result.selected_depth == 2
    assert result.selected_hits == 32
    assert result.final_depth_hits == 28
    assert result.selected_minus_final_ppm == 125_000
    assert result.fold_wins == 4
    assert result.replay_equal is True
    assert result.passed is True

    tied = score_intermediate_readout_depths(
        source,
        labels,
        (planes[1], planes[1], planes[2]),
        split_authority=_authority(source),
        checkpoint_sha256="2" * 64,
        feature_manifest_sha256="3" * 64,
        expected_depth_count=3,
        output_dimensions=8,
        fold_count=4,
    )
    assert validate_intermediate_readout_result_bytes(tied).selected_depth == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(selected_depth=3),
        lambda value: value.update(selected_hits=value["selected_hits"] - 1),
        lambda value: value.update(fold_wins=3),
        lambda value: value.update(passed=False),
        lambda value: value["depths"].pop(),
        lambda value: value["depths"][0].update(replay_hits=0),
    ],
)
def test_result_rejects_selection_count_replay_and_gate_mutations(mutation) -> None:
    source, labels, planes = _fixture()
    raw = score_intermediate_readout_depths(
        source,
        labels,
        planes,
        split_authority=_authority(source),
        checkpoint_sha256="2" * 64,
        feature_manifest_sha256="3" * 64,
        expected_depth_count=3,
        output_dimensions=8,
        fold_count=4,
    )
    value = json.loads(raw)
    mutation(value)
    mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

    with pytest.raises(ValueError):
        validate_intermediate_readout_result_bytes(mutated)


def test_descriptor_authority_rejects_topology_norm_nonfinite_and_evaluation_role() -> None:
    source, labels, planes = _fixture()
    cases = (
        planes[:-1],
        (*planes[:-1], planes[-1][:, :-1]),
        (*planes[:-1], planes[-1] * 2.0),
        (*planes[:-1], planes[-1].clone()),
    )
    cases[-1][-1][0, 0] = torch.nan
    for candidate in cases:
        with pytest.raises(ValueError, match="intermediate descriptor authority differs"):
            score_intermediate_readout_depths(
                source,
                labels,
                candidate,
                split_authority=_authority(source),
                checkpoint_sha256="2" * 64,
                feature_manifest_sha256="3" * 64,
                expected_depth_count=3,
                output_dimensions=8,
                fold_count=4,
            )

    clean_authority = build_feature_split_authority(
        source_manifest_sha256="1" * 64,
        role="clean-validation",
        official_test_access=False,
        ordered_example_ids=tuple(f"row-{row:03d}" for row in range(source.shape[0])),
        features=source,
    )
    with pytest.raises(ValueError):
        score_intermediate_readout_depths(
            source,
            labels,
            planes,
            split_authority=clean_authority,
            checkpoint_sha256="2" * 64,
            feature_manifest_sha256="3" * 64,
            expected_depth_count=3,
            output_dimensions=8,
            fold_count=4,
        )
