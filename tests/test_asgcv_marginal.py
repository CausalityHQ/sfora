from __future__ import annotations

import numpy as np
import pytest

from sfora.asgcv_marginal import (
    AsgcvVisionCutAuthority,
    canonical_marginal_gradient_sample_bytes,
    validate_marginal_gradient_sample_inputs,
)


def _cut() -> AsgcvVisionCutAuthority:
    return AsgcvVisionCutAuthority(
        boundary_names=("merger", "deepstack-0", "deepstack-1", "deepstack-2"),
        images=2,
        patches_per_boundary=3,
        channel_dimensions=4,
    ).validated()


def _arrays(*, zero: bool) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4) / 17
    gradient = np.zeros_like(tokens) if zero else np.flip(tokens, axis=-1).copy() / 13
    return tokens, gradient


@pytest.mark.parametrize("zero", [False, True])
def test_marginal_gradient_sample_binds_complete_cut_and_zero_semantics(zero: bool) -> None:
    tokens, gradient = _arrays(zero=zero)
    raw = canonical_marginal_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256="4" * 64,
        completion_protocol_sha256="5" * 64,
        marginal_schedule_sha256="6" * 64,
        pooler_state_sha256="7" * 64,
        candidate_pair_ordinal=5,
        pair_ordinals=(17, 29),
        relation_sign=-1,
        zero_semantic_target=zero,
        grpo_loss=0.0 if zero else 0.125,
        attention_kl=0.0 if zero else 0.375,
        generated_tokens=0 if zero else 64,
        vision_cut_authority=_cut(),
        patch_tokens=tokens,
        exact_gradient=gradient,
    )
    value = validate_marginal_gradient_sample_inputs(
        raw,
        patch_tokens=tokens,
        exact_gradient=gradient,
    )
    assert value["schema"] == "sfora-asgcv-marginal-gradient-sample-v1"
    assert value["zero_semantic_target"] is zero
    assert value["replay_branch_count"] == (0 if zero else 8)
    assert value["vision_cut_authority"] == _cut().to_mapping()
    assert value["arrays"]["patch_tokens"]["shape"] == [2, 12, 4]

    wrong = gradient.copy()
    wrong[0, 0, 0] = np.float32(1.0 if zero else 0.0)
    with pytest.raises(ValueError):
        validate_marginal_gradient_sample_inputs(
            raw,
            patch_tokens=tokens,
            exact_gradient=wrong,
        )


def test_vision_cut_refuses_missing_reordered_or_shape_drift() -> None:
    mapping = _cut().to_mapping()
    for mutation in (
        {**mapping, "boundary_names": mapping["boundary_names"][:3]},
        {
            **mapping,
            "boundary_names": [
                "deepstack-0",
                "merger",
                "deepstack-1",
                "deepstack-2",
            ],
        },
        {**mapping, "images": True},
        {**mapping, "patches_per_boundary": 0},
    ):
        with pytest.raises(ValueError):
            AsgcvVisionCutAuthority.from_mapping(mutation)
