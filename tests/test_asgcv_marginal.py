from __future__ import annotations

import numpy as np
import pytest

from sfora.asgcv_marginal import (
    AsgcvVisionCutAuthority,
    canonical_marginal_gradient_sample_bytes,
    validate_marginal_gradient_sample_context,
    validate_marginal_gradient_sample_inputs,
)
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvRolloutAuthority,
    assemble_asgcv_marginal_schedule,
    build_asgcv_pair_schedule,
    classify_asgcv_completion_group,
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


def test_marginal_sample_context_binds_candidate_and_zero_outcome() -> None:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11,), different_prefix_ids=(21,), terminal_token_ids=(99,)
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="8" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    example_ids = tuple(f"cars-{index:02d}" for index in range(16))
    labels = tuple(index // 2 for index in range(16))
    candidates = build_asgcv_pair_schedule(
        example_ids, labels, schedule_seed_sha256="9" * 64, pair_count=8
    )
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11,) if pair.relation_sign == 1 else (21,)),
                    30 + rollout_ordinal,
                    99,
                )
                for rollout_ordinal in range(8)
            ),
            pair.relation_sign,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=pair.ordinal,
        )
        for pair in candidates.pairs
    )
    schedule = assemble_asgcv_marginal_schedule(candidates, groups)
    pair = candidates.pairs[0]
    tokens, gradient = _arrays(zero=True)
    raw = canonical_marginal_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=groups[0].sha256(),
        completion_protocol_sha256=protocol.sha256(),
        marginal_schedule_sha256=schedule.sha256(),
        pooler_state_sha256="7" * 64,
        candidate_pair_ordinal=0,
        pair_ordinals=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        zero_semantic_target=True,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=0,
        vision_cut_authority=_cut(),
        patch_tokens=tokens,
        exact_gradient=gradient,
    )
    assert validate_marginal_gradient_sample_context(
        raw,
        marginal_schedule=schedule,
        candidate_schedule=candidates,
        completion_groups=groups,
    )["candidate_pair_ordinal"] == 0

    with pytest.raises(ValueError):
        validate_marginal_gradient_sample_context(
            raw,
            marginal_schedule=schedule,
            candidate_schedule=candidates,
            completion_groups=groups[1:] + groups[:1],
        )
