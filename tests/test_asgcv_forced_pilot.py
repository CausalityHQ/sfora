from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sfora.asgcv_forced_pilot import (
    ASGCV_FORCED_ACCURACY_GATE_PPM,
    ASGCV_FORCED_AUC_GATE_PPM,
    AsgcvForcedObservation,
    AsgcvForcedResult,
    canonical_asgcv_forced_observation_bytes,
    canonical_asgcv_forced_result_bytes,
)


def _observation(ordinal: int, *, correct: bool = True) -> AsgcvForcedObservation:
    relation = 1 if ordinal % 2 == 0 else -1
    signed_gap = 1.0 + ordinal / 100.0
    if not correct:
        signed_gap = -signed_gap
    gap = relation * signed_gap
    return AsgcvForcedObservation(
        source_commit="1" * 40,
        launch_authority_sha256="2" * 64,
        pilot_schedule_sha256="3" * 64,
        model_revision="4" * 40,
        fixture_sha256="5" * 64,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(2 * ordinal, 2 * ordinal + 1),
        relation_sign=relation,
        same_score=-0.5 + gap / 2.0,
        different_score=-0.5 - gap / 2.0,
        gradient_sha256=f"{ordinal + 1:064x}",
        gradient_norm=2.0,
        boundary_norms=(1.0, 1.0, 1.0, 1.0),
        prepare_elapsed_ns=10,
        replay_elapsed_ns=20,
        peak_cuda_reserved_bytes=30,
        peak_rss_bytes=40,
    ).validated()


def test_forced_observation_is_canonical_claim_ineligible_and_strict() -> None:
    observation = _observation(0)
    raw = canonical_asgcv_forced_observation_bytes(observation)
    assert raw.endswith(b"\n")
    assert json.loads(raw)["claim_eligible"] is False
    assert json.loads(raw)["official_test_access"] is False
    assert json.loads(raw)["generated_tokens"] == 0
    assert json.loads(raw)["branch_order"] == ["same", "different"]
    assert AsgcvForcedObservation.from_mapping(json.loads(raw)) == observation

    mapping = json.loads(raw)
    for mutation in (
        {**mapping, "claim_eligible": True},
        {**mapping, "official_test_access": True},
        {**mapping, "generated_tokens": 1},
        {**mapping, "branch_order": ["different", "same"]},
        {**mapping, "same_score": float("nan")},
        {**mapping, "gradient_norm": 0.0},
    ):
        with pytest.raises(ValueError):
            AsgcvForcedObservation.from_mapping(mutation)


def test_forced_result_recomputes_balanced_accuracy_auc_and_hard_gates() -> None:
    observations = tuple(_observation(ordinal) for ordinal in range(32))
    result = AsgcvForcedResult.from_observations(
        observations,
        repeat_checked_ordinals=(0, 31),
        repeat_gradient_sha256s=(observations[0].gradient_sha256, observations[31].gradient_sha256),
    )
    assert result.accuracy_ppm == 1_000_000
    assert result.same_recall_ppm == 1_000_000
    assert result.different_recall_ppm == 1_000_000
    assert result.auc_ppm == 1_000_000
    assert result.passed is True
    assert ASGCV_FORCED_ACCURACY_GATE_PPM == 625_000
    assert ASGCV_FORCED_AUC_GATE_PPM == 700_000
    raw = canonical_asgcv_forced_result_bytes(result)
    assert raw.endswith(b"\n")
    assert AsgcvForcedResult.from_mapping(json.loads(raw)) == result

    weak = tuple(
        replace(observation, same_score=-0.5, different_score=-0.5).validated()
        for observation in observations
    )
    failed = AsgcvForcedResult.from_observations(
        weak,
        repeat_checked_ordinals=(0, 31),
        repeat_gradient_sha256s=(weak[0].gradient_sha256, weak[31].gradient_sha256),
    )
    assert failed.accuracy_ppm == 0
    assert failed.auc_ppm == 500_000
    assert failed.passed is False


def test_forced_result_rejects_missing_balance_order_identity_and_repeat_drift() -> None:
    observations = tuple(_observation(ordinal) for ordinal in range(32))
    with pytest.raises(ValueError, match="ordinal"):
        AsgcvForcedResult.from_observations(
            observations[:-1],
            repeat_checked_ordinals=(0, 31),
            repeat_gradient_sha256s=(
                observations[0].gradient_sha256,
                observations[31].gradient_sha256,
            ),
        )
    with pytest.raises(ValueError, match="identity"):
        AsgcvForcedResult.from_observations(
            (replace(observations[0], source_commit="9" * 40), *observations[1:]),
            repeat_checked_ordinals=(0, 31),
            repeat_gradient_sha256s=(
                observations[0].gradient_sha256,
                observations[31].gradient_sha256,
            ),
        )
    with pytest.raises(ValueError, match="repeat"):
        AsgcvForcedResult.from_observations(
            observations,
            repeat_checked_ordinals=(0, 31),
            repeat_gradient_sha256s=("f" * 64, observations[31].gradient_sha256),
        )
