from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, replace

import numpy as np
import pytest

from sfora import prism_measurement as prism
from sfora.prism_measurement import (
    PRISM_CHANNELS,
    PrismChannelCalibration,
    PrismCueResult,
    PrismExample,
    PrismMeasurementAuthority,
    PrismMeasurementEvidence,
    PrismObservation,
    PrismObservationRow,
    PrismScoringRow,
    PrismTokenProtocol,
    build_prism_schedules,
    calibrate_prism_channels,
    canonical_prism_cue_result_bytes,
    invalid_prism_observation,
    parse_prism_completion,
    prism_calibration_receipt_sha256,
    release_prism_observation_capability,
    score_prism_cue_panel,
    validate_prism_cue_result,
    validate_prism_cue_result_bytes,
    validate_prism_observation,
    validate_prism_schedules,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _optimization_examples() -> tuple[PrismExample, ...]:
    return tuple(
        PrismExample(
            example_id=f"secret/path/optimization-{label}-{ordinal}.jpg",
            label=label,
            image_sha256=_digest(1 + label * 8 + ordinal),
        )
        for label in range(49)
        for ordinal in range(8)
    )


def _caliber_examples() -> tuple[PrismExample, ...]:
    return tuple(
        PrismExample(
            example_id=f"secret/path/caliber-{label}-{ordinal}.jpg",
            label=label,
            image_sha256=_digest(10_000 + (label - 82) * 32 + ordinal),
        )
        for label in (82, 83)
        for ordinal in range(32)
    )


def _token_protocol() -> PrismTokenProtocol:
    return PrismTokenProtocol(
        channel_prefixes=tuple((100 + index,) for index in range(8)),
        visibility_prefixes=((200,), (201,), (202,), (203,)),
        relation_prefixes=((300,), (301,), (302,)),
        confidence_prefixes=((400,), (401,), (402,)),
        evidence_separator=(500,),
        terminal_tokens=(600, 601),
        max_evidence_tokens=4,
    )


def _score_panel(
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring: tuple[PrismScoringRow, ...],
    *,
    bootstrap_seed: bytes,
    source_identity: str,
) -> PrismCueResult:
    protocol = _token_protocol()
    return score_prism_cue_panel(
        calibrations,
        observations,
        scoring,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
        calibration_receipt_sha256=prism_calibration_receipt_sha256(
            calibrations, protocol
        ),
        protocol=protocol,
    )


def _validate_panel_result(
    result: PrismCueResult,
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring: tuple[PrismScoringRow, ...],
    *,
    bootstrap_seed: bytes,
    source_identity: str,
) -> None:
    protocol = _token_protocol()
    validate_prism_cue_result(
        result,
        calibrations,
        observations,
        scoring,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
        calibration_receipt_sha256=prism_calibration_receipt_sha256(
            calibrations, protocol
        ),
        protocol=protocol,
    )


def _completion_for(row: PrismObservationRow, relation: str) -> tuple[int, ...]:
    channel = row.channel
    return (
        100 + PRISM_CHANNELS.index(channel),
        200,
        300 if relation == "same" else 301,
        402,
        9,
        500,
        10,
        600,
        601,
    )


def test_prism_token_protocol_parses_exact_ids_and_rejects_ambiguity() -> None:
    observations, _scoring = build_prism_schedules(
        _optimization_examples(),
        _caliber_examples(),
        source_identity="token-source",
    )
    row = observations[0]
    channel_index = PRISM_CHANNELS.index(row.channel)
    completion = (
        100 + channel_index,
        200,
        301,
        402,
        300,
        9,
        500,
        10,
        11,
        600,
        601,
    )
    parsed = parse_prism_completion(row, completion, _token_protocol())
    assert isinstance(parsed, PrismObservation)
    assert parsed.pair_ordinal == row.pair_ordinal
    assert parsed.fold == row.fold
    assert parsed.channel == row.channel
    assert parsed.left_first is row.left_first
    assert parsed.left_payload_sha256 == row.left_payload_sha256
    assert parsed.right_payload_sha256 == row.right_payload_sha256
    assert parsed.generation_seed == row.generation_seed
    assert parsed.protocol_valid is True
    assert parsed.left_visible is True
    assert parsed.right_visible is True
    assert parsed.relation == "different"
    assert parsed.confidence == "high"
    assert parsed.evidence_left_token_ids == (300, 9)
    assert parsed.evidence_right_token_ids == (10, 11)
    expected_digest = hashlib.sha256(
        struct.pack("<Q", len(completion))
        + b"".join(struct.pack("<I", token) for token in completion)
    ).hexdigest()
    assert parsed.completion_sha256 == expected_digest
    validate_prism_observation(parsed, row, completion, _token_protocol())
    with pytest.raises(ValueError, match="digest"):
        validate_prism_observation(
            replace(parsed, completion_sha256="0" * 64),
            row,
            completion,
            _token_protocol(),
        )

    malformed = (
        ((), "token"),
        (completion[:0] + (101,) + completion[1:], "channel"),
        (completion[:6] + completion[7:], "separator"),
        (completion[:7] + (500,) + completion[7:], "separator"),
        (completion + (999,), "terminal"),
        (completion[:4] + (1, 2, 3, 4, 5) + completion[6:], "evidence"),
        (completion[:-2], "terminal"),
        (completion[:4] + (True,) + completion[5:], "token"),
        (completion[:4] + (-1,) + completion[5:], "token"),
    )
    for changed, message in malformed:
        with pytest.raises((TypeError, ValueError), match=message):
            parse_prism_completion(row, changed, _token_protocol())

    overlapping = replace(
        _token_protocol(),
        channel_prefixes=((100,), (100, 1), *((102 + index,) for index in range(6))),
    )
    with pytest.raises(ValueError, match="overlap"):
        parse_prism_completion(row, completion, overlapping)
    for changed_protocol in (
        replace(_token_protocol(), evidence_separator=(True,)),
        replace(_token_protocol(), terminal_tokens=(-1,)),
        replace(_token_protocol(), max_evidence_tokens=True),
    ):
        with pytest.raises(ValueError, match="protocol"):
            parse_prism_completion(row, completion, changed_protocol)
    with pytest.raises(ValueError, match="row"):
        parse_prism_completion(
            replace(row, generation_seed=True),
            completion,
            _token_protocol(),
        )
    invalid = invalid_prism_observation(row, (999,))
    assert invalid.protocol_valid is False
    assert invalid.left_first is row.left_first
    assert invalid.relation == "indeterminate"
    assert invalid.evidence_left_token_ids == ()
    assert invalid.evidence_right_token_ids == ()
    validate_prism_observation(invalid, row, (999,), _token_protocol())
    empty_invalid = invalid_prism_observation(row, ())
    validate_prism_observation(empty_invalid, row, (), _token_protocol())

    with pytest.raises(ValueError, match="protocol"):
        validate_prism_observation(
            invalid,
            row,
            (999,),
            replace(_token_protocol(), terminal_tokens=(-1,)),
        )
    with pytest.raises(ValueError, match="row"):
        validate_prism_observation(
            invalid,
            replace(row, left_payload_sha256="not-a-digest"),
            (999,),
            _token_protocol(),
        )


def _perfect_panel(*, source_identity: str = "panel-source") -> tuple[
    tuple[PrismObservation, ...],
    tuple[object, ...],
]:
    schedule, scoring = build_prism_schedules(
        _optimization_examples(),
        _caliber_examples(),
        source_identity=source_identity,
    )
    observations = tuple(
        parse_prism_completion(
            row,
            _completion_for(row, scoring[row.pair_ordinal].relation),
            _token_protocol(),
        )
        for row in schedule
    )
    return observations, scoring


def _invalid_observation(observation: PrismObservation) -> PrismObservation:
    return replace(
        observation,
        protocol_valid=False,
        left_visible=False,
        right_visible=False,
        relation="indeterminate",
        confidence="low",
        evidence_left_token_ids=(),
        evidence_right_token_ids=(),
    )


def _bootstrap_lowers(result: PrismCueResult, seed: bytes) -> tuple[float, float, float]:
    scores = np.asarray(result.pair_scores, dtype=np.float64)
    truth = np.asarray(result.pair_truth, dtype=np.int64)
    probabilities = np.asarray(
        [
            1.0 / (1.0 + math.exp(-score))
            if score >= 0.0
            else math.exp(score) / (1.0 + math.exp(score))
            for score in scores
        ]
    )
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    losses = -np.log(np.where(truth == 1, clipped, 1.0 - clipped))
    material = b"sfora-prism-cue-bootstrap-v1\0" + len(seed).to_bytes(8, "little") + seed
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(hashlib.sha256(material).digest()[:16], "little"))
    )
    improvements: list[float] = []
    aucs: list[float] = []
    while len(improvements) < 10_000:
        indexes = generator.integers(0, 32, size=32)
        labels = truth[indexes]
        if set(labels.tolist()) != {0, 1}:
            continue
        sampled_scores = scores[indexes]
        positive = sampled_scores[labels == 1]
        negative = sampled_scores[labels == 0]
        comparisons = positive[:, None] - negative[None, :]
        improvements.append(math.log(2.0) - float(losses[indexes].mean()))
        aucs.append(
            (
                np.count_nonzero(comparisons > 0)
                + 0.5 * np.count_nonzero(comparisons == 0)
            )
            / comparisons.size
        )
    improvements.sort()
    aucs.sort()
    return improvements[499], aucs[499], improvements[0]


def test_prism_calibration_and_cue_result_use_fixed_jeffreys_arithmetic() -> None:
    observations, scoring = _perfect_panel()
    calibrations = calibrate_prism_channels(
        observations, scoring, source_identity="panel-source"
    )
    assert len(calibrations) == len(PRISM_CHANNELS)
    assert all(isinstance(row, PrismChannelCalibration) for row in calibrations)
    for channel, calibration in zip(PRISM_CHANNELS, calibrations, strict=True):
        assert calibration.channel == channel
        assert calibration.counts == ((48, 0), (0, 48), (0, 0))
        assert calibration.visibility_ppm == 1_000_000
        assert calibration.loo_log_loss_improvement > 0.02
        assert all(value > 0.0 for value in calibration.fold_log_loss_improvements)
        assert calibration.eligible is True

    result = _score_panel(
        calibrations,
        observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert isinstance(result, PrismCueResult)
    assert result.calibration_receipt_sha256 == prism_calibration_receipt_sha256(
        calibrations, _token_protocol()
    )
    assert len(result.pair_scores) == 32
    assert result.pair_truth == tuple(
        int(row.relation == "different") for row in scoring[128:160]
    )
    assert result.pair_truth.count(0) == 16
    assert result.pair_truth.count(1) == 16
    assert result.mean_log_loss_improvement_lower_95 >= 0.05
    assert result.auc == 1.0
    assert result.auc_lower_95 >= 0.80
    assert result.valid_orientation_ppm == (1_000_000, 1_000_000)
    assert result.orientation_auc_gap == 0.0
    assert result.eligible_channels == PRISM_CHANNELS
    assert result.log_loss_gate_passed is True
    assert result.auc_gate_passed is True
    assert result.channel_gate_passed is True
    assert result.orientation_gate_passed is True
    assert result.cue_classification == "cue-pass"
    assert result.passed is True
    same_probability = (0.5 / 49.5) / ((47.5 / 48.5) + (0.5 / 49.5))
    expected_loo = math.log(2.0) + math.log1p(-same_probability)
    assert calibrations[0].loo_log_loss_improvement == pytest.approx(
        expected_loo, abs=1e-15
    )

    invalid_calibration_ordinal = next(
        index
        for index, observation in enumerate(observations)
        if observation.fold == 1 and observation.channel == PRISM_CHANNELS[0]
    )
    one_invalid_calibration = tuple(
        _invalid_observation(observation) if index == invalid_calibration_ordinal else observation
        for index, observation in enumerate(observations)
    )
    invalid_calibration = calibrate_prism_channels(
        one_invalid_calibration, scoring, source_identity="panel-source"
    )[0]
    channel_rows = [
        observation
        for observation in one_invalid_calibration
        if observation.fold in {1, 2, 3} and observation.channel == PRISM_CHANNELS[0]
    ]
    relation_indexes = {"same": 0, "different": 1, "indeterminate": 2}
    manual_counts = np.zeros((3, 2), dtype=np.int64)
    for observation in channel_rows:
        if observation.protocol_valid:
            truth = int(scoring[observation.pair_ordinal].relation == "different")
            manual_counts[relation_indexes[observation.relation], truth] += 1
    manual_losses: list[float] = []
    for observation in channel_rows:
        truth = int(scoring[observation.pair_ordinal].relation == "different")
        if not observation.protocol_valid:
            probability = 0.5
        else:
            remaining = manual_counts.copy()
            relation_index = relation_indexes[observation.relation]
            remaining[relation_index, truth] -= 1
            likelihood_same = (remaining[relation_index, 0] + 0.5) / (
                float(remaining[:, 0].sum()) + 1.5
            )
            likelihood_different = (remaining[relation_index, 1] + 0.5) / (
                float(remaining[:, 1].sum()) + 1.5
            )
            probability = likelihood_different / (likelihood_same + likelihood_different)
        manual_losses.append(
            -math.log(probability if truth else 1.0 - probability)
        )
    assert invalid_calibration.loo_log_loss_improvement == pytest.approx(
        math.log(2.0) - math.fsum(manual_losses) / len(manual_losses),
        abs=1e-15,
    )
    _validate_panel_result(
        result,
        calibrations,
        observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )

    forged = replace(result, passed=False)
    with pytest.raises(ValueError, match="derivation"):
        _validate_panel_result(
            forged,
            calibrations,
            observations,
            scoring,
            bootstrap_seed=b"fixed-panel-bootstrap",
            source_identity="panel-source",
        )
    forged_receipt = replace(result, calibration_receipt_sha256="0" * 64)
    with pytest.raises(ValueError, match="derivation"):
        _validate_panel_result(
            forged_receipt,
            calibrations,
            observations,
            scoring,
            bootstrap_seed=b"fixed-panel-bootstrap",
            source_identity="panel-source",
        )
    changed_protocol = replace(_token_protocol(), max_evidence_tokens=9)
    with pytest.raises(ValueError, match="receipt"):
        score_prism_cue_panel(
            calibrations,
            observations,
            scoring,
            bootstrap_seed=b"fixed-panel-bootstrap",
            source_identity="panel-source",
            calibration_receipt_sha256=prism_calibration_receipt_sha256(
                calibrations, _token_protocol()
            ),
            protocol=changed_protocol,
        )
    insufficient_observations = tuple(
        _invalid_observation(observation)
        if observation.fold in {1, 2, 3} and observation.channel not in PRISM_CHANNELS[:3]
        else observation
        for observation in observations
    )
    insufficient = calibrate_prism_channels(
        insufficient_observations, scoring, source_identity="panel-source"
    )
    failed = _score_panel(
        insufficient,
        insufficient_observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert failed.passed is False

    no_anchor_observations = tuple(
        _invalid_observation(observation)
        if observation.fold in {1, 2, 3}
        and observation.channel in {PRISM_CHANNELS[0], PRISM_CHANNELS[1], PRISM_CHANNELS[7]}
        else observation
        for observation in observations
    )
    no_anchor_calibrations = calibrate_prism_channels(
        no_anchor_observations, scoring, source_identity="panel-source"
    )
    no_anchor = _score_panel(
        no_anchor_calibrations,
        no_anchor_observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert len(no_anchor.eligible_channels) == 5
    assert no_anchor.auc_lower_95 >= 0.80
    assert no_anchor.mean_log_loss_improvement_lower_95 >= 0.05
    assert no_anchor.passed is False

    diagnostic_inverted = tuple(
        replace(
            observation,
            relation=("different" if observation.relation == "same" else "same"),
        )
        if observation.fold == 4
        else observation
        for observation in observations
    )
    inverted_result = _score_panel(
        calibrations,
        diagnostic_inverted,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert inverted_result.auc == 0.0
    assert inverted_result.passed is False

    one_error_observations = tuple(
        replace(
            observation,
            relation=("different" if observation.relation == "same" else "same"),
        )
        if observation.pair_ordinal == 128
        else observation
        for observation in observations
    )
    one_error_result = _score_panel(
        calibrations,
        one_error_observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    expected_improvement_lower, expected_auc_lower, bootstrap_minimum = _bootstrap_lowers(
        one_error_result, b"fixed-panel-bootstrap"
    )
    assert one_error_result.mean_log_loss_improvement_lower_95 == pytest.approx(
        expected_improvement_lower, abs=1e-15
    )
    assert one_error_result.auc_lower_95 == pytest.approx(expected_auc_lower, abs=1e-15)
    assert one_error_result.mean_log_loss_improvement_lower_95 > bootstrap_minimum
    assert one_error_result.auc_gate_passed is True
    assert one_error_result.log_loss_gate_passed is True
    assert one_error_result.cue_classification == "cue-pass"
    assert one_error_result.passed is True

    rank_cue_observations = tuple(
        replace(
            observation,
            relation=("different" if observation.relation == "same" else "same"),
        )
        if observation.pair_ordinal in {128, 129, 130}
        else observation
        for observation in observations
    )
    rank_cue_result = _score_panel(
        calibrations,
        rank_cue_observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert rank_cue_result.auc_gate_passed is True
    assert rank_cue_result.log_loss_gate_passed is False
    assert rank_cue_result.cue_classification == "rank-cue-only"
    assert rank_cue_result.passed is False

    diagnostic_ties = tuple(
        replace(observation, relation="indeterminate")
        if observation.fold == 4
        else observation
        for observation in observations
    )
    tie_result = _score_panel(
        calibrations,
        diagnostic_ties,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert tie_result.auc == 0.5
    assert tie_result.mean_log_loss_improvement == 0.0
    assert tie_result.passed is False

    one_orientation_inverted = tuple(
        replace(
            observation,
            relation=("different" if observation.relation == "same" else "same"),
        )
        if observation.fold == 4 and observation.left_first is False
        else observation
        for observation in observations
    )
    orientation_result = _score_panel(
        calibrations,
        one_orientation_inverted,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert orientation_result.orientation_auc_gap == 1.0
    assert orientation_result.passed is False

    invalid_false_ordinals = {
        observation.pair_ordinal
        for observation in observations
        if observation.fold == 4
        and observation.left_first is False
        and observation.channel == PRISM_CHANNELS[0]
    }
    invalid_false_ordinals = set(sorted(invalid_false_ordinals)[:5])
    low_validity = tuple(
        _invalid_observation(observation)
        if observation.fold == 4
        and observation.left_first is False
        and observation.pair_ordinal in invalid_false_ordinals
        else observation
        for observation in observations
    )
    low_validity_result = _score_panel(
        calibrations,
        low_validity,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    assert min(low_validity_result.valid_orientation_ppm) < 750_000
    assert low_validity_result.passed is False

    diagnostic_abstention = tuple(
        _invalid_observation(observation)
        if observation.fold == 4 and observation.channel in PRISM_CHANNELS[:2]
        else observation
        for observation in observations
    )
    abstention_result = _score_panel(
        calibrations,
        diagnostic_abstention,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    first_agreement = next(
        row
        for row in abstention_result.conditional_agreement
        if row[:2] == PRISM_CHANNELS[:2] and row[2] == "same"
    )
    assert first_agreement[3:] == (0, None)

    asymmetric_calibration_observations = tuple(
        replace(observation, relation="indeterminate")
        if observation.fold in {1, 2, 3}
        and observation.channel == PRISM_CHANNELS[0]
        and scoring[observation.pair_ordinal].relation == "same"
        else _invalid_observation(observation)
        if observation.fold == 4 and observation.channel == PRISM_CHANNELS[0]
        else observation
        for observation in observations
    )
    asymmetric_calibrations = calibrate_prism_channels(
        asymmetric_calibration_observations,
        scoring,
        source_identity="panel-source",
    )
    assert asymmetric_calibrations[0].eligible is True
    neutral_invalid_result = _score_panel(
        asymmetric_calibrations,
        asymmetric_calibration_observations,
        scoring,
        bootstrap_seed=b"fixed-panel-bootstrap",
        source_identity="panel-source",
    )
    expected_magnitude = 7.0 * math.log(97.0) / 8.0
    for pair_score, pair_truth in zip(
        neutral_invalid_result.pair_scores,
        neutral_invalid_result.pair_truth,
        strict=True,
    ):
        assert pair_score == pytest.approx(
            expected_magnitude if pair_truth else -expected_magnitude,
            abs=1e-12,
        )

    with pytest.raises(ValueError, match="cardinality"):
        calibrate_prism_channels(
            observations[:-1], scoring, source_identity="panel-source"
        )
    duplicated = (*observations[:-1], observations[0])
    with pytest.raises(ValueError, match="order"):
        calibrate_prism_channels(duplicated, scoring, source_identity="panel-source")
    stale_orientation = tuple(
        replace(observation, left_first=not observation.left_first)
        if observation.fold == 4
        else observation
        for observation in observations
    )
    with pytest.raises(ValueError, match="seed authority"):
        calibrate_prism_channels(
            stale_orientation, scoring, source_identity="panel-source"
        )
    unbalanced_orientation = tuple(
        replace(
            observation,
            left_first=True,
            generation_seed=prism._generation_seed(
                "panel-source",
                observation.pair_ordinal,
                observation.channel,
                observation.left_payload_sha256,
                observation.right_payload_sha256,
                True,
            ),
        )
        if observation.fold == 4
        else observation
        for observation in observations
    )
    with pytest.raises(ValueError, match="orientation balance"):
        calibrate_prism_channels(
            unbalanced_orientation, scoring, source_identity="panel-source"
        )
    with pytest.raises(ValueError, match="payload binding"):
        calibrate_prism_channels(
            (replace(observations[0], left_payload_sha256="f" * 64), *observations[1:]),
            scoring,
            source_identity="panel-source",
        )
    with pytest.raises(ValueError, match="seed"):
        calibrate_prism_channels(
            (
                observations[0],
                replace(
                    observations[1], generation_seed=observations[0].generation_seed
                ),
                *observations[2:],
            ),
            scoring,
            source_identity="panel-source",
        )
    with pytest.raises(ValueError, match="seed authority"):
        calibrate_prism_channels(
            observations,
            scoring,
            source_identity="wrong-panel-source",
        )
    for changed in (
        (replace(calibrations[0], counts=((True, 0), (0, 48), (0, 0))), *calibrations[1:]),
        (replace(calibrations[0], loo_log_loss_improvement=float("nan")), *calibrations[1:]),
        (calibrations[1], calibrations[0], *calibrations[2:]),
    ):
        with pytest.raises(ValueError, match="calibration"):
            _score_panel(
                tuple(changed),
                observations,
                scoring,
                bootstrap_seed=b"fixed-panel-bootstrap",
                source_identity="panel-source",
            )

    invisible = tuple(
        replace(observation, left_visible=False)
        if observation.fold in {1, 2, 3} and observation.channel == PRISM_CHANNELS[0]
        else observation
        for observation in observations
    )
    assert (
        calibrate_prism_channels(
            invisible, scoring, source_identity="panel-source"
        )[0].eligible
        is False
    )
    one_bad_fold = tuple(
        replace(
            observation,
            relation=("different" if observation.relation == "same" else "same"),
        )
        if observation.fold == 1 and observation.channel == PRISM_CHANNELS[0]
        else observation
        for observation in observations
    )
    one_bad = calibrate_prism_channels(
        one_bad_fold, scoring, source_identity="panel-source"
    )[0]
    assert one_bad.loo_log_loss_improvement > 0.02
    assert one_bad.fold_log_loss_improvements[0] < 0.0
    assert one_bad.eligible is False

    failed_pilot = tuple(
        _invalid_observation(observation)
        if observation.fold == 0 and observation.left_first is False
        else observation
        for observation in observations
    )
    with pytest.raises(ValueError, match="pilot"):
        _score_panel(
            calibrations,
            failed_pilot,
            scoring,
            bootstrap_seed=b"fixed-panel-bootstrap",
            source_identity="panel-source",
        )

    false_pilot_ordinals = sorted(
        {
            observation.pair_ordinal
            for observation in observations
            if observation.fold == 0
            and observation.left_first is False
            and observation.channel == PRISM_CHANNELS[0]
        }
    )[:8]
    boundary_pilot = tuple(
        _invalid_observation(observation)
        if observation.fold == 0
        and observation.pair_ordinal in false_pilot_ordinals
        else observation
        for observation in observations
    )
    assert (
        _score_panel(
            calibrations,
            boundary_pilot,
            scoring,
            bootstrap_seed=b"fixed-panel-bootstrap",
            source_identity="panel-source",
        ).passed
        is True
    )


def test_prism_cue_gate_requires_every_literal_threshold() -> None:
    passing = dict(
        improvement_lower=0.05,
        auc_lower=0.80,
        eligible_channels=PRISM_CHANNELS[:4],
        valid_orientation_ppm=(750_000, 750_000),
        orientation_gap=0.10,
    )
    assert prism._passes_prism_cue_gates(**passing) is True
    failures = (
        {**passing, "improvement_lower": math.nextafter(0.05, 0.0)},
        {**passing, "auc_lower": math.nextafter(0.80, 0.0)},
        {**passing, "eligible_channels": PRISM_CHANNELS[:3]},
        {
            **passing,
            "eligible_channels": (
                PRISM_CHANNELS[0],
                PRISM_CHANNELS[2],
                PRISM_CHANNELS[3],
                PRISM_CHANNELS[4],
            ),
        },
        {**passing, "valid_orientation_ppm": (749_999, 1_000_000)},
        {**passing, "orientation_gap": math.nextafter(0.10, 1.0)},
    )
    for changed in failures:
        assert prism._passes_prism_cue_gates(**changed) is False


def test_prism_schedules_are_balanced_anonymous_disjoint_and_source_bound() -> None:
    optimization = _optimization_examples()
    caliber = _caliber_examples()
    observations, scoring = build_prism_schedules(
        optimization,
        caliber,
        source_identity="source-a",
    )

    assert PRISM_CHANNELS == (
        "grille-fascia",
        "lamps",
        "wheels",
        "silhouette-roofline",
        "trim-badging",
        "stance-proportions",
        "interior-dashboard",
        "model-year-evidence",
    )
    assert len(scoring) == 160
    assert len(observations) == 160 * len(PRISM_CHANNELS)
    assert tuple(row.pair_ordinal for row in scoring) == tuple(range(160))
    assert len({row.generation_seed for row in observations}) == len(observations)

    used_ids = [
        example_id for row in scoring for example_id in (row.left_example_id, row.right_example_id)
    ]
    assert len(used_ids) == 320
    assert len(set(used_ids)) == 320
    for fold in range(4):
        rows = [row for row in scoring if row.fold == fold]
        assert len(rows) == 32
        assert sum(row.relation == "same" for row in rows) == 16
        assert sum(row.relation == "different" for row in rows) == 16
    caliber_rows = [row for row in scoring if row.fold == 4]
    assert len(caliber_rows) == 32
    assert sum((row.left_label, row.right_label) == (82, 82) for row in caliber_rows) == 8
    assert sum((row.left_label, row.right_label) == (83, 83) for row in caliber_rows) == 8
    assert sum(row.left_label != row.right_label for row in caliber_rows) == 16

    for pair_ordinal in range(160):
        pair = [row for row in observations if row.pair_ordinal == pair_ordinal]
        assert tuple(row.channel for row in pair) == PRISM_CHANNELS
        assert len({row.left_first for row in pair}) == 1
        assert len({(row.left_payload_sha256, row.right_payload_sha256) for row in pair}) == 1
        assert pair[0].left_payload_sha256 == scoring[pair_ordinal].left_payload_sha256
        assert pair[0].right_payload_sha256 == scoring[pair_ordinal].right_payload_sha256
    for fold, relation in (
        *((fold, relation) for fold in range(4) for relation in ("same", "different")),
        (4, "same"),
        (4, "different"),
    ):
        ordinals = {
            row.pair_ordinal for row in scoring if row.fold == fold and row.relation == relation
        }
        orientations = {
            row.left_first
            for row in observations
            if row.channel == PRISM_CHANNELS[0] and row.pair_ordinal in ordinals
        }
        assert orientations == {False, True}

    calibration_capability = release_prism_observation_capability(
        observations,
        scoring,
        phase="calibration",
        source_identity="source-a",
    )
    assert len(calibration_capability) == 128 * len(PRISM_CHANNELS)
    assert all(len(row.pair_handle) == 64 for row in calibration_capability)
    assert len({row.pair_handle for row in calibration_capability}) == 128
    assert all("fold" not in asdict(row) for row in calibration_capability)
    assert all("ordinal" not in asdict(row) for row in calibration_capability)
    anonymous = json.dumps([asdict(row) for row in calibration_capability], sort_keys=True)
    for forbidden in (
        "label",
        "relation",
        "example_id",
        "class_name",
        "fold",
        "secret/path",
        "optimization-",
        "caliber-",
    ):
        assert forbidden not in anonymous
    with pytest.raises(ValueError, match="receipt"):
        release_prism_observation_capability(
            observations,
            scoring,
            phase="diagnostic",
            source_identity="source-a",
        )
    with pytest.raises(ValueError, match="schedule"):
        release_prism_observation_capability(
            tuple(row for row in observations if row.fold != 0),
            scoring,
            phase="diagnostic",
            source_identity="source-a",
            calibration_receipt_sha256="0" * 64,
            calibrations=(),
            pilot_observations=(),
            pilot_completion_ids=(),
            protocol=_token_protocol(),
        )
    panel_observations, panel_scoring = _perfect_panel(source_identity="source-a")
    panel_calibrations = calibrate_prism_channels(
        panel_observations,
        panel_scoring,
        source_identity="source-a",
    )
    calibration_receipt = prism_calibration_receipt_sha256(
        panel_calibrations, _token_protocol()
    )
    pilot_schedule_rows = tuple(row for row in observations if row.fold == 0)
    pilot_observations = tuple(row for row in panel_observations if row.fold == 0)
    pilot_completion_ids = tuple(
        _completion_for(row, panel_scoring[row.pair_ordinal].relation)
        for row in pilot_schedule_rows
    )
    with pytest.raises(ValueError, match="pilot"):
        release_prism_observation_capability(
            observations,
            scoring,
            phase="diagnostic",
            source_identity="source-a",
            calibration_receipt_sha256=calibration_receipt,
            calibrations=panel_calibrations,
            protocol=_token_protocol(),
        )
    diagnostic_capability = release_prism_observation_capability(
        observations,
        scoring,
        phase="diagnostic",
        source_identity="source-a",
        calibration_receipt_sha256=calibration_receipt,
        calibrations=panel_calibrations,
        pilot_observations=pilot_observations,
        pilot_completion_ids=pilot_completion_ids,
        protocol=_token_protocol(),
    )
    assert len(diagnostic_capability) == 32 * len(PRISM_CHANNELS)
    assert all(len(row.pair_handle) == 64 for row in diagnostic_capability)
    assert len({row.pair_handle for row in diagnostic_capability}) == 32
    assert all("fold" not in asdict(row) for row in diagnostic_capability)
    assert all("ordinal" not in asdict(row) for row in diagnostic_capability)
    failed_pilot = tuple(
        invalid_prism_observation(schedule_row, ()) if index < 65 else observed
        for index, (schedule_row, observed) in enumerate(
            zip(pilot_schedule_rows, pilot_observations, strict=True)
        )
    )
    failed_pilot_completion_ids = tuple(
        () if index < 65 else completion_ids
        for index, completion_ids in enumerate(pilot_completion_ids)
    )
    with pytest.raises(ValueError, match="pilot"):
        release_prism_observation_capability(
            observations,
            scoring,
            phase="diagnostic",
            source_identity="source-a",
            calibration_receipt_sha256=calibration_receipt,
            calibrations=panel_calibrations,
            pilot_observations=failed_pilot,
            pilot_completion_ids=failed_pilot_completion_ids,
            protocol=_token_protocol(),
        )
    with pytest.raises(ValueError, match="digest"):
        release_prism_observation_capability(
            observations,
            scoring,
            phase="diagnostic",
            source_identity="source-a",
            calibration_receipt_sha256=calibration_receipt,
            calibrations=panel_calibrations,
            pilot_observations=(
                replace(pilot_observations[0], completion_sha256="0" * 64),
                *pilot_observations[1:],
            ),
            pilot_completion_ids=pilot_completion_ids,
            protocol=_token_protocol(),
        )
    with pytest.raises(ValueError, match="receipt"):
        release_prism_observation_capability(
            observations,
            scoring,
            phase="diagnostic",
            source_identity="source-a",
            calibration_receipt_sha256="0" * 64,
            calibrations=panel_calibrations,
            pilot_observations=pilot_observations,
            pilot_completion_ids=pilot_completion_ids,
            protocol=_token_protocol(),
        )

    reordered = build_prism_schedules(
        tuple(reversed(optimization)),
        tuple(reversed(caliber)),
        source_identity="source-a",
    )
    assert reordered == (observations, scoring)
    changed_source = build_prism_schedules(
        optimization,
        caliber,
        source_identity="source-b",
    )
    assert changed_source != (observations, scoring)
    validate_prism_schedules(observations, scoring, source_identity="source-a")

    schedule_mutations = []
    changed = list(observations)
    changed[0], changed[1] = changed[1], changed[0]
    schedule_mutations.append((tuple(changed), scoring))
    changed = list(observations)
    changed[0] = replace(changed[0], generation_seed=changed[0].generation_seed ^ 1)
    schedule_mutations.append((tuple(changed), scoring))
    changed = list(observations)
    first_pair = changed[: len(PRISM_CHANNELS)]
    caliber_pair = changed[128 * len(PRISM_CHANNELS) : 129 * len(PRISM_CHANNELS)]
    for index in range(len(PRISM_CHANNELS)):
        changed[index] = replace(
            first_pair[index],
            left_payload_sha256=caliber_pair[index].left_payload_sha256,
            right_payload_sha256=caliber_pair[index].right_payload_sha256,
        )
    schedule_mutations.append((tuple(changed), scoring))
    changed = list(observations)
    changed[0] = replace(changed[0], generation_seed=True)
    schedule_mutations.append((tuple(changed), scoring))
    changed_scoring = list(scoring)
    changed_scoring[0] = replace(changed_scoring[0], left_label=True)
    schedule_mutations.append((observations, tuple(changed_scoring)))
    changed_scoring = list(scoring)
    changed_scoring[0], changed_scoring[1] = changed_scoring[1], changed_scoring[0]
    schedule_mutations.append((observations, tuple(changed_scoring)))
    for changed_observations, changed_scoring in schedule_mutations:
        with pytest.raises((TypeError, ValueError)):
            validate_prism_schedules(
                changed_observations,
                changed_scoring,
                source_identity="source-a",
            )
    changed_scoring = list(scoring)
    changed_scoring[0] = replace(
        changed_scoring[0],
        left_payload_sha256=scoring[1].left_payload_sha256,
    )
    with pytest.raises(ValueError, match="payload binding"):
        validate_prism_schedules(
            observations,
            tuple(changed_scoring),
            source_identity="source-a",
        )
    changed_scoring = list(scoring)
    changed_observations = list(observations)
    changed_scoring[0] = replace(changed_scoring[0], fold=1)
    for index in range(len(PRISM_CHANNELS)):
        changed_observations[index] = replace(changed_observations[index], fold=1)
    with pytest.raises(ValueError, match="fold balance"):
        validate_prism_schedules(
            tuple(changed_observations),
            tuple(changed_scoring),
            source_identity="source-a",
        )
    with pytest.raises(ValueError, match="seed"):
        validate_prism_schedules(
            observations,
            scoring,
            source_identity="source-b",
        )


def test_prism_optimization_folds_are_class_stratified_under_imbalance() -> None:
    optimization = tuple(
        PrismExample(
            example_id=f"imbalanced-{label}-{ordinal}",
            label=label,
            image_sha256=_digest(20_000 + label * 64 + ordinal),
        )
        for label in range(49)
        for ordinal in range(40 if label < 4 else 6)
    )
    observations, scoring = build_prism_schedules(
        optimization,
        _caliber_examples(),
        source_identity="imbalanced-source",
    )
    validate_prism_schedules(observations, scoring, source_identity="imbalanced-source")
    same_rows = [row for row in scoring if row.fold < 4 and row.relation == "same"]
    counts = {label: sum(row.left_label == label for row in same_rows) for label in range(49)}
    assert min(counts.values()) >= 1
    assert max(counts.values()) <= 2
    for fold in range(4):
        fold_labels = [row.left_label for row in same_rows if row.fold == fold]
        assert len(fold_labels) == 16
        assert max(fold_labels.count(label) for label in set(fold_labels)) <= 1


def test_prism_schedule_rejects_invalid_capability_inputs() -> None:
    optimization = _optimization_examples()
    caliber = _caliber_examples()

    mutations = (
        (optimization[:-200], caliber),
        (optimization, caliber[:-1]),
        (
            (replace(optimization[0], example_id=optimization[1].example_id), *optimization[1:]),
            caliber,
        ),
        (
            (
                replace(optimization[0], image_sha256=optimization[1].image_sha256),
                *optimization[1:],
            ),
            caliber,
        ),
        ((replace(optimization[0], label=True), *optimization[1:]), caliber),
        ((replace(optimization[0], label=49), *optimization[1:]), caliber),
        (optimization, (replace(caliber[0], label=84), *caliber[1:])),
    )
    for changed_optimization, changed_caliber in mutations:
        with pytest.raises((TypeError, ValueError)):
            build_prism_schedules(
                tuple(changed_optimization),
                tuple(changed_caliber),
                source_identity="source-a",
            )

    with pytest.raises(TypeError):
        build_prism_schedules(  # type: ignore[arg-type]
            list(optimization),
            caliber,
            source_identity="source-a",
        )
    with pytest.raises((TypeError, ValueError)):
        build_prism_schedules(
            optimization,
            caliber,
            source_identity="",
        )


def _measurement_authority() -> PrismMeasurementAuthority:
    return PrismMeasurementAuthority(
        source_commit="1" * 40,
        source_tree_sha256="2" * 64,
        dataset_revision="dataset-revision",
        dataset_manifest_sha256="3" * 64,
        model_revision="model-revision",
        processor_revision="processor-revision",
        tokenizer_revision="tokenizer-revision",
        prompt_bundle_sha256="4" * 64,
        token_protocol_sha256="5" * 64,
        observation_manifest_sha256="6" * 64,
        scoring_manifest_sha256="7" * 64,
        completion_bundle_sha256="8" * 64,
    )


def test_prism_canonical_result_recomputes_authenticated_primitive_evidence() -> None:
    source_identity = "canonical-panel-source"
    bootstrap_seed = b"canonical-panel-bootstrap"
    observations, scoring = _perfect_panel(source_identity=source_identity)
    calibrations = calibrate_prism_channels(
        observations,
        scoring,
        source_identity=source_identity,
    )
    protocol = _token_protocol()
    calibration_receipt = prism_calibration_receipt_sha256(calibrations, protocol)
    result = score_prism_cue_panel(
        calibrations,
        observations,
        scoring,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
        calibration_receipt_sha256=calibration_receipt,
        protocol=protocol,
    )
    evidence = PrismMeasurementEvidence(
        authority=_measurement_authority(),
        observations=observations,
        scoring_rows=scoring,
        protocol=protocol,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
    )

    raw = canonical_prism_cue_result_bytes(evidence, calibrations, result)
    with pytest.raises(ValueError, match="calibration derivation"):
        canonical_prism_cue_result_bytes(evidence, calibrations[:-1], result)
    with pytest.raises(ValueError, match="result derivation"):
        canonical_prism_cue_result_bytes(
            evidence,
            calibrations,
            replace(result, passed=not result.passed),
        )
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert json.dumps(
        json.loads(raw),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode() + b"\n" == raw
    restored_calibrations, restored_result = validate_prism_cue_result_bytes(
        raw,
        expected=evidence,
    )
    assert restored_calibrations == calibrations
    assert restored_result == result

    payload = json.loads(raw)
    payload["result"]["mean_log_loss_improvement_lower_95"] = 0.123
    mutated = (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    with pytest.raises(ValueError, match="artifact differs"):
        validate_prism_cue_result_bytes(mutated, expected=evidence)

    with pytest.raises(ValueError, match="authority"):
        validate_prism_cue_result_bytes(
            raw,
            expected=replace(
                evidence,
                authority=replace(evidence.authority, source_commit="9" * 40),
            ),
        )

    for invalid_authority in (
        replace(evidence.authority, source_commit="1" * 39),
        replace(evidence.authority, source_tree_sha256=True),
        replace(evidence.authority, dataset_revision=""),
    ):
        with pytest.raises((TypeError, ValueError), match="authority"):
            validate_prism_cue_result_bytes(
                raw,
                expected=replace(evidence, authority=invalid_authority),
            )

    wrong_schema = json.loads(raw)
    wrong_schema["schema"] = "wrong-schema"
    missing_calibration = json.loads(raw)
    missing_calibration["calibrations"].pop()
    wrong_truth = json.loads(raw)
    wrong_truth["result"]["pair_truth"][0] ^= 1
    for changed in (wrong_schema, missing_calibration, wrong_truth):
        changed_raw = (
            json.dumps(changed, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
            + b"\n"
        )
        with pytest.raises(ValueError, match="artifact differs"):
            validate_prism_cue_result_bytes(changed_raw, expected=evidence)
