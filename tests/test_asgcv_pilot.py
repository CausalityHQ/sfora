from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from sfora.asgcv_pilot import (
    ASGCV_P32_BOUNDARY_NAMES,
    ASGCV_P32_PAIR_COUNT,
    AsgcvP32Candidate,
    AsgcvP32Result,
    asgcv_p32_branch_exchange_energy_ppm,
    asgcv_p32_collapsed_exact_cosine,
    asgcv_p32_field_authority,
    canonical_asgcv_p32_candidate_bytes,
    canonical_asgcv_p32_result_bytes,
    derive_asgcv_p32_schedule_seed,
    validate_asgcv_p32_candidate_bytes,
    validate_asgcv_p32_candidate_context,
    validate_asgcv_p32_pilot_schedule,
    validate_asgcv_p32_result_bundle,
    validate_asgcv_p32_result_bytes,
)
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    build_asgcv_pair_schedule,
    classify_asgcv_completion_group,
)


def _candidate(ordinal: int) -> AsgcvP32Candidate:
    relation = 1 if ordinal % 2 == 0 else -1
    verdicts = (relation,) * 4 + (-relation,) * 4
    return AsgcvP32Candidate(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        launch_authority_sha256="c" * 64,
        predictor_initialization_seed_sha256="d" * 64,
        partition_authority_sha256="4" * 64,
        pilot_schedule_sha256="5" * 64,
        completion_protocol_sha256="6" * 64,
        rollout_authority_sha256="7" * 64,
        completion_group_sha256=f"{ordinal + 1:064x}",
        pooler_state_sha256="8" * 64,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(2 * ordinal, 2 * ordinal + 1),
        relation_sign=relation,
        generation_seeds=tuple(range(8)),
        rewards=(1, 1, 1, 1, 0, 0, 0, 0),
        valid_flags=(True,) * 8,
        verdict_relation_signs=verdicts,
        attribute_span_lengths=(3,) * 8,
        generated_token_counts=(16,) * 8,
        completion_scores=(-0.10, -0.11, -0.09, -0.10, -1.10, -1.11, -1.09, -1.10),
        lowest_branch_indices=(0, 4),
        highest_branch_indices=(3, 7),
        branch_exchange_distinct=True,
        collapsed_branch_scores=(-0.10, -1.10),
        collapsed_backend_coefficient_ppm=393_256,
        highest_branch_scores=(-0.10, -1.10),
        highest_backend_coefficient_ppm=393_256,
        lowest_gradient_sha256="9" * 64,
        highest_gradient_sha256="a" * 64,
        lowest_gradient_norm=2.0,
        highest_gradient_norm=2.0,
        branch_exchange_energy_ppm=100_000,
        boundary_names=ASGCV_P32_BOUNDARY_NAMES,
        boundary_norms=(1.0, 1.0, 1.0, 1.0),
        exact_gradient_sha256="b" * 64 if ordinal < 4 else None,
        exact_gradient_norm=2.0 if ordinal < 4 else None,
        exact_replay_generated_tokens=128 if ordinal < 4 else None,
        collapsed_exact_cosine=0.5 if ordinal < 4 else None,
        prepare_elapsed_ns=1_000_000,
        generate_elapsed_ns=100_000_000,
        score_elapsed_ns=2_000_000,
        collapsed_replay_elapsed_ns=1_000_000,
        branch_exchange_replay_elapsed_ns=1_000_000,
        exact_replay_elapsed_ns=400_000_000 if ordinal < 4 else 0,
        predictor_forward_elapsed_ns=1_000_000,
        candidate_total_elapsed_ns=506_000_000 if ordinal < 4 else 106_000_000,
        peak_cuda_reserved_bytes=1_000_000,
        peak_rss_bytes=2_000_000,
    ).validated()


def test_p32_field_math_has_one_fp32_digest_norm_energy_and_cosine_authority() -> None:
    lowest = np.array([1.0, 0.0], dtype=np.float32)
    highest = np.array([1.0, 1.0], dtype=np.float32)
    digest, norm = asgcv_p32_field_authority(lowest, role="lowest")
    assert len(digest) == 64
    assert norm == 1.0
    assert asgcv_p32_branch_exchange_energy_ppm(lowest, highest) == 333_333
    assert asgcv_p32_branch_exchange_energy_ppm(lowest, lowest) == 0
    assert asgcv_p32_collapsed_exact_cosine(lowest, highest) == pytest.approx(1.0 / math.sqrt(2.0))
    assert ASGCV_P32_BOUNDARY_NAMES == (
        "merger",
        "deepstack-0",
        "deepstack-1",
        "deepstack-2",
    )

    with pytest.raises(ValueError):
        asgcv_p32_field_authority(lowest.astype(np.float64), role="lowest")
    with pytest.raises(ValueError):
        asgcv_p32_collapsed_exact_cosine(np.zeros(2, dtype=np.float32), highest)


def test_p32_schedule_is_domain_separated_and_bound_to_training_partition() -> None:
    partition = AsgcvPartitionAuthority(
        source_manifest_sha256="a" * 64,
        partition_seed_sha256="b" * 64,
        predictor_train_class_ids=tuple(range(16)),
        e0_validation_class_ids=tuple(range(16, 20)),
        e1_optimization_class_ids=tuple(range(20, 24)),
    ).validated()
    source_commit = "1" * 40
    seed = derive_asgcv_p32_schedule_seed(
        partition_authority=partition,
        source_commit=source_commit,
    )
    predictor_ids = tuple(f"train-{ordinal:03d}" for ordinal in range(64))
    predictor_labels = tuple(ordinal // 4 for ordinal in range(64))
    validation_ids = tuple(f"valid-{ordinal:03d}" for ordinal in range(16))
    validation_labels = tuple(16 + ordinal // 4 for ordinal in range(16))
    optimization_ids = tuple(f"optim-{ordinal:03d}" for ordinal in range(16))
    optimization_labels = tuple(20 + ordinal // 4 for ordinal in range(16))
    schedule = build_asgcv_pair_schedule(
        predictor_ids,
        predictor_labels,
        schedule_seed_sha256=seed,
        pair_count=ASGCV_P32_PAIR_COUNT,
    )
    validate_asgcv_p32_pilot_schedule(
        schedule,
        partition_authority=partition,
        source_commit=source_commit,
        predictor_train=(predictor_ids, predictor_labels),
        e0_validation=(validation_ids, validation_labels),
        e1_optimization=(optimization_ids, optimization_labels),
    )

    with pytest.raises(ValueError):
        validate_asgcv_p32_pilot_schedule(
            replace(schedule, schedule_seed_sha256="c" * 64).validated(),
            partition_authority=partition,
            source_commit=source_commit,
            predictor_train=(predictor_ids, predictor_labels),
            e0_validation=(validation_ids, validation_labels),
            e1_optimization=(optimization_ids, optimization_labels),
        )


def test_p32_candidate_seals_usable_incorrect_verdicts_and_rejects_relation_drift() -> None:
    candidate = _candidate(0)
    raw = canonical_asgcv_p32_candidate_bytes(candidate)
    assert raw.endswith(b"\n")
    assert validate_asgcv_p32_candidate_bytes(raw) == candidate
    assert candidate.both_verdicts_valid is True
    assert candidate.exchange_evaluable is True
    assert candidate.valid_correct_count == 4
    assert candidate.valid_incorrect_count == 4
    assert candidate.empirical_correct_probability_ppm == 500_000
    assert candidate.probability_calibration_ppm < 250_000
    assert candidate.within_verdict_dispersion_ratio_ppm < 25_000

    branch_sensitive = replace(
        candidate,
        completion_scores=(-0.10, 4.0, 4.0, 4.0, -1.10, -4.0, -4.0, -4.0),
        highest_branch_scores=(4.0, -4.0),
        highest_backend_coefficient_ppm=887,
    ).validated()
    assert branch_sensitive.score_probability == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))

    replace(
        candidate,
        collapsed_branch_scores=(-0.1000005, -1.1000005),
        collapsed_backend_coefficient_ppm=393_256,
        highest_branch_scores=(-0.1000005, -1.1000005),
        highest_backend_coefficient_ppm=393_256,
    ).validated()
    with pytest.raises(ValueError, match="backend evidence"):
        replace(candidate, collapsed_branch_scores=(-0.101, -1.10)).validated()

    mapping = candidate.to_mapping()
    for mutation in (
        {**mapping, "claim_eligible": True},
        {**mapping, "official_test_access": True},
        {**mapping, "valid_flags": [False, *mapping["valid_flags"][1:]]},
        {**mapping, "verdict_relation_signs": [None, *mapping["verdict_relation_signs"][1:]]},
        {**mapping, "rewards": [0, *mapping["rewards"][1:]]},
        {**mapping, "lowest_branch_indices": [1, 4]},
        {**mapping, "collapsed_branch_scores": [-1.10, -0.10]},
        {**mapping, "collapsed_backend_coefficient_ppm": 1},
        {**mapping, "highest_branch_scores": [-0.11, -1.10]},
        {**mapping, "highest_backend_coefficient_ppm": 1},
        {**mapping, "branch_exchange_energy_ppm": 1_000_001},
        {**mapping, "prepare_elapsed_ns": 0},
        {**mapping, "collapsed_replay_elapsed_ns": "x"},
        {**mapping, "candidate_total_elapsed_ns": 1},
    ):
        with pytest.raises(ValueError):
            AsgcvP32Candidate.from_mapping(mutation)
    with pytest.raises(ValueError, match="timing"):
        replace(candidate, candidate_total_elapsed_ns=1).validated()
    with pytest.raises(ValueError, match="boundary"):
        replace(candidate, boundary_names=tuple(reversed(ASGCV_P32_BOUNDARY_NAMES))).validated()


def test_p32_candidate_context_rebuilds_group_schedule_and_rollout_seed_bindings() -> None:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11,),
        different_prefix_ids=(21,),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="d" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    example_ids = tuple(f"cars-{ordinal:03d}" for ordinal in range(64))
    labels = tuple(ordinal // 4 for ordinal in range(64))
    schedule = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="e" * 64,
        pair_count=ASGCV_P32_PAIR_COUNT,
    )
    groups = []
    candidate_rows = []
    for pair in schedule.pairs:
        completions = tuple(
            (
                *((11,) if verdict == 1 else (21,)),
                30 + ordinal,
                99,
            )
            for ordinal, verdict in enumerate(
                (pair.relation_sign,) * 4 + (-pair.relation_sign,) * 4
            )
        )
        group = classify_asgcv_completion_group(
            completions,
            pair.relation_sign,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=pair.ordinal,
        )
        candidate = replace(
            _candidate(pair.ordinal),
            model_revision=rollout.model_revision,
            pilot_schedule_sha256=schedule.sha256(),
            completion_protocol_sha256=protocol.sha256(),
            rollout_authority_sha256=rollout.sha256(),
            completion_group_sha256=group.sha256(),
            pair_ordinals=(pair.left_index, pair.right_index),
            relation_sign=pair.relation_sign,
            generation_seeds=group.generation_seeds,
            rewards=group.rewards,
            valid_flags=group.valid_flags,
            verdict_relation_signs=group.verdict_relation_signs,
            attribute_span_lengths=tuple(
                0 if span is None else span[1] - span[0] for span in group.attribute_spans
            ),
            generated_token_counts=tuple(len(completion) for completion in group.completion_ids),
            exact_replay_generated_tokens=(
                sum(len(completion) for completion in group.completion_ids)
                if pair.ordinal < 4
                else None
            ),
        ).validated()
        groups.append(group)
        candidate_rows.append(candidate)
    candidate = candidate_rows[0]
    group = groups[0]
    validate_asgcv_p32_candidate_context(
        candidate,
        completion_group=group,
        rollout_authority=rollout,
        pilot_schedule=schedule,
    )
    candidate_receipts = tuple(canonical_asgcv_p32_candidate_bytes(row) for row in candidate_rows)
    result_raw = canonical_asgcv_p32_result_bytes(tuple(candidate_rows))
    assert (
        validate_asgcv_p32_result_bundle(
            result_raw,
            candidate_receipts=candidate_receipts,
            completion_groups=tuple(groups),
            rollout_authority=rollout,
            pilot_schedule=schedule,
        ).passed
        is True
    )

    with pytest.raises(ValueError):
        validate_asgcv_p32_candidate_context(
            replace(candidate, generation_seeds=tuple(range(8))).validated(),
            completion_group=group,
            rollout_authority=rollout,
            pilot_schedule=schedule,
        )


def test_p32_result_recomputes_all_gates_and_rejects_receipt_order_or_metric_drift() -> None:
    candidates = tuple(_candidate(ordinal) for ordinal in range(ASGCV_P32_PAIR_COUNT))
    result = AsgcvP32Result.from_candidates(candidates)
    assert result.passed is True
    assert result.branch_yield_ppm == 1_000_000
    assert result.variance_yield_ppm == 1_000_000
    assert result.completion_validity_ppm == 1_000_000
    assert result.exchange_evaluable_candidates == ASGCV_P32_PAIR_COUNT
    assert result.coefficient_evaluable_candidates == ASGCV_P32_PAIR_COUNT
    assert result.calibration_evaluable_candidates == ASGCV_P32_PAIR_COUNT
    assert result.dispersion_evaluable_candidates == ASGCV_P32_PAIR_COUNT
    assert result.collapsed_timing_candidates == ASGCV_P32_PAIR_COUNT
    assert result.exact_timing_candidates == 4
    assert result.projected_step_wall_ratio_ppm == 27_944
    assert result.projected_step_wall_ratio_p90_ppm == 27_944
    assert result.projected_exact_capture_wall_ns == 513_024_000_000
    assert result.projected_collapsed_capture_wall_ns == 106_496_000_000
    assert result.exact_diagnostic_ordinals == (0, 1, 2, 3)
    candidate_receipts = tuple(canonical_asgcv_p32_candidate_bytes(row) for row in candidates)
    result_raw = canonical_asgcv_p32_result_bytes(candidates)
    assert validate_asgcv_p32_result_bytes(result_raw, candidate_receipts) == result

    with pytest.raises(ValueError):
        AsgcvP32Result.from_candidates(tuple(reversed(candidates)))
    mixed = list(candidates)
    mixed[5] = replace(
        mixed[5],
        source_commit="f" * 40,
        pilot_schedule_sha256="e" * 64,
    ).validated()
    with pytest.raises(ValueError, match="identity"):
        AsgcvP32Result.from_candidates(tuple(mixed))
    forged = {
        **result.to_mapping(),
        "median_coefficient_ppm": 999_999,
        "median_probability_calibration_ppm": 0,
        "median_branch_exchange_energy_ppm": 0,
        "projected_step_wall_ratio_ppm": 1,
    }
    with pytest.raises(ValueError):
        validate_asgcv_p32_result_bytes(
            (json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            candidate_receipts,
        )
    count_drift = {**result.to_mapping(), "collapsed_timing_candidates": 31}
    with pytest.raises(ValueError):
        validate_asgcv_p32_result_bytes(
            (json.dumps(count_drift, sort_keys=True, separators=(",", ":")) + "\n").encode(),
            candidate_receipts,
        )


def test_p32_exact_diagnostics_follow_first_four_variance_eligible_ordinals() -> None:
    rows = [_candidate(ordinal) for ordinal in range(ASGCV_P32_PAIR_COUNT)]
    first = rows[0]
    rows[0] = replace(
        first,
        rewards=(1,) * 8,
        verdict_relation_signs=(first.relation_sign,) * 8,
        lowest_branch_indices=None,
        highest_branch_indices=None,
        branch_exchange_distinct=False,
        collapsed_branch_scores=None,
        collapsed_backend_coefficient_ppm=None,
        highest_branch_scores=None,
        highest_backend_coefficient_ppm=None,
        lowest_gradient_sha256=None,
        highest_gradient_sha256=None,
        lowest_gradient_norm=None,
        highest_gradient_norm=None,
        branch_exchange_energy_ppm=None,
        boundary_norms=None,
        exact_gradient_sha256=None,
        exact_gradient_norm=None,
        exact_replay_generated_tokens=None,
        collapsed_exact_cosine=None,
        collapsed_replay_elapsed_ns=0,
        branch_exchange_replay_elapsed_ns=0,
        exact_replay_elapsed_ns=0,
        predictor_forward_elapsed_ns=0,
        candidate_total_elapsed_ns=104_000_000,
    ).validated()
    fifth = rows[4]
    rows[4] = replace(
        fifth,
        exact_gradient_sha256="c" * 64,
        exact_gradient_norm=2.0,
        exact_replay_generated_tokens=128,
        collapsed_exact_cosine=0.5,
        exact_replay_elapsed_ns=400_000_000,
        candidate_total_elapsed_ns=506_000_000,
    ).validated()

    result = AsgcvP32Result.from_candidates(tuple(rows))
    assert result.exact_diagnostic_ordinals == (1, 2, 3, 4)


def test_p32_exact_diagnostic_can_record_variance_without_collapsed_field() -> None:
    candidate = replace(
        _candidate(0),
        rewards=(1, 1, 0, 0, 0, 0, 0, 0),
        valid_flags=(True, True, False, False, False, False, False, False),
        verdict_relation_signs=(1, 1, None, None, None, None, None, None),
        attribute_span_lengths=(3, 3, 0, 0, 0, 0, 0, 0),
        lowest_branch_indices=None,
        highest_branch_indices=None,
        branch_exchange_distinct=False,
        collapsed_branch_scores=None,
        collapsed_backend_coefficient_ppm=None,
        highest_branch_scores=None,
        highest_backend_coefficient_ppm=None,
        lowest_gradient_sha256=None,
        highest_gradient_sha256=None,
        lowest_gradient_norm=None,
        highest_gradient_norm=None,
        branch_exchange_energy_ppm=None,
        boundary_norms=None,
        collapsed_replay_elapsed_ns=0,
        branch_exchange_replay_elapsed_ns=0,
        collapsed_exact_cosine=None,
        predictor_forward_elapsed_ns=0,
        candidate_total_elapsed_ns=504_000_000,
    ).validated()
    assert candidate.nonzero_reward_variance is True
    assert candidate.both_verdicts_valid is False
    assert candidate.exact_diagnostic is True
    assert candidate.collapsed_exact_cosine is None
    assert candidate.predictor_forward_elapsed_ns == 0
    with pytest.raises(ValueError, match="predictor"):
        replace(candidate, predictor_forward_elapsed_ns=1).validated()


def test_p32_zero_gradient_evidence_is_recorded_as_degenerate_not_rejected() -> None:
    candidate = replace(
        _candidate(0),
        lowest_gradient_norm=0.0,
        highest_gradient_norm=0.0,
        branch_exchange_energy_ppm=1_000_000,
        boundary_norms=(0.0, 0.0, 0.0, 0.0),
        exact_gradient_norm=0.0,
        collapsed_exact_cosine=None,
    ).validated()
    assert candidate.lowest_gradient_norm == 0.0
    assert candidate.exact_gradient_norm == 0.0
    with pytest.raises(ValueError, match="exact diagnostic"):
        replace(candidate, exact_replay_generated_tokens=127).validated()


def test_p32_result_fails_closed_when_branch_exchange_is_not_evaluable() -> None:
    candidates = tuple(
        replace(
            _candidate(ordinal),
            rewards=(1, 0, 0, 0, 0, 0, 0, 0),
            valid_flags=(True, True, False, False, False, False, False, False),
            verdict_relation_signs=(
                _candidate(ordinal).relation_sign,
                -_candidate(ordinal).relation_sign,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
            lowest_branch_indices=(0, 1),
            highest_branch_indices=None,
            branch_exchange_distinct=False,
            collapsed_branch_scores=(-0.10, -0.11),
            collapsed_backend_coefficient_ppm=463_906,
            highest_branch_scores=None,
            highest_backend_coefficient_ppm=None,
            highest_gradient_sha256=None,
            highest_gradient_norm=None,
            branch_exchange_energy_ppm=None,
            branch_exchange_replay_elapsed_ns=0,
        ).validated()
        for ordinal in range(ASGCV_P32_PAIR_COUNT)
    )
    result = AsgcvP32Result.from_candidates(candidates)
    assert result.exchange_evaluable_candidates == 0
    assert result.branch_exchange_gate_passed is False
    assert result.dispersion_gate_passed is False
    assert result.passed is False


def test_p32_total_semantic_failure_still_emits_a_canonical_terminal() -> None:
    candidates = tuple(
        replace(
            _candidate(ordinal),
            rewards=(1,) * 8,
            verdict_relation_signs=(_candidate(ordinal).relation_sign,) * 8,
            lowest_branch_indices=None,
            highest_branch_indices=None,
            branch_exchange_distinct=False,
            collapsed_branch_scores=None,
            collapsed_backend_coefficient_ppm=None,
            highest_branch_scores=None,
            highest_backend_coefficient_ppm=None,
            lowest_gradient_sha256=None,
            highest_gradient_sha256=None,
            lowest_gradient_norm=None,
            highest_gradient_norm=None,
            branch_exchange_energy_ppm=None,
            boundary_norms=None,
            exact_gradient_sha256=None,
            exact_gradient_norm=None,
            exact_replay_generated_tokens=None,
            collapsed_exact_cosine=None,
            collapsed_replay_elapsed_ns=0,
            branch_exchange_replay_elapsed_ns=0,
            exact_replay_elapsed_ns=0,
            predictor_forward_elapsed_ns=0,
            candidate_total_elapsed_ns=104_000_000,
        ).validated()
        for ordinal in range(ASGCV_P32_PAIR_COUNT)
    )
    result = AsgcvP32Result.from_candidates(candidates)
    assert result.exact_diagnostic_ordinals == ()
    assert result.branch_yield_ppm == 0
    assert result.projected_step_wall_ratio_ppm == 1_000_000
    assert result.passed is False
    raw = canonical_asgcv_p32_result_bytes(candidates)
    receipts = tuple(canonical_asgcv_p32_candidate_bytes(row) for row in candidates)
    assert validate_asgcv_p32_result_bytes(raw, receipts) == result


def test_p32_dispersion_and_exchange_gates_require_powered_evidence() -> None:
    rows = []
    for ordinal in range(ASGCV_P32_PAIR_COUNT):
        candidate = _candidate(ordinal)
        if ordinal >= 7:
            relation = candidate.relation_sign
            candidate = replace(
                candidate,
                rewards=(1, 0, 0, 0, 0, 0, 0, 0),
                valid_flags=(True, True, False, False, False, False, False, False),
                verdict_relation_signs=(relation, -relation, None, None, None, None, None, None),
                lowest_branch_indices=(0, 1),
                highest_branch_indices=None,
                branch_exchange_distinct=False,
                collapsed_branch_scores=(-0.10, -0.11),
                collapsed_backend_coefficient_ppm=463_906,
                highest_branch_scores=None,
                highest_backend_coefficient_ppm=None,
                highest_gradient_sha256=None,
                highest_gradient_norm=None,
                branch_exchange_energy_ppm=None,
                branch_exchange_replay_elapsed_ns=0,
                exact_gradient_sha256=None if ordinal < 4 else candidate.exact_gradient_sha256,
                exact_gradient_norm=None if ordinal < 4 else candidate.exact_gradient_norm,
                exact_replay_generated_tokens=None
                if ordinal < 4
                else candidate.exact_replay_generated_tokens,
                collapsed_exact_cosine=None if ordinal < 4 else candidate.collapsed_exact_cosine,
                exact_replay_elapsed_ns=0 if ordinal < 4 else candidate.exact_replay_elapsed_ns,
            ).validated()
        rows.append(candidate)
    result = AsgcvP32Result.from_candidates(tuple(rows))
    assert result.exchange_evaluable_candidates == 7
    assert result.branch_exchange_gate_passed is False
    assert result.dispersion_gate_passed is False

    high_tail = tuple(
        replace(
            _candidate(ordinal),
            completion_scores=(-0.1, 10.0, -10.0, 0.0, -1.1, 10.0, -10.0, 0.0),
            highest_branch_scores=(0.0, 0.0),
            highest_backend_coefficient_ppm=463_914,
        ).validated()
        if ordinal >= 28
        else _candidate(ordinal)
        for ordinal in range(ASGCV_P32_PAIR_COUNT)
    )
    tail_result = AsgcvP32Result.from_candidates(high_tail)
    assert tail_result.median_dispersion_ratio_ppm < 250_000
    assert tail_result.p90_dispersion_ratio_ppm > 500_000
    assert tail_result.dispersion_gate_passed is False
