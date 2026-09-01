from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sfora.asgcv import canonical_gradient_sample_bytes
from sfora.asgcv_marginal import (
    ASGCV_VISION_BOUNDARIES,
    AsgcvVisionCutAuthority,
    canonical_marginal_gradient_sample_bytes,
)
from sfora.asgcv_pilot import (
    ASGCV_P32_BOUNDARY_NAMES,
    ASGCV_P32_PAIR_COUNT,
    AsgcvP32Candidate,
    validate_asgcv_p32_result_bytes,
)
from sfora.asgcv_protocol import (
    AsgcvCompletionGroup,
    AsgcvCompletionProtocol,
    AsgcvMarginalSchedule,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    assemble_asgcv_marginal_schedule,
    build_asgcv_pair_schedule,
    canonical_asgcv_completion_group_bytes,
    classify_asgcv_completion_group,
    derive_asgcv_rollout_seeds,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_e0.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_e0_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_p32_atomic_receipt_fsyncs_file_and_parent_directory(tmp_path: Path, monkeypatch) -> None:
    original = os.fsync
    calls: list[int] = []

    def recording_fsync(descriptor: int) -> None:
        calls.append(descriptor)
        original(descriptor)

    monkeypatch.setattr(_MODULE.os, "fsync", recording_fsync)
    _MODULE._write_atomic_bytes(tmp_path / "candidate.json", b"{}\n")
    assert (tmp_path / "candidate.json").read_bytes() == b"{}\n"
    assert len(calls) == 2


def _sample(ordinal: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    patch = np.full((2, 49, 4), ordinal + 1, dtype=np.float32)
    gradient = patch * np.float32(0.25)
    receipt = canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=f"{ordinal + 1:064x}",
        completion_protocol_sha256="4" * 64,
        eligible_schedule_sha256="5" * 64,
        pooler_state_sha256="6" * 64,
        eligible_pair_ordinal=ordinal,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(ordinal * 2, ordinal * 2 + 1),
        relation_sign=1 if ordinal % 2 == 0 else -1,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=8,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    return receipt, patch, gradient


def _marginal_sample(
    *,
    candidate_ordinal: int,
    candidate_schedule: AsgcvPairSchedule,
    completion_groups: tuple[AsgcvCompletionGroup, ...],
    marginal_schedule: AsgcvMarginalSchedule,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    pair = candidate_schedule.pairs[candidate_ordinal]
    group = completion_groups[candidate_ordinal]
    cut = AsgcvVisionCutAuthority(
        boundary_names=ASGCV_VISION_BOUNDARIES,
        images=2,
        patches_per_boundary=49,
        channel_dimensions=4,
    ).validated()
    patch = np.full((2, 196, 4), candidate_ordinal + 1, dtype=np.float32)
    zero = marginal_schedule.zero_target_flags[candidate_ordinal]
    gradient = np.zeros_like(patch) if zero else patch * np.float32(0.25)
    receipt = canonical_marginal_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=group.sha256(),
        completion_protocol_sha256=group.protocol_sha256,
        marginal_schedule_sha256=marginal_schedule.sha256(),
        pooler_state_sha256="6" * 64,
        candidate_pair_ordinal=candidate_ordinal,
        pair_ordinals=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        zero_semantic_target=zero,
        replay_branch_count=0 if zero else 2,
        branch_completion_indices=None if zero else (0, 4),
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=0,
        vision_cut_authority=cut,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    return receipt, patch, gradient


def test_capture_triples_are_atomic_idempotent_and_resume_from_first_absent(tmp_path: Path) -> None:
    for ordinal in range(2):
        receipt, patch, gradient = _sample(ordinal)
        assert _MODULE.write_capture_triple(
            tmp_path,
            ordinal=ordinal,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        ) == ("written" if ordinal == 0 else "written")
        assert not tuple(tmp_path.glob("*.partial"))
    assert _MODULE.validated_capture_prefix(tmp_path, expected_count=4) == 2

    receipt, patch, gradient = _sample(1)
    assert (
        _MODULE.write_capture_triple(
            tmp_path,
            ordinal=1,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        == "reused"
    )


def test_capture_resume_rejects_partial_gap_corruption_and_shape_drift(tmp_path: Path) -> None:
    receipt, patch, gradient = _sample(0)
    _MODULE.write_capture_triple(
        tmp_path,
        ordinal=0,
        receipt=receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    np.save(tmp_path / "patch-000001.npy", patch, allow_pickle=False)
    with pytest.raises(ValueError, match="partial"):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    (tmp_path / "patch-000001.npy").unlink()
    np.save(tmp_path / "gradient-000000.npy", gradient + np.float32(1.0), allow_pickle=False)
    with pytest.raises(ValueError):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    other = tmp_path / "other"
    other.mkdir()
    bad_patch = np.zeros((2, 48, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt,
            patch_tokens=bad_patch,
            exact_gradient=np.zeros_like(bad_patch),
        )

    receipt_one, patch_one, gradient_one = _sample(1)
    with pytest.raises(ValueError, match="ordinal"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt_one,
            patch_tokens=patch_one,
            exact_gradient=gradient_one,
        )
    assert not tuple(other.iterdir())


def _protocol_bundle() -> tuple[object, ...]:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="8" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="9" * 64,
        pair_count=16,
    )
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11, 12) if pair.relation_sign == 1 else (21, 22)),
                    30 + rollout_ordinal,
                    99,
                )
                if rollout_ordinal < 4
                else (
                    *((21, 22) if pair.relation_sign == 1 else (11, 12)),
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
    eligible = assemble_asgcv_eligible_schedule(candidates, groups, target_pair_count=8)
    return protocol, rollout, example_ids, labels, candidates, groups, eligible


def _p32_bundle() -> tuple[
    AsgcvCompletionProtocol,
    AsgcvRolloutAuthority,
    AsgcvPartitionAuthority,
    str,
    tuple[tuple[str, ...], tuple[int, ...]],
    tuple[tuple[str, ...], tuple[int, ...]],
    tuple[tuple[str, ...], tuple[int, ...]],
    AsgcvPairSchedule,
    tuple[AsgcvCompletionGroup, ...],
]:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="8" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    partition = AsgcvPartitionAuthority(
        source_manifest_sha256="a" * 64,
        partition_seed_sha256="b" * 64,
        predictor_train_class_ids=tuple(range(16)),
        e0_validation_class_ids=tuple(range(16, 20)),
        e1_optimization_class_ids=tuple(range(20, 24)),
    ).validated()
    source_commit = "1" * 40
    example_ids = tuple(f"p32-{index:03d}" for index in range(64))
    labels = tuple(index // 4 for index in range(64))
    predictor_train = (example_ids, labels)
    e0_validation = (
        tuple(f"valid-{index:03d}" for index in range(16)),
        tuple(16 + index // 4 for index in range(16)),
    )
    e1_optimization = (
        tuple(f"optim-{index:03d}" for index in range(16)),
        tuple(20 + index // 4 for index in range(16)),
    )
    from sfora.asgcv_pilot import derive_asgcv_p32_schedule_seed

    schedule = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256=derive_asgcv_p32_schedule_seed(
            partition_authority=partition,
            source_commit=source_commit,
        ),
        pair_count=ASGCV_P32_PAIR_COUNT,
    )
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11, 12) if pair.relation_sign == 1 else (21, 22)),
                    30 + rollout_ordinal,
                    99,
                )
                if rollout_ordinal < 4
                else (
                    *((21, 22) if pair.relation_sign == 1 else (11, 12)),
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
        for pair in schedule.pairs
    )
    return (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    )


def _p32_candidate(
    ordinal: int,
    *,
    protocol: AsgcvCompletionProtocol,
    rollout: AsgcvRolloutAuthority,
    partition: AsgcvPartitionAuthority,
    schedule: AsgcvPairSchedule,
    group: AsgcvCompletionGroup,
    exact: bool,
) -> AsgcvP32Candidate:
    pair = schedule.pairs[ordinal]
    return AsgcvP32Candidate(
        source_commit="1" * 40,
        model_revision=rollout.model_revision,
        fixture_sha256="3" * 64,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        partition_authority_sha256=partition.sha256(),
        pilot_schedule_sha256=schedule.sha256(),
        completion_protocol_sha256=protocol.sha256(),
        rollout_authority_sha256=rollout.sha256(),
        completion_group_sha256=group.sha256(),
        pooler_state_sha256="6" * 64,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        generation_seeds=group.generation_seeds,
        rewards=group.rewards,
        valid_flags=group.valid_flags,
        verdict_relation_signs=group.verdict_relation_signs,
        attribute_span_lengths=tuple(
            0 if span is None else span[1] - span[0] for span in group.attribute_spans
        ),
        generated_token_counts=tuple(len(row) for row in group.completion_ids),
        completion_scores=(-0.10, -0.11, -0.09, -0.10, -1.10, -1.11, -1.09, -1.10),
        lowest_branch_indices=(0, 4),
        highest_branch_indices=(3, 7),
        branch_exchange_distinct=True,
        collapsed_branch_scores=(-0.10, -1.10),
        collapsed_backend_coefficient_ppm=393_256,
        highest_branch_scores=(-0.10, -1.10),
        highest_backend_coefficient_ppm=393_256,
        lowest_gradient_sha256="7" * 64,
        highest_gradient_sha256="8" * 64,
        lowest_gradient_norm=2.0,
        highest_gradient_norm=2.0,
        branch_exchange_energy_ppm=100_000,
        boundary_names=ASGCV_P32_BOUNDARY_NAMES,
        boundary_norms=(1.0, 1.0, 1.0, 1.0),
        exact_gradient_sha256="9" * 64 if exact else None,
        exact_gradient_norm=2.0 if exact else None,
        exact_replay_generated_tokens=32 if exact else None,
        collapsed_exact_cosine=0.5 if exact else None,
        prepare_elapsed_ns=1_000_000,
        generate_elapsed_ns=100_000_000,
        score_elapsed_ns=2_000_000,
        collapsed_replay_elapsed_ns=1_000_000,
        branch_exchange_replay_elapsed_ns=1_000_000,
        exact_replay_elapsed_ns=400_000_000 if exact else 0,
        predictor_forward_elapsed_ns=1_000_000,
        candidate_total_elapsed_ns=506_000_000 if exact else 106_000_000,
        peak_cuda_reserved_bytes=1_000_000,
        peak_rss_bytes=2_000_000,
    ).validated()


def test_p32_campaign_publishes_atomic_receipts_and_resumes_without_reexecution(
    tmp_path: Path,
) -> None:
    (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    ) = _p32_bundle()
    calls: list[tuple[int, bool]] = []

    def execute_one(ordinal: int, exact_diagnostic: bool) -> tuple[object, object]:
        calls.append((ordinal, exact_diagnostic))
        group = groups[ordinal]
        return group, _p32_candidate(
            ordinal,
            protocol=protocol,
            rollout=rollout,
            partition=partition,
            schedule=schedule,
            group=group,
            exact=exact_diagnostic,
        )

    result = _MODULE.run_p32_campaign(
        tmp_path,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        partition_authority=partition,
        source_commit=source_commit,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
        execute_one=execute_one,
    )
    assert calls == [(ordinal, ordinal < 4) for ordinal in range(ASGCV_P32_PAIR_COUNT)]
    receipts = tuple(
        _MODULE._validate_p32_row_bytes((tmp_path / f"candidate-{ordinal:06d}.json").read_bytes())[
            2
        ]
        for ordinal in range(ASGCV_P32_PAIR_COUNT)
    )
    assert validate_asgcv_p32_result_bytes(result, receipts).passed is True
    assert (tmp_path / "result.json").read_bytes() == result
    assert not tuple(tmp_path.glob("*.partial"))

    assert (
        _MODULE.run_p32_campaign(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=partition,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("authenticated prefix must not reexecute"),
        )
        == result
    )


def test_p32_campaign_publishes_canonical_failure_terminal_without_partial_row(
    tmp_path: Path,
) -> None:
    (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    ) = _p32_bundle()

    def execute_one(ordinal: int, exact_diagnostic: bool) -> tuple[object, object]:
        if ordinal == 1:
            raise MemoryError("bounded fixture OOM")
        group = groups[ordinal]
        return group, _p32_candidate(
            ordinal,
            protocol=protocol,
            rollout=rollout,
            partition=partition,
            schedule=schedule,
            group=group,
            exact=exact_diagnostic,
        )

    raw = _MODULE.run_p32_campaign_with_failure_terminal(
        tmp_path,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        partition_authority=partition,
        source_commit=source_commit,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
        execute_one=execute_one,
    )
    value = _MODULE._validate_p32_failure_bytes(raw)
    assert value["failed_candidate_ordinal"] == 1
    assert value["failure_kind"] == "memory-error"
    assert value["failure_phase"] == "candidate-execution"
    assert len(value["completed_candidate_sha256s"]) == 1
    assert value["claim_eligible"] is False
    assert value["official_test_access"] is False
    assert (tmp_path / "failure.json").read_bytes() == raw
    assert (tmp_path / "candidate-000000.json").is_file()
    assert not (tmp_path / "candidate-000001.json").exists()
    assert not (tmp_path / "result.json").exists()
    assert not tuple(tmp_path.glob("*.partial"))

    with pytest.raises(ValueError, match="failure terminal"):
        _MODULE.run_p32_campaign_with_failure_terminal(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=partition,
            source_commit=source_commit,
            launch_authority_sha256="4" * 64,
            predictor_initialization_seed_sha256="5" * 64,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("a terminal campaign must not restart"),
        )
    with pytest.raises(ValueError, match="context"):
        _MODULE.run_p32_campaign_with_failure_terminal(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=partition,
            source_commit=source_commit,
            launch_authority_sha256="e" * 64,
            predictor_initialization_seed_sha256="5" * 64,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("a terminal campaign must not restart"),
        )


def test_p32_campaign_preserves_result_assembly_failure_after_all_candidates(
    tmp_path: Path,
) -> None:
    (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    ) = _p32_bundle()

    def execute_one(ordinal: int, exact_diagnostic: bool) -> tuple[object, object]:
        candidate = _p32_candidate(
            ordinal,
            protocol=protocol,
            rollout=rollout,
            partition=partition,
            schedule=schedule,
            group=groups[ordinal],
            exact=exact_diagnostic,
        )
        if ordinal == ASGCV_P32_PAIR_COUNT - 1:
            candidate = replace(candidate, pooler_state_sha256="f" * 64).validated()
        return groups[ordinal], candidate

    raw = _MODULE.run_p32_campaign_with_failure_terminal(
        tmp_path,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        partition_authority=partition,
        source_commit=source_commit,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
        execute_one=execute_one,
    )
    value = _MODULE._validate_p32_failure_bytes(raw)
    assert value["failure_phase"] == "result-assembly"
    assert value["failed_candidate_ordinal"] == ASGCV_P32_PAIR_COUNT
    assert len(value["completed_candidate_sha256s"]) == ASGCV_P32_PAIR_COUNT
    assert (tmp_path / "failure.json").read_bytes() == raw
    assert not (tmp_path / "result.json").exists()


def test_p32_campaign_rejects_partial_or_context_drift_before_execution(tmp_path: Path) -> None:
    (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    ) = _p32_bundle()
    (tmp_path / "group-000000.json").write_bytes(canonical_asgcv_completion_group_bytes(groups[0]))
    with pytest.raises(ValueError, match="partial"):
        _MODULE.run_p32_campaign(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=partition,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("partial campaign must not execute"),
        )

    (tmp_path / "group-000000.json").unlink()
    group = groups[0]
    candidate = _p32_candidate(
        0,
        protocol=protocol,
        rollout=rollout,
        partition=partition,
        schedule=schedule,
        group=group,
        exact=True,
    )
    (tmp_path / "candidate-000000.json").write_bytes(
        _MODULE._canonical_p32_row_bytes(group, candidate)
    )
    drift = replace(schedule, schedule_seed_sha256="b" * 64).validated()
    with pytest.raises(ValueError):
        _MODULE.run_p32_campaign(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=drift,
            partition_authority=partition,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("drifted campaign must not execute"),
        )

    with pytest.raises(ValueError, match="schedule|partition"):
        _MODULE.run_p32_campaign(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=replace(partition, partition_seed_sha256="c" * 64).validated(),
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=lambda *_: pytest.fail("drifted partition must not execute"),
        )


def test_p32_campaign_commits_each_group_and_candidate_as_one_resumable_row(
    tmp_path: Path,
) -> None:
    (
        protocol,
        rollout,
        partition,
        source_commit,
        predictor_train,
        e0_validation,
        e1_optimization,
        schedule,
        groups,
    ) = _p32_bundle()
    interrupted_calls: list[int] = []

    def interrupt_after_one(ordinal: int, exact: bool) -> tuple[object, object]:
        interrupted_calls.append(ordinal)
        if ordinal == 1:
            raise RuntimeError("interrupted")
        return groups[ordinal], _p32_candidate(
            ordinal,
            protocol=protocol,
            rollout=rollout,
            partition=partition,
            schedule=schedule,
            group=groups[ordinal],
            exact=exact,
        )

    with pytest.raises(RuntimeError, match="interrupted"):
        _MODULE.run_p32_campaign(
            tmp_path,
            rollout_authority=rollout,
            pilot_schedule=schedule,
            partition_authority=partition,
            source_commit=source_commit,
            predictor_train=predictor_train,
            e0_validation=e0_validation,
            e1_optimization=e1_optimization,
            execute_one=interrupt_after_one,
        )
    assert interrupted_calls == [0, 1]
    assert tuple(path.name for path in tmp_path.iterdir()) == ("candidate-000000.json",)

    resumed_calls: list[int] = []

    def resume(ordinal: int, exact: bool) -> tuple[object, object]:
        resumed_calls.append(ordinal)
        return groups[ordinal], _p32_candidate(
            ordinal,
            protocol=protocol,
            rollout=rollout,
            partition=partition,
            schedule=schedule,
            group=groups[ordinal],
            exact=exact,
        )

    _MODULE.run_p32_campaign(
        tmp_path,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        partition_authority=partition,
        source_commit=source_commit,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
        execute_one=resume,
    )
    assert resumed_calls == list(range(1, ASGCV_P32_PAIR_COUNT))
    assert not tuple(tmp_path.glob("group-*.json"))


def test_capture_schedule_validates_context_and_skips_authenticated_prefix(tmp_path: Path) -> None:
    protocol, rollout, example_ids, labels, candidates, groups, eligible = _protocol_bundle()
    calls: list[int] = []

    def capture_one(
        eligible_ordinal: int, candidate_ordinal: int
    ) -> tuple[bytes, np.ndarray, np.ndarray]:
        calls.append(eligible_ordinal)
        pair = candidates.pairs[candidate_ordinal]
        group = groups[candidate_ordinal]
        patch = np.full((2, 49, 4), eligible_ordinal + 1, dtype=np.float32)
        gradient = patch * np.float32(0.25)
        receipt = canonical_gradient_sample_bytes(
            source_commit="1" * 40,
            model_revision="2" * 40,
            fixture_sha256="3" * 64,
            completion_group_sha256=group.sha256(),
            completion_protocol_sha256=protocol.sha256(),
            eligible_schedule_sha256=eligible.sha256(),
            pooler_state_sha256="6" * 64,
            eligible_pair_ordinal=eligible_ordinal,
            candidate_pair_ordinal=candidate_ordinal,
            pair_ordinals=(pair.left_index, pair.right_index),
            relation_sign=pair.relation_sign,
            grpo_loss=0.0,
            attention_kl=0.0,
            generated_tokens=8,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        return receipt, patch, gradient

    assert (
        _MODULE.capture_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            eligible_schedule=eligible,
            example_ids=example_ids,
            labels=labels,
            capture_one=capture_one,
        )
        == 8
    )
    assert calls == list(range(8))
    assert (
        _MODULE.capture_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            eligible_schedule=eligible,
            example_ids=example_ids,
            labels=labels,
            capture_one=lambda *_: pytest.fail("authenticated prefix must not recapture"),
        )
        == 8
    )


def test_eligibility_generates_exact_source_seeded_groups_before_capture() -> None:
    protocol, rollout, example_ids, labels, candidates, _, _ = _protocol_bundle()

    class Adapter:
        def __init__(self) -> None:
            self.group = -1
            self.calls: list[int] = []

        def prepare_image_pair(self, images: object, *args: object) -> int:
            assert isinstance(images, tuple) and len(images) == 2
            assert all(isinstance(image, np.ndarray) for image in images)
            self.group += 1
            return self.group

        def generate(self, pair: int, seed: int, **kwargs: object) -> tuple[int, ...]:
            assert kwargs == {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 128}
            self.calls.append(seed)
            relation = candidates.pairs[pair].relation_sign
            local = len(self.calls) % 8
            correct = local in {1, 2, 3, 4}
            verdict = relation if correct else -relation
            prefix = (11, 12) if verdict == 1 else (21, 22)
            return (*prefix, 77, 99)

    adapter = Adapter()
    images = tuple(np.full((8, 8, 3), index, dtype=np.uint8) for index in range(32))
    groups, eligible = _MODULE.build_eligibility_schedule(
        adapter,
        images=images,
        prompt_utf8="Describe the relation.",
        attribute_token_span=(2, 3),
        patch_tokens_per_image=4,
        protocol=protocol,
        rollout_authority=rollout,
        candidate_schedule=candidates,
        target_pair_count=8,
        example_ids=example_ids,
        labels=labels,
    )
    assert len(groups) == 16
    assert eligible.target_pair_count == 8
    assert all(group.nonzero_reward_variance for group in groups)
    assert adapter.calls == [
        seed
        for pair in candidates.pairs
        for seed in derive_asgcv_rollout_seeds(
            rollout,
            candidate_pair_ordinal=pair.ordinal,
        )
    ]


def test_marginal_schedule_keeps_zero_variance_and_duplicate_completion_groups() -> None:
    protocol, rollout, example_ids, labels, candidates, _, _ = _protocol_bundle()

    class Adapter:
        def __init__(self) -> None:
            self.group = -1

        def prepare_image_pair(self, images: object, *args: object) -> int:
            assert isinstance(images, tuple) and len(images) == 2
            self.group += 1
            return self.group

        def generate(self, pair: int, seed: int, **kwargs: object) -> tuple[int, ...]:
            del seed, kwargs
            relation = candidates.pairs[pair].relation_sign
            if pair % 2 == 0:
                prefix = (11, 12) if relation == 1 else (21, 22)
                return (*prefix, 77, 99)
            rollout = getattr(self, "rollout", 0)
            self.rollout = rollout + 1
            correct = rollout % 8 < 4
            verdict = relation if correct else -relation
            prefix = (11, 12) if verdict == 1 else (21, 22)
            return (*prefix, 77 + rollout % 8, 99)

    images = tuple(np.full((8, 8, 3), index, dtype=np.uint8) for index in range(32))
    groups, schedule = _MODULE.build_marginal_schedule(
        Adapter(),
        images=images,
        prompt_utf8="Describe the relation.",
        attribute_token_span=(2, 3),
        patch_tokens_per_image=4,
        protocol=protocol,
        rollout_authority=rollout,
        candidate_schedule=candidates,
        example_ids=example_ids,
        labels=labels,
    )
    assert schedule.candidate_ordinals == tuple(range(candidates.pair_count))
    assert schedule.zero_target_flags == tuple(
        not group.nonzero_reward_variance for group in groups
    )
    assert schedule.zero_target_flags[0] is True
    assert len(set(groups[0].completion_ids)) == 1


def test_marginal_capture_triples_store_complete_cut_and_resume_zero_targets(
    tmp_path: Path,
) -> None:
    protocol, rollout, example_ids, labels, candidates, _, _ = _protocol_bundle()

    class Adapter:
        def prepare_image_pair(self, images: object, *args: object) -> int:
            assert isinstance(images, tuple) and len(images) == 2
            return 0

        def generate(self, pair: int, seed: int, **kwargs: object) -> tuple[int, ...]:
            del pair, seed, kwargs
            return 11, 12, 77, 99

    images = tuple(np.full((8, 8, 3), index, dtype=np.uint8) for index in range(32))
    groups, marginal = _MODULE.build_marginal_schedule(
        Adapter(),
        images=images,
        prompt_utf8="Describe the relation.",
        attribute_token_span=(2, 3),
        patch_tokens_per_image=4,
        protocol=protocol,
        rollout_authority=rollout,
        candidate_schedule=candidates,
        example_ids=example_ids,
        labels=labels,
    )
    receipt, patch, gradient = _marginal_sample(
        candidate_ordinal=0,
        candidate_schedule=candidates,
        completion_groups=groups,
        marginal_schedule=marginal,
    )
    assert candidates.pairs[0].relation_sign == 1
    assert marginal.zero_target_flags[0] is True
    assert (
        _MODULE.write_marginal_capture_triple(
            tmp_path,
            ordinal=0,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        == "written"
    )
    assert _MODULE.validated_marginal_capture_prefix(tmp_path, expected_count=16) == 1
    assert (
        _MODULE.write_marginal_capture_triple(
            tmp_path,
            ordinal=0,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        == "reused"
    )
    with pytest.raises(ValueError, match="complete-cut"):
        _MODULE.write_marginal_capture_triple(
            tmp_path,
            ordinal=1,
            receipt=receipt,
            patch_tokens=patch[:, :49],
            exact_gradient=gradient[:, :49],
        )


def test_marginal_capture_schedule_preserves_candidate_order_and_zero_semantics(
    tmp_path: Path,
) -> None:
    protocol, rollout, example_ids, labels, candidates, _, _ = _protocol_bundle()
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11, 12) if pair.relation_sign == 1 else (21, 22)),
                    30 + rollout_ordinal,
                    99,
                )
                if pair.ordinal % 2 == 0 or rollout_ordinal < 4
                else (
                    *((21, 22) if pair.relation_sign == 1 else (11, 12)),
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
    marginal = assemble_asgcv_marginal_schedule(candidates, groups)
    calls: list[tuple[int, bool]] = []

    def capture_one(
        candidate_ordinal: int, zero_semantic_target: bool
    ) -> tuple[bytes, np.ndarray, np.ndarray]:
        calls.append((candidate_ordinal, zero_semantic_target))
        return _marginal_sample(
            candidate_ordinal=candidate_ordinal,
            candidate_schedule=candidates,
            completion_groups=groups,
            marginal_schedule=marginal,
        )

    assert (
        _MODULE.capture_marginal_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            marginal_schedule=marginal,
            example_ids=example_ids,
            labels=labels,
            capture_one=capture_one,
        )
        == 16
    )
    assert calls == list(enumerate(marginal.zero_target_flags))
    assert (
        _MODULE.capture_marginal_schedule(
            tmp_path,
            protocol=protocol,
            rollout_authority=rollout,
            candidate_schedule=candidates,
            completion_groups=groups,
            marginal_schedule=marginal,
            example_ids=example_ids,
            labels=labels,
            capture_one=lambda *_: pytest.fail("authenticated prefix must not recapture"),
        )
        == 16
    )
