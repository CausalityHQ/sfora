from __future__ import annotations

from dataclasses import replace

import pytest

from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    assemble_asgcv_eligible_schedule,
    build_asgcv_pair_schedule,
    classify_asgcv_completion,
    classify_asgcv_completion_group,
    derive_asgcv_rollout_seeds,
    validate_asgcv_partition_bundle,
    validate_asgcv_protocol_bundle,
)


def _protocol() -> AsgcvCompletionProtocol:
    return AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22, 23),
        terminal_token_ids=(0, 99),
    ).validated()


def _rollout_authority() -> AsgcvRolloutAuthority:
    return AsgcvRolloutAuthority(
        master_seed_sha256="12" * 32,
        model_revision="3" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=1024,
    ).validated()


def _partition_authority() -> AsgcvPartitionAuthority:
    return AsgcvPartitionAuthority(
        source_manifest_sha256="9a" * 32,
        partition_seed_sha256="bc" * 32,
        predictor_train_class_ids=(0, 1),
        e0_validation_class_ids=(2, 3),
        e1_optimization_class_ids=(4, 5),
    ).validated()


def test_partition_authority_seals_disjoint_class_bands_and_image_identities() -> None:
    authority = _partition_authority()
    assert authority.to_mapping() == {
        "schema": "sfora-asgcv-partition-authority-v1",
        "source_manifest_sha256": "9a" * 32,
        "partition_seed_sha256": "bc" * 32,
        "predictor_train_class_ids": [0, 1],
        "e0_validation_class_ids": [2, 3],
        "e1_optimization_class_ids": [4, 5],
        "official_test_accessible": False,
    }
    assert type(authority).from_mapping(authority.to_mapping()) == authority
    assert len(authority.sha256()) == 64

    validate_asgcv_partition_bundle(
        authority,
        predictor_train=(('train-a', 'train-b'), (0, 1)),
        e0_validation=(('valid-a', 'valid-b'), (2, 3)),
        e1_optimization=(('optim-a', 'optim-b'), (4, 5)),
    )

    with pytest.raises(ValueError):
        validate_asgcv_partition_bundle(
            authority,
            predictor_train=(('shared', 'train-b'), (0, 1)),
            e0_validation=(('valid-a', 'shared'), (2, 3)),
            e1_optimization=(('optim-a', 'optim-b'), (4, 5)),
        )


def test_partition_authority_rejects_class_overlap_schema_and_role_drift() -> None:
    authority = _partition_authority()
    for mutation in (
        {**authority.to_mapping(), "official_test_accessible": True},
        {**authority.to_mapping(), "e0_validation_class_ids": [2, True]},
        {**authority.to_mapping(), "e1_optimization_class_ids": [3, 5]},
        {**authority.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(authority).from_mapping(mutation)

    with pytest.raises(ValueError):
        validate_asgcv_partition_bundle(
            authority,
            predictor_train=(('train-a', 'train-b'), (0, 2)),
            e0_validation=(('valid-a', 'valid-b'), (2, 3)),
            e1_optimization=(('optim-a', 'optim-b'), (4, 5)),
        )


def test_completion_protocol_is_exact_and_content_addressed() -> None:
    protocol = _protocol()
    assert protocol.to_mapping() == {
        "schema": "sfora-asgcv-completion-protocol-v1",
        "same_prefix_ids": [11, 12],
        "different_prefix_ids": [21, 22, 23],
        "terminal_token_ids": [0, 99],
    }
    assert AsgcvCompletionProtocol.from_mapping(protocol.to_mapping()) == protocol
    assert len(protocol.sha256()) == 64
    assert AsgcvCompletionProtocol.from_mapping(protocol.to_mapping()).sha256() == protocol.sha256()


def test_rollout_authority_binds_sampler_and_derives_pair_unique_seed_blocks() -> None:
    authority = _rollout_authority()
    assert authority.to_mapping() == {
        "schema": "sfora-asgcv-rollout-authority-v1",
        "master_seed_sha256": "12" * 32,
        "model_revision": "3" * 40,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 1024,
        "rollouts_per_pair": 8,
    }
    assert type(authority).from_mapping(authority.to_mapping()) == authority
    first = derive_asgcv_rollout_seeds(authority, candidate_pair_ordinal=5)
    assert len(first) == len(set(first)) == 8
    assert first == derive_asgcv_rollout_seeds(authority, candidate_pair_ordinal=5)
    assert first != derive_asgcv_rollout_seeds(authority, candidate_pair_ordinal=6)
    changed = replace(authority, model_revision="4" * 40).validated()
    assert first != derive_asgcv_rollout_seeds(changed, candidate_pair_ordinal=5)

    for mutation in (
        {**authority.to_mapping(), "temperature": True},
        {**authority.to_mapping(), "top_p": 0.0},
        {**authority.to_mapping(), "max_new_tokens": True},
        {**authority.to_mapping(), "rollouts_per_pair": 7},
        {**authority.to_mapping(), "extra": 1},
    ):
        with pytest.raises(ValueError):
            type(authority).from_mapping(mutation)
    with pytest.raises(ValueError):
        derive_asgcv_rollout_seeds(authority, candidate_pair_ordinal=True)


def test_completion_classification_binds_reward_verdict_and_attribute_span() -> None:
    protocol = _protocol()

    same = classify_asgcv_completion((11, 12, 31, 32, 99), 1, protocol)
    assert same.reward == 1
    assert same.verdict_relation_sign == 1
    assert same.attribute_span == (2, 4)
    assert same.valid is True

    wrong = classify_asgcv_completion((21, 22, 23, 41, 0), 1, protocol)
    assert wrong.reward == 0
    assert wrong.verdict_relation_sign == -1
    assert wrong.attribute_span == (3, 4)
    assert wrong.valid is True

    malformed = classify_asgcv_completion((77, 78, 99), -1, protocol)
    assert malformed.reward == 0
    assert malformed.verdict_relation_sign is None
    assert malformed.attribute_span is None
    assert malformed.valid is False


def test_completion_protocol_rejects_ambiguous_empty_and_concrete_type_drift() -> None:
    for kwargs in (
        {"same_prefix_ids": (), "different_prefix_ids": (2,), "terminal_token_ids": (0,)},
        {
            "same_prefix_ids": (1,),
            "different_prefix_ids": (1, 2),
            "terminal_token_ids": (0,),
        },
        {"same_prefix_ids": (True,), "different_prefix_ids": (2,), "terminal_token_ids": (0,)},
        {"same_prefix_ids": (1,), "different_prefix_ids": (2,), "terminal_token_ids": (0, 0)},
    ):
        with pytest.raises(ValueError):
            AsgcvCompletionProtocol(**kwargs).validated()

    mapping = _protocol().to_mapping()
    with pytest.raises(ValueError):
        AsgcvCompletionProtocol.from_mapping({**mapping, "extra": 1})
    with pytest.raises(ValueError):
        AsgcvCompletionProtocol.from_mapping({**mapping, "same_prefix_ids": [11, True]})


def test_completion_classification_rejects_empty_attributes_and_input_drift() -> None:
    protocol = _protocol()
    for completion in ((11, 12, 99), (11, 12, 0, 99), (21, 22, 23)):
        observed = classify_asgcv_completion(completion, 1, protocol)
        assert observed.valid is False
        assert observed.reward == 0
        assert observed.attribute_span is None

    with pytest.raises(ValueError):
        classify_asgcv_completion([11, 12, 31], 1, protocol)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        classify_asgcv_completion((11, True, 31), 1, protocol)
    with pytest.raises(ValueError):
        classify_asgcv_completion((11, 12, 31), 0, protocol)


def test_completion_group_derives_variance_eligibility_and_exact_teacher_spans() -> None:
    protocol = _protocol()
    completions = tuple(
        (11, 12, 30 + index, 99) if index < 4 else (21, 22, 23, 40 + index, 0)
        for index in range(8)
    )
    rollout = _rollout_authority()
    group = classify_asgcv_completion_group(
        completions,
        1,
        protocol,
        rollout_authority=rollout,
        candidate_pair_ordinal=7,
    )

    assert group.rewards == (1, 1, 1, 1, 0, 0, 0, 0)
    assert group.correct_rollouts == (True, True, True, True, False, False, False, False)
    assert group.attribute_spans == ((2, 3), (2, 3), (2, 3), (2, 3), None, None, None, None)
    assert group.nonzero_reward_variance is True
    assert group.candidate_pair_ordinal == 7
    assert group.rollout_authority_sha256 == rollout.sha256()
    assert group.generation_seeds == derive_asgcv_rollout_seeds(
        rollout,
        candidate_pair_ordinal=7,
    )
    assert len(group.sha256()) == 64
    assert (
        classify_asgcv_completion_group(
            completions,
            1,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=7,
        )
        == group
    )
    assert type(group).from_mapping(group.to_mapping()) == group
    for mutation in (
        {**group.to_mapping(), "candidate_pair_ordinal": True},
        {**group.to_mapping(), "rollout_authority_sha256": True},
        {**group.to_mapping(), "generation_seeds": [True, *group.generation_seeds[1:]]},
    ):
        with pytest.raises(ValueError):
            type(group).from_mapping(mutation)

    all_correct = tuple((11, 12, 30 + index, 99) for index in range(8))
    assert classify_asgcv_completion_group(
        all_correct,
        1,
        protocol,
        rollout_authority=rollout,
        candidate_pair_ordinal=7,
    ).nonzero_reward_variance is False

    with pytest.raises(ValueError):
        classify_asgcv_completion_group(
            completions[:7],
            1,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=7,
        )


def test_eligible_schedule_refills_before_gradients_and_preserves_relation_balance() -> None:
    protocol = _protocol()
    rollout = _rollout_authority()
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    )
    groups = []
    eligible_seen = {-1: 0, 1: 0}
    for pair in candidates.pairs:
        eligible = eligible_seen[pair.relation_sign] < 4
        if eligible:
            eligible_seen[pair.relation_sign] += 1
        correct_prefix = (11, 12) if pair.relation_sign == 1 else (21, 22, 23)
        wrong_prefix = (21, 22, 23) if pair.relation_sign == 1 else (11, 12)
        completions = tuple(
            (*correct_prefix, 30 + index, 99)
            if eligible and index < 4
            else (*wrong_prefix, 50 + index, 0)
            for index in range(8)
        )
        groups.append(
            classify_asgcv_completion_group(
                completions,
                pair.relation_sign,
                protocol,
                rollout_authority=rollout,
                candidate_pair_ordinal=pair.ordinal,
            )
        )

    selected = assemble_asgcv_eligible_schedule(
        candidates,
        tuple(groups),
        target_pair_count=8,
    )
    assert selected.target_pair_count == 8
    assert len(selected.candidate_ordinals) == 8
    assert len(set(selected.candidate_ordinals)) == 8
    signs = [candidates.pairs[index].relation_sign for index in selected.candidate_ordinals]
    assert signs.count(1) == signs.count(-1) == 4
    assert type(selected).from_mapping(selected.to_mapping()) == selected
    assert len(selected.sha256()) == 64

    with pytest.raises(ValueError, match="eligible pair capacity"):
        assemble_asgcv_eligible_schedule(
            candidates,
            tuple(groups),
            target_pair_count=16,
        )


def test_protocol_bundle_rebuilds_schedule_groups_and_eligibility() -> None:
    protocol = _protocol()
    rollout = _rollout_authority()
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    candidates = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    )
    groups = tuple(
        classify_asgcv_completion_group(
            tuple(
                (
                    *((11, 12) if index < 4 else (21, 22, 23)),
                    30 + index,
                    99,
                )
                if pair.relation_sign == 1
                else (
                    *((21, 22, 23) if index < 4 else (11, 12)),
                    30 + index,
                    99,
                )
                for index in range(8)
            ),
            pair.relation_sign,
            protocol,
            rollout_authority=rollout,
            candidate_pair_ordinal=pair.ordinal,
        )
        for pair in candidates.pairs
    )
    eligible = assemble_asgcv_eligible_schedule(candidates, groups, target_pair_count=8)

    validate_asgcv_protocol_bundle(
        protocol,
        rollout,
        candidates,
        groups,
        eligible,
        example_ids=example_ids,
        labels=labels,
    )
    with pytest.raises(ValueError):
        validate_asgcv_protocol_bundle(
            protocol,
            replace(rollout, model_revision="4" * 40).validated(),
            candidates,
            groups,
            eligible,
            example_ids=example_ids,
            labels=labels,
        )

    first = groups[0]
    forged_group = replace(
        first,
        completion_ids=tuple((77, 30 + index, 99) for index in range(8)),
        rewards=(0,) * 8,
        correct_rollouts=(False,) * 8,
        attribute_spans=(None,) * 8,
        nonzero_reward_variance=False,
    ).validated()
    mutations = (
        (candidates, (forged_group, *groups[1:]), eligible),
        (
            replace(candidates, example_manifest_sha256="cd" * 32).validated(),
            groups,
            eligible,
        ),
        (
            candidates,
            groups,
            replace(
                eligible,
                candidate_ordinals=tuple(reversed(eligible.candidate_ordinals)),
            ).validated(),
        ),
    )
    for mutated_candidates, mutated_groups, mutated_eligible in mutations:
        with pytest.raises(ValueError):
            validate_asgcv_protocol_bundle(
                protocol,
                rollout,
                mutated_candidates,
                mutated_groups,
                mutated_eligible,
                example_ids=example_ids,
                labels=labels,
            )


def test_pair_schedule_is_balanced_disjoint_stratified_and_replayable() -> None:
    example_ids = tuple(f"cars-{index:02d}" for index in range(32))
    labels = tuple(index // 4 for index in range(32))
    first = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    )
    second = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    )

    assert first == second
    assert type(first).from_mapping(first.to_mapping()) == first
    assert first.pair_count == 16
    assert len(first.pairs) == 16
    assert len(first.sha256()) == 64
    assert [pair.ordinal for pair in first.pairs] == list(range(16))
    assert [pair.relation_sign for pair in first.pairs].count(1) == 8
    assert [pair.relation_sign for pair in first.pairs].count(-1) == 8
    for offset in range(0, first.pair_count, 8):
        signs = [pair.relation_sign for pair in first.pairs[offset : offset + 8]]
        assert signs.count(1) == signs.count(-1) == 4
    used = [index for pair in first.pairs for index in (pair.left_index, pair.right_index)]
    assert sorted(used) == list(range(32))
    for pair in first.pairs:
        same = labels[pair.left_index] == labels[pair.right_index]
        assert same is (pair.relation_sign == 1)

    changed = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="cd" * 32,
        pair_count=16,
    )
    assert changed.sha256() != first.sha256()


def test_pair_schedule_rejects_identity_label_capacity_and_count_drift() -> None:
    example_ids = tuple(f"cars-{index:02d}" for index in range(16))
    labels = tuple(index // 4 for index in range(16))
    mutations = (
        (example_ids[:-1] + (example_ids[0],), labels, 8),
        (example_ids, labels[:-1] + (True,), 8),
        (example_ids, (0,) * 16, 8),
        (example_ids, labels, 7),
        (example_ids[:8], labels[:8], 8),
    )
    for ids, values, count in mutations:
        with pytest.raises(ValueError):
            build_asgcv_pair_schedule(
                ids,
                values,
                schedule_seed_sha256="ab" * 32,
                pair_count=count,
            )

    baseline = build_asgcv_pair_schedule(
        tuple(f"cars-{index:02d}" for index in range(32)),
        tuple(index // 4 for index in range(32)),
        schedule_seed_sha256="ab" * 32,
        pair_count=16,
    ).to_mapping()
    for mutation in (
        {**baseline, "extra": 1},
        {**baseline, "example_manifest_sha256": True},
        {**baseline, "pairs": baseline["pairs"][:-1]},
        {
            **baseline,
            "pairs": [{**baseline["pairs"][0], "left_index": True}, *baseline["pairs"][1:]],
        },
    ):
        with pytest.raises(ValueError):
            AsgcvPairSchedule.from_mapping(mutation)
