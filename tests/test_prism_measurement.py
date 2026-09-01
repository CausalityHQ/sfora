from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from sfora.prism_measurement import (
    PRISM_CHANNELS,
    PrismExample,
    build_prism_schedules,
    release_prism_observation_capability,
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
        phase="calibration",
    )
    assert len(calibration_capability) == 128 * len(PRISM_CHANNELS)
    assert max(row.pair_ordinal for row in calibration_capability) == 127
    assert all("fold" not in asdict(row) for row in calibration_capability)
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
        release_prism_observation_capability(observations, phase="diagnostic")
    diagnostic_capability = release_prism_observation_capability(
        observations,
        phase="diagnostic",
        calibration_receipt_sha256="f" * 64,
    )
    assert len(diagnostic_capability) == 32 * len(PRISM_CHANNELS)
    assert min(row.pair_ordinal for row in diagnostic_capability) == 128
    assert all("fold" not in asdict(row) for row in diagnostic_capability)

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
