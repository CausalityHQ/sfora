from __future__ import annotations

from dataclasses import replace

import pytest

from sfora.prism_measurement import (
    PRISM_CHANNELS,
    PrismExample,
    PrismObservationCapabilityRow,
    PrismTokenProtocol,
    build_prism_schedules,
    release_prism_observation_capability,
)
from sfora.prism_observer import (
    PrismCompletionBundle,
    PrismCompletionRow,
    bind_prism_diagnostic_capability,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _schedules() -> tuple:
    optimization = tuple(
        PrismExample(
            example_id=f"private/optimization-{label}-{ordinal}.png",
            label=label,
            image_sha256=_digest(1 + label * 8 + ordinal),
        )
        for label in range(49)
        for ordinal in range(8)
    )
    diagnostic = tuple(
        PrismExample(
            example_id=f"private/diagnostic-{label}-{ordinal}.png",
            label=label,
            image_sha256=_digest(10_000 + (label - 82) * 32 + ordinal),
        )
        for label in (82, 83)
        for ordinal in range(32)
    )
    return build_prism_schedules(
        optimization,
        diagnostic,
        source_identity="prism-binder-fixture-v1",
    )


def _protocol() -> PrismTokenProtocol:
    return PrismTokenProtocol(
        channel_prefixes=tuple((100 + index,) for index in range(8)),
        visibility_prefixes=((200,), (201,), (202,), (203,)),
        relation_prefixes=((300,), (301,), (302,)),
        confidence_prefixes=((400,), (401,), (402,)),
        evidence_separator=(500,),
        terminal_tokens=(600,),
        max_evidence_tokens=4,
    )


def _completion_ids(channel: str, relation: str) -> tuple[int, ...]:
    return (
        100 + PRISM_CHANNELS.index(channel),
        200,
        300 if relation == "same" else 301,
        402,
        500,
        600,
    )


def _calibration_bundle(
    capability: tuple[PrismObservationCapabilityRow, ...], scoring: tuple
) -> PrismCompletionBundle:
    rows = []
    for row in capability:
        private = next(
            scoring_row
            for scoring_row in scoring
            if scoring_row.left_payload_sha256 == row.left_payload_sha256
            and scoring_row.right_payload_sha256 == row.right_payload_sha256
        )
        rows.append(
            PrismCompletionRow(
                pair_handle=row.pair_handle,
                channel=row.channel,
                completion_ids=_completion_ids(row.channel, private.relation),
            )
        )
    return PrismCompletionBundle(
        schema="sfora-prism-completion-bundle-v1",
        phase="calibration",
        observer_authority_sha256="a" * 64,
        token_protocol_sha256="b" * 64,
        rows=tuple(rows),
    )


def test_calibration_bundle_releases_only_anonymous_diagnostic_capability() -> None:
    observations, scoring = _schedules()
    calibration_capability = release_prism_observation_capability(
        observations,
        scoring,
        source_identity="prism-binder-fixture-v1",
        phase="calibration",
    )

    release = bind_prism_diagnostic_capability(
        observations,
        scoring,
        calibration_capability,
        _calibration_bundle(calibration_capability, scoring),
        _protocol(),
        source_identity="prism-binder-fixture-v1",
    )

    assert len(release.calibrations) == 8
    assert len(release.calibration_receipt_sha256) == 64
    assert len(release.capability) == 256
    assert all(type(row) is PrismObservationCapabilityRow for row in release.capability)
    assert tuple(row.channel for row in release.capability[:8]) == PRISM_CHANNELS


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reordered", "handle", "validity"))
def test_calibration_binder_rejects_incomplete_or_drifted_completion_rows(
    mutation: str,
) -> None:
    observations, scoring = _schedules()
    capability = release_prism_observation_capability(
        observations,
        scoring,
        source_identity="prism-binder-fixture-v1",
        phase="calibration",
    )
    bundle = _calibration_bundle(capability, scoring)
    rows = list(bundle.rows)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    elif mutation == "reordered":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "handle":
        rows[0] = replace(rows[0], pair_handle="f" * 64)
    else:
        for index in range(65):
            rows[index] = replace(rows[index], completion_ids=(999,))

    with pytest.raises(ValueError):
        bind_prism_diagnostic_capability(
            observations,
            scoring,
            capability,
            replace(bundle, rows=tuple(rows)),
            _protocol(),
            source_identity="prism-binder-fixture-v1",
        )
