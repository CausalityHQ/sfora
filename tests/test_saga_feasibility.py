from __future__ import annotations

from dataclasses import replace

import pytest

from sfora.saga_feasibility import (
    FeasibilityEvidence,
    FeasibilityOutcome,
    ObjectAuthority,
    PhaseMeasurement,
    ResourceEnvelope,
    canonical_feasibility_result_bytes,
    parse_canonical_object,
    project_best_case_step_ns,
)


def _phase(name: str, *, elapsed_ns: int) -> PhaseMeasurement:
    return PhaseMeasurement(
        name=name,
        completed=True,
        elapsed_ns=elapsed_ns,
        peak_cuda_reserved_bytes=1_000,
        peak_rss_bytes=2_000,
    )


def coherent_feasibility_evidence() -> FeasibilityEvidence:
    return FeasibilityEvidence(
        source_commit="a" * 40,
        controller_commit="b" * 40,
        binary_sha256="c" * 64,
        environment_sha256="d" * 64,
        host="spark-fixture",
        model=ObjectAuthority(
            role="model-snapshot-manifest",
            relative_path="model/manifest.json",
            byte_length=101,
            sha256="e" * 64,
        ),
        fixture=ObjectAuthority(
            role="synthetic-fixture",
            relative_path="fixture.json",
            byte_length=202,
            sha256="f" * 64,
        ),
        envelope=ResourceEnvelope(
            cuda_reserved_limit_bytes=103_079_215_104,
            rss_limit_bytes=118_111_600_640,
            wall_limit_ns=7_200_000_000_000,
            progress_limit_ns=300_000_000_000,
        ),
        load=_phase("load", elapsed_ns=5),
        rollout=_phase("rollout", elapsed_ns=20),
        replay=_phase("replay", elapsed_ns=30),
        attention=_phase("attention", elapsed_ns=40),
        dml=_phase("dml", elapsed_ns=10),
        deterministic=True,
        attention_available=True,
        backend_valid=True,
        authority_valid=True,
        memory_within_envelope=True,
        time_within_envelope=True,
        dataset_reads=0,
        label_reads=0,
        evaluation_reads=0,
        optimizer_steps=0,
    )


def test_projection_uses_one_dml_microbatch_and_eight_pair_groups() -> None:
    assert (
        project_best_case_step_ns(
            dml_microbatch_ns=10,
            rollout_group_ns=20,
            replay_pair_ns=30,
            attention_pair_ns=40,
        )
        == 730
    )


@pytest.mark.parametrize("bad", [0, -1, True, 1.0])
def test_projection_rejects_nonpositive_or_nonconcrete_timings(bad: object) -> None:
    with pytest.raises(ValueError, match="timing authority"):
        project_best_case_step_ns(
            dml_microbatch_ns=bad,  # type: ignore[arg-type]
            rollout_group_ns=20,
            replay_pair_ns=30,
            attention_pair_ns=40,
        )


def test_result_recomputes_outcome_and_rejects_incomplete_phase() -> None:
    evidence = coherent_feasibility_evidence()
    raw = canonical_feasibility_result_bytes(evidence)
    value = parse_canonical_object(raw, role="SAGA feasibility result")

    assert raw.endswith(b"\n")
    assert value["claim_eligible"] is False
    assert value["quality_metrics"] == []
    assert value["outcome"] == FeasibilityOutcome.FITS.value
    assert value["best_case_step_ns"] == 730
    assert len(value["result_sha256"]) == 64

    with pytest.raises(ValueError, match="phase evidence"):
        canonical_feasibility_result_bytes(
            replace(evidence, replay=replace(evidence.replay, completed=False))
        )


def test_result_recomputes_outcome_precedence() -> None:
    evidence = coherent_feasibility_evidence()
    cases = (
        (replace(evidence, time_within_envelope=False), "TIME_BUDGET_FAIL"),
        (replace(evidence, attention_available=False), "ATTENTION_UNAVAILABLE"),
        (replace(evidence, memory_within_envelope=False), "MEMORY_FAIL"),
        (replace(evidence, deterministic=False), "DETERMINISM_FAIL"),
        (replace(evidence, backend_valid=False), "BACKEND_INVALID"),
        (replace(evidence, authority_valid=False), "AUTHORITY_INVALID"),
    )
    for mutated, expected in cases:
        value = parse_canonical_object(
            canonical_feasibility_result_bytes(mutated), role="result"
        )
        assert value["outcome"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_reads", 1),
        ("label_reads", 1),
        ("evaluation_reads", 1),
        ("optimizer_steps", 1),
        ("dataset_reads", False),
    ],
)
def test_result_rejects_nonzero_or_nonconcrete_capability_counters(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match="capability counters"):
        canonical_feasibility_result_bytes(
            replace(coherent_feasibility_evidence(), **{field: value})
        )


def test_object_authority_rejects_schema_type_and_digest_drift() -> None:
    valid = {
        "role": "model",
        "relative_path": "manifest.json",
        "byte_length": 1,
        "sha256": "0" * 64,
    }
    assert ObjectAuthority.from_mapping(valid).byte_length == 1

    mutations = (
        {**valid, "extra": 1},
        {**valid, "byte_length": True},
        {**valid, "sha256": "0" * 63},
        {**valid, "relative_path": "../manifest.json"},
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="object authority"):
            ObjectAuthority.from_mapping(mutation)


def test_parse_canonical_object_rejects_noncanonical_bytes() -> None:
    assert parse_canonical_object(b'{"a":1}\n', role="fixture") == {"a": 1}
    for raw in (b'{"a": 1}\n', b'{"a":1}', b"[]\n"):
        with pytest.raises(ValueError, match="canonical JSON"):
            parse_canonical_object(raw, role="fixture")
