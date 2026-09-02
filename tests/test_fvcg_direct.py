from __future__ import annotations

import json
from dataclasses import replace

import pytest

from sfora.fvcg_direct import (
    FvcgPhaseAResult,
    FvcgStepAuthority,
    FvcgStepEvidence,
    canonical_fvcg_phase_a_result_bytes,
    select_stratum_pair,
    validate_fvcg_phase_a_result_bytes,
)


def _authority() -> FvcgStepAuthority:
    return FvcgStepAuthority(
        source_commit="1" * 40,
        launch_authority_sha256="2" * 64,
        model_revision="3" * 40,
        fixture_sha256="4" * 64,
        selection_seed_sha256="5" * 64,
        semantic_weight=0.25,
        gradient_clip_norm=1.0,
        direct_vjp_atol=1.0e-5,
        direct_vjp_rtol=1.0e-4,
    ).validated()


def _step(ordinal: int, *, selected_pair: int | None = None) -> FvcgStepEvidence:
    authority = _authority()
    pair = select_stratum_pair(
        tuple(range(8)), seed_sha256=authority.selection_seed_sha256, step=ordinal
    )
    return FvcgStepEvidence(
        ordinal=ordinal,
        selected_pair=pair if selected_pair is None else selected_pair,
        correct_score=-0.1,
        incorrect_score=-1.1,
        correct_probability_ppm=731_059,
        coefficient_ppm=393_256,
        loss=-0.39325649134544005,
        generated_tokens=0,
        vision_nonzero_gradient_parameters=4,
        pooler_nonzero_gradient_parameters=2,
        proxy_nonzero_gradient_parameters=1,
        language_gradient_parameters=0,
        gradients_finite=True,
        dml_gradient_norm=2.0,
        semantic_gradient_norm=0.5,
        combined_gradient_cosine_distance_ppm=20_000,
        clip_activated=False,
        vision_state_changed=True,
        pooler_state_changed=True,
        proxy_state_changed=True,
        combined_elapsed_ns=10_000_000_000 + ordinal,
        semantic_elapsed_ns=1_000_000_000 + ordinal,
        peak_cuda_reserved_bytes=80 * 1024**3,
        peak_rss_bytes=80 * 1024**3,
        memory_psi_full_avg10_ppm=0,
        direct_vjp_max_abs_error=1.0e-6,
        direct_vjp_max_rel_error=1.0e-5,
        gradient_sha256=f"{ordinal + 6:064x}",
        updated_state_sha256=f"{ordinal + 10:064x}",
        optimizer_state_sha256=f"{ordinal + 14:064x}",
        language_state_sha256="f" * 64,
    ).validated(authority)


def _passing_result() -> FvcgPhaseAResult:
    authority = _authority()
    steps = (_step(0), _step(1), _step(2))
    return FvcgPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=replace(steps[0]),
        initial_language_state_sha256="f" * 64,
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_select_stratum_pair_is_seeded_repeatable_and_covers_only_registered_pairs() -> None:
    selections = tuple(
        select_stratum_pair(tuple(range(8)), seed_sha256="1" * 64, step=step)
        for step in range(64)
    )
    assert selections == tuple(
        select_stratum_pair(tuple(range(8)), seed_sha256="1" * 64, step=step)
        for step in range(64)
    )
    assert set(selections) == set(range(8))
    assert selections != tuple(
        select_stratum_pair(tuple(range(8)), seed_sha256="2" * 64, step=step)
        for step in range(64)
    )
    with pytest.raises(ValueError, match="stratum"):
        select_stratum_pair(tuple(range(7)), seed_sha256="1" * 64, step=0)


def test_step_recomputes_forced_scalar_and_rejects_pair_or_type_drift() -> None:
    authority = _authority()
    step = _step(0)
    assert step.correct_probability_ppm == 731_059
    assert step.coefficient_ppm == 393_256
    for mutation in (
        replace(step, selected_pair=(step.selected_pair + 1) % 8),
        replace(step, generated_tokens=False),
        replace(step, correct_probability_ppm=step.correct_probability_ppm + 1),
        replace(step, coefficient_ppm=step.coefficient_ppm + 1),
    ):
        with pytest.raises(ValueError):
            mutation.validated(authority)


def test_phase_a_result_recomputes_gates_and_rejects_mutations() -> None:
    result = _passing_result()
    raw = canonical_fvcg_phase_a_result_bytes(result)
    reopened = validate_fvcg_phase_a_result_bytes(raw)
    assert reopened == result
    assert reopened.passed is True
    assert reopened.combined_p90_ns == 10_000_000_002
    assert reopened.semantic_p90_ns == 1_000_000_002
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")

    for mutate in (
        lambda value: value["steps"][0].__setitem__("generated_tokens", False),
        lambda value: value.__setitem__("passed", False),
        lambda value: value.__setitem__("combined_p90_ns", 1),
        lambda value: value.__setitem__("result_sha256", "0" * 64),
        lambda value: value["authority"].__setitem__("semantic_weight", 1),
    ):
        value = json.loads(raw)
        mutate(value)
        with pytest.raises(ValueError):
            validate_fvcg_phase_a_result_bytes(_canonical(value))


def test_phase_a_fails_resource_liveness_determinism_and_timing_gates() -> None:
    authority = _authority()
    passing = (_step(0), _step(1), _step(2))
    mutations = (
        replace(passing[0], peak_cuda_reserved_bytes=96 * 1024**3 + 1),
        replace(passing[0], combined_elapsed_ns=15_000_000_001),
        replace(passing[0], semantic_elapsed_ns=2_000_000_001),
        replace(passing[0], vision_nonzero_gradient_parameters=0),
        replace(passing[0], language_gradient_parameters=1),
        replace(passing[0], generated_tokens=1),
        replace(passing[0], gradients_finite=False),
        replace(passing[0], direct_vjp_max_abs_error=1.0),
        replace(passing[0], vision_state_changed=False),
    )
    for first in mutations:
        result = FvcgPhaseAResult.from_steps(
            authority=authority,
            steps=(first, passing[1], passing[2]),
            repeated_step_zero=replace(first),
            initial_language_state_sha256="f" * 64,
        )
        assert result.passed is False

    nondeterministic = replace(passing[0], gradient_sha256="0" * 64)
    result = FvcgPhaseAResult.from_steps(
        authority=authority,
        steps=passing,
        repeated_step_zero=nondeterministic,
        initial_language_state_sha256="f" * 64,
    )
    assert result.passed is False
