from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
import torch

from sfora.fvcg_direct import FvcgStepAuthority, FvcgStepEvidence, select_stratum_pair
from sfora.fvcg_norm import (
    FvcgNormAuthority,
    FvcgNormPhaseAResult,
    FvcgNormStepEvidence,
    canonical_fvcg_norm_phase_a_result_bytes,
    combine_norm_stabilized_gradients,
    remeasure_stored_combined_gradients,
    validate_fvcg_norm_phase_a_result_bytes,
)


def _flat(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat(tuple(value.flatten() for value in values))


def test_norm_stabilized_field_is_invariant_to_positive_semantic_scale() -> None:
    dml = (torch.tensor([3.0, 4.0]),)
    semantic = (torch.tensor([0.0, 2.0]),)

    small = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    large = combine_norm_stabilized_gradients(
        dml, tuple(value * 1_000_000.0 for value in semantic), rho=0.25
    )

    assert torch.equal(_flat(small.gradients), _flat(large.gradients))
    assert small.dml_norm == pytest.approx(5.0)
    assert small.applied_semantic_norm == pytest.approx(1.25)
    assert small.applied_to_dml_ratio_ppm == 250_000


def test_norm_stabilized_field_removes_only_conflicting_component() -> None:
    dml = (torch.tensor([2.0, 0.0]),)
    semantic = (torch.tensor([-3.0, 4.0]),)

    result = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)

    assert result.raw_dot == pytest.approx(-6.0)
    assert result.projected_dot == pytest.approx(0.0, abs=1.0e-7)
    assert result.safe_semantic_norm == pytest.approx(4.0)
    assert torch.allclose(result.gradients[0], torch.tensor([2.0, 0.5]))
    assert 5_000 <= result.combined_cosine_distance_ppm <= 50_000


def test_norm_stabilized_field_retains_nonconflicting_parallel_component() -> None:
    result = combine_norm_stabilized_gradients(
        (torch.tensor([2.0, 0.0]),),
        (torch.tensor([3.0, 4.0]),),
        rho=0.25,
    )

    assert result.raw_dot == pytest.approx(6.0)
    assert result.projected_dot == pytest.approx(6.0)
    assert torch.allclose(result.gradients[0], torch.tensor([2.3, 0.4]))


@pytest.mark.parametrize(
    ("dml", "semantic", "rho"),
    [
        ((torch.zeros(2),), (torch.ones(2),), 0.25),
        ((torch.ones(2),), (torch.zeros(2),), 0.25),
        ((torch.ones(2),), (torch.tensor([math.nan, 1.0]),), 0.25),
        ((torch.ones(2),), (torch.ones(3),), 0.25),
        ((torch.ones(2),), (torch.ones(2),), 0.0),
    ],
)
def test_norm_stabilized_field_fails_closed(
    dml: tuple[torch.Tensor, ...],
    semantic: tuple[torch.Tensor, ...],
    rho: float,
) -> None:
    with pytest.raises(ValueError, match="FVCG-Norm"):
        combine_norm_stabilized_gradients(dml, semantic, rho=rho)


def test_norm_stabilized_field_is_deterministic_and_fp32() -> None:
    dml = (torch.tensor([1.0, 2.0], dtype=torch.bfloat16),)
    semantic = (torch.tensor([2.0, -0.5], dtype=torch.bfloat16),)

    first = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    second = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)

    assert first == second
    assert first.gradients[0].dtype == torch.float32


def test_norm_stabilized_field_accepts_fp32_projection_roundoff() -> None:
    generator = torch.Generator().manual_seed(1)
    dml = (torch.randn(128, generator=generator, dtype=torch.float32) * 1.0e5,)
    semantic = (torch.randn(128, generator=generator, dtype=torch.float32) * 1.0e-2,)
    if float(torch.dot(dml[0], semantic[0])) >= 0.0:
        semantic = (-semantic[0],)

    result = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)

    normalized_residual = result.projected_dot / (
        result.dml_norm * result.safe_semantic_norm
    )
    assert -1.0e-7 <= normalized_residual < -1.0e-10


def test_norm_stabilized_field_scales_roundoff_to_unprojected_semantic_norm() -> None:
    generator = torch.Generator().manual_seed(0)
    dml_value = torch.randn(128, generator=generator, dtype=torch.float32) * 1.0e5
    orthogonal = torch.randn(128, generator=generator, dtype=torch.float32)
    scale = float(torch.sum(dml_value.double() * orthogonal.double())) / float(
        torch.sum(dml_value.double().square())
    )
    orthogonal = orthogonal - scale * dml_value
    semantic_value = -1.0e-7 * dml_value + 1.0e-4 * orthogonal

    result = combine_norm_stabilized_gradients((dml_value,), (semantic_value,), rho=0.25)

    safe_normalized_residual = result.projected_dot / (
        result.dml_norm * result.safe_semantic_norm
    )
    semantic_normalized_residual = result.projected_dot / (
        result.dml_norm * result.semantic_norm
    )
    assert safe_normalized_residual < -(2.0**-23)
    assert semantic_normalized_residual >= -8.0 * 2.0**-23


def test_norm_stabilized_field_reports_the_applied_fp32_increment() -> None:
    dml = (torch.tensor([1.0e8, 3.0, -7.0], dtype=torch.float32),)
    semantic = (torch.tensor([0.3, -0.7, 0.2], dtype=torch.float32),)
    result = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    applied = tuple(
        combined - source for combined, source in zip(result.gradients, dml, strict=True)
    )
    actual_norm = math.sqrt(math.fsum(float(torch.sum(value.double() ** 2)) for value in applied))
    assert result.applied_semantic_norm == actual_norm
    assert result.applied_to_dml_ratio_ppm == round(actual_norm / result.dml_norm * 1_000_000)


def test_norm_stabilized_field_remeasures_the_optimizer_dtype_store() -> None:
    dml = (torch.tensor([1.0e8, 3.0, -7.0], dtype=torch.float32),)
    semantic = (torch.tensor([0.3, -0.7, 0.2], dtype=torch.float32),)
    calculated = combine_norm_stabilized_gradients(dml, semantic, rho=0.25)
    stored = tuple(value.to(torch.bfloat16).float() for value in calculated.gradients)

    measured = remeasure_stored_combined_gradients(calculated, dml, stored)

    actual_increment = stored[0] - dml[0]
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(measured.gradients, stored, strict=True)
    )
    assert measured.applied_semantic_norm == pytest.approx(
        float(torch.linalg.vector_norm(actual_increment.double()))
    )
    assert measured.applied_to_dml_ratio_ppm != calculated.applied_to_dml_ratio_ppm


def _authority() -> FvcgNormAuthority:
    return FvcgNormAuthority(
        base=FvcgStepAuthority(
            source_commit="1" * 40,
            launch_authority_sha256="2" * 64,
            model_revision="3" * 40,
            fixture_sha256="4" * 64,
            selection_seed_sha256="5" * 64,
            semantic_weight=0.25,
            gradient_clip_norm=10.0,
            direct_vjp_atol=0.05,
            direct_vjp_rtol=0.01,
        ),
        rho=0.25,
        fixture_source_commit="6" * 40,
    ).validated()


def _step(ordinal: int) -> FvcgNormStepEvidence:
    authority = _authority()
    base = FvcgStepEvidence(
        ordinal=ordinal,
        selected_pair=select_stratum_pair(
            tuple(range(8)), seed_sha256=authority.base.selection_seed_sha256, step=ordinal
        ),
        correct_score=-0.1,
        incorrect_score=-1.1,
        correct_probability_ppm=731_059,
        coefficient_ppm=393_256,
        loss=-0.39325649134544005,
        generated_tokens=0,
        vision_nonzero_gradient_parameters=2,
        pooler_nonzero_gradient_parameters=1,
        proxy_nonzero_gradient_parameters=1,
        language_gradient_parameters=0,
        gradients_finite=True,
        dml_gradient_norm=4.0,
        semantic_gradient_norm=2.0,
        combined_gradient_cosine_distance_ppm=29_857,
        clip_activated=False,
        vision_state_changed=True,
        pooler_state_changed=True,
        proxy_state_changed=True,
        combined_elapsed_ns=4_000_000_000,
        semantic_elapsed_ns=800_000_000,
        peak_cuda_reserved_bytes=60 * 1024**3,
        peak_rss_bytes=20 * 1024**3,
        memory_psi_full_avg10_ppm=0,
        direct_vjp_max_abs_error=0.01,
        direct_vjp_max_rel_error=0.005,
        gradient_sha256=f"{ordinal + 6:064x}",
        updated_state_sha256=f"{ordinal + 10:064x}",
        optimizer_state_sha256=f"{ordinal + 14:064x}",
        language_state_sha256="f" * 64,
    )
    return FvcgNormStepEvidence(
        base=base,
        safe_semantic_norm=2.0,
        raw_dot=0.0,
        projected_dot=0.0,
        applied_semantic_norm=1.0,
        applied_to_dml_ratio_ppm=250_000,
    ).validated(authority)


def test_norm_phase_a_canonical_result_recomputes_norm_gates_and_digest() -> None:
    authority = _authority()
    steps = tuple(_step(index) for index in range(3))
    result = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=replace(steps[0]),
        initial_language_state_sha256="f" * 64,
    )

    raw = canonical_fvcg_norm_phase_a_result_bytes(result)
    reopened = validate_fvcg_norm_phase_a_result_bytes(raw)

    assert reopened == result
    assert reopened.passed is True
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")


def test_norm_phase_a_rejects_derived_scalar_and_pass_mutations() -> None:
    authority = _authority()
    steps = tuple(_step(index) for index in range(3))
    result = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=steps[0],
        initial_language_state_sha256="f" * 64,
    )
    raw = canonical_fvcg_norm_phase_a_result_bytes(result)

    for path in ("ratio", "projected", "safe", "cauchy", "direction", "passed"):
        value = json.loads(raw)
        if path == "ratio":
            value["steps"][0]["applied_to_dml_ratio_ppm"] += 1
        elif path == "projected":
            value["steps"][0]["projected_dot"] = 7.0
        elif path == "safe":
            value["steps"][0]["safe_semantic_norm"] = 1.0
        elif path == "cauchy":
            value["steps"][0]["raw_dot"] = 9.0
        elif path == "direction":
            value["steps"][0]["base"]["combined_gradient_cosine_distance_ppm"] = 1
        else:
            value["passed"] = False
        mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(ValueError, match="FVCG-Norm"):
            validate_fvcg_norm_phase_a_result_bytes(mutated)


def test_norm_step_rejects_impossible_projection_evidence_directly() -> None:
    authority = _authority()
    step = _step(0)
    for mutation in (
        replace(step, projected_dot=7.0),
        replace(step, projected_dot=-7.9e-6),
        replace(step, safe_semantic_norm=1.0),
        replace(step, raw_dot=9.0, projected_dot=9.0),
    ):
        with pytest.raises(ValueError, match="step arithmetic"):
            mutation.validated(authority)


def test_norm_phase_a_fails_out_of_band_direction_without_rewriting_evidence() -> None:
    authority = _authority()
    steps = tuple(_step(index) for index in range(3))
    failed = replace(
        steps[0],
        base=replace(steps[0].base, combined_gradient_cosine_distance_ppm=4_999),
    )
    result = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=(failed, *steps[1:]),
        repeated_step_zero=failed,
        initial_language_state_sha256="f" * 64,
    )
    assert result.passed is False


def test_norm_phase_a_replay_covers_norms_and_direction() -> None:
    authority = _authority()
    steps = tuple(_step(index) for index in range(3))
    for replay in (
        replace(
            steps[0], base=replace(steps[0].base, dml_gradient_norm=5.0), applied_semantic_norm=1.25
        ),
        replace(
            steps[0],
            base=replace(steps[0].base, semantic_gradient_norm=3.0),
            safe_semantic_norm=3.0,
        ),
        replace(
            steps[0],
            base=replace(steps[0].base, combined_gradient_cosine_distance_ppm=6_000),
        ),
    ):
        result = FvcgNormPhaseAResult.from_steps(
            authority=authority,
            steps=steps,
            repeated_step_zero=replay,
            initial_language_state_sha256="f" * 64,
        )
        assert result.passed is False


def test_norm_result_publishes_recomputed_base_resource_aggregates() -> None:
    authority = _authority()
    steps = tuple(_step(index) for index in range(3))
    result = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=steps[0],
        initial_language_state_sha256="f" * 64,
    )
    value = json.loads(canonical_fvcg_norm_phase_a_result_bytes(result))
    assert value["combined_p90_ns"] == 4_000_000_000
    assert value["semantic_p90_ns"] == 800_000_000
    assert value["peak_cuda_reserved_bytes"] == 60 * 1024**3
    assert value["peak_rss_bytes"] == 20 * 1024**3
    assert value["peak_memory_psi_full_avg10_ppm"] == 0
    assert value["deterministic_step_zero"] is True
    assert value["gates"]["combined_p90_ns"] == 15_000_000_000
