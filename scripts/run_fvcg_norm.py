#!/usr/bin/env python3
"""Run the local-only FVCG-Norm Phase-A combined-step falsifier."""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts import run_fvcg_direct as direct  # noqa: E402
from scripts.prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from scripts.run_asgcv_p32 import load_p32_local_authority  # noqa: E402
from sfora.fvcg_direct import FvcgStepAuthority, FvcgStepEvidence  # noqa: E402
from sfora.fvcg_norm import (  # noqa: E402
    FVCG_NORM_RHO,
    FvcgNormAuthority,
    FvcgNormPhaseAResult,
    FvcgNormStepEvidence,
    canonical_fvcg_norm_phase_a_result_bytes,
    combine_norm_stabilized_gradients,
    remeasure_stored_combined_gradients,
    validate_fvcg_norm_phase_a_result_bytes,
)
from sfora.pfml import pfml_potential_loss  # noqa: E402
from sfora.saga_feasibility import (  # noqa: E402
    load_fixture_authority,
    load_snapshot_authority,
)


class CombinedStepAdapter(Protocol):
    pooler: torch.nn.Module

    def vision_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...

    def language_parameters(self) -> tuple[torch.nn.Parameter, ...]: ...

    def vision_pool(self, microbatch: object) -> torch.Tensor: ...

    def direct_collapsed_verdict_backward(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_norm_combined_step(
    adapter: CombinedStepAdapter,
    *,
    authority: FvcgNormAuthority,
    optimizer: object,
    proxies: torch.nn.Parameter,
    proxy_labels: torch.Tensor,
    dml_microbatch: object,
    dml_labels: torch.Tensor,
    semantic_pair: object,
    correct_completion_ids: tuple[int, ...],
    incorrect_completion_ids: tuple[int, ...],
    ordinal: int,
    direct_vjp_errors: tuple[float, float],
    memory_psi_full_avg10_ppm: int,
    direct_vjp_reference: tuple[torch.Tensor | None, ...] | None = None,
) -> FvcgNormStepEvidence:
    """Capture independent fields, normalize semantics, clip once, and step."""

    authority.validated()
    vision = adapter.vision_parameters()
    pooler = tuple(adapter.pooler.parameters())
    language = adapter.language_parameters()
    if not vision or not pooler or proxies.ndim != 2:
        raise ValueError("FVCG-Norm trainable role authority differs")
    if any(parameter.grad is not None for parameter in language):
        raise ValueError("FVCG-Norm language gradient differs")
    if (
        type(ordinal) is not int
        or ordinal < 0
        or type(memory_psi_full_avg10_ppm) is not int
        or memory_psi_full_avg10_ppm < 0
    ):
        raise ValueError("FVCG-Norm step authority differs")

    initial_vision = direct.parameter_state_sha256(vision)
    initial_pooler = direct.parameter_state_sha256(pooler)
    initial_proxy = direct.parameter_state_sha256((proxies,))
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _synchronize()
    started = perf_counter_ns()
    embeddings = adapter.vision_pool(dml_microbatch)
    dml_loss = pfml_potential_loss(
        embeddings,
        dml_labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.2,
        alpha=direct.FVCG_PFML_ALPHA,
        torch_module=torch,
    )
    dml_loss.backward()
    dml_gradients = tuple(
        torch.zeros_like(parameter, dtype=torch.float32)
        if parameter.grad is None
        else parameter.grad.detach().float().clone()
        for parameter in vision
    )
    for parameter in vision:
        parameter.grad = None
    protected_gradients = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in (*pooler, proxies)
    )

    _synchronize()
    semantic_started = perf_counter_ns()
    semantic = adapter.direct_collapsed_verdict_backward(
        semantic_pair,
        correct_completion_ids=correct_completion_ids,
        incorrect_completion_ids=incorrect_completion_ids,
    )
    _synchronize()
    semantic_elapsed_ns = max(1, perf_counter_ns() - semantic_started)
    if any(
        (before is None) != (parameter.grad is None)
        or (
            before is not None
            and parameter.grad is not None
            and not torch.equal(before, parameter.grad)
        )
        for before, parameter in zip(protected_gradients, (*pooler, proxies), strict=True)
    ):
        raise ValueError("FVCG-Norm non-vision gradient differs")
    if any(parameter.grad is None for parameter in vision):
        raise ValueError("FVCG-Norm semantic gradient is absent")
    semantic_gradients = tuple(parameter.grad.detach().float().clone() for parameter in vision)
    field = combine_norm_stabilized_gradients(dml_gradients, semantic_gradients, rho=authority.rho)
    for parameter, gradient in zip(vision, field.gradients, strict=True):
        parameter.grad.copy_(gradient.to(dtype=parameter.dtype))
    field = remeasure_stored_combined_gradients(
        field,
        dml_gradients,
        tuple(parameter.grad.detach().float().clone() for parameter in vision),
    )

    all_trainable = (*vision, *pooler, proxies)
    preclip_norm = float(
        torch.nn.utils.clip_grad_norm_(all_trainable, authority.base.gradient_clip_norm)
    )
    optimizer.step()
    _synchronize()
    combined_elapsed_ns = max(1, perf_counter_ns() - started)
    if direct_vjp_reference is not None:
        direct_vjp_errors = direct._direct_vjp_errors(
            semantic_gradients,
            direct_vjp_reference,
            absolute_floor=authority.base.direct_vjp_atol,
        )

    vision_count, vision_finite = direct._nonzero_finite_count(vision)
    pooler_count, pooler_finite = direct._nonzero_finite_count(pooler)
    proxy_count, proxy_finite = direct._nonzero_finite_count((proxies,))
    language_count, language_finite = direct._nonzero_finite_count(language)
    if language_count:
        raise ValueError("FVCG-Norm language gradient differs")
    gradient_sha256 = hashlib.sha256(
        direct._gradient_vector(all_trainable).detach().cpu().numpy().astype("<f4").tobytes()
    ).hexdigest()
    vision_state = direct.parameter_state_sha256(vision)
    pooler_state = direct.parameter_state_sha256(pooler)
    proxy_state = direct.parameter_state_sha256((proxies,))
    updated_state = hashlib.sha256(
        bytes.fromhex(vision_state) + bytes.fromhex(pooler_state) + bytes.fromhex(proxy_state)
    ).hexdigest()
    observed_psi = max(memory_psi_full_avg10_ppm, direct._memory_psi_full_avg10_ppm())
    base = FvcgStepEvidence(
        ordinal=ordinal,
        selected_pair=direct.select_stratum_pair(
            tuple(range(8)),
            seed_sha256=authority.base.selection_seed_sha256,
            step=ordinal,
        ),
        correct_score=float(semantic.branch_scores[0]),
        incorrect_score=float(semantic.branch_scores[1]),
        correct_probability_ppm=round(float(semantic.correct_probability) * 1_000_000),
        coefficient_ppm=round(float(semantic.coefficient) * 1_000_000),
        loss=float(semantic.loss),
        generated_tokens=int(semantic.generated_tokens),
        vision_nonzero_gradient_parameters=vision_count,
        pooler_nonzero_gradient_parameters=pooler_count,
        proxy_nonzero_gradient_parameters=proxy_count,
        language_gradient_parameters=language_count,
        gradients_finite=(vision_finite and pooler_finite and proxy_finite and language_finite),
        dml_gradient_norm=field.dml_norm,
        semantic_gradient_norm=field.semantic_norm,
        combined_gradient_cosine_distance_ppm=field.combined_cosine_distance_ppm,
        clip_activated=preclip_norm > authority.base.gradient_clip_norm,
        vision_state_changed=vision_state != initial_vision,
        pooler_state_changed=pooler_state != initial_pooler,
        proxy_state_changed=proxy_state != initial_proxy,
        combined_elapsed_ns=combined_elapsed_ns,
        semantic_elapsed_ns=semantic_elapsed_ns,
        peak_cuda_reserved_bytes=direct._peak_cuda_reserved_bytes(),
        peak_rss_bytes=direct._peak_rss_bytes(),
        memory_psi_full_avg10_ppm=observed_psi,
        direct_vjp_max_abs_error=float(direct_vjp_errors[0]),
        direct_vjp_max_rel_error=float(direct_vjp_errors[1]),
        gradient_sha256=gradient_sha256,
        updated_state_sha256=updated_state,
        optimizer_state_sha256=direct._optimizer_state_sha256(optimizer),
        language_state_sha256=direct.parameter_state_sha256(language),
    ).validated(authority.base)
    return FvcgNormStepEvidence(
        base=base,
        safe_semantic_norm=field.safe_semantic_norm,
        raw_dot=field.raw_dot,
        projected_dot=field.projected_dot,
        applied_semantic_norm=field.applied_semantic_norm,
        applied_to_dml_ratio_ppm=field.applied_to_dml_ratio_ppm,
    ).validated(authority)


def run_phase_a(
    *,
    authority: FvcgNormAuthority,
    context_factory: Callable[[int], direct.PhaseAContext],
    output_directory: Path,
) -> bytes:
    """Run one warm-up, three restored measured steps, and a restored repeat."""

    result_path = output_directory / "result.json"
    if result_path.exists():
        reopened = validate_fvcg_norm_phase_a_result_bytes(result_path.read_bytes())
        if reopened.authority != authority:
            raise ValueError("FVCG-Norm resumed authority differs")
        return canonical_fvcg_norm_phase_a_result_bytes(reopened)

    def one(ordinal: int) -> tuple[str, FvcgNormStepEvidence]:
        context = context_factory(ordinal)
        try:
            initial_language = direct.parameter_state_sha256(context.adapter.language_parameters())
            evidence = run_norm_combined_step(
                context.adapter,
                authority=authority,
                optimizer=context.optimizer,
                proxies=context.proxies,
                proxy_labels=context.proxy_labels,
                dml_microbatch=context.dml_microbatch,
                dml_labels=context.dml_labels,
                semantic_pair=context.semantic_pair,
                correct_completion_ids=context.correct_completion_ids,
                incorrect_completion_ids=context.incorrect_completion_ids,
                ordinal=ordinal,
                direct_vjp_errors=context.direct_vjp_errors,
                memory_psi_full_avg10_ppm=context.memory_psi_full_avg10_ppm,
                direct_vjp_reference=context.direct_vjp_reference,
            )
            return initial_language, evidence
        finally:
            del context
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    one(0)
    measured = tuple(one(ordinal) for ordinal in range(3))
    repeated_initial_language, repeated = one(0)
    initial_language = measured[0][0]
    if any(value != initial_language for value, _step in measured) or (
        repeated_initial_language != initial_language
    ):
        raise ValueError("FVCG-Norm restored language state differs")
    steps = tuple(step for _language, step in measured)
    result = FvcgNormPhaseAResult.from_steps(
        authority=authority,
        steps=steps,
        repeated_step_zero=repeated,
        initial_language_state_sha256=initial_language,
    )
    raw = canonical_fvcg_norm_phase_a_result_bytes(result)
    direct._write_new_atomic(result_path, raw)
    if validate_fvcg_norm_phase_a_result_bytes(result_path.read_bytes()) != result:
        raise ValueError("FVCG-Norm persisted result differs")
    return raw


def _reject_duplicate_options(argv: list[str]) -> None:
    options = [token.split("=", 1)[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate FVCG-Norm option")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only FVCG-Norm boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    _reject_duplicate_options(values)
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fixture-source-commit", required=True)
    parser.add_argument("--selection-seed-sha256", required=True)
    parser.add_argument("--execute-phase-a", required=True, action="store_true")
    parsed = parser.parse_args(values)
    for name, length in (
        ("source_commit", 40),
        ("fixture_source_commit", 40),
        ("selection_seed_sha256", 64),
    ):
        value = getattr(parsed, name)
        if (
            type(value) is not str
            or len(value) != length
            or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"{name.replace('_', ' ')} differs")
    for name in ("snapshot_manifest", "fixture", "p32_authority", "train_manifest"):
        path = getattr(parsed, name)
        if path.is_symlink() or not path.is_file():
            parser.error(f"{name.replace('_', ' ')} differs")
    if parsed.model_root.is_symlink() or not parsed.model_root.is_dir():
        parser.error("model root differs")
    if parsed.output_directory.is_symlink() or not parsed.output_directory.is_dir():
        parser.error("output directory differs")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs and run one FVCG-Norm campaign."""

    args = parse_args(argv)
    if _authenticated_source_commit(_REPOSITORY_ROOT) != args.source_commit:
        raise ValueError("FVCG-Norm executing source commit differs")
    direct._configure_determinism()
    local = load_p32_local_authority(
        args.p32_authority,
        args.train_manifest,
        source_commit=args.fixture_source_commit,
    )
    snapshot = load_snapshot_authority(root=args.model_root, manifest_path=args.snapshot_manifest)
    fixture = load_fixture_authority(args.fixture)
    if (
        fixture.source_commit != args.fixture_source_commit
        or getattr(snapshot, "model_revision", None) != local.rollout_authority.model_revision
    ):
        raise ValueError("FVCG-Norm local authority binding differs")
    base = FvcgStepAuthority(
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        model_revision=local.rollout_authority.model_revision,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        selection_seed_sha256=args.selection_seed_sha256,
        semantic_weight=FVCG_NORM_RHO,
        gradient_clip_norm=10.0,
        direct_vjp_atol=0.05,
        direct_vjp_rtol=0.01,
    ).validated()
    authority = FvcgNormAuthority(
        base=base,
        rho=FVCG_NORM_RHO,
        fixture_source_commit=args.fixture_source_commit,
    ).validated()
    raw = run_phase_a(
        authority=authority,
        context_factory=direct._real_context_factory(
            snapshot=snapshot, fixture=fixture, local=local, authority=base
        ),
        output_directory=args.output_directory,
    )
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
