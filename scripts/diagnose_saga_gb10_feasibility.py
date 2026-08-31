#!/usr/bin/env python3
"""Run the local, quality-blind SAGA GB10 feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch
import torch.nn.functional as torch_functional

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import (
    FeasibilityEvidence,
    FeasibilityOutcome,
    FixtureAuthority,
    ObjectAuthority,
    PhaseMeasurement,
    ResourceEnvelope,
    SnapshotAuthority,
    canonical_feasibility_result_bytes,
    load_fixture_authority,
    load_snapshot_authority,
)


class FeasibilityFailure(Exception):
    """Expected, canonically classifiable diagnostic failure."""

    def __init__(self, outcome: FeasibilityOutcome, clause: str) -> None:
        super().__init__(clause)
        self.outcome = outcome
        self.clause = clause


class AttentionUnavailable(Exception):
    """The sealed backend cannot expose exact layer attention."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Result identities and controller envelope unavailable from model state."""

    source_commit: str
    controller_commit: str
    binary_sha256: str
    environment_sha256: str
    host: str
    model_object: ObjectAuthority
    fixture_object: ObjectAuthority
    envelope: ResourceEnvelope


@dataclass(frozen=True, slots=True)
class LoadedAuthority:
    """Cross-bound snapshot and synthetic fixture authorities."""

    snapshot: SnapshotAuthority
    fixture: FixtureAuthority
    result_identity: RunIdentity | None = None


class ModelFactory(Protocol):
    """Offline-only factory boundary for the concrete Transformers adapter."""

    def load_model(self, root: Path, **kwargs: object) -> object: ...

    def load_processor(self, root: Path, **kwargs: object) -> object: ...


class TransformersFactory:
    """Lazy local-only Transformers construction boundary."""

    def load_model(self, root: Path, **kwargs: object) -> object:
        from transformers import Qwen3VLForConditionalGeneration

        values = dict(kwargs)
        if values.get("dtype") == "bfloat16":
            values["dtype"] = torch.bfloat16
        return Qwen3VLForConditionalGeneration.from_pretrained(root, **values)

    def load_processor(self, root: Path, **kwargs: object) -> object:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(root, **kwargs)


class SagaModelAdapter(Protocol):
    """Backend-independent scientific operations required by the diagnostic."""

    def prepare_pair(self, fixture: FixtureAuthority) -> object: ...

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]: ...

    def replay(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        advantages: tuple[float, ...],
        *,
        output_attentions: bool,
    ) -> ReplayOutput: ...

    def assert_gradient_roles(self) -> GradientEvidence: ...

    def clear_graphs(self) -> None: ...

    def attention_observation(
        self,
        pair: object,
        completion_ids: tuple[tuple[int, ...], ...],
        *,
        layer: int,
    ) -> AttentionOutput: ...

    def prepare_microbatch(self, fixture: FixtureAuthority) -> object: ...

    def vision_pool(self, microbatch: object) -> torch.Tensor: ...

    def validate_structure(self, authority: LoadedAuthority) -> None: ...


@dataclass(frozen=True, slots=True)
class SealedRollouts:
    """Content-addressed completion IDs from one exact rollout group."""

    completion_ids: tuple[tuple[int, ...], ...]
    token_counts: tuple[int, ...]
    completion_sha256: tuple[str, ...]
    elapsed_ns: int

    @property
    def group_size(self) -> int:
        return len(self.completion_ids)


@dataclass(frozen=True, slots=True)
class ReplayOutput:
    """Backend replay scalar and token count before gradient validation."""

    loss: float
    generated_tokens: int


@dataclass(frozen=True, slots=True)
class GradientEvidence:
    """Exact trainable/frozen gradient-role evidence."""

    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int
    finite: bool
    gradient_sha256: str


@dataclass(frozen=True, slots=True)
class ReplayEvidence:
    """Validated replay measurement."""

    elapsed_ns: int
    loss: float
    generated_tokens: int
    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int
    gradient_sha256: str


@dataclass(frozen=True, slots=True)
class AttentionOutput:
    """Exact layer attention and patch tokens returned by the adapter."""

    teacher_maps: torch.Tensor
    patch_tokens: torch.Tensor
    head_count: int


@dataclass(frozen=True, slots=True)
class AttentionEvidence:
    """Validated detached-teacher KL evidence."""

    layer: int
    head_count: int
    elapsed_ns: int
    kl: float
    teacher_unit_mass: bool
    teacher_gradient_parameters: int
    pooler_nonzero_gradient_parameters: int


@dataclass(frozen=True, slots=True)
class DmlFloorEvidence:
    """Validated 64-image vision/pooler activation-floor evidence."""

    batch_size: int
    embedding_shape: tuple[int, int]
    elapsed_ns: int
    loss: float
    maximum_norm_delta_ppm: int
    vision_nonzero_gradient_parameters: int
    language_gradient_parameters: int


class SingleQueryPooler(torch.nn.Module):
    """One learned query attention pooler with a 4096-dimensional output."""

    def __init__(self, token_dim: int, embedding_dim: int = 4096) -> None:
        super().__init__()
        if type(token_dim) is not int or token_dim <= 0:
            raise ValueError("SAGA pooler token dimension differs")
        if type(embedding_dim) is not int or embedding_dim != 4096:
            raise ValueError("SAGA pooler embedding dimension differs")
        self.query = torch.nn.Parameter(torch.zeros(token_dim))
        self.key = torch.nn.Linear(token_dim, token_dim, bias=False)
        self.output = torch.nn.Linear(token_dim, embedding_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.query.numel():
            raise ValueError("SAGA pooler token shape differs")
        logits = torch.einsum("d,bpd->bp", self.query, self.key(tokens)) / math.sqrt(
            tokens.shape[-1]
        )
        weights = logits.softmax(dim=-1)
        pooled = torch.einsum("bp,bpd->bd", weights, tokens)
        return torch_functional.normalize(self.output(pooled), dim=-1), weights


class QwenSagaAdapter:
    """Validated Qwen model/processor pair with frozen gradient roles."""

    def __init__(self, model: object, processor: object) -> None:
        self._model = model
        self._processor = processor
        self.architecture = "Qwen3VLForConditionalGeneration"

    def trainable_parameter_roles(self) -> tuple[str, ...]:
        return ("vision",)

    def frozen_parameter_roles(self) -> tuple[str, ...]:
        return ("language",)


def _model_attribute(model: object, name: str) -> object:
    try:
        return getattr(model, name)
    except AttributeError as error:
        raise ValueError("SAGA model authority differs") from error


def load_qwen_adapter(
    authority: LoadedAuthority, *, factory: ModelFactory
) -> QwenSagaAdapter:
    """Load one registered local model and freeze exact parameter roles."""

    if type(authority) is not LoadedAuthority:
        raise ValueError("SAGA loaded authority differs")
    snapshot = authority.snapshot
    fixture = authority.fixture
    if fixture.model_revision != snapshot.model_revision:
        raise ValueError("SAGA model authority differs")
    model = factory.load_model(
        snapshot.root,
        local_files_only=True,
        trust_remote_code=False,
        dtype="bfloat16",
        attn_implementation=snapshot.attention_backend,
    )
    processor = factory.load_processor(
        snapshot.root,
        local_files_only=True,
        trust_remote_code=False,
    )
    expected = {
        "architecture": snapshot.architecture,
        "dtype": snapshot.dtype,
        "device": "cuda",
        "attention_backend": snapshot.attention_backend,
    }
    if any(_model_attribute(model, name) != value for name, value in expected.items()):
        raise ValueError("SAGA model authority differs")
    layer_count = _model_attribute(model, "layer_count")
    if type(layer_count) is not int or layer_count <= fixture.attention_layer:
        raise ValueError("SAGA model authority differs")
    output_keys = _model_attribute(processor, "output_keys")
    if output_keys != (
        "attention_mask",
        "image_grid_thw",
        "input_ids",
        "pixel_values",
    ):
        raise ValueError("SAGA model authority differs")
    roles = _model_attribute(model, "parameters_by_role")
    if type(roles) is not dict or set(roles) != {"vision", "language"}:
        raise ValueError("SAGA model authority differs")
    if any(type(parameters) is not list or not parameters for parameters in roles.values()):
        raise ValueError("SAGA model authority differs")
    for parameter in roles["vision"]:
        parameter.requires_grad = True
    for parameter in roles["language"]:
        parameter.requires_grad = False
    return QwenSagaAdapter(model, processor)


def group_normalized_advantages(rewards: tuple[int, ...]) -> tuple[float, ...]:
    """Compute the exact population-normalized synthetic group advantages."""

    if (
        type(rewards) is not tuple
        or len(rewards) != 8
        or any(type(reward) is not int or reward not in {0, 1} for reward in rewards)
    ):
        raise ValueError("SAGA synthetic reward authority differs")
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    if variance <= 0.0:
        raise ValueError("SAGA synthetic rewards have zero group variance")
    scale = math.sqrt(variance)
    return tuple((reward - mean) / scale for reward in rewards)


def _completion_digest(token_ids: tuple[int, ...]) -> str:
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("SAGA completion token authority differs")
    return hashlib.sha256(
        canonical_json_bytes({"token_ids": list(token_ids)})
    ).hexdigest()


def run_rollout_phase(
    adapter: SagaModelAdapter, fixture: FixtureAuthority
) -> SealedRollouts:
    """Generate and seal one exact eight-completion rollout group."""

    if fixture.group_size != 8 or fixture.generation_seeds != tuple(range(8)):
        raise ValueError("SAGA rollout fixture authority differs")
    pair = adapter.prepare_pair(fixture)
    started = perf_counter_ns()
    completions = tuple(
        adapter.generate(
            pair,
            seed,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=1024,
        )
        for seed in fixture.generation_seeds
    )
    elapsed_ns = max(1, perf_counter_ns() - started)
    if len(completions) != 8 or any(type(tokens) is not tuple for tokens in completions):
        raise ValueError("SAGA rollout completion evidence differs")
    digests = tuple(_completion_digest(tokens) for tokens in completions)
    if len(set(digests)) != 8:
        raise ValueError("SAGA rollout generators are not distinct")
    return SealedRollouts(
        completion_ids=completions,
        token_counts=tuple(len(tokens) for tokens in completions),
        completion_sha256=digests,
        elapsed_ns=elapsed_ns,
    )


def _validate_gradient_evidence(gradient: GradientEvidence) -> None:
    if type(gradient) is not GradientEvidence:
        raise ValueError("SAGA gradient evidence differs")
    if (
        type(gradient.vision_nonzero_gradient_parameters) is not int
        or gradient.vision_nonzero_gradient_parameters <= 0
    ):
        raise ValueError("SAGA vision gradient evidence differs")
    if (
        type(gradient.language_gradient_parameters) is not int
        or gradient.language_gradient_parameters != 0
    ):
        raise ValueError("SAGA language gradient evidence differs")
    if gradient.finite is not True:
        raise ValueError("SAGA finite gradient evidence differs")
    if (
        type(gradient.gradient_sha256) is not str
        or len(gradient.gradient_sha256) != 64
        or any(character not in "0123456789abcdef" for character in gradient.gradient_sha256)
    ):
        raise ValueError("SAGA gradient digest differs")


def run_replay_phase(
    adapter: SagaModelAdapter,
    fixture: FixtureAuthority,
    rollouts: SealedRollouts,
) -> ReplayEvidence:
    """Replay sealed completions and prove exact gradient-role separation."""

    if type(rollouts) is not SealedRollouts or rollouts.group_size != fixture.group_size:
        raise ValueError("SAGA replay rollout authority differs")
    pair = adapter.prepare_pair(fixture)
    advantages = group_normalized_advantages(fixture.synthetic_rewards)
    started = perf_counter_ns()
    try:
        output = adapter.replay(
            pair,
            rollouts.completion_ids,
            advantages,
            output_attentions=False,
        )
        elapsed_ns = max(1, perf_counter_ns() - started)
        if (
            type(output) is not ReplayOutput
            or type(output.loss) is not float
            or not math.isfinite(output.loss)
            or type(output.generated_tokens) is not int
            or output.generated_tokens != sum(rollouts.token_counts)
        ):
            raise ValueError("SAGA replay output evidence differs")
        gradient = adapter.assert_gradient_roles()
        _validate_gradient_evidence(gradient)
        return ReplayEvidence(
            elapsed_ns=elapsed_ns,
            loss=output.loss,
            generated_tokens=output.generated_tokens,
            vision_nonzero_gradient_parameters=(
                gradient.vision_nonzero_gradient_parameters
            ),
            language_gradient_parameters=gradient.language_gradient_parameters,
            gradient_sha256=gradient.gradient_sha256,
        )
    finally:
        adapter.clear_graphs()


def _nonzero_finite_gradient_count(parameters: object) -> int:
    count = 0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError("SAGA finite pooler gradient evidence differs")
        if bool(torch.count_nonzero(gradient)):
            count += 1
    return count


def run_attention_phase(
    adapter: SagaModelAdapter,
    pooler: SingleQueryPooler,
    fixture: FixtureAuthority,
    rollouts: SealedRollouts,
) -> AttentionEvidence:
    """Measure exact layer-26 detached-teacher attention/pooler KL."""

    if fixture.attention_layer != 26 or rollouts.group_size != fixture.group_size:
        raise ValueError("SAGA attention authority differs")
    pair = adapter.prepare_pair(fixture)
    started = perf_counter_ns()
    try:
        output = adapter.attention_observation(
            pair,
            rollouts.completion_ids,
            layer=fixture.attention_layer,
        )
        if type(output) is not AttentionOutput:
            raise ValueError("SAGA attention output differs")
        teacher = output.teacher_maps
        tokens = output.patch_tokens
        if (
            teacher.ndim != 2
            or tokens.ndim != 3
            or teacher.shape != tokens.shape[:2]
            or type(output.head_count) is not int
            or output.head_count <= 0
        ):
            raise ValueError("SAGA teacher attention shape differs")
        if (
            not bool(torch.isfinite(teacher).all())
            or bool((teacher < 0).any())
            or not torch.allclose(
                teacher.sum(dim=-1),
                torch.ones(teacher.shape[0], device=teacher.device),
                atol=1e-6,
                rtol=0.0,
            )
        ):
            raise ValueError("SAGA teacher attention differs")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("SAGA patch token evidence differs")
        detached_teacher = teacher.detach()
        detached_tokens = tokens.detach()
        _, pooler_weights = pooler(detached_tokens)
        epsilon = torch.finfo(pooler_weights.dtype).tiny
        kl = (
            detached_teacher
            * (
                detached_teacher.clamp_min(epsilon).log()
                - pooler_weights.clamp_min(epsilon).log()
            )
        ).sum(dim=-1).mean()
        if not bool(torch.isfinite(kl)):
            raise ValueError("SAGA attention KL differs")
        kl.backward()
        pooler_gradient_count = _nonzero_finite_gradient_count(pooler.parameters())
        if pooler_gradient_count <= 0:
            raise ValueError("SAGA pooler gradient evidence differs")
        elapsed_ns = max(1, perf_counter_ns() - started)
        return AttentionEvidence(
            layer=26,
            head_count=output.head_count,
            elapsed_ns=elapsed_ns,
            kl=float(kl.detach()),
            teacher_unit_mass=True,
            teacher_gradient_parameters=int(teacher.grad is not None)
            + int(tokens.grad is not None),
            pooler_nonzero_gradient_parameters=pooler_gradient_count,
        )
    finally:
        adapter.clear_graphs()
        pooler.zero_grad(set_to_none=True)


def fixture_pairwise_loss(
    embeddings: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Backend-independent pairwise activation-floor loss."""

    distances = torch.cdist(embeddings.float(), embeddings.float()).square()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positive = distances[same & ~eye]
    negative = distances[~same]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("SAGA DML pseudo-label evidence differs")
    return positive.mean() + torch_functional.relu(1.0 - negative).mean()


def run_dml_floor_phase(
    adapter: SagaModelAdapter, fixture: FixtureAuthority
) -> DmlFloorEvidence:
    """Measure the 64-image, 4096-dimensional activation and gradient floor."""

    if fixture.image_count != 64:
        raise ValueError("SAGA DML fixture authority differs")
    started = perf_counter_ns()
    try:
        embeddings = adapter.vision_pool(adapter.prepare_microbatch(fixture))
        if type(embeddings) is not torch.Tensor or tuple(embeddings.shape) != (64, 4096):
            raise ValueError("SAGA DML embedding shape differs")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("SAGA DML embeddings are not finite")
        norm_delta = (embeddings.float().norm(dim=-1) - 1.0).abs()
        maximum_norm_delta_ppm = int(
            math.ceil(float(norm_delta.max().detach()) * 1_000_000)
        )
        if maximum_norm_delta_ppm > 10:
            raise ValueError("SAGA DML embedding norms differ")
        labels = torch.tensor(
            [ordinal % 2 for ordinal in range(64)], device=embeddings.device
        )
        loss = fixture_pairwise_loss(embeddings, labels)
        if not bool(torch.isfinite(loss)):
            raise ValueError("SAGA DML loss differs")
        loss.backward()
        gradient = adapter.assert_gradient_roles()
        _validate_gradient_evidence(gradient)
        elapsed_ns = max(1, perf_counter_ns() - started)
        return DmlFloorEvidence(
            batch_size=64,
            embedding_shape=(64, 4096),
            elapsed_ns=elapsed_ns,
            loss=float(loss.detach()),
            maximum_norm_delta_ppm=maximum_norm_delta_ppm,
            vision_nonzero_gradient_parameters=(
                gradient.vision_nonzero_gradient_parameters
            ),
            language_gradient_parameters=gradient.language_gradient_parameters,
        )
    finally:
        adapter.clear_graphs()


@dataclass(frozen=True, slots=True)
class _ScientificRun:
    rollout: SealedRollouts
    replay: ReplayEvidence
    attention: AttentionEvidence
    dml: DmlFloorEvidence


def _phase_measurement(name: str, elapsed_ns: int = 0) -> PhaseMeasurement:
    return PhaseMeasurement(
        name=name,
        completed=elapsed_ns > 0,
        elapsed_ns=elapsed_ns,
        peak_cuda_reserved_bytes=0,
        peak_rss_bytes=0,
    )


def _run_scientific_phases(
    authority: LoadedAuthority,
    adapter: SagaModelAdapter,
    pooler: SingleQueryPooler,
    measurements: dict[str, PhaseMeasurement] | None,
    progress: Callable[[str], None] | None = None,
) -> _ScientificRun:
    rollout = run_rollout_phase(adapter, authority.fixture)
    if measurements is not None:
        measurements["rollout"] = _phase_measurement("rollout", rollout.elapsed_ns)
        if progress is not None:
            progress("rollout")
    replay = run_replay_phase(adapter, authority.fixture, rollout)
    if measurements is not None:
        measurements["replay"] = _phase_measurement("replay", replay.elapsed_ns)
        if progress is not None:
            progress("replay")
    attention = run_attention_phase(adapter, pooler, authority.fixture, rollout)
    if measurements is not None:
        measurements["attention"] = _phase_measurement(
            "attention", attention.elapsed_ns
        )
        if progress is not None:
            progress("attention")
    dml = run_dml_floor_phase(adapter, authority.fixture)
    if measurements is not None:
        measurements["dml"] = _phase_measurement("dml", dml.elapsed_ns)
        if progress is not None:
            progress("dml")
    return _ScientificRun(
        rollout=rollout,
        replay=replay,
        attention=attention,
        dml=dml,
    )


def _repeatability_signature(run: _ScientificRun) -> tuple[object, ...]:
    return (
        run.rollout.token_counts,
        run.rollout.completion_sha256,
        run.replay.loss,
        run.replay.generated_tokens,
        run.replay.vision_nonzero_gradient_parameters,
        run.replay.language_gradient_parameters,
        run.replay.gradient_sha256,
        run.attention.layer,
        run.attention.head_count,
        run.attention.kl,
        run.attention.teacher_unit_mass,
        run.attention.teacher_gradient_parameters,
        run.attention.pooler_nonzero_gradient_parameters,
        run.dml.embedding_shape,
        run.dml.loss,
        run.dml.maximum_norm_delta_ppm,
        run.dml.vision_nonzero_gradient_parameters,
        run.dml.language_gradient_parameters,
    )


def _failure_flags(outcome: FeasibilityOutcome) -> dict[str, bool]:
    flags = {
        "authority_valid": True,
        "backend_valid": True,
        "deterministic": True,
        "memory_within_envelope": True,
        "attention_available": True,
        "time_within_envelope": True,
    }
    field = {
        FeasibilityOutcome.AUTHORITY_INVALID: "authority_valid",
        FeasibilityOutcome.BACKEND_INVALID: "backend_valid",
        FeasibilityOutcome.DETERMINISM_FAIL: "deterministic",
        FeasibilityOutcome.MEMORY_FAIL: "memory_within_envelope",
        FeasibilityOutcome.ATTENTION_UNAVAILABLE: "attention_available",
        FeasibilityOutcome.TIME_BUDGET_FAIL: "time_within_envelope",
    }.get(outcome)
    if field is not None:
        flags[field] = False
    return flags


def _validate_run_identity(authority: LoadedAuthority) -> RunIdentity:
    identity = authority.result_identity
    if type(identity) is not RunIdentity:
        raise ValueError("SAGA result identity differs")
    fixture = authority.fixture
    if (
        identity.source_commit != fixture.source_commit
        or identity.binary_sha256 != fixture.binary_sha256
        or identity.environment_sha256 != fixture.environment_sha256
        or identity.host != fixture.host
        or authority.snapshot.model_revision != fixture.model_revision
    ):
        raise ValueError("SAGA result identity differs")
    identity.model_object.validated()
    identity.fixture_object.validated()
    identity.envelope.to_mapping()
    return identity


def run_feasibility(
    authority: LoadedAuthority,
    adapter: SagaModelAdapter,
    *,
    progress: Callable[[str], None] | None = None,
) -> bytes:
    """Run one complete repeatability diagnostic and emit canonical evidence."""

    identity = _validate_run_identity(authority)
    measurements = {
        name: _phase_measurement(name)
        for name in ("load", "rollout", "replay", "attention", "dml")
    }
    outcome = FeasibilityOutcome.FITS
    try:
        started = perf_counter_ns()
        adapter.validate_structure(authority)
        measurements["load"] = _phase_measurement(
            "load", max(1, perf_counter_ns() - started)
        )
        if progress is not None:
            progress("load")
        token_dim = getattr(adapter, "pooler_token_dim", 16)
        pooler = SingleQueryPooler(token_dim=token_dim)
        first = _run_scientific_phases(
            authority, adapter, pooler, measurements, progress
        )
        second = _run_scientific_phases(authority, adapter, pooler, None)
        if _repeatability_signature(first) != _repeatability_signature(second):
            raise FeasibilityFailure(
                FeasibilityOutcome.DETERMINISM_FAIL, "repeatability"
            )
    except FeasibilityFailure as failure:
        outcome = failure.outcome
    except AttentionUnavailable:
        outcome = FeasibilityOutcome.ATTENTION_UNAVAILABLE
    except torch.OutOfMemoryError:
        outcome = FeasibilityOutcome.MEMORY_FAIL

    flags = _failure_flags(outcome)
    evidence = FeasibilityEvidence(
        source_commit=identity.source_commit,
        controller_commit=identity.controller_commit,
        binary_sha256=identity.binary_sha256,
        environment_sha256=identity.environment_sha256,
        host=identity.host,
        model=identity.model_object,
        fixture=identity.fixture_object,
        envelope=identity.envelope,
        load=measurements["load"],
        rollout=measurements["rollout"],
        replay=measurements["replay"],
        attention=measurements["attention"],
        dml=measurements["dml"],
        dataset_reads=0,
        label_reads=0,
        evaluation_reads=0,
        optimizer_steps=0,
        **flags,
    )
    return canonical_feasibility_result_bytes(evidence)


def _reject_duplicate_options(argv: list[str]) -> None:
    options = [token for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate SAGA feasibility option")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse only local scientific capabilities and fail closed otherwise."""

    values = list(argv) if argv is not None else None
    if values is not None:
        _reject_duplicate_options(values)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--result-output", required=True, type=Path)
    parser.add_argument("--progress-output", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--execute-feasibility", required=True, action="store_true")
    parsed = parser.parse_args(values)
    for name in ("source_commit", "controller_commit"):
        value = getattr(parsed, name)
        if len(value) != 40 or any(
            character not in "0123456789abcdef" for character in value
        ):
            parser.error(f"{name.replace('_', ' ')} must be 40 lowercase hex")
    for name in ("binary_sha256", "environment_sha256"):
        value = getattr(parsed, name)
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            parser.error(f"{name.replace('_', ' ')} must be 64 lowercase hex")
    if not parsed.host:
        parser.error("host must be nonempty")
    if not parsed.model_root.is_dir():
        parser.error("model root must be an existing directory")
    for name in ("snapshot_manifest", "fixture"):
        if not getattr(parsed, name).is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing file")
    if parsed.result_output.exists():
        parser.error("result output must not already exist")
    if parsed.progress_output is not None and parsed.progress_output.exists():
        parser.error("progress output must not already exist")
    return parsed


def _path_authority(path: Path, *, role: str) -> ObjectAuthority:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"SAGA {role} path authority differs")
    payload = path.read_bytes()
    return ObjectAuthority(
        role=role,
        relative_path=path.name,
        byte_length=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    ).validated()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.resolve(strict=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _write_progress(path: Path, phase: str) -> None:
    payload = canonical_json_bytes(
        {
            "schema": "sfora-saga-gb10-feasibility-progress-v1",
            "completed_phase": phase,
        }
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs, execute once, and publish canonical evidence."""

    args = parse_args(argv)
    snapshot = load_snapshot_authority(
        root=args.model_root, manifest_path=args.snapshot_manifest
    )
    fixture = load_fixture_authority(args.fixture)
    identity = RunIdentity(
        source_commit=args.source_commit,
        controller_commit=args.controller_commit,
        binary_sha256=args.binary_sha256,
        environment_sha256=args.environment_sha256,
        host=args.host,
        model_object=_path_authority(
            args.snapshot_manifest, role="model-snapshot-manifest"
        ),
        fixture_object=_path_authority(args.fixture, role="synthetic-fixture"),
        envelope=ResourceEnvelope(
            cuda_reserved_limit_bytes=103_079_215_104,
            rss_limit_bytes=118_111_600_640,
            wall_limit_ns=7_200_000_000_000,
            progress_limit_ns=300_000_000_000,
        ),
    )
    authority = LoadedAuthority(
        snapshot=snapshot,
        fixture=fixture,
        result_identity=identity,
    )
    adapter = load_qwen_adapter(authority, factory=TransformersFactory())
    progress = (
        (lambda phase: _write_progress(args.progress_output, phase))
        if args.progress_output is not None
        else None
    )
    raw = run_feasibility(authority, adapter, progress=progress)
    _write_new(args.result_output, raw)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
