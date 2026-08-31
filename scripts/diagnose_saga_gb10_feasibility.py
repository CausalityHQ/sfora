#!/usr/bin/env python3
"""Run the local, quality-blind SAGA GB10 feasibility diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol

import torch
import torch.nn.functional as torch_functional

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import FixtureAuthority, SnapshotAuthority


@dataclass(frozen=True, slots=True)
class LoadedAuthority:
    """Cross-bound snapshot and synthetic fixture authorities."""

    snapshot: SnapshotAuthority
    fixture: FixtureAuthority


class ModelFactory(Protocol):
    """Offline-only factory boundary for the concrete Transformers adapter."""

    def load_model(self, root: Path, **kwargs: object) -> object: ...

    def load_processor(self, root: Path, **kwargs: object) -> object: ...


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
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-feasibility", required=True, action="store_true")
    parsed = parser.parse_args(values)
    if len(parsed.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in parsed.source_commit
    ):
        parser.error("source commit must be 40 lowercase hexadecimal characters")
    if not parsed.model_root.is_dir():
        parser.error("model root must be an existing directory")
    for name in ("snapshot_manifest", "fixture"):
        if not getattr(parsed, name).is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing file")
    if parsed.result_output.exists():
        parser.error("result output must not already exist")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Parse the dedicated boundary; later tasks add the scientific runner."""

    parse_args(argv)
    raise RuntimeError("SAGA feasibility scientific runner is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
