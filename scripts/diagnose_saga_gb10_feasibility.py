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
