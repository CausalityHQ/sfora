#!/usr/bin/env python3
"""Run the local, quality-blind SAGA GB10 feasibility diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
