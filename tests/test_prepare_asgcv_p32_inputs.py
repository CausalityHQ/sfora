from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.data import ImageExample

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_asgcv_p32_inputs.py"
_SPEC = importlib.util.spec_from_file_location("prepare_asgcv_p32_inputs_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _PromptResolver:
    def resolve(self, _model_root: Path, prompt: str) -> object:
        assert prompt == _MODULE.P32_PROMPT
        return _MODULE.P32PromptAuthority(
            same_prefix_ids=(11,),
            different_prefix_ids=(22,),
            terminal_token_ids=(99,),
            attribute_token_span=(101, 109),
            patch_tokens_per_image=49,
        )


class _Tokenizer:
    eos_token_id = 99

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return {
            "SAME": [11],
            "DIFFERENT": [22],
            _MODULE.P32_PROMPT: [31, 32, 33, 34],
        }[text]


class _Processor:
    tokenizer = _Tokenizer()

    def apply_chat_template(self, messages: object, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert content[2]["text"] == _MODULE.P32_PROMPT
        return {
            "input_ids": torch.tensor([[90, 91, 31, 32, 33, 34, 92, 93]]),
            "mm_token_type_ids": torch.tensor([[1, 1, 0, 0, 0, 0, 1, 1]]),
        }


def test_prompt_resolver_derives_token_and_fixed_image_boundaries() -> None:
    resolver = _MODULE.TransformersPromptResolver(
        processor_loader=lambda root: _Processor() if root == Path("/model") else None
    )
    authority = resolver.resolve(Path("/model"), _MODULE.P32_PROMPT)
    assert authority == _MODULE.P32PromptAuthority(
        same_prefix_ids=(11,),
        different_prefix_ids=(22,),
        terminal_token_ids=(99,),
        attribute_token_span=(2, 6),
        patch_tokens_per_image=2,
    )


def _cars_train_examples() -> list[ImageExample]:
    rows = []
    for label in range(98):
        for ordinal in range(4):
            rows.append(
                ImageExample(
                    example_id=f"cars-{label:03d}-{ordinal:02d}",
                    image=np.full((5, 7, 3), label + ordinal, dtype=np.uint8),
                    label=label,
                )
            )
    return rows


def test_preparer_seals_train_only_arrays_partition_schedule_and_prompt(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    result = _MODULE.prepare_p32_inputs(
        output_root=output,
        model_root=tmp_path / "model",
        source_commit="1" * 40,
        model_revision="2" * 40,
        partition_seed_sha256="3" * 64,
        rollout_seed_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        examples=_cars_train_examples(),
        prompt_resolver=_PromptResolver(),
    )

    assert result.train_manifest == output / "train-manifest.json"
    assert result.launch_authority == output / "p32-authority.json"
    manifest = json.loads(result.train_manifest.read_bytes())
    authority = json.loads(result.launch_authority.read_bytes())
    assert manifest["schema"] == "sfora-cars-train-p32-manifest-v1"
    assert manifest["official_test_access"] is False
    assert len(manifest["predictor_train"]) == 256
    assert len(manifest["e0_validation"]) == 68
    assert len(manifest["e1_optimization"]) == 68
    assert len({row["label"] for row in manifest["predictor_train"]}) == 64
    assert len({row["label"] for row in manifest["e0_validation"]}) == 17
    assert len({row["label"] for row in manifest["e1_optimization"]}) == 17
    assert (
        max(row["label"] for role in manifest.values() if isinstance(role, list) for row in role)
        < 98
    )
    assert all((output / row["array_path"]).is_file() for row in manifest["predictor_train"])
    first = np.load(output / manifest["predictor_train"][0]["array_path"])
    assert first.shape == (224, 224, 3)
    assert first.dtype == np.uint8
    assert authority["schema"] == "sfora-asgcv-p32-launch-v1"
    assert authority["source_commit"] == "1" * 40
    assert authority["prompt_utf8"] == _MODULE.P32_PROMPT
    assert authority["attribute_token_span"] == [101, 109]
    assert authority["patch_tokens_per_image"] == 49
    assert authority["completion_protocol"]["same_prefix_ids"] == [11]
    assert authority["completion_protocol"]["different_prefix_ids"] == [22]
    assert len(authority["pilot_schedule"]["pairs"]) == 32
    assert result.fixture.is_file()
    assert result.fixture.read_bytes().endswith(b"\n")
    assert all(
        path.read_bytes().endswith(b"\n")
        for path in (result.train_manifest, result.launch_authority, result.fixture)
    )


def test_preparer_rejects_official_test_rows_and_existing_outputs(tmp_path: Path) -> None:
    examples = _cars_train_examples()
    common = dict(
        model_root=tmp_path / "model",
        source_commit="1" * 40,
        model_revision="2" * 40,
        partition_seed_sha256="3" * 64,
        rollout_seed_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        prompt_resolver=_PromptResolver(),
    )
    with np.testing.assert_raises(ValueError):
        _MODULE.prepare_p32_inputs(
            output_root=tmp_path / "official-test",
            examples=[*examples, ImageExample("test-row", np.zeros((2, 2, 3), np.uint8), 98)],
            **common,
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    with np.testing.assert_raises(ValueError):
        _MODULE.prepare_p32_inputs(output_root=existing, examples=examples, **common)


def test_cli_authenticates_snapshot_and_loads_cars_train_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    model_root = tmp_path / "model"
    snapshot = tmp_path / "snapshot.json"
    output = tmp_path / "prepared"

    class _Snapshot:
        model_revision = "2" * 40

    monkeypatch.setattr(
        _MODULE,
        "load_snapshot_authority",
        lambda *, root, manifest_path: calls.append((root, manifest_path)) or _Snapshot(),
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE,
        "load_image_retrieval_examples",
        lambda **kwargs: calls.append(kwargs) or _cars_train_examples(),
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE,
        "TransformersPromptResolver",
        lambda: _PromptResolver(),
    )

    assert (
        _MODULE.main(
            [
                "--output-root",
                str(output),
                "--model-root",
                str(model_root),
                "--snapshot-manifest",
                str(snapshot),
                "--model-revision",
                "2" * 40,
                "--source-commit",
                "1" * 40,
                "--partition-seed-sha256",
                "3" * 64,
                "--rollout-seed-sha256",
                "4" * 64,
                "--predictor-initialization-seed-sha256",
                "5" * 64,
                "--execute-preparation",
            ]
        )
        == 0
    )
    assert calls == [
        (model_root, snapshot),
        {
            "dataset_name": "cars",
            "split": "train",
            "limit_per_class": None,
            "min_per_class": None,
            "max_classes": None,
            "seed": 0,
        },
    ]
    assert output.is_dir()


def test_cli_requires_explicit_execution_and_rejects_unknown_flags() -> None:
    common = [
        "--output-root",
        "/output",
        "--model-root",
        "/model",
        "--snapshot-manifest",
        "/snapshot.json",
        "--model-revision",
        "2" * 40,
        "--source-commit",
        "1" * 40,
        "--partition-seed-sha256",
        "3" * 64,
        "--rollout-seed-sha256",
        "4" * 64,
        "--predictor-initialization-seed-sha256",
        "5" * 64,
    ]
    with pytest.raises(SystemExit):
        _MODULE.parse_args(common)
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*common, "--execute-preparation", "--dataset-split", "test"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*common, "--output-root", "/different", "--execute-preparation"])
