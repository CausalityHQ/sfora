from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from sfora.saga_feasibility import FixtureAuthority, SnapshotAuthority

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnose_saga_gb10_feasibility.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_saga_gb10_feasibility", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _valid_cli_args(tmp_path: Path) -> list[str]:
    model_root = tmp_path / "model"
    model_root.mkdir(exist_ok=True)
    snapshot = tmp_path / "snapshot.json"
    fixture = tmp_path / "fixture.json"
    snapshot.touch()
    fixture.touch()
    return [
        "--model-root",
        str(model_root),
        "--snapshot-manifest",
        str(snapshot),
        "--fixture",
        str(fixture),
        "--result-output",
        str(tmp_path / "result.json"),
        "--source-commit",
        "a" * 40,
        "--execute-feasibility",
    ]


def test_cli_accepts_only_local_scientific_capabilities(tmp_path: Path) -> None:
    args = _MODULE.parse_args(_valid_cli_args(tmp_path))
    assert args.execute_feasibility is True
    assert args.model_root == tmp_path / "model"
    for forbidden in (
        "--dataset",
        "--labels",
        "--checkpoint",
        "--hub-token",
        "--model-uri",
        "--aws-profile",
        "--train",
        "--evaluate",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*_valid_cli_args(tmp_path), forbidden, "x"])


@pytest.mark.parametrize("missing_index", range(0, 12, 2))
def test_cli_rejects_missing_required_flags(
    tmp_path: Path, missing_index: int
) -> None:
    argv = _valid_cli_args(tmp_path)
    del argv[missing_index : missing_index + 2]
    with pytest.raises(SystemExit):
        _MODULE.parse_args(argv)


def test_cli_rejects_duplicate_flags(tmp_path: Path) -> None:
    argv = _valid_cli_args(tmp_path)
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*argv, "--fixture", str(tmp_path / "fixture.json")])


@dataclass
class _Parameter:
    requires_grad: bool


class _FakeModel:
    architecture = "Qwen3VLForConditionalGeneration"
    dtype = "bfloat16"
    device = "cuda"
    attention_backend = "eager"
    layer_count = 36

    def __init__(self) -> None:
        self.parameters_by_role = {
            "vision": [_Parameter(False)],
            "language": [_Parameter(True), _Parameter(True)],
        }


class _FakeProcessor:
    output_keys = ("attention_mask", "image_grid_thw", "input_ids", "pixel_values")


class _FakeFactory:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.model_kwargs: dict[str, object] | None = None
        self.processor_kwargs: dict[str, object] | None = None

    def load_model(self, root: Path, **kwargs: object) -> _FakeModel:
        assert root.is_dir()
        self.model_kwargs = kwargs
        return self.model

    def load_processor(self, root: Path, **kwargs: object) -> _FakeProcessor:
        assert root.is_dir()
        self.processor_kwargs = kwargs
        return _FakeProcessor()


def _loaded_authority(tmp_path: Path) -> object:
    root = tmp_path / "model"
    root.mkdir()
    snapshot = SnapshotAuthority(
        root=root,
        repository_id="Qwen/Qwen3-VL-8B-Instruct",
        model_revision="1" * 40,
        processor_revision="2" * 40,
        tokenizer_revision="3" * 40,
        snapshot_tree_sha256="4" * 64,
        architecture="Qwen3VLForConditionalGeneration",
        dtype="bfloat16",
        attention_backend="eager",
        files=(),
    )
    fixture = FixtureAuthority(
        source_commit="5" * 40,
        model_revision="1" * 40,
        binary_sha256="6" * 64,
        environment_sha256="7" * 64,
        host="spark-fixture",
        group_size=8,
        image_count=64,
        generation_seeds=tuple(range(8)),
        synthetic_rewards=(0, 1, 0, 1, 0, 1, 0, 1),
        attention_layer=26,
        prompt_sha256="8" * 64,
        message_serialization_sha256="9" * 64,
    )
    return _MODULE.LoadedAuthority(snapshot=snapshot, fixture=fixture)


def test_model_load_freezes_language_and_leaves_only_vision_trainable(
    tmp_path: Path,
) -> None:
    factory = _FakeFactory()
    adapter = _MODULE.load_qwen_adapter(_loaded_authority(tmp_path), factory=factory)
    assert adapter.architecture == "Qwen3VLForConditionalGeneration"
    assert adapter.trainable_parameter_roles() == ("vision",)
    assert adapter.frozen_parameter_roles() == ("language",)
    assert all(
        parameter.requires_grad
        for parameter in factory.model.parameters_by_role["vision"]
    )
    assert all(
        not parameter.requires_grad
        for parameter in factory.model.parameters_by_role["language"]
    )
    assert factory.model_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "attn_implementation": "eager",
    }
    assert factory.processor_kwargs == {
        "local_files_only": True,
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("architecture", "WrongModel"),
        ("dtype", "float32"),
        ("device", "cpu"),
        ("attention_backend", "flash_attention_2"),
        ("layer_count", 26),
    ],
)
def test_model_load_rejects_structural_drift(
    tmp_path: Path, attribute: str, value: object
) -> None:
    factory = _FakeFactory()
    setattr(factory.model, attribute, value)
    with pytest.raises(ValueError, match="model authority"):
        _MODULE.load_qwen_adapter(_loaded_authority(tmp_path), factory=factory)
