from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_siglip_compatibility_capacity.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probe_siglip_compatibility_capacity", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

parse_args = _MODULE.parse_args
write_capacity_descriptor_artifact = _MODULE.write_capacity_descriptor_artifact
load_capacity_descriptor_artifact = _MODULE.load_capacity_descriptor_artifact


def _args(tmp_path: Path) -> list[str]:
    return [
        "--control-binding",
        str(tmp_path / "binding.json"),
        "--control-binding-sha256",
        "1" * 64,
        "--checkpoint-seed17",
        str(tmp_path / "checkpoint.pt"),
        "--optimization-manifest",
        str(tmp_path / "optimization.json"),
        "--optimization-manifest-sha256",
        "2" * 64,
        "--optimization-image-root",
        str(tmp_path / "images"),
        "--spatial-artifact",
        str(tmp_path / "tail.safetensors"),
        "--spatial-artifact-sha256",
        "3" * 64,
        "--descriptor-artifact",
        str(tmp_path / "descriptors.safetensors"),
        "--result",
        str(tmp_path / "result.json"),
        "--execute-capacity-diagnostic",
    ]


def test_capacity_cli_is_local_burned_data_only_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    assert parse_args(_args(tmp_path)).execute_capacity_diagnostic is True
    for mutation in (
        _args(tmp_path)[:-1],
        _args(tmp_path) + ["--spatial-artifact", str(tmp_path / "duplicate")],
        _args(tmp_path) + ["--evaluation-manifest", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--class-names", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--aws-profile", "forbidden"],
    ):
        with pytest.raises(SystemExit):
            parse_args(mutation)


def test_capacity_descriptor_artifact_roundtrips_identity_and_partition(
    tmp_path: Path,
) -> None:
    generator = torch.Generator().manual_seed(91)
    student = torch.nn.functional.normalize(
        torch.randn(98, 8, generator=generator), dim=1
    )
    teacher = torch.nn.functional.normalize(
        torch.randn(98, 8, generator=generator), dim=1
    )
    ids = tuple(f"cars-train-{index // 2}-{index}" for index in range(98))
    labels = tuple(index // 2 for index in range(98))
    path = tmp_path / "descriptors.safetensors"

    digest = write_capacity_descriptor_artifact(path, student, teacher, ids, labels)
    restored = load_capacity_descriptor_artifact(path, ids=ids, labels=labels)

    assert len(digest) == 64
    assert torch.equal(restored.student, student)
    assert torch.equal(restored.teacher, teacher)
    assert restored.ids == ids
    assert restored.labels == labels
    assert set(restored.fit_labels) | set(restored.development_labels) == set(range(49))
    assert set(restored.fit_labels).isdisjoint(restored.development_labels)
    with pytest.raises(FileExistsError):
        write_capacity_descriptor_artifact(path, student, teacher, ids, labels)


def test_capacity_descriptor_artifact_rejects_identity_and_tensor_drift(
    tmp_path: Path,
) -> None:
    student = torch.nn.functional.normalize(
        torch.arange(1, 98 * 4 + 1, dtype=torch.float32).reshape(98, 4), dim=1
    )
    teacher = student.clone()
    ids = tuple(f"id-{index}" for index in range(98))
    labels = tuple(index // 2 for index in range(98))
    path = tmp_path / "descriptors.safetensors"
    write_capacity_descriptor_artifact(path, student, teacher, ids, labels)

    with pytest.raises(ValueError, match="capacity descriptor artifact"):
        load_capacity_descriptor_artifact(
            path,
            ids=ids[:-1] + ("changed",),
            labels=labels,
        )
    with pytest.raises(ValueError, match="capacity descriptor artifact"):
        write_capacity_descriptor_artifact(
            tmp_path / "bad.safetensors",
            torch.full_like(student, float("nan")),
            teacher,
            ids,
            labels,
        )
