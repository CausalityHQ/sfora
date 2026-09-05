from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "probe_siglip_gallery_compatibility_alignment.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "probe_siglip_gallery_compatibility_alignment", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

parse_args = _MODULE.parse_args
stream_alignment_descriptor_pairs = _MODULE.stream_alignment_descriptor_pairs
write_alignment_artifact = _MODULE.write_alignment_artifact
load_alignment_artifact = _MODULE.load_alignment_artifact
materialize_registered_optimization_images = (
    _MODULE.materialize_registered_optimization_images
)


class _Output:
    def __init__(self, source: torch.Tensor, final: torch.Tensor) -> None:
        self.hidden_states = tuple(source if index == 18 else final for index in range(28))
        self.pooler_output = final.mean(dim=1)


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.post_layernorm = nn.Identity()
        self.head = _Mean()

    def forward(self, *, pixel_values: torch.Tensor, **_: object) -> _Output:
        source = pixel_values.reshape(pixel_values.shape[0], -1, pixel_values.shape[-1])
        return _Output(source, source * 2)


class _Mean(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.mean(dim=1)


class _Readout(nn.Module):
    def __init__(self, projection: nn.Linear) -> None:
        super().__init__()
        self.projection = projection

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = torch.nn.functional.normalize(self.projection(value.mean(dim=1)), dim=1)
        return (normalized * 0.99).to(torch.bfloat16)


def test_stream_produces_paired_normalized_student_teacher_descriptors() -> None:
    projection = nn.Linear(4, 3, bias=False).eval()
    torch.manual_seed(7)
    nn.init.normal_(projection.weight)
    student, teacher = stream_alignment_descriptor_pairs(
        _Vision().eval(),
        projection,
        nn.Identity().eval(),
        _Readout(projection).eval(),
        (torch.arange(32, dtype=torch.float32).reshape(2, 1, 4, 4) + 1,),
        source_depth=18,
        device=torch.device("cpu"),
    )
    assert student.shape == teacher.shape == (2, 3)
    assert student.dtype == teacher.dtype == torch.float32
    assert torch.allclose(
        torch.linalg.vector_norm(student, dim=1), torch.ones(2), atol=1e-6, rtol=0
    )
    assert torch.allclose(
        torch.linalg.vector_norm(teacher, dim=1), torch.ones(2), atol=1e-6, rtol=0
    )


def test_alignment_artifact_roundtrips_exclusively(tmp_path: Path) -> None:
    matrix = torch.eye(8, dtype=torch.float64)
    development_student = torch.eye(8, dtype=torch.float32)
    development_aligned = development_student.clone()
    path = tmp_path / "alignment.safetensors"
    digest = write_alignment_artifact(
        path, matrix, development_student, development_aligned
    )
    assert len(digest) == 64
    assert torch.equal(load_alignment_artifact(path), matrix)
    with pytest.raises(FileExistsError):
        write_alignment_artifact(path, matrix, development_student, development_aligned)
    with pytest.raises(ValueError, match="alignment descriptor evidence differs"):
        write_alignment_artifact(
            tmp_path / "bad.safetensors",
            matrix,
            development_student,
            development_aligned.roll(1, dims=0),
        )


def test_materializer_reads_only_registered_optimization_rows(tmp_path: Path) -> None:
    from PIL import Image

    class _Rows:
        def __init__(self, rows: list[object]) -> None:
            self.rows = rows
            self.reads: list[int] = []

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> object:
            self.reads.append(index)
            value = self.rows[index]
            if isinstance(value, BaseException):
                raise value
            return value

    train = _Rows(
        [
            {"image": Image.new("RGB", (2, 2), "red"), "label": 0},
            AssertionError("unregistered train row was accessed"),
        ]
    )
    test = _Rows(
        [
            {"image": Image.new("RGB", (2, 2), "blue"), "label": 1},
            AssertionError("unregistered test row was accessed"),
        ]
    )
    manifest = tmp_path / "optimization.json"
    manifest.write_text(
        json.dumps(
            {
                "claim_eligible": False,
                "dataset_id": "tanganke/stanford_cars",
                "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
                "examples": [
                    {"example_id": "cars-train-0-0", "label": 0},
                    {"example_id": "cars-train-1-2", "label": 1},
                ],
                "schema": "rsta-optimization-manifest-v1",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )

    def loader(_dataset_id: str, *, revision: str, split: str):  # type: ignore[no-untyped-def]
        assert revision == "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
        return {"train": train, "test": test}[split]

    output = tmp_path / "images"
    materialize_registered_optimization_images(manifest, output, dataset_loader=loader)
    assert train.reads == [0]
    assert test.reads == [0]
    assert len(tuple(output.iterdir())) == 2


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
        "--alignment-artifact",
        str(tmp_path / "alignment.safetensors"),
        "--result",
        str(tmp_path / "result.json"),
        "--execute-gallery-alignment",
    ]


def test_cli_is_local_optimization_only_and_rejects_duplicates(tmp_path: Path) -> None:
    assert parse_args(_args(tmp_path)).execute_gallery_alignment is True
    for mutation in (
        _args(tmp_path)[:-1],
        _args(tmp_path) + ["--spatial-artifact", str(tmp_path / "duplicate")],
        _args(tmp_path) + ["--evaluation-manifest", str(tmp_path / "forbidden")],
        _args(tmp_path) + ["--aws-profile", "forbidden"],
    ):
        with pytest.raises(SystemExit):
            parse_args(mutation)
