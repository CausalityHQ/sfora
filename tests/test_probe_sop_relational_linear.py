from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

_EXPORT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_unicom_sop_embeddings.py"
_EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_unicom_sop_embeddings", _EXPORT_SCRIPT
)
assert _EXPORT_SPEC is not None and _EXPORT_SPEC.loader is not None
_EXPORT_MODULE = importlib.util.module_from_spec(_EXPORT_SPEC)
sys.modules[_EXPORT_SPEC.name] = _EXPORT_MODULE
_EXPORT_SPEC.loader.exec_module(_EXPORT_MODULE)
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_sop_relational_linear.py"
_SPEC = importlib.util.spec_from_file_location("probe_sop_relational_linear", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC.loader.exec_module(_MODULE)


def _write_archive(path: Path, root: Path, model: str, offset: float = 0.0) -> None:
    (root / "a").mkdir(parents=True)
    for name in ("a/1.jpg", "a/2.jpg", "a/3.jpg", "a/4.jpg"):
        (root / name).write_bytes(name.encode())
    (root / "Ebay_train.txt").write_text(
        "image_id class_id super_class_id path\n1 1 1 a/1.jpg\n2 1 1 a/2.jpg\n",
        encoding="utf-8",
    )
    (root / "Ebay_test.txt").write_text(
        "image_id class_id super_class_id path\n3 2 1 a/3.jpg\n4 2 1 a/4.jpg\n",
        encoding="utf-8",
    )
    records = _EXPORT_MODULE.parse_sop_records(
        root, expected_counts=(2, 2), expected_classes=(1, 1)
    )

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        return np.asarray(
            [[float(index) + offset, 1.0, 2.0] for index, _path in enumerate(paths, 1)],
            dtype=np.float32,
        )

    _EXPORT_MODULE.export_sop_embeddings(
        records,
        encode,
        {
            "model_identifier": model,
            "model_revision": "ab" * 20,
            "checkpoint_sha256": "cd" * 32,
            "transform": "fixture",
        },
        path,
        expected_counts=(2, 2),
        expected_classes=(1, 1),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_score_symmetric_excludes_self_and_matches_hand_derived_metrics() -> None:
    values = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.99, 0.10],
                [0.10, 0.99],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )

    result = _MODULE.score_symmetric(
        values,
        (1, 1, 2, 2),
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert result["map_at_r"] == 0.0
    assert result["r1"] == 0.0
    assert result["per_query_ap"] == (0.0, 0.0, 0.0, 0.0)
    assert result["per_query_r1"] == (0.0, 0.0, 0.0, 0.0)


def test_score_symmetric_packed_and_float_agree_on_exact_geometry() -> None:
    values = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = (1, 1, 2, 2)

    floating = _MODULE.score_symmetric(
        values, labels, candidate_width=2, device=torch.device("cpu")
    )
    packed = _MODULE.score_symmetric(
        pack_int8_unit_embeddings(values),
        labels,
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert floating == packed
    assert floating["map_at_r"] == 1.0
    assert floating["r1"] == 1.0


@pytest.mark.parametrize(
    ("labels", "candidate_width", "match"),
    (
        ((1, 2, 2), 2, "singleton"),
        ((1, 1, 1, 2, 2, 2), 1, "candidate"),
        ((1, 1, 2, 2), 4, "candidate"),
    ),
)
def test_score_symmetric_rejects_invalid_positive_or_candidate_authority(
    labels: tuple[int, ...], candidate_width: int, match: str
) -> None:
    values = F.normalize(
        torch.arange(1, len(labels) * 3 + 1).reshape(len(labels), 3).float(), dim=1
    )
    with pytest.raises(ValueError, match=match):
        _MODULE.score_symmetric(
            values,
            labels,
            candidate_width=candidate_width,
            device=torch.device("cpu"),
        )


def test_load_paired_archives_authenticates_and_binds_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    _write_archive(source, tmp_path / "source-data", "source", 0.0)
    _write_archive(teacher, tmp_path / "teacher-data", "teacher", 10.0)

    pair = _MODULE.load_paired_archives(
        source,
        _sha256(source),
        teacher,
        _sha256(teacher),
        expected_counts=(2, 2),
        expected_classes=(1, 1),
        expected_dimension=3,
        expected_identifiers=("source", "teacher"),
    )

    assert pair["source_train"].shape == (2, 3)
    assert pair["teacher_train"].shape == (2, 3)
    assert pair["source_test"].shape == (2, 3)
    assert pair["teacher_test"].shape == (2, 3)
    assert pair["train_labels"] == (1, 1)
    assert pair["test_labels"] == (2, 2)

    with pytest.raises(ValueError, match="digest"):
        _MODULE.load_paired_archives(
            source,
            "00" * 32,
            teacher,
            _sha256(teacher),
            expected_counts=(2, 2),
            expected_classes=(1, 1),
            expected_dimension=3,
            expected_identifiers=("source", "teacher"),
        )


def test_parse_args_requires_authenticated_archives_and_three_outputs() -> None:
    args = _MODULE.parse_args(
        [
            "--source-embeddings",
            "/source.npz",
            "--source-embeddings-sha256",
            "11" * 32,
            "--teacher-embeddings",
            "/teacher.npz",
            "--teacher-embeddings-sha256",
            "22" * 32,
            "--output",
            "/quality.json",
            "--model-output",
            "/model.sfora-rl1",
            "--latency-output",
            "/latency.json",
            "--execute-relational-linear",
        ]
    )
    assert args.source_embeddings == Path("/source.npz")
    assert args.source_embeddings_sha256 == "11" * 32
    assert args.teacher_embeddings == Path("/teacher.npz")
    assert args.teacher_embeddings_sha256 == "22" * 32
    assert args.output == Path("/quality.json")
    assert args.model_output == Path("/model.sfora-rl1")
    assert args.latency_output == Path("/latency.json")
    assert args.execute_relational_linear is True

    with pytest.raises(SystemExit):
        _MODULE.parse_args(["--unknown"])


def test_main_publishes_three_seed_quality_model_and_latency_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.1, 0.0],
                [0.9, 0.1, 0.0, 0.1],
                [0.0, 1.0, 0.1, 0.0],
                [0.1, 0.9, 0.0, 0.1],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    pair = {
        "source_metadata": {},
        "teacher_metadata": {},
        "source_train": source,
        "teacher_train": source,
        "source_test": source,
        "teacher_test": source,
        "train_labels": (1, 1, 2, 2),
        "test_labels": (3, 3, 4, 4),
    }
    basis = torch.eye(4, dtype=torch.float32)[:3].contiguous()

    monkeypatch.setattr(_MODULE, "DIMENSIONS", 2)
    monkeypatch.setattr(_MODULE, "PCA_CONTROL_DIMENSIONS", 3)
    monkeypatch.setattr(_MODULE, "CANDIDATE_WIDTH", 2)
    monkeypatch.setattr(_MODULE, "LATENCY_WARMUP_PAIRS", 0)
    monkeypatch.setattr(_MODULE, "LATENCY_MEASURED_PAIRS", 2)
    monkeypatch.setattr(_MODULE, "MAP_IMPROVEMENT_GATE", -1.0)
    monkeypatch.setattr(_MODULE, "load_paired_archives", lambda *_args, **_kwargs: pair)
    monkeypatch.setattr(
        _MODULE,
        "_fit_uncentered_covariance_basis",
        lambda *_args, **_kwargs: basis,
    )
    monkeypatch.setattr(
        _MODULE,
        "fit_relational_linear_encoder",
        lambda *_args, **_kwargs: (
            _MODULE.RelationalLinearEncoder(basis[:2].contiguous()),
            (0.5, 0.25),
        ),
    )
    monkeypatch.setattr(
        _MODULE,
        "_encode_floating",
        lambda model, values, **_kwargs: model(values).detach().cpu(),
    )
    monkeypatch.setattr(
        _MODULE,
        "_encode_deployed",
        lambda model, values, **_kwargs: (
            model(values).detach().cpu(),
            pack_int8_unit_embeddings(model(values).detach().cpu()),
        ),
    )
    monkeypatch.setattr(_MODULE, "_familywise_lower_bound", lambda *_args, **_kwargs: 0.1)
    monkeypatch.setattr(
        _MODULE,
        "_profile_paired_latency",
        lambda *_args, **_kwargs: {
            "alternating_order": True,
            "baseline": {"p95_ns": 100, "samples_ns": [100, 100]},
            "measured_pairs": 2,
            "treatment": {"p95_ns": 101, "samples_ns": [101, 101]},
            "warmup_pairs": 0,
        },
    )
    quality = tmp_path / "quality.json"
    model = tmp_path / "model.sfora-rl1"
    latency = tmp_path / "latency.json"
    arguments = [
        "--source-embeddings",
        "/source.npz",
        "--source-embeddings-sha256",
        "11" * 32,
        "--teacher-embeddings",
        "/teacher.npz",
        "--teacher-embeddings-sha256",
        "22" * 32,
        "--output",
        str(quality),
        "--model-output",
        str(model),
        "--latency-output",
        str(latency),
        "--execute-relational-linear",
    ]

    assert _MODULE.main(arguments) == 0

    receipt = json.loads(quality.read_bytes())
    latency_receipt = json.loads(latency.read_bytes())
    assert set(receipt["arms"]) == {"17", "1729", "65537"}
    assert receipt["passes_quality"] is True
    assert receipt["passes_latency"] is True
    assert receipt["persistent_bytes_per_item"] == 66
    assert latency_receipt["measured_pairs"] == 2
    assert model.read_bytes().startswith(b"SFORA-RL1")
    assert quality.read_bytes().endswith(b"\n")
    assert latency.read_bytes().endswith(b"\n")
    assert not tuple(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError, match="output exists"):
        _MODULE.main(arguments)
