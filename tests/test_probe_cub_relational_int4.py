from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.packed_int4 import pack_int4_unit_embeddings

_ROOT = Path(__file__).resolve().parents[1]
for name in ("export_unicom_cub_embeddings",):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
_EXPORT = sys.modules["export_unicom_cub_embeddings"]
_SCRIPT = _ROOT / "scripts/probe_cub_relational_int4.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("probe_cub_relational_int4", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _unit(rows: int, dimensions: int, offset: int = 0) -> torch.Tensor:
    values = torch.arange(offset + 1, offset + rows * dimensions + 1).reshape(rows, dimensions)
    return F.normalize(values.float(), dim=1)


def _write_archive(
    path: Path,
    root: Path,
    model: str,
    offset: float,
    *,
    revision: str = "ab" * 20,
    checkpoint_sha256: str = "cd" * 32,
) -> None:
    from test_export_unicom_cub_embeddings import _write_cub_fixture

    root.mkdir()
    _write_cub_fixture(root)
    records = _EXPORT.parse_cub_records(
        root,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_total_classes=4,
        train_class_max=2,
    )

    def encode(paths: tuple[Path, ...]) -> np.ndarray:
        values = np.asarray(
            [[index + offset, 1.0, 2.0, 3.0] for index, _ in enumerate(paths, 1)],
            dtype=np.float32,
        )
        return np.ascontiguousarray(values / np.linalg.norm(values, axis=1, keepdims=True))

    _EXPORT.export_cub_embeddings(
        records,
        encode,
        {
            "model_identifier": model,
            "model_revision": revision,
            "checkpoint_sha256": checkpoint_sha256,
            "transform": "fixture",
            "dataset_archive_sha256": _EXPORT.CUB_ARCHIVE_SHA256,
            "dataset_archive_md5": _EXPORT.CUB_ARCHIVE_MD5,
            "dataset_content_sha256": "ef" * 32,
            "cub_exporter_source_sha256": _sha(_ROOT / "scripts/export_unicom_cub_embeddings.py"),
            "sop_exporter_source_sha256": _sha(_ROOT / "scripts/export_unicom_sop_embeddings.py"),
        },
        path,
        expected_counts=(4, 4),
        expected_classes=(2, 2),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cub_pair_loader_authenticates_and_binds_all_row_identities(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    _write_archive(source, tmp_path / "source", "source", 0.0)
    _write_archive(teacher, tmp_path / "teacher", "teacher", 10.0)

    pair = _MODULE.load_paired_cub_archives(
        source,
        _sha(source),
        teacher,
        _sha(teacher),
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_dimension=4,
        expected_identifiers=("source", "teacher"),
        expected_revision="ab" * 20,
        expected_checkpoint_sha256=("cd" * 32, "cd" * 32),
    )

    assert pair["source_train"].shape == (4, 4)
    assert pair["teacher_test"].shape == (4, 4)
    assert pair["test_labels"] == (3, 3, 4, 4)
    assert pair["source_archive_sha256"] == _sha(source)
    assert pair["teacher_archive_sha256"] == _sha(teacher)
    authority = _MODULE.cub_evidence_authority(pair, source_commit="78" * 20)
    assert authority["source_embeddings_sha256"] == _sha(source)
    assert authority["teacher_embeddings_sha256"] == _sha(teacher)
    assert authority["source_model_identifier"] == "source"
    assert authority["teacher_model_identifier"] == "teacher"
    assert authority["source_commit"] == "78" * 20
    assert authority["dataset_archive_sha256"] == _EXPORT.CUB_ARCHIVE_SHA256
    assert authority["ordered_record_sha256"] == pair["source_metadata"]["ordered_record_sha256"]
    assert len(authority["evaluator_source_sha256"]) == 64
    assert len(authority["packed_int4_source_sha256"]) == 64
    assert len(authority["relational_compaction_source_sha256"]) == 64
    with pytest.raises(ValueError, match="digest"):
        _MODULE.load_paired_cub_archives(
            source,
            "00" * 32,
            teacher,
            _sha(teacher),
            expected_counts=(4, 4),
            expected_classes=(2, 2),
            expected_dimension=4,
            expected_identifiers=("source", "teacher"),
            expected_revision="ab" * 20,
            expected_checkpoint_sha256=("cd" * 32, "cd" * 32),
        )


def test_source_commit_must_contain_every_executing_scientific_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def committed(args: list[str], **_kwargs: object) -> SimpleNamespace:
        relative = args[-1].split(":", 1)[1]
        return SimpleNamespace(stdout=(_ROOT / relative).read_bytes())

    monkeypatch.setattr(_MODULE.subprocess, "run", committed)
    _MODULE.verify_source_commit("12" * 20)

    def drifted(args: list[str], **kwargs: object) -> SimpleNamespace:
        result = committed(args, **kwargs)
        if args[-1].endswith("scripts/probe_cub_relational_int4.py"):
            result.stdout += b"drift"
        return result

    monkeypatch.setattr(_MODULE.subprocess, "run", drifted)
    with pytest.raises(ValueError, match="registered commit"):
        _MODULE.verify_source_commit("12" * 20)


def test_pair_loader_requires_frozen_official_models_and_checkpoints(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    _write_archive(
        source,
        tmp_path / "source",
        "UNICOM-ViT-B/16",
        0.0,
        revision=_MODULE.UNICOM_REVISION,
        checkpoint_sha256=_MODULE.OFFICIAL_CHECKPOINT_SHA256[0],
    )
    _write_archive(
        teacher,
        tmp_path / "teacher",
        "UNICOM-ViT-L/14@336px",
        10.0,
        revision=_MODULE.UNICOM_REVISION,
        checkpoint_sha256=_MODULE.OFFICIAL_CHECKPOINT_SHA256[1],
    )

    _MODULE.load_paired_cub_archives(
        source,
        _sha(source),
        teacher,
        _sha(teacher),
        expected_counts=(4, 4),
        expected_classes=(2, 2),
        expected_dimension=4,
    )

    wrong = tmp_path / "wrong.npz"
    _write_archive(
        wrong,
        tmp_path / "wrong",
        "UNICOM-ViT-B/16",
        0.0,
        revision=_MODULE.UNICOM_REVISION,
        checkpoint_sha256="00" * 32,
    )
    with pytest.raises(ValueError, match="paired metadata"):
        _MODULE.load_paired_cub_archives(
            wrong,
            _sha(wrong),
            teacher,
            _sha(teacher),
            expected_counts=(4, 4),
            expected_classes=(2, 2),
            expected_dimension=4,
        )


def test_int4_symmetric_score_uses_packed_geometry_and_excludes_self() -> None:
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32)
    result = _MODULE.score_int4_symmetric(
        pack_int4_unit_embeddings(values),
        (1, 1, 2, 2),
        candidate_width=2,
        device=torch.device("cpu"),
    )
    assert result["map_at_r"] == 1.0
    assert result["r1"] == 1.0
    assert result["per_query_ap"] == (1.0, 1.0, 1.0, 1.0)


def test_ridge_teacher_control_is_deterministic_finite_and_train_only() -> None:
    source = _unit(12, 6)
    teacher = _unit(12, 7, 100)
    test = _unit(5, 6, 300)

    first = _MODULE.fit_ridge_teacher_control(source, teacher, test, dimensions=4)
    second = _MODULE.fit_ridge_teacher_control(source, teacher, test, dimensions=4)

    assert first.shape == (5, 4)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(torch.linalg.vector_norm(first, dim=1), torch.ones(5))
    changed_test = _MODULE.fit_ridge_teacher_control(source, teacher, test.roll(1, 0), dimensions=4)
    assert not torch.equal(first, changed_test)


def test_fixed_random_rotation_is_orthogonal_deterministic_and_seeded() -> None:
    first = _MODULE.fixed_random_rotation(16, seed=17)
    second = _MODULE.fixed_random_rotation(16, seed=17)
    other = _MODULE.fixed_random_rotation(16, seed=1729)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    torch.testing.assert_close(first @ first.T, torch.eye(16), atol=2e-6, rtol=2e-6)


def test_deployed_retrieval_uses_exact_packed_geometry_and_ordinal_ties() -> None:
    model = _MODULE.RelationalLinearEncoder(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    )
    gallery = pack_int4_unit_embeddings(
        torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        )
    )

    result = _MODULE.retrieve_int4_candidates(
        model,
        gallery,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        exclude_index=1,
        candidate_width=2,
    )

    assert result.tolist() == [[0, 2]]


def test_cub_cli_requires_explicit_local_outputs() -> None:
    args = _MODULE.parse_args(
        [
            "--source-embeddings",
            "/source.npz",
            "--source-embeddings-sha256",
            "12" * 32,
            "--teacher-embeddings",
            "/teacher.npz",
            "--teacher-embeddings-sha256",
            "34" * 32,
            "--source-commit",
            "56" * 20,
            "--output",
            "/result.json",
            "--model-output",
            "/model.bin",
            "--latency-output",
            "/latency.json",
            "--execute-relational-int4",
        ]
    )
    assert args.execute_relational_int4 is True
    assert args.source_commit == "56" * 20
    with pytest.raises(SystemExit):
        _MODULE.parse_args(["--source-embeddings", "/source.npz"])


def test_main_authenticates_then_exclusively_publishes_three_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "verify_source_commit", lambda _commit: None)
    pair = {"authenticated": True}
    quality_raw = b'{"claim_eligible":false,"schema":"fixture"}\n'
    model_raw = b"SFORA-RL1fixture"
    latency_raw = b'{"claim_eligible":false,"schema":"latency"}\n'
    monkeypatch.setattr(_MODULE, "load_paired_cub_archives", lambda *_args: pair)
    monkeypatch.setattr(
        _MODULE,
        "run_sealed_cub_evaluation",
        lambda value, *, source_commit: (
            (quality_raw, model_raw, latency_raw)
            if value is pair and source_commit == "56" * 20
            else None
        ),
    )
    quality = tmp_path / "quality.json"
    model = tmp_path / "model.bin"
    latency = tmp_path / "latency.json"
    arguments = [
        "--source-embeddings",
        "/source.npz",
        "--source-embeddings-sha256",
        "12" * 32,
        "--teacher-embeddings",
        "/teacher.npz",
        "--teacher-embeddings-sha256",
        "34" * 32,
        "--source-commit",
        "56" * 20,
        "--output",
        str(quality),
        "--model-output",
        str(model),
        "--latency-output",
        str(latency),
        "--execute-relational-int4",
    ]

    assert _MODULE.main(arguments) == 0
    assert json.loads(quality.read_bytes())["schema"] == "fixture"
    assert model.read_bytes() == model_raw
    assert json.loads(latency.read_bytes())["schema"] == "latency"
    assert not tuple(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        _MODULE.main(arguments)


def test_main_removes_every_partial_and_final_when_publication_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "verify_source_commit", lambda _commit: None)
    monkeypatch.setattr(_MODULE, "load_paired_cub_archives", lambda *_args: {})
    monkeypatch.setattr(
        _MODULE,
        "run_sealed_cub_evaluation",
        lambda _pair, *, source_commit: (
            (b"quality\n", b"model\n", b"latency\n") if source_commit == "56" * 20 else None
        ),
    )
    real_link = os.link
    calls = 0

    def fail_second_link(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture publication failure")
        real_link(source, destination)

    monkeypatch.setattr(_MODULE.os, "link", fail_second_link)
    outputs = tuple(tmp_path / name for name in ("quality.json", "model.bin", "latency.json"))
    arguments = [
        "--source-embeddings",
        "/source.npz",
        "--source-embeddings-sha256",
        "12" * 32,
        "--teacher-embeddings",
        "/teacher.npz",
        "--teacher-embeddings-sha256",
        "34" * 32,
        "--source-commit",
        "56" * 20,
        "--output",
        str(outputs[0]),
        "--model-output",
        str(outputs[1]),
        "--latency-output",
        str(outputs[2]),
        "--execute-relational-int4",
    ]

    with pytest.raises(OSError, match="fixture publication failure"):
        _MODULE.main(arguments)
    assert all(not path.exists() for path in outputs)
    assert not tuple(tmp_path.glob("*.partial"))


def test_main_never_deletes_concurrent_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "verify_source_commit", lambda _commit: None)
    monkeypatch.setattr(_MODULE, "load_paired_cub_archives", lambda *_args: {})
    monkeypatch.setattr(
        _MODULE,
        "run_sealed_cub_evaluation",
        lambda _pair, *, source_commit: (b"quality\n", b"model\n", b"latency\n"),
    )
    outputs = tuple(tmp_path / name for name in ("quality.json", "model.bin", "latency.json"))
    target = outputs[0].with_name(outputs[0].name + ".partial")
    real_open = Path.open

    def racing_open(path: Path, mode: str = "r", *args: object, **kwargs: object):
        if path == target and mode == "xb" and not target.exists():
            target.write_bytes(b"other-run")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    with pytest.raises(FileExistsError):
        _MODULE.main(
            [
                "--source-embeddings",
                "/source.npz",
                "--source-embeddings-sha256",
                "12" * 32,
                "--teacher-embeddings",
                "/teacher.npz",
                "--teacher-embeddings-sha256",
                "34" * 32,
                "--source-commit",
                "56" * 20,
                "--output",
                str(outputs[0]),
                "--model-output",
                str(outputs[1]),
                "--latency-output",
                str(outputs[2]),
                "--execute-relational-int4",
            ]
        )
    assert target.read_bytes() == b"other-run"
    assert all(not output.exists() for output in outputs)


def test_main_rejects_duplicate_outputs_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def load(*_args: object) -> object:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(_MODULE, "load_paired_cub_archives", load)
    shared = tmp_path / "shared.bin"
    arguments = [
        "--source-embeddings",
        "/source.npz",
        "--source-embeddings-sha256",
        "12" * 32,
        "--teacher-embeddings",
        "/teacher.npz",
        "--teacher-embeddings-sha256",
        "34" * 32,
        "--source-commit",
        "56" * 20,
        "--output",
        str(shared),
        "--model-output",
        str(shared),
        "--latency-output",
        str(tmp_path / "latency.json"),
        "--execute-relational-int4",
    ]

    with pytest.raises(ValueError, match="distinct"):
        _MODULE.main(arguments)
    assert called is False

    arguments[arguments.index("--model-output") + 1] = str(
        shared.with_name(shared.name + ".partial")
    )
    with pytest.raises(ValueError, match="distinct"):
        _MODULE.main(arguments)
    assert called is False


def test_sealed_evaluation_runs_real_small_pipeline_and_binds_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE, "RELATIONAL_DIMENSIONS", 4)
    monkeypatch.setattr(_MODULE, "RIDGE_DIMENSIONS", 4)
    monkeypatch.setattr(_MODULE, "PCA_INT8_DIMENSIONS", 2)
    monkeypatch.setattr(_MODULE, "CANDIDATE_WIDTH", 2)
    monkeypatch.setattr(_MODULE, "SEEDS", (17,))
    monkeypatch.setattr(_MODULE, "BATCH", 4)
    monkeypatch.setattr(_MODULE, "EPOCHS", 1)
    monkeypatch.setattr(_MODULE, "BOOTSTRAP_SAMPLES", 10)
    monkeypatch.setattr(_MODULE, "BOOTSTRAP_COMPARISONS", 6)
    monkeypatch.setattr(_MODULE, "LATENCY_WARMUP_PAIRS", 1)
    monkeypatch.setattr(_MODULE, "LATENCY_MEASURED_PAIRS", 2)
    metadata = {
        "checkpoint_sha256": "cd" * 32,
        "dataset_archive_sha256": _EXPORT.CUB_ARCHIVE_SHA256,
        "dataset_content_sha256": "ef" * 32,
        "cub_exporter_source_sha256": _sha(_ROOT / "scripts/export_unicom_cub_embeddings.py"),
        "sop_exporter_source_sha256": _sha(_ROOT / "scripts/export_unicom_sop_embeddings.py"),
        "model_identifier": "fixture",
        "model_revision": "ab" * 20,
        "ordered_record_sha256": "12" * 32,
    }
    pair = {
        "source_metadata": metadata,
        "teacher_metadata": {**metadata, "model_identifier": "teacher"},
        "source_archive_sha256": "34" * 32,
        "teacher_archive_sha256": "56" * 32,
        "source_train": _unit(8, 6),
        "teacher_train": _unit(8, 7, 100),
        "source_test": _unit(6, 6, 200),
        "teacher_test": _unit(6, 7, 300),
        "train_labels": (1, 1, 2, 2, 3, 3, 4, 4),
        "test_labels": (5, 5, 6, 6, 7, 7),
    }

    quality_raw, model_raw, latency_raw = _MODULE.run_sealed_cub_evaluation(
        pair, source_commit="78" * 20
    )
    quality = json.loads(quality_raw)
    latency = json.loads(latency_raw)

    assert quality_raw.endswith(b"\n") and not quality_raw.endswith(b"\n\n")
    assert latency_raw.endswith(b"\n") and not latency_raw.endswith(b"\n\n")
    assert model_raw.startswith(b"SFORA-RL1")
    assert quality["evidence_authority"]["source_embeddings_sha256"] == "34" * 32
    assert quality["optimizer_steps"] == 2
    assert quality["examples_processed"] == 8
    assert quality["cpu_quality_code_parity"] is True
    assert quality["training_device"] in {"cpu", "cuda"}
    assert quality["persistent_bytes_per_item"] == 4
    assert "pca64_int8" in quality["controls"]
    assert "ridge_teacher128_int4" in quality["controls"]
    assert "ridge_teacher64_int8" not in quality["controls"]
    assert latency["selection_policy"] == "score-descending-ordinal-ascending"
