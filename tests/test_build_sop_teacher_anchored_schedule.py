from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from sfora.teacher_anchored_schedule_io import parse_teacher_anchored_schedule_for_inputs


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_sop_teacher_anchored_schedule.py"
    spec = importlib.util.spec_from_file_location("build_sop_teacher_anchored_schedule", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()
TeacherAnchoredTrainPair = SUBJECT.TeacherAnchoredTrainPair


def _pair(*, dimensions: int = 128) -> TeacherAnchoredTrainPair:
    generator = np.random.Generator(np.random.PCG64(4401))
    source = generator.normal(size=(1280, dimensions)).astype(np.float32)
    teacher = generator.normal(size=(1280, dimensions)).astype(np.float32)
    source /= np.linalg.norm(source.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    teacher /= np.linalg.norm(teacher.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    return TeacherAnchoredTrainPair(
        source_embeddings=np.ascontiguousarray(source),
        teacher_embeddings=np.ascontiguousarray(teacher),
        labels=np.repeat(np.arange(20, dtype=np.int64), 64),
        image_ids=np.arange(1000, 2280, dtype=np.int64),
        relative_paths=tuple(f"class/image-{row}.jpg" for row in range(1280)),
        source_metadata={"model_revision": "3" * 40},
        teacher_metadata={"model_revision": "3" * 40},
        source_snapshot_sha256="1" * 64,
        teacher_snapshot_sha256="2" * 64,
    )


def test_schedule_builder_cli_is_explicit_local_and_fail_closed(tmp_path: Path) -> None:
    argv = (
        "--source-snapshot",
        str((tmp_path / "source.npz").resolve()),
        "--source-snapshot-sha256",
        "1" * 64,
        "--teacher-snapshot",
        str((tmp_path / "teacher.npz").resolve()),
        "--teacher-snapshot-sha256",
        "2" * 64,
        "--source-revision",
        "3" * 40,
        "--seed",
        "17",
        "--output",
        str((tmp_path / "schedule.bin").resolve()),
        "--execute-build-schedule",
    )
    arguments = SUBJECT.parse_schedule_build_args(argv)
    assert arguments.seed == 17
    assert arguments.output == (tmp_path / "schedule.bin").resolve()
    for forbidden in ("--dataset", "--labels", "--arm", "--s3-uri", "--resume"):
        with pytest.raises(SystemExit):
            SUBJECT.parse_schedule_build_args((*argv, forbidden, "x"))


def test_builder_seals_fitting_only_codes_and_canonical_receipt() -> None:
    pair = _pair()
    artifact = SUBJECT.build_sop_teacher_anchored_schedule(
        pair,
        seed=17,
        source_revision="3" * 40,
    )
    receipt = json.loads(artifact.receipt_bytes)
    assert artifact.receipt_bytes.endswith(b"\n")
    assert artifact.receipt_bytes == SUBJECT.canonical_schedule_build_receipt_bytes(artifact)
    assert receipt["schema"] == "sfora-teacher-anchored-schedule-build-result-v1"
    assert receipt["claim_eligible"] is False
    assert receipt["schedule_sha256"] == hashlib.sha256(artifact.payload).hexdigest()
    assert receipt["schedule_bytes"] == len(artifact.payload)
    assert receipt["fitting_row_count"] < pair.labels.size
    assert receipt["blas_threads"] == 2
    assert receipt["cpu_threads"] == 2
    assert receipt["numpy_version"] == np.__version__
    assert receipt["torch_version"] == torch.__version__
    assert torch.get_num_threads() == 2
    assert set(receipt).isdisjoint({"dataset", "labels", "arm"})

    fitting_rows = artifact.split.fitting_rows
    teacher_codes = artifact.teacher_codes
    sealed = parse_teacher_anchored_schedule_for_inputs(
        artifact.payload,
        expected_sha256=receipt["schedule_sha256"],
        teacher_codes=teacher_codes,
        sample_ids=tuple(int(pair.image_ids[row]) for row in fitting_rows),
        source_revision="3" * 40,
        source_snapshot_sha256="1" * 64,
        teacher_snapshot_sha256="2" * 64,
        split_sha256=artifact.split.sha256,
        seed=17,
    )
    assert sealed.binding.fitting_row_count == len(fitting_rows)

    with pytest.raises(ValueError, match="schedule build result differs"):
        SUBJECT.canonical_schedule_build_receipt_bytes(
            artifact._replace(teacher_codes=np.zeros_like(artifact.teacher_codes))
        )
    with pytest.raises(ValueError, match="schedule build result differs"):
        SUBJECT.canonical_schedule_build_receipt_bytes(
            artifact._replace(teacher_pca_sha256="9" * 64)
        )


def test_builder_matches_768d_initializer_and_excludes_validation_rows() -> None:
    import train_sop_teacher_anchored_distillation as trainer

    pair = _pair(dimensions=768)
    torch.set_num_threads(1)
    artifact = SUBJECT.build_sop_teacher_anchored_schedule(pair, seed=17, source_revision="3" * 40)
    fitting = np.asarray(artifact.split.fitting_rows, dtype=np.int64)
    initialization = trainer.initialize_teacher_anchored_student(
        torch.from_numpy(np.ascontiguousarray(pair.source_embeddings[fitting])),
        torch.from_numpy(np.ascontiguousarray(pair.teacher_embeddings[fitting])),
        expected_teacher_pca_sha256=artifact.teacher_pca_sha256,
    )
    expected_codes = (
        initialization.projection.apply_teacher(
            trainer._normalize_snapshot_rows(
                torch.from_numpy(np.ascontiguousarray(pair.teacher_embeddings[fitting]))
            )
        )
        .detach()
        .numpy()
    )
    np.testing.assert_array_equal(artifact.teacher_codes, expected_codes)

    changed_teacher = pair.teacher_embeddings.copy()
    changed_teacher[np.asarray(artifact.split.validation_rows, dtype=np.int64)] *= -1.0
    torch.set_num_threads(4)
    changed = SUBJECT.build_sop_teacher_anchored_schedule(
        replace(pair, teacher_embeddings=changed_teacher),
        seed=17,
        source_revision="3" * 40,
    )
    assert changed.teacher_pca_sha256 == artifact.teacher_pca_sha256
    np.testing.assert_array_equal(changed.teacher_codes, artifact.teacher_codes)
    assert changed.payload == artifact.payload


def test_builder_publishes_once_without_overwrite(tmp_path: Path) -> None:
    artifact = SUBJECT.build_sop_teacher_anchored_schedule(
        _pair(), seed=17, source_revision="3" * 40
    )
    output = tmp_path / "schedule.bin"
    SUBJECT.publish_sop_teacher_anchored_schedule(output, artifact)
    assert output.read_bytes() == artifact.payload
    with pytest.raises(FileExistsError):
        SUBJECT.publish_sop_teacher_anchored_schedule(output, artifact)
    assert output.read_bytes() == artifact.payload

    malformed_output = tmp_path / "malformed.bin"
    with pytest.raises(ValueError, match="schedule publication differs"):
        SUBJECT.publish_sop_teacher_anchored_schedule(
            malformed_output, artifact._replace(payload=b"fabricated")
        )
    assert not malformed_output.exists()


def test_main_rejects_existing_output_before_loading_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "schedule.bin"
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        SUBJECT,
        "load_authenticated_train_pair",
        lambda *_args: pytest.fail("snapshots must not be read for an occupied output"),
    )
    result = SUBJECT.main(
        (
            "--source-snapshot",
            str((tmp_path / "source.npz").resolve()),
            "--source-snapshot-sha256",
            "1" * 64,
            "--teacher-snapshot",
            str((tmp_path / "teacher.npz").resolve()),
            "--teacher-snapshot-sha256",
            "2" * 64,
            "--source-revision",
            "3" * 40,
            "--seed",
            "17",
            "--output",
            str(output.resolve()),
            "--execute-build-schedule",
        )
    )
    assert result == 2
    assert output.read_bytes() == b"existing"
