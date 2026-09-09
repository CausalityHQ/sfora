from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sfora.nested_rank_protocol import ordered_training_records_sha256

SCRIPT = Path(__file__).parents[1] / "scripts" / "build_sop_teacher_anchored_source_snapshot.py"
SPEC = importlib.util.spec_from_file_location("build_sop_teacher_anchored_source_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
sys.modules[SPEC.name] = SUBJECT
try:
    SPEC.loader.exec_module(SUBJECT)
finally:
    sys.modules.pop(SPEC.name, None)
    sys.path.pop(0)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _teacher_snapshot(path: Path) -> tuple[str, dict[str, np.ndarray]]:
    arrays = {
        "train_embeddings": np.ones((4, 3), dtype=np.float32),
        "train_labels": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "train_image_ids": np.asarray([11, 12, 13, 14], dtype=np.int64),
        "train_relative_paths": np.asarray(
            ["train/11.png", "train/12.png", "train/13.png", "train/14.png"]
        ),
    }
    metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": "1" * 64,
        "model_identifier": "UNICOM-ViT-L/14@336px",
        "model_revision": "2" * 40,
        "checkpoint_sha256": "3" * 64,
        "embedding_dimension": 3,
        "train_rows": 4,
        "train_classes": 2,
        "train_array_sha256": {
            name: _sha256(value.tobytes(order="C")) for name, value in arrays.items()
        },
        "excluded_test_array_sha256": {
            "test_embeddings": "4" * 64,
            "test_labels": "5" * 64,
            "test_image_ids": "6" * 64,
            "test_relative_paths": "7" * 64,
        },
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(value) for value in arrays["train_relative_paths"]),
        ),
        "transform": "fixture teacher transform",
    }
    buffer = io.BytesIO()
    np.savez(
        buffer,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    path.write_bytes(buffer.getvalue())
    return _sha256(buffer.getvalue()), arrays


class _FixtureEncoder(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        means = batch.float().mean(dim=(1, 2, 3))
        return torch.stack((means, means + 1.0, means + 2.0), dim=1)


class _RecordingEncoder(_FixtureEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(batch.shape[0])
        return super().forward(batch)


def _source_model(encoder: torch.nn.Module | None = None) -> object:
    return SUBJECT.SourceSnapshotModel(
        encoder=_FixtureEncoder() if encoder is None else encoder,
        transform=lambda image: torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        ).permute(2, 0, 1),
        model_identifier="UNICOM-ViT-B/16",
        model_revision="2" * 40,
        checkpoint_sha256="8" * 64,
        source_archive_sha256="9" * 64,
        excluded_test_embeddings_sha256="a" * 64,
        transform_name="fixture source transform",
    )


def _image_tree(root: Path) -> None:
    train = root / "train"
    train.mkdir()
    for index in range(11, 15):
        Image.new("RGB", (2, 2), color=(index, index, index)).save(train / f"{index}.png")


def test_build_source_snapshot_reads_only_teacher_named_training_images(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.npz"
    teacher_sha256, arrays = _teacher_snapshot(teacher)
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    output = tmp_path / "source.npz"

    expected_embeddings = np.asarray(
        [[11.0, 12.0, 13.0], [12.0, 13.0, 14.0], [13.0, 14.0, 15.0], [14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    result = SUBJECT.build_source_snapshot(
        teacher_snapshot=teacher,
        teacher_snapshot_sha256=teacher_sha256,
        image_root=image_root,
        output=output,
        source_model=_source_model(),
        expected_train_embeddings_sha256=_sha256(expected_embeddings.tobytes()),
        batch_size=2,
    )

    assert result["train_rows"] == 4
    assert result["train_embeddings_sha256"] == _sha256(expected_embeddings.tobytes())
    with np.load(output, allow_pickle=False) as snapshot:
        assert set(snapshot.files) == {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "train_image_ids",
            "train_relative_paths",
        }
        assert np.array_equal(snapshot["train_embeddings"], expected_embeddings)
        assert np.array_equal(snapshot["train_labels"], arrays["train_labels"])
        assert np.array_equal(snapshot["train_image_ids"], arrays["train_image_ids"])
        assert np.array_equal(snapshot["train_relative_paths"], arrays["train_relative_paths"])
        metadata = json.loads(str(snapshot["metadata_json"].item()))
    assert metadata["model_identifier"] == "UNICOM-ViT-B/16"
    assert metadata["source_archive_sha256"] == "9" * 64
    assert metadata["excluded_test_array_sha256"] == {
        "test_embeddings": "a" * 64,
        "test_labels": "5" * 64,
        "test_image_ids": "6" * 64,
        "test_relative_paths": "7" * 64,
    }


def test_build_source_snapshot_pads_tail_with_authorized_training_rows(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.npz"
    teacher_sha256, _arrays = _teacher_snapshot(teacher)
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    encoder = _RecordingEncoder()
    expected = np.asarray(
        [[11.0, 12.0, 13.0], [12.0, 13.0, 14.0], [13.0, 14.0, 15.0], [14.0, 15.0, 16.0]],
        dtype=np.float32,
    )

    SUBJECT.build_source_snapshot(
        teacher_snapshot=teacher,
        teacher_snapshot_sha256=teacher_sha256,
        image_root=image_root,
        output=tmp_path / "source.npz",
        source_model=_source_model(encoder),
        expected_train_embeddings_sha256=_sha256(expected.tobytes()),
        batch_size=3,
    )

    assert encoder.batch_sizes == [3, 3]


def test_build_source_snapshot_rejects_digest_output_and_image_authority(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.npz"
    teacher_sha256, _arrays = _teacher_snapshot(teacher)
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    kwargs = {
        "teacher_snapshot": teacher,
        "teacher_snapshot_sha256": teacher_sha256,
        "image_root": image_root,
        "source_model": _source_model(),
        "expected_train_embeddings_sha256": "b" * 64,
        "batch_size": 2,
    }
    with pytest.raises(ValueError, match="teacher snapshot"):
        SUBJECT.build_source_snapshot(
            output=tmp_path / "bad-digest.npz",
            **{
                **kwargs,
                "teacher_snapshot_sha256": "0" * 64,
            },
        )
    with pytest.raises(ValueError, match="embedding digest"):
        SUBJECT.build_source_snapshot(output=tmp_path / "bad-embedding.npz", **kwargs)
    assert not (tmp_path / "bad-embedding.npz").exists()

    occupied = tmp_path / "occupied.npz"
    occupied.write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        SUBJECT.build_source_snapshot(output=occupied, **kwargs)

    (image_root / "train" / "12.png").unlink()
    (image_root / "train" / "12.png").symlink_to(image_root / "train" / "11.png")
    with pytest.raises(ValueError, match="training image"):
        SUBJECT.build_source_snapshot(output=tmp_path / "symlink.npz", **kwargs)


def test_build_source_snapshot_does_not_accept_or_delete_replaced_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher = tmp_path / "teacher.npz"
    teacher_sha256, _arrays = _teacher_snapshot(teacher)
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    output = tmp_path / "source.npz"
    displaced = tmp_path / "displaced.npz"
    expected = np.asarray(
        [[11.0, 12.0, 13.0], [12.0, 13.0, 14.0], [13.0, 14.0, 15.0], [14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    publish = SUBJECT.publish_large_writer_noreplace

    def replace_after_publish(*args: object, **kwargs: object) -> object:
        retained = publish(*args, **kwargs)
        output.rename(displaced)
        output.write_bytes(b"foreign-output")
        return retained

    monkeypatch.setattr(SUBJECT, "publish_large_writer_noreplace", replace_after_publish)
    with pytest.raises(ValueError, match="publication ownership"):
        SUBJECT.build_source_snapshot(
            teacher_snapshot=teacher,
            teacher_snapshot_sha256=teacher_sha256,
            image_root=image_root,
            output=output,
            source_model=_source_model(),
            expected_train_embeddings_sha256=_sha256(expected.tobytes()),
            batch_size=2,
        )
    assert output.read_bytes() == b"foreign-output"
    assert displaced.is_file()


@pytest.mark.parametrize("corruption", ["overwrite", "append"])
def test_build_source_snapshot_rejects_same_inode_corruption_after_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    teacher = tmp_path / "teacher.npz"
    teacher_sha256, _arrays = _teacher_snapshot(teacher)
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    output = tmp_path / "source.npz"
    expected = np.asarray(
        [[11.0, 12.0, 13.0], [12.0, 13.0, 14.0], [13.0, 14.0, 15.0], [14.0, 15.0, 16.0]],
        dtype=np.float32,
    )
    publish = SUBJECT.publish_large_writer_noreplace

    def corrupt_after_publish(*args: object, **kwargs: object) -> object:
        retained = publish(*args, **kwargs)
        with output.open("r+b") as stream:
            stream.seek(
                0 if corruption == "overwrite" else 0,
                os.SEEK_SET if corruption == "overwrite" else os.SEEK_END,
            )
            stream.write(b"corrupt!")
            stream.flush()
        return retained

    monkeypatch.setattr(SUBJECT, "publish_large_writer_noreplace", corrupt_after_publish)
    with pytest.raises(ValueError, match="publication content"):
        SUBJECT.build_source_snapshot(
            teacher_snapshot=teacher,
            teacher_snapshot_sha256=teacher_sha256,
            image_root=image_root,
            output=output,
            source_model=_source_model(),
            expected_train_embeddings_sha256=_sha256(expected.tobytes()),
            batch_size=2,
        )


def test_training_image_walk_uses_retained_no_follow_directory_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    _image_tree(image_root)
    calls: list[tuple[object, int | None]] = []
    open_file = SUBJECT.os.open

    def observe_open(
        path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        calls.append((path, dir_fd))
        if dir_fd is None:
            return open_file(path, flags, mode)
        return open_file(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(SUBJECT.os, "open", observe_open)
    payload, size, digest = SUBJECT._read_training_image(image_root, "train/11.png")

    assert size == len(payload)
    assert digest == hashlib.sha256(payload).digest()
    assert calls[0] == (image_root, None)
    assert calls[1][0] == "train" and calls[1][1] is not None
    assert calls[2][0] == "11.png" and calls[2][1] is not None


def test_source_snapshot_cli_is_explicit_local_only(tmp_path: Path) -> None:
    values = [
        "--teacher-snapshot",
        str((tmp_path / "teacher.npz").resolve()),
        "--teacher-snapshot-sha256",
        "b0da9f6097646ffad78c21c84970751ae9a7da003e0eb2979e28f72009785264",
        "--image-root",
        str((tmp_path / "images").resolve()),
        "--unicom-checkout",
        str((tmp_path / "unicom").resolve()),
        "--checkpoint",
        str((tmp_path / "FP16-ViT-B-16.pt").resolve()),
        "--output",
        str((tmp_path / "source.npz").resolve()),
        "--execute-build-source-snapshot",
    ]
    args = SUBJECT.parse_args(values)
    assert args.batch_size == 64
    with pytest.raises(SystemExit):
        SUBJECT.parse_args(
            [
                *values[:3],
                "0" * 64,
                *values[4:],
            ]
        )
    for forbidden in ("--test-root", "--test-snapshot", "--url", "--s3", "--class-names"):
        with pytest.raises(SystemExit):
            SUBJECT.parse_args([*values, forbidden, "value"])
