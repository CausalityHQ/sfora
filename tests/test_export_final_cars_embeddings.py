"""Cars196 final-state PFML exporter tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "export_final_cars_embeddings",
    Path(__file__).resolve().parents[1] / "scripts" / "export_final_cars_embeddings.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _example(label: int, suffix: str, root: Path) -> SimpleNamespace:
    path = root / f"{suffix}.jpg"
    path.write_bytes(suffix.encode())
    return SimpleNamespace(label=label, example_id=suffix, image=str(path))


class _DecodedImage:
    mode = "RGB"
    size = (2, 1)

    def __init__(self, content: bytes) -> None:
        self._content = content

    def tobytes(self) -> bytes:
        return self._content


def test_independent_leave_one_out_recall_excludes_self() -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1])
    assert _module.independent_leave_one_out_recall_at_1(
        embeddings, labels, chunk_size=1
    ) == pytest.approx(1.0)


def test_independent_leave_one_out_recall_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _module.independent_leave_one_out_recall_at_1(
            np.asarray([[1.0, 0.0], [np.nan, 1.0]]), np.asarray([0, 0])
        )


def test_partition_verifier_accepts_disjoint_official_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)}
    )
    train = [_example(0, "train-a", tmp_path), _example(0, "train-b", tmp_path)]
    test = [_example(1, "test-a", tmp_path), _example(1, "test-b", tmp_path)]
    audit = _module.verify_official_partition(train, test)
    assert audit["train_test_identity_overlap"] == 0
    assert audit["train_test_example_id_overlap"] == 0


def test_partition_verifier_rejects_identity_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)}
    )
    train = [_example(0, "train-a", tmp_path), _example(0, "train-b", tmp_path)]
    test = [_example(0, "test-a", tmp_path), _example(0, "test-b", tmp_path)]
    with pytest.raises(ValueError, match="identities overlap"):
        _module.verify_official_partition(train, test)


def test_partition_verifier_hashes_decoded_images_and_rejects_cross_split_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        _module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)}
    )
    train = [
        SimpleNamespace(label=0, example_id="train-a", image=_DecodedImage(b"a")),
        SimpleNamespace(label=0, example_id="train-b", image=_DecodedImage(b"shared")),
    ]
    test = [
        SimpleNamespace(label=1, example_id="test-a", image=_DecodedImage(b"b")),
        SimpleNamespace(label=1, example_id="test-b", image=_DecodedImage(b"shared")),
    ]
    with pytest.raises(ValueError, match="decoded image content overlaps"):
        _module.verify_official_partition(train, test)


def test_reported_final_recall_requires_one_completed_pfml_method() -> None:
    report = {
        "methods": {
            "pfml_end_to_end:resnet50": {
                "objective": "pfml",
                "executed_train_steps": 16_200,
                "recall_at_1": 0.91,
            }
        }
    }
    assert _module.reported_final_recall_at_1(report, 16_200) == 0.91

    report["methods"]["pfml_end_to_end:resnet50"]["executed_train_steps"] = 16_199
    with pytest.raises(ValueError, match="resolved final training step"):
        _module.reported_final_recall_at_1(report, 16_200)
