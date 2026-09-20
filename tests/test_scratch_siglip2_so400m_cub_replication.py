from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "_scratch_siglip2_so400m_cub_replication.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "scratch_siglip2_so400m_cub_replication", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_index(path: Path, rows: list[tuple[int, object]]) -> None:
    path.write_text("".join(f"{index} {value}\n" for index, value in rows))


def test_cub_protocol_uses_frozen_classes_official_split_and_image_order(
    tmp_path: Path,
) -> None:
    subject = _load_subject()
    root = tmp_path / "CUB_200_2011"
    (root / "images").mkdir(parents=True)
    classes = {
        1: "001.Alpha",
        2: "002.Beta",
        3: "003.Gamma",
        4: "004.Delta",
    }
    images = [
        (1, 1, 1, "001.Alpha/a.jpg"),
        (2, 1, 0, "001.Alpha/b.jpg"),
        (3, 2, 0, "002.Beta/c.jpg"),
        (4, 2, 1, "002.Beta/d.jpg"),
        (5, 3, 1, "003.Gamma/e.jpg"),
        (6, 3, 1, "003.Gamma/f.jpg"),
        (7, 4, 0, "004.Delta/g.jpg"),
        (8, 4, 0, "004.Delta/h.jpg"),
    ]
    _write_index(root / "classes.txt", list(classes.items()))
    _write_index(root / "images.txt", [(i, path) for i, _, _, path in images])
    _write_index(root / "image_class_labels.txt", [(i, label) for i, label, _, _ in images])
    _write_index(root / "train_test_split.txt", [(i, split) for i, _, split, _ in images])
    for _, _, _, relative in images:
        path = root / "images" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    fit_paths, fit_labels, evaluation_paths, evaluation_labels = (
        subject.select_cub_protocol_rows(
            root,
            {
                "fit_classes": ["003.Gamma", "001.Alpha"],
                "evaluation_classes": ["004.Delta", "002.Beta"],
                "protocol": "sha256-name-ordered-100-fit-train-100-eval-test-classes",
            },
            expected_counts=(3, 3),
            expected_classes=(2, 2),
        )
    )

    assert [path.name for path in fit_paths] == ["a.jpg", "e.jpg", "f.jpg"]
    assert np.array_equal(fit_labels, np.asarray([0, 1, 1], dtype=np.int64))
    assert [path.name for path in evaluation_paths] == ["c.jpg", "g.jpg", "h.jpg"]
    assert np.array_equal(
        evaluation_labels, np.asarray([0, 1, 1], dtype=np.int64)
    )


def test_cub_gate_is_the_unchanged_cars_gate() -> None:
    subject = _load_subject()

    decision = subject.classify_candidate(
        {"map_at_r": 0.72, "recall_at_1": 0.91},
        {"map_at_r": 0.726, "recall_at_1": 0.911},
        {"lower": 0.001, "median": 0.006, "upper": 0.011},
    )

    assert decision == {
        "map_at_r_delta": 0.006000000000000005,
        "recall_at_1_delta": 0.0010000000000000009,
        "minimum_map_at_r_delta": 0.005,
        "require_positive_interval_lower": True,
        "minimum_recall_at_1_delta": 0.0,
        "passed": True,
    }
