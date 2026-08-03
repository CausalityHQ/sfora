from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "analyze_sop_official_structure",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_sop_official_structure.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_analyze_reports_hierarchy_of_nearest_negative() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.7, 0.7, 0.0],
            [0.69, 0.71, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.99, 0.01, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([10, 10, 20, 20, 30, 30])
    superclasses = np.asarray([1, 1, 1, 1, 2, 2])

    result = _MODULE.analyze(embeddings, labels, superclasses, chunk_size=2)

    assert result["training_leave_one_out_r1"] == 1.0
    assert result["nearest_negative_same_superclass_fraction"] == 4 / 6
    assert result["classes"] == 3
    assert result["superclasses"] == 2


def test_load_superclasses_reads_product_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "Ebay_train.txt"
    metadata.write_text(
        "image_id class_id super_class_id path\n"
        "1 1 4 mug_final/123_0.JPG\n"
        "2 1 4 mug_final/123_1.JPG\n"
        "3 2 7 lamp_final/987_0.JPG\n",
        encoding="utf-8",
    )
    assert _MODULE.load_superclasses(metadata) == {123: 4, 987: 7}
