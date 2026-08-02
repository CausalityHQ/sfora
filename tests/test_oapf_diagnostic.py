from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "diagnose_oapf",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_oapf.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_report_objective_uses_payload_not_model_qualified_key() -> None:
    report = {
        "methods": {
            "proxy_anchor_end_to_end:bn_inception": {"objective": "proxy_anchor"}
        }
    }
    assert _module.report_has_objective(report, "proxy_anchor")
    assert not _module.report_has_objective(report, "hist")


def test_train_loader_excludes_query_and_gallery_identities(tmp_path: Path) -> None:
    root = tmp_path / "inshop"
    (root / "Eval").mkdir(parents=True)
    (root / "Img").mkdir()
    partition = """3
image_name item_id evaluation_status
train_a.jpg train_item train
query.jpg evaluation_item query
gallery.jpg evaluation_item gallery
"""
    (root / "Eval" / "list_eval_partition.txt").write_text(partition, encoding="utf-8")
    (root / "Img" / "train_a.jpg").touch()

    records = _module.load_inshop_train_only(root)

    assert [record.relative_path for record in records] == ["train_a.jpg"]
    assert [record.item_id for record in records] == ["train_item"]
    # Labels match the official loader's sort over manifest identities, even
    # though no evaluation image is opened.
    assert [record.label for record in records] == [1]


def test_train_loader_rejects_partition_count_or_row_drift(tmp_path: Path) -> None:
    root = tmp_path / "inshop"
    (root / "Eval").mkdir(parents=True)
    (root / "Img").mkdir()
    (root / "Eval" / "list_eval_partition.txt").write_text(
        "2\nheader\ntrain_a.jpg item train\n", encoding="utf-8"
    )
    (root / "Img" / "train_a.jpg").touch()

    try:
        _module.load_inshop_train_only(root)
    except ValueError as error:
        assert "declares 2 rows" in str(error)
    else:
        raise AssertionError("partition count drift must be rejected")


def test_digest_binding_rejects_a_different_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"registered")
    expected = hashlib.sha256(b"registered").hexdigest()
    assert _module.require_sha256(artifact, expected, "artifact") == expected

    artifact.write_bytes(b"changed")
    try:
        _module.require_sha256(artifact, expected, "artifact")
    except ValueError as error:
        assert "SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("a changed provenance artifact must be rejected")


def test_pair_features_use_registered_min_max_endpoint_layout() -> None:
    canonical = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    left = np.asarray([0], dtype=np.int64)
    right = np.asarray([1], dtype=np.int64)
    controls, radii = _module.pair_features(
        canonical,
        np.asarray([2.0, 4.0]),
        np.asarray([1.0, np.e]),
        np.asarray([-0.2, 0.3]),
        np.asarray([0.1, 0.4]),
        np.asarray([0.2, 0.5]),
        np.asarray([0.1, 0.3]),
        np.asarray([0.5, 2.0]),
        left,
        right,
    )

    assert controls.shape == (1, 13)
    assert np.allclose(
        controls[0],
        [
            np.sqrt(2.0),
            2.0,
            4.0,
            0.0,
            1.0,
            -0.2,
            0.3,
            0.1,
            0.4,
            0.2,
            0.5,
            0.1,
            0.3,
        ],
    )
    assert np.allclose(radii[0], [np.log(0.5), np.log(2.0)])


def test_weighted_hedges_g_has_registered_positive_direction() -> None:
    high = np.asarray([0.8, 1.0, 1.2], dtype=np.float64)
    low = np.asarray([-0.2, 0.0, 0.2], dtype=np.float64)
    weights = np.ones(3, dtype=np.float64)

    assert _module.weighted_hedges_g(high, weights, low, weights) > 0.0
