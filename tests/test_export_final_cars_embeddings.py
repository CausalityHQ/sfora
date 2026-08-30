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
    embeddings = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.asarray([0, 0, 1, 1])
    assert _module.independent_leave_one_out_recall_at_1(
        embeddings, labels, chunk_size=1
    ) == pytest.approx(1.0)


def test_independent_recall_preserves_exported_l2_near_ties() -> None:
    # These vectors differ from unit norm only at float32-normalization scale.
    # Exported-coordinate squared L2 gives 2/8 correct, while normalizing again
    # in float64 changes the ranking to 3/8. The verifier must reproduce the
    # benchmark metric, not silently substitute idealized cosine geometry.
    embeddings = np.asarray(
        [
            [0.999999740485382, 2.939206875696321e-6, -7.36626053402835e-8, -3.794201970999852e-6],
            [
                0.9999998573013525,
                -3.118063378160621e-6,
                3.763879338067382e-6,
                -5.068315744825331e-6,
            ],
            [
                1.0000002267022614,
                6.328542689090506e-8,
                -1.1832455536608093e-6,
                -1.3681275262267475e-7,
            ],
            [0.9999998456849051, 4.067739915150416e-7, 1.4553019288128912e-6, 2.58763818282212e-7],
            [
                0.9999999537389478,
                -1.1386382926666143e-6,
                -1.0319240403881874e-7,
                1.830239947936335e-6,
            ],
            [
                1.0000000908603262,
                -2.3069892336120913e-6,
                -1.2268802944470123e-6,
                3.729797547454824e-7,
            ],
            [
                1.0000001256918891,
                -1.602603896057274e-6,
                -4.0959457772671215e-6,
                5.824869307477138e-7,
            ],
            [0.9999999321193099, 4.1391602154160095e-6, 2.380637514787195e-6, 8.930047013783728e-7],
        ],
        dtype=np.float64,
    )
    labels = np.repeat(np.arange(4), 2)
    assert _module.independent_leave_one_out_recall_at_1(embeddings, labels) == 0.25


def test_independent_leave_one_out_recall_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _module.independent_leave_one_out_recall_at_1(
            np.asarray([[1.0, 0.0], [np.nan, 1.0]]), np.asarray([0, 0])
        )


def test_partition_verifier_accepts_disjoint_official_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)})
    train = [_example(0, "train-a", tmp_path), _example(0, "train-b", tmp_path)]
    test = [_example(1, "test-a", tmp_path), _example(1, "test-b", tmp_path)]
    audit = _module.verify_official_partition(train, test)
    assert audit["train_test_identity_overlap"] == 0
    assert audit["train_test_example_id_overlap"] == 0


def test_partition_verifier_rejects_identity_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)})
    train = [_example(0, "train-a", tmp_path), _example(0, "train-b", tmp_path)]
    test = [_example(0, "test-a", tmp_path), _example(0, "test-b", tmp_path)]
    with pytest.raises(ValueError, match="identities overlap"):
        _module.verify_official_partition(train, test)


def test_partition_verifier_rejects_singleton_test_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 2)})
    train = [_example(0, "train-a", tmp_path), _example(0, "train-b", tmp_path)]
    test = [_example(1, "test-a", tmp_path), _example(2, "test-b", tmp_path)]
    with pytest.raises(ValueError, match="fewer than two examples"):
        _module.verify_official_partition(train, test)


def test_partition_verifier_hashes_decoded_images_and_rejects_cross_split_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_module, "EXPECTED_CARS_PARTITION", {"train": (2, 1), "test": (2, 1)})
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


def test_reported_final_recall_accepts_exact_proxy_anchor_objective() -> None:
    report = {
        "methods": {
            "proxy_anchor_end_to_end:resnet50": {
                "objective": "proxy_anchor",
                "executed_train_steps": 4_080,
                "recall_at_1": 0.876,
            }
        }
    }
    assert (
        _module.reported_final_recall_at_1(report, 4_080, expected_objective="proxy_anchor")
        == 0.876
    )
    with pytest.raises(ValueError, match="not proxy_anchor"):
        _module.reported_final_recall_at_1(
            {
                "methods": {
                    "wrong": {
                        "objective": "pfml",
                        "executed_train_steps": 4_080,
                        "recall_at_1": 0.9,
                    }
                }
            },
            4_080,
            expected_objective="proxy_anchor",
        )


def test_validate_export_recipe_requires_exact_id_and_digest() -> None:
    config = SimpleNamespace(
        recipe_id="proxy_anchor.cars.official-51db570",
        recipe_digest="d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a",
    )
    _module.validate_export_recipe(
        config,
        expected_recipe_id="proxy_anchor.cars.official-51db570",
        expected_recipe_digest="d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a",
    )
    with pytest.raises(ValueError, match="recipe digest"):
        _module.validate_export_recipe(
            config,
            expected_recipe_id="proxy_anchor.cars.official-51db570",
            expected_recipe_digest="0" * 64,
        )
