from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from sfora.product_quantization import balanced_product_quantization_spec

TRANSFER_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/evaluate_rate_matched_pq_transfer.py"
)
TRANSFER_SPEC = importlib.util.spec_from_file_location(
    "evaluate_rate_matched_pq_transfer", TRANSFER_SCRIPT
)
assert TRANSFER_SPEC is not None and TRANSFER_SPEC.loader is not None
TRANSFER = importlib.util.module_from_spec(TRANSFER_SPEC)
sys.modules[TRANSFER_SPEC.name] = TRANSFER
TRANSFER_SPEC.loader.exec_module(TRANSFER)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rate_matched_pq_mechanism.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rate_matched_pq_mechanism", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUBJECT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUBJECT
SPEC.loader.exec_module(SUBJECT)


def _archive() -> Any:
    generator = torch.Generator().manual_seed(991)
    train = TRANSFER.normalize_embedding_rows(torch.randn((128, 96), generator=generator))
    test = TRANSFER.normalize_embedding_rows(torch.randn((12, 96), generator=generator))
    return TRANSFER.TransferArchive(
        archive_keys=("test_embeddings", "test_labels", "train_embeddings", "train_labels"),
        sha256="1" * 64,
        test_embeddings=test,
        test_labels=(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
        train_embeddings=train,
        train_labels=tuple(index // 2 for index in range(128)),
    )


def test_build_representations_freezes_geometry_and_uses_train_only_statistics() -> None:
    archive = _archive()

    representations = SUBJECT.build_representations(archive, dimensions=80, random_seed=50)

    assert tuple(item.name for item in representations) == (
        "original-unit",
        "centered",
        "centered-unit",
        "pca",
        "pca-unit",
        "random-orthogonal-unit",
    )
    by_name = {item.name: item for item in representations}
    assert by_name["original-unit"].train_embeddings is archive.train_embeddings
    assert by_name["original-unit"].test_embeddings is archive.test_embeddings

    mean = archive.train_embeddings.double().mean(dim=0)
    expected_centered_train = (archive.train_embeddings.double() - mean).float()
    expected_centered_test = (archive.test_embeddings.double() - mean).float()
    torch.testing.assert_close(
        by_name["centered"].train_embeddings,
        expected_centered_train,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        by_name["centered"].test_embeddings,
        expected_centered_test,
        rtol=0.0,
        atol=0.0,
    )
    original_distances = torch.cdist(
        archive.test_embeddings.double(), archive.test_embeddings.double()
    ).square()
    centered_distances = torch.cdist(
        by_name["centered"].test_embeddings.double(),
        by_name["centered"].test_embeddings.double(),
    ).square()
    torch.testing.assert_close(centered_distances, original_distances, rtol=0.0, atol=2e-6)

    for name in ("centered-unit", "pca-unit", "random-orthogonal-unit"):
        assert by_name[name].train_embeddings.is_contiguous()
        assert by_name[name].test_embeddings.is_contiguous()
        torch.testing.assert_close(
            torch.linalg.vector_norm(by_name[name].test_embeddings.double(), dim=1),
            torch.ones(12, dtype=torch.float64),
            rtol=0.0,
            atol=2e-6,
        )
    assert by_name["pca"].train_embeddings.shape == (128, 80)
    assert by_name["pca"].test_embeddings.shape == (12, 80)
    assert by_name["pca-unit"].spec == balanced_product_quantization_spec(
        dimensions=80,
        bytes_per_vector=24,
        codebook_size=256,
    )


def test_random_representation_is_seed_deterministic() -> None:
    archive = _archive()

    first = SUBJECT.build_representations(archive, dimensions=80, random_seed=50)[-1]
    second = SUBJECT.build_representations(archive, dimensions=80, random_seed=50)[-1]
    different = SUBJECT.build_representations(archive, dimensions=80, random_seed=51)[-1]

    torch.testing.assert_close(first.train_embeddings, second.train_embeddings, rtol=0.0, atol=0.0)
    assert not torch.equal(first.train_embeddings, different.train_embeddings)


def test_fit_quantized_arms_uses_frozen_matrix_and_train_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive()
    representations = SUBJECT.build_representations(archive, dimensions=80, random_seed=50)
    calls: list[tuple[str, torch.Tensor, Any, dict[str, int]]] = []

    class FakeCodec:
        pass

    def fit_opq(values: torch.Tensor, spec: object, **kwargs: int) -> FakeCodec:
        calls.append(("opq", values, spec, kwargs))
        return FakeCodec()

    def fit_pq(values: torch.Tensor, spec: object, **kwargs: int) -> FakeCodec:
        calls.append(("pq", values, spec, kwargs))
        return FakeCodec()

    monkeypatch.setattr(SUBJECT, "fit_optimized_product_quantizer", fit_opq)
    monkeypatch.setattr(SUBJECT, "fit_product_quantizer", fit_pq)

    arms = SUBJECT.fit_quantized_arms(
        representations,
        seed=53,
        maximum_iterations=20,
        rotation_iterations=4,
    )

    assert tuple(arm.name for arm in arms) == (
        "original-unit-opq24",
        "original-unit-opq32",
        "centered-unit-opq24",
        "pca-opq24",
        "pca-unit-opq24",
        "pca-unit-pq24",
        "random-orthogonal-unit-opq24",
    )
    by_representation = {item.name: item for item in representations}
    expected = (
        ("opq", "original-unit", 24),
        ("opq", "original-unit", 32),
        ("opq", "centered-unit", 24),
        ("opq", "pca", 24),
        ("opq", "pca-unit", 24),
        ("pq", "pca-unit", 24),
        ("opq", "random-orthogonal-unit", 24),
    )
    assert len(calls) == len(expected)
    for (kind, values, spec, kwargs), (want_kind, representation_name, width) in zip(
        calls, expected, strict=True
    ):
        assert kind == want_kind
        assert values is by_representation[representation_name].train_embeddings
        assert spec.bytes_per_vector == width
        assert kwargs == (
            {"maximum_iterations": 20, "seed": 53}
            if kind == "pq"
            else {"maximum_iterations": 20, "rotation_iterations": 4, "seed": 53}
        )


def test_score_quantized_arm_encodes_only_test_rows_and_uses_exact_adc() -> None:
    values = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 2)
    distances = torch.tensor(
        [
            [0.0, 1.0, 1.0, 4.0],
            [1.0, 0.0, 4.0, 1.0],
            [0.5, 3.0, 0.0, 2.0],
            [3.0, 0.5, 2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    class FakeCodec:
        def __init__(self) -> None:
            self.spec = balanced_product_quantization_spec(
                dimensions=2, bytes_per_vector=2, codebook_size=256
            )

        def to(self, device: torch.device) -> FakeCodec:
            assert device == torch.device("cpu")
            return self

        def hard_encode(self, batch: torch.Tensor) -> torch.Tensor:
            assert batch.shape[1] == 2
            return batch.to(torch.uint8)

        def asymmetric_squared_distances(
            self, queries: torch.Tensor, codes: torch.Tensor
        ) -> torch.Tensor:
            assert codes.shape == (4, 2)
            return distances[queries[:, 0].long()]

    representation = SUBJECT.Representation(
        name="fixture",
        spec=FakeCodec().spec,
        test_embeddings=values,
        train_embeddings=torch.full((256, 2), -99.0, dtype=torch.float32),
    )
    arm = SUBJECT.QuantizedArm(
        codec=FakeCodec(),
        fit_seconds=1.25,
        name="fixture-opq2",
        representation=representation,
    )

    result = SUBJECT.score_quantized_arm(
        arm,
        labels=(1, 2, 1, 2),
        device=torch.device("cpu"),
        encode_batch_size=2,
        query_batch_size=2,
    )

    assert result == {
        "code_bytes_per_vector": 2,
        "fit_seconds": 1.25,
        "map_at_r": 0.5,
        "name": "fixture-opq2",
        "per_query_ap": (0.0, 0.0, 1.0, 1.0),
        "per_query_r1": (0.0, 0.0, 1.0, 1.0),
        "r1": 0.5,
    }
