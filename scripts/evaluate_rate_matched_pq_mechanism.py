#!/usr/bin/env python3
"""Diagnose geometry and quantization mechanisms for a fixed-rate similarity codec."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

import torch

from sfora.product_quantization import (
    ProductQuantizationSpec,
    balanced_product_quantization_spec,
    fit_optimized_product_quantizer,
    fit_product_quantizer,
)
from sfora.representation_ceiling import fit_centered_pca

_transfer_module = import_module(
    "scripts.evaluate_rate_matched_pq_transfer"
    if __package__
    else "evaluate_rate_matched_pq_transfer"
)
TransferArchive = _transfer_module.TransferArchive
normalize_embedding_rows = cast(
    Callable[[torch.Tensor], torch.Tensor], _transfer_module.normalize_embedding_rows
)
encode_in_batches = _transfer_module.encode_in_batches
score_adc_retrieval = _transfer_module.score_adc_retrieval
score_float_retrieval = _transfer_module.score_float_retrieval


class _TransferArchiveLike(Protocol):
    test_embeddings: torch.Tensor
    train_embeddings: torch.Tensor


class _QuantizerLike(Protocol):
    spec: ProductQuantizationSpec

    def to(self, device: torch.device) -> _QuantizerLike: ...

    def hard_encode(self, values: torch.Tensor) -> torch.Tensor: ...

    def asymmetric_squared_distances(
        self, queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class Representation:
    """One train-fitted geometry and its transformed train/test rows."""

    name: str
    spec: ProductQuantizationSpec
    test_embeddings: torch.Tensor
    train_embeddings: torch.Tensor


@dataclass(frozen=True, slots=True)
class QuantizedArm:
    """One fitted quantizer bound to its input representation."""

    codec: _QuantizerLike
    fit_seconds: float
    name: str
    representation: Representation


def _representation(name: str, train: torch.Tensor, test: torch.Tensor) -> Representation:
    return Representation(
        name=name,
        spec=balanced_product_quantization_spec(
            dimensions=train.shape[1], bytes_per_vector=24, codebook_size=256
        ),
        test_embeddings=test.contiguous(),
        train_embeddings=train.contiguous(),
    )


def _random_orthonormal_components(
    input_dimensions: int, output_dimensions: int, *, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(
        (input_dimensions, output_dimensions), dtype=torch.float64, generator=generator
    )
    components = torch.linalg.qr(matrix, mode="reduced").Q.T.contiguous()
    for component in components:
        pivot = int(torch.argmax(torch.abs(component)))
        if float(component[pivot]) < 0.0:
            component.neg_()
    return cast(torch.Tensor, components.float().contiguous())


def build_representations(
    archive: _TransferArchiveLike, *, dimensions: int, random_seed: int
) -> tuple[Representation, ...]:
    """Construct the fixed train-only geometry-control panel."""

    if (
        type(archive) is not TransferArchive
        or type(dimensions) is not int
        or not 1
        <= dimensions
        <= min(
            archive.train_embeddings.shape[0] - 1,
            archive.train_embeddings.shape[1],
        )
        or type(random_seed) is not int
        or random_seed < 0
    ):
        raise ValueError("mechanism representation authority differs")
    train = archive.train_embeddings
    test = archive.test_embeddings
    mean = train.double().mean(dim=0)
    centered_train = (train.double() - mean).float().contiguous()
    centered_test = (test.double() - mean).float().contiguous()
    centered_unit_train = normalize_embedding_rows(centered_train)
    centered_unit_test = normalize_embedding_rows(centered_test)

    pca = fit_centered_pca(train, dimensions=dimensions)
    components = pca.components.double()
    pca_train = (centered_train.double() @ components.T).float().contiguous()
    pca_test = (centered_test.double() @ components.T).float().contiguous()
    pca_unit_train = normalize_embedding_rows(pca_train)
    pca_unit_test = normalize_embedding_rows(pca_test)

    random_components = _random_orthonormal_components(
        train.shape[1], dimensions, seed=random_seed
    ).double()
    random_train = normalize_embedding_rows(
        (centered_train.double() @ random_components.T).float().contiguous()
    )
    random_test = normalize_embedding_rows(
        (centered_test.double() @ random_components.T).float().contiguous()
    )
    return (
        _representation("original-unit", train, test),
        _representation("centered", centered_train, centered_test),
        _representation("centered-unit", centered_unit_train, centered_unit_test),
        _representation("pca", pca_train, pca_test),
        _representation("pca-unit", pca_unit_train, pca_unit_test),
        _representation("random-orthogonal-unit", random_train, random_test),
    )


def fit_quantized_arms(
    representations: tuple[Representation, ...],
    *,
    seed: int,
    maximum_iterations: int,
    rotation_iterations: int,
) -> tuple[QuantizedArm, ...]:
    """Fit the frozen quantizer panel using representation training rows only."""

    expected_names = (
        "original-unit",
        "centered",
        "centered-unit",
        "pca",
        "pca-unit",
        "random-orthogonal-unit",
    )
    if (
        type(representations) is not tuple
        or tuple(item.name for item in representations) != expected_names
        or type(seed) is not int
        or seed < 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
        or type(rotation_iterations) is not int
        or rotation_iterations < 1
    ):
        raise ValueError("mechanism quantizer authority differs")
    by_name = {item.name: item for item in representations}
    matrix = (
        ("original-unit-opq24", "original-unit", "opq", 24),
        ("original-unit-opq32", "original-unit", "opq", 32),
        ("centered-unit-opq24", "centered-unit", "opq", 24),
        ("pca-opq24", "pca", "opq", 24),
        ("pca-unit-opq24", "pca-unit", "opq", 24),
        ("pca-unit-pq24", "pca-unit", "pq", 24),
        ("random-orthogonal-unit-opq24", "random-orthogonal-unit", "opq", 24),
    )
    arms: list[QuantizedArm] = []
    for arm_name, representation_name, kind, width in matrix:
        representation = by_name[representation_name]
        spec = (
            representation.spec
            if width == 24
            else balanced_product_quantization_spec(
                dimensions=representation.train_embeddings.shape[1],
                bytes_per_vector=width,
                codebook_size=256,
            )
        )
        started = time.perf_counter_ns()
        codec: _QuantizerLike
        if kind == "pq":
            codec = cast(
                _QuantizerLike,
                fit_product_quantizer(
                    representation.train_embeddings,
                    spec,
                    seed=seed,
                    maximum_iterations=maximum_iterations,
                ),
            )
        else:
            codec = cast(
                _QuantizerLike,
                fit_optimized_product_quantizer(
                    representation.train_embeddings,
                    spec,
                    seed=seed,
                    maximum_iterations=maximum_iterations,
                    rotation_iterations=rotation_iterations,
                ),
            )
        arms.append(
            QuantizedArm(
                codec=codec,
                fit_seconds=(time.perf_counter_ns() - started) / 1_000_000_000.0,
                name=arm_name,
                representation=representation,
            )
        )
    return tuple(arms)


def score_quantized_arm(
    arm: QuantizedArm,
    *,
    labels: tuple[int, ...],
    device: torch.device,
    encode_batch_size: int,
    query_batch_size: int,
) -> dict[str, Any]:
    """Encode and score one fitted arm on its bound evaluation rows."""

    if (
        type(arm) is not QuantizedArm
        or type(device) is not torch.device
        or device.type not in ("cpu", "cuda")
        or type(encode_batch_size) is not int
        or encode_batch_size < 1
        or type(query_batch_size) is not int
        or query_batch_size < 1
    ):
        raise ValueError("mechanism arm scoring authority differs")
    codec = arm.codec.to(device)
    queries = arm.representation.test_embeddings.to(device)

    def encode(batch: torch.Tensor) -> torch.Tensor:
        return codec.hard_encode(batch.to(device)).cpu()

    with torch.inference_mode():
        codes = encode_in_batches(
            arm.representation.test_embeddings,
            encoder=encode,
            batch_size=encode_batch_size,
        ).to(device)
        score = cast(
            dict[str, Any],
            score_adc_retrieval(
                queries,
                codes,
                labels,
                distance=codec.asymmetric_squared_distances,
                batch_size=query_batch_size,
            ),
        )
    return {
        "code_bytes_per_vector": codec.spec.bytes_per_vector,
        "fit_seconds": arm.fit_seconds,
        "map_at_r": score["map_at_r"],
        "name": arm.name,
        "per_query_ap": score["per_query_ap"],
        "per_query_r1": score["per_query_r1"],
        "r1": score["r1"],
    }


def score_float_representations(
    representations: tuple[Representation, ...],
    *,
    labels: tuple[int, ...],
    batch_size: int,
    neighbor_width: int,
) -> tuple[dict[str, Any], ...]:
    """Score every float geometry and its overlap with original-space neighbors."""

    if (
        type(representations) is not tuple
        or not representations
        or representations[0].name != "original-unit"
        or type(neighbor_width) is not int
        or neighbor_width < 1
    ):
        raise ValueError("mechanism float panel authority differs")
    scores: list[dict[str, Any]] = []
    original_neighbors: tuple[tuple[int, ...], ...] | None = None
    for representation in representations:
        score = cast(
            dict[str, Any],
            score_float_retrieval(
                representation.test_embeddings,
                representation.test_embeddings,
                labels,
                batch_size=batch_size,
                neighbor_width=neighbor_width,
            ),
        )
        neighbors = cast(tuple[tuple[int, ...], ...], score["neighbor_ordinals"])
        if original_neighbors is None:
            original_neighbors = neighbors
        overlap = math.fsum(
            len(set(reference).intersection(candidate)) / neighbor_width
            for reference, candidate in zip(original_neighbors, neighbors, strict=True)
        ) / len(neighbors)
        scores.append(
            {
                "map_at_r": score["map_at_r"],
                "name": representation.name,
                "original_neighbor_overlap": overlap,
                "per_query_ap": score["per_query_ap"],
                "per_query_r1": score["per_query_r1"],
                "r1": score["r1"],
            }
        )
    return tuple(scores)
