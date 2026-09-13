#!/usr/bin/env python3
"""Diagnose geometry and quantization mechanisms for a fixed-rate similarity codec."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import torch

from sfora.product_quantization import (
    ProductQuantizationSpec,
    balanced_product_quantization_spec,
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


class _TransferArchiveLike(Protocol):
    test_embeddings: torch.Tensor
    train_embeddings: torch.Tensor


@dataclass(frozen=True, slots=True)
class Representation:
    """One train-fitted geometry and its transformed train/test rows."""

    name: str
    spec: ProductQuantizationSpec
    test_embeddings: torch.Tensor
    train_embeddings: torch.Tensor


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
