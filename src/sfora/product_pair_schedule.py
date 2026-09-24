"""Replayable full-coverage product batches with distinct positive images."""

from __future__ import annotations

import numpy as np


def product_pair_batches(
    labels: np.ndarray, *, products_per_batch: int, passes: int, seed: int
) -> np.ndarray:
    """Visit every product once per pass and draw two distinct images each visit.

    A short final batch is filled with different products already visited in
    that pass. Each batch still contains exactly one pair per product.
    """

    if (
        type(labels) is not np.ndarray
        or labels.dtype != np.int64
        or labels.ndim != 1
        or labels.size < 4
        or type(products_per_batch) is not int
        or products_per_batch < 2
        or type(passes) is not int
        or passes < 1
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("product-pair schedule inventory differs")
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < products_per_batch or np.any(counts < 2):
        raise ValueError("product-pair schedule needs two images per product")
    grouped = [np.flatnonzero(inverse == index) for index in range(len(classes))]
    generator = np.random.Generator(np.random.PCG64(seed))
    batches: list[np.ndarray] = []
    for _pass in range(passes):
        shuffled = generator.permutation(len(classes))
        for start in range(0, len(shuffled), products_per_batch):
            chosen = shuffled[start : start + products_per_batch]
            if len(chosen) < products_per_batch:
                padding = generator.choice(
                    shuffled[:start], products_per_batch - len(chosen), replace=False
                )
                chosen = np.concatenate((chosen, padding))
            rows = np.concatenate(
                [generator.choice(grouped[int(index)], 2, replace=False) for index in chosen]
            )
            batches.append(rows)
    return np.ascontiguousarray(np.stack(batches), dtype=np.int64)


__all__ = ["product_pair_batches"]
