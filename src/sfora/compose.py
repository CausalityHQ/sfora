"""Composable, type-safe projection *bricks* for building and comparing metric-learning
pipelines.

Everything is a :class:`Projection`: something you ``fit`` on train embeddings/labels and
``transform`` onto any embeddings. Bricks compose:

- **leaves** — :class:`Identity`, :class:`L2Normalize`, :class:`Pca`, :class:`Head`
  (a trainable projection head wrapping the :class:`~sfora.api.SforaProjector` objectives);
- **combinators** — :class:`Pipeline` (run bricks in sequence) and :class:`Join`
  (fan out to several bricks and *join* their outputs by concatenation, mean, or
  Procrustes-aligned mean — a feature-concatenation / aligned-average ensemble);
- **evaluation** — :func:`evaluate` fits on train and scores retrieval on test
  (train and test classes are disjoint in the zero-shot protocol, so nothing is fit
  on the test split), and :func:`compare` ranks a whole set of candidates so a user
  can pick the best approach for their data.

Example::

    from sfora.compose import Head, Join, Pca, Pipeline, compare, grid

    candidates = {
        "proxy_anchor": Head(objective="proxy_anchor"),
        "group+pca128": Pipeline([Head(objective="group_supcon_xbm_radius"), Pca(128)]),
        "ensemble3": Join("concat", [Head(seed=s) for s in range(3)]),
        **grid("head", Head, {"objective": ["triplet", "group_supcon"], "steps": [40, 80]}),
    }
    ranking = compare(candidates, train=(Xtr, ytr), test=(Xte, yte))
    print(ranking[0].name, ranking[0].recall_at_1)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sfora.api import SforaProjector
from sfora.image_benchmark import image_self_retrieval_score
from sfora.training import Objective

Embeddings = NDArray[np.floating]
Labels = NDArray[np.integer]
Data = tuple[Embeddings, Labels]
Floats = NDArray[np.float64]


@runtime_checkable
class Projection(Protocol):
    """A fit/transform brick. `fit` learns everything from the TRAIN split only."""

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = ...
    ) -> Self: ...

    def transform(self, embeddings: Embeddings) -> Floats: ...


def _l2(matrix: Floats) -> Floats:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1.0e-12)


# ─────────────────────────────  leaf bricks  ─────────────────────────────


@dataclass
class Identity:
    """Pass embeddings through unchanged (a useful baseline / placeholder)."""

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        del embeddings, labels, validation
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        return np.asarray(embeddings, dtype=np.float64)


@dataclass
class L2Normalize:
    """L2-normalise each row to the unit sphere (cosine geometry)."""

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        del embeddings, labels, validation
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        return _l2(np.asarray(embeddings, dtype=np.float64))


@dataclass
class Pca:
    """Fit PCA axes on the TRAIN embeddings, project any embeddings to `dim` dims."""

    dim: int
    whiten: bool = False
    _mean: Floats | None = field(default=None, init=False, repr=False)
    _components: Floats | None = field(default=None, init=False, repr=False)
    _scale: Floats | None = field(default=None, init=False, repr=False)

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        del labels, validation
        features = np.asarray(embeddings, dtype=np.float64)
        self._mean = features.mean(axis=0, keepdims=True)
        _, singular, vt = np.linalg.svd(features - self._mean, full_matrices=False)
        keep = min(self.dim, vt.shape[0])
        self._components = vt[:keep]
        self._scale = singular[:keep] if self.whiten else None
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        if self._mean is None or self._components is None:
            raise ValueError("Pca.fit must be called before transform")
        projected = (np.asarray(embeddings, dtype=np.float64) - self._mean) @ self._components.T
        if self._scale is not None:
            projected = projected / (self._scale + 1.0e-8)
        return projected


@dataclass
class Head:
    """A trainable projection head — thin, typed wrapper over `SforaProjector`.

    Trains one of the library's metric-learning objectives on frozen embeddings and
    applies the learned matrix. Any objective/param the projector accepts is settable
    through `objective` and `params`.
    """

    objective: Objective = "group_supcon_xbm_radius"
    steps: int = 80
    seed: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)
    _projector: SforaProjector | None = field(default=None, init=False, repr=False)

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        projector = SforaProjector(
            objective=self.objective, steps=self.steps, seed=self.seed, **dict(self.params)
        )
        if validation is not None:
            projector.fit(
                embeddings,
                labels,
                validation_embeddings=validation[0],
                validation_labels=validation[1],
            )
        else:
            projector.fit(embeddings, labels)
        self._projector = projector
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        if self._projector is None:
            raise ValueError("Head.fit must be called before transform")
        return self._projector.transform(embeddings)


# ─────────────────────────────  combinators  ─────────────────────────────


@dataclass
class Pipeline:
    """Run bricks in sequence: each is fit on the previous brick's train output."""

    steps: Sequence[Projection]

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        current = np.asarray(embeddings, dtype=np.float64)
        val = validation
        for step in self.steps:
            step.fit(current, labels, validation=val)
            current = step.transform(current)
            if val is not None:
                val = (step.transform(val[0]), val[1])
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        current = np.asarray(embeddings, dtype=np.float64)
        for step in self.steps:
            current = step.transform(current)
        return current


JoinKind = Literal["concat", "mean", "aligned_mean"]


@dataclass
class Join:
    """Fan out to several branches, fit each on the same input, and join their outputs.

    - ``concat`` — feature-concatenation ensemble (branches may differ in kind/dim);
      the concatenation is L2-normalised.
    - ``mean`` — average the branch outputs (branches must share output dim).
    - ``aligned_mean`` — Procrustes-align every branch's TRAIN output to the first,
      store the rotations, and average (the alignment is fit on train, then frozen).
    """

    kind: JoinKind
    branches: Sequence[Projection]
    _rotations: list[Floats] = field(default_factory=list, init=False, repr=False)

    def fit(
        self, embeddings: Embeddings, labels: Labels, *, validation: Data | None = None
    ) -> Self:
        for branch in self.branches:
            branch.fit(embeddings, labels, validation=validation)
        if self.kind == "aligned_mean":
            outputs = [_l2(branch.transform(embeddings)) for branch in self.branches]
            reference = outputs[0]
            self._rotations = [np.eye(reference.shape[1])]
            for output in outputs[1:]:
                u, _, vt = np.linalg.svd(output.T @ reference, full_matrices=False)
                self._rotations.append(u @ vt)
        return self

    def transform(self, embeddings: Embeddings) -> Floats:
        outputs = [branch.transform(embeddings) for branch in self.branches]
        if self.kind == "concat":
            return _l2(np.concatenate([_l2(o) for o in outputs], axis=1))
        if self.kind == "mean":
            return _l2(np.mean([_l2(o) for o in outputs], axis=0))
        if not self._rotations:
            raise ValueError("Join(aligned_mean).fit must be called before transform")
        aligned = [_l2(o) @ r for o, r in zip(outputs, self._rotations, strict=True)]
        return _l2(np.mean(aligned, axis=0))


# ─────────────────────────────  evaluation  ─────────────────────────────


@dataclass(frozen=True)
class RetrievalReport:
    """Retrieval metrics for one candidate (the set DML papers report)."""

    name: str
    recall_at_1: float
    recall_at_2: float
    recall_at_4: float
    recall_at_8: float
    map_at_r: float
    output_dim: int


def evaluate(
    projection: Projection,
    *,
    train: Data,
    test: Data,
    name: str = "candidate",
    random_state: int = 0,
) -> RetrievalReport:
    """Fit `projection` on the TRAIN split, then self-retrieval-score it on TEST.

    Nothing is fit on the test split — train and test classes are disjoint in the
    zero-shot retrieval protocol, so this is an honest, non-transductive estimate.
    """
    projection.fit(train[0], train[1])
    projected = _l2(projection.transform(test[0]))
    metrics = image_self_retrieval_score(projected, test[1], random_state=random_state)
    return RetrievalReport(
        name=name,
        recall_at_1=metrics.recall_at_1,
        recall_at_2=metrics.recall_at_2,
        recall_at_4=metrics.recall_at_4,
        recall_at_8=metrics.recall_at_8,
        map_at_r=metrics.map_at_r,
        output_dim=int(projected.shape[1]),
    )


def compare(
    candidates: Mapping[str, Projection],
    *,
    train: Data,
    test: Data,
    rank_by: Literal["recall_at_1", "recall_at_2", "recall_at_4", "recall_at_8", "map_at_r"] = (
        "recall_at_1"
    ),
    random_state: int = 0,
) -> list[RetrievalReport]:
    """Fit and score every candidate on the same split, ranked best-first by `rank_by`."""
    reports = [
        evaluate(projection, train=train, test=test, name=name, random_state=random_state)
        for name, projection in candidates.items()
    ]
    return sorted(reports, key=lambda report: getattr(report, rank_by), reverse=True)


def grid(
    prefix: str,
    factory: type[Projection],
    param_grid: Mapping[str, Sequence[Any]],
) -> dict[str, Projection]:
    """Expand a parameter grid into named candidates for :func:`compare`.

    ``grid("head", Head, {"objective": ["triplet", "group_supcon"], "steps": [40, 80]})``
    yields four named ``Head`` bricks. Requires the factory to accept the grid keys as
    keyword arguments.
    """
    keys = list(param_grid)
    candidates: dict[str, Projection] = {}
    for values in product(*(param_grid[key] for key in keys)):
        combo = dict(zip(keys, values, strict=True))
        label = ",".join(f"{key}={value}" for key, value in combo.items())
        candidates[f"{prefix}[{label}]"] = factory(**combo)
    return candidates
