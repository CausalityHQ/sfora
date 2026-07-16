from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from sfora.evaluation import RetrievalScore, retrieval_score_on_split
from sfora.training import (
    Objective,
    ProjectionHeadTrainingConfig,
    ProjectionHeadTrainingResult,
    train_projection_head,
)


class SforaProjector:
    """Reusable projection-head API for group-aware metric learning.

    The class wraps :func:`train_projection_head` with a small fit/transform
    interface suitable for applications that already have embeddings and labels.
    """

    def __init__(
        self,
        *,
        objective: Objective = "group_supcon_xbm_radius",
        group_size: int = 4,
        steps: int = 80,
        learning_rate: float = 0.01,
        margin: float = 0.5,
        hard_weight: float = 0.5,
        spread_weight: float = 0.1,
        triplet_weight: float = 1.0,
        group_weight: float = 1.0,
        xbm_weight: float = 0.25,
        xbm_memory_size: int = 1024,
        radius_weight: float = 0.05,
        variance_weight: float = 0.05,
        output_dimensions: int | None = None,
        normalize_embeddings: bool = True,
        shuffle_groups_each_step: bool = False,
        seed: int = 0,
    ) -> None:
        self.config = ProjectionHeadTrainingConfig(
            objective=objective,
            group_size=group_size,
            steps=steps,
            learning_rate=learning_rate,
            margin=margin,
            hard_weight=hard_weight,
            spread_weight=spread_weight,
            triplet_weight=triplet_weight,
            group_weight=group_weight,
            xbm_weight=xbm_weight,
            xbm_memory_size=xbm_memory_size,
            radius_weight=radius_weight,
            variance_weight=variance_weight,
            output_dimensions=output_dimensions,
            normalize_projected_embeddings=normalize_embeddings,
            shuffle_groups_each_step=shuffle_groups_each_step,
            seed=seed,
        )
        self.training_result: ProjectionHeadTrainingResult | None = None

    @property
    def projection_matrix(self) -> NDArray[np.float64]:
        """Learned projection matrix after :meth:`fit`."""
        if self.training_result is None:
            raise ValueError("fit must be called before accessing projection_matrix")
        return self.training_result.projection_matrix

    @property
    def selected_step(self) -> int:
        """Training step selected by validation scoring, or final step without scoring."""
        if self.training_result is None:
            raise ValueError("fit must be called before accessing selected_step")
        return self.training_result.selected_step

    @property
    def selection_score(self) -> float | None:
        """Best validation MAP@R used for checkpoint selection, when supplied."""
        if self.training_result is None:
            raise ValueError("fit must be called before accessing selection_score")
        return self.training_result.selection_score

    def fit(
        self,
        embeddings: NDArray[np.floating],
        labels: NDArray[np.integer],
        *,
        validation_embeddings: NDArray[np.floating] | None = None,
        validation_labels: NDArray[np.integer] | None = None,
        validation_query_limit: int | None = None,
        random_state: int = 0,
    ) -> SforaProjector:
        """Fit the projection head on labeled embeddings.

        When a validation split is supplied, the best projection is selected by
        MAP@R from transformed validation queries into the transformed training
        gallery. This mirrors the benchmark protocol and avoids choosing a
        final step that improves objective loss while hurting retrieval quality.
        """
        input_embeddings = np.asarray(embeddings, dtype=np.float64)
        label_array = np.asarray(labels)
        selection_callback = _validation_selection_callback(
            train_embeddings=input_embeddings,
            train_labels=label_array,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            normalize_embeddings=self.config.normalize_projected_embeddings,
            query_limit=validation_query_limit,
            random_state=random_state,
        )
        # ``config`` is a public attribute a caller may mutate after construction, and
        # ProjectionHeadTrainingConfig does not validate on assignment — so revalidate
        # here before training, catching e.g. negative steps or an unknown objective.
        validated = ProjectionHeadTrainingConfig.model_validate(
            self.config.model_dump(exclude={"selection_score_callback"})
        )
        training_config = validated.model_copy(
            update={"selection_score_callback": selection_callback}
        )
        self.training_result = train_projection_head(input_embeddings, label_array, training_config)
        return self

    def transform(self, embeddings: NDArray[np.floating]) -> NDArray[np.float64]:
        """Project embeddings with the fitted projection matrix."""
        if self.training_result is None:
            raise ValueError("fit must be called before transform")
        projected = (
            np.asarray(embeddings, dtype=np.float64) @ self.training_result.projection_matrix
        )
        if self.config.normalize_projected_embeddings:
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            projected = projected / np.maximum(norms, 1e-12)
        return projected

    def fit_transform(
        self,
        embeddings: NDArray[np.floating],
        labels: NDArray[np.integer],
    ) -> NDArray[np.float64]:
        """Fit the projection head and return transformed training embeddings."""
        return self.fit(embeddings, labels).transform(embeddings)

    def score_retrieval(
        self,
        gallery_embeddings: NDArray[np.floating],
        gallery_labels: NDArray[np.integer],
        query_embeddings: NDArray[np.floating],
        query_labels: NDArray[np.integer],
        *,
        query_limit: int | None = None,
        random_state: int = 0,
    ) -> RetrievalScore:
        """Evaluate transformed query embeddings against a transformed gallery."""
        return retrieval_score_on_split(
            self.transform(gallery_embeddings),
            gallery_labels,
            self.transform(query_embeddings),
            query_labels,
            query_limit=query_limit,
            random_state=random_state,
        )


def fit_sfora_projection(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    objective: Objective = "group_supcon_xbm_radius",
    group_size: int = 4,
    steps: int = 80,
    learning_rate: float = 0.01,
    margin: float = 0.5,
    hard_weight: float = 0.5,
    spread_weight: float = 0.1,
    triplet_weight: float = 1.0,
    group_weight: float = 1.0,
    xbm_weight: float = 0.25,
    xbm_memory_size: int = 1024,
    radius_weight: float = 0.05,
    variance_weight: float = 0.05,
    output_dimensions: int | None = None,
    normalize_embeddings: bool = True,
    shuffle_groups_each_step: bool = False,
    seed: int = 0,
    validation_embeddings: NDArray[np.floating] | None = None,
    validation_labels: NDArray[np.integer] | None = None,
    validation_query_limit: int | None = None,
    random_state: int = 0,
) -> SforaProjector:
    """Fit and return a :class:`SforaProjector`."""
    return SforaProjector(
        objective=objective,
        group_size=group_size,
        steps=steps,
        learning_rate=learning_rate,
        margin=margin,
        hard_weight=hard_weight,
        spread_weight=spread_weight,
        triplet_weight=triplet_weight,
        group_weight=group_weight,
        xbm_weight=xbm_weight,
        xbm_memory_size=xbm_memory_size,
        radius_weight=radius_weight,
        variance_weight=variance_weight,
        output_dimensions=output_dimensions,
        normalize_embeddings=normalize_embeddings,
        shuffle_groups_each_step=shuffle_groups_each_step,
        seed=seed,
    ).fit(
        embeddings,
        labels,
        validation_embeddings=validation_embeddings,
        validation_labels=validation_labels,
        validation_query_limit=validation_query_limit,
        random_state=random_state,
    )


def _validation_selection_callback(
    *,
    train_embeddings: NDArray[np.float64],
    train_labels: NDArray[np.integer],
    validation_embeddings: NDArray[np.floating] | None,
    validation_labels: NDArray[np.integer] | None,
    normalize_embeddings: bool,
    query_limit: int | None,
    random_state: int,
) -> None | Callable[[NDArray[np.float64], int], float]:
    if validation_embeddings is None and validation_labels is None:
        return None
    if validation_embeddings is None or validation_labels is None:
        raise ValueError("validation_embeddings and validation_labels must be supplied together")

    validation_array = np.asarray(validation_embeddings, dtype=np.float64)
    validation_label_array = np.asarray(validation_labels)

    def score(projection: NDArray[np.float64], step: int) -> float:
        del step
        gallery = train_embeddings @ projection
        queries = validation_array @ projection
        if normalize_embeddings:
            gallery = _normalize(gallery)
            queries = _normalize(queries)
        return retrieval_score_on_split(
            gallery,
            train_labels,
            queries,
            validation_label_array,
            query_limit=query_limit,
            random_state=random_state,
        ).map_at_r

    return score


def _normalize(embeddings: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)
