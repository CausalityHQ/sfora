from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ProbeScore:
    """Metrics from a linear classifier trained on frozen embeddings."""

    accuracy: float
    macro_f1: float
    confusion_matrix: NDArray[np.int64]
    train_accuracy: float
    train_macro_f1: float


@dataclass(frozen=True)
class RetrievalScore:
    """Nearest-neighbor retrieval metrics for a frozen embedding space."""

    precision_at_1: float
    map_at_r: float
    mean_relevant_items: float
    evaluated_queries: int
    total_queries: int


@dataclass(frozen=True)
class EmbeddingSpaceDiagnostics:
    """Class-geometry diagnostics for interpreting downstream probe behavior."""

    train_within_class_radius: float
    test_within_class_radius: float
    train_centroid_gap: float
    train_test_centroid_drift: float
    signal_to_noise_ratio: float
    drift_to_gap_ratio: float


def linear_probe_score(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    test_size: float = 0.25,
    random_state: int = 0,
    max_iter: int = 1_000,
) -> ProbeScore:
    """Evaluate embedding separability with a multinomial logistic regression probe."""
    embedding_array = np.asarray(embeddings, dtype=float)
    label_array = np.asarray(labels)
    classes = np.unique(label_array)

    if embedding_array.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if label_array.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if embedding_array.shape[0] != label_array.shape[0]:
        raise ValueError("embeddings and labels must contain the same number of examples")
    if classes.shape[0] < 2:
        raise ValueError("linear probe requires at least two classes")

    x_train, x_test, y_train, y_test = train_test_split(
        embedding_array,
        label_array,
        test_size=test_size,
        random_state=random_state,
        stratify=label_array,
    )
    return linear_probe_score_on_split(
        x_train,
        y_train,
        x_test,
        y_test,
        random_state=random_state,
        max_iter=max_iter,
    )


def linear_probe_score_on_split(
    train_embeddings: NDArray[np.floating],
    train_labels: NDArray[np.integer],
    test_embeddings: NDArray[np.floating],
    test_labels: NDArray[np.integer],
    *,
    random_state: int = 0,
    max_iter: int = 1_000,
) -> ProbeScore:
    """Evaluate separability by training a linear probe on an explicit split."""
    x_train = np.asarray(train_embeddings, dtype=float)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(test_embeddings, dtype=float)
    y_test = np.asarray(test_labels)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("train_embeddings and test_embeddings must be 2D arrays")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("train_labels and test_labels must be 1D arrays")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train embeddings and labels must contain the same number of examples")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("test embeddings and labels must contain the same number of examples")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test embeddings must have the same dimensionality")

    classes = np.unique(np.concatenate([y_train, y_test]))
    train_classes = np.unique(y_train)
    if train_classes.shape[0] < 2:
        raise ValueError("linear probe requires at least two train classes")

    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(x_train, y_train)
    train_predictions = classifier.predict(x_train)
    predictions = classifier.predict(x_test)

    return ProbeScore(
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        confusion_matrix=confusion_matrix(y_test, predictions, labels=classes),
        train_accuracy=float(accuracy_score(y_train, train_predictions)),
        train_macro_f1=float(f1_score(y_train, train_predictions, average="macro")),
    )


def retrieval_score_on_split(
    gallery_embeddings: NDArray[np.floating],
    gallery_labels: NDArray[np.integer],
    query_embeddings: NDArray[np.floating],
    query_labels: NDArray[np.integer],
    *,
    query_limit: int | None = None,
    random_state: int = 0,
) -> RetrievalScore:
    """Evaluate nearest-neighbor retrieval from query embeddings into a gallery."""
    gallery = np.asarray(gallery_embeddings, dtype=float)
    gallery_label_array = np.asarray(gallery_labels)
    queries = np.asarray(query_embeddings, dtype=float)
    query_label_array = np.asarray(query_labels)

    if gallery.ndim != 2 or queries.ndim != 2:
        raise ValueError("gallery_embeddings and query_embeddings must be 2D arrays")
    if gallery_label_array.ndim != 1 or query_label_array.ndim != 1:
        raise ValueError("gallery_labels and query_labels must be 1D arrays")
    if gallery.shape[0] != gallery_label_array.shape[0]:
        raise ValueError("gallery embeddings and labels must contain the same number of examples")
    if queries.shape[0] != query_label_array.shape[0]:
        raise ValueError("query embeddings and labels must contain the same number of examples")
    if gallery.shape[1] != queries.shape[1]:
        raise ValueError("gallery and query embeddings must have the same dimensionality")
    if gallery.shape[0] == 0 or queries.shape[0] == 0:
        raise ValueError("retrieval scoring requires non-empty gallery and query embeddings")
    if query_limit is not None and query_limit < 1:
        raise ValueError("query_limit must be at least 1")

    total_queries = int(queries.shape[0])
    if query_limit is not None and query_limit < total_queries:
        query_indices = _stratified_query_indices(
            query_label_array,
            query_limit=query_limit,
            random_state=random_state,
        )
        queries = queries[query_indices]
        query_label_array = query_label_array[query_indices]

    precision_at_1_values: list[float] = []
    average_precisions: list[float] = []
    relevant_counts: list[int] = []
    for query, query_label in zip(queries, query_label_array, strict=True):
        distances = np.linalg.norm(gallery - query[np.newaxis, :], axis=1)
        order = np.argsort(distances, kind="stable")
        ordered_matches = gallery_label_array[order] == query_label
        relevant_count = int(ordered_matches.sum())
        if relevant_count == 0:
            continue
        top_r_matches = ordered_matches[:relevant_count]
        relevant_ranks = np.flatnonzero(top_r_matches) + 1
        precisions = [float(top_r_matches[:rank].sum() / rank) for rank in relevant_ranks]
        precision_at_1_values.append(float(ordered_matches[0]))
        average_precisions.append(float(sum(precisions) / relevant_count))
        relevant_counts.append(relevant_count)

    if not average_precisions:
        raise ValueError("retrieval scoring requires at least one relevant gallery item")

    return RetrievalScore(
        precision_at_1=float(np.mean(precision_at_1_values)),
        map_at_r=float(np.mean(average_precisions)),
        mean_relevant_items=float(np.mean(relevant_counts)),
        evaluated_queries=len(average_precisions),
        total_queries=total_queries,
    )


def _stratified_query_indices(
    labels: NDArray[np.integer],
    *,
    query_limit: int,
    random_state: int,
) -> NDArray[np.int64]:
    rng = np.random.default_rng(random_state)
    label_array = np.asarray(labels)
    grouped = {
        label: np.flatnonzero(label_array == label)
        for label in sorted(np.unique(label_array).tolist())
    }
    class_count = len(grouped)
    base_quota = query_limit // class_count
    remainder = query_limit % class_count
    selected: list[int] = []
    for position, label in enumerate(grouped):
        indices = grouped[label].copy()
        rng.shuffle(indices)
        quota = min(len(indices), base_quota + (1 if position < remainder else 0))
        selected.extend(int(index) for index in indices[:quota])

    if len(selected) < query_limit:
        selected_set = set(selected)
        remaining = np.array(
            [index for index in range(label_array.shape[0]) if index not in selected_set],
            dtype=np.int64,
        )
        rng.shuffle(remaining)
        selected.extend(int(index) for index in remaining[: query_limit - len(selected)])

    return np.asarray(sorted(selected), dtype=np.int64)


def embedding_space_diagnostics_on_split(
    train_embeddings: NDArray[np.floating],
    train_labels: NDArray[np.integer],
    test_embeddings: NDArray[np.floating],
    test_labels: NDArray[np.integer],
) -> EmbeddingSpaceDiagnostics:
    """Measure class geometry that affects linear-probe generalization."""
    x_train = np.asarray(train_embeddings, dtype=float)
    y_train = np.asarray(train_labels)
    x_test = np.asarray(test_embeddings, dtype=float)
    y_test = np.asarray(test_labels)

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("train_embeddings and test_embeddings must be 2D arrays")
    if y_train.ndim != 1 or y_test.ndim != 1:
        raise ValueError("train_labels and test_labels must be 1D arrays")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("train embeddings and labels must contain the same number of examples")
    if x_test.shape[0] != y_test.shape[0]:
        raise ValueError("test embeddings and labels must contain the same number of examples")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train and test embeddings must have the same dimensionality")

    train_classes = np.unique(y_train)
    if train_classes.shape[0] < 2:
        raise ValueError("embedding diagnostics require at least two train classes")
    missing_test_classes = set(np.unique(y_test).tolist()) - set(train_classes.tolist())
    if missing_test_classes:
        raise ValueError("test labels must be present in train labels")

    centroids = {
        label.item() if hasattr(label, "item") else label: x_train[y_train == label].mean(axis=0)
        for label in train_classes
    }
    train_distances = [
        _distance_to_centroid(row, centroids[label.item() if hasattr(label, "item") else label])
        for row, label in zip(x_train, y_train, strict=True)
    ]
    test_distances = [
        _distance_to_centroid(row, centroids[label.item() if hasattr(label, "item") else label])
        for row, label in zip(x_test, y_test, strict=True)
    ]
    centroid_values = list(centroids.values())
    centroid_gaps = [
        float(np.linalg.norm(left - right))
        for index, left in enumerate(centroid_values)
        for right in centroid_values[index + 1 :]
    ]
    drift_values = [
        float(np.linalg.norm(x_test[y_test == label].mean(axis=0) - centroids[key]))
        for label in train_classes
        for key in [label.item() if hasattr(label, "item") else label]
        if np.any(y_test == label)
    ]

    train_radius = float(np.mean(train_distances))
    test_radius = float(np.mean(test_distances))
    centroid_gap = float(np.mean(centroid_gaps))
    centroid_drift = float(np.mean(drift_values)) if drift_values else 0.0
    denominator = train_radius + test_radius
    signal_to_noise = centroid_gap / denominator if denominator > 0.0 else float("inf")
    drift_to_gap = centroid_drift / centroid_gap if centroid_gap > 0.0 else float("inf")
    return EmbeddingSpaceDiagnostics(
        train_within_class_radius=train_radius,
        test_within_class_radius=test_radius,
        train_centroid_gap=centroid_gap,
        train_test_centroid_drift=centroid_drift,
        signal_to_noise_ratio=float(signal_to_noise),
        drift_to_gap_ratio=float(drift_to_gap),
    )


def _distance_to_centroid(
    embedding: NDArray[np.floating],
    centroid: NDArray[np.floating],
) -> float:
    return float(np.linalg.norm(embedding - centroid))
