"""Low-cost diagnostics for the SigLIP Proxy Anchor head bottleneck."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class CotangentRankEvidence:
    """Analytic rank ceilings for one pooled Proxy Anchor configuration."""

    class_count: int
    logical_batch_size: int
    embedding_dimensions: int
    tower_dimensions: int
    maximum_per_example_cotangent_rank: int
    maximum_projection_gradient_rank: int
    per_example_tower_fraction: float
    projection_gradient_fraction: float


@dataclass(frozen=True, slots=True)
class SpectralProjectionEvidence:
    """One uncentered spectral projection and its rank-cut stability evidence."""

    weight: torch.Tensor
    retained_singular_value: float
    discarded_singular_value: float | None
    cut_ratio: float | None


@dataclass(frozen=True, slots=True)
class FeatureSplitAuthority:
    """Authority proving data-dependent head state uses optimization data only."""

    source_manifest_sha256: str
    ordered_example_ids_sha256: str
    feature_matrix_sha256: str
    role: str
    official_test_access: bool
    example_count: int

    def validated(self, *, features: torch.Tensor) -> FeatureSplitAuthority:
        """Reject evaluation data, malformed identities, and feature drift."""

        if (
            type(self.source_manifest_sha256) is not str
            or len(self.source_manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_manifest_sha256)
            or len(self.ordered_example_ids_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.ordered_example_ids_sha256
            )
            or len(self.feature_matrix_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.feature_matrix_sha256)
            or self.role != "optimization-train"
            or type(self.official_test_access) is not bool
            or self.official_test_access
            or type(self.example_count) is not int
            or self.example_count != features.shape[0]
            or self.feature_matrix_sha256 != _feature_matrix_sha256(features)
        ):
            raise ValueError("feature split authority differs")
        return self


def _feature_matrix_sha256(features: torch.Tensor) -> str:
    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.device.type != "cpu"
        or not features.is_floating_point()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("feature matrix authority differs")
    payload = bytearray(b"sfora-feature-matrix-v1\0")
    payload.extend(features.shape[0].to_bytes(8, "big"))
    payload.extend(features.shape[1].to_bytes(8, "big"))
    canonical = features.detach().to(dtype=torch.float64).contiguous().numpy()
    payload.extend(canonical.astype("<f8", copy=False).tobytes(order="C"))
    return hashlib.sha256(payload).hexdigest()


def _ordered_example_ids_sha256(example_ids: tuple[str, ...]) -> str:
    payload = bytearray(b"sfora-ordered-example-ids-v1\0")
    payload.extend(len(example_ids).to_bytes(8, "big"))
    for example_id in example_ids:
        encoded = example_id.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    return hashlib.sha256(payload).hexdigest()


def build_feature_split_authority(
    *,
    source_manifest_sha256: str,
    role: str,
    official_test_access: bool,
    ordered_example_ids: tuple[str, ...],
    features: torch.Tensor,
) -> FeatureSplitAuthority:
    """Bind an ordered feature matrix to its immutable source split identity."""

    if (
        type(ordered_example_ids) is not tuple
        or len(ordered_example_ids) != features.shape[0]
        or len(set(ordered_example_ids)) != len(ordered_example_ids)
        or any(type(value) is not str or not value for value in ordered_example_ids)
    ):
        raise ValueError("feature example identity differs")
    return FeatureSplitAuthority(
        source_manifest_sha256=source_manifest_sha256,
        ordered_example_ids_sha256=_ordered_example_ids_sha256(ordered_example_ids),
        feature_matrix_sha256=_feature_matrix_sha256(features),
        role=role,
        official_test_access=official_test_access,
        example_count=len(ordered_example_ids),
    )


@dataclass(frozen=True, slots=True)
class SubclassAssignments:
    """Frozen per-row subclass IDs and their original-class parents."""

    assignments: torch.Tensor
    parents: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class CachedHeadTrainingEvidence:
    """Deterministic train-only state for one frozen-tower head screen."""

    split_authority: FeatureSplitAuthority
    subclass_assignments: SubclassAssignments
    initial_projection_weight: torch.Tensor
    projection_weight: torch.Tensor
    subclass_proxies: torch.Tensor
    initial_loss: float
    final_loss: float
    train_steps: int
    logical_batch_size: int
    loss_trajectory: tuple[float, ...]


def _subclass_assignment_sha256(
    assignments: torch.Tensor,
    parents: tuple[int, ...],
    *,
    master_seed_sha256: str,
    subclasses_per_class: int,
    iterations: int,
    split_authority: FeatureSplitAuthority,
) -> str:
    payload = bytearray(b"sfora-cosine-subclass-assignments-v1\0")
    values = tuple(int(value) for value in assignments.tolist())
    payload.extend(len(values).to_bytes(8, "big"))
    for value in values:
        payload.extend(value.to_bytes(8, "big", signed=True))
    payload.extend(len(parents).to_bytes(8, "big"))
    for value in parents:
        payload.extend(value.to_bytes(8, "big", signed=True))
    payload.extend(bytes.fromhex(master_seed_sha256))
    payload.extend(subclasses_per_class.to_bytes(8, "big"))
    payload.extend(iterations.to_bytes(8, "big"))
    payload.extend(bytes.fromhex(split_authority.source_manifest_sha256))
    payload.extend(bytes.fromhex(split_authority.ordered_example_ids_sha256))
    payload.extend(bytes.fromhex(split_authority.feature_matrix_sha256))
    return hashlib.sha256(payload).hexdigest()


def cotangent_rank_evidence(
    *,
    class_count: int,
    logical_batch_size: int,
    embedding_dimensions: int,
    tower_dimensions: int,
) -> CotangentRankEvidence:
    """Compute the exact structural rank ceilings implied by proxy scoring."""

    values = (class_count, logical_batch_size, embedding_dimensions, tower_dimensions)
    if any(type(value) is not int or value <= 0 for value in values) or class_count < 2:
        raise ValueError("cotangent-rank dimensions differ")
    per_example = min(tower_dimensions, embedding_dimensions, class_count + 1)
    projection = min(
        tower_dimensions,
        embedding_dimensions,
        logical_batch_size,
        class_count + logical_batch_size,
    )
    return CotangentRankEvidence(
        class_count=class_count,
        logical_batch_size=logical_batch_size,
        embedding_dimensions=embedding_dimensions,
        tower_dimensions=tower_dimensions,
        maximum_per_example_cotangent_rank=per_example,
        maximum_projection_gradient_rank=projection,
        per_example_tower_fraction=per_example / tower_dimensions,
        projection_gradient_fraction=projection / embedding_dimensions,
    )


def uncentered_spectral_projection_evidence(
    features: torch.Tensor,
    *,
    output_dimensions: int,
    split_authority: FeatureSplitAuthority,
) -> SpectralProjectionEvidence:
    """Return sign-canonical directions and the observed spectral-cut ratio."""

    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or not features.is_floating_point()
        or features.dtype not in (torch.float32, torch.float64)
        or type(output_dimensions) is not int
        or output_dimensions <= 0
        or output_dimensions > min(features.shape)
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("spectral projection input differs")
    if features.device.type != "cpu":
        raise ValueError("spectral projection requires authoritative CPU features")
    if type(split_authority) is not FeatureSplitAuthority:
        raise ValueError("feature split authority differs")
    split_authority.validated(features=features)

    matrix = features.to(dtype=torch.float64)
    _left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
    if (
        singular_values.shape[0] < output_dimensions
        or not bool(torch.isfinite(singular_values).all())
        or bool((singular_values[:output_dimensions] <= 0).any())
    ):
        raise ValueError("spectral projection rank differs")
    weight = right[:output_dimensions].clone()
    for row in weight:
        nonzero = torch.nonzero(row != 0, as_tuple=False)
        if nonzero.numel() == 0:
            raise ValueError("spectral projection direction differs")
        if bool(row[int(nonzero[0])] < 0):
            row.neg_()
    result = cast(torch.Tensor, weight.to(dtype=torch.float32))
    if not bool(torch.isfinite(result).all()):
        raise ValueError("spectral projection result differs")
    retained = float(singular_values[output_dimensions - 1])
    discarded = (
        float(singular_values[output_dimensions])
        if output_dimensions < singular_values.shape[0]
        else None
    )
    ratio = retained / discarded if discarded is not None and discarded > 0.0 else None
    return SpectralProjectionEvidence(
        weight=result,
        retained_singular_value=retained,
        discarded_singular_value=discarded,
        cut_ratio=ratio,
    )


def uncentered_spectral_projection(
    features: torch.Tensor,
    *,
    output_dimensions: int,
    split_authority: FeatureSplitAuthority,
) -> torch.Tensor:
    """Return sign-canonical top right singular vectors without centering."""

    return uncentered_spectral_projection_evidence(
        features,
        output_dimensions=output_dimensions,
        split_authority=split_authority,
    ).weight


def initialize_spectral_projection_(
    projection: nn.Linear,
    features: torch.Tensor,
    *,
    split_authority: FeatureSplitAuthority,
) -> nn.Linear:
    """Install the deterministic uncentered spectral projection in-place."""

    if (
        type(projection) is not nn.Linear
        or projection.bias is not None
        or projection.weight.dtype != torch.float32
    ):
        raise ValueError("spectral projection layer authority differs")
    weight = uncentered_spectral_projection(
        features,
        output_dimensions=projection.out_features,
        split_authority=split_authority,
    )
    if weight.shape != projection.weight.shape:
        raise ValueError("spectral projection shape differs")
    with torch.no_grad():
        projection.weight.copy_(weight.to(device=projection.weight.device))
    return projection


def subclass_proxy_anchor_loss(
    class_scores: torch.Tensor,
    original_labels: torch.Tensor,
    subclass_assignments: torch.Tensor,
    subclass_parents: torch.Tensor,
    *,
    alpha: float,
    delta: float,
) -> torch.Tensor:
    """Proxy Anchor over subclasses without repelling sibling subclasses."""

    batch, subclasses = class_scores.shape if class_scores.ndim == 2 else (-1, -1)
    if (
        batch <= 0
        or subclasses < 2
        or original_labels.shape != (batch,)
        or subclass_assignments.shape != (batch,)
        or subclass_parents.shape != (subclasses,)
        or original_labels.dtype not in (torch.int32, torch.int64)
        or subclass_assignments.dtype not in (torch.int32, torch.int64)
        or subclass_parents.dtype not in (torch.int32, torch.int64)
        or not class_scores.is_floating_point()
        or not bool(torch.isfinite(class_scores).all())
        or type(alpha) is not float
        or alpha <= 0.0
        or type(delta) is not float
        or not 0.0 <= delta < 1.0
    ):
        raise ValueError("subclass Proxy Anchor authority differs")
    labels = original_labels.to(dtype=torch.int64, device=class_scores.device)
    assignments = subclass_assignments.to(dtype=torch.int64, device=class_scores.device)
    parents = subclass_parents.to(dtype=torch.int64, device=class_scores.device)
    if (
        bool((assignments < 0).any())
        or bool((assignments >= subclasses).any())
        or bool((labels < 0).any())
        or bool((parents < 0).any())
        or not torch.equal(parents[assignments], labels)
    ):
        raise ValueError("subclass Proxy Anchor label authority differs")

    positives = F.one_hot(assignments, num_classes=subclasses).to(torch.bool)
    negatives = labels[:, None] != parents[None, :]
    zero = torch.zeros((1, subclasses), dtype=class_scores.dtype, device=class_scores.device)
    positive_logits = (-alpha * (class_scores - delta)).masked_fill(~positives, -torch.inf)
    negative_logits = (alpha * (class_scores + delta)).masked_fill(~negatives, -torch.inf)
    positive_terms = torch.logsumexp(torch.cat((zero, positive_logits), dim=0), dim=0)
    negative_terms = torch.logsumexp(torch.cat((zero, negative_logits), dim=0), dim=0)
    return positive_terms[positives.any(dim=0)].mean() + negative_terms.mean()


def cosine_subclass_assignments(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    subclasses_per_class: int,
    master_seed_sha256: str,
    iterations: int,
    split_authority: FeatureSplitAuthority,
) -> SubclassAssignments:
    """Freeze deterministic class-local spherical k-means++ assignments."""

    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.device.type != "cpu"
        or not features.is_floating_point()
        or features.dtype not in (torch.float32, torch.float64)
        or labels.shape != (features.shape[0],)
        or labels.device.type != "cpu"
        or labels.dtype not in (torch.int32, torch.int64)
        or type(subclasses_per_class) is not int
        or subclasses_per_class < 2
        or type(iterations) is not int
        or iterations <= 0
        or type(master_seed_sha256) is not str
        or len(master_seed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in master_seed_sha256)
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("subclass assignment authority differs")
    norms = torch.linalg.vector_norm(features.double(), dim=1)
    if type(split_authority) is not FeatureSplitAuthority:
        raise ValueError("feature split authority differs")
    split_authority.validated(features=features)
    if bool((norms <= 0).any()):
        raise ValueError("subclass feature norm differs")
    normalized = features.double() / norms[:, None]
    class_ids = tuple(sorted({int(value) for value in labels.tolist()}))
    assignments = torch.empty(features.shape[0], dtype=torch.int64)
    parents: list[int] = []
    offset = 0
    for class_id in class_ids:
        indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        if indices.numel() < 2 * subclasses_per_class:
            raise ValueError("subclass class population differs")
        values = normalized[indices]
        seed_bytes = hashlib.sha256(
            b"sfora-cosine-subclasses-v1\0"
            + bytes.fromhex(master_seed_sha256)
            + class_id.to_bytes(8, "big", signed=True)
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(seed_bytes[:8], "big") % (2**63 - 1))
        first = int(torch.randint(values.shape[0], (1,), generator=generator))
        centroid_rows = [first]
        while len(centroid_rows) < subclasses_per_class:
            similarities = values @ values[centroid_rows].T
            distances = (1.0 - similarities.max(dim=1).values).clamp_min(0.0).square()
            distances[centroid_rows] = 0.0
            if float(distances.sum()) <= 0.0:
                remaining = [row for row in range(values.shape[0]) if row not in centroid_rows]
                centroid_rows.append(remaining[0])
            else:
                centroid_rows.append(
                    int(torch.multinomial(distances, 1, generator=generator).item())
                )
        centroids = values[centroid_rows].clone()
        local = torch.zeros(values.shape[0], dtype=torch.int64)
        for _ in range(iterations):
            local = torch.argmax(values @ centroids.T, dim=1)
            updated = []
            for cluster in range(subclasses_per_class):
                members = values[local == cluster]
                if members.shape[0] == 0:
                    assigned_similarity = (values * centroids[local]).sum(dim=1)
                    repair = int(torch.argmin(assigned_similarity))
                    center = values[repair]
                else:
                    center = members.mean(dim=0)
                updated.append(center / torch.linalg.vector_norm(center))
            centroids = torch.stack(updated)
        order = sorted(range(subclasses_per_class), key=lambda row: tuple(centroids[row].tolist()))
        remap = torch.empty(subclasses_per_class, dtype=torch.int64)
        for canonical, original in enumerate(order):
            remap[original] = canonical
        final_local = torch.argmax(values @ centroids.T, dim=1)
        if len(set(int(value) for value in final_local.tolist())) != subclasses_per_class:
            raise ValueError("subclass cluster population differs")
        assignments[indices] = remap[final_local] + offset
        parents.extend([class_id] * subclasses_per_class)
        offset += subclasses_per_class
    frozen_parents = tuple(parents)
    return SubclassAssignments(
        assignments=assignments,
        parents=frozen_parents,
        sha256=_subclass_assignment_sha256(
            assignments,
            frozen_parents,
            master_seed_sha256=master_seed_sha256,
            subclasses_per_class=subclasses_per_class,
            iterations=iterations,
            split_authority=split_authority,
        ),
    )


def train_cached_subclass_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    split_authority: FeatureSplitAuthority,
    initial_projection_weight: torch.Tensor,
    batches: tuple[tuple[int, ...], ...],
    master_seed_sha256: str,
    output_dimensions: int,
    subclasses_per_class: int,
    cluster_iterations: int,
    train_steps: int,
    projection_learning_rate: float,
    proxy_learning_rate: float,
    weight_decay: float,
    alpha: float,
    delta: float,
    device: torch.device | None = None,
) -> CachedHeadTrainingEvidence:
    """Train only a spectral projection and subclass proxies on cached features."""

    if (
        type(features) is not torch.Tensor
        or features.ndim != 2
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or labels.shape != (features.shape[0],)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or type(split_authority) is not FeatureSplitAuthority
        or type(initial_projection_weight) is not torch.Tensor
        or initial_projection_weight.device.type != "cpu"
        or initial_projection_weight.dtype != torch.float32
        or initial_projection_weight.shape != (output_dimensions, features.shape[1])
        or not bool(torch.isfinite(initial_projection_weight).all())
        or type(batches) is not tuple
        or len(batches) != train_steps
        or not batches
        or type(output_dimensions) is not int
        or not 1 <= output_dimensions <= min(features.shape)
        or type(train_steps) is not int
        or train_steps < 1
        or type(projection_learning_rate) is not float
        or projection_learning_rate <= 0.0
        or type(proxy_learning_rate) is not float
        or proxy_learning_rate <= 0.0
        or type(weight_decay) is not float
        or weight_decay < 0.0
        or type(alpha) is not float
        or alpha <= 0.0
        or type(delta) is not float
        or not 0.0 <= delta < 1.0
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("cached subclass head authority differs")
    split_authority.validated(features=features)
    logical_batch_size = len(batches[0])
    if logical_batch_size < 2 or any(
        type(batch) is not tuple
        or len(batch) != logical_batch_size
        or len(set(batch)) != len(batch)
        or any(type(row) is not int or not 0 <= row < features.shape[0] for row in batch)
        for batch in batches
    ):
        raise ValueError("cached subclass batch authority differs")
    gram = initial_projection_weight.double() @ initial_projection_weight.double().T
    if not torch.allclose(
        gram,
        torch.eye(output_dimensions, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-5,
    ):
        raise ValueError("cached subclass spectral initialization differs")
    if bool((torch.linalg.vector_norm(features.double(), dim=1) <= 0).any()):
        raise ValueError("cached subclass feature norm differs")
    assignments = cosine_subclass_assignments(
        features,
        labels,
        subclasses_per_class=subclasses_per_class,
        master_seed_sha256=master_seed_sha256,
        iterations=cluster_iterations,
        split_authority=split_authority,
    )
    execution_device = torch.device("cpu") if device is None else device
    if type(execution_device) is not torch.device or execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("cached subclass execution device differs")
    projection = nn.Linear(features.shape[1], output_dimensions, bias=False, dtype=torch.float32)
    with torch.no_grad():
        projection.weight.copy_(initial_projection_weight)
    projection = projection.to(execution_device)
    training_features = features.to(execution_device)
    training_labels = labels.to(execution_device)
    training_assignments = assignments.assignments.to(execution_device)
    parents = torch.tensor(assignments.parents, dtype=torch.int64, device=execution_device)

    with torch.no_grad():
        initial_embeddings = F.normalize(projection(training_features), dim=1)
        proxy_rows = []
        for subclass in range(len(assignments.parents)):
            members = initial_embeddings[training_assignments == subclass]
            if members.shape[0] == 0:
                raise ValueError("cached subclass proxy population differs")
            center = members.mean(dim=0)
            norm = torch.linalg.vector_norm(center)
            if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
                raise ValueError("cached subclass proxy norm differs")
            proxy_rows.append(center / norm)
        proxies = nn.Parameter(torch.stack(proxy_rows))

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [projection.weight],
                "lr": projection_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": [proxies],
                "lr": proxy_learning_rate,
                "weight_decay": weight_decay,
            },
        ]
    )

    def loss_value(rows: torch.Tensor | None = None) -> torch.Tensor:
        selected_features = training_features if rows is None else training_features[rows]
        selected_labels = training_labels if rows is None else training_labels[rows]
        selected_assignments = training_assignments if rows is None else training_assignments[rows]
        embeddings = F.normalize(projection(selected_features), dim=1)
        scores = embeddings @ F.normalize(proxies, dim=1).T
        return subclass_proxy_anchor_loss(
            scores,
            selected_labels,
            selected_assignments,
            parents,
            alpha=alpha,
            delta=delta,
        )

    with torch.no_grad():
        initial_loss = float(loss_value())
    trajectory = [initial_loss]
    trajectory_interval = max(1, train_steps // 10)
    for step, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        rows = torch.tensor(batch, dtype=torch.int64, device=execution_device)
        loss = loss_value(rows)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("cached subclass head loss became nonfinite")
        torch.autograd.backward(loss)
        gradient_norm = torch.nn.utils.clip_grad_norm_((projection.weight, proxies), 10.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("cached subclass head gradient became nonfinite")
        optimizer.step()
        if (step + 1) % trajectory_interval == 0 or step + 1 == train_steps:
            with torch.no_grad():
                trajectory.append(float(loss_value()))
    with torch.no_grad():
        final_loss = float(loss_value())
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise RuntimeError("cached subclass head evidence became nonfinite")
    return CachedHeadTrainingEvidence(
        split_authority=split_authority,
        subclass_assignments=assignments,
        initial_projection_weight=initial_projection_weight.detach().clone().contiguous(),
        projection_weight=projection.weight.detach().float().cpu().contiguous(),
        subclass_proxies=proxies.detach().float().cpu().contiguous(),
        initial_loss=initial_loss,
        final_loss=final_loss,
        train_steps=train_steps,
        logical_batch_size=logical_batch_size,
        loss_trajectory=tuple(trajectory),
    )


def principal_angles_degrees(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Measure ordered principal angles between two row spans."""

    if (
        type(left) is not torch.Tensor
        or type(right) is not torch.Tensor
        or left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[1]
        or left.shape[0] == 0
        or right.shape[0] == 0
        or left.shape[0] > left.shape[1]
        or right.shape[0] > right.shape[1]
        or not left.is_floating_point()
        or not right.is_floating_point()
        or left.device.type != "cpu"
        or right.device.type != "cpu"
        or not bool(torch.isfinite(left).all())
        or not bool(torch.isfinite(right).all())
    ):
        raise ValueError("principal-angle authority differs")
    if (
        int(torch.linalg.matrix_rank(left.double())) != left.shape[0]
        or int(torch.linalg.matrix_rank(right.double())) != right.shape[0]
    ):
        raise ValueError("principal-angle span rank differs")
    left_basis = torch.linalg.qr(left.double().T, mode="reduced").Q
    right_basis = torch.linalg.qr(right.double().T, mode="reduced").Q
    if left_basis.shape[1] != left.shape[0] or right_basis.shape[1] != right.shape[0]:
        raise ValueError("principal-angle span rank differs")
    cosines = torch.linalg.svdvals(left_basis.T @ right_basis).clamp(0.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosines))
    return angles.to(dtype=torch.float32)
