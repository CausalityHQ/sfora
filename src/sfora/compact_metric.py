"""Compact supervised affine metric learning for frozen teacher embeddings."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from sfora.joint_relational_compaction import fixed_int8_unit_codes
from sfora.representation_ceiling import fit_centered_pca
from sfora.teacher_anchored_distillation import (
    class_balanced_anchor_schedule,
    positive_coverage_hard_negative_loss,
    stable_different_class_topk,
)

_COMPACT_METRIC_ARTIFACT_MAGIC = b"SFORA-COMPACT-METRIC-v1\0"
_SHA256_BYTES = 32


@dataclass(frozen=True, slots=True)
class CompactMetricConfig:
    """Deterministic PCA-initialized affine metric-learning recipe."""

    output_dimensions: int = 128
    cycles: int = 4
    anchor_epochs_per_cycle: float = 5.366426295488848
    classes_per_update: int = 64
    rows_per_class: int = 2
    hard_negatives: int = 256
    learning_rate: float = 1e-3
    temperature: float = 0.05
    margin: float = 0.02
    anchor_weight: float = 10.0
    gradient_clip: float = 1.0
    mining_block_size: int = 128
    seed: int = 17

    def __post_init__(self) -> None:
        integers = (
            self.output_dimensions,
            self.cycles,
            self.classes_per_update,
            self.rows_per_class,
            self.hard_negatives,
            self.mining_block_size,
        )
        floats = (
            self.anchor_epochs_per_cycle,
            self.learning_rate,
            self.temperature,
            self.margin,
            self.anchor_weight,
            self.gradient_clip,
        )
        if (
            any(type(value) is not int or value < 1 for value in integers)
            or self.output_dimensions < 2
            or self.rows_per_class < 2
            or any(type(value) is not float or not math.isfinite(value) for value in floats)
            or self.anchor_epochs_per_cycle <= 0.0
            or self.learning_rate <= 0.0
            or self.temperature <= 0.0
            or self.margin < 0.0
            or self.anchor_weight < 0.0
            or self.gradient_clip <= 0.0
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("compact metric config differs")


@dataclass(frozen=True, init=False, slots=True)
class CompactMetricEncoder:
    """One affine projection followed by unit normalization and int8 coding."""

    _weight: torch.Tensor
    _bias: torch.Tensor

    def __init__(self, *, weight: torch.Tensor, bias: torch.Tensor) -> None:
        if (
            type(weight) is not torch.Tensor
            or weight.device.type != "cpu"
            or weight.dtype != torch.float32
            or weight.ndim != 2
            or min(weight.shape) < 2
            or not weight.is_contiguous()
            or not bool(torch.isfinite(weight).all())
            or type(bias) is not torch.Tensor
            or bias.device.type != "cpu"
            or bias.dtype != torch.float32
            or bias.shape != (weight.shape[0],)
            or not bias.is_contiguous()
            or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("compact metric encoder authority differs")
        object.__setattr__(self, "_weight", weight.detach().clone())
        object.__setattr__(self, "_bias", bias.detach().clone())

    @property
    def weight(self) -> torch.Tensor:
        """Return an isolated output-by-input affine weight."""

        return self._weight.clone()

    @property
    def bias(self) -> torch.Tensor:
        """Return an isolated affine bias."""

        return self._bias.clone()

    @property
    def sha256(self) -> str:
        """Return the digest of the canonical weight-then-bias float32 bytes."""

        digest = hashlib.sha256()
        digest.update(_COMPACT_METRIC_ARTIFACT_MAGIC)
        digest.update(struct.pack("<QQ", *self._weight.shape))
        digest.update(self._weight.numpy().astype("<f4", copy=False).tobytes(order="C"))
        digest.update(self._bias.numpy().astype("<f4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        """Persist the encoder in a strict authenticated little-endian format."""

        if not isinstance(path, Path):
            raise ValueError("compact metric artifact path differs")
        payload = bytearray(_COMPACT_METRIC_ARTIFACT_MAGIC)
        payload.extend(struct.pack("<QQ", *self._weight.shape))
        payload.extend(self._weight.numpy().astype("<f4", copy=False).tobytes(order="C"))
        payload.extend(self._bias.numpy().astype("<f4", copy=False).tobytes(order="C"))
        payload.extend(hashlib.sha256(payload).digest())
        path.write_bytes(payload)

    @classmethod
    def load(cls, path: Path) -> CompactMetricEncoder:
        """Load an encoder only when its framing, length, and digest are exact."""

        if not isinstance(path, Path):
            raise ValueError("compact metric artifact path differs")
        artifact = path.read_bytes()
        header_bytes = len(_COMPACT_METRIC_ARTIFACT_MAGIC) + 16
        if len(artifact) < header_bytes + _SHA256_BYTES:
            raise ValueError("compact metric artifact differs")
        payload = artifact[:-_SHA256_BYTES]
        if (
            not payload.startswith(_COMPACT_METRIC_ARTIFACT_MAGIC)
            or hashlib.sha256(payload).digest() != artifact[-_SHA256_BYTES:]
        ):
            raise ValueError("compact metric artifact differs")
        output_dimensions, input_dimensions = struct.unpack(
            "<QQ", payload[len(_COMPACT_METRIC_ARTIFACT_MAGIC) : header_bytes]
        )
        expected_bytes = header_bytes + 4 * (
            output_dimensions * input_dimensions + output_dimensions
        )
        if output_dimensions < 2 or input_dimensions < 2 or len(payload) != expected_bytes:
            raise ValueError("compact metric artifact differs")
        values = np.frombuffer(payload, dtype="<f4", offset=header_bytes).copy()
        split = output_dimensions * input_dimensions
        weight = torch.from_numpy(values[:split].reshape(output_dimensions, input_dimensions))
        bias = torch.from_numpy(values[split:])
        try:
            return cls(weight=weight, bias=bias)
        except ValueError as error:
            raise ValueError("compact metric artifact differs") from error

    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project finite nonzero CPU float32 embeddings into the unit code space."""

        if (
            type(embeddings) is not torch.Tensor
            or embeddings.device.type != "cpu"
            or embeddings.dtype != torch.float32
            or embeddings.ndim != 2
            or embeddings.shape[0] < 1
            or embeddings.shape[1] != self._weight.shape[1]
            or not embeddings.is_contiguous()
            or not bool(torch.isfinite(embeddings).all())
            or bool((torch.linalg.vector_norm(embeddings, dim=1) < 1e-12).any())
        ):
            raise ValueError("compact metric transform authority differs")
        with torch.autocast(device_type="cpu", enabled=False):
            normalized = torch.nn.functional.normalize(embeddings, dim=1)
            projected = torch.nn.functional.linear(
                normalized,
                self._weight,
                self._bias,
            )
            return _normalize_projected(projected)

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return exactly one signed byte per output dimension."""

        codes, _restored = fixed_int8_unit_codes(self.transform(embeddings))
        return codes

    def to_module(self, device: torch.device) -> CompactMetricModule:
        """Build a frozen module for repeated inference beside a backbone."""

        if type(device) is not torch.device:
            raise ValueError("compact metric module authority differs")
        return CompactMetricModule(self._weight, self._bias, device=device)


class CompactMetricModule(torch.nn.Module):
    """Frozen fast path for trusted finite nonzero backbone embeddings."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, *, device: torch.device) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(
            weight.shape[1], weight.shape[0], device=device, dtype=torch.float32
        )
        with torch.no_grad():
            self.linear.weight.copy_(weight.to(device))
            self.linear.bias.copy_(bias.to(device))
        self.requires_grad_(False)
        self.eval()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            normalized = torch.nn.functional.normalize(embeddings.float(), dim=-1)
            return torch.nn.functional.normalize(self.linear(normalized), dim=-1)

    def encode(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return exactly one signed byte per output dimension on this device."""

        projected = self(embeddings)
        return torch.round(projected * 127.0).clamp(-127, 127).to(torch.int8).contiguous()


@dataclass(frozen=True, slots=True)
class CompactMetricFitResult:
    """Learned encoder and deterministic schedule evidence."""

    encoder: CompactMetricEncoder
    eligible_class_count: int
    classes_per_update: int
    updates_per_cycle: int
    total_updates: int
    effective_hard_negatives: int
    schedule_sha256: str
    parameter_sha256: str
    losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class WithinClassWhiteningFitResult:
    """Experimental opt-in whitening encoder and covariance evidence."""

    encoder: CompactMetricEncoder
    shrinkage: float
    within_minimum_eigenvalue: float
    within_maximum_eigenvalue: float


@dataclass(frozen=True, slots=True)
class PowerWhiteningFitResult:
    """Explicit covariance-power projection and its fit-only authority."""

    encoder: CompactMetricEncoder
    alpha: float
    regularization: float
    output_dimensions: int
    within_minimum_eigenvalue: float
    within_maximum_eigenvalue: float


@dataclass(frozen=True, slots=True)
class CompactMetricSelectionFold:
    """One unseen-class fold from fit-only compact metric selection."""

    fold: int
    training_class_count: int
    training_row_count: int
    validation_class_count: int
    validation_row_count: int
    learned_map_at_r: float
    pca_map_at_r: float
    learned_recall_at_1: float
    pca_recall_at_1: float


@dataclass(frozen=True, slots=True)
class CompactMetricSelectionResult:
    """Selected full-fit encoder and its class-disjoint fit-only evidence."""

    encoder: CompactMetricEncoder
    selected: Literal["learned_projection", "pca_fallback"]
    map_at_r_delta: float
    recall_at_1_delta: float
    folds: tuple[CompactMetricSelectionFold, ...]


def choose_compact_metric_projection(
    *,
    map_at_r_delta: float,
    recall_at_1_delta: float,
    minimum_map_gain: float = 0.003,
) -> Literal["learned_projection", "pca_fallback"]:
    """Choose the learned code only when fit-only retrieval clears both gates."""

    if (
        type(map_at_r_delta) is not float
        or type(recall_at_1_delta) is not float
        or type(minimum_map_gain) is not float
        or not math.isfinite(map_at_r_delta)
        or not math.isfinite(recall_at_1_delta)
        or not math.isfinite(minimum_map_gain)
        or minimum_map_gain < 0.0
    ):
        raise ValueError("compact metric selection authority differs")
    if map_at_r_delta >= minimum_map_gain and recall_at_1_delta >= 0.0:
        return "learned_projection"
    return "pca_fallback"


def select_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CompactMetricConfig | None = None,
    minimum_map_gain: float = 0.003,
    device: torch.device | None = None,
) -> CompactMetricSelectionResult:
    """Select learned affine metric or PCA using three unseen-class fit folds."""

    resolved = config or CompactMetricConfig()
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] < 6
        or embeddings.shape[1] < 2
        or not embeddings.is_contiguous()
        or not bool(torch.isfinite(embeddings).all())
        or bool((torch.linalg.vector_norm(embeddings, dim=1) < 1e-12).any())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or not labels.is_contiguous()
        or bool((labels < 0).any())
        or type(minimum_map_gain) is not float
        or not math.isfinite(minimum_map_gain)
        or minimum_map_gain < 0.0
        or (device is not None and type(device) is not torch.device)
    ):
        raise ValueError("compact metric selector authority differs")
    destination = device or torch.device("cpu")
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("compact metric selector authority differs")

    with (
        torch.inference_mode(False),
        torch.enable_grad(),
        torch.autocast(device_type="cpu", enabled=False),
    ):
        return _select_compact_metric_projection(
            embeddings.detach(),
            labels,
            resolved=resolved,
            minimum_map_gain=minimum_map_gain,
            destination=destination,
        )


def _select_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    resolved: CompactMetricConfig,
    minimum_map_gain: float,
    destination: torch.device,
) -> CompactMetricSelectionResult:
    """Run selection with caller autograd and CPU autocast disabled."""

    normalized = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    label_array = labels.numpy()
    label_values = tuple(int(value) for value in np.unique(label_array))
    counts = {value: int((label_array == value).sum()) for value in label_values}
    eligible_values = tuple(value for value in label_values if counts[value] >= 2)
    label_folds = {value: _compact_metric_fold(value) for value in eligible_values}
    if len(eligible_values) < 3:
        raise ValueError("compact metric selector requires at least three eligible classes")
    missing_folds = {0, 1, 2} - set(label_folds.values())
    if missing_folds:
        raise ValueError(
            "compact metric selector has no eligible classes in deterministic fold "
            f"{min(missing_folds)}"
        )
    eligible_mask = np.isin(label_array, np.asarray(eligible_values, dtype=label_array.dtype))

    folds: list[CompactMetricSelectionFold] = []
    learned_ap: list[float] = []
    pca_ap: list[float] = []
    learned_r1: list[float] = []
    pca_r1: list[float] = []
    for fold in range(3):
        validation_mask = np.asarray(
            [
                bool(eligible) and label_folds[int(value)] == fold
                for value, eligible in zip(label_array, eligible_mask, strict=True)
            ],
            dtype=np.bool_,
        )
        training_mask = eligible_mask & ~validation_mask
        training_labels = labels[torch.from_numpy(training_mask)].contiguous()
        validation_labels = labels[torch.from_numpy(validation_mask)].contiguous()
        if len(torch.unique(training_labels)) < 2 or len(torch.unique(validation_labels)) < 1:
            raise ValueError("compact metric selector authority differs")
        training = normalized[torch.from_numpy(training_mask)].contiguous()
        validation = normalized[torch.from_numpy(validation_mask)].contiguous()
        learned = fit_compact_metric_projection(
            training,
            training_labels,
            config=resolved,
            device=destination,
        ).encoder.encode(validation)
        pca = fit_centered_pca(training, dimensions=resolved.output_dimensions)
        pca_projection = _normalize_projected(
            (validation - pca.mean.float()) @ pca.components.T.float()
        )
        pca_codes, _restored = fixed_int8_unit_codes(pca_projection)
        learned_score = _score_compact_metric_codes(learned, validation_labels, device=destination)
        pca_score = _score_compact_metric_codes(pca_codes, validation_labels, device=destination)
        learned_ap.extend(learned_score[2])
        pca_ap.extend(pca_score[2])
        learned_r1.extend(learned_score[3])
        pca_r1.extend(pca_score[3])
        folds.append(
            CompactMetricSelectionFold(
                fold=fold,
                training_class_count=len(torch.unique(training_labels)),
                training_row_count=len(training),
                validation_class_count=len(torch.unique(validation_labels)),
                validation_row_count=len(validation),
                learned_map_at_r=learned_score[0],
                pca_map_at_r=pca_score[0],
                learned_recall_at_1=learned_score[1],
                pca_recall_at_1=pca_score[1],
            )
        )

    map_delta = math.fsum(a - b for a, b in zip(learned_ap, pca_ap, strict=True)) / len(learned_ap)
    recall_delta = math.fsum(a - b for a, b in zip(learned_r1, pca_r1, strict=True)) / len(
        learned_r1
    )
    selected = choose_compact_metric_projection(
        map_at_r_delta=map_delta,
        recall_at_1_delta=recall_delta,
        minimum_map_gain=minimum_map_gain,
    )
    if selected == "learned_projection":
        encoder = fit_compact_metric_projection(
            normalized,
            labels,
            config=resolved,
            device=destination,
        ).encoder
    else:
        pca = fit_centered_pca(normalized, dimensions=resolved.output_dimensions)
        encoder = CompactMetricEncoder(
            weight=pca.components,
            bias=(-(pca.components.double() @ pca.mean.double())).float().contiguous(),
        )
    return CompactMetricSelectionResult(
        encoder=encoder,
        selected=selected,
        map_at_r_delta=map_delta,
        recall_at_1_delta=recall_delta,
        folds=tuple(folds),
    )


def fit_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CompactMetricConfig | None = None,
    initial_encoder: CompactMetricEncoder | None = None,
    device: torch.device | None = None,
) -> CompactMetricFitResult:
    """Fit the compact affine metric recipe on labeled embeddings."""

    resolved = config or CompactMetricConfig()
    if (
        type(embeddings) is not torch.Tensor
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] < 2
        or embeddings.shape[1] < 2
        or not embeddings.is_contiguous()
        or not bool(torch.isfinite(embeddings).all())
        or bool((torch.linalg.vector_norm(embeddings, dim=1) < 1e-12).any())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or not labels.is_contiguous()
        or bool((labels < 0).any())
        or resolved.output_dimensions > min(embeddings.shape[0] - 1, embeddings.shape[1])
        or (
            initial_encoder is not None
            and (
                type(initial_encoder) is not CompactMetricEncoder
                or initial_encoder.weight.shape != (resolved.output_dimensions, embeddings.shape[1])
            )
        )
        or (device is not None and type(device) is not torch.device)
    ):
        raise ValueError("compact metric training authority differs")
    destination = device or torch.device("cpu")
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("compact metric training authority differs")

    with (
        torch.inference_mode(False),
        torch.enable_grad(),
        torch.autocast(device_type=destination.type, enabled=False),
    ):
        return _fit_compact_metric_projection(
            embeddings.detach(),
            labels,
            config=resolved,
            initial_encoder=initial_encoder,
            device=destination,
        )


def _fit_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CompactMetricConfig,
    initial_encoder: CompactMetricEncoder | None,
    device: torch.device,
) -> CompactMetricFitResult:
    resolved = config
    destination = device

    normalized = torch.nn.functional.normalize(embeddings, dim=1).contiguous()
    label_array = labels.numpy()
    groups = {
        int(label): np.flatnonzero(label_array == int(label)).astype(np.int64)
        for label in np.unique(label_array)
    }
    eligible = {
        label: rows for label, rows in groups.items() if len(rows) >= resolved.rows_per_class
    }
    if len(eligible) < 2:
        raise ValueError("compact metric training authority differs")
    classes_per_update = min(resolved.classes_per_update, len(eligible))
    eligible_rows = sum(len(rows) for rows in eligible.values())
    updates_per_cycle = math.ceil(
        resolved.anchor_epochs_per_cycle
        * eligible_rows
        / (classes_per_update * resolved.rows_per_class)
    )
    schedule = class_balanced_anchor_schedule(
        label_array,
        seed=resolved.seed,
        updates=updates_per_cycle,
        classes_per_update=classes_per_update,
        rows_per_class=resolved.rows_per_class,
    )

    model = torch.nn.Linear(
        embeddings.shape[1], resolved.output_dimensions, device=destination, dtype=torch.float32
    )
    with torch.no_grad():
        if initial_encoder is None:
            pca = fit_centered_pca(normalized, dimensions=resolved.output_dimensions)
            model.weight.copy_(pca.components.to(destination))
            model.bias.copy_(
                (-(pca.components.double() @ pca.mean.double())).float().to(destination)
            )
        else:
            model.weight.copy_(initial_encoder.weight.to(destination))
            model.bias.copy_(initial_encoder.bias.to(destination))
    bank = normalized.to(destination)
    with torch.inference_mode():
        start = torch.nn.functional.normalize(model(bank), dim=1).detach().contiguous()
    max_group = max(len(groups[label]) for label in eligible)
    hard_negatives = min(resolved.hard_negatives, len(labels) - max_group)
    if hard_negatives < 1:
        raise ValueError("compact metric training authority differs")
    frozen_negatives = _blockwise_hard_negatives(
        start,
        labels.to(destination),
        k=hard_negatives,
        block_size=resolved.mining_block_size,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=resolved.learning_rate * math.sqrt(resolved.output_dimensions / embeddings.shape[1]),
    )
    losses: list[float] = []
    rows = tuple(row.copy() for row in schedule.row_indexes)
    for _cycle in range(resolved.cycles):
        for anchors_numpy in rows:
            anchors = torch.from_numpy(anchors_numpy).to(destination)
            positive_rows, positive_mask = _positive_rows(anchors_numpy, label_array, eligible)
            positives = torch.from_numpy(positive_rows).to(destination)
            mask = torch.from_numpy(positive_mask).to(destination).contiguous()
            negatives = frozen_negatives[anchors]
            anchor_codes = torch.nn.functional.normalize(model(bank[anchors]), dim=1)
            positive_codes = torch.nn.functional.normalize(
                model(bank[positives.reshape(-1)]), dim=1
            ).reshape(len(anchors), positives.shape[1], -1)
            negative_codes = torch.nn.functional.normalize(
                model(bank[negatives.reshape(-1)]), dim=1
            ).reshape(len(anchors), negatives.shape[1], -1)
            loss = positive_coverage_hard_negative_loss(
                torch.einsum("bd,bpd->bp", anchor_codes, positive_codes).contiguous(),
                mask,
                torch.einsum("bd,bnd->bn", anchor_codes, negative_codes).contiguous(),
                torch.einsum("bd,bd->b", anchor_codes, start[anchors]).contiguous(),
                temperature=resolved.temperature,
                margin=resolved.margin,
                anchor_weight=resolved.anchor_weight,
                positive_aggregation="mean_logit",
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))

    encoder = CompactMetricEncoder(
        weight=model.weight.detach().cpu().contiguous(),
        bias=model.bias.detach().cpu().contiguous(),
    )
    return CompactMetricFitResult(
        encoder=encoder,
        eligible_class_count=len(eligible),
        classes_per_update=classes_per_update,
        updates_per_cycle=updates_per_cycle,
        total_updates=resolved.cycles * updates_per_cycle,
        effective_hard_negatives=hard_negatives,
        schedule_sha256=schedule.sha256,
        parameter_sha256=encoder.sha256,
        losses=tuple(losses),
    )


def fit_within_class_whitening_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    output_dimensions: int = 128,
) -> WithinClassWhiteningFitResult:
    """Fit an experimental shrinkage-whitened projection, failing on ambiguous spectra."""

    if (
        type(embeddings) is not torch.Tensor
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] < 3
        or embeddings.shape[1] < 2
        or not embeddings.is_contiguous()
        or not bool(torch.isfinite(embeddings).all())
        or bool((torch.linalg.vector_norm(embeddings, dim=1) < 1e-12).any())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or not labels.is_contiguous()
        or bool((labels < 0).any())
        or len(torch.unique(labels)) < 2
        or type(output_dimensions) is not int
        or not 2 <= output_dimensions <= min(embeddings.shape[0] - 1, embeddings.shape[1])
    ):
        raise ValueError("within-class whitening authority differs")

    normalized = torch.nn.functional.normalize(embeddings.detach().double(), dim=1).numpy()
    label_array = labels.numpy()
    values = np.unique(label_array)
    mean = normalized.mean(axis=0)
    class_means = np.stack([normalized[label_array == value].mean(axis=0) for value in values])
    residuals = np.concatenate(
        [
            normalized[label_array == value] - class_means[index]
            for index, value in enumerate(values)
        ]
    )
    estimator = LedoitWolf(assume_centered=True, store_precision=False).fit(residuals)
    within = np.asarray(estimator.covariance_, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0] <= 0.0:
        raise ValueError("within-class whitening covariance differs")
    inverse_root = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    centered = normalized - mean
    whitened = centered @ inverse_root
    total_white = whitened.T @ whitened / len(whitened)
    variance, directions = np.linalg.eigh(total_white)
    order = np.argsort(variance, kind="stable")[::-1]
    retained_boundary = min(output_dimensions + 1, len(variance))
    boundary_values = variance[order[:retained_boundary]]
    selected_values = boundary_values[:output_dimensions]
    tolerance = max(normalized.shape) * np.finfo(np.float64).eps * max(float(variance[-1]), 1.0)
    if (
        not np.isfinite(variance).all()
        or selected_values[-1] <= tolerance
        or any(
            abs(float(boundary_values[index] - boundary_values[index + 1])) <= tolerance
            for index in range(len(boundary_values) - 1)
        )
    ):
        raise ValueError("within-class whitening projection differs")
    selected = directions[:, order[:output_dimensions]].copy()
    for direction in selected.T:
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
    weight = np.ascontiguousarray((selected.T @ inverse_root).astype(np.float32))
    bias = np.ascontiguousarray((-(weight.astype(np.float64) @ mean)).astype(np.float32))
    encoder = CompactMetricEncoder(weight=torch.from_numpy(weight), bias=torch.from_numpy(bias))
    return WithinClassWhiteningFitResult(
        encoder=encoder,
        shrinkage=float(estimator.shrinkage_),
        within_minimum_eigenvalue=float(eigenvalues[0]),
        within_maximum_eigenvalue=float(eigenvalues[-1]),
    )


def fit_power_whitening_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    regularization: float,
    output_dimensions: int = 128,
) -> PowerWhiteningFitResult:
    """Fit one explicit member of the replicated covariance-power family."""

    if (
        type(embeddings) is not torch.Tensor
        or embeddings.device.type != "cpu"
        or embeddings.dtype != torch.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] < 3
        or embeddings.shape[1] < 2
        or not embeddings.is_contiguous()
        or not bool(torch.isfinite(embeddings).all())
        or bool((torch.linalg.vector_norm(embeddings, dim=1) < 1e-12).any())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or not labels.is_contiguous()
        or bool((labels < 0).any())
        or len(torch.unique(labels)) < 2
        or type(alpha) is not float
        or not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
        or type(regularization) is not float
        or not math.isfinite(regularization)
        or regularization < 0.0
        or type(output_dimensions) is not int
        or not 2 <= output_dimensions <= min(embeddings.shape[0] - 1, embeddings.shape[1])
    ):
        raise ValueError("power whitening authority differs")

    normalized = torch.nn.functional.normalize(embeddings.detach().double(), dim=1).numpy()
    label_array = labels.numpy()
    mean = normalized.mean(axis=0)
    centered = normalized - mean
    residuals = np.empty_like(normalized)
    for label in np.unique(label_array):
        mask = label_array == label
        residuals[mask] = normalized[mask] - normalized[mask].mean(axis=0)
    within = residuals.T @ residuals / len(residuals)
    eigenvalues, eigenvectors = np.linalg.eigh(within)
    if eigenvalues[0] <= 0.0 or not np.isfinite(eigenvalues).all():
        raise ValueError("power whitening covariance differs")
    scale = float(np.trace(within) / within.shape[0])
    adjusted = eigenvalues + regularization * scale
    powers = np.power(adjusted, -alpha)
    if not np.isfinite(powers).all():
        raise ValueError("power whitening covariance differs")
    transform = (eigenvectors * powers) @ eigenvectors.T
    transformed = centered @ transform
    total = transformed.T @ transformed / len(transformed)
    variance, directions = np.linalg.eigh(total)
    order = np.argsort(variance, kind="stable")[::-1]
    retained_boundary = min(output_dimensions + 1, len(variance))
    boundary_values = variance[order[:retained_boundary]]
    tolerance = max(normalized.shape) * np.finfo(np.float64).eps * max(float(variance[-1]), 1.0)
    if (
        not np.isfinite(variance).all()
        or boundary_values[output_dimensions - 1] <= tolerance
        or any(
            abs(float(boundary_values[index] - boundary_values[index + 1])) <= tolerance
            for index in range(len(boundary_values) - 1)
        )
    ):
        raise ValueError("power whitening projection differs")
    selected = directions[:, order[:output_dimensions]].copy()
    for direction in selected.T:
        pivot = int(np.argmax(np.abs(direction)))
        if direction[pivot] < 0.0:
            direction *= -1.0
    weight64 = selected.T @ transform
    weight = np.ascontiguousarray(weight64.astype(np.float32))
    bias = np.ascontiguousarray((-(weight64 @ mean)).astype(np.float32))
    return PowerWhiteningFitResult(
        encoder=CompactMetricEncoder(weight=torch.from_numpy(weight), bias=torch.from_numpy(bias)),
        alpha=alpha,
        regularization=regularization,
        output_dimensions=output_dimensions,
        within_minimum_eigenvalue=float(eigenvalues[0]),
        within_maximum_eigenvalue=float(eigenvalues[-1]),
    )


def _normalize_projected(projected: torch.Tensor) -> torch.Tensor:
    norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True)
    if bool((~torch.isfinite(projected)).any()) or bool((~torch.isfinite(norms)).any()):
        raise ValueError("compact metric transform geometry differs")
    if bool((norms < 1e-12).any()):
        raise ValueError("compact metric transform geometry differs")
    normalized = projected / norms
    if bool((~torch.isfinite(normalized)).any()):
        raise ValueError("compact metric transform geometry differs")
    return cast(torch.Tensor, normalized.contiguous())


def _blockwise_hard_negatives(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    k: int,
    block_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(embeddings), block_size):
        stop = min(start + block_size, len(embeddings))
        parts.append(
            stable_different_class_topk(
                embeddings[start:stop],
                embeddings,
                labels[start:stop],
                labels,
                k=k,
            )
        )
    return torch.cat(parts).contiguous()


def _positive_rows(
    anchors: np.ndarray,
    labels: np.ndarray,
    groups: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        groups[int(labels[anchor])][groups[int(labels[anchor])] != anchor] for anchor in anchors
    ]
    width = max(len(value) for value in rows)
    padded = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.bool_)
    for index, value in enumerate(rows):
        padded[index, : len(value)] = value
        mask[index, : len(value)] = True
    return padded, mask


def _compact_metric_fold(label: int) -> int:
    digest = hashlib.sha256(f"compact-selector-v1:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % 3


def _score_compact_metric_codes(
    codes: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[float, float, tuple[float, ...], tuple[float, ...]]:
    label_values = tuple(int(value) for value in labels.tolist())
    counts = {value: label_values.count(value) for value in set(label_values)}
    if any(count < 2 for count in counts.values()):
        raise ValueError("compact metric selector authority differs")
    width = max(counts.values()) - 1
    rankings = []
    with torch.autocast(device_type=device.type, enabled=False), torch.inference_mode():
        gallery = torch.nn.functional.normalize(codes.float(), dim=1).to(device)
        for start in range(0, len(gallery), 256):
            stop = min(start + 256, len(gallery))
            scores = gallery[start:stop] @ gallery.T
            rows = torch.arange(stop - start, device=device)
            scores[rows, torch.arange(start, stop, device=device)] = -torch.inf
            rankings.append(_compact_metric_lexicographic_topk(scores, width).cpu())
    average_precision = []
    recall_at_1 = []
    for ranking, label in zip(torch.cat(rankings).tolist(), label_values, strict=True):
        positives = counts[label] - 1
        found = 0
        terms = []
        for rank, row in enumerate(ranking[:positives], start=1):
            if label_values[row] == label:
                found += 1
                terms.append(found / rank)
        average_precision.append(math.fsum(terms) / positives)
        recall_at_1.append(float(label_values[ranking[0]] == label))
    return (
        math.fsum(average_precision) / len(average_precision),
        math.fsum(recall_at_1) / len(recall_at_1),
        tuple(average_precision),
        tuple(recall_at_1),
    )


def _compact_metric_lexicographic_topk(scores: torch.Tensor, width: int) -> torch.Tensor:
    retained = min(width + 1, scores.shape[1])
    values, indexes = torch.topk(scores, k=retained, dim=1, largest=True, sorted=False)
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    indexes = indexes.gather(1, score_order)
    values = values.gather(1, score_order)
    if retained > width:
        ambiguous = values[:, width - 1] == values[:, width]
        for row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
            boundary = values[row, width - 1]
            candidates = torch.nonzero(scores[row] >= boundary, as_tuple=False).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = scores[row, candidates]
            order = torch.argsort(candidate_values, descending=True, stable=True)
            indexes[row, :width] = candidates[order[:width]]
    return indexes[:, :width]
