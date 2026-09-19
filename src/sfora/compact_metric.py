"""Compact supervised affine metric learning for frozen teacher embeddings."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from sfora.joint_relational_compaction import fixed_int8_unit_codes
from sfora.representation_ceiling import fit_centered_pca
from sfora.teacher_anchored_distillation import (
    class_balanced_anchor_schedule,
    positive_coverage_hard_negative_loss,
    stable_different_class_topk,
)


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
        digest.update(b"SFORA-COMPACT-METRIC-v1\0")
        digest.update(struct.pack("<QQ", *self._weight.shape))
        digest.update(self._weight.numpy().astype("<f4", copy=False).tobytes(order="C"))
        digest.update(self._bias.numpy().astype("<f4", copy=False).tobytes(order="C"))
        return digest.hexdigest()

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
            or bool((torch.linalg.vector_norm(embeddings, dim=1) == 0).any())
        ):
            raise ValueError("compact metric transform authority differs")
        with torch.autocast(device_type="cpu", enabled=False):
            normalized = torch.nn.functional.normalize(embeddings, dim=1)
            projected = normalized @ self._weight.T + self._bias
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


def fit_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CompactMetricConfig | None = None,
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
        or bool((torch.linalg.vector_norm(embeddings, dim=1) == 0).any())
        or type(labels) is not torch.Tensor
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or labels.shape != (embeddings.shape[0],)
        or not labels.is_contiguous()
        or bool((labels < 0).any())
        or resolved.output_dimensions > min(embeddings.shape[0] - 1, embeddings.shape[1])
        or (device is not None and type(device) is not torch.device)
    ):
        raise ValueError("compact metric training authority differs")
    destination = device or torch.device("cpu")
    if destination.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("compact metric training authority differs")

    with torch.autocast(device_type=destination.type, enabled=False):
        return _fit_compact_metric_projection(
            embeddings.detach(), labels, config=resolved, device=destination
        )


def _fit_compact_metric_projection(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    config: CompactMetricConfig,
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

    pca = fit_centered_pca(normalized, dimensions=resolved.output_dimensions)
    model = torch.nn.Linear(
        embeddings.shape[1], resolved.output_dimensions, device=destination, dtype=torch.float32
    )
    with torch.no_grad():
        model.weight.copy_(pca.components.to(destination))
        model.bias.copy_((-(pca.components.double() @ pca.mean.double())).float().to(destination))
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
