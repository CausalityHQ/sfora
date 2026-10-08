"""Shared compact encoders trained from teacher neighborhood relations."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sfora.packed_int8 import (
    _PACKED_INT8_ARTIFACT_MAGIC as _PACKED_INT8_ARTIFACT_MAGIC,
    _SHA256_BYTES as _SHA256_BYTES,
    PackedInt8Embeddings as PackedInt8Embeddings,
    _unit_rows,
    fixed_int8_unit_codes as fixed_int8_unit_codes,
    pack_int8_unit_embeddings as pack_int8_unit_embeddings,
)


def _validate_basis(basis: torch.Tensor) -> None:
    if (
        type(basis) is not torch.Tensor
        or basis.device.type != "cpu"
        or basis.dtype != torch.float32
        or basis.ndim != 2
        or basis.shape[0] < 2
        or basis.shape[1] <= basis.shape[0]
        or not bool(torch.isfinite(basis).all())
    ):
        raise ValueError("joint relational basis authority differs")


def _validate_input(value: torch.Tensor, dimensions: int) -> None:
    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] != dimensions
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("joint relational input authority differs")


def _fit_uncentered_covariance_basis(descriptors: torch.Tensor, *, dimensions: int) -> torch.Tensor:
    """Fit leading uncentered axes without depending on experimental modules."""

    if (
        type(descriptors) is not torch.Tensor
        or descriptors.dtype != torch.float32
        or descriptors.device.type != "cpu"
        or descriptors.ndim != 2
        or descriptors.shape[0] < 2
        or not bool(torch.isfinite(descriptors).all())
        or type(dimensions) is not int
        or not 1 < dimensions < descriptors.shape[1]
    ):
        raise ValueError("relational linear covariance authority differs")
    values = descriptors.detach().double()
    covariance = values.T @ values
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[-1]) <= 0.0:
        raise ValueError("relational linear covariance geometry differs")
    return cast(torch.Tensor, eigenvectors[:, -dimensions:].T.flip(0).float().contiguous())


class RelationalLinearEncoder(nn.Module):
    """One shared bias-free projection optimized for neighborhood relations."""

    def __init__(self, basis: torch.Tensor) -> None:
        super().__init__()
        _validate_basis(basis)
        self.projection = nn.Linear(basis.shape[1], basis.shape[0], bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(basis)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_input(value, self.projection.in_features)
        encoded = self.projection(value)
        norms = torch.linalg.vector_norm(encoded, dim=1)
        if not bool(torch.isfinite(encoded).all()) or bool((norms <= 1e-8).any()):
            raise ValueError("joint relational output authority differs")
        return F.normalize(encoded, dim=1)

    def to_bytes(self) -> bytes:
        """Serialize dimensions and little-endian float32 weights canonically."""

        weight = self.projection.weight.detach().cpu().contiguous().numpy()
        header = b"SFORA-RL1" + struct.pack("<II", weight.shape[0], weight.shape[1])
        return header + weight.astype("<f4", copy=False).tobytes(order="C")

    @classmethod
    def from_bytes(cls, wire: bytes) -> RelationalLinearEncoder:
        """Restore one canonical relational-linear encoder artifact."""

        if type(wire) is not bytes or len(wire) < 17 or wire[:9] != b"SFORA-RL1":
            raise ValueError("relational linear encoder byte authority differs")
        output_dimensions, input_dimensions = struct.unpack("<II", wire[9:17])
        expected = 17 + output_dimensions * input_dimensions * 4
        if len(wire) != expected or output_dimensions < 2 or input_dimensions <= output_dimensions:
            raise ValueError("relational linear encoder byte authority differs")
        values = np.frombuffer(wire, dtype="<f4", offset=17).astype(np.float32, copy=True)
        basis = torch.from_numpy(values.reshape(output_dimensions, input_dimensions))
        return cls(basis).eval()


@dataclass(frozen=True, slots=True)
class RelationalLinearTrainingConfig:
    """Frozen optimizer recipe for label-free relational compaction."""

    batch_size: int = 1024
    epochs: int = 20
    learning_rate: float = 1e-4
    seed: int = 17
    temperature: float = 0.05
    weight_decay: float = 1e-4
    updates: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.batch_size) is not int
            or self.batch_size < 3
            or type(self.epochs) is not int
            or self.epochs < 1
            or type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or type(self.seed) is not int
            or self.seed < 0
            or type(self.temperature) is not float
            or not math.isfinite(self.temperature)
            or self.temperature <= 0.0
            or (self.updates is not None and (type(self.updates) is not int or self.updates < 1))
            or type(self.weight_decay) is not float
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("relational linear training config differs")


class JointRelationalEncoder(nn.Module):
    """PCA-initialized shared encoder with a zero-initialized nonlinear residual."""

    def __init__(self, basis: torch.Tensor, *, hidden_dimensions: int, seed: int) -> None:
        super().__init__()
        _validate_basis(basis)
        if (
            type(hidden_dimensions) is not int
            or hidden_dimensions < 2
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError("joint relational basis authority differs")
        self.primary = nn.Linear(basis.shape[1], basis.shape[0], bias=False)
        self.residual_input = nn.Linear(basis.shape[1], hidden_dimensions, bias=False)
        self.residual_output = nn.Linear(hidden_dimensions, basis.shape[0], bias=False)
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.primary.weight.copy_(basis)
            nn.init.kaiming_uniform_(
                self.residual_input.weight, a=math.sqrt(5), generator=generator
            )
            self.residual_output.weight.zero_()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _validate_input(value, self.primary.in_features)
        encoded = self.primary(value) + self.residual_output(F.relu(self.residual_input(value)))
        norms = torch.linalg.vector_norm(encoded, dim=1)
        if not bool(torch.isfinite(encoded).all()) or bool((norms <= 1e-8).any()):
            raise ValueError("joint relational output authority differs")
        return F.normalize(encoded, dim=1)


def neighborhood_distribution_kl(
    student: torch.Tensor, teacher: torch.Tensor, *, temperature: float
) -> torch.Tensor:
    """Match off-diagonal teacher neighborhood probabilities."""

    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.ndim != 2
        or teacher.ndim != 2
        or student.shape[0] != teacher.shape[0]
        or student.shape[0] < 2
        or student.device != teacher.device
        or not _unit_rows(student)
        or not _unit_rows(teacher)
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError("joint relational relation authority differs")
    student_scores = student @ student.T / temperature
    teacher_scores = teacher @ teacher.T / temperature
    off_diagonal = ~torch.eye(len(student), dtype=torch.bool, device=student.device)
    student_scores = student_scores[off_diagonal].reshape(len(student), -1)
    teacher_scores = teacher_scores[off_diagonal].reshape(len(teacher), -1)
    return F.kl_div(
        F.log_softmax(student_scores, dim=1),
        F.softmax(teacher_scores, dim=1),
        reduction="batchmean",
    )


def fit_relational_linear_encoder(
    source: torch.Tensor,
    teacher: torch.Tensor,
    basis: torch.Tensor,
    *,
    config: RelationalLinearTrainingConfig,
    device: torch.device,
) -> tuple[RelationalLinearEncoder, tuple[float, ...]]:
    """Fit one shared linear encoder to unlabeled teacher neighborhood relations."""

    _validate_basis(basis)
    if type(config) is not RelationalLinearTrainingConfig or type(device) is not torch.device:
        raise ValueError("relational linear training authority differs")
    if (
        type(source) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or source.device.type != "cpu"
        or teacher.device.type != "cpu"
        or source.dtype != torch.float32
        or teacher.dtype != torch.float32
        or source.ndim != 2
        or teacher.ndim != 2
        or source.shape[0] != teacher.shape[0]
        or source.shape[0] < config.batch_size
        or source.shape[1] != basis.shape[1]
        or not bool(torch.isfinite(source).all())
        or not bool(torch.isfinite(teacher).all())
        or bool((torch.linalg.vector_norm(source, dim=1) <= 1e-12).any())
        or bool((torch.linalg.vector_norm(teacher, dim=1) <= 1e-12).any())
    ):
        raise ValueError("relational linear training authority differs")
    source_unit = F.normalize(source.detach(), dim=1)
    teacher_unit = F.normalize(teacher.detach(), dim=1)
    model = RelationalLinearEncoder(basis).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        foreach=False,
    )
    steps_per_epoch = len(source) // config.batch_size
    maximum_updates = config.epochs * steps_per_epoch
    update_count = maximum_updates if config.updates is None else config.updates
    if update_count > maximum_updates:
        raise ValueError("relational linear training authority differs")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=update_count)
    generator = torch.Generator().manual_seed(config.seed)
    losses = []
    completed_updates = 0
    for _epoch in range(config.epochs):
        order = torch.randperm(len(source), generator=generator)[
            : steps_per_epoch * config.batch_size
        ]
        epoch_losses = []
        for start in range(0, len(order), config.batch_size):
            if completed_updates == update_count:
                break
            indexes = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            encoded = model(source_unit[indexes].to(device))
            loss = neighborhood_distribution_kl(
                encoded,
                teacher_unit[indexes].to(device),
                temperature=config.temperature,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("relational linear training loss is nonfinite")
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            completed_updates += 1
            epoch_losses.append(float(loss.detach()))
        losses.append(math.fsum(epoch_losses) / len(epoch_losses))
        if completed_updates == update_count:
            break
    return model.cpu().eval(), tuple(losses)


def fit_relational_linear_compaction(
    source: torch.Tensor,
    teacher: torch.Tensor,
    *,
    output_dimensions: int,
    config: RelationalLinearTrainingConfig,
    device: torch.device,
) -> tuple[RelationalLinearEncoder, tuple[float, ...]]:
    """Fit a train-only covariance basis and its relational linear projection."""

    basis = _fit_uncentered_covariance_basis(source, dimensions=output_dimensions)
    return fit_relational_linear_encoder(
        source,
        teacher,
        basis,
        config=config,
        device=device,
    )
