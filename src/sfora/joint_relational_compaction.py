"""Shared compact encoders trained from teacher neighborhood relations."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sfora.split_code_anchor import fit_uncentered_covariance_basis


def _unit_rows(value: torch.Tensor) -> bool:
    if not bool(torch.isfinite(value).all()):
        return False
    norms = torch.linalg.vector_norm(value.detach().double(), dim=1)
    return bool((torch.abs(norms - 1.0) <= 2e-5).all())


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
            or type(self.weight_decay) is not float
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("relational linear training config differs")


@dataclass(frozen=True, slots=True)
class PackedInt8Embeddings:
    """Row-major int8 embeddings with one little-endian f16 inverse norm per row."""

    codes: torch.Tensor
    inverse_norms: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.codes) is not torch.Tensor
            or self.codes.device.type != "cpu"
            or self.codes.dtype != torch.int8
            or self.codes.ndim != 2
            or self.codes.shape[0] < 1
            or self.codes.shape[1] < 2
            or not self.codes.is_contiguous()
            or type(self.inverse_norms) is not torch.Tensor
            or self.inverse_norms.device.type != "cpu"
            or self.inverse_norms.dtype != torch.float16
            or self.inverse_norms.shape != (self.codes.shape[0],)
            or not self.inverse_norms.is_contiguous()
            or not bool(torch.isfinite(self.inverse_norms).all())
            or bool((self.inverse_norms <= 0).any())
        ):
            raise ValueError("packed int8 embedding authority differs")
        norms = torch.linalg.vector_norm(self.codes.float(), dim=1)
        expected_inverse_norms = norms.reciprocal().to(torch.float16)
        lower = torch.nextafter(
            expected_inverse_norms,
            torch.full_like(expected_inverse_norms, -torch.inf),
        )
        upper = torch.nextafter(
            expected_inverse_norms,
            torch.full_like(expected_inverse_norms, torch.inf),
        )
        if (
            bool((norms <= 0).any())
            or bool((self.inverse_norms < lower).any())
            or bool((self.inverse_norms > upper).any())
        ):
            raise ValueError("packed int8 embedding authority differs")

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact wire width of one packed vector."""

        return self.codes.shape[1] + 2

    def restore(self) -> torch.Tensor:
        """Restore unit-like float32 rows without retaining expanded storage."""

        return self.codes.float() * self.inverse_norms.float().unsqueeze(1)

    def cosine_similarity(
        self,
        other: PackedInt8Embeddings,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Compute pairwise cosine scores directly from packed row metadata."""

        if type(other) is not PackedInt8Embeddings or other.codes.shape[1] != self.codes.shape[1]:
            raise ValueError("packed int8 similarity authority differs")
        if device is None:
            device = torch.device("cpu")
        if type(device) is not torch.device:
            raise ValueError("packed int8 similarity authority differs")
        integer_dots = (
            self.codes.to(device=device, dtype=torch.float32)
            @ other.codes.to(device=device, dtype=torch.float32).T
        )
        return (
            integer_dots
            * self.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(1)
            * other.inverse_norms.to(device=device, dtype=torch.float32).unsqueeze(0)
        )

    def to_bytes(self) -> bytes:
        """Serialize each row as signed code bytes followed by one little-endian f16."""

        count, dimensions = self.codes.shape
        wire = np.empty((count, dimensions + 2), dtype=np.uint8)
        wire[:, :dimensions] = self.codes.numpy().view(np.uint8)
        inverse_bytes = self.inverse_norms.numpy().astype("<f2", copy=False).view(np.uint8)
        wire[:, dimensions:] = inverse_bytes.reshape(count, 2)
        return wire.tobytes(order="C")

    @classmethod
    def from_bytes(cls, wire: bytes, *, count: int, dimensions: int) -> PackedInt8Embeddings:
        """Parse an exact packed batch with no trailing or missing bytes."""

        if (
            type(wire) is not bytes
            or type(count) is not int
            or count < 1
            or type(dimensions) is not int
            or dimensions < 2
            or len(wire) != count * (dimensions + 2)
        ):
            raise ValueError("packed int8 byte authority differs")
        rows = np.frombuffer(wire, dtype=np.uint8).reshape(count, dimensions + 2)
        codes = torch.from_numpy(rows[:, :dimensions].copy().view(np.int8))
        inverse_values = (
            rows[:, dimensions:].copy().reshape(-1).view("<f2").astype(np.float16, copy=True)
        )
        inverse = torch.from_numpy(inverse_values)
        return cls(codes=codes.contiguous(), inverse_norms=inverse.contiguous())


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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * steps_per_epoch
    )
    generator = torch.Generator().manual_seed(config.seed)
    losses = []
    for _epoch in range(config.epochs):
        order = torch.randperm(len(source), generator=generator)[
            : steps_per_epoch * config.batch_size
        ]
        epoch_losses = []
        for start in range(0, len(order), config.batch_size):
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
            epoch_losses.append(float(loss.detach()))
        losses.append(math.fsum(epoch_losses) / len(epoch_losses))
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

    basis = fit_uncentered_covariance_basis(source.detach(), dimensions=output_dimensions).float()
    return fit_relational_linear_encoder(
        source,
        teacher,
        basis,
        config=config,
        device=device,
    )


def fixed_int8_unit_codes(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a unit embedding to fixed-scale int8 and return its unit restoration."""

    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.float32
        or value.ndim != 2
        or value.shape[0] < 1
        or value.shape[1] < 2
        or not _unit_rows(value)
    ):
        raise ValueError("joint relational quantization authority differs")
    codes = torch.round(value * 127.0).clamp(-127, 127).to(torch.int8).contiguous()
    restored = F.normalize(codes.float(), dim=1).contiguous()
    if not bool(torch.isfinite(restored).all()):
        raise ValueError("joint relational quantization geometry differs")
    return codes, restored


def pack_int8_unit_embeddings(value: torch.Tensor) -> PackedInt8Embeddings:
    """Quantize unit rows into the exact dimensions-plus-two-byte wire format."""

    codes, _restored = fixed_int8_unit_codes(value)
    norms = torch.linalg.vector_norm(codes.float(), dim=1)
    inverse_norms = norms.reciprocal().to(torch.float16).contiguous()
    return PackedInt8Embeddings(codes=codes, inverse_norms=inverse_norms)
