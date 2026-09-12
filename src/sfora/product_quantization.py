"""Hard product quantization with deployment-identical asymmetric training scores."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ProductQuantizationSpec:
    """Wire geometry for one-byte-per-block product codes."""

    block_dimensions: tuple[int, ...]
    codebook_size: int = 256

    def __post_init__(self) -> None:
        if (
            type(self.block_dimensions) is not tuple
            or len(self.block_dimensions) < 1
            or any(type(width) is not int or width < 1 for width in self.block_dimensions)
            or type(self.codebook_size) is not int
            or not 2 <= self.codebook_size <= 256
        ):
            raise ValueError("product quantization spec differs")

    @property
    def dimensions(self) -> int:
        """Return the input width represented by all blocks."""

        return sum(self.block_dimensions)

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact hard-code width."""

        return len(self.block_dimensions)


@dataclass(frozen=True, slots=True)
class NeighborhoodAdcDistillationLoss:
    """Observable components of neighborhood distillation through hard ADC."""

    total: torch.Tensor
    adc_kl: torch.Tensor
    float_kl: torch.Tensor
    reconstruction: torch.Tensor


class _HardReconstructionStraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
        return hard.clone()

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return gradient, gradient


class ProductQuantizer(nn.Module):
    """Variable-block product quantizer with one unsigned byte per block."""

    def __init__(self, spec: ProductQuantizationSpec, codebooks: Sequence[torch.Tensor]) -> None:
        super().__init__()
        if type(spec) is not ProductQuantizationSpec or len(codebooks) != len(
            spec.block_dimensions
        ):
            raise ValueError("product quantization codebook authority differs")
        validated = []
        device: torch.device | None = None
        for width, codebook in zip(spec.block_dimensions, codebooks, strict=True):
            if (
                type(codebook) is not torch.Tensor
                or codebook.dtype != torch.float32
                or codebook.ndim != 2
                or codebook.shape != (spec.codebook_size, width)
                or not bool(torch.isfinite(codebook).all())
            ):
                raise ValueError("product quantization codebook authority differs")
            if device is None:
                device = codebook.device
            elif codebook.device != device:
                raise ValueError("product quantization codebook authority differs")
            validated.append(nn.Parameter(codebook.detach().clone().contiguous()))
        self.spec = spec
        self.codebooks = nn.ParameterList(validated)

    @classmethod
    def from_codebooks(
        cls, spec: ProductQuantizationSpec, codebooks: Sequence[torch.Tensor]
    ) -> ProductQuantizer:
        """Construct a trainable quantizer from exact floating codebooks."""

        return cls(spec, codebooks)

    def detached_codebooks(self) -> tuple[torch.Tensor, ...]:
        """Return canonical CPU float32 codebook values."""

        return tuple(
            codebook.detach().cpu().float().contiguous().clone() for codebook in self.codebooks
        )

    def _validate_values(self, values: torch.Tensor) -> None:
        if (
            type(values) is not torch.Tensor
            or values.dtype != torch.float32
            or values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != self.spec.dimensions
            or values.device != self.codebooks[0].device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("product quantization input authority differs")

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if (
            type(codes) is not torch.Tensor
            or codes.dtype != torch.uint8
            or codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] != self.spec.bytes_per_vector
            or codes.device != self.codebooks[0].device
            or bool((codes.to(torch.int16) >= self.spec.codebook_size).any())
        ):
            raise ValueError("product quantization code authority differs")

    def hard_encode(self, values: torch.Tensor) -> torch.Tensor:
        """Assign every block to its nearest codeword with stable lowest-index ties."""

        self._validate_values(values)
        codes = []
        start = 0
        with torch.no_grad():
            for width, codebook in zip(self.spec.block_dimensions, self.codebooks, strict=True):
                block = values[:, start : start + width]
                distances = (block[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
                codes.append(torch.argmin(distances, dim=1).to(torch.uint8))
                start += width
        return torch.stack(codes, dim=1).contiguous()

    def hard_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode one hard code batch without vector renormalization."""

        self._validate_codes(codes)
        blocks = [codebook[codes[:, index].long()] for index, codebook in enumerate(self.codebooks)]
        return torch.cat(blocks, dim=1).contiguous()

    def straight_through(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return hard reconstructions with identity input and selected-codeword gradients."""

        self._validate_values(values)
        codes = self.hard_encode(values)
        hard = self.hard_decode(codes)
        reconstructed = _HardReconstructionStraightThrough.apply(values, hard)  # type: ignore[no-untyped-call]
        return reconstructed.contiguous(), codes

    def asymmetric_squared_distances(
        self, queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor:
        """Sum per-block float-query lookup distances for a coded gallery batch."""

        self._validate_values(queries)
        self._validate_codes(gallery_codes)
        distances = torch.zeros(
            (queries.shape[0], gallery_codes.shape[0]),
            dtype=torch.float32,
            device=queries.device,
        )
        start = 0
        for block_index, (width, codebook) in enumerate(
            zip(self.spec.block_dimensions, self.codebooks, strict=True)
        ):
            query_block = queries[:, start : start + width]
            table = (query_block[:, None, :] - codebook[None, :, :]).square().sum(dim=-1)
            distances = distances + table[:, gallery_codes[:, block_index].long()]
            start += width
        return distances.contiguous()


class OptimizedProductQuantizer(nn.Module):
    """Product quantization behind one fixed orthogonal input rotation."""

    _rotation: torch.Tensor

    def __init__(
        self,
        spec: ProductQuantizationSpec,
        rotation: torch.Tensor,
        codebooks: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        if (
            type(rotation) is not torch.Tensor
            or rotation.dtype != torch.float32
            or rotation.ndim != 2
            or rotation.shape != (spec.dimensions, spec.dimensions)
            or not bool(torch.isfinite(rotation).all())
        ):
            raise ValueError("optimized product quantization rotation differs")
        identity = torch.eye(spec.dimensions, dtype=torch.float64, device=rotation.device)
        gram = rotation.double().T @ rotation.double()
        if float(torch.linalg.matrix_norm(gram - identity, ord="fro")) > 2e-5:
            raise ValueError("optimized product quantization rotation differs")
        quantizer = ProductQuantizer(spec, codebooks)
        if rotation.device != quantizer.codebooks[0].device:
            raise ValueError("optimized product quantization rotation differs")
        self.spec = spec
        self.register_buffer("_rotation", rotation.detach().clone().contiguous())
        self.quantizer = quantizer

    @classmethod
    def from_components(
        cls,
        spec: ProductQuantizationSpec,
        rotation: torch.Tensor,
        codebooks: Sequence[torch.Tensor],
    ) -> OptimizedProductQuantizer:
        """Construct an OPQ codec from exact rotation and codebook components."""

        return cls(spec, rotation, codebooks)

    def detached_rotation(self) -> torch.Tensor:
        """Return the canonical CPU float32 orthogonal rotation."""

        return self._rotation.detach().cpu().float().contiguous().clone()

    def detached_codebooks(self) -> tuple[torch.Tensor, ...]:
        """Return the canonical CPU float32 product codebooks."""

        return self.quantizer.detached_codebooks()

    def _rotated(self, values: torch.Tensor) -> torch.Tensor:
        self.quantizer._validate_values(values)
        return torch.matmul(values, self._rotation).contiguous()

    def hard_encode(self, values: torch.Tensor) -> torch.Tensor:
        """Rotate and assign every block to one hard codeword."""

        return self.quantizer.hard_encode(self._rotated(values))

    def hard_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode hard codes back into the original vector coordinates."""

        transformed = self.quantizer.hard_decode(codes)
        return torch.matmul(transformed, self._rotation.T).contiguous()

    def straight_through(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Use the exact rotated hard codec with straight-through row gradients."""

        transformed, codes = self.quantizer.straight_through(self._rotated(values))
        return torch.matmul(transformed, self._rotation.T).contiguous(), codes

    def asymmetric_squared_distances(
        self, queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor:
        """Score original-coordinate float queries against rotated hard codes."""

        return self.quantizer.asymmetric_squared_distances(self._rotated(queries), gallery_codes)


def fit_product_quantizer(
    values: torch.Tensor,
    spec: ProductQuantizationSpec,
    *,
    seed: int,
    maximum_iterations: int,
) -> ProductQuantizer:
    """Fit deterministic independent Lloyd codebooks for every registered block."""

    if (
        type(values) is not torch.Tensor
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < spec.codebook_size
        or values.shape[1] != spec.dimensions
        or not bool(torch.isfinite(values).all())
        or type(seed) is not int
        or seed < 0
        or type(maximum_iterations) is not int
        or maximum_iterations < 1
    ):
        raise ValueError("product quantization fit authority differs")
    array = values.detach().numpy().astype(np.float32, copy=False)
    codebooks = []
    start = 0
    for block_index, width in enumerate(spec.block_dimensions):
        estimator = KMeans(
            n_clusters=spec.codebook_size,
            init="k-means++",
            n_init=1,
            max_iter=maximum_iterations,
            random_state=seed + block_index,
            algorithm="lloyd",
        )
        with threadpool_limits(limits=1):
            estimator.fit(array[:, start : start + width])
        centers = estimator.cluster_centers_.astype(np.float32, copy=True)
        if centers.shape != (spec.codebook_size, width) or not np.isfinite(centers).all():
            raise RuntimeError("product quantization fit produced invalid codebooks")
        codebooks.append(torch.from_numpy(centers))
        start += width
    return ProductQuantizer.from_codebooks(spec, tuple(codebooks))


def fit_optimized_product_quantizer(
    values: torch.Tensor,
    spec: ProductQuantizationSpec,
    *,
    seed: int,
    maximum_iterations: int,
    rotation_iterations: int,
) -> OptimizedProductQuantizer:
    """Fit deterministic OPQ alternations and retain the best observed codec."""

    if (
        type(values) is not torch.Tensor
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < spec.codebook_size
        or values.shape[1] != spec.dimensions
        or not bool(torch.isfinite(values).all())
        or type(rotation_iterations) is not int
        or rotation_iterations < 1
    ):
        raise ValueError("optimized product quantization fit authority differs")
    rotation = torch.eye(spec.dimensions, dtype=torch.float32)
    baseline = fit_product_quantizer(values, spec, seed=seed, maximum_iterations=maximum_iterations)
    with torch.inference_mode():
        baseline_codes = baseline.hard_encode(values)
        best_error = float((values - baseline.hard_decode(baseline_codes)).double().square().sum())
    best_rotation = rotation.clone()
    best_codebooks = baseline.detached_codebooks()
    candidate = baseline
    for _ in range(rotation_iterations):
        transformed = torch.matmul(values, rotation).float().contiguous()
        with torch.inference_mode():
            codes = candidate.hard_encode(transformed)
            reconstructed = candidate.hard_decode(codes)
            left, _singular, right = torch.linalg.svd(
                values.double().T @ reconstructed.double(), full_matrices=False
            )
            rotation = torch.matmul(left, right).float().contiguous()
        transformed = torch.matmul(values, rotation).float().contiguous()
        candidate = fit_product_quantizer(
            transformed, spec, seed=seed, maximum_iterations=maximum_iterations
        )
        with torch.inference_mode():
            codes = candidate.hard_encode(transformed)
            error = float((transformed - candidate.hard_decode(codes)).double().square().sum())
            if error < best_error:
                best_error = error
                best_rotation = rotation
                best_codebooks = candidate.detached_codebooks()
    return OptimizedProductQuantizer.from_components(spec, best_rotation, best_codebooks)


def _unit_rows(values: torch.Tensor) -> bool:
    if not bool(torch.isfinite(values).all()):
        return False
    norms = torch.linalg.vector_norm(values.detach().double(), dim=-1)
    return bool((torch.abs(norms - 1.0) <= 2e-5).all())


def neighborhood_adc_distillation_loss(
    student_queries: torch.Tensor,
    student_gallery: torch.Tensor,
    teacher_queries: torch.Tensor,
    teacher_gallery: torch.Tensor,
    quantizer: ProductQuantizer,
    *,
    temperature: float,
    float_weight: float,
    reconstruction_weight: float,
) -> NeighborhoodAdcDistillationLoss:
    """Distill teacher neighborhoods through the exact hard asymmetric score."""

    if (
        type(quantizer) is not ProductQuantizer
        or type(student_queries) is not torch.Tensor
        or type(student_gallery) is not torch.Tensor
        or type(teacher_queries) is not torch.Tensor
        or type(teacher_gallery) is not torch.Tensor
        or student_queries.dtype != torch.float32
        or student_gallery.dtype != torch.float32
        or teacher_queries.dtype != torch.float32
        or teacher_gallery.dtype != torch.float32
        or student_queries.ndim != 2
        or student_gallery.ndim != 3
        or teacher_queries.shape != student_queries.shape
        or teacher_gallery.shape != student_gallery.shape
        or student_gallery.shape[0] != student_queries.shape[0]
        or student_gallery.shape[2] != student_queries.shape[1]
        or student_queries.shape[1] != quantizer.spec.dimensions
        or student_queries.device != student_gallery.device
        or teacher_queries.device != student_queries.device
        or teacher_gallery.device != student_queries.device
        or quantizer.codebooks[0].device != student_queries.device
        or not _unit_rows(student_queries)
        or not _unit_rows(student_gallery)
        or not _unit_rows(teacher_queries)
        or not _unit_rows(teacher_gallery)
        or type(temperature) is not float
        or not math.isfinite(temperature)
        or temperature <= 0.0
        or type(float_weight) is not float
        or not math.isfinite(float_weight)
        or float_weight < 0.0
        or type(reconstruction_weight) is not float
        or not math.isfinite(reconstruction_weight)
        or reconstruction_weight < 0.0
    ):
        raise ValueError("neighborhood ADC distillation authority differs")
    batch, candidates, dimensions = student_gallery.shape
    hard_gallery, codes = quantizer.straight_through(
        student_gallery.reshape(batch * candidates, dimensions)
    )
    hard_gallery = hard_gallery.reshape_as(student_gallery)
    reconstruction_gallery = quantizer.hard_decode(codes).reshape_as(student_gallery)
    teacher_logits = (
        torch.einsum("bd,bcd->bc", teacher_queries.detach(), teacher_gallery.detach()) / temperature
    )
    adc_logits = (
        -0.5 * (student_queries[:, None, :] - hard_gallery).square().sum(dim=-1) / temperature
    )
    float_logits = torch.einsum("bd,bcd->bc", student_queries, student_gallery) / temperature
    teacher_probabilities = F.softmax(teacher_logits, dim=-1)
    adc_kl = F.kl_div(
        F.log_softmax(adc_logits, dim=-1), teacher_probabilities, reduction="batchmean"
    )
    float_kl = F.kl_div(
        F.log_softmax(float_logits, dim=-1), teacher_probabilities, reduction="batchmean"
    )
    reconstruction = F.mse_loss(reconstruction_gallery, student_gallery)
    total = adc_kl + float_weight * float_kl + reconstruction_weight * reconstruction
    if not bool(torch.isfinite(total)):
        raise RuntimeError("neighborhood ADC distillation loss is nonfinite")
    return NeighborhoodAdcDistillationLoss(
        total=total,
        adc_kl=adc_kl,
        float_kl=float_kl,
        reconstruction=reconstruction,
    )
