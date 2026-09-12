"""Full-dimensional additive quantization with exact one-byte stage codes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class AdditiveQuantizationSpec:
    """Wire and shared-codebook geometry for an additive quantizer."""

    dimensions: int
    stages: int
    codebook_size: int = 256

    def __post_init__(self) -> None:
        if (
            type(self.dimensions) is not int
            or self.dimensions < 1
            or type(self.stages) is not int
            or self.stages < 1
            or type(self.codebook_size) is not int
            or not 2 <= self.codebook_size <= 256
        ):
            raise ValueError("additive quantization spec differs")

    @property
    def bytes_per_vector(self) -> int:
        """Return the exact one-byte-per-stage database width."""

        return self.stages

    @property
    def codebook_shape(self) -> tuple[int, int, int]:
        """Return stages, codewords, and full vector dimensions."""

        return (self.stages, self.codebook_size, self.dimensions)


@dataclass(frozen=True, slots=True)
class AdditiveFitSpec:
    """Deterministic alternating-fit schedule for an additive quantizer."""

    rounds: int
    assignment_sweeps: int
    codebook_epochs: int
    batch_size: int
    learning_rates: tuple[float, ...]
    parallel_weight: float
    seed: int

    def __post_init__(self) -> None:
        if (
            type(self.rounds) is not int
            or self.rounds < 1
            or type(self.assignment_sweeps) is not int
            or self.assignment_sweeps < 1
            or type(self.codebook_epochs) is not int
            or self.codebook_epochs < 1
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.learning_rates) is not tuple
            or len(self.learning_rates) != self.rounds
            or any(
                type(rate) is not float or not math.isfinite(rate) or rate <= 0.0
                for rate in self.learning_rates
            )
            or type(self.parallel_weight) is not float
            or not math.isfinite(self.parallel_weight)
            or self.parallel_weight < 0.0
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("additive fit spec differs")


@dataclass(frozen=True, slots=True)
class AdditiveFitResult:
    """Best monotone state and diagnostics from alternating fitting."""

    quantizer: AdditiveQuantizer
    codes: torch.Tensor
    objectives: tuple[float, ...]
    assignment_churn_ppm: tuple[int, ...]
    fit_stage_utilization_ppm: tuple[tuple[int, ...], ...]


def padded_product_codebooks(
    block_dimensions: tuple[int, ...], codebooks: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Embed disjoint product codebooks into full-dimensional additive supports."""

    if (
        type(block_dimensions) is not tuple
        or len(block_dimensions) < 1
        or any(type(width) is not int or width < 1 for width in block_dimensions)
        or len(codebooks) != len(block_dimensions)
    ):
        raise ValueError("additive product initialization differs")
    dimensions = sum(block_dimensions)
    codebook_size: int | None = None
    device: torch.device | None = None
    result = []
    start = 0
    for width, codebook in zip(block_dimensions, codebooks, strict=True):
        if (
            type(codebook) is not torch.Tensor
            or codebook.dtype != torch.float32
            or codebook.ndim != 2
            or codebook.shape[1] != width
            or codebook.shape[0] < 2
            or codebook.shape[0] > 256
            or not bool(torch.isfinite(codebook).all())
        ):
            raise ValueError("additive product initialization differs")
        if codebook_size is None:
            codebook_size = codebook.shape[0]
            device = codebook.device
        elif codebook.shape[0] != codebook_size or codebook.device != device:
            raise ValueError("additive product initialization differs")
        expanded = torch.zeros(
            (codebook.shape[0], dimensions), dtype=torch.float32, device=codebook.device
        )
        expanded[:, start : start + width] = codebook.detach()
        result.append(expanded)
        start += width
    return torch.stack(result).contiguous()


def additive_reconstruction_loss(
    values: torch.Tensor,
    reconstructed: torch.Tensor,
    *,
    parallel_weight: float,
) -> torch.Tensor:
    """Return MSE plus a residual penalty parallel to unit-normalized inputs."""

    if (
        type(values) is not torch.Tensor
        or type(reconstructed) is not torch.Tensor
        or values.dtype != torch.float32
        or reconstructed.dtype != torch.float32
        or values.ndim != 2
        or values.shape != reconstructed.shape
        or values.shape[0] < 1
        or values.shape[1] < 1
        or values.device != reconstructed.device
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(reconstructed).all())
        or type(parallel_weight) is not float
        or not math.isfinite(parallel_weight)
        or parallel_weight < 0.0
    ):
        raise ValueError("additive reconstruction authority differs")
    residual = values - reconstructed
    per_row = residual.square().sum(dim=1)
    parallel = (values * residual).sum(dim=1).square()
    result = (per_row + parallel_weight * parallel).mean()
    if not bool(torch.isfinite(result)):
        raise RuntimeError("additive reconstruction loss is nonfinite")
    return result


class AdditiveQuantizer(nn.Module):
    """Full-dimensional additive codebooks with deterministic hard encoding."""

    def __init__(self, spec: AdditiveQuantizationSpec, codebooks: torch.Tensor) -> None:
        super().__init__()
        if (
            type(spec) is not AdditiveQuantizationSpec
            or type(codebooks) is not torch.Tensor
            or codebooks.dtype != torch.float32
            or codebooks.shape != spec.codebook_shape
            or not bool(torch.isfinite(codebooks).all())
        ):
            raise ValueError("additive quantization codebook authority differs")
        self.spec = spec
        self.codebooks = nn.Parameter(codebooks.detach().clone().contiguous())

    @classmethod
    def from_codebooks(
        cls, spec: AdditiveQuantizationSpec, codebooks: torch.Tensor
    ) -> AdditiveQuantizer:
        """Construct a trainable additive quantizer from exact shared codebooks."""

        return cls(spec, codebooks)

    def detached_codebooks(self) -> torch.Tensor:
        """Return canonical CPU float32 codebooks."""

        return self.codebooks.detach().cpu().float().contiguous().clone()

    def _validate_values(self, values: torch.Tensor) -> None:
        if (
            type(values) is not torch.Tensor
            or values.dtype != torch.float32
            or values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] != self.spec.dimensions
            or values.device != self.codebooks.device
            or not bool(torch.isfinite(values).all())
        ):
            raise ValueError("additive quantization input authority differs")

    def _validate_codes(self, codes: torch.Tensor, *, rows: int | None = None) -> None:
        if (
            type(codes) is not torch.Tensor
            or codes.dtype != torch.uint8
            or codes.ndim != 2
            or codes.shape[0] < 1
            or codes.shape[1] != self.spec.bytes_per_vector
            or (rows is not None and codes.shape[0] != rows)
            or codes.device != self.codebooks.device
            or bool((codes.to(torch.int16) >= self.spec.codebook_size).any())
        ):
            raise ValueError("additive quantization code authority differs")

    def hard_decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the selected full-dimensional codeword from every stage."""

        self._validate_codes(codes)
        decoded = torch.zeros(
            (codes.shape[0], self.spec.dimensions),
            dtype=torch.float32,
            device=codes.device,
        )
        for stage, codebook in enumerate(self.codebooks):
            decoded = decoded + codebook[codes[:, stage].long()]
        return decoded.contiguous()

    def asymmetric_dot_scores(
        self, queries: torch.Tensor, gallery_codes: torch.Tensor
    ) -> torch.Tensor:
        """Score float queries by exactly one lookup per additive stage."""

        self._validate_values(queries)
        self._validate_codes(gallery_codes)
        scores = torch.zeros(
            (queries.shape[0], gallery_codes.shape[0]),
            dtype=torch.float32,
            device=queries.device,
        )
        with torch.no_grad():
            for stage, codebook in enumerate(self.codebooks):
                table = queries @ codebook.T
                scores = scores + table[:, gallery_codes[:, stage].long()]
        return scores.contiguous()

    def coordinate_descent_encode(
        self,
        values: torch.Tensor,
        *,
        initial_codes: torch.Tensor,
        sweeps: int,
        parallel_weight: float,
        stage_order: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Reassign every stage against the complete additive residual."""

        self._validate_values(values)
        self._validate_codes(initial_codes, rows=values.shape[0])
        if (
            type(sweeps) is not int
            or sweeps < 1
            or type(parallel_weight) is not float
            or not math.isfinite(parallel_weight)
            or parallel_weight < 0.0
            or (
                stage_order is not None
                and (
                    type(stage_order) is not tuple
                    or len(stage_order) != self.spec.stages
                    or set(stage_order) != set(range(self.spec.stages))
                )
            )
        ):
            raise ValueError("additive coordinate descent authority differs")
        ordered_stages = tuple(range(self.spec.stages)) if stage_order is None else stage_order
        codes = initial_codes.detach().clone().contiguous()
        with torch.no_grad():
            for _ in range(sweeps):
                decoded = self.hard_decode(codes)
                for stage in ordered_stages:
                    codebook = self.codebooks[stage]
                    incumbent = codes[:, stage].long()
                    residual_without_stage = values - (decoded - codebook[incumbent])
                    residual_norms = residual_without_stage.square().sum(dim=1, keepdim=True)
                    residual_dot_code = residual_without_stage @ codebook.T
                    losses = (
                        residual_norms
                        - 2.0 * residual_dot_code
                        + codebook.square().sum(dim=1).unsqueeze(0)
                    )
                    if parallel_weight != 0.0:
                        value_dot_residual = (values * residual_without_stage).sum(
                            dim=1, keepdim=True
                        )
                        value_dot_code = values @ codebook.T
                        losses = (
                            losses
                            + parallel_weight * (value_dot_residual - value_dot_code).square()
                        )
                    candidate = torch.argmin(losses, dim=1)
                    current_loss = losses.gather(1, incumbent[:, None]).squeeze(1)
                    candidate_loss = losses.gather(1, candidate[:, None]).squeeze(1)
                    selected = torch.where(candidate_loss < current_loss, candidate, incumbent)
                    decoded = decoded - codebook[incumbent] + codebook[selected]
                    codes[:, stage] = selected.to(torch.uint8)
            if not bool(torch.isfinite(decoded).all()):
                raise RuntimeError("additive coordinate descent is nonfinite")
        return codes.contiguous()


def _complete_objective(
    quantizer: AdditiveQuantizer,
    values: torch.Tensor,
    codes: torch.Tensor,
    *,
    parallel_weight: float,
) -> float:
    with torch.no_grad():
        loss = additive_reconstruction_loss(
            values,
            quantizer.hard_decode(codes),
            parallel_weight=parallel_weight,
        )
    return float(loss)


def fit_additive_quantizer(
    values: torch.Tensor,
    *,
    initial_codebooks: torch.Tensor,
    initial_codes: torch.Tensor,
    fit_spec: AdditiveFitSpec,
) -> AdditiveFitResult:
    """Alternate hard reassignment and guarded fixed-codebook gradient updates."""

    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
        or type(initial_codebooks) is not torch.Tensor
        or initial_codebooks.dtype != torch.float32
        or initial_codebooks.ndim != 3
        or initial_codebooks.shape[0] < 1
        or initial_codebooks.shape[1] < 2
        or initial_codebooks.shape[1] > 256
        or initial_codebooks.shape[2] != values.shape[1]
        or initial_codebooks.device != values.device
        or type(fit_spec) is not AdditiveFitSpec
        or not bool(torch.isfinite(values).all())
        or not bool(torch.isfinite(initial_codebooks).all())
    ):
        raise ValueError("additive fit authority differs")
    spec = AdditiveQuantizationSpec(
        dimensions=values.shape[1],
        stages=initial_codebooks.shape[0],
        codebook_size=initial_codebooks.shape[1],
    )
    quantizer = AdditiveQuantizer.from_codebooks(spec, initial_codebooks)
    quantizer._validate_codes(initial_codes, rows=values.shape[0])
    codes = initial_codes.detach().clone().contiguous()
    objectives = [
        _complete_objective(quantizer, values, codes, parallel_weight=fit_spec.parallel_weight)
    ]
    churn = []
    utilization = []
    for round_index, learning_rate in enumerate(fit_spec.learning_rates):
        previous_codes = codes.detach().clone()
        previous_objective = objectives[-1]
        stage_generator = torch.Generator(device="cpu")
        stage_generator.manual_seed(fit_spec.seed + 100_000 + round_index)
        stage_order = tuple(
            int(value) for value in torch.randperm(spec.stages, generator=stage_generator).tolist()
        )
        codes = quantizer.coordinate_descent_encode(
            values,
            initial_codes=codes,
            sweeps=fit_spec.assignment_sweeps,
            parallel_weight=fit_spec.parallel_weight,
            stage_order=stage_order,
        )
        assignment_objective = _complete_objective(
            quantizer, values, codes, parallel_weight=fit_spec.parallel_weight
        )
        if assignment_objective > previous_objective:
            codes = previous_codes
            assignment_objective = previous_objective
        changed = int((codes != previous_codes).sum())
        churn.append((changed * 1_000_000) // codes.numel())
        utilization.append(
            tuple(
                int(torch.unique(codes[:, stage]).numel()) * 1_000_000 // spec.codebook_size
                for stage in range(spec.stages)
            )
        )
        accepted: AdditiveQuantizer | None = None
        accepted_objective = assignment_objective
        original_codebooks = quantizer.detached_codebooks().to(values.device)
        for attempt, rate in enumerate((learning_rate, learning_rate * 0.25)):
            candidate = AdditiveQuantizer.from_codebooks(spec, original_codebooks)
            optimizer = torch.optim.Adam(candidate.parameters(), lr=rate, weight_decay=0.0)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(fit_spec.seed + 200_000 + round_index * 2 + attempt)
            for _ in range(fit_spec.codebook_epochs):
                order = torch.randperm(values.shape[0], generator=generator)
                for start in range(0, values.shape[0], fit_spec.batch_size):
                    indexes = order[start : start + fit_spec.batch_size].to(values.device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = additive_reconstruction_loss(
                        values[indexes],
                        candidate.hard_decode(codes[indexes]),
                        parallel_weight=fit_spec.parallel_weight,
                    )
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
            candidate_objective = _complete_objective(
                candidate, values, codes, parallel_weight=fit_spec.parallel_weight
            )
            if candidate_objective <= assignment_objective:
                accepted = candidate
                accepted_objective = candidate_objective
                break
        if accepted is not None:
            quantizer = accepted
        objectives.append(accepted_objective)
    return AdditiveFitResult(
        quantizer=quantizer,
        codes=codes,
        objectives=tuple(objectives),
        assignment_churn_ppm=tuple(churn),
        fit_stage_utilization_ppm=tuple(utilization),
    )
