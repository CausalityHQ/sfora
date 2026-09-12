"""Joint projection and hard product-quantization training primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sfora.product_quantization import (
    NeighborhoodAdcDistillationLoss,
    ProductQuantizer,
    neighborhood_adc_distillation_loss,
)


@dataclass(frozen=True, slots=True)
class JointPqTrainingSpec:
    """Fixed optimizer and loss schedule for one joint-PQ fit."""

    updates: int
    batch_size: int
    projection_learning_rate: float
    codebook_learning_rate: float
    weight_decay: float
    temperature: float
    float_weight: float
    reconstruction_weight: float
    differential_weight: float
    gradient_norm_cap: float
    seed: int

    def __post_init__(self) -> None:
        positive = (
            self.projection_learning_rate,
            self.codebook_learning_rate,
            self.temperature,
            self.gradient_norm_cap,
        )
        nonnegative = (
            self.weight_decay,
            self.float_weight,
            self.reconstruction_weight,
            self.differential_weight,
        )
        if (
            type(self.updates) is not int
            or self.updates < 1
            or type(self.batch_size) is not int
            or self.batch_size < 1
            or type(self.seed) is not int
            or self.seed < 0
            or any(
                type(value) is not float or not math.isfinite(value) or value <= 0.0
                for value in positive
            )
            or any(
                type(value) is not float or not math.isfinite(value) or value < 0.0
                for value in nonnegative
            )
        ):
            raise ValueError("joint PQ training spec differs")


@dataclass(frozen=True, slots=True)
class JointPqLossTrace:
    """Detached scalar objective components for one optimizer update."""

    total: float
    adc_kl: float
    float_kl: float
    reconstruction: float
    differential: float


@dataclass(frozen=True, slots=True)
class JointPqFitResult:
    """A fitted model and its deterministic total-loss trace."""

    model: JointPqProjection
    losses: tuple[float, ...]
    trace: tuple[JointPqLossTrace, ...]


class JointPqProjection(nn.Module):
    """One normalized linear projection followed by a trainable hard PQ codec."""

    _initial_weight: torch.Tensor
    _row_space_basis: torch.Tensor | None

    def __init__(
        self,
        projection: nn.Linear,
        quantizer: ProductQuantizer,
        row_space_basis: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if (
            type(projection) is not nn.Linear
            or projection.bias is None
            or type(quantizer) is not ProductQuantizer
            or projection.out_features != quantizer.spec.dimensions
            or projection.weight.dtype != torch.float32
            or projection.bias.dtype != torch.float32
            or projection.weight.device != quantizer.codebooks[0].device
            or not bool(torch.isfinite(projection.weight).all())
            or not bool(torch.isfinite(projection.bias).all())
            or (
                row_space_basis is not None
                and (
                    type(row_space_basis) is not torch.Tensor
                    or row_space_basis.dtype != torch.float32
                    or row_space_basis.ndim != 2
                    or row_space_basis.shape[1] != projection.in_features
                    or row_space_basis.device != projection.weight.device
                    or not bool(torch.isfinite(row_space_basis).all())
                )
            )
        ):
            raise ValueError("joint PQ projection authority differs")
        self.projection = projection
        self.quantizer = quantizer
        self.register_buffer("_initial_weight", projection.weight.detach().clone())
        self.register_buffer(
            "_row_space_basis",
            None if row_space_basis is None else row_space_basis.detach().clone(),
        )

    @classmethod
    def from_components(
        cls,
        *,
        input_dimensions: int,
        initial_weight: torch.Tensor,
        initial_bias: torch.Tensor,
        quantizer: ProductQuantizer,
        row_space_basis: torch.Tensor | None = None,
    ) -> JointPqProjection:
        """Clone an affine projection and quantizer into an independent trainable arm."""

        if (
            type(input_dimensions) is not int
            or input_dimensions < 1
            or type(initial_weight) is not torch.Tensor
            or type(initial_bias) is not torch.Tensor
            or initial_weight.dtype != torch.float32
            or initial_bias.dtype != torch.float32
            or initial_weight.shape != (quantizer.spec.dimensions, input_dimensions)
            or initial_bias.shape != (quantizer.spec.dimensions,)
            or initial_weight.device != initial_bias.device
            or initial_weight.device != quantizer.codebooks[0].device
            or not bool(torch.isfinite(initial_weight).all())
            or not bool(torch.isfinite(initial_bias).all())
        ):
            raise ValueError("joint PQ projection authority differs")
        projection = nn.Linear(
            input_dimensions,
            quantizer.spec.dimensions,
            bias=True,
            device=initial_weight.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            projection.weight.copy_(initial_weight)
            projection.bias.copy_(initial_bias)
        cloned_quantizer = ProductQuantizer.from_codebooks(
            quantizer.spec,
            tuple(codebook.detach().clone() for codebook in quantizer.codebooks),
        )
        return cls(projection, cloned_quantizer, row_space_basis)

    def effective_weight(self) -> torch.Tensor:
        """Return the deployable weight, constrained to the registered row space if present."""

        delta = self.projection.weight - self._initial_weight
        basis = self._row_space_basis
        if basis is not None:
            delta = (delta @ basis.T) @ basis
        return self._initial_weight + delta

    def _project_prevalidated(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            F.linear(inputs, self.effective_weight(), self.projection.bias), dim=-1
        ).contiguous()

    def project(self, inputs: torch.Tensor) -> torch.Tensor:
        """Project and unit-normalize one row matrix."""

        if (
            type(inputs) is not torch.Tensor
            or inputs.dtype != torch.float32
            or inputs.ndim != 2
            or inputs.shape[0] < 1
            or inputs.shape[1] != self.projection.in_features
            or inputs.device != self.projection.weight.device
            or not bool(torch.isfinite(inputs).all())
        ):
            raise ValueError("joint PQ projection input differs")
        result = self._project_prevalidated(inputs)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("joint PQ projection is nonfinite")
        return result.contiguous()

    def _loss_prevalidated(
        self,
        query_inputs: torch.Tensor,
        candidate_inputs: torch.Tensor,
        teacher_queries: torch.Tensor,
        teacher_candidates: torch.Tensor,
        *,
        neighbor_pairs: torch.Tensor,
        spec: JointPqTrainingSpec,
    ) -> NeighborhoodAdcDistillationLoss:
        batch, candidates, input_dimensions = candidate_inputs.shape
        projected_queries = self._project_prevalidated(query_inputs)
        projected_candidates = self._project_prevalidated(
            candidate_inputs.reshape(batch * candidates, input_dimensions)
        ).reshape(batch, candidates, -1)
        return neighborhood_adc_distillation_loss(
            projected_queries,
            projected_candidates,
            teacher_queries,
            teacher_candidates,
            self.quantizer,
            temperature=spec.temperature,
            float_weight=spec.float_weight,
            reconstruction_weight=spec.reconstruction_weight,
            differential_weight=spec.differential_weight,
            neighbor_pairs=neighbor_pairs,
            _validated=True,
        )

    def loss(
        self,
        query_inputs: torch.Tensor,
        candidate_inputs: torch.Tensor,
        teacher_queries: torch.Tensor,
        teacher_candidates: torch.Tensor,
        *,
        neighbor_pairs: torch.Tensor,
        temperature: float,
        float_weight: float,
        reconstruction_weight: float,
        differential_weight: float,
    ) -> NeighborhoodAdcDistillationLoss:
        """Evaluate the deployment-identical joint training objective."""

        if (
            type(candidate_inputs) is not torch.Tensor
            or candidate_inputs.dtype != torch.float32
            or candidate_inputs.ndim != 3
            or candidate_inputs.shape[0] != query_inputs.shape[0]
            or candidate_inputs.shape[2] != self.projection.in_features
            or candidate_inputs.device != query_inputs.device
            or not bool(torch.isfinite(candidate_inputs).all())
        ):
            raise ValueError("joint PQ projection input differs")
        batch, candidates, input_dimensions = candidate_inputs.shape
        projected_candidates = self.project(
            candidate_inputs.reshape(batch * candidates, input_dimensions)
        ).reshape(batch, candidates, -1)
        return neighborhood_adc_distillation_loss(
            self.project(query_inputs),
            projected_candidates,
            teacher_queries,
            teacher_candidates,
            self.quantizer,
            temperature=temperature,
            float_weight=float_weight,
            reconstruction_weight=reconstruction_weight,
            differential_weight=differential_weight,
            neighbor_pairs=neighbor_pairs,
        )


def fit_joint_pq_projection(
    model: JointPqProjection,
    *,
    inputs: torch.Tensor,
    teacher_values: torch.Tensor,
    candidate_indexes: torch.Tensor,
    neighbor_pairs: torch.Tensor,
    spec: JointPqTrainingSpec,
) -> JointPqFitResult:
    """Fit one joint arm with a deterministic without-replacement row schedule."""

    device = model.projection.weight.device
    if (
        type(model) is not JointPqProjection
        or type(spec) is not JointPqTrainingSpec
        or type(inputs) is not torch.Tensor
        or inputs.dtype != torch.float32
        or inputs.ndim != 2
        or inputs.shape[0] < spec.batch_size
        or inputs.shape[1] != model.projection.in_features
        or inputs.device != device
        or not bool(torch.isfinite(inputs).all())
        or type(teacher_values) is not torch.Tensor
        or teacher_values.dtype != torch.float32
        or teacher_values.ndim != 2
        or teacher_values.shape[0] != inputs.shape[0]
        or teacher_values.shape[1] < 2
        or teacher_values.device != device
        or not bool(torch.isfinite(teacher_values).all())
        or not bool(
            (
                torch.abs(torch.linalg.vector_norm(teacher_values.detach().double(), dim=1) - 1.0)
                <= 2e-5
            ).all()
        )
        or type(candidate_indexes) is not torch.Tensor
        or candidate_indexes.dtype != torch.int64
        or candidate_indexes.ndim != 2
        or candidate_indexes.shape[0] != inputs.shape[0]
        or candidate_indexes.device != device
        or bool((candidate_indexes < 0).any())
        or bool((candidate_indexes >= inputs.shape[0]).any())
        or type(neighbor_pairs) is not torch.Tensor
        or neighbor_pairs.dtype != torch.int64
        or neighbor_pairs.ndim != 2
        or neighbor_pairs.shape[0] < 1
        or neighbor_pairs.shape[1] != 2
        or neighbor_pairs.device != device
        or bool((neighbor_pairs < 0).any())
        or bool((neighbor_pairs >= candidate_indexes.shape[1]).any())
        or bool((neighbor_pairs[:, 0] == neighbor_pairs[:, 1]).any())
    ):
        raise ValueError("joint PQ training authority differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    scheduled_rows: list[torch.Tensor] = []
    required_rows = spec.updates * spec.batch_size
    accumulated_rows = 0
    while accumulated_rows < required_rows:
        permutation = torch.randperm(inputs.shape[0], generator=generator)
        scheduled_rows.append(permutation)
        accumulated_rows += permutation.numel()
    schedule = torch.cat(scheduled_rows)[:required_rows].reshape(spec.updates, spec.batch_size)
    optimizer = torch.optim.AdamW(
        (
            {
                "params": model.projection.parameters(),
                "lr": spec.projection_learning_rate,
            },
            {
                "params": model.quantizer.parameters(),
                "lr": spec.codebook_learning_rate,
            },
        ),
        weight_decay=spec.weight_decay,
    )
    losses: list[float] = []
    trace: list[JointPqLossTrace] = []
    model.train()
    for batch_cpu in schedule:
        batch = batch_cpu.to(device)
        candidates = candidate_indexes[batch]
        optimizer.zero_grad(set_to_none=True)
        observed = model._loss_prevalidated(
            inputs[batch],
            inputs[candidates],
            teacher_values[batch],
            teacher_values[candidates],
            neighbor_pairs=neighbor_pairs,
            spec=spec,
        )
        observed.total.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), spec.gradient_norm_cap, error_if_nonfinite=True
        )
        optimizer.step()
        value = float(observed.total.detach())
        if not math.isfinite(value) or any(
            not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
        ):
            raise RuntimeError("joint PQ training loss is nonfinite")
        losses.append(value)
        trace.append(
            JointPqLossTrace(
                total=value,
                adc_kl=float(observed.adc_kl.detach()),
                float_kl=float(observed.float_kl.detach()),
                reconstruction=float(observed.reconstruction.detach()),
                differential=float(observed.differential.detach()),
            )
        )
    model.eval()
    return JointPqFitResult(model=model, losses=tuple(losses), trace=tuple(trace))
