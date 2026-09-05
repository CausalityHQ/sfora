"""Descriptor-only capacity diagnostics for SigLIP gallery compatibility."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _validated_pair(
    student: torch.Tensor, teacher: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        type(student) is not torch.Tensor
        or type(teacher) is not torch.Tensor
        or student.device.type != "cpu"
        or teacher.device.type != "cpu"
        or student.dtype != torch.float32
        or teacher.dtype != torch.float32
        or student.ndim != 2
        or teacher.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] < 2
        or student.shape[1] < 2
        or not bool(torch.isfinite(student).all())
        or not bool(torch.isfinite(teacher).all())
        or bool((torch.linalg.vector_norm(student, dim=1) <= 0).any())
        or bool((torch.linalg.vector_norm(teacher, dim=1) <= 0).any())
    ):
        raise ValueError("compatibility descriptor authority differs")
    return student, teacher


def compatibility_folds(labels: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Return the registered three-way class folds."""

    if (
        type(labels) is not tuple
        or len(labels) != 39
        or any(type(label) is not int for label in labels)
        or set(labels) != set(range(39))
    ):
        raise ValueError("compatibility class authority differs")
    ranked = sorted(
        (
            hashlib.sha256(
                b"sfora-compatibility-capacity-fold-v1\0" + str(label).encode("ascii")
            ).digest(),
            label,
        )
        for label in labels
    )
    return tuple(
        tuple(label for _, label in ranked[start : start + 13])
        for start in range(0, 39, 13)
    )


@dataclass(frozen=True, slots=True)
class AffineMap:
    """A validated FP64 affine descriptor map with normalized FP32 output."""

    weight: torch.Tensor
    bias: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.weight) is not torch.Tensor
            or type(self.bias) is not torch.Tensor
            or self.weight.device.type != "cpu"
            or self.bias.device.type != "cpu"
            or self.weight.dtype != torch.float64
            or self.bias.dtype != torch.float64
            or self.weight.ndim != 2
            or self.weight.shape[0] != self.weight.shape[1]
            or self.bias.shape != (self.weight.shape[0],)
            or not bool(torch.isfinite(self.weight).all())
            or not bool(torch.isfinite(self.bias).all())
        ):
            raise ValueError("compatibility affine authority differs")

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply this map and return normalized CPU FP32 descriptors."""

        if (
            type(descriptors) is not torch.Tensor
            or descriptors.device.type != "cpu"
            or descriptors.dtype != torch.float32
            or descriptors.ndim != 2
            or descriptors.shape[0] < 2
            or descriptors.shape[1] != self.weight.shape[0]
            or not bool(torch.isfinite(descriptors).all())
            or bool((torch.linalg.vector_norm(descriptors, dim=1) <= 0).any())
        ):
            raise ValueError("compatibility descriptor authority differs")
        mapped = F.normalize(descriptors.double() @ self.weight + self.bias, dim=1)
        if not bool(torch.isfinite(mapped).all()):
            raise ValueError("compatibility affine authority differs")
        return mapped.float().contiguous()


def fit_centered_similarity(student: torch.Tensor, teacher: torch.Tensor) -> AffineMap:
    """Fit the registered centered orthogonal descriptor map."""

    student, teacher = _validated_pair(student, teacher)
    source = student.double()
    target = teacher.double()
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    left, _, right_t = torch.linalg.svd(
        (source - source_mean).T @ (target - target_mean), full_matrices=False
    )
    weight = (left @ right_t).contiguous()
    bias = (target_mean - source_mean @ weight).contiguous()
    return AffineMap(weight, bias)


def fit_regularized_affine(
    student: torch.Tensor, teacher: torch.Tensor, regularization: float
) -> AffineMap:
    """Fit the registered identity-regularized affine ridge map."""

    student, teacher = _validated_pair(student, teacher)
    if type(regularization) is not float or regularization not in {1e-4, 1e-2, 1.0}:
        raise ValueError("compatibility regularization differs")
    source = student.double()
    target = teacher.double()
    rows, dimensions = source.shape
    augmented = torch.cat((source, torch.ones(rows, 1, dtype=torch.float64)), dim=1)
    identity_target = torch.cat(
        (torch.eye(dimensions, dtype=torch.float64), torch.zeros(1, dimensions)), dim=0
    )
    system = augmented.T @ augmented / rows
    system += regularization * torch.eye(dimensions + 1, dtype=torch.float64)
    right = augmented.T @ target / rows + regularization * identity_target
    solution = torch.linalg.solve(system, right)
    return AffineMap(solution[:-1].contiguous(), solution[-1].contiguous())


class CompatibilityResidual(nn.Module):
    """The fixed low-rank residual descriptor adapter."""

    def __init__(
        self,
        dimensions: int,
        *,
        rank: int,
        seed: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        if (
            type(dimensions) is not int
            or dimensions < 2
            or type(rank) is not int
            or rank != 32
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError("compatibility residual authority differs")
        target = torch.device("cpu") if device is None else device
        with torch.random.fork_rng(devices=[] if target.type == "cpu" else [target]):
            torch.manual_seed(seed)
            self.down = nn.Linear(dimensions, rank, bias=False, dtype=torch.float64, device=target)
            self.up = nn.Linear(rank, dimensions, bias=True, dtype=torch.float64, device=target)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply the unnormalized residual so zero initialization is exact identity."""

        return descriptors + self.up(F.gelu(self.down(descriptors)))


@dataclass(frozen=True, slots=True)
class ResidualFit:
    """A frozen fitted residual and its complete optimization evidence."""

    state_dict: dict[str, torch.Tensor]
    losses: dict[str, tuple[float, ...]]
    anchor_ids: tuple[str, ...]
    relational: bool
    seed: int

    def apply(self, descriptors: torch.Tensor) -> torch.Tensor:
        """Apply the frozen residual to normalized CPU FP32 descriptors."""

        if type(descriptors) is not torch.Tensor or descriptors.ndim != 2:
            raise ValueError("compatibility descriptor authority differs")
        dimensions = self.state_dict["up.bias"].shape[0]
        model = CompatibilityResidual(dimensions, rank=32, seed=self.seed)
        model.load_state_dict(self.state_dict, strict=True)
        model.eval()
        with torch.inference_mode():
            return F.normalize(model(descriptors.double()), dim=1).float().contiguous()


def _masked_score_loss(
    actual: torch.Tensor,
    expected: torch.Tensor,
    query_ids: tuple[str, ...],
    anchor_ids: tuple[str, ...],
) -> torch.Tensor:
    mask = torch.tensor(
        [[query_id != anchor_id for anchor_id in anchor_ids] for query_id in query_ids],
        dtype=torch.bool,
        device=actual.device,
    )
    if not bool(mask.any()):
        raise ValueError("compatibility residual authority differs")
    return (actual[mask] - expected[mask]).square().mean()


def fit_teacher_anchored_residual(
    student: torch.Tensor,
    teacher: torch.Tensor,
    ids: tuple[str, ...],
    *,
    relational: bool,
    seed: int,
) -> ResidualFit:
    """Fit the registered paired or teacher-anchored residual adapter."""

    student, teacher = _validated_pair(student, teacher)
    if (
        type(ids) is not tuple
        or len(ids) != student.shape[0]
        or len(ids) < 256
        or len(set(ids)) != len(ids)
        or any(type(value) is not str or not value for value in ids)
        or type(relational) is not bool
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("compatibility residual authority differs")
    anchor_indexes = sorted(
        range(len(ids)),
        key=lambda index: hashlib.sha256(
            b"sfora-compatibility-anchor-v1\0" + ids[index].encode("utf-8")
        ).digest(),
    )[:256]
    anchor_ids = tuple(ids[index] for index in anchor_indexes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = student.double().to(device)
    target = teacher.double().to(device)
    anchor_index_tensor = torch.tensor(anchor_indexes, dtype=torch.int64, device=device)
    source_anchors = source[anchor_index_tensor]
    target_anchors = target[anchor_index_tensor]
    model = CompatibilityResidual(student.shape[1], rank=32, seed=seed, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    trajectories: dict[str, list[float]] = {
        "paired": [],
        "forward": [],
        "reverse": [],
        "self": [],
    }
    rows = student.shape[0]
    for update in range(2_000):
        query_indexes = (torch.arange(256, device=device) + update * 256) % rows
        query_ids = tuple(ids[int(index)] for index in query_indexes.cpu())
        source_query = source[query_indexes]
        target_query = target[query_indexes]
        mapped_query = F.normalize(model(source_query), dim=1)
        paired_loss = (1.0 - (mapped_query * target_query).sum(dim=1)).mean()
        if relational:
            mapped_anchors = F.normalize(model(source_anchors), dim=1)
            teacher_scores = target_query @ target_anchors.T
            forward_loss = _masked_score_loss(
                mapped_query @ target_anchors.T, teacher_scores, query_ids, anchor_ids
            )
            reverse_loss = _masked_score_loss(
                target_query @ mapped_anchors.T, teacher_scores, query_ids, anchor_ids
            )
            self_loss = _masked_score_loss(
                mapped_query @ mapped_anchors.T,
                source_query @ source_anchors.T,
                query_ids,
                anchor_ids,
            )
            loss = paired_loss + forward_loss + reverse_loss + self_loss
        else:
            forward_loss = reverse_loss = self_loss = paired_loss.new_zeros(())
            loss = paired_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        values = {
            "paired": float(paired_loss.detach().cpu()),
            "forward": float(forward_loss.detach().cpu()),
            "reverse": float(reverse_loss.detach().cpu()),
            "self": float(self_loss.detach().cpu()),
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("compatibility residual authority differs")
        for name, value in values.items():
            trajectories[name].append(value)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return ResidualFit(
        state_dict=state,
        losses={name: tuple(values) for name, values in trajectories.items()},
        anchor_ids=anchor_ids,
        relational=relational,
        seed=seed,
    )
