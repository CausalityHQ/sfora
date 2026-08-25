"""Pure protocol and optimizer helpers for the UniCOM ProxyMuon screen."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.optim import _muon

LR_GRID = (0.000025, 0.00005, 0.0001, 0.0002, 0.0004)
PHASE1_SEEDS = (0, 1, 2)
PHASE2_SEEDS = (3, 4, 5)
RETAINED_STEPS = (0, 64, 128, 192, 256, 307, 384, 435, 512)
VALIDATION_STEPS = (307, 435, 512)
OPTIMIZERS = ("adamw", "proxy_muon")

_PHASE1_ROW_KEYS = (
    "optimizer",
    "learning_rate",
    "fit_seed",
    "step_64_diagnostic_mean",
)
_ADAMW_REFERENCE_ROW_KEYS = (
    "variant",
    "learning_rate",
    "fit_seed",
    "step_512_diagnostic_mean",
    "step_512_accuracy",
)
_DECISION_KEYS = (
    "structural_valid",
    "adamw_selected_lr_interior",
    "proxy_muon_selected_lr_interior",
    "proxy_muon_reach_steps",
    "proxy_muon_noninferior_at_reach",
    "proxy_muon_step512_noninferior",
    "fp32_reach_steps",
    "fp32_noninferior_at_reach",
    "fp32_step512_noninferior",
)


@dataclass(frozen=True)
class LearningRateSelection:
    """Registered Phase-1 learning-rate selection."""

    learning_rate: float
    mean_step_64_loss: float
    interior: bool


@dataclass(frozen=True)
class AdamWReference:
    """Per-seed AdamW reference chosen only by registered fitting loss."""

    variant: str
    learning_rate: float
    step_512_diagnostic_mean: float
    step_512_accuracy: float


@dataclass(frozen=True)
class MuonTrace:
    """Read-only evidence for one pending built-in Muon update."""

    orthogonal_update: torch.Tensor
    update_dtype: str
    polar_factor_residual: float


def build_head_optimizer(
    head: torch.nn.Parameter, optimizer_name: str, learning_rate: float
) -> torch.optim.Optimizer:
    """Construct one fresh optimizer over the sole registered proxy matrix."""

    if (
        type(head) is not torch.nn.Parameter
        or head.dtype != torch.float32
        or head.ndim != 2
        or head.shape[0] == 0
        or head.shape[1] == 0
        or not head.is_contiguous()
        or not head.requires_grad
        or not torch.isfinite(head).all()
        or type(optimizer_name) is not str
        or optimizer_name not in OPTIMIZERS
        or type(learning_rate) is not float
        or learning_rate not in LR_GRID
    ):
        raise ValueError("ProxyMuon optimizer input differs")
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            [head],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
    return torch.optim.Muon(
        [head],
        lr=learning_rate,
        momentum=0.95,
        nesterov=True,
        ns_coefficients=(3.4445, -4.775, 2.0315),
        eps=1e-7,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
        weight_decay=0.0,
    )


def _newton_schulz_zeropower(
    update: torch.Tensor, *, ns_dtype: torch.dtype
) -> torch.Tensor:
    coefficients = (3.4445, -4.775, 2.0315)
    ortho = update.to(dtype=ns_dtype, copy=True)
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        ortho = ortho.T
    ortho.div_(ortho.norm().clamp(min=1e-7))
    a, b, c = coefficients
    for _ in range(5):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


class PrecisionMuon(torch.optim.Optimizer):
    """Pinned single-parameter Muon with an explicit NS precision switch."""

    def __init__(
        self, head: torch.nn.Parameter, *, lr: float, ns_dtype: torch.dtype
    ) -> None:
        if (
            type(head) is not torch.nn.Parameter
            or head.dtype != torch.float32
            or head.ndim != 2
            or head.shape[0] == 0
            or head.shape[1] == 0
            or not head.is_contiguous()
            or not head.requires_grad
            or not torch.isfinite(head).all()
            or torch.any(torch.linalg.vector_norm(head, dim=1) == 0.0)
            or type(lr) is not float
            or lr not in LR_GRID
            or ns_dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("ProxyMuon precision adapter input differs")
        self.ns_dtype = ns_dtype
        super().__init__(
            [head],
            {
                "lr": lr,
                "weight_decay": 0.0,
                "momentum": 0.95,
                "nesterov": True,
                "ns_coefficients": (3.4445, -4.775, 2.0315),
                "eps": 1e-7,
                "ns_steps": 5,
                "adjust_lr_fn": "match_rms_adamw",
            },
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("ProxyMuon precision adapter closure differs")
        group = self.param_groups[0]
        head = group["params"][0]
        gradient = head.grad
        if (
            type(gradient) is not torch.Tensor
            or gradient.dtype != torch.float32
            or gradient.shape != head.shape
            or gradient.is_sparse
            or not torch.isfinite(gradient).all()
        ):
            raise ValueError("ProxyMuon precision adapter gradient differs")
        state = self.state[head]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(
                gradient, memory_format=torch.preserve_format
            )
        if tuple(state) != ("momentum_buffer",):
            raise ValueError("ProxyMuon precision adapter state differs")
        momentum_buffer = state["momentum_buffer"]
        if (
            type(momentum_buffer) is not torch.Tensor
            or momentum_buffer.dtype != torch.float32
            or momentum_buffer.shape != head.shape
            or not torch.isfinite(momentum_buffer).all()
        ):
            raise ValueError("ProxyMuon precision adapter momentum differs")
        momentum_buffer.lerp_(gradient, 0.05)
        update = gradient.lerp(momentum_buffer, 0.95)
        orthogonal_update = _newton_schulz_zeropower(
            update, ns_dtype=self.ns_dtype
        )
        if (
            type(orthogonal_update) is not torch.Tensor
            or orthogonal_update.dtype != self.ns_dtype
            or orthogonal_update.shape != head.shape
            or not torch.isfinite(orthogonal_update).all()
        ):
            raise ValueError("ProxyMuon precision adapter update differs")
        adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(head.shape))
        head.mul_(1 - group["lr"] * group["weight_decay"])
        head.add_(orthogonal_update, alpha=-adjusted_lr)
        if not torch.isfinite(head).all():
            raise ValueError("ProxyMuon precision adapter parameter differs")
        return None


def trace_builtin_muon_step(
    head: torch.nn.Parameter, optimizer: torch.optim.Optimizer
) -> MuonTrace:
    """Reconstruct a pending built-in Muon update without mutating live state."""

    if (
        type(head) is not torch.nn.Parameter
        or head.dtype != torch.float32
        or head.ndim != 2
        or not head.is_contiguous()
        or type(optimizer) is not torch.optim.Muon
        or len(optimizer.param_groups) != 1
    ):
        raise ValueError("ProxyMuon trace input differs")
    group = optimizer.param_groups[0]
    if (
        group["params"] != [head]
        or type(group["lr"]) is not float
        or group["lr"] not in LR_GRID
        or group["momentum"] != 0.95
        or group["nesterov"] is not True
        or group["ns_coefficients"] != (3.4445, -4.775, 2.0315)
        or group["eps"] != 1e-7
        or group["ns_steps"] != 5
        or group["adjust_lr_fn"] != "match_rms_adamw"
        or group["weight_decay"] != 0.0
    ):
        raise ValueError("ProxyMuon trace optimizer differs")
    gradient = head.grad
    if (
        type(gradient) is not torch.Tensor
        or gradient.dtype != torch.float32
        or gradient.shape != head.shape
        or gradient.is_sparse
        or not torch.isfinite(gradient).all()
    ):
        raise ValueError("ProxyMuon trace gradient differs")
    state = optimizer.state.get(head, {})
    if tuple(state) not in ((), ("momentum_buffer",)):
        raise ValueError("ProxyMuon trace state differs")
    prior_momentum = state.get("momentum_buffer")
    if prior_momentum is None:
        prior_momentum = torch.zeros_like(
            gradient, memory_format=torch.preserve_format
        )
    elif (
        type(prior_momentum) is not torch.Tensor
        or prior_momentum.dtype != torch.float32
        or prior_momentum.shape != head.shape
        or not torch.isfinite(prior_momentum).all()
    ):
        raise ValueError("ProxyMuon trace momentum differs")
    next_momentum = prior_momentum.clone().lerp_(gradient, 0.05)
    effective_update = gradient.clone().lerp(next_momentum, 0.95)
    orthogonal_update = _muon._zeropower_via_newtonschulz(
        effective_update,
        (3.4445, -4.775, 2.0315),
        5,
        1e-7,
    )
    floating = orthogonal_update.float()
    gram = (
        floating.T @ floating
        if floating.shape[0] >= floating.shape[1]
        else floating @ floating.T
    )
    identity = torch.eye(gram.shape[0], dtype=torch.float32, device=gram.device)
    residual = float(torch.linalg.vector_norm(gram - identity))
    if not math.isfinite(residual):
        raise ValueError("ProxyMuon trace residual differs")
    return MuonTrace(
        orthogonal_update=orthogonal_update,
        update_dtype=str(orthogonal_update.dtype),
        polar_factor_residual=residual,
    )


def _validated_phase1_rows(
    rows: list[dict[str, object]],
) -> dict[tuple[str, float, int], float]:
    if type(rows) is not list:
        raise ValueError("ProxyMuon Phase-1 rows differ")
    expected_order = tuple(
        (optimizer, learning_rate, fit_seed)
        for optimizer in OPTIMIZERS
        for learning_rate in LR_GRID
        for fit_seed in PHASE1_SEEDS
    )
    if len(rows) != len(expected_order):
        raise ValueError("ProxyMuon Phase-1 row count differs")

    observed: dict[tuple[str, float, int], float] = {}
    for row, expected_key in zip(rows, expected_order, strict=True):
        if type(row) is not dict or tuple(row) != _PHASE1_ROW_KEYS:
            raise ValueError("ProxyMuon Phase-1 row schema differs")
        optimizer = row["optimizer"]
        learning_rate = row["learning_rate"]
        fit_seed = row["fit_seed"]
        loss = row["step_64_diagnostic_mean"]
        if (
            type(optimizer) is not str
            or type(learning_rate) is not float
            or type(fit_seed) is not int
            or type(loss) is not float
            or not math.isfinite(loss)
        ):
            raise ValueError("ProxyMuon Phase-1 row value differs")
        key = (optimizer, learning_rate, fit_seed)
        if key != expected_key or key in observed:
            raise ValueError("ProxyMuon Phase-1 row order differs")
        observed[key] = loss
    return observed


def select_learning_rate(
    rows: list[dict[str, object]], *, optimizer: str
) -> LearningRateSelection:
    """Select the registered LR by three-seed mean with smaller-LR ties."""

    if type(optimizer) is not str or optimizer not in OPTIMIZERS:
        raise ValueError("ProxyMuon optimizer differs")
    observed = _validated_phase1_rows(rows)
    means = tuple(
        (
            math.fsum(observed[(optimizer, learning_rate, seed)] for seed in PHASE1_SEEDS)
            / len(PHASE1_SEEDS),
            learning_rate,
        )
        for learning_rate in LR_GRID
    )
    mean_loss, learning_rate = min(means)
    return LearningRateSelection(
        learning_rate=learning_rate,
        mean_step_64_loss=mean_loss,
        interior=learning_rate not in (LR_GRID[0], LR_GRID[-1]),
    )


def select_adamw_reference(
    rows: list[dict[str, object]], *, selected_learning_rate: float, fit_seed: int
) -> AdamWReference:
    """Choose selected-vs-anchor AdamW by loss, with smaller-LR ties."""

    if (
        type(rows) is not list
        or type(selected_learning_rate) is not float
        or selected_learning_rate not in LR_GRID
        or type(fit_seed) is not int
        or fit_seed not in PHASE2_SEEDS
    ):
        raise ValueError("ProxyMuon AdamW reference inventory differs")
    variants = (
        (("adamw_selected", selected_learning_rate),)
        if selected_learning_rate == 0.0001
        else (
            ("adamw_selected", selected_learning_rate),
            ("adamw_anchor", 0.0001),
        )
    )
    if len(rows) != len(variants):
        raise ValueError("ProxyMuon AdamW reference row count differs")

    validated: list[AdamWReference] = []
    for row, (expected_variant, expected_lr) in zip(rows, variants, strict=True):
        if type(row) is not dict or tuple(row) != _ADAMW_REFERENCE_ROW_KEYS:
            raise ValueError("ProxyMuon AdamW reference row schema differs")
        variant = row["variant"]
        learning_rate = row["learning_rate"]
        row_seed = row["fit_seed"]
        loss = row["step_512_diagnostic_mean"]
        accuracy = row["step_512_accuracy"]
        if (
            type(variant) is not str
            or variant != expected_variant
            or type(learning_rate) is not float
            or learning_rate != expected_lr
            or type(row_seed) is not int
            or row_seed != fit_seed
            or type(loss) is not float
            or not math.isfinite(loss)
            or type(accuracy) is not float
            or not math.isfinite(accuracy)
        ):
            raise ValueError("ProxyMuon AdamW reference row value differs")
        validated.append(
            AdamWReference(
                variant=variant,
                learning_rate=learning_rate,
                step_512_diagnostic_mean=loss,
                step_512_accuracy=accuracy,
            )
        )
    return min(
        validated,
        key=lambda row: (row.step_512_diagnostic_mean, row.learning_rate),
    )


def compute_reach_step(
    losses: dict[int, float], reference_loss: float
) -> int | str:
    """Return the first registered step whose loss reaches the reference."""

    if (
        type(losses) is not dict
        or tuple(losses) != VALIDATION_STEPS
        or type(reference_loss) is not float
        or not math.isfinite(reference_loss)
        or any(type(value) is not float or not math.isfinite(value) for value in losses.values())
    ):
        raise ValueError("ProxyMuon reach evidence differs")
    for step in VALIDATION_STEPS:
        if losses[step] <= reference_loss:
            return step
    return ">512"


def accuracy_noninferior(candidate_accuracy: float, reference_accuracy: float) -> bool:
    """Apply the registered accuracy-loss boundary without subtraction drift."""

    if (
        type(candidate_accuracy) is not float
        or type(reference_accuracy) is not float
        or not math.isfinite(candidate_accuracy)
        or not math.isfinite(reference_accuracy)
        or not 0.0 <= candidate_accuracy <= 1.0
        or not 0.0 <= reference_accuracy <= 1.0
    ):
        raise ValueError("ProxyMuon accuracy evidence differs")
    return candidate_accuracy >= reference_accuracy - 0.002


def _validated_reaches(value: object) -> dict[int, int | str]:
    if type(value) is not dict or tuple(value) != PHASE2_SEEDS:
        raise ValueError("ProxyMuon decision reach schema differs")
    if any(
        not (
            (type(reach) is int and reach in VALIDATION_STEPS)
            or (type(reach) is str and reach == ">512")
        )
        for reach in value.values()
    ):
        raise ValueError("ProxyMuon decision reach value differs")
    return value


def _validated_seed_bools(value: object) -> dict[int, bool]:
    if (
        type(value) is not dict
        or tuple(value) != PHASE2_SEEDS
        or any(type(flag) is not bool for flag in value.values())
    ):
        raise ValueError("ProxyMuon decision predicate differs")
    return value


def _meets_route(
    reaches: dict[int, int | str],
    at_reach: dict[int, bool],
    at_step_512: dict[int, bool],
    *,
    maximum_step: int,
) -> bool:
    return all(
        type(reaches[seed]) is int
        and reaches[seed] <= maximum_step
        and at_reach[seed]
        and at_step_512[seed]
        for seed in PHASE2_SEEDS
    )


def decide_proxy_muon_f0(evidence: dict[str, object]) -> str:
    """Apply the preregistered structural-to-scientific decision cascade."""

    if type(evidence) is not dict or tuple(evidence) != _DECISION_KEYS:
        raise ValueError("ProxyMuon decision schema differs")
    structural_valid = evidence["structural_valid"]
    adamw_interior = evidence["adamw_selected_lr_interior"]
    proxy_muon_interior = evidence["proxy_muon_selected_lr_interior"]
    if any(
        type(value) is not bool
        for value in (structural_valid, adamw_interior, proxy_muon_interior)
    ):
        raise ValueError("ProxyMuon decision scalar differs")
    proxy_reaches = _validated_reaches(evidence["proxy_muon_reach_steps"])
    proxy_at_reach = _validated_seed_bools(
        evidence["proxy_muon_noninferior_at_reach"]
    )
    proxy_at_512 = _validated_seed_bools(
        evidence["proxy_muon_step512_noninferior"]
    )
    fp32_reaches = _validated_reaches(evidence["fp32_reach_steps"])
    fp32_at_reach = _validated_seed_bools(evidence["fp32_noninferior_at_reach"])
    fp32_at_512 = _validated_seed_bools(evidence["fp32_step512_noninferior"])

    if not structural_valid:
        return "STRUCTURAL_FAILURE"
    if not adamw_interior or not proxy_muon_interior:
        return "UNRESOLVED_LR_BOUNDARY"
    if _meets_route(
        proxy_reaches, proxy_at_reach, proxy_at_512, maximum_step=307
    ):
        return "PROCEED_TRAINING"
    if _meets_route(fp32_reaches, fp32_at_reach, fp32_at_512, maximum_step=307):
        return "ROUTE_FP32_ORTHOGONALIZATION"
    if _meets_route(
        proxy_reaches, proxy_at_reach, proxy_at_512, maximum_step=435
    ):
        return "ROUTE_MATCHED_LR"
    return "CLOSE_PROXY_MUON"
