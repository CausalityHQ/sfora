"""Pair-conditioned rank-16 patch-gradient predictor for ASG-CV."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from sfora.asgcv import (
    ASGCV_PREDICTOR_RANK,
    ASGCV_STRATUM_SIZE,
    AsgcvSrhtAuthority,
    select_stratum_index,
    srht_signs_and_rows,
)

_STATE_DOMAIN = b"sfora-asgcv-predictor-state-v1\0"
_PREDICTION_DOMAIN = b"sfora-asgcv-prediction-v1\0"


@dataclass(frozen=True, slots=True)
class PreparedAsgcvStratum:
    """Detached predictor output sealed before one registered selection."""

    predicted: torch.Tensor
    selected_index: int
    predictor_state_sha256: str
    prediction_sha256: str


class AsgcvPatchGradientPredictor(nn.Module):
    """Predict an exchange-equivariant low-rank gradient field for one pair."""

    def __init__(self, *, channel_dimensions: int, predictor_rank: int = 16) -> None:
        super().__init__()
        if type(channel_dimensions) is not int or channel_dimensions < ASGCV_PREDICTOR_RANK:
            raise ValueError("ASG-CV predictor channel dimensions differ")
        if type(predictor_rank) is not int or predictor_rank != ASGCV_PREDICTOR_RANK:
            raise ValueError("ASG-CV predictor rank differs")
        self.channel_dimensions = channel_dimensions
        self.predictor_rank = predictor_rank
        self.normalization = nn.LayerNorm(channel_dimensions)
        self.patch_projection = nn.Linear(channel_dimensions, predictor_rank, bias=False)
        self.context_projection = nn.Linear(
            4 * channel_dimensions + 1,
            2 * predictor_rank,
        )
        self.channel_basis = nn.Parameter(torch.empty(channel_dimensions, predictor_rank))
        nn.init.xavier_uniform_(self.channel_basis)

    def _validated_inputs(
        self,
        tokens: object,
        relation_signs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            type(tokens) is not torch.Tensor
            or tokens.dtype != torch.float32
            or tokens.ndim != 4
            or tokens.shape[1] != 2
            or tokens.shape[2] <= 0
            or tokens.shape[3] != self.channel_dimensions
            or tokens.requires_grad
            or not bool(torch.isfinite(tokens).all())
        ):
            raise ValueError("ASG-CV predictor token authority differs")
        if (
            type(relation_signs) is not torch.Tensor
            or relation_signs.dtype != torch.int8
            or relation_signs.ndim != 1
            or relation_signs.shape[0] != tokens.shape[0]
            or relation_signs.device != tokens.device
            or not bool(((relation_signs == -1) | (relation_signs == 1)).all())
        ):
            raise ValueError("ASG-CV predictor relation authority differs")
        return tokens, relation_signs

    def forward(self, tokens: object, relation_signs: object) -> torch.Tensor:
        """Return the trainable rank-16 patch/channel gradient prediction."""

        token_tensor, sign_tensor = self._validated_inputs(tokens, relation_signs)
        normalized = self.normalization(token_tensor)
        pooled = normalized.mean(dim=2)
        counterpart = pooled[:, [1, 0]]
        sign_feature = sign_tensor.to(dtype=normalized.dtype)[:, None, None].expand(-1, 2, -1)
        context = torch.cat(
            (
                pooled,
                counterpart,
                pooled - counterpart,
                pooled * counterpart,
                sign_feature,
            ),
            dim=-1,
        )
        patch_bias, rank_modulation = self.context_projection(context).chunk(2, dim=-1)
        patch_factors = self.patch_projection(normalized) + patch_bias[:, :, None, :]
        channel_factors = self.channel_basis[None, None, :, :] * (
            1.0 + torch.tanh(rank_modulation[:, :, None, :])
        )
        result = torch.einsum("bipr,bidr->bipd", patch_factors, channel_factors)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("ASG-CV predictor result is not finite")
        return result

    def predict_detached(self, tokens: object, relation_signs: object) -> torch.Tensor:
        """Return the student-injection field with predictor autograd detached."""

        with torch.no_grad():
            return self.forward(tokens, relation_signs).detach()


def predictor_state_sha256(predictor: object) -> str:
    """Hash every named predictor tensor with exact shape and little-endian fp32 bytes."""

    if type(predictor) is not AsgcvPatchGradientPredictor:
        raise ValueError("ASG-CV predictor state authority differs")
    frame = bytearray(_STATE_DOMAIN)
    state = predictor.state_dict()
    for name in sorted(state):
        tensor = state[name]
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("ASG-CV predictor state tensor differs")
        name_bytes = name.encode("utf-8")
        frame.extend(len(name_bytes).to_bytes(8, "big"))
        frame.extend(name_bytes)
        frame.extend(tensor.ndim.to_bytes(8, "big"))
        for size in tensor.shape:
            frame.extend(int(size).to_bytes(8, "big"))
        array = tensor.detach().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
        frame.extend(array.tobytes(order="C"))
    return hashlib.sha256(frame).hexdigest()


def _prediction_sha256(value: torch.Tensor) -> str:
    if value.dtype != torch.float32 or value.ndim != 4 or not bool(torch.isfinite(value).all()):
        raise ValueError("ASG-CV prediction authority differs")
    frame = bytearray(_PREDICTION_DOMAIN)
    for size in value.shape:
        frame.extend(int(size).to_bytes(8, "big"))
    array = value.detach().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
    frame.extend(array.tobytes(order="C"))
    return hashlib.sha256(frame).hexdigest()


def prepare_asgcv_stratum(
    predictor: object,
    tokens: object,
    relation_signs: object,
    *,
    selection_seed: object,
    optimizer_step: int,
    stratum_ordinal: int,
) -> PreparedAsgcvStratum:
    """Freeze predictor output before revealing the registered selected pair."""

    if type(predictor) is not AsgcvPatchGradientPredictor:
        raise ValueError("ASG-CV prepared predictor authority differs")
    if type(tokens) is not torch.Tensor or tokens.shape[0] != ASGCV_STRATUM_SIZE:
        raise ValueError("ASG-CV prepared stratum shape differs")
    state_digest = predictor_state_sha256(predictor)
    predicted = predictor.predict_detached(tokens, relation_signs)
    if predictor_state_sha256(predictor) != state_digest:
        raise ValueError("ASG-CV predictor changed during preparation")
    selected_index = select_stratum_index(
        selection_seed,
        optimizer_step=optimizer_step,
        stratum_ordinal=stratum_ordinal,
    )
    return PreparedAsgcvStratum(
        predicted=predicted,
        selected_index=selected_index,
        predictor_state_sha256=state_digest,
        prediction_sha256=_prediction_sha256(predicted),
    )


def torch_asgcv_stratum_gradient(
    prepared: object,
    exact_selected: object,
    *,
    predictor: object,
) -> torch.Tensor:
    """Form the torch estimator only while the prepared predictor state is unchanged."""

    if type(prepared) is not PreparedAsgcvStratum:
        raise ValueError("ASG-CV prepared stratum authority differs")
    if type(predictor) is not AsgcvPatchGradientPredictor:
        raise ValueError("ASG-CV prepared predictor authority differs")
    if predictor_state_sha256(predictor) != prepared.predictor_state_sha256:
        raise ValueError("ASG-CV predictor changed before estimator formation")
    predicted = prepared.predicted
    if _prediction_sha256(predicted) != prepared.prediction_sha256:
        raise ValueError("ASG-CV prepared prediction changed")
    if (
        type(exact_selected) is not torch.Tensor
        or exact_selected.dtype != torch.float32
        or exact_selected.shape != predicted.shape[1:]
        or exact_selected.device != predicted.device
        or exact_selected.requires_grad
        or not bool(torch.isfinite(exact_selected).all())
    ):
        raise ValueError("ASG-CV selected exact gradient authority differs")
    result = predicted.mean(dim=0) + exact_selected - predicted[prepared.selected_index]
    if not bool(torch.isfinite(result).all()):
        raise ValueError("ASG-CV torch stratum gradient is not finite")
    return result


def torch_srht_gradient_sketch(
    field: object,
    authority: AsgcvSrhtAuthority,
) -> torch.Tensor:
    """Apply the differentiable fixed-order SRHT on the final tensor dimension."""

    if (
        type(field) is not torch.Tensor
        or field.dtype not in (torch.float32, torch.float64)
        or field.ndim < 2
        or not bool(torch.isfinite(field).all())
    ):
        raise ValueError("ASG-CV torch SRHT field authority differs")
    if type(authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV torch SRHT authority differs")
    authority.validated()
    field_tensor: torch.Tensor = field
    if field_tensor.shape[-1] != authority.input_dimensions:
        raise ValueError("ASG-CV torch SRHT shape differs")

    numpy_signs, numpy_rows = srht_signs_and_rows(authority)
    signs = torch.from_numpy(numpy_signs).to(
        device=field_tensor.device,
        dtype=field_tensor.dtype,
    )
    rows = torch.from_numpy(numpy_rows).to(device=field_tensor.device)
    work = torch.nn.functional.pad(
        field_tensor * signs[: authority.input_dimensions],
        (0, authority.padded_dimensions - authority.input_dimensions),
    )
    width = 1
    while width < authority.padded_dimensions:
        groups = work.reshape(*work.shape[:-1], -1, width * 2)
        left = groups[..., :width]
        right = groups[..., width:]
        work = torch.cat((left + right, left - right), dim=-1).reshape(*work.shape)
        width *= 2
    work = work / float(authority.padded_dimensions) ** 0.5
    selected: torch.Tensor = torch.index_select(work, dim=-1, index=rows)
    result: torch.Tensor = (
        selected * (float(authority.padded_dimensions) / authority.output_dimensions) ** 0.5
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("ASG-CV torch SRHT result is not finite")
    return result


def predictor_training_loss(
    predicted: object,
    exact: object,
    authority: AsgcvSrhtAuthority,
) -> torch.Tensor:
    """Return the fixed dense-plus-SRHT normalized predictor objective."""

    if (
        type(predicted) is not torch.Tensor
        or type(exact) is not torch.Tensor
        or predicted.dtype != torch.float32
        or exact.dtype != torch.float32
        or predicted.ndim != 4
        or predicted.shape != exact.shape
        or predicted.shape[1] != 2
        or exact.requires_grad
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(exact).all())
    ):
        raise ValueError("ASG-CV predictor training tensor authority differs")
    if type(authority) is not AsgcvSrhtAuthority:
        raise ValueError("ASG-CV predictor training SRHT authority differs")
    authority.validated()
    if predicted.shape[-1] != authority.input_dimensions:
        raise ValueError("ASG-CV predictor training shape differs")

    exact_energy = exact.square().sum()
    if not bool(torch.isfinite(exact_energy)) or float(exact_energy) <= 0.0:
        raise ValueError("ASG-CV predictor exact gradient energy differs")
    dense_loss = (predicted - exact).square().sum() / exact_energy
    projected_exact = torch_srht_gradient_sketch(exact, authority)
    projected_predicted = torch_srht_gradient_sketch(predicted, authority)
    projected_energy = projected_exact.square().sum()
    if not bool(torch.isfinite(projected_energy)) or float(projected_energy) <= 0.0:
        raise ValueError("ASG-CV predictor projected gradient energy differs")
    projected_loss = (projected_predicted - projected_exact).square().sum() / projected_energy
    result = dense_loss + projected_loss
    if not bool(torch.isfinite(result)):
        raise ValueError("ASG-CV predictor training loss is not finite")
    return result
