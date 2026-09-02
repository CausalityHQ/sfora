"""Pair-conditioned rank-16 patch-gradient predictor for ASG-CV."""

from __future__ import annotations

import hashlib
import math
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
_INITIALIZATION_DOMAIN = b"sfora-asgcv-predictor-initialization-v1\0"
ASGCV_EXCHANGE_MAX_ABS_ERROR_GATE_PPM = 1


@dataclass(frozen=True, slots=True)
class PreparedAsgcvStratum:
    """Detached predictor output sealed before one registered selection."""

    predicted: torch.Tensor
    selected_index: int
    predictor_state_sha256: str
    prediction_sha256: str


@dataclass(frozen=True, slots=True)
class AsgcvRelationControls:
    """Deterministic exchange-equivariance and relation-liveness evidence."""

    exchange_max_abs_error_ppm: int
    relation_response_energy_ppm: int

    def validated(self) -> AsgcvRelationControls:
        if (
            type(self.exchange_max_abs_error_ppm) is not int
            or not 0 <= self.exchange_max_abs_error_ppm <= ASGCV_EXCHANGE_MAX_ABS_ERROR_GATE_PPM
        ):
            raise ValueError("ASG-CV predictor exchange-equivariance differs")
        if (
            type(self.relation_response_energy_ppm) is not int
            or self.relation_response_energy_ppm <= 0
        ):
            raise ValueError("ASG-CV predictor relation liveness differs")
        return self


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
            1.0 + 0.5 * torch.tanh(rank_modulation[:, :, None, :])
        )
        result = torch.einsum("bipr,bidr->bipd", patch_factors, channel_factors)
        if not bool(torch.isfinite(result).all()):
            raise ValueError("ASG-CV predictor result is not finite")
        return result

    def predict_detached(self, tokens: object, relation_signs: object) -> torch.Tensor:
        """Return the student-injection field with predictor autograd detached."""

        with torch.no_grad():
            return self.forward(tokens, relation_signs).detach()


def evaluate_predictor_relation_controls(
    predictor: object,
    tokens: object,
) -> AsgcvRelationControls:
    """Measure exact pair-swap behavior and nonzero relation conditioning."""

    if (
        type(predictor) is not AsgcvPatchGradientPredictor
        or type(tokens) is not torch.Tensor
        or tokens.ndim != 4
        or tokens.shape[0] <= 0
    ):
        raise ValueError("ASG-CV predictor relation-control authority differs")
    signs = torch.ones(tokens.shape[0], dtype=torch.int8, device=tokens.device)
    positive = predictor.predict_detached(tokens, signs)
    negative = predictor.predict_detached(tokens, -signs)
    swapped = predictor.predict_detached(tokens[:, [1, 0]], signs)
    positive64 = positive.detach().cpu().double()
    negative64 = negative.detach().cpu().double()
    exchange_error = float(
        torch.max(torch.abs(swapped.detach().cpu().double() - positive64[:, [1, 0]]))
    )
    exchange_scale = float(torch.max(torch.abs(positive64)))
    if not math.isfinite(exchange_error) or not math.isfinite(exchange_scale):
        raise ValueError("ASG-CV predictor relation-control finiteness differs")
    exchange_ppm = (
        0
        if exchange_error == 0.0
        else math.ceil(exchange_error / max(exchange_scale, np.finfo(np.float64).tiny) * 1_000_000)
    )
    response_energy = float(torch.sum((positive64 - negative64).square()))
    reference_energy = float(0.5 * torch.sum(positive64.square() + negative64.square()))
    if (
        not math.isfinite(response_energy)
        or not math.isfinite(reference_energy)
        or response_energy <= 0.0
        or reference_energy <= 0.0
    ):
        raise ValueError("ASG-CV predictor relation liveness differs")
    relation_ppm = math.ceil(response_energy / reference_energy * 1_000_000)
    return AsgcvRelationControls(
        exchange_max_abs_error_ppm=exchange_ppm,
        relation_response_energy_ppm=relation_ppm,
    ).validated()


def source_bound_predictor(
    *,
    channel_dimensions: int,
    seed_sha256: object,
) -> AsgcvPatchGradientPredictor:
    """Initialize a replayable predictor without changing caller RNG state."""

    if (
        type(seed_sha256) is not str
        or len(seed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in seed_sha256)
    ):
        raise ValueError("ASG-CV predictor initialization seed differs")
    material = hashlib.sha256(_INITIALIZATION_DOMAIN + bytes.fromhex(seed_sha256)).digest()
    seed = int.from_bytes(material[:8], "big") & (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return AsgcvPatchGradientPredictor(
            channel_dimensions=channel_dimensions,
            predictor_rank=ASGCV_PREDICTOR_RANK,
        )


def canonical_predictor_state_bytes(predictor: object) -> bytes:
    """Serialize every named state tensor with exact framing and fp32 bytes."""

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
    return bytes(frame)


def predictor_state_sha256(predictor: object) -> str:
    """Hash the canonical predictor state bytes."""

    return hashlib.sha256(canonical_predictor_state_bytes(predictor)).hexdigest()


def predictor_from_state_bytes(
    raw: object,
    *,
    channel_dimensions: int,
) -> AsgcvPatchGradientPredictor:
    """Restore one canonical predictor state without mutating caller RNG state."""

    if type(raw) is not bytes or not raw.startswith(_STATE_DOMAIN):
        raise ValueError("ASG-CV predictor state bytes differ")
    seed = hashlib.sha256(raw).hexdigest()
    predictor = source_bound_predictor(
        channel_dimensions=channel_dimensions,
        seed_sha256=seed,
    )
    expected_state = predictor.state_dict()
    offset = len(_STATE_DOMAIN)
    restored: dict[str, torch.Tensor] = {}

    def read_u64() -> int:
        nonlocal offset
        if offset + 8 > len(raw):
            raise ValueError("ASG-CV predictor state frame differs")
        value = int.from_bytes(raw[offset : offset + 8], "big")
        offset += 8
        return value

    while offset < len(raw):
        name_length = read_u64()
        if name_length <= 0 or offset + name_length > len(raw):
            raise ValueError("ASG-CV predictor state name differs")
        try:
            name = raw[offset : offset + name_length].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ASG-CV predictor state name differs") from error
        offset += name_length
        if name not in expected_state or name in restored:
            raise ValueError("ASG-CV predictor state name differs")
        if restored and name <= max(restored):
            raise ValueError("ASG-CV predictor state order differs")
        dimensions = read_u64()
        expected_shape = tuple(int(size) for size in expected_state[name].shape)
        if dimensions != len(expected_shape):
            raise ValueError("ASG-CV predictor state shape differs")
        shape = tuple(read_u64() for _ in range(dimensions))
        if shape != expected_shape:
            raise ValueError("ASG-CV predictor state shape differs")
        count = math.prod(shape)
        byte_count = count * np.dtype("<f4").itemsize
        if offset + byte_count > len(raw):
            raise ValueError("ASG-CV predictor state payload differs")
        array = np.frombuffer(raw, dtype=np.dtype("<f4"), count=count, offset=offset).copy()
        offset += byte_count
        if not bool(np.isfinite(array).all()):
            raise ValueError("ASG-CV predictor state tensor differs")
        restored[name] = torch.from_numpy(array.reshape(shape))
    if offset != len(raw) or set(restored) != set(expected_state):
        raise ValueError("ASG-CV predictor state frame differs")
    predictor.load_state_dict(restored, strict=True)
    if canonical_predictor_state_bytes(predictor) != raw:
        raise ValueError("ASG-CV predictor state canonicalization differs")
    return predictor


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
