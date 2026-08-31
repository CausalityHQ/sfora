"""Pair-conditioned rank-16 patch-gradient predictor for ASG-CV."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from sfora.asgcv import ASGCV_PREDICTOR_RANK

_STATE_DOMAIN = b"sfora-asgcv-predictor-state-v1\0"


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
