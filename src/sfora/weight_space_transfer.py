"""Pure weight-space transfer interpolation authority."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

INTERPOLATION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class FoldedInferenceState:
    """One deterministic tower fold with trained retrieval-head state."""

    alpha: float
    state: OrderedDict[str, torch.Tensor]
    sha256: str
    tower_squared_displacement: float


def _canonical_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def _validate_state(state: object, *, role: str) -> OrderedDict[str, torch.Tensor]:
    if type(state) is not OrderedDict or not state:
        raise ValueError(f"{role} state must be a nonempty concrete OrderedDict")
    typed = state
    for name, tensor in typed.items():
        if type(name) is not str or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{role} state schema differs")
        if tensor.layout != torch.strided:
            raise ValueError(f"{role} state tensors must use strided layout")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{role} state tensors must be finite")
    if "projection.weight" not in typed or "proxies" not in typed:
        raise ValueError(f"{role} state lacks the registered retrieval head")
    if not any(name.startswith("tower.") for name in typed):
        raise ValueError(f"{role} state lacks the registered tower")
    return typed


def model_state_sha256(state: object) -> str:
    """Hash tensor names, metadata, and bytes using the control-run authority."""

    typed = _validate_state(state, role="model")
    digest = hashlib.sha256()
    for name, tensor in sorted(typed.items()):
        metadata = _canonical_bytes(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)}
        )
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def interpolate_inference_state(
    initial_state: object,
    trained_state: object,
    *,
    alpha: float,
) -> FoldedInferenceState:
    """Interpolate only the tower and carry the trained projection/proxies."""

    if type(alpha) is not float or alpha not in INTERPOLATION_ALPHAS:
        raise ValueError("alpha differs from the registered interpolation grid")
    initial = _validate_state(initial_state, role="initial")
    trained = _validate_state(trained_state, role="trained")
    if set(initial) != set(trained):
        raise ValueError("endpoint state names differ")

    folded: OrderedDict[str, torch.Tensor] = OrderedDict()
    tower_squared_displacement = 0.0
    for name, left in initial.items():
        right = trained[name]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError("endpoint tensor metadata differs")
        if left.is_floating_point() != right.is_floating_point():
            raise ValueError("endpoint tensor kinds differ")
        if not left.is_floating_point() and not torch.equal(left, right):
            raise ValueError("non-floating endpoint tensors differ")

        if name.startswith("tower.") and left.is_floating_point():
            if alpha == 0.0:
                value = left.detach().cpu().clone()
            elif alpha == 1.0:
                value = right.detach().cpu().clone()
            else:
                value = torch.lerp(left.detach().cpu().float(), right.detach().cpu().float(), alpha)
                value = value.to(left.dtype)
            delta = value.double() - left.detach().cpu().double()
            tower_squared_displacement += float(torch.sum(delta * delta))
        else:
            value = right.detach().cpu().clone()
        folded[name] = value

    if not math.isfinite(tower_squared_displacement):
        raise ValueError("tower displacement must be finite")
    return FoldedInferenceState(
        alpha=alpha,
        state=folded,
        sha256=model_state_sha256(folded),
        tower_squared_displacement=tower_squared_displacement,
    )

