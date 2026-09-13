"""Deterministic averaging for compatible deployable model states."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import torch

_MISSING_METADATA = object()


def _metadata_equal(left: Any, right: Any) -> bool:
    if left is _MISSING_METADATA or right is _MISSING_METADATA:
        return left is right
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        left_keys = tuple(left)
        return left_keys == tuple(right) and all(
            _metadata_equal(left[key], right[key]) for key in left_keys
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _metadata_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if left is None or type(left) in (bool, int, float, str):
        return bool(left == right)
    return False


def average_compatible_model_states(
    states: tuple[Mapping[str, torch.Tensor], ...],
) -> OrderedDict[str, torch.Tensor]:
    """Average floating state in FP64 while requiring exact discrete buffers.

    The input order is authoritative. All replicas must expose the same ordered
    state-dict schema and tensor metadata. Integer and Boolean buffers are copied
    only when every replica agrees exactly, which keeps the result independent of
    an arbitrary "last checkpoint" convention.
    """

    if (
        type(states) is not tuple
        or not states
        or any(not isinstance(state, Mapping) for state in states)
    ):
        raise ValueError("compatible model state differs")
    keys = tuple(states[0])
    if (
        not keys
        or any(type(key) is not str or not key for key in keys)
        or any(tuple(state) != keys for state in states[1:])
    ):
        raise ValueError("compatible model state differs")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    first_metadata = getattr(states[0], "_metadata", _MISSING_METADATA)
    if any(
        not _metadata_equal(first_metadata, getattr(state, "_metadata", _MISSING_METADATA))
        for state in states[1:]
    ):
        raise ValueError("compatible model state differs")
    with torch.no_grad():
        for key in keys:
            values = tuple(state[key] for state in states)
            if any(type(value) is not torch.Tensor for value in values):
                raise TypeError(f"model state value is not a tensor: {key}")
            first = values[0]
            if (
                first.layout != torch.strided
                or first.is_complex()
                or any(
                    value.shape != first.shape
                    or value.dtype != first.dtype
                    or value.device != first.device
                    or value.layout != first.layout
                    for value in values[1:]
                )
            ):
                raise ValueError("compatible model state differs")
            if first.is_floating_point():
                if any(not bool(torch.isfinite(value).all()) for value in values):
                    raise ValueError("compatible model state differs")
                accumulator = first.detach().to(torch.float64).clone()
                for count, value in enumerate(values[1:], start=2):
                    accumulator.mul_((count - 1) / count).add_(
                        value.detach().to(torch.float64), alpha=1 / count
                    )
                averaged = accumulator.to(first.dtype).contiguous()
                if not bool(torch.isfinite(averaged).all()):
                    raise ValueError("compatible model state differs")
                result[key] = averaged
            else:
                if any(not torch.equal(first, value) for value in values[1:]):
                    raise ValueError("compatible model state differs")
                result[key] = first.detach().clone().contiguous()
    if first_metadata is not _MISSING_METADATA:
        result._metadata = deepcopy(first_metadata)  # type: ignore[attr-defined]
    return result
