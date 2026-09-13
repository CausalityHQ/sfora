from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

import sfora
from sfora.model_soup import average_compatible_model_states


def test_average_compatible_model_states_uses_fp64_and_preserves_exact_buffers() -> None:
    states = (
        OrderedDict(
            weight=torch.tensor([16_777_216.0, 1.0], dtype=torch.float32),
            counter=torch.tensor(7, dtype=torch.int64),
        ),
        OrderedDict(
            weight=torch.tensor([1.0, 2.0], dtype=torch.float32),
            counter=torch.tensor(7, dtype=torch.int64),
        ),
        OrderedDict(
            weight=torch.tensor([-16_777_216.0, 3.0], dtype=torch.float32),
            counter=torch.tensor(7, dtype=torch.int64),
        ),
    )

    averaged = average_compatible_model_states(states)

    assert sfora.average_compatible_model_states is average_compatible_model_states
    assert tuple(averaged) == ("weight", "counter")
    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["weight"], torch.tensor([1.0 / 3.0, 2.0]))
    assert averaged["counter"].item() == 7


def test_average_compatible_model_states_avoids_fp64_sum_overflow() -> None:
    maximum = torch.finfo(torch.float64).max
    states = (
        OrderedDict(weight=torch.tensor([maximum], dtype=torch.float64)),
        OrderedDict(weight=torch.tensor([maximum], dtype=torch.float64)),
    )

    averaged = average_compatible_model_states(states)

    assert torch.equal(averaged["weight"], torch.tensor([maximum], dtype=torch.float64))


def test_average_compatible_model_states_preserves_compatible_metadata() -> None:
    first = OrderedDict(weight=torch.tensor([1.0]))
    second = OrderedDict(weight=torch.tensor([3.0]))
    first._metadata = OrderedDict(  # type: ignore[attr-defined]
        (("", {"version": 2}), ("layer", {"version": 1}))
    )
    second._metadata = OrderedDict(  # type: ignore[attr-defined]
        (("", {"version": 2}), ("layer", {"version": 1}))
    )

    averaged = average_compatible_model_states((first, second))

    assert averaged._metadata == first._metadata  # type: ignore[attr-defined]
    assert averaged._metadata is not first._metadata  # type: ignore[attr-defined]


def test_average_compatible_model_states_rejects_metadata_drift() -> None:
    first = OrderedDict(weight=torch.tensor([1.0]))
    second = OrderedDict(weight=torch.tensor([3.0]))
    first._metadata = OrderedDict((("", {"version": 2}),))  # type: ignore[attr-defined]
    second._metadata = OrderedDict((("", {"version": 1}),))  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="compatible model state"):
        average_compatible_model_states((first, second))


def test_average_compatible_model_states_returns_detached_non_aliasing_state() -> None:
    source = torch.tensor([[1.0, 3.0]], requires_grad=True)
    states = (
        OrderedDict(weight=source, mask=torch.tensor([True, False])),
        OrderedDict(weight=torch.tensor([[3.0, 5.0]]), mask=torch.tensor([True, False])),
    )

    averaged = average_compatible_model_states(states)
    averaged["weight"].add_(10.0)
    averaged["mask"][0] = False

    assert not averaged["weight"].requires_grad
    assert torch.equal(source, torch.tensor([[1.0, 3.0]]))
    assert torch.equal(states[0]["mask"], torch.tensor([True, False]))


@pytest.mark.parametrize(
    "states",
    (
        (),
        (OrderedDict(),),
        (
            OrderedDict(weight=torch.ones(2)),
            OrderedDict(bias=torch.ones(2)),
        ),
        (
            OrderedDict(weight=torch.ones(2)),
            OrderedDict(weight=torch.ones(3)),
        ),
        (
            OrderedDict(weight=torch.ones(2, dtype=torch.float32)),
            OrderedDict(weight=torch.ones(2, dtype=torch.float64)),
        ),
        (
            OrderedDict(counter=torch.tensor(1)),
            OrderedDict(counter=torch.tensor(2)),
        ),
        (
            OrderedDict(weight=torch.tensor([float("nan")])),
            OrderedDict(weight=torch.ones(1)),
        ),
    ),
)
def test_average_compatible_model_states_rejects_incompatible_or_nonfinite_state(
    states: tuple[OrderedDict[str, torch.Tensor], ...],
) -> None:
    with pytest.raises(ValueError, match="compatible model state"):
        average_compatible_model_states(states)


def test_average_compatible_model_states_rejects_non_tensor_values() -> None:
    with pytest.raises(TypeError, match="model state value is not a tensor"):
        average_compatible_model_states((OrderedDict(weight=object()),))  # type: ignore[arg-type]


def test_average_compatible_model_states_rejects_non_mapping_state() -> None:
    with pytest.raises(ValueError, match="compatible model state"):
        average_compatible_model_states((object(),))  # type: ignore[arg-type]
