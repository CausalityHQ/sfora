from collections import OrderedDict

import pytest
import torch

from sfora.weight_space_transfer import (
    INTERPOLATION_ALPHAS,
    FoldedInferenceState,
    interpolate_inference_state,
    model_state_sha256,
)


def _states() -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    initial = OrderedDict(
        {
            "tower.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "projection.weight": torch.tensor([[2.0, -2.0]]),
            "tower.position_ids": torch.tensor([0, 1], dtype=torch.int64),
            "proxies": torch.tensor([[9.0, 8.0]]),
        }
    )
    trained = OrderedDict(
        {
            "tower.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "projection.weight": torch.tensor([[6.0, 2.0]]),
            "tower.position_ids": torch.tensor([0, 1], dtype=torch.int64),
            "proxies": torch.tensor([[1.0, 2.0]]),
        }
    )
    return initial, trained


def test_fixed_alpha_authority_and_exact_endpoints() -> None:
    assert INTERPOLATION_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    initial, trained = _states()

    start = interpolate_inference_state(initial, trained, alpha=0.0)
    middle = interpolate_inference_state(initial, trained, alpha=0.5)
    end = interpolate_inference_state(initial, trained, alpha=1.0)

    assert type(start) is FoldedInferenceState
    assert tuple(start.state) == (
        "tower.weight",
        "projection.weight",
        "tower.position_ids",
        "proxies",
    )
    assert torch.equal(start.state["tower.weight"], initial["tower.weight"])
    assert torch.equal(end.state["tower.weight"], trained["tower.weight"])
    assert torch.equal(middle.state["tower.weight"], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(start.state["projection.weight"], trained["projection.weight"])
    assert torch.equal(middle.state["projection.weight"], trained["projection.weight"])
    assert torch.equal(middle.state["proxies"], trained["proxies"])
    assert middle.tower_squared_displacement == pytest.approx(16.0)
    assert start.sha256 == model_state_sha256(start.state)
    assert end.sha256 == model_state_sha256(end.state)
    replay = interpolate_inference_state(initial, trained, alpha=0.5)
    assert replay.sha256 == middle.sha256
    assert replay.tower_squared_displacement == middle.tower_squared_displacement
    assert all(torch.equal(replay.state[name], value) for name, value in middle.state.items())


@pytest.mark.parametrize("alpha", [-0.1, 0.1, 1.1, True, 1])
def test_interpolation_rejects_unregistered_or_concrete_type_alpha(alpha: object) -> None:
    initial, trained = _states()
    with pytest.raises(ValueError, match="alpha"):
        interpolate_inference_state(initial, trained, alpha=alpha)  # type: ignore[arg-type]


def test_interpolation_rejects_schema_shape_dtype_and_nonfinite_drift() -> None:
    initial, trained = _states()
    mutations = []

    missing = trained.copy()
    del missing["tower.weight"]
    mutations.append(missing)

    shape = trained.copy()
    shape["tower.weight"] = torch.zeros(3)
    mutations.append(shape)

    dtype = trained.copy()
    dtype["tower.weight"] = trained["tower.weight"].double()
    mutations.append(dtype)

    nonfinite = trained.copy()
    nonfinite["tower.weight"] = trained["tower.weight"].clone()
    nonfinite["tower.weight"][0, 0] = float("nan")
    mutations.append(nonfinite)

    integer = trained.copy()
    integer["tower.position_ids"] = torch.tensor([0, 2], dtype=torch.int64)
    mutations.append(integer)

    for mutation in mutations:
        with pytest.raises(ValueError):
            interpolate_inference_state(initial, mutation, alpha=0.5)


def test_interpolation_rejects_invalid_mapping_names_and_proxy_shape() -> None:
    initial, trained = _states()

    with pytest.raises(ValueError):
        interpolate_inference_state(dict(initial), trained, alpha=0.5)  # type: ignore[arg-type]

    bad_name = initial.copy()
    bad_name[""] = bad_name.pop("tower.weight")
    with pytest.raises(ValueError):
        interpolate_inference_state(bad_name, trained, alpha=0.5)

    proxy_shape = trained.copy()
    proxy_shape["proxies"] = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        interpolate_inference_state(initial, proxy_shape, alpha=0.5)


def test_state_digest_binds_order_metadata_and_bytes() -> None:
    initial, _ = _states()
    state = initial
    same = OrderedDict(reversed(tuple(state.items())))
    changed = state.copy()
    changed["tower.weight"] = changed["tower.weight"].clone()
    changed["tower.weight"][0, 0] += 1.0

    assert model_state_sha256(state) == model_state_sha256(same)
    assert model_state_sha256(state) != model_state_sha256(changed)
    with pytest.raises(ValueError):
        model_state_sha256({"tower.weight": state["tower.weight"]})  # type: ignore[arg-type]
