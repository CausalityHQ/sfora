from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_unicom_checkpoint_soup.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_unicom_checkpoint_soup", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suffix_windows_are_ordered_from_endpoint_to_longest_soup() -> None:
    module = _load_script()
    paths = tuple(Path(f"epoch-{epoch:04d}.pt") for epoch in (4, 8, 12, 16))

    assert module.suffix_windows(paths) == (
        (paths[3],),
        (paths[2], paths[3]),
        (paths[1], paths[2], paths[3]),
        paths,
    )


def test_average_model_states_uses_fp64_accumulation_and_exact_buffers() -> None:
    module = _load_script()
    left = OrderedDict(
        weight=torch.tensor([16_777_216.0, 1.0], dtype=torch.float32),
        counter=torch.tensor(7, dtype=torch.int64),
    )
    right = OrderedDict(
        weight=torch.tensor([-16_777_216.0, 3.0], dtype=torch.float32),
        counter=torch.tensor(7, dtype=torch.int64),
    )

    averaged = module.average_model_states((left, right))

    assert tuple(averaged) == ("weight", "counter")
    assert averaged["weight"].dtype == torch.float32
    assert torch.equal(averaged["weight"], torch.tensor([0.0, 2.0]))
    assert averaged["counter"].item() == 7


def test_average_model_states_rejects_changed_integer_buffer() -> None:
    module = _load_script()
    left = OrderedDict(counter=torch.tensor(1, dtype=torch.int64))
    right = OrderedDict(counter=torch.tensor(2, dtype=torch.int64))

    with pytest.raises(ValueError, match="non-floating tensor differs"):
        module.average_model_states((left, right))


def test_interpolation_uses_initial_plus_alpha_times_soup_delta() -> None:
    module = _load_script()
    initial = OrderedDict(
        weight=torch.tensor([2.0, -2.0]), counter=torch.tensor(3, dtype=torch.int64)
    )
    soup = OrderedDict(weight=torch.tensor([6.0, 2.0]), counter=torch.tensor(3, dtype=torch.int64))

    result = module.interpolate_model_states(initial, soup, alpha=0.25)

    assert torch.equal(result["weight"], torch.tensor([3.0, -1.0]))
    assert result["counter"].item() == 3


def test_interpolation_carries_trained_nonfloating_buffers() -> None:
    module = _load_script()
    initial = OrderedDict(counter=torch.tensor(0, dtype=torch.int64))
    soup = OrderedDict(counter=torch.tensor(16, dtype=torch.int64))

    result = module.interpolate_model_states(initial, soup, alpha=0.5)

    assert result["counter"].item() == 16


def test_candidate_selection_is_metric_first_and_stable_on_ties() -> None:
    module = _load_script()
    candidates = [
        {"name": "first", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.7}},
        {"name": "map-win", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.71}},
        {"name": "recall-win", "metrics": {"recall_at_1": 0.81, "map_at_r": 0.6}},
        {"name": "same", "metrics": {"recall_at_1": 0.81, "map_at_r": 0.6}},
    ]

    assert module.select_candidate(candidates)["name"] == "recall-win"


def test_evaluate_grid_loads_each_real_interpolated_state() -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    initial = OrderedDict(weight=torch.tensor([[0.0]]))
    checkpoints = (
        (Path("epoch-0001.pt"), OrderedDict(weight=torch.tensor([[1.0]]))),
        (Path("epoch-0002.pt"), OrderedDict(weight=torch.tensor([[3.0]]))),
    )

    candidates = module.evaluate_grid(
        model,
        initial,
        checkpoints,
        alphas=(0.5, 1.0),
        evaluate=lambda: {
            "recall_at_1": float(model.weight.item()),
            "map_at_r": float(model.weight.item()) / 10.0,
        },
    )

    assert [row["name"] for row in candidates] == [
        "epochs-2-alpha-0.5",
        "epochs-2-alpha-1",
        "epochs-1_2-alpha-0.5",
        "epochs-1_2-alpha-1",
    ]
    assert [row["metrics"]["recall_at_1"] for row in candidates] == [
        1.5,
        3.0,
        1.0,
        2.0,
    ]
