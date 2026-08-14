from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

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


def test_average_model_states_carries_latest_nonfloating_buffer() -> None:
    module = _load_script()
    left = OrderedDict(counter=torch.tensor(1, dtype=torch.int64))
    right = OrderedDict(counter=torch.tensor(2, dtype=torch.int64))

    result = module.average_model_states((left, right))

    assert result["counter"].item() == 2


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


def test_candidate_selection_is_map_first_then_recall_and_stable_on_ties() -> None:
    module = _load_script()
    candidates = [
        {"name": "first", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.7}},
        {"name": "map-win", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.71}},
        {"name": "recall-win", "metrics": {"recall_at_1": 0.81, "map_at_r": 0.6}},
        {"name": "same-map", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.71}},
    ]

    assert module.select_candidate(candidates)["name"] == "map-win"


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


def test_evaluate_grid_rejects_duplicate_epoch_names_across_directories() -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    initial = OrderedDict(weight=torch.tensor([[0.0]]))
    checkpoints = (
        (Path("run-a/epoch-0004.pt"), OrderedDict(weight=torch.tensor([[1.0]]))),
        (Path("run-b/epoch-0004.pt"), OrderedDict(weight=torch.tensor([[2.0]]))),
    )

    with pytest.raises(ValueError, match="unique increasing epochs"):
        module.evaluate_grid(
            model,
            initial,
            checkpoints,
            alphas=(1.0,),
            prepare=lambda: None,
            evaluate=lambda: {"recall_at_1": 1.0, "map_at_r": 1.0},
        )


def test_evaluate_grid_prepares_every_candidate_before_scoring() -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    initial = OrderedDict(weight=torch.tensor([[0.0]]))
    checkpoints = (
        (Path("epoch-0001.pt"), OrderedDict(weight=torch.tensor([[1.0]]))),
        (Path("epoch-0002.pt"), OrderedDict(weight=torch.tensor([[3.0]]))),
    )
    prepared: list[float] = []

    candidates = module.evaluate_grid(
        model,
        initial,
        checkpoints,
        alphas=(1.0,),
        prepare=lambda: prepared.append(float(model.weight.item())),
        evaluate=lambda: {"recall_at_1": prepared[-1], "map_at_r": 0.0},
    )

    assert prepared == [3.0, 2.0]
    assert [row["metrics"]["recall_at_1"] for row in candidates] == prepared


def test_recalibrate_batch_norm_resets_and_updates_only_bn_statistics() -> None:
    module = _load_script()
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.BatchNorm1d(2, momentum=0.1),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[1].running_mean.fill_(99.0)
    model.eval()
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            torch.arange(4),
        ),
        batch_size=2,
    )

    module.recalibrate_batch_norm(model, loader, device=torch.device("cpu"))

    assert torch.equal(model[1].running_mean, torch.tensor([4.0, 5.0]))
    assert model[1].num_batches_tracked.item() == 2
    assert model[1].momentum == 0.1
    assert not model.training and not model[1].training


def test_checkpoint_loader_rejects_holdout_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    path = tmp_path / "epoch-0001.pt"
    torch.save(
        {
            "epoch": 1,
            "model": OrderedDict(weight=torch.tensor([1.0])),
            "selection_holdout": {"seed": 5, "fraction": 0.2},
        },
        path,
    )

    with pytest.raises(ValueError, match="selection holdout differs"):
        module._load_checkpoint_states((path,), holdout_seed=0, holdout_fraction=0.2)


def test_checkpoint_loader_rejects_mixed_training_protocols(tmp_path: Path) -> None:
    module = _load_script()
    paths = tuple(tmp_path / f"epoch-{epoch:04d}.pt" for epoch in (1, 2))
    for epoch, path in enumerate(paths, start=1):
        torch.save(
            {
                "epoch": epoch,
                "model": OrderedDict(weight=torch.tensor([float(epoch)])),
                "selection_holdout": {"seed": 0, "fraction": 0.2},
                "training_protocol": {"seed": epoch - 1, "objective": "official-eight-mask"},
            },
            path,
        )

    with pytest.raises(ValueError, match="training protocol differs"):
        module._load_checkpoint_states(paths, holdout_seed=0, holdout_fraction=0.2)
