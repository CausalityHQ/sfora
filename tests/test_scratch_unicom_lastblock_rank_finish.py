from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "_scratch_unicom_lastblock_rank_finish.py"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location(
        "scratch_unicom_lastblock_rank_finish", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ToyUniCOM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(24))
        self.norm = torch.nn.LayerNorm(4)
        self.feature = torch.nn.Linear(4, 3)


def test_freeze_except_final_block_exposes_only_block_23() -> None:
    subject = _load_subject()
    model = _ToyUniCOM()

    authority = subject.freeze_except_final_block(model)

    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    assert trainable == ("blocks.23.weight", "blocks.23.bias")
    assert authority == {
        "block_index": 23,
        "parameter_names": ["blocks.23.weight", "blocks.23.bias"],
        "trainable_parameters": 20,
        "total_parameters": sum(value.numel() for value in model.parameters()),
    }


def test_classify_candidate_requires_effect_ci_and_recall_nonregression() -> None:
    subject = _load_subject()
    baseline = {"map_at_r": 0.86, "recall_at_1": 0.97}

    passing = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.971},
        {"lower": 0.001, "median": 0.006, "upper": 0.011},
    )
    small = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.8649, "recall_at_1": 0.971},
        {"lower": 0.001, "median": 0.0049, "upper": 0.009},
    )
    uncertain = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.971},
        {"lower": 0.0, "median": 0.006, "upper": 0.012},
    )
    recall_loss = subject.classify_candidate(
        baseline,
        {"map_at_r": 0.866, "recall_at_1": 0.9699},
        {"lower": 0.001, "median": 0.006, "upper": 0.011},
    )

    assert passing == {
        "map_at_r_delta": 0.006000000000000005,
        "recall_at_1_delta": 0.0010000000000000009,
        "minimum_map_at_r_delta": 0.005,
        "require_positive_interval_lower": True,
        "minimum_recall_at_1_delta": 0.0,
        "passed": True,
    }
    assert small["passed"] is False
    assert uncertain["passed"] is False
    assert recall_loss["passed"] is False
