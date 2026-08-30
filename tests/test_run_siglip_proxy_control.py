"""Authenticated runner tests for the SigLIP pooled control."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_siglip_proxy_control.py"
_SPEC = importlib.util.spec_from_file_location("run_siglip_proxy_control", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_canonical_bytes_and_create_new_publication(tmp_path: Path) -> None:
    payload = {"z": 1, "a": {"finite": True}, "claim_eligible": False}
    expected = b'{"a":{"finite":true},"claim_eligible":false,"z":1}\n'
    assert _MODULE._canonical_bytes(payload) == expected

    output = tmp_path / "receipt.json"
    _MODULE._write_new(output, expected)
    assert output.read_bytes() == expected
    with pytest.raises(FileExistsError):
        _MODULE._write_new(output, expected)
    assert not list(tmp_path.glob("*.partial"))


def test_schedule_is_warmup_inclusive_and_epoch_bound() -> None:
    config = SiglipProxyControlConfig()
    steps_per_epoch = 2

    assert _MODULE._learning_rate_multiplier(config, step=0, steps_per_epoch=2) == 0.1
    assert _MODULE._learning_rate_multiplier(config, step=9, steps_per_epoch=2) == 1.0
    assert _MODULE._learning_rate_multiplier(config, step=19, steps_per_epoch=2) == 1.0
    assert _MODULE._learning_rate_multiplier(config, step=20, steps_per_epoch=2) == 0.5
    assert _MODULE._learning_rate_multiplier(config, step=40, steps_per_epoch=2) == 0.25
    assert _MODULE._learning_rate_multiplier(config, step=120, steps_per_epoch=2) == 0.03125
    with pytest.raises(ValueError, match="step"):
        _MODULE._learning_rate_multiplier(config, step=-1, steps_per_epoch=steps_per_epoch)


def _sampler_fixture() -> tuple[tuple[str, ...], torch.Tensor]:
    example_ids = tuple(
        f"cars-{label:02d}-{position:02d}" for label in range(49) for position in range(7)
    )
    labels = torch.tensor(
        [label for label in range(49) for _ in range(7)],
        dtype=torch.int64,
    )
    return example_ids, labels


def test_sampler_has_exact_stateless_classes_and_persistent_example_cycles() -> None:
    example_ids, labels = _sampler_fixture()
    state = _MODULE.SamplerState.initial()

    batches, after = _MODULE._build_epoch_batches(
        example_ids=example_ids,
        labels=labels,
        seed=17,
        epoch=0,
        steps_per_epoch=2,
        state=state,
    )
    repeated, repeated_after = _MODULE._build_epoch_batches(
        example_ids=example_ids,
        labels=labels,
        seed=17,
        epoch=0,
        steps_per_epoch=2,
        state=state,
    )

    assert batches == repeated
    assert after == repeated_after
    assert len(batches) == 2
    for batch in batches:
        assert len(batch) == 120
        assert len(set(batch)) == 120
        batch_labels = labels[list(batch)]
        unique, counts = torch.unique(batch_labels, return_counts=True)
        assert unique.numel() == 30
        assert counts.tolist() == [4] * 30
    next_epoch, _ = _MODULE._build_epoch_batches(
        example_ids=example_ids,
        labels=labels,
        seed=17,
        epoch=1,
        steps_per_epoch=2,
        state=after,
    )
    assert next_epoch != batches


class _TinyTower(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(5, 4)
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.norm(self.linear(inputs)))


def test_optimizer_groups_cover_every_parameter_exactly_once() -> None:
    model = PooledProxyAnchorModel(
        tower=_TinyTower(),
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=49,
    )

    groups = _MODULE._optimizer_groups(model, SiglipProxyControlConfig())

    grouped_ids = [id(parameter) for group in groups for parameter in group["params"]]
    trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
    assert sorted(grouped_ids) == sorted(trainable_ids)
    assert len(grouped_ids) == len(set(grouped_ids))
    assert {float(group["lr"]) for group in groups} == {1.0e-5, 1.0e-4, 1.0e-2}
    proxy_group = next(group for group in groups if float(group["lr"]) == 1.0e-2)
    assert float(proxy_group["weight_decay"]) == 0.0
    assert any(float(group["weight_decay"]) == 1.0e-4 for group in groups)


@pytest.mark.parametrize(
    "payload",
    [
        {"claim_eligible": 0},
        {"claim_eligible": False, "value": float("nan")},
        {"claim_eligible": False, "value": object()},
    ],
)
def test_canonical_bytes_rejects_type_and_value_drift(payload: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _MODULE._canonical_bytes(payload)


def _passing_smoke_observation(microbatch_size: int) -> Any:
    return _MODULE.SmokeObservation(
        microbatch_size=microbatch_size,
        steps_completed=3,
        peak_process_rss_bytes=8 * 1024**3,
        peak_cuda_allocated_bytes=16 * 1024**3,
        peak_cuda_reserved_bytes=20 * 1024**3,
        memory_psi_growth=0.0,
        swap_growth_bytes=0,
        examples_per_second=10.0,
        final_loss=2.5,
        complete_tower_gradient_coverage=True,
        maximum_score_disagreement=1.0e-6,
    )


def test_memory_smoke_selects_first_rung_passing_every_registered_gate() -> None:
    calls: list[int] = []

    def run_rung(microbatch_size: int) -> Any:
        calls.append(microbatch_size)
        observation = _passing_smoke_observation(microbatch_size)
        if microbatch_size == 120:
            return replace(
                observation,
                peak_process_rss_bytes=60 * 1024**3,
                peak_cuda_reserved_bytes=40 * 1024**3,
            )
        if microbatch_size == 60:
            return replace(observation, maximum_score_disagreement=3.0e-5)
        if microbatch_size == 40:
            return replace(observation, examples_per_second=0.01)
        return observation

    receipt = _MODULE.run_memory_smoke(
        config=SiglipProxyControlConfig(),
        steps_per_epoch=2,
        run_rung=run_rung,
    )

    assert calls == [120, 60, 40, 30]
    assert receipt.selected_microbatch_size == 30
    assert tuple(row.microbatch_size for row in receipt.observations) == (120, 60, 40, 30)
    assert receipt.projected_seed_seconds == pytest.approx(60 * 2 * 120 / 10.0)


def test_memory_smoke_fails_closed_when_no_rung_passes() -> None:
    def run_rung(microbatch_size: int) -> Any:
        return replace(
            _passing_smoke_observation(microbatch_size),
            complete_tower_gradient_coverage=False,
        )

    with pytest.raises(RuntimeError, match="no smoke microbatch"):
        _MODULE.run_memory_smoke(
            config=SiglipProxyControlConfig(),
            steps_per_epoch=2,
            run_rung=run_rung,
        )


def test_checkpoint_publication_rotates_only_after_new_authority_is_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=1_000, free=9_000),
    )
    first = _MODULE.publish_epoch_checkpoint(
        directory=tmp_path,
        seed=17,
        epoch=1,
        write_checkpoint=lambda path: path.write_bytes(b"checkpoint-one"),
        maximum_checkpoint_bytes=100,
    )
    assert first.path.read_bytes() == b"checkpoint-one"
    assert first.receipt_path.is_file()
    assert _MODULE.latest_authenticated_checkpoint(tmp_path, seed=17) == first

    second = _MODULE.publish_epoch_checkpoint(
        directory=tmp_path,
        seed=17,
        epoch=2,
        write_checkpoint=lambda path: path.write_bytes(b"checkpoint-two"),
        maximum_checkpoint_bytes=100,
    )
    assert not first.path.exists()
    assert not first.receipt_path.exists()
    assert second.path.read_bytes() == b"checkpoint-two"
    assert _MODULE.latest_authenticated_checkpoint(tmp_path, seed=17) == second
    assert not list(tmp_path.glob("*.partial"))


def test_checkpoint_resume_rejects_corruption_and_free_space_shortfall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10_000, used=1_000, free=9_000),
    )
    authority = _MODULE.publish_epoch_checkpoint(
        directory=tmp_path,
        seed=29,
        epoch=7,
        write_checkpoint=lambda path: path.write_bytes(b"authenticated-state"),
        maximum_checkpoint_bytes=100,
    )
    authority.path.write_bytes(b"corrupted-state")
    with pytest.raises(ValueError, match="checkpoint digest"):
        _MODULE.latest_authenticated_checkpoint(tmp_path, seed=29)

    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )
    with pytest.raises(OSError, match="free space"):
        _MODULE.publish_epoch_checkpoint(
            directory=empty,
            seed=43,
            epoch=1,
            write_checkpoint=lambda path: path.write_bytes(b"state"),
            maximum_checkpoint_bytes=100,
        )
    assert not list(empty.iterdir())
