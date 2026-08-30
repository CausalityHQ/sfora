"""Authenticated runner tests for the SigLIP pooled control."""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from collections import UserDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sfora.data import ImageExample
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
    assert _MODULE._learning_rate_multiplier(config, step=17, steps_per_epoch=2) == 1.0
    assert _MODULE._learning_rate_multiplier(config, step=18, steps_per_epoch=2) == 0.5
    assert _MODULE._learning_rate_multiplier(config, step=19, steps_per_epoch=2) == 0.5
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


def test_optimizer_decay_excludes_layer_norm_by_module_type_not_name_spelling() -> None:
    class LayerNormSpellingTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(5, 4)
            self.layer_norm1 = torch.nn.LayerNorm(4)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.layer_norm1(self.linear(inputs)))

    tower = LayerNormSpellingTower()
    model = PooledProxyAnchorModel(
        tower=tower,
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=49,
    )
    groups = _MODULE._optimizer_groups(model, SiglipProxyControlConfig())
    decay_by_parameter = {
        id(parameter): float(group["weight_decay"])
        for group in groups
        for parameter in group["params"]
    }

    assert decay_by_parameter[id(tower.layer_norm1.weight)] == 0.0
    assert decay_by_parameter[id(tower.layer_norm1.bias)] == 0.0
    assert decay_by_parameter[id(tower.linear.weight)] == 1.0e-4


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


def test_memory_smoke_records_cuda_oom_and_descends_to_next_rung(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    cache_clears: list[None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: cache_clears.append(None))

    def run_rung(microbatch_size: int) -> Any:
        calls.append(microbatch_size)
        if microbatch_size == 120:
            raise torch.cuda.OutOfMemoryError("registered smoke OOM")
        return _passing_smoke_observation(microbatch_size)

    receipt = _MODULE.run_memory_smoke(
        config=SiglipProxyControlConfig(),
        steps_per_epoch=2,
        run_rung=run_rung,
    )

    assert calls == [120, 60]
    assert cache_clears == [None]
    assert receipt.observations[0].microbatch_size == 120
    assert receipt.observations[0].failure_reason == "cuda-out-of-memory"
    assert receipt.selected_microbatch_size == 60


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

    orphan = tmp_path / "seed-017-epoch-002.pt"
    orphan.write_bytes(b"interrupted-before-receipt")
    partial = tmp_path / "seed-017-epoch-002.pt.partial"
    partial.write_bytes(b"interrupted-write")

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


def _control_examples() -> list[ImageExample]:
    minimums = {**{label: 4 for label in range(49)}, **{label: 2 for label in range(49, 98)}}
    return [
        ImageExample(
            example_id=f"cars-{label:03d}-{position:03d}",
            image=object(),
            label=label,
        )
        for label, count in minimums.items()
        for position in range(count)
    ]


def _run_authority(*, manifest_sha256: str = "1" * 64) -> Any:
    return _MODULE.ControlRunAuthority(
        source_revision="2" * 40,
        source_tree_digest="3" * 64,
        manifest_sha256=manifest_sha256,
        torch_version=str(torch.__version__),
        transformers_version="4.test",
        torchvision_version="0.test",
        cuda_runtime=torch.version.cuda,
        device_name="cpu",
        microbatch_size=30,
        steps_per_epoch=1,
    )


def test_control_data_boundary_loads_only_cars_train_and_never_test_classes() -> None:
    calls: list[tuple[str, str]] = []

    def loader(*, dataset_name: str, split: str) -> list[ImageExample]:
        calls.append((dataset_name, split))
        return _control_examples()

    bands = _MODULE.load_control_examples(loader=loader)

    assert calls == [("cars", "train")]
    assert {example.label for example in bands.optimization} == set(range(49))
    assert {example.label for example in bands.clean_validation} == set(range(49, 82))
    assert {example.label for example in bands.burned_diagnostic} == set(range(82, 98))
    assert max(example.label for example in bands.ordered_manifest) == 97
    with pytest.raises(ValueError, match="official test"):
        _MODULE.load_control_examples(
            loader=lambda **_kwargs: (
                _control_examples()
                + [ImageExample(example_id="forbidden", image=object(), label=98)]
            )
        )


def test_siglip_component_boundary_is_pinned_local_eager_and_nonreentrant() -> None:
    calls: dict[str, Any] = {}

    class FakeVision(torch.nn.Module):
        config = SimpleNamespace(
            _attn_implementation="eager",
            _commit_hash=SiglipProxyControlConfig().model_revision,
        )

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeVision:
            calls["vision"] = (model_name, kwargs)
            return cls()

        def gradient_checkpointing_enable(self, **kwargs: object) -> None:
            calls["checkpointing"] = kwargs

        def forward(self, *, pixel_values: torch.Tensor, return_dict: bool) -> object:
            assert return_dict is True
            return SimpleNamespace(pooler_output=pixel_values.mean(dim=(-2, -1)))

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeProcessor:
            calls["processor"] = (model_name, kwargs)
            return cls()

    config = SiglipProxyControlConfig()
    tower, processor = _MODULE.load_siglip_control_components(
        config=config,
        vision_model_cls=FakeVision,
        processor_cls=FakeProcessor,
    )

    expected_load = {
        "revision": config.model_revision,
        "local_files_only": True,
        "attn_implementation": "eager",
    }
    assert calls["vision"] == (config.model_name, expected_load)
    assert calls["processor"] == (
        config.model_name,
        {"revision": config.model_revision, "local_files_only": True},
    )
    assert calls["checkpointing"] == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert processor.__class__ is FakeProcessor
    pixels = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    assert torch.equal(tower(pixels), pixels.mean(dim=(-2, -1)))


def test_control_loaders_reject_resolved_dataset_and_model_revision_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SiglipProxyControlConfig()
    monkeypatch.setitem(_MODULE._HF_DATASET_REVISIONS, config.dataset_name, "0" * 40)
    with pytest.raises(RuntimeError, match="dataset revision"):
        _MODULE.load_control_examples(loader=lambda **_kwargs: _control_examples())

    class DriftedVision(torch.nn.Module):
        config = SimpleNamespace(_attn_implementation="eager", _commit_hash="0" * 40)

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> DriftedVision:
            return cls()

        def gradient_checkpointing_enable(self, **_kwargs: object) -> None:
            return None

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> FakeProcessor:
            return cls()

    with pytest.raises(ValueError, match="resolved model revision"):
        _MODULE.load_siglip_control_components(
            config=config,
            vision_model_cls=DriftedVision,
            processor_cls=FakeProcessor,
        )


def test_determinism_boundary_sets_and_verifies_every_torch_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    _MODULE.require_control_determinism(torch.device("cpu"))

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False


def test_control_preprocessing_is_the_frozen_train_and_processor_eval_path() -> None:
    from torchvision.transforms import InterpolationMode

    train = _MODULE.build_control_train_transform()
    names = [transform.__class__.__name__ for transform in train.transforms]
    assert names == ["RandomResizedCrop", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    crop, flip, _, normalize = train.transforms
    assert crop.size == (384, 384)
    assert crop.scale == (0.16, 1.0)
    assert crop.interpolation is InterpolationMode.BICUBIC
    assert flip.p == 0.5
    assert tuple(normalize.mean) == (0.5, 0.5, 0.5)
    assert tuple(normalize.std) == (0.5, 0.5, 0.5)

    calls: list[tuple[list[object], str]] = []

    class FakeProcessor:
        def __call__(self, *, images: list[object], return_tensors: str) -> dict[str, object]:
            calls.append((images, return_tensors))
            return {"pixel_values": torch.ones(len(images), 3, 384, 384)}

    images = [object(), object()]
    pixels = _MODULE.preprocess_control_evaluation(FakeProcessor(), images)
    assert calls == [(images, "pt")]
    assert pixels.shape == (2, 3, 384, 384)
    assert pixels.dtype == torch.float32


def test_control_evaluation_accepts_the_processor_mapping_contract() -> None:
    class BatchFeatureLike(UserDict[str, object]):
        pass

    class FakeProcessor:
        def __call__(self, *, images: list[object], return_tensors: str) -> object:
            assert return_tensors == "pt"
            return BatchFeatureLike({"pixel_values": torch.ones(len(images), 3, 384, 384)})

    pixels = _MODULE.preprocess_control_evaluation(FakeProcessor(), [object()])
    assert pixels.shape == (1, 3, 384, 384)


def test_batch_materialization_augments_each_selected_image_exactly_once() -> None:
    examples = tuple(
        ImageExample(example_id=f"id-{index}", image=index, label=index % 2) for index in range(4)
    )
    calls: list[int] = []

    def transform(image: object) -> torch.Tensor:
        assert type(image) is int
        calls.append(image)
        return torch.full((3, 2, 2), float(image))

    pixels, labels = _MODULE.materialize_control_training_batch(
        examples=examples,
        positions=(3, 1, 2),
        transform=transform,
        materialize=lambda image: image,
    )

    assert calls == [3, 1, 2]
    assert pixels.shape == (3, 3, 2, 2)
    assert pixels[:, 0, 0, 0].tolist() == [3.0, 1.0, 2.0]
    assert labels.tolist() == [1, 1, 0]


def test_control_embedding_reports_raw_and_projected_normalized_fp32() -> None:
    class TinyTower(torch.nn.Module):
        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return pixels[:, :2, 0, 0]

    model = PooledProxyAnchorModel(
        tower=TinyTower(),
        input_dimensions=2,
        embedding_dimensions=2,
        class_count=2,
    )
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 0.5]]))
    examples = tuple(
        ImageExample(example_id=f"id-{index}", image=index, label=index // 2) for index in range(4)
    )

    class FakeProcessor:
        def __call__(self, *, images: list[object], return_tensors: str) -> dict[str, object]:
            assert return_tensors == "pt"
            pixels = torch.zeros(len(images), 3, 384, 384)
            for row, image in enumerate(images):
                assert type(image) is int
                pixels[row, 0] = float(image + 1)
                pixels[row, 1] = 1.0
            return {"pixel_values": pixels}

    raw, projected, labels = _MODULE.embed_control_examples(
        model=model,
        examples=examples,
        processor=FakeProcessor(),
        device=torch.device("cpu"),
        batch_size=3,
        materialize=lambda image: image,
    )

    assert raw.dtype == projected.dtype == torch.float32
    assert raw.shape == projected.shape == (4, 2)
    torch.testing.assert_close(torch.linalg.vector_norm(raw, dim=1), torch.ones(4))
    torch.testing.assert_close(torch.linalg.vector_norm(projected, dim=1), torch.ones(4))
    assert labels.tolist() == [0, 0, 1, 1]
    assert model.training is False
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_control_epoch_uses_exact_sampler_replay_schedule_and_one_step() -> None:
    class FlatTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(5, 4)

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.linear(pixels.flatten(1)))

    examples = tuple(
        ImageExample(
            example_id=f"cars-{label:02d}-{position:02d}",
            image=torch.tensor(
                [[[label / 49.0, position / 4.0, 0.25, 0.5, 1.0]]],
                dtype=torch.float32,
            ),
            label=label,
        )
        for label in range(49)
        for position in range(4)
    )
    torch.manual_seed(7)
    model = PooledProxyAnchorModel(
        tower=FlatTower(),
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=49,
    )
    config = SiglipProxyControlConfig()
    optimizer = torch.optim.AdamW(_MODULE._optimizer_groups(model, config))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    evidence = _MODULE.train_control_epoch(
        model=model,
        optimizer=optimizer,
        examples=examples,
        transform=lambda image: cast(torch.Tensor, image).clone(),
        seed=17,
        epoch=0,
        steps_per_epoch=1,
        sampler_state=_MODULE.SamplerState.initial(),
        microbatch_size=30,
        config=config,
        device=torch.device("cpu"),
        materialize=lambda image: image,
    )

    assert evidence.optimizer_steps == 1
    assert len(evidence.losses) == 1
    assert evidence.maximum_score_disagreement <= config.replay_score_tolerance
    assert evidence.sampler_state != _MODULE.SamplerState.initial()
    assert all(math.isfinite(loss) for loss in evidence.losses)
    assert any(
        not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
    )
    expected_multiplier = 1.0 / config.warmup_epochs
    assert {float(group["lr"]) for group in optimizer.param_groups} == {
        config.tower_learning_rate * expected_multiplier,
        config.projection_learning_rate * expected_multiplier,
        config.proxy_learning_rate * expected_multiplier,
    }


def test_control_checkpoint_restores_model_optimizer_sampler_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = PooledProxyAnchorModel(
        tower=_TinyTower(),
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=49,
    )
    config = SiglipProxyControlConfig()
    optimizer = torch.optim.AdamW(_MODULE._optimizer_groups(model, config))
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_model = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_rng = torch.random.get_rng_state().clone()
    sampler = _MODULE.SamplerState(
        cycles=tuple(index % 3 for index in range(49)),
        positions=tuple(index % 4 for index in range(49)),
    )
    path = tmp_path / "state.pt"
    run_authority = _run_authority()
    _MODULE.write_control_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=config,
        seed=17,
        completed_epoch=4,
        sampler_state=sampler,
        final_objective=0.25,
        maximum_score_disagreement=1.0e-6,
        run_authority=run_authority,
        initial_snapshot_sha256="4" * 64,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    torch.manual_seed(999)
    restored = _MODULE.restore_control_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        config=config,
        expected_seed=17,
        expected_run_authority=run_authority,
    )

    assert restored.completed_epoch == 4
    assert restored.final_objective == 0.25
    assert restored.maximum_score_disagreement == 1.0e-6
    assert restored.initial_snapshot_sha256 == "4" * 64
    assert restored.sampler_state == sampler
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected_model[name])

    with pytest.raises(ValueError, match="run authority"):
        _MODULE.restore_control_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            config=config,
            expected_seed=17,
            expected_run_authority=replace(run_authority, manifest_sha256="5" * 64),
        )


def test_control_seed_lifecycle_evaluates_only_initial_and_final_and_checkpoints_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bands = _MODULE.load_control_examples(loader=lambda **_kwargs: _control_examples())
    events: list[tuple[str, int]] = []

    class FakeTower(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            return torch.ones(pixels.shape[0], 1152) * self.scale

    monkeypatch.setattr(
        _MODULE,
        "load_siglip_control_components",
        lambda **_kwargs: (FakeTower(), object()),
    )

    def fake_embed(*, examples: tuple[ImageExample, ...], **_kwargs: object) -> Any:
        events.append(("embed", min(example.label for example in examples)))
        labels = torch.tensor([example.label for example in examples])
        angles = labels.float() * 0.03125
        embeddings = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
        return embeddings, embeddings, labels

    def fake_train(*, epoch: int, sampler_state: Any, **_kwargs: object) -> Any:
        events.append(("train", epoch))
        return _MODULE.EpochTrainingEvidence(
            epoch=epoch,
            optimizer_steps=1,
            losses=(1.0 / (epoch + 1),),
            maximum_score_disagreement=0.0,
            sampler_state=sampler_state,
        )

    def fake_publish(*, seed: int, epoch: int, directory: Path, **_kwargs: object) -> Any:
        events.append(("checkpoint", epoch))
        return _MODULE.CheckpointAuthority(
            seed=seed,
            epoch=epoch,
            path=directory / f"epoch-{epoch}.pt",
            receipt_path=directory / f"epoch-{epoch}.json",
            sha256="0" * 64,
            bytes=1,
        )

    monkeypatch.setattr(_MODULE, "embed_control_examples", fake_embed)
    monkeypatch.setattr(_MODULE, "train_control_epoch", fake_train)
    monkeypatch.setattr(_MODULE, "publish_epoch_checkpoint", fake_publish)
    monkeypatch.setattr(_MODULE, "latest_authenticated_checkpoint", lambda *_args, **_kwargs: None)

    result = _MODULE.run_control_seed(
        config=SiglipProxyControlConfig(),
        seed=17,
        bands=bands,
        checkpoint_directory=tmp_path,
        maximum_checkpoint_bytes=1_000_000,
        microbatch_size=30,
        evaluation_batch_size=64,
        query_block=64,
        device=torch.device("cpu"),
        smoke_receipt=_MODULE.SmokeReceipt(
            observations=(_passing_smoke_observation(30),),
            selected_microbatch_size=30,
            projected_seed_seconds=1.0,
        ),
        run_authority=replace(
            _run_authority(),
            manifest_sha256=_MODULE.control_manifest_sha256(bands.ordered_manifest),
        ),
    )

    assert events[:3] == [("embed", 0), ("embed", 49), ("embed", 82)]
    assert events[-3:] == [("embed", 0), ("embed", 49), ("embed", 82)]
    assert [value for kind, value in events if kind == "train"] == list(range(60))
    assert [value for kind, value in events if kind == "checkpoint"] == list(range(1, 61))
    assert sum(kind == "embed" for kind, _ in events) == 6
    assert result.optimizer_steps == 60
    assert result.final_checkpoint.epoch == 60
    assert result.seed_evidence.seed == 17
