from __future__ import annotations

import gc
import math
import random
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora import unicom_probe as probe_module
from sfora.unicom_inshop import InshopRecord
from sfora.unicom_probe import (
    PROBE_LEARNING_RATE,
    ProbeFit,
    ProbeGradientMetrics,
    ProbeMetrics,
    ProbeSeedStatistics,
    ProbeSplit,
    class_mean_head,
    compare_probe_gradients,
    evaluate_probe_heads,
    fit_spherical_probe,
    fit_spherical_probe_trajectory,
    probe_decision,
    split_probe_records,
)
from sfora.unicom_training import (
    experiment_stream_seed,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_logits,
    sharded_mask_arcface_loss,
)


def _record(label: str, name: str) -> InshopRecord:
    return InshopRecord(split="train", image_path=Path("/dataset") / name, label=label)


def test_split_probe_records_uses_seeded_validation_and_keeps_singletons() -> None:
    records = (
        _record("c", "03_1_front.jpg"),
        _record("a", "02_3_side.jpg"),
        _record("a", "01_1_front.jpg"),
        _record("b", "07_1_front.jpg"),
        _record("c", "03_2_side.jpg"),
        _record("a", "01_2_back.jpg"),
    )

    actual = split_probe_records(records, {"a": 0, "b": 1, "c": 2})

    assert type(actual) is ProbeSplit
    assert actual.fitting == (
        _record("a", "02_3_side.jpg"),
        _record("b", "07_1_front.jpg"),
        _record("c", "03_1_front.jpg"),
    )
    assert actual.validation == (
        _record("a", "01_1_front.jpg"),
        _record("c", "03_2_side.jpg"),
    )
    assert actual.validation_group_represented == (False, True)
    assert actual.validation_class_count == 2
    assert actual.singleton_class_count == 1
    assert actual.excluded_same_series_count == 1


def test_split_probe_records_reports_unrepresented_acquisition_series() -> None:
    actual = split_probe_records(
        (
            _record("a", "01_1_front.jpg"),
            _record("a", "02_1_front.jpg"),
            _record("a", "03_1_front.jpg"),
        ),
        {"a": 0},
    )

    assert actual.validation == (_record("a", "01_1_front.jpg"),)
    assert actual.validation_group_represented == (False,)
    assert actual.excluded_same_series_count == 0


@pytest.mark.parametrize(
    ("records", "labels"),
    (
        ((_record("a", "a-1.jpg"), _record("a", "a-2.jpg")), {"a": 1}),
        ((_record("a", "a-1.jpg"), _record("a", "a-2.jpg")), {"a": 0, "b": 1}),
        ((_record("a", "a-1.jpg"), InshopRecord("query", Path("/dataset/a-2.jpg"), "a")), {"a": 0}),
    ),
)
def test_split_probe_records_rejects_invalid_class_inventory(
    records: tuple[InshopRecord, ...], labels: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        split_probe_records(records, labels)


def test_class_mean_head_matches_fp64_normalized_class_means() -> None:
    features = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)

    actual = class_mean_head(features, labels, class_count=2)

    normalized = torch.nn.functional.normalize(features, dim=1).double()
    expected = torch.stack((normalized[:2].mean(dim=0), normalized[2:].mean(dim=0)))
    expected = torch.nn.functional.normalize(expected, dim=1).float()
    expected *= 0.01 * (4.0**0.5)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"


@pytest.mark.parametrize(
    ("features", "labels", "class_count"),
    (
        (torch.ones(2, 4, dtype=torch.float64), torch.tensor([0, 1]), 2),
        (torch.ones(2, 4), torch.tensor([0, 1], dtype=torch.int32), 2),
        (torch.ones(2, 4), torch.tensor([0, 2]), 2),
        (torch.ones(2, 4), torch.tensor([0, 0]), 2),
        (torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.tensor([0, 1]), 2),
    ),
)
def test_class_mean_head_rejects_invalid_features_or_labels(
    features: torch.Tensor, labels: torch.Tensor, class_count: int
) -> None:
    with pytest.raises((TypeError, ValueError)):
        class_mean_head(features, labels, class_count)


def _separable_probe_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    class_count = 8
    dimension = 768
    labels = torch.arange(class_count, dtype=torch.int64).repeat_interleave(4)
    features = torch.zeros(labels.numel(), dimension, dtype=torch.float32)
    features[torch.arange(labels.numel()), labels] = 1.0
    features[:, class_count] = torch.linspace(0.01, 0.04, labels.numel())
    features = features.contiguous()
    initial = torch.roll(torch.eye(class_count, dimension, dtype=torch.float32), shifts=1, dims=0)
    initial = torch.nn.functional.normalize(initial, dim=1) * (0.01 * dimension**0.5)
    return features, labels, initial


def test_fit_spherical_probe_decreases_loss_and_preserves_exact_row_norm() -> None:
    features, labels, initial = _separable_probe_fixture()
    original_features = features.clone()
    original_initial = initial.clone()

    result = fit_spherical_probe(
        features,
        labels,
        initial,
        steps=64,
        batch_size=8,
    )

    assert result.steps == 64
    assert PROBE_LEARNING_RATE == 1e-4
    assert result.final_loss < result.initial_loss
    assert not torch.equal(result.head, initial)
    expected_norm = 0.01 * features.shape[1] ** 0.5
    torch.testing.assert_close(
        torch.linalg.vector_norm(result.head, dim=1),
        torch.full((8,), expected_norm),
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.equal(features, original_features)
    assert torch.equal(initial, original_initial)
    cosine = torch.nn.functional.cosine_similarity(result.head, initial, dim=1)
    assert result.row_cosine_min == float(cosine.min())
    assert result.row_cosine_p05 == float(torch.quantile(cosine.double(), 0.05))
    assert result.row_cosine_median == float(torch.quantile(cosine.double(), 0.5))
    assert result.row_cosine_mean == float(cosine.double().mean())


def test_fit_spherical_probe_uses_independent_diagnostic_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_probe as probe_module

    features, labels, initial = _separable_probe_fixture()
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = probe_module.sharded_mask_arcface_loss

    def capture(
        batch_features: torch.Tensor,
        head: torch.Tensor,
        batch_labels: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((batch_features.detach().clone(), masks.detach().clone()))
        return original(batch_features, head, batch_labels, masks)

    monkeypatch.setattr(probe_module, "sharded_mask_arcface_loss", capture)
    fit_spherical_probe(features, labels, initial, steps=1, batch_size=8)

    assert len(calls) == 3
    assert not torch.equal(calls[0][0], calls[1][0])
    assert not torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[0][0], calls[2][0])
    assert torch.equal(calls[0][1], calls[2][1])


def test_fit_spherical_probe_is_byte_deterministic_for_registered_streams() -> None:
    features, labels, initial = _separable_probe_fixture()

    first = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)
    second = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)

    assert torch.equal(first.head, second.head)
    assert first.initial_loss == second.initial_loss
    assert first.final_loss == second.final_loss
    assert first.steps == second.steps


def test_probe_trajectory_reuses_one_optimizer_and_matches_legacy_final() -> None:
    features, labels, initial = _separable_probe_fixture()

    fit, snapshots = fit_spherical_probe_trajectory(
        features,
        labels,
        initial,
        steps=8,
        snapshot_steps=(0, 1, 2, 4, 8),
        batch_size=8,
    )
    legacy = fit_spherical_probe(features, labels, initial, steps=8, batch_size=8)

    assert tuple(snapshots) == (0, 1, 2, 4, 8)
    assert torch.equal(snapshots[0], initial)
    assert all(snapshot.is_contiguous() for snapshot in snapshots.values())
    assert all(not snapshot.requires_grad for snapshot in snapshots.values())
    assert len({snapshot.data_ptr() for snapshot in snapshots.values()}) == 5
    assert snapshots[8] is fit.head
    assert torch.equal(fit.head, legacy.head)


def test_injected_adamw_trajectory_is_byte_identical_to_public_path_and_rng_neutral() -> None:
    features, labels, initial = _separable_probe_fixture()
    snapshot_steps = (0, 1, 2, 4, 8)

    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    legacy_fit, legacy_snapshots = probe_module._fit_spherical_probe(
        features,
        labels,
        initial,
        snapshot_steps=snapshot_steps,
        steps=8,
        batch_size=8,
    )
    assert random.getstate() == python_before
    assert np.random.get_state()[0] == numpy_before[0]
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)

    factory_heads: list[torch.nn.Parameter] = []

    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        factory_heads.append(head)
        return torch.optim.AdamW(
            [head],
            lr=PROBE_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()
    injected_fit, injected_snapshots, traces = (
        probe_module._fit_probe_trajectory_with_optimizer(
            features,
            labels,
            initial,
            snapshot_steps=snapshot_steps,
            optimizer_factory=optimizer_factory,
            trace_factory=None,
            steps=8,
            batch_size=8,
        )
    )

    assert len(factory_heads) == 1
    assert traces == {0: None, 1: None, 2: None, 4: None, 8: None}
    assert torch.equal(injected_fit.head, legacy_fit.head)
    assert (
        injected_fit.initial_loss,
        injected_fit.final_loss,
        injected_fit.steps,
        injected_fit.row_cosine_min,
        injected_fit.row_cosine_p05,
        injected_fit.row_cosine_median,
        injected_fit.row_cosine_mean,
    ) == (
        legacy_fit.initial_loss,
        legacy_fit.final_loss,
        legacy_fit.steps,
        legacy_fit.row_cosine_min,
        legacy_fit.row_cosine_p05,
        legacy_fit.row_cosine_median,
        legacy_fit.row_cosine_mean,
    )
    assert tuple(injected_snapshots) == tuple(legacy_snapshots)
    assert all(
        torch.equal(injected_snapshots[step], legacy_snapshots[step])
        for step in snapshot_steps
    )
    assert random.getstate() == python_before
    assert np.random.get_state()[0] == numpy_before[0]
    assert np.array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_injected_trajectory_reports_each_completed_step_and_stops_on_callback() -> None:
    features, labels, initial = _separable_probe_fixture()
    completed_steps: list[int] = []

    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            [head],
            lr=PROBE_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

    def progress_callback(completed: int) -> None:
        completed_steps.append(completed)
        if completed == 3:
            raise RuntimeError("registered execution budget exceeded")

    with pytest.raises(RuntimeError, match="registered execution budget exceeded"):
        probe_module._fit_probe_trajectory_with_optimizer(
            features,
            labels,
            initial,
            snapshot_steps=(0, 2, 4, 8),
            optimizer_factory=optimizer_factory,
            trace_factory=None,
            progress_callback=progress_callback,
            steps=8,
            batch_size=8,
        )

    assert completed_steps == [1, 2, 3]


def test_real_trajectory_supports_explicit_reduced_feature_width_for_cpu_e2e() -> None:
    generator = torch.Generator().manual_seed(205)
    labels = torch.arange(8, dtype=torch.int64)
    features = torch.nn.functional.normalize(
        torch.randn(8, 16, generator=generator), dim=1
    ).float().contiguous()
    initial = probe_module.class_mean_head(features, labels, class_count=8)

    fit, snapshots, traces = probe_module._fit_probe_trajectory_with_optimizer(
        features,
        labels,
        initial,
        snapshot_steps=(0, 2),
        optimizer_factory=lambda head: torch.optim.AdamW([head], lr=0.0001),
        trace_factory=None,
        steps=2,
        batch_size=8,
        selected_features=16,
    )

    assert fit.steps == 2
    assert tuple(snapshots) == (0, 2)
    assert traces == {0: None, 2: None}


def test_proxy_muon_trajectory_forwards_completed_step_callback() -> None:
    features, labels, initial = _separable_probe_fixture()
    completed_steps: list[int] = []

    def progress_callback(completed: int) -> None:
        completed_steps.append(completed)
        if completed == 2:
            raise RuntimeError("registered execution budget exceeded")

    with pytest.raises(RuntimeError, match="registered execution budget exceeded"):
        probe_module.fit_proxy_muon_trajectory(
            features,
            labels,
            initial,
            learning_rate=1e-4,
            ns_dtype=torch.float32,
            snapshot_steps=(0, 2, 4),
            progress_callback=progress_callback,
            steps=4,
            batch_size=8,
        )

    assert completed_steps == [1, 2]


def test_diagnostic_panel_is_exact_batch_major_mask_minor_cartesian_product() -> None:
    features, labels, head = _separable_probe_fixture()
    seed = experiment_stream_seed(2, 23_004)

    actual = probe_module.diagnostic_panel_losses(
        features,
        labels,
        head,
        fit_seed=2,
        batch_size=8,
    )

    epoch_indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=8,
        epoch=0,
        seed=seed,
        shards=8,
    )
    batches = tuple(
        torch.tensor(epoch_indices[start : start + 8], dtype=torch.int64)
        for start in range(0, 32, 8)
    )
    generator = torch.Generator().manual_seed(seed)
    masks = tuple(
        sample_shard_masks(
            dimension=768,
            selected=512,
            shards=8,
            generator=generator,
            device=torch.device("cpu"),
        )
        for _ in range(4)
    )
    expected = tuple(
        float(
            sharded_mask_arcface_loss(
                features.index_select(0, batch),
                head,
                labels.index_select(0, batch),
                mask,
            )
        )
        for batch in batches
        for mask in masks
    )

    assert actual == expected
    assert len(actual) == 16
    assert actual[0] == probe_module._diagnostic_loss(
        features,
        labels,
        head,
        batch_size=8,
        batch_seed=seed,
        mask_seed=seed,
    )
    assert math.fsum(actual) / 16 == math.fsum(expected) / 16


def test_proxy_muon_trajectory_records_producing_step_trace_and_projects_rows() -> None:
    features, labels, initial = _separable_probe_fixture()

    fit, snapshots, traces = probe_module.fit_proxy_muon_trajectory(
        features,
        labels,
        initial,
        learning_rate=0.0002,
        fit_seed=3,
        snapshot_steps=(0, 1, 2),
        steps=2,
        batch_size=8,
    )

    assert tuple(snapshots) == (0, 1, 2)
    assert tuple(traces) == (0, 1, 2)
    assert traces[0] is None
    assert type(traces[1]) is probe_module.MuonTrace
    assert type(traces[2]) is probe_module.MuonTrace
    assert traces[1].update_dtype == "torch.bfloat16"
    assert traces[2].update_dtype == "torch.bfloat16"
    assert len(traces[1].orthogonal_update_sha256) == 64
    assert len(traces[2].orthogonal_update_sha256) == 64
    assert snapshots[2] is fit.head
    target_norms = torch.linalg.vector_norm(initial, dim=1)
    for snapshot in snapshots.values():
        assert snapshot.device.type == "cpu"
        assert snapshot.is_contiguous()
        torch.testing.assert_close(
            torch.linalg.vector_norm(snapshot, dim=1),
            target_norms,
            rtol=2e-6,
            atol=2e-7,
        )


def test_proxy_muon_float32_sensitivity_records_float32_trace() -> None:
    features, labels, initial = _separable_probe_fixture()

    _fit, _snapshots, traces = probe_module.fit_proxy_muon_trajectory(
        features,
        labels,
        initial,
        learning_rate=0.0002,
        ns_dtype=torch.float32,
        fit_seed=3,
        snapshot_steps=(0, 1),
        steps=1,
        batch_size=8,
    )

    assert traces[0] is None
    assert type(traces[1]) is probe_module.MuonTrace
    assert traces[1].update_dtype == "torch.float32"


def test_injected_trajectory_releases_optimizer_parameter_gradient_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels, initial = _separable_probe_fixture()
    head_refs: list[weakref.ReferenceType[torch.nn.Parameter]] = []
    optimizer_refs: list[weakref.ReferenceType[torch.optim.Optimizer]] = []
    gradient_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    mask_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_masks = probe_module.sample_shard_masks

    def capture_masks(*args, **kwargs):
        masks = original_masks(*args, **kwargs)
        mask_refs.append(weakref.ref(masks))
        return masks

    def optimizer_factory(head: torch.nn.Parameter) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            [head],
            lr=PROBE_LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        head_refs.append(weakref.ref(head))
        optimizer_refs.append(weakref.ref(optimizer))
        return optimizer

    def trace_factory(
        head: torch.nn.Parameter, _optimizer: torch.optim.Optimizer
    ) -> None:
        gradient_refs.append(weakref.ref(head.grad))

    monkeypatch.setattr(probe_module, "sample_shard_masks", capture_masks)
    fit, snapshots, traces = probe_module._fit_probe_trajectory_with_optimizer(
        features,
        labels,
        initial,
        snapshot_steps=(0, 2),
        optimizer_factory=optimizer_factory,
        trace_factory=trace_factory,
        steps=2,
        batch_size=8,
    )

    assert fit.head is snapshots[2]
    assert traces == {0: None, 2: None}
    gc.collect()
    assert all(reference() is None for reference in head_refs)
    assert all(reference() is None for reference in optimizer_refs)
    assert all(reference() is None for reference in gradient_refs)
    assert all(reference() is None for reference in mask_refs)


@pytest.mark.parametrize(
    "mutation", ("feature_dtype", "label_dtype", "zero_row", "nan", "empty_class")
)
def test_fit_spherical_probe_rejects_invalid_tensor_contract(mutation: str) -> None:
    features, labels, initial = _separable_probe_fixture()
    if mutation == "feature_dtype":
        features = features.double()
    elif mutation == "label_dtype":
        labels = labels.int()
    elif mutation == "zero_row":
        initial[0].zero_()
    elif mutation == "nan":
        features[0, 0] = float("nan")
    elif mutation == "empty_class":
        labels[labels == 7] = 6

    with pytest.raises((TypeError, ValueError)):
        fit_spherical_probe(features, labels, initial, steps=2, batch_size=8)


def test_evaluate_probe_heads_uses_margin_free_accuracy_and_retains_paired_losses() -> None:
    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()
    represented = tuple(index % 2 == 0 for index in range(labels.numel()))

    actual = evaluate_probe_heads(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        validation_group_represented=represented,
        mask_sets=2,
    )

    assert tuple(actual) == ("class_mean", "spherical_probe")
    assert all(type(value) is ProbeMetrics for value in actual.values())
    expected: dict[str, tuple[tuple[float, ...], int, float, float]] = {}
    generator = torch.Generator().manual_seed(23_003)
    per_mask = {"class_mean": [], "spherical_probe": []}
    correct = {"class_mean": 0, "spherical_probe": 0}
    represented_loss = {"class_mean": 0.0, "spherical_probe": 0.0}
    unrepresented_loss = {"class_mean": 0.0, "spherical_probe": 0.0}
    represented_tensor = torch.tensor(represented, dtype=torch.bool)
    for _index in range(2):
        masks = sample_shard_masks(
            dimension=768,
            selected=512,
            shards=8,
            generator=generator,
            device=torch.device("cpu"),
        )
        for name, head in {"class_mean": initial, "spherical_probe": probe}.items():
            logits = sharded_mask_arcface_logits(features, head, labels, masks)
            losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            per_mask[name].append(float(losses.double().mean()))
            represented_loss[name] += float(losses[represented_tensor].double().sum())
            unrepresented_loss[name] += float(losses[~represented_tensor].double().sum())
            margin_free = sharded_mask_arcface_logits(
                features, head, labels, masks, margin=0.0
            )
            correct[name] += int(torch.count_nonzero(margin_free.argmax(dim=1) == labels))
    observations = labels.numel() * 2
    represented_observations = int(represented_tensor.sum()) * 2
    unrepresented_observations = int((~represented_tensor).sum()) * 2
    for name in per_mask:
        expected[name] = (
            tuple(per_mask[name]),
            correct[name],
            represented_loss[name] / represented_observations,
            unrepresented_loss[name] / unrepresented_observations,
        )

    for name in actual:
        assert actual[name].per_mask_mean_losses == expected[name][0]
        assert actual[name].mean_loss == sum(expected[name][0]) / 2
        assert actual[name].correct_count == expected[name][1]
        assert actual[name].observation_count == observations
        assert actual[name].accuracy == expected[name][1] / observations
        assert actual[name].represented_mean_loss == expected[name][2]
        assert actual[name].unrepresented_mean_loss == expected[name][3]
        assert actual[name].represented_mean_loss == (
            math.fsum(actual[name].per_mask_represented_mean_losses) / 2
        )
        assert actual[name].unrepresented_mean_loss == (
            math.fsum(actual[name].per_mask_unrepresented_mean_losses) / 2
        )


def test_evaluate_probe_head_matches_candidate_metric_without_baseline_replay() -> None:
    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()
    represented = tuple(index % 2 == 0 for index in range(labels.numel()))
    paired = evaluate_probe_heads(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        validation_group_represented=represented,
        mask_sets=2,
    )

    candidate = probe_module.evaluate_probe_head(
        features,
        labels,
        probe,
        validation_group_represented=represented,
        mask_sets=2,
    )

    assert candidate == paired["spherical_probe"]


def test_evaluate_probe_heads_allows_singleton_classes_absent_from_validation() -> None:
    features, labels, initial = _separable_probe_fixture()
    keep = labels != 7
    validation_features = features[keep].contiguous()
    validation_labels = labels[keep].contiguous()
    represented = tuple(index % 2 == 0 for index in range(validation_labels.numel()))

    actual = evaluate_probe_heads(
        validation_features,
        validation_labels,
        {"class_mean": initial, "spherical_probe": initial.clone()},
        validation_group_represented=represented,
        mask_sets=2,
    )

    assert actual["class_mean"].observation_count == validation_labels.numel() * 2


def test_evaluate_probe_heads_uses_validator_exact_stratum_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_probe as probe_module

    labels = torch.arange(12, dtype=torch.int64) % 8
    features = torch.ones(12, 768, dtype=torch.float32)
    heads = {
        "class_mean": torch.ones(8, 768, dtype=torch.float32),
        "spherical_probe": torch.ones(8, 768, dtype=torch.float32),
    }
    represented = tuple(index < 7 for index in range(12))
    generator = torch.Generator().manual_seed(65)

    monkeypatch.setattr(
        probe_module,
        "sharded_mask_arcface_logits",
        lambda *_args, **_kwargs: torch.zeros(12, 8, dtype=torch.float32),
    )
    monkeypatch.setattr(
        probe_module.F,
        "cross_entropy",
        lambda *_args, **_kwargs: torch.rand(
            12, generator=generator, dtype=torch.float32
        )
        * 100.0,
    )
    monkeypatch.setattr(probe_module, "PROBE_VALIDATION_IMAGES", 12)

    actual = evaluate_probe_heads(
        features,
        labels,
        heads,
        validation_group_represented=represented,
        mask_sets=64,
    )

    for name, metrics in actual.items():
        probe_module._validate_metrics(metrics, name)


def test_fit_spherical_probe_clamps_fp32_cosine_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_probe as probe_module

    features, labels, initial = _separable_probe_fixture()
    monkeypatch.setattr(
        probe_module.F,
        "cosine_similarity",
        lambda *_args, **_kwargs: torch.full((8,), 1.0000004, dtype=torch.float32),
    )

    result = fit_spherical_probe(features, labels, initial, steps=1, batch_size=8)

    assert result.row_cosine_min == 1.0
    assert result.row_cosine_p05 == 1.0
    assert result.row_cosine_median == 1.0
    assert result.row_cosine_mean == 1.0


def test_compare_probe_gradients_clamps_fp32_cosine_before_persisting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.unicom_probe as probe_module

    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()
    monkeypatch.setattr(
        probe_module.F,
        "cosine_similarity",
        lambda left, *_args, **_kwargs: torch.full(
            (left.shape[0],), 1.0000004, dtype=torch.float32
        ),
    )

    result = compare_probe_gradients(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        fit_seed=2,
        batch_size=8,
    )

    assert result.cosine_min == 1.0
    assert result.cosine_p05 == 1.0
    assert result.cosine_median == 1.0
    assert result.cosine_mean == 1.0


def test_compare_probe_gradients_matches_independent_autograd_oracle() -> None:
    from sfora.unicom_training import experiment_stream_seed, padded_epoch_indices

    features, labels, initial = _separable_probe_fixture()
    probe = torch.roll(initial, shifts=-1, dims=0).contiguous()

    actual = compare_probe_gradients(
        features,
        labels,
        {"class_mean": initial, "spherical_probe": probe},
        fit_seed=2,
        batch_size=8,
    )

    indices = padded_epoch_indices(
        size=features.shape[0],
        global_batch=8,
        epoch=0,
        seed=experiment_stream_seed(2, 23_005),
        shards=8,
    )[:8]
    batch_labels = labels[list(indices)]
    generator = torch.Generator().manual_seed(experiment_stream_seed(2, 23_005))
    masks = sample_shard_masks(
        dimension=768,
        selected=512,
        shards=8,
        generator=generator,
        device=torch.device("cpu"),
    )
    gradients = []
    for head in (initial, probe):
        batch = features[list(indices)].detach().clone().requires_grad_(True)
        logits = sharded_mask_arcface_logits(batch, head, batch_labels, masks)
        loss = torch.nn.functional.cross_entropy(logits, batch_labels)
        gradients.append(torch.autograd.grad(loss, batch)[0])
    mean_gradient, probe_gradient = gradients
    per_row = torch.nn.functional.cosine_similarity(mean_gradient, probe_gradient, dim=1)

    assert type(actual) is ProbeGradientMetrics
    assert actual.class_mean_l2 == float(torch.linalg.vector_norm(mean_gradient).double())
    assert actual.spherical_probe_l2 == float(torch.linalg.vector_norm(probe_gradient).double())
    assert actual.relative_l2_difference == float(
        torch.linalg.vector_norm(probe_gradient - mean_gradient).double()
        / torch.linalg.vector_norm(mean_gradient).double()
    )
    assert actual.cosine_min == float(per_row.double().min())
    assert actual.cosine_p05 == float(torch.quantile(per_row.double(), 0.05))
    assert actual.cosine_median == float(torch.quantile(per_row.double(), 0.5))
    assert actual.cosine_mean == float(per_row.double().mean())


def _metrics(loss: float, accuracy: float, *, count: int = 3_188) -> ProbeMetrics:
    observations = count * 64
    correct = round(accuracy * observations)
    return ProbeMetrics(
        mean_loss=loss,
        accuracy=correct / observations,
        correct_count=correct,
        observation_count=observations,
        per_mask_mean_losses=tuple(loss for _ in range(64)),
        per_mask_represented_mean_losses=tuple(loss for _ in range(64)),
        per_mask_unrepresented_mean_losses=tuple(loss for _ in range(64)),
        per_image_mean_losses=tuple(loss for _ in range(count)),
        represented_mean_loss=loss,
        unrepresented_mean_loss=loss,
    )


def _metrics_from_deltas(deltas: tuple[float, ...]) -> ProbeMetrics:
    losses = tuple(1.0 - delta for delta in deltas)
    mean_loss = math.fsum(losses) / len(losses)
    return ProbeMetrics(
        mean_loss=mean_loss,
        accuracy=round(0.8 * 3_188 * 64) / (3_188 * 64),
        correct_count=round(0.8 * 3_188 * 64),
        observation_count=3_188 * 64,
        per_mask_mean_losses=losses,
        per_mask_represented_mean_losses=losses,
        per_mask_unrepresented_mean_losses=losses,
        per_image_mean_losses=tuple(mean_loss for _ in range(3_188)),
        represented_mean_loss=mean_loss,
        unrepresented_mean_loss=mean_loss,
    )


def _fit() -> ProbeFit:
    return ProbeFit(
        head=torch.ones(8, 768, dtype=torch.float32),
        initial_loss=2.0,
        final_loss=1.9,
        steps=512,
        row_cosine_min=0.8,
        row_cosine_p05=0.9,
        row_cosine_median=0.97,
        row_cosine_mean=0.95,
    )


def _gradient() -> ProbeGradientMetrics:
    return ProbeGradientMetrics(
        class_mean_l2=1.0,
        spherical_probe_l2=1.0,
        relative_l2_difference=0.1,
        spherical_to_class_mean_l2_ratio=1.0,
        cosine_min=0.98,
        cosine_p05=0.985,
        cosine_median=0.99,
        cosine_mean=0.99,
        zero_gradient_row_count=0,
    )


def test_probe_decision_promotes_at_registered_inclusive_boundaries() -> None:
    decision = probe_decision(
        class_mean=_metrics(1.0, 0.8),
        probe_fits={seed: _fit() for seed in range(3)},
        probe_metrics={seed: _metrics(0.9, 0.8) for seed in range(3)},
        probe_gradients={seed: _gradient() for seed in range(3)},
    )

    assert decision.status == "PROMOTE"
    assert tuple(decision.per_seed_predicates) == (0, 1, 2)
    assert all(all(values.values()) for values in decision.per_seed_predicates.values())
    assert all(
        type(value) is ProbeSeedStatistics
        for value in decision.per_seed_statistics.values()
    )
    assert decision.per_seed_statistics[0] == ProbeSeedStatistics(
        mean_paired_loss_delta=pytest.approx(0.1),
        paired_95_lower_bound=pytest.approx(0.1),
        identity_95_lower_bound=pytest.approx(0.1),
        positive_mask_count=64,
        unrepresented_paired_95_lower_bound=pytest.approx(0.1),
        unrepresented_positive_mask_count=64,
        accuracy_delta=0.0,
        represented_loss_delta=pytest.approx(0.1),
        unrepresented_loss_delta=pytest.approx(0.1),
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "fit",
        "loss",
        "paired_ci",
        "majority",
        "accuracy",
        "represented",
        "unrepresented",
        "head_min",
        "head_mean",
        "cosine",
    ),
)
def test_probe_decision_closes_each_failed_scientific_predicate(mutation: str) -> None:
    fits = {seed: _fit() for seed in range(3)}
    metrics = {seed: _metrics(0.9, 0.8) for seed in range(3)}
    gradients = {seed: _gradient() for seed in range(3)}
    arguments = {
        "class_mean": _metrics(1.0, 0.8),
        "probe_fits": fits,
        "probe_metrics": metrics,
        "probe_gradients": gradients,
    }
    if mutation == "fit":
        fits[1] = replace(fits[1], final_loss=fits[1].initial_loss)
    elif mutation == "loss":
        metrics[1] = _metrics(1.0, 0.8)
    elif mutation == "paired_ci":
        metrics[1] = _metrics_from_deltas((0.2,) * 48 + (-0.5,) * 16)
    elif mutation == "majority":
        metrics[1] = _metrics_from_deltas((0.2,) * 47 + (-0.01,) * 17)
    elif mutation == "accuracy":
        metrics[1] = _metrics(0.9, 0.79)
    elif mutation == "represented":
        value = _metrics(0.9, 0.8)
        metrics[1] = replace(
            value,
            represented_mean_loss=1.0,
            per_mask_represented_mean_losses=(1.0,) * 64,
        )
    elif mutation == "unrepresented":
        value = _metrics(0.9, 0.8)
        metrics[1] = replace(
            value,
            unrepresented_mean_loss=1.0,
            per_mask_unrepresented_mean_losses=(1.0,) * 64,
        )
    elif mutation == "head_min":
        fits[1] = replace(_fit(), row_cosine_min=0.799)
    elif mutation == "head_mean":
        fits[1] = replace(_fit(), row_cosine_mean=0.949)
    elif mutation == "cosine":
        gradients[1] = replace(_gradient(), cosine_median=0.996)

    assert probe_decision(**arguments).status == "CLOSE_DIRECTION"


def test_probe_metrics_preserve_mask_strata_and_per_image_loss_distributions() -> None:
    features = torch.zeros(2, 512, dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    labels = torch.tensor([0, 1], dtype=torch.int64)
    class_mean = torch.zeros(8, 512, dtype=torch.float32)
    class_mean[:8, :8] = torch.eye(8, dtype=torch.float32)
    probe = class_mean.clone()
    heads = {
        "class_mean": class_mean,
        "spherical_probe": probe,
    }

    actual = evaluate_probe_heads(
        features,
        labels,
        heads,
        validation_group_represented=(True, False),
        mask_sets=64,
    )

    for metrics in actual.values():
        assert len(metrics.per_mask_represented_mean_losses) == 64
        assert len(metrics.per_mask_unrepresented_mean_losses) == 64
        assert len(metrics.per_image_mean_losses) == 2
        assert metrics.represented_mean_loss == pytest.approx(
            math.fsum(metrics.per_mask_represented_mean_losses) / 64
        )
        assert metrics.unrepresented_mean_loss == pytest.approx(
            math.fsum(metrics.per_mask_unrepresented_mean_losses) / 64
        )


def test_probe_decision_requires_identity_and_unrepresented_uncertainty() -> None:
    class_mean = _metrics(1.0, 0.8)
    fits = {seed: _fit() for seed in range(3)}
    gradients = {seed: _gradient() for seed in range(3)}
    metrics = {seed: _metrics(0.9, 0.8) for seed in range(3)}
    weak = replace(
        metrics[1],
        per_mask_unrepresented_mean_losses=(0.8,) * 47 + (1.01,) * 17,
        unrepresented_mean_loss=(0.8 * 47 + 1.01 * 17) / 64,
        per_image_mean_losses=(0.5,) * 3_187 + (1275.7,),
    )
    metrics[1] = weak

    decision = probe_decision(
        class_mean=class_mean,
        probe_fits=fits,
        probe_metrics=metrics,
        probe_gradients=gradients,
    )

    assert decision.status == "CLOSE_DIRECTION"
    assert decision.per_seed_predicates[1]["unrepresented_positive_mask_majority"] is False
    assert decision.per_seed_predicates[1]["identity_95_lower_positive"] is False


def test_probe_decision_rejects_nonregistered_identity_distribution_size() -> None:
    class_mean = replace(_metrics(1.0, 0.8), per_image_mean_losses=(1.0,) * 100)

    with pytest.raises(ValueError, match="probe metrics differ"):
        probe_decision(
            class_mean=class_mean,
            probe_fits={seed: _fit() for seed in range(3)},
            probe_metrics={seed: _metrics(0.9, 0.8) for seed in range(3)},
            probe_gradients={seed: _gradient() for seed in range(3)},
        )


def test_gradient_metrics_exclude_matched_zero_rows_and_report_norm_ratio() -> None:
    features = torch.zeros(8, 512, dtype=torch.float32)
    features[:, :8] = torch.eye(8, dtype=torch.float32)
    labels = torch.arange(8, dtype=torch.int64)
    head = features.clone()

    actual = compare_probe_gradients(
        features,
        labels,
        {"class_mean": head, "spherical_probe": head.clone()},
        fit_seed=0,
        batch_size=8,
    )

    assert actual.spherical_to_class_mean_l2_ratio == pytest.approx(1.0)
    assert type(actual.zero_gradient_row_count) is int
    assert 0 <= actual.zero_gradient_row_count < 8
