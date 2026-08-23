from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_unicom_full_width_objective.py"


def _load_script():
    if not SCRIPT.is_file():
        pytest.fail("full-width objective evaluator is absent")
    spec = importlib.util.spec_from_file_location("evaluate_unicom_full_width_objective", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_registered_experiment_axes_are_exact() -> None:
    module = _load_script()

    assert module.ARMS == ("sampled_512", "full_768")
    assert module.EPOCHS == (4, 8, 12, 16)
    assert module.SELECTION_SEEDS == (0,)
    assert module.CONFIRMATION_SEEDS == (2, 3, 4, 5, 6)
    assert tuple(range(768)) == module.PRIMARY_COORDINATES
    assert tuple(range(512)) == module.LEGACY_COORDINATES


def test_seed0_quality_miss_is_reported_but_cannot_close_confirmation() -> None:
    module = _load_script()

    decision = module.selection_decision(
        primary_map_delta=-0.001,
        control_top1_count=758,
        candidate_top1_count=756,
        candidate_primary_by_epoch={4: 0.80, 8: 0.84, 12: 0.89, 16: 0.90},
        control_epoch16_primary=0.91,
        abba_step_time_ratio=1.01,
        peak_allocated_ratio=1.0,
        peak_reserved_ratio=1.01,
        control_checkpoint_bytes=100,
        candidate_checkpoint_bytes=100,
    )

    assert decision["prediction_matched"] is False
    assert decision["operational_passed"] is True
    assert decision["decision"] == "PROMOTE_CONFIRMATION"


def test_seed0_resource_failure_closes_current_implementation() -> None:
    module = _load_script()

    decision = module.selection_decision(
        primary_map_delta=0.01,
        control_top1_count=758,
        candidate_top1_count=758,
        candidate_primary_by_epoch={4: 0.80, 8: 0.90, 12: 0.92, 16: 0.93},
        control_epoch16_primary=0.91,
        abba_step_time_ratio=1.0200000000000002,
        peak_allocated_ratio=1.0,
        peak_reserved_ratio=1.0,
        control_checkpoint_bytes=100,
        candidate_checkpoint_bytes=100,
    )

    assert decision["prediction_matched"] is True
    assert decision["operational_passed"] is False
    assert decision["decision"] == "CLOSE_RESOURCE"


def test_trajectory_prediction_requires_control_endpoint_by_epoch_twelve() -> None:
    module = _load_script()

    assert module.first_epoch_reaching(
        {4: 0.8, 8: 0.89, 12: 0.91, 16: 0.92}, 0.90
    ) == 12
    assert module.first_epoch_reaching(
        {4: 0.8, 8: 0.89, 12: 0.899, 16: 0.92}, 0.90
    ) == 16
    assert module.first_epoch_reaching(
        {4: 0.8, 8: 0.89, 12: 0.899, 16: 0.899}, 0.90
    ) is None


def test_paired_t_interval_uses_exact_five_seed_critical_value() -> None:
    module = _load_script()

    interval = module.paired_t_interval(
        (0.004, 0.005, 0.003, 0.006, 0.0045), critical=2.7764451052
    )

    assert interval == pytest.approx((0.0031117774473999995, 0.0058882225526))


@pytest.mark.parametrize(
    "values",
    [(), (0.1,), (0.1, float("nan")), (0.1, True), [0.1, 0.2]],
)
def test_paired_t_interval_rejects_unregistered_values(values) -> None:
    module = _load_script()

    with pytest.raises((TypeError, ValueError)):
        module.paired_t_interval(values, critical=2.7764451052)


def test_paired_query_bootstrap_uses_one_shared_draw() -> None:
    module = _load_script()
    control = (0.0, 0.2, 0.4, 0.6)
    candidate = (0.1, 0.2, 0.6, 0.7)

    interval = module.paired_query_bootstrap(
        control, candidate, seed=768, samples=10_000
    )

    generator = np.random.Generator(np.random.PCG64(768))
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    indices = generator.integers(0, 4, size=(10_000, 4))
    expected = tuple(float(value) for value in np.percentile(delta[indices].mean(1), (2.5, 97.5)))
    assert interval == expected


def _confirmation_rows(delta: float = 0.004) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "seed": seed,
            "control_epoch16_primary": 0.91,
            "candidate_primary_by_epoch": {
                4: 0.84,
                8: 0.89,
                12: 0.91 + delta,
                16: 0.91 + delta,
            },
            "control_top1_count": 758,
            "candidate_top1_count": 758,
        }
        for seed in (2, 3, 4, 5, 6)
    )


def test_confirmation_decision_requires_quality_speed_and_exact_deployment() -> None:
    module = _load_script()

    result = module.confirmation_decision(
        _confirmation_rows(),
        mean_abba_step_time_ratio=1.01,
        mean_peak_allocated_ratio=1.01,
        mean_peak_reserved_ratio=1.0,
        checkpoint_bytes_equal=True,
        deployed_parameters_equal=True,
        inference_operations_equal=True,
        deployment_storage_equal=True,
    )

    assert result["primary_map_deltas"] == pytest.approx([0.004] * 5)
    assert result["positive_seed_count"] == 5
    assert result["epoch12_reach_count"] == 5
    assert all(result["predicates"].values())
    assert result["decision"] == "SUPPORTED_HOLDOUT"


def test_confirmation_decision_closes_when_mean_effect_misses_floor() -> None:
    module = _load_script()

    result = module.confirmation_decision(
        _confirmation_rows(delta=0.0029),
        mean_abba_step_time_ratio=1.0,
        mean_peak_allocated_ratio=1.0,
        mean_peak_reserved_ratio=1.0,
        checkpoint_bytes_equal=True,
        deployed_parameters_equal=True,
        inference_operations_equal=True,
        deployment_storage_equal=True,
    )

    assert result["predicates"]["mean_primary_map_delta_at_least_0_003"] is False
    assert result["decision"] == "CLOSE_FULL_WIDTH"


def test_pair_embeddings_use_identical_registered_views_and_row_order() -> None:
    module = _load_script()
    query = np.zeros((2, 768), dtype=np.float32)
    gallery = np.zeros((4, 768), dtype=np.float32)
    query[0, 0] = 1.0
    query[1, 1] = 1.0
    gallery[0, 0] = 1.0
    gallery[1, 0] = 0.9
    gallery[1, 2] = 0.1
    gallery[2, 1] = 1.0
    gallery[3, 1] = 0.9
    gallery[3, 2] = 0.1
    query_labels = np.asarray(["a", "b"])
    gallery_labels = np.asarray(["a", "a", "b", "b"])
    query_ids = ("query/a.jpg", "query/b.jpg")
    gallery_ids = ("gallery/a1.jpg", "gallery/a2.jpg", "gallery/b1.jpg", "gallery/b2.jpg")

    result = module.evaluate_pair_embeddings(
        control_query=query,
        control_gallery=gallery,
        candidate_query=query.copy(),
        candidate_gallery=gallery.copy(),
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        query_ids=query_ids,
        gallery_ids=gallery_ids,
    )

    expected_query_ids = hashlib.sha256("\n".join(query_ids).encode()).hexdigest()
    expected_gallery_ids = hashlib.sha256("\n".join(gallery_ids).encode()).hexdigest()
    assert result["query_ids_sha256"] == expected_query_ids
    assert result["gallery_ids_sha256"] == expected_gallery_ids
    assert tuple(result["arms"]) == module.ARMS
    for arm in module.ARMS:
        assert tuple(result["arms"][arm]) == (
            "query_embedding_sha256",
            "gallery_embedding_sha256",
            "primary",
            "legacy",
        )
        assert result["arms"][arm]["primary"]["map_at_r"] == 1.0
        assert result["arms"][arm]["legacy"]["map_at_r"] == 1.0
        assert result["arms"][arm]["primary"]["top1_correct"] == [True, True]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda values: values.__setitem__("candidate_query", values["candidate_query"][:, :512]),
        lambda values: values.__setitem__("query_ids", ("query/a.jpg", "query/a.jpg")),
        lambda values: values.__setitem__("query_labels", np.asarray(["a", "z"])),
    ],
)
def test_pair_embeddings_reject_shape_order_or_label_drift(mutation) -> None:
    module = _load_script()
    values = {
        "control_query": np.ones((2, 768), dtype=np.float32),
        "control_gallery": np.ones((4, 768), dtype=np.float32),
        "candidate_query": np.ones((2, 768), dtype=np.float32),
        "candidate_gallery": np.ones((4, 768), dtype=np.float32),
        "query_labels": np.asarray(["a", "b"]),
        "gallery_labels": np.asarray(["a", "a", "b", "b"]),
        "query_ids": ("query/a.jpg", "query/b.jpg"),
        "gallery_ids": (
            "gallery/a1.jpg",
            "gallery/a2.jpg",
            "gallery/b1.jpg",
            "gallery/b2.jpg",
        ),
    }
    mutation(values)

    with pytest.raises(ValueError):
        module.evaluate_pair_embeddings(**values)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    module = _load_script()
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n')
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n')

    with pytest.raises(ValueError):
        module.strict_json_object(duplicate)
    with pytest.raises(ValueError):
        module.strict_json_object(nonfinite)


def test_publish_result_is_exclusive_mode_0600_and_strict_reloaded(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "result.json"
    observed: list[dict[str, object]] = []

    module.publish_result(
        {"schema_version": "test-v1", "passed": True},
        output,
        validate=lambda value: observed.append(value.copy()),
    )

    assert observed == [{"schema_version": "test-v1", "passed": True}]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert output.read_bytes() == b'{"schema_version":"test-v1","passed":true}\n'
    assert not list(tmp_path.glob(".*.tmp"))
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.publish_result(
            {"schema_version": "test-v1", "passed": False},
            output,
            validate=lambda _value: None,
        )
    assert output.read_bytes() == before


def test_publish_result_rolls_back_when_reloaded_validation_fails(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "failed.json"

    with pytest.raises(ValueError, match="injected validation failure"):
        module.publish_result(
            json.loads('{"schema_version":"test-v1","passed":false}'),
            output,
            validate=lambda _value: (_ for _ in ()).throw(
                ValueError("injected validation failure")
            ),
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_arm_protocols_bind_training_and_evaluation_widths_exactly() -> None:
    module = _load_script()
    sampled = {
        "objective": "official-eight-mask",
        "selected_features": 512,
        "evaluation_features": 768,
    }
    full = {
        "objective": "official-eight-mask",
        "selected_features": 768,
        "evaluation_features": 768,
    }

    module.validate_arm_protocol(sampled, "sampled_512")
    module.validate_arm_protocol(full, "full_768")
    for arm, protocol in (("sampled_512", sampled), ("full_768", full)):
        for key, replacement in (
            ("objective", "prefix-512"),
            ("selected_features", 256),
            ("evaluation_features", 512),
        ):
            mutated = protocol.copy()
            mutated[key] = replacement
            with pytest.raises(ValueError):
                module.validate_arm_protocol(mutated, arm)


def test_historical_protocol_without_evaluation_width_is_rejected() -> None:
    module = _load_script()

    with pytest.raises(ValueError):
        module.validate_arm_protocol(
            {"objective": "official-eight-mask", "selected_features": 512},
            "sampled_512",
        )
