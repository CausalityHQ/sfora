from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import stat
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


def _valid_pair_payload(module):
    inventory = []
    for epoch in module.EPOCHS:
        for arm in module.ARMS:
            digest = f"{len(inventory) + 1:x}" * 64
            inventory.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "path": f"/{arm}/epoch-{epoch:04d}.pt",
                    "sha256": digest,
                    "bytes": 1000 + epoch,
                }
            )
    config = {
        "schema_version": "unicom-full-width-pair-config-v1",
        "seed": 0,
        "inventory": inventory,
    }
    query = np.zeros((2, 768), dtype=np.float32)
    gallery = np.zeros((4, 768), dtype=np.float32)
    query[0, 0] = query[1, 1] = 1.0
    gallery[0, 0] = gallery[1, 0] = 1.0
    gallery[2, 1] = gallery[3, 1] = 1.0

    def load_checkpoint(row: dict[str, object]) -> dict[str, object]:
        selected = 512 if row["arm"] == "sampled_512" else 768
        return {
            "epoch": row["epoch"],
            "model": {"arm": row["arm"], "epoch": row["epoch"]},
            "selection_holdout": {"seed": 0, "fraction": 0.2},
            "training_protocol": {
                "objective": "official-eight-mask",
                "selected_features": selected,
                "evaluation_features": 768,
            },
        }

    def encode(_model_state, _row):
        return {
            "query": query.copy(),
            "gallery": gallery.copy(),
            "query_labels": np.asarray(["a", "b"]),
            "gallery_labels": np.asarray(["a", "a", "b", "b"]),
            "query_ids": ("query/a.jpg", "query/b.jpg"),
            "gallery_ids": (
                "gallery/a1.jpg",
                "gallery/a2.jpg",
                "gallery/b1.jpg",
                "gallery/b2.jpg",
            ),
            "elapsed_seconds": 0.25,
            "peak_allocated_bytes": 123,
        }

    return config, module.evaluate_pair(config, load_checkpoint, encode)


def _walk_schema(value, path=()):
    yield path, value
    if type(value) is dict:
        for key, child in value.items():
            yield from _walk_schema(child, (*path, key))
    elif type(value) is list:
        for index, child in enumerate(value):
            yield from _walk_schema(child, (*path, index))


def _at_path(value, path):
    for part in path:
        value = value[part]
    return value


def _set_path(value, path, replacement):
    _at_path(value, path[:-1])[path[-1]] = replacement


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


@pytest.mark.parametrize(
    "failure_phase",
    ("directory_open", "first_directory_fsync", "temporary_unlink", "final_directory_fsync"),
)
def test_publish_result_preserves_valid_output_after_durable_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    module = _load_script()
    output = tmp_path / "durable.json"
    temporary = output.with_name(f".{output.name}.{module.os.getpid()}.tmp")
    real_open = module.os.open
    real_fsync = module.os.fsync
    real_unlink = module.Path.unlink
    fsync_calls = 0
    directory_open_calls = 0
    unlink_calls = 0

    def fail_directory_open(path, flags, *args):
        nonlocal directory_open_calls
        if Path(path) == output.parent:
            directory_open_calls += 1
            if failure_phase == "directory_open" and directory_open_calls == 1:
                raise OSError("injected directory open failure")
        return real_open(path, flags, *args)

    def fail_directory_fsync(descriptor):
        nonlocal fsync_calls
        fsync_calls += 1
        if failure_phase == "first_directory_fsync" and fsync_calls == 2:
            raise OSError("injected first directory fsync failure")
        if failure_phase == "final_directory_fsync" and fsync_calls == 3:
            raise OSError("injected final directory fsync failure")
        return real_fsync(descriptor)

    def fail_temporary_unlink(path, *args, **kwargs):
        nonlocal unlink_calls
        if Path(path) == temporary:
            unlink_calls += 1
            if failure_phase == "temporary_unlink" and unlink_calls == 1:
                raise OSError("injected temporary unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", fail_directory_open)
    monkeypatch.setattr(module.os, "fsync", fail_directory_fsync)
    monkeypatch.setattr(module.Path, "unlink", fail_temporary_unlink)

    with pytest.raises(OSError, match="injected"):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
        )

    assert output.read_bytes() == b'{"schema_version":"test-v1","passed":true}\n'
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_result_preserves_preexisting_exact_temporary(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "result.json"
    temporary = output.with_name(f".{output.name}.{module.os.getpid()}.tmp")
    temporary.write_bytes(b"foreign temporary\n")

    with pytest.raises(FileExistsError):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
        )

    assert not output.exists()
    assert temporary.read_bytes() == b"foreign temporary\n"


def test_publish_result_link_race_preserves_foreign_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "result.json"

    def lose_link_race(_source, destination):
        Path(destination).write_bytes(b"foreign destination\n")
        raise FileExistsError(destination)

    monkeypatch.setattr(module.os, "link", lose_link_race)

    with pytest.raises(FileExistsError):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
        )

    assert output.read_bytes() == b"foreign destination\n"
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_result_prelink_fsync_failure_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "result.json"

    def fail_file_fsync(_descriptor):
        raise OSError("injected file fsync failure")

    monkeypatch.setattr(module.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="injected file fsync failure"):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_result_open_failure_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "result.json"

    def fail_open(_path, _flags, _mode):
        raise OSError("injected temporary open failure")

    monkeypatch.setattr(module.os, "open", fail_open)

    with pytest.raises(OSError, match="injected temporary open failure"):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_result_write_failure_leaves_no_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    output = tmp_path / "result.json"

    class FailingWriter:
        def __init__(self, descriptor: int) -> None:
            self.descriptor = descriptor

        def __enter__(self):
            return self

        def __exit__(self, _error_type, _error, _traceback) -> None:
            module.os.close(self.descriptor)

        def write(self, _value):
            raise OSError("injected temporary write failure")

    monkeypatch.setattr(
        module.os,
        "fdopen",
        lambda descriptor, _mode, *, closefd: FailingWriter(descriptor),
    )

    with pytest.raises(OSError, match="injected temporary write failure"):
        module.publish_result(
            {"schema_version": "test-v1", "passed": True},
            output,
            validate=lambda _value: None,
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


def test_evaluate_pair_loads_raw_checkpoints_in_frozen_order_without_history_metrics() -> None:
    module = _load_script()
    inventory = []
    for epoch in module.EPOCHS:
        for arm in module.ARMS:
            digest = f"{len(inventory) + 1:x}" * 64
            inventory.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "path": f"/{arm}/epoch-{epoch:04d}.pt",
                    "sha256": digest,
                    "bytes": 1000 + epoch,
                }
            )
    config = {
        "schema_version": "unicom-full-width-pair-config-v1",
        "seed": 0,
        "inventory": inventory,
    }
    calls: list[tuple[str, str, int]] = []
    poison_history = object()

    def load_checkpoint(row: dict[str, object]) -> dict[str, object]:
        calls.append(("load", row["arm"], row["epoch"]))
        selected = 512 if row["arm"] == "sampled_512" else 768
        return {
            "epoch": row["epoch"],
            "model": {"arm": row["arm"], "epoch": row["epoch"]},
            "selection_holdout": {"seed": 0, "fraction": 0.2},
            "training_protocol": {
                "objective": "official-eight-mask",
                "selected_features": selected,
                "evaluation_features": 768,
            },
            "history": poison_history,
        }

    query = np.zeros((2, 768), dtype=np.float32)
    gallery = np.zeros((4, 768), dtype=np.float32)
    query[0, 0] = query[1, 1] = 1.0
    gallery[0, 0] = gallery[1, 0] = 1.0
    gallery[2, 1] = gallery[3, 1] = 1.0

    def encode(
        model_state: dict[str, object], row: dict[str, object]
    ) -> dict[str, object]:
        calls.append(("encode", row["arm"], row["epoch"]))
        assert model_state == {"arm": row["arm"], "epoch": row["epoch"]}
        return {
            "query": query.copy(),
            "gallery": gallery.copy(),
            "query_labels": np.asarray(["a", "b"]),
            "gallery_labels": np.asarray(["a", "a", "b", "b"]),
            "query_ids": ("query/a.jpg", "query/b.jpg"),
            "gallery_ids": (
                "gallery/a1.jpg",
                "gallery/a2.jpg",
                "gallery/b1.jpg",
                "gallery/b2.jpg",
            ),
            "elapsed_seconds": 0.25,
            "peak_allocated_bytes": 123,
        }

    result = module.evaluate_pair(config, load_checkpoint, encode)

    assert calls == [
        (operation, arm, epoch)
        for epoch in module.EPOCHS
        for arm in module.ARMS
        for operation in ("load", "encode")
    ]
    assert [row["epoch"] for row in result["rows"]] == list(module.EPOCHS)
    assert result["seed"] == 0
    for row in result["rows"]:
        assert tuple(row["arms"]) == module.ARMS
        assert row["arms"]["sampled_512"]["primary"]["map_at_r"] == 1.0
        assert row["arms"]["full_768"]["primary"]["map_at_r"] == 1.0
    module.validate_pair_result(result, config)

    result["rows"][0]["arms"]["sampled_512"]["primary"]["recall"]["10"] = 0.5
    with pytest.raises(ValueError, match="paired retrieval view aggregates differ"):
        module.validate_pair_result(result, config)


@pytest.mark.parametrize(
    "drift",
    ("query_ids", "gallery_ids", "query_labels", "gallery_labels"),
)
def test_evaluate_pair_rejects_cross_arm_record_drift(drift: str) -> None:
    module = _load_script()
    config, _result = _valid_pair_payload(module)
    query = np.zeros((2, 768), dtype=np.float32)
    gallery = np.zeros((4, 768), dtype=np.float32)
    query[0, 0] = query[1, 1] = 1.0
    gallery[0, 0] = gallery[1, 0] = 1.0
    gallery[2, 1] = gallery[3, 1] = 1.0

    def load_checkpoint(row):
        selected = 512 if row["arm"] == "sampled_512" else 768
        return {
            "epoch": row["epoch"],
            "model": {"arm": row["arm"]},
            "selection_holdout": {"seed": 0, "fraction": 0.2},
            "training_protocol": {
                "objective": "official-eight-mask",
                "selected_features": selected,
                "evaluation_features": 768,
            },
        }

    def encode(_model_state, row):
        value = {
            "query": query.copy(),
            "gallery": gallery.copy(),
            "query_labels": np.asarray(["a", "b"]),
            "gallery_labels": np.asarray(["a", "a", "b", "b"]),
            "query_ids": ("query/a.jpg", "query/b.jpg"),
            "gallery_ids": (
                "gallery/a1.jpg",
                "gallery/a2.jpg",
                "gallery/b1.jpg",
                "gallery/b2.jpg",
            ),
            "elapsed_seconds": 0.25,
            "peak_allocated_bytes": 123,
        }
        if row["arm"] == "full_768":
            replacements = {
                "query_ids": ("query/a.jpg", "query/c.jpg"),
                "gallery_ids": (
                    "gallery/a1.jpg",
                    "gallery/a2.jpg",
                    "gallery/b1.jpg",
                    "gallery/b3.jpg",
                ),
                "query_labels": np.asarray(["b", "a"]),
                "gallery_labels": np.asarray(["a", "b", "a", "b"]),
            }
            value[drift] = replacements[drift]
        return value

    with pytest.raises(ValueError, match="paired checkpoint evaluation records differ"):
        module.evaluate_pair(config, load_checkpoint, encode)


@pytest.mark.parametrize("drift", ("epoch", "model", "selection_holdout"))
def test_evaluate_pair_rejects_raw_checkpoint_drift(drift: str) -> None:
    module = _load_script()
    config, _result = _valid_pair_payload(module)

    def load_checkpoint(row):
        selected = 512 if row["arm"] == "sampled_512" else 768
        value = {
            "epoch": row["epoch"],
            "model": {"arm": row["arm"]},
            "selection_holdout": {"seed": 0, "fraction": 0.2},
            "training_protocol": {
                "objective": "official-eight-mask",
                "selected_features": selected,
                "evaluation_features": 768,
            },
        }
        replacements = {
            "epoch": row["epoch"] + 1,
            "model": None,
            "selection_holdout": {"seed": 1, "fraction": 0.2},
        }
        value[drift] = replacements[drift]
        return value

    with pytest.raises(ValueError, match="paired raw checkpoint binding differs"):
        module.evaluate_pair(
            config,
            load_checkpoint,
            lambda _model, _row: pytest.fail("invalid checkpoint reached encoding"),
        )


def test_pair_result_rejects_cross_row_and_cross_view_drift() -> None:
    module = _load_script()
    config, result = _valid_pair_payload(module)

    mutated = copy.deepcopy(result)
    mutated["rows"][1]["query_ids_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="paired evaluator record order differs"):
        module.validate_pair_result(mutated, config)

    def truncate(view):
        view["average_precision"] = [1.0]
        view["top1_correct"] = [True]
        view["map_at_r"] = 1.0
        view["recall"] = {str(key): 1.0 for key in module.RECALL_AT_K}
        view["recall_counts"] = {str(key): 1 for key in module.RECALL_AT_K}

    mutated = copy.deepcopy(result)
    truncate(mutated["rows"][0]["arms"]["sampled_512"]["primary"])
    with pytest.raises(ValueError, match="paired evaluator view counts differ"):
        module.validate_pair_result(mutated, config)

    mutated = copy.deepcopy(result)
    for view in ("primary", "legacy"):
        truncate(mutated["rows"][1]["arms"]["sampled_512"][view])
    with pytest.raises(ValueError, match="paired evaluator query count differs"):
        module.validate_pair_result(mutated, config)


def test_pair_config_generated_mutations_cover_every_schema_path() -> None:
    module = _load_script()
    config, _result = _valid_pair_payload(module)
    traversed = {path for path, _value in _walk_schema(config)}
    exercised = set()

    removed_mapping_members = set()
    for path, original in _walk_schema(config):
        mutated = copy.deepcopy(config)
        target = _at_path(mutated, path) if path else mutated
        if type(original) is dict:
            for key in original:
                member_mutated = copy.deepcopy(config)
                member_target = _at_path(member_mutated, path) if path else member_mutated
                member_target.pop(key)
                with pytest.raises((TypeError, ValueError), match="paired evaluator"):
                    module._validate_pair_config(member_mutated)
                removed_mapping_members.add((*path, key))
            exercised.add(path)
            continue
        elif type(original) is list:
            target.pop()
        elif type(original) is bool:
            _set_path(mutated, path, 1)
        elif type(original) is int:
            for nonfinite in (float("nan"), float("inf"), float("-inf")):
                nonfinite_mutated = copy.deepcopy(config)
                _set_path(nonfinite_mutated, path, nonfinite)
                with pytest.raises((TypeError, ValueError), match="paired evaluator"):
                    module._validate_pair_config(nonfinite_mutated)
            _set_path(mutated, path, True)
        elif type(original) is str:
            _set_path(mutated, path, None)
        else:
            raise AssertionError((path, type(original)))
        with pytest.raises((TypeError, ValueError), match="paired evaluator"):
            module._validate_pair_config(mutated)
        exercised.add(path)

    for path, original in _walk_schema(config):
        if type(original) is not dict:
            continue
        mutated = copy.deepcopy(config)
        target = _at_path(mutated, path) if path else mutated
        target["__extra__"] = None
        with pytest.raises((TypeError, ValueError), match="paired evaluator"):
            module._validate_pair_config(mutated)

        mutated = copy.deepcopy(config)
        target = _at_path(mutated, path) if path else mutated
        reordered = dict(reversed(tuple(target.items())))
        if path:
            _set_path(mutated, path, reordered)
        else:
            mutated = reordered
        with pytest.raises((TypeError, ValueError), match="paired evaluator"):
            module._validate_pair_config(mutated)

    assert exercised == traversed
    assert removed_mapping_members


def test_pair_result_generated_mutations_cover_every_schema_path() -> None:
    module = _load_script()
    config, result = _valid_pair_payload(module)
    traversed = {path for path, _value in _walk_schema(result)}
    exercised = set()

    removed_mapping_members = set()
    for path, original in _walk_schema(result):
        mutated = copy.deepcopy(result)
        target = _at_path(mutated, path) if path else mutated
        if type(original) is dict:
            for key in original:
                member_mutated = copy.deepcopy(result)
                member_target = _at_path(member_mutated, path) if path else member_mutated
                member_target.pop(key)
                with pytest.raises((TypeError, ValueError)):
                    module.validate_pair_result(member_mutated, config)
                removed_mapping_members.add((*path, key))
            exercised.add(path)
            continue
        elif type(original) is list:
            target.pop()
        elif type(original) is bool:
            _set_path(mutated, path, 1)
        elif type(original) is int:
            for nonfinite in (float("nan"), float("inf"), float("-inf")):
                nonfinite_mutated = copy.deepcopy(result)
                _set_path(nonfinite_mutated, path, nonfinite)
                with pytest.raises((TypeError, ValueError)):
                    module.validate_pair_result(nonfinite_mutated, config)
            _set_path(mutated, path, True)
        elif type(original) is float:
            for nonfinite in (float("nan"), float("inf"), float("-inf")):
                nonfinite_mutated = copy.deepcopy(result)
                _set_path(nonfinite_mutated, path, nonfinite)
                with pytest.raises((TypeError, ValueError)):
                    module.validate_pair_result(nonfinite_mutated, config)
            exercised.add(path)
            continue
        elif type(original) is str:
            _set_path(mutated, path, None)
        else:
            raise AssertionError((path, type(original)))
        with pytest.raises((TypeError, ValueError)):
            module.validate_pair_result(mutated, config)
        exercised.add(path)

    for path, original in _walk_schema(result):
        if type(original) is not dict:
            continue
        mutated = copy.deepcopy(result)
        target = _at_path(mutated, path) if path else mutated
        target["__extra__"] = None
        with pytest.raises((TypeError, ValueError)):
            module.validate_pair_result(mutated, config)

        mutated = copy.deepcopy(result)
        target = _at_path(mutated, path) if path else mutated
        reordered = dict(reversed(tuple(target.items())))
        if path:
            _set_path(mutated, path, reordered)
        else:
            mutated = reordered
        with pytest.raises((TypeError, ValueError)):
            module.validate_pair_result(mutated, config)

    assert exercised == traversed
    assert removed_mapping_members


def test_pair_result_rejects_finite_relational_drift() -> None:
    module = _load_script()
    config, result = _valid_pair_payload(module)
    view_paths = [
        (row_index, arm, view)
        for row_index in range(len(module.EPOCHS))
        for arm in module.ARMS
        for view in ("primary", "legacy")
    ]

    for row_index, arm, view in view_paths:
        path = ("rows", row_index, "arms", arm, view)
        for key in module.RECALL_AT_K:
            mutated = copy.deepcopy(result)
            _at_path(mutated, path)["recall"][str(key)] = 0.5
            with pytest.raises(ValueError, match="aggregates differ"):
                module.validate_pair_result(mutated, config)

            mutated = copy.deepcopy(result)
            _at_path(mutated, path)["recall_counts"][str(key)] = 1
            with pytest.raises(ValueError, match="aggregates differ"):
                module.validate_pair_result(mutated, config)

        mutated = copy.deepcopy(result)
        _at_path(mutated, path)["map_at_r"] = 0.5
        with pytest.raises(ValueError, match="aggregates differ"):
            module.validate_pair_result(mutated, config)

        mutated = copy.deepcopy(result)
        _at_path(mutated, path)["average_precision"][0] = 0.5
        with pytest.raises(ValueError, match="aggregates differ"):
            module.validate_pair_result(mutated, config)

        mutated = copy.deepcopy(result)
        _at_path(mutated, path)["top1_correct"][0] = False
        with pytest.raises(ValueError, match="aggregates differ"):
            module.validate_pair_result(mutated, config)


def test_historical_protocol_without_evaluation_width_is_rejected() -> None:
    module = _load_script()

    with pytest.raises(ValueError):
        module.validate_arm_protocol(
            {"objective": "official-eight-mask", "selected_features": 512},
            "sampled_512",
        )


def test_cli_runs_registered_pair_and_publishes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    config = tmp_path / "pair.json"
    output = tmp_path / "result.json"
    config.write_text("{}\n", encoding="utf-8")
    observed: list[object] = []

    def run_registered_pair(args):
        observed.append(
            (
                args.config,
                args.unicom_checkout,
                args.initial_checkpoint,
                args.dataset_root,
                args.batch_size,
                args.workers,
            )
        )
        return {"schema_version": "synthetic-pair-v1"}, {"expected": "config"}

    def publish_result(payload, destination, *, validate):
        observed.append((payload, destination))
        validate(payload)

    monkeypatch.setattr(module, "run_registered_pair", run_registered_pair, raising=False)
    monkeypatch.setattr(module, "publish_result", publish_result)
    monkeypatch.setattr(
        module,
        "validate_pair_result",
        lambda value, expected: observed.append((value, expected)),
    )

    status = module.main(
        [
            "--config",
            str(config),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output",
            str(output),
            "--batch-size",
            "64",
            "--workers",
            "2",
        ]
    )

    assert status == 0
    assert observed == [
        (
            config,
            tmp_path / "unicom",
            tmp_path / "initial.pt",
            tmp_path / "dataset",
            64,
            2,
        ),
        ({"schema_version": "synthetic-pair-v1"}, output),
        ({"schema_version": "synthetic-pair-v1"}, {"expected": "config"}),
    ]


def test_cli_rejects_preexisting_output_before_loading_any_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    config = tmp_path / "pair.json"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "result.json"
    output.write_text("do not clobber\n", encoding="utf-8")
    called = False

    def forbidden_runner(_args):
        nonlocal called
        called = True
        raise AssertionError("checkpoint evaluation started before output preflight")

    monkeypatch.setattr(module, "run_registered_pair", forbidden_runner)

    status = module.main(
        [
            "--config",
            str(config),
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output",
            str(output),
        ]
    )

    assert status == 2
    assert called is False
    assert output.read_text(encoding="utf-8") == "do not clobber\n"


def test_registered_checkpoint_authenticates_file_before_deserialization(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint_path = tmp_path / "epoch-0004.pt"
    checkpoint_path.write_bytes(b"registered checkpoint bytes")
    inventory = {
        "arm": "sampled_512",
        "epoch": 4,
        "path": str(checkpoint_path),
        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "bytes": checkpoint_path.stat().st_size,
    }
    checkpoint = {
        "epoch": 4,
        "model": {"weight": object()},
        "classifier": object(),
        "ema": object(),
        "optimizer": object(),
        "scheduler": object(),
        "scaler": object(),
        "mask_generator": object(),
        "torch_rng_state": object(),
        "cuda_rng_states": object(),
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": {
            "seed": 0,
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        "history": object(),
    }
    calls = []

    def torch_load(path, *, map_location, weights_only, mmap):
        calls.append((path, map_location, weights_only, mmap))
        return checkpoint

    assert module.load_registered_checkpoint(inventory, torch_load=torch_load) is checkpoint
    assert calls == [(checkpoint_path, "cpu", False, True)]

    checkpoint_path.write_bytes(b"altered checkpoint bytes")
    with pytest.raises(ValueError, match="checkpoint file binding differs"):
        module.load_registered_checkpoint(inventory, torch_load=torch_load)
    assert len(calls) == 1


def test_registered_pair_uses_authenticated_config_and_real_callback_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script()
    inventory = []
    for epoch in module.EPOCHS:
        for arm in module.ARMS:
            inventory.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "path": f"/{arm}/epoch-{epoch:04d}.pt",
                    "sha256": f"{len(inventory) + 1:x}" * 64,
                    "bytes": 1000 + epoch,
                }
            )
    config = {
        "schema_version": "unicom-full-width-pair-config-v1",
        "seed": 0,
        "inventory": inventory,
    }
    config_path = tmp_path / "pair.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    args = Namespace(
        config=config_path,
        unicom_checkout=tmp_path / "unicom",
        initial_checkpoint=tmp_path / "initial.pt",
        dataset_root=tmp_path / "dataset",
        output=tmp_path / "result.json",
        batch_size=128,
        workers=4,
    )
    callbacks = (object(), object())
    expected_result = {"schema_version": "result"}
    observed = []
    monkeypatch.setattr(
        module,
        "build_real_pair_callbacks",
        lambda received_args, received_config: (
            observed.append((received_args, received_config)) or callbacks
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "evaluate_pair",
        lambda received_config, load, encode: (
            observed.append((received_config, load, encode)) or expected_result
        ),
    )

    result, authenticated_config = module.run_registered_pair(args)

    assert result is expected_result
    assert authenticated_config == config
    assert observed == [
        (args, config),
        (config, callbacks[0], callbacks[1]),
    ]


def test_real_callback_factory_uses_only_train_holdout_and_raw_model_state(
    tmp_path: Path,
) -> None:
    module = _load_script()
    inventory = []
    for epoch in module.EPOCHS:
        for arm in module.ARMS:
            path = tmp_path / arm / f"epoch-{epoch:04d}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(f"{arm}-{epoch}".encode())
            inventory.append(
                {
                    "arm": arm,
                    "epoch": epoch,
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                }
            )
    config = {
        "schema_version": "unicom-full-width-pair-config-v1",
        "seed": 0,
        "inventory": inventory,
    }
    initial = tmp_path / "FP16-ViT-L-14-336px.pt"
    initial.write_bytes(b"initial checkpoint")
    checkout = tmp_path / "unicom"
    checkout.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "Eval").mkdir()
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.write_bytes(b"registered partition")
    args = Namespace(
        unicom_checkout=checkout,
        initial_checkpoint=initial,
        dataset_root=dataset,
        batch_size=2,
        workers=0,
    )
    train_a = SimpleNamespace(split="train", image_path=tmp_path / "a.jpg", label="a")
    train_b = SimpleNamespace(split="train", image_path=tmp_path / "b.jpg", label="a")
    train_c = SimpleNamespace(split="train", image_path=tmp_path / "c.jpg", label="a")
    forbidden = SimpleNamespace(split="query", image_path=tmp_path / "official.jpg", label="z")
    encoded_records = []
    loaded_states = []

    class Model:
        def to(self, device):
            assert device == "cuda"
            return self

        def load_state_dict(self, state, *, strict):
            assert strict is True
            loaded_states.append(state)

    model = Model()

    def encode_records(received_model, records, transform, *, device, batch_size, workers):
        assert received_model is model
        assert transform == "transform"
        assert (device, batch_size, workers) == ("cuda", 2, 0)
        encoded_records.append(records)
        values = np.zeros((len(records), 768), dtype=np.float32)
        for index in range(len(records)):
            values[index, index] = 1.0
        return values, np.asarray([row.label for row in records])

    trainer_source = tmp_path / "train_unicom_inshop.py"
    trainer_source.write_bytes(b"registered trainer source")
    trainer = SimpleNamespace(
        __file__=str(trainer_source),
        UNICOM_REVISION="revision",
        UNICOM_L14_336_SHA256=hashlib.sha256(initial.read_bytes()).hexdigest(),
        _load_official_model=lambda received_checkout, received_initial: (
            model,
            "transform",
        ),
        _encode_records=encode_records,
    )
    checkpoint = {
        "epoch": 4,
        "model": {"weight": "raw"},
        "classifier": object(),
        "ema": object(),
        "optimizer": object(),
        "scheduler": object(),
        "scaler": object(),
        "mask_generator": object(),
        "torch_rng_state": object(),
        "cuda_rng_states": object(),
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": {
            "seed": 0,
            "trainer_sha256": hashlib.sha256(trainer_source.read_bytes()).hexdigest(),
            "unicom_revision": "revision",
            "initial_checkpoint_sha256": hashlib.sha256(initial.read_bytes()).hexdigest(),
            "partition_sha256": hashlib.sha256(partition.read_bytes()).hexdigest(),
            "objective": "official-eight-mask",
            "selected_features": 512,
            "evaluation_features": 768,
        },
        "history": object(),
    }

    class Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def reset_peak_memory_stats():
            return None

        @staticmethod
        def max_memory_allocated():
            return 321

    torch_module = SimpleNamespace(
        cuda=Cuda(),
        device=lambda name: name,
        load=lambda *args, **kwargs: checkpoint,
    )

    def split_holdout(records, *, fraction, seed):
        assert records == (train_a, train_b, train_c)
        assert (fraction, seed) == (0.2, 0)
        return (train_a,), (train_b,), (train_a, train_c), {"a": 0}

    load_checkpoint, encode = module.build_real_pair_callbacks(
        args,
        config,
        torch_module=torch_module,
        trainer=trainer,
        parse_partition=lambda root: (train_a, forbidden, train_b, train_c),
        split_holdout=split_holdout,
        git_revision=lambda path: "revision",
    )
    loaded = load_checkpoint(inventory[0])
    encoded = encode(loaded["model"], inventory[0])

    assert loaded is checkpoint
    assert loaded_states == [{"weight": "raw"}]
    assert encoded_records == [(train_b,), (train_a, train_c)]
    assert encoded["query_ids"] == (str(train_b.image_path),)
    assert encoded["gallery_ids"] == (str(train_a.image_path), str(train_c.image_path))
    assert encoded["peak_allocated_bytes"] == 321

    for key in (
        "trainer_sha256",
        "unicom_revision",
        "initial_checkpoint_sha256",
        "partition_sha256",
    ):
        registered = checkpoint["training_protocol"][key]
        checkpoint["training_protocol"][key] = "wrong"
        with pytest.raises(ValueError, match="checkpoint provenance differs"):
            load_checkpoint(inventory[0])
        checkpoint["training_protocol"][key] = registered

    checkpoint["training_protocol"]["seed"] = 1
    with pytest.raises(ValueError, match="checkpoint seed binding differs"):
        load_checkpoint(inventory[0])
    checkpoint["training_protocol"]["seed"] = 0

    wrong_name = tmp_path / "renamed-initial.pt"
    wrong_name.write_bytes(initial.read_bytes())
    args.initial_checkpoint = wrong_name
    with pytest.raises(ValueError, match="initial checkpoint filename differs"):
        module.build_real_pair_callbacks(
            args,
            config,
            torch_module=torch_module,
            trainer=trainer,
            parse_partition=lambda root: (train_a, forbidden, train_b, train_c),
            split_holdout=split_holdout,
            git_revision=lambda path: "revision",
        )
