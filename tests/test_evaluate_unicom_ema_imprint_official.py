from __future__ import annotations

import copy
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "scripts" / "evaluate_unicom_ema_imprint_official.py"
SUMMARY_PATH = (
    PROJECT
    / "reports"
    / "generated"
    / "unicom_ema_imprint_replication_summary_3a64952_seeds1-6.json"
)


def _load_module():
    if not MODULE_PATH.is_file():
        pytest.fail("official retained-checkpoint evaluator is absent")
    spec = importlib.util.spec_from_file_location(
        "evaluate_unicom_ema_imprint_official", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_row_order_keeps_seed_one_sensitivity_out_of_gate() -> None:
    module = _load_module()

    rows = module.registered_rows()

    assert len(rows) == 48
    assert [(row["seed"], row["arm"], row["epoch"], row["gating"]) for row in rows[:9]] == [
        (2, "random_raw", 4, True),
        (2, "random_raw", 8, True),
        (2, "random_raw", 12, True),
        (2, "random_raw", 16, True),
        (2, "imprinted_raw", 4, True),
        (2, "imprinted_raw", 8, True),
        (2, "imprinted_raw", 12, True),
        (2, "imprinted_raw", 16, True),
        (3, "random_raw", 4, True),
    ]
    assert [row["seed"] for row in rows[-8:]] == [1] * 8
    assert all(row["gating"] is False for row in rows[-8:])


def test_paired_t_interval_uses_frozen_builtin_float_arithmetic() -> None:
    module = _load_module()
    deltas = (
        0.019490767976936607,
        0.017513576043862833,
        0.01655111383146657,
        0.02171094991505773,
        0.015771041571864153,
    )

    interval = module.paired_t_interval(deltas, critical=2.7764451052)

    assert type(interval) is tuple
    assert len(interval) == 2
    assert all(type(value) is float for value in interval)
    assert interval == pytest.approx((0.015224754666755504, 0.021190225068919427))


@pytest.mark.parametrize(
    "values",
    [
        (),
        (0.1,),
        (0.1, float("nan")),
        (0.1, float("inf")),
        (True, 0.1),
        (1, 0.1),
    ],
)
def test_paired_t_interval_rejects_unregistered_types_and_nonfinite_values(values) -> None:
    module = _load_module()

    with pytest.raises((TypeError, ValueError)):
        module.paired_t_interval(values, critical=2.7764451052)


def test_paired_query_bootstrap_uses_one_shared_query_draw_across_seeds() -> None:
    module = _load_module()
    baseline = np.asarray(
        [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.bool_,
    )
    candidate = np.asarray(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.bool_,
    )

    interval = module.paired_query_bootstrap_interval(baseline, candidate)

    assert interval == pytest.approx((0.1, 0.4))


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        (np.ones((5, 4), dtype=np.float32), np.ones((5, 4), dtype=np.bool_)),
        (np.ones((4, 4), dtype=np.bool_), np.ones((4, 4), dtype=np.bool_)),
        (np.ones((5, 4), dtype=np.bool_), np.ones((5, 3), dtype=np.bool_)),
        (np.ones((5, 0), dtype=np.bool_), np.ones((5, 0), dtype=np.bool_)),
    ],
)
def test_paired_query_bootstrap_rejects_wrong_shape_or_dtype(baseline, candidate) -> None:
    module = _load_module()

    with pytest.raises((TypeError, ValueError)):
        module.paired_query_bootstrap_interval(baseline, candidate)


def test_trajectory_uses_the_random_endpoint_as_one_shared_target() -> None:
    module = _load_module()
    by_seed = {
        2: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.31, 16: 0.3},
            "imprinted_raw": {4: 0.2, 8: 0.3, 12: 0.32, 16: 0.33},
        },
        3: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.25, 16: 0.3},
            "imprinted_raw": {4: 0.31, 8: 0.32, 12: 0.33, 16: 0.34},
        },
        4: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.25, 16: 0.3},
            "imprinted_raw": {4: 0.2, 8: 0.31, 12: 0.32, 16: 0.33},
        },
        5: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.25, 16: 0.3},
            "imprinted_raw": {4: 0.2, 8: 0.31, 12: 0.32, 16: 0.33},
        },
        6: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.25, 16: 0.3},
            "imprinted_raw": {4: 0.1, 8: 0.2, 12: 0.25, 16: 0.3},
        },
    }

    result = module.trajectory_decision(by_seed)

    assert result["per_seed"] == [
        {"seed": 2, "random_epoch": 12, "imprinted_epoch": 8, "speedup": 1.5},
        {"seed": 3, "random_epoch": 16, "imprinted_epoch": 4, "speedup": 4.0},
        {"seed": 4, "random_epoch": 16, "imprinted_epoch": 8, "speedup": 2.0},
        {"seed": 5, "random_epoch": 16, "imprinted_epoch": 8, "speedup": 2.0},
        {"seed": 6, "random_epoch": 16, "imprinted_epoch": 16, "speedup": 1.0},
    ]
    assert result["supported"] is True


def test_trajectory_fails_when_candidate_never_reaches_random_endpoint() -> None:
    module = _load_module()
    by_seed = {
        seed: {
            "random_raw": {4: 0.1, 8: 0.2, 12: 0.3, 16: 0.4},
            "imprinted_raw": {4: 0.1, 8: 0.2, 12: 0.3, 16: 0.39},
        }
        for seed in (2, 3, 4, 5, 6)
    }

    result = module.trajectory_decision(by_seed)

    assert all(row["imprinted_epoch"] is None for row in result["per_seed"])
    assert result["supported"] is False


def test_quality_decision_recomputes_all_frozen_gates_and_sensitivity() -> None:
    module = _load_module()
    baseline = np.zeros((5, 20), dtype=np.bool_)
    candidate = np.zeros((5, 20), dtype=np.bool_)

    result = module.quality_decision(
        primary_map_deltas=(0.01, 0.011, 0.009, 0.012, 0.01),
        legacy_map_deltas=(0.008, 0.009, 0.007, 0.01, 0.008),
        recall_at_1_deltas=(0.0, 0.0, 0.0, 0.0, 0.0),
        baseline_top1=baseline,
        candidate_top1=candidate,
        sensitivity_map_delta=0.009,
    )

    assert result["primary_map_interval"][0] > 0.002
    assert result["positive_primary_seed_count"] == 5
    assert result["recall_at_1_seed_interval"][0] > -0.001
    assert result["recall_at_1_query_interval"][0] > -0.001
    assert result["legacy_sign_agrees"] is True
    assert result["anomaly_reaudit_required"] is False
    assert result["exclusion_sensitive"] is False
    assert result["supported"] is True


def test_quality_decision_marks_five_seed_pass_as_exclusion_sensitive() -> None:
    module = _load_module()
    baseline = np.zeros((5, 20), dtype=np.bool_)

    result = module.quality_decision(
        primary_map_deltas=(0.006, 0.006, 0.006, 0.006, 0.006),
        legacy_map_deltas=(0.003, 0.003, 0.003, 0.003, 0.003),
        recall_at_1_deltas=(0.0, 0.0, 0.0, 0.0, 0.0),
        baseline_top1=baseline,
        candidate_top1=baseline.copy(),
        sensitivity_map_delta=-0.02,
    )

    assert result["supported"] is True
    assert result["six_seed_map_interval"][0] <= 0.002
    assert result["exclusion_sensitive"] is True


@pytest.mark.parametrize(
    ("change", "expected_key"),
    [
        ({"legacy_map_deltas": (-0.01,) * 5}, "legacy_sign_agrees"),
        ({"primary_map_deltas": (0.001,) * 5}, "primary_map_interval"),
        ({"recall_at_1_deltas": (-0.004, 0.0, 0.0, 0.0, 0.0)}, "r1_per_seed_guard"),
        ({"primary_map_deltas": (0.031,) * 5}, "anomaly_reaudit_required"),
    ],
)
def test_quality_decision_falsifiers(change, expected_key) -> None:
    module = _load_module()
    baseline = np.zeros((5, 20), dtype=np.bool_)
    arguments = {
        "primary_map_deltas": (0.01,) * 5,
        "legacy_map_deltas": (0.008,) * 5,
        "recall_at_1_deltas": (0.0,) * 5,
        "baseline_top1": baseline,
        "candidate_top1": baseline.copy(),
        "sensitivity_map_delta": 0.009,
    }
    arguments.update(change)

    result = module.quality_decision(**arguments)

    if expected_key == "anomaly_reaudit_required":
        assert result[expected_key] is True
        assert result["supported"] is True
    else:
        assert result[expected_key] is False or result[expected_key][0] <= 0.002
        assert result["supported"] is False


def test_registered_inventory_binds_real_summary_and_remote_paths() -> None:
    module = _load_module()
    summary = module.load_registered_summary(SUMMARY_PATH)

    inventory = module.registered_checkpoint_inventory(summary, Path("/home/riomus"))

    assert len(inventory) == 48
    assert inventory[0] == {
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "gating": True,
        "path": "/home/riomus/unicom-ema-random-387d697-seed2-e16/epoch-0004.pt",
        "sha256": "3101537c853161daf4eb3f66bd39d68a4b0537ac1d290af7bcebf8b51911f587",
    }
    assert inventory[-1] == {
        "seed": 1,
        "arm": "imprinted_raw",
        "epoch": 16,
        "gating": False,
        "path": ("/home/riomus/unicom-ema-imprinted-81f3f48-seed1-e16-retry1/epoch-0016.pt"),
        "sha256": "d5118bb97d3ab91ba2a21a145eda7dd59af0dd79141e2fc40104d146d4418c9f",
    }
    assert len({row["sha256"] for row in inventory}) == 48


@pytest.mark.parametrize("mutation", ["seed_order", "duplicate_hash", "selected_cell"])
def test_registered_inventory_rejects_summary_drift(mutation: str) -> None:
    module = _load_module()
    summary = json.loads(SUMMARY_PATH.read_text())
    changed = copy.deepcopy(summary)
    if mutation == "seed_order":
        changed["reports"][0], changed["reports"][1] = (
            changed["reports"][1],
            changed["reports"][0],
        )
    elif mutation == "duplicate_hash":
        changed["reports"][1]["random_raw"]["checkpoint_sha256_by_epoch"][0] = changed["reports"][
            0
        ]["random_raw"]["checkpoint_sha256_by_epoch"][0]
    else:
        changed["selected_cell"] = "random_raw"

    with pytest.raises(ValueError):
        module.registered_checkpoint_inventory(changed, Path("/home/riomus"))


def test_embedding_views_use_full_unit_primary_and_legacy_prefix_bias() -> None:
    module = _load_module()
    query = np.zeros((1, 768), dtype=np.float32)
    gallery = np.zeros((2, 768), dtype=np.float32)
    query[0, [0, 1, 600]] = [
        3.3229994773864746,
        0.22578661143779755,
        -1.7631540298461914,
    ]
    gallery[:, [0, 1, 600]] = [
        [-0.28128743171691895, -0.6680463552474976, 2.409726858139038],
        [-1.0551505088806152, -0.39080098271369934, -1.1927679777145386],
    ]

    result = module.evaluate_embedding_views(
        query,
        gallery,
        np.asarray(["a"]),
        np.asarray(["b", "a"]),
    )

    assert result["query_embedding_sha256"] == (
        "0463be0bf8ee2f948e6bacc19e0eb3ab70225f3f0bc19f5555fa6e1b20820702"
    )
    assert result["gallery_embedding_sha256"] == (
        "96d43f991ef730bcbad07a951a1ded2b2344bbb0e0376a48a25c10c9a208a23c"
    )
    assert result["primary"]["map_at_r"] == 1.0
    assert result["primary"]["top1_correct"] == [True]
    assert result["legacy"]["map_at_r"] == 0.0
    assert result["legacy"]["top1_correct"] == [False]
    assert result["primary"]["recall"] == {
        "1": 1.0,
        "10": 1.0,
        "20": 1.0,
        "30": 1.0,
        "40": 1.0,
        "50": 1.0,
    }
    assert result["primary"]["topk_correct_counts"] == {
        "1": 1,
        "10": 1,
        "20": 1,
        "30": 1,
        "40": 1,
        "50": 1,
    }
    assert result["query_prefix_energy_mean"] == pytest.approx(0.7811077062183105)
    assert result["gallery_prefix_energy_mean"] == pytest.approx(0.2769239500460373)


@pytest.mark.parametrize(
    ("query_shape", "gallery_shape"),
    [((0, 768), (2, 768)), ((1, 767), (2, 767)), ((1, 768), (2, 512))],
)
def test_embedding_views_reject_wrong_or_empty_embedding_shapes(query_shape, gallery_shape) -> None:
    module = _load_module()
    query = np.zeros(query_shape, dtype=np.float32)
    gallery = np.zeros(gallery_shape, dtype=np.float32)

    with pytest.raises((TypeError, ValueError)):
        module.evaluate_embedding_views(
            query,
            gallery,
            np.asarray(["a"] * query_shape[0]),
            np.asarray(["a"] * gallery_shape[0]),
        )


def test_identity_uniform_map_uses_lexicographically_first_query_per_identity() -> None:
    module = _load_module()

    value = module.identity_uniform_map(
        (0.0, 1.0, 0.25, 0.75),
        ("id_a", "id_a", "id_b", "id_b"),
        ("z/a2.jpg", "a/a1.jpg", "x/b2.jpg", "b/b1.jpg"),
    )

    assert value == 0.875


@pytest.mark.parametrize(
    ("average_precision", "labels", "paths"),
    [
        ((0.5,), ("a", "a"), ("a", "b")),
        ((0.5,), ("",), ("a",)),
        ((0.5, 0.6), ("a", "b"), ("same", "same")),
        ((float("nan"),), ("a",), ("a",)),
    ],
)
def test_identity_uniform_map_rejects_malformed_query_evidence(
    average_precision, labels, paths
) -> None:
    module = _load_module()

    with pytest.raises((TypeError, ValueError)):
        module.identity_uniform_map(average_precision, labels, paths)


def test_evaluation_model_must_be_eval_fp32_and_batchnorm_free() -> None:
    module = _load_module()
    model = torch.nn.Linear(3, 2).float().eval()

    module.validate_evaluation_model(model)

    with pytest.raises(ValueError, match="eval mode"):
        module.validate_evaluation_model(model.train())
    with pytest.raises(ValueError, match="FP32"):
        module.validate_evaluation_model(torch.nn.Linear(3, 2).half().eval())
    with pytest.raises(ValueError, match="BatchNorm"):
        module.validate_evaluation_model(torch.nn.BatchNorm1d(3).float().eval())


def _literal_valid_view() -> dict[str, object]:
    return {
        "recall": {"1": 0.5, "10": 1.0, "20": 1.0, "30": 1.0, "40": 1.0, "50": 1.0},
        "topk_correct_counts": {"1": 1, "10": 2, "20": 2, "30": 2, "40": 2, "50": 2},
        "map_at_r": 0.75,
        "average_precision": [0.5, 1.0],
        "top1_correct": [True, False],
    }


def _literal_valid_row() -> dict[str, object]:
    return {
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "gating": True,
        "checkpoint_path": "/home/riomus/run/epoch-0004.pt",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_model_key": "model",
        "query_embedding_sha256": "b" * 64,
        "gallery_embedding_sha256": "c" * 64,
        "query_prefix_energy_mean": 0.8,
        "gallery_prefix_energy_mean": 0.7,
        "primary": _literal_valid_view(),
        "legacy": _literal_valid_view(),
        "identity_uniform_map": {"primary": 0.7, "legacy": 0.7},
        "elapsed_seconds": 12.5,
        "peak_gpu_mib": 1234,
    }


def test_evaluation_row_validator_recomputes_query_metrics_and_counts() -> None:
    module = _load_module()
    row = _literal_valid_row()
    expected = {
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "gating": True,
        "path": "/home/riomus/run/epoch-0004.pt",
        "sha256": "a" * 64,
    }

    module.validate_evaluation_row(row, expected, query_count=2)


def test_view_validator_accepts_production_numpy_mean_at_official_scale() -> None:
    module = _load_module()
    values = np.random.default_rng(1).random(14_218).astype(np.float64)
    top1 = values > 0.5
    top1_count = int(top1.sum())
    view = {
        "recall": {
            "1": top1_count / 14_218,
            "10": 1.0,
            "20": 1.0,
            "30": 1.0,
            "40": 1.0,
            "50": 1.0,
        },
        "topk_correct_counts": {
            "1": top1_count,
            "10": 14_218,
            "20": 14_218,
            "30": 14_218,
            "40": 14_218,
            "50": 14_218,
        },
        "map_at_r": float(np.mean(values)),
        "average_precision": values.tolist(),
        "top1_correct": top1.tolist(),
    }

    module._validate_view(view, query_count=14_218)


def test_recall_extension_preserves_existing_retrieval_bytes() -> None:
    from sfora.unicom_retrieval_audit import retrieval_metrics_from_score_chunks

    scores = np.random.default_rng(205).integers(-3, 4, size=(7, 80)).astype(np.float64)
    query_labels = np.asarray([f"id-{index}" for index in range(7)])
    gallery_labels = np.asarray(
        [f"id-{index % 7}" for index in range(70)] + [f"other-{index}" for index in range(10)]
    )
    baseline = retrieval_metrics_from_score_chunks([scores], query_labels, gallery_labels)
    extended = retrieval_metrics_from_score_chunks(
        [scores], query_labels, gallery_labels, recall_at_k=(1, 10, 20, 30, 40, 50)
    )

    assert extended.map_at_r.hex() == baseline.map_at_r.hex()
    assert {key: extended.recall[key].hex() for key in (1, 10, 20, 30)} == {
        key: baseline.recall[key].hex() for key in (1, 10, 20, 30)
    }
    assert extended.average_precision.tobytes() == baseline.average_precision.tobytes()
    assert extended.top1_indices.tobytes() == baseline.top1_indices.tobytes()
    assert extended.top1_correct.tobytes() == baseline.top1_correct.tobytes()


@pytest.mark.parametrize("recall_at_k", [(), (1, 1), (10, 1), (0, 1), (True, 10), [1, 10]])
def test_recall_extension_rejects_unregistered_contract(recall_at_k) -> None:
    from sfora.unicom_retrieval_audit import retrieval_metrics_from_score_chunks

    with pytest.raises(ValueError, match="recall-at-k"):
        retrieval_metrics_from_score_chunks(
            [np.ones((1, 2), dtype=np.float64)],
            np.asarray(["a"]),
            np.asarray(["a", "b"]),
            recall_at_k=recall_at_k,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_key",
        "map_inconsistent",
        "recall_count_inconsistent",
        "ap_length",
        "top1_type",
        "hash",
        "gating",
        "energy",
        "model_key",
        "identity_map",
        "elapsed",
        "peak_type",
        "view_order",
    ],
)
def test_evaluation_row_validator_rejects_recursive_relational_drift(mutation: str) -> None:
    module = _load_module()
    row = _literal_valid_row()
    expected = {
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "gating": True,
        "path": "/home/riomus/run/epoch-0004.pt",
        "sha256": "a" * 64,
    }
    if mutation == "extra_key":
        row["extra"] = None
    elif mutation == "map_inconsistent":
        row["primary"]["map_at_r"] = 0.8
    elif mutation == "recall_count_inconsistent":
        row["primary"]["recall"]["10"] = 0.5
    elif mutation == "ap_length":
        row["primary"]["average_precision"] = [0.5]
    elif mutation == "top1_type":
        row["primary"]["top1_correct"][0] = 1
    elif mutation == "hash":
        row["query_embedding_sha256"] = "z" * 64
    elif mutation == "gating":
        row["gating"] = False
    elif mutation == "energy":
        row["gallery_prefix_energy_mean"] = -0.1
    elif mutation == "model_key":
        row["checkpoint_model_key"] = "ema"
    elif mutation == "identity_map":
        row["identity_uniform_map"]["primary"] = 1.1
    elif mutation == "elapsed":
        row["elapsed_seconds"] = -1.0
    elif mutation == "peak_type":
        row["peak_gpu_mib"] = True
    else:
        view = row["primary"]
        row["primary"] = {
            "map_at_r": view["map_at_r"],
            **{key: value for key, value in view.items() if key != "map_at_r"},
        }

    with pytest.raises((TypeError, ValueError)):
        module.validate_evaluation_row(row, expected, query_count=2)


def test_loaded_checkpoint_uses_only_raw_model_under_inference_mode() -> None:
    module = _load_module()
    source = torch.nn.Linear(3, 768, bias=False).float()
    with torch.no_grad():
        source.weight.zero_()
        source.weight[:3, :3] = torch.eye(3)
    target = torch.nn.Linear(3, 768, bias=False).float().train()
    checkpoint = {
        "model": source.state_dict(),
        "ema": {"weight": torch.full_like(source.weight, 99.0)},
    }

    def encode(model):
        assert model.training is False
        assert torch.is_grad_enabled() is False
        query = model(torch.tensor([[1.0, 0.0, 0.0]])).numpy()
        gallery = model(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])).numpy()
        return (
            np.ascontiguousarray(query, dtype=np.float32),
            np.ascontiguousarray(gallery, dtype=np.float32),
            np.asarray(["a"]),
            np.asarray(["a", "b"]),
            ("q/a.jpg",),
        )

    result = module.evaluate_loaded_checkpoint(target, checkpoint, encode)

    assert result["primary"]["map_at_r"] == 1.0
    assert torch.equal(target.weight, source.weight)
    assert not torch.all(target.weight == 99.0)


def test_loaded_checkpoint_requires_strict_raw_model_state() -> None:
    module = _load_module()
    target = torch.nn.Linear(3, 768, bias=False).float()

    with pytest.raises((KeyError, RuntimeError, ValueError)):
        module.evaluate_loaded_checkpoint(target, {"model": {}}, lambda _model: None)


def test_progress_line_has_exact_metric_free_schema() -> None:
    module = _load_module()
    row = {
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "sha256": "a" * 64,
    }

    line = module.progress_line(
        row_index=1,
        inventory_row=row,
        elapsed_seconds=12.5,
        peak_gpu_mib=3456,
    )

    payload = json.loads(line)
    assert tuple(payload) == (
        "row_index",
        "seed",
        "arm",
        "epoch",
        "checkpoint_sha256",
        "elapsed_seconds",
        "peak_gpu_mib",
    )
    assert payload == {
        "row_index": 1,
        "seed": 2,
        "arm": "random_raw",
        "epoch": 4,
        "checkpoint_sha256": "a" * 64,
        "elapsed_seconds": 12.5,
        "peak_gpu_mib": 3456,
    }
    assert not any(token in line for token in ("map_at_r", "recall", "delta", "quality"))


def test_atomic_result_publication_strict_reloads_mode_0600_and_never_clobbers(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "result.json"
    validations: list[dict[str, object]] = []

    module.publish_result(
        {"schema_version": "test-v1", "status": "PASS"},
        output,
        validate=lambda value: validations.append(value),
    )

    assert json.loads(output.read_text()) == {"schema_version": "test-v1", "status": "PASS"}
    assert validations == [{"schema_version": "test-v1", "status": "PASS"}]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.tmp"))
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.publish_result(
            {"schema_version": "test-v1", "status": "FAIL"},
            output,
            validate=lambda _value: None,
        )
    assert output.read_bytes() == original


def test_atomic_result_publication_cleans_owned_temp_after_reload_failure(
    tmp_path: Path,
) -> None:
    module = _load_module()
    output = tmp_path / "result.json"

    with pytest.raises(ValueError, match="rejected"):
        module.publish_result(
            {"schema_version": "test-v1", "status": "PASS"},
            output,
            validate=lambda _value: (_ for _ in ()).throw(ValueError("rejected")),
        )

    assert not output.exists()
    assert not list(tmp_path.iterdir())


def _synthetic_inventory_and_rows():
    inventory = []
    rows = []
    for index, registered in enumerate(
        [
            (seed, arm, epoch, seed != 1)
            for seed in (2, 3, 4, 5, 6, 1)
            for arm in ("random_raw", "imprinted_raw")
            for epoch in (4, 8, 12, 16)
        ]
    ):
        seed, arm, epoch, gating = registered
        digest = f"{index + 1:064x}"
        expected = {
            "seed": seed,
            "arm": arm,
            "epoch": epoch,
            "gating": gating,
            "path": f"/checkpoints/seed{seed}/{arm}/epoch-{epoch:04d}.pt",
            "sha256": digest,
        }
        inventory.append(expected)
        if arm == "random_raw":
            value = {4: 0.6, 8: 0.7, 12: 0.74, 16: 0.75}[epoch]
        else:
            value = {4: 0.74, 8: 0.76, 12: 0.77, 16: 0.76}[epoch]
        view = {
            "recall": {"1": 0.5, "10": 1.0, "20": 1.0, "30": 1.0, "40": 1.0, "50": 1.0},
            "topk_correct_counts": {"1": 1, "10": 2, "20": 2, "30": 2, "40": 2, "50": 2},
            "map_at_r": value,
            "average_precision": [value, value],
            "top1_correct": [True, False],
        }
        legacy_value = value - (0.01 if arm == "random_raw" else 0.005)
        legacy = copy.deepcopy(view)
        legacy["map_at_r"] = legacy_value
        legacy["average_precision"] = [legacy_value, legacy_value]
        rows.append(
            {
                "seed": seed,
                "arm": arm,
                "epoch": epoch,
                "gating": gating,
                "checkpoint_path": expected["path"],
                "checkpoint_sha256": digest,
                "checkpoint_model_key": "model",
                "query_embedding_sha256": f"{index + 101:064x}",
                "gallery_embedding_sha256": f"{index + 201:064x}",
                "query_prefix_energy_mean": 0.8,
                "gallery_prefix_energy_mean": 0.7,
                "primary": view,
                "legacy": legacy,
                "identity_uniform_map": {"primary": value, "legacy": legacy_value},
                "elapsed_seconds": 10.0,
                "peak_gpu_mib": 1000,
            }
        )
    return inventory, rows


def _result_metadata():
    return {
        "design_commit": "6bc00b1" + "0" * 33,
        "source_commit": "d" * 40,
        "dataset": {
            "partition_sha256": "c" * 64,
            "query_count": 2,
            "gallery_count": 2,
            "test_identity_count": 2,
        },
        "environment": {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "numpy": "2.5.0",
            "cuda": "13.0",
            "cublas_workspace_config": ":4096:8",
        },
        "evaluation": {
            "transform_repr": "Compose(...)\n",
            "transform_sha256": "e" * 64,
            "batch_size": 128,
            "workers": 4,
            "dtype": "float32",
        },
        "attempts": [{"attempt": 1, "exit_status": 0}],
    }


def test_result_builder_and_validator_recompute_gate_from_all_48_rows() -> None:
    module = _load_module()
    inventory, rows = _synthetic_inventory_and_rows()

    result = module.build_result(inventory=inventory, rows=rows, **_result_metadata())

    module.validate_result(result)
    assert result["status"] == "PASS"
    assert result["decisions"]["quality"]["supported"] is True
    assert result["decisions"]["trajectory"]["supported"] is True
    assert result["decisions"]["retained_checkpoint_gate_passed"] is True
    assert len(result["rows"]) == 48


def test_complete_result_validates_after_exact_json_roundtrip(tmp_path: Path) -> None:
    module = _load_module()
    inventory, rows = _synthetic_inventory_and_rows()
    result = module.build_result(inventory=inventory, rows=rows, **_result_metadata())
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")

    persisted = module.strict_json_object(path)

    module.validate_result(persisted)


@pytest.mark.parametrize(
    "mutation",
    [
        "row_order",
        "row_metric",
        "decision",
        "status",
        "attempt",
        "dataset_count",
        "extra_key",
    ],
)
def test_result_validator_rejects_posthoc_or_relational_drift(mutation: str) -> None:
    module = _load_module()
    inventory, rows = _synthetic_inventory_and_rows()
    result = module.build_result(inventory=inventory, rows=rows, **_result_metadata())
    changed = copy.deepcopy(result)
    if mutation == "row_order":
        changed["rows"][0], changed["rows"][1] = changed["rows"][1], changed["rows"][0]
    elif mutation == "row_metric":
        changed["rows"][0]["primary"]["map_at_r"] = 0.61
    elif mutation == "decision":
        changed["decisions"]["quality"]["supported"] = False
    elif mutation == "status":
        changed["status"] = "FAIL"
    elif mutation == "attempt":
        changed["attempts"] = [{"attempt": 1, "exit_status": 1}]
    elif mutation == "dataset_count":
        changed["dataset"]["query_count"] = 3
    else:
        changed["extra"] = None

    with pytest.raises((TypeError, ValueError)):
        module.validate_result(changed)


def _run_config(inventory: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "unicom-ema-imprint-official-run-v1",
        "design_commit": "6bc00b1" + "0" * 33,
        "source_commit": "d" * 40,
        "dataset_root": "/datasets/inshop_official_standard",
        "partition_sha256": "e" * 64,
        "checkpoint_root": "/checkpoints",
        "summary_path": (
            "reports/generated/"
            "unicom_ema_imprint_replication_summary_3a64952_seeds1-6.json"
        ),
        "unicom_checkout": "/home/riomus/UNICOM",
        "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "trainer_sha256_by_seed": [
            {
                "seed": seed,
                "sha256": (
                    "6596994c96e6834bafd98a84834d877bc97c3a0965238dd12dda7aec191073ae"
                    if seed == 1
                    else "991a48bff567547f115cd23268357c4691514c2f64ab9b80a36cfe84d213ddc4"
                ),
            }
            for seed in range(1, 7)
        ],
        "initial_checkpoint": "/home/riomus/models/FP16-ViT-L-14-336px.pt",
        "initial_checkpoint_sha256": (
            "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
        ),
        "output": "reports/generated/unicom_ema_imprint_official.json",
        "batch_size": 128,
        "workers": 4,
        "attempt": 1,
        "prior_attempt_exit_statuses": [],
        "inventory": inventory,
    }


def test_run_config_is_exact_and_binds_all_48_rows() -> None:
    module = _load_module()
    inventory, _rows = _synthetic_inventory_and_rows()
    config = _run_config(inventory)

    module.validate_run_config(config)

    changed = copy.deepcopy(config)
    changed["inventory"][0]["sha256"] = changed["inventory"][1]["sha256"]
    with pytest.raises(ValueError):
        module.validate_run_config(changed)


def test_preflight_hashes_every_checkpoint_before_opening_partition(
    tmp_path: Path,
) -> None:
    module = _load_module()
    inventory, _rows = _synthetic_inventory_and_rows()
    config = _run_config(inventory)
    config["partition_sha256"] = module.hashlib.sha256(b"partition").hexdigest()
    opened: list[str] = []
    hashed: list[str] = []

    def hash_file(path: Path) -> str:
        hashed.append(str(path))
        if str(path) == config["initial_checkpoint"]:
            return config["initial_checkpoint_sha256"]
        return next(row["sha256"] for row in inventory if row["path"] == str(path))

    def open_partition(_path: Path) -> bytes:
        opened.append("partition")
        assert len(hashed) == 49
        return b"partition"

    module.preflight_registered_inputs(
        config,
        hash_file=hash_file,
        read_partition=open_partition,
    )

    assert len(hashed) == 49
    assert opened == ["partition"]


def test_summary_authority_binds_dataset_unicom_and_initial_model() -> None:
    module = _load_module()
    summary = module.load_registered_summary(SUMMARY_PATH)
    inventory, _rows = _synthetic_inventory_and_rows()
    config = _run_config(inventory)
    config["partition_sha256"] = (
        "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
    )

    module.validate_summary_authority(summary, config)

    changed = copy.deepcopy(config)
    changed["partition_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="summary authority"):
        module.validate_summary_authority(summary, changed)


def test_run_config_requires_exact_initial_checkpoint_basename() -> None:
    module = _load_module()
    inventory, _rows = _synthetic_inventory_and_rows()
    config = _run_config(inventory)
    config["initial_checkpoint"] = "/home/riomus/models/renamed.pt"

    with pytest.raises(ValueError, match="initial checkpoint"):
        module.validate_run_config(config)


def test_authorized_second_attempt_is_preserved_in_result_history() -> None:
    module = _load_module()
    inventory, _rows = _synthetic_inventory_and_rows()
    config = _run_config(inventory)
    config["attempt"] = 2
    config["prior_attempt_exit_statuses"] = [2]

    assert module.attempt_history(config) == [
        {"attempt": 1, "exit_status": 2},
        {"attempt": 2, "exit_status": 0},
    ]


def test_deterministic_runtime_is_strict_and_records_exact_environment(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    environment = module.configure_deterministic_runtime(torch)

    assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert tuple(environment) == ("python", "torch", "numpy", "cuda", "cublas_workspace_config")
    assert environment["cublas_workspace_config"] == ":4096:8"


def test_deterministic_runtime_rejects_wrong_cublas_setting(monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(ValueError, match="CUBLAS"):
        module.configure_deterministic_runtime(torch)


def test_registered_evaluation_runs_exactly_48_rows_in_frozen_order() -> None:
    module = _load_module()
    inventory, rows = _synthetic_inventory_and_rows()
    calls: list[tuple[int, str, int]] = []
    progress: list[str] = []

    def evaluate(row: dict[str, object]) -> dict[str, object]:
        calls.append((row["seed"], row["arm"], row["epoch"]))
        return copy.deepcopy(rows[len(calls) - 1])

    result = module.evaluate_registered_rows(
        inventory,
        query_count=2,
        evaluate=evaluate,
        emit_progress=progress.append,
    )

    assert result == rows
    assert calls == [
        (row["seed"], row["arm"], row["epoch"]) for row in inventory
    ]
    assert len(progress) == 48
    assert all("map_at_r" not in line and "recall" not in line for line in progress)


def test_registered_evaluation_stops_without_partial_result_on_row_failure() -> None:
    module = _load_module()
    inventory, rows = _synthetic_inventory_and_rows()
    calls = 0

    def evaluate(_row: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 7:
            raise RuntimeError("row failed")
        return copy.deepcopy(rows[calls - 1])

    with pytest.raises(RuntimeError, match="row failed"):
        module.evaluate_registered_rows(
            inventory,
            query_count=2,
            evaluate=evaluate,
            emit_progress=lambda _line: None,
        )
    assert calls == 7


def test_cli_refuses_nonisolated_process_before_opening_config(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--config", str(missing)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "isolated" in completed.stderr
    assert not missing.exists()


def test_isolated_process_activates_only_the_executing_checkout_source() -> None:
    program = f"""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('official_eval', Path({str(MODULE_PATH)!r}))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.activate_project_source(Path({str(PROJECT)!r}))
import sfora.unicom_inshop
print(Path(sfora.unicom_inshop.__file__).resolve())
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == (
        PROJECT / "src" / "sfora" / "unicom_inshop.py"
    ).resolve()
