from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from sfora import unicom_retrieval_audit

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_unicom_ftra_falsifier.py"
SPEC = importlib.util.spec_from_file_location("evaluate_unicom_ftra_falsifier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ftra = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ftra
SPEC.loader.exec_module(ftra)


def test_deployment_scores_match_full_normalize_then_prefix_distance() -> None:
    query = np.ascontiguousarray([[3.0, 4.0, 12.0], [4.0, 0.0, 3.0]], dtype=np.float32)
    gallery = np.ascontiguousarray(
        [[0.0, 5.0, 12.0], [4.0, 3.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float32
    )

    scores = ftra.deployment_scores(query, gallery, selected_features=2)

    query_prefix = np.asarray([[3 / 13, 4 / 13], [4 / 5, 0.0]], dtype=np.float32).astype(np.float64)
    gallery_prefix = np.asarray(
        [[0.0, 5 / 13], [4 / 5, 3 / 5], [0.0, 0.0]], dtype=np.float32
    ).astype(np.float64)
    expected = -np.sum(
        (query_prefix[:, None, :] - gallery_prefix[None, :, :]) ** 2,
        axis=2,
        dtype=np.float64,
    )
    np.testing.assert_array_equal(scores, expected)
    assert scores.dtype == np.float64
    assert scores.flags.c_contiguous


def test_deployment_scores_match_registered_retrieval_view_chunks(monkeypatch) -> None:
    generator = np.random.Generator(np.random.PCG64(0))
    query = np.ascontiguousarray(generator.normal(size=(513, 768)).astype(np.float32))
    gallery = np.ascontiguousarray(generator.normal(size=(700, 768)).astype(np.float32))
    query_labels = np.asarray(["A"] * query.shape[0])
    gallery_labels = np.asarray(["A"] * gallery.shape[0])
    captured: list[np.ndarray] = []
    original_reducer = unicom_retrieval_audit.retrieval_metrics_from_score_chunks

    def capture(score_chunks, query_values, gallery_values):
        chunks = tuple(score_chunks)
        captured.extend(chunks)
        return original_reducer(chunks, query_values, gallery_values)

    monkeypatch.setattr(unicom_retrieval_audit, "retrieval_metrics_from_score_chunks", capture)

    scores = ftra.deployment_scores(query, gallery, selected_features=512)
    unicom_retrieval_audit.retrieval_view(
        query,
        gallery,
        query_labels,
        gallery_labels,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    expected = np.ascontiguousarray(np.concatenate(captured, axis=0), dtype=np.float64)

    assert ftra.SCREEN_QUERY_CHUNK_SIZE == 256
    assert [chunk.shape[0] for chunk in captured] == [256, 256, 1]
    np.testing.assert_array_equal(scores, expected)


def test_blend_grid_uses_repository_metrics_and_prefers_lower_alpha_on_ties() -> None:
    endpoint = np.ascontiguousarray([[0.5, 0.4, 0.6, 0.3], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    teacher = np.ascontiguousarray([[0.9, 0.8, 0.1, 0.0], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    query_labels = np.asarray(["A", "B"])
    gallery_labels = np.asarray(["A", "A", "B", "B"])

    candidates = ftra.evaluate_blend_grid(
        endpoint,
        teacher,
        query_labels,
        gallery_labels,
        alphas=(0.0, 0.5, 1.0),
    )

    assert [row["alpha"] for row in candidates] == [0.0, 0.5, 1.0]
    assert candidates[0]["metrics"] == {
        "recall_at_1": 0.5,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_30": 1.0,
        "map_at_r": 0.625,
    }
    assert candidates[1]["metrics"]["map_at_r"] == 0.75
    assert candidates[2]["metrics"]["map_at_r"] == 1.0
    assert ftra.select_blend_candidate(candidates)["alpha"] == 1.0


def test_blend_grid_is_invariant_to_each_arms_positive_affine_score_scale() -> None:
    endpoint = np.ascontiguousarray([[0.5, 0.4, 0.6, 0.3], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    teacher = np.ascontiguousarray([[0.9, 0.8, 0.1, 0.0], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    query_labels = np.asarray(["A", "B"])
    gallery_labels = np.asarray(["A", "A", "B", "B"])

    reference = ftra.evaluate_blend_grid(
        endpoint, teacher, query_labels, gallery_labels, alphas=(0.0, 0.5, 1.0)
    )
    rescaled = ftra.evaluate_blend_grid(
        np.ascontiguousarray(endpoint * 100.0 + 17.0),
        np.ascontiguousarray(teacher * 0.01 - 9.0),
        query_labels,
        gallery_labels,
        alphas=(0.0, 0.5, 1.0),
    )

    assert rescaled == reference


def test_blend_grid_rejects_missing_endpoint_alpha() -> None:
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    labels = np.asarray(["A"])

    try:
        ftra.evaluate_blend_grid(scores, scores, labels, labels, alphas=(0.05, 1.0))
    except ValueError as error:
        assert str(error) == "blend alpha grid must begin with the endpoint"
    else:
        raise AssertionError("accepted a blend grid without alpha zero")


def _candidate(map_at_r: float, recall_at_1: float, *, alpha: float = 0.2) -> dict[str, object]:
    return {
        "alpha": alpha,
        "metrics": {"map_at_r": map_at_r, "recall_at_1": recall_at_1},
    }


def _provenance() -> dict[str, object]:
    return {
        "screen_report": "/registered/selection.json",
        "screen_report_sha256": "a" * 64,
        "endpoint_state": "/registered/endpoint.pt",
        "endpoint_state_sha256": "b" * 64,
        "epoch8_checkpoint": "/registered/epoch-0008.pt",
        "epoch8_checkpoint_sha256": "c" * 64,
        "initial_checkpoint_sha256": "d" * 64,
        "batch_norm_recalibration": "full-optimization-cumulative-batches-all-arms",
        "training_protocol": {"protocol": "frozen-training"},
    }


def test_falsifier_gate_passes_only_unique_safe_teacher_gain() -> None:
    endpoint = _candidate(0.900, 0.980)
    teacher = _candidate(0.906, 0.979)
    pure_teacher = _candidate(0.902, 0.978, alpha=1.0)
    weak_control = _candidate(0.902, 0.981)

    gate = ftra.falsifier_gate(
        endpoint,
        teacher,
        pure_teacher,
        weak_control,
        map_bootstrap_interval=(0.001, 0.011),
    )

    assert gate == {
        "minimum_map_gain": 0.003,
        "maximum_control_gain_fraction": 0.7,
        "recall_at_1_guard": -0.00125,
        "winner_is_interior": True,
        "pretrained_map_gain_over_endpoint": 0.006000000000000005,
        "pretrained_map_gain_over_pure_teacher": 0.0040000000000000036,
        "epoch8_map_gain": 0.0020000000000000018,
        "epoch8_gain_fraction": 0.3333333333333333,
        "pretrained_recall_at_1_delta": -0.0010000000000000009,
        "bootstrap_seed": 0,
        "bootstrap_samples": 10_000,
        "pretrained_map_delta_query_bootstrap_95_interval": [0.001, 0.011],
        "interpretation": "exploratory_seed0_train_holdout_falsifier_only",
        "passed": True,
    }


def test_falsifier_gate_rejects_each_registered_boundary_independently() -> None:
    endpoint = _candidate(0.900, 0.980)
    passing_teacher = _candidate(0.906, 0.979)
    passing_pure_teacher = _candidate(0.902, 0.978, alpha=1.0)
    passing_control = _candidate(0.902, 0.981)
    passing_interval = (0.001, 0.011)
    cases = (
        (
            _candidate(0.9029, 0.979),
            _candidate(0.898, 0.978, alpha=1.0),
            _candidate(0.900, 0.980),
            passing_interval,
        ),
        (
            passing_teacher,
            _candidate(0.9031, 0.978, alpha=1.0),
            _candidate(0.900, 0.980),
            passing_interval,
        ),
        (
            _candidate(0.906, 0.979, alpha=1.0),
            passing_pure_teacher,
            passing_control,
            passing_interval,
        ),
        (
            passing_teacher,
            passing_pure_teacher,
            _candidate(0.90421, 0.981),
            passing_interval,
        ),
        (
            _candidate(0.906, 0.9787),
            passing_pure_teacher,
            passing_control,
            passing_interval,
        ),
        (
            passing_teacher,
            passing_pure_teacher,
            passing_control,
            (0.0, 0.011),
        ),
    )

    for teacher, pure_teacher, control, interval in cases:
        gate = ftra.falsifier_gate(
            endpoint,
            teacher,
            pure_teacher,
            control,
            map_bootstrap_interval=interval,
        )
        assert gate["passed"] is False


def test_falsifier_gate_rejects_control_explanation_over_boundary_and_r1_damage() -> None:
    endpoint = _candidate(0.900, 0.980)
    teacher = _candidate(0.904, 0.978)
    pure_teacher = _candidate(0.899, 0.979, alpha=1.0)
    control_boundary = _candidate(0.90281, 0.981)

    gate = ftra.falsifier_gate(
        endpoint,
        teacher,
        pure_teacher,
        control_boundary,
        map_bootstrap_interval=(0.001, 0.008),
    )

    assert gate["epoch8_gain_fraction"] > 0.7
    assert gate["passed"] is False


def test_falsifier_gate_rejects_pure_teacher_winner_and_nonpositive_interval() -> None:
    endpoint = _candidate(0.900, 0.980)
    pure_teacher = _candidate(0.910, 0.981, alpha=1.0)
    control = _candidate(0.900, 0.980)

    pure_gate = ftra.falsifier_gate(
        endpoint,
        pure_teacher,
        pure_teacher,
        control,
        map_bootstrap_interval=(0.005, 0.015),
    )
    uncertain_gate = ftra.falsifier_gate(
        endpoint,
        _candidate(0.905, 0.981),
        _candidate(0.900, 0.980, alpha=1.0),
        control,
        map_bootstrap_interval=(-0.001, 0.011),
    )

    assert pure_gate["winner_is_interior"] is False
    assert pure_gate["passed"] is False
    assert uncertain_gate["passed"] is False


def test_evaluate_falsifier_builds_control_bound_report() -> None:
    endpoint = np.ascontiguousarray([[0.5, 0.4, 0.6, 0.3], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    pretrained = np.ascontiguousarray(
        [[0.9, 0.8, 0.1, 0.0], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64
    )
    epoch8 = np.ascontiguousarray([[0.6, 0.5, 0.55, 0.3], [0.2, 0.1, 0.9, 0.8]], dtype=np.float64)
    query_labels = np.asarray(["A", "B"])
    gallery_labels = np.asarray(["A", "A", "B", "B"])

    report = ftra.evaluate_falsifier(
        endpoint,
        pretrained,
        epoch8,
        query_labels,
        gallery_labels,
        selected_features=512,
        provenance=_provenance(),
    )

    assert tuple(report) == (
        "protocol",
        "alphas",
        "selected_features",
        "score_preprocessing",
        "provenance",
        "endpoint",
        "pretrained",
        "epoch8_control",
        "gate",
    )
    assert report["endpoint"] == report["pretrained"]["candidates"][0]
    assert 0.0 < report["pretrained"]["best"]["alpha"] < 1.0
    assert report["epoch8_control"]["best"]["alpha"] == 0.75
    ftra.validate_falsifier_report(report)


def test_report_validator_recomputes_winner_evidence_and_gate() -> None:
    labels = np.asarray(["A"])
    gallery_labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    report = ftra.evaluate_falsifier(
        scores,
        scores,
        scores,
        labels,
        gallery_labels,
        selected_features=512,
        provenance=_provenance(),
    )
    report["pretrained"]["best"] = report["pretrained"]["candidates"][1]

    try:
        ftra.validate_falsifier_report(report)
    except ValueError as error:
        assert str(error) == "pretrained winner differs"
    else:
        raise AssertionError("validator accepted a noncanonical tied winner")


def test_report_validator_rejects_provenance_and_feature_mutations() -> None:
    labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    report = ftra.evaluate_falsifier(
        scores,
        scores,
        scores,
        labels,
        labels,
        selected_features=512,
        provenance=_provenance(),
    )
    mutations = (
        lambda value: value["provenance"].update(extra=True),
        lambda value: value["provenance"].update(screen_report=""),
        lambda value: value["provenance"].update(screen_report_sha256="A" * 64),
        lambda value: value["provenance"].update(screen_report_sha256="a" * 63),
        lambda value: value["provenance"].update(batch_norm_recalibration="partial"),
        lambda value: value["provenance"].update(training_protocol={}),
        lambda value: value.update(selected_features=256),
    )

    for mutate in mutations:
        changed = json.loads(json.dumps(report))
        mutate(changed)
        try:
            ftra.validate_falsifier_report(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted mutated registered evidence")


def test_report_validator_binds_both_groups_to_the_same_endpoint() -> None:
    labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    report = ftra.evaluate_falsifier(
        scores,
        scores,
        scores,
        labels,
        labels,
        selected_features=512,
        provenance=_provenance(),
    )
    wrong_report_endpoint = json.loads(json.dumps(report))
    wrong_report_endpoint["endpoint"] = wrong_report_endpoint["pretrained"]["candidates"][1]
    wrong_control_endpoint = json.loads(json.dumps(report))
    wrong_control_endpoint["epoch8_control"]["candidates"][0]["metrics"]["recall_at_10"] = 0.9

    for changed in (wrong_report_endpoint, wrong_control_endpoint):
        try:
            ftra.validate_falsifier_report(changed)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted inconsistent endpoint evidence")


def test_candidate_validator_rejects_inconsistent_query_evidence() -> None:
    labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    report = ftra.evaluate_falsifier(
        scores,
        scores,
        scores,
        labels,
        labels,
        selected_features=512,
        provenance=_provenance(),
    )
    top1_changed = json.loads(json.dumps(report["endpoint"]))
    top1_changed["query_evidence"]["top1_correct"][0] = False
    map_changed = json.loads(json.dumps(report["endpoint"]))
    map_changed["metrics"]["map_at_r"] = 0.9

    for candidate in (top1_changed, map_changed):
        try:
            ftra._validate_candidate(candidate)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted inconsistent query evidence")


def test_candidate_validator_uses_producer_numpy_mean_reduction() -> None:
    average_precision = [0.9350724237877682, 0.8158535541215322, 0.002738500170148095]
    candidate = {
        "alpha": 0.0,
        "metrics": {
            "recall_at_1": 2 / 3,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "recall_at_30": 1.0,
            "map_at_r": float(np.mean(np.asarray(average_precision, dtype=np.float64))),
        },
        "query_evidence": {
            "top1_correct": [True, True, False],
            "average_precision": average_precision,
        },
    }

    assert ftra._validate_candidate(candidate) == 3


def test_atomic_report_publication_reloads_and_never_clobbers(tmp_path: Path) -> None:
    labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    report = ftra.evaluate_falsifier(
        scores,
        scores,
        scores,
        labels,
        labels,
        selected_features=512,
        provenance=_provenance(),
    )
    output = tmp_path / "falsifier.json"

    ftra.write_report_atomic(report, output)

    assert json.loads(output.read_text()) == report
    original = output.read_bytes()
    try:
        ftra.write_report_atomic(report, output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("writer clobbered an existing report")
    assert output.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def _screen_binding() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    candidates = []
    for epochs in ([16], [12, 16], [8, 12, 16], [4, 8, 12, 16]):
        for alpha in (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0):
            winner = epochs == [16] and alpha == 1.0
            candidates.append(
                {
                    "name": f"epochs-{'_'.join(map(str, epochs))}-alpha-{alpha}",
                    "epochs": epochs,
                    "checkpoints": [f"/registered/epoch-{epoch:04d}.pt" for epoch in epochs],
                    "alpha": alpha,
                    "metrics": {
                        "recall_at_1": 1.0 if winner else 0.0,
                        "recall_at_10": 1.0,
                        "recall_at_20": 1.0,
                        "recall_at_30": 1.0,
                        "map_at_r": 1.0 if winner else 0.5,
                    },
                    "query_evidence": {
                        "top1_correct": [winner],
                        "average_precision": [1.0 if winner else 0.5],
                    },
                }
            )
    candidate = candidates[6]
    training_protocol = {"protocol": "frozen-training"}
    screen = {
        "protocol": "unicom-train-identity-holdout-suffix-soup-wise-v3",
        "mode": "screen",
        "training_seed": 0,
        "alphas": [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0],
        "batch_size": 128,
        "workers": 4,
        "selected_features": 512,
        "screen_report": None,
        "screen_report_sha256": None,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "training_protocol": training_protocol,
        "batch_norm_recalibration": "full-optimization-cumulative-batches",
        "selected": candidate,
        "endpoint": candidate,
        "gate": {"promoted": False},
        "candidates": candidates,
        "model_path": "/registered/selected-model.pt",
    }
    endpoint = {
        "model": {"weight": "endpoint"},
        "endpoint": candidate,
        "training_protocol": training_protocol,
    }
    epoch8 = {
        "epoch": 8,
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": training_protocol,
        "model": {"weight": "epoch8"},
    }
    return screen, endpoint, epoch8


def test_screen_binding_requires_canonical_failed_endpoint_screen() -> None:
    screen, endpoint, epoch8 = _screen_binding()
    ftra.validate_screen_binding(screen, endpoint, epoch8)

    screen["gate"] = {"promoted": True}
    try:
        ftra.validate_screen_binding(screen, endpoint, epoch8)
    except ValueError as error:
        assert str(error) == "screen did not close soup selection"
    else:
        raise AssertionError("accepted a promoted soup screen")


def test_screen_binding_requires_the_registered_epoch16_training_endpoint() -> None:
    screen, endpoint_payload, epoch8 = _screen_binding()
    old_endpoint = screen["endpoint"]
    old_endpoint["metrics"]["map_at_r"] = 0.5
    old_endpoint["metrics"]["recall_at_1"] = 0.5
    replacement = screen["candidates"][13]
    replacement["metrics"]["map_at_r"] = 0.9
    replacement["metrics"]["recall_at_1"] = 0.9
    screen["selected"] = replacement
    screen["endpoint"] = replacement
    endpoint_payload["endpoint"] = replacement

    try:
        ftra.validate_screen_binding(screen, endpoint_payload, epoch8)
    except ValueError as error:
        assert str(error) == "screen endpoint closure differs"
    else:
        raise AssertionError("accepted a non-epoch16 training endpoint")

    screen, endpoint_payload, epoch8 = _screen_binding()
    old_endpoint = screen["endpoint"]
    old_endpoint["metrics"]["map_at_r"] = 0.5
    old_endpoint["metrics"]["recall_at_1"] = 0.5
    replacement = screen["candidates"][5]
    replacement["metrics"]["map_at_r"] = 0.9
    replacement["metrics"]["recall_at_1"] = 0.9
    screen["selected"] = replacement
    screen["endpoint"] = replacement
    endpoint_payload["endpoint"] = replacement

    try:
        ftra.validate_screen_binding(screen, endpoint_payload, epoch8)
    except ValueError as error:
        assert str(error) == "screen endpoint closure differs"
    else:
        raise AssertionError("accepted a non-alpha1 training endpoint")


def test_screen_binding_rejects_truncated_grid_and_unregistered_epoch8_path() -> None:
    screen, endpoint, epoch8 = _screen_binding()
    screen["candidates"].pop()
    try:
        ftra.validate_screen_binding(screen, endpoint, epoch8)
    except ValueError as error:
        assert str(error) == "screen candidate grid differs"
    else:
        raise AssertionError("accepted a truncated soup screen")

    screen, endpoint, epoch8 = _screen_binding()
    try:
        ftra.validate_epoch8_path(screen, Path("/registered/not-epoch8.pt"))
    except ValueError as error:
        assert str(error) == "epoch-8 control path is absent from screen"
    else:
        raise AssertionError("accepted an unregistered epoch-8 path")


def test_endpoint_reproduction_must_match_screen_metrics() -> None:
    screen, _endpoint, _epoch8 = _screen_binding()
    reproduced = dict(screen["endpoint"])
    reproduced["metrics"] = dict(reproduced["metrics"])
    ftra.validate_endpoint_reproduction(screen, reproduced)

    reproduced["metrics"]["map_at_r"] -= 0.001
    try:
        ftra.validate_endpoint_reproduction(screen, reproduced)
    except ValueError as error:
        assert str(error) == "endpoint metrics do not reproduce the soup screen"
    else:
        raise AssertionError("accepted a mismatched endpoint reproduction")


def test_each_model_arm_is_recalibrated_before_encoding() -> None:
    events: list[object] = []

    class Model:
        def load_state_dict(self, state, *, strict):
            events.append(("load", state, strict))

    class Soup:
        @staticmethod
        def recalibrate_batch_norm(model, loader, *, device):
            events.append(("recalibrate", model, loader, device))

    def encode():
        events.append("encode")
        return "scores"

    model = Model()
    assert (
        ftra.encode_recalibrated_arm(Soup(), model, None, "loader", device="cuda", encode=encode)
        == "scores"
    )
    assert (
        ftra.encode_recalibrated_arm(
            Soup(), model, {"weight": 1}, "loader", device="cuda", encode=encode
        )
        == "scores"
    )
    assert events == [
        ("recalibrate", model, "loader", "cuda"),
        "encode",
        ("load", {"weight": 1}, True),
        ("recalibrate", model, "loader", "cuda"),
        "encode",
    ]


def test_main_runs_one_encoded_falsifier_and_refuses_second_publication(
    tmp_path: Path, monkeypatch
) -> None:
    labels = np.asarray(["A"])
    scores = np.ascontiguousarray([[1.0]], dtype=np.float64)
    calls = 0

    def load(_args):
        nonlocal calls
        calls += 1
        return scores, scores, scores, labels, labels, _provenance()

    monkeypatch.setattr(ftra, "load_registered_scores", load)
    output = tmp_path / "falsifier.json"
    arguments = [
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--initial-checkpoint",
        str(tmp_path / "initial.pt"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--screen-report",
        str(tmp_path / "selection.json"),
        "--endpoint-state",
        str(tmp_path / "endpoint.pt"),
        "--epoch8-checkpoint",
        str(tmp_path / "epoch-0008.pt"),
        "--output",
        str(output),
    ]

    assert ftra.main(arguments) == 1
    assert calls == 1
    ftra.validate_falsifier_report(json.loads(output.read_text()))
    assert ftra.main(arguments) == 2
    assert calls == 1
