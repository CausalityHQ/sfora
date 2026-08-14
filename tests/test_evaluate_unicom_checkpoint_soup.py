from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

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


def test_selected_model_payload_carries_training_provenance() -> None:
    module = _load_script()
    state = OrderedDict(weight=torch.tensor([1.0]))
    selected = {"epochs": [12, 16], "alpha": 0.9}
    training_protocol = {
        "initial_checkpoint_sha256": module.REGISTERED_INITIAL_CHECKPOINT_SHA256,
    }

    payload = module.selected_model_payload(state, selected, training_protocol)

    assert tuple(payload) == ("model", "selection", "training_protocol")
    assert payload["model"] is state
    assert payload["selection"] is selected
    assert payload["training_protocol"] is training_protocol


def test_endpoint_model_payload_carries_recalibrated_candidate_provenance() -> None:
    module = _load_script()
    state = OrderedDict(weight=torch.tensor([1.0]))
    endpoint = {"epochs": [16], "alpha": 1.0}
    training_protocol = {
        "initial_checkpoint_sha256": module.REGISTERED_INITIAL_CHECKPOINT_SHA256,
    }

    payload = module.endpoint_model_payload(state, endpoint, training_protocol)

    assert tuple(payload) == ("model", "endpoint", "training_protocol")
    assert payload["model"] is state
    assert payload["endpoint"] is endpoint
    assert payload["training_protocol"] is training_protocol


def test_candidate_selection_is_map_first_then_recall_and_stable_on_ties() -> None:
    module = _load_script()
    candidates = [
        {"name": "first", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.7}},
        {"name": "map-win", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.71}},
        {"name": "recall-win", "metrics": {"recall_at_1": 0.81, "map_at_r": 0.6}},
        {"name": "same-map", "metrics": {"recall_at_1": 0.8, "map_at_r": 0.71}},
    ]

    assert module.select_candidate(candidates)["name"] == "map-win"


@pytest.mark.parametrize("alphas", [(0.5, 0.5), (1.0, 0.5)])
def test_candidate_grid_rejects_duplicate_or_unsorted_alphas(alphas) -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    state = OrderedDict(weight=torch.tensor([[1.0]]))

    with pytest.raises(ValueError, match="unique increasing"):
        module.evaluate_grid(
            model,
            state,
            ((Path("epoch-0001.pt"), state),),
            alphas=alphas,
            evaluate=lambda: {"recall_at_1": 1.0, "map_at_r": 1.0},
        )


def test_cli_requires_an_explicit_alpha_grid() -> None:
    module = _load_script()

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--unicom-checkout",
                "/tmp/unicom",
                "--initial-checkpoint",
                "/tmp/model.pt",
                "--dataset-root",
                "/tmp/inshop",
                "--output-dir",
                "/tmp/output",
                "--trajectory-checkpoints",
                "/tmp/epoch-0004.pt",
            ]
        )


def test_screen_protocol_rejects_every_trajectory_drift() -> None:
    module = _load_script()
    args = SimpleNamespace(
        batch_size=64,
        workers=0,
        holdout_seed=5,
        holdout_fraction=0.3,
        selected_features=768,
        training_seed=0,
        dataset_root=Path("/tmp/inshop"),
    )
    trainer = SimpleNamespace(
        UNICOM_REVISION="official-revision",
        UNICOM_L14_336_SHA256="checkpoint-sha",
        __file__="/tmp/train_unicom_inshop.py",
        _sha256_file=lambda path: {
            Path("/tmp/train_unicom_inshop.py"): "trainer-sha",
            Path("/tmp/inshop/Eval/list_eval_partition.txt"): "partition-sha",
        }[Path(path)],
    )
    expected = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "trainer-sha",
        "unicom_revision": "official-revision",
        "initial_checkpoint_sha256": "checkpoint-sha",
        "partition_sha256": "partition-sha",
        "seed": 0,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }

    assert module._expected_screen_training_protocol(trainer, args) == expected
    module._validate_screen_training_protocol(expected, trainer, args)
    for key in expected:
        drifted = dict(expected)
        drifted[key] = object()
        with pytest.raises(ValueError, match="protocol differs"):
            module._validate_screen_training_protocol(drifted, trainer, args)


def test_selection_modes_freeze_screen_and_bind_replication_to_winner(tmp_path: Path) -> None:
    module = _load_script()
    paths = tuple(Path(f"epoch-{epoch:04d}.pt") for epoch in (4, 8, 12, 16))
    selected = {
        "name": "epochs-12_16-alpha-0.9",
        "epochs": [12, 16],
        "checkpoints": [str(path) for path in paths[2:]],
        "alpha": 0.9,
        "metrics": {"recall_at_1": 0.8, "map_at_r": 0.7},
    }
    screen_report = tmp_path / "selection.json"
    screen_report.write_text(
        json.dumps(
            {
                "protocol": "unicom-train-identity-holdout-suffix-soup-wise-v3",
                "mode": "screen",
                "training_seed": 0,
                "alphas": list(module.REGISTERED_SCREEN_ALPHAS),
                "batch_size": 128,
                "workers": 4,
                "selected_features": 512,
                "holdout_seed": 0,
                "holdout_fraction": 0.2,
                "selected": selected,
                "candidates": [selected],
                "gate": {"promoted": True},
            }
        )
    )
    screen = SimpleNamespace(
        mode="screen",
        training_seed=0,
        trajectory_checkpoints=paths,
        alphas=list(module.REGISTERED_SCREEN_ALPHAS),
        holdout_seed=0,
        holdout_fraction=0.2,
        batch_size=128,
        workers=4,
        selected_features=512,
        screen_report=None,
    )
    fixed = SimpleNamespace(
        mode="fixed",
        training_seed=1,
        trajectory_checkpoints=paths,
        alphas=[0.9],
        holdout_seed=0,
        holdout_fraction=0.2,
        batch_size=128,
        workers=4,
        selected_features=512,
        screen_report=screen_report,
    )

    module._validate_selection_args(screen)
    module._validate_selection_args(fixed)
    for seed in (2, 3, 4, 5, 6):
        replication = SimpleNamespace(**vars(fixed))
        replication.training_seed = seed
        module._validate_selection_args(replication)
    out_of_protocol = SimpleNamespace(**vars(fixed))
    out_of_protocol.training_seed = 7
    with pytest.raises(ValueError, match="frozen protocol"):
        module._validate_selection_args(out_of_protocol)
    for field, value in (
        ("training_seed", 2),
        ("alphas", [0.25, 0.5, 0.75, 1.0]),
        ("holdout_seed", 1),
        ("holdout_fraction", 0.3),
        ("batch_size", 64),
        ("workers", 0),
        ("selected_features", 768),
    ):
        drifted = SimpleNamespace(**vars(screen))
        setattr(drifted, field, value)
        with pytest.raises(ValueError, match="frozen protocol"):
            module._validate_selection_args(drifted)
    wrong_winner = SimpleNamespace(**vars(fixed))
    wrong_winner.alphas = [0.85]
    with pytest.raises(ValueError, match="screen winner"):
        module._validate_selection_args(wrong_winner)


def test_fixed_replication_evaluates_only_the_registered_full_window() -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    initial = OrderedDict(weight=torch.tensor([[0.0]]))
    checkpoints = tuple(
        (Path(f"epoch-{epoch:04d}.pt"), OrderedDict(weight=torch.tensor([[float(epoch)]])))
        for epoch in (12, 16)
    )

    candidates = module.evaluate_grid(
        model,
        initial,
        checkpoints,
        alphas=(0.9,),
        registered_specs=(
            (tuple(path for path, _state in checkpoints), 0.9),
            ((checkpoints[1][0],), 1.0),
        ),
        evaluate=lambda: {"recall_at_1": 1.0, "map_at_r": 1.0},
    )

    assert [row["epochs"] for row in candidates] == [[12, 16], [16]]


def test_query_evidence_is_persisted_and_drives_the_paired_gate() -> None:
    module = _load_script()
    model = torch.nn.Linear(1, 1, bias=False)
    state = OrderedDict(weight=torch.tensor([[1.0]]))
    evidence = {
        "top1_correct": [True, False, True, False],
        "average_precision": [0.52, 0.4, 0.32, 0.2],
    }
    candidates = module.evaluate_grid(
        model,
        state,
        ((Path("epoch-0016.pt"), state),),
        alphas=(1.0,),
        evaluate=lambda: ({"recall_at_1": 0.5, "map_at_r": 0.36}, evidence),
    )
    endpoint = {
        **candidates[0],
        "query_evidence": {
            "top1_correct": [True, False, True, False],
            "average_precision": [0.5, 0.4, 0.3, 0.2],
        },
    }

    gate = module.paired_candidate_gate(candidates[0], endpoint, seed=7, samples=100)

    assert candidates[0]["query_evidence"] == evidence
    assert gate["map_delta"] == pytest.approx(0.01)
    assert gate["resampling_unit"] == "holdout_query_within_training_seed"
    lower, upper = gate["map_delta_query_bootstrap_95_interval"]
    assert lower < gate["map_delta"] < upper
    assert gate["recall_at_1_delta"] == 0.0
    assert gate["promoted"] is True


@pytest.mark.parametrize(
    ("map_value", "top1_failures", "promoted"),
    [(0.503, 1, True), (0.5029, 0, False), (0.5031, 2, False)],
)
def test_paired_gate_enforces_both_promotion_boundaries(
    map_value: float, top1_failures: int, promoted: bool
) -> None:
    module = _load_script()
    endpoint_top1 = [True] * 800
    selected_top1 = [False] * top1_failures + [True] * (800 - top1_failures)
    endpoint = {
        "query_evidence": {
            "top1_correct": endpoint_top1,
            "average_precision": [0.5] * 800,
        }
    }
    selected = {
        "query_evidence": {
            "top1_correct": selected_top1,
            "average_precision": [map_value] * 800,
        }
    }

    gate = module.paired_candidate_gate(selected, endpoint, samples=100)

    assert gate["promoted"] is promoted


def _candidate(
    *, seed: int, map_delta: float, recall_failures: int, endpoint: bool
) -> dict[str, object]:
    count = 800
    top1 = [True] * count
    if not endpoint and recall_failures > 0:
        top1[:recall_failures] = [False] * recall_failures
    return {
        "name": "epochs-16-alpha-1" if endpoint else "epochs-12_16-alpha-0.9",
        "epochs": [16] if endpoint else [12, 16],
        "checkpoints": (
            [f"seed-{seed}/epoch-0016.pt"]
            if endpoint
            else [f"seed-{seed}/epoch-0012.pt", f"seed-{seed}/epoch-0016.pt"]
        ),
        "alpha": 1.0 if endpoint else 0.9,
        "metrics": {
            "recall_at_1": sum(top1) / count,
            "map_at_r": 0.5 if endpoint else 0.5 + map_delta,
        },
        "query_evidence": {
            "top1_correct": top1,
            "average_precision": [0.5 if endpoint else 0.5 + map_delta] * count,
        },
    }


def _training_protocol(seed: int) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1",
        "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "initial_checkpoint_sha256": (
            "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
        ),
        "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }


def _screen_report() -> dict[str, object]:
    candidates = []
    for epochs in ([16], [12, 16], [8, 12, 16], [4, 8, 12, 16]):
        for alpha in (0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0):
            is_selected = epochs == [12, 16] and alpha == 0.9
            candidate = _candidate(
                seed=0,
                map_delta=0.01 if is_selected else 0.0,
                recall_failures=0,
                endpoint=epochs == [16] and alpha == 1.0,
            )
            candidate["name"] = f"epochs-{'_'.join(str(epoch) for epoch in epochs)}-alpha-{alpha:g}"
            candidate["epochs"] = epochs
            candidate["checkpoints"] = [f"seed-0/epoch-{epoch:04d}.pt" for epoch in epochs]
            candidate["alpha"] = float(alpha)
            candidates.append(candidate)
    selected = next(
        candidate
        for candidate in candidates
        if candidate["epochs"] == [12, 16] and candidate["alpha"] == 0.9
    )
    endpoint = next(
        candidate
        for candidate in candidates
        if candidate["epochs"] == [16] and candidate["alpha"] == 1.0
    )
    return {
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
        "training_protocol": _training_protocol(0),
        "selected": selected,
        "endpoint": endpoint,
        "gate": {
            "resampling_unit": "holdout_query_within_training_seed",
            "map_delta": 0.01,
            "recall_at_1_delta": 0.0,
            "promoted": True,
        },
        "candidates": candidates,
    }


def _replication_report(seed: int, map_delta: float, recall_failures: int = 0) -> dict[str, object]:
    selected = _candidate(
        seed=seed,
        map_delta=map_delta,
        recall_failures=recall_failures,
        endpoint=False,
    )
    endpoint = _candidate(seed=seed, map_delta=0.0, recall_failures=0, endpoint=True)
    recall_delta = -recall_failures / 800
    return {
        "protocol": "unicom-train-identity-holdout-suffix-soup-wise-v3",
        "mode": "fixed",
        "training_seed": seed,
        "alphas": [0.9],
        "batch_size": 128,
        "workers": 4,
        "selected_features": 512,
        "screen_report": "seed-0/selection.json",
        "screen_report_sha256": "a" * 64,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "training_protocol": _training_protocol(seed),
        "selected": selected,
        "endpoint": endpoint,
        "gate": {
            "resampling_unit": "holdout_query_within_training_seed",
            "map_delta": map_delta,
            "recall_at_1_delta": recall_delta,
            "promoted": map_delta >= 0.003 and recall_delta >= -0.00125,
        },
        "candidates": [selected, endpoint],
    }


def _write_replication_bundle(
    tmp_path: Path,
    reports: list[dict[str, object]],
    *,
    screen: dict[str, object] | None = None,
) -> tuple[Path, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    screen_path = tmp_path / "seed-0-selection.json"
    screen_path.write_text(json.dumps(_screen_report() if screen is None else screen) + "\n")
    screen_sha256 = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    report_paths = []
    for report in reports:
        checkpoint_paths: dict[str, str] = {}
        for candidate in (report["selected"], report["endpoint"]):
            candidate_checkpoints = []
            for checkpoint in candidate["checkpoints"]:
                if checkpoint not in checkpoint_paths:
                    checkpoint_path = tmp_path / checkpoint
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path.write_bytes(
                        f"seed={report['training_seed']};name={checkpoint_path.name}\n".encode()
                    )
                    checkpoint_paths[checkpoint] = str(checkpoint_path)
                candidate_checkpoints.append(checkpoint_paths[checkpoint])
            candidate["checkpoints"] = candidate_checkpoints
        report["screen_report"] = str(screen_path)
        report["screen_report_sha256"] = screen_sha256
        report_path = tmp_path / f"seed-{report['training_seed']}-selection.json"
        report_path.write_text(json.dumps(report) + "\n")
        report_paths.append(report_path)
    return screen_path, report_paths


def test_training_seed_summary_uses_six_paired_runs_and_exact_sign_test(tmp_path: Path) -> None:
    module = _load_script()
    reports = [
        _replication_report(seed, delta, recall_failures)
        for seed, delta, recall_failures in zip(
            range(1, 7),
            (0.010, 0.011, 0.012, 0.013, 0.014, 0.015),
            (0, 0, 0, 0, 1, 0),
            strict=True,
        )
    ]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)

    summary = module.summarize_training_seed_replications(screen_path, report_paths)

    assert summary["training_seeds"] == [1, 2, 3, 4, 5, 6]
    assert summary["resampling_unit"] == "paired_training_seed"
    assert summary["winner"] == {"epochs": [12, 16], "alpha": 0.9}
    assert summary["map_delta_mean"] == pytest.approx(0.0125)
    assert summary["student_t_degrees_of_freedom"] == 5
    assert summary["student_t_critical_two_sided_95"] == pytest.approx(2.5705818356363146)
    lower, upper = summary["map_delta_paired_student_t_95_interval"]
    assert lower > 0.0
    assert lower < summary["map_delta_mean"] < upper
    assert summary["exact_two_sided_sign_p_value"] == 0.03125
    assert summary["all_map_deltas_positive"] is True
    assert summary["all_recall_at_1_deltas_above_guard"] is True
    assert summary["nondegenerate_training_seed_variation"] is True
    assert summary["claim_supported"] is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda reports: reports.pop(), "training seeds"),
        (lambda reports: reports[5].update(training_seed=5), "training seeds"),
        (
            lambda reports: reports[3]["selected"].update(alpha=0.85),
            "winner",
        ),
        (
            lambda reports: reports[2]["gate"].update(resampling_unit="training_seed"),
            "query evidence",
        ),
        (lambda reports: reports[0].update(alphas=[0.85]), "evaluation protocol"),
        (lambda reports: reports[1].update(batch_size=64), "evaluation protocol"),
        (lambda reports: reports[2].update(workers=0), "evaluation protocol"),
        (lambda reports: reports[3].update(selected_features=256), "evaluation protocol"),
        (lambda reports: reports[4].update(holdout_seed=1), "evaluation protocol"),
        (lambda reports: reports[5].update(holdout_fraction=0.5), "evaluation protocol"),
    ],
)
def test_training_seed_summary_rejects_replication_drift(
    tmp_path: Path, mutator, message: str
) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    mutator(reports)
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)

    with pytest.raises(ValueError, match=message):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_training_seed_summary_recomputes_evidence_and_binds_screen(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    reports[2]["gate"]["map_delta"] = 0.02
    screen_path, report_paths = _write_replication_bundle(tmp_path / "delta", reports)
    with pytest.raises(ValueError, match="recomputed"):
        module.summarize_training_seed_replications(screen_path, report_paths)

    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    for candidate in (reports[3]["selected"], reports[3]["endpoint"]):
        candidate["query_evidence"]["top1_correct"].pop()
        candidate["query_evidence"]["average_precision"].pop()
    screen_path, report_paths = _write_replication_bundle(tmp_path / "query-count", reports)
    with pytest.raises(ValueError, match="query count"):
        module.summarize_training_seed_replications(screen_path, report_paths)

    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path / "screen", reports)
    report = json.loads(report_paths[4].read_text())
    report["screen_report_sha256"] = "c" * 64
    report_paths[4].write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="screen report"):
        module.summarize_training_seed_replications(screen_path, report_paths)

    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    reports[1]["training_protocol"]["trainer_sha256"] = "d" * 64
    screen_path, report_paths = _write_replication_bundle(tmp_path / "protocol", reports)
    with pytest.raises(ValueError, match="training protocol"):
        module.summarize_training_seed_replications(screen_path, report_paths)


@pytest.mark.parametrize(
    "mutation",
    [
        {"protocol": "wrong"},
        {"trainer_sha256": "a" * 64},
        {"unicom_revision": "0" * 40},
        {"initial_checkpoint_sha256": "a" * 64},
        {"partition_sha256": "a" * 64},
        {"epochs": 12},
        {"extra": True},
    ],
)
def test_training_seed_summary_rejects_coordinated_training_protocol_drift(
    tmp_path: Path, mutation: dict[str, object]
) -> None:
    module = _load_script()
    screen = _screen_report()
    screen["training_protocol"].update(mutation)
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    for report in reports:
        report["training_protocol"].update(mutation)
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports, screen=screen)

    with pytest.raises(ValueError, match="training protocol"):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_training_seed_summary_rejects_direction_or_recall_guard_failure(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    reports[5] = _replication_report(6, -0.001)
    screen_path, report_paths = _write_replication_bundle(tmp_path / "negative", reports)
    negative = module.summarize_training_seed_replications(screen_path, report_paths)
    assert negative["exact_two_sided_sign_p_value"] == 0.21875
    assert negative["all_map_deltas_positive"] is False
    assert negative["claim_supported"] is False

    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    reports[4] = _replication_report(5, 0.01, 2)
    screen_path, report_paths = _write_replication_bundle(tmp_path / "recall", reports)
    recall_failure = module.summarize_training_seed_replications(screen_path, report_paths)
    assert recall_failure["all_recall_at_1_deltas_above_guard"] is False
    assert recall_failure["claim_supported"] is False


def test_training_seed_summary_rejects_relabelled_checkpoint_evidence(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    reports[1]["selected"]["checkpoints"] = reports[0]["selected"]["checkpoints"]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)

    with pytest.raises(ValueError, match="checkpoint evidence"):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_training_seed_summary_rejects_copied_checkpoint_bytes(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)
    first = json.loads(report_paths[0].read_text())
    second = json.loads(report_paths[1].read_text())
    copied_path = tmp_path / "copied-seed-2" / "epoch-0012.pt"
    copied_path.parent.mkdir()
    copied_path.write_bytes(Path(first["selected"]["checkpoints"][0]).read_bytes())
    second["selected"]["checkpoints"][0] = str(copied_path)
    second["candidates"][0]["checkpoints"][0] = str(copied_path)
    report_paths[1].write_text(json.dumps(second) + "\n")

    with pytest.raises(ValueError, match="checkpoint evidence"):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_training_seed_summary_rejects_checkpoint_epoch_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)
    report = json.loads(report_paths[3].read_text())
    wrong_epoch = tmp_path / "seed-4" / "epoch-0008.pt"
    wrong_epoch.write_bytes(b"distinct epoch-mismatched checkpoint\n")
    report["selected"]["checkpoints"][0] = str(wrong_epoch)
    report["candidates"][0]["checkpoints"][0] = str(wrong_epoch)
    report_paths[3].write_text(json.dumps(report) + "\n")

    with pytest.raises(ValueError, match="checkpoint evidence"):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_training_seed_summary_does_not_claim_zero_seed_variation(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports)

    summary = module.summarize_training_seed_replications(screen_path, report_paths)

    assert summary["map_delta_sample_standard_deviation"] == 0.0
    assert summary["nondegenerate_training_seed_variation"] is False
    assert summary["claim_supported"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda screen: screen.update(protocol="wrong"),
        lambda screen: screen.update(mode="fixed"),
        lambda screen: screen.update(training_seed=1),
        lambda screen: screen.update(screen_report="prior.json"),
        lambda screen: screen.update(screen_report_sha256="a" * 64),
        lambda screen: screen["training_protocol"].update(seed=1),
        lambda screen: screen.update(selected=screen["endpoint"]),
        lambda screen: screen.update(candidates=[screen["endpoint"]]),
        lambda screen: screen["gate"].update(promoted=False),
        lambda screen: screen["gate"].update(resampling_unit="training_seed"),
        lambda screen: screen.update(endpoint=screen["candidates"][0]),
        lambda screen: screen.update(alphas=[0.25, 1.0]),
        lambda screen: screen["candidates"].pop(),
        lambda screen: screen.update(holdout_seed=1),
        lambda screen: screen.update(holdout_fraction=0.1),
        lambda screen: screen.update(selected_features=256),
        lambda screen: screen["training_protocol"].update(protocol="wrong"),
        lambda screen: screen["training_protocol"].update(trainer_sha256="a" * 64),
        lambda screen: screen["training_protocol"].update(unicom_revision="0" * 40),
        lambda screen: screen["training_protocol"].update(initial_checkpoint_sha256="a" * 64),
        lambda screen: screen["training_protocol"].update(partition_sha256="a" * 64),
        lambda screen: screen["training_protocol"].update(epochs=12),
        lambda screen: screen["training_protocol"].update(extra=True),
    ],
)
def test_training_seed_summary_rejects_invalid_screen_report(tmp_path: Path, mutator) -> None:
    module = _load_script()
    screen = _screen_report()
    mutator(screen)
    reports = [_replication_report(seed, 0.01) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path, reports, screen=screen)

    with pytest.raises(ValueError, match="seed-0 screen|training protocol"):
        module.summarize_training_seed_replications(screen_path, report_paths)


def test_aggregate_cli_reads_paths_and_publishes_once(tmp_path: Path) -> None:
    module = _load_script()
    reports = [_replication_report(seed, 0.01 + seed / 1000) for seed in range(1, 7)]
    screen_path, report_paths = _write_replication_bundle(tmp_path / "inputs", reports)
    output = tmp_path / "summary.json"
    arguments = [
        "--aggregate",
        "--screen-report",
        str(screen_path),
        "--replication-reports",
        *(str(path) for path in report_paths),
        "--output",
        str(output),
    ]

    assert module.main(arguments) == 0
    assert json.loads(output.read_text())["claim_supported"] is True
    original = output.read_bytes()
    assert module.main(arguments) == 2
    assert output.read_bytes() == original


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


def test_recalibrated_model_state_uses_candidate_weights_and_returns_updated_bn() -> None:
    module = _load_script()
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.BatchNorm1d(2, momentum=0.1),
    )
    candidate = OrderedDict(
        (name, value.detach().clone()) for name, value in model.state_dict().items()
    )
    candidate["0.weight"] = torch.eye(2)
    candidate["1.running_mean"] = torch.full((2,), 99.0)
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            torch.arange(4),
        ),
        batch_size=2,
    )

    finalized = module.recalibrated_model_state(
        model, candidate, loader, device=torch.device("cpu")
    )

    assert torch.equal(finalized["0.weight"], torch.eye(2))
    assert torch.equal(finalized["1.running_mean"], torch.tensor([4.0, 5.0]))
    assert finalized["1.num_batches_tracked"].item() == 2
    assert all(value.device.type == "cpu" for value in finalized.values())


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
