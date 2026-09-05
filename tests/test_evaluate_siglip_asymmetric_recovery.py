"""Asymmetric recovery evaluation is isolated from the frozen paired evaluator."""

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sfora.data import ImageExample

SPEC = importlib.util.spec_from_file_location(
    "evaluate_siglip_asymmetric_recovery",
    Path(__file__).parents[1] / "scripts/evaluate_siglip_asymmetric_recovery.py",
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _cell(vectors: torch.Tensor, labels: tuple[int, ...]) -> dict[str, object]:
    return subject.recovery._retrieval_cell(vectors, labels)


def test_cross_cells_reproduce_frozen_controls_before_cross_model_metrics() -> None:
    labels = (49, 49, 50, 50)
    ids = ("a", "b", "c", "d")
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    pa = teacher.clone()
    relational = teacher.clone()
    descriptors = {"teacher": teacher, "pa": pa, "relational": relational}
    prior = {"cells": {name: _cell(value, labels) for name, value in descriptors.items()}}

    result = subject.cross_cells(descriptors, labels=labels, example_ids=ids, prior=prior)

    assert set(result) == {"pa", "relational"}
    assert result["pa"]["map_at_r"] == 1.0
    assert result["pa"]["retrieval"]["nearest_ordinals"] == (1, 0, 3, 2)

    drifted = copy.deepcopy(prior)
    drifted["cells"]["pa"]["retrieval"]["nearest_ordinals"] = [2, 0, 3, 2]
    with pytest.raises(ValueError, match="control"):
        subject.cross_cells(descriptors, labels=labels, example_ids=ids, prior=drifted)


def test_prior_result_is_exactly_bound_to_recovery_authority(tmp_path: Path) -> None:
    value = {
        "schema": "sfora-siglip-depth-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "pair_sha256": "a" * 64,
        "monitor_sha256": "b" * 64,
        "cells": {name: {"queries": 2746} for name in ("teacher", "pa", "relational")},
        "resources": {"elapsed_seconds": 400.0},
        "source_sha256": {
            "runner": subject.probe._file_sha(Path(str(subject.recovery.__file__))),
            "retrieval_core": subject.probe._file_sha(Path(str(subject.retrieval_core.__file__))),
        },
    }
    path = tmp_path / "prior.json"
    path.write_bytes(subject.recovery.probe._canonical(value))
    digest = subject.recovery.probe._file_sha(path)
    assert subject.read_prior_result(path, digest, "a" * 64, "b" * 64) == value
    for mutate in (
        lambda x: x.update(claim_eligible=True),
        lambda x: x.update(pair_sha256="c" * 64),
        lambda x: x["cells"].pop("pa"),
        lambda x: x["source_sha256"].update(retrieval_core="d" * 64),
    ):
        bad = copy.deepcopy(value)
        mutate(bad)
        path.write_bytes(subject.recovery.probe._canonical(bad))
        bad_digest = subject.recovery.probe._file_sha(path)
        with pytest.raises(ValueError):
            subject.read_prior_result(path, bad_digest, "a" * 64, "b" * 64)


def test_prior_evaluation_time_is_deducted_from_both_caps() -> None:
    prior = {"elapsed_s": 400.0}
    assert subject.asymmetric_budget_seconds(2000.0, prior) == 1600.0
    assert subject.asymmetric_budget_seconds(5000.0, prior) == 1800.0
    for elapsed in (True, 0.0, float("nan"), 2000.0):
        with pytest.raises(ValueError):
            subject.asymmetric_budget_seconds(2000.0, {"elapsed_s": elapsed})


def test_prior_monitor_binds_successful_whole_process_time(tmp_path: Path) -> None:
    value = {
        "schema": "sfora-recovery-evaluation-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "elapsed_s": 475.06047469202895,
        "result_sha256": subject.PRIOR_EVALUATION_SHA256,
    }
    path = tmp_path / "prior.monitor.json"
    path.write_bytes(subject.probe._canonical(value))
    digest = subject.probe._file_sha(path)
    assert subject.read_prior_monitor(path, digest)["elapsed_s"] == value["elapsed_s"]
    for key, changed in (("exit_code", 125), ("stop_reason", "psi-cap"), ("elapsed_s", 0.0)):
        bad = {**value, key: changed}
        path.write_bytes(subject.probe._canonical(bad))
        with pytest.raises(ValueError):
            subject.read_prior_monitor(path, subject.probe._file_sha(path))


def test_direct_cli_requires_prior_authority_and_explicit_execution(tmp_path: Path) -> None:
    output = tmp_path / "cross.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.main(
            [
                "--pair-directory",
                str(tmp_path),
                "--pair-sha256",
                "a" * 64,
                "--smoke-result",
                str(tmp_path / "smoke.json"),
                "--audit-result",
                str(tmp_path / "audit.json"),
                "--pair-monitor",
                str(tmp_path / "monitor.json"),
                "--monitor-sha256",
                "b" * 64,
                "--prior-evaluation",
                str(tmp_path / "prior.json"),
                "--prior-evaluation-sha256",
                "c" * 64,
                "--prior-evaluation-monitor",
                str(tmp_path / "prior.monitor.json"),
                "--prior-evaluation-monitor-sha256",
                "d" * 64,
                "--control-root",
                str(tmp_path / "control"),
                "--output",
                str(output),
                "--execute-asymmetric-evaluation",
            ]
        )
    assert not output.exists()
    with pytest.raises(SystemExit):
        subject.parse_args(["--updates", "1"])


def test_runner_authenticates_then_extracts_teacher_and_both_students_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = {
        "teacher_state_sha256": "f" * 64,
        "teacher_checkpoint_sha256": "e" * 64,
        "checkpoints": {name: {"sha256": name * 8} for name in ("pa", "relational")},
        "arms": {
            "pa": {"final_state_sha256": "a" * 64},
            "relational": {"final_state_sha256": "b" * 64},
        },
    }
    audit = {
        "common_image_ids": [f"id-{i}" for i in range(2746)],
        "common_to_native": list(range(2746)),
        "decoded_native_sha256": "d" * 64,
    }
    args = SimpleNamespace(
        pair_directory=tmp_path,
        pair_sha256="1" * 64,
        smoke_result=tmp_path / "smoke.json",
        audit_result=tmp_path / "audit.json",
        pair_monitor=tmp_path / "monitor.json",
        monitor_sha256="2" * 64,
        prior_evaluation=tmp_path / "prior.json",
        prior_evaluation_sha256=subject.PRIOR_EVALUATION_SHA256,
        prior_evaluation_monitor=tmp_path / "prior.monitor.json",
        prior_evaluation_monitor_sha256=subject.PRIOR_EVALUATION_MONITOR_SHA256,
        control_root=tmp_path,
    )
    monkeypatch.setattr(
        subject.recovery,
        "_read_json",
        lambda path, digest: audit if path == args.audit_result else receipt,
    )
    monkeypatch.setattr(subject.recovery.pair, "read_smoke_authority", lambda *args: {})
    monkeypatch.setattr(subject.recovery, "validate_pair_receipt", lambda *args: None)
    monkeypatch.setattr(subject.recovery, "verify_inference_dependencies", lambda *args: None)
    monkeypatch.setattr(subject.recovery, "evaluation_budget_seconds", lambda *args: 1700.0)
    monkeypatch.setattr(
        subject,
        "read_prior_result",
        lambda *args: {"cells": {}, "resources": {"elapsed_seconds": 100.0}},
    )
    monkeypatch.setattr(subject, "read_prior_monitor", lambda *args: {"elapsed_s": 101.0})
    monkeypatch.setattr(
        subject.recovery,
        "authenticate_checkpoint_files",
        lambda *args: {
            "pa": tmp_path / "pa-final.pt",
            "relational": tmp_path / "relational-final.pt",
        },
    )
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: {"model_state": {}})
    monkeypatch.setattr(subject.recovery, "validate_student_payload", lambda *args: None)
    monkeypatch.setattr(subject.recovery, "evaluation_device", lambda: torch.device("cpu"))

    class Model:
        name = "teacher"

        def to(self, device: torch.device) -> "Model":
            return self

        def eval(self) -> "Model":
            return self

        def requires_grad_(self, value: bool) -> "Model":
            return self

        def load_state_dict(self, state: dict[str, Any], strict: bool) -> None:
            assert strict and state == {}

    teacher = Model()
    monkeypatch.setattr(
        subject.recovery, "load_teacher_and_processor", lambda root: (teacher, None)
    )
    monkeypatch.setattr(
        subject.recovery.control,
        "_model_state_sha256",
        lambda model: (
            receipt["teacher_state_sha256"]
            if model.name == "teacher"
            else receipt["arms"][model.name]["final_state_sha256"]
        ),
    )
    examples = tuple(ImageExample(f"id-{i}", object(), 49 + i % 33) for i in range(2746))
    monkeypatch.setattr(
        subject.evaluation_core, "load_recovery_evaluation_images", lambda: examples
    )
    monkeypatch.setattr(subject.recovery, "decoded_native_digest", lambda *args: "d" * 64)
    extracted: list[str] = []
    vectors = torch.zeros(2746, 512)
    vectors[:, 0] = 1

    def embed(model: Model, *args: Any) -> torch.Tensor:
        extracted.append(model.name)
        return vectors.clone()

    monkeypatch.setattr(subject.recovery, "embed_recovery_model", embed)

    def prune(teacher: Model) -> Model:
        name = "pa" if extracted == ["teacher"] else "relational"
        return type("Student", (Model,), {"name": name})()

    monkeypatch.setattr(subject, "prune_siglip_student", prune)
    observed: dict[str, Any] = {}

    def cells(descriptors: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, Any]:
        observed.update(descriptors)
        return {
            name: {"queries": 2746, "map_at_r": 0.7, "correct": 2000, "retrieval": {}}
            for name in ("pa", "relational")
        }

    monkeypatch.setattr(subject, "cross_cells", cells)
    result = subject.run_evaluation(args)
    assert extracted == ["teacher", "pa", "relational"]
    assert set(observed) == {"teacher", "pa", "relational"}
    assert result["decision"]["selected_arm"] == "pa"
    assert result["claim_eligible"] is False
    assert "retrieval_core" in result["source_sha256"]
