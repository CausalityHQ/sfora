from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "diagnose_pass201_cis_operator.py"
SPEC = importlib.util.spec_from_file_location("diagnose_pass201_cis_operator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LITERAL_ROWS = [
    {"example_id": "7-a", "sample_index": 70, "label": 7},
    {"example_id": "3-a", "sample_index": 31, "label": 3},
    {"example_id": "7-d", "sample_index": 72, "label": 7},
    {"example_id": "5-a", "sample_index": 50, "label": 5},
]
LITERAL_MANIFEST = LITERAL_ROWS + [
    {"example_id": "7-b", "sample_index": 71, "label": 7},
    {"example_id": "7-c", "sample_index": 73, "label": 7},
    {"example_id": "3-b", "sample_index": 30, "label": 3},
    {"example_id": "5-b", "sample_index": 51, "label": 5},
]

HEX_A = "a" * 64
HEX_B = "b" * 64
OPERATORS = (
    "proxy_anchor",
    "atomic_one_hot",
    "atomic_complementary",
    "atomic_full_union",
    "summed_union",
    "summed_dropout",
)
PANELS = ("network_only", "joint_including_proxies")
REGIMES = ("configured_loss_stateless", "equal_norm")
METRICS = ("R_F", "Delta_M", "D_F", "D_M")
THRESHOLDS = {
    "shared_confuser_excess": 0.010,
    "network_equal_union_advantage_foreign": 0.001,
    "network_equal_union_advantage_margin": 0.001,
    "network_equal_union_foreign_suppression": 0.001,
    "network_equal_union_margin_change": 0.000,
    "network_equal_union_predicted_suppression": 0.001,
    "network_equal_union_predicted_margin_change": 0.000,
    "joint_equal_union_advantage_foreign": 0.000,
    "joint_equal_union_advantage_margin": 0.000,
    "joint_equal_union_foreign_suppression": 0.000,
    "joint_equal_union_margin_change": 0.000,
}

FROZEN_DRAFT_SHA256 = "310f194ee28727caa5908e877338afed82c7ac8be5f2f446affb08f402ef8066"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _literal_tree_digest(root: Path, relative_root: str) -> tuple[int, str]:
    paths = sorted(
        (path for path in (root / relative_root).rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    framed = b"".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode()
        for path in paths
    )
    return len(paths), hashlib.sha256(framed).hexdigest()


def _literal_data_tree_digest(root: Path, relative_root: str) -> tuple[int, int, str]:
    paths = sorted(
        (path for path in (root / relative_root).rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    framed = b"".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode()
        for path in paths
    )
    return (
        len(paths),
        sum(path.stat().st_size for path in paths),
        hashlib.sha256(framed).hexdigest(),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_binding_fixture(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "source"
    (root / "src/sfora").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "docs").mkdir()
    (root / "artifacts").mkdir()
    (root / "dataset/img/a").mkdir(parents=True)
    (root / "src/sfora/core.py").write_text("BOUND = True\n", encoding="utf-8")
    (root / "scripts/launch.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    diagnostic = root / "scripts/diagnostic.py"
    diagnostic.write_bytes(MODULE_PATH.read_bytes())
    partition = root / "dataset/partition.txt"
    partition.write_text("train a/one.jpg 0\ntrain a/two.jpg 0\n", encoding="utf-8")
    (root / "dataset/img/a/one.jpg").write_bytes(b"one")
    (root / "dataset/img/a/two.jpg").write_bytes(b"two")

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Pass201 Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "bound source"], cwd=root, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    source_count, source_tree = _literal_tree_digest(root, "src/sfora")
    data_count, data_bytes, data_tree = _literal_data_tree_digest(root / "dataset", "img")
    argv = [
        ".venv/bin/sfora",
        "image-end-to-end",
        "--dataset-name",
        "inshop",
        "--objectives",
        "proxy_anchor",
        "--seed",
        "0",
    ]
    bind_output_postcondition = (
        "bind_output_hashes_in_separate_activation_manifest_before_any_"
        "pass201_tensor_or_gradient"
    )
    prelaunch = {
        "schema_version": "pass201-pa-source-prelaunch-v1",
        "status": "frozen_before_training",
        "local_source_revision": revision,
        "source": {
            "python_tree_root": "src/sfora",
            "python_file_count": source_count,
            "python_tree_merkle_sha256": source_tree,
            "files": {
                "scripts/launch.sh": {
                    "bytes": (root / "scripts/launch.sh").stat().st_size,
                    "sha256": _sha256_file(root / "scripts/launch.sh"),
                }
            },
        },
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "dataset": {
            "root": str((root / "dataset").resolve()),
            "partition_path": "partition.txt",
            "partition_bytes": partition.stat().st_size,
            "partition_sha256": _sha256_file(partition),
            "partition_line_count": 2,
            "image_root": "img",
            "image_file_count": data_count,
            "image_total_bytes": data_bytes,
            "image_tree_merkle_sha256": data_tree,
        },
        "execution": {
            "working_directory": str(root.resolve()),
            "environment": {"PYTHONPATH": "src"},
            "argv": argv,
        },
        "outputs": {
            "report": "artifacts/report.json",
            "checkpoint": "artifacts/checkpoint.json",
            "all_absent_when_frozen": True,
        },
        "postconditions": {
            "recompute_every_source_and_dataset_digest_before_launch": True,
            "recompute_every_source_and_dataset_digest_after_exit": True,
            "require_report_checkpoint_config_identity": True,
            "require_ordinary_proxy_anchor_seed0_final_training_state_8580_steps": True,
            bind_output_postcondition: True,
        },
    }
    prelaunch_path = root / "docs/pass201_pa_source_prelaunch_manifest.json"
    _write_json(prelaunch_path, prelaunch)

    config = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": 0,
        "batch_size": 180,
        "samples_per_class": 0,
        "hard_class_fraction": 0.0,
        "drop_last_train_batch": True,
        "freeze_batch_norm": False,
        "freeze_batch_norm_affine": False,
        "checkpoint_selection_interval": 0,
        "learning_rate": 0.001,
        "coalition_weight": 0.2,
        "proxy_learning_rate_multiplier": 10.0,
        "proxy_count_per_class": 1,
        "num_workers": 0,
    }
    config_path = root / "artifacts/config.json"
    _write_json(config_path, config)
    train_manifest_path = root / "artifacts/train.json"
    _write_json(
        train_manifest_path,
        {
            "schema_version": "pass201-train-manifest-v1",
            "rows": [
                {"example_id": "a/one.jpg", "sample_index": 0, "label": 0},
                {"example_id": "a/two.jpg", "sample_index": 1, "label": 0},
            ],
        },
    )
    checkpoint_path = root / "artifacts/checkpoint.json"
    checkpoint = {
        "artifact_selection": "final_training_state",
        "evaluation_model_source": "student",
        "training_step": 8580,
        "checkpoint_epoch": 30,
        "objective": "proxy_anchor",
        "seed": 0,
        "training_config": config,
        "resolved_config_sha256": _sha256_file(config_path),
        "train_manifest_sha256": _sha256_file(train_manifest_path),
    }
    _write_json(checkpoint_path, checkpoint)
    report_path = root / "artifacts/report.json"
    report = {
        "config": config,
        "methods": {
            "proxy_anchor_end_to_end:bn_inception": {
                "objective": "proxy_anchor",
                "executed_train_steps": 8580,
            }
        },
        "source_binding": {
            "prelaunch_source_manifest_sha256": _sha256_file(prelaunch_path),
            "source_revision": revision,
            "source_python_tree_sha256_before": source_tree,
            "source_python_tree_sha256_after": source_tree,
            "dataset_image_tree_sha256_before": data_tree,
            "dataset_image_tree_sha256_after": data_tree,
            "argv": argv,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "resolved_config_sha256": _sha256_file(config_path),
            "train_manifest_sha256": _sha256_file(train_manifest_path),
        },
    }
    _write_json(report_path, report)
    return SimpleNamespace(
        root=root,
        git_root=root,
        prelaunch_manifest=prelaunch_path,
        expected_prelaunch_sha256=_sha256_file(prelaunch_path),
        source_report=report_path,
        checkpoint=checkpoint_path,
        resolved_config=config_path,
        train_manifest=train_manifest_path,
        dataset_root=root / "dataset",
        diagnostic_path=diagnostic,
        activated_preregistration=root / "docs/pass201_cis_operator_activated_preregistration.json",
        source_manifest=root / "docs/pass201_cis_operator_source_manifest.json",
    )


def test_source_activation_emits_two_atomic_acyclic_artifacts_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _source_binding_fixture(tmp_path)
    writes: list[Path] = []
    real_write = MODULE._atomic_write_json

    def recording_write(path: Path, payload: object) -> bytes:
        writes.append(path)
        return real_write(path, payload)

    monkeypatch.setattr(MODULE, "_atomic_write_json", recording_write)
    preregistration, manifest = MODULE.activate_source(args)

    assert writes == [args.activated_preregistration, args.source_manifest]
    assert set(preregistration) == {
        "schema_version",
        "frozen_draft_path",
        "frozen_draft_sha256",
        "result_path",
        "source",
        "constants",
        "thresholds",
        "authorized_action",
    }
    assert preregistration["schema_version"] == "pass201-cis-activated-preregistration-v1"
    assert preregistration["frozen_draft_path"] == (
        "docs/pass201_cis_operator_diagnostic_draft_2026-08-09.md"
    )
    assert preregistration["frozen_draft_sha256"] == FROZEN_DRAFT_SHA256
    assert preregistration["result_path"] == (
        "reports/generated/pass201_cis_operator/pass201_inshop_seed0.json"
    )
    assert preregistration["authorized_action"] == (
        "binding_and_integrity_smoke_then_scientific_if_green"
    )
    assert set(manifest) == {
        "schema_version",
        "status",
        "prelaunch_source_manifest_path",
        "prelaunch_source_manifest_sha256",
        "source_report_path",
        "source_report_sha256",
        "source_revision",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_epoch",
        "objective",
        "seed",
        "resolved_config_path",
        "resolved_config_sha256",
        "train_manifest_path",
        "train_manifest_sha256",
        "diagnostic_source_sha256",
        "activated_preregistration_sha256",
        "torch_version",
        "numpy_version",
    }
    assert manifest["schema_version"] == "pass201-source-v1"
    assert manifest["status"] == "frozen"
    assert manifest["activated_preregistration_sha256"] == _sha256_file(
        args.activated_preregistration
    )
    assert all(value is not None for value in manifest.values())
    assert "activated_preregistration_sha256" not in preregistration["source"]
    assert "source_manifest_sha256" not in preregistration["source"]
    assert not list(args.activated_preregistration.parent.glob("*.tmp-*"))
    assert json.loads(args.activated_preregistration.read_text(encoding="utf-8")) == preregistration
    assert json.loads(args.source_manifest.read_text(encoding="utf-8")) == manifest


@pytest.mark.parametrize(
    "mutation",
    [
        "prelaunch_digest",
        "source_tree",
        "source_revision",
        "source_revision_tree",
        "run_command",
        "prelaunch_output_path",
        "dataset_tree",
        "report_checkpoint_digest",
        "checkpoint_config_digest",
        "seed",
        "objective",
        "batch_size",
        "bn_mode",
        "proxy_count",
        "post_source_replay",
        "post_data_replay",
    ],
)
def test_source_binding_mutations_fail_before_torch_model_or_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    args = _source_binding_fixture(tmp_path)
    prelaunch = json.loads(args.prelaunch_manifest.read_text(encoding="utf-8"))
    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    config = json.loads(args.resolved_config.read_text(encoding="utf-8"))
    if mutation == "prelaunch_digest":
        args.expected_prelaunch_sha256 = "0" * 64
    elif mutation == "source_tree":
        (args.root / "src/sfora/core.py").write_text("DRIFT = True\n", encoding="utf-8")
    elif mutation == "source_revision":
        report["source_binding"]["source_revision"] = "0" * 40
        _write_json(args.source_report, report)
    elif mutation == "source_revision_tree":
        source_path = args.root / "src/sfora/core.py"
        original = source_path.read_text(encoding="utf-8")
        source_path.write_text("REVISION_DRIFT = True\n", encoding="utf-8")
        subprocess.run(["git", "add", "src/sfora/core.py"], cwd=args.root, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "revision drift"], cwd=args.root, check=True
        )
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=args.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        source_path.write_text(original, encoding="utf-8")
        prelaunch["local_source_revision"] = revision
        _write_json(args.prelaunch_manifest, prelaunch)
        args.expected_prelaunch_sha256 = _sha256_file(args.prelaunch_manifest)
        report["source_binding"]["prelaunch_source_manifest_sha256"] = (
            args.expected_prelaunch_sha256
        )
        report["source_binding"]["source_revision"] = revision
        _write_json(args.source_report, report)
    elif mutation == "run_command":
        report["source_binding"]["argv"][-1] = "1"
        _write_json(args.source_report, report)
    elif mutation == "prelaunch_output_path":
        prelaunch["outputs"]["report"] = "artifacts/substitute.json"
        _write_json(args.prelaunch_manifest, prelaunch)
        args.expected_prelaunch_sha256 = _sha256_file(args.prelaunch_manifest)
        report["source_binding"]["prelaunch_source_manifest_sha256"] = (
            args.expected_prelaunch_sha256
        )
        _write_json(args.source_report, report)
    elif mutation == "dataset_tree":
        (args.dataset_root / "img/a/one.jpg").write_bytes(b"changed")
    elif mutation == "report_checkpoint_digest":
        report["source_binding"]["checkpoint_sha256"] = "0" * 64
        _write_json(args.source_report, report)
    elif mutation == "checkpoint_config_digest":
        checkpoint["resolved_config_sha256"] = "0" * 64
        _write_json(args.checkpoint, checkpoint)
    elif mutation in {"seed", "objective", "batch_size", "bn_mode", "proxy_count"}:
        key, value = {
            "seed": ("seed", 1),
            "objective": ("objectives", ["proxy_anchor_coalition"]),
            "batch_size": ("batch_size", 179),
            "bn_mode": ("freeze_batch_norm", True),
            "proxy_count": ("proxy_count_per_class", 2),
        }[mutation]
        config[key] = value
        _write_json(args.resolved_config, config)
    elif mutation == "post_source_replay":
        report["source_binding"]["source_python_tree_sha256_after"] = "0" * 64
        _write_json(args.source_report, report)
    elif mutation == "post_data_replay":
        report["source_binding"]["dataset_image_tree_sha256_after"] = "0" * 64
        _write_json(args.source_report, report)
    else:  # pragma: no cover
        raise AssertionError(mutation)

    touched: list[str] = []
    monkeypatch.setattr(MODULE, "_import_torch", lambda: touched.append("torch"))
    monkeypatch.setattr(MODULE, "_load_process_runtime", lambda *args: touched.append("model"))
    monkeypatch.setattr(MODULE, "score_context", lambda **kwargs: touched.append("score"))
    with pytest.raises(ValueError):
        MODULE.activate_source(args)
    assert touched == []
    assert not args.activated_preregistration.exists()
    assert not args.source_manifest.exists()


def test_activation_failure_preserves_prior_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _source_binding_fixture(tmp_path)
    MODULE.activate_source(args)
    before_preregistration = args.activated_preregistration.read_bytes()
    before_manifest = args.source_manifest.read_bytes()

    def fail_second_write(path: Path, payload: object) -> bytes:
        if path == args.source_manifest:
            raise OSError("injected source manifest write failure")
        return MODULE.canonical_json_bytes(payload) + b"\n"

    monkeypatch.setattr(MODULE, "_atomic_write_json", fail_second_write)
    with pytest.raises(OSError, match="injected"):
        MODULE.activate_source(args)
    assert args.activated_preregistration.read_bytes() == before_preregistration
    assert args.source_manifest.read_bytes() == before_manifest


def test_source_activation_reads_bound_torch_checkpoint_metadata_without_import_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _source_binding_fixture(tmp_path)
    metadata = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    metadata["state_dict"] = {"weight": torch.tensor([1.0])}
    torch.save(metadata, args.checkpoint)
    report = json.loads(args.source_report.read_text(encoding="utf-8"))
    report["source_binding"]["checkpoint_sha256"] = _sha256_file(args.checkpoint)
    _write_json(args.source_report, report)
    touched: list[str] = []
    monkeypatch.setattr(MODULE, "_import_torch", lambda: touched.append("torch"))

    _, manifest = MODULE.activate_source(args)

    assert touched == []
    assert manifest["checkpoint_sha256"] == _sha256_file(args.checkpoint)


class _FakeCuda:
    def __init__(self, events: list[str]):
        self.events = events

    def device_count(self) -> int:
        self.events.append("enumerate")
        return 1

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(name="fake-gpu", pci_bus_id="0000:01:00.0")

    def manual_seed_all(self, seed: int) -> None:
        assert seed == 2010812
        self.events.append("cuda_seed")

    def get_rng_state(self, index: int) -> np.ndarray:
        assert index == 0
        return np.asarray([4, 5, 6], dtype=np.uint8)


class _FakeBackendFlags:
    benchmark = True
    deterministic = False
    allow_tf32 = True


class _FakeTorch:
    __version__ = "test-torch"
    version = SimpleNamespace(cuda="test-cuda")

    def __init__(self, events: list[str]):
        self.events = events
        self.cuda = _FakeCuda(events)
        self.backends = SimpleNamespace(
            cudnn=_FakeBackendFlags(),
            cuda=SimpleNamespace(matmul=_FakeBackendFlags()),
        )

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        assert enabled is True and warn_only is False
        self.events.append("deterministic")

    def manual_seed(self, seed: int) -> None:
        assert seed == 2010812
        self.events.append("torch_seed")

    def get_rng_state(self) -> np.ndarray:
        return np.asarray([1, 2, 3], dtype=np.uint8)


def _process_digest(index: int) -> dict:
    record = {
        "context_index": index,
        "s_tensor_sha256": hashlib.sha256(f"s-{index}".encode()).hexdigest(),
        "s_prime_tensor_sha256": hashlib.sha256(f"sp-{index}".encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(f"m-{index}".encode()).hexdigest(),
    }
    record["combined_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record


class _FakeProcessRuntime:
    def __init__(self, events: list[str], role: str):
        self.events = events
        self.role = role

    def prepare_contexts(self) -> list[dict]:
        self.events.append("prepare")
        return [
            {
                "digest_record": _process_digest(index),
                "context": {"context_index": index, "operator_value": float(index)},
            }
            for index in range(32)
        ]

    def score_context(self, index: int, prepared: dict) -> dict:
        self.events.append(f"score:{index}")
        return {
            "context": prepared["context"],
            "replay_tensors": {"gradient": [float(index), 1.0]},
            "replay_scalars": {"loss": float(index)},
        }

    def finalize(self, contexts: list[dict]) -> None:
        self.events.append(f"finalize:{len(contexts)}")
        return None


def _synthetic_task2_score() -> dict:
    gradients = {}
    updates = {
        regime: {operator: {} for operator in OPERATORS} for regime in REGIMES
    }
    outcomes = {
        regime: {operator: {} for operator in OPERATORS} for regime in REGIMES
    }
    for operator in OPERATORS:
        for panel in PANELS:
            gradients[f"{operator}.{panel}"] = MODULE.PanelGradient(
                parameter_names=("weight",),
                named_gradients=(("weight", np.asarray([0.1])),),
                named_update_gradients=(("weight", np.asarray([0.1])),),
                parameter_count=2,
                gradient_sha256=HEX_A,
                raw_gradient_norm=0.1,
                update_space_norm=0.1,
                auxiliary_to_pa_norm_ratio=1.0,
                cosine_with_pa=1.0,
                cosine_with_atomic_full_union=1.0,
                cosine_with_summed_dropout=1.0,
                scale_residual_to_summed_union=(
                    1.0 if operator.startswith("atomic_") else None
                ),
            )
            for regime in REGIMES:
                equal = regime == "equal_norm"
                updates[regime][operator][panel] = MODULE.StatelessUpdate(
                    parameter_names=("weight",),
                    named_updates=(("weight", np.asarray([-0.01])),),
                    update_sha256=HEX_B,
                    parameter_update_norm=0.1,
                    reference_pa_norm=0.1 if equal else None,
                    norm_match_absolute_error=0.0 if equal else None,
                )
                outcomes[regime][operator][panel] = MODULE.OutcomeFields(
                    R_F=0.02,
                    Delta_M=0.02,
                    D_F=0.02,
                    D_M=0.02,
                )
    return {
        "losses": {operator: np.float64(0.5) for operator in OPERATORS},
        "gradients": gradients,
        "updates": updates,
        "outcomes": outcomes,
        "shared_confuser": {
            "foreign_proxy_rows": 3,
            "A_aligned": 0.2,
            "null_mean": 0.1,
            "E_shared": 1.0,
            "null_distribution": np.zeros(256, dtype="<f8"),
            "null_distribution_sha256": HEX_A,
        },
    }


def test_process_role_materializes_the_complete_task2_context_schema() -> None:
    expected = _context(0)
    base = {
        key: deepcopy(value)
        for key, value in expected.items()
        if key
        not in {
            "foreign_proxy_rows",
            "shared_confuser",
            "operators",
        }
    }
    score = _synthetic_task2_score()
    score["train_graph"] = object()
    actual = MODULE.materialize_scored_context(base, score)
    MODULE._validate_context(actual, 0, [])
    assert actual == expected


def test_process_role_uses_the_bound_in_module_runtime_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.delenv("PASS201_RUNTIME_FACTORY", raising=False)
    monkeypatch.setattr(
        MODULE,
        "_ProductionRuntime",
        lambda source_manifest, torch_module: sentinel,
    )
    assert MODULE._load_process_runtime({"schema_version": "pass201-source-v1"}, object()) is (
        sentinel
    )


def test_process_role_sets_determinism_before_model_and_seeds_once_without_rewind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    fake_torch = _FakeTorch(events)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PASS201_PROCESS_MODE", "smoke")
    monkeypatch.setattr(MODULE, "_import_torch", lambda: fake_torch)

    def load_runtime(source_manifest: dict, torch_module: object) -> _FakeProcessRuntime:
        assert source_manifest == {"bound": True}
        assert torch_module is fake_torch
        assert events[:2] == ["enumerate", "deterministic"]
        events.append("model")
        return _FakeProcessRuntime(events, "scientific")

    monkeypatch.setattr(MODULE, "_load_process_runtime", load_runtime)
    output = tmp_path / "scientific.json"
    MODULE.run_process_role("scientific", {"bound": True}, output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert events.count("enumerate") == 1
    assert events.count("torch_seed") == 1
    assert events.count("cuda_seed") == 1
    assert events.index("deterministic") < events.index("model")
    assert events.index("prepare") < events.index("score:0")
    assert [event for event in events if event.startswith("score:")] == [
        f"score:{index}" for index in range(32)
    ]
    assert payload["process_record"]["prepared_context_count"] == 32
    assert payload["aggregate_context_indices"] == list(range(32))
    assert len(payload["contexts"]) == 32


@pytest.mark.parametrize(
    ("role", "expected_scores"),
    [
        ("integrity_replay_a", [0]),
        ("integrity_replay_b", [0]),
        ("scientific", list(range(32))),
    ],
)
def test_process_role_scores_only_the_role_authorized_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_scores: list[int],
) -> None:
    events: list[str] = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PASS201_PROCESS_MODE", "smoke")
    monkeypatch.setattr(MODULE, "_import_torch", lambda: _FakeTorch(events))
    monkeypatch.setattr(
        MODULE,
        "_load_process_runtime",
        lambda source, torch_module: _FakeProcessRuntime(events, role),
    )
    output = tmp_path / f"{role}.json"
    MODULE.run_process_role(role, {"bound": True}, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [int(event.split(":")[1]) for event in events if event.startswith("score:")] == (
        expected_scores
    )
    assert payload["aggregate_context_indices"] == (
        expected_scores if role == "scientific" else []
    )


def _comparison_process(
    role: str, *, tensor: float = 0.0, scalar: float = 1.0
) -> dict:
    context = {"context_index": 0, "operator_value": 1.0}
    contexts = (
        [context]
        + [{"context_index": index, "operator_value": float(index)} for index in range(1, 32)]
        if role == "scientific"
        else [context]
    )
    return {
        "schema_version": "pass201-process-v1",
        "status": "ok",
        "role": role,
        "process_record": {
            "role": role,
            "pid": 1,
            "accelerator": "fake-gpu",
            "python_version": "3.13.9",
            "torch_version": "2.12.1",
            "cuda_version": "13.0",
            "cudnn_version": "9",
            "visible_cuda_devices": ["0000:01:00.0 fake-gpu"],
            "initial_python_rng_sha256": HEX_A,
            "initial_numpy_rng_sha256": HEX_A,
            "initial_torch_cpu_rng_sha256": HEX_A,
            "initial_torch_cuda_rng_sha256_by_device": {"0": HEX_A},
            "prepared_context_count": 32,
            "input_context_digest_records": [_process_digest(i) for i in range(32)],
            "context0_record_sha256": hashlib.sha256(
                json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "contexts": contexts,
        "aggregate_context_indices": [] if role != "scientific" else list(range(32)),
        "replay_tensors": {"gradient": [tensor]},
        "replay_scalars": {"loss": scalar},
        "result": None,
    }


def test_replay_tensor_and_scalar_tolerances_are_exact() -> None:
    a = _comparison_process("integrity_replay_a")
    b = _comparison_process("integrity_replay_b", tensor=2e-6)
    scalar_at_limit = 1.0 / (1.0 - 1e-5)
    scientific = _comparison_process("scientific", scalar=scalar_at_limit)
    MODULE.compare_integrity_records(a, b, scientific)

    b["replay_tensors"]["gradient"][0] = float(np.nextafter(2e-6, np.inf))
    with pytest.raises(ValueError, match="tensor replay tolerance"):
        MODULE.compare_integrity_records(a, b, scientific)
    b["replay_tensors"]["gradient"][0] = 0.0
    scientific["replay_scalars"]["loss"] = float(np.nextafter(scalar_at_limit, np.inf))
    with pytest.raises(ValueError, match="scalar replay tolerance"):
        MODULE.compare_integrity_records(a, b, scientific)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("accelerator", "other-gpu", "environment replay"),
        ("initial_python_rng_sha256", HEX_B, "RNG replay"),
    ],
)
def test_replay_rejects_environment_and_initial_rng_drift(
    field: str, replacement: str, message: str
) -> None:
    a = _comparison_process("integrity_replay_a")
    b = _comparison_process("integrity_replay_b")
    scientific = _comparison_process("scientific")
    b["process_record"][field] = replacement
    with pytest.raises(ValueError, match=message):
        MODULE.compare_integrity_records(a, b, scientific)


def test_process_role_output_rejects_a_self_inconsistent_context0_hash() -> None:
    payload = _comparison_process("integrity_replay_a")
    payload["contexts"][0]["operator_value"] = 2.0
    with pytest.raises(ValueError, match="context-0 record digest"):
        MODULE._validate_process_output(payload, "integrity_replay_a")


def test_process_role_failure_preserves_atomic_prior_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingRuntime(_FakeProcessRuntime):
        def score_context(self, index: int, prepared: dict) -> dict:
            if index == 3:
                raise RuntimeError("injected scorer failure")
            return super().score_context(index, prepared)

    events: list[str] = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("PASS201_PROCESS_MODE", "smoke")
    monkeypatch.setattr(MODULE, "_import_torch", lambda: _FakeTorch(events))
    monkeypatch.setattr(
        MODULE,
        "_load_process_runtime",
        lambda source, torch_module: FailingRuntime(events, "scientific"),
    )
    output = tmp_path / "scientific.json"
    output.write_bytes(b"prior-committed-output\n")
    with pytest.raises(RuntimeError, match="injected"):
        MODULE.run_process_role("scientific", {"bound": True}, output)
    assert output.read_bytes() == b"prior-committed-output\n"
    assert not list(tmp_path.glob("*.tmp-*"))


def _write_fake_runtime_module(path: Path) -> None:
    path.write_text(
        """
import hashlib
import json
import os

def _digest(index):
    record = {
        "context_index": index,
        "s_tensor_sha256": hashlib.sha256(f"s-{index}".encode()).hexdigest(),
        "s_prime_tensor_sha256": hashlib.sha256(f"sp-{index}".encode()).hexdigest(),
        "metadata_sha256": hashlib.sha256(f"m-{index}".encode()).hexdigest(),
    }
    record["combined_sha256"] = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return record

class Runtime:
    def __init__(self, torch):
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        self.role = os.environ["PASS201_PROCESS_ROLE"]

    def prepare_contexts(self):
        return [
            {
                "digest_record": _digest(index),
                "context": {"context_index": index, "operator_value": float(index)},
            }
            for index in range(32)
        ]

    def score_context(self, index, prepared):
        if self.role == "scientific":
            assert os.environ.get("PASS201_BINDING_AUTHORIZED_SHA256")
            assert os.environ.get("PASS201_REPLAY_AUTHORIZED_SHA256")
        return {
            "context": prepared["context"],
            "replay_tensors": {"gradient": [float(index), 1.0]},
            "replay_scalars": {"loss": float(index)},
        }

    def finalize(self, contexts):
        return None

def build_runtime(source_manifest, torch):
    assert source_manifest["schema_version"] == "pass201-source-v1"
    return Runtime(torch)
""".lstrip(),
        encoding="utf-8",
    )


def test_process_role_controller_uses_exactly_three_fresh_ordered_children(
    tmp_path: Path,
) -> None:
    args = _source_binding_fixture(tmp_path)
    MODULE.activate_source(args)
    runtime_path = tmp_path / "fake_runtime.py"
    _write_fake_runtime_module(runtime_path)
    output = tmp_path / "smoke.json"
    args.smoke_only = True
    args.scientific = False
    args.binding_only = False
    args.output = output
    args.runtime_factory = runtime_path

    MODULE.run_controller(args)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "pass201-controller-smoke-v1"
    assert [process["role"] for process in payload["processes"]] == [
        "integrity_replay_a",
        "integrity_replay_b",
        "scientific",
    ]
    assert len({process["process_record"]["pid"] for process in payload["processes"]}) == 3
    records = [
        process["process_record"]["input_context_digest_records"]
        for process in payload["processes"]
    ]
    assert records[0] == records[1] == records[2]
    assert [len(process["contexts"]) for process in payload["processes"]] == [1, 1, 32]
    assert payload["processes"][0]["aggregate_context_indices"] == []
    assert payload["processes"][1]["aggregate_context_indices"] == []
    assert payload["processes"][2]["aggregate_context_indices"] == list(range(32))
    assert not list(output.parent.glob(".pass201-process-*"))


def test_replay_controller_failure_emits_only_the_launched_process_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _source_binding_fixture(tmp_path)
    MODULE.activate_source(args)
    output = tmp_path / "invalid.json"
    args.smoke_only = True
    args.scientific = False
    args.binding_only = False
    args.output = output
    args.runtime_factory = None
    bound_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    bound_manifest["prelaunch_source_manifest_sha256"] = (
        "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803"
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_controller_binding",
        lambda controller_args: (bound_manifest, _constants()),
    )
    a = _comparison_process("integrity_replay_a")
    b = _comparison_process("integrity_replay_b")
    a["process_record"]["pid"] = 101
    b["process_record"]["pid"] = 102
    b["process_record"]["input_context_digest_records"][1]["metadata_sha256"] = HEX_B
    combined = {
        key: b["process_record"]["input_context_digest_records"][1][key]
        for key in (
            "context_index",
            "s_tensor_sha256",
            "s_prime_tensor_sha256",
            "metadata_sha256",
        )
    }
    b["process_record"]["input_context_digest_records"][1]["combined_sha256"] = (
        hashlib.sha256(
            json.dumps(combined, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    launched: list[str] = []

    def spawn(role: str, **kwargs: object) -> dict:
        launched.append(role)
        return {"integrity_replay_a": a, "integrity_replay_b": b}[role]

    monkeypatch.setattr(MODULE, "_spawn_process_role", spawn)
    MODULE.run_controller(args)
    payload = json.loads(output.read_text(encoding="utf-8"))
    MODULE.validate_payload_structure(payload)
    assert launched == ["integrity_replay_a", "integrity_replay_b"]
    assert payload["status"] == "INVALID"
    assert payload["integrity"]["stage"] == "integrity_replay_b"
    assert [record["role"] for record in payload["integrity"]["process_records"]] == launched
    assert "contexts" not in payload
    assert "aggregates" not in payload


def test_cli_parser_exposes_only_the_frozen_execution_modes() -> None:
    parser = MODULE._build_cli_parser()
    actions = {option for action in parser._actions for option in action.option_strings}
    assert {
        "--activate-source",
        "--process-role",
        "--binding-only",
        "--smoke-only",
        "--scientific",
    } <= actions


def _literal_tensor_frame(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    dtype_bytes = contiguous.dtype.str.encode("utf-8")
    payload = contiguous.tobytes(order="C")
    return b"".join(
        [
            struct.pack("<I", len(dtype_bytes)),
            dtype_bytes,
            struct.pack("<I", contiguous.ndim),
            *(struct.pack("<q", dimension) for dimension in contiguous.shape),
            struct.pack("<Q", len(payload)),
            payload,
        ]
    )


def test_canonical_json_bytes_uses_frozen_encoding():
    value = {"z": "café", "a": [1, True, None]}
    assert MODULE.canonical_json_bytes(value) == b'{"a":[1,true,null],"z":"caf\xc3\xa9"}'
    with pytest.raises(ValueError):
        MODULE.canonical_json_bytes({"bad": float("nan")})


def test_tensor_digest_frames_dtype_shape_and_payload():
    base = np.array([[1.0, 2.0]], dtype="<f8")
    expected = hashlib.sha256(_literal_tensor_frame(base)).hexdigest()
    assert MODULE.sha256_tensor_frame(base) == expected
    assert MODULE.sha256_tensor_frame(base) != MODULE.sha256_tensor_frame(base.astype("<f4"))
    assert MODULE.sha256_tensor_frame(base) != MODULE.sha256_tensor_frame(base.reshape(2, 1))


def test_named_tensor_digest_frames_names_shapes_and_lengths():
    left = [("ab", np.array([1.0], dtype="<f8"))]
    right = [("a", np.array([98.0, 1.0], dtype="<f8"))]
    assert MODULE.sha256_named_tensors(left) != MODULE.sha256_named_tensors(right)


def test_named_tensor_digest_has_literal_framing_and_little_endian_float64():
    array = np.array([1.0, -2.0], dtype=">f4")
    name = "w.β".encode()
    normalized = np.asarray(array, dtype="<f8")
    frame = b"".join(
        [
            struct.pack("<I", len(name)),
            name,
            struct.pack("<I", 1),
            struct.pack("<q", 2),
            struct.pack("<Q", normalized.nbytes),
            normalized.tobytes(order="C"),
        ]
    )
    assert MODULE.sha256_named_tensors([("w.β", array)]) == hashlib.sha256(frame).hexdigest()


def test_gradient_hash_accepts_a_live_requires_grad_tensor_via_cpu_bytes():
    live = torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=True)
    expected = MODULE.sha256_named_tensors(
        [("weight", np.array([1.0, -2.0], dtype=np.float32))]
    )
    assert MODULE.sha256_named_tensors([("weight", live)]) == expected


def test_s_prime_is_disjoint_and_preserves_literal_label_sequence():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    assert context["row_labels"] == [7, 3, 7, 5]
    assert context["s_prime_example_ids"] == ["7-b", "3-b", "7-c", "5-b"]
    assert set(context["row_example_ids"]).isdisjoint(context["s_prime_example_ids"])


def test_representatives_use_sorted_labels_and_minimum_stable_index():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    assert context["representative_row_indices"] == [1, 3, 0]
    assert context["representative_sample_indices"] == [31, 50, 70]


def test_cross_context_reuse_is_causal_prefix_only():
    first = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    second_rows = [
        {"example_id": "7-a", "sample_index": 70, "label": 7},
        {"example_id": "8-a", "sample_index": 80, "label": 8},
    ]
    manifest = second_rows + [
        {"example_id": "7-e", "sample_index": 74, "label": 7},
        {"example_id": "8-b", "sample_index": 81, "label": 8},
    ]
    second = MODULE.construct_one_context(
        rows=second_rows,
        train_manifest=manifest,
        context_index=1,
        prior_contexts=[first],
    )
    assert first["cross_context_reuse"] == {
        "prior_context_indices_sharing_s_ids": [],
        "prior_context_indices_sharing_s_prime_ids": [],
        "prior_context_indices_sharing_any_ids": [],
        "reused_s_image_count": 0,
        "reused_s_prime_image_count": 0,
        "reused_any_image_count": 0,
        "reused_label_count": 0,
    }
    assert second["cross_context_reuse"] == {
        "prior_context_indices_sharing_s_ids": [0],
        "prior_context_indices_sharing_s_prime_ids": [],
        "prior_context_indices_sharing_any_ids": [0],
        "reused_s_image_count": 1,
        "reused_s_prime_image_count": 0,
        "reused_any_image_count": 1,
        "reused_label_count": 1,
    }


def test_rejected_batch_remains_in_partial_audit():
    infeasible = [
        {"example_id": "9-a", "sample_index": 90, "label": 9},
        {"example_id": "9-b", "sample_index": 91, "label": 9},
    ]
    manifest = (
        LITERAL_MANIFEST + infeasible + [{"example_id": "9-c", "sample_index": 92, "label": 9}]
    )
    accepted, audit = MODULE.construct_context_audit(
        batches=[LITERAL_ROWS, infeasible], train_manifest=manifest, target_count=2
    )
    assert len(accepted) == 1
    assert [entry["status"] for entry in audit] == ["accepted", "rejected"]
    rejected = audit[1]
    assert rejected["context_index"] == 1
    assert rejected["rejection_code"] == "INSUFFICIENT_DISJOINT_S_PRIME"
    assert rejected["row_example_ids"] == ["9-a", "9-b"]
    assert rejected["s_prime_example_ids"] == []
    assert rejected["s_prime_sample_indices"] == []


def test_input_context_digest_excludes_process_metadata():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    context["s_tensor"] = np.arange(8, dtype="<f4").reshape(2, 4)
    context["s_prime_tensor"] = np.arange(8, 16, dtype="<f4").reshape(2, 4)
    context["pid"] = 111
    left = MODULE.build_input_context_digest(context)
    context["pid"] = 222
    right = MODULE.build_input_context_digest(context)
    assert left == right
    assert set(left) == {
        "context_index",
        "s_tensor_sha256",
        "s_prime_tensor_sha256",
        "metadata_sha256",
        "combined_sha256",
    }
    combined = {key: value for key, value in left.items() if key != "combined_sha256"}
    assert (
        left["combined_sha256"]
        == hashlib.sha256(
            json.dumps(
                combined,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def _summary(*, lcb: float = 0.02, ucb: float = 0.03) -> dict:
    return {
        "n": 32,
        "mean": 0.025,
        "median": 0.025,
        "sample_sd": 0.001,
        "q25": 0.024,
        "q75": 0.026,
        "lcb_0_005": lcb,
        "ucb_0_995": ucb,
    }


def _update(*, equal_norm: bool) -> dict:
    return {
        "update_sha256": HEX_B,
        "parameter_update_norm": 0.1,
        "R_F": 0.02,
        "Delta_M": 0.02,
        "D_F": 0.02,
        "D_M": 0.02,
        "reference_pa_norm": 0.1 if equal_norm else None,
        "norm_match_absolute_error": 0.0 if equal_norm else None,
    }


def _operator(name: str) -> dict:
    panels = {}
    for panel in PANELS:
        panels[panel] = {
            "parameter_count": 2,
            "gradient_sha256": HEX_A,
            "raw_gradient_norm": 0.1,
            "update_space_norm": 0.1,
            "auxiliary_to_pa_norm_ratio": 1.0,
            "cosine_with_pa": 1.0,
            "cosine_with_atomic_full_union": 1.0,
            "cosine_with_summed_dropout": 1.0,
            "scale_residual_to_summed_union": (
                1.0
                if name in ("atomic_one_hot", "atomic_complementary", "atomic_full_union")
                else None
            ),
            "updates": {
                "configured_loss_stateless": _update(equal_norm=False),
                "equal_norm": _update(equal_norm=True),
            },
        }
    return {
        "name": name,
        "loss": 0.5,
        "representative_count": 2,
        "panels": panels,
    }


def _context(context_index: int) -> dict:
    row_ids = [f"{context_index}-s-{index}" for index in range(180)]
    s_prime_ids = [f"{context_index}-p-{index}" for index in range(180)]
    return {
        "context_index": context_index,
        "production_epoch": 0,
        "production_batch_index": context_index,
        "batch_size": 180,
        "m_unique": 2,
        "row_example_ids": row_ids,
        "row_sample_indices": list(range(180)),
        "row_labels": [1, 2] * 90,
        "class_multiplicities": {"1": 90, "2": 90},
        "representative_row_indices": [0, 1],
        "representative_sample_indices": [0, 1],
        "s_tensor_sha256": HEX_A,
        "s_prime_example_ids": s_prime_ids,
        "s_prime_sample_indices": list(range(180, 360)),
        "s_prime_tensor_sha256": HEX_B,
        "cross_context_reuse": {
            "prior_context_indices_sharing_s_ids": [],
            "prior_context_indices_sharing_s_prime_ids": [],
            "prior_context_indices_sharing_any_ids": [],
            "reused_s_image_count": 0,
            "reused_s_prime_image_count": 0,
            "reused_any_image_count": 0,
            "reused_label_count": 0 if context_index == 0 else 2,
        },
        "foreign_proxy_rows": 3,
        "shared_confuser": {
            "A_aligned": 0.2,
            "null_mean": 0.1,
            "E_shared": 1.0,
            "null_distribution_sha256": HEX_A,
        },
        "operators": {name: _operator(name) for name in OPERATORS},
    }


def _digest_record(context: dict) -> dict:
    metadata_keys = (
        "context_index",
        "production_epoch",
        "production_batch_index",
        "row_example_ids",
        "row_sample_indices",
        "row_labels",
        "class_multiplicities",
        "representative_row_indices",
        "representative_sample_indices",
        "s_prime_example_ids",
        "s_prime_sample_indices",
        "cross_context_reuse",
    )
    record = {
        "context_index": context["context_index"],
        "s_tensor_sha256": context["s_tensor_sha256"],
        "s_prime_tensor_sha256": context["s_prime_tensor_sha256"],
        "metadata_sha256": hashlib.sha256(
            json.dumps(
                {key: context[key] for key in metadata_keys},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    record["combined_sha256"] = hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return record


def test_process_role_scientific_aggregation_uses_the_32_retained_contexts() -> None:
    contexts = [_context(index) for index in range(32)]
    aggregates, distributions = MODULE.aggregate_scored_contexts(contexts)
    MODULE._validate_aggregates(aggregates)
    assert aggregates["m_unique"]["n"] == 32
    assert aggregates["equal_norm"]["network_only"]["operators"]["summed_union"][
        "R_F"
    ]["mean"] == 0.02
    assert set(distributions) == set(aggregates["bootstrap"]["distribution_sha256_by_metric"])


def test_named_tensor_digest_enforces_lexicographic_utf8_name_order():
    tensors = [
        ("z.weight", np.array([2.0])),
        ("ä.weight", np.array([3.0])),
        ("a.weight", np.array([1.0])),
    ]
    expected_order = [tensors[2], tensors[0], tensors[1]]
    assert MODULE.sha256_named_tensors(tensors) == MODULE.sha256_named_tensors(expected_order)


def _metric_paths() -> list[str]:
    paths = ["m_unique", "shared_confuser.E_shared"]
    for regime in REGIMES:
        for panel in PANELS:
            for operator in OPERATORS:
                for metric in METRICS:
                    paths.append(f"{regime}.{panel}.operators.{operator}.{metric}")
            for metric in ("A_F", "A_M"):
                paths.append(f"{regime}.{panel}.paired_advantages.{metric}")
    return paths


def _aggregates() -> dict:
    aggregates = {"m_unique": _summary(), "shared_confuser": _summary(lcb=0.01)}
    for regime in REGIMES:
        aggregates[regime] = {}
        for panel in PANELS:
            aggregates[regime][panel] = {
                "operators": {
                    operator: {metric: _summary() for metric in METRICS} for operator in OPERATORS
                },
                "paired_advantages": {
                    "A_F": _summary(),
                    "A_M": _summary(),
                },
            }
    aggregates["bootstrap"] = {
        "seed": 2010811,
        "replicates": 20000,
        "quantile_method": "linear",
        "joint_context_index_sha256": HEX_A,
        "distribution_sha256_by_metric": {path: HEX_B for path in _metric_paths()},
    }
    return aggregates


def _source() -> dict:
    return {
        "prelaunch_source_manifest_path": "docs/pass201_pa_source_prelaunch_manifest.json",
        "prelaunch_source_manifest_sha256": (
            "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803"
        ),
        "source_report_path": "reports/source.json",
        "source_report_sha256": HEX_A,
        "source_revision": "f" * 40,
        "checkpoint_path": "checkpoints/source.pt",
        "checkpoint_sha256": HEX_A,
        "checkpoint_bytes": 123,
        "checkpoint_epoch": 2,
        "resolved_config_path": "reports/config.json",
        "resolved_config_sha256": HEX_A,
        "train_manifest_path": "reports/train.json",
        "train_manifest_sha256": HEX_A,
        "diagnostic_source_sha256": HEX_A,
        "activated_preregistration_sha256": HEX_A,
        "python_version": "3.11.0",
        "torch_version": "2.0.0",
        "numpy_version": "2.0.0",
        "cuda_version": "12.0",
        "cudnn_version": "9",
    }


def _constants() -> dict:
    return {
        "batch_size": 180,
        "context_pairs": 32,
        "null_replicates": 256,
        "bootstrap_replicates": 20000,
        "s_prime_rank_seed": 2010809,
        "null_seed": 2010810,
        "bootstrap_seed": 2010811,
        "model_forward_seed": 2010812,
        "learning_rate": 0.001,
        "coalition_weight": 0.2,
        "proxy_learning_rate_multiplier": 10.0,
        "owner_margin_temperature": 0.05,
    }


def _process_record(role: str, contexts: list[dict]) -> dict:
    return {
        "role": role,
        "pid": 100,
        "accelerator": "synthetic",
        "python_version": "3.11.0",
        "torch_version": "2.0.0",
        "cuda_version": "12.0",
        "cudnn_version": "9",
        "visible_cuda_devices": ["0000:00:01.0"],
        "initial_python_rng_sha256": HEX_A,
        "initial_numpy_rng_sha256": HEX_A,
        "initial_torch_cpu_rng_sha256": HEX_A,
        "initial_torch_cuda_rng_sha256_by_device": {"0": HEX_A},
        "prepared_context_count": 32,
        "input_context_digest_records": [_digest_record(context) for context in contexts],
        "context0_record_sha256": hashlib.sha256(
            json.dumps(
                contexts[0],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def literal_valid_scored_payload() -> dict:
    contexts = [_context(index) for index in range(32)]
    records = [
        _process_record(role, contexts)
        for role in ("integrity_replay_a", "integrity_replay_b", "scientific")
    ]
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "PASS",
        "reason_codes": [],
        "candidate_values_computed": True,
        "uses_test_data": "artifact_binding_only",
        "source": _source(),
        "constants": _constants(),
        "contexts": contexts,
        "aggregates": _aggregates(),
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "component_decisions": {key: "PASS" for key in THRESHOLDS},
            "overall": "PASS",
            "authorized_next_action": "write_separate_gpu_preregistration",
        },
        "integrity": {
            "accepted_context_count": 32,
            "rejected_context_count": 0,
            "invalid_context_count": 0,
            "input_replay_verified": True,
            "parameter_hash_before": HEX_A,
            "parameter_hash_after": HEX_A,
            "buffer_hash_before": HEX_B,
            "buffer_hash_after": HEX_B,
            "training_flags_restored": True,
            "deterministic_process_verified": True,
            "first_context_operator_replay_verified": True,
            "deterministic_settings": {
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "autocast": False,
                "dtype": "float32",
            },
            "process_records": records,
            "replay_residuals": {
                "pair_count": 3,
                "tensor_max_absolute": 2e-6,
                "scalar_max_relative": 1e-5,
                "tensor_tolerance": 2e-6,
                "scalar_tolerance": 1e-5,
                "scalar_denominator": "max(abs(a),abs(b),1e-12)",
            },
            "all_finite": True,
        },
    }


def _failure_digest(status: str, reason_codes: list[str], integrity: dict) -> str:
    process_records = integrity["process_records"]
    evidence = {
        "status": status,
        "reason_codes": sorted(reason_codes),
        "stage": integrity["stage"],
        "accepted_context_count": integrity["accepted_context_count"],
        "rejected_context_count": integrity["rejected_context_count"],
        "invalid_context_count": integrity["invalid_context_count"],
        "last_process_record": process_records[-1] if process_records else None,
    }
    return hashlib.sha256(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def literal_valid_blocked_payload() -> dict:
    reason_codes = ["BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE"]
    integrity = {
        "stage": "source_activation",
        "accepted_context_count": 0,
        "rejected_context_count": 0,
        "invalid_context_count": 0,
        "input_replay_verified": False,
        "deterministic_process_verified": False,
        "process_records": [],
        "failure_evidence_sha256": "",
        "all_finite": False,
    }
    integrity["failure_evidence_sha256"] = _failure_digest("BLOCKED", reason_codes, integrity)
    source = _source()
    for key in source.keys() - {
        "prelaunch_source_manifest_path",
        "prelaunch_source_manifest_sha256",
    }:
        source[key] = None
    constants = _constants()
    for key in (
        "learning_rate",
        "coalition_weight",
        "proxy_learning_rate_multiplier",
    ):
        constants[key] = None
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "BLOCKED",
        "reason_codes": reason_codes,
        "candidate_values_computed": False,
        "uses_test_data": "artifact_binding_only",
        "source": source,
        "constants": constants,
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "overall": "BLOCKED",
            "authorized_next_action": "none",
        },
        "integrity": integrity,
    }


def literal_valid_invalid_payload() -> dict:
    payload = literal_valid_blocked_payload()
    payload["status"] = "INVALID"
    payload["reason_codes"] = ["INVALID_OPERATING_POINT_MISMATCH"]
    payload["decision"]["overall"] = "INVALID"
    payload["integrity"]["failure_evidence_sha256"] = _failure_digest(
        "INVALID", payload["reason_codes"], payload["integrity"]
    )
    return payload


def literal_valid_early_unresolved_payload() -> dict:
    full = _context(0)
    contexts = []
    for index in range(32):
        entry = {
            key: deepcopy(value)
            for key, value in full.items()
            if key
            in {
                "context_index",
                "production_epoch",
                "production_batch_index",
                "row_example_ids",
                "row_sample_indices",
                "row_labels",
                "class_multiplicities",
                "representative_row_indices",
                "representative_sample_indices",
                "s_prime_example_ids",
                "s_prime_sample_indices",
            }
        }
        entry["context_index"] = index
        entry["production_batch_index"] = index
        entry["status"] = "accepted" if index < 31 else "rejected"
        entry["rejection_code"] = None if index < 31 else "INSUFFICIENT_DISJOINT_S_PRIME"
        if index == 31:
            entry["s_prime_example_ids"] = []
            entry["s_prime_sample_indices"] = []
        contexts.append(entry)
    process = _process_record("integrity_replay_a", [_context(index) for index in range(32)])
    process["prepared_context_count"] = 31
    process["input_context_digest_records"] = process["input_context_digest_records"][:31]
    process["context0_record_sha256"] = None
    reason_codes = ["UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS"]
    integrity = {
        "stage": "context_construction",
        "accepted_context_count": 31,
        "rejected_context_count": 1,
        "invalid_context_count": 0,
        "input_replay_verified": False,
        "deterministic_process_verified": False,
        "process_records": [process],
        "failure_evidence_sha256": "",
        "all_finite": True,
    }
    integrity["failure_evidence_sha256"] = _failure_digest("UNRESOLVED", reason_codes, integrity)
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "UNRESOLVED",
        "reason_codes": reason_codes,
        "candidate_values_computed": False,
        "uses_test_data": "artifact_binding_only",
        "source": _source(),
        "constants": _constants(),
        "contexts": contexts,
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "overall": "UNRESOLVED",
            "authorized_next_action": "none",
        },
        "integrity": integrity,
    }


def test_bootstrap_indices_are_the_frozen_shared_matrix_and_digest():
    indices = MODULE.bootstrap_indices()
    expected = np.random.Generator(np.random.PCG64(2010811)).integers(
        0, 32, size=(20000, 32), dtype=np.int64
    )
    assert indices.dtype.str == "<i8"
    assert indices.flags.c_contiguous
    assert np.array_equal(indices, expected)
    assert hashlib.sha256(indices.tobytes(order="C")).hexdigest() == (
        MODULE.sha256_bootstrap_indices(indices)
    )


def test_summary_uses_sample_sd_and_frozen_bootstrap_bounds():
    values = np.arange(32, dtype=np.float64)
    indices = np.tile(np.arange(32, dtype="<i8"), (20000, 1))
    summary = MODULE.summarize_metric(values, indices)
    assert summary == {
        "n": 32,
        "mean": 15.5,
        "median": 15.5,
        "sample_sd": pytest.approx(np.std(values, ddof=1)),
        "q25": 7.75,
        "q75": 23.25,
        "lcb_0_005": 15.5,
        "ucb_0_995": 15.5,
    }
    distribution = MODULE.bootstrap_mean_distribution(values, indices)
    assert distribution.dtype.str == "<f8"
    assert hashlib.sha256(distribution.tobytes(order="C")).hexdigest() == (
        MODULE.sha256_bootstrap_distribution(distribution)
    )


@pytest.mark.parametrize(
    "payload_factory",
    [
        literal_valid_scored_payload,
        literal_valid_early_unresolved_payload,
        literal_valid_blocked_payload,
        literal_valid_invalid_payload,
    ],
)
def test_payload_validator_accepts_each_conditional_family(payload_factory):
    MODULE.validate_payload_structure(payload_factory())


def test_early_payload_rejects_internally_wrong_partial_audit_metadata():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["class_multiplicities"] = {"3": 2, "5": 1, "7": 1}
    with pytest.raises(ValueError, match="class_multiplicities"):
        MODULE.validate_payload_structure(payload)


def test_accepted_partial_requires_180_integer_s_prime_indices():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["s_prime_sample_indices"].pop()
    with pytest.raises(ValueError, match="s_prime_sample_indices"):
        MODULE.validate_payload_structure(payload)


def test_accepted_partial_requires_integer_s_prime_index_elements():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["s_prime_sample_indices"][10] = False
    with pytest.raises(ValueError, match="s_prime_sample_indices"):
        MODULE.validate_payload_structure(payload)


def test_accepted_partial_requires_distinct_s_prime_ids():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["s_prime_example_ids"][1] = payload["contexts"][0][
        "s_prime_example_ids"
    ][0]
    with pytest.raises(ValueError, match="s_prime_example_ids"):
        MODULE.validate_payload_structure(payload)


def test_accepted_partial_requires_s_prime_ids_disjoint_from_s():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["s_prime_example_ids"][0] = payload["contexts"][0]["row_example_ids"][0]
    with pytest.raises(ValueError, match="s_prime_example_ids"):
        MODULE.validate_payload_structure(payload)


def test_partial_audit_production_batches_are_strictly_ordered():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][1]["production_batch_index"] = 0
    with pytest.raises(ValueError, match="production_batch_index"):
        MODULE.validate_payload_structure(payload)


@pytest.mark.parametrize(
    "payload_factory", [literal_valid_blocked_payload, literal_valid_invalid_payload]
)
def test_reduced_payload_requires_literal_false_candidate_flag(payload_factory):
    payload = payload_factory()
    payload["candidate_values_computed"] = 0
    with pytest.raises(ValueError, match="candidate_values_computed"):
        MODULE.validate_payload_structure(payload)


def test_reduced_payload_rejects_arbitrary_reason_code():
    payload = literal_valid_blocked_payload()
    payload["reason_codes"] = ["BLOCKED_SOURCE_UNAVAILABLE"]
    payload["integrity"]["failure_evidence_sha256"] = _failure_digest(
        payload["status"], payload["reason_codes"], payload["integrity"]
    )
    with pytest.raises(ValueError, match="reason_codes"):
        MODULE.validate_payload_structure(payload)


def test_source_activation_invalid_allows_only_operating_point_mismatch():
    payload = literal_valid_invalid_payload()
    payload["reason_codes"] = ["INVALID_NONDETERMINISTIC_TRAIN_INPUT"]
    payload["integrity"]["failure_evidence_sha256"] = _failure_digest(
        payload["status"], payload["reason_codes"], payload["integrity"]
    )
    with pytest.raises(ValueError, match="reason_codes"):
        MODULE.validate_payload_structure(payload)


def _extra_key(payload):
    payload["unexpected"] = True


def _missing_key(payload):
    del payload["source"]


def _wrong_null(payload):
    payload["contexts"][0]["operators"]["proxy_anchor"]["panels"]["network_only"][
        "scale_residual_to_summed_union"
    ] = 1.0


def _nonfinite(payload):
    payload["contexts"][0]["shared_confuser"]["E_shared"] = float("inf")


def _status_mismatch(payload):
    payload["status"] = "FAIL"


def _wrong_process_prefix(payload):
    payload["integrity"]["process_records"][0]["role"] = "scientific"


def _unsorted_visible_devices(payload):
    record = payload["integrity"]["process_records"][0]
    record["visible_cuda_devices"] = ["0000:00:02.0", "0000:00:01.0"]
    record["initial_torch_cuda_rng_sha256_by_device"] = {"0": HEX_A, "1": HEX_A}


def _source_activation_invalid_count(payload):
    blocked = literal_valid_blocked_payload()
    payload.clear()
    payload.update(blocked)
    payload["status"] = "INVALID"
    payload["reason_codes"] = ["INVALID_OPERATING_POINT_MISMATCH"]
    payload["decision"]["overall"] = "INVALID"
    payload["integrity"]["invalid_context_count"] = 1
    payload["integrity"]["failure_evidence_sha256"] = _failure_digest(
        "INVALID", payload["reason_codes"], payload["integrity"]
    )


def test_source_activation_invalid_count_mutation_reaches_count_rule():
    payload = literal_valid_scored_payload()
    _source_activation_invalid_count(payload)
    with pytest.raises(ValueError, match="source activation counts"):
        MODULE.validate_payload_structure(payload)


def _component_reason_mixing(payload):
    payload["reason_codes"] = ["FAIL_NO_SHARED_CONFOUNDER"]


def _malformed_digest(payload):
    payload["source"]["checkpoint_sha256"] = "ABC"


def _wrong_prelaunch_digest(payload):
    payload["source"]["prelaunch_source_manifest_sha256"] = "0" * 64


def _context0_hash_includes_process_metadata(payload):
    record = payload["integrity"]["process_records"][0]
    record["context0_record_sha256"] = hashlib.sha256(
        json.dumps(
            {**payload["contexts"][0], "pid": record["pid"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _self_consistent_wrong_class_multiplicities(payload):
    context = payload["contexts"][0]
    context["class_multiplicities"] = {"1": 89, "2": 91}
    digest_record = _digest_record(context)
    context_sha256 = hashlib.sha256(
        json.dumps(
            context,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    for record in payload["integrity"]["process_records"]:
        record["input_context_digest_records"][0] = deepcopy(digest_record)
        record["context0_record_sha256"] = context_sha256


def _self_consistent_false_causal_reuse(payload):
    context = payload["contexts"][1]
    context["cross_context_reuse"]["prior_context_indices_sharing_any_ids"] = [0]
    context["cross_context_reuse"]["reused_any_image_count"] = 1
    digest_record = _digest_record(context)
    for record in payload["integrity"]["process_records"]:
        record["input_context_digest_records"][1] = deepcopy(digest_record)


def _rehash_input_context(payload: dict, context_index: int) -> None:
    digest_record = _digest_record(payload["contexts"][context_index])
    for record in payload["integrity"]["process_records"]:
        record["input_context_digest_records"][context_index] = deepcopy(digest_record)


def test_scored_production_batch_indices_are_strictly_increasing():
    payload = literal_valid_scored_payload()
    payload["contexts"][1]["production_batch_index"] = 100
    _rehash_input_context(payload, 1)
    assert [context["production_batch_index"] for context in payload["contexts"][:3]] == [0, 100, 2]
    with pytest.raises(ValueError, match="production_batch_index"):
        MODULE.validate_payload_structure(payload)


def _integer_in_float_field(payload):
    payload["contexts"][1]["shared_confuser"]["E_shared"] = 1


def _integer_in_summary_float_field(payload):
    payload["aggregates"]["m_unique"]["mean"] = 2


def _float_representative_count(payload):
    payload["contexts"][1]["operators"]["proxy_anchor"]["representative_count"] = 2.0


def _bool_row_sample_index(payload):
    payload["contexts"][1]["row_sample_indices"][10] = False
    _rehash_input_context(payload, 1)


def _bool_row_label(payload):
    payload["contexts"][1]["row_labels"][10] = True
    _rehash_input_context(payload, 1)


def _non_string_row_id(payload):
    payload["contexts"][1]["row_example_ids"][10] = 10
    _rehash_input_context(payload, 1)


def _integer_threshold_float(payload):
    payload["decision"]["thresholds"]["joint_equal_union_margin_change"] = 0


def _bool_production_epoch(payload):
    payload["contexts"][1]["production_epoch"] = False
    _rehash_input_context(payload, 1)


def _integer_operator_loss(payload):
    payload["contexts"][1]["operators"]["proxy_anchor"]["loss"] = 1


def _float_bootstrap_seed(payload):
    payload["aggregates"]["bootstrap"]["seed"] = 2010811.0


@pytest.mark.parametrize(
    "mutation",
    [
        _integer_in_float_field,
        _integer_in_summary_float_field,
        _float_representative_count,
        _bool_row_sample_index,
        _bool_row_label,
        _non_string_row_id,
        _integer_threshold_float,
        _bool_production_epoch,
        _integer_operator_loss,
        _float_bootstrap_seed,
    ],
)
def test_payload_validator_enforces_strict_numeric_and_element_types(mutation):
    payload = literal_valid_scored_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        MODULE.validate_payload_structure(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        _extra_key,
        _missing_key,
        _wrong_null,
        _nonfinite,
        _status_mismatch,
        _wrong_process_prefix,
        _unsorted_visible_devices,
        _source_activation_invalid_count,
        _component_reason_mixing,
        _malformed_digest,
        _wrong_prelaunch_digest,
        _context0_hash_includes_process_metadata,
        _self_consistent_wrong_class_multiplicities,
        _self_consistent_false_causal_reuse,
    ],
)
def test_payload_validator_fails_closed(mutation):
    payload = literal_valid_scored_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        MODULE.validate_payload_structure(payload)


def test_threshold_boundaries_are_inclusive_and_failure_reasons_have_precedence():
    payload = literal_valid_scored_payload()
    aggregates = payload["aggregates"]
    aggregates["shared_confuser"]["lcb_0_005"] = 0.010
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["lcb_0_005"] = 0.001
    MODULE.validate_payload_structure(payload)

    aggregates["shared_confuser"]["lcb_0_005"] = -0.1
    aggregates["shared_confuser"]["ucb_0_995"] = 0.0
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["ucb_0_995"] = 0.0
    payload["status"] = "FAIL"
    payload["reason_codes"] = [
        "FAIL_NO_SHARED_CONFOUNDER",
        "FAIL_NO_COALITION_SPECIFIC_ACTION",
        "FAIL_PROXY_ONLY",
    ]
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["shared_confuser_excess"] = "FAIL"
    payload["decision"]["component_decisions"]["network_equal_union_advantage_foreign"] = "FAIL"
    MODULE.validate_payload_structure(payload)


def test_any_failed_component_makes_overall_fail_even_without_a_reason_predicate():
    payload = literal_valid_scored_payload()
    equal_joint = payload["aggregates"]["equal_norm"]["joint_including_proxies"]
    equal_joint["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    equal_joint["paired_advantages"]["A_F"]["ucb_0_995"] = 0.0
    configured_joint = payload["aggregates"]["configured_loss_stateless"]["joint_including_proxies"]
    configured_joint["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    payload["status"] = "FAIL"
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["joint_equal_union_advantage_foreign"] = "FAIL"
    MODULE.validate_payload_structure(payload)


def _set_network_predicted_suppression_failure(payload: dict) -> None:
    summary = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"][
        "D_F"
    ]
    summary["lcb_0_005"] = -0.1
    summary["ucb_0_995"] = 0.0
    payload["status"] = "FAIL"
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_predicted_suppression"] = "FAIL"


def test_predicted_suppression_failure_is_not_fail_not_viable():
    payload = literal_valid_scored_payload()
    _set_network_predicted_suppression_failure(payload)
    joint_advantage = payload["aggregates"]["equal_norm"]["joint_including_proxies"][
        "paired_advantages"
    ]["A_F"]
    joint_advantage["lcb_0_005"] = -0.1
    joint_advantage["ucb_0_995"] = 0.1
    payload["decision"]["component_decisions"]["joint_equal_union_advantage_foreign"] = "UNRESOLVED"
    payload["reason_codes"] = []
    MODULE.validate_payload_structure(payload)


def test_proxy_only_predicate_includes_network_predicted_components():
    payload = literal_valid_scored_payload()
    _set_network_predicted_suppression_failure(payload)
    payload["reason_codes"] = ["FAIL_PROXY_ONLY"]
    MODULE.validate_payload_structure(payload)


def test_proxy_only_includes_network_d_m_without_owner_damage():
    payload = literal_valid_scored_payload()
    network_union = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"]
    network_union["D_F"]["lcb_0_005"] = -0.1
    network_union["D_F"]["ucb_0_995"] = 0.1
    network_union["D_M"]["lcb_0_005"] = -0.1
    network_union["D_M"]["ucb_0_995"] = float(np.nextafter(0.0, -np.inf))
    payload["status"] = "FAIL"
    payload["reason_codes"] = ["FAIL_PROXY_ONLY"]
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_predicted_suppression"] = (
        "UNRESOLVED"
    )
    payload["decision"]["component_decisions"]["network_equal_union_predicted_margin_change"] = (
        "FAIL"
    )
    MODULE.validate_payload_structure(payload)


def _component_summaries(payload: dict) -> dict[str, dict]:
    aggregates = payload["aggregates"]
    network = aggregates["equal_norm"]["network_only"]
    joint = aggregates["equal_norm"]["joint_including_proxies"]
    return {
        "shared_confuser_excess": aggregates["shared_confuser"],
        "network_equal_union_advantage_foreign": network["paired_advantages"]["A_F"],
        "network_equal_union_advantage_margin": network["paired_advantages"]["A_M"],
        "network_equal_union_foreign_suppression": network["operators"]["summed_union"]["R_F"],
        "network_equal_union_margin_change": network["operators"]["summed_union"]["Delta_M"],
        "network_equal_union_predicted_suppression": network["operators"]["summed_union"]["D_F"],
        "network_equal_union_predicted_margin_change": network["operators"]["summed_union"]["D_M"],
        "joint_equal_union_advantage_foreign": joint["paired_advantages"]["A_F"],
        "joint_equal_union_advantage_margin": joint["paired_advantages"]["A_M"],
        "joint_equal_union_foreign_suppression": joint["operators"]["summed_union"]["R_F"],
        "joint_equal_union_margin_change": joint["operators"]["summed_union"]["Delta_M"],
    }


def test_every_pass_lcb_boundary_is_inclusive():
    payload = literal_valid_scored_payload()
    for key, summary in _component_summaries(payload).items():
        threshold = THRESHOLDS[key]
        summary["lcb_0_005"] = threshold
    MODULE.validate_payload_structure(payload)


def test_owner_margin_zero_ucb_is_unresolved_not_fail():
    payload = literal_valid_scored_payload()
    summary = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"][
        "Delta_M"
    ]
    summary["lcb_0_005"] = -0.1
    summary["ucb_0_995"] = 0.0
    payload["status"] = "UNRESOLVED"
    payload["decision"]["overall"] = "UNRESOLVED"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_margin_change"] = "UNRESOLVED"
    MODULE.validate_payload_structure(payload)


def test_owner_margin_next_float_below_zero_is_fail():
    payload = literal_valid_scored_payload()
    summary = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"][
        "D_M"
    ]
    summary["lcb_0_005"] = -0.1
    summary["ucb_0_995"] = float(np.nextafter(0.0, -np.inf))
    payload["status"] = "FAIL"
    payload["reason_codes"] = ["FAIL_PROXY_ONLY", "FAIL_OWNER_DAMAGE"]
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_predicted_margin_change"] = (
        "FAIL"
    )
    MODULE.validate_payload_structure(payload)


def literal_valid_scored_payload_and_evidence() -> tuple[dict, dict]:
    payload = literal_valid_scored_payload()
    gradient_tensors = {}
    update_tensors = {}
    null_distributions = {}
    for context in payload["contexts"]:
        context_index = context["context_index"]
        for operator in OPERATORS:
            for panel in PANELS:
                key = f"{context_index}.{operator}.{panel}"
                tensors = [("weight", np.array([1.0, 2.0]))]
                gradient_tensors[key] = tensors
                context["operators"][operator]["panels"][panel]["gradient_sha256"] = (
                    MODULE.sha256_named_tensors(tensors)
                )
                for regime in REGIMES:
                    update_key = f"{key}.{regime}"
                    update_tensors[update_key] = [("weight", np.array([1.0, 2.0]))]
                    context["operators"][operator]["panels"][panel]["updates"][regime][
                        "update_sha256"
                    ] = MODULE.sha256_named_tensors(update_tensors[update_key])
        null = np.linspace(0.0, 1.0, 256, dtype="<f8")
        null_distributions[str(context_index)] = null
        context["shared_confuser"]["null_distribution_sha256"] = hashlib.sha256(
            null.tobytes(order="C")
        ).hexdigest()
    indices = MODULE.bootstrap_indices()
    payload["aggregates"]["bootstrap"]["joint_context_index_sha256"] = hashlib.sha256(
        indices.tobytes(order="C")
    ).hexdigest()
    bootstrap_distributions = {}
    for metric_path in _metric_paths():
        distribution = np.linspace(0.0, 1.0, 20000, dtype="<f8")
        bootstrap_distributions[metric_path] = distribution
        payload["aggregates"]["bootstrap"]["distribution_sha256_by_metric"][metric_path] = (
            hashlib.sha256(distribution.tobytes(order="C")).hexdigest()
        )
    context0_sha256 = hashlib.sha256(
        json.dumps(
            payload["contexts"][0],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    for record in payload["integrity"]["process_records"]:
        record["context0_record_sha256"] = context0_sha256
    return payload, {
        "gradient_tensors": gradient_tensors,
        "update_tensors": update_tensors,
        "null_distributions": null_distributions,
        "bootstrap_indices": indices,
        "bootstrap_distributions": bootstrap_distributions,
    }


def test_construction_validation_rejects_digest_raw_evidence_mismatch():
    payload, evidence = literal_valid_scored_payload_and_evidence()
    MODULE.validate_construction_evidence(payload, evidence)
    evidence["gradient_tensors"]["0.proxy_anchor.network_only"][0][1][0] += 1.0
    with pytest.raises(ValueError, match="gradient_sha256"):
        MODULE.validate_construction_evidence(payload, evidence)


def _mutate_update_evidence(evidence):
    evidence["update_tensors"]["0.proxy_anchor.network_only.configured_loss_stateless"][0][1][
        0
    ] += 1.0


def _mutate_null_evidence(evidence):
    evidence["null_distributions"]["0"][0] += 1.0


def _mutate_bootstrap_indices(evidence):
    evidence["bootstrap_indices"][0, 0] = (evidence["bootstrap_indices"][0, 0] + 1) % 32


def _mutate_bootstrap_distribution(evidence):
    evidence["bootstrap_distributions"]["m_unique"][0] += 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_mutate_update_evidence, "update_sha256"),
        (_mutate_null_evidence, "null_distribution_sha256"),
        (_mutate_bootstrap_indices, "bootstrap indices"),
        (_mutate_bootstrap_distribution, "distribution_sha256"),
    ],
)
def test_construction_validation_checks_every_live_evidence_family(mutation, message):
    payload, evidence = literal_valid_scored_payload_and_evidence()
    mutation(evidence)
    with pytest.raises(ValueError, match=message):
        MODULE.validate_construction_evidence(payload, evidence)


def _literal_operator_fixture():
    root_half = 2.0**-0.5
    return {
        "embeddings": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [root_half, root_half]],
            dtype=torch.float64,
        ),
        "labels": torch.tensor([10, 20, 10, 30]),
        "sample_indices": torch.tensor([8, 9, 4, 1]),
        "proxies": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [root_half, root_half]], dtype=torch.float64
        ),
        "proxy_labels": torch.tensor([10, 20, 30]),
    }


def test_operator_losses_match_hand_derived_six_loss_fixture():
    losses = MODULE.coalition_losses(**_literal_operator_fixture(), alpha=1.0, delta=0.0)
    assert tuple(losses) == OPERATORS
    expected = {
        "proxy_anchor": 2.252048189835015,
        "atomic_one_hot": 0.7834148747907791,
        "atomic_complementary": 0.737391145638213,
        "atomic_full_union": 0.6262800345271019,
        "summed_union": 0.5146653905391636,
        "summed_dropout": 0.7071154802690388,
    }
    for name, want in expected.items():
        assert losses[name].shape == ()
        assert losses[name].item() == pytest.approx(want, abs=1e-12)


def test_atomic_full_union_is_per_image_and_summed_union_is_cross_image():
    members = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    proxies = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]], dtype=torch.float64
    )
    losses = MODULE.coalition_losses(
        embeddings=members,
        labels=torch.tensor([10, 20]),
        sample_indices=torch.tensor([4, 9]),
        proxies=proxies,
        proxy_labels=torch.tensor([10, 20, 30]),
        alpha=1.0,
        delta=0.0,
    )
    assert losses["atomic_full_union"].item() == pytest.approx(
        0.7047830585784727, abs=1e-12
    )
    assert losses["summed_union"].item() == pytest.approx(0.7049762468198759, abs=1e-12)


def test_operator_losses_select_minimum_index_and_are_aligned_permutation_invariant():
    fixture = _literal_operator_fixture()
    original = MODULE.coalition_losses(**fixture, alpha=1.0, delta=0.0)
    order = torch.tensor([3, 0, 2, 1])
    permuted = MODULE.coalition_losses(
        embeddings=fixture["embeddings"][order],
        labels=fixture["labels"][order],
        sample_indices=fixture["sample_indices"][order],
        proxies=fixture["proxies"],
        proxy_labels=fixture["proxy_labels"],
        alpha=1.0,
        delta=0.0,
    )
    for name in OPERATORS:
        assert permuted[name].item() == pytest.approx(original[name].item(), abs=1e-12)


def test_summed_dropout_removes_only_largest_sorted_label_target():
    fixture = _literal_operator_fixture()
    losses = MODULE.coalition_losses(**fixture, alpha=1.0, delta=0.0)
    assert losses["summed_dropout"].item() == pytest.approx(0.7071154802690388, abs=1e-12)
    assert losses["summed_dropout"].item() != pytest.approx(losses["summed_union"].item())


class _TinyOperatorModel(torch.nn.Module):
    def __init__(self, *, disconnected: bool = False):
        super().__init__()
        self.metric_proxies = torch.nn.Parameter(
            torch.tensor([[1.0, 0.1], [-0.2, 1.0], [0.8, 0.6]], dtype=torch.float64)
        )
        self.encoder = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.encoder.weight.data.copy_(
            torch.tensor([[1.0, 0.2], [-0.1, 0.9]], dtype=torch.float64)
        )
        if disconnected:
            self.unused = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float64))

    def forward(self, values):
        return self.encoder(values)


def _tiny_gradient_fixture(*, disconnected=False):
    model = _TinyOperatorModel(disconnected=disconnected)
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-0.8, 0.6], [0.6, 0.8]], dtype=torch.float64
    )
    labels = torch.tensor([10, 20, 10, 30])
    sample_indices = torch.tensor([8, 9, 4, 1])
    proxy_labels = torch.tensor([10, 20, 30])
    losses = MODULE.coalition_losses(
        embeddings=model(inputs),
        labels=labels,
        sample_indices=sample_indices,
        proxies=model.metric_proxies,
        proxy_labels=proxy_labels,
        alpha=1.3,
        delta=0.2,
    )
    return model, inputs, labels, sample_indices, proxy_labels, losses


def _loss_at_current_parameters(model, inputs, labels, sample_indices, proxy_labels, operator):
    return MODULE.coalition_losses(
        embeddings=model(inputs),
        labels=labels,
        sample_indices=sample_indices,
        proxies=model.metric_proxies,
        proxy_labels=proxy_labels,
        alpha=1.3,
        delta=0.2,
    )[operator]


@pytest.mark.parametrize("operator", OPERATORS)
def test_operator_gradient_matches_finite_differences(operator):
    model, inputs, labels, sample_indices, proxy_labels, losses = _tiny_gradient_fixture()
    expected_names = ("encoder.weight", "metric_proxies")
    gradients = MODULE.operator_gradients(
        losses,
        dict(model.named_parameters()),
        expected_trainable_parameter_names=expected_names,
        proxy_parameter_name="metric_proxies",
        proxy_learning_rate_multiplier=2.5,
    )
    record = gradients[f"{operator}.joint_including_proxies"]
    assert record.parameter_names == expected_names
    analytical = dict(record.named_gradients)
    epsilon = 1e-6
    for name, parameter in sorted(model.named_parameters()):
        for flat_index in range(parameter.numel()):
            with torch.no_grad():
                flat = parameter.view(-1)
                original = flat[flat_index].item()
                flat[flat_index] = original + epsilon
                plus = _loss_at_current_parameters(
                    model, inputs, labels, sample_indices, proxy_labels, operator
                ).item()
                flat[flat_index] = original - epsilon
                minus = _loss_at_current_parameters(
                    model, inputs, labels, sample_indices, proxy_labels, operator
                ).item()
                flat[flat_index] = original
            finite_difference = (plus - minus) / (2.0 * epsilon)
            assert analytical[name].view(-1)[flat_index].item() == pytest.approx(
                finite_difference, abs=2e-7
            )


def test_complementary_has_no_cross_member_derivative_but_summed_union_does():
    fixture = _literal_operator_fixture()

    def first_representative_gradient(operator, second_row):
        embeddings = fixture["embeddings"].clone()
        embeddings[1] = torch.tensor(second_row, dtype=torch.float64)
        embeddings.requires_grad_()
        loss = MODULE.coalition_losses(
            embeddings=embeddings,
            labels=fixture["labels"],
            sample_indices=fixture["sample_indices"],
            proxies=fixture["proxies"],
            proxy_labels=fixture["proxy_labels"],
            alpha=1.0,
            delta=0.0,
        )[operator]
        return torch.autograd.grad(loss, embeddings)[0][2]

    complementary_a = first_representative_gradient("atomic_complementary", [0.0, 1.0])
    complementary_b = first_representative_gradient("atomic_complementary", [0.6, 0.8])
    summed_a = first_representative_gradient("summed_union", [0.0, 1.0])
    summed_b = first_representative_gradient("summed_union", [0.6, 0.8])
    assert torch.equal(complementary_a, complementary_b)
    assert not torch.allclose(summed_a, summed_b, atol=1e-12, rtol=0.0)


def test_gradient_panels_use_exact_lexical_membership_and_proxy_multiplier():
    model, _, _, _, _, losses = _tiny_gradient_fixture()
    gradients = MODULE.operator_gradients(
        losses,
        dict(model.named_parameters()),
        expected_trainable_parameter_names=("encoder.weight", "metric_proxies"),
        proxy_parameter_name="metric_proxies",
        proxy_learning_rate_multiplier=2.5,
    )
    network = gradients["summed_union.network_only"]
    joint = gradients["summed_union.joint_including_proxies"]
    assert network.parameter_names == ("encoder.weight",)
    assert joint.parameter_names == ("encoder.weight", "metric_proxies")
    assert tuple(name for name, _ in network.named_update_gradients) == ("encoder.weight",)
    raw_joint = dict(joint.named_gradients)
    update_joint = dict(joint.named_update_gradients)
    assert torch.equal(update_joint["encoder.weight"], raw_joint["encoder.weight"])
    assert torch.equal(update_joint["metric_proxies"], 2.5 * raw_joint["metric_proxies"])


def test_gradient_panel_measurements_use_update_space_and_context_multiplicity():
    model, _, _, _, _, losses = _tiny_gradient_fixture()
    gradients = MODULE.operator_gradients(
        losses,
        dict(model.named_parameters()),
        expected_trainable_parameter_names=("encoder.weight", "metric_proxies"),
        proxy_parameter_name="metric_proxies",
        proxy_learning_rate_multiplier=2.5,
        representative_count=3,
    )
    for panel in PANELS:
        pa = gradients[f"proxy_anchor.{panel}"]
        full = gradients[f"atomic_full_union.{panel}"]
        dropout = gradients[f"summed_dropout.{panel}"]
        summed = gradients[f"summed_union.{panel}"]
        for operator in OPERATORS:
            record = gradients[f"{operator}.{panel}"]
            flattened = torch.cat(
                [
                    value.detach().to(torch.float64).view(-1)
                    for _, value in record.named_update_gradients
                ]
            )
            pa_flattened = torch.cat(
                [
                    value.detach().to(torch.float64).view(-1)
                    for _, value in pa.named_update_gradients
                ]
            )
            full_flattened = torch.cat(
                [
                    value.detach().to(torch.float64).view(-1)
                    for _, value in full.named_update_gradients
                ]
            )
            dropout_flattened = torch.cat(
                [
                    value.detach().to(torch.float64).view(-1)
                    for _, value in dropout.named_update_gradients
                ]
            )
            assert record.auxiliary_to_pa_norm_ratio == pytest.approx(
                record.update_space_norm / pa.update_space_norm, abs=1e-12
            )
            assert record.cosine_with_pa == pytest.approx(
                torch.dot(flattened, pa_flattened).item()
                / (
                    torch.linalg.vector_norm(flattened)
                    * torch.linalg.vector_norm(pa_flattened)
                ).item(),
                abs=1e-12,
            )
            assert record.cosine_with_atomic_full_union == pytest.approx(
                torch.dot(flattened, full_flattened).item()
                / (
                    torch.linalg.vector_norm(flattened)
                    * torch.linalg.vector_norm(full_flattened)
                ).item(),
                abs=1e-12,
            )
            assert record.cosine_with_summed_dropout == pytest.approx(
                torch.dot(flattened, dropout_flattened).item()
                / (
                    torch.linalg.vector_norm(flattened)
                    * torch.linalg.vector_norm(dropout_flattened)
                ).item(),
                abs=1e-12,
            )
            if operator.startswith("atomic_"):
                assert record.scale_residual_to_summed_union == pytest.approx(
                    summed.update_space_norm / (3.0**0.5 * record.update_space_norm),
                    abs=1e-12,
                )
            else:
                assert record.scale_residual_to_summed_union is None


@pytest.mark.parametrize(
    ("expected_names", "message"),
    [
        (("metric_proxies",), "unexpected trainable parameter"),
        (("encoder.weight", "ghost", "metric_proxies"), "missing trainable parameter"),
        (("metric_proxies", "encoder.weight"), "lexicographic"),
        (("encoder.weight", "encoder.weight", "metric_proxies"), "duplicate"),
    ],
)
def test_operator_gradients_reject_parameter_membership_mismatch(expected_names, message):
    model, _, _, _, _, losses = _tiny_gradient_fixture()
    with pytest.raises(ValueError, match=message):
        MODULE.operator_gradients(
            losses,
            dict(model.named_parameters()),
            expected_trainable_parameter_names=expected_names,
            proxy_parameter_name="metric_proxies",
            proxy_learning_rate_multiplier=2.5,
        )


def test_operator_gradients_reject_disconnected_required_parameter():
    model, _, _, _, _, losses = _tiny_gradient_fixture(disconnected=True)
    with pytest.raises(ValueError, match="disconnected.*unused"):
        MODULE.operator_gradients(
            losses,
            dict(model.named_parameters()),
            expected_trainable_parameter_names=(
                "encoder.weight",
                "metric_proxies",
                "unused",
            ),
            proxy_parameter_name="metric_proxies",
            proxy_learning_rate_multiplier=2.5,
        )


def test_stateless_updates_zero_network_proxy_and_match_panel_specific_equal_norm():
    model, _, _, _, _, losses = _tiny_gradient_fixture()
    gradients = MODULE.operator_gradients(
        losses,
        dict(model.named_parameters()),
        expected_trainable_parameter_names=("encoder.weight", "metric_proxies"),
        proxy_parameter_name="metric_proxies",
        proxy_learning_rate_multiplier=2.5,
    )
    updates = MODULE.make_stateless_updates(
        gradients, learning_rate=0.2, coalition_weight=0.3
    )
    for panel in PANELS:
        reference = updates["equal_norm"]["proxy_anchor"][panel].reference_pa_norm
        for operator in OPERATORS:
            update = updates["equal_norm"][operator][panel]
            assert update.parameter_update_norm == pytest.approx(reference, abs=1e-12)
            assert update.norm_match_absolute_error <= 1e-10 * max(reference, 1e-12)
    network_update = updates["configured_loss_stateless"]["summed_union"]["network_only"]
    joint_update = updates["configured_loss_stateless"]["summed_union"][
        "joint_including_proxies"
    ]
    assert "metric_proxies" not in dict(network_update.named_updates)
    assert "metric_proxies" in dict(joint_update.named_updates)
    assert updates["equal_norm"]["proxy_anchor"]["network_only"].reference_pa_norm != (
        updates["equal_norm"]["proxy_anchor"]["joint_including_proxies"].reference_pa_norm
    )


class _LiteralOutcomeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_proxies = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [-2**-0.5, -2**-0.5]], dtype=torch.float64
            )
        )
        self.encoder = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.encoder.weight.data.copy_(torch.eye(2, dtype=torch.float64))

    def forward(self, values):
        return self.encoder(values)


def test_owner_outcomes_match_literal_values_and_directional_signs():
    model = _LiteralOutcomeModel().train()
    before_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    update = {
        "encoder.weight": torch.tensor([[0.0, 0.02], [0.02, 0.0]], dtype=torch.float64)
    }
    outcome = MODULE.owner_outcomes(
        model=model,
        clean_inputs=torch.eye(2, dtype=torch.float64),
        labels=torch.tensor([10, 20]),
        proxy_labels=torch.tensor([10, 20, 30]),
        named_updates=update,
        proxy_parameter_name="metric_proxies",
        temperature=0.05,
    )
    assert pytest.approx(0.009352896580780748, abs=1e-12) == outcome.R_F
    assert outcome.Delta_M == pytest.approx(-0.020195923426626905, abs=1e-12)
    assert pytest.approx(0.009471858666137749, abs=1e-12) == outcome.D_F
    assert pytest.approx(-0.019999975371446453, abs=1e-12) == outcome.D_M
    assert outcome.R_F * outcome.D_F > 0.0
    assert outcome.Delta_M * outcome.D_M > 0.0
    assert model.training is True
    for name, value in model.named_parameters():
        assert torch.equal(value, before_parameters[name])


def test_shared_confuser_uses_exact_independent_row_null_stream():
    foreign = torch.tensor(
        [[1.0, 1.0], [1.0, -2.0], [-2.0, 1.0], [-1.0, -0.2]], dtype=torch.float64
    )
    result = MODULE.shared_confuser_statistic(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64),
        labels=torch.tensor([10, 20, 30]),
        sample_indices=torch.tensor([1, 2, 3]),
        proxies=torch.cat(
            (torch.eye(2, dtype=torch.float64), torch.tensor([[-1.0, 0.0]]), foreign)
        ),
        proxy_labels=torch.tensor([10, 20, 30, 40, 50, 60, 70]),
        context_index=0,
    )
    representative_rows = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    normalized_foreign = foreign.numpy().copy()
    normalized_foreign /= np.linalg.norm(normalized_foreign, axis=1, keepdims=True)
    q = 1.0 / (1.0 + np.exp(-(representative_rows @ normalized_foreign.T)))

    continuous_generator = np.random.Generator(np.random.PCG64(2010810))
    continuous = np.empty(256, dtype="<f8")
    for replicate in range(256):
        permuted = np.stack(
            [row[continuous_generator.permutation(q.shape[1])] for row in q], axis=0
        )
        continuous[replicate] = np.exp(np.log(permuted).mean(axis=0)).mean()

    legacy_reseeded = np.empty(256, dtype="<f8")
    for replicate in range(256):
        legacy_generator = np.random.Generator(np.random.PCG64(2010810 + replicate))
        permuted = np.stack(
            [row[legacy_generator.permutation(q.shape[1])] for row in q], axis=0
        )
        legacy_reseeded[replicate] = np.exp(np.log(permuted).mean(axis=0)).mean()

    assert result["A_aligned"] == pytest.approx(0.4718767931766301, abs=1e-12)
    assert result["null_mean"] == pytest.approx(0.48001819844444316, abs=1e-12)
    assert result["E_shared"] == pytest.approx(-0.016960617939478638, abs=1e-12)
    assert result["null_distribution"][:3].tolist() == pytest.approx(
        [0.4826049207350477, 0.47584428019889907, 0.47672352362500003], abs=1e-12
    )
    assert np.array_equal(result["null_distribution"], continuous)
    assert not np.array_equal(result["null_distribution"], legacy_reseeded)
    assert result["null_distribution"][0] != pytest.approx(
        0.4718767931766301, abs=1e-12
    )
    expected_digest = hashlib.sha256(
        np.asarray(result["null_distribution"], dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert result["null_distribution_sha256"] == expected_digest


class _TinyBatchNormOperatorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_proxies = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.2], [-0.1, 1.0], [0.7, -0.6], [-0.8, -0.3]],
                dtype=torch.float64,
            )
        )
        self.bn = torch.nn.BatchNorm1d(2, dtype=torch.float64)
        self.head = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
        self.head.weight.data.copy_(
            torch.tensor([[1.0, 0.3], [-0.2, 0.8]], dtype=torch.float64)
        )
        self.forward_modes = []

    def forward(self, values):
        self.forward_modes.append(bool(self.training))
        return self.head(self.bn(values))


def _tiny_bn_rows():
    return torch.tensor(
        [[2.0, -0.5], [-0.4, 1.7], [0.3, -1.1], [1.1, 0.8]], dtype=torch.float64
    )


def _tensor_byte_identity(value):
    array = value.detach().cpu().contiguous().numpy()
    return str(value.dtype), tuple(value.shape), array.tobytes(order="C")


class _SignedZeroMutatingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sentinel = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))

    def forward(self, values):
        with torch.no_grad():
            self.sentinel[0] = -0.0
        return values + self.sentinel * 0.0


def test_train_graph_rejects_signed_zero_parameter_bit_mutation():
    model = _SignedZeroMutatingModel()
    before = _tensor_byte_identity(model.sentinel)
    negative_zero = torch.tensor([-0.0], dtype=torch.float64)
    assert torch.equal(model.sentinel, negative_zero)
    assert _tensor_byte_identity(negative_zero) != before
    with pytest.raises(ValueError, match="parameter.*byte-identical"):
        MODULE.bufferless_train_embeddings(model, torch.ones((2, 1), dtype=torch.float64))
    assert _tensor_byte_identity(model.sentinel) == before


def test_train_bn_forward_uses_disposable_buffers_and_restores_flags():
    model = _TinyBatchNormOperatorModel().train()
    inputs = _tiny_bn_rows()
    before_parameters = {
        name: _tensor_byte_identity(value) for name, value in model.named_parameters()
    }
    before_buffers = {
        name: _tensor_byte_identity(value) for name, value in model.named_buffers()
    }
    before_flags = tuple(module.training for module in model.modules())
    graph = MODULE.bufferless_train_embeddings(model, inputs)
    assert graph.changed_buffer_names == (
        "bn.num_batches_tracked",
        "bn.running_mean",
        "bn.running_var",
    )
    for name, value in model.named_parameters():
        assert _tensor_byte_identity(value) == before_parameters[name]
    for name, value in model.named_buffers():
        assert _tensor_byte_identity(value) == before_buffers[name]
    assert tuple(module.training for module in model.modules()) == before_flags
    model.eval()
    with torch.no_grad():
        eval_embeddings = model(inputs)
    model.train()
    assert not torch.allclose(graph.embeddings, eval_embeddings, atol=1e-12, rtol=0.0)


def test_score_context_uses_one_shared_train_graph_and_one_clean_baseline():
    model = _TinyBatchNormOperatorModel().train()
    before_parameters = {
        name: _tensor_byte_identity(value) for name, value in model.named_parameters()
    }
    before_buffers = {
        name: _tensor_byte_identity(value) for name, value in model.named_buffers()
    }
    before_flags = tuple(module.training for module in model.modules())
    result = MODULE.score_context(
        model=model,
        train_inputs=_tiny_bn_rows(),
        clean_inputs=torch.tensor(
            [[1.7, -0.2], [-0.1, 1.4], [0.5, -0.8], [0.9, 0.6]], dtype=torch.float64
        ),
        labels=torch.tensor([10, 20, 10, 30]),
        sample_indices=torch.tensor([8, 9, 4, 1]),
        proxy_labels=torch.tensor([10, 20, 30, 40]),
        expected_trainable_parameter_names=(
            "bn.bias",
            "bn.weight",
            "head.weight",
            "metric_proxies",
        ),
        proxy_parameter_name="metric_proxies",
        alpha=1.3,
        delta=0.2,
        learning_rate=0.01,
        coalition_weight=0.2,
        proxy_learning_rate_multiplier=2.0,
        context_index=0,
    )
    assert model.forward_modes.count(True) == 1
    assert model.forward_modes.count(False) == 25
    assert tuple(result["losses"]) == OPERATORS
    assert set(result["gradients"]) == {
        f"{operator}.{panel}" for operator in OPERATORS for panel in PANELS
    }
    assert sum(
        len(result["outcomes"][regime][operator])
        for regime in REGIMES
        for operator in OPERATORS
    ) == 24
    assert result["train_graph"].changed_buffer_names
    assert result["train_graph"].embeddings.requires_grad is False
    assert all(loss.requires_grad is False for loss in result["losses"].values())
    assert result["shared_confuser"]["null_distribution"].shape == (256,)
    for name, value in model.named_parameters():
        assert _tensor_byte_identity(value) == before_parameters[name]
    for name, value in model.named_buffers():
        assert _tensor_byte_identity(value) == before_buffers[name]
    assert tuple(module.training for module in model.modules()) == before_flags
