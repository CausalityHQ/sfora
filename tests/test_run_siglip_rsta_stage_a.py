"""Tests for the outcome-blind SigLIP RSTA Stage-A execution controller."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from sfora.pass209_m4 import canonical_json_bytes

_CONTROL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_siglip_proxy_control.py"
_CONTROL_SPEC = importlib.util.spec_from_file_location("run_siglip_proxy_control", _CONTROL_SCRIPT)
assert _CONTROL_SPEC is not None and _CONTROL_SPEC.loader is not None
_CONTROL_MODULE = importlib.util.module_from_spec(_CONTROL_SPEC)
sys.modules[_CONTROL_SPEC.name] = _CONTROL_MODULE
_CONTROL_SPEC.loader.exec_module(_CONTROL_MODULE)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_siglip_rsta_stage_a.py"
_SCIENTIFIC_CLI = _SCRIPT.with_name("diagnose_siglip_rsta_stage_a.py")
_SPEC = importlib.util.spec_from_file_location("run_siglip_rsta_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_direct_controller_script_reaches_authority_validation_without_name_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--seed-receipt",
            str(missing / "17.json"),
            "--seed-receipt",
            str(missing / "29.json"),
            "--seed-receipt",
            str(missing / "43.json"),
            "--aggregate-receipt",
            str(missing / "aggregate.json"),
            "--checkpoint",
            str(missing / "17.pt"),
            "--checkpoint",
            str(missing / "29.pt"),
            "--checkpoint",
            str(missing / "43.pt"),
            "--control-manifest",
            str(missing / "manifest.json"),
            "--optimization-image-root",
            str(missing / "images"),
            "--scientific-cli",
            str(missing / "diagnose.py"),
            "--scratch-root",
            str(tmp_path / "scratch"),
            "--result-output",
            str(tmp_path / "result.json"),
            "--terminal-output",
            str(tmp_path / "terminal.json"),
            "--expected-hostname",
            socket.gethostname(),
            "--expected-source-commit",
            "1" * 40,
            "--expected-controller-source-commit",
            "2" * 40,
            "--execute-controller",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "NameError" not in completed.stderr
    assert json.loads((tmp_path / "result.json").read_bytes()) == {
        "claim_eligible": False,
        "first_decisive_clause": "authority-mismatch",
        "schema": "siglip-rsta-stage-a-result-v1",
        "verdict": "INVALID",
    }
    assert not (tmp_path / "terminal.json").exists()


def _environment() -> dict[str, object]:
    return {
        "source_revision": "1" * 40,
        "source_tree_digest": "2" * 64,
        "manifest_sha256": "3" * 64,
        "torch_version": "2.8.0",
        "transformers_version": "4.56.2",
        "torchvision_version": "0.23.0",
        "cuda_runtime": "13.0",
        "device_name": "NVIDIA GB10",
        "microbatch_size": 30,
        "steps_per_epoch": 4,
        "evaluation_batch_size": 64,
        "query_block": 4096,
    }


def _seed_receipt(seed: int, checkpoint: bytes) -> bytes:
    def band(recall: float) -> dict[str, dict[str, float]]:
        return {
            "raw": {"recall_at_1": recall},
            "projected": {"recall_at_1": recall},
        }

    config = _CONTROL_MODULE.SiglipProxyControlConfig()
    return canonical_json_bytes(
        {
            "schema": "sfora-siglip-proxy-control-seed-v1",
            "claim_eligible": False,
            "seed": seed,
            "source": {"revision": "1" * 40, "tree_digest": "2" * 64, "dirty": False},
            "dataset": {
                "name": config.dataset_name,
                "revision": config.dataset_revision,
                "manifest_sha256": "3" * 64,
                "optimization_examples": 4,
                "clean_validation_examples": 1,
                "burned_diagnostic_examples": 1,
            },
            "model": {
                "name": config.model_name,
                "revision": config.model_revision,
                "resolved_revision": config.model_revision,
                "initial_state_sha256": f"{seed:064x}",
            },
            "config": _CONTROL_MODULE._json_compatible(vars(config)),
            "config_sha256": _CONTROL_MODULE._config_sha256(config),
            "smoke": {
                "observations": [],
                "projected_seed_seconds": 1.0,
                "selected_microbatch_size": 30,
                "sha256": "7" * 64,
            },
            "evaluation": {
                "initial": {"clean_validation": band(0.90)},
                "final": {"clean_validation": band(0.94)},
            },
            "changes": {"memorization_to_transfer_ratio": None},
            "training": {
                "optimizer_steps": 240,
                "steps_per_epoch": 4,
                "microbatch_size": 30,
                "final_objective": float(seed),
                "maximum_score_disagreement": 0.0,
            },
            "checkpoint": {
                "basename": f"seed-{seed}-epoch-060.pt",
                "receipt_basename": f"seed-{seed}-epoch-060.checkpoint.json",
                "sha256": hashlib.sha256(checkpoint).hexdigest(),
                "bytes": len(checkpoint),
                "epoch": 60,
            },
            "resources": {
                "wall_seconds": 1.0,
                "examples_per_second": 1.0,
                "peak_process_rss_bytes": 1,
                "peak_cuda_allocated_bytes": 1,
                "peak_cuda_reserved_bytes": 1,
            },
            "environment": _environment(),
        }
    )


def _authority_bundle(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [
        {"example_id": "opt-0-a", "label": 0},
        {"example_id": "opt-0-b", "label": 0},
        {"example_id": "opt-1-a", "label": 1},
        {"example_id": "opt-1-b", "label": 1},
        {"example_id": "clean-a", "label": 49},
        {"example_id": "burned-a", "label": 82},
    ]
    environment = _environment()
    environment["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes({"examples": rows})
    ).hexdigest()
    run_authority = _CONTROL_MODULE.ControlRunAuthority(**environment)
    run_authority_sha256 = _CONTROL_MODULE._run_authority_sha256(run_authority)
    config_sha256 = _CONTROL_MODULE._config_sha256(_CONTROL_MODULE.SiglipProxyControlConfig())

    def checkpoint(seed: int) -> bytes:
        stream = io.BytesIO()
        torch.save(
            {
                "claim_eligible": False,
                "completed_epoch": 60,
                "config_sha256": config_sha256,
                "cpu_rng_state": torch.zeros(4, dtype=torch.uint8),
                "cuda_rng_states": (torch.zeros(4, dtype=torch.uint8),),
                "final_objective": float(seed),
                "initial_snapshot_sha256": f"{seed:064x}",
                "maximum_score_disagreement": 0.0,
                "model_state": OrderedDict({"weight": torch.zeros(1)}),
                "optimizer_state": {},
                "run_authority_sha256": run_authority_sha256,
                "sampler_cycles": (0,) * 49,
                "sampler_positions": (0,) * 49,
                "schema": "sfora-siglip-proxy-checkpoint-payload-v1",
                "seed": seed,
            },
            stream,
        )
        return stream.getvalue()

    checkpoints = tuple(checkpoint(seed) for seed in (17, 29, 43))

    def receipt(seed: int, checkpoint: bytes) -> bytes:
        value = json.loads(_seed_receipt(seed, checkpoint))
        value["dataset"]["manifest_sha256"] = environment["manifest_sha256"]
        value["environment"] = environment
        return canonical_json_bytes(value)

    receipts = tuple(
        receipt(seed, raw) for seed, raw in zip((17, 29, 43), checkpoints, strict=True)
    )
    receipt_paths = tuple(tmp_path / f"seed-{seed}.json" for seed in (17, 29, 43))
    checkpoint_paths = tuple(tmp_path / f"seed-{seed}-epoch-060.pt" for seed in (17, 29, 43))
    for path, raw in zip(receipt_paths, receipts, strict=True):
        path.write_bytes(raw)
    for path, raw in zip(checkpoint_paths, checkpoints, strict=True):
        path.write_bytes(raw)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_bytes(_CONTROL_MODULE.control_aggregate_receipt_bytes(receipts))
    manifest_path = tmp_path / "control-manifest.json"
    manifest_path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "sfora-siglip-proxy-control-manifest-v1",
                "claim_eligible": False,
                "dataset_id": _CONTROL_MODULE.SiglipProxyControlConfig().dataset_name,
                "dataset_revision": _CONTROL_MODULE.SiglipProxyControlConfig().dataset_revision,
                "examples": rows,
            }
        )
    )
    return receipt_paths, aggregate_path, checkpoint_paths, manifest_path


def test_authority_projection_authenticates_aggregate_and_emits_only_optimization_roles(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)

    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )

    binding = json.loads(projected.control_binding_bytes)
    manifest = json.loads(projected.optimization_manifest_bytes)
    assert binding["schema"] == "rsta-control-binding-v1"
    assert binding["selected_microbatch_size"] == 30
    assert [row["seed"] for row in binding["checkpoints"]] == [17, 29, 43]
    assert manifest == {
        "claim_eligible": False,
        "dataset_id": "tanganke/stanford_cars",
        "dataset_revision": _CONTROL_MODULE.SiglipProxyControlConfig().dataset_revision,
        "examples": [
            {"example_id": "opt-0-a", "label": 0},
            {"example_id": "opt-0-b", "label": 0},
            {"example_id": "opt-1-a", "label": 1},
            {"example_id": "opt-1-b", "label": 1},
        ],
        "schema": "rsta-optimization-manifest-v1",
    }
    forbidden = {"evaluation", "changes", "clean", "burned", "test", "recall", "verdict"}
    assert not any(word in projected.control_binding_bytes.decode() for word in forbidden)
    assert not any(word in projected.optimization_manifest_bytes.decode() for word in forbidden)


def test_authority_projection_rejects_aggregate_checkpoint_and_microbatch_drift(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)

    aggregate_path.write_bytes(aggregate_path.read_bytes().replace(b"0.94", b"0.93", 1))
    with pytest.raises(ValueError, match="aggregate"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_rejects_manifest_dataset_identity_drift(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["dataset_revision"] = "9" * 40
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(ValueError, match="dataset identity"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_binds_dataset_identity_to_frozen_config(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    for path in receipt_paths:
        receipt = json.loads(path.read_bytes())
        receipt["dataset"]["revision"] = "9" * 40
        path.write_bytes(canonical_json_bytes(receipt))
    manifest = json.loads(manifest_path.read_bytes())
    manifest["dataset_revision"] = "9" * 40
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="dataset identity"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_binds_model_identity_to_frozen_config(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    for path in receipt_paths:
        receipt = json.loads(path.read_bytes())
        receipt["model"]["name"] = "wrong/model"
        path.write_bytes(canonical_json_bytes(receipt))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="model identity"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_reconstructs_frozen_config_and_run_authority(
    tmp_path: Path,
) -> None:
    for role in ("config", "run"):
        receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
            tmp_path / role
        )
        for path in receipt_paths:
            receipt = json.loads(path.read_bytes())
            if role == "config":
                receipt["config"]["train_epochs"] = 59
            else:
                receipt["environment"]["steps_per_epoch"] = False
            path.write_bytes(canonical_json_bytes(receipt))
        aggregate_path.write_bytes(
            _CONTROL_MODULE.control_aggregate_receipt_bytes(
                tuple(path.read_bytes() for path in receipt_paths)
            )
        )

        with pytest.raises(ValueError, match=role):
            _MODULE.project_stage_a_authority(
                seed_receipts=receipt_paths,
                aggregate_receipt=aggregate_path,
                checkpoints=checkpoint_paths,
                control_manifest=manifest_path,
            )


def test_authority_projection_cross_binds_run_authority_to_source_and_manifest(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    for path in receipt_paths:
        receipt = json.loads(path.read_bytes())
        receipt["source"]["revision"] = "8" * 40
        path.write_bytes(canonical_json_bytes(receipt))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="run authority"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_rejects_checkpoint_payload_finality_drift(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    payload = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=True)
    payload["completed_epoch"] = 59
    stream = io.BytesIO()
    torch.save(payload, stream)
    drifted = stream.getvalue()
    checkpoint_paths[0].write_bytes(drifted)
    receipt = json.loads(receipt_paths[0].read_bytes())
    receipt["checkpoint"]["sha256"] = hashlib.sha256(drifted).hexdigest()
    receipt["checkpoint"]["bytes"] = len(drifted)
    receipt_paths[0].write_bytes(canonical_json_bytes(receipt))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="checkpoint payload"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_rejects_official_test_labels_in_full_manifest(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_bytes())
    next(row for row in manifest["examples"] if row["label"] == 49)["label"] = 98
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes({"examples": manifest["examples"]})
    ).hexdigest()
    for receipt_path, checkpoint_path in zip(receipt_paths, checkpoint_paths, strict=True):
        receipt = json.loads(receipt_path.read_bytes())
        receipt["dataset"]["manifest_sha256"] = manifest_sha256
        receipt["environment"]["manifest_sha256"] = manifest_sha256
        run_authority = _CONTROL_MODULE.ControlRunAuthority(**receipt["environment"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        payload["run_authority_sha256"] = _CONTROL_MODULE._run_authority_sha256(run_authority)
        stream = io.BytesIO()
        torch.save(payload, stream)
        checkpoint = stream.getvalue()
        checkpoint_path.write_bytes(checkpoint)
        receipt["checkpoint"]["sha256"] = hashlib.sha256(checkpoint).hexdigest()
        receipt["checkpoint"]["bytes"] = len(checkpoint)
        receipt_path.write_bytes(canonical_json_bytes(receipt))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="manifest row"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_authority_projection_recomputes_all_control_band_counts(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_bytes())
    next(row for row in manifest["examples"] if row["label"] == 49)["label"] = 82
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    manifest_sha256 = hashlib.sha256(
        canonical_json_bytes({"examples": manifest["examples"]})
    ).hexdigest()
    for receipt_path, checkpoint_path in zip(receipt_paths, checkpoint_paths, strict=True):
        receipt = json.loads(receipt_path.read_bytes())
        receipt["dataset"]["manifest_sha256"] = manifest_sha256
        receipt["environment"]["manifest_sha256"] = manifest_sha256
        run_authority = _CONTROL_MODULE.ControlRunAuthority(**receipt["environment"])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        payload["run_authority_sha256"] = _CONTROL_MODULE._run_authority_sha256(run_authority)
        stream = io.BytesIO()
        torch.save(payload, stream)
        checkpoint = stream.getvalue()
        checkpoint_path.write_bytes(checkpoint)
        receipt["checkpoint"]["sha256"] = hashlib.sha256(checkpoint).hexdigest()
        receipt["checkpoint"]["bytes"] = len(checkpoint)
        receipt_path.write_bytes(canonical_json_bytes(receipt))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )

    with pytest.raises(ValueError, match="band counts"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )

    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "checkpoint"
    )
    checkpoint_paths[0].write_bytes(b"drift")
    with pytest.raises(ValueError, match="checkpoint"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )

    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "microbatch"
    )
    for path in receipt_paths:
        value = json.loads(path.read_bytes())
        value["smoke"]["selected_microbatch_size"] = 20
        path.write_bytes(canonical_json_bytes(value))
    aggregate_path.write_bytes(
        _CONTROL_MODULE.control_aggregate_receipt_bytes(
            tuple(path.read_bytes() for path in receipt_paths)
        )
    )
    with pytest.raises(ValueError, match="microbatch"):
        _MODULE.project_stage_a_authority(
            seed_receipts=receipt_paths,
            aggregate_receipt=aggregate_path,
            checkpoints=checkpoint_paths,
            control_manifest=manifest_path,
        )


def test_controller_cli_is_closed_and_refuses_scientific_or_forbidden_overrides() -> None:
    argv = [
        "--seed-receipt",
        "/evidence/17.json",
        "--seed-receipt",
        "/evidence/29.json",
        "--seed-receipt",
        "/evidence/43.json",
        "--aggregate-receipt",
        "/evidence/aggregate.json",
        "--checkpoint",
        "/evidence/17.pt",
        "--checkpoint",
        "/evidence/29.pt",
        "--checkpoint",
        "/evidence/43.pt",
        "--control-manifest",
        "/evidence/manifest.json",
        "--optimization-image-root",
        "/evidence/optimization-images",
        "--scientific-cli",
        "/source/scripts/diagnose_siglip_rsta_stage_a.py",
        "--scratch-root",
        "/scratch",
        "--result-output",
        "/terminal/result.json",
        "--terminal-output",
        "/terminal/failure.json",
        "--expected-hostname",
        "dgx-stage-a",
        "--expected-source-commit",
        "1" * 40,
        "--expected-controller-source-commit",
        "2" * 40,
        "--execute-controller",
    ]
    parsed = _MODULE.parse_controller_args(argv)
    assert parsed.seed_receipt == [Path(value) for value in argv[1:6:2]]
    assert parsed.checkpoint == [Path(value) for value in argv[9:14:2]]
    assert parsed.expected_hostname == "dgx-stage-a"
    assert parsed.expected_source_commit == "1" * 40
    assert parsed.expected_controller_source_commit == "2" * 40

    for forbidden in (
        "--clean-root",
        "--burned-root",
        "--test-root",
        "--backend",
        "--seed",
        "--threshold",
        "--restart",
        "--s3-uri",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_controller_args([*argv, forbidden, "value"])
    with pytest.raises(SystemExit):
        _MODULE.parse_controller_args([*argv, "--result-output", "/other.json"])


def test_main_rejects_wrong_host_before_reading_any_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE.socket, "gethostname", lambda: "actual-host")
    monkeypatch.setattr(
        _MODULE,
        "project_stage_a_authority",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("authority was read")),
    )
    missing = tmp_path / "missing"
    argv = [
        "--seed-receipt",
        str(missing / "17.json"),
        "--seed-receipt",
        str(missing / "29.json"),
        "--seed-receipt",
        str(missing / "43.json"),
        "--aggregate-receipt",
        str(missing / "aggregate.json"),
        "--checkpoint",
        str(missing / "17.pt"),
        "--checkpoint",
        str(missing / "29.pt"),
        "--checkpoint",
        str(missing / "43.pt"),
        "--control-manifest",
        str(missing / "manifest.json"),
        "--optimization-image-root",
        str(missing / "images"),
        "--scientific-cli",
        str(_SCIENTIFIC_CLI),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--result-output",
        str(tmp_path / "result.json"),
        "--terminal-output",
        str(tmp_path / "terminal.json"),
        "--expected-hostname",
        "expected-host",
        "--expected-source-commit",
        "1" * 40,
        "--expected-controller-source-commit",
        "2" * 40,
        "--execute-controller",
    ]

    assert _MODULE.main(argv) == 1
    assert json.loads((tmp_path / "result.json").read_bytes())["first_decisive_clause"] == (
        "authority-mismatch"
    )
    assert not (tmp_path / "terminal.json").exists()


def test_main_rejects_wrong_controller_source_before_reading_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_MODULE.socket, "gethostname", lambda: "expected-host")
    monkeypatch.setattr(_MODULE, "_current_controller_source", lambda: ("3" * 40, True))
    monkeypatch.setattr(
        _MODULE,
        "project_stage_a_authority",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("authority was read")),
    )
    missing = tmp_path / "missing"
    argv = [
        "--seed-receipt",
        str(missing / "17.json"),
        "--seed-receipt",
        str(missing / "29.json"),
        "--seed-receipt",
        str(missing / "43.json"),
        "--aggregate-receipt",
        str(missing / "aggregate.json"),
        "--checkpoint",
        str(missing / "17.pt"),
        "--checkpoint",
        str(missing / "29.pt"),
        "--checkpoint",
        str(missing / "43.pt"),
        "--control-manifest",
        str(missing / "manifest.json"),
        "--optimization-image-root",
        str(missing / "images"),
        "--scientific-cli",
        str(_SCIENTIFIC_CLI),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--result-output",
        str(tmp_path / "result.json"),
        "--terminal-output",
        str(tmp_path / "terminal.json"),
        "--expected-hostname",
        "expected-host",
        "--expected-source-commit",
        "1" * 40,
        "--expected-controller-source-commit",
        "2" * 40,
        "--execute-controller",
    ]

    assert _MODULE.main(argv) == 1
    assert json.loads((tmp_path / "result.json").read_bytes())["first_decisive_clause"] == (
        "authority-mismatch"
    )


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (_MODULE.StageAProcessObservation(96 * 1024**3, 1, 0, 0, 1, 1), "memory-cap"),
        (_MODULE.StageAProcessObservation(1, 1, 1, 0, 1, 1), "memory-pressure"),
        (_MODULE.StageAProcessObservation(1, 1, 0, 1, 1, 1), "swap-growth"),
        (_MODULE.StageAProcessObservation(1, 1, 0, 0, 3_600_000_000_001, 1), "timeout"),
        (_MODULE.StageAProcessObservation(1, 1, 0, 0, 1, 300_000_000_001), "progress"),
    ],
)
def test_process_stop_authority_is_exact_and_fail_closed(observation, reason: str) -> None:
    assert _MODULE.stage_a_process_stop_reason(observation) == reason
    assert (
        _MODULE.stage_a_process_stop_reason(_MODULE.StageAProcessObservation(1, 1, 0, 0, 1, 1))
        is None
    )


def test_process_stop_rejects_the_exact_control_memory_limit() -> None:
    assert (
        _MODULE.stage_a_process_stop_reason(
            _MODULE.StageAProcessObservation(96 * 1024**3, 0, 0, 0, 1, 1)
        )
        == "memory-cap"
    )


def test_controller_failure_preserves_terminal_cleans_named_scratch_and_never_restarts(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "authority"
    )
    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    result = tmp_path / "result.json"
    terminal = tmp_path / "terminal.json"
    calls: list[tuple[str, ...]] = []
    failure_bytes = canonical_json_bytes(
        {"claim_eligible": False, "reason": "child-exit", "schema": "rsta-terminal-v1"}
    )

    def child(argv: tuple[str, ...], _cwd: Path) -> bytes:
        calls.append(argv)
        raise _MODULE.StageAChildFailure(failure_bytes)

    controller = _MODULE.StageAController(
        scratch_root=tmp_path / "scratch",
        result_output=result,
        terminal_output=terminal,
    )
    with pytest.raises(_MODULE.StageAChildFailure):
        controller.execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=_SCIENTIFIC_CLI,
            child_runner=child,
        )
    assert len(calls) == 1
    child_argv = calls[0]
    assert str(_SCIENTIFIC_CLI) in child_argv
    assert not any(str(path) in child_argv for path in receipt_paths)
    assert str(aggregate_path) not in child_argv
    assert not any(
        flag in child_argv
        for flag in ("--clean-root", "--burned-root", "--test-root", "--backend", "--threshold")
    )
    assert terminal.read_bytes() == failure_bytes
    assert not result.exists()
    assert list((tmp_path / "scratch").iterdir()) == []
    with pytest.raises(RuntimeError, match="already started"):
        controller.execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=_SCIENTIFIC_CLI,
            child_runner=child,
        )
    assert len(calls) == 1


def test_controller_rejects_any_unregistered_scientific_cli(tmp_path: Path) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "authority"
    )
    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    wrong_cli = tmp_path / "diagnose_siglip_rsta_stage_a.py"
    wrong_cli.write_text("raise SystemExit(0)\n")

    with pytest.raises(ValueError, match="scientific CLI"):
        _MODULE.StageAController(
            scratch_root=tmp_path / "scratch",
            result_output=tmp_path / "result.json",
            terminal_output=tmp_path / "terminal.json",
        ).execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=wrong_cli,
            child_runner=lambda _argv, _cwd: canonical_json_bytes({"complete": True}),
        )


def test_child_process_is_one_owned_group_and_stops_before_publishing_partial() -> None:
    launches: list[tuple[tuple[str, ...], dict[str, object]]] = []
    terminated: list[int] = []

    class FakeProcess:
        pid = 9123

        def poll(self):
            return None

        def communicate(self, timeout=None):
            return b"partial-science", b"resource stop"

    def popen(argv, **kwargs):
        launches.append((tuple(argv), kwargs))
        return FakeProcess()

    with pytest.raises(_MODULE.StageAChildFailure) as captured:
        _MODULE.run_stage_a_child_process(
            ("python", "diagnose.py"),
            cwd=Path("/scratch"),
            popen_factory=popen,
            sample=lambda _pid: _MODULE.StageAProcessObservation(96 * 1024**3, 1, 0, 0, 1, 1),
            terminate_group=lambda pid: terminated.append(pid),
            sleep=lambda _seconds: None,
        )
    assert len(launches) == 1
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["cwd"] == Path("/scratch")
    assert launches[0][1]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert launches[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert launches[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "AWS_PROFILE" not in launches[0][1]["env"]
    assert terminated == [9123]
    terminal = json.loads(captured.value.terminal_bytes)
    assert terminal["reason"] == "memory-cap"
    assert "partial-science" not in captured.value.terminal_bytes.decode()


def test_child_process_terminates_group_when_resource_sampling_fails() -> None:
    terminated: list[int] = []

    class FakeProcess:
        pid = 7123

        def poll(self):
            return None

        def communicate(self, timeout=None):
            return b"partial-science", b"monitor failed"

    with pytest.raises(_MODULE.StageAChildFailure) as captured:
        _MODULE.run_stage_a_child_process(
            ("python", "diagnose.py"),
            cwd=Path("/scratch"),
            popen_factory=lambda *_args, **_kwargs: FakeProcess(),
            sample=lambda _pid: (_ for _ in ()).throw(RuntimeError("nvidia-smi failed")),
            terminate_group=lambda pid: terminated.append(pid),
            sleep=lambda _seconds: None,
        )

    assert terminated == [7123]
    assert json.loads(captured.value.terminal_bytes)["reason"] == "monitor-error"


def test_child_process_escalates_to_sigkill_when_sigterm_does_not_complete() -> None:
    terminated: list[int] = []
    killed: list[int] = []

    class FakeProcess:
        pid = 8123

        def __init__(self) -> None:
            self.communications = 0

        def poll(self):
            return None

        def communicate(self, timeout=None):
            self.communications += 1
            if self.communications == 1:
                raise subprocess.TimeoutExpired(("python", "diagnose.py"), timeout)
            return b"partial-science", b"resource stop"

    with pytest.raises(_MODULE.StageAChildFailure):
        _MODULE.run_stage_a_child_process(
            ("python", "diagnose.py"),
            cwd=Path("/scratch"),
            popen_factory=lambda *_args, **_kwargs: FakeProcess(),
            sample=lambda _pid: _MODULE.StageAProcessObservation(96 * 1024**3, 1, 0, 0, 1, 1),
            terminate_group=lambda pid: terminated.append(pid),
            kill_group=lambda pid: killed.append(pid),
            sleep=lambda _seconds: None,
        )

    assert terminated == [8123]
    assert killed == [8123]


def test_controller_preserves_terminal_when_child_returns_invalid_partial(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "authority"
    )
    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    terminal = tmp_path / "terminal.json"
    controller = _MODULE.StageAController(
        scratch_root=tmp_path / "scratch",
        result_output=tmp_path / "result.json",
        terminal_output=terminal,
    )

    with pytest.raises(_MODULE.StageAChildFailure):
        controller.execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=_SCIENTIFIC_CLI,
            child_runner=lambda _argv, _cwd: b"partial",
        )
    assert json.loads(terminal.read_bytes())["reason"] == "invalid-child-result"
    assert not (tmp_path / "result.json").exists()
    assert list((tmp_path / "scratch").iterdir()) == []


def test_controller_rejects_canonical_output_that_is_not_a_stage_a_result(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "authority"
    )
    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    result = tmp_path / "result.json"
    terminal = tmp_path / "terminal.json"

    with pytest.raises(_MODULE.StageAChildFailure):
        _MODULE.StageAController(
            scratch_root=tmp_path / "scratch",
            result_output=result,
            terminal_output=terminal,
        ).execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=_SCIENTIFIC_CLI,
            child_runner=lambda _argv, _cwd: canonical_json_bytes({"complete": True}),
        )

    assert json.loads(terminal.read_bytes())["reason"] == "invalid-child-result"
    assert not result.exists()


def test_controller_cleanup_does_not_mask_terminal_if_child_writes_scratch(
    tmp_path: Path,
) -> None:
    receipt_paths, aggregate_path, checkpoint_paths, manifest_path = _authority_bundle(
        tmp_path / "authority"
    )
    projected = _MODULE.project_stage_a_authority(
        seed_receipts=receipt_paths,
        aggregate_receipt=aggregate_path,
        checkpoints=checkpoint_paths,
        control_manifest=manifest_path,
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    terminal = tmp_path / "terminal.json"
    failure = canonical_json_bytes(
        {"claim_eligible": False, "reason": "child-exit", "schema": "rsta-terminal-v1"}
    )

    def child(_argv: tuple[str, ...], cwd: Path) -> bytes:
        (cwd / "unexpected-child-file").write_bytes(b"partial")
        raise _MODULE.StageAChildFailure(failure)

    with pytest.raises(_MODULE.StageAChildFailure):
        _MODULE.StageAController(
            scratch_root=tmp_path / "scratch",
            result_output=tmp_path / "result.json",
            terminal_output=terminal,
        ).execute(
            projected,
            optimization_image_root=image_root,
            scientific_cli=_SCIENTIFIC_CLI,
            child_runner=child,
        )

    assert terminal.read_bytes() == failure
    assert list((tmp_path / "scratch").iterdir()) == []
