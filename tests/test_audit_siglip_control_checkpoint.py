from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch

_ROOT = Path(__file__).parents[1]
_CONTROL_SCRIPT = _ROOT / "scripts" / "run_siglip_proxy_control.py"
_CONTROL_SPEC = importlib.util.spec_from_file_location(
    "audit_fixture_run_siglip_proxy_control", _CONTROL_SCRIPT
)
assert _CONTROL_SPEC is not None and _CONTROL_SPEC.loader is not None
_CONTROL = importlib.util.module_from_spec(_CONTROL_SPEC)
sys.modules[_CONTROL_SPEC.name] = _CONTROL
_CONTROL_SPEC.loader.exec_module(_CONTROL)

_SCRIPT = _ROOT / "scripts" / "audit_siglip_control_checkpoint.py"
_SPEC = importlib.util.spec_from_file_location("audit_siglip_control_checkpoint", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _band(correct: int, queries: int) -> dict[str, float | int]:
    return {
        "correct": correct,
        "queries": queries,
        "recall_at_1": correct / queries,
        "mean_nearest_positive_cosine": 0.9,
        "mean_nearest_negative_cosine": 0.8,
        "mean_margin": 0.1,
    }


def _snapshot(*, burned_correct: int) -> dict[str, object]:
    return {
        "optimization": {"raw": _band(3_880, 3_963), "projected": _band(3_885, 3_963)},
        "clean_validation": {"raw": _band(2_596, 2_746), "projected": _band(2_596, 2_746)},
        "burned_diagnostic": {
            "raw": _band(burned_correct, 1_345),
            "projected": _band(burned_correct, 1_345),
        },
    }


def _campaign_fixture(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = _CONTROL.SiglipProxyControlConfig()
    config_payload = cast(dict[str, object], _CONTROL._json_compatible(vars(config)))
    run_authority = _CONTROL.ControlRunAuthority(
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        manifest_sha256="3" * 64,
        torch_version=str(torch.__version__),
        transformers_version="fixture-transformers",
        torchvision_version="fixture-torchvision",
        cuda_runtime=None,
        device_name="fixture-device",
        microbatch_size=30,
        steps_per_epoch=33,
        evaluation_batch_size=32,
        query_block=128,
    )
    checkpoint_directory = tmp_path / "checkpoints"
    checkpoint_directory.mkdir()
    checkpoint = checkpoint_directory / "seed-017-epoch-060.pt"
    checkpoint.write_bytes(b"authenticated-checkpoint-fixture")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_receipt = checkpoint_directory / "seed-017-epoch-060.checkpoint.json"
    checkpoint_receipt.write_bytes(
        _CONTROL._canonical_bytes(
            {
                "bytes": checkpoint.stat().st_size,
                "checkpoint": checkpoint.name,
                "claim_eligible": False,
                "epoch": 60,
                "schema": "sfora-siglip-proxy-checkpoint-v1",
                "seed": 17,
                "sha256": checkpoint_sha256,
            }
        )
    )

    receipt_values: list[dict[str, Any]] = []
    receipt_paths: list[Path] = []
    for seed, burned_correct in ((17, 1_258), (29, 1_248), (43, 1_250)):
        checkpoint_value = {
            "basename": f"seed-{seed:03d}-epoch-060.pt",
            "receipt_basename": f"seed-{seed:03d}-epoch-060.checkpoint.json",
            "sha256": checkpoint_sha256 if seed == 17 else f"{seed:064x}",
            "bytes": checkpoint.stat().st_size if seed == 17 else 100 + seed,
            "epoch": 60,
        }
        value = {
            "schema": "sfora-siglip-proxy-control-seed-v1",
            "claim_eligible": False,
            "seed": seed,
            "source": {"revision": "1" * 40, "tree_digest": "2" * 64, "dirty": False},
            "dataset": {
                "name": config.dataset_name,
                "revision": config.dataset_revision,
                "manifest_sha256": "3" * 64,
                "optimization_examples": 3_963,
                "clean_validation_examples": 2_746,
                "burned_diagnostic_examples": 1_345,
            },
            "model": {
                "name": config.model_name,
                "revision": config.model_revision,
                "resolved_revision": config.model_revision,
                "initial_state_sha256": f"{seed:064x}",
            },
            "config": config_payload,
            "config_sha256": _CONTROL._config_sha256(config),
            "smoke": {
                "observations": [],
                "projected_seed_seconds": 1.0,
                "selected_microbatch_size": 30,
                "sha256": "4" * 64,
            },
            "evaluation": {
                "initial": _snapshot(burned_correct=1_240),
                "final": _snapshot(burned_correct=burned_correct),
            },
            "changes": {
                "train_margin_change": 0.1,
                "clean_recall_change": 0.01,
                "clean_margin_change": 0.01,
                "burned_margin_change": 0.01,
                "memorization_to_transfer_ratio": 0.1,
                "transfer_mechanism_conclusion_supported": False,
            },
            "training": {
                "optimizer_steps": 1_980,
                "steps_per_epoch": 33,
                "microbatch_size": 30,
                "final_objective": 0.2,
                "maximum_score_disagreement": 0.0,
            },
            "checkpoint": checkpoint_value,
            "resources": {
                "wall_seconds": 1.0,
                "examples_per_second": 2.0,
                "peak_process_rss_bytes": 3,
                "peak_cuda_allocated_bytes": 4,
                "peak_cuda_reserved_bytes": 5,
            },
            "environment": vars(run_authority),
        }
        path = tmp_path / f"seed-{seed:03d}.receipt.json"
        path.write_bytes(_CONTROL._canonical_bytes(value))
        receipt_values.append(value)
        receipt_paths.append(path)
    receipt_raws = tuple(path.read_bytes() for path in receipt_paths)
    aggregate = tmp_path / "aggregate.json"
    aggregate.write_bytes(_CONTROL.control_aggregate_receipt_bytes(receipt_raws))
    return {
        "aggregate": aggregate,
        "checkpoint": checkpoint,
        "checkpoint_directory": checkpoint_directory,
        "checkpoint_receipt": checkpoint_receipt,
        "receipt_paths": tuple(receipt_paths),
        "receipt_values": receipt_values,
        "run_authority": run_authority,
    }


def test_campaign_authenticates_aggregate_selected_seed_and_checkpoint(tmp_path: Path) -> None:
    fixture = _campaign_fixture(tmp_path)
    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )

    assert campaign.seed == 17
    assert campaign.checkpoint.epoch == 60
    assert campaign.checkpoint.path == fixture["checkpoint"]
    assert vars(campaign.run_authority) == vars(fixture["run_authority"])
    assert campaign.final_burned_correct == {"raw": 1_258, "projected": 1_258}
    assert campaign.aggregate_sha256 == hashlib.sha256(
        cast(Path, fixture["aggregate"]).read_bytes()
    ).hexdigest()


def test_campaign_rejects_stale_aggregate_and_semantic_authority_drift(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    aggregate = cast(Path, fixture["aggregate"])
    receipt_paths = cast(tuple[Path, ...], fixture["receipt_paths"])
    checkpoint_directory = cast(Path, fixture["checkpoint_directory"])

    aggregate.write_bytes(aggregate.read_bytes().replace(b'"seeds":[17', b'"seeds": [17'))
    with pytest.raises(ValueError, match="aggregate"):
        _MODULE.read_authenticated_control_campaign(
            aggregate=aggregate,
            seed_receipts=receipt_paths,
            checkpoint_directory=checkpoint_directory,
            selected_seed=17,
        )

    fixture = _campaign_fixture(tmp_path / "semantic")
    receipt_paths = cast(tuple[Path, ...], fixture["receipt_paths"])
    values = cast(list[dict[str, Any]], fixture["receipt_values"])
    for value, path in zip(values, receipt_paths, strict=True):
        mutated = copy.deepcopy(value)
        mutated["model"]["name"] = "wrong-model"
        path.write_bytes(_CONTROL._canonical_bytes(mutated))
    aggregate = cast(Path, fixture["aggregate"])
    aggregate.write_bytes(
        _CONTROL.control_aggregate_receipt_bytes(tuple(path.read_bytes() for path in receipt_paths))
    )
    with pytest.raises(ValueError, match="model"):
        _MODULE.read_authenticated_control_campaign(
            aggregate=aggregate,
            seed_receipts=receipt_paths,
            checkpoint_directory=cast(Path, fixture["checkpoint_directory"]),
            selected_seed=17,
        )


def test_campaign_rejects_checkpoint_and_concrete_type_drift(tmp_path: Path) -> None:
    fixture = _campaign_fixture(tmp_path)
    receipt_paths = cast(tuple[Path, ...], fixture["receipt_paths"])
    selected = json.loads(receipt_paths[0].read_bytes())
    selected["checkpoint"]["epoch"] = True
    receipt_paths[0].write_bytes(_CONTROL._canonical_bytes(selected))
    aggregate = cast(Path, fixture["aggregate"])
    aggregate.write_bytes(
        _CONTROL.control_aggregate_receipt_bytes(tuple(path.read_bytes() for path in receipt_paths))
    )
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        _MODULE.read_authenticated_control_campaign(
            aggregate=aggregate,
            seed_receipts=receipt_paths,
            checkpoint_directory=cast(Path, fixture["checkpoint_directory"]),
            selected_seed=17,
        )
