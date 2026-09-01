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

from sfora.siglip_checkpoint_audit import validate_siglip_checkpoint_audit_bytes
from sfora.substrate_screen import (
    SubstrateRetrievalError,
    SubstrateScreenEvidence,
    SubstrateScreenMetrics,
)
from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    validate_twin_reachability_artifact_bytes,
    validate_twin_reachability_inference_artifact_bytes,
)

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
    assert campaign.seed_receipt_sha256 == hashlib.sha256(
        cast(tuple[Path, ...], fixture["receipt_paths"])[0].read_bytes()
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


def _burned_examples() -> tuple[Any, ...]:
    return tuple(
        _CONTROL.ImageExample(
            example_id=f"cars-burned-{position:04d}",
            image=object(),
            label=82 + position % 16,
        )
        for position in range(1_345)
    )


def _terminal_screen(correct: int) -> SubstrateScreenEvidence:
    error_count = 1_345 - correct
    errors = tuple(
        SubstrateRetrievalError(
            query_position=position,
            nearest_position=position + 1,
            query_label=82 + position % 16,
            nearest_label=82 + (position + 1) % 16,
        )
        for position in range(error_count)
    )
    return SubstrateScreenEvidence(
        metrics=SubstrateScreenMetrics(
            correct=correct,
            queries=1_345,
            recall_at_1=correct / 1_345,
        ),
        errors=errors,
    )


def test_runner_restores_once_scores_only_burned_and_self_validates(tmp_path: Path) -> None:
    fixture = _campaign_fixture(tmp_path)
    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )
    examples = _burned_examples()
    calls: list[object] = []
    raw_descriptors = torch.tensor([[1.0, 0.0]]).repeat(1_345, 1)
    projected_descriptors = torch.tensor([[0.0, 1.0]]).repeat(1_345, 1)
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)
    twins: list[tuple[bytes, bytes, bytes, bytes]] = []

    def restore(**kwargs: object) -> tuple[object, object]:
        calls.append(("restore", kwargs["campaign"]))
        return object(), object()

    def embed(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = cast(tuple[Any, ...], kwargs["examples"])
        calls.append(("embed", tuple(row.label for row in observed)))
        return raw_descriptors, projected_descriptors, labels

    def score(descriptors: torch.Tensor, observed_labels: torch.Tensor, **_kwargs: object) -> Any:
        calls.append(("score", descriptors, observed_labels))
        return _terminal_screen(1_258)

    result = _MODULE.run_checkpoint_error_audit(
        campaign=campaign,
        burned_examples=examples,
        device=torch.device("cuda"),
        restore_model=restore,
        embed_examples=embed,
        score_descriptors=score,
        twin_sink=lambda raw, projected, raw_inference, projected_inference: twins.append(
            (raw, projected, raw_inference, projected_inference)
        ),
    )

    assert [call[0] for call in calls] == ["restore", "embed", "score", "score"]
    assert all(82 <= label <= 97 for label in cast(tuple[int, ...], calls[1][1]))
    assert calls[2][1] is raw_descriptors
    assert calls[3][1] is projected_descriptors
    assert calls[2][2] is labels
    assert calls[3][2] is labels
    assert cast(torch.Tensor, calls[2][1]).device.type == "cpu"
    assert cast(torch.Tensor, calls[2][2]).device.type == "cpu"
    example_ids = tuple(row.example_id for row in examples)
    example_labels = tuple(row.label for row in examples)
    validate_siglip_checkpoint_audit_bytes(
        result,
        expected_authority=_MODULE.audit_authority(campaign, examples),
        expected_example_ids=example_ids,
        expected_labels=example_labels,
    )
    assert b"clean_validation" not in result
    assert b"optimization" not in result
    assert len(twins) == 1
    for expected_plane, twin, inference_raw in zip(
        ("trained-raw", "trained-projected"),
        twins[0][:2],
        twins[0][2:],
        strict=True,
    ):
        authority = TwinReachabilityAuthority(**json.loads(twin)["authority"])
        evidence = validate_twin_reachability_artifact_bytes(twin, expected=authority)
        inference_evidence, inference = validate_twin_reachability_inference_artifact_bytes(
            inference_raw,
            expected=authority,
        )
        assert inference_evidence == evidence
        assert inference.bootstrap_draws == 10_000
        assert inference.permutation_draws == 64
        assert evidence.plane == expected_plane
        assert evidence.source_count == 169
        assert authority.producer_kind == "trained-checkpoint"
        assert authority.producer_identity == campaign.checkpoint.sha256
        selected_descriptors = (
            raw_descriptors if expected_plane == "trained-raw" else projected_descriptors
        )[[index for index, example in enumerate(examples) if example.label in {82, 83}]]
        descriptor_header = b'{"dtype":"float32-le","shape":[169,2]}\n'
        assert authority.descriptor_sha256 == hashlib.sha256(
            descriptor_header
            + selected_descriptors.numpy().astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()


def test_runner_rejects_terminal_metric_mismatch_and_cleans_publication(
    tmp_path: Path,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )
    examples = _burned_examples()
    labels = torch.tensor([example.label for example in examples], dtype=torch.int64)

    with pytest.raises(ValueError, match="terminal metrics"):
        _MODULE.run_checkpoint_error_audit(
            campaign=campaign,
            burned_examples=examples,
            device=torch.device("cpu"),
            restore_model=lambda **_kwargs: (object(), object()),
            embed_examples=lambda **_kwargs: (
                torch.ones(1_345, 2),
                torch.ones(1_345, 2),
                labels,
            ),
            score_descriptors=lambda *_args, **_kwargs: _terminal_screen(1_257),
        )

    output = tmp_path / "result.json"
    payload = b'{"sealed":true}\n'
    _MODULE.publish_new_result(output, payload)
    assert output.read_bytes() == payload
    with pytest.raises(FileExistsError):
        _MODULE.publish_new_result(output, payload)
    assert not output.with_name(f".{output.name}.partial").exists()


def test_multi_result_publication_rolls_back_and_preflight_rejects_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    third = tmp_path / "third.json"
    original_link = _MODULE.os.link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected multi-result failure")
        original_link(source, target)

    monkeypatch.setattr(_MODULE.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        _MODULE.publish_new_results(
            ((first, b"first\n"), (second, b"second\n"), (third, b"third\n"))
        )
    assert not first.exists() and not second.exists() and not third.exists()
    assert not any(tmp_path.glob(".*.partial"))

    first.write_bytes(b"sealed\n")
    with pytest.raises(FileExistsError):
        _MODULE.require_new_result_paths((first, second, third))


def test_runner_rejects_wrong_band_before_restore(tmp_path: Path) -> None:
    fixture = _campaign_fixture(tmp_path)
    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )
    examples = list(_burned_examples())
    examples[0] = _CONTROL.ImageExample(
        example_id="cars-clean-forbidden",
        image=object(),
        label=81,
    )
    called = False

    def restore(**_kwargs: object) -> tuple[object, object]:
        nonlocal called
        called = True
        return object(), object()

    with pytest.raises(ValueError, match="burned"):
        _MODULE.run_checkpoint_error_audit(
            campaign=campaign,
            burned_examples=tuple(examples),
            device=torch.device("cpu"),
            restore_model=restore,
        )
    assert not called


@pytest.mark.parametrize("mutation", ("short", "long", "duplicate-id"))
def test_runner_rejects_burned_cardinality_and_identity_drift_before_restore(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _campaign_fixture(tmp_path)
    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )
    examples = list(_burned_examples())
    if mutation == "short":
        examples.pop()
    elif mutation == "long":
        examples.append(
            _CONTROL.ImageExample(
                example_id="cars-burned-extra",
                image=object(),
                label=82,
            )
        )
    else:
        examples[1] = _CONTROL.ImageExample(
            example_id=examples[0].example_id,
            image=object(),
            label=examples[1].label,
        )
    called = False

    def restore(**_kwargs: object) -> tuple[object, object]:
        nonlocal called
        called = True
        return object(), object()

    with pytest.raises(ValueError, match="burned"):
        _MODULE.run_checkpoint_error_audit(
            campaign=campaign,
            burned_examples=tuple(examples),
            device=torch.device("cpu"),
            restore_model=restore,
        )
    assert not called


def test_cli_requires_exact_local_capability_and_main_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _campaign_fixture(tmp_path)
    output = tmp_path / "result.json"
    raw_twin_output = tmp_path / "raw-twin.json"
    projected_twin_output = tmp_path / "projected-twin.json"
    raw_twin_inference_output = tmp_path / "raw-twin-inference.json"
    projected_twin_inference_output = tmp_path / "projected-twin-inference.json"
    receipt_paths = cast(tuple[Path, ...], fixture["receipt_paths"])
    arguments = [
        "--aggregate",
        str(fixture["aggregate"]),
        "--checkpoint-directory",
        str(fixture["checkpoint_directory"]),
        "--selected-seed",
        "17",
        "--output",
        str(output),
        "--raw-twin-output",
        str(raw_twin_output),
        "--projected-twin-output",
        str(projected_twin_output),
        "--raw-twin-inference-output",
        str(raw_twin_inference_output),
        "--projected-twin-inference-output",
        str(projected_twin_inference_output),
    ]
    for path in receipt_paths:
        arguments.extend(("--seed-receipt", str(path)))
    arguments.append("--execute-checkpoint-audit")

    parsed = _MODULE.parse_args(arguments)
    assert parsed.seed_receipt == list(receipt_paths)
    assert parsed.selected_seed == 17
    with pytest.raises(SystemExit):
        _MODULE.parse_args(arguments + ["--output", str(tmp_path / "other.json")])
    with pytest.raises(SystemExit):
        _MODULE.parse_args(arguments[:-3] + ["--execute-checkpoint-audit"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args(arguments + ["--clean-errors", str(tmp_path / "forbidden.json")])
    projected_flag = arguments.index("--projected-twin-output")
    with pytest.raises(SystemExit):
        _MODULE.parse_args(arguments[:projected_flag] + arguments[projected_flag + 2 :])

    campaign = _MODULE.read_authenticated_control_campaign(
        aggregate=fixture["aggregate"],
        seed_receipts=fixture["receipt_paths"],
        checkpoint_directory=fixture["checkpoint_directory"],
        selected_seed=17,
    )
    examples = _burned_examples()
    bands = cast(
        Any,
        type(
            "Bands",
            (),
            {
                "burned_diagnostic": examples,
                "ordered_manifest": examples,
            },
        )(),
    )
    monkeypatch.setattr(_MODULE, "read_authenticated_control_campaign", lambda **_kwargs: campaign)
    monkeypatch.setattr(_MODULE, "load_control_examples", lambda: bands)
    monkeypatch.setattr(_MODULE, "control_manifest_sha256", lambda _rows: "3" * 64)
    payload = b'{"claim_eligible":false,"sealed":true}\n'
    raw_twin = b'{"plane":"trained-raw"}\n'
    projected_twin = b'{"plane":"trained-projected"}\n'
    raw_twin_inference = b'{"plane":"trained-raw","inference":true}\n'
    projected_twin_inference = b'{"plane":"trained-projected","inference":true}\n'
    calls: list[tuple[object, ...] | str] = []

    def deterministic(device: torch.device) -> None:
        assert device == torch.device("cuda")
        calls.append("deterministic")

    def run(**kwargs: object) -> bytes:
        calls.append((kwargs["campaign"], kwargs["burned_examples"], kwargs["device"]))
        cast(Any, kwargs["twin_sink"])(
            raw_twin,
            projected_twin,
            raw_twin_inference,
            projected_twin_inference,
        )
        return payload

    monkeypatch.setattr(_MODULE, "require_control_determinism", deterministic)
    monkeypatch.setattr(_MODULE, "run_checkpoint_error_audit", run)
    assert _MODULE.main(arguments) == 0
    assert output.read_bytes() == payload
    assert raw_twin_output.read_bytes() == raw_twin
    assert projected_twin_output.read_bytes() == projected_twin
    assert raw_twin_inference_output.read_bytes() == raw_twin_inference
    assert projected_twin_inference_output.read_bytes() == projected_twin_inference
    assert calls == ["deterministic", (campaign, examples, torch.device("cuda"))]
    terminal = json.loads(capsys.readouterr().out)
    assert terminal == {
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "raw_twin_output": str(raw_twin_output),
        "raw_twin_sha256": hashlib.sha256(raw_twin).hexdigest(),
        "projected_twin_output": str(projected_twin_output),
        "projected_twin_sha256": hashlib.sha256(projected_twin).hexdigest(),
        "raw_twin_inference_output": str(raw_twin_inference_output),
        "raw_twin_inference_sha256": hashlib.sha256(raw_twin_inference).hexdigest(),
        "projected_twin_inference_output": str(projected_twin_inference_output),
        "projected_twin_inference_sha256": hashlib.sha256(
            projected_twin_inference
        ).hexdigest(),
    }

    output.unlink()
    raw_twin_output.unlink()
    projected_twin_output.unlink()
    raw_twin_inference_output.unlink()
    projected_twin_inference_output.unlink()
    bands.burned_diagnostic = examples[:-1]
    with pytest.raises(ValueError, match="manifest"):
        _MODULE.main(arguments)
    assert calls == ["deterministic", (campaign, examples, torch.device("cuda"))]

    bands.burned_diagnostic = examples
    monkeypatch.setattr(_MODULE, "control_manifest_sha256", lambda _rows: "0" * 64)
    with pytest.raises(ValueError, match="manifest"):
        _MODULE.main(arguments)
