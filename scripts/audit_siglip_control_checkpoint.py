#!/usr/bin/env python3
"""Authenticate and audit one terminal SigLIP control checkpoint locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from sfora.siglip_checkpoint_audit import (
    SiglipCheckpointAuditAuthority,
    build_siglip_checkpoint_audit,
    canonical_siglip_checkpoint_audit_bytes,
)
from sfora.substrate_screen import SubstrateScreenEvidence, score_frozen_substrate_evidence
from sfora.token_set_screen import F1_TRAIN_CLASSES
from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    build_twin_reachability,
    canonical_twin_reachability_artifact_bytes,
)

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from run_siglip_proxy_control import (  # noqa: E402
    CheckpointAuthority,
    ControlRunAuthority,
    PooledProxyAnchorModel,
    SiglipProxyControlConfig,
    _checkpoint_authority_from_receipt,
    _config_sha256,
    _json_compatible,
    _optimizer_groups,
    control_aggregate_receipt_bytes,
    control_manifest_sha256,
    embed_control_examples,
    load_control_examples,
    load_siglip_control_components,
    read_control_seed_receipt,
    require_control_determinism,
    restore_control_checkpoint,
)

_SELECTED_SEED = 17
_FINAL_EPOCH = 60
_BURNED_QUERIES = 1_345


@dataclass(frozen=True, slots=True)
class AuthenticatedControlCampaign:
    """Exact terminal campaign and selected checkpoint authority."""

    seed: int
    seed_receipt: dict[str, Any]
    seed_receipt_sha256: str
    aggregate_sha256: str
    run_authority: ControlRunAuthority
    checkpoint: CheckpointAuthority
    final_burned_correct: dict[str, int]


def _read_regular(path: Path, *, role: str) -> bytes:
    if not isinstance(path, Path) or path.is_symlink() or not path.exists():
        raise ValueError(f"{role} must be a regular file")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular file")
    return path.read_bytes()


def _require_keys(value: object, expected: set[str], *, role: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{role} schema differs")
    return cast(dict[str, Any], value)


def _burned_correct(seed_receipt: dict[str, Any]) -> dict[str, int]:
    evaluation = _require_keys(
        seed_receipt.get("evaluation"), {"initial", "final"}, role="evaluation"
    )
    final = _require_keys(
        evaluation["final"],
        {"optimization", "clean_validation", "burned_diagnostic"},
        role="final evaluation",
    )
    burned = _require_keys(
        final["burned_diagnostic"], {"raw", "projected"}, role="burned evaluation"
    )
    result: dict[str, int] = {}
    scalar_keys = {
        "correct",
        "queries",
        "recall_at_1",
        "mean_nearest_positive_cosine",
        "mean_nearest_negative_cosine",
        "mean_margin",
    }
    for name in ("raw", "projected"):
        row = _require_keys(burned[name], scalar_keys, role=f"{name} burned metrics")
        correct = row["correct"]
        queries = row["queries"]
        recall = row["recall_at_1"]
        if (
            type(correct) is not int
            or type(queries) is not int
            or type(recall) is not float
            or queries != _BURNED_QUERIES
            or not 0 <= correct <= queries
            or recall != correct / queries
        ):
            raise ValueError(f"{name} burned metrics differ")
        result[name] = correct
    return result


def _validate_selected_seed_receipt(
    value: dict[str, Any],
) -> tuple[ControlRunAuthority, dict[str, int]]:
    config = SiglipProxyControlConfig()
    source = _require_keys(value.get("source"), {"revision", "tree_digest", "dirty"}, role="source")
    dataset = _require_keys(
        value.get("dataset"),
        {
            "name",
            "revision",
            "manifest_sha256",
            "optimization_examples",
            "clean_validation_examples",
            "burned_diagnostic_examples",
        },
        role="dataset",
    )
    model = _require_keys(
        value.get("model"),
        {"name", "revision", "resolved_revision", "initial_state_sha256"},
        role="model",
    )
    config_payload = _json_compatible(vars(config))
    source_manifest = dataset["manifest_sha256"]
    if (
        value.get("seed") != _SELECTED_SEED
        or type(value.get("seed")) is not int
        or source["dirty"] is not False
        or dataset
        != {
            "name": config.dataset_name,
            "revision": config.dataset_revision,
            "manifest_sha256": source_manifest,
            "optimization_examples": 3_963,
            "clean_validation_examples": 2_746,
            "burned_diagnostic_examples": _BURNED_QUERIES,
        }
        or model["name"] != config.model_name
        or model["revision"] != config.model_revision
        or model["resolved_revision"] != config.model_revision
        or value.get("config") != config_payload
        or value.get("config_sha256") != _config_sha256(config)
    ):
        raise ValueError("selected seed model, dataset, or config authority differs")
    environment = _require_keys(
        value.get("environment"), set(ControlRunAuthority.__dataclass_fields__), role="environment"
    )
    try:
        run_authority = ControlRunAuthority(**environment)
    except (TypeError, ValueError) as error:
        raise ValueError("selected seed environment differs") from error
    training = _require_keys(
        value.get("training"),
        {
            "optimizer_steps",
            "steps_per_epoch",
            "microbatch_size",
            "final_objective",
            "maximum_score_disagreement",
        },
        role="training",
    )
    if (
        run_authority.source_revision != source["revision"]
        or run_authority.source_tree_digest != source["tree_digest"]
        or run_authority.manifest_sha256 != source_manifest
        or run_authority.steps_per_epoch != 33
        or run_authority.evaluation_batch_size != 32
        or run_authority.query_block != 128
        or training["steps_per_epoch"] != run_authority.steps_per_epoch
        or training["microbatch_size"] != run_authority.microbatch_size
    ):
        raise ValueError("selected seed run authority differs")
    return run_authority, _burned_correct(value)


def read_authenticated_control_campaign(
    *,
    aggregate: Path,
    seed_receipts: tuple[Path, ...],
    checkpoint_directory: Path,
    selected_seed: int,
) -> AuthenticatedControlCampaign:
    """Cross-bind the terminal aggregate, seed receipt, and final checkpoint."""

    if (
        type(seed_receipts) is not tuple
        or len(seed_receipts) != 3
        or type(selected_seed) is not int
        or selected_seed != _SELECTED_SEED
        or not isinstance(checkpoint_directory, Path)
        or checkpoint_directory.is_symlink()
        or not checkpoint_directory.is_dir()
    ):
        raise ValueError("control campaign paths or selected seed differ")
    aggregate_raw = _read_regular(aggregate, role="control aggregate")
    receipt_raws = tuple(
        _read_regular(path, role="control seed receipt") for path in seed_receipts
    )
    expected_aggregate = control_aggregate_receipt_bytes(receipt_raws)
    if aggregate_raw != expected_aggregate:
        raise ValueError("control aggregate authority differs")
    parsed = tuple(read_control_seed_receipt(raw) for raw in receipt_raws)
    selected_entry = next(
        (
            (value, raw)
            for value, raw in zip(parsed, receipt_raws, strict=True)
            if value.get("seed") == selected_seed
        ),
        None,
    )
    if selected_entry is None:
        raise ValueError("selected seed receipt is absent")
    selected, selected_raw = selected_entry
    run_authority, final_burned_correct = _validate_selected_seed_receipt(selected)
    checkpoint_value = _require_keys(
        selected.get("checkpoint"),
        {"basename", "receipt_basename", "sha256", "bytes", "epoch"},
        role="checkpoint",
    )
    if (
        type(checkpoint_value["epoch"]) is not int
        or checkpoint_value["epoch"] != _FINAL_EPOCH
        or type(checkpoint_value["basename"]) is not str
        or type(checkpoint_value["receipt_basename"]) is not str
        or type(checkpoint_value["sha256"]) is not str
        or type(checkpoint_value["bytes"]) is not int
    ):
        raise ValueError("selected checkpoint authority differs")
    checkpoint = _checkpoint_authority_from_receipt(
        checkpoint_directory / checkpoint_value["receipt_basename"],
        directory=checkpoint_directory,
        expected_seed=selected_seed,
    )
    if (
        checkpoint.epoch != checkpoint_value["epoch"]
        or checkpoint.path.name != checkpoint_value["basename"]
        or checkpoint.receipt_path.name != checkpoint_value["receipt_basename"]
        or checkpoint.sha256 != checkpoint_value["sha256"]
        or checkpoint.bytes != checkpoint_value["bytes"]
    ):
        raise ValueError("selected checkpoint binding differs")
    return AuthenticatedControlCampaign(
        seed=selected_seed,
        seed_receipt=selected,
        seed_receipt_sha256=hashlib.sha256(selected_raw).hexdigest(),
        aggregate_sha256=hashlib.sha256(aggregate_raw).hexdigest(),
        run_authority=run_authority,
        checkpoint=checkpoint,
        final_burned_correct=final_burned_correct,
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity_sha256(role: str, values: Sequence[object]) -> str:
    return hashlib.sha256(_canonical({role: list(values)})).hexdigest()


def audit_authority(
    campaign: AuthenticatedControlCampaign,
    burned_examples: Sequence[object],
) -> SiglipCheckpointAuditAuthority:
    """Derive the pure result authority from authenticated campaign inputs."""

    if type(campaign) is not AuthenticatedControlCampaign:
        raise TypeError("checkpoint audit campaign has the wrong concrete type")
    example_ids = tuple(getattr(example, "example_id", None) for example in burned_examples)
    labels = tuple(getattr(example, "label", None) for example in burned_examples)
    dataset = cast(dict[str, Any], campaign.seed_receipt["dataset"])
    model = cast(dict[str, Any], campaign.seed_receipt["model"])
    return SiglipCheckpointAuditAuthority(
        source_revision=campaign.run_authority.source_revision,
        source_tree_digest=campaign.run_authority.source_tree_digest,
        aggregate_sha256=campaign.aggregate_sha256,
        seed_receipt_sha256=campaign.seed_receipt_sha256,
        dataset_revision=dataset["revision"],
        dataset_manifest_sha256=dataset["manifest_sha256"],
        model_name=model["name"],
        model_revision=model["revision"],
        config_sha256=campaign.seed_receipt["config_sha256"],
        seed=campaign.seed,
        checkpoint_sha256=campaign.checkpoint.sha256,
        checkpoint_bytes=campaign.checkpoint.bytes,
        checkpoint_epoch=campaign.checkpoint.epoch,
        evaluation_batch_size=campaign.run_authority.evaluation_batch_size,
        query_block=campaign.run_authority.query_block,
        ordered_example_ids_sha256=_identity_sha256("example_ids", example_ids),
        label_vector_sha256=_identity_sha256("labels", labels),
    )


def restore_audit_model(
    *,
    campaign: AuthenticatedControlCampaign,
    device: torch.device,
) -> tuple[PooledProxyAnchorModel, object]:
    """Restore the exact final checkpoint through the existing strict loader."""

    config = SiglipProxyControlConfig()
    tower, processor = load_siglip_control_components(config=config)
    torch.manual_seed(campaign.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(campaign.seed)
    model = PooledProxyAnchorModel(
        tower=tower,
        input_dimensions=config.input_dimensions,
        embedding_dimensions=config.embedding_dimensions,
        class_count=len(F1_TRAIN_CLASSES),
        projection_initialization=config.projection_initialization,
        proxy_initialization=config.proxy_initialization,
    ).to(device)
    optimizer = torch.optim.AdamW(_optimizer_groups(model, config))
    restored = restore_control_checkpoint(
        campaign.checkpoint.path,
        model=model,
        optimizer=optimizer,
        config=config,
        expected_seed=campaign.seed,
        expected_run_authority=campaign.run_authority,
    )
    if restored.seed != campaign.seed or restored.completed_epoch != _FINAL_EPOCH:
        raise ValueError("restored checkpoint terminal authority differs")
    return model.eval(), processor


def _descriptor_sha256(descriptors: torch.Tensor) -> str:
    canonical = descriptors.detach().to(device="cpu", dtype=torch.float32).contiguous()
    header = _canonical(
        {"dtype": "float32-le", "shape": [int(size) for size in canonical.shape]}
    )
    values = canonical.numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(header + values).hexdigest()


def _build_checkpoint_twin_artifacts(
    *,
    campaign: AuthenticatedControlCampaign,
    burned_examples: tuple[object, ...],
    raw_descriptors: torch.Tensor,
    projected_descriptors: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[bytes, bytes]:
    positions = [
        index
        for index, example in enumerate(burned_examples)
        if getattr(example, "label", None) in {82, 83}
    ]
    example_ids = tuple(getattr(burned_examples[index], "example_id", None) for index in positions)
    selected_labels = tuple(getattr(burned_examples[index], "label", None) for index in positions)
    if (
        len(positions) < 40
        or min(selected_labels.count(82), selected_labels.count(83)) < 20
        or len(set(example_ids)) != len(example_ids)
        or any(type(value) is not str or not value for value in example_ids)
        or set(selected_labels) != {82, 83}
        or any(type(value) is not int for value in selected_labels)
        or not torch.equal(labels[positions], torch.tensor(selected_labels, dtype=torch.int64))
    ):
        raise ValueError("checkpoint twin example authority differs")
    dataset = cast(dict[str, Any], campaign.seed_receipt["dataset"])
    model = cast(dict[str, Any], campaign.seed_receipt["model"])
    artifacts: list[bytes] = []
    for plane, descriptors in (
        ("trained-raw", raw_descriptors),
        ("trained-projected", projected_descriptors),
    ):
        selected = (
            descriptors[positions]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        evidence = build_twin_reachability(
            plane,
            selected.numpy().astype(np.float32, copy=False),
            np.asarray(selected_labels, dtype=np.int64),
        )
        authority = TwinReachabilityAuthority(
            plane=plane,
            source_revision=campaign.run_authority.source_revision,
            source_tree_digest=campaign.run_authority.source_tree_digest,
            dataset_revision=dataset["revision"],
            dataset_manifest_sha256=dataset["manifest_sha256"],
            model_name=model["name"],
            model_revision=model["revision"],
            producer_kind="trained-checkpoint",
            producer_identity=campaign.checkpoint.sha256,
            ordered_example_ids_sha256=_identity_sha256("example_ids", example_ids),
            label_vector_sha256=_identity_sha256("labels", selected_labels),
            descriptor_sha256=_descriptor_sha256(selected),
        )
        artifacts.append(canonical_twin_reachability_artifact_bytes(authority, evidence))
    return artifacts[0], artifacts[1]


def run_checkpoint_error_audit(
    *,
    campaign: AuthenticatedControlCampaign,
    burned_examples: tuple[object, ...],
    device: torch.device,
    restore_model: Callable[..., tuple[object, object]] = restore_audit_model,
    embed_examples: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
        embed_control_examples
    ),
    score_descriptors: Callable[..., SubstrateScreenEvidence] = score_frozen_substrate_evidence,
    twin_sink: Callable[[bytes, bytes], None] | None = None,
) -> bytes:
    """Restore once, score two descriptor planes, and serialize no clean identities."""

    if type(campaign) is not AuthenticatedControlCampaign or type(burned_examples) is not tuple:
        raise TypeError("checkpoint audit inputs have the wrong concrete type")
    if (
        len(burned_examples) != _BURNED_QUERIES
        or any(
            type(getattr(example, "example_id", None)) is not str
            or not example.example_id
            or type(getattr(example, "label", None)) is not int
            or not 82 <= example.label <= 97
            for example in burned_examples
        )
        or len({example.example_id for example in burned_examples}) != _BURNED_QUERIES
    ):
        raise ValueError("checkpoint audit burned example authority differs")
    model, processor = restore_model(campaign=campaign, device=device)
    raw_descriptors, projected_descriptors, labels = embed_examples(
        model=model,
        examples=burned_examples,
        processor=processor,
        device=device,
        batch_size=campaign.run_authority.evaluation_batch_size,
    )
    expected_labels = torch.tensor(
        [getattr(example, "label", None) for example in burned_examples], dtype=torch.int64
    )
    if (
        not isinstance(raw_descriptors, torch.Tensor)
        or not isinstance(projected_descriptors, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or raw_descriptors.ndim != 2
        or projected_descriptors.ndim != 2
        or raw_descriptors.shape[0] != len(burned_examples)
        or projected_descriptors.shape[0] != len(burned_examples)
        or labels.device.type != "cpu"
        or labels.dtype != torch.int64
        or not torch.equal(labels, expected_labels)
    ):
        raise ValueError("checkpoint audit embedding authority differs")
    raw_evidence = score_descriptors(
        raw_descriptors,
        labels,
        query_block=campaign.run_authority.query_block,
    )
    projected_evidence = score_descriptors(
        projected_descriptors,
        labels,
        query_block=campaign.run_authority.query_block,
    )
    if {
        "raw": raw_evidence.metrics.correct,
        "projected": projected_evidence.metrics.correct,
    } != campaign.final_burned_correct:
        raise ValueError("recomputed checkpoint terminal metrics differ")
    authority = audit_authority(campaign, burned_examples)
    evidence = build_siglip_checkpoint_audit(
        authority=authority,
        examples=burned_examples,
        raw=raw_evidence,
        projected=projected_evidence,
    )
    if twin_sink is not None:
        if not callable(twin_sink):
            raise TypeError("checkpoint twin sink differs")
        twin_sink(
            *_build_checkpoint_twin_artifacts(
                campaign=campaign,
                burned_examples=burned_examples,
                raw_descriptors=raw_descriptors,
                projected_descriptors=projected_descriptors,
                labels=labels,
            )
        )
    return canonical_siglip_checkpoint_audit_bytes(
        evidence,
        authority=authority,
        expected_example_ids=tuple(example.example_id for example in burned_examples),
        expected_labels=tuple(example.label for example in burned_examples),
    )


def publish_new_result(path: Path, payload: bytes) -> None:
    """Publish one create-new result and remove its partial on every failure."""

    if not isinstance(path, Path) or type(payload) is not bytes or not payload:
        raise TypeError("checkpoint audit publication inputs differ")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path)
        partial.unlink()
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def require_new_result_paths(paths: tuple[Path, ...]) -> None:
    """Fail before model work unless every distinct result path is create-new."""

    if (
        type(paths) is not tuple
        or not paths
        or any(not isinstance(path, Path) for path in paths)
        or len({path.resolve() for path in paths}) != len(paths)
    ):
        raise TypeError("checkpoint audit result paths differ")
    for path in paths:
        partial = path.with_name(f".{path.name}.partial")
        if path.exists() or path.is_symlink() or partial.exists() or partial.is_symlink():
            raise FileExistsError(path)


def publish_new_results(outputs: tuple[tuple[Path, bytes], ...]) -> None:
    """Atomically publish a distinct create-new result set or roll it all back."""

    if (
        type(outputs) is not tuple
        or not outputs
        or any(
            not isinstance(path, Path) or type(payload) is not bytes or not payload
            for path, payload in outputs
        )
    ):
        raise TypeError("checkpoint audit publication set differs")
    require_new_result_paths(tuple(path for path, _payload in outputs))
    partials: list[Path] = []
    published: list[Path] = []
    try:
        for path, payload in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_name(f".{path.name}.partial")
            with partial.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            partials.append(partial)
        for (path, _payload), partial in zip(outputs, partials, strict=True):
            os.link(partial, path)
            published.append(path)
            partial.unlink()
    except BaseException:
        for path in reversed(published):
            path.unlink(missing_ok=True)
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the exact local-only terminal checkpoint audit capability."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--aggregate", required=True, type=_absolute_path)
    parser.add_argument("--seed-receipt", action="append", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-directory", required=True, type=_absolute_path)
    parser.add_argument("--selected-seed", required=True, type=int, choices=(_SELECTED_SEED,))
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--raw-twin-output", required=True, type=_absolute_path)
    parser.add_argument("--projected-twin-output", required=True, type=_absolute_path)
    parser.add_argument("--execute-checkpoint-audit", required=True, action="store_true")
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted(
        {flag for flag in flags if flag != "--seed-receipt" and flags.count(flag) > 1}
    )
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    parsed = parser.parse_args(effective)
    if len(parsed.seed_receipt) != 3:
        parser.error("exactly three seed receipts are required")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate, execute once, and publish one canonical error audit."""

    arguments = parse_args(argv)
    require_new_result_paths(
        (
            arguments.output,
            arguments.raw_twin_output,
            arguments.projected_twin_output,
        )
    )
    campaign = read_authenticated_control_campaign(
        aggregate=arguments.aggregate,
        seed_receipts=tuple(arguments.seed_receipt),
        checkpoint_directory=arguments.checkpoint_directory,
        selected_seed=arguments.selected_seed,
    )
    bands = load_control_examples()
    if (
        control_manifest_sha256(bands.ordered_manifest)
        != campaign.run_authority.manifest_sha256
        or len(bands.burned_diagnostic) != _BURNED_QUERIES
    ):
        raise ValueError("checkpoint audit dataset manifest authority differs")
    device = torch.device("cuda")
    require_control_determinism(device)
    twins: list[tuple[bytes, bytes]] = []
    payload = run_checkpoint_error_audit(
        campaign=campaign,
        burned_examples=cast(tuple[object, ...], bands.burned_diagnostic),
        device=device,
        twin_sink=lambda raw, projected: twins.append((raw, projected)),
    )
    if len(twins) != 1:
        raise RuntimeError("checkpoint twin artifacts were not produced exactly once")
    raw_twin, projected_twin = twins[0]
    publish_new_results(
        (
            (arguments.output, payload),
            (arguments.raw_twin_output, raw_twin),
            (arguments.projected_twin_output, projected_twin),
        )
    )
    sys.stdout.write(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "raw_twin_output": str(arguments.raw_twin_output),
                "raw_twin_sha256": hashlib.sha256(raw_twin).hexdigest(),
                "projected_twin_output": str(arguments.projected_twin_output),
                "projected_twin_sha256": hashlib.sha256(projected_twin).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
