#!/usr/bin/env python3
"""Run an authenticated local-only cached-feature SigLIP head screen."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import (
    build_feature_split_authority,
    cotangent_rank_evidence,
    principal_angles_degrees,
    train_cached_subclass_head,
    uncentered_spectral_projection_evidence,
)
from sfora.siglip_proxy_control import evaluate_control_band
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES
from sfora.token_set_screen import F1_TRAIN_CLASSES, F1_VALIDATION_CLASSES

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from run_siglip_proxy_control import SamplerState, _build_epoch_batches  # noqa: E402

_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_CLEAN_RECALL_GATE = 0.974
_PROJECTION_LEARNING_RATE = 1.0e-2
_PROXY_LEARNING_RATE = 2.0e-2
_WEIGHT_DECAY = 0.0
_ALPHA = 32.0
_DELTA = 0.1
_OPTIMIZATION_RECALL_VALIDITY_GATE = 0.99
_LOSS_REDUCTION_VALIDITY_RATIO = 0.95


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the closed local-file capability surface without reading inputs."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=_absolute_path)
    parser.add_argument("--feature-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--result", required=True, type=_absolute_path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--execute-head-screen", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


def _validate_json(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON refuses nonfinite values")
        return
    if type(value) is list:
        for item in cast(list[object], value):
            _validate_json(item)
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError("canonical JSON keys must be strings")
            _validate_json(item)
        return
    raise TypeError("canonical JSON value differs")


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    _validate_json(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_regular(path: Path, *, role: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{role} path differs")
    return path.read_bytes()


@dataclass(frozen=True, slots=True)
class CachedFeatureBand:
    """One authenticated role-local feature matrix and ordered labels."""

    role: str
    example_ids: tuple[str, ...]
    labels: torch.Tensor
    features: torch.Tensor
    feature_sha256: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class CachedFeatureAuthority:
    """All three disjoint Cars-train feature roles."""

    source_manifest_sha256: str
    source_commit: str
    model_name: str
    model_revision: str
    optimization: CachedFeatureBand
    clean_validation: CachedFeatureBand
    burned_diagnostic: CachedFeatureBand


def _band_semantic_sha256(
    role: str,
    example_ids: tuple[str, ...],
    labels: torch.Tensor,
    features: torch.Tensor,
) -> str:
    if (
        type(role) is not str
        or not role
        or type(example_ids) is not tuple
        or labels.dtype != torch.int64
        or labels.shape != (len(example_ids),)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or features.ndim != 2
        or features.shape[0] != len(example_ids)
        or not features.is_contiguous()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("feature band semantic authority differs")
    payload = bytearray(b"sfora-siglip-head-feature-band-v1\0")
    for value in (role, *example_ids):
        encoded = value.encode("utf-8")
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    payload.extend(labels.contiguous().numpy().astype("<i8", copy=False).tobytes())
    payload.extend(features.contiguous().numpy().astype("<f4", copy=False).tobytes())
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(role: str, value: torch.Tensor) -> str:
    if (
        type(role) is not str
        or not role
        or type(value) is not torch.Tensor
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError("head tensor authority differs")
    payload = bytearray(b"sfora-siglip-head-tensor-v1\0")
    encoded = role.encode("utf-8")
    payload.extend(len(encoded).to_bytes(8, "big"))
    payload.extend(encoded)
    payload.extend(value.ndim.to_bytes(8, "big"))
    for dimension in value.shape:
        payload.extend(int(dimension).to_bytes(8, "big"))
    payload.extend(value.numpy().astype("<f4", copy=False).tobytes())
    return hashlib.sha256(payload).hexdigest()


def _load_band(
    root: Path,
    value: object,
    *,
    expected_role: str,
    expected_classes: frozenset[int],
) -> CachedFeatureBand:
    if type(value) is not dict:
        raise ValueError("feature band schema differs")
    item = cast(dict[str, object], value)
    if set(item) != {"role", "file", "sha256", "shape", "example_ids", "labels"}:
        raise ValueError("feature band schema differs")
    filename = item["file"]
    digest = item["sha256"]
    shape = item["shape"]
    example_ids = item["example_ids"]
    labels = item["labels"]
    if (
        item["role"] != expected_role
        or type(filename) is not str
        or not filename
        or Path(filename).name != filename
        or type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or type(shape) is not list
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
        or type(example_ids) is not list
        or type(labels) is not list
        or len(example_ids) != shape[0]
        or len(labels) != shape[0]
        or len(set(cast(list[str], example_ids))) != len(example_ids)
        or any(type(value) is not str or not value for value in example_ids)
        or any(type(value) is not int for value in labels)
        or frozenset(cast(list[int], labels)) != expected_classes
    ):
        raise ValueError("feature band authority differs")
    raw = _read_regular(root / filename, role=f"{expected_role} feature matrix")
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("feature matrix digest differs")
    try:
        matrix = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as error:
        raise ValueError("feature matrix encoding differs") from error
    if (
        type(matrix) is not np.ndarray
        or matrix.dtype != np.dtype("<f4")
        or list(matrix.shape) != shape
        or not matrix.flags.c_contiguous
        or not bool(np.isfinite(matrix).all())
        or bool((np.linalg.norm(matrix.astype(np.float64), axis=1) <= 0).any())
    ):
        raise ValueError("feature matrix authority differs")
    frozen_ids = tuple(cast(list[str], example_ids))
    frozen_labels = torch.tensor(cast(list[int], labels), dtype=torch.int64)
    frozen_features = torch.from_numpy(matrix.copy())
    return CachedFeatureBand(
        role=expected_role,
        example_ids=frozen_ids,
        labels=frozen_labels,
        features=frozen_features,
        feature_sha256=digest,
        semantic_sha256=_band_semantic_sha256(
            expected_role,
            frozen_ids,
            frozen_labels,
            frozen_features,
        ),
    )


def load_feature_cache(path: Path, *, expected_sha256: str) -> CachedFeatureAuthority:
    """Authenticate and load a complete image-free feature cache."""

    raw = _read_regular(path, role="feature manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("feature manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("feature manifest JSON differs") from error
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise ValueError("feature manifest canonical bytes differ")
    if set(value) != {
        "schema",
        "claim_eligible",
        "source_manifest_sha256",
        "control_manifest_file",
        "source_commit",
        "model_name",
        "model_revision",
        "bands",
    }:
        raise ValueError("feature manifest schema differs")
    source_digest = value["source_manifest_sha256"]
    source_commit = value["source_commit"]
    control_filename = value["control_manifest_file"]
    bands = value["bands"]
    if (
        value["schema"] != "sfora-siglip-head-feature-cache-v1"
        or value["claim_eligible"] is not False
        or type(source_digest) is not str
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
        or control_filename != "control-manifest.json"
        or type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or value["model_name"] != _MODEL_NAME
        or value["model_revision"] != _MODEL_REVISION
        or type(bands) is not dict
        or set(bands) != {"optimization", "clean_validation", "burned_diagnostic"}
    ):
        raise ValueError("feature manifest authority differs")
    root = path.parent
    control_raw = _read_regular(root / control_filename, role="control manifest")
    if hashlib.sha256(control_raw).hexdigest() != source_digest:
        raise ValueError("control manifest digest differs")
    try:
        control = json.loads(control_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("control manifest JSON differs") from error
    if (
        type(control) is not dict
        or control_raw != _canonical_bytes(control)
        or set(control)
        != {"schema", "claim_eligible", "dataset_id", "dataset_revision", "examples"}
        or control["schema"] != "sfora-siglip-proxy-control-manifest-v1"
        or control["claim_eligible"] is not False
        or control["dataset_id"] != "tanganke/stanford_cars"
        or control["dataset_revision"] != "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
        or type(control["examples"]) is not list
    ):
        raise ValueError("control manifest authority differs")
    loaded = CachedFeatureAuthority(
        source_manifest_sha256=source_digest,
        source_commit=source_commit,
        model_name=_MODEL_NAME,
        model_revision=_MODEL_REVISION,
        optimization=_load_band(
            root,
            bands["optimization"],
            expected_role="optimization-train",
            expected_classes=F1_TRAIN_CLASSES,
        ),
        clean_validation=_load_band(
            root,
            bands["clean_validation"],
            expected_role="clean-validation",
            expected_classes=F1_VALIDATION_CLASSES,
        ),
        burned_diagnostic=_load_band(
            root,
            bands["burned_diagnostic"],
            expected_role="burned-diagnostic",
            expected_classes=SUBSTRATE_F0_CLASSES,
        ),
    )
    control_rows = control["examples"]
    observed_rows = [
        {"example_id": example_id, "label": int(label)}
        for band in (loaded.optimization, loaded.clean_validation, loaded.burned_diagnostic)
        for example_id, label in zip(band.example_ids, band.labels.tolist(), strict=True)
    ]
    if (
        any(
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            for row in control_rows
        )
        or sorted(observed_rows, key=lambda row: cast(str, row["example_id"])) != control_rows
        or len({row["example_id"] for row in observed_rows}) != len(observed_rows)
    ):
        raise ValueError("feature cache differs from control manifest")
    return loaded


def _metric_payload(features: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    evidence = evaluate_control_band(F.normalize(features.float(), dim=1), labels, query_block=512)
    return {
        "correct": evidence.retrieval.correct,
        "queries": evidence.retrieval.queries,
        "recall_at_1": evidence.retrieval.recall_at_1,
        "mean_nearest_positive_cosine": evidence.margins.mean_nearest_positive_cosine,
        "mean_nearest_negative_cosine": evidence.margins.mean_nearest_negative_cosine,
        "mean_margin": evidence.margins.mean_margin,
    }


def _require_determinism(device: torch.device) -> None:
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("torch refused the head-screen deterministic envelope")


def run_head_screen(
    cache: CachedFeatureAuthority,
    *,
    master_seed_sha256: str,
    output_dimensions: int,
    subclasses_per_class: int,
    cluster_iterations: int,
    train_steps: int,
    device: str,
) -> bytes:
    """Train one fixed head and score all roles without fitting on evaluation rows."""

    if type(cache) is not CachedFeatureAuthority or device not in {"cpu", "cuda"}:
        raise ValueError("head screen execution authority differs")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA head screen requested without CUDA")
    execution_device = torch.device(device)
    _require_determinism(execution_device)
    for band in (cache.optimization, cache.clean_validation, cache.burned_diagnostic):
        if band.semantic_sha256 != _band_semantic_sha256(
            band.role,
            band.example_ids,
            band.labels,
            band.features,
        ):
            raise ValueError("feature band changed after authentication")
    train = cache.optimization
    split_authority = build_feature_split_authority(
        source_manifest_sha256=cache.source_manifest_sha256,
        role=train.role,
        official_test_access=False,
        ordered_example_ids=train.example_ids,
        features=train.features,
    )
    spectral = uncentered_spectral_projection_evidence(
        train.features,
        output_dimensions=output_dimensions,
        split_authority=split_authority,
    )
    spectral_weight = spectral.weight
    seed = int.from_bytes(bytes.fromhex(master_seed_sha256)[:8], "big")
    batches, _state = _build_epoch_batches(
        example_ids=train.example_ids,
        labels=train.labels,
        seed=seed,
        epoch=0,
        steps_per_epoch=train_steps,
        state=SamplerState.initial(),
    )
    trained = train_cached_subclass_head(
        train.features,
        train.labels,
        split_authority=split_authority,
        initial_projection_weight=spectral_weight,
        batches=batches,
        master_seed_sha256=master_seed_sha256,
        output_dimensions=output_dimensions,
        subclasses_per_class=subclasses_per_class,
        cluster_iterations=cluster_iterations,
        train_steps=train_steps,
        projection_learning_rate=_PROJECTION_LEARNING_RATE,
        proxy_learning_rate=_PROXY_LEARNING_RATE,
        weight_decay=_WEIGHT_DECAY,
        alpha=_ALPHA,
        delta=_DELTA,
        device=execution_device,
    )
    payloads: dict[str, object] = {}
    for name, band in (
        ("optimization", cache.optimization),
        ("clean_validation", cache.clean_validation),
        ("burned_diagnostic", cache.burned_diagnostic),
    ):
        payloads[name] = {
            "raw": _metric_payload(band.features, band.labels),
            "spectral": _metric_payload(band.features @ spectral_weight.T, band.labels),
            "trained": _metric_payload(
                band.features @ trained.projection_weight.T,
                band.labels,
            ),
        }
    clean_trained = cast(
        dict[str, object], cast(dict[str, object], payloads["clean_validation"])["trained"]
    )
    clean_recall = clean_trained["recall_at_1"]
    if type(clean_recall) is not float:
        raise RuntimeError("head-screen clean metric authority differs")
    optimization = cast(dict[str, object], payloads["optimization"])
    optimization_spectral = cast(dict[str, object], optimization["spectral"])["recall_at_1"]
    optimization_trained = cast(dict[str, object], optimization["trained"])["recall_at_1"]
    if type(optimization_spectral) is not float or type(optimization_trained) is not float:
        raise RuntimeError("head-screen optimization metric authority differs")
    valid = (
        trained.final_loss <= trained.initial_loss * _LOSS_REDUCTION_VALIDITY_RATIO
        and optimization_trained >= _OPTIMIZATION_RECALL_VALIDITY_GATE
        and optimization_trained >= optimization_spectral
    )
    angles = principal_angles_degrees(spectral_weight, trained.projection_weight)
    control_rank = cotangent_rank_evidence(
        class_count=49,
        logical_batch_size=trained.logical_batch_size,
        embedding_dimensions=output_dimensions,
        tower_dimensions=train.features.shape[1],
    )
    subclass_rank = cotangent_rank_evidence(
        class_count=49 * subclasses_per_class,
        logical_batch_size=trained.logical_batch_size,
        embedding_dimensions=output_dimensions,
        tower_dimensions=train.features.shape[1],
    )
    return _canonical_bytes(
        {
            "schema": "sfora-siglip-cached-head-screen-v1",
            "claim_eligible": False,
            "official_test_access": False,
            "source_manifest_sha256": cache.source_manifest_sha256,
            "source_commit": cache.source_commit,
            "model_name": cache.model_name,
            "model_revision": cache.model_revision,
            "clean_recall_at_1_gate": _CLEAN_RECALL_GATE,
            "valid": valid,
            "passed": valid and clean_recall >= _CLEAN_RECALL_GATE,
            "feature_sha256": {
                "optimization": cache.optimization.feature_sha256,
                "clean_validation": cache.clean_validation.feature_sha256,
                "burned_diagnostic": cache.burned_diagnostic.feature_sha256,
            },
            "procedure": {
                "master_seed_sha256": master_seed_sha256,
                "output_dimensions": output_dimensions,
                "subclasses_per_class": subclasses_per_class,
                "cluster_iterations": cluster_iterations,
                "train_steps": train_steps,
                "device": device,
                "projection_learning_rate": _PROJECTION_LEARNING_RATE,
                "proxy_learning_rate": _PROXY_LEARNING_RATE,
                "weight_decay": _WEIGHT_DECAY,
                "alpha": _ALPHA,
                "delta": _DELTA,
                "logical_batch_size": trained.logical_batch_size,
                "control_cotangent_rank_evidence": asdict(control_rank),
                "subclass_cotangent_rank_evidence": asdict(subclass_rank),
                "spectral_projection_sha256": _tensor_sha256(
                    "spectral-projection", spectral_weight.contiguous()
                ),
                "retained_singular_value": spectral.retained_singular_value,
                "discarded_singular_value": spectral.discarded_singular_value,
                "spectral_cut_ratio": spectral.cut_ratio,
            },
            "training": {
                "initial_loss": trained.initial_loss,
                "final_loss": trained.final_loss,
                "loss_trajectory": list(trained.loss_trajectory),
                "loss_reduction_validity_ratio": _LOSS_REDUCTION_VALIDITY_RATIO,
                "optimization_recall_validity_gate": _OPTIMIZATION_RECALL_VALIDITY_GATE,
                "maximum_principal_angle_degrees": float(angles.max()),
                "mean_principal_angle_degrees": float(angles.mean()),
                "subclass_assignment_sha256": trained.subclass_assignments.sha256,
                "projection_sha256": _tensor_sha256(
                    "trained-projection", trained.projection_weight
                ),
                "subclass_proxies_sha256": _tensor_sha256(
                    "trained-subclass-proxies", trained.subclass_proxies
                ),
            },
            "bands": payloads,
        }
    )


def _write_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path, follow_symlinks=False)
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Execute the fixed scientific cell from authenticated local files."""

    arguments = parse_args(argv)
    cache = load_feature_cache(
        arguments.feature_manifest,
        expected_sha256=arguments.feature_manifest_sha256,
    )
    source_commit = _authenticated_source_commit(Path(__file__).resolve().parents[1])
    if cache.source_commit != source_commit:
        raise ValueError("feature cache source revision differs from executing source")
    result = run_head_screen(
        cache,
        master_seed_sha256=hashlib.sha256(b"sfora-siglip-head-screen-v1").hexdigest(),
        output_dimensions=512,
        subclasses_per_class=4,
        cluster_iterations=10,
        train_steps=100,
        device=arguments.device,
    )
    _write_new(arguments.result, result)
    sys.stdout.buffer.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
