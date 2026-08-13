"""Reproducible foundation-encoder screening primitives."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any, Literal

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION = re.compile(r"[0-9a-f]{40,64}")
_REMOTE_ALLOW_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
)


def _require_nonempty(name: str, value: str) -> None:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty builtin string")


def _require_sha256(name: str, value: str) -> None:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class RemoteFoundationModelSpec:
    """Immutable authority for one remote foundation encoder."""

    arm: str
    model_id: str
    revision: str
    weight_sha256: str
    processor_sha256: str
    config_sha256: str
    pooling: Literal["image_features", "pooler", "cls"]
    resolution: int
    embedding_width: int
    license: str
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("model_id", self.model_id)
        if type(self.revision) is not str or _GIT_REVISION.fullmatch(self.revision) is None:
            raise ValueError("revision must be an immutable lowercase Git object ID")
        _require_sha256("weight_sha256", self.weight_sha256)
        _require_sha256("processor_sha256", self.processor_sha256)
        _require_sha256("config_sha256", self.config_sha256)
        if self.pooling not in {"image_features", "pooler", "cls"}:
            raise ValueError("unsupported pooling rule")
        if type(self.resolution) is not int or self.resolution <= 0:
            raise ValueError("resolution must be a positive builtin integer")
        if type(self.embedding_width) is not int or self.embedding_width <= 0:
            raise ValueError("embedding_width must be a positive builtin integer")
        _require_nonempty("license", self.license)
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be a builtin boolean")


@dataclass(frozen=True)
class LocalCheckpointFoundationSpec:
    """Immutable authority for one trained local retrieval encoder."""

    arm: str
    checkpoint_path: Path
    pretrained_backbone_path: Path
    checkpoint_sha256: str
    resolved_config_sha256: str
    pretrained_backbone_sha256: str
    transform_id: str
    embedding_width: int
    pooling: Literal["embedding"]
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        for name in ("checkpoint_path", "pretrained_backbone_path"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a concrete pathlib.Path")
        _require_sha256("checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("resolved_config_sha256", self.resolved_config_sha256)
        _require_sha256("pretrained_backbone_sha256", self.pretrained_backbone_sha256)
        _require_nonempty("transform_id", self.transform_id)
        if type(self.embedding_width) is not int or self.embedding_width <= 0:
            raise ValueError("embedding_width must be a positive builtin integer")
        if self.pooling != "embedding":
            raise ValueError("local checkpoint pooling must be embedding")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("unsupported dtype")
        if type(self.normalize) is not bool:
            raise ValueError("normalize must be a builtin boolean")
        if self.normalize is not True:
            raise ValueError("local BN-Inception comparator requires normalized embeddings")


@dataclass(frozen=True)
class FoundationEncoderAudit:
    """Observed source authority for a foundation encoder."""

    status: Literal["available", "unavailable"]
    model_id: str
    revision: str
    weight_sha256: str | None
    processor_sha256: str | None
    config_sha256: str | None
    reason: str | None

    def __post_init__(self) -> None:
        if self.status not in {"available", "unavailable"}:
            raise ValueError("audit status differs from exact schema")
        _require_nonempty("audit model_id", self.model_id)
        if type(self.revision) is not str or _GIT_REVISION.fullmatch(self.revision) is None:
            raise ValueError("audit revision must be an immutable lowercase Git object ID")
        if self.status == "available":
            _require_sha256("audit weight_sha256", self.weight_sha256)  # type: ignore[arg-type]
            _require_sha256("audit processor_sha256", self.processor_sha256)  # type: ignore[arg-type]
            _require_sha256("audit config_sha256", self.config_sha256)  # type: ignore[arg-type]
            if self.reason is not None:
                raise ValueError("available audit reason must be null")
        else:
            for name in ("weight_sha256", "processor_sha256", "config_sha256"):
                digest = getattr(self, name)
                if digest is not None:
                    _require_sha256(f"audit {name}", digest)
            if type(self.reason) is not str or not self.reason:
                raise ValueError("unavailable audit reason must be a nonempty builtin string")


@dataclass(frozen=True)
class TransformersFoundationEncoder:
    """Loaded remote components bound to their authenticated audit."""

    spec: RemoteFoundationModelSpec
    processor: Any
    model: Any
    device: Any
    audit: FoundationEncoderAudit

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        import numpy as np
        import torch

        _validate_encode_request(images, batch_size, normalize_embeddings)
        if normalize_embeddings is not self.spec.normalize:
            raise ValueError("requested normalization differs from registered normalization")
        dtype = torch.float32 if self.spec.dtype == "float32" else torch.bfloat16
        batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                values = self.processor(
                    images=images[start : start + batch_size],
                    return_tensors="pt",
                    size={"height": self.spec.resolution, "width": self.spec.resolution},
                )
                values = {
                    name: (
                        value.to(device=self.device, dtype=dtype)
                        if torch.is_tensor(value) and value.is_floating_point()
                        else value.to(self.device)
                        if hasattr(value, "to")
                        else value
                    )
                    for name, value in values.items()
                }
                if self.spec.pooling == "image_features":
                    output = self.model.get_image_features(**values)
                else:
                    result = self.model(**values)
                    output = (
                        result.pooler_output
                        if self.spec.pooling == "pooler"
                        else result.last_hidden_state[:, 0, :]
                    )
                _validate_embedding_output(
                    output,
                    batch_count=len(images[start : start + batch_size]),
                    embedding_width=self.spec.embedding_width,
                )
                if normalize_embeddings:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                batches.append(output.detach().cpu().float().numpy().astype(np.float64))
        return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class LocalFoundationEncoderAudit:
    """Observed byte authority for a trained local retrieval encoder."""

    checkpoint_sha256: str
    resolved_config_sha256: str
    pretrained_backbone_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("local audit checkpoint_sha256", self.checkpoint_sha256)
        _require_sha256("local audit resolved_config_sha256", self.resolved_config_sha256)
        _require_sha256(
            "local audit pretrained_backbone_sha256",
            self.pretrained_backbone_sha256,
        )


@dataclass(frozen=True)
class LocalCheckpointFoundationEncoder:
    """Loaded local model bound to all of its authenticated inputs."""

    spec: LocalCheckpointFoundationSpec
    model: Any
    transform: Callable[[object], Any]
    device: Any
    audit: LocalFoundationEncoderAudit

    def encode(
        self,
        images: list[object],
        *,
        batch_size: int,
        normalize_embeddings: bool,
    ) -> Any:
        import numpy as np
        import torch

        _validate_encode_request(images, batch_size, normalize_embeddings)
        if normalize_embeddings is not self.spec.normalize:
            raise ValueError("requested normalization differs from registered normalization")
        dtype = torch.float32 if self.spec.dtype == "float32" else torch.bfloat16
        batches: list[Any] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                values = torch.stack(
                    [self.transform(image) for image in images[start : start + batch_size]]
                ).to(device=self.device, dtype=dtype)
                output = self.model(values)
                _validate_embedding_output(
                    output,
                    batch_count=len(images[start : start + batch_size]),
                    embedding_width=self.spec.embedding_width,
                )
                if normalize_embeddings:
                    output = torch.nn.functional.normalize(output, p=2, dim=-1)
                batches.append(output.detach().cpu().float().numpy().astype(np.float64))
        return np.concatenate(batches, axis=0)


def _validate_encode_request(
    images: list[object],
    batch_size: int,
    normalize_embeddings: bool,
) -> None:
    if type(images) is not list or not images:
        raise ValueError("images must be a nonempty builtin list")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive builtin integer")
    if type(normalize_embeddings) is not bool:
        raise ValueError("normalize_embeddings must be a builtin boolean")


def _validate_embedding_output(
    output: Any,
    *,
    batch_count: int,
    embedding_width: int,
) -> None:
    import torch

    if not torch.is_tensor(output) or not output.is_floating_point():
        raise ValueError("encoder output must be a floating-point torch tensor")
    if output.ndim != 2:
        raise ValueError("encoder output must be a rank-2 embedding tensor")
    if tuple(output.shape) != (batch_count, embedding_width):
        raise ValueError("encoder output shape differs from registered embedding width")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("encoder output must contain only finite values")


@dataclass(frozen=True)
class NativeFixtureRecord:
    arm: str
    metric: str
    native_value: float | None
    input_sha256: str
    source_sha256: str
    native_cross_check: Literal["available", "unavailable"]
    reason: str | None

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        _require_sha256("input_sha256", self.input_sha256)
        _require_sha256("source_sha256", self.source_sha256)
        if self.native_cross_check == "available":
            if type(self.native_value) is not float or not isfinite(self.native_value):
                raise ValueError("available native fixture value must be a finite builtin float")
            if self.reason is not None:
                raise ValueError("available native fixture cannot have an unavailable reason")
        elif self.native_cross_check == "unavailable":
            if self.native_value is not None:
                raise ValueError("unavailable native fixture cannot carry a value")
            _require_nonempty("reason", self.reason)  # type: ignore[arg-type]
        else:
            raise ValueError("unsupported native cross-check state")


@dataclass(frozen=True)
class MetricToleranceRecord:
    arm: str
    metric: str
    tolerance: float
    frozen_before_execution: bool

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        if type(self.tolerance) is not float or not isfinite(self.tolerance):
            raise ValueError("tolerance must be a finite builtin float")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be nonnegative")
        if self.frozen_before_execution is not True:
            raise ValueError("tolerance must be frozen before execution")


@dataclass(frozen=True)
class FoundationFidelityAudit:
    arm: str
    metric: str
    native_value: float | None
    repository_value: float
    tolerance: float
    provenance: Literal["native_cross_check", "unavailable"]
    passed: bool | None


@dataclass(frozen=True)
class PublishedMetricRecord:
    arm: str
    metric: str
    native_value: float | None
    tolerance: float | None
    source: str
    provenance: Literal["native_cross_check", "repository_only"]

    def __post_init__(self) -> None:
        _require_nonempty("arm", self.arm)
        _require_nonempty("metric", self.metric)
        _require_nonempty("source", self.source)
        if self.provenance == "native_cross_check":
            if type(self.native_value) is not float or not isfinite(self.native_value):
                raise ValueError("published native value must be a finite builtin float")
            if type(self.tolerance) is not float or not isfinite(self.tolerance):
                raise ValueError("published tolerance must be a finite builtin float")
            if self.tolerance < 0.0:
                raise ValueError("published tolerance must be nonnegative")
        elif self.provenance == "repository_only":
            if self.native_value is not None or self.tolerance is not None:
                raise ValueError("repository-only metric cannot carry native comparison values")
        else:
            raise ValueError("unsupported published metric provenance")


@dataclass(frozen=True)
class PublishedMetricAudit:
    arm: str
    metric: str
    native_value: float | None
    repository_value: float
    tolerance: float | None
    provenance: Literal["native_cross_check", "repository_only"]
    passed: bool | None
    invalidates_confirmatory_claim: bool


def _record_keys(records: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    return tuple((record.arm, record.metric) for record in records)


def validate_native_fixture_authority(
    fixtures: Sequence[NativeFixtureRecord],
    tolerances: Sequence[MetricToleranceRecord],
    *,
    registered_pairs: Sequence[tuple[str, str]],
) -> None:
    """Require complete, exact, ordered fixture and tolerance authorities."""

    expected = tuple(registered_pairs)
    fixture_keys = _record_keys(fixtures)
    tolerance_keys = _record_keys(tolerances)
    if set(fixture_keys) != set(expected):
        raise ValueError("fixture key set differs from registered arm/metric pairs")
    if set(tolerance_keys) != set(expected):
        raise ValueError("tolerance key set differs from registered arm/metric pairs")
    if fixture_keys != expected or tolerance_keys != expected:
        raise ValueError("fixture and tolerance ordered keys differ from registered order")


def _finite_repository_value(values: Mapping[str, float], metric: str) -> float:
    try:
        value = values[metric]
    except KeyError as error:
        raise ValueError(f"repository value missing for metric {metric}") from error
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"repository value for {metric} must be a finite builtin float")
    return value


def verify_native_fixture(
    *,
    arm: str,
    encoder: object,
    fixture_inputs: Mapping[str, Path],
    native_sources: Mapping[str, Path],
    repository_metric: Callable[[object, Path, Path, str], float],
    fixtures: Sequence[NativeFixtureRecord],
    tolerances: Sequence[MetricToleranceRecord],
    registered_pairs: Sequence[tuple[str, str]],
) -> tuple[FoundationFidelityAudit, ...]:
    encoder_spec = getattr(encoder, "spec", None)
    if getattr(encoder_spec, "arm", None) != arm:
        raise ValueError("encoder arm differs from requested fixture arm")
    validate_native_fixture_authority(
        fixtures,
        tolerances,
        registered_pairs=registered_pairs,
    )
    expected_arm_pairs = tuple(pair for pair in registered_pairs if pair[0] == arm)
    if not expected_arm_pairs:
        raise ValueError("requested arm has no registered native fixture pairs")
    fixture_rows = tuple(row for row in fixtures if row.arm == arm)
    tolerance_rows = tuple(row for row in tolerances if row.arm == arm)
    if _record_keys(fixture_rows) != expected_arm_pairs:
        raise ValueError("requested arm fixture order differs from registered order")
    if _record_keys(tolerance_rows) != expected_arm_pairs:
        raise ValueError("requested arm tolerance order differs from registered order")
    audits: list[FoundationFidelityAudit] = []
    for fixture, tolerance in zip(fixture_rows, tolerance_rows, strict=True):
        try:
            input_path = fixture_inputs[fixture.metric]
            source_path = native_sources[fixture.metric]
        except KeyError as error:
            raise ValueError(
                f"registered fixture path missing for metric {fixture.metric}"
            ) from error
        if _sha256_file(input_path, field="fixture input") != fixture.input_sha256:
            raise ValueError(f"fixture input digest differs for metric {fixture.metric}")
        if _sha256_file(source_path, field="native source") != fixture.source_sha256:
            raise ValueError(f"native source digest differs for metric {fixture.metric}")
        repository_value = repository_metric(
            encoder,
            input_path,
            source_path,
            fixture.metric,
        )
        if type(repository_value) is not float or not isfinite(repository_value):
            raise ValueError(
                f"repository value for {fixture.metric} must be a finite builtin float"
            )
        available = fixture.native_cross_check == "available"
        passed = (
            abs(repository_value - fixture.native_value) <= tolerance.tolerance
            if available and fixture.native_value is not None
            else None
        )
        audits.append(
            FoundationFidelityAudit(
                arm=arm,
                metric=fixture.metric,
                native_value=fixture.native_value,
                repository_value=repository_value,
                tolerance=tolerance.tolerance,
                provenance="native_cross_check" if available else "unavailable",
                passed=passed,
            )
        )
    return tuple(audits)


def cross_check_published_metrics(
    *,
    arm: str,
    repository_values: Mapping[str, float],
    records: Sequence[PublishedMetricRecord],
    registered_pairs: Sequence[tuple[str, str]],
) -> tuple[PublishedMetricAudit, ...]:
    expected = tuple(registered_pairs)
    if _record_keys(records) != expected:
        raise ValueError("published metric ordered keys differ from registered order")
    expected_arm_pairs = tuple(pair for pair in expected if pair[0] == arm)
    if not expected_arm_pairs:
        raise ValueError("requested arm has no registered published metric pairs")
    audits: list[PublishedMetricAudit] = []
    for record in records:
        if record.arm != arm:
            continue
        repository_value = _finite_repository_value(repository_values, record.metric)
        available = record.provenance == "native_cross_check"
        passed = (
            abs(repository_value - record.native_value) <= record.tolerance
            if available and record.native_value is not None and record.tolerance is not None
            else None
        )
        audits.append(
            PublishedMetricAudit(
                arm=arm,
                metric=record.metric,
                native_value=record.native_value,
                repository_value=repository_value,
                tolerance=record.tolerance,
                provenance=record.provenance,
                passed=passed,
                invalidates_confirmatory_claim=passed is False,
            )
        )
    return tuple(audits)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("JSON authority must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("authority is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError("authority root must be a JSON object")
    return value


def _require_ordered_keys(value: dict[str, Any], expected: tuple[str, ...], *, name: str) -> None:
    if tuple(value) != expected:
        raise ValueError(f"{name} keys differ from exact schema")


def load_native_fixture_authority(
    fixture_path: Path,
    tolerance_path: Path,
    *,
    registered_pairs: Sequence[tuple[str, str]],
    require_frozen: bool = True,
) -> tuple[tuple[NativeFixtureRecord, ...], tuple[MetricToleranceRecord, ...]]:
    fixture_value = _load_strict_json(fixture_path)
    tolerance_value = _load_strict_json(tolerance_path)
    _require_ordered_keys(
        fixture_value,
        ("schema_version", "status", "records"),
        name="fixture authority",
    )
    _require_ordered_keys(
        tolerance_value,
        ("schema_version", "status", "records"),
        name="tolerance authority",
    )
    if fixture_value["schema_version"] != "foundation-native-fixtures-v1":
        raise ValueError("fixture schema version differs")
    if tolerance_value["schema_version"] != "foundation-metric-tolerances-v1":
        raise ValueError("tolerance schema version differs")
    for name, value in (("fixture", fixture_value), ("tolerance", tolerance_value)):
        if value["status"] not in {"prospective_unfrozen", "frozen"}:
            raise ValueError(f"{name} authority status differs")
        if require_frozen and value["status"] != "frozen":
            raise ValueError(f"{name} authority is not frozen")
    if fixture_value["status"] != tolerance_value["status"]:
        raise ValueError("fixture and tolerance authority statuses differ")
    if fixture_value["status"] == "prospective_unfrozen" and (
        fixture_value["records"] != [] or tolerance_value["records"] != []
    ):
        raise ValueError("prospective unfrozen fixture authorities must be empty")
    if type(fixture_value["records"]) is not list:
        raise ValueError("fixture records must be a JSON array")
    if type(tolerance_value["records"]) is not list:
        raise ValueError("tolerance records must be a JSON array")
    fixtures: list[NativeFixtureRecord] = []
    for record in fixture_value["records"]:
        if type(record) is not dict:
            raise ValueError("fixture record must be a JSON object")
        _require_ordered_keys(
            record,
            (
                "arm",
                "metric",
                "native_value",
                "input_sha256",
                "source_sha256",
                "native_cross_check",
                "reason",
            ),
            name="fixture record",
        )
        fixtures.append(NativeFixtureRecord(**record))
    tolerances: list[MetricToleranceRecord] = []
    for record in tolerance_value["records"]:
        if type(record) is not dict:
            raise ValueError("tolerance record must be a JSON object")
        _require_ordered_keys(
            record,
            ("arm", "metric", "tolerance", "frozen_before_execution"),
            name="tolerance record",
        )
        tolerances.append(MetricToleranceRecord(**record))
    validate_native_fixture_authority(
        fixtures,
        tolerances,
        registered_pairs=registered_pairs,
    )
    return tuple(fixtures), tuple(tolerances)


def load_published_metric_register(
    path: Path,
    *,
    require_frozen: bool = True,
) -> tuple[PublishedMetricRecord, ...]:
    value = _load_strict_json(path)
    _require_ordered_keys(
        value,
        ("schema_version", "status", "records"),
        name="published metric authority",
    )
    if value["schema_version"] != "foundation-published-metrics-v1":
        raise ValueError("published metric schema version differs")
    if value["status"] not in {"prospective_unfrozen", "frozen"}:
        raise ValueError("published metric authority status differs")
    if require_frozen and value["status"] != "frozen":
        raise ValueError("published metric authority is not frozen")
    if value["status"] == "prospective_unfrozen" and value["records"] != []:
        raise ValueError("prospective unfrozen published authority must be empty")
    if type(value["records"]) is not list:
        raise ValueError("published metric records must be a JSON array")
    records: list[PublishedMetricRecord] = []
    for record in value["records"]:
        if type(record) is not dict:
            raise ValueError("published metric record must be a JSON object")
        _require_ordered_keys(
            record,
            ("arm", "metric", "native_value", "tolerance", "source", "provenance"),
            name="published metric record",
        )
        records.append(PublishedMetricRecord(**record))
    return tuple(records)


def _load_transformers_dependencies() -> tuple[Any, Any]:
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("install the research extra to load foundation encoders") from error
    return AutoImageProcessor, AutoModel


def _snapshot_download(
    repo_id: str,
    *,
    revision: str,
    allow_patterns: tuple[str, ...],
) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("install the research extra to authenticate remote artifacts") from error
    return str(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(allow_patterns),
        )
    )


def _artifact_set_sha256(root: Path, paths: Sequence[Path]) -> str:
    if len(paths) == 1:
        return sha256(_remote_artifact_bytes(root, paths[0])).hexdigest()
    digest = sha256(b"foundation-artifact-set-v1\0")
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        name = path.relative_to(root).as_posix().encode("utf-8")
        payload = _remote_artifact_bytes(root, path)
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _remote_artifact_bytes(root: Path, path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"remote artifact is not a file: {path}")
    resolved_root = root.resolve(strict=True)
    cache_scope = (
        resolved_root.parent.parent
        if resolved_root.parent.name == "snapshots"
        else resolved_root
    )
    resolved_path = path.resolve(strict=True)
    if not resolved_path.is_relative_to(cache_scope):
        raise ValueError("remote artifact symlink escapes authenticated cache scope")
    return resolved_path.read_bytes()


def _observe_remote_snapshot(
    spec: RemoteFoundationModelSpec,
) -> tuple[Path | None, FoundationEncoderAudit]:
    try:
        root = Path(
            _snapshot_download(
                spec.model_id,
                revision=spec.revision,
                allow_patterns=_REMOTE_ALLOW_PATTERNS,
            )
        )
    except OSError as error:
        return None, FoundationEncoderAudit(
            status="unavailable",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=None,
            processor_sha256=None,
            config_sha256=None,
            reason=f"snapshot unavailable: {error}",
        )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("snapshot root must be a real directory")
    config_path = root / "config.json"
    processor_paths = tuple(
        path
        for name in ("preprocessor_config.json", "processor_config.json")
        if (path := root / name).is_file()
    )
    weight_paths = tuple(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".safetensors.index.json")
            or (path.name.startswith("pytorch_model") and path.name.endswith(".bin"))
            or (path.name.startswith("pytorch_model") and path.name.endswith(".bin.index.json"))
        )
    )
    missing = []
    if not config_path.is_file():
        missing.append("config.json")
    if not processor_paths:
        missing.append("processor configuration")
    if not weight_paths:
        missing.append("model weights")
    if missing:
        return None, FoundationEncoderAudit(
            status="unavailable",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=None,
            processor_sha256=None,
            config_sha256=None,
            reason=f"snapshot lacks {', '.join(missing)}",
        )
    return root, FoundationEncoderAudit(
        status="available",
        model_id=spec.model_id,
        revision=spec.revision,
        weight_sha256=_artifact_set_sha256(root, weight_paths),
        processor_sha256=_artifact_set_sha256(root, processor_paths),
        config_sha256=sha256(_remote_artifact_bytes(root, config_path)).hexdigest(),
        reason=None,
    )


def _observe_remote_artifacts(spec: RemoteFoundationModelSpec) -> FoundationEncoderAudit:
    """Return the structured availability audit without loading model code."""

    return _observe_remote_snapshot(spec)[1]


def _sha256_file(path: Path, *, field: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field} path must be a regular non-symlink file")
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("resolved training config is not canonical JSON data") from error
    return sha256(payload).hexdigest()


def _observe_local_artifacts(
    spec: LocalCheckpointFoundationSpec,
    checkpoint: Mapping[str, Any],
) -> LocalFoundationEncoderAudit:
    if type(checkpoint) is not dict or type(checkpoint.get("training_config")) is not dict:
        raise ValueError("local checkpoint lacks builtin training_config authority")
    return LocalFoundationEncoderAudit(
        checkpoint_sha256=_sha256_file(spec.checkpoint_path, field="checkpoint"),
        resolved_config_sha256=_canonical_json_sha256(checkpoint["training_config"]),
        pretrained_backbone_sha256=_sha256_file(
            spec.pretrained_backbone_path,
            field="pretrained backbone",
        ),
    )


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("install the research extra to load the local comparator") from error
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict:
        raise ValueError("local checkpoint root must be a builtin dictionary")
    return value


def _build_local_bn_inception(*, embedding_size: int, add_gmp: bool) -> Any:
    from sfora.bn_inception import build_bn_inception

    return build_bn_inception(
        embedding_size=embedding_size,
        pretrained=False,
        add_gmp=add_gmp,
    )


def _build_local_eval_transform(spec: LocalCheckpointFoundationSpec) -> Callable[[object], Any]:
    if spec.transform_id != "proxy-anchor-eval-224-v1":
        raise ValueError("unsupported local evaluation transform")
    from sfora.image_end_to_end import ImageEndToEndConfig, _default_transform_factory

    return _default_transform_factory(
        ImageEndToEndConfig(
            backbone_name="bn_inception",
            input_size=224,
        ),
        False,
    )


def _load_local_checkpoint_model(
    spec: LocalCheckpointFoundationSpec,
    checkpoint: dict[str, Any] | None = None,
) -> Any:
    if checkpoint is None:
        checkpoint = _torch_load_checkpoint(spec.checkpoint_path)
    required = ("state_dict", "arch", "training_config")
    if any(name not in checkpoint for name in required):
        raise ValueError("local checkpoint lacks required trained-model fields")
    arch = checkpoint["arch"]
    training_config = checkpoint["training_config"]
    if type(arch) is not dict or type(training_config) is not dict:
        raise ValueError("local checkpoint architecture/config must be builtin dictionaries")
    if _canonical_json_sha256(training_config) != spec.resolved_config_sha256:
        raise ValueError("checkpoint training_config digest differs from resolved authority")
    arch_fields = (
        "backbone_name",
        "pretrained_weights",
        "head_pooling",
        "embedding_dimensions",
        "embedding_head_init",
        "embedding_layer_norm",
    )
    if set(arch) != set(arch_fields):
        raise ValueError("checkpoint architecture key set differs")
    expected_arch = {field: training_config.get(field) for field in arch_fields}
    if arch != expected_arch:
        raise ValueError("checkpoint architecture differs from resolved training config")
    supported_arch = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "head_pooling": "avg_max",
        "embedding_dimensions": spec.embedding_width,
        "embedding_head_init": "default",
        "embedding_layer_norm": False,
    }
    if expected_arch != supported_arch:
        raise ValueError("checkpoint architecture is unsupported by the local comparator")
    model = _build_local_bn_inception(
        embedding_size=spec.embedding_width,
        add_gmp=True,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def _require_matching_remote_audit(
    spec: RemoteFoundationModelSpec,
    audit: FoundationEncoderAudit,
) -> None:
    if audit.status != "available" or audit.reason is not None:
        raise ValueError("remote encoder authority is unavailable")
    expected = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "weight_sha256": spec.weight_sha256,
        "processor_sha256": spec.processor_sha256,
        "config_sha256": spec.config_sha256,
    }
    for field, value in expected.items():
        if getattr(audit, field) != value:
            raise ValueError(f"observed {field} differs from registered authority")


def load_foundation_encoder(
    spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec,
) -> TransformersFoundationEncoder | LocalCheckpointFoundationEncoder:
    """Load one encoder only after its immutable source authority matches."""

    if isinstance(spec, LocalCheckpointFoundationSpec):
        import torch

        if _sha256_file(spec.checkpoint_path, field="checkpoint") != spec.checkpoint_sha256:
            raise ValueError("observed checkpoint_sha256 differs from registered authority")
        if (
            _sha256_file(spec.pretrained_backbone_path, field="pretrained backbone")
            != spec.pretrained_backbone_sha256
        ):
            raise ValueError(
                "observed pretrained_backbone_sha256 differs from registered authority"
            )
        checkpoint = _torch_load_checkpoint(spec.checkpoint_path)
        local_audit = _observe_local_artifacts(spec, checkpoint)
        for field in (
            "checkpoint_sha256",
            "resolved_config_sha256",
            "pretrained_backbone_sha256",
        ):
            if getattr(local_audit, field) != getattr(spec, field):
                raise ValueError(f"observed {field} differs from registered authority")
        dtype = torch.float32 if spec.dtype == "float32" else torch.bfloat16
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_local_checkpoint_model(spec, checkpoint)
        model.to(device=device, dtype=dtype)
        model.eval()
        return LocalCheckpointFoundationEncoder(
            spec=spec,
            model=model,
            transform=_build_local_eval_transform(spec),
            device=device,
            audit=local_audit,
        )
    root, remote_audit = _observe_remote_snapshot(spec)
    _require_matching_remote_audit(spec, remote_audit)
    if root is None:
        raise ValueError("available remote audit lacks an authenticated snapshot root")
    processor_class, model_class = _load_transformers_dependencies()
    import torch

    dtype = torch.float32 if spec.dtype == "float32" else torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = processor_class.from_pretrained(
        str(root),
        revision=spec.revision,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        str(root),
        revision=spec.revision,
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device=device, dtype=dtype)
    model.eval()
    return TransformersFoundationEncoder(
        spec=spec,
        processor=processor,
        model=model,
        device=device,
        audit=remote_audit,
    )
