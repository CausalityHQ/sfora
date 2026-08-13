import gc
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sfora.foundation_pareto as foundation_pareto
from sfora.data import ImageExample
from sfora.foundation_pareto import (
    EmbeddingCacheKeyV2,
    FoundationEncoderAudit,
    LocalCheckpointFoundationEncoder,
    LocalCheckpointFoundationSpec,
    LocalFoundationEncoderAudit,
    MetricToleranceRecord,
    NativeFixtureRecord,
    PublishedMetricRecord,
    RemoteFoundationModelSpec,
    TransformersFoundationEncoder,
    cross_check_published_metrics,
    evaluate_foundation_geometries,
    export_embeddings_v2,
    load_embeddings_v2,
    load_foundation_encoder,
    load_native_fixture_authority,
    load_published_metric_register,
    profile_foundation_encoder,
    validate_native_fixture_authority,
    verify_native_fixture,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _remote_spec(**changes: object) -> RemoteFoundationModelSpec:
    values: dict[str, object] = {
        "arm": "dino-v3-s",
        "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "weight_sha256": _SHA_A,
        "processor_sha256": _SHA_B,
        "config_sha256": _SHA_C,
        "pooling": "cls",
        "resolution": 224,
        "embedding_width": 2,
        "license": "Apache-2.0",
        "dtype": "float32",
        "normalize": True,
    }
    values.update(changes)
    return RemoteFoundationModelSpec(**values)  # type: ignore[arg-type]


def _local_spec(**changes: object) -> LocalCheckpointFoundationSpec:
    values: dict[str, object] = {
        "arm": "comparator",
        "checkpoint_path": Path("artifacts/comparator.pt"),
        "pretrained_backbone_path": Path("artifacts/backbone.pth"),
        "checkpoint_sha256": "e" * 64,
        "resolved_config_sha256": "f" * 64,
        "pretrained_backbone_sha256": "1" * 64,
        "transform_id": "proxy-anchor-eval-224-v1",
        "embedding_width": 8,
        "pooling": "embedding",
        "dtype": "float32",
        "normalize": True,
    }
    values.update(changes)
    return LocalCheckpointFoundationSpec(**values)  # type: ignore[arg-type]


def _geometry_rows() -> list[dict[str, object]]:
    metrics = {
        "precision_at_1": 0.5,
        "recall_at_1": 0.5,
        "recall_at_2": 0.5,
        "recall_at_4": 0.5,
        "recall_at_8": 0.5,
        "map_at_r": 0.5,
        "mean_relevant_items": 1.0,
        "evaluated_queries": 2,
        "total_queries": 2,
        "recall_at_10": 0.5,
        "recall_at_20": 0.5,
        "recall_at_30": 0.5,
        "recall_at_100": 0.5,
    }
    return [
        {"geometry": geometry, "metrics": dict(metrics)}
        for geometry in (
            "normalized_cosine",
            "normalized_euclidean",
            "native_unnormalized_euclidean",
        )
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", ""),
        ("revision", ""),
        ("revision", "main"),
        ("weight_sha256", "A" * 64),
        ("processor_sha256", "short"),
        ("config_sha256", ""),
        ("resolution", 0),
        ("embedding_width", 0),
        ("license", ""),
    ],
)
def test_remote_model_spec_rejects_mutable_or_malformed_authority(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        _remote_spec(**{field: value})


def test_literal_remote_registry_uses_verified_release_facts() -> None:
    siglip2 = RemoteFoundationModelSpec(
        arm="siglip2-base-patch16-256",
        model_id="google/siglip2-base-patch16-256",
        revision="3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab",
        weight_sha256="6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863",
        processor_sha256="d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
        config_sha256="7b5aedcb8893e31376e129c1ffd7a5392f1a806dbc793ce53eda220c2ec59edf",
        pooling="image_features",
        resolution=256,
        embedding_width=768,
        license="Apache-2.0",
        dtype="float32",
        normalize=True,
    )
    gated = (
        FoundationEncoderAudit(
            status="unavailable",
            model_id="facebook/dinov3-vits16-pretrain-lvd1689m",
            revision="114c1379950215c8b35dfcd4e90a5c251dde0d32",
            weight_sha256="4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
            processor_sha256=None,
            config_sha256=None,
            reason="manual gate prevents authenticating config and processor bytes",
        ),
        FoundationEncoderAudit(
            status="unavailable",
            model_id="facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
            revision="10d30274b4d445111e2d5bf75ac93bbd94db274b",
            weight_sha256="bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2",
            processor_sha256=None,
            config_sha256=None,
            reason="manual gate prevents authenticating config and processor bytes",
        ),
    )

    assert siglip2.embedding_width == 768
    assert [row.status for row in gated] == ["unavailable", "unavailable"]


def test_local_checkpoint_spec_requires_complete_trained_model_authority(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "anchor.pt"
    backbone = tmp_path / "bn-inception.pth"
    checkpoint.write_bytes(b"trained-anchor")
    backbone.write_bytes(b"pretrained-backbone")

    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_SHA_A,
        resolved_config_sha256=_SHA_B,
        pretrained_backbone_sha256=_SHA_C,
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )

    assert spec.checkpoint_path == checkpoint
    assert spec.embedding_width == 512

    with pytest.raises(ValueError, match="requires normalized embeddings"):
        LocalCheckpointFoundationSpec(
            arm="proxy-anchor",
            checkpoint_path=checkpoint,
            pretrained_backbone_path=backbone,
            checkpoint_sha256=_SHA_A,
            resolved_config_sha256=_SHA_B,
            pretrained_backbone_sha256=_SHA_C,
            transform_id="proxy-anchor-eval-224-v1",
            embedding_width=512,
            pooling="embedding",
            dtype="float32",
            normalize=False,
        )


def test_local_loader_authenticates_all_trained_model_inputs_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "anchor.pt"
    backbone = tmp_path / "bn-inception.pth"
    checkpoint.write_bytes(b"trained-anchor")
    backbone.write_bytes(b"pretrained-backbone")
    training_config = {"embedding_width": 512}
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_sha256(checkpoint),
        resolved_config_sha256=_canonical_json_digest(training_config),
        pretrained_backbone_sha256=_sha256(backbone),
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )
    loaded: list[LocalCheckpointFoundationSpec] = []

    class RuntimeModel:
        moved: tuple[object, object] | None = None
        evaluated = False

        def to(self, *, device: object, dtype: object) -> object:
            self.moved = (device, dtype)
            return self

        def eval(self) -> object:
            self.evaluated = True
            return self

    model = RuntimeModel()
    monkeypatch.setattr(
        foundation_pareto,
        "_torch_load_checkpoint",
        lambda path: {"training_config": training_config},
    )

    def load_model(
        value: LocalCheckpointFoundationSpec,
        checkpoint: dict[str, object],
    ) -> RuntimeModel:
        loaded.append(value)
        return model

    monkeypatch.setattr(
        foundation_pareto,
        "_load_local_checkpoint_model",
        load_model,
    )

    encoder = load_foundation_encoder(spec)

    assert isinstance(encoder, LocalCheckpointFoundationEncoder)
    assert encoder.model is model
    assert encoder.audit.checkpoint_sha256 == _sha256(checkpoint)
    assert loaded == [spec]
    assert encoder.device == torch.device("cpu")
    assert model.moved == (torch.device("cpu"), torch.float32)
    assert model.evaluated is True


def test_local_loader_rejects_checkpoint_drift_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "anchor.pt"
    backbone = tmp_path / "bn-inception.pth"
    checkpoint.write_bytes(b"registered")
    backbone.write_bytes(b"backbone")
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_sha256(checkpoint),
        resolved_config_sha256=_canonical_json_digest({}),
        pretrained_backbone_sha256=_sha256(backbone),
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )
    checkpoint.write_bytes(b"drifted")
    loaded = False

    def must_not_load(value: LocalCheckpointFoundationSpec, checkpoint: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError(value)

    monkeypatch.setattr(foundation_pareto, "_load_local_checkpoint_model", must_not_load)

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        load_foundation_encoder(spec)

    assert loaded is False


def test_local_model_loader_reconstructs_exact_checkpoint_architecture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "anchor.pt"
    backbone = tmp_path / "bn-inception.pth"
    training_config = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "embedding_dimensions": 512,
        "head_pooling": "avg_max",
        "embedding_head_init": "kaiming_normal",
        "embedding_layer_norm": False,
    }
    checkpoint.write_bytes(b"authenticated-bytes")
    backbone.write_bytes(b"pretrained-backbone")
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_sha256(checkpoint),
        resolved_config_sha256=_canonical_json_digest(training_config),
        pretrained_backbone_sha256=_sha256(backbone),
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )
    state_dict = {"embedding.weight": object()}
    monkeypatch.setattr(
        foundation_pareto,
        "_torch_load_checkpoint",
        lambda path: {
            "state_dict": state_dict,
            "arch": {
                "backbone_name": "bn_inception",
                "pretrained_weights": "bn_inception_52deb4733",
                "embedding_dimensions": 512,
                "head_pooling": "avg_max",
                "embedding_head_init": "kaiming_normal",
                "embedding_layer_norm": False,
            },
            "training_config": training_config,
        },
    )

    class FakeModel:
        loaded: tuple[object, bool] | None = None
        evaluated = False

        def load_state_dict(self, value: object, strict: bool) -> None:
            self.loaded = (value, strict)

        def eval(self) -> None:
            self.evaluated = True

    model = FakeModel()
    builds: list[tuple[int, bool]] = []

    def build_model(*, embedding_size: int, add_gmp: bool) -> FakeModel:
        builds.append((embedding_size, add_gmp))
        return model

    monkeypatch.setattr(
        foundation_pareto,
        "_build_local_bn_inception",
        build_model,
    )

    loaded = foundation_pareto._load_local_checkpoint_model(spec)

    assert loaded is model
    assert builds == [(512, True)]
    assert model.loaded == (state_dict, True)
    assert model.evaluated is True


def test_local_model_loader_rejects_architecture_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "anchor.pt"
    backbone = tmp_path / "bn-inception.pth"
    training_config = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "embedding_dimensions": 512,
        "head_pooling": "avg_max",
        "embedding_head_init": "kaiming_normal",
        "embedding_layer_norm": False,
    }
    checkpoint.write_bytes(b"authenticated-bytes")
    backbone.write_bytes(b"pretrained-backbone")
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_sha256(checkpoint),
        resolved_config_sha256=_canonical_json_digest(training_config),
        pretrained_backbone_sha256=_sha256(backbone),
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_torch_load_checkpoint",
        lambda path: {
            "state_dict": {},
            "arch": {**training_config, "embedding_layer_norm": True},
            "training_config": training_config,
        },
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_build_local_bn_inception",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(kwargs)),
    )

    with pytest.raises(ValueError, match="architecture"):
        foundation_pareto._load_local_checkpoint_model(spec)


def test_local_checkpoint_loader_reconstructs_real_bn_inception_state(
    tmp_path: Path,
) -> None:
    training_config = {
        "backbone_name": "bn_inception",
        "pretrained_weights": "bn_inception_52deb4733",
        "embedding_dimensions": 2,
        "head_pooling": "avg_max",
        "embedding_head_init": "kaiming_normal",
        "embedding_layer_norm": False,
    }
    source_model = foundation_pareto._build_local_bn_inception(
        embedding_size=2,
        add_gmp=True,
    )
    expected = source_model.state_dict()["model.embedding.bias"].clone()
    checkpoint = tmp_path / "anchor.pt"
    torch.save(
        {
            "state_dict": source_model.state_dict(),
            "arch": dict(training_config),
            "training_config": dict(training_config),
        },
        checkpoint,
    )
    del source_model
    gc.collect()
    backbone = tmp_path / "bn-inception.pth"
    backbone.write_bytes(b"registered-upstream-backbone-authority")
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=checkpoint,
        pretrained_backbone_path=backbone,
        checkpoint_sha256=_sha256(checkpoint),
        resolved_config_sha256=_canonical_json_digest(training_config),
        pretrained_backbone_sha256=_sha256(backbone),
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=2,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )

    encoder = load_foundation_encoder(spec)

    torch.testing.assert_close(
        encoder.model.state_dict()["model.embedding.bias"],
        expected,
    )
    assert encoder.model.training is False


def test_remote_loader_uses_exact_revision_and_authenticated_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _remote_spec()
    calls: list[tuple[str, str, dict[str, object]]] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            calls.append(("processor", model_id, kwargs))
            return SimpleNamespace(name="processor")

    class FakeModel:
        moved: tuple[object, object] | None = None
        evaluated = False

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            calls.append(("model", model_id, kwargs))
            return cls()

        def to(self, *, device: object, dtype: object) -> object:
            self.moved = (device, dtype)
            return self

        def eval(self) -> object:
            self.evaluated = True
            return self

    monkeypatch.setattr(
        foundation_pareto,
        "_load_transformers_dependencies",
        lambda: (FakeProcessor, FakeModel),
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_observe_remote_snapshot",
        lambda value: (
            Path("/authenticated/snapshot"),
            FoundationEncoderAudit(
                status="available",
                model_id=value.model_id,
                revision=value.revision,
                weight_sha256=value.weight_sha256,
                processor_sha256=value.processor_sha256,
                config_sha256=value.config_sha256,
                reason=None,
            ),
        ),
    )

    encoder = load_foundation_encoder(spec)

    assert isinstance(encoder, TransformersFoundationEncoder)
    assert encoder.audit.revision == spec.revision
    assert calls == [
        (
            "processor",
            "/authenticated/snapshot",
            {"revision": spec.revision, "local_files_only": True},
        ),
        (
            "model",
            "/authenticated/snapshot",
            {
                "revision": spec.revision,
                "local_files_only": True,
                "torch_dtype": torch.float32,
            },
        ),
    ]
    assert encoder.model.moved == (torch.device("cpu"), torch.float32)
    assert encoder.model.evaluated is True


def test_remote_loader_rejects_observed_digest_drift_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _remote_spec()
    loaded = False

    class MustNotLoad:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
            nonlocal loaded
            loaded = True
            raise AssertionError((model_id, kwargs))

    monkeypatch.setattr(
        foundation_pareto,
        "_load_transformers_dependencies",
        lambda: (MustNotLoad, MustNotLoad),
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_observe_remote_snapshot",
        lambda value: (
            Path("/authenticated/snapshot"),
            FoundationEncoderAudit(
                status="available",
                model_id=value.model_id,
                revision=value.revision,
                weight_sha256=value.weight_sha256,
                processor_sha256=value.processor_sha256,
                config_sha256="d" * 64,
                reason=None,
            ),
        ),
    )

    with pytest.raises(ValueError, match="config_sha256"):
        load_foundation_encoder(spec)

    assert loaded is False


def test_remote_artifact_observer_uses_exact_revision_and_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = b'{"architectures":["TinyVision"]}'
    processor = b'{"size":224}'
    weights = b"safetensor-bytes"
    (tmp_path / "config.json").write_bytes(config)
    (tmp_path / "preprocessor_config.json").write_bytes(processor)
    (tmp_path / "model.safetensors").write_bytes(weights)
    stale = tmp_path / "onnx"
    stale.mkdir()
    (stale / "model.safetensors").write_bytes(b"stale-nested-cache")
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def snapshot(repo_id: str, *, revision: str, allow_patterns: tuple[str, ...]) -> str:
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(foundation_pareto, "_snapshot_download", snapshot)
    spec = _remote_spec(
        weight_sha256=hashlib.sha256(weights).hexdigest(),
        processor_sha256=hashlib.sha256(processor).hexdigest(),
        config_sha256=hashlib.sha256(config).hexdigest(),
    )

    audit = foundation_pareto._observe_remote_artifacts(spec)

    assert audit.status == "available"
    assert audit.weight_sha256 == spec.weight_sha256
    assert audit.processor_sha256 == spec.processor_sha256
    assert audit.config_sha256 == spec.config_sha256
    assert calls == [
        (
            spec.model_id,
            spec.revision,
            (
                "config.json",
                "preprocessor_config.json",
                "processor_config.json",
                "*.safetensors",
                "*.safetensors.index.json",
                "pytorch_model*.bin",
                "pytorch_model*.bin.index.json",
            ),
        )
    ]


def test_remote_artifact_observer_records_gated_snapshot_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        foundation_pareto,
        "_snapshot_download",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("401 gated repository")),
    )

    audit = foundation_pareto._observe_remote_artifacts(_remote_spec())

    assert audit.status == "unavailable"
    assert audit.reason == "snapshot unavailable: 401 gated repository"
    assert audit.model_id == _remote_spec().model_id
    assert audit.revision == _remote_spec().revision
    assert audit.weight_sha256 is None
    assert audit.processor_sha256 is None
    assert audit.config_sha256 is None


def test_remote_artifact_observer_authenticates_huggingface_cache_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "models--org--model"
    root = model_root / "snapshots" / "revision"
    blobs = model_root / "blobs"
    root.mkdir(parents=True)
    blobs.mkdir()
    payloads = {
        "config.json": b'{"model_type":"tiny"}',
        "preprocessor_config.json": b'{"size":224}',
        "model.safetensors": b"registered-weights",
    }
    for name, payload in payloads.items():
        (blobs / name).write_bytes(payload)
        (root / name).symlink_to(Path("../../blobs") / name)
    monkeypatch.setattr(
        foundation_pareto,
        "_snapshot_download",
        lambda *args, **kwargs: str(root),
    )
    spec = _remote_spec(
        weight_sha256=hashlib.sha256(payloads["model.safetensors"]).hexdigest(),
        processor_sha256=hashlib.sha256(payloads["preprocessor_config.json"]).hexdigest(),
        config_sha256=hashlib.sha256(payloads["config.json"]).hexdigest(),
    )

    observed_root, audit = foundation_pareto._observe_remote_snapshot(spec)

    assert observed_root == root
    assert audit.status == "available"
    foundation_pareto._require_matching_remote_audit(spec, audit)


def test_remote_artifact_observer_rejects_cache_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "models--org--model" / "snapshots" / "revision"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_bytes(b"{}")
    (root / "config.json").symlink_to(outside / "config.json")
    (root / "preprocessor_config.json").write_bytes(b"{}")
    (root / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        foundation_pareto,
        "_snapshot_download",
        lambda *args, **kwargs: str(root),
    )

    with pytest.raises(ValueError, match="escapes authenticated cache scope"):
        foundation_pareto._observe_remote_snapshot(_remote_spec())


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "drift"},
        {"revision": "main"},
        {"weight_sha256": ""},
        {"status": "available", "reason": "unexpected"},
        {"status": "unavailable", "reason": None},
    ],
)
def test_remote_audit_rejects_malformed_status_and_authority(
    changes: dict[str, object],
) -> None:
    spec = _remote_spec()
    values: dict[str, object] = {
        "status": "available",
        "model_id": spec.model_id,
        "revision": spec.revision,
        "weight_sha256": spec.weight_sha256,
        "processor_sha256": spec.processor_sha256,
        "config_sha256": spec.config_sha256,
        "reason": None,
    }
    values.update(changes)
    with pytest.raises(ValueError):
        FoundationEncoderAudit(**values)  # type: ignore[arg-type]


def test_unavailable_remote_audit_is_rejected_before_digest_comparison() -> None:
    spec = _remote_spec()
    audit = FoundationEncoderAudit(
        status="unavailable",
        model_id=spec.model_id,
        revision=spec.revision,
        weight_sha256="d" * 64,
        processor_sha256=spec.processor_sha256,
        config_sha256=spec.config_sha256,
        reason="gated",
    )
    with pytest.raises(ValueError, match="authority is unavailable"):
        foundation_pareto._require_matching_remote_audit(spec, audit)


@pytest.mark.parametrize(
    ("pooling", "expected"),
    [
        ("image_features", [[3.0, 4.0], [0.0, 5.0], [4.0, 3.0]]),
        ("pooler", [[4.0, 3.0], [5.0, 0.0], [3.0, 4.0]]),
        ("cls", [[13.0, 14.0], [10.0, 15.0], [14.0, 13.0]]),
    ],
)
def test_remote_encoder_batches_and_uses_only_registered_pooling(
    pooling: str,
    expected: list[list[float]],
) -> None:
    class Processor:
        def __call__(
            self,
            *,
            images: list[object],
            return_tensors: str,
            size: dict[str, int],
        ) -> dict[str, object]:
            assert return_tensors == "pt"
            assert size == {"height": 224, "width": 224}
            return {"pixel_values": torch.tensor(images, dtype=torch.float32)}

    class Model:
        def get_image_features(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
            return pixel_values

        def __call__(self, *, pixel_values: torch.Tensor) -> object:
            return SimpleNamespace(
                pooler_output=torch.flip(pixel_values, dims=(1,)),
                last_hidden_state=torch.stack(
                    (pixel_values + 10.0, pixel_values),
                    dim=1,
                ),
            )

    spec = _remote_spec(pooling=pooling, normalize=False)
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )

    actual = encoder.encode(
        [[3.0, 4.0], [0.0, 5.0], [4.0, 3.0]],
        batch_size=2,
        normalize_embeddings=False,
    )

    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float32))
    assert actual.dtype == np.float32


def test_remote_image_features_accepts_transformers_pooling_output() -> None:
    class Processor:
        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"pixel_values": torch.tensor([[3.0, 4.0]], dtype=torch.float32)}

    class Model:
        def get_image_features(self, *, pixel_values: torch.Tensor) -> object:
            return SimpleNamespace(pooler_output=pixel_values)

    spec = _remote_spec(pooling="image_features", normalize=False)
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )

    actual = encoder.encode([[3.0, 4.0]], batch_size=1, normalize_embeddings=False)

    np.testing.assert_array_equal(actual, np.asarray([[3.0, 4.0]], dtype=np.float32))


def test_local_encoder_applies_registered_transform_batches_and_normalizes() -> None:
    transformed: list[object] = []

    def transform(image: object) -> torch.Tensor:
        transformed.append(image)
        return torch.tensor(image, dtype=torch.float32)

    class Model:
        def __call__(self, value: torch.Tensor) -> torch.Tensor:
            return value * 2.0

    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=Path("anchor.pt"),
        pretrained_backbone_path=Path("backbone.pt"),
        checkpoint_sha256=_SHA_A,
        resolved_config_sha256=_SHA_B,
        pretrained_backbone_sha256=_SHA_C,
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=2,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )
    encoder = LocalCheckpointFoundationEncoder(
        spec=spec,
        model=Model(),
        transform=transform,
        device=torch.device("cpu"),
        audit=LocalFoundationEncoderAudit(
            checkpoint_sha256=_SHA_A,
            resolved_config_sha256=_SHA_B,
            pretrained_backbone_sha256=_SHA_C,
        ),
    )

    actual = encoder.encode(
        [[3.0, 4.0], [0.0, 5.0], [4.0, 3.0]],
        batch_size=2,
        normalize_embeddings=True,
    )

    np.testing.assert_allclose(
        actual,
        np.asarray([[0.6, 0.8], [0.0, 1.0], [0.8, 0.6]], dtype=np.float32),
    )
    raw = encoder.encode(
        [[3.0, 4.0], [0.0, 5.0], [4.0, 3.0]],
        batch_size=2,
        normalize_embeddings=False,
    )
    np.testing.assert_allclose(
        raw,
        np.asarray([[6.0, 8.0], [0.0, 10.0], [8.0, 6.0]], dtype=np.float32),
    )
    assert transformed == [
        [3.0, 4.0],
        [0.0, 5.0],
        [4.0, 3.0],
        [3.0, 4.0],
        [0.0, 5.0],
        [4.0, 3.0],
    ]


def test_remote_encoder_supports_registered_and_raw_cache_normalization() -> None:
    class Processor:
        def __call__(
            self,
            *,
            images: list[object],
            return_tensors: str,
            size: dict[str, int],
        ) -> dict[str, object]:
            assert return_tensors == "pt"
            assert size == {"height": 224, "width": 224}
            return {"pixel_values": torch.tensor(images, dtype=torch.float32)}

    class Model:
        def get_image_features(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
            assert pixel_values.dtype is torch.bfloat16
            return pixel_values

    spec = _remote_spec(pooling="image_features", dtype="bfloat16", normalize=True)
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )

    actual = encoder.encode([[3.0, 4.0]], batch_size=1, normalize_embeddings=True)

    np.testing.assert_allclose(actual, np.asarray([[0.6, 0.8]]), atol=0.005)
    assert actual.dtype == np.float32
    raw = encoder.encode([[3.0, 4.0]], batch_size=1, normalize_embeddings=False)
    np.testing.assert_allclose(raw, np.asarray([[3.0, 4.0]]), atol=0.005)


def test_remote_encoder_rejects_wrong_rank_or_embedding_width() -> None:
    class Processor:
        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"pixel_values": torch.ones((1, 2), dtype=torch.float32)}

    class Model:
        def get_image_features(self, **kwargs: object) -> torch.Tensor:
            return torch.ones((1, 2, 1), dtype=torch.float32)

    spec = _remote_spec(pooling="image_features", normalize=False)
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )
    with pytest.raises(ValueError, match="rank-2"):
        encoder.encode([[1.0, 2.0]], batch_size=1, normalize_embeddings=False)


def test_remote_encoder_rejects_nonfinite_output() -> None:
    class Processor:
        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"pixel_values": torch.ones((1, 2), dtype=torch.float32)}

    class Model:
        def get_image_features(self, **kwargs: object) -> torch.Tensor:
            return torch.tensor([[float("nan"), 1.0]], dtype=torch.float32)

    spec = _remote_spec(pooling="image_features", normalize=False)
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )
    with pytest.raises(ValueError, match="finite"):
        encoder.encode([[1.0, 2.0]], batch_size=1, normalize_embeddings=False)


def _fixture_records(
    tmp_path: Path | None = None,
) -> tuple[list[NativeFixtureRecord], list[MetricToleranceRecord]]:
    input_sha = _SHA_A
    source_sha = _SHA_B
    if tmp_path is not None:
        input_path = tmp_path / "fixture.bin"
        source_path = tmp_path / "native-source.bin"
        input_path.write_bytes(b"registered-fixture-input")
        source_path.write_bytes(b"registered-native-source")
        input_sha = _sha256(input_path)
        source_sha = _sha256(source_path)
    fixtures = [
        NativeFixtureRecord(
            arm="dino-v3-s",
            metric="embedding_cosine",
            native_value=0.75,
            input_sha256=input_sha,
            source_sha256=source_sha,
            native_cross_check="available",
            reason=None,
        ),
        NativeFixtureRecord(
            arm="dino-v3-s",
            metric="recall_at_1",
            native_value=None,
            input_sha256=input_sha,
            source_sha256=source_sha,
            native_cross_check="unavailable",
            reason="upstream fixture has embeddings but no labels",
        ),
    ]
    tolerances = [
        MetricToleranceRecord(
            arm="dino-v3-s",
            metric="embedding_cosine",
            tolerance=0.01,
            frozen_before_execution=True,
        ),
        MetricToleranceRecord(
            arm="dino-v3-s",
            metric="recall_at_1",
            tolerance=0.0,
            frozen_before_execution=True,
        ),
    ]
    return fixtures, tolerances


def test_native_fixture_authority_requires_exact_ordered_arm_metric_pairs() -> None:
    fixtures, tolerances = _fixture_records()
    pairs = (("dino-v3-s", "embedding_cosine"), ("dino-v3-s", "recall_at_1"))

    validate_native_fixture_authority(fixtures, tolerances, registered_pairs=pairs)

    with pytest.raises(ValueError, match="fixture key set"):
        validate_native_fixture_authority(fixtures[:-1], tolerances, registered_pairs=pairs)
    with pytest.raises(ValueError, match="tolerance key set"):
        validate_native_fixture_authority(fixtures, tolerances[:-1], registered_pairs=pairs)
    with pytest.raises(ValueError, match="ordered keys"):
        validate_native_fixture_authority(fixtures[::-1], tolerances, registered_pairs=pairs)


def test_native_fixture_failure_gates_before_export(tmp_path: Path) -> None:
    fixtures, tolerances = _fixture_records(tmp_path)
    pairs = (("dino-v3-s", "embedding_cosine"), ("dino-v3-s", "recall_at_1"))
    values = {"embedding_cosine": 0.70, "recall_at_1": 0.5}
    calls: list[tuple[object, str]] = []

    def metric(encoder: object, input_path: Path, source_path: Path, name: str) -> float:
        assert input_path.read_bytes() == b"registered-fixture-input"
        assert source_path.read_bytes() == b"registered-native-source"
        calls.append((encoder, name))
        return values[name]

    audits = verify_native_fixture(
        arm="dino-v3-s",
        encoder=SimpleNamespace(spec=SimpleNamespace(arm="dino-v3-s")),
        fixture_inputs={
            "embedding_cosine": tmp_path / "fixture.bin",
            "recall_at_1": tmp_path / "fixture.bin",
        },
        native_sources={
            "embedding_cosine": tmp_path / "native-source.bin",
            "recall_at_1": tmp_path / "native-source.bin",
        },
        repository_metric=metric,
        fixtures=fixtures,
        tolerances=tolerances,
        registered_pairs=pairs,
    )

    assert audits[0].passed is False
    assert audits[0].provenance == "native_cross_check"
    assert audits[1].passed is None
    assert audits[1].provenance == "unavailable"
    assert [name for _, name in calls] == ["embedding_cosine", "recall_at_1"]


def test_native_fixture_rejects_unregistered_or_missing_arm_and_accepts_boundary(
    tmp_path: Path,
) -> None:
    fixtures, tolerances = _fixture_records(tmp_path)
    pairs = (("dino-v3-s", "embedding_cosine"), ("dino-v3-s", "recall_at_1"))
    inputs = {
        "embedding_cosine": tmp_path / "fixture.bin",
        "recall_at_1": tmp_path / "fixture.bin",
    }
    sources = {
        "embedding_cosine": tmp_path / "native-source.bin",
        "recall_at_1": tmp_path / "native-source.bin",
    }
    values = {"embedding_cosine": 0.755, "recall_at_1": 0.5}

    def metric(encoder: object, input_path: Path, source_path: Path, name: str) -> float:
        return values[name]

    audits = verify_native_fixture(
        arm="dino-v3-s",
        encoder=SimpleNamespace(spec=SimpleNamespace(arm="dino-v3-s")),
        fixture_inputs=inputs,
        native_sources=sources,
        repository_metric=metric,
        fixtures=fixtures,
        tolerances=tolerances,
        registered_pairs=pairs,
    )
    assert audits[0].passed is True

    with pytest.raises(ValueError, match="no registered native fixture pairs"):
        verify_native_fixture(
            arm="missing",
            encoder=SimpleNamespace(spec=SimpleNamespace(arm="missing")),
            fixture_inputs={},
            native_sources={},
            repository_metric=metric,
            fixtures=fixtures,
            tolerances=tolerances,
            registered_pairs=pairs,
        )

    with pytest.raises(ValueError, match="encoder arm differs"):
        verify_native_fixture(
            arm="dino-v3-s",
            encoder=SimpleNamespace(spec=SimpleNamespace(arm="siglip2")),
            fixture_inputs=inputs,
            native_sources=sources,
            repository_metric=metric,
            fixtures=fixtures,
            tolerances=tolerances,
            registered_pairs=pairs,
        )

    with pytest.raises(ValueError, match="fixture key set"):
        verify_native_fixture(
            arm="dino-v3-s",
            encoder=SimpleNamespace(spec=SimpleNamespace(arm="dino-v3-s")),
            fixture_inputs=inputs,
            native_sources=sources,
            repository_metric=metric,
            fixtures=fixtures[:-1],
            tolerances=tolerances,
            registered_pairs=pairs,
        )

    (tmp_path / "fixture.bin").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="fixture input digest differs"):
        verify_native_fixture(
            arm="dino-v3-s",
            encoder=SimpleNamespace(spec=SimpleNamespace(arm="dino-v3-s")),
            fixture_inputs=inputs,
            native_sources=sources,
            repository_metric=lambda *args: (_ for _ in ()).throw(AssertionError(args)),
            fixtures=fixtures,
            tolerances=tolerances,
            registered_pairs=pairs,
        )


def test_published_metric_cross_check_is_post_evaluation_and_repository_only() -> None:
    records = [
        PublishedMetricRecord(
            arm="unicom",
            metric="recall_at_1",
            native_value=90.0,
            tolerance=0.25,
            source="official table 1",
            provenance="native_cross_check",
        ),
        PublishedMetricRecord(
            arm="unicom",
            metric="recall_at_100",
            native_value=None,
            tolerance=None,
            source="official table 1 omits R@100",
            provenance="repository_only",
        ),
    ]

    audits = cross_check_published_metrics(
        arm="unicom",
        repository_values={"recall_at_1": 89.5, "recall_at_100": 99.0},
        records=records,
        registered_pairs=(("unicom", "recall_at_1"), ("unicom", "recall_at_100")),
    )

    assert audits[0].passed is False
    assert audits[0].invalidates_confirmatory_claim is True
    assert audits[1].passed is None
    assert audits[1].provenance == "repository_only"


def test_published_metrics_require_registered_complete_arm_and_accept_boundary() -> None:
    records = [
        PublishedMetricRecord(
            arm="unicom",
            metric="recall_at_1",
            native_value=90.0,
            tolerance=0.25,
            source="official table 1",
            provenance="native_cross_check",
        )
    ]
    pairs = (("unicom", "recall_at_1"),)
    audits = cross_check_published_metrics(
        arm="unicom",
        repository_values={"recall_at_1": 89.75},
        records=records,
        registered_pairs=pairs,
    )
    assert audits[0].passed is True

    with pytest.raises(ValueError, match="no registered published metric pairs"):
        cross_check_published_metrics(
            arm="missing",
            repository_values={},
            records=records,
            registered_pairs=pairs,
        )
    with pytest.raises(ValueError, match="ordered keys"):
        cross_check_published_metrics(
            arm="unicom",
            repository_values={"recall_at_1": 90.0},
            records=[*records, records[0]],
            registered_pairs=pairs,
        )


def test_native_fixture_register_strictly_loads_complete_ordered_json(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    tolerance_path = tmp_path / "tolerances.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-native-fixtures-v1",
                "status": "frozen",
                "records": [
                    {
                        "arm": "dino-v3-s",
                        "metric": "embedding_cosine",
                        "native_value": 0.75,
                        "input_sha256": _SHA_A,
                        "source_sha256": _SHA_B,
                        "native_cross_check": "available",
                        "reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tolerance_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-metric-tolerances-v1",
                "status": "frozen",
                "records": [
                    {
                        "arm": "dino-v3-s",
                        "metric": "embedding_cosine",
                        "tolerance": 0.01,
                        "frozen_before_execution": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    fixtures, tolerances = load_native_fixture_authority(
        fixture_path,
        tolerance_path,
        registered_pairs=(("dino-v3-s", "embedding_cosine"),),
    )

    assert fixtures[0].native_value == 0.75
    assert tolerances[0].tolerance == 0.01

    fixture_path.write_text(
        fixture_path.read_text(encoding="utf-8").replace(
            '"reason": null',
            '"reason": null, "extra": 1',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fixture record keys"):
        load_native_fixture_authority(
            fixture_path,
            tolerance_path,
            registered_pairs=(("dino-v3-s", "embedding_cosine"),),
        )


def test_published_metric_register_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "published.json"
    path.write_text(
        '{"schema_version":"foundation-published-metrics-v1",'
        '"schema_version":"drift","records":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_published_metric_register(path)


def test_repository_fidelity_authorities_are_strict_and_prospectively_empty() -> None:
    root = Path(__file__).resolve().parents[1]

    fixtures, tolerances = load_native_fixture_authority(
        root / "docs/foundation_native_fixtures.json",
        root / "docs/foundation_metric_tolerances.json",
        registered_pairs=(),
        require_frozen=False,
    )
    published = load_published_metric_register(
        root / "docs/foundation_published_metric_register.json",
        require_frozen=False,
    )
    test_reads = foundation_pareto.load_test_read_register(
        root / "docs/foundation_test_read_register.json",
        require_frozen=False,
    )

    assert fixtures == ()
    assert tolerances == ()
    assert published == ()
    assert test_reads.records == ()

    with pytest.raises(ValueError, match="authority is not frozen"):
        load_native_fixture_authority(
            root / "docs/foundation_native_fixtures.json",
            root / "docs/foundation_metric_tolerances.json",
            registered_pairs=(),
        )
    with pytest.raises(ValueError, match="authority is not frozen"):
        load_published_metric_register(root / "docs/foundation_published_metric_register.json")
    with pytest.raises(ValueError, match="authority is not frozen"):
        foundation_pareto.load_test_read_register(root / "docs/foundation_test_read_register.json")


def test_unfrozen_authorities_cannot_carry_values_even_for_inspection(
    tmp_path: Path,
) -> None:
    fixture_path = tmp_path / "fixtures.json"
    tolerance_path = tmp_path / "tolerances.json"
    published_path = tmp_path / "published.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-native-fixtures-v1",
                "status": "prospective_unfrozen",
                "records": [
                    {
                        "arm": "arm",
                        "metric": "metric",
                        "native_value": 1.0,
                        "input_sha256": _SHA_A,
                        "source_sha256": _SHA_B,
                        "native_cross_check": "available",
                        "reason": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    tolerance_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-metric-tolerances-v1",
                "status": "prospective_unfrozen",
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    published_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-published-metrics-v1",
                "status": "prospective_unfrozen",
                "records": [
                    {
                        "arm": "arm",
                        "metric": "metric",
                        "native_value": 1.0,
                        "tolerance": 99.0,
                        "source": "post-hoc",
                        "provenance": "native_cross_check",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be empty"):
        load_native_fixture_authority(
            fixture_path,
            tolerance_path,
            registered_pairs=(("arm", "metric"),),
            require_frozen=False,
        )
    with pytest.raises(ValueError, match="must be empty"):
        load_published_metric_register(published_path, require_frozen=False)


def _cache_key(**changes: object) -> EmbeddingCacheKeyV2:
    values: dict[str, object] = {
        "arm": "siglip2",
        "model_revision": "1" * 40,
        "weight_sha256": _SHA_A,
        "processor_sha256": _SHA_B,
        "transform_id": "official-eval-view-v1",
        "resolution": 256,
        "dtype": "float32",
        "storage_dtype": "float32",
        "normalize": True,
        "dataset_rows_sha256": _SHA_C,
        "split": "query",
    }
    values.update(changes)
    return EmbeddingCacheKeyV2(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "2" * 40),
        ("weight_sha256", "d" * 64),
        ("processor_sha256", "e" * 64),
        ("transform_id", "official-eval-view-v2"),
        ("resolution", 224),
        ("dtype", "bfloat16"),
        ("normalize", False),
        ("dataset_rows_sha256", "f" * 64),
        ("split", "gallery"),
    ],
)
def test_cache_v2_every_registered_identity_mutation_changes_path(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    baseline = _cache_key()
    mutated = _cache_key(**{field: value})

    assert baseline.cache_path(tmp_path) != mutated.cache_path(tmp_path)


def test_cache_v2_rejects_legacy_schema_and_row_order_drift(tmp_path: Path) -> None:
    key = _cache_key()
    path = key.cache_path(tmp_path)
    np.savez(
        path,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ids=np.asarray(["a", "b"]),
        labels=np.asarray(["x", "y"]),
    )
    with pytest.raises(ValueError, match="cache-v2"):
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=("a", "b"),
            expected_labels=("x", "y"),
        )

    path.unlink()
    export_embeddings_v2(
        path,
        key=key,
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ids=("a", "b"),
        labels=("x", "y"),
    )
    with pytest.raises(ValueError, match="row IDs"):
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=("b", "a"),
            expected_labels=("x", "y"),
        )
    with pytest.raises(ValueError, match="labels"):
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=("a", "b"),
            expected_labels=("y", "x"),
        )


def test_cache_v2_maps_local_checkpoint_identity_without_remote_coercion() -> None:
    spec = LocalCheckpointFoundationSpec(
        arm="proxy-anchor",
        checkpoint_path=Path("anchor.pt"),
        pretrained_backbone_path=Path("backbone.pt"),
        checkpoint_sha256=_SHA_A,
        resolved_config_sha256=_SHA_B,
        pretrained_backbone_sha256=_SHA_C,
        transform_id="proxy-anchor-eval-224-v1",
        embedding_width=512,
        pooling="embedding",
        dtype="float32",
        normalize=True,
    )

    key = EmbeddingCacheKeyV2.from_model_spec(
        spec,
        dataset_rows_sha256="d" * 64,
        split="gallery",
        resolution=224,
    )

    assert key.model_revision == spec.checkpoint_sha256
    assert key.weight_sha256 == spec.pretrained_backbone_sha256
    assert key.processor_sha256 == spec.resolved_config_sha256
    assert key.transform_id == spec.transform_id


def test_cache_v2_publication_is_no_clobber_and_strictly_reloads(tmp_path: Path) -> None:
    key = _cache_key()
    path = key.cache_path(tmp_path)
    embeddings = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    export_embeddings_v2(
        path,
        key=key,
        embeddings=embeddings,
        ids=("a", "b"),
        labels=("x", "y"),
    )
    original = path.read_bytes()

    actual = load_embeddings_v2(
        path,
        key=key,
        expected_ids=("a", "b"),
        expected_labels=("x", "y"),
    )
    np.testing.assert_array_equal(actual, embeddings)
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(FileExistsError):
        export_embeddings_v2(
            path,
            key=key,
            embeddings=np.zeros_like(embeddings),
            ids=("a", "b"),
            labels=("x", "y"),
        )
    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.tmp.*")) == []


def test_cache_v2_rolls_back_owned_publication_when_strict_reload_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = _cache_key()
    path = key.cache_path(tmp_path)
    monkeypatch.setattr(
        foundation_pareto,
        "load_embeddings_v2",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("reload failed")),
    )

    with pytest.raises(ValueError, match="reload failed"):
        export_embeddings_v2(
            path,
            key=key,
            embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
            ids=("a",),
            labels=("x",),
        )

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.tmp.*")) == []


def test_cache_v2_roundtrips_real_bfloat_compute_encoder_output(tmp_path: Path) -> None:
    class Processor:
        def __call__(self, **kwargs: object) -> dict[str, torch.Tensor]:
            return {"pixel_values": torch.tensor([[3.0, 4.0]], dtype=torch.float32)}

    class Model:
        def get_image_features(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
            assert pixel_values.dtype is torch.bfloat16
            return pixel_values

    spec = _remote_spec(
        pooling="image_features",
        dtype="bfloat16",
        normalize=True,
    )
    encoder = TransformersFoundationEncoder(
        spec=spec,
        processor=Processor(),
        model=Model(),
        device=torch.device("cpu"),
        audit=FoundationEncoderAudit(
            status="available",
            model_id=spec.model_id,
            revision=spec.revision,
            weight_sha256=spec.weight_sha256,
            processor_sha256=spec.processor_sha256,
            config_sha256=spec.config_sha256,
            reason=None,
        ),
    )
    embeddings = encoder.encode(
        [[3.0, 4.0]],
        batch_size=1,
        normalize_embeddings=True,
    )
    key = EmbeddingCacheKeyV2.from_model_spec(
        spec,
        dataset_rows_sha256="d" * 64,
        split="query",
    )
    path = key.cache_path(tmp_path)

    export_embeddings_v2(
        path,
        key=key,
        embeddings=embeddings,
        ids=("a",),
        labels=("x",),
    )
    actual = load_embeddings_v2(
        path,
        key=key,
        expected_ids=("a",),
        expected_labels=("x",),
    )

    assert key.dtype == "bfloat16"
    assert key.storage_dtype == "float32"
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, embeddings)


def test_cache_v2_rejects_row_count_and_nonobject_metadata(tmp_path: Path) -> None:
    key = _cache_key()
    path = key.cache_path(tmp_path)
    embeddings = np.zeros((3, 2), dtype=np.float32)
    metadata = foundation_pareto._cache_metadata(
        key=key,
        embeddings=embeddings,
        ids=("a", "b"),
        labels=("x", "y"),
    )
    np.savez(
        path,
        embeddings=embeddings,
        metadata_json=np.frombuffer(
            foundation_pareto._canonical_json_bytes(metadata),
            dtype=np.uint8,
        ),
    )
    with pytest.raises(ValueError, match="row counts"):
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=("a", "b"),
            expected_labels=("x", "y"),
        )

    path.unlink()
    np.savez(
        path,
        embeddings=np.zeros((2, 2), dtype=np.float32),
        metadata_json=np.frombuffer(b"[]", dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="metadata root"):
        load_embeddings_v2(
            path,
            key=key,
            expected_ids=("a", "b"),
            expected_labels=("x", "y"),
        )


def test_geometry_evaluator_returns_all_registered_rankings_without_selection() -> None:
    query = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    query_labels = np.asarray([10, 20], dtype=np.int64)
    gallery = np.asarray(
        [
            [100.0, 1.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    gallery_labels = np.asarray([10, 99, 20, 98], dtype=np.int64)

    rows = evaluate_foundation_geometries(
        query,
        query_labels,
        gallery,
        gallery_labels,
    )

    assert [row.geometry for row in rows] == [
        "normalized_cosine",
        "normalized_euclidean",
        "native_unnormalized_euclidean",
    ]
    # Hand-computed: after normalization, query 0 ranks collinear gallery 1 then 0;
    # native Euclidean instead ranks gallery 1 then gallery 3, with huge gallery 0 last.
    assert rows[0].gallery_order[0] == (1, 0, 3, 2)
    assert rows[1].gallery_order == rows[0].gallery_order
    assert rows[2].gallery_order[0] == (1, 3, 2, 0)
    assert rows[0].metrics.recall_at_2 == 1.0
    assert rows[1].metrics.recall_at_2 == 1.0
    assert rows[2].metrics.recall_at_2 == 0.5
    assert rows[0].metrics is not rows[1].metrics


def test_geometry_evaluator_retains_only_registered_retrieval_depth() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    query_labels = np.asarray([7], dtype=np.int64)
    gallery = np.column_stack(
        (
            np.arange(1.0, 152.0, dtype=np.float32),
            np.ones(151, dtype=np.float32),
        )
    )
    gallery_labels = np.arange(1000, 1151, dtype=np.int64)
    gallery_labels[99] = 7

    rows = evaluate_foundation_geometries(
        query,
        query_labels,
        gallery,
        gallery_labels,
    )

    assert all(len(row.gallery_order[0]) == 100 for row in rows)
    assert all(row.metrics.recall_at_100 == 1.0 for row in rows)


def test_profile_foundation_encoder_excludes_warmups_and_records_exact_costs() -> None:
    events: list[str] = []

    class Parameter:
        def __init__(self, count: int) -> None:
            self.count = count

        def numel(self) -> int:
            return self.count

    class Encoder:
        model = SimpleNamespace(parameters=lambda: (Parameter(5), Parameter(7)))
        device = torch.device("cpu")

        def encode(
            self,
            images: list[object],
            *,
            batch_size: int,
            normalize_embeddings: bool,
        ) -> np.ndarray:
            assert batch_size == len(images)
            assert normalize_embeddings is True
            events.append(f"encode:{batch_size}")
            return np.ones((len(images), 3), dtype=np.float32)

    clock_values = iter(value * 1_000_000 for value in (0, 1, 2, 6, 7, 16, 20, 22, 23, 28, 30, 38))
    peak_values = iter((111, 222))

    def read_peak_memory_bytes() -> int:
        events.append("read-memory")
        return next(peak_values)

    def count_macs(_encoder: object, images: Sequence[object]) -> int:
        events.append("count-macs")
        return len(images) * 100

    profile = profile_foundation_encoder(
        Encoder(),
        fixtures=("a", "b", "c", "d"),
        batch_sizes=(1, 2),
        warmup_iterations=2,
        measured_iterations=3,
        clock_ns=lambda: next(clock_values),
        synchronize=lambda: events.append("sync"),
        reset_peak_memory=lambda: events.append("reset"),
        read_peak_memory_bytes=read_peak_memory_bytes,
        mac_counter=count_macs,
    )

    assert [row.batch_size for row in profile.batches] == [1, 2]
    assert profile.batches[0].latency_samples_ms == (1.0, 4.0, 9.0)
    assert profile.batches[0].latency_p50_ms == 4.0
    assert profile.batches[0].latency_p95_ms == 8.5
    assert profile.batches[0].peak_memory_bytes == 111
    assert profile.batches[0].macs == 100
    assert profile.batches[0].mac_status == "available"
    assert profile.batches[1].latency_samples_ms == (2.0, 5.0, 8.0)
    assert profile.batches[1].latency_p50_ms == 5.0
    assert profile.batches[1].latency_p95_ms == pytest.approx(7.7)
    assert profile.batches[1].peak_memory_bytes == 222
    assert profile.batches[1].macs == 200
    assert profile.parameter_count == 12
    assert profile.warmup_iterations == 2
    assert profile.measured_iterations == 3
    assert profile.descriptor_rows == 4
    assert profile.descriptor_width == 3
    assert profile.descriptor_dtype == "float32"
    assert profile.descriptor_bytes == 48
    assert profile.device_type == "cpu"
    assert profile.torch_version == str(torch.__version__)
    assert profile.numpy_version == str(np.__version__)
    assert profile.python_version == platform.python_version()
    assert profile.cuda_version == (
        str(torch.version.cuda) if torch.version.cuda is not None else None
    )
    assert profile.transformers_version == version("transformers")
    assert profile.device_name != "cpu"
    assert [event for event in events if event.startswith("encode:")] == [
        *(["encode:1"] * 5),
        *(["encode:2"] * 5),
    ]
    assert events == [
        "encode:1",
        "encode:1",
        "reset",
        "sync",
        "encode:1",
        "sync",
        "sync",
        "encode:1",
        "sync",
        "sync",
        "encode:1",
        "sync",
        "read-memory",
        "count-macs",
        "encode:2",
        "encode:2",
        "reset",
        "sync",
        "encode:2",
        "sync",
        "sync",
        "encode:2",
        "sync",
        "sync",
        "encode:2",
        "sync",
        "read-memory",
        "count-macs",
    ]


def test_profile_foundation_encoder_records_missing_macs_as_unavailable() -> None:
    class Encoder:
        model = SimpleNamespace(parameters=lambda: ())
        device = torch.device("cpu")

        def encode(
            self,
            images: list[object],
            *,
            batch_size: int,
            normalize_embeddings: bool,
        ) -> np.ndarray:
            return np.zeros((len(images), 2), dtype=np.float32)

    times = iter((0, 1_000_000))
    profile = profile_foundation_encoder(
        Encoder(),
        fixtures=("a",),
        batch_sizes=(1,),
        warmup_iterations=0,
        measured_iterations=1,
        clock_ns=lambda: next(times),
        synchronize=lambda: None,
        reset_peak_memory=lambda: None,
        read_peak_memory_bytes=lambda: 0,
    )

    assert profile.batches[0].mac_status == "unavailable"
    assert profile.batches[0].macs is None


def _identity_blocks(*, identities: int = 8, examples_per_identity: int = 4) -> list[ImageExample]:
    return [
        ImageExample(
            example_id=f"identity-{label}-example-{index}",
            image=(label, index),
            label=label,
        )
        for label in range(identities)
        for index in range(examples_per_identity)
    ]


def test_probe_validation_split_is_identity_disjoint_repeatable_and_uses_registered_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    examples = _identity_blocks()
    calls: list[tuple[Sequence[ImageExample], float, int]] = []
    registered = foundation_pareto.class_disjoint_recipe_selection_split

    def observed_split(
        values: Sequence[ImageExample],
        *,
        fraction: float,
        seed: int,
    ) -> object:
        calls.append((values, fraction, seed))
        return registered(values, fraction=fraction, seed=seed)

    monkeypatch.setattr(
        foundation_pareto,
        "class_disjoint_recipe_selection_split",
        observed_split,
    )

    first = foundation_pareto.build_identity_disjoint_validation_split(
        examples,
        fraction=0.25,
        seed=17,
    )
    second = foundation_pareto.build_identity_disjoint_validation_split(
        examples,
        fraction=0.25,
        seed=17,
    )

    assert calls == [(examples, 0.25, 17), (examples, 0.25, 17)]
    optimization_labels = {row.label for row in first.optimization}
    validation_labels = {row.label for row in first.query + first.gallery}
    assert optimization_labels.isdisjoint(validation_labels)
    assert [row.example_id for row in first.optimization] == [
        row.example_id for row in second.optimization
    ]
    assert [row.example_id for row in first.query] == [row.example_id for row in second.query]
    assert [row.example_id for row in first.gallery] == [row.example_id for row in second.gallery]


@pytest.mark.parametrize(
    (
        "candidate_recall_points",
        "candidate_p95_ms",
        "candidate_descriptor_bytes_per_image",
        "expected_status",
        "expected_pareto",
        "expected_kind",
    ),
    [
        (1.14, 12.0, 2048, "CONTINUE", False, "quality_margin"),
        (79.6, 9.0, 2048, "CONTINUE", True, "both"),
        (79.75, 10.0, 2048, "CONTINUE", False, "quality_margin"),
        (80.5, 9.0, 2048, "CONTINUE", True, "quality_margin"),
        (78.999999, 1.0, 1, "CLOSE_FOUNDATION_TRANSFER", True, "none"),
    ],
)
def test_f1_decision_uses_exact_quality_and_strict_pareto_boundaries(
    candidate_recall_points: float,
    candidate_p95_ms: float,
    candidate_descriptor_bytes_per_image: int,
    expected_status: str,
    expected_pareto: bool,
    expected_kind: str,
    decision_probes: tuple[object, object],
) -> None:
    candidate_base, comparator_base = decision_probes
    comparator_points = 2.14 if candidate_recall_points == 1.14 else 80.0
    decision = foundation_pareto.decide_f1(
        candidate_probe=replace(
            candidate_base,
            validation_recall_at_1_points=candidate_recall_points,
        ),
        comparator_probe=replace(
            comparator_base,
            validation_recall_at_1_points=comparator_points,
        ),
        candidate_encoder_p95_ms=candidate_p95_ms,
        comparator_encoder_p95_ms=10.0,
        candidate_descriptor_bytes_per_image=candidate_descriptor_bytes_per_image,
        comparator_descriptor_bytes_per_image=2048,
    )

    assert decision.status == expected_status
    assert decision.cost_pareto_dominant is expected_pareto
    assert decision.fidelity_only is (expected_status == "CLOSE_FOUNDATION_TRANSFER")
    assert decision.continuation_kind == expected_kind


def test_f1_decision_accepts_realistic_decimal_point_four_pareto_boundary(
    decision_probes: tuple[object, object],
) -> None:
    candidate_base, comparator_base = decision_probes
    decision = foundation_pareto.decide_f1(
        candidate_probe=replace(candidate_base, validation_recall_at_1_points=80.4),
        comparator_probe=replace(comparator_base, validation_recall_at_1_points=80.0),
        candidate_encoder_p95_ms=9.0,
        comparator_encoder_p95_ms=10.0,
        candidate_descriptor_bytes_per_image=2048,
        comparator_descriptor_bytes_per_image=2048,
    )

    assert decision.quality_within_point_four is True
    assert decision.cost_pareto_dominant is True
    assert decision.continuation_kind == "both"


def test_f1_decision_rejects_unavailable_comparator() -> None:
    decision = foundation_pareto.decide_f1(
        candidate_probe=None,
        comparator_probe=None,
        candidate_encoder_p95_ms=-1.0,
        comparator_encoder_p95_ms=None,
        candidate_descriptor_bytes_per_image=-1,
        comparator_descriptor_bytes_per_image=None,
    )

    assert decision.status == "UNAVAILABLE_COMPARATOR"
    assert decision.quality_gap_points is None
    assert decision.fidelity_only is False
    assert decision.authorized_followup == "resolve_local_comparator"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config", "different_config"),
        ("dataset", "sop"),
        ("split_sha256", "b" * 64),
        ("device_type", "cuda"),
    ],
)
def test_f1_decision_rejects_mismatched_probe_protocol_or_split(
    field: str,
    replacement: object,
    decision_probes: tuple[object, object],
) -> None:
    candidate, comparator = decision_probes
    if replacement == "different_config":
        replacement = replace(comparator.config, epochs=comparator.config.epochs + 1)
    with pytest.raises(ValueError, match="same probe protocol"):
        foundation_pareto.decide_f1(
            candidate_probe=candidate,
            comparator_probe=replace(comparator, **{field: replacement}),
            candidate_encoder_p95_ms=9.0,
            comparator_encoder_p95_ms=10.0,
            candidate_descriptor_bytes_per_image=2048,
            comparator_descriptor_bytes_per_image=2048,
        )


@pytest.mark.parametrize("invalid_side", ["candidate", "comparator"])
def test_f1_decision_requires_exact_probe_result_types(
    invalid_side: str,
    decision_probes: tuple[object, object],
) -> None:
    candidate, comparator = decision_probes
    invalid = SimpleNamespace(**vars(candidate))

    with pytest.raises(ValueError, match="exact BiasFreeProbeResult"):
        foundation_pareto.decide_f1(
            candidate_probe=invalid if invalid_side == "candidate" else candidate,
            comparator_probe=invalid if invalid_side == "comparator" else comparator,
            candidate_encoder_p95_ms=9.0,
            comparator_encoder_p95_ms=10.0,
            candidate_descriptor_bytes_per_image=2048,
            comparator_descriptor_bytes_per_image=2048,
        )


def _probe_train_only_fixture() -> tuple[object, dict[str, np.ndarray]]:
    examples = _identity_blocks(identities=8, examples_per_identity=2)
    examples.append(ImageExample(example_id="singleton", image=None, label=99))
    split = foundation_pareto.build_identity_disjoint_validation_split(
        examples,
        fraction=0.25,
        seed=23,
    )
    embeddings: dict[str, np.ndarray] = {}
    for example in examples:
        vector = np.zeros(8, dtype=np.float32)
        vector[example.label % 8] = 1.0
        embeddings[example.example_id] = vector
    return split, embeddings


@pytest.fixture(scope="module")
def decision_probes() -> tuple[object, object]:
    split, embeddings = _probe_train_only_fixture()
    config = foundation_pareto.ProbeTrainingConfig(
        identities_per_batch=4,
        epochs=1,
        learning_rates=(0.001,),
        temperatures=(0.05,),
        protocol_id="f1-bias-free-512-supcon-decision-test-v1",
    )
    candidate = foundation_pareto.fit_bias_free_probe_512(
        embeddings,
        split,
        arm_key="candidate",
        dataset="cars",
        split_seed=23,
        config=config,
    )
    comparator = foundation_pareto.fit_bias_free_probe_512(
        embeddings,
        split,
        arm_key="comparator",
        dataset="cars",
        split_seed=23,
        config=config,
    )
    return candidate, comparator


def test_probe_training_config_freezes_registered_supcon_protocol() -> None:
    config = foundation_pareto.ProbeTrainingConfig()

    assert config.output_dim == 512
    assert config.identities_per_batch == 32
    assert config.images_per_identity == 2
    assert config.epochs == 20
    assert config.learning_rates == (0.001, 0.003, 0.01)
    assert config.temperatures == (0.05, 0.10)
    assert config.adam_betas == (0.9, 0.999)
    assert config.adam_eps == 1e-8
    assert config.weight_decay == 0.0
    assert config.protocol_id == "f1-bias-free-512-supcon-v1"


def test_test_read_register_authenticates_and_allows_exactly_one_registered_read(
    tmp_path: Path,
) -> None:
    register_path = tmp_path / "test-read.json"
    register_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-test-read-register-v1",
                "status": "frozen",
                "records": [
                    {
                        "arm": "candidate",
                        "model_revision": "a" * 40,
                        "checkpoint_sha256": "b" * 64,
                        "metrics": ["recall_at_1", "recall_at_10"],
                        "purpose": "confirmatory_published_metric_cross_check",
                        "permitted_evaluations": 1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ledger = foundation_pareto.load_test_read_register(register_path)
    audit = ledger.consume(
        arm="candidate",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        metrics=("recall_at_1", "recall_at_10"),
        purpose="confirmatory_published_metric_cross_check",
    )

    assert audit.arm == "candidate"
    assert audit.evaluation_number == 1
    assert audit.metrics == ("recall_at_1", "recall_at_10")
    with pytest.raises(ValueError, match="already consumed"):
        ledger.consume(
            arm="candidate",
            model_revision="a" * 40,
            checkpoint_sha256="b" * 64,
            metrics=("recall_at_1", "recall_at_10"),
            purpose="confirmatory_published_metric_cross_check",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "c" * 40),
        ("checkpoint_sha256", "d" * 64),
        ("metrics", ("recall_at_1",)),
        ("purpose", "selection"),
    ],
)
def test_test_read_register_rejects_unregistered_official_read(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    register_path = tmp_path / "test-read.json"
    register_path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-test-read-register-v1",
                "status": "frozen",
                "records": [
                    {
                        "arm": "candidate",
                        "model_revision": "a" * 40,
                        "checkpoint_sha256": "b" * 64,
                        "metrics": ["recall_at_1", "recall_at_10"],
                        "purpose": "confirmatory_published_metric_cross_check",
                        "permitted_evaluations": 1,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger = foundation_pareto.load_test_read_register(register_path)
    request = {
        "arm": "candidate",
        "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "metrics": ("recall_at_1", "recall_at_10"),
        "purpose": "confirmatory_published_metric_cross_check",
    }
    request[field] = value

    with pytest.raises(ValueError, match="registered test read"):
        ledger.consume(**request)


def test_model_spec_authority_freezes_arm_order_roles_and_local_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "foundation-model-specs-v1",
                "status": "frozen",
                "arms": [
                    {
                        "kind": "remote",
                        "spec": {
                            "arm": "candidate",
                            "model_id": "org/model",
                            "revision": "a" * 40,
                            "weight_sha256": "b" * 64,
                            "processor_sha256": "c" * 64,
                            "config_sha256": "d" * 64,
                            "pooling": "cls",
                            "resolution": 224,
                            "embedding_width": 8,
                            "license": "Apache-2.0",
                            "dtype": "float32",
                            "normalize": True,
                        },
                        "cache_resolution": 224,
                        "role": "candidate",
                    },
                    {
                        "kind": "local",
                        "spec": {
                            "arm": "comparator",
                            "checkpoint_path": "artifacts/comparator.pt",
                            "pretrained_backbone_path": "artifacts/backbone.pth",
                            "checkpoint_sha256": "e" * 64,
                            "resolved_config_sha256": "f" * 64,
                            "pretrained_backbone_sha256": "1" * 64,
                            "transform_id": "proxy-anchor-eval-224-v1",
                            "embedding_width": 8,
                            "pooling": "embedding",
                            "dtype": "float32",
                            "normalize": True,
                        },
                        "cache_resolution": 224,
                        "role": "comparator",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    arms = foundation_pareto.load_foundation_model_specs(path)

    assert [arm.spec.arm for arm in arms] == ["candidate", "comparator"]
    assert [arm.role for arm in arms] == ["candidate", "comparator"]
    assert isinstance(arms[0].spec, RemoteFoundationModelSpec)
    assert isinstance(arms[1].spec, LocalCheckpointFoundationSpec)
    assert arms[1].cache_resolution == 224
    assert arms[1].spec.checkpoint_path == Path("artifacts/comparator.pt")

    remote_comparator = json.loads(path.read_text(encoding="utf-8"))
    remote_comparator["arms"][1] = {
        **remote_comparator["arms"][0],
        "spec": {**remote_comparator["arms"][0]["spec"], "arm": "comparator"},
        "role": "comparator",
    }
    path.write_text(json.dumps(remote_comparator) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local anchor"):
        foundation_pareto.load_foundation_model_specs(path)


def test_official_test_read_receipt_is_durable_no_clobber_and_precedes_loader(
    tmp_path: Path,
) -> None:
    record = foundation_pareto.FoundationTestReadRecord(
        arm="candidate",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        metrics=("recall_at_1",),
        purpose="confirmatory_published_metric_cross_check",
        permitted_evaluations=1,
    )
    first_ledger = foundation_pareto.FoundationTestReadLedger((record,))
    audit = first_ledger.consume(
        arm="candidate",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        metrics=("recall_at_1",),
        purpose="confirmatory_published_metric_cross_check",
    )
    receipt = foundation_pareto.publish_official_test_read_receipt(
        tmp_path,
        audit,
        decision_sha256="c" * 64,
    )
    loader_calls: list[str] = []
    value = foundation_pareto.load_registered_official_test(
        receipt,
        arm="candidate",
        metrics=("recall_at_1",),
        loader=lambda: loader_calls.append("loaded") or "official rows",
    )

    assert value == "official rows"
    assert loader_calls == ["loaded"]
    assert receipt.decision_sha256 == "c" * 64
    assert receipt.receipt_path.is_file()
    assert receipt.receipt_path.stat().st_mode & 0o777 == 0o600

    assert list(tmp_path.glob(f".{receipt.receipt_path.name}.tmp.*")) == []
    persisted = json.loads(receipt.receipt_path.read_text(encoding="utf-8"))
    assert persisted["decision_sha256"] == "c" * 64

    second_ledger = foundation_pareto.FoundationTestReadLedger((record,))
    second_audit = second_ledger.consume(
        arm="candidate",
        model_revision="a" * 40,
        checkpoint_sha256="b" * 64,
        metrics=("recall_at_1",),
        purpose="confirmatory_published_metric_cross_check",
    )
    with pytest.raises(FileExistsError):
        foundation_pareto.publish_official_test_read_receipt(
            tmp_path,
            second_audit,
            decision_sha256="c" * 64,
        )


def test_source_commit_uses_module_checkout_and_rejects_dirty_tracked_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def clean_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        output = "a" * 40 + "\n" if command[-2:] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(foundation_pareto.subprocess, "run", clean_run)
    assert foundation_pareto._source_commit() == "a" * 40
    repository = str(Path(foundation_pareto.__file__).resolve().parents[2])
    assert all(command[1:3] == ["-C", repository] for command in calls)

    def dirty_run(command: list[str], **kwargs: object) -> object:
        output = "a" * 40 + "\n" if command[-2:] == ["rev-parse", "HEAD"] else " M src/x.py\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(foundation_pareto.subprocess, "run", dirty_run)
    with pytest.raises(ValueError, match="dirty tracked bytes"):
        foundation_pareto._source_commit()


def test_official_test_loader_is_unreachable_for_mismatched_receipt(tmp_path: Path) -> None:
    ledger = foundation_pareto.FoundationTestReadLedger(
        (
            foundation_pareto.FoundationTestReadRecord(
                arm="candidate",
                model_revision="a" * 40,
                checkpoint_sha256="b" * 64,
                metrics=("recall_at_1",),
                purpose="confirmatory_published_metric_cross_check",
                permitted_evaluations=1,
            ),
        )
    )
    receipt = foundation_pareto.publish_official_test_read_receipt(
        tmp_path,
        ledger.consume(
            arm="candidate",
            model_revision="a" * 40,
            checkpoint_sha256="b" * 64,
            metrics=("recall_at_1",),
            purpose="confirmatory_published_metric_cross_check",
        ),
        decision_sha256="c" * 64,
    )
    called = False

    def forbidden_loader() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(ValueError, match="receipt differs"):
        foundation_pareto.load_registered_official_test(
            receipt,
            arm="other",
            metrics=("recall_at_1",),
            loader=forbidden_loader,
        )
    assert called is False


def test_foundation_screen_orders_f0_probe_decision_and_strict_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision_probes: tuple[object, object],
) -> None:
    candidate_probe, comparator_probe = decision_probes
    candidate_spec = _remote_spec(arm="candidate")
    comparator_spec = _local_spec()
    arms = (
        foundation_pareto.FoundationScreenArmSpec(
            kind="remote",
            spec=candidate_spec,
            cache_resolution=224,
            role="candidate",
        ),
        foundation_pareto.FoundationScreenArmSpec(
            kind="local",
            spec=comparator_spec,
            cache_resolution=224,
            role="comparator",
        ),
    )
    train = _identity_blocks(identities=8, examples_per_identity=2)
    trace: list[str] = []

    monkeypatch.setattr(foundation_pareto, "load_foundation_model_specs", lambda path: arms)
    monkeypatch.setattr(
        foundation_pareto,
        "load_native_fixture_authority",
        lambda *args, **kwargs: ((), ()),
    )

    def forbidden_published_register(*args: object, **kwargs: object) -> object:
        raise AssertionError("published outcomes are unreachable before an official test read")

    monkeypatch.setattr(
        foundation_pareto,
        "load_published_metric_register",
        forbidden_published_register,
    )
    monkeypatch.setattr(
        foundation_pareto,
        "load_test_read_register",
        lambda *args, **kwargs: foundation_pareto.FoundationTestReadLedger(()),
    )

    def fake_train_examples(**kwargs: object) -> object:
        assert kwargs["dataset_root"] == tmp_path / "data"
        return train

    monkeypatch.setattr(
        foundation_pareto,
        "load_image_retrieval_examples",
        fake_train_examples,
        raising=False,
    )

    def fake_load(spec: object) -> object:
        trace.append(f"{spec.arm}:authenticate")
        if isinstance(spec, LocalCheckpointFoundationSpec):
            return SimpleNamespace(
                spec=spec,
                audit=foundation_pareto.LocalFoundationEncoderAudit(
                    checkpoint_sha256=spec.checkpoint_sha256,
                    resolved_config_sha256=spec.resolved_config_sha256,
                    pretrained_backbone_sha256=spec.pretrained_backbone_sha256,
                ),
            )
        return SimpleNamespace(
            spec=spec,
            audit=FoundationEncoderAudit(
                status="available",
                model_id=spec.model_id,
                revision=spec.revision,
                weight_sha256=spec.weight_sha256,
                processor_sha256=spec.processor_sha256,
                config_sha256=spec.config_sha256,
                reason=None,
            ),
        )

    def fake_fixture(**kwargs: object) -> tuple[object, ...]:
        arm = str(kwargs["arm"])
        trace.append(f"{arm}:fixture")
        return (
            foundation_pareto.FoundationFidelityAudit(
                arm=arm,
                metric="embedding_cosine",
                native_value=1.0,
                repository_value=1.0,
                tolerance=0.0,
                provenance="native_cross_check",
                passed=True,
            ),
        )

    def fake_cache(**kwargs: object) -> object:
        arm = kwargs["arm_spec"].spec.arm
        trace.append(f"{arm}:cache")
        embeddings = {row.example_id: np.eye(8, dtype=np.float32)[row.label % 8] for row in train}
        return SimpleNamespace(
            train_embeddings=embeddings,
            records=(
                {
                    "arm": arm,
                    "split": "train",
                    "status": "exported",
                    "path": f"cache/{arm}.npz",
                    "rows": len(train),
                    "embedding_sha256": "a" * 64,
                },
            ),
        )

    def fake_profile(encoder: object, fixtures: object) -> object:
        trace.append(f"{encoder.spec.arm}:profile")
        return foundation_pareto.EncoderCostProfile(
            batches=(),
            parameter_count=1,
            warmup_iterations=10,
            measured_iterations=50,
            descriptor_rows=1,
            descriptor_width=8,
            descriptor_dtype="float32",
            descriptor_bytes=32,
            python_version="3.12.3",
            torch_version="2.12.1",
            numpy_version="2.5.0",
            transformers_version=None,
            cuda_version=None,
            device_type="cpu",
            device_name="test-cpu",
        )

    def fake_probe(*args: object, **kwargs: object) -> object:
        arm = str(kwargs["arm_key"])
        trace.append(f"{arm}:probe")
        return candidate_probe if arm == "candidate" else comparator_probe

    real_decide = foundation_pareto.decide_f1

    def traced_decision(**kwargs: object) -> object:
        trace.append("candidate:decision")
        return real_decide(**kwargs)

    monkeypatch.setattr(foundation_pareto, "load_foundation_encoder", fake_load)
    monkeypatch.setattr(foundation_pareto, "verify_native_fixture", fake_fixture)
    monkeypatch.setattr(
        foundation_pareto,
        "_prepare_foundation_train_cache",
        fake_cache,
        raising=False,
    )
    monkeypatch.setattr(foundation_pareto, "profile_foundation_encoder", fake_profile)
    monkeypatch.setattr(foundation_pareto, "fit_bias_free_probe_512", fake_probe)
    monkeypatch.setattr(foundation_pareto, "decide_f1", traced_decision)
    monkeypatch.setattr(foundation_pareto, "_source_commit", lambda: "f" * 40, raising=False)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    report = tmp_path / "report.json"

    written = foundation_pareto.run_foundation_screen(
        dataset="cars",
        dataset_root=tmp_path / "data",
        model_specs_path=tmp_path / "models.json",
        cache_dir=cache_dir,
        report_path=report,
        fixture_authority_path=tmp_path / "fixtures.json",
        tolerance_authority_path=tmp_path / "tolerances.json",
        published_register_path=tmp_path / "published.json",
        test_read_register_path=tmp_path / "test-reads.json",
        validation_seed=23,
        validation_fraction=0.25,
        allow_registered_test_read=False,
    )

    assert written == report
    assert trace == [
        "candidate:authenticate",
        "candidate:fixture",
        "candidate:cache",
        "candidate:profile",
        "candidate:probe",
        "comparator:authenticate",
        "comparator:fixture",
        "comparator:cache",
        "comparator:profile",
        "comparator:probe",
        "candidate:decision",
    ]
    persisted = foundation_pareto.load_foundation_screen_report(report)
    assert persisted["overall_status"] == "CONTINUE"
    assert persisted["registered_arms"] == ["candidate", "comparator"]
    assert persisted["official_test_reads"] == []
    assert persisted["published_metric_audits"] == []
    original_bytes = report.read_bytes()
    trace_before_no_clobber = list(trace)
    with pytest.raises(FileExistsError):
        foundation_pareto.run_foundation_screen(
            dataset="cars",
            dataset_root=tmp_path / "data",
            model_specs_path=tmp_path / "models.json",
            cache_dir=cache_dir,
            report_path=report,
            fixture_authority_path=tmp_path / "fixtures.json",
            tolerance_authority_path=tmp_path / "tolerances.json",
            published_register_path=tmp_path / "published.json",
            test_read_register_path=tmp_path / "test-reads.json",
            validation_seed=23,
            validation_fraction=0.25,
            allow_registered_test_read=True,
        )
    assert trace == trace_before_no_clobber
    with pytest.raises(FileExistsError):
        foundation_pareto.publish_foundation_screen_report(report, persisted)
    assert report.read_bytes() == original_bytes

    drifted = json.loads(json.dumps(persisted))
    drifted["f1_decisions"][0]["status"] = "CLOSE_FOUNDATION_TRANSFER"
    with pytest.raises(ValueError, match="decision digest"):
        foundation_pareto.validate_foundation_screen_report(drifted)

    forbidden = json.loads(json.dumps(persisted))
    forbidden["encoder_audits"][0]["kernel"] = "not-computed"
    with pytest.raises(ValueError, match="forbidden report key"):
        foundation_pareto.validate_foundation_screen_report(forbidden)

    nested_mutations: list[dict[str, object]] = []
    extra_encoder = json.loads(json.dumps(persisted))
    extra_encoder["encoder_audits"][0]["drift"] = True
    nested_mutations.append(extra_encoder)
    empty_encoder_audit = json.loads(json.dumps(persisted))
    empty_encoder_audit["encoder_audits"][0]["audit"] = {}
    nested_mutations.append(empty_encoder_audit)
    inconsistent_fixture = json.loads(json.dumps(persisted))
    inconsistent_fixture["fixture_fidelity_audits"][0]["passed"] = False
    nested_mutations.append(inconsistent_fixture)
    bad_cache_digest = json.loads(json.dumps(persisted))
    bad_cache_digest["cache_records"][0]["embedding_sha256"] = "BAD"
    nested_mutations.append(bad_cache_digest)
    bad_probe_digest = json.loads(json.dumps(persisted))
    bad_probe_digest["probe_audits"][0]["weight_sha256"] = "BAD"
    nested_mutations.append(bad_probe_digest)
    bad_profile_batch = json.loads(json.dumps(persisted))
    bad_profile_batch["cost_profiles"][0]["profile"]["batches"] = [{}]
    nested_mutations.append(bad_profile_batch)
    extra_decision = json.loads(json.dumps(persisted))
    extra_decision["f1_decisions"][0]["drift"] = True
    extra_decision["decision_sha256"] = foundation_pareto._decision_sha256(
        extra_decision["f1_decisions"], extra_decision["overall_status"]
    )
    nested_mutations.append(extra_decision)
    for mutation in nested_mutations:
        with pytest.raises(ValueError):
            foundation_pareto.validate_foundation_screen_report(mutation)

    def failing_candidate_fixture(**kwargs: object) -> tuple[object, ...]:
        arm = str(kwargs["arm"])
        trace.append(f"{arm}:fixture")
        return (
            foundation_pareto.FoundationFidelityAudit(
                arm=arm,
                metric="embedding_cosine",
                native_value=1.0,
                repository_value=0.0 if arm == "candidate" else 1.0,
                tolerance=0.0,
                provenance="native_cross_check",
                passed=arm != "candidate",
            ),
        )

    monkeypatch.setattr(
        foundation_pareto,
        "verify_native_fixture",
        failing_candidate_fixture,
    )
    trace.clear()
    closed_report = tmp_path / "closed.json"
    with pytest.raises(ValueError, match="candidate decision"):
        foundation_pareto.run_foundation_screen(
            dataset="cars",
            dataset_root=tmp_path / "data",
            model_specs_path=tmp_path / "models.json",
            cache_dir=cache_dir,
            report_path=closed_report,
            fixture_authority_path=tmp_path / "fixtures.json",
            tolerance_authority_path=tmp_path / "tolerances.json",
            published_register_path=tmp_path / "published.json",
            test_read_register_path=tmp_path / "test-reads.json",
            validation_seed=23,
            validation_fraction=0.25,
            allow_registered_test_read=True,
        )
    assert trace[:2] == ["candidate:authenticate", "candidate:fixture"]
    assert "candidate:cache" not in trace
    assert "candidate:probe" not in trace
    assert not closed_report.exists()
    assert not tuple(tmp_path.glob("official-test-read-*.json"))

    def candidate_unavailable(spec: object) -> object:
        trace.append(f"{spec.arm}:authenticate")
        if spec.arm == "candidate":
            raise OSError("gated weights unavailable")
        return fake_load(spec)

    monkeypatch.setattr(foundation_pareto, "load_foundation_encoder", candidate_unavailable)
    monkeypatch.setattr(foundation_pareto, "verify_native_fixture", fake_fixture)
    trace.clear()
    unavailable_report = tmp_path / "unavailable.json"
    with pytest.raises(ValueError, match="candidate decision"):
        foundation_pareto.run_foundation_screen(
            dataset="cars",
            dataset_root=tmp_path / "data",
            model_specs_path=tmp_path / "models.json",
            cache_dir=cache_dir,
            report_path=unavailable_report,
            fixture_authority_path=tmp_path / "fixtures.json",
            tolerance_authority_path=tmp_path / "tolerances.json",
            published_register_path=tmp_path / "published.json",
            test_read_register_path=tmp_path / "test-reads.json",
            validation_seed=23,
            validation_fraction=0.25,
            allow_registered_test_read=False,
        )
    assert not unavailable_report.exists()
    assert "candidate:fixture" not in trace

    published_records = tuple(
        PublishedMetricRecord(
            arm=arm.spec.arm,
            metric=metric,
            native_value=None,
            tolerance=None,
            source="repository",
            provenance="repository_only",
        )
        for arm in arms
        for metric in foundation_pareto.FOUNDATION_PUBLISHED_METRICS
    )
    monkeypatch.setattr(foundation_pareto, "load_foundation_encoder", fake_load)
    monkeypatch.setattr(
        foundation_pareto,
        "load_published_metric_register",
        lambda *args, **kwargs: published_records,
    )
    monkeypatch.setattr(
        foundation_pareto,
        "load_test_read_register",
        lambda *args, **kwargs: foundation_pareto.FoundationTestReadLedger(
            tuple(
                foundation_pareto.FoundationTestReadRecord(
                    arm=arm.spec.arm,
                    model_revision=foundation_pareto._test_read_identity(arm.spec)[0],
                    checkpoint_sha256=foundation_pareto._test_read_identity(arm.spec)[1],
                    metrics=foundation_pareto.FOUNDATION_PUBLISHED_METRICS,
                    purpose="confirmatory_published_metric_cross_check",
                    permitted_evaluations=1,
                )
                for arm in arms
            )
        ),
    )
    monkeypatch.setattr(
        foundation_pareto,
        "load_image_retrieval_bundle",
        lambda **kwargs: SimpleNamespace(protocol="query_gallery"),
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_evaluate_official_foundation_arm",
        lambda **kwargs: (
            {metric: 0.5 for metric in foundation_pareto.FOUNDATION_PUBLISHED_METRICS},
            (
                {
                    "arm": kwargs["arm_spec"].spec.arm,
                    "split": "official",
                    "status": "exported",
                    "path": f"cache/{kwargs['arm_spec'].spec.arm}-official.npz",
                    "rows": 2,
                    "embedding_sha256": "d" * 64,
                },
            ),
            _geometry_rows(),
        ),
    )
    trace.clear()
    official_report = tmp_path / "official.json"
    foundation_pareto.run_foundation_screen(
        dataset="cars",
        dataset_root=tmp_path / "data",
        model_specs_path=tmp_path / "models.json",
        cache_dir=cache_dir,
        report_path=official_report,
        fixture_authority_path=tmp_path / "fixtures.json",
        tolerance_authority_path=tmp_path / "tolerances.json",
        published_register_path=tmp_path / "published.json",
        test_read_register_path=tmp_path / "test-reads.json",
        validation_seed=23,
        validation_fraction=0.25,
        allow_registered_test_read=True,
    )
    official = foundation_pareto.load_foundation_screen_report(official_report)
    assert len(official["official_test_reads"]) == 2
    assert len(official["published_metric_audits"]) == 12
    assert all(
        row["decision_sha256"] == official["decision_sha256"]
        for row in official["official_test_reads"]
    )


def test_registered_official_reads_publish_receipts_before_loading_and_cross_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = (
        foundation_pareto.FoundationScreenArmSpec(
            kind="remote",
            spec=_remote_spec(arm="candidate"),
            cache_resolution=224,
            role="candidate",
        ),
        foundation_pareto.FoundationScreenArmSpec(
            kind="local",
            spec=_local_spec(),
            cache_resolution=224,
            role="comparator",
        ),
    )
    metrics = foundation_pareto.FOUNDATION_PUBLISHED_METRICS
    ledger = foundation_pareto.FoundationTestReadLedger(
        tuple(
            foundation_pareto.FoundationTestReadRecord(
                arm=arm.spec.arm,
                model_revision=foundation_pareto._test_read_identity(arm.spec)[0],
                checkpoint_sha256=foundation_pareto._test_read_identity(arm.spec)[1],
                metrics=metrics,
                purpose="confirmatory_published_metric_cross_check",
                permitted_evaluations=1,
            )
            for arm in specs
        )
    )
    records = tuple(
        PublishedMetricRecord(
            arm=arm.spec.arm,
            metric=metric,
            native_value=(0.9 if arm.spec.arm == "candidate" and metric == "recall_at_1" else None),
            tolerance=(0.0 if arm.spec.arm == "candidate" and metric == "recall_at_1" else None),
            source="repository",
            provenance=(
                "native_cross_check"
                if arm.spec.arm == "candidate" and metric == "recall_at_1"
                else "repository_only"
            ),
        )
        for arm in specs
        for metric in metrics
    )
    trace: list[str] = []
    bundle = SimpleNamespace(protocol="query_gallery")

    def fake_bundle(**kwargs: object) -> object:
        trace.append("load")
        return bundle

    def fake_evaluate(
        **kwargs: object,
    ) -> tuple[dict[str, float], tuple[dict[str, object], ...], list[dict[str, object]]]:
        arm = kwargs["arm_spec"].spec.arm
        trace.append(f"{arm}:official")
        return (
            {metric: 0.5 for metric in metrics},
            (
                {
                    "arm": arm,
                    "split": "official",
                    "status": "exported",
                    "path": f"cache/{arm}-official.npz",
                    "rows": 2,
                    "embedding_sha256": "d" * 64,
                },
            ),
            _geometry_rows(),
        )

    real_cross_check = foundation_pareto.cross_check_published_metrics

    def traced_cross_check(**kwargs: object) -> object:
        trace.append(f"{kwargs['arm']}:published")
        return real_cross_check(**kwargs)

    def fake_published_register(path: Path) -> tuple[PublishedMetricRecord, ...]:
        assert path == tmp_path / "published.json"
        trace.append("published-register")
        return records

    monkeypatch.setattr(foundation_pareto, "load_image_retrieval_bundle", fake_bundle)
    monkeypatch.setattr(
        foundation_pareto,
        "load_published_metric_register",
        fake_published_register,
    )
    monkeypatch.setattr(
        foundation_pareto,
        "_evaluate_official_foundation_arm",
        fake_evaluate,
        raising=False,
    )
    monkeypatch.setattr(
        foundation_pareto,
        "cross_check_published_metrics",
        traced_cross_check,
    )

    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    official_cache = tmp_path / "cache"
    official_cache.mkdir()
    reads, audits, cache_rows = foundation_pareto._run_registered_official_reads(
        dataset="cars",
        dataset_root=tmp_path / "data",
        validation_seed=17,
        arms=specs,
        encoders={"candidate": object(), "comparator": object()},
        cache_dir=official_cache,
        receipt_root=receipt_root,
        decision_sha256="c" * 64,
        ledger=ledger,
        published_register_path=tmp_path / "published.json",
    )

    assert trace == [
        "load",
        "candidate:official",
        "load",
        "comparator:official",
        "published-register",
        "candidate:published",
        "comparator:published",
    ]
    assert len(reads) == 2
    assert len(audits) == 12
    assert len(cache_rows) == 2
    assert all(row["decision_sha256"] == "c" * 64 for row in reads)
    assert all(Path(row["receipt_path"]).is_file() for row in reads)
    assert audits[0]["invalidates_confirmatory_claim"] is True
    assert audits[0]["passed"] is False


def test_foundation_geometry_excludes_self_for_self_retrieval_protocol() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    rows = foundation_pareto.evaluate_foundation_geometries(
        embeddings,
        labels,
        embeddings,
        labels,
        exclude_self=True,
    )

    assert all(index not in row.gallery_order[index] for row in rows for index in range(4))
    assert all(row.metrics.recall_at_1 == 1.0 for row in rows)
    assert all(row.metrics.mean_relevant_items == 1.0 for row in rows)


def test_official_split_cache_preserves_native_unnormalized_embeddings(tmp_path: Path) -> None:
    import numpy as np

    calls: list[bool] = []
    encoder = SimpleNamespace(
        encode=lambda images, *, batch_size, normalize_embeddings: (
            calls.append(normalize_embeddings)
            or np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
        )
    )
    examples = [
        ImageExample("a", np.zeros((2, 2, 3), dtype=np.uint8), 0),
        ImageExample("b", np.ones((2, 2, 3), dtype=np.uint8), 1),
    ]
    cache = tmp_path / "cache"
    cache.mkdir()
    embeddings, _ = foundation_pareto._official_split_cache(
        arm_spec=foundation_pareto.FoundationScreenArmSpec(
            kind="remote",
            spec=_remote_spec(arm="candidate", normalize=True),
            cache_resolution=224,
            role="candidate",
        ),
        encoder=encoder,
        examples=examples,
        cache_dir=cache,
        dataset="cars",
        split_name="query",
    )

    assert calls == [False]
    np.testing.assert_array_equal(embeddings, np.asarray([[3.0, 4.0], [0.0, 2.0]]))


def test_bias_free_probe_is_scale_and_mapping_order_invariant_and_auditable() -> None:
    split, embeddings = _probe_train_only_fixture()
    config = foundation_pareto.ProbeTrainingConfig(
        identities_per_batch=4,
        epochs=2,
        learning_rates=(0.001,),
        temperatures=(0.05,),
        protocol_id="f1-bias-free-512-supcon-test-v1",
    )
    first = foundation_pareto.fit_bias_free_probe_512(
        embeddings,
        split,
        arm_key="synthetic-arm",
        dataset="cars",
        split_seed=23,
        config=config,
    )
    scaled = {
        key: embeddings[key] * float(2 ** (index % 4 + 1))
        for index, key in enumerate(reversed(tuple(embeddings)))
    }
    second = foundation_pareto.fit_bias_free_probe_512(
        scaled,
        split,
        arm_key="synthetic-arm",
        dataset="cars",
        split_seed=23,
        config=config,
    )

    assert first == second
    assert first.bias is False
    assert first.input_normalized is True
    assert first.output_normalized is True
    assert first.output_dim == 512
    assert first.parameter_count == 512 * 8
    assert first.excluded_singleton_identities == 1
    assert first.dropped_tail_identities == 2
    assert first.validation_recall_at_1 == 1.0
    assert first.validation_recall_at_1_points == 100.0
    assert len(first.grid_evaluations) == 1
    assert [row.epoch for row in first.grid_evaluations[0].epochs] == [0, 1, 2]
    assert first.selected_epoch == 1
    assert first.fitted is True
    assert first.selection_evaluations == 2
    assert first.config == config
    assert hashlib.sha256(first.selected_weight_bytes).hexdigest() == first.weight_sha256
    assert not any("official" in name for name in first.__dataclass_fields__)


def test_bias_free_probe_rejects_optimization_validation_identity_overlap() -> None:
    split, embeddings = _probe_train_only_fixture()
    overlap_label = split.optimization[0].label
    query = ImageExample(example_id="overlap-query", image=None, label=overlap_label)
    gallery = ImageExample(example_id="overlap-gallery", image=None, label=overlap_label)
    embeddings[query.example_id] = embeddings[split.optimization[0].example_id]
    embeddings[gallery.example_id] = embeddings[split.optimization[1].example_id]
    overlapping = type(split)(
        optimization=split.optimization,
        query=[query],
        gallery=[gallery],
    )

    with pytest.raises(ValueError, match="identity-disjoint"):
        foundation_pareto.fit_bias_free_probe_512(
            embeddings,
            overlapping,
            arm_key="synthetic-arm",
            dataset="cars",
            split_seed=23,
            config=foundation_pareto.ProbeTrainingConfig(
                identities_per_batch=4,
                epochs=1,
                learning_rates=(0.001,),
                temperatures=(0.05,),
                protocol_id="f1-bias-free-512-supcon-test-v1",
            ),
        )


def test_bias_free_probe_learns_transferable_metric_on_held_out_identities() -> None:
    optimization: list[ImageExample] = []
    embeddings: dict[str, np.ndarray] = {}
    for label, signal in enumerate((-4.0, -2.0, 2.0, 4.0)):
        for index, nuisance in enumerate((-10.0, 10.0)):
            example = ImageExample(
                example_id=f"optimization-{label}-{index}",
                image=None,
                label=label,
            )
            optimization.append(example)
            embeddings[example.example_id] = np.asarray([signal, nuisance], dtype=np.float32)
    query = [
        ImageExample(example_id="query-100", image=None, label=100),
        ImageExample(example_id="query-101", image=None, label=101),
    ]
    gallery = [
        ImageExample(example_id="gallery-100", image=None, label=100),
        ImageExample(example_id="gallery-101", image=None, label=101),
    ]
    embeddings.update(
        {
            "query-100": np.asarray([1.0, 10.0], dtype=np.float32),
            "query-101": np.asarray([-1.0, -10.0], dtype=np.float32),
            "gallery-100": np.asarray([1.0, -10.0], dtype=np.float32),
            "gallery-101": np.asarray([-1.0, 10.0], dtype=np.float32),
        }
    )
    split_type = type(_probe_train_only_fixture()[0])

    result = foundation_pareto.fit_bias_free_probe_512(
        embeddings,
        split_type(optimization=optimization, query=query, gallery=gallery),
        arm_key="metric-learning-canary",
        dataset="cars",
        split_seed=31,
        config=foundation_pareto.ProbeTrainingConfig(
            identities_per_batch=4,
            epochs=20,
            learning_rates=(0.01,),
            temperatures=(0.05,),
            protocol_id="f1-bias-free-512-supcon-canary-v1",
        ),
    )

    epoch_recalls = [row.validation_recall_at_1 for row in result.grid_evaluations[0].epochs]
    assert epoch_recalls[0] == 0.0
    assert result.validation_recall_at_1 == 1.0
    assert result.selected_epoch > 0


def test_probe_custom_config_cannot_claim_registered_protocol_id() -> None:
    with pytest.raises(ValueError, match="custom probe config"):
        foundation_pareto.ProbeTrainingConfig(epochs=1)


def test_probe_grid_uses_one_matched_sampler_stream() -> None:
    split, embeddings = _probe_train_only_fixture()
    result = foundation_pareto.fit_bias_free_probe_512(
        embeddings,
        split,
        arm_key="matched-grid",
        dataset="cars",
        split_seed=23,
        config=foundation_pareto.ProbeTrainingConfig(
            identities_per_batch=4,
            epochs=1,
            learning_rates=(0.001, 0.003),
            temperatures=(0.05, 0.10),
            protocol_id="f1-bias-free-512-supcon-grid-test-v1",
        ),
    )

    assert len({row.sampler_seed for row in result.grid_evaluations}) == 1


def test_probe_rejects_incompatible_cublas_determinism_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split, embeddings = _probe_train_only_fixture()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        foundation_pareto.fit_bias_free_probe_512(
            embeddings,
            split,
            arm_key="environment-canary",
            dataset="cars",
            split_seed=23,
            config=foundation_pareto.ProbeTrainingConfig(
                identities_per_batch=4,
                epochs=1,
                learning_rates=(0.001,),
                temperatures=(0.05,),
                protocol_id="f1-bias-free-512-supcon-env-test-v1",
            ),
        )
