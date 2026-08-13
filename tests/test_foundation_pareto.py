import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import sfora.foundation_pareto as foundation_pareto
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
    export_embeddings_v2,
    load_embeddings_v2,
    load_foundation_encoder,
    load_native_fixture_authority,
    load_published_metric_register,
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
        "embedding_head_init": "default",
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
                "embedding_head_init": "default",
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
        "embedding_head_init": "default",
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
        "embedding_head_init": "default",
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
        processor_sha256=hashlib.sha256(
            payloads["preprocessor_config.json"]
        ).hexdigest(),
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
    assert transformed == [[3.0, 4.0], [0.0, 5.0], [4.0, 3.0]]


def test_encoder_enforces_registered_normalization_and_compute_dtype() -> None:
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
    with pytest.raises(ValueError, match="normalization"):
        encoder.encode([[3.0, 4.0]], batch_size=1, normalize_embeddings=False)


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

    assert fixtures == ()
    assert tolerances == ()
    assert published == ()

    with pytest.raises(ValueError, match="authority is not frozen"):
        load_native_fixture_authority(
            root / "docs/foundation_native_fixtures.json",
            root / "docs/foundation_metric_tolerances.json",
            registered_pairs=(),
        )
    with pytest.raises(ValueError, match="authority is not frozen"):
        load_published_metric_register(
            root / "docs/foundation_published_metric_register.json"
        )


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
