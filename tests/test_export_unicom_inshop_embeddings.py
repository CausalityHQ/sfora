from __future__ import annotations

import importlib.util
import json
import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sfora.unicom_audit_io import load_embedding_bundle

SCRIPT = Path(__file__).parents[1] / "scripts/export_unicom_inshop_embeddings.py"
SOUP_SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_unicom_checkpoint_soup.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_unicom_inshop_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_soup_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_unicom_checkpoint_soup_for_export_test", SOUP_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(module, root: Path):
    records = []
    for split, labels in (
        ("train", ["a", "a", "b", "b"]),
        ("query", ["x", "y"]),
        ("gallery", ["x", "y", "x"]),
    ):
        for index, label in enumerate(labels):
            path = root / f"{split}-{index}.jpg"
            path.write_bytes(bytes([len(records) + 1]))
            records.append(module.InshopRecord(split=split, image_path=path, label=label))
    return tuple(records)


def _metadata(model_identifier: str = "UNICOM-ViT-B/16") -> dict[str, object]:
    return {
        "model_identifier": model_identifier,
        "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "image_list_sha256": "c" * 64,
        "transform": "official-unicom-eval-v1",
    }


def _training_protocol(module, *, seed: int = 0) -> dict[str, object]:
    return {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": "b2cfdaed33d46ec445141bb40b1a3f28aed0d3ca859101843ddf825866640bb1",
        "unicom_revision": module.UNICOM_REVISION,
        "initial_checkpoint_sha256": module.UNICOM_L14_336_SHA256,
        "partition_sha256": "p" * 64,
        "seed": seed,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 1e-5,
        "classifier_learning_rate": 1e-4,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 512,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
    }


def _selected_candidate() -> dict[str, object]:
    return {
        "name": "epochs-12_16-alpha-0.9",
        "epochs": [12, 16],
        "checkpoints": ["epoch-0012.pt", "epoch-0016.pt"],
        "alpha": 0.9,
        "metrics": {
            "recall_at_1": 0.5,
            "recall_at_10": 0.75,
            "recall_at_20": 0.8,
            "recall_at_30": 0.9,
            "map_at_r": 0.7,
        },
        "query_evidence": {
            "top1_correct": [True, False],
            "average_precision": [0.8, 0.6],
        },
    }


def test_parse_partition_preserves_official_row_order_and_labels(tmp_path: Path) -> None:
    module = _load_script()
    root = tmp_path / "dataset"
    (root / "Eval").mkdir(parents=True)
    (root / "Img" / "img").mkdir(parents=True)
    rows = [
        ("img/a.jpg", "item_2", "query"),
        ("img/b.jpg", "item_1", "train"),
        ("img/c.jpg", "item_2", "gallery"),
    ]
    for name, _, _ in rows:
        (root / "Img" / name).write_bytes(b"pixel")
    (root / "Eval" / "list_eval_partition.txt").write_text(
        "3\nimage_name item_id evaluation_status\n"
        + "\n".join(" ".join(row) for row in rows)
        + "\n"
    )

    records = module.parse_inshop_partition(root, expected_counts=None)

    assert [(row.split, row.image_path.name, row.label) for row in records] == [
        ("query", "a.jpg", "item_2"),
        ("train", "b.jpg", "item_1"),
        ("gallery", "c.jpg", "item_2"),
    ]


def test_export_embeddings_batches_in_official_order_and_roundtrips(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)
    observed: list[str] = []

    def encode_batch(paths: tuple[Path, ...]) -> np.ndarray:
        observed.extend(path.name for path in paths)
        rows = []
        for path in paths:
            value = float(path.read_bytes()[0])
            rows.append([value, 1.0, value / 2.0])
        return np.asarray(rows, dtype=np.float32)

    output = tmp_path / "bundle.npz"
    module.export_embeddings(
        records,
        encode_batch,
        _metadata(),
        output,
        batch_size=2,
        expected_counts=(4, 2, 3),
    )

    assert observed == [record.image_path.name for record in records]
    bundle = load_embedding_bundle(
        output, expected_counts=(4, 2, 3), expected_dimension=3
    )
    assert bundle.train_labels.tolist() == ["a", "a", "b", "b"]
    assert bundle.query_labels.tolist() == ["x", "y"]
    assert bundle.gallery_labels.tolist() == ["x", "y", "x"]
    assert json.loads(str(np.load(output, allow_pickle=False)["metadata_json"]))[
        "embedding_dimension"
    ] == 3
    assert "torch" not in module.__dict__


def test_export_is_no_clobber_and_preserves_existing_bytes(tmp_path: Path) -> None:
    module = _load_script()
    output = tmp_path / "bundle.npz"
    output.write_bytes(b"existing")
    records = _records(module, tmp_path)

    with pytest.raises(FileExistsError):
        module.export_embeddings(
            records,
            lambda paths: np.ones((len(paths), 3), dtype=np.float32),
            _metadata(),
            output,
            batch_size=2,
            expected_counts=(4, 2, 3),
        )
    assert output.read_bytes() == b"existing"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_export_rejects_nonfinite_or_wrong_batch_rows(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)

    with pytest.raises(ValueError, match="batch"):
        module.export_embeddings(
            records,
            lambda paths: np.full((len(paths) - 1, 3), np.nan, dtype=np.float32),
            _metadata(),
            tmp_path / "bundle.npz",
            batch_size=2,
            expected_counts=(4, 2, 3),
        )


def test_official_encoder_loads_named_architecture_from_checkpoint_directory(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-B-16.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: list[tuple[object, ...]] = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: observed.append((*args, kwargs))
        or (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(checkout, checkpoint)

    assert observed == [
        ("ViT-B/16", {"download_root": str(checkpoint.parent)})
    ]


def test_official_encoder_rejects_checkpoint_filename_upstream_would_ignore(
    tmp_path: Path,
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "renamed-unicom.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="FP16-ViT-B-16.pt"):
        module._official_encoder(tmp_path / "checkout", checkpoint)


def test_l14_336_model_spec_binds_official_name_file_and_digest() -> None:
    module = _load_script()

    spec = module._model_spec("ViT-L/14@336px")

    assert spec.model_identifier == "UNICOM-ViT-L/14@336px"
    assert spec.checkpoint_filename == "FP16-ViT-L-14-336px.pt"
    assert spec.checkpoint_sha256 == (
        "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
    )


def test_official_encoder_loads_l14_336_from_matching_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"checkpoint")
    observed: list[tuple[object, ...]] = []

    class Model:
        def cuda(self):
            return self

        def eval(self):
            return self

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: observed.append((*args, kwargs))
        or (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(checkout, checkpoint, model_name="ViT-L/14@336px")

    assert observed == [("ViT-L/14@336px", {"download_root": str(checkpoint.parent)})]


def test_official_encoder_loads_finetuned_state_before_cuda(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"initial")
    finetuned = tmp_path / "selected-model.pt"
    finetuned.write_bytes(b"finetuned")
    observed: list[tuple[object, ...]] = []

    class Model:
        def load_state_dict(self, state, *, strict):
            observed.append(("load_state_dict", state, strict))

        def cuda(self):
            observed.append(("cuda",))
            return self

        def eval(self):
            observed.append(("eval",))
            return self

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: (Model(), object()),
    )
    def load(*args, **kwargs):
        observed.append(("torch.load", *args, kwargs))
        return {
            "model": {"weight": "trained"},
            "selection": _selected_candidate(),
            "training_protocol": _training_protocol(module),
        }

    fake_torch = SimpleNamespace(load=load)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(
        checkout,
        checkpoint,
        model_name="ViT-L/14@336px",
        finetuned_state=finetuned,
        finetuned_kind="selected",
        partition_sha256="p" * 64,
        training_seed=0,
    )

    assert observed == [
        (
            "torch.load",
            finetuned,
            {"map_location": "cpu", "weights_only": True, "mmap": True},
        ),
        ("load_state_dict", {"weight": "trained"}, True),
        ("cuda",),
        ("eval",),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        OrderedDict(),
        {},
        {"model": {}},
        {"model": []},
        {"model": {"weight": "trained"}},
        {
            "model": {"weight": "trained"},
            "training_protocol": {},
        },
        {
            "model": {"weight": "trained"},
            "training_protocol": {"initial_checkpoint_sha256": "0" * 64},
        },
    ],
)
def test_official_encoder_rejects_unbound_finetuned_payload(
    tmp_path: Path, monkeypatch, payload
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"initial")
    finetuned = tmp_path / "selected-model.pt"
    finetuned.write_bytes(b"finetuned")

    class Model:
        def load_state_dict(self, state, *, strict):
            raise AssertionError("invalid state must not be loaded")

    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    with pytest.raises(ValueError, match="fine-tuned model state differs"):
        module._official_encoder(
            checkout,
            checkpoint,
            model_name="ViT-L/14@336px",
            finetuned_state=finetuned,
            finetuned_kind="selected",
            partition_sha256="p" * 64,
            training_seed=0,
        )


def test_official_encoder_accepts_only_final_endpoint_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"initial")
    finetuned = tmp_path / "epoch-0016.pt"
    finetuned.write_bytes(b"finetuned")
    loaded: list[object] = []

    class Model:
        def load_state_dict(self, state, *, strict):
            loaded.append(state)

        def cuda(self):
            return self

        def eval(self):
            return self

    protocol = _training_protocol(module)
    payload = {
        "model": {"weight": "trained"},
        "endpoint": {
            **_selected_candidate(),
            "name": "epochs-16-alpha-1",
            "epochs": [16],
            "checkpoints": ["runs/seed-0/epoch-0016.pt"],
            "alpha": 1.0,
        },
        "training_protocol": protocol,
    }
    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: (Model(), object()),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    module._official_encoder(
        checkout,
        checkpoint,
        model_name="ViT-L/14@336px",
        finetuned_state=finetuned,
        finetuned_kind="endpoint",
        partition_sha256="p" * 64,
        training_seed=0,
    )
    assert loaded == [{"weight": "trained"}]

    payload["endpoint"]["epochs"] = [4]
    with pytest.raises(ValueError, match="fine-tuned model state differs"):
        module._official_encoder(
            checkout,
            checkpoint,
            model_name="ViT-L/14@336px",
            finetuned_state=finetuned,
            finetuned_kind="endpoint",
            partition_sha256="p" * 64,
            training_seed=0,
        )


def test_soup_endpoint_payload_is_consumable_by_exporter() -> None:
    exporter = _load_script()
    soup = _load_soup_script()
    model = torch.nn.Linear(1, 1, bias=False)
    initial = OrderedDict(weight=torch.tensor([[0.0]]))
    endpoint_state = OrderedDict(weight=torch.tensor([[1.0]]))
    endpoint = soup.evaluate_grid(
        model,
        initial,
        ((Path("runs/seed-0/epoch-0016.pt"), endpoint_state),),
        alphas=(1.0,),
        evaluate=lambda: (
            {
                "recall_at_1": 0.5,
                "recall_at_10": 0.75,
                "recall_at_20": 0.8,
                "recall_at_30": 0.9,
                "map_at_r": 0.7,
            },
            {
                "top1_correct": [True, False],
                "average_precision": [0.8, 0.6],
            },
        ),
    )[0]
    payload = soup.endpoint_model_payload(
        endpoint_state, endpoint, _training_protocol(exporter)
    )

    loaded = exporter._finetuned_model_state(
        payload,
        kind="endpoint",
        model_spec=exporter._model_spec("ViT-L/14@336px"),
        partition_sha256="p" * 64,
        training_seed=0,
    )

    assert loaded is endpoint_state


def test_official_encoder_rejects_raw_training_checkpoint_as_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"initial")
    payload = {
        "epoch": 16,
        "model": {"weight": "raw-training-state"},
        "classifier": object(),
        "optimizer": {},
        "scheduler": {},
        "scaler": None,
        "mask_generator": object(),
        "torch_rng_state": object(),
        "cuda_rng_states": None,
        "selection_holdout": {"seed": 0, "fraction": 0.2},
        "training_protocol": _training_protocol(module),
        "history": [],
    }
    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: pytest.fail(
            "raw endpoint must fail before model construction"
        ),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload)
    )
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    with pytest.raises(ValueError, match="fine-tuned model state differs"):
        module._official_encoder(
            checkout,
            checkpoint,
            model_name="ViT-L/14@336px",
            finetuned_state=tmp_path / "epoch-0016.pt",
            finetuned_kind="endpoint",
            partition_sha256="p" * 64,
            training_seed=0,
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("initial_checkpoint_sha256", "0" * 64),
        ("unicom_revision", "0" * 40),
        ("partition_sha256", "0" * 64),
        ("seed", 1),
        ("learning_rate", 2e-5),
    ],
)
def test_official_encoder_rejects_training_source_drift(
    tmp_path: Path, monkeypatch, field: str, wrong_value: str
) -> None:
    module = _load_script()
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    package_file = package / "__init__.py"
    package_file.write_text("")
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"initial")
    protocol = _training_protocol(module)
    protocol[field] = wrong_value
    payload = {
        "model": {"weight": "trained"},
        "selection": _selected_candidate(),
        "training_protocol": protocol,
    }
    fake_unicom = SimpleNamespace(
        __file__=str(package_file),
        load=lambda *args, **kwargs: pytest.fail("drift must fail before model construction"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=lambda *args, **kwargs: payload))
    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=object()))
    monkeypatch.setattr(module.importlib, "import_module", lambda name: fake_unicom)

    with pytest.raises(ValueError, match="fine-tuned model state differs"):
        module._official_encoder(
            checkout,
            checkpoint,
            model_name="ViT-L/14@336px",
            finetuned_state=tmp_path / "selected-model.pt",
            finetuned_kind="selected",
            partition_sha256="p" * 64,
            training_seed=0,
        )


@pytest.mark.parametrize("metric", ["recall_at_1", "map_at_r"])
def test_finetuned_selection_rejects_metrics_inconsistent_with_query_evidence(
    metric: str,
) -> None:
    module = _load_script()
    selection = _selected_candidate()
    selection["metrics"][metric] += 0.01
    payload = {
        "model": {"weight": "trained"},
        "selection": selection,
        "training_protocol": _training_protocol(module),
    }

    with pytest.raises(ValueError, match="fine-tuned model state differs"):
        module._finetuned_model_state(
            payload,
            kind="selected",
            model_spec=module._model_spec("ViT-L/14@336px"),
            partition_sha256="p" * 64,
            training_seed=0,
        )


def test_finetuned_state_rejects_every_frozen_training_protocol_drift() -> None:
    module = _load_script()
    expected = _training_protocol(module)
    model_spec = module._model_spec("ViT-L/14@336px")
    clean_payload = {
        "model": {"weight": "trained"},
        "selection": _selected_candidate(),
        "training_protocol": expected,
    }
    assert module._finetuned_model_state(
        clean_payload,
        kind="selected",
        model_spec=model_spec,
        partition_sha256="p" * 64,
        training_seed=0,
    ) == {"weight": "trained"}

    for key in expected:
        drifted = dict(expected)
        drifted[key] = object()
        payload = dict(clean_payload, training_protocol=drifted)
        with pytest.raises(ValueError, match="fine-tuned model state differs"):
            module._finetuned_model_state(
                payload,
                kind="selected",
                model_spec=model_spec,
                partition_sha256="p" * 64,
                training_seed=0,
            )


def test_main_binds_finetuned_state_digest_into_embedding_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkout = tmp_path / "checkout"
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    finetuned = tmp_path / "selected-model.pt"
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    output = tmp_path / "bundle.npz"
    partition.parent.mkdir(parents=True)
    for path in (checkpoint, finetuned, partition):
        path.write_bytes(path.name.encode())
    observed: dict[str, object] = {}
    digests = {
        checkpoint: module.UNICOM_L14_336_SHA256,
        finetuned: "f" * 64,
        partition: "p" * 64,
    }
    monkeypatch.setattr(module, "_git_revision", lambda path: module.UNICOM_REVISION)
    monkeypatch.setattr(module, "_sha256_file", lambda path: digests[path])
    monkeypatch.setattr(module, "parse_inshop_partition", lambda path: ("records",))
    monkeypatch.setattr(
        module,
        "_official_encoder",
        lambda *args, **kwargs: observed.update(encoder_kwargs=kwargs) or object(),
    )
    monkeypatch.setattr(
        module,
        "export_embeddings",
        lambda records, encode, metadata, path, **kwargs: observed.update(
            records=records, metadata=metadata, output=path
        ),
    )

    exit_code = module.main(
        [
            "--unicom-checkout",
            str(checkout),
            "--checkpoint",
            str(checkpoint),
            "--finetuned-state",
            str(finetuned),
            "--finetuned-kind",
            "selected",
            "--training-seed",
            "0",
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--model-name",
            "ViT-L/14@336px",
        ]
    )

    assert exit_code == 0
    assert observed["encoder_kwargs"]["finetuned_state"] == finetuned
    assert observed["encoder_kwargs"]["finetuned_kind"] == "selected"
    assert observed["encoder_kwargs"]["partition_sha256"] == "p" * 64
    assert observed["encoder_kwargs"]["training_seed"] == 0
    assert observed["metadata"]["checkpoint_sha256"] == "f" * 64
    assert observed["metadata"]["transform"].endswith(
        f"initial-checkpoint-sha256={module.UNICOM_L14_336_SHA256};"
        "finetuned-state-kind=selected;"
        "training-seed=0;"
        f"finetuned-state-sha256={'f' * 64}"
    )


def test_main_preserves_frozen_official_checkpoint_metadata(tmp_path: Path, monkeypatch) -> None:
    module = _load_script()
    checkout = tmp_path / "checkout"
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    output = tmp_path / "bundle.npz"
    partition.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    partition.write_bytes(b"partition")
    observed: dict[str, object] = {}
    digests = {
        checkpoint: module.UNICOM_L14_336_SHA256,
        partition: "p" * 64,
    }
    monkeypatch.setattr(module, "_git_revision", lambda path: module.UNICOM_REVISION)
    monkeypatch.setattr(module, "_sha256_file", lambda path: digests[path])
    monkeypatch.setattr(module, "parse_inshop_partition", lambda path: ("records",))
    monkeypatch.setattr(
        module,
        "_official_encoder",
        lambda *args, **kwargs: observed.update(encoder_kwargs=kwargs) or object(),
    )
    monkeypatch.setattr(
        module,
        "export_embeddings",
        lambda records, encode, metadata, path, **kwargs: observed.update(metadata=metadata),
    )

    exit_code = module.main(
        [
            "--unicom-checkout",
            str(checkout),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset),
            "--output",
            str(output),
            "--model-name",
            "ViT-L/14@336px",
        ]
    )

    assert exit_code == 0
    assert observed["encoder_kwargs"]["finetuned_state"] is None
    assert observed["metadata"]["checkpoint_sha256"] == module.UNICOM_L14_336_SHA256
    assert observed["metadata"]["transform"] == (
        "official UNICOM ViT-L/14@336px load_model_and_transform"
    )


def test_main_rejects_wrong_official_checkpoint_digest_before_encoder(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    checkpoint.write_bytes(b"wrong")
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    partition.write_bytes(b"partition")
    monkeypatch.setattr(module, "_git_revision", lambda path: module.UNICOM_REVISION)
    monkeypatch.setattr(module, "_sha256_file", lambda path: "0" * 64)
    monkeypatch.setattr(
        module,
        "_official_encoder",
        lambda *args, **kwargs: pytest.fail("encoder must not load an unregistered checkpoint"),
    )

    assert (
        module.main(
            [
                "--unicom-checkout",
                str(tmp_path / "checkout"),
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(dataset),
                "--output",
                str(tmp_path / "bundle.npz"),
                "--model-name",
                "ViT-L/14@336px",
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--finetuned-state", "selected-model.pt", "--training-seed", "0"],
        ["--finetuned-state", "selected-model.pt", "--finetuned-kind", "selected"],
        ["--finetuned-kind", "selected", "--training-seed", "0"],
    ],
)
def test_main_requires_state_kind_and_training_seed_together(
    tmp_path: Path, monkeypatch, extra: list[str]
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    partition.write_bytes(b"partition")
    monkeypatch.setattr(module, "_git_revision", lambda path: module.UNICOM_REVISION)
    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda path: module.UNICOM_L14_336_SHA256 if path == checkpoint else "p" * 64,
    )
    monkeypatch.setattr(
        module,
        "_official_encoder",
        lambda *args, **kwargs: pytest.fail("incomplete fine-tuned args must fail before encoder"),
    )

    assert (
        module.main(
            [
                "--unicom-checkout",
                str(tmp_path / "checkout"),
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(dataset),
                "--output",
                str(tmp_path / "bundle.npz"),
                "--model-name",
                "ViT-L/14@336px",
                *extra,
            ]
        )
        == 2
    )


def test_main_rejects_finetuned_state_changed_while_loading(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    checkpoint = tmp_path / "FP16-ViT-L-14-336px.pt"
    finetuned = tmp_path / "selected-model.pt"
    dataset = tmp_path / "dataset"
    partition = dataset / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    for path in (checkpoint, finetuned, partition):
        path.write_bytes(path.name.encode())
    fine_digests = iter(("f" * 64, "e" * 64))

    def digest(path: Path) -> str:
        if path == checkpoint:
            return module.UNICOM_L14_336_SHA256
        if path == partition:
            return "p" * 64
        return next(fine_digests)

    monkeypatch.setattr(module, "_git_revision", lambda path: module.UNICOM_REVISION)
    monkeypatch.setattr(module, "_sha256_file", digest)
    monkeypatch.setattr(module, "parse_inshop_partition", lambda path: ("records",))
    monkeypatch.setattr(module, "_official_encoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module,
        "export_embeddings",
        lambda *args, **kwargs: pytest.fail("changed fine-tuned state must not export"),
    )

    assert (
        module.main(
            [
                "--unicom-checkout",
                str(tmp_path / "checkout"),
                "--checkpoint",
                str(checkpoint),
                "--finetuned-state",
                str(finetuned),
                "--finetuned-kind",
                "selected",
                "--training-seed",
                "0",
                "--dataset-root",
                str(dataset),
                "--output",
                str(tmp_path / "bundle.npz"),
                "--model-name",
                "ViT-L/14@336px",
            ]
        )
        == 2
    )


def test_export_metadata_accepts_registered_l14_336_identifier(tmp_path: Path) -> None:
    module = _load_script()
    records = _records(module, tmp_path)

    module.export_embeddings(
        records,
        lambda paths: np.ones((len(paths), 3), dtype=np.float32),
        _metadata("UNICOM-ViT-L/14@336px"),
        tmp_path / "bundle.npz",
        batch_size=2,
        expected_counts=(4, 2, 3),
    )
