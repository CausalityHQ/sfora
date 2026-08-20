from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import random
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from sfora.unicom_inshop import InshopRecord

SCRIPT = Path(__file__).parents[1] / "scripts/train_unicom_inshop.py"
INITIALIZATION_KEYS = (
    "schema_version",
    "seed",
    "classifier_init",
    "trainer_sha256",
    "algorithm",
    "classifier_tensor_sha256",
    "classifier_shape",
    "classifier_dtype",
    "optimizer_steps_per_epoch",
    "initialization_seconds",
    "post_initialization_rng",
)


def _load_script():
    spec = importlib.util.spec_from_file_location("train_unicom_inshop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _state_digest(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + b"\0" + pickle.dumps(value, protocol=5)).hexdigest()


def test_initialization_receipt_binds_exact_classifier_bytes_and_preserves_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing classifier bytes or consuming RNG must change/falsify the receipt."""
    module = _load_script()
    classifier = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cuda_states = [torch.tensor([5, 6, 7], dtype=torch.uint8)]
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "get_rng_state_all", lambda: cuda_states)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=classifier,
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.25,
        trainer_sha256="a" * 64,
    )

    assert tuple(receipt) == INITIALIZATION_KEYS
    assert receipt["classifier_tensor_sha256"] == hashlib.sha256(
        classifier.numpy().tobytes(order="C")
    ).hexdigest()
    assert receipt["classifier_shape"] == [3, 4]
    assert receipt["classifier_dtype"] == "torch.float32"
    assert receipt["post_initialization_rng"] == {
        "python_sha256": _state_digest(b"python-random-v1", python_state),
        "numpy_sha256": _state_digest(b"numpy-random-v1", numpy_state),
        "torch_cpu_sha256": hashlib.sha256(
            b"torch-cpu-random-v1\0" + bytes(torch_state.tolist())
        ).hexdigest(),
        "torch_cuda_sha256_by_device": [
            hashlib.sha256(b"torch-cuda-random-v1:0\0" + bytes(cuda_states[0].tolist())).hexdigest()
        ],
    }
    assert random.getstate() == python_state
    current_numpy = np.random.get_state()
    assert current_numpy[0] == numpy_state[0]
    assert np.array_equal(current_numpy[1], numpy_state[1])
    assert current_numpy[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    module.validate_initialization_receipt(receipt, expected_shape=[3, 4])


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("seed", True),
        ("classifier_shape", [3200, 767]),
        ("classifier_dtype", "float32"),
        ("optimizer_steps_per_epoch", 0),
        ("initialization_seconds", float("nan")),
    ),
)
def test_initialization_receipt_rejects_schema_or_scalar_drift(
    field: str, replacement: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed receipt scalars must fail before training can use their evidence."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3200, 768),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    receipt[field] = replacement
    with pytest.raises((TypeError, ValueError)):
        module.validate_initialization_receipt(receipt, expected_shape=[3200, 768])


def test_initialization_receipt_atomic_publication_reloads_and_never_clobbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A published receipt must be durable, mode-0600, strict, and immutable."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3, 4),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    output = tmp_path / "initialization-receipt.json"

    module.write_initialization_receipt_atomic(
        receipt, output, expected_shape=[3, 4]
    )

    persisted = module.strict_json_object(output.read_bytes())
    module.validate_initialization_receipt(persisted, expected_shape=[3, 4])
    assert persisted == receipt
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        module.write_initialization_receipt_atomic(
            receipt, output, expected_shape=[3, 4]
        )
    assert output.read_bytes() == original
    assert json.loads(original) == receipt


def test_initialization_receipt_atomic_publication_completes_short_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legal short write must be retried until every registered byte is durable."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    receipt = module.classifier_initialization_receipt(
        seed=2,
        classifier_init="random",
        classifier=torch.ones(3, 4),
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.0,
        trainer_sha256="a" * 64,
    )
    real_write = module.os.write

    def short_write(descriptor: int, payload: bytes | memoryview) -> int:
        limit = max(1, len(payload) // 2)
        return real_write(descriptor, payload[:limit])

    monkeypatch.setattr(module.os, "write", short_write)
    output = tmp_path / "initialization-receipt.json"

    module.write_initialization_receipt_atomic(
        receipt, output, expected_shape=[3, 4]
    )

    assert module.strict_json_object(output.read_bytes()) == receipt


def test_resume_reauthenticates_initializer_bytes_without_retiming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume must reuse duration but reject a changed deterministic initializer."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )
    classifier = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    output = tmp_path / "initialization-receipt.json"
    fresh = module.bind_initialization_receipt(
        output=output,
        resume=False,
        seed=2,
        classifier_init="random",
        classifier=classifier,
        optimizer_steps_per_epoch=161,
        initialization_seconds=1.25,
        trainer_sha256="a" * 64,
        expected_shape=[3, 4],
    )
    original = output.read_bytes()

    resumed = module.bind_initialization_receipt(
        output=output,
        resume=True,
        seed=2,
        classifier_init="random",
        classifier=classifier.clone(),
        optimizer_steps_per_epoch=161,
        initialization_seconds=None,
        trainer_sha256="a" * 64,
        expected_shape=[3, 4],
    )

    assert resumed == fresh
    assert resumed["initialization_seconds"] == 1.25
    assert output.read_bytes() == original
    changed = classifier.clone()
    changed[0, 0] += 1.0
    with pytest.raises(ValueError, match="resume initialization receipt"):
        module.bind_initialization_receipt(
            output=output,
            resume=True,
            seed=2,
            classifier_init="random",
            classifier=changed,
            optimizer_steps_per_epoch=161,
            initialization_seconds=None,
            trainer_sha256="a" * 64,
            expected_shape=[3, 4],
        )


def test_initialization_binding_rejects_wrong_registered_classifier_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production shape authority must not be inferred from a wrong live tensor."""
    module = _load_script()
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        module.torch.cuda,
        "get_rng_state_all",
        lambda: [torch.tensor([1], dtype=torch.uint8)],
    )

    with pytest.raises(ValueError, match="classifier shape"):
        module.bind_initialization_receipt(
            output=tmp_path / "initialization-receipt.json",
            resume=False,
            seed=2,
            classifier_init="random",
            classifier=torch.ones(3, 4),
            optimizer_steps_per_epoch=161,
            initialization_seconds=1.0,
            trainer_sha256="a" * 64,
            expected_shape=[3200, 768],
        )


def test_epoch_sampler_matches_padded_global_order() -> None:
    module = _load_script()
    sampler = module.PaddedEpochSampler(size=10, batch_size=8, seed=0)
    sampler.set_epoch(3)

    generator = torch.Generator().manual_seed(1_003)
    shuffled = torch.randperm(10, generator=generator).tolist()
    assert list(sampler) == (shuffled * 2)[:16]
    assert len(sampler) == 16


def test_objective_masks_bind_official_eight_shards_and_prefix_controls() -> None:
    module = _load_script()
    generator = torch.Generator().manual_seed(7)

    eight = module.objective_masks(
        "official-eight-mask",
        dimension=8,
        selected=4,
        generator=generator,
        device=torch.device("cpu"),
    )
    one = module.objective_masks(
        "official-one-mask",
        dimension=8,
        selected=4,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    prefix = module.objective_masks(
        "prefix-512",
        dimension=8,
        selected=4,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )

    assert eight.shape == (8, 4)
    assert one.shape == (1, 4)
    assert torch.equal(eight[0], one[0])
    assert torch.equal(prefix, torch.arange(4)[None])


def test_cli_defaults_match_official_336_recipe() -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--unicom-checkout",
            "/tmp/unicom",
            "--checkpoint",
            "/tmp/FP16-ViT-L-14-336px.pt",
            "--dataset-root",
            "/tmp/inshop",
            "--output-dir",
            "/tmp/output",
        ]
    )

    assert args.epochs == 128
    assert args.batch_size == 128
    assert args.learning_rate == 1e-5
    assert args.classifier_learning_rate == 1e-4
    assert args.margin == 0.25
    assert args.scale == 32.0
    assert args.objective == "official-eight-mask"
    assert args.selected_features == 512
    assert args.workers == 4
    assert args.seed == 1024
    assert args.holdout_seed == 0
    assert args.holdout_fraction == 0.2
    assert args.eval_every == 4
    assert args.checkpoint_every == 4
    assert args.max_steps is None
    assert args.resume is None
    assert not args.bf16
    assert not args.compile
    assert not args.fused
    assert args.classifier_init == "random"


def test_worker_seed_preserves_the_epoch_varying_dataloader_seed(monkeypatch) -> None:
    module = _load_script()
    python_seeds: list[int] = []
    numpy_seeds: list[int] = []
    initial_seeds = iter((2**32 + 123, 2**32 + 456))
    monkeypatch.setattr(module.torch, "initial_seed", lambda: next(initial_seeds))
    monkeypatch.setattr(
        module.torch,
        "manual_seed",
        lambda _seed: pytest.fail("worker init must not overwrite PyTorch's worker seed"),
    )
    monkeypatch.setattr(module.random, "seed", python_seeds.append)
    monkeypatch.setattr(np.random, "seed", numpy_seeds.append)

    module._seed_worker(0)
    module._seed_worker(0)

    assert python_seeds == [123, 456]
    assert numpy_seeds == [123, 456]


def test_training_loader_seed_is_epoch_derived_and_global_rng_independent() -> None:
    module = _load_script()
    loader = SimpleNamespace(generator=torch.Generator())
    global_state = torch.get_rng_state().clone()

    module._seed_training_loader(loader, seed=7, epoch=3)

    assert loader.generator.initial_seed() == module.experiment_stream_seed(7, 2_003)
    assert torch.equal(torch.get_rng_state(), global_state)


def test_train_dataset_uses_optimization_label_mapping(tmp_path: Path) -> None:
    module = _load_script()
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (3, 2), (10, 20, 30)).save(image_path)
    records = (InshopRecord(split="train", image_path=image_path, label="item_b"),)

    dataset = module.InshopTrainDataset(
        records,
        {"item_b": 7},
        lambda image: torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()),
    )

    image, label = dataset[0]
    assert image.shape == (2, 3, 3)
    assert label == 7
    assert len(dataset) == 1


def test_optimizer_binds_separate_backbone_and_classifier_rates() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(3, 2)
    classifier = torch.nn.Parameter(torch.ones(4, 2))

    optimizer = module.build_optimizer(
        backbone,
        classifier,
        learning_rate=1e-5,
        classifier_learning_rate=1e-4,
        fused=False,
    )

    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    assert optimizer.defaults["weight_decay"] == 0.0


def test_step_ema_tracks_only_parameters_on_their_live_device() -> None:
    module = _load_script()

    class BufferedLinear(torch.nn.Linear):
        def __init__(self) -> None:
            super().__init__(2, 2, bias=False)
            self.register_buffer("running", torch.tensor([3.0], dtype=torch.float32))
            self.register_buffer("counter", torch.tensor(4, dtype=torch.int64))

    backbone = BufferedLinear()
    classifier = torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    with torch.no_grad():
        backbone.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    ema = module.StepEMA(backbone, classifier)

    initial = ema.state_dict()
    assert tuple(initial) == ("decay", "updates", "backbone", "classifier")
    assert initial["decay"] == 0.999
    assert initial["updates"] == 0
    assert tuple(initial["backbone"]) == ("weight",)
    assert initial["backbone"]["weight"].device.type == "cpu"

    with torch.no_grad():
        backbone.weight.add_(10.0)
        classifier.add_(20.0)
        backbone.running.fill_(99.0)
        backbone.counter.fill_(7)
    ema.update()

    state = ema.state_dict()
    assert state["updates"] == 1
    assert torch.equal(state["backbone"]["weight"], initial["backbone"]["weight"] + 0.01)
    assert torch.equal(state["classifier"], initial["classifier"] + 0.02)
    materialized = ema.materialize_backbone_state()
    assert torch.equal(materialized["running"], torch.tensor([99.0]))
    assert torch.equal(materialized["counter"], torch.tensor(7))
    assert torch.equal(materialized["weight"], state["backbone"]["weight"])

    with torch.no_grad():
        backbone.weight.zero_()
        classifier.zero_()
    assert not torch.equal(state["backbone"]["weight"], backbone.weight)
    assert not torch.equal(state["classifier"], classifier)


def test_step_ema_optimizer_hook_runs_only_for_executed_steps() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.SGD([*backbone.parameters(), classifier], lr=0.1)
    ema = module.StepEMA(backbone, classifier)

    hook = ema.register_step_hook(optimizer)
    with pytest.raises(RuntimeError, match="already registered"):
        ema.register_step_hook(optimizer)

    optimizer.zero_grad(set_to_none=True)
    (backbone(torch.ones(1, 2)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1

    # This models GradScaler's overflow branch: it does not call optimizer.step().
    optimizer.zero_grad(set_to_none=True)
    assert ema.state_dict()["updates"] == 1

    hook.remove()
    ema.release_step_hook()
    optimizer.zero_grad(set_to_none=True)
    (backbone(torch.ones(1, 2)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1


def test_step_ema_hook_does_not_run_for_grad_scaler_overflow() -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    optimizer = torch.optim.SGD([*backbone.parameters(), classifier], lr=0.1)
    scaler = torch.amp.GradScaler("cpu")
    ema = module.StepEMA(backbone, classifier)
    ema.register_step_hook(optimizer)

    optimizer.zero_grad(set_to_none=True)
    loss = (backbone(torch.ones(1, 2)).sum() + classifier.sum()) * torch.tensor(float("inf"))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert ema.state_dict()["updates"] == 0
    ema.release_step_hook()


@pytest.mark.parametrize(
    "mutation",
    ("decay", "updates", "backbone_keys", "backbone_dtype", "classifier_shape"),
)
def test_step_ema_rejects_invalid_serialized_state(mutation: str) -> None:
    module = _load_script()
    backbone = torch.nn.Linear(2, 1, bias=False)
    classifier = torch.nn.Parameter(torch.ones(1, 1))
    ema = module.StepEMA(backbone, classifier)
    state = ema.state_dict()
    if mutation == "decay":
        state["decay"] = 0.9
    elif mutation == "updates":
        state["updates"] = True
    elif mutation == "backbone_keys":
        state["backbone"]["extra"] = torch.ones(1)
    elif mutation == "backbone_dtype":
        state["backbone"]["weight"] = state["backbone"]["weight"].double()
    else:
        state["classifier"] = torch.ones(2, 1)

    with pytest.raises((TypeError, ValueError)):
        ema.load_state_dict(state)


def test_imprinted_classifier_matches_independent_class_mean_formula_and_preserves_state(
    tmp_path: Path,
) -> None:
    module = _load_script()

    def record(name: str, label: str, color: tuple[int, int, int]) -> InshopRecord:
        path = tmp_path / name
        Image.new("RGB", (2, 2), color).save(path)
        return InshopRecord(split="train", image_path=path, label=label)

    records = (
        record("b0.png", "b", (0, 255, 0)),
        record("a0.png", "a", (255, 0, 0)),
        record("b1.png", "b", (0, 0, 255)),
        record("a1.png", "a", (255, 255, 0)),
    )

    class RepeatedMean(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.register_buffer("running", torch.tensor([5.0]))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            rgb = images.mean(dim=(2, 3)) * self.scale
            return rgb.repeat(1, 256)

    def transform(image: Image.Image) -> torch.Tensor:
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values).permute(2, 0, 1)

    model = RepeatedMean().train()
    model_before = {name: value.clone() for name, value in model.state_dict().items()}
    random.seed(19)
    np.random.seed(23)
    torch.manual_seed(29)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    values = module.imprinted_classifier_values(
        model,
        records,
        {"a": 0, "b": 1},
        transform,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
    )

    per_record = []
    for row in records:
        with Image.open(row.image_path) as image:
            embedding = model(transform(image.convert("RGB"))[None])[0]
        per_record.append(embedding / torch.linalg.vector_norm(embedding))
    expected_rows = []
    for label in ("a", "b"):
        mean = (
            torch.stack(
                [
                    value.double()
                    for row, value in zip(records, per_record, strict=True)
                    if row.label == label
                ]
            ).sum(dim=0, dtype=torch.float64)
            / 2.0
        )
        expected_rows.append((mean / torch.linalg.vector_norm(mean)).float())
    expected = torch.stack(expected_rows) * (0.01 * np.sqrt(768.0))

    assert values.dtype == torch.float32
    assert values.device.type == "cpu"
    assert torch.equal(values, expected)
    assert model.training
    assert all(torch.equal(model.state_dict()[name], value) for name, value in model_before.items())
    assert random.getstate() == python_state
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    assert np.array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


@pytest.mark.parametrize(
    "mutation",
    ("missing_class", "noncontiguous_labels", "zero_embedding", "wrong_dimension"),
)
def test_imprinted_classifier_rejects_invalid_inputs(tmp_path: Path, mutation: str) -> None:
    module = _load_script()
    path = tmp_path / "row.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(path)
    records = (InshopRecord(split="train", image_path=path, label="a"),)
    labels = {"a": 0}

    class Output(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            dimension = 767 if mutation == "wrong_dimension" else 768
            if mutation == "zero_embedding":
                return torch.zeros(images.shape[0], dimension)
            return torch.ones(images.shape[0], dimension)

    if mutation == "missing_class":
        labels = {"a": 0, "b": 1}
    elif mutation == "noncontiguous_labels":
        labels = {"a": 1}

    with pytest.raises(ValueError):
        module.imprinted_classifier_values(
            Output(),
            records,
            labels,
            lambda _image: torch.ones(3, 2, 2),
            device=torch.device("cpu"),
            batch_size=1,
            workers=0,
        )


def test_classifier_initialization_consumes_identical_rng_for_both_arms() -> None:
    module = _load_script()

    torch.manual_seed(37)
    random_values = module.initialize_classifier_values(
        labels=3,
        mode="random",
        imprinted=lambda: pytest.fail("random arm must not build imprints"),
    )
    random_next = torch.rand(4)

    torch.manual_seed(37)
    imprint = torch.full((3, 768), 0.25, dtype=torch.float32)
    imprinted_values = module.initialize_classifier_values(
        labels=3,
        mode="imprinted",
        imprinted=lambda: imprint,
    )
    imprinted_next = torch.rand(4)

    assert not torch.equal(random_values, imprinted_values)
    assert torch.equal(imprinted_values, imprint)
    assert torch.equal(random_next, imprinted_next)


def test_official_train_transform_emits_336_fp32_tensor() -> None:
    module = _load_script()
    transform = module.build_train_transform(336)
    torch.manual_seed(0)

    result = transform(Image.new("RGB", (400, 500), (10, 20, 30)))

    assert result.shape == (3, 336, 336)
    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()


def test_training_epoch_updates_backbone_with_official_mask_objective() -> None:
    module = _load_script()
    torch.manual_seed(4)
    backbone = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = module.build_optimizer(
        backbone,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )
    before = backbone.weight.detach().clone()
    scaler = torch.amp.GradScaler("cpu")

    result = module.run_training_epoch(
        backbone,
        classifier,
        loader,
        optimizer,
        scheduler=None,
        mask_generator=torch.Generator().manual_seed(11),
        device=torch.device("cpu"),
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=scaler,
    )

    assert result["steps"] == 1
    assert np.isfinite(result["mean_loss"])
    assert not torch.equal(backbone.weight, before)
    assert scaler.state_dict()["_growth_tracker"] == 1


def test_holdout_evaluation_uses_official_normalize_then_prefix_geometry(
    tmp_path: Path,
) -> None:
    module = _load_script()

    def record(name: str, split: str, label: str, color: tuple[int, int, int]):
        path = tmp_path / name
        Image.new("RGB", (2, 2), color).save(path)
        return InshopRecord(split=split, image_path=path, label=label)

    query = (
        record("qa.png", "query", "a", (255, 0, 0)),
        record("qb.png", "query", "b", (0, 255, 0)),
    )
    gallery = (
        record("ga.png", "gallery", "a", (255, 0, 0)),
        record("gb.png", "gallery", "b", (0, 255, 0)),
    )

    class MeanColor(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    def transform(image: Image.Image) -> torch.Tensor:
        values = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(values).permute(2, 0, 1)

    result = module.evaluate_holdout(
        MeanColor(),
        query,
        gallery,
        transform,
        device=torch.device("cpu"),
        batch_size=2,
        workers=0,
        selected_features=3,
    )

    assert result == {
        "recall_at_1": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_30": 1.0,
        "map_at_r": 1.0,
    }


def test_fit_writes_sparse_raw_model_checkpoint_and_metrics(tmp_path: Path) -> None:
    module = _load_script()
    torch.manual_seed(5)
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )
    evaluations: list[int] = []
    data_seeds: list[int] = []

    def evaluate(epoch: int) -> dict[str, float]:
        evaluations.append(epoch)
        data_seeds.append(loader.generator.initial_seed())
        return {"recall_at_1": epoch / 2}

    history = module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=2,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=1,
        checkpoint_every=1,
        output_dir=tmp_path,
        evaluate=evaluate,
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert evaluations == [1, 2]
    assert data_seeds == [
        module.experiment_stream_seed(0, 2_000),
        module.experiment_stream_seed(0, 2_001),
    ]
    assert [row["epoch"] for row in history] == [1, 2]
    assert [row["metrics"] for row in history] == [
        {"recall_at_1": 0.5},
        {"recall_at_1": 1.0},
    ]
    checkpoint = torch.load(tmp_path / "epoch-0002.pt", weights_only=False)
    assert tuple(checkpoint) == (
        "epoch",
        "model",
        "classifier",
        "ema",
        "optimizer",
        "scheduler",
        "scaler",
        "mask_generator",
        "torch_rng_state",
        "cuda_rng_states",
        "selection_holdout",
        "training_protocol",
        "history",
    )
    assert checkpoint["epoch"] == 2
    assert set(checkpoint["model"]) == set(raw_model.state_dict())
    assert checkpoint["history"] == history
    assert not list(tmp_path.glob("*.tmp"))


def test_ema_checkpoint_roundtrip_restores_shadow_and_update_count(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    ema = module.StepEMA(raw_model, classifier)
    with torch.no_grad():
        raw_model.weight.add_(2.0)
        classifier.add_(3.0)
    ema.update()
    expected_ema = ema.state_dict()
    protocol = {
        "seed": 0,
        "objective": "official-eight-mask",
        "ema_decay": 0.999,
        "classifier_init": "random",
    }
    path = tmp_path / "ema.pt"

    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        step_ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
        history=[],
    )
    with torch.no_grad():
        raw_model.weight.zero_()
        classifier.zero_()
    ema.update()

    epoch, history = module.restore_training_checkpoint(
        path,
        raw_model=raw_model,
        classifier=classifier,
        step_ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(2),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
    )

    assert epoch == 1
    assert history == []
    restored = ema.state_dict()
    assert restored["updates"] == expected_ema["updates"]
    assert torch.equal(restored["classifier"], expected_ema["classifier"])
    assert all(
        torch.equal(restored["backbone"][name], value)
        for name, value in expected_ema["backbone"].items()
    )


def test_fit_always_checkpoints_final_and_evaluated_epochs(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )

    module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=3,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=3,
        checkpoint_every=2,
        output_dir=tmp_path,
        evaluate=lambda epoch: {"recall_at_1": epoch / 3},
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert sorted(path.name for path in tmp_path.glob("epoch-*.pt")) == [
        "epoch-0002.pt",
        "epoch-0003.pt",
    ]


def test_fit_registers_ema_for_training_and_releases_hook_afterward(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(3, 8, bias=False)
    classifier = torch.nn.Parameter(torch.randn(4, 8))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    ema = module.StepEMA(raw_model, classifier)
    loader = DataLoader(
        TensorDataset(torch.randn(4, 3), torch.arange(4, dtype=torch.int64)),
        batch_size=4,
        generator=torch.Generator(),
    )

    module.fit_model(
        raw_model=raw_model,
        train_model=raw_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=None,
        sampler=None,
        mask_generator=torch.Generator().manual_seed(17),
        device=torch.device("cpu"),
        epochs=1,
        start_epoch=0,
        objective="official-eight-mask",
        selected_features=4,
        margin=0.25,
        scale=32.0,
        max_steps=1,
        bf16=False,
        scaler=None,
        eval_every=0,
        checkpoint_every=1,
        output_dir=tmp_path,
        evaluate=lambda _epoch: pytest.fail("evaluation is disabled"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={
            "seed": 0,
            "objective": "official-eight-mask",
            "ema_decay": 0.999,
            "classifier_init": "random",
        },
        step_ema=ema,
    )

    assert ema.state_dict()["updates"] == 1
    checkpoint = torch.load(tmp_path / "epoch-0001.pt", weights_only=False)
    assert checkpoint["ema"]["updates"] == 1
    optimizer.zero_grad(set_to_none=True)
    (raw_model(torch.ones(1, 3)).sum() + classifier.sum()).backward()
    optimizer.step()
    assert ema.state_dict()["updates"] == 1


def test_restore_checkpoint_recovers_training_state_and_history(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    mask_generator = torch.Generator().manual_seed(41)
    scaler = torch.amp.GradScaler("cpu")
    expected_model = {
        name: value.detach().clone() for name, value in raw_model.state_dict().items()
    }
    expected_classifier = classifier.detach().clone()
    expected_mask_state = mask_generator.get_state().clone()
    expected_next_random = torch.rand(3, generator=torch.Generator().manual_seed(123))
    torch.manual_seed(123)
    path = tmp_path / "resume.pt"
    module.save_training_checkpoint(
        path,
        epoch=7,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        mask_generator=mask_generator,
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
        history=[{"epoch": 7, "train": {"steps": 1, "mean_loss": 2.0}, "metrics": None}],
    )
    with torch.no_grad():
        raw_model.weight.zero_()
        classifier.zero_()
    mask_generator.manual_seed(99)
    torch.rand(19)

    epoch, history = module.restore_training_checkpoint(
        path,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        mask_generator=mask_generator,
        device=torch.device("cuda" if torch.cuda.is_available() else "meta"),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
    )

    assert epoch == 7
    assert history == [{"epoch": 7, "train": {"steps": 1, "mean_loss": 2.0}, "metrics": None}]
    for name, value in raw_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    assert torch.equal(classifier, expected_classifier)
    assert torch.equal(mask_generator.get_state(), expected_mask_state)
    assert torch.equal(torch.rand(3), expected_next_random)


def test_restore_checkpoint_rejects_training_protocol_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    path = tmp_path / "resume.pt"
    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol={"seed": 0, "objective": "official-eight-mask"},
        history=[],
    )

    with pytest.raises(ValueError, match="training protocol differs"):
        module.restore_training_checkpoint(
            path,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            mask_generator=torch.Generator().manual_seed(2),
            device=torch.device("cpu"),
            selection_holdout={"seed": 0, "fraction": 0.2},
            training_protocol={"seed": 1, "objective": "official-eight-mask"},
        )


def test_restore_checkpoint_rejects_selection_holdout_mismatch(tmp_path: Path) -> None:
    module = _load_script()
    raw_model = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    optimizer = module.build_optimizer(
        raw_model,
        classifier,
        learning_rate=1e-3,
        classifier_learning_rate=2e-3,
        fused=False,
    )
    path = tmp_path / "resume.pt"
    protocol = {"seed": 0, "objective": "official-eight-mask"}
    module.save_training_checkpoint(
        path,
        epoch=1,
        raw_model=raw_model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        mask_generator=torch.Generator().manual_seed(1),
        selection_holdout={"seed": 0, "fraction": 0.2},
        training_protocol=protocol,
        history=[],
    )

    with pytest.raises(ValueError, match="selection holdout differs"):
        module.restore_training_checkpoint(
            path,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            mask_generator=torch.Generator().manual_seed(2),
            device=torch.device("cpu"),
            selection_holdout={"seed": 1, "fraction": 0.2},
            training_protocol=protocol,
        )


def test_main_fails_before_training_when_inputs_are_missing(tmp_path: Path, capsys) -> None:
    module = _load_script()

    exit_code = module.main(
        [
            "--unicom-checkout",
            str(tmp_path / "unicom"),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--dataset-root",
            str(tmp_path / "inshop"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert "training failed:" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()
