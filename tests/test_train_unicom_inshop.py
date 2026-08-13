from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from sfora.unicom_inshop import InshopRecord

SCRIPT = Path(__file__).parents[1] / "scripts/train_unicom_inshop.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_unicom_inshop", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_epoch_sampler_matches_padded_global_order() -> None:
    module = _load_script()
    sampler = module.PaddedEpochSampler(size=10, batch_size=8, seed=1024)
    sampler.set_epoch(3)

    generator = torch.Generator().manual_seed(1027)
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
    assert args.holdout_fraction == 0.2
    assert args.eval_every == 4
    assert args.checkpoint_every == 4
    assert args.max_steps is None
    assert not args.bf16
    assert not args.compile
    assert not args.fused


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
    )
    evaluations: list[int] = []

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
        evaluate=lambda epoch: evaluations.append(epoch) or {"recall_at_1": epoch / 2},
    )

    assert evaluations == [1, 2]
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
        "optimizer",
        "scheduler",
    )
    assert checkpoint["epoch"] == 2
    assert set(checkpoint["model"]) == set(raw_model.state_dict())


def test_main_fails_before_training_when_inputs_are_missing(
    tmp_path: Path, capsys
) -> None:
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
