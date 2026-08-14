#!/usr/bin/env python3
"""Train the pinned UNICOM ViT-L/14@336 In-Shop recipe on one device."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from sfora.unicom_inshop import InshopRecord
from sfora.unicom_retrieval_audit import retrieval_view
from sfora.unicom_training import (
    experiment_stream_seed,
    identity_holdout,
    padded_epoch_indices,
    sample_shard_masks,
    sharded_mask_arcface_loss,
)

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_L14_336_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
UNICOM_MEAN = (0.48145466, 0.4578275, 0.40821073)
UNICOM_STD = (0.26862954, 0.26130258, 0.27577711)
EMA_DECAY = 0.999


class StepEMA:
    """Same-device FP32 exponential average of trainable retrieval weights."""

    def __init__(
        self,
        backbone: torch.nn.Module,
        classifier: torch.nn.Parameter,
        *,
        decay: float = EMA_DECAY,
    ) -> None:
        if type(decay) is not float or decay != EMA_DECAY:
            raise ValueError("EMA decay differs from the frozen protocol")
        parameters = dict(backbone.named_parameters())
        if not parameters or any(
            parameter.dtype != torch.float32 for parameter in parameters.values()
        ):
            raise ValueError("EMA backbone parameters must be nonempty FP32 tensors")
        if classifier.dtype != torch.float32:
            raise ValueError("EMA classifier must be FP32")
        self._backbone = backbone
        self._classifier_source = classifier
        self._parameter_names = tuple(parameters)
        self._shadow = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        self._classifier = classifier.detach().clone()
        self._updates = 0
        self._hook = None

    @torch.no_grad()
    def update(self) -> None:
        parameters = dict(self._backbone.named_parameters())
        if tuple(parameters) != self._parameter_names:
            raise ValueError("EMA backbone parameter order differs")
        for name in self._parameter_names:
            source = parameters[name]
            shadow = self._shadow[name]
            if (
                source.dtype != torch.float32
                or source.shape != shadow.shape
                or source.device != shadow.device
            ):
                raise ValueError("EMA backbone parameter contract differs")
            shadow.mul_(EMA_DECAY).add_(source, alpha=1.0 - EMA_DECAY)
        source_classifier = self._classifier_source
        if (
            source_classifier.dtype != torch.float32
            or source_classifier.shape != self._classifier.shape
            or source_classifier.device != self._classifier.device
        ):
            raise ValueError("EMA classifier contract differs")
        self._classifier.mul_(EMA_DECAY).add_(source_classifier, alpha=1.0 - EMA_DECAY)
        self._updates += 1

    def register_step_hook(self, optimizer: torch.optim.Optimizer):
        if self._hook is not None:
            raise RuntimeError("EMA optimizer hook is already registered")
        self._hook = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: self.update()
        )
        return self._hook

    def release_step_hook(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": EMA_DECAY,
            "updates": self._updates,
            "backbone": {
                name: self._shadow[name].detach().cpu().clone() for name in self._parameter_names
            },
            "classifier": self._classifier.detach().cpu().clone(),
        }

    def load_state_dict(self, state: object) -> None:
        if type(state) is not dict or tuple(state) != (
            "decay",
            "updates",
            "backbone",
            "classifier",
        ):
            raise ValueError("EMA state schema differs")
        if type(state["decay"]) is not float or state["decay"] != EMA_DECAY:
            raise ValueError("EMA state decay differs")
        updates = state["updates"]
        if type(updates) is not int or updates < 0:
            raise TypeError("EMA state update count differs")
        backbone = state["backbone"]
        if type(backbone) is not dict or tuple(backbone) != self._parameter_names:
            raise ValueError("EMA state parameter order differs")
        for name in self._parameter_names:
            value = backbone[name]
            target = self._shadow[name]
            if (
                type(value) is not torch.Tensor
                or value.dtype != torch.float32
                or value.shape != target.shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError("EMA state parameter differs")
        classifier = state["classifier"]
        if (
            type(classifier) is not torch.Tensor
            or classifier.dtype != torch.float32
            or classifier.shape != self._classifier.shape
            or not torch.isfinite(classifier).all()
        ):
            raise ValueError("EMA state classifier differs")
        with torch.no_grad():
            for name in self._parameter_names:
                self._shadow[name].copy_(backbone[name])
            self._classifier.copy_(classifier)
        self._updates = updates

    def materialize_backbone_state(self) -> dict[str, torch.Tensor]:
        state = {
            name: value.detach().cpu().clone()
            for name, value in self._backbone.state_dict().items()
        }
        for name in self._parameter_names:
            state[name] = self._shadow[name].detach().cpu().clone()
        return state


class InshopTrainDataset(Dataset[tuple[torch.Tensor, int]]):
    """Optimization rows with a train-only contiguous identity mapping."""

    def __init__(
        self,
        records: tuple[InshopRecord, ...],
        label_indices: Mapping[str, int],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._records = records
        self._label_indices = dict(label_indices)
        self._transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        with Image.open(record.image_path) as image:
            tensor = self._transform(image.convert("RGB"))
        return tensor, self._label_indices[record.label]

    def __len__(self) -> int:
        return len(self._records)


class InshopEvalDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(
        self,
        records: tuple[InshopRecord, ...],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self._records = records
        self._transform = transform

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self._records[index]
        with Image.open(record.image_path) as image:
            tensor = self._transform(image.convert("RGB"))
        return tensor, record.label

    def __len__(self) -> int:
        return len(self._records)


class PaddedEpochSampler(Sampler[int]):
    """Single-process equivalent of UNICOM's padded global sampler."""

    def __init__(self, *, size: int, batch_size: int, seed: int) -> None:
        self._size = size
        self._batch_size = batch_size
        self._seed = seed
        self._epoch = 0
        self._indices = padded_epoch_indices(
            size=size,
            global_batch=batch_size,
            epoch=self._epoch,
            seed=seed,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._indices = padded_epoch_indices(
            size=self._size,
            global_batch=self._batch_size,
            epoch=epoch,
            seed=self._seed,
        )

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def objective_masks(
    objective: str,
    *,
    dimension: int,
    selected: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Build the registered UNICOM feature mask or its ablation controls."""

    if objective == "official-eight-mask":
        shards = 8
    elif objective == "official-one-mask":
        shards = 1
    elif objective == "prefix-512":
        if dimension <= 0 or not 0 < selected <= dimension:
            raise ValueError("mask dimensions differ")
        return torch.arange(selected, device=device, dtype=torch.int64)[None]
    else:
        raise ValueError(f"unsupported objective: {objective}")
    return sample_shard_masks(
        dimension=dimension,
        selected=selected,
        shards=shards,
        generator=generator,
        device=device,
    )


def build_train_transform(image_size: int):
    """Return the augmentation pipeline from UNICOM's retrieval trainer."""

    from timm.data import create_transform

    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=UNICOM_MEAN,
        std=UNICOM_STD,
    )


def build_optimizer(
    backbone: torch.nn.Module,
    classifier: torch.nn.Parameter,
    *,
    learning_rate: float,
    classifier_learning_rate: float,
    fused: bool,
) -> torch.optim.AdamW:
    """Build the two-rate zero-decay AdamW optimizer used by UNICOM."""

    return torch.optim.AdamW(
        [
            {"params": backbone.parameters(), "lr": learning_rate},
            {"params": [classifier], "lr": classifier_learning_rate},
        ],
        lr=learning_rate,
        weight_decay=0.0,
        fused=fused,
    )


@torch.inference_mode()
def imprinted_classifier_values(
    model: torch.nn.Module,
    records: tuple[InshopRecord, ...],
    labels: Mapping[str, int],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    """Build norm-matched class means without perturbing later training streams."""

    import numpy as np

    if (
        type(records) is not tuple
        or not records
        or type(labels) is not dict
        or not labels
        or tuple(labels.values()) != tuple(range(len(labels)))
        or any(type(label) is not str or not label for label in labels)
    ):
        raise ValueError("imprinted classifier label mapping differs")
    if type(batch_size) is not int or batch_size <= 0 or type(workers) is not int or workers < 0:
        raise ValueError("imprinted classifier loader configuration differs")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = model.training
    sums = [torch.zeros(768, dtype=torch.float64) for _label in labels]
    counts = [0 for _label in labels]
    try:
        loader = torch.utils.data.DataLoader(
            InshopEvalDataset(records, transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            generator=torch.Generator().manual_seed(experiment_stream_seed(0, 4_000)),
        )
        model.eval()
        for images, batch_labels in loader:
            embeddings = model(images.to(device)).float()
            if (
                embeddings.ndim != 2
                or embeddings.shape != (len(batch_labels), 768)
                or not torch.isfinite(embeddings).all()
            ):
                raise ValueError("imprinted classifier embeddings differ")
            norms = torch.linalg.vector_norm(embeddings, dim=1)
            if torch.any(norms == 0.0):
                raise ValueError("imprinted classifier embedding has zero norm")
            normalized = (embeddings / norms[:, None]).cpu()
            for value, label in zip(normalized, batch_labels, strict=True):
                if label not in labels:
                    raise ValueError("imprinted classifier record label differs")
                index = labels[label]
                sums[index].add_(value.double())
                counts[index] += 1
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    if any(count <= 0 for count in counts):
        raise ValueError("imprinted classifier class is empty")
    rows = []
    for total, count in zip(sums, counts, strict=True):
        mean = total / count
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm) or norm == 0.0:
            raise ValueError("imprinted classifier class mean differs")
        rows.append((mean / norm).float())
    return torch.stack(rows) * (0.01 * math.sqrt(768.0))


def initialize_classifier_values(
    *,
    labels: int,
    mode: str,
    imprinted: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Consume the official random-init stream in both factorial arms."""

    if type(labels) is not int or labels <= 0 or mode not in ("random", "imprinted"):
        raise ValueError("classifier initialization configuration differs")
    values = torch.empty(labels, 768, dtype=torch.float32)
    torch.nn.init.normal_(values, std=0.01)
    if mode == "imprinted":
        replacement = imprinted()
        if (
            type(replacement) is not torch.Tensor
            or replacement.dtype != torch.float32
            or replacement.device.type != "cpu"
            or replacement.shape != values.shape
            or not torch.isfinite(replacement).all()
        ):
            raise ValueError("imprinted classifier values differ")
        values.copy_(replacement)
    return values


def run_training_epoch(
    backbone: torch.nn.Module,
    classifier: torch.nn.Parameter,
    loader,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler,
    mask_generator: torch.Generator,
    device: torch.device,
    objective: str,
    selected_features: int,
    margin: float,
    scale: float,
    max_steps: int | None,
    bf16: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float | int]:
    """Run one train epoch with the registered single-device loss."""

    backbone.train()
    losses: list[float] = []
    for images, labels in loader:
        if max_steps is not None and len(losses) >= max_steps:
            break
        images = images.to(device)
        labels = labels.to(device=device, dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=bf16,
        ):
            embeddings = backbone(images)
        embeddings = embeddings.float()
        masks = objective_masks(
            objective,
            dimension=embeddings.shape[1],
            selected=selected_features,
            generator=mask_generator,
            device=device,
        )
        loss = sharded_mask_arcface_loss(
            embeddings,
            classifier,
            labels,
            masks,
            margin=margin,
            scale=scale,
        )
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("training loader produced no steps")
    return {"steps": len(losses), "mean_loss": math.fsum(losses) / len(losses)}


@torch.inference_mode()
def _encode_records(
    model: torch.nn.Module,
    records: tuple[InshopRecord, ...],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> tuple[object, object]:
    import numpy as np

    loader = torch.utils.data.DataLoader(
        InshopEvalDataset(records, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    chunks: list[np.ndarray] = []
    labels: list[str] = []
    model.eval()
    for images, batch_labels in loader:
        values = model(images.to(device)).float().cpu().numpy()
        chunks.append(np.ascontiguousarray(values, dtype=np.float32))
        labels.extend(batch_labels)
    return np.ascontiguousarray(np.concatenate(chunks)), np.asarray(labels)


def evaluate_holdout(
    model: torch.nn.Module,
    query: tuple[InshopRecord, ...],
    gallery: tuple[InshopRecord, ...],
    transform: Callable[[Image.Image], torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
    selected_features: int,
) -> dict[str, float]:
    """Evaluate the identity-disjoint holdout with official deployment geometry."""

    import numpy as np

    query_values, query_labels = _encode_records(
        model, query, transform, device=device, batch_size=batch_size, workers=workers
    )
    gallery_values, gallery_labels = _encode_records(
        model, gallery, transform, device=device, batch_size=batch_size, workers=workers
    )
    result = retrieval_view(
        query_values,
        gallery_values,
        query_labels,
        gallery_labels,
        coordinates=np.arange(selected_features),
        normalize_before=True,
    )
    return {
        "recall_at_1": result.recall[1],
        "recall_at_10": result.recall[10],
        "recall_at_20": result.recall[20],
        "recall_at_30": result.recall[30],
        "map_at_r": result.map_at_r,
    }


def fit_model(
    *,
    raw_model: torch.nn.Module,
    train_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    sampler,
    mask_generator: torch.Generator,
    device: torch.device,
    epochs: int,
    start_epoch: int,
    objective: str,
    selected_features: int,
    margin: float,
    scale: float,
    max_steps: int | None,
    bf16: bool,
    scaler: torch.amp.GradScaler | None,
    eval_every: int,
    checkpoint_every: int,
    output_dir: Path,
    evaluate: Callable[[int], dict[str, float]],
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    history: list[dict[str, object]] | None = None,
    step_ema: StepEMA | None = None,
) -> list[dict[str, object]]:
    """Fit and persist sparse raw-model checkpoints for later trajectory soups."""

    output_dir.mkdir(parents=True, exist_ok=True)
    history = [] if history is None else list(history)
    if step_ema is not None:
        step_ema.register_step_hook(optimizer)
    try:
        for epoch in range(start_epoch, epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            _seed_training_loader(loader, seed=int(training_protocol["seed"]), epoch=epoch)
            train_result = run_training_epoch(
                train_model,
                classifier,
                loader,
                optimizer,
                scheduler=scheduler,
                mask_generator=mask_generator,
                device=device,
                objective=objective,
                selected_features=selected_features,
                margin=margin,
                scale=scale,
                max_steps=max_steps,
                bf16=bf16,
                scaler=scaler,
            )
            completed_epoch = epoch + 1
            metrics = (
                evaluate(completed_epoch)
                if eval_every > 0 and completed_epoch % eval_every == 0
                else None
            )
            row = {"epoch": completed_epoch, "train": train_result, "metrics": metrics}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if (
                completed_epoch % checkpoint_every == 0
                or metrics is not None
                or completed_epoch == epochs
            ):
                save_training_checkpoint(
                    output_dir / f"epoch-{completed_epoch:04d}.pt",
                    epoch=completed_epoch,
                    raw_model=raw_model,
                    classifier=classifier,
                    step_ema=step_ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    mask_generator=mask_generator,
                    selection_holdout=selection_holdout,
                    training_protocol=training_protocol,
                    history=history,
                )
    finally:
        if step_ema is not None:
            step_ema.release_step_hook()
    return history


def _seed_training_loader(loader, *, seed: int, epoch: int) -> None:
    data_generator = getattr(loader, "generator", None)
    if type(data_generator) is not torch.Generator:
        raise ValueError("training loader must expose its dedicated generator")
    data_generator.manual_seed(experiment_stream_seed(seed, 2_000 + epoch))


def save_training_checkpoint(
    path: Path,
    *,
    epoch: int,
    raw_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    mask_generator: torch.Generator,
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    history: list[dict[str, object]],
    step_ema: StepEMA | None = None,
) -> None:
    """Atomically persist all mutable state needed to resume an epoch boundary."""

    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "classifier": classifier.detach(),
        "ema": None if step_ema is None else step_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "mask_generator": mask_generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "selection_holdout": selection_holdout,
        "training_protocol": training_protocol,
        "history": history,
    }
    temporary = path.with_name(f"{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"checkpoint temporary already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def restore_training_checkpoint(
    path: Path,
    *,
    raw_model: torch.nn.Module,
    classifier: torch.nn.Parameter,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler | None,
    mask_generator: torch.Generator,
    device: torch.device,
    selection_holdout: dict[str, int | float],
    training_protocol: dict[str, object],
    step_ema: StepEMA | None = None,
) -> tuple[int, list[dict[str, object]]]:
    """Restore an epoch-boundary checkpoint and return its epoch and history."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = (
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
    if type(checkpoint) is not dict or tuple(checkpoint) != expected:
        raise ValueError("training checkpoint schema differs")
    if checkpoint["selection_holdout"] != selection_holdout:
        raise ValueError("training checkpoint selection holdout differs")
    if checkpoint["training_protocol"] != training_protocol:
        raise ValueError("training checkpoint training protocol differs")
    registered_ema = training_protocol.get("ema_decay")
    if registered_ema is None:
        if checkpoint["ema"] is not None or step_ema is not None:
            raise ValueError("training checkpoint EMA state differs")
    elif registered_ema != EMA_DECAY or step_ema is None:
        raise ValueError("training checkpoint EMA protocol differs")
    else:
        step_ema.load_state_dict(checkpoint["ema"])
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint["classifier"].shape != classifier.shape:
        raise ValueError("training checkpoint classifier shape differs")
    with torch.no_grad():
        classifier.copy_(checkpoint["classifier"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is None:
        if checkpoint["scheduler"] is not None:
            raise ValueError("training checkpoint scheduler differs")
    else:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is None:
        if checkpoint["scaler"] is not None:
            raise ValueError("training checkpoint scaler differs")
    else:
        scaler.load_state_dict(checkpoint["scaler"])
    mask_generator.set_state(checkpoint["mask_generator"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    cuda_rng_states = checkpoint["cuda_rng_states"]
    if cuda_rng_states is not None:
        if not torch.cuda.is_available() or device.type != "cuda":
            raise ValueError("training checkpoint CUDA RNG state differs")
        torch.cuda.set_rng_state_all(cuda_rng_states)
    epoch = checkpoint["epoch"]
    history = checkpoint["history"]
    if type(epoch) is not int or epoch < 1 or type(history) is not list:
        raise ValueError("training checkpoint progress differs")
    return epoch, history


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_official_model(checkout: Path, checkpoint: Path):
    package_root = (checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    if Path(unicom.__file__).resolve().parent != package_root / "unicom":
        raise ValueError("imported UNICOM package does not come from the pinned checkout")
    return unicom.load("ViT-L/14@336px", download_root=str(checkpoint.parent))


def _seed_process(seed: int) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    from sfora.unicom_inshop import parse_inshop_partition

    if _git_revision(args.unicom_checkout) != UNICOM_REVISION:
        raise ValueError("UNICOM checkout revision differs")
    if args.checkpoint.name != "FP16-ViT-L-14-336px.pt":
        raise ValueError("UNICOM checkpoint filename differs")
    if _sha256_file(args.checkpoint) != UNICOM_L14_336_SHA256:
        raise ValueError("UNICOM checkpoint SHA-256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for UNICOM training")
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs/batch must be positive and workers nonnegative")
    if args.eval_every < 0 or args.checkpoint_every <= 0:
        raise ValueError("evaluation cadence must be nonnegative and checkpoint cadence positive")
    if args.holdout_fraction == 0.0 and args.eval_every != 0:
        raise ValueError("full-train mode requires --eval-every 0 to avoid test leakage")

    _seed_process(args.seed)
    device = torch.device("cuda")
    records = parse_inshop_partition(args.dataset_root)
    train_records = tuple(row for row in records if row.split == "train")
    optimization, query, gallery, labels = identity_holdout(
        train_records,
        fraction=args.holdout_fraction,
        seed=args.holdout_seed,
    )
    raw_model, eval_transform = _load_official_model(args.unicom_checkout, args.checkpoint)
    raw_model = raw_model.to(device)
    train_model = torch.compile(raw_model, mode="reduce-overhead") if args.compile else raw_model
    classifier_values = initialize_classifier_values(
        labels=len(labels),
        mode=args.classifier_init,
        imprinted=lambda: imprinted_classifier_values(
            raw_model,
            optimization,
            labels,
            eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
        ),
    )
    classifier = torch.nn.Parameter(classifier_values.to(device))
    step_ema = StepEMA(raw_model, classifier)
    optimizer = build_optimizer(
        raw_model,
        classifier,
        learning_rate=args.learning_rate,
        classifier_learning_rate=args.classifier_learning_rate,
        fused=args.fused,
    )
    sampler = PaddedEpochSampler(size=len(optimization), batch_size=args.batch_size, seed=args.seed)
    data_generator = torch.Generator().manual_seed(experiment_stream_seed(args.seed, 2_000))
    loader = torch.utils.data.DataLoader(
        InshopTrainDataset(optimization, labels, build_train_transform(336)),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=data_generator,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.learning_rate, args.classifier_learning_rate],
        steps_per_epoch=len(loader),
        epochs=args.epochs,
        pct_start=0.1,
    )
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(args.seed, 3_000)
    )
    scaler = None if args.bf16 else torch.amp.GradScaler("cuda", growth_interval=200)
    start_epoch = 0
    history: list[dict[str, object]] = []
    selection_holdout = {
        "seed": args.holdout_seed,
        "fraction": args.holdout_fraction,
    }
    training_protocol: dict[str, object] = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": _sha256_file(Path(__file__)),
        "unicom_revision": UNICOM_REVISION,
        "initial_checkpoint_sha256": UNICOM_L14_336_SHA256,
        "partition_sha256": _sha256_file(args.dataset_root / "Eval" / "list_eval_partition.txt"),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "classifier_learning_rate": args.classifier_learning_rate,
        "margin": args.margin,
        "scale": args.scale,
        "objective": args.objective,
        "selected_features": args.selected_features,
        "holdout_seed": args.holdout_seed,
        "holdout_fraction": args.holdout_fraction,
        "max_steps": args.max_steps,
        "bf16": args.bf16,
        "compile": args.compile,
        "fused": args.fused,
        "classifier_init": args.classifier_init,
        "ema_decay": EMA_DECAY,
    }
    if args.resume is not None:
        start_epoch, history = restore_training_checkpoint(
            args.resume,
            raw_model=raw_model,
            classifier=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            mask_generator=mask_generator,
            device=device,
            selection_holdout=selection_holdout,
            training_protocol=training_protocol,
            step_ema=step_ema,
        )
        if start_epoch >= args.epochs:
            raise ValueError("resume checkpoint already reached requested epochs")
    return fit_model(
        raw_model=raw_model,
        train_model=train_model,
        classifier=classifier,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        mask_generator=mask_generator,
        device=device,
        epochs=args.epochs,
        start_epoch=start_epoch,
        objective=args.objective,
        selected_features=args.selected_features,
        margin=args.margin,
        scale=args.scale,
        max_steps=args.max_steps,
        bf16=args.bf16,
        scaler=scaler,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        evaluate=lambda _epoch: evaluate_holdout(
            raw_model,
            query,
            gallery,
            eval_transform,
            device=device,
            batch_size=args.batch_size,
            workers=args.workers,
            selected_features=args.selected_features,
        ),
        history=history,
        selection_holdout=selection_holdout,
        training_protocol=training_protocol,
        step_ema=step_ema,
    )


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    import numpy as np

    np.random.seed(worker_seed)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=32.0)
    parser.add_argument(
        "--objective",
        choices=("official-eight-mask", "official-one-mask", "prefix-512"),
        default="official-eight-mask",
    )
    parser.add_argument("--selected-features", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fused", action="store_true")
    parser.add_argument("--classifier-init", choices=("random", "imprinted"), default="random")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        history = run(args)
    except Exception as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 2
    summary = args.output_dir / "history.json"
    summary.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"training complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
