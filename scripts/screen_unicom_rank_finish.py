#!/usr/bin/env python3
"""Run the bounded, claim-ineligible UniCOM Smooth-AP rank-finish screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from sfora.atomic_publication import (
    publish_bytes_noreplace,
    publish_large_writer_noreplace,
)
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import (
    identity_balanced_batches,
    smooth_ap_finish_loss,
)
from sfora.unicom_training import experiment_stream_seed, identity_holdout

BASELINE = {
    "map_at_r": 0.8975116742477199,
    "recall_at_1": 0.986198243412798,
    "recall_at_10": 0.9974905897114178,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: Mapping[str, object], key: str) -> float:
    result = value.get(key)
    if type(result) is not float or not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("rank-finish metric differs")
    return result


def classify_rank_finish(
    epoch6: Mapping[str, object], epoch8: Mapping[str, object] | None
) -> dict[str, object]:
    """Apply the frozen epoch-6 abort and epoch-8 promotion gates."""

    epoch6_delta = _metric(epoch6, "map_at_r") - BASELINE["map_at_r"]
    if epoch8 is None and epoch6_delta <= -0.003:
        return {"status": "ABORT_EPOCH6", "epoch6_delta_map": epoch6_delta}
    if epoch8 is None:
        return {"status": "CONTINUE_EPOCH6", "epoch6_delta_map": epoch6_delta}
    deltas = {
        key: _metric(epoch8, key) - baseline for key, baseline in BASELINE.items()
    }
    recall_pass = (
        deltas["recall_at_1"] >= -0.001
        and deltas["recall_at_10"] >= -0.001
    )
    if not recall_pass or deltas["map_at_r"] < 0.003:
        status = "REJECT"
    elif deltas["map_at_r"] >= 0.010:
        status = "PROMOTE"
    else:
        status = "EXPLORATORY_IMPROVEMENT"
    return {
        "status": status,
        "epoch6_delta_map": epoch6_delta,
        "epoch8_deltas": deltas,
    }


def canonical_result_bytes(result: object) -> bytes:
    """Serialize one canonical claim-ineligible screen receipt."""

    if (
        type(result) is not dict
        or result.get("schema") != "unicom-rank-finish-screen-v1"
        or result.get("claim_eligible") is not False
    ):
        raise ValueError("rank-finish result authority differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def seed_rank_finish_rng(finish_seed: int) -> None:
    """Reset stochastic finish streams after restoring the common parent."""

    if type(finish_seed) is not int or finish_seed < 0:
        raise ValueError("rank-finish seed differs")
    seed = experiment_stream_seed(finish_seed, 4_000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_publication_required(finish_seed: int, *, skip: bool) -> bool:
    """Allow only the non-candidate robustness seed to omit model publication."""

    if type(finish_seed) is not int or finish_seed not in {0, 1, 2}:
        raise ValueError("rank-finish seed differs")
    if type(skip) is not bool or (skip and finish_seed != 2):
        raise ValueError("rank-finish model publication policy differs")
    return not skip


def publish_inference_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    finish_seed: int,
    source_commit: str,
    parent_checkpoint_sha256: str,
) -> dict[str, object]:
    """Publish one immutable CPU inference state and return its authority."""

    if type(finish_seed) is not int or finish_seed < 0:
        raise ValueError("rank-finish seed differs")
    if any(
        type(value) is not str
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
        for value, length in (
            (source_commit, 40),
            (parent_checkpoint_sha256, 64),
        )
    ):
        raise ValueError("rank-finish inference authority differs")
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    payload = {
        "schema": "unicom-rank-finish-inference-v1",
        "finish_seed": finish_seed,
        "source_commit": source_commit,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "model": state,
    }

    def writer(descriptor: int) -> None:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            torch.save(payload, handle)
            handle.flush()

    def validator(descriptor: int, size: int) -> None:
        if size <= 0:
            raise ValueError("rank-finish inference checkpoint is empty")
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            restored = torch.load(handle, map_location="cpu", weights_only=True)
        if (
            type(restored) is not dict
            or tuple(restored) != tuple(payload)
            or any(restored[key] != payload[key] for key in tuple(payload)[:-1])
            or type(restored["model"]) is not dict
            or tuple(restored["model"]) != tuple(state)
            or any(
                not torch.equal(restored["model"][name], tensor)
                for name, tensor in state.items()
            )
        ):
            raise ValueError("rank-finish inference checkpoint differs")

    published = publish_large_writer_noreplace(path, writer, validator=validator)
    published.close()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--official-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--partition-sha256", required=True)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--resume-checkpoint-sha256", required=True)
    parser.add_argument("--resume-run-receipt", required=True, type=Path)
    parser.add_argument("--resume-run-receipt-sha256", required=True)
    parser.add_argument("--finish-seed", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--model-output", required=True, type=Path)
    parser.add_argument("--skip-model-publication", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-rank-finish", action="store_true", required=True)
    return parser.parse_args(arguments)


def _load_trainer(repository: Path):
    path = repository / "scripts" / "train_unicom_inshop.py"
    specification = importlib.util.spec_from_file_location("rank_finish_trainer", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _require_sha256(path: Path, expected: str, name: str) -> None:
    if (
        type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != expected
    ):
        raise ValueError(f"{name} authority differs")


def run(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source_commit != args.source_commit or dirty:
        raise ValueError("rank-finish source authority differs")
    partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
    for path, expected, name in (
        (args.official_checkpoint, args.official_checkpoint_sha256, "official checkpoint"),
        (partition, args.partition_sha256, "partition"),
        (args.resume_checkpoint, args.resume_checkpoint_sha256, "resume checkpoint"),
        (args.resume_run_receipt, args.resume_run_receipt_sha256, "run receipt"),
    ):
        _require_sha256(path, expected, name)
    publish_model = model_publication_required(
        args.finish_seed, skip=args.skip_model_publication
    )
    for path in (args.output, *((args.model_output,) if publish_model else ())):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)

    trainer = _load_trainer(repository)
    receipt = json.loads(args.resume_run_receipt.read_bytes())
    trainer.validate_training_run_receipt_v2(
        receipt, evidence_root=args.resume_run_receipt.parent
    )
    protocol = receipt["training_protocol"]
    if (
        receipt["mode"] != "imprinted"
        or receipt["training_seed"] != 0
        or receipt["holdout_seed"] != 0
        or receipt["stop_after_epoch"] != 4
        or protocol["compile"] is not False
        or protocol["fused"] is not False
        or protocol["bf16"] is not False
        or protocol["epochs"] != 16
        or protocol["batch_size"] != 128
        or protocol["selected_features"] != 512
        or protocol["evaluation_features"] != 512
    ):
        raise ValueError("rank-finish parent protocol differs")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("rank-finish screen requires CUDA")
    records = parse_inshop_partition(args.dataset_root)
    training = tuple(row for row in records if row.split == "train")
    optimization, query, gallery, label_indices = identity_holdout(
        training, fraction=receipt["holdout_fraction"], seed=receipt["holdout_seed"]
    )
    labels = tuple(row.label for row in optimization)
    original_sampler = trainer.PaddedEpochSampler(
        size=len(optimization), batch_size=128, seed=receipt["training_seed"]
    )
    steps = len(original_sampler) // 128

    model, eval_transform = trainer._load_official_model(
        args.unicom_checkout, args.official_checkpoint
    )
    model = model.to(device)
    classifier = torch.nn.Parameter(
        torch.empty((len(label_indices), 768), device=device, dtype=torch.float32)
    )
    optimizer = trainer.build_optimizer(
        model,
        classifier,
        learning_rate=protocol["learning_rate"],
        classifier_learning_rate=protocol["classifier_learning_rate"],
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[protocol["learning_rate"], protocol["classifier_learning_rate"]],
        steps_per_epoch=steps,
        epochs=16,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", growth_interval=200)
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(receipt["training_seed"], 3_000)
    )
    step_ema = trainer.StepEMA(model, classifier)
    start_epoch, _ = trainer.restore_training_checkpoint(
        args.resume_checkpoint,
        raw_model=model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mask_generator=mask_generator,
        device=device,
        selection_holdout={
            "seed": receipt["holdout_seed"],
            "fraction": receipt["holdout_fraction"],
        },
        training_protocol=protocol,
        step_ema=step_ema,
    )
    if start_epoch != 4:
        raise ValueError("rank-finish resume epoch differs")
    seed_rank_finish_rng(args.finish_seed)

    dataset = trainer.InshopTrainDataset(
        optimization, label_indices, trainer.build_train_transform(336)
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    history = []
    step_ema.register_step_hook(optimizer)
    try:
        for epoch in range(4, 8):
            batches = identity_balanced_batches(
                labels,
                batch_size=128,
                images_per_identity=4,
                seed=args.finish_seed,
                epoch=epoch + 1,
                steps=steps,
            )
            generator = torch.Generator().manual_seed(
                experiment_stream_seed(args.finish_seed, 2_000 + epoch)
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batches,
                num_workers=protocol["workers"],
                pin_memory=True,
                worker_init_fn=trainer._seed_worker,
                generator=generator,
            )
            model.train()
            losses = []
            for images, batch_labels in loader:
                optimizer.zero_grad(set_to_none=True)
                embeddings = model(images.to(device)).float()
                loss = smooth_ap_finish_loss(
                    embeddings, tuple(int(value) for value in batch_labels.tolist())
                )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                losses.append(float(loss.detach()))
            completed_epoch = epoch + 1
            metrics = None
            if completed_epoch in {6, 8}:
                metrics = trainer.evaluate_holdout(
                    model,
                    query,
                    gallery,
                    eval_transform,
                    device=device,
                    batch_size=128,
                    workers=protocol["workers"],
                    evaluation_features=512,
                )
            row = {
                "epoch": completed_epoch,
                "steps": len(losses),
                "mean_loss": math.fsum(losses) / len(losses),
                "metrics": metrics,
            }
            history.append(row)
            print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
            if (
                args.finish_seed == 0
                and
                completed_epoch == 6
                and classify_rank_finish(metrics, None)["status"] == "ABORT_EPOCH6"
            ):
                break
    finally:
        step_ema.release_step_hook()
    torch.cuda.synchronize()

    epoch6 = next(row["metrics"] for row in history if row["epoch"] == 6)
    epoch8 = next(
        (row["metrics"] for row in history if row["epoch"] == 8), None
    )
    decision = classify_rank_finish(epoch6, epoch8)
    model_artifact = (
        publish_inference_checkpoint(
            args.model_output,
            model=model,
            finish_seed=args.finish_seed,
            source_commit=source_commit,
            parent_checkpoint_sha256=args.resume_checkpoint_sha256,
        )
        if publish_model
        else None
    )
    result = {
        "schema": "unicom-rank-finish-screen-v1",
        "claim_eligible": False,
        "source_commit": source_commit,
        "finish_seed": args.finish_seed,
        "script_sha256": _sha256_file(Path(__file__)),
        "rank_finish_module_sha256": _sha256_file(
            repository / "src" / "sfora" / "unicom_rank_finish.py"
        ),
        "checkpoint": {
            "path": str(args.resume_checkpoint.resolve()),
            "sha256": args.resume_checkpoint_sha256,
        },
        "run_receipt": {
            "path": str(args.resume_run_receipt.resolve()),
            "sha256": args.resume_run_receipt_sha256,
        },
        "partition_sha256": args.partition_sha256,
        "model_artifact": model_artifact,
        "method": {
            "loss": "smooth-ap-deployment-prefix-v1",
            "temperature": 0.01,
            "dimensions": 512,
            "identities_per_batch": 32,
            "images_per_identity": 4,
            "start_epoch": 4,
            "terminal_epoch": history[-1]["epoch"],
        },
        "baseline": BASELINE,
        "history": history,
        "decision": decision,
        "status": decision["status"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    payload = canonical_result_bytes(result)
    published = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda persisted: (
            None
            if persisted == payload
            else (_ for _ in ()).throw(ValueError("rank-finish persisted bytes differ"))
        ),
    )
    published.close()
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        result = run(args)
    except Exception as error:
        print(f"rank-finish screen failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
