#!/usr/bin/env python3
"""Run one matched arm of the UniCOM finish causal panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.unicom_finish_evidence import bind_finish_evidence
from sfora.unicom_finish_protocol import (
    FinishArm,
    build_finish_batches,
    capture_rng_state,
    restore_rng_state,
    schedule_sha256,
    validate_finish_config,
)
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import smooth_ap_finish_loss
from sfora.unicom_training import experiment_stream_seed, identity_holdout


def finish_loss(
    trainer,
    *,
    arm: FinishArm,
    embeddings: torch.Tensor,
    classifier: torch.nn.Parameter,
    labels: torch.Tensor,
    mask_generator: torch.Generator,
    margin: float,
    scale: float,
) -> torch.Tensor:
    """Compute the registered arm objective without changing optimizer topology."""

    if arm is FinishArm.SMOOTH_AP_PK:
        return smooth_ap_finish_loss(
            embeddings, tuple(int(value) for value in labels.tolist())
        )
    if arm not in {
        FinishArm.CLASSIFICATION_PADDED,
        FinishArm.CLASSIFICATION_PK,
    }:
        raise ValueError("finish ablation arm differs")
    masks = trainer.objective_masks(
        "official-eight-mask",
        dimension=embeddings.shape[1],
        selected=512,
        generator=mask_generator,
        device=embeddings.device,
    )
    return trainer.sharded_mask_arcface_loss(
        embeddings,
        classifier,
        labels,
        masks,
        margin=margin,
        scale=scale,
    )


def isolated_evaluation[Result](evaluate: Callable[[], Result]) -> Result:
    """Run evaluation without consuming any continuation RNG stream."""

    state = capture_rng_state()
    try:
        return evaluate()
    finally:
        restore_rng_state(state)


def execute_scaled_step(
    scaler: torch.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    scheduler,
) -> int:
    """Execute exactly one finite update and advance OneCycle only afterward."""

    observed = 0

    def count_step(_optimizer, _args, _kwargs) -> None:
        nonlocal observed
        observed += 1

    hook = optimizer.register_step_post_hook(count_step)
    try:
        scaler.step(optimizer)
    finally:
        hook.remove()
    scaler.update()
    if observed != 1:
        raise ValueError("finish ablation GradScaler skipped an optimizer step")
    scheduler.step()
    return observed


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", required=True, type=FinishArm, choices=tuple(FinishArm))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--unicom-checkout", required=True, type=Path)
    parser.add_argument("--official-checkpoint", required=True, type=Path)
    parser.add_argument("--official-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--partition-sha256", required=True)
    parser.add_argument("--resume-checkpoint", required=True, type=Path)
    parser.add_argument("--resume-checkpoint-sha256", required=True)
    parser.add_argument("--resume-run-receipt", required=True, type=Path)
    parser.add_argument("--resume-run-receipt-sha256", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-finish-ablation", action="store_true", required=True)
    return parser.parse_args(arguments)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_digest(digest, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode())
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))


def model_state_sha256(model: torch.nn.Module) -> str:
    """Hash an ordered finite model state without retaining a CPU clone."""

    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if (
            type(name) is not str
            or type(value) is not torch.Tensor
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finish ablation model state differs")
        _tensor_digest(digest, name, value)
    return digest.hexdigest()


def classifier_state_sha256(
    classifier: torch.nn.Parameter, optimizer: torch.optim.Optimizer
) -> str:
    """Hash the classifier tensor and its complete AdamW state."""

    digest = hashlib.sha256()
    _tensor_digest(digest, "classifier", classifier)
    state = optimizer.state.get(classifier)
    if type(state) is not dict or not state:
        raise ValueError("finish ablation classifier optimizer state differs")
    for key in sorted(state):
        value = state[key]
        if (
            type(key) is not str
            or type(value) is not torch.Tensor
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finish ablation classifier optimizer state differs")
        _tensor_digest(digest, key, value)
    return digest.hexdigest()


def _require_file(path: Path, digest: str, name: str) -> None:
    if (
        type(digest) is not str
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != digest
    ):
        raise ValueError(f"finish ablation {name} differs")


def _load_trainer(repository: Path):
    path = repository / "scripts" / "train_unicom_inshop.py"
    specification = importlib.util.spec_from_file_location("finish_parent_trainer", path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def canonical_result_bytes(result: object) -> bytes:
    """Serialize one claim-ineligible ablation result canonically."""

    if (
        type(result) is not dict
        or result.get("schema") != "unicom-finish-ablation-result-v1"
        or result.get("claim_eligible") is not False
    ):
        raise ValueError("finish ablation result differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def _validate_parent(receipt: object) -> dict[str, object]:
    if type(receipt) is not dict:
        raise ValueError("finish ablation parent differs")
    protocol = receipt.get("training_protocol")
    if (
        type(protocol) is not dict
        or receipt.get("mode") != "imprinted"
        or receipt.get("training_seed") != 0
        or receipt.get("holdout_seed") != 0
        or receipt.get("stop_after_epoch") != 4
        or protocol.get("compile") is not False
        or protocol.get("fused") is not False
        or protocol.get("bf16") is not False
        or protocol.get("epochs") != 16
        or protocol.get("batch_size") != 128
        or protocol.get("selected_features") != 512
        or protocol.get("evaluation_features") != 512
        or protocol.get("objective") != "official-eight-mask"
        or protocol.get("margin") != 0.25
        or protocol.get("scale") != 32.0
    ):
        raise ValueError("finish ablation parent protocol differs")
    return receipt


def require_parent_checkpoint_binding(
    receipt: object,
    *,
    receipt_path: Path,
    checkpoint: Path,
    expected_sha256: str,
) -> None:
    """Require the supplied checkpoint to be the receipt's sole epoch-four state."""

    if type(receipt) is not dict or type(receipt.get("checkpoints")) is not list:
        raise ValueError("finish ablation parent checkpoint binding differs")
    checkpoints = receipt["checkpoints"]
    if len(checkpoints) != 1 or type(checkpoints[0]) is not dict:
        raise ValueError("finish ablation parent checkpoint binding differs")
    binding = checkpoints[0]
    expected_path = receipt_path.parent.resolve() / "epoch-0004.pt"
    if (
        set(binding) != {"epoch", "root", "path", "sha256", "bytes"}
        or binding["epoch"] != 4
        or type(binding["epoch"]) is not int
        or binding["root"] != "current"
        or binding["path"] != "epoch-0004.pt"
        or binding["sha256"] != expected_sha256
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or checkpoint.resolve() != expected_path
        or checkpoint.stat().st_size != binding["bytes"]
    ):
        raise ValueError("finish ablation parent checkpoint binding differs")


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one matched continuation from the common epoch-four parent."""

    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
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
    if source != args.source_commit or dirty:
        raise ValueError("finish ablation source differs")
    partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
    for path, digest, name in (
        (args.config, args.config_sha256, "config"),
        (args.official_checkpoint, args.official_checkpoint_sha256, "official checkpoint"),
        (partition, args.partition_sha256, "partition"),
        (args.resume_checkpoint, args.resume_checkpoint_sha256, "parent checkpoint"),
        (args.resume_run_receipt, args.resume_run_receipt_sha256, "parent receipt"),
    ):
        _require_file(path, digest, name)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    evidence_root = args.evidence_root.resolve()
    if (
        args.evidence_root.absolute() != evidence_root
        or not evidence_root.is_dir()
        or evidence_root.is_symlink()
        or any(evidence_root.iterdir())
    ):
        raise ValueError("finish ablation evidence root differs")
    config = validate_finish_config(json.loads(args.config.read_bytes()))
    if (
        config["partition_sha256"] != args.partition_sha256
        or config["parent_checkpoint_sha256"] != args.resume_checkpoint_sha256
        or config["parent_receipt_sha256"] != args.resume_run_receipt_sha256
        or config["source_commit"] != source
        or config["official_checkpoint_sha256"] != args.official_checkpoint_sha256
    ):
        raise ValueError("finish ablation config binding differs")

    trainer = _load_trainer(repository)
    parent_receipt = json.loads(args.resume_run_receipt.read_bytes())
    trainer.validate_training_run_receipt_v2(
        parent_receipt, evidence_root=args.resume_run_receipt.parent
    )
    parent_receipt = _validate_parent(parent_receipt)
    require_parent_checkpoint_binding(
        parent_receipt,
        receipt_path=args.resume_run_receipt,
        checkpoint=args.resume_checkpoint,
        expected_sha256=args.resume_checkpoint_sha256,
    )
    protocol = parent_receipt["training_protocol"]
    if (
        protocol["partition_sha256"] != args.partition_sha256
        or protocol["initial_checkpoint_sha256"] != args.official_checkpoint_sha256
        or protocol["unicom_revision"] != config["unicom_revision"]
        or protocol["environment_sha256"] != config["environment_sha256"]
    ):
        raise ValueError("finish ablation parent authority differs")
    checkout_source = subprocess.run(
        ["git", "-C", str(args.unicom_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    checkout_dirty = subprocess.run(
        ["git", "-C", str(args.unicom_checkout), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if checkout_source != config["unicom_revision"] or checkout_dirty:
        raise ValueError("finish ablation UniCOM authority differs")
    records = parse_inshop_partition(args.dataset_root)
    training = tuple(record for record in records if record.split == "train")
    optimization, query, gallery, label_indices = identity_holdout(
        training,
        fraction=parent_receipt["holdout_fraction"],
        seed=parent_receipt["holdout_seed"],
    )
    if len(optimization) != 20_650 or len(label_indices) != 3_200:
        raise ValueError("finish ablation optimization inventory differs")
    labels = tuple(record.label for record in optimization)

    if not torch.cuda.is_available():
        raise RuntimeError("finish ablation requires CUDA")
    device = torch.device("cuda")
    if trainer.registered_runtime_environment(device) != protocol["environment"]:
        raise ValueError("finish ablation runtime environment differs")
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
        steps_per_epoch=config["steps_per_epoch"],
        epochs=16,
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", growth_interval=200)
    mask_generator = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(parent_receipt["training_seed"], 3_000)
    )
    step_ema = trainer.StepEMA(model, classifier)
    start_epoch, history = trainer.restore_training_checkpoint(
        args.resume_checkpoint,
        raw_model=model,
        classifier=classifier,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        mask_generator=mask_generator,
        device=device,
        selection_holdout={
            "seed": parent_receipt["holdout_seed"],
            "fraction": parent_receipt["holdout_fraction"],
        },
        training_protocol=protocol,
        step_ema=step_ema,
    )
    if start_epoch != 4 or not history or history[-1].get("epoch") != 4:
        raise ValueError("finish ablation resume boundary differs")
    initial_ema_updates = step_ema._updates
    initial_scheduler_step = scheduler.last_epoch
    initial_model_sha256 = model_state_sha256(model)
    initial_classifier_sha256 = classifier_state_sha256(classifier, optimizer)

    finish_seed = config["finish_seed"]
    torch.manual_seed(experiment_stream_seed(finish_seed, 4_000))
    torch.cuda.manual_seed_all(experiment_stream_seed(finish_seed, 4_000))
    dataset = trainer.InshopTrainDataset(
        optimization, label_indices, trainer.build_train_transform(336)
    )
    schedules = tuple(
        build_finish_batches(
            labels,
            arm=args.arm,
            seed=finish_seed,
            epoch=epoch,
            steps=config["steps_per_epoch"],
        )
        for epoch in config["epochs"]
    )
    combined_schedule_sha256 = schedule_sha256(
        tuple(batch for epoch_batches in schedules for batch in epoch_batches)
    )
    if combined_schedule_sha256 != config["schedule_sha256"][args.arm.value]:
        raise ValueError("finish ablation schedule binding differs")
    arm_history: list[dict[str, object]] = []
    attempted_steps = 0
    successful_steps = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    step_ema.register_step_hook(optimizer)
    try:
        for epoch, batches in zip(config["epochs"], schedules, strict=True):
            epoch_index = epoch - 1
            generator = torch.Generator().manual_seed(
                experiment_stream_seed(finish_seed, 2_000 + epoch_index)
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
            losses: list[float] = []
            for images, batch_labels in loader:
                attempted_steps += 1
                optimizer.zero_grad(set_to_none=True)
                embeddings = model(images.to(device)).float()
                batch_labels = batch_labels.to(device=device, dtype=torch.int64)
                loss = finish_loss(
                    trainer,
                    arm=args.arm,
                    embeddings=embeddings,
                    classifier=classifier,
                    labels=batch_labels,
                    mask_generator=mask_generator,
                    margin=protocol["margin"],
                    scale=protocol["scale"],
                )
                scaler.scale(loss).backward()
                if args.arm is FinishArm.SMOOTH_AP_PK and classifier.grad is not None:
                    raise ValueError("finish ablation classifier gradient differs")
                successful_steps += execute_scaled_step(scaler, optimizer, scheduler)
                losses.append(float(loss.detach()))
            if len(losses) != config["steps_per_epoch"]:
                raise ValueError("finish ablation update count differs")
            metrics = None
            if epoch in {6, 8}:
                metrics = isolated_evaluation(
                    lambda epoch=epoch: trainer.evaluate_holdout(
                        model,
                        query,
                        gallery,
                        eval_transform,
                        device=device,
                        batch_size=128,
                        workers=protocol["workers"],
                        evaluation_features=512,
                        dataset_root=args.dataset_root if epoch == 8 else None,
                        evidence_root=evidence_root if epoch == 8 else None,
                        epoch=epoch if epoch == 8 else None,
                    )
                )
            row = {
                "epoch": epoch,
                "steps": len(losses),
                "mean_loss": math.fsum(losses) / len(losses),
                "metrics": metrics,
            }
            arm_history.append(row)
            print(json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
    finally:
        step_ema.release_step_hook()
    torch.cuda.synchronize()
    final_ema_updates = step_ema._updates
    if (
        attempted_steps != 644
        or successful_steps != 644
        or final_ema_updates - initial_ema_updates != 644
        or scheduler.last_epoch - initial_scheduler_step != 644
    ):
        raise ValueError("finish ablation terminal update count differs")
    final_model_sha256 = model_state_sha256(model)
    final_classifier_sha256 = classifier_state_sha256(classifier, optimizer)
    if (
        initial_model_sha256 == final_model_sha256
        or (
            args.arm is FinishArm.SMOOTH_AP_PK
            and initial_classifier_sha256 != final_classifier_sha256
        )
        or (
            args.arm is not FinishArm.SMOOTH_AP_PK
            and initial_classifier_sha256 == final_classifier_sha256
        )
    ):
        raise ValueError("finish ablation terminal state differs")
    evidence = bind_finish_evidence(
        arm=args.arm.value,
        finish_seed=finish_seed,
        schedule_sha256=combined_schedule_sha256,
        evidence_root=evidence_root,
    )
    result = {
        "schema": "unicom-finish-ablation-result-v1",
        "claim_eligible": False,
        "source_commit": source,
        "arm": args.arm.value,
        "finish_seed": finish_seed,
        "config_sha256": args.config_sha256,
        "parent_checkpoint_sha256": args.resume_checkpoint_sha256,
        "parent_receipt_sha256": args.resume_run_receipt_sha256,
        "partition_sha256": args.partition_sha256,
        "schedule_sha256": combined_schedule_sha256,
        "history": arm_history,
        "updates": {
            "attempted": attempted_steps,
            "successful": successful_steps,
            "skipped": attempted_steps - successful_steps,
            "initial_ema": initial_ema_updates,
            "final_ema": final_ema_updates,
            "initial_scheduler_step": initial_scheduler_step,
            "final_scheduler_step": scheduler.last_epoch,
        },
        "state_sha256": {
            "initial_model": initial_model_sha256,
            "final_model": final_model_sha256,
            "initial_classifier_optimizer": initial_classifier_sha256,
            "final_classifier_optimizer": final_classifier_sha256,
        },
        "evidence": evidence,
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    payload = canonical_result_bytes(result)
    publication = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("finish ablation bytes differ"))
        ),
    )
    publication.close()
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(arguments))
    except Exception as error:
        print(f"finish ablation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"arm": result["arm"], "status": "COMPLETE"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
