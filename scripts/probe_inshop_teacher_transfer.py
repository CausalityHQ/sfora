#!/usr/bin/env python3
"""Fit-only teacher/student information and encoder-gradient wiring falsifier."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from probe_inshop_siglip2_base_pilot import BASE_HASHES, BASE_REVISION
from probe_inshop_spatial_parts import CHECKPOINT_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, sha256
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import member_bank_positive_ordinals
from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.representation_ceiling import fit_centered_pca
from sfora.sop_compact_training import compact_head_features
from sfora.teacher_anchored_distillation import (
    cross_dimensional_relational_distillation_loss,
    embedding_geometry_diagnostics,
)
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_training import sharded_mask_arcface_loss


def product_sham(teacher):
    """Shift complete consecutive two-image products, preserving positive pairs."""
    if len(teacher) < 6 or len(teacher) % 2:
        raise ValueError("teacher sham needs paired products")
    return teacher.reshape(-1, 2, teacher.shape[1]).roll(1, 0).reshape_as(teacher).contiguous()


def information(teacher, sham, temperature=0.20):
    mask = ~torch.eye(len(teacher), dtype=torch.bool, device=teacher.device)
    logits = (teacher @ teacher.T)[mask].reshape(len(teacher), -1) / temperature
    other = (sham @ sham.T)[mask].reshape(len(sham), -1) / temperature
    log_p, log_q = F.log_softmax(logits, 1), F.log_softmax(other, 1)
    probabilities = log_p.exp()
    return {
        "teacher_uniform_kl": float(
            (probabilities * (log_p + np.log(len(teacher) - 1))).sum(1).mean()
        ),
        "teacher_sham_kl": float((probabilities * (log_p - log_q)).sum(1).mean()),
    }


def squared_norm(values):
    return sum(
        float(value.detach().double().square().sum()) for value in values if value is not None
    )


def training_smoke(student, head, classifier, teacher, teacher_head, pixels, bank, positives):
    """Three serial eight-update arms; cached targets apply only to fixed smoke pixels."""
    initial_vision = {
        name: value.detach().cpu().clone() for name, value in student.state_dict().items()
    }
    initial_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    initial_classifier = classifier.detach().clone()
    target_started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_values = [teacher(pixel_values=value).pooler_output for value in pixels]
    teacher_codes = (
        torch.cat(
            [
                F.normalize(compact_head_features(value, teacher_head), dim=1)
                for value in teacher_values
            ]
        )
        .detach()
        .contiguous()
    )
    torch.cuda.synchronize()
    target_wall = time.perf_counter() - target_started
    generator = np.random.Generator(np.random.PCG64(179024))
    batches = [
        np.asarray(
            [
                2 * row + offset
                for row in generator.choice(128, 16, replace=False)
                for offset in (0, 1)
            ],
            dtype=np.int64,
        )
        for _ in range(8)
    ]
    all_pixels = torch.cat(pixels)
    arms, witnesses = {}, {}

    def codes_for(indexes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            source = student(pixel_values=all_pixels[indexes]).pooler_output
        return compact_head_features(source, head)

    def geometry():
        with torch.no_grad():
            codes = F.normalize(codes_for(torch.arange(32, device="cuda")), dim=1).contiguous()
        return {
            "variance": float(codes.var(0, correction=0).sum()),
            "effective_rank": embedding_geometry_diagnostics(codes).effective_rank,
        }

    for arm in ("main_only", "true_teacher", "pair_sham"):
        student.load_state_dict(initial_vision, strict=True)
        head.load_state_dict(initial_head, strict=True)
        with torch.no_grad():
            classifier.copy_(initial_classifier)
        torch.manual_seed(179024)
        student.train()
        current_bank = bank.detach().clone()
        optimizer = torch.optim.AdamW(
            [
                {"params": student.parameters(), "lr": 1e-5},
                {"params": head.parameters(), "lr": 1e-4},
                {"params": [classifier], "lr": 1e-4},
            ],
            weight_decay=0.05,
        )
        parameters = tuple(student.parameters()) + tuple(head.parameters()) + (classifier,)
        initial_geometry = geometry()
        history = []
        for batch in batches:
            indexes = torch.tensor(batch, device="cuda")
            target = indexes // 2
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            projected = codes_for(indexes)
            codes = F.normalize(projected, dim=1).contiguous()
            classification = sharded_mask_arcface_loss(
                projected,
                classifier,
                target,
                torch.arange(128, device="cuda").reshape(1, -1),
                margin=0.3,
                scale=64.0,
            )
            main = classification + 8 * smooth_ap_bank_loss(
                codes, current_bank, positives[indexes], indexes
            )
            targets = teacher_codes[indexes].contiguous()
            if arm == "pair_sham":
                targets = product_sham(targets)
            relation = cross_dimensional_relational_distillation_loss(
                codes, targets, temperatures=(0.20,)
            )
            loss = main + (0.0 if arm == "main_only" else 0.1) * relation
            if not torch.isfinite(loss):
                raise ValueError("transfer smoke nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                raise ValueError("transfer smoke nonfinite parameter")
            with torch.no_grad():
                current_bank[indexes] = codes.detach()
            torch.cuda.synchronize()
            step_wall = time.perf_counter() - step_started
            measured_geometry = geometry()
            if measured_geometry["variance"] < 0.5 * initial_geometry[
                "variance"
            ] or measured_geometry["effective_rank"] < max(
                2, 0.5 * initial_geometry["effective_rank"]
            ):
                raise ValueError("transfer smoke compact geometry collapse")
            history.append(
                {
                    "classification_loss": float(classification.detach()),
                    "relation_loss": float(relation.detach()),
                    "total_loss": float(loss.detach()),
                    "preclip_gradient_norm": float(norm),
                    "step_wall_seconds": step_wall,
                    **measured_geometry,
                }
            )
        probe_codes = F.normalize(codes_for(torch.arange(32, device="cuda")), dim=1).contiguous()
        terminal_relation = cross_dimensional_relational_distillation_loss(
            probe_codes, teacher_codes[:32].contiguous(), temperatures=(0.20,)
        )
        gradients = torch.autograd.grad(0.1 * terminal_relation, parameters, allow_unused=True)
        routes = {}
        for name, module in (
            ("first_block", student.encoder.layers[0]),
            ("last_block", student.encoder.layers[-1]),
            ("head", head),
        ):
            selected = {id(parameter) for parameter in module.parameters()}
            routes[name] = (
                squared_norm(
                    tuple(
                        gradient
                        for parameter, gradient in zip(parameters, gradients, strict=True)
                        if id(parameter) in selected
                    )
                )
                ** 0.5
            )
        if (
            any(not bool(torch.isfinite(value).all()) for value in gradients if value is not None)
            or not all(value > 0 for value in routes.values())
            or gradients[-1] is not None
        ):
            raise ValueError("terminal transfer gradient route failed")
        witnesses[arm] = {
            "encoder": student.encoder.layers[0].self_attn.q_proj.weight.detach().cpu().clone(),
            "head": head.weight.detach().cpu().clone(),
        }
        arms[arm] = {
            "stable_updates": len(history),
            "images": 8 * 32,
            "history": history,
            "initial_geometry": initial_geometry,
            "terminal_gradient_norms": routes,
            "median_step_wall_seconds": float(
                np.median([row["step_wall_seconds"] for row in history])
            ),
        }
        del optimizer, gradients
    distinct = {
        name: float((witnesses["true_teacher"][name] - witnesses["main_only"][name]).norm())
        for name in ("encoder", "head")
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        replay = teacher(pixel_values=pixels[0]).pooler_output
    replay_codes = F.normalize(compact_head_features(replay, teacher_head), dim=1).detach()
    criteria = {
        "teacher_unchanged": torch.equal(replay_codes, teacher_codes[:32]),
        "treatment_update_differs": all(value > 0 for value in distinct.values()),
        "update_overhead": arms["true_teacher"]["median_step_wall_seconds"]
        <= 3 * arms["main_only"]["median_step_wall_seconds"],
    }
    return {
        "arms": arms,
        "criteria": criteria,
        "parameter_update_difference_norms": distinct,
        "teacher_target_generation_wall_seconds": target_wall,
        "schedule_sha256": hashlib.sha256(np.asarray(batches, dtype="<i8").tobytes()).hexdigest(),
        "teacher_targets": "bounded fixed-pixel cache; augmentation requires new online targets",
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "base-snapshot", "large-snapshot", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--training-smoke", action="store_true")
    args = parser.parse_args()
    from export_inshop_siglip2_train_features import MODEL_HASHES
    from export_sop_siglip2_train import MODEL_REVISION
    from preflight_inshop_siglip2_unseen_gallery import digest_rows, split

    if args.output.exists() or not torch.cuda.is_available():
        raise ValueError("transfer device/output authority differs")
    if (
        sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("teacher or partition authority differs")
    for path, revision, hashes in (
        (args.base_snapshot, BASE_REVISION, BASE_HASHES),
        (args.large_snapshot, MODEL_REVISION, MODEL_HASHES),
    ):
        if path.name != revision or any(
            sha256(path / name) != digest for name, digest in hashes.items()
        ):
            raise ValueError("transfer snapshot authority differs")
    records = tuple(
        row for row in parse_inshop_partition(args.dataset_root) if row.split == "train"
    )
    labels = tuple(row.label for row in records)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("transfer fit authority differs")
    counts = Counter(labels[row] for row in fit)
    names = sorted(
        (name for name, count in counts.items() if count >= 2),
        key=lambda name: hashlib.sha256(b"inshop-teacher-transfer-v1\0" + name.encode()).digest(),
    )[:128]
    rows = [
        row
        for name in names
        for row in sorted(
            (row for row in fit if labels[row] == name),
            key=lambda row: hashlib.sha256(
                str(records[row].image_path.relative_to(args.dataset_root)).encode()
            ).digest(),
        )[:2]
    ]
    if len(rows) != 256:
        raise ValueError("transfer panel inventory differs")
    manifest = [
        {
            "train_row": row,
            "relative_path": str(records[row].image_path.relative_to(args.dataset_root)),
            "image_sha256": sha256(records[row].image_path),
        }
        for row in rows
    ]
    torch.set_num_threads(16)
    torch.manual_seed(179024)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    processors = [
        AutoImageProcessor.from_pretrained(path, local_files_only=True, backend="torchvision")
        for path in (args.base_snapshot, args.large_snapshot)
    ]
    student = (
        SiglipVisionModel.from_pretrained(
            args.base_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    teacher = SiglipVisionModel(
        AutoConfig.from_pretrained(args.large_snapshot, local_files_only=True).vision_config
    ).float()
    teacher_head = nn.Linear(1024, 128)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    teacher.load_state_dict(state["vision"], strict=True)
    teacher_head.load_state_dict(state["head"], strict=True)
    if any(
        not torch.equal(value, state["vision"][name])
        for name, value in teacher.state_dict().items()
    ) or any(
        not torch.equal(value, state["head"][name])
        for name, value in teacher_head.state_dict().items()
    ):
        raise ValueError("teacher FP32 load parity differs")
    del state
    teacher, teacher_head = (
        teacher.cuda().eval().requires_grad_(False),
        teacher_head.cuda().eval().requires_grad_(False),
    )
    if len(student.encoder.layers) != 12 or student.config.hidden_size != 768:
        raise ValueError("student architecture differs")
    source, pixels = [], []
    with torch.no_grad():
        for start in range(0, 256, 32):
            images = []
            for row in rows[start : start + 32]:
                with Image.open(records[row].image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = [
                processor(images=images, return_tensors="pt")["pixel_values"]
                for processor in processors
            ]
            if not torch.equal(*inputs):
                raise ValueError("teacher/student pixel parity differs")
            batch = inputs[0].cuda()
            pixels.append(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                source.append(
                    F.normalize(student(pixel_values=batch).pooler_output.float(), dim=1).cpu()
                )
    source = torch.cat(source)
    pca = fit_centered_pca(source, dimensions=128)
    head = nn.Linear(768, 128).cuda()
    with torch.no_grad():
        head.weight.copy_(pca.components)
        head.bias.copy_(-(pca.components @ pca.mean))
        initial = F.normalize(head(source.cuda()), dim=1)
    classifier = nn.Parameter(F.normalize(initial.reshape(128, 2, 128).mean(1), dim=1))
    target = torch.arange(16, device="cuda").repeat_interleave(2)
    ordinals = torch.arange(32, device="cuda")
    positives = member_bank_positive_ordinals(np.repeat(np.arange(128), 2).astype(np.int64)).cuda()
    teacher_started = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        teacher_source = teacher(pixel_values=pixels[0]).pooler_output
    teacher_codes = (
        F.normalize(compact_head_features(teacher_source, teacher_head), dim=1)
        .detach()
        .contiguous()
    )
    torch.cuda.synchronize()
    teacher_wall = time.perf_counter() - teacher_started
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = student(pixel_values=pixels[0]).pooler_output
    projected = compact_head_features(values, head)
    codes = F.normalize(projected, dim=1).contiguous()
    main_loss = sharded_mask_arcface_loss(
        projected,
        classifier,
        target,
        torch.arange(128, device="cuda").reshape(1, -1),
        margin=0.3,
        scale=64.0,
    ) + 8 * smooth_ap_bank_loss(codes, initial.detach(), positives[:32], ordinals)
    relation = cross_dimensional_relational_distillation_loss(
        codes, teacher_codes, temperatures=(0.20,)
    )
    sham = product_sham(teacher_codes)
    sham_loss = cross_dimensional_relational_distillation_loss(codes, sham, temperatures=(0.20,))
    parameters = tuple(student.parameters()) + tuple(head.parameters()) + (classifier,)
    true_grad = torch.autograd.grad(
        0.1 * relation, parameters, retain_graph=True, allow_unused=True
    )
    main_grad = torch.autograd.grad(main_loss, parameters, retain_graph=True, allow_unused=True)
    sham_grad = torch.autograd.grad(0.1 * sham_loss, parameters, allow_unused=True)
    n = len(tuple(student.parameters()))
    ratio = (squared_norm(true_grad[:n]) / squared_norm(main_grad[:n])) ** 0.5
    difference = (
        squared_norm(
            tuple(
                a - b
                for a, b in zip(true_grad[:n], sham_grad[:n], strict=True)
                if a is not None and b is not None
            )
        )
        / squared_norm(true_grad[:n])
    ) ** 0.5
    routed = {}
    for name, module in (
        ("first_block", student.encoder.layers[0]),
        ("last_block", student.encoder.layers[-1]),
        ("head", head),
    ):
        selected = {id(parameter) for parameter in module.parameters()}
        routed[name] = (
            squared_norm(
                tuple(
                    gradient
                    for parameter, gradient in zip(parameters, true_grad, strict=True)
                    if id(parameter) in selected
                )
            )
            ** 0.5
        )
    measured = information(teacher_codes, sham)
    measured.update(
        weighted_relation_to_main_encoder_gradient_ratio=ratio,
        true_sham_encoder_gradient_relative_difference=difference,
        gradient_norms=routed,
        main_loss=float(main_loss.detach()),
        relation_loss=float(relation.detach()),
        sham_loss=float(sham_loss.detach()),
        teacher_forward_wall_seconds=teacher_wall,
    )
    finite = all(
        bool(torch.isfinite(value).all())
        for gradients in (true_grad, main_grad, sham_grad)
        for value in gradients
        if value is not None
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    criteria = {
        "finite_gradients": finite,
        "student_encoder_head_route": all(value > 0 for value in routed.values()),
        "classifier_no_relation_gradient": true_grad[-1] is None,
        "teacher_no_gradient": all(
            parameter.grad is None
            for parameter in tuple(teacher.parameters()) + tuple(teacher_head.parameters())
        ),
        "weighted_gradient_ratio": 1e-4 <= ratio <= 1,
        "teacher_information": measured["teacher_uniform_kl"] > 1e-4
        and measured["teacher_sham_kl"] > 1e-4,
        "sham_gradient_differs": difference > 1e-3,
        "resource": wall <= 120 and peak < 16 * 1024**3,
    }
    receipt = {
        "schema": "sfora-inshop-teacher-transfer-gradient-v1",
        "decision": "GO_BOUNDED_SMOKE" if all(criteria.values()) else "KILL",
        "claim_eligible": False,
        "split": "official TRAIN original fit only",
        "panel": manifest,
        "fit_sha256": digest_rows(fit),
        "partition_sha256": PARTITION_SHA,
        "teacher_checkpoint_sha256": CHECKPOINT_SHA,
        "base_model_sha256": BASE_HASHES,
        "source_sha256": {
            str(Path(__file__).name): sha256(Path(__file__)),
            "teacher_anchored_distillation.py": sha256(
                Path(cross_dimensional_relational_distillation_loss.__code__.co_filename)
            ),
        },
        "hardware": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "dtype": "FP32 parameters/BF16 autocast",
        "coefficient": 0.1,
        "temperature": 0.20,
        "measured": measured,
        "criteria": criteria,
        "main_wall_seconds": wall,
        "peak_allocated_cuda_bytes": peak,
        "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(),
        "pixels_sha256": hashlib.sha256(pixels[0].cpu().numpy().tobytes()).hexdigest(),
    }
    if args.training_smoke and all(criteria.values()):
        del (
            true_grad,
            main_grad,
            sham_grad,
            main_loss,
            relation,
            sham_loss,
            projected,
            codes,
            values,
        )
        result = training_smoke(
            student, head, classifier, teacher, teacher_head, pixels, initial, positives
        )
        receipt["training_smoke"] = result
        torch.cuda.synchronize()
        receipt["main_wall_seconds"] = time.perf_counter() - started
        receipt["peak_allocated_cuda_bytes"] = torch.cuda.max_memory_allocated()
        receipt["peak_reserved_cuda_bytes"] = torch.cuda.max_memory_reserved()
        result["criteria"]["aggregate_resource"] = (
            receipt["main_wall_seconds"] <= 120
            and receipt["peak_allocated_cuda_bytes"] < 16 * 1024**3
        )
        receipt["decision"] = (
            "GO_INDEPENDENT_GATE_DESIGN" if all(result["criteria"].values()) else "KILL"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "decision",
                    "measured",
                    "criteria",
                    "main_wall_seconds",
                    "peak_allocated_cuda_bytes",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
