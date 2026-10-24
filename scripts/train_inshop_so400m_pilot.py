#!/usr/bin/env python3
"""TRAIN-only In-Shop pilot: the SOP So400m 512-D frontier recipe on DeepFashion In-Shop TRAIN, crop 256 or 384.

Recipe = the SOP 384/512-D run (sfora-sop-384d512-driver-v1.sh): train_sop_so400m_384_dim512_pilot.py over the
unmodified pair trainer with --arm float_rank --member-bank --train-vision-dtype bf16 --coverage-first-schedule
--freeze-lower-stack --rank-coefficient 8 --batch-size 64 --workers 2. Every training op is a pair-module function
(ImageRows, FixedBatches, make_collate, initialize_head_and_classifier, member_bank_*, compact_head_features,
compact_training_terms, export_all, score_packed_full_gallery), called in the same order as pair.main().

Deviations from pair.main (all data plumbing, none in the optimization):
- rows/split: In-Shop TRAIN (25,882 rows) with the partition of screen_inshop_so400m_resolution_v1.py
  (products with >= 2 images split fit 0.5 seed 179019; single-image products in fit). Single-image fit
  products are NOT trained on (no positive for ArcFace batches or the member bank); they stay in the gallery.
- head PCA init / bank seed from the screen's pretrained features at the same crop (sha-pinned).
- SOP-only pins dropped: UNICOM archive, member-bank preflight/cost receipts, cutile native 128-D cross-check.
- crop 256: position embeddings interpolated in every vision forward (as the SOP 280 wrapper); 384 native.

Pre-registered decision lives in the companion doc (docs/inshop_so400m_384_pilot_2026-10-24.md).
"""
import argparse
import hashlib
import json
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as dim512  # noqa: E402  (patches pair to 512-D)
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402
from sfora.sop_compact_training import CompactTrainingArm  # noqa: E402
from sfora.unicom_inshop import parse_inshop_partition  # noqa: E402
from sfora.unicom_rank_finish import identity_balanced_batches  # noqa: E402

pair = dim512.pair
FEATURES_SHA256 = {256: "e119f6721210963fbbfcca7bf6e049d6ed80dd6e30454b5940acc68a9d95d5e1",
                   384: "b56aec2e3a2ff19860131b2ef71800f23c0a12fdb6d2fd8643eba3fda0cdf6d3"}
ORDERED_ROWS_SHA256 = "2d43ad5cee694b2f38d7e3d806c5f983b9161ede48b54bd9cc7841b31d53ee5c"
RANK_COEFFICIENT = 8.0


def inshop_split(dataset_root: Path):
    rows = [r for r in parse_inshop_partition(dataset_root) if r.split == "train"]
    ids = {label: i for i, label in enumerate(sorted({r.label for r in rows}))}
    labels = np.asarray([ids[r.label] for r in rows], dtype=np.int64)
    ordered = hashlib.sha256("\n".join(
        f"{r.image_path.relative_to(dataset_root)}\0{r.label}" for r in rows).encode()).hexdigest()
    counts = np.bincount(labels)
    multi = np.flatnonzero(counts[labels] >= 2)
    part = deterministic_class_partition(tuple(map(int, labels[multi])), fit_fraction=0.5, seed=179019)
    held = multi[np.asarray(part.validation_row_indexes, dtype=np.int64)]
    fit = np.setdiff1d(np.arange(len(rows), dtype=np.int64), held)
    train = fit[counts[labels[fit]] >= 2]
    return tuple(r.image_path for r in rows), labels, fit, held, train, ordered


@torch.inference_mode()
def official_read(vision, head, processor, dataset_root: Path, workers: int) -> dict:
    """Official In-Shop: 14,218 queries vs 12,612 gallery, packed int8, exact torch scores (R@1, mAP@R)."""
    records = parse_inshop_partition(dataset_root)
    query = [r for r in records if r.split == "query"]
    gallery = [r for r in records if r.split == "gallery"]
    ids = {label: i for i, label in enumerate(sorted({r.label for r in query + gallery}))}

    def encode(rows):
        values = pair.export_all(vision, head, tuple(r.image_path for r in rows), tuple(ids[r.label] for r in rows),
                                 processor, workers=workers, batch_size=64)
        packed = pair.pack_int8_unit_embeddings(values)
        return packed.codes.float().cuda(), packed.inverse_norms.float().cuda()

    q, qn = encode(query)
    g, gn = encode(gallery)
    ql = torch.tensor([ids[r.label] for r in query], device="cuda")
    gl = torch.tensor([ids[r.label] for r in gallery], device="cuda")
    relevant = torch.bincount(gl, minlength=len(ids))[ql]
    assert int(relevant.min()) >= 1
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device="cuda")
    r1, ap = [], []
    for start in range(0, len(q), 64):
        sl = slice(start, start + 64)
        scores = (q[sl] @ g.T) * qn[sl, None] * gn[None, :]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = gl[ranked] == ql[sl, None]
        rel = relevant[sl]
        precision = matches.cumsum(1) / ranks[None, :]
        ap.extend(((precision * matches * (ranks[None, :] <= rel[:, None])).sum(1) / rel).tolist())
        r1.extend(matches[:, 0].float().tolist())
    return {"official": True, "queries": len(query), "gallery": len(gallery),
            "recall_at_1": float(np.mean(r1)), "map_at_r": float(np.mean(ap)),
            "per_query_r1": r1, "per_query_ap": ap}


def patch_crop(crop: int) -> None:
    pair.ARM = {**pair.ARMS["so400"], "crop": crop}
    pair.MODEL_HASHES, pair.WIDTH = pair.ARM["model_hashes"], pair.ARM["width"]
    if crop != 384:
        from transformers.models.siglip import modeling_siglip

        embed = modeling_siglip.SiglipVisionEmbeddings.forward
        modeling_siglip.SiglipVisionEmbeddings.forward = (
            lambda self, pixel_values, interpolate_pos_encoding=False: embed(self, pixel_values, True))


def main() -> None:
    started_program = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--crop", type=int, choices=(256, 384), required=True)
    p.add_argument("--seed", type=int, default=179024)
    p.add_argument("--updates", type=int, default=1000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--pretrained-features", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--full-train", action="store_true",
                   help="train on ALL TRAIN products (>= 2 images), then ONE official query->gallery read")
    a = p.parse_args()
    patch_crop(a.crop)
    if (a.output_dir.exists() or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or pair.sha256(a.pretrained_features) != FEATURES_SHA256[a.crop]
            or any(pair.sha256(a.model_snapshot / n) != v for n, v in pair.MODEL_HASHES.items())):
        raise ValueError("In-Shop pilot authority differs")
    paths, labels, fit_rows, held_rows, train_rows, ordered = inshop_split(a.dataset_root)
    if ordered != ORDERED_ROWS_SHA256 or (len(fit_rows), len(held_rows)) != (12_761, 13_121):
        raise ValueError("In-Shop TRAIN rows/partition differ from the resolution screen")
    source = np.load(a.pretrained_features, mmap_mode="r")
    assert source.shape == (len(paths), pair.WIDTH)
    a.output_dir.mkdir(parents=True)
    if a.full_train:
        counts = np.bincount(labels)
        train_rows = np.flatnonzero(counts[labels] >= 2)

    train_labels = tuple(map(int, labels[train_rows]))
    names = tuple(sorted(set(train_labels)))
    class_index = {name: i for i, name in enumerate(names)}
    class_ids = np.asarray([class_index[x] for x in train_labels], dtype=np.int64)
    schedule = identity_balanced_batches(tuple(map(str, train_labels)), batch_size=64,
                                         images_per_identity=pair.IMAGES_PER_IDENTITY, seed=a.seed,
                                         epoch=1, steps=a.updates, coverage_first=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    train_features = torch.from_numpy(np.asarray(source[train_rows]).copy()).float()
    head, classifier, pca_sha = pair.initialize_head_and_classifier(train_features, train_labels)
    bank = pair.member_bank_initial_values(train_features, head, live_head=False)

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(a.model_snapshot, local_files_only=True, backend="torchvision")
    processor.size = {"height": a.crop, "width": a.crop}
    full = AutoModel.from_pretrained(a.model_snapshot, local_files_only=True, use_safetensors=True,
                                     dtype=torch.float16)
    vision = full.vision_model.float().cuda().train()
    del full
    vision.embeddings.requires_grad_(False)
    assert len(vision.encoder.layers) == pair.ARM["blocks"]
    for block in vision.encoder.layers[: pair.ARM["frozen_blocks"]]:
        block.requires_grad_(False)
    head = head.cuda().train()
    classifier = nn.Parameter(classifier.cuda())
    bank = bank.cuda()
    positive_table = pair.member_bank_positive_ordinals(class_ids).cuda()
    schedule_ordinals = torch.tensor(schedule, dtype=torch.long, device="cuda")
    class_ids_gpu = torch.from_numpy(class_ids).cuda()
    loader = DataLoader(
        pair.ImageRows(tuple(paths[r] for r in train_rows), tuple(class_ids.tolist()), augment=True),
        batch_sampler=pair.FixedBatches(schedule), num_workers=a.workers, pin_memory=True,
        generator=torch.Generator().manual_seed(a.seed), collate_fn=pair.make_collate(processor))
    optimizer = torch.optim.AdamW(
        [{"params": vision.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-4},
         {"params": [classifier], "lr": 1e-4}], weight_decay=0.05)
    dtype, scaler = pair.training_precision("bf16", device="cuda")
    masks = torch.arange(pair.OUTPUT_WIDTH, device="cuda", dtype=torch.int64).unsqueeze(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses = []
    for step, (batch, target) in enumerate(loader, start=1):
        tensors = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):
            source_features = vision(**tensors).pooler_output
        features = pair.compact_head_features(source_features, head)
        control, _ = pair.compact_training_terms(features, classifier, target, masks,
                                                 arm=CompactTrainingArm.ARCFACE, arcface_margin=0.3,
                                                 arcface_scale=64.0)
        ordinals = schedule_ordinals[step - 1]
        if not torch.equal(target, class_ids_gpu[ordinals]):
            raise ValueError("In-Shop member-bank schedule labels differ")
        rank = pair.member_bank_rank_loss(features, bank, head, positive_table[ordinals], ordinals,
                                          live_head=False)
        loss = control + RANK_COEFFICIENT * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError("In-Shop train loss is nonfinite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(vision.parameters()) + list(head.parameters()) + [classifier],
                                       1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        rows, positions = pair.member_bank_refresh_rows(schedule[step - 1])
        bank[torch.tensor(rows, dtype=torch.long, device="cuda")] = pair.member_bank_refresh_values(
            source_features, features, torch.tensor(positions, dtype=torch.long, device="cuda"), live_head=False)
        losses.append((float(loss.detach()), float(control.detach()), float(rank.detach())))
        if step % 50 == 0 or step == a.updates:
            print(json.dumps({"step": step, "loss": losses[-1], "elapsed_seconds": time.perf_counter() - started}),
                  flush=True)
    training_wall = time.perf_counter() - started
    training_peak = torch.cuda.max_memory_allocated()
    torch.save({"vision": {k: v.detach().cpu() for k, v in vision.state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "classifier": classifier.detach().cpu(), "seed": a.seed, "updates": a.updates, "crop": a.crop},
               a.output_dir / "checkpoint.pt")
    export_started = time.perf_counter()
    values = pair.export_all(vision, head, paths, tuple(map(int, labels)), processor, workers=a.workers,
                             batch_size=64)
    export_seconds = time.perf_counter() - export_started
    np.save(a.output_dir / "train_embeddings.npy", values.numpy(), allow_pickle=False)
    packed = pair.pack_int8_unit_embeddings(values)
    if a.full_train:  # held rows were trained on: the TRAIN-held score is meaningless, read official once
        quality = official_read(vision, head, processor, a.dataset_root, a.workers)
    else:
        quality = pair.score_packed_full_gallery(packed.codes.float(), packed.inverse_norms,
                                                 torch.from_numpy(labels.copy()),
                                                 torch.from_numpy(held_rows.copy()), device=torch.device("cuda"))
    quality["output_width"] = pair.OUTPUT_WIDTH
    quality["gallery_wire_bytes_per_row"] = len(packed.to_bytes()) // len(packed.codes)
    receipt = {
        "schema": "sfora-inshop-so400m-pilot-v1", "claim_eligible": False,
        "split": ("In-Shop full TRAIN; ONE official query->gallery read" if a.full_train else
                  "In-Shop official TRAIN product-disjoint fit/held (screen partition); no query/gallery rows"),
        "full_train": a.full_train,
        "seed": a.seed, "updates": a.updates, "crop": a.crop,
        "substrate_config": {k: v for k, v in pair.ARM.items() if k != "model_hashes"},
        "fit_images": len(fit_rows), "train_images": len(train_rows), "train_products": len(names),
        "holdout_queries": len(held_rows), "gallery_images": len(labels),
        "ordered_rows_sha256": ordered, "pretrained_features_sha256": FEATURES_SHA256[a.crop],
        "source_pca_sha256": pca_sha,
        "schedule_sha256": hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest(),
        "wrapper_sha256": pair.sha256(Path(__file__)), "pair_sha256": pair.sha256(Path(pair.__file__)),
        "dim512_sha256": pair.sha256(Path(dim512.__file__)),
        "training_objective": "ArcFace 0.3/64 + 8.0 x full-fit float SmoothAP member bank (512-D)",
        "optimizer": "AdamW wd 0.05; vision 1e-5; head/classifier 1e-4; clip 1.0; bf16 autocast",
        "frozen": f"embeddings + first {pair.ARM['frozen_blocks']} of {pair.ARM['blocks']} blocks",
        "training_wall_seconds": training_wall, "training_peak_cuda_allocated_bytes": training_peak,
        "export_seconds": export_seconds, "first_last_loss": [losses[0], losses[-1]],
        "total_wall_seconds": time.perf_counter() - started_program,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "quality": quality,
    }
    (a.output_dir / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
    print(json.dumps({"crop": a.crop, "r1": quality["recall_at_1"], "map_at_r": quality["map_at_r"]}), flush=True)


if __name__ == "__main__":
    main()
