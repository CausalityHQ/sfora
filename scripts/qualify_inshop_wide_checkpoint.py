#!/usr/bin/env python3
"""Matched32 qualification of fixed17-update artifacts; no training or refitting."""

import argparse
import json
import time
from pathlib import Path

import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import split
from probe_inshop_wide_training_head import fold_uncentered_head
from run_inshop_wide_head_smoke import packed_fit_images
from score_inshop_crop_view_pair import sha256
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition


def matched_checkpoint_checks(model_snapshot, raw, checkpoint, raw_sha, folded_sha, image_paths):
    if sha256(raw) != raw_sha or len(image_paths) != 64:
        raise ValueError("matched checkpoint authority differs")
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=model_snapshot,
        checkpoint=checkpoint,
        expected_checkpoint_sha256=folded_sha,
        model_file_sha256=MODEL_HASHES,
        precision="fp32_autocast",
        device=torch.device("cuda"),
    )
    parent = torch.load(raw, map_location="cpu", weights_only=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if checkpoint != raw and state["parent_checkpoint_sha256"] != raw_sha:
        raise ValueError("matched qualification fold lineage differs")
    same_vision = all(
        torch.equal(value.cpu(), parent["vision"][name])
        for name, value in encoder.vision.state_dict().items()
    )
    head = nn.Linear(1024, parent["head"]["weight"].shape[0])
    head.load_state_dict(parent["head"], strict=True)
    folded = fold_uncentered_head(head, state["compactor"]) if checkpoint != raw else head
    same_head = all(
        torch.equal(value, encoder.head.state_dict()[name].cpu())
        for name, value in folded.state_dict().items()
    )
    folded = folded.cuda().eval()
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    expected = []
    with torch.no_grad():
        for start in (0, 32):
            pixels = encoder.processor(images=images[start : start + 32], return_tensors="pt")[
                "pixel_values"
            ].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                pooled = encoder.vision(pixel_values=pixels).pooler_output
            expected.append(F.normalize(compact_head_features(pooled, folded), dim=1).cpu())
    reference = pack_int8_unit_embeddings(torch.cat(expected))
    codes, norms = packed_fit_images(encoder, images)
    return {
        "same_parent_vision": same_vision,
        "same_folded_head": same_head,
        "exact_codes": torch.equal(codes, reference.codes),
        "exact_inverse_norms": torch.equal(norms, reference.inverse_norms),
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("artifact-dir", "dataset-root", "model-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("matched qualification output exists")
    old = args.artifact_dir / "receipt.json"
    if sha256(old) != "e23a5f2c646ea437885e56a049b7611c3bf3f56fefb73b070cb0840928274921":
        raise ValueError("matched qualification parent evidence differs")
    receipt = json.loads(old.read_text())
    if not all(
        value for name, value in receipt["criteria"].items() if name != "exact_native_reload"
    ):
        raise ValueError("matched qualification requires original stability guards")
    wide = receipt["arms"]["256"]
    directory = args.artifact_dir / "256"
    raw = directory / "checkpoint.pt"
    checkpoint = directory / "folded_checkpoint.pt"
    if sha256(raw) != wide["checkpoint_sha256"]:
        raise ValueError("matched qualification raw checkpoint differs")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, _ = split(tuple(row.label for row in train))
    checks = matched_checkpoint_checks(
        args.model_snapshot,
        raw,
        checkpoint,
        wide["checkpoint_sha256"],
        wide["width_fold"]["folded_checkpoint_sha256"],
        tuple(train[row].image_path for row in fit[:64]),
    )
    criteria = {
        **checks,
        "budget25s": time.perf_counter() - started <= 25,
    }
    result = {
        "schema": "sfora-inshop-wide-matched32-qualification-v1",
        "claim_eligible": False,
        "decision": "GO_FROZEN_100_GATE" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "batch_size": 32,
        "fit_probe_images": 64,
        "training_updates_per_arm": 17,
        "new_training_updates": 0,
        "parent_smoke_sha256": sha256(old),
        "raw_checkpoint_sha256": wide["checkpoint_sha256"],
        "folded_checkpoint_sha256": wide["width_fold"]["folded_checkpoint_sha256"],
        "source_sha256": sha256(Path(__file__)),
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
