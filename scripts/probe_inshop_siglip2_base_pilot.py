#!/usr/bin/env python3
"""One matched fit-only source quality/encoder-cost pilot; no fine-tuning."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from PIL import Image
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from transformers import AutoImageProcessor, AutoModel

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_inshop import parse_inshop_partition

BASE_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
BASE_HASHES = {
    "config.json": "7b5aedcb8893e31376e129c1ffd7a5392f1a806dbc793ce53eda220c2ec59edf",
    "model.safetensors": "6125cacc01fa93bdc98a0c5101cefcd69b2ed1f8ab4f38d86f4ad5984f5dc863",
    "preprocessor_config.json": "d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff",
}


def packed_hits(query, gallery):
    q, g = (pack_int8_unit_embeddings(value) for value in (query, gallery))
    scores = (
        (q.codes.float() @ g.codes.float().T)
        * q.inverse_norms.float()[:, None]
        * g.inverse_norms.float()[None, :]
    )
    return (scores.argmax(1) == torch.arange(len(query), device=query.device)).int()


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "large-snapshot", "base-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
    ):
        raise ValueError("Base pilot dataset/output authority differs")
    for path, revision, hashes in (
        (args.large_snapshot, MODEL_REVISION, MODEL_HASHES),
        (args.base_snapshot, BASE_REVISION, BASE_HASHES),
    ):
        if path.name != revision or any(
            sha256(path / name) != digest for name, digest in hashes.items()
        ):
            raise ValueError("Base pilot snapshot authority differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("Base pilot fit authority differs")
    counts = Counter(labels[row] for row in fit)
    selected = sorted(
        (name for name, count in counts.items() if count >= 2),
        key=lambda name: hashlib.sha256(b"inshop-siglip2-base-pilot-v1\0" + name.encode()).digest(),
    )[:512]
    if len(selected) != 512:
        raise ValueError("Base pilot product inventory differs")
    pairs = [
        sorted(
            (row for row in fit if labels[row] == name),
            key=lambda row: hashlib.sha256(
                str(train[row].image_path.relative_to(args.dataset_root)).encode()
            ).digest(),
        )[:2]
        for name in selected
    ]
    rows = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    manifest = [
        {
            "train_row": row,
            "relative_path": str(train[row].image_path.relative_to(args.dataset_root)),
            "image_sha256": sha256(train[row].image_path),
        }
        for row in rows
    ]
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    models, processors = {}, {}
    for arm, path, layers, width in (
        ("large", args.large_snapshot, 24, 1024),
        ("base", args.base_snapshot, 12, 768),
    ):
        processors[arm] = AutoImageProcessor.from_pretrained(
            path, local_files_only=True, backend="torchvision"
        )
        full = AutoModel.from_pretrained(
            path, local_files_only=True, use_safetensors=True, dtype=torch.float16
        )
        vision = full.vision_model
        del full
        if len(vision.encoder.layers) != layers or vision.config.hidden_size != width:
            raise ValueError("Base pilot model shape differs")
        models[arm] = vision.float().cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    outputs = {arm: [] for arm in models}
    fixed = None
    export_started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(rows), 32):
            images = []
            for row in rows[start : start + 32]:
                with Image.open(train[row].image_path) as image:
                    images.append(image.convert("RGB"))
            inputs = {arm: processors[arm](images=images, return_tensors="pt") for arm in models}
            if not torch.equal(inputs["base"]["pixel_values"], inputs["large"]["pixel_values"]):
                raise ValueError("Base pilot native processor pixel mismatch")
            for arm in models:
                tensors = {key: value.cuda() for key, value in inputs[arm].items()}
                if fixed is None:
                    fixed = tensors
                values = models[arm](**tensors).pooler_output
                if not torch.isfinite(values).all() or bool(
                    (values.float().norm(dim=1) == 0).any()
                ):
                    raise ValueError("Base pilot source numerical failure")
                outputs[arm].append(torch.nn.functional.normalize(values.float(), dim=1).cpu())
        torch.cuda.synchronize()
        export_wall = time.perf_counter() - export_started
        for _ in range(3):
            for model in models.values():
                model(**fixed)
        torch.cuda.synchronize()
        times = {arm: [] for arm in models}
        for block in range(10):
            for arm in ("large", "base") if block % 2 == 0 else ("base", "large"):
                torch.cuda.synchronize()
                call_started = time.perf_counter()
                models[arm](**fixed)
                torch.cuda.synchronize()
                times[arm].append(1000 * (time.perf_counter() - call_started))
    arms = {}
    for arm in models:
        values = torch.cat(outputs[arm])
        query, gallery = values[:512], values[512:]
        pca = fit_centered_pca(gallery, dimensions=128)
        hits = packed_hits(
            torch.nn.functional.normalize(pca.apply(query), dim=1),
            torch.nn.functional.normalize(pca.apply(gallery), dim=1),
        ).numpy()
        raw_hits = ((query @ gallery.T).argmax(1) == torch.arange(512)).int().numpy()
        arms[arm] = {
            "packed_r1": float(hits.mean()),
            "raw_float_r1": float(raw_hits.mean()),
            "packed_hits": hits.tolist(),
            "encoder_batch32_ms": times[arm],
            "encoder_batch32_p50_ms": float(np.median(times[arm])),
            "vision_parameters": sum(p.numel() for p in models[arm].parameters()),
        }
    delta = np.array(arms["base"]["packed_hits"]) - np.array(arms["large"]["packed_hits"])
    lower = bootstrap_lower(delta, np.asarray(selected))
    ratio = arms["base"]["encoder_batch32_p50_ms"] / arms["large"]["encoder_batch32_p50_ms"]
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    criteria = {
        "quality_point": float(delta.mean()) >= -0.03,
        "quality_lower": lower >= -0.05,
        "encoder_time": ratio <= 0.8,
        "wall": wall <= 120,
        "cuda": peak < 10_000_000_000,
    }
    report = {
        "schema": "sfora-inshop-siglip2-base-pilot-v1",
        "claim_eligible": False,
        "split": "official TRAIN fit only;512 products/512 query+512 gallery images",
        "arms": arms,
        "criteria": criteria,
        "advance": all(criteria.values()),
        "gain_pp": 100 * float(delta.mean()),
        "gain_lower95_pp": 100 * lower,
        "encoder_batch32_p50_ratio": ratio,
        "main_wall_seconds": wall,
        "export_both_arms_wall_seconds": export_wall,
        "peak_cuda_allocated_bytes": peak,
        "pixel_parity_all_images": True,
        "manifest": manifest,
        "partition_sha256": PARTITION_SHA,
        "fit_sha256": digest_rows(fit),
        "base_model_hashes": BASE_HASHES,
        "large_model_hashes": MODEL_HASHES,
        "source_sha256": sha256(Path(__file__)),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("arms", "manifest")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
