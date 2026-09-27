#!/usr/bin/env python3
"""Frozen three-seed TRAIN-only patch-token fusion screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from score_inshop_crop_view_pair import NATIVE_SHA, bootstrap_lower, packed_hits, roles, sha256
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition

SEED_ORDER = (179026, 179024, 179027)
CHECKPOINT_SHAS = {
    179024: "dc5025998a3ea1902cb8251dedb1a4d4fd2659ffc20413c3425178b3e311f376",
    179026: "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089",
    179027: "4816912a52ed939e4461ba566abfcc0f3e4994b839ddc90e9c68355c554869a0",
}
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def save_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "native-library", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    if (
        len(args.checkpoint) != 3
        or len(args.receipt) != 3
        or (args.output_dir / "receipt.json").exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.native_library) != NATIVE_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("trained-token source authority differs")
    sources = {}
    for seed, checkpoint, receipt in zip(
        sorted(CHECKPOINT_SHAS), args.checkpoint, args.receipt, strict=True
    ):
        meta = json.loads(receipt.read_text())
        expected = CHECKPOINT_SHAS[seed]
        if (
            sha256(checkpoint) != expected
            or meta.get("checkpoint_sha256") != expected
            or meta.get("seed") != seed
            or meta.get("arm") != "freeze_emb"
            or meta.get("updates") != 1_000
            or meta.get("fit_rows_sha256") != FIT_SHA
            or meta.get("held_rows_sha256") != HELD_SHA
            or len(meta.get("quality", {}).get("per_query_r1", ())) != 12_599
        ):
            raise ValueError(f"trained-token checkpoint {seed} differs")
        sources[seed] = (checkpoint, receipt, meta)
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    if (
        len(fit) != 13_283
        or len(held) != 12_599
        or digest_rows(fit) != FIT_SHA
        or digest_rows(held) != HELD_SHA
    ):
        raise ValueError("trained-token TRAIN partition differs")
    paths = tuple(train[row].image_path for row in held)
    labels = tuple(train[row].label for row in held)
    classes = {label: row for row, label in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([classes[label] for label in labels], dtype=torch.int64)
    query, gallery = roles(labels, paths, args.dataset_root)
    if (len(query), len(gallery)) != (6_354, 6_245):
        raise ValueError("trained-token fixed roles differ")
    if any(
        hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
        for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
    ):
        raise ValueError("trained-token role digests differ")

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    reports: list[dict[str, Any]] = []
    deltas: list[np.ndarray] = []
    for seed in SEED_ORDER:
        checkpoint_path, receipt_path, meta = sources[seed]
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        vision.load_state_dict(state["vision"], strict=True)
        head.load_state_dict(state["head"], strict=True)
        if any(
            not torch.equal(value.cpu(), state["vision"][name])
            for name, value in vision.state_dict().items()
        ) or any(
            not torch.equal(value.cpu(), state["head"][name])
            for name, value in head.state_dict().items()
        ):
            raise ValueError(f"trained-token FP32 checkpoint {seed} rounded")
        del state
        token_means: list[torch.Tensor] = []

        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            captured: list[torch.Tensor] = token_means,
        ) -> None:
            tokens = output.last_hidden_state
            if tokens is None or tokens.ndim != 3 or tokens.shape[1:] != (256, 1024):
                raise ValueError("trained-token final patch geometry differs")
            captured.append(tokens.float().mean(dim=1).cpu())

        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        hook = vision.register_forward_hook(capture)
        try:
            base = export_all(
                vision, head, paths, tuple(range(len(paths))), processor, workers=4, batch_size=64
            )
        finally:
            hook.remove()
        torch.cuda.synchronize()
        export_wall = time.perf_counter() - started
        means = torch.cat(token_means)
        if means.shape != (len(held), 1024) or not bool(torch.isfinite(means).all()):
            raise ValueError("trained-token mean export differs")
        token_heads = []
        with torch.inference_mode():
            for chunk in means.split(64):
                token_heads.append(
                    torch.nn.functional.normalize(
                        compact_head_features(chunk.cuda(), head), dim=1
                    ).cpu()
                )
        token = torch.cat(token_heads)
        candidate = torch.nn.functional.normalize(0.75 * base + 0.25 * token, dim=1)
        scored = {}
        for name, values in (("base", base), ("candidate", candidate)):
            score_started = time.perf_counter()
            packed = pack_int8_unit_embeddings(values)
            symmetric = score_packed_full_gallery(
                packed.codes.float(),
                packed.inverse_norms,
                label_ids,
                torch.arange(len(held)),
                device=torch.device("cuda"),
            )
            asymmetric = packed_hits(values.numpy(), labels, query, gallery, args.native_library)
            scored[name] = {
                "symmetric_r1": symmetric["recall_at_1"],
                "symmetric_map_at_r": symmetric["map_at_r"],
                "asymmetric_r1": float(asymmetric.mean()),
                "asymmetric_hits": asymmetric.astype(int).tolist(),
                "packed_bytes_per_image": 130,
                "score_wall_seconds": time.perf_counter() - score_started,
            }
            if name == "base" and (
                symmetric["per_query_r1"] != meta["quality"]["per_query_r1"]
                or abs(symmetric["map_at_r"] - meta["quality"]["map_at_r"]) > 1e-7
            ):
                raise ValueError(f"trained-token base packed replay {seed} differs")
        asym_delta = 100 * (scored["candidate"]["asymmetric_r1"] - scored["base"]["asymmetric_r1"])
        map_delta = scored["candidate"]["symmetric_map_at_r"] - scored["base"]["symmetric_map_at_r"]
        report = {
            "seed": seed,
            "checkpoint_sha256": sha256(checkpoint_path),
            "receipt_sha256": sha256(receipt_path),
            "export_wall_seconds": export_wall,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scores": scored,
            "asymmetric_delta_pp": asym_delta,
            "symmetric_map_delta": map_delta,
        }
        save_new(args.output_dir / f"seed-{seed}.json", report)
        reports.append(report)
        deltas.append(
            np.asarray(scored["candidate"]["asymmetric_hits"], dtype=np.float64)
            - np.asarray(scored["base"]["asymmetric_hits"], dtype=np.float64)
        )
        print(
            json.dumps(
                {"seed": seed, "asymmetric_delta_pp": asym_delta, "symmetric_map_delta": map_delta}
            ),
            flush=True,
        )
        if seed == 179026 and (asym_delta <= 0 or map_delta < -0.002):
            break
    mean_delta = np.mean(np.stack(deltas), axis=0)
    lower_pp = 100 * bootstrap_lower(mean_delta, np.asarray(labels)[query])
    gates = {
        "three_seeds": len(reports) == 3,
        "every_asymmetric_nonnegative": all(row["asymmetric_delta_pp"] >= 0 for row in reports),
        "mean_asymmetric_gain": 100 * float(mean_delta.mean()) >= 0.25,
        "paired_lower_positive": lower_pp > 0,
        "every_symmetric_map": all(row["symmetric_map_delta"] >= -0.002 for row in reports),
    }
    save_new(
        args.output_dir / "receipt.json",
        {
            "schema": "sfora-inshop-trained-token-mean-train-v1",
            "claim_eligible": False,
            "source_sha256": sha256(Path(__file__)),
            "helper_source_sha256": {
                name: sha256(Path(importlib.import_module(name).__file__))
                for name in (
                    "score_inshop_crop_view_pair",
                    "train_sop_siglip2_compact",
                    "sfora.joint_relational_compaction",
                    "sfora.sop_compact_training",
                )
            },
            "native_library_sha256": NATIVE_SHA,
            "partition_sha256": PARTITION_SHA,
            "query_rows_sha256": QUERY_SHA,
            "gallery_rows_sha256": GALLERY_SHA,
            "seed_order": list(SEED_ORDER),
            "seeds_run": [row["seed"] for row in reports],
            "mean_asymmetric_delta_pp": 100 * float(mean_delta.mean()),
            "product_bootstrap_lower_pp": lower_pp,
            "gates": gates,
            "advance": all(gates.values()),
            "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
        },
    )


if __name__ == "__main__":
    main()
