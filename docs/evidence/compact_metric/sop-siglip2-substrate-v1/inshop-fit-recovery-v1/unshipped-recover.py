#!/usr/bin/env python3
"""Measure the trained In-Shop fit-product positive tail before changing the loss."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from sfora.unicom_inshop import parse_inshop_partition
from torch import nn
from train_sop_siglip2_compact import export_all, sha256

RECEIPT_SHA = "76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc"
TRAIN_HELPER_SHA = "e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6"
TAIL_RECEIPT_SHA = "953be9e6eab846f22e3e187f043eebfaf3161af0bbe4a752f6c68987ffb7fb49"


def save_exact_features(values: np.ndarray, expected_sha: str, output: Path) -> dict:
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if (
        values.dtype != np.dtype("<f4")
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
        or digest != expected_sha
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("recovered feature authority differs")
    with output.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"features_sha256": digest, "npy_sha256": sha256(output)}


@torch.inference_mode()
def positive_tail(features: torch.Tensor, labels: torch.Tensor) -> tuple[list[float], list[int]]:
    """Return best-negative minus worst-positive for every non-singleton anchor."""

    if (
        features.ndim != 2
        or labels.shape != (len(features),)
        or features.device != labels.device
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("positive-tail geometry differs")
    margins: list[float] = []
    ordinals: list[int] = []
    all_rows = torch.arange(len(features), device=features.device)
    for start in range(0, len(features), 64):
        stop = min(start + 64, len(features))
        rows = all_rows[start:stop]
        same = labels[start:stop, None] == labels[None, :]
        positive = same & (rows[:, None] != all_rows[None, :])
        eligible = positive.any(dim=1)
        scores = features[start:stop] @ features.T
        best_negative = scores.masked_fill(same, -torch.inf).max(dim=1).values
        worst_positive = scores.masked_fill(~positive, torch.inf).min(dim=1).values
        margins.extend((best_negative[eligible] - worst_positive[eligible]).cpu().tolist())
        ordinals.extend(rows[eligible].cpu().tolist())
    return margins, ordinals


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        x = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.9, 0.4], [0.0, 1.0]])
        margins, rows = positive_tail(x, torch.tensor([0, 0, 1, 2]))
        assert rows == [0, 1] and np.allclose(margins, [0.1, 0.16])
        values = np.arange(256, dtype="<f4").reshape(2, 128)
        digest = hashlib.sha256(values.tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.npy"
            save_exact_features(values, digest, output)
            assert np.array_equal(np.load(output, allow_pickle=False), values)
            for target, expected in [
                (output, digest),
                (output.with_name("bad.npy"), "0" * 64),
            ]:
                try:
                    save_exact_features(values, expected, target)
                except ValueError:
                    pass
                else:
                    raise AssertionError("overwrite or hash mismatch accepted")
            assert not output.with_name("bad.npy").exists()
        print("positive-tail self-test passed")
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-reference", type=Path)
    args = parser.parse_args()
    reference = None
    if args.recover_reference is not None:
        if (
            sha256(args.recover_reference) != TAIL_RECEIPT_SHA
            or args.output.suffix != ".npy"
            or args.output.is_symlink()
            or args.output.with_suffix(".json").exists()
            or args.output.with_suffix(".json").is_symlink()
        ):
            raise ValueError("fit recovery reference/output differs")
        reference = json.loads(args.recover_reference.read_text())
    receipt_path = args.run / "receipt.json"
    checkpoint_path = args.run / "checkpoint.pt"
    helper_file = sys.modules[export_all.__module__].__file__
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(receipt_path) != RECEIPT_SHA
        or helper_file is None
        or sha256(Path(helper_file)) != TRAIN_HELPER_SHA
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("positive-tail authority differs")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("seed") != 179026
        or receipt.get("arm") != "freeze_emb"
        or receipt.get("updates") != 1_000
        or receipt.get("checkpoint_sha256") != sha256(checkpoint_path)
    ):
        raise ValueError("positive-tail training receipt differs")
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    fit, held = split(tuple(row.label for row in train))
    if (
        len(fit) != 13_283
        or len(held) != 12_599
        or receipt.get("fit_rows_sha256") != digest_rows(fit)
    ):
        raise ValueError("positive-tail fit rows differ")
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot / "preprocessor_config.json",
        local_files_only=True,
        backend="torchvision",
    )
    if type(processor).__name__ != "SiglipImageProcessor":
        raise ValueError("positive-tail processor differs")
    model = AutoModel.from_pretrained(
        args.model_snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float16,
    )
    vision = model.vision_model.float()
    del model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head = nn.Linear(1024, 128)
    head.load_state_dict(checkpoint["head"], strict=True)
    vision, head = vision.cuda().eval(), head.cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = export_all(
        vision,
        head,
        tuple(train[i].image_path for i in fit),
        tuple(range(len(fit))),
        processor,
        workers=4,
        batch_size=64,
    )
    if reference is not None:
        if (
            tuple(values.shape) != (13_283, 128)
            or reference["checkpoint_sha256"] != receipt["checkpoint_sha256"]
            or reference["fit_rows_sha256"] != digest_rows(fit)
            or torch.cuda.max_memory_allocated()
            > reference["peak_cuda_allocated_bytes"] + (1 << 30)
        ):
            raise ValueError("fit recovery geometry/resource authority differs")
        report = save_exact_features(values.numpy(), reference["features_sha256"], args.output)
        report.update(
            claim_eligible=False,
            decision="PASS_EXACT_FIT_ARTIFACT_ONLY",
            split="official In-Shop TRAIN fit only; no held/test encoding or scoring",
            shape=list(values.shape),
            checkpoint_sha256=receipt["checkpoint_sha256"],
            fit_rows_sha256=digest_rows(fit),
            reference_receipt_sha256=TAIL_RECEIPT_SHA,
            source_sha256=sha256(Path(__file__)),
            helper_sha256=TRAIN_HELPER_SHA,
            wall_seconds=time.perf_counter() - started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        with args.output.with_suffix(".json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(report), flush=True)
        return
    names = tuple(train[i].label for i in fit)
    encoded = {name: i for i, name in enumerate(sorted(set(names)))}
    labels = torch.tensor([encoded[name] for name in names], device="cuda")
    margins, ordinals = positive_tail(values.cuda(), labels)
    eligible = sum(count for count in Counter(names).values() if count > 1)
    if (
        len(ordinals) != eligible
        or len(set(ordinals)) != eligible
        or not np.isfinite(margins).all()
    ):
        raise ValueError("positive-tail result differs")
    tail = sum(margin > 0.03 for margin in margins)
    report = {
        "schema": "sfora-inshop-fit-positive-tail-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit products only; self-excluded full fit gallery",
        "seed": 179026,
        "fit_rows": len(fit),
        "eligible_queries": eligible,
        "tail_margin_threshold": 0.03,
        "tail_count": tail,
        "tail_fraction": tail / eligible,
        "advance": tail / eligible >= 0.10,
        "margins": margins,
        "query_ordinals": ordinals,
        "training_receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "model_file_sha256": MODEL_HASHES,
        "source_sha256": sha256(Path(__file__)),
        "helper_sha256": TRAIN_HELPER_SHA,
        "fit_rows_sha256": digest_rows(fit),
        "features_sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    with args.output.open("xb") as stream:
        stream.write((json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"eligible": eligible, "tail": tail, "advance": report["advance"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
