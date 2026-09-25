#!/usr/bin/env python3
"""Post-selection SOP TRAIN holdout-only packed-gallery sensitivity check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from compare_sop_siglip2_member_bank_arms import product_bootstrap
from train_sop_siglip2_compact import score_packed_full_gallery

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
CONTROL_SHA256 = "d88167bfcbf8152ee912c8382061afaf248e45fe52ae24cf5f1a5da739477fc3"
BANK_SHA256 = "2e73ee0e6252c91d54c815d52581abd0d303de09a56776a2fcd5b5e7908c981c"
SEED = 179019


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def score_reduced_gallery(
    packed: PackedInt8Embeddings,
    labels: np.ndarray,
    held: np.ndarray,
    native_library: Path,
) -> dict:
    reduced = PackedInt8Embeddings(
        packed.codes[held].contiguous(), packed.inverse_norms[held].contiguous()
    )
    scored = score_packed_full_gallery(
        reduced.codes.float(),
        reduced.inverse_norms,
        torch.from_numpy(labels[held].copy()),
        torch.arange(len(held), dtype=torch.long),
        device=torch.device("cuda:0"),
    )
    code = reduced.codes.float().cuda()
    inverse = reduced.inverse_norms.float().cuda()
    max_score_delta = 0.0
    with CutilePackedInt8Gallery.open_packed(native_library, reduced) as gallery:
        for start in range(0, len(held), 32):
            stop = min(start + 32, len(held))
            block = torch.arange(start, stop, device="cuda:0")
            query = PackedInt8Embeddings(
                reduced.codes[start:stop].contiguous(),
                reduced.inverse_norms[start:stop].contiguous(),
            )
            ordinals, native_scores = gallery.search_packed(query)
            oracle = (code[block] @ code.T) * inverse[block, None] * inverse[None, :]
            expected = torch.argsort(oracle, dim=1, descending=True, stable=True)[:, :10]
            if not np.array_equal(ordinals, expected.cpu().numpy()):
                raise ValueError("SOP reduced-gallery native ordinals differ")
            expected_scores = oracle.gather(1, expected).cpu().numpy()
            max_score_delta = max(
                max_score_delta,
                float(np.max(np.abs(native_scores - expected_scores))),
            )
    if max_score_delta > 1e-6:
        raise ValueError("SOP reduced-gallery native scores differ")
    scored["native_top10_exact"] = True
    scored["native_top10_max_score_abs_delta"] = max_score_delta
    return scored


@torch.inference_mode()
def control_error_origin(
    packed: PackedInt8Embeddings,
    labels: np.ndarray,
    fit: np.ndarray,
    held: np.ndarray,
    expected_r1: list[float],
) -> dict[str, int]:
    code = packed.codes.float().cuda()
    inverse = packed.inverse_norms.float().cuda()
    gallery_labels = torch.from_numpy(labels.copy()).cuda()
    fit_mask = torch.zeros(len(labels), device="cuda:0", dtype=torch.bool)
    fit_mask[torch.from_numpy(fit.copy()).cuda()] = True
    origin = {"fit_product_impostor": 0, "holdout_product_impostor": 0}
    observed = []
    for start in range(0, len(held), 64):
        rows = torch.from_numpy(held[start : start + 64].copy()).cuda()
        scores = (code[rows] @ code.T) * inverse[rows, None] * inverse[None, :]
        scores[torch.arange(len(rows), device="cuda:0"), rows] = -torch.inf
        nearest = torch.argmax(scores, dim=1)
        correct = gallery_labels[nearest] == gallery_labels[rows]
        observed.extend(correct.cpu().tolist())
        wrong = ~correct
        origin["fit_product_impostor"] += int(fit_mask[nearest[wrong]].sum())
        origin["holdout_product_impostor"] += int((~fit_mask[nearest[wrong]]).sum())
    if observed != [bool(value) for value in expected_r1]:
        raise ValueError("SOP reduced-gallery control origin differs from full receipt")
    return origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--control-embeddings", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--bank-embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.native_library) != NATIVE_SHA256
        or sha256(args.control_receipt) != CONTROL_SHA256
        or sha256(args.bank_receipt) != BANK_SHA256
    ):
        raise ValueError("SOP reduced-gallery authority differs")
    control = json.loads(args.control_receipt.read_text())
    bank = json.loads(args.bank_receipt.read_text())
    if (
        sha256(args.control_embeddings) != control["train_embeddings_sha256"]
        or sha256(args.bank_embeddings) != bank["train_embeddings_sha256"]
        or control["query_image_ids_sha256"] != bank["query_image_ids_sha256"]
    ):
        raise ValueError("SOP reduced-gallery embedding authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != (59_551,) or ids.shape != labels.shape:
        raise ValueError("SOP reduced-gallery archive differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (
        len(fit) != 53_700
        or len(held) != 5_851
        or len(np.unique(labels[held])) != 1_132
        or hashlib.sha256(ids[held].tobytes()).hexdigest() != control["query_image_ids_sha256"]
    ):
        raise ValueError("SOP reduced-gallery partition differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    arms = {}
    control_packed = None
    for name, path in (("control", args.control_embeddings), ("bank", args.bank_embeddings)):
        values = np.load(path, allow_pickle=False)
        if values.shape != (59_551, 128) or values.dtype != np.float32:
            raise ValueError("SOP reduced-gallery embedding geometry differs")
        packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))
        if name == "control":
            control_packed = packed
        arms[name] = score_reduced_gallery(packed, labels, held, args.native_library)
    origin = control_error_origin(
        control_packed, labels, fit, held, control["quality"]["per_query_r1"]
    )
    intervals = {}
    for metric, key in (("recall_at_1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        difference = np.asarray(arms["bank"][key]) - np.asarray(arms["control"][key])
        intervals[metric] = product_bootstrap(difference, labels[held])
    gate = bool(
        intervals["recall_at_1"]["point"] > 0
        and intervals["recall_at_1"]["lower_95"] > 0
        and intervals["map_at_r"]["point"] >= -0.005
    )
    result = {
        "schema": "sfora-sop-siglip2-unseen-gallery-sensitivity-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN holdout products only; no fit rows in reduced gallery",
        "source_sha256": sha256(Path(__file__)),
        "control_receipt_sha256": CONTROL_SHA256,
        "bank_receipt_sha256": BANK_SHA256,
        "source_archive_sha256": ARCHIVE_SHA256,
        "native_library_sha256": NATIVE_SHA256,
        "holdout_queries": len(held),
        "holdout_products": 1_132,
        "gallery_rows": len(held),
        "control_error_origin_full_gallery": origin,
        "arms": arms,
        "paired_product_bootstrap": intervals,
        "sensitivity_gate_pass": gate,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "control_r1": arms["control"]["recall_at_1"],
                "bank_r1": arms["bank"]["recall_at_1"],
                "paired_r1": intervals["recall_at_1"],
                "control_error_origin": origin,
                "sensitivity_gate_pass": gate,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
