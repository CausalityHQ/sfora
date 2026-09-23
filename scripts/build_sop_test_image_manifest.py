#!/usr/bin/env python3
"""Pin official SOP test-image bytes in metadata order before model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from train_sop_compact_backbone import publish_file_noreplace, sha256

EXPECTED_SOP_RECORD_SHA256 = "ea323eb87568d3f6ab88372ca5d1bf9a7033811cc958f899c532886f526fe992"


def build_manifest_bytes(records: Sequence[object]) -> tuple[bytes, int]:
    """Concatenate one raw SHA-256 digest per image in official record order."""

    if not records:
        raise ValueError("SOP test image inventory differs")
    manifest = bytearray()
    total_bytes = 0
    for record in records:
        image = Path(record.image_path).read_bytes()
        manifest.extend(hashlib.sha256(image).digest())
        total_bytes += len(image)
    return bytes(manifest), total_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--execute-sop-test-image-manifest", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.receipt_output.exists() or args.output == args.receipt_output:
        parser.error("SOP test image manifest destinations differ")
    return args


def main() -> None:
    args = parse_args()
    for destination in (args.output, args.receipt_output):
        destination.parent.mkdir(parents=True, exist_ok=True)
    script_sha256 = sha256(Path(__file__))
    parser_sha256 = sha256(Path(sys.modules[parse_sop_records.__module__].__file__))
    started = time.perf_counter()
    records = parse_sop_records(args.dataset_root)
    if ordered_record_sha256(records) != EXPECTED_SOP_RECORD_SHA256:
        raise ValueError("SOP official image inventory differs")
    test_records = tuple(record for record in records if record.split == "test")
    if len(test_records) != 60_502 or len({record.label for record in test_records}) != 11_316:
        raise ValueError("SOP official test inventory differs")
    manifest, total_bytes = build_manifest_bytes(test_records)
    if len(manifest) != 60_502 * 32 or sha256(Path(__file__)) != script_sha256:
        raise ValueError("SOP test image manifest build differs")
    publish_file_noreplace(args.output, lambda stream: stream.write(manifest))
    receipt = {
        "schema": "sfora-sop-test-image-content-manifest-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official test",
        "images": len(test_records),
        "classes": len({record.label for record in test_records}),
        "digest_format": "concatenated raw SHA-256 per image in official Ebay_test.txt order",
        "manifest_bytes": len(manifest),
        "source_image_bytes": total_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "inputs": {
            "sop_train_metadata_sha256": sha256(args.dataset_root / "Ebay_train.txt"),
            "sop_test_metadata_sha256": sha256(args.dataset_root / "Ebay_test.txt"),
            "ordered_records_sha256": EXPECTED_SOP_RECORD_SHA256,
            "ordered_test_records_sha256": ordered_record_sha256(test_records),
            "parser_source_sha256": parser_sha256,
            "builder_source_sha256": script_sha256,
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        },
        "argv": sys.argv,
    }
    publish_file_noreplace(
        args.receipt_output,
        lambda stream: stream.write(
            (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(
        json.dumps(
            {
                "manifest_sha256": receipt["inputs"]["manifest_sha256"],
                "source_image_bytes": total_bytes,
                "receipt": str(args.receipt_output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
