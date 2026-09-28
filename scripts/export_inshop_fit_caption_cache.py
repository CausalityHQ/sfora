#!/usr/bin/env python3
"""Export the qualified frozen text teacher for TRAIN fit products only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_description_alignment import TOKENIZER_HASHES, encode_captions
from probe_inshop_description_grounding import FIT_SHA, METADATA_SHA, fit_captions
from score_inshop_crop_view_pair import PARTITION_SHA, sha256

ALIGNMENT_SHA = "b6f77d2eaacac7e170e2d688b03db9e1ff3455f1ebfaed5829e107856f554aab"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "metadata", "model-snapshot", "alignment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    hashes = {**MODEL_HASHES, **TOKENIZER_HASHES}
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.metadata) != METADATA_SHA
        or sha256(args.alignment) != ALIGNMENT_SHA
        or not json.loads(args.alignment.read_text())["advance"]
        or args.model_snapshot.name != MODEL_REVISION
        or any(sha256(args.model_snapshot / name) != value for name, value in hashes.items())
    ):
        raise ValueError("fit caption export authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != FIT_SHA:
        raise ValueError("caption export fit rows differ")
    captions = fit_captions(json.loads(args.metadata.read_text()), {labels[row] for row in fit})
    caption_sha = hashlib.sha256(json.dumps(captions, sort_keys=True).encode()).hexdigest()
    if caption_sha != json.loads(args.alignment.read_text())["caption_sha256"]:
        raise ValueError("caption export logical metadata differs")
    torch.set_num_threads(8)
    features = encode_captions(args.model_snapshot, captions)
    if features.shape != (2003, 1024) or not torch.allclose(
        features.norm(dim=1), torch.ones(2003), atol=2e-6, rtol=0
    ):
        raise ValueError("caption teacher unit geometry differs")
    cache = {
        "schema": "sfora-inshop-fit-caption-cache-v1",
        "labels": sorted(captions),
        "features": features,
        "fit_rows_sha256": FIT_SHA,
        "metadata_sha256": METADATA_SHA,
        "caption_sha256": caption_sha,
        "model_tokenizer_sha256": hashes,
        "source_sha256": sha256(Path(__file__)),
        "encoder_source_sha256": sha256(Path(encode_captions.__code__.co_filename)),
        "alignment_sha256": ALIGNMENT_SHA,
        "token_length": 64,
    }
    with args.output.open("xb") as stream:
        torch.save(cache, stream)
    receipt = {key: value for key, value in cache.items() if key not in ("features", "labels")}
    receipt.update(
        {
            "cache_sha256": sha256(args.output),
            "fit_products": len(captions),
            "export_wall_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "held_or_official_rows": 0,
        }
    )
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
