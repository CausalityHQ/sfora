#!/usr/bin/env python3
"""Bounded TRAIN-fit image/text alignment with a category-matched sham."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_description_grounding import FIT_SHA, METADATA_SHA, fit_captions
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256

TOKENIZER_HASHES = {
    "tokenizer.json": "cb9140fae3ac5122c972d37adf83e1248471a38147ad76f8215c8872c6fd8322",
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "14afe629fe4959b9e0d51e1852b8d9f7ad074f90a1a7125a4fcdd17f06e78fc8",
    "special_tokens_map.json": "baec30ea10906f16adb8c18af7a34023002c1746542612b8b41c9f09e1351351",
}
ELIGIBILITY_SHA = "8c288436623814f79bf3aa38f05c0b1f3bfd2b1af2e32c6788b75835f0ab664a"


def shuffled_caption_names(categories: dict[str, str], rng: np.random.Generator) -> dict[str, str]:
    groups = defaultdict(list)
    for name, category in sorted(categories.items()):
        groups[category].append(name)
    mapping = {}
    for names in groups.values():
        if len(names) < 2:
            raise ValueError("caption-control category has fewer than two products")
        order = rng.permutation(names).tolist()
        mapping.update(zip(order, order[1:] + order[:1], strict=True))
    return mapping


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "partition",
        "metadata",
        "eligibility",
        "feature-cache",
        "model-snapshot",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or not torch.cuda.is_available()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.metadata) != METADATA_SHA
        or sha256(args.eligibility) != ELIGIBILITY_SHA
        or not json.loads(args.eligibility.read_text())["advance"]
        or sha256(args.feature_cache) != SOURCE_CACHE_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest
            for name, digest in {**MODEL_HASHES, **TOKENIZER_HASHES}.items()
        )
    ):
        raise ValueError("description alignment authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != FIT_SHA:
        raise ValueError("description alignment fit split differs")
    captions = fit_captions(json.loads(args.metadata.read_text()), {labels[row] for row in fit})
    categories = {
        labels[row]: "/".join(Path(train[row][0]).parts[:3])
        for row in fit
        if labels[row] in captions
    }
    shuffled = shuffled_caption_names(categories, np.random.default_rng(179019))
    selected_names = sorted(
        captions,
        key=lambda name: hashlib.sha256(b"inshop-description-pilot-v1\0" + name.encode()).digest(),
    )[:512]
    grouped = defaultdict(list)
    for row in fit:
        grouped[labels[row]].append(row)
    query = [
        min(grouped[name], key=lambda row: hashlib.sha256(train[row][0].encode()).digest())
        for name in selected_names
    ]
    values = np.load(args.feature_cache, mmap_mode="r", allow_pickle=False)
    if values.shape != (25_882, 1024) or values.dtype != np.float32:
        raise ValueError("description image cache differs")
    torch.set_num_threads(8)
    image = torch.nn.functional.normalize(torch.from_numpy(values[list(fit)].copy()), dim=1)
    positions = {row: place for place, row in enumerate(fit)}
    query_image = image[[positions[row] for row in query]]
    similarities = query_image @ image.T
    same = np.asarray([[labels[q] == labels[g] for g in fit] for q in query])
    similarities.masked_fill_(torch.from_numpy(same), -torch.inf)
    negatives = [fit[ordinal] for ordinal in similarities.argmax(dim=1).tolist()]
    if any(labels[row] not in captions for row in negatives):
        raise ValueError("selected hard impostor has no usable description")
    manifest = [
        {
            "query_train_row": q,
            "negative_train_row": n,
            "query_path": train[q][0],
            "negative_path": train[n][0],
        }
        for q, n in zip(query, negatives, strict=True)
    ]
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True)
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    text_model = full.text_model.cuda().eval()
    del full
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    names = sorted(captions)
    text_parts = []
    for start in range(0, len(names), 64):
        inputs = tokenizer(
            [captions[name] for name in names[start : start + 64]],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        text_parts.append(
            text_model(**{key: value.cuda() for key, value in inputs.items()})
            .pooler_output.float()
            .cpu()
        )
    text = torch.nn.functional.normalize(torch.cat(text_parts), dim=1)
    if text.shape != (len(names), 1024) or not bool(torch.isfinite(text).all()):
        raise ValueError("description text representation differs")
    indices = {name: place for place, name in enumerate(names)}
    actual, sham = [], []
    for q, n, feature in zip(query, negatives, query_image, strict=True):
        actual.append(float(feature @ (text[indices[labels[q]]] - text[indices[labels[n]]])))
        sham.append(
            float(
                feature @ (text[indices[shuffled[labels[q]]]] - text[indices[shuffled[labels[n]]]])
            )
        )
    actual_hits, sham_hits = (np.asarray(margins) > 0 for margins in (actual, sham))
    products = np.asarray(selected_names)
    lower = bootstrap_lower(actual_hits.astype(float), products)
    delta = actual_hits.astype(int) - sham_hits.astype(int)
    paired_lower = bootstrap_lower(delta, products)
    wall = time.perf_counter() - started
    report = {
        "schema": "sfora-inshop-description-alignment-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit products only; 512 distinct products",
        "source_sha256": sha256(Path(__file__)),
        "helper_source_sha256": {
            function.__name__: sha256(Path(function.__code__.co_filename))
            for function in (split, digest_rows, fit_captions, bootstrap_lower, sha256)
        },
        "partition_sha256": PARTITION_SHA,
        "eligibility_sha256": ELIGIBILITY_SHA,
        "metadata_sha256": METADATA_SHA,
        "features_sha256": SOURCE_CACHE_SHA,
        "fit_rows_sha256": FIT_SHA,
        "model_tokenizer_sha256": {**MODEL_HASHES, **TOKENIZER_HASHES},
        "caption_sha256": hashlib.sha256(json.dumps(captions, sort_keys=True).encode()).hexdigest(),
        "manifest": manifest,
        "shuffled_caption_names": shuffled,
        "true_caption_wins": int(actual_hits.sum()),
        "shuffled_caption_wins": int(sham_hits.sum()),
        "true_caption_win_fraction": float(actual_hits.mean()),
        "true_caption_win_lower95": lower,
        "paired_gain_pp": float(100 * delta.mean()),
        "paired_gain_lower95_pp": 100 * paired_lower,
        "true_margins": actual,
        "shuffled_margins": sham,
        "alignment_wall_seconds": wall,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "advance": actual_hits.mean() >= 0.65
        and lower > 0.50
        and delta.mean() >= 0.10
        and paired_lower > 0
        and wall <= 120,
        "training_run": False,
        "serving_latency_measured": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "true_caption_wins",
                    "shuffled_caption_wins",
                    "paired_gain_pp",
                    "paired_gain_lower95_pp",
                    "alignment_wall_seconds",
                    "advance",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
