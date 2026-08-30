#!/usr/bin/env python3
"""Run the preregistered paired Cars train-class TSPA mechanism screen."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

import torch

from sfora.data import (
    _HF_DATASET_REVISIONS,
    ImageExample,
    load_image_retrieval_examples,
    materialize_image,
)
from sfora.token_set_proxy_anchor import select_attention_tokens
from sfora.token_set_screen import (
    F1_TRAIN_CLASSES,
    F1_VALIDATION_CLASSES,
    validate_f1_class_partition,
)
from sfora.token_set_training import (
    F1ArmResult,
    F1TrainingConfig,
    FrozenTokenSetSplit,
    initialize_paired_f1_heads,
    train_f1_arm,
)

_MODEL_NAME = "google/siglip-base-patch16-224"
_MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_SEEDS = (17, 29, 43)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def validate_f0_receipt(receipt: dict[str, Any]) -> None:
    """Require the exact successful train-only F0 prerequisite."""

    expected = {
        "schema": "sfora-siglip-token-set-screen-v1",
        "claim_eligible": False,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "model_revision": _MODEL_REVISION,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "model_name": _MODEL_NAME,
        "top_k": 32,
        "set_weight": 0.25,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("F0 prerequisite authority differs")
    if receipt.get("passed") is not True:
        raise ValueError("F0 did not pass; F1 remains fenced")
    for key in ("dataset_examples_sha256", "source_tree_digest"):
        if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(key))) is None:
            raise ValueError(f"F0 {key} authority differs")
    if re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_revision"))) is None:
        raise ValueError("F0 source revision authority differs")


def summarize_f1_results(results: list[F1ArmResult]) -> dict[str, Any]:
    """Recompute paired F1 gates from exactly three seeds and three arms."""

    by_key = {(result.arm, result.seed): result for result in results}
    expected = {
        (arm, seed)
        for arm in ("pooled", "tspa", "token-shuffled-tspa")
        for seed in _SEEDS
    }
    if len(results) != len(expected) or set(by_key) != expected:
        raise ValueError("F1 results do not contain the exact paired seeds and arms")
    pooled = [by_key[("pooled", seed)].validation_recall_at_1 for seed in _SEEDS]
    tspa = [by_key[("tspa", seed)].validation_recall_at_1 for seed in _SEEDS]
    shuffled = [
        by_key[("token-shuffled-tspa", seed)].validation_recall_at_1 for seed in _SEEDS
    ]
    tspa_gain = sum(value - base for value, base in zip(tspa, pooled, strict=True)) / len(_SEEDS)
    shuffle_gain = sum(value - base for value, base in zip(tspa, shuffled, strict=True)) / len(
        _SEEDS
    )
    pooled_gains = [value - base for value, base in zip(tspa, pooled, strict=True)]
    shuffled_gains = [value - base for value, base in zip(tspa, shuffled, strict=True)]
    collapse = any(
        by_key[(arm, seed)].collapse_exceeded is not False
        for arm in ("tspa", "token-shuffled-tspa")
        for seed in _SEEDS
    )
    gates = {
        "minimum_mean_gain_over_pooled": 0.005,
        "minimum_mean_gain_over_token_shuffled": 0.005,
        "token_proxy_collapse_forbidden": True,
        "every_seed_gain_nonnegative": True,
    }
    return {
        "mean_pooled_recall_at_1": sum(pooled) / len(pooled),
        "mean_tspa_recall_at_1": sum(tspa) / len(tspa),
        "mean_token_shuffled_recall_at_1": sum(shuffled) / len(shuffled),
        "mean_tspa_gain_over_pooled": tspa_gain,
        "mean_tspa_gain_over_shuffled": shuffle_gain,
        "paired_tspa_gains_over_pooled": pooled_gains,
        "paired_tspa_gains_over_shuffled": shuffled_gains,
        "paired_tspa_gain_over_pooled_pstdev": statistics.pstdev(pooled_gains),
        "paired_tspa_gain_over_shuffled_pstdev": statistics.pstdev(shuffled_gains),
        "any_token_proxy_collapse": collapse,
        "gates": gates,
        "passed": (
            tspa_gain >= gates["minimum_mean_gain_over_pooled"]
            and shuffle_gain >= gates["minimum_mean_gain_over_token_shuffled"]
            and all(gain >= 0.0 for gain in pooled_gains)
            and all(gain >= 0.0 for gain in shuffled_gains)
            and not collapse
        ),
    }


def _encode_features(
    examples: list[ImageExample],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(_MODEL_NAME, revision=_MODEL_REVISION)
    model = AutoModel.from_pretrained(_MODEL_NAME, revision=_MODEL_REVISION).eval().to(device)
    globals_: list[torch.Tensor] = []
    tokens_: list[torch.Tensor] = []
    attention_: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            images = [
                materialize_image(example.image) for example in examples[start : start + batch_size]
            ]
            features = processor(images=images, return_tensors="pt")
            pixel_values = features["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.vision_model(pixel_values=pixel_values)
                hidden = output.last_hidden_state
                pooled = output.pooler_output
                head = model.vision_model.head
                probe = head.probe.expand(hidden.shape[0], -1, -1)
                _, weights = head.attention(
                    probe,
                    hidden,
                    hidden,
                    need_weights=True,
                    average_attn_weights=True,
                )
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            selected, attention, _ = select_attention_tokens(
                hidden,
                weights[:, 0, :].float(),
                top_k=32,
            )
            globals_.append(pooled.cpu())
            tokens_.append(selected.cpu())
            attention_.append(attention.cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(globals_), torch.cat(tokens_), torch.cat(attention_)


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("xb") as stream:
        stream.write(payload)
    try:
        os.link(partial, path)
    finally:
        partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0-receipt", type=Path, required=True)
    parser.add_argument("--f0-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    args = parser.parse_args()
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be an exact commit")
    if re.fullmatch(r"[0-9a-f]{64}", args.source_tree_digest) is None:
        raise ValueError("source tree digest must be an exact SHA-256")
    if args.feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("F1 requires CUDA")
    if _HF_DATASET_REVISIONS["tanganke/stanford_cars"] != _DATASET_REVISION:
        raise RuntimeError("Cars dataset revision authority differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    f0_bytes = args.f0_receipt.read_bytes()
    if re.fullmatch(r"[0-9a-f]{64}", args.f0_receipt_sha256) is None or hashlib.sha256(
        f0_bytes
    ).hexdigest() != args.f0_receipt_sha256:
        raise ValueError("F0 receipt SHA-256 differs")
    f0 = json.loads(f0_bytes)
    if not isinstance(f0, dict) or _canonical_bytes(f0) != f0_bytes:
        raise ValueError("F0 receipt is not canonical JSON")
    validate_f0_receipt(f0)

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    dataset_digest = hashlib.sha256(
        _canonical_bytes(
            {
                "examples": sorted(
                    (str(example.example_id), int(example.label)) for example in examples
                )
            }
        )
    ).hexdigest()
    if f0["dataset_examples_sha256"] != dataset_digest:
        raise ValueError("F0 and F1 Cars dataset identities differ")
    selected = [example for example in examples if int(example.label) < 82]
    labels = torch.tensor([int(example.label) for example in selected], dtype=torch.int64)
    train_mask = labels < 49
    validation_mask = ~train_mask
    validate_f1_class_partition(
        train_labels=labels[train_mask],
        validation_labels=labels[validation_mask],
    )
    globals_, tokens, attention = _encode_features(
        selected,
        batch_size=args.feature_batch_size,
        device=torch.device("cuda"),
    )
    train = FrozenTokenSetSplit(
        global_features=globals_[train_mask],
        token_features=tokens[train_mask],
        pretrained_attention=attention[train_mask],
        labels=labels[train_mask],
    )
    validation = FrozenTokenSetSplit(
        global_features=globals_[validation_mask],
        token_features=tokens[validation_mask],
        pretrained_attention=attention[validation_mask],
        labels=labels[validation_mask],
    )
    config = F1TrainingConfig()
    results: list[F1ArmResult] = []
    device = torch.device("cuda")
    for seed in _SEEDS:
        pooled, tspa, shuffled = initialize_paired_f1_heads(
            input_dimensions=globals_.shape[1],
            classes=49,
            global_dimensions=config.global_dimensions,
            token_dimensions=config.token_dimensions,
            token_proxies_per_class=config.token_proxies_per_class,
            set_weight=config.set_weight,
            seed=seed,
            device=device,
        )
        for arm, head in (
            ("pooled", pooled),
            ("tspa", tspa),
            ("token-shuffled-tspa", shuffled),
        ):
            results.append(
                train_f1_arm(
                    arm=arm,  # type: ignore[arg-type]
                    head=head,
                    train=train,
                    validation=validation,
                    config=config,
                    seed=seed,
                    device=device,
                )
            )
    summary = summarize_f1_results(results)
    result = {
        "schema": "sfora-siglip-token-set-f1-v1",
        "claim_eligible": False,
        "source_revision": args.source_revision,
        "source_tree_digest": args.source_tree_digest,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": dataset_digest,
        "split": "train",
        "train_classes": sorted(F1_TRAIN_CLASSES),
        "validation_classes": sorted(F1_VALIDATION_CLASSES),
        "model_name": _MODEL_NAME,
        "model_revision": _MODEL_REVISION,
        "f0_receipt_sha256": hashlib.sha256(f0_bytes).hexdigest(),
        "seeds": list(_SEEDS),
        "training": dataclasses.asdict(config),
        "feature_extraction": {
            "batch_size": args.feature_batch_size,
            "top_k": 32,
            "token_storage_dtype": "float32",
        },
        "determinism": {
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "cuda_matmul_tf32": False,
            "triton_dot_input_precision": "ieee",
        },
        "arms": [dataclasses.asdict(row) for row in results],
        "summary": summary,
        "passed": summary["passed"],
    }
    payload = _canonical_bytes(result)
    _write_new(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "passed": result["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
