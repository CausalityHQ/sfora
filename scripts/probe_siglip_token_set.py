#!/usr/bin/env python3
"""Leakage-safe Cars train-holdout screen for token-set late interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch

from sfora.data import _HF_DATASET_REVISIONS, load_image_retrieval_examples, materialize_image
from sfora.kernels.set_maxsim import fused_set_maxsim
from sfora.token_set_proxy_anchor import select_attention_tokens

_MODEL_NAME = "google/siglip-base-patch16-224"
_MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_HOLDOUT_CLASSES = tuple(range(82, 98))


def validate_train_holdout(*, split: str, labels: torch.Tensor) -> None:
    """Reject any evaluation surface other than Cars train classes 82..97."""

    if split != "train":
        raise ValueError("the token-set screen is restricted to the train split")
    observed = set(int(label) for label in labels.tolist())
    if observed != set(_HOLDOUT_CLASSES):
        raise ValueError("the holdout must contain exactly Cars train classes 82 through 97")
    counts = torch.bincount(labels.to(torch.int64), minlength=98)[82:98]
    if bool((counts < 2).any()):
        raise ValueError("every holdout class needs at least two images")


def score_token_sets(
    global_embeddings: torch.Tensor,
    token_sets: torch.Tensor,
    token_weights: torch.Tensor,
    labels: torch.Tensor,
    *,
    set_weight: float,
    query_block: int,
) -> dict[str, float]:
    """Compute exact leave-one-out R@1 for global, set, and fixed hybrid scores."""

    if not 0.0 <= set_weight <= 1.0:
        raise ValueError("set_weight must lie in [0, 1]")
    if query_block < 1:
        raise ValueError("query_block must be positive")
    count = int(global_embeddings.shape[0])
    if token_sets.shape[0] != count or token_weights.shape[0] != count or labels.shape != (count,):
        raise ValueError("embedding, token, weight, and label counts differ")
    global_embeddings = torch.nn.functional.normalize(global_embeddings, dim=-1)
    token_sets = torch.nn.functional.normalize(token_sets, dim=-1)
    correct = {"pooled": 0, "set": 0, "hybrid": 0}
    for start in range(0, count, query_block):
        stop = min(start + query_block, count)
        pooled_scores = global_embeddings[start:stop] @ global_embeddings.T
        set_scores = fused_set_maxsim(
            token_sets[start:stop],
            token_sets,
            query_weights=token_weights[start:stop],
            gallery_weights=token_weights,
        )
        hybrid_scores = (1.0 - set_weight) * pooled_scores + set_weight * set_scores
        rows = torch.arange(stop - start, device=labels.device)
        columns = torch.arange(start, stop, device=labels.device)
        for scores in (pooled_scores, set_scores, hybrid_scores):
            scores[rows, columns] = -torch.inf
        expected = labels[start:stop]
        correct["pooled"] += int((labels[pooled_scores.argmax(dim=1)] == expected).sum())
        correct["set"] += int((labels[set_scores.argmax(dim=1)] == expected).sum())
        correct["hybrid"] += int((labels[hybrid_scores.argmax(dim=1)] == expected).sum())
    return {
        "pooled_recall_at_1": correct["pooled"] / count,
        "set_recall_at_1": correct["set"] / count,
        "hybrid_recall_at_1": correct["hybrid"] / count,
    }


def _encode_holdout(
    images: list[object],
    *,
    model_name: str,
    model_revision: str,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name, revision=model_revision)
    model = AutoModel.from_pretrained(model_name, revision=model_revision).eval().to(device)
    globals_: list[torch.Tensor] = []
    tokens_: list[torch.Tensor] = []
    weights_: list[torch.Tensor] = []
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            features = processor(images=batch, return_tensors="pt")
            pixel_values = features["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model.vision_model(pixel_values=pixel_values)
                hidden = output.last_hidden_state
                pooled = output.pooler_output
                head = model.vision_model.head
                probe = head.probe.expand(hidden.shape[0], -1, -1)
                _, attention = head.attention(
                    probe,
                    hidden,
                    hidden,
                    need_weights=True,
                    average_attn_weights=True,
                )
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            selected, weights, _ = select_attention_tokens(
                hidden,
                attention[:, 0, :].float(),
                top_k=top_k,
            )
            globals_.append(pooled.cpu())
            tokens_.append(selected.to(torch.float16).cpu())
            weights_.append(weights.cpu())
    del model
    return torch.cat(globals_), torch.cat(tokens_), torch.cat(weights_)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    if partial.exists():
        raise FileExistsError(f"refusing pre-existing partial {partial}")
    partial.write_bytes(payload)
    partial.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-digest", required=True)
    parser.add_argument("--model-name", default=_MODEL_NAME)
    parser.add_argument("--model-revision", default=_MODEL_REVISION)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--set-weight", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-block", type=int, default=32)
    args = parser.parse_args()
    if args.model_name != _MODEL_NAME:
        raise ValueError(f"model name must remain {_MODEL_NAME}")
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        raise ValueError("source revision must be an exact 40-character commit")
    if re.fullmatch(r"[0-9a-f]{64}", args.source_tree_digest) is None:
        raise ValueError("source tree digest must be an exact SHA-256")
    if re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is None:
        raise ValueError("model revision must be an exact 40-character commit")
    if not torch.cuda.is_available():
        raise RuntimeError("the token-set screen requires CUDA")
    if _HF_DATASET_REVISIONS["tanganke/stanford_cars"] != _DATASET_REVISION:
        raise RuntimeError("the Cars dataset revision authority differs")

    examples = load_image_retrieval_examples(dataset_name="cars", split="train")
    dataset_examples_sha256 = hashlib.sha256(
        _canonical_bytes(
            {
                "examples": sorted(
                    (str(example.example_id), int(example.label)) for example in examples
                )
            }
        )
    ).hexdigest()
    holdout = [example for example in examples if int(example.label) in _HOLDOUT_CLASSES]
    labels_cpu = torch.tensor([int(example.label) for example in holdout], dtype=torch.int64)
    validate_train_holdout(split="train", labels=labels_cpu)
    global_cpu, token_cpu, weight_cpu = _encode_holdout(
        [materialize_image(example.image) for example in holdout],
        model_name=args.model_name,
        model_revision=args.model_revision,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=torch.device("cuda"),
    )
    device = torch.device("cuda")
    metrics = score_token_sets(
        global_cpu.to(device),
        token_cpu.to(device),
        weight_cpu.to(device),
        labels_cpu.to(device),
        set_weight=args.set_weight,
        query_block=args.query_block,
    )
    gates = {
        "pooled_minimum": 0.82,
        "set_minimum_relative_to_pooled": -0.01,
        "hybrid_minimum_relative_to_pooled": 0.0,
    }
    passed = (
        metrics["pooled_recall_at_1"] >= gates["pooled_minimum"]
        and metrics["set_recall_at_1"]
        >= metrics["pooled_recall_at_1"] + gates["set_minimum_relative_to_pooled"]
        and metrics["hybrid_recall_at_1"]
        >= metrics["pooled_recall_at_1"] + gates["hybrid_minimum_relative_to_pooled"]
    )
    result = {
        "schema": "sfora-siglip-token-set-screen-v1",
        "claim_eligible": False,
        "source_revision": args.source_revision,
        "source_tree_digest": args.source_tree_digest,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "dataset_examples_sha256": dataset_examples_sha256,
        "split": "train",
        "holdout_classes": list(_HOLDOUT_CLASSES),
        "examples": len(holdout),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "top_k": args.top_k,
        "set_weight": args.set_weight,
        "metrics": metrics,
        "gates": gates,
        "passed": passed,
    }
    payload = _canonical_bytes(result)
    _write_new(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "passed": passed,
                **metrics,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
