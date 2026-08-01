"""Train-fit test-apply diagnostic: can one seed predict a five-seed GPA consensus?"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from sfora.image_benchmark import image_self_retrieval_score


def _l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def _load(pattern: str) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    blocks: list[np.ndarray] = []
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            blocks.append(_l2(np.asarray(payload["embeddings"], dtype=np.float64)))
            current_labels = np.asarray(payload["labels"], dtype=np.int64)
            current_ids = np.asarray(payload["example_ids"])
        if labels is None:
            labels, ids = current_labels, current_ids
        elif not np.array_equal(labels, current_labels) or not np.array_equal(ids, current_ids):
            raise SystemExit(f"row mismatch in {path}")
    assert labels is not None
    return blocks, labels, paths


def _gpa_rotations(blocks: list[np.ndarray], iterations: int = 20) -> list[np.ndarray]:
    aligned = [block.copy() for block in blocks]
    rotations = [np.eye(block.shape[1]) for block in blocks]
    for _ in range(iterations):
        consensus = _l2(np.mean(aligned, axis=0))
        for index, block in enumerate(blocks):
            left, _, right = np.linalg.svd(block.T @ consensus, full_matrices=False)
            correction = np.eye(right.shape[0])
            correction[-1, -1] = np.sign(np.linalg.det(left @ right))
            rotations[index] = left @ correction @ right
            aligned[index] = block @ rotations[index]
    return rotations


def _score(values: np.ndarray, labels: np.ndarray) -> float:
    return float(image_self_retrieval_score(_l2(values), labels).recall_at_1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train, _, train_paths = _load(args.train)
    test, test_labels, test_paths = _load(args.test)
    train_stems = [Path(path).name.replace(".train.npz", "") for path in train_paths]
    test_stems = [Path(path).name.replace(".test.npz", "") for path in test_paths]
    if train_stems != test_stems:
        raise SystemExit(f"train/test pack mismatch: {train_stems} != {test_stems}")

    rotations = _gpa_rotations(train)
    train_target = _l2(np.mean([block @ rotation for block, rotation in zip(train, rotations)], axis=0))
    test_target = _l2(np.mean([block @ rotation for block, rotation in zip(test, rotations)], axis=0))

    x_train = train[0]
    dimension = x_train.shape[1]
    gram = x_train.T @ x_train
    ridge_lambda = 1e-3 * float(np.trace(gram)) / dimension
    ridge = np.linalg.solve(gram + ridge_lambda * np.eye(dimension), x_train.T @ train_target)

    payload = {
        "train_paths": train_paths,
        "test_paths": test_paths,
        "ridge_lambda": ridge_lambda,
        "seed0_r1": _score(test[0], test_labels),
        "concat_r1": _score(np.concatenate(test, axis=1), test_labels),
        "train_fit_gpa_r1": _score(test_target, test_labels),
        "orthogonal_control_r1": _score(test[0] @ rotations[0], test_labels),
        "ridge_r1": _score(test[0] @ ridge, test_labels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
