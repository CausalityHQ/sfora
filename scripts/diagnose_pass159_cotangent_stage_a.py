#!/usr/bin/env python3
"""Preregistered Stage-A diagnostic for Pass159 cotangent transplant.

The CLI and artifact-binding layer are added in later TDD steps.  The functions
below are deliberately NumPy-only so their geometry can be tested in isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_VECTOR_EPS = 1.0e-12
_ANTIPODAL_EPS = 1.0e-6
_PARTITION_DOMAIN = "pass159-stage-a-v1|"
_OFFICIAL_PARTITION = {
    "train": (25_882, 3_997),
    "query": (14_218, 3_985),
    "gallery": (12_612, 3_985),
}


@dataclass(frozen=True)
class BoundSeed:
    seed: int
    train_embeddings: np.ndarray
    train_raw_norms: np.ndarray
    train_labels: np.ndarray
    train_example_ids: np.ndarray
    proxies: np.ndarray
    proxy_labels: np.ndarray
    alpha: float
    delta: float
    official_recall_at_1: float
    artifact_binding: dict[str, Any]


def _require_unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite unit vector")
    norm = float(np.linalg.norm(array))
    if norm <= _VECTOR_EPS or not np.isclose(norm, 1.0, atol=1.0e-8, rtol=1.0e-8):
        raise ValueError(f"{name} must be a finite unit vector")
    return array


def _unit_rows(rows: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= _VECTOR_EPS):
        raise ValueError(f"{name} contains a zero vector")
    return array / norms[:, None]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    nonnegative = values >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exp_values = np.exp(values[~nonnegative])
    result[~nonnegative] = exp_values / (1.0 + exp_values)
    return result


def angular_proxy_anchor_cotangent(
    descriptor: np.ndarray,
    label: int,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    alpha: float,
    delta: float,
) -> np.ndarray:
    """Return the singleton Proxy Anchor cotangent on the descriptor sphere."""
    z = _require_unit(descriptor, name="descriptor")
    normalized_proxies = _unit_rows(proxies, name="proxies")
    labels = np.asarray(proxy_labels, dtype=np.int64)
    if labels.shape != (normalized_proxies.shape[0],):
        raise ValueError("proxy_labels must align with proxies")
    positive = labels == int(label)
    if int(positive.sum()) != 1:
        raise ValueError("singleton diagnostic requires exactly one proxy for the label")
    if normalized_proxies.shape[0] < 2:
        raise ValueError("singleton diagnostic requires at least two classes")
    if alpha <= 0.0 or delta < 0.0:
        raise ValueError("alpha must be positive and delta nonnegative")

    similarities = normalized_proxies @ z
    own_index = int(np.flatnonzero(positive)[0])
    ambient = (
        -float(alpha)
        * _sigmoid(np.asarray([float(alpha) * (float(delta) - similarities[own_index])]))[0]
        * normalized_proxies[own_index]
    )
    foreign = ~positive
    foreign_coefficients = (
        float(alpha)
        * _sigmoid(float(alpha) * (similarities[foreign] + float(delta)))
        / int(foreign.sum())
    )
    ambient = ambient + foreign_coefficients @ normalized_proxies[foreign]
    tangent = ambient - float(np.dot(ambient, z)) * z
    if not np.isfinite(tangent).all():
        raise ValueError("Proxy Anchor cotangent is non-finite")
    return tangent


def parallel_transport(
    tangent: np.ndarray,
    origin: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    """Parallel-transport a tangent vector along the shortest sphere geodesic."""
    x = _require_unit(origin, name="origin")
    y = _require_unit(target, name="target")
    vector = np.asarray(tangent, dtype=np.float64)
    if vector.shape != x.shape or not np.isfinite(vector).all():
        raise ValueError("tangent must be a finite vector aligned with origin")
    vector = vector - float(np.dot(vector, x)) * x
    norm = float(np.linalg.norm(vector))
    if norm <= _VECTOR_EPS:
        raise ValueError("tangent must have nonzero norm")
    denominator = 1.0 + float(np.dot(x, y))
    if denominator <= _ANTIPODAL_EPS:
        raise ValueError("origin and target are antipodal or numerically unresolved")
    transported = vector - (float(np.dot(vector, y)) / denominator) * (x + y)
    transported = transported - float(np.dot(transported, y)) * y
    if not np.isfinite(transported).all() or np.linalg.norm(transported) <= _VECTOR_EPS:
        raise ValueError("parallel transport produced an invalid vector")
    return transported


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def smooth_margin_gradient(
    receiver: np.ndarray,
    positive_supports: np.ndarray,
    frozen_foreign_supports: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Return the sphere gradient of the frozen smooth retrieval margin."""
    z = _require_unit(receiver, name="receiver")
    positives = _unit_rows(positive_supports, name="positive_supports")
    foreign = _unit_rows(frozen_foreign_supports, name="frozen_foreign_supports")
    if positives.shape[1] != z.shape[0] or foreign.shape[1] != z.shape[0]:
        raise ValueError("supports and receiver must share a descriptor dimension")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    positive_weights = _softmax((positives @ z) / float(tau))
    foreign_weights = _softmax((foreign @ z) / float(tau))
    ambient = positive_weights @ positives - foreign_weights @ foreign
    tangent = ambient - float(np.dot(ambient, z)) * z
    if np.linalg.norm(tangent) <= _VECTOR_EPS:
        raise ValueError("smooth margin gradient has zero norm")
    return tangent


def _partition_hash(example_id: str) -> bytes:
    return hashlib.sha256((_PARTITION_DOMAIN + str(example_id)).encode("utf-8")).digest()


def partition_identity(example_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two support positions and the remaining controller positions."""
    ids = np.asarray(example_ids)
    if ids.ndim != 1 or len(ids) < 5:
        raise ValueError("Pass159 identities require at least five images")
    as_text = [str(value) for value in ids.tolist()]
    if len(set(as_text)) != len(as_text):
        raise ValueError("Pass159 identity contains duplicate example IDs")
    order = sorted(range(len(ids)), key=lambda index: (_partition_hash(as_text[index]), as_text[index]))
    return np.asarray(order[:2], dtype=np.int64), np.asarray(order[2:], dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(array: np.ndarray, *, name: str) -> Any:
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return value.item()


def _canonical_query_gallery_recall_at_1(
    query: np.ndarray,
    query_labels: np.ndarray,
    gallery: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    query64 = np.asarray(query, dtype=np.float64)
    gallery64 = np.asarray(gallery, dtype=np.float64)
    query_sq = np.sum(query64 * query64, axis=1)
    gallery_sq = np.sum(gallery64 * gallery64, axis=1)
    correct = 0
    for start in range(0, len(query64), chunk_size):
        stop = min(start + chunk_size, len(query64))
        distances = (
            query_sq[start:stop, None]
            + gallery_sq[None, :]
            - 2.0 * (query64[start:stop] @ gallery64.T)
        )
        nearest = np.argmin(distances, axis=1)
        correct += int(np.sum(gallery_labels[nearest] == query_labels[start:stop]))
    return correct / len(query64)


def _manifest_paths(entry: dict[str, dict[str, str]]) -> dict[str, Path]:
    required = {
        "prehead_npz",
        "checkpoint_pt",
        "report_json",
        "train_npz",
        "query_npz",
        "gallery_npz",
        "retrieval_json",
    }
    if set(entry) != required:
        raise ValueError(f"manifest entry keys differ: {set(entry)} != {required}")
    paths: dict[str, Path] = {}
    for name in sorted(required):
        item = entry[name]
        if set(item) != {"path", "sha256"}:
            raise ValueError(f"manifest {name} must contain path and sha256")
        path = Path(item["path"])
        if not path.is_file():
            raise ValueError(f"manifest artifact is missing: {path}")
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ValueError(
                f"manifest SHA-256 mismatch for {name}: {observed} != {item['sha256']}"
            )
        paths[name] = path
    return paths


def _load_final_pack(
    path: Path,
    *,
    split: str,
    checkpoint_digest: str,
    report_digest: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "embeddings",
            "labels",
            "example_ids",
            "source_paths",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
            "report_sha256",
        }
        if set(archive.files) != required:
            raise ValueError(f"{split} final-pack keys differ from the frozen schema")
        result = {name: np.asarray(archive[name]) for name in archive.files}
    if _scalar(result["artifact_selection"], name=f"{split} artifact_selection") != "final_training_state":
        raise ValueError(f"{split} final pack is not a final training state")
    if _scalar(result["split"], name=f"{split} split") != split:
        raise ValueError(f"{split} final pack has the wrong split marker")
    if _scalar(result["checkpoint_sha256"], name=f"{split} checkpoint_sha256") != checkpoint_digest:
        raise ValueError(f"{split} final pack checkpoint digest differs")
    if _scalar(result["report_sha256"], name=f"{split} report_sha256") != report_digest:
        raise ValueError(f"{split} final pack report digest differs")
    return result


def _reconstruct_head(
    prehead: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(prehead, dtype=np.float32)
    raw = features @ np.asarray(weight, dtype=np.float32).T + np.asarray(bias, dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1)
    if not np.isfinite(raw).all() or np.any(norms <= _VECTOR_EPS):
        raise ValueError(f"{split} reconstructed head has nonfinite or zero-norm rows")
    return raw / norms[:, None], norms


def load_bound_seed(
    entry: dict[str, dict[str, str]],
    *,
    seed: int,
    expected_partition: dict[str, tuple[int, int]] | None = None,
) -> BoundSeed:
    """Fail-closed artifact binding before any Pass159 training statistic."""
    import torch

    expected = _OFFICIAL_PARTITION if expected_partition is None else expected_partition
    paths = _manifest_paths(entry)
    report_digest = entry["report_json"]["sha256"]
    checkpoint_digest = entry["checkpoint_pt"]["sha256"]
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    config = report.get("config")
    if not isinstance(config, dict):
        raise ValueError("report lacks a config object")
    frozen_config = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": int(seed),
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "checkpoint_selection_interval": 0,
        "backbone_name": "bn_inception",
        "head_pooling": "avg_max",
    }
    for key, value in frozen_config.items():
        if config.get(key) != value:
            raise ValueError(f"report config {key}={config.get(key)!r} != {value!r}")

    checkpoint = torch.load(paths["checkpoint_pt"], map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not a final training state")
    if checkpoint.get("training_config") != config:
        raise ValueError("checkpoint training_config differs from report config")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks a state_dict")
    required_state = {
        "model.embedding.weight",
        "model.embedding.bias",
        "metric_proxies",
        "metric_proxy_labels",
    }
    if not required_state.issubset(state):
        raise ValueError("checkpoint lacks Pass159 head or proxy tensors")
    weight = state["model.embedding.weight"].detach().cpu().numpy()
    bias = state["model.embedding.bias"].detach().cpu().numpy()
    proxies = _unit_rows(state["metric_proxies"].detach().cpu().numpy(), name="metric_proxies")
    proxy_labels = state["metric_proxy_labels"].detach().cpu().numpy().astype(np.int64)
    if len(np.unique(proxy_labels)) != len(proxy_labels):
        raise ValueError("checkpoint does not have exactly one proxy per class")

    packs = {
        split: _load_final_pack(
            paths[f"{split}_npz"],
            split=split,
            checkpoint_digest=checkpoint_digest,
            report_digest=report_digest,
        )
        for split in ("train", "query", "gallery")
    }
    with np.load(paths["prehead_npz"], allow_pickle=False) as archive:
        expected_prehead_keys = {
            "train",
            "train_labels",
            "query",
            "query_labels",
            "gallery",
            "gallery_labels",
        }
        if set(archive.files) != expected_prehead_keys:
            raise ValueError("pre-head pack keys differ from the frozen schema")
        prehead = {name: np.asarray(archive[name]) for name in archive.files}

    reconstructed: dict[str, np.ndarray] = {}
    raw_norms: dict[str, np.ndarray] = {}
    prehead_checks: dict[str, dict[str, Any]] = {}
    all_ids: list[set[str]] = []
    for split in ("train", "query", "gallery"):
        labels = np.asarray(packs[split]["labels"], dtype=np.int64)
        observed_partition = (len(labels), len(np.unique(labels)))
        if observed_partition != expected[split]:
            raise ValueError(
                f"{split} partition {observed_partition} != expected {expected[split]}"
            )
        prehead_labels = np.asarray(prehead[f"{split}_labels"], dtype=np.int64)
        if not np.array_equal(prehead_labels, labels):
            raise ValueError(f"{split} labels differ between pre-head and final packs")
        ids = np.asarray(packs[split]["example_ids"]).astype(str)
        if len(set(ids.tolist())) != len(ids):
            raise ValueError(f"{split} final pack has duplicate example IDs")
        all_ids.append(set(ids.tolist()))
        embeddings, norms = _reconstruct_head(prehead[split], weight, bias, split=split)
        exported = np.asarray(packs[split]["embeddings"], dtype=np.float32)
        if embeddings.shape != exported.shape:
            raise ValueError(f"{split} reconstructed embedding shape differs from final pack")
        if not np.isfinite(exported).all():
            raise ValueError(f"{split} final pack contains nonfinite embeddings")
        exported_norms = np.linalg.norm(exported, axis=1)
        if not np.allclose(exported_norms, 1.0, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError(f"{split} final-pack embeddings are not unit normalized")
        absolute_difference = np.abs(embeddings - exported)
        within_tolerance = bool(
            np.allclose(embeddings, exported, atol=2.0e-5, rtol=2.0e-5)
        )
        prehead_checks[split] = {
            "within_tolerance": within_tolerance,
            "max_abs_difference": float(absolute_difference.max()),
            "rows_above_2e_5": int(np.sum(absolute_difference.max(axis=1) > 2.0e-5)),
            "used_for_official_r1": False,
        }
        if split == "train" and not within_tolerance:
            raise ValueError("train reconstructed embeddings differ from final pack")
        reconstructed[split] = embeddings
        raw_norms[split] = norms
    if any(all_ids[left] & all_ids[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("example IDs overlap across official splits")

    official_r1 = _canonical_query_gallery_recall_at_1(
        np.asarray(packs["query"]["embeddings"], dtype=np.float32),
        np.asarray(packs["query"]["labels"], dtype=np.int64),
        np.asarray(packs["gallery"]["embeddings"], dtype=np.float32),
        np.asarray(packs["gallery"]["labels"], dtype=np.int64),
    )
    methods = report.get("methods")
    if not isinstance(methods, dict) or len(methods) != 1:
        raise ValueError("report must contain exactly one method")
    reported_r1 = float(next(iter(methods.values()))["recall_at_1"])
    retrieval = json.loads(paths["retrieval_json"].read_text(encoding="utf-8"))
    if retrieval.get("artifact_selection") != "final_training_state":
        raise ValueError("retrieval audit is not a final training state")
    if retrieval.get("checkpoint_sha256") != checkpoint_digest or retrieval.get("report_sha256") != report_digest:
        raise ValueError("retrieval audit artifact digests differ")
    if retrieval.get("resolved_training_steps") != checkpoint.get("training_step"):
        raise ValueError("retrieval audit training step differs from checkpoint")
    bound_r1_values = [
        reported_r1,
        float(retrieval.get("reported_final_recall_at_1")),
        float(retrieval.get("independent_recall_at_1")),
        float(retrieval.get("canonical_float64_euclidean_recall_at_1")),
    ]
    if any(value != official_r1 for value in bound_r1_values):
        raise ValueError(f"official R@1 mismatch: reconstructed={official_r1}, bound={bound_r1_values}")

    return BoundSeed(
        seed=int(seed),
        train_embeddings=reconstructed["train"],
        train_raw_norms=raw_norms["train"],
        train_labels=np.asarray(packs["train"]["labels"], dtype=np.int64),
        train_example_ids=np.asarray(packs["train"]["example_ids"]).astype(str),
        proxies=proxies,
        proxy_labels=proxy_labels,
        alpha=float(config["proxy_anchor_alpha"]),
        delta=float(config["proxy_anchor_delta"]),
        official_recall_at_1=official_r1,
        artifact_binding={
            "artifacts": {
                name: {"path": str(paths[name]), "sha256": entry[name]["sha256"]}
                for name in sorted(paths)
            },
            "prehead_reconstruction": prehead_checks,
            "official_r1_source": "digest_bound_final_query_gallery_packs",
        },
    )


def _hash_hex(value: str) -> str:
    return _partition_hash(str(value)).hex()


def _select_extreme(
    indices: np.ndarray,
    values: np.ndarray,
    example_ids: np.ndarray,
    *,
    largest: bool,
) -> int:
    if len(indices) == 0:
        raise ValueError("cannot select from an empty controller set")
    ordered = sorted(
        (int(index) for index in indices),
        key=lambda index: (
            -float(values[index]) if largest else float(values[index]),
            _hash_hex(str(example_ids[index])),
            str(example_ids[index]),
        ),
    )
    return ordered[0]


def _batch_angular_cotangents(
    descriptors: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    alpha: float,
    delta: float,
    chunk_size: int = 64,
) -> np.ndarray:
    z = _unit_rows(descriptors, name="cotangent descriptors")
    p = _unit_rows(proxies, name="cotangent proxies")
    labels = np.asarray(labels, dtype=np.int64)
    proxy_labels = np.asarray(proxy_labels, dtype=np.int64)
    proxy_index = {int(label): index for index, label in enumerate(proxy_labels.tolist())}
    if len(proxy_index) != len(proxy_labels):
        raise ValueError("cotangent batch requires one proxy per class")
    try:
        own_indices = np.asarray([proxy_index[int(label)] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"descriptor label lacks a proxy: {error.args[0]}") from error
    outputs = np.empty_like(z, dtype=np.float64)
    foreign_count = len(proxy_labels) - 1
    if foreign_count <= 0:
        raise ValueError("cotangent batch requires at least two proxies")
    for start in range(0, len(z), chunk_size):
        stop = min(start + chunk_size, len(z))
        rows = z[start:stop]
        own = own_indices[start:stop]
        similarities = rows @ p.T
        coefficients = (
            float(alpha)
            * _sigmoid(float(alpha) * (similarities + float(delta)))
            / foreign_count
        )
        coefficients[np.arange(stop - start), own] = 0.0
        ambient = coefficients @ p
        own_similarities = similarities[np.arange(stop - start), own]
        own_coefficients = -float(alpha) * _sigmoid(
            float(alpha) * (float(delta) - own_similarities)
        )
        ambient += own_coefficients[:, None] * p[own]
        outputs[start:stop] = ambient - np.sum(ambient * rows, axis=1)[:, None] * rows
    return outputs


def _normalize_direction(vector: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or norm <= _VECTOR_EPS:
        raise ValueError(f"{name} has zero or nonfinite norm")
    return array / norm


def _alignment(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.dot(
            _normalize_direction(left, name="alignment left"),
            _normalize_direction(right, name="alignment right"),
        )
    )


def _top_foreign_supports(
    receiver: np.ndarray,
    label: int,
    support_embeddings: np.ndarray,
    support_labels: np.ndarray,
    support_ids: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    eligible = np.flatnonzero(np.asarray(support_labels) != int(label))
    if len(eligible) < top_k:
        raise ValueError(f"only {len(eligible)} foreign supports are available for top-{top_k}")
    scores = np.asarray(support_embeddings, dtype=np.float64) @ np.asarray(
        receiver, dtype=np.float64
    )
    hashes = np.asarray([_hash_hex(str(value)) for value in support_ids[eligible]])
    ids = np.asarray(support_ids[eligible]).astype(str)
    order = np.lexsort((ids, hashes, -scores[eligible]))
    return eligible[order[:top_k]]


def compute_seed_rows(bound: BoundSeed, *, top_k: int = 32) -> dict[str, Any]:
    """Compute one preregistered training-only Stage-A row per eligible identity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    descriptors = np.asarray(bound.train_embeddings, dtype=np.float64)
    norms = np.asarray(bound.train_raw_norms, dtype=np.float64)
    labels = np.asarray(bound.train_labels, dtype=np.int64)
    example_ids = np.asarray(bound.train_example_ids).astype(str)
    if not (
        descriptors.shape[0] == norms.shape[0] == labels.shape[0] == example_ids.shape[0]
    ):
        raise ValueError("bound training arrays are misaligned")
    proxy_by_label = {
        int(label): np.asarray(bound.proxies[index], dtype=np.float64)
        for index, label in enumerate(bound.proxy_labels.tolist())
    }

    roles: dict[int, dict[str, Any]] = {}
    for label in sorted(np.unique(labels).tolist()):
        identity_indices = np.flatnonzero(labels == int(label))
        if len(identity_indices) < 5:
            continue
        local_support, local_controllers = partition_identity(example_ids[identity_indices])
        supports = identity_indices[local_support]
        controllers = identity_indices[local_controllers]
        receiver = _select_extreme(controllers, norms, example_ids, largest=False)
        donor = _select_extreme(controllers, norms, example_ids, largest=True)
        if receiver == donor:
            raise ValueError("receiver and candidate donor unexpectedly coincide")
        roles[int(label)] = {
            "indices": identity_indices,
            "supports": supports,
            "controllers": controllers,
            "receiver": receiver,
            "donor": donor,
        }
    if not roles:
        raise ValueError("no identity has the preregistered minimum of five images")

    feature_rows: list[tuple[float, float]] = []
    for label, role in roles.items():
        receiver = int(role["receiver"])
        own_proxy = proxy_by_label[label]
        for index in role["controllers"]:
            index = int(index)
            if index == receiver:
                continue
            feature_rows.append(
                (
                    float(np.dot(descriptors[receiver], descriptors[index])),
                    float(np.dot(descriptors[index], own_proxy)),
                )
            )
    feature_scales = np.std(np.asarray(feature_rows, dtype=np.float64), axis=0)
    feature_scales = np.maximum(feature_scales, _VECTOR_EPS)

    random_matches_candidate = 0
    singleton_match_alternatives = 0
    for label, role in roles.items():
        receiver = int(role["receiver"])
        donor = int(role["donor"])
        controllers = np.asarray(role["controllers"], dtype=np.int64)
        alternatives = controllers[controllers != receiver]
        fixed_hash = min(
            (int(index) for index in alternatives),
            key=lambda index: (_hash_hex(example_ids[index]), example_ids[index]),
        )
        random_matches_candidate += int(fixed_hash == donor)

        ordered = sorted(
            (int(index) for index in controllers),
            key=lambda index: (_hash_hex(example_ids[index]), example_ids[index]),
        )
        observed_norms = np.asarray([norms[index] for index in ordered], dtype=np.float64)
        shift_digest = hashlib.sha256(f"pass159-norm-permute-v1|{label}".encode("utf-8")).hexdigest()
        shift = int(shift_digest[:8], 16) % len(ordered)
        assigned_norms = np.roll(observed_norms, shift)
        permuted_candidates = [position for position, index in enumerate(ordered) if index != receiver]
        permuted_position = min(
            permuted_candidates,
            key=lambda position: (
                -float(assigned_norms[position]),
                _hash_hex(example_ids[ordered[position]]),
                example_ids[ordered[position]],
            ),
        )
        norm_permuted = ordered[permuted_position]

        match_candidates = [
            int(index) for index in controllers if int(index) not in {receiver, donor}
        ]
        if not match_candidates:
            raise ValueError("cosine matching lacks an alternative controller")
        singleton_match_alternatives += int(len(match_candidates) == 1)
        own_proxy = proxy_by_label[label]
        target_features = np.asarray(
            [
                np.dot(descriptors[receiver], descriptors[donor]),
                np.dot(descriptors[donor], own_proxy),
            ],
            dtype=np.float64,
        )

        def match_key(index: int) -> tuple[float, str, str]:
            features = np.asarray(
                [
                    np.dot(descriptors[receiver], descriptors[index]),
                    np.dot(descriptors[index], own_proxy),
                ],
                dtype=np.float64,
            )
            distance = float(np.sum(((features - target_features) / feature_scales) ** 2))
            return distance, _hash_hex(example_ids[index]), example_ids[index]

        matched = min(match_candidates, key=match_key)
        role.update(
            {
                "fixed_hash": fixed_hash,
                "norm_permuted": norm_permuted,
                "matched": matched,
                "match_distance": match_key(matched)[0],
            }
        )

    needed_indices = sorted(
        {
            int(role[key])
            for role in roles.values()
            for key in ("receiver", "donor", "fixed_hash", "norm_permuted", "matched")
        }
    )
    needed_cotangents = _batch_angular_cotangents(
        descriptors[needed_indices],
        labels[needed_indices],
        bound.proxies,
        bound.proxy_labels,
        alpha=bound.alpha,
        delta=bound.delta,
    )
    cotangent_by_index = {
        index: needed_cotangents[position] for position, index in enumerate(needed_indices)
    }
    role_labels = np.asarray(sorted(roles), dtype=np.int64)
    proxy_origins = np.asarray([proxy_by_label[int(label)] for label in role_labels])
    proxy_cotangents = _batch_angular_cotangents(
        proxy_origins,
        role_labels,
        bound.proxies,
        bound.proxy_labels,
        alpha=bound.alpha,
        delta=bound.delta,
    )
    proxy_cotangent_by_label = {
        int(label): proxy_cotangents[position] for position, label in enumerate(role_labels)
    }

    support_indices = np.concatenate([role["supports"] for role in roles.values()])
    support_embeddings = descriptors[support_indices]
    support_labels = labels[support_indices]
    support_ids = example_ids[support_indices]
    rows: list[dict[str, Any]] = []
    exclusion_reasons: dict[str, int] = {}
    for label, role in roles.items():
        try:
            receiver = int(role["receiver"])
            donor = int(role["donor"])
            receiver_descriptor = descriptors[receiver]
            foreign_local = _top_foreign_supports(
                receiver_descriptor,
                label,
                support_embeddings,
                support_labels,
                support_ids,
                top_k=top_k,
            )
            q = smooth_margin_gradient(
                receiver_descriptor,
                descriptors[np.asarray(role["supports"], dtype=np.int64)],
                support_embeddings[foreign_local],
                tau=0.05,
            )

            def transported(index: int) -> np.ndarray:
                return parallel_transport(
                    -cotangent_by_index[int(index)],
                    descriptors[int(index)],
                    receiver_descriptor,
                )

            candidate = transported(donor)
            own = -cotangent_by_index[receiver]
            projection = -cotangent_by_index[donor]
            projection = projection - float(np.dot(projection, receiver_descriptor)) * receiver_descriptor
            proxy_only = parallel_transport(
                -proxy_cotangent_by_label[label],
                proxy_by_label[label],
                receiver_descriptor,
            )
            directions = {
                "candidate": candidate,
                "receiver_own": own,
                "fixed_hash_donor": transported(int(role["fixed_hash"])),
                "norm_permuted_donor": transported(int(role["norm_permuted"])),
                "cosine_matched_donor": transported(int(role["matched"])),
                "ambient_projection": projection,
                "proxy_only": proxy_only,
            }
            unit_candidate = _normalize_direction(candidate, name="candidate")
            unit_own = _normalize_direction(own, name="receiver own")
            orthogonal = unit_candidate - float(np.dot(unit_candidate, unit_own)) * unit_own
            row = {
                "seed": int(bound.seed),
                "label": int(label),
                "support_ids": [str(example_ids[index]) for index in role["supports"]],
                "controller_ids": [str(example_ids[index]) for index in role["controllers"]],
                "receiver_id": str(example_ids[receiver]),
                "candidate_donor_id": str(example_ids[donor]),
                "fixed_hash_donor_id": str(example_ids[int(role["fixed_hash"])]),
                "norm_permuted_donor_id": str(example_ids[int(role["norm_permuted"])]),
                "cosine_matched_donor_id": str(example_ids[int(role["matched"])]),
                "foreign_support_ids": [str(support_ids[index]) for index in foreign_local],
                "receiver_norm": float(norms[receiver]),
                "candidate_donor_norm": float(norms[donor]),
                "cosine_match_standardized_squared_distance": float(role["match_distance"]),
                "alignments": {name: _alignment(q, direction) for name, direction in directions.items()},
                "orthogonal_fraction": float(np.linalg.norm(orthogonal)),
            }
            rows.append(row)
        except ValueError as error:
            reason = str(error)
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    return {
        "seed": int(bound.seed),
        "eligible_identities": len(roles),
        "excluded_identities": len(roles) - len(rows),
        "exclusion_reasons": exclusion_reasons,
        "feature_scales": feature_scales.tolist(),
        "fixed_hash_matches_candidate": random_matches_candidate,
        "single_cosine_match_alternative": singleton_match_alternatives,
        "identity_rows": rows,
    }


_CONTROL_ORDER = (
    "receiver_own",
    "fixed_hash_donor",
    "norm_permuted_donor",
    "cosine_matched_donor",
    "ambient_projection",
    "proxy_only",
)


def clustered_verdict(
    identity_rows: list[dict[str, Any]],
    *,
    seed: int = 159,
    replicates: int = 1_000,
) -> dict[str, Any]:
    """Apply the frozen joint-identity bootstrap and Stage-A decision rule."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    by_label: dict[int, dict[int, dict[str, Any]]] = {}
    for row in identity_rows:
        label = int(row["label"])
        row_seed = int(row["seed"])
        if row_seed in by_label.setdefault(label, {}):
            raise ValueError(f"duplicate row for label={label}, seed={row_seed}")
        alignments = row.get("alignments")
        expected_arms = {"candidate", *_CONTROL_ORDER}
        if not isinstance(alignments, dict) or set(alignments) != expected_arms:
            raise ValueError("identity row alignment arms differ from the frozen set")
        by_label[label][row_seed] = row
    required_seeds = {0, 1, 2, 3}
    complete_labels = sorted(
        label for label, seed_rows in by_label.items() if set(seed_rows) == required_seeds
    )
    if not complete_labels:
        raise ValueError("no identity is present in all four seeds")
    incomplete_count = len(by_label) - len(complete_labels)
    arms = ("candidate", *_CONTROL_ORDER)
    values = np.asarray(
        [
            [
                [float(by_label[label][row_seed]["alignments"][arm]) for arm in arms]
                for row_seed in range(4)
            ]
            for label in complete_labels
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("verdict received nonfinite alignments")
    pooled_means = values.mean(axis=(0, 1))
    control_means = pooled_means[1:]
    strongest_index = int(np.argmax(control_means))
    strongest_control = _CONTROL_ORDER[strongest_index]
    strongest_arm_index = strongest_index + 1
    candidate_delta = float(pooled_means[0] - pooled_means[strongest_arm_index])
    seed_deltas = {
        str(row_seed): float(
            values[:, row_seed, 0].mean() - values[:, row_seed, strongest_arm_index].mean()
        )
        for row_seed in range(4)
    }

    rng = np.random.default_rng(int(seed))
    bootstrap_deltas = np.empty(replicates, dtype=np.float64)
    identity_count = len(complete_labels)
    for replicate in range(replicates):
        sampled = rng.integers(0, identity_count, size=identity_count)
        sampled_means = values[sampled].mean(axis=(0, 1))
        bootstrap_deltas[replicate] = sampled_means[0] - float(np.max(sampled_means[1:]))
    lower_bound = float(np.percentile(bootstrap_deltas, 2.5))

    complete_rows = [
        by_label[label][row_seed] for label in complete_labels for row_seed in range(4)
    ]
    median_orthogonal = float(
        np.median([float(row["orthogonal_fraction"]) for row in complete_rows])
    )
    seed_ge = sum(delta >= 0.02 for delta in seed_deltas.values())
    seed_nonpositive = sum(delta <= 0.0 for delta in seed_deltas.values())
    projection_delta = float(pooled_means[0] - pooled_means[arms.index("ambient_projection")])
    criteria = {
        "noncollapse_ge_0_20": median_orthogonal >= 0.20,
        "candidate_delta_ge_0_03": candidate_delta >= 0.03,
        "bootstrap_lower_bound_positive": lower_bound > 0.0,
        "at_least_three_seed_deltas_ge_0_02": seed_ge >= 3,
        "beats_ambient_projection": projection_delta > 0.0,
    }
    fail_reasons: list[str] = []
    if median_orthogonal < 0.10:
        fail_reasons.append("median orthogonal fraction is below 0.10")
    if candidate_delta <= 0.0:
        fail_reasons.append("pooled candidate delta is nonpositive")
    if seed_nonpositive >= 3:
        fail_reasons.append("at least three seed deltas are nonpositive")
    if fail_reasons:
        stage_a = "FAIL"
    elif all(criteria.values()):
        stage_a = "PASS_ONWARD"
    else:
        stage_a = "UNRESOLVED"
    return {
        "stage_a": stage_a,
        "reasons": fail_reasons,
        "criteria": criteria,
        "complete_identity_count": identity_count,
        "incomplete_identity_count": incomplete_count,
        "pooled_means": {arm: float(pooled_means[index]) for index, arm in enumerate(arms)},
        "strongest_control": strongest_control,
        "candidate_delta": candidate_delta,
        "projection_delta": projection_delta,
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(replicates),
        "bootstrap_95_lower_bound": lower_bound,
        "seed_deltas": seed_deltas,
        "seed_deltas_ge_0_02": int(seed_ge),
        "seed_deltas_nonpositive": int(seed_nonpositive),
        "median_orthogonal_fraction": median_orthogonal,
        "beats_projection": projection_delta > 0.0,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=1_000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or set(manifest.get("seeds", {})) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise ValueError("Pass159 manifest must contain schema 1 and exactly seeds 0-3")
    bound_seeds: list[BoundSeed] = []
    seed_results: list[dict[str, Any]] = []
    for seed in range(4):
        bound = load_bound_seed(manifest["seeds"][str(seed)], seed=seed)
        if bound_seeds and not np.array_equal(
            bound.train_example_ids, bound_seeds[0].train_example_ids
        ):
            raise ValueError("training example-ID order differs across seeds")
        bound_seeds.append(bound)
        seed_results.append(compute_seed_rows(bound, top_k=args.top_k))
    identity_rows = [
        row for seed_result in seed_results for row in seed_result["identity_rows"]
    ]
    verdict = clustered_verdict(
        identity_rows,
        seed=159,
        replicates=args.bootstrap_replicates,
    )
    per_seed = []
    for bound, result in zip(bound_seeds, seed_results, strict=True):
        rows = result["identity_rows"]
        arm_names = ("candidate", *_CONTROL_ORDER)
        arm_means = {
            arm: float(np.mean([row["alignments"][arm] for row in rows]))
            for arm in arm_names
        }
        per_seed.append(
            {
                **{key: value for key, value in result.items() if key != "identity_rows"},
                "official_recall_at_1": bound.official_recall_at_1,
                "artifact_binding": bound.artifact_binding,
                "alignment_means": arm_means,
            }
        )
    payload = {
        "schema_version": 1,
        "preregistration": {
            "document": "docs/pass159_gradient_transplant_search_2026-08-08.md",
            "manifest": str(args.manifest),
            "tau": 0.05,
            "top_k": int(args.top_k),
            "bootstrap_seed": 159,
            "bootstrap_replicates": int(args.bootstrap_replicates),
            "uses_test_data": "artifact_binding_only",
            "stage_a_cannot_clear_gate_1": True,
        },
        "per_seed": per_seed,
        "identity_rows": identity_rows,
        "verdict": verdict,
    }
    write_json_atomic(args.output, payload)
    print(json.dumps(verdict, sort_keys=True))


if __name__ == "__main__":
    main()
