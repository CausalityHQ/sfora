#!/usr/bin/env python3
"""Run the prospective, training-only OAPF provenance diagnostic.

This script deliberately has no retrieval-evaluation inputs.  It parses only
In-Shop ``train`` rows, freezes the supplied epoch-10 model, and never loads a
query or gallery image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr  # type: ignore[import-untyped]

from sfora.image_end_to_end import ImageEndToEndConfig, _torchvision_model_factory
from sfora.oapf import (
    class_balanced_pair_weights,
    deterministic_view_seed,
    fit_standardizer,
    fit_weighted_logistic,
    fixed_class_folds,
    logistic_probabilities,
    orbit_radius_and_rms,
    same_class_pairs,
    weighted_auc,
    within_class_derangement,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
REGISTERED_CHECKPOINT_SHA256 = "31307c9e0ce816397e3d3b3ff3f0084dc84b3ef47e8e9847ecbc71fa3b97fcbd"
REGISTERED_REPORT_SHA256 = "e84aa1b7a0e3ee052b5bd4ce13a6a8e77396cb4f4738797a83853c4f4ded92cc"


@dataclass(frozen=True)
class TrainRecord:
    relative_path: str
    item_id: str
    label: int
    path: Path


def load_inshop_train_only(dataset_root: Path) -> list[TrainRecord]:
    """Return train rows with the official loader's global identity labels.

    Evaluation images are never opened. Their item IDs must be read from the
    official partition manifest solely because the training code defines labels
    by sorting identities across that manifest.
    """

    partition = dataset_root / "Eval" / "list_eval_partition.txt"
    lines = partition.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"invalid In-Shop partition: {partition}")
    try:
        declared_count = int(lines[0])
    except ValueError as error:
        raise ValueError("In-Shop partition first line must be an image count") from error
    all_rows: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(lines[2:], start=3):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"invalid In-Shop partition row {line_number}: {line!r}")
        relative_path, item_id, split = fields
        if split not in {"train", "query", "gallery"}:
            raise ValueError(f"invalid In-Shop split {split!r} on row {line_number}")
        all_rows.append((relative_path, item_id, split))
    if len(all_rows) != declared_count:
        raise ValueError(
            f"In-Shop partition declares {declared_count} rows but contains {len(all_rows)}"
        )
    labels = {item_id: index for index, item_id in enumerate(sorted({row[1] for row in all_rows}))}
    train_rows = [(path, item) for path, item, split in all_rows if split == "train"]
    records = [
        TrainRecord(
            relative_path=relative_path,
            item_id=item_id,
            label=labels[item_id],
            path=dataset_root / "Img" / relative_path,
        )
        for relative_path, item_id in train_rows
    ]
    if not records or any(not record.path.is_file() for record in records):
        raise ValueError("In-Shop training split is empty or references missing images")
    return records


def pair_features(
    canonical: FloatArray,
    raw_norm: FloatArray,
    local_density: FloatArray,
    proxy_margin: FloatArray,
    rms: FloatArray,
    mean_canonical_displacement: FloatArray,
    centroid_canonical_displacement: FloatArray,
    radius: FloatArray,
    left: IntArray,
    right: IntArray,
) -> tuple[FloatArray, FloatArray]:
    """Build exactly the registered M0 controls and two endpoint-radius features."""

    distance = np.linalg.norm(canonical[left] - canonical[right], axis=1)
    controls = [distance]
    for values in (
        raw_norm,
        np.log(np.maximum(local_density, 1.0e-12)),
        proxy_margin,
        rms,
        mean_canonical_displacement,
        centroid_canonical_displacement,
    ):
        controls.extend(
            [np.minimum(values[left], values[right]), np.maximum(values[left], values[right])]
        )
    radii = np.log(np.maximum(radius, 1.0e-6))
    radius_features = np.column_stack(
        [np.minimum(radii[left], radii[right]), np.maximum(radii[left], radii[right])]
    )
    return np.column_stack(controls), radius_features


def weighted_quantiles(
    values: FloatArray, weights: FloatArray, probabilities: FloatArray
) -> FloatArray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    positions = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return np.interp(probabilities, positions, sorted_values)


def weighted_hedges_g(
    high: FloatArray, high_w: FloatArray, low: FloatArray, low_w: FloatArray
) -> float:
    """Class-weighted Hedges g using weighted moments and effective sample sizes."""

    def moments(values: FloatArray, weights: FloatArray) -> tuple[float, float, float]:
        normalized = weights / weights.sum()
        mean = float(np.sum(normalized * values))
        effective_n = float(weights.sum() ** 2 / np.sum(weights**2))
        population_variance = float(np.sum(normalized * (values - mean) ** 2))
        variance = population_variance * effective_n / max(effective_n - 1.0, 1.0)
        return mean, variance, effective_n

    high_mean, high_var, high_n = moments(high, high_w)
    low_mean, low_var, low_n = moments(low, low_w)
    denominator = max(high_n + low_n - 2.0, 1.0)
    pooled = np.sqrt(max(((high_n - 1) * high_var + (low_n - 1) * low_var) / denominator, 1e-12))
    correction = 1.0 - 3.0 / max(4.0 * (high_n + low_n) - 9.0, 1.0)
    return float(correction * (high_mean - low_mean) / pooled)


def residualize_by_class(values: FloatArray, labels: IntArray) -> FloatArray:
    result = values.copy()
    for label in np.unique(labels):
        mask = labels == label
        result[mask] -= result[mask].mean()
    return result


def _linear_residuals(
    train_x: FloatArray,
    train_y: FloatArray,
    train_w: FloatArray,
    test_x: FloatArray,
    test_y: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    standardizer = fit_standardizer(train_x, train_w)
    train_design = np.column_stack([np.ones(train_x.shape[0]), standardizer.apply(train_x)])
    test_design = np.column_stack([np.ones(test_x.shape[0]), standardizer.apply(test_x)])
    root_w = np.sqrt(train_w / train_w.sum())
    coefficients = np.linalg.lstsq(train_design * root_w[:, None], train_y * root_w, rcond=None)[0]
    return (
        np.asarray(train_y - train_design @ coefficients, dtype=np.float64),
        np.asarray(test_y - test_design @ coefficients, dtype=np.float64),
    )


def evaluate_models(
    controls: FloatArray,
    radius_features: FloatArray,
    continuous: FloatArray,
    targets: IntArray,
    pair_labels: IntArray,
    weights: FloatArray,
    endpoint_radius: FloatArray,
    endpoint_labels: IntArray,
    left: IntArray,
    right: IntArray,
) -> dict[str, Any]:
    """Run all preregistered folds, direction, permutation, and decile tests."""

    fold_by_label = fixed_class_folds(endpoint_labels)
    pair_folds = np.asarray([fold_by_label[int(label)] for label in pair_labels], dtype=np.int64)
    fold_rows: list[dict[str, float]] = []
    compatibility_residual = np.empty(targets.size, dtype=np.float64)
    crossfit_q = np.empty(targets.size, dtype=np.float64)
    crossfit_bins = np.empty(targets.size, dtype=np.int64)
    permutation_auc = np.empty((100, 5), dtype=np.float64)

    for fold in range(5):
        train = pair_folds != fold
        test = ~train
        standardizer = fit_standardizer(controls[train], weights[train])
        train_m0 = standardizer.apply(controls[train])
        test_m0 = standardizer.apply(controls[test])
        beta0 = fit_weighted_logistic(train_m0, targets[train], weights[train])
        p0 = logistic_probabilities(test_m0, beta0)
        auc0 = weighted_auc(targets[test], p0, weights[test])
        _, compatibility_residual[test] = _linear_residuals(
            controls[train],
            continuous[train],
            weights[train],
            controls[test],
            continuous[test],
        )

        combined = np.column_stack([controls, radius_features])
        full_standardizer = fit_standardizer(combined[train], weights[train])
        train_full = full_standardizer.apply(combined[train])
        test_full = full_standardizer.apply(combined[test])
        beta1 = fit_weighted_logistic(train_full, targets[train], weights[train])
        auc1 = weighted_auc(targets[test], logistic_probabilities(test_full, beta1), weights[test])

        real_beta = fit_weighted_logistic(
            train_full, targets[train], weights[train], nonnegative_from=controls.shape[1]
        )
        real_auc = weighted_auc(
            targets[test], logistic_probabilities(test_full, real_beta), weights[test]
        )
        inverse = np.column_stack([controls, -radius_features])
        inverse_standardizer = fit_standardizer(inverse[train], weights[train])
        inverse_train = inverse_standardizer.apply(inverse[train])
        inverse_test = inverse_standardizer.apply(inverse[test])
        inverse_beta = fit_weighted_logistic(
            inverse_train, targets[train], weights[train], nonnegative_from=controls.shape[1]
        )
        inverse_auc = weighted_auc(
            targets[test], logistic_probabilities(inverse_test, inverse_beta), weights[test]
        )
        fold_rows.append(
            {
                "fold": float(fold),
                "auc_m0": auc0,
                "auc_m1": auc1,
                "auc_real_direction": real_auc,
                "auc_inverse_direction": inverse_auc,
            }
        )

        mean_log_radius = radius_features.mean(axis=1)
        train_q, test_q = _linear_residuals(
            controls[train],
            mean_log_radius[train],
            weights[train],
            controls[test],
            mean_log_radius[test],
        )
        train_q_median = weighted_quantiles(
            train_q, weights[train], np.asarray([0.5], dtype=np.float64)
        )[0]
        crossfit_q[test] = test_q - train_q_median
        distance_edges = weighted_quantiles(
            controls[train, 0], weights[train], np.linspace(0.1, 0.9, 9)
        )
        crossfit_bins[test] = np.searchsorted(distance_edges, controls[test, 0], side="right")

        for permutation_index, seed in enumerate(range(174000, 174100)):
            permuted = within_class_derangement(endpoint_radius, endpoint_labels, seed)
            _, permuted_radius = pair_features(
                np.zeros((endpoint_radius.size, 1)),
                np.ones(endpoint_radius.size),
                np.ones(endpoint_radius.size),
                np.zeros(endpoint_radius.size),
                np.ones(endpoint_radius.size),
                np.ones(endpoint_radius.size),
                np.ones(endpoint_radius.size),
                permuted,
                left,
                right,
            )
            permuted_combined = np.column_stack([controls, permuted_radius])
            perm_standardizer = fit_standardizer(permuted_combined[train], weights[train])
            perm_train = perm_standardizer.apply(permuted_combined[train])
            perm_test = perm_standardizer.apply(permuted_combined[test])
            perm_beta = fit_weighted_logistic(perm_train, targets[train], weights[train])
            permutation_auc[permutation_index, fold] = weighted_auc(
                targets[test], logistic_probabilities(perm_test, perm_beta), weights[test]
            )

    decile_g: list[float] = []
    for decile in range(10):
        mask = crossfit_bins == decile
        # Each held-out pair uses the median q from its corresponding training folds.
        high = mask & (crossfit_q >= 0.0)
        low = mask & (crossfit_q < 0.0)
        high_weights = (
            class_balanced_pair_weights(pair_labels[high])
            if high.any()
            else np.empty(0, dtype=np.float64)
        )
        low_weights = (
            class_balanced_pair_weights(pair_labels[low])
            if low.any()
            else np.empty(0, dtype=np.float64)
        )
        decile_g.append(
            weighted_hedges_g(
                compatibility_residual[high],
                high_weights,
                compatibility_residual[low],
                low_weights,
            )
            if high.any() and low.any()
            else float("nan")
        )

    auc_m0 = np.asarray([row["auc_m0"] for row in fold_rows])
    auc_m1 = np.asarray([row["auc_m1"] for row in fold_rows])
    real_auc_array = np.asarray([row["auc_real_direction"] for row in fold_rows])
    inverse_auc_array = np.asarray([row["auc_inverse_direction"] for row in fold_rows])
    permutation_mean_by_fold = permutation_auc.mean(axis=0)
    serializable_decile_g = [value if np.isfinite(value) else None for value in decile_g]
    return {
        "folds": fold_rows,
        "macro_auc_m1_minus_m0": float(np.mean(auc_m1 - auc_m0)),
        "direction_folds_real_better": int(np.sum(real_auc_array > inverse_auc_array)),
        "direction_macro_auc_advantage": float(np.mean(real_auc_array - inverse_auc_array)),
        "permutation_folds_real_better_than_mean": int(np.sum(auc_m1 > permutation_mean_by_fold)),
        "permutation_macro_auc_advantage": float(np.mean(auc_m1 - permutation_mean_by_fold)),
        "permutation_auc_mean_by_fold": permutation_mean_by_fold.tolist(),
        "distance_decile_hedges_g": serializable_decile_g,
        "distance_deciles_passing": int(np.sum(np.asarray(decile_g) >= 0.20)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(path: Path, expected: str, artifact: str) -> str:
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"{artifact} SHA-256 mismatch: {actual} != {expected.lower()}")
    return actual


def report_has_objective(report: dict[str, Any], objective: str) -> bool:
    """Match an objective by payload value, not its model-qualified method key."""

    methods = report.get("methods", {})
    return isinstance(methods, dict) and any(
        isinstance(method, dict) and method.get("objective") == objective
        for method in methods.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=REGISTERED_CHECKPOINT_SHA256)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--training-report-sha256", default=REGISTERED_REPORT_SHA256)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view-metadata-output", type=Path)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--distance-chunk-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate frozen provenance/model/split and exit before CUDA or image encoding",
    )
    args = parser.parse_args()

    metadata_path = args.view_metadata_output or args.output.with_suffix(".views.json")
    artifact_paths = [args.checkpoint, args.training_report, args.output, metadata_path]
    if args.embedding_cache is not None:
        artifact_paths.append(args.embedding_cache)
    resolved_artifacts = [path.expanduser().resolve() for path in artifact_paths]
    if len(set(resolved_artifacts)) != len(resolved_artifacts):
        raise ValueError("checkpoint, report, output, metadata, and cache paths must be distinct")

    import torch
    import torch.nn.functional as torch_f
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import InterpolationMode, RandomResizedCrop
    from torchvision.transforms import functional as tv_f

    if args.checkpoint_sha256.lower() != REGISTERED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint override does not match the prospectively registered digest")
    if args.training_report_sha256.lower() != REGISTERED_REPORT_SHA256:
        raise ValueError("report override does not match the prospectively registered digest")
    actual_checkpoint_sha256 = require_sha256(
        args.checkpoint, REGISTERED_CHECKPOINT_SHA256, "checkpoint"
    )
    actual_report_sha256 = require_sha256(
        args.training_report, REGISTERED_REPORT_SHA256, "training-report"
    )
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    if config.dataset_name != "inshop" or config.seed != 0:
        raise ValueError("OAPF diagnostic requires official In-Shop seed 0")
    if (
        config.dataset_selection_policy != "full_official_partition"
        or config.limit_per_class is not None
        or config.max_classes is not None
    ):
        raise ValueError("OAPF diagnostic requires the unabridged official training partition")
    if "proxy_anchor" not in config.objectives or not report_has_objective(
        report, "proxy_anchor"
    ):
        raise ValueError("training report is not an official Proxy Anchor run")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    for field, value in checkpoint.get("arch", {}).items():
        if hasattr(config, field) and getattr(config, field) != value:
            raise ValueError(
                f"checkpoint architecture field {field!r} does not match training report"
            )
    state = checkpoint["state_dict"]
    proxy_tensor = state.get("metric_proxies")
    proxy_label_tensor = state.get("metric_proxy_labels")
    if proxy_tensor is None or proxy_label_tensor is None:
        raise ValueError("checkpoint lacks Proxy Anchor proxies")
    records = load_inshop_train_only(args.dataset_root)
    unique_proxy_labels = sorted({int(value) for value in proxy_label_tensor.tolist()})
    expected_labels = sorted({record.label for record in records})
    if expected_labels != unique_proxy_labels:
        raise ValueError("official training identity labels do not match checkpoint proxy labels")
    if int(report.get("train_examples", -1)) != len(records):
        raise ValueError("training report example count does not match official training partition")

    model: Any = _torchvision_model_factory(config.model_copy(update=checkpoint["arch"]))
    model.load_state_dict(
        {
            key: value
            for key, value in state.items()
            if key not in {"metric_proxies", "metric_proxy_labels"}
        },
        strict=True,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "checkpoint_sha256": actual_checkpoint_sha256,
                    "training_report_sha256": actual_report_sha256,
                    "dataset": "inshop.train",
                    "examples": len(records),
                    "classes": len(expected_labels),
                    "query_gallery_images_loaded": False,
                },
                indent=2,
            )
        )
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    metadata: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}

    def sampled_view(
        image: Any, record: TrainRecord, pack: str, view_index: int
    ) -> tuple[Any, dict[str, Any]]:
        seed = deterministic_view_seed(pack, record.relative_path, view_index)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            top, left, height, width = RandomResizedCrop.get_params(
                image, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            flip = bool(torch.rand(()) < 0.5)
        transformed = tv_f.resized_crop(
            image, top, left, height, width, [224, 224], InterpolationMode.BILINEAR
        )
        if flip:
            transformed = tv_f.hflip(transformed)
        return transformed, {
            "relative_path": record.relative_path,
            "view_index": view_index,
            "seed": seed,
            "crop": [top, left, height, width],
            "flip": flip,
        }

    class ViewDataset(Dataset[Any]):
        def __init__(self, pack: str | None, view_index: int = 0) -> None:
            self.pack = pack
            self.view_index = view_index

        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            record = records[index]
            with Image.open(record.path) as opened:
                image = opened.convert("RGB")
            if self.pack is None:
                image = tv_f.resize(image, 256, interpolation=InterpolationMode.BILINEAR)
                image = tv_f.center_crop(image, [224, 224])
            else:
                image, _ = sampled_view(image, record, self.pack, self.view_index)
            channels = [image.getchannel(channel) for channel in range(3)]
            bgr = Image.merge("RGB", list(reversed(channels)))
            tensor = tv_f.pil_to_tensor(bgr).float()
            tensor = tv_f.normalize(tensor, (104.0, 117.0, 128.0), (1.0, 1.0, 1.0))
            return tensor, record.label

    raw_batches: list[Any] = []
    embedding_layer = getattr(getattr(model, "model", None), "embedding", None)
    if embedding_layer is None:
        raise ValueError("OAPF operating checkpoint must use BN-Inception")
    hook = embedding_layer.register_forward_hook(
        lambda _module, _inputs, output: raw_batches.append(output.detach().cpu())
    )

    def encode(pack: str | None, view_index: int = 0) -> tuple[FloatArray, IntArray, FloatArray]:
        raw_batches.clear()
        loader = DataLoader(
            ViewDataset(pack, view_index),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        outputs: list[Any] = []
        labels: list[Any] = []
        with torch.no_grad():
            for images, batch_labels in loader:
                outputs.append(model(images.to(device, non_blocking=True)).detach().cpu())
                labels.append(batch_labels)
        normalized = torch.cat(outputs).numpy().astype(np.float64)
        raw = torch.cat(raw_batches).numpy().astype(np.float64)
        return normalized, torch.cat(labels).numpy().astype(np.int64), np.linalg.norm(raw, axis=1)

    canonical, labels, raw_norm = encode(None)
    pack_a = np.stack([encode("A", index)[0] for index in range(6)], axis=1)
    pack_b = np.stack([encode("B", index)[0] for index in range(6)], axis=1)
    hook.remove()
    # Worker processes cannot mutate the parent's metadata. Recreate the same
    # pure pinned samples in the parent, then persist every crop and flip.
    for pack in ("A", "B"):
        for view_index in range(6):
            for record in records:
                with Image.open(record.path) as opened:
                    # Crop sampling uses dimensions only; avoid decoding 12 extra
                    # copies of every image solely to recreate metadata.
                    image = Image.new("RGB", opened.size)
                _, specification = sampled_view(image, record, pack, view_index)
                metadata[pack].append(specification)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")
    if args.embedding_cache is not None:
        args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.embedding_cache,
            canonical=canonical,
            labels=labels,
            raw_norm=raw_norm,
            pack_a=pack_a,
            pack_b=pack_b,
        )

    radius_a, rms_a = orbit_radius_and_rms(canonical, pack_a)
    radius_b, _ = orbit_radius_and_rms(canonical, pack_b)
    pack_a_displacements = np.linalg.norm(pack_a - canonical[:, None, :], axis=2)
    mean_canonical_displacement = pack_a_displacements.mean(axis=1)
    centroid_canonical_displacement = np.linalg.norm(pack_a.mean(axis=1) - canonical, axis=1)

    canonical_gpu = torch.as_tensor(canonical, dtype=torch.float32, device=device)
    labels_gpu = torch.as_tensor(labels, dtype=torch.long, device=device)
    local_density = np.empty(labels.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, labels.size, args.distance_chunk_size):
            stop = min(start + args.distance_chunk_size, labels.size)
            distances = torch.cdist(canonical_gpu[start:stop], canonical_gpu)
            rows = torch.arange(stop - start, device=device)
            distances[rows, torch.arange(start, stop, device=device)] = torch.inf
            local_density[start:stop] = (
                distances.topk(20, largest=False).values.mean(dim=1).cpu().numpy()
            )

    proxies = torch_f.normalize(proxy_tensor.to(device), p=2, dim=1)
    proxy_labels = proxy_label_tensor.to(device)
    proxy_margin = np.empty(labels.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, labels.size, args.distance_chunk_size):
            stop = min(start + args.distance_chunk_size, labels.size)
            similarities = canonical_gpu[start:stop] @ proxies.T
            batch_labels = labels_gpu[start:stop]
            own = (
                (
                    similarities.masked_fill(
                        proxy_labels[None, :] != batch_labels[:, None], -torch.inf
                    )
                )
                .max(dim=1)
                .values
            )
            rival = (
                (
                    similarities.masked_fill(
                        proxy_labels[None, :] == batch_labels[:, None], -torch.inf
                    )
                )
                .max(dim=1)
                .values
            )
            proxy_margin[start:stop] = (own - rival).cpu().numpy()

    nearest_negative_gpu = torch.empty((labels.size, 6), dtype=torch.float32, device=device)
    with torch.no_grad():
        for view_index in range(6):
            views_gpu = torch.as_tensor(pack_b[:, view_index], dtype=torch.float32, device=device)
            for start in range(0, labels.size, args.distance_chunk_size):
                stop = min(start + args.distance_chunk_size, labels.size)
                distances = torch.cdist(views_gpu[start:stop], canonical_gpu)
                distances.masked_fill_(
                    labels_gpu[None, :] == labels_gpu[start:stop, None], torch.inf
                )
                nearest_negative_gpu[start:stop, view_index] = distances.min(dim=1).values

    left, right, pair_labels = same_class_pairs(labels)
    success_count = np.empty(left.size, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, left.size, 4096):
            stop = min(start + 4096, left.size)
            chunk_left = torch.as_tensor(left[start:stop], dtype=torch.long, device=device)
            chunk_right = torch.as_tensor(right[start:stop], dtype=torch.long, device=device)
            left_views = torch.as_tensor(
                pack_b[left[start:stop]], dtype=torch.float32, device=device
            )
            right_views = torch.as_tensor(
                pack_b[right[start:stop]], dtype=torch.float32, device=device
            )
            forward = torch.linalg.vector_norm(
                left_views - canonical_gpu[chunk_right, None, :], dim=2
            )
            reverse = torch.linalg.vector_norm(
                right_views - canonical_gpu[chunk_left, None, :], dim=2
            )
            successes = (nearest_negative_gpu[chunk_left] - forward > 0.0).sum(dim=1)
            successes += (nearest_negative_gpu[chunk_right] - reverse > 0.0).sum(dim=1)
            success_count[start:stop] = successes.cpu().numpy()
    continuous = success_count.astype(np.float64) / 12.0
    targets = (success_count >= 10).astype(np.int64)
    weights = class_balanced_pair_weights(pair_labels)
    controls, radius_features = pair_features(
        canonical,
        raw_norm,
        local_density,
        proxy_margin,
        rms_a,
        mean_canonical_displacement,
        centroid_canonical_displacement,
        radius_a,
        left,
        right,
    )
    global_spearman = float(spearmanr(radius_a, radius_b).statistic)
    residual_spearman = float(
        spearmanr(
            residualize_by_class(radius_a, labels), residualize_by_class(radius_b, labels)
        ).statistic
    )
    prevalence = float(np.sum(weights * targets) / np.sum(weights))
    reliability_ok = bool(
        np.isfinite(global_spearman)
        and np.isfinite(residual_spearman)
        and global_spearman >= 0.50
        and residual_spearman >= 0.50
    )
    prevalence_ok = 0.10 <= prevalence <= 0.90
    analysis: dict[str, Any] = {}
    analysis_error: str | None = None
    if reliability_ok and prevalence_ok:
        try:
            analysis = evaluate_models(
                controls,
                radius_features,
                continuous,
                targets,
                pair_labels,
                weights,
                radius_a,
                labels,
                left,
                right,
            )
        except (ValueError, RuntimeError, FloatingPointError) as error:
            analysis_error = f"registered analysis undefined: {error}"
    else:
        failed = []
        if not prevalence_ok:
            failed.append("binary prevalence outside [0.10, 0.90]")
        if not reliability_ok:
            failed.append("radius reliability below 0.50 or undefined")
        analysis_error = "; ".join(failed)
    gates = {
        "binary_prevalence": prevalence_ok,
        "global_spearman": bool(np.isfinite(global_spearman) and global_spearman >= 0.50),
        "within_class_residual_spearman": bool(
            np.isfinite(residual_spearman) and residual_spearman >= 0.50
        ),
        "incremental_auc": bool(analysis.get("macro_auc_m1_minus_m0", -np.inf) >= 0.03),
        "direction": bool(
            analysis.get("direction_folds_real_better", -1) >= 4
            and analysis.get("direction_macro_auc_advantage", -np.inf) >= 0.03
        ),
        "permutation": bool(
            analysis.get("permutation_folds_real_better_than_mean", -1) >= 4
            and analysis.get("permutation_macro_auc_advantage", -np.inf) >= 0.03
        ),
        "distance_deciles": bool(analysis.get("distance_deciles_passing", -1) >= 7),
    }
    payload = {
        "candidate": "OAPF",
        "candidate_number": 174,
        "dataset": "inshop.train",
        "training_only": True,
        "query_gallery_used": False,
        "evaluation_images_loaded": False,
        "evaluation_manifest_ids_used_only_for_official_label_map": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "training_report": str(args.training_report),
        "training_report_sha256": actual_report_sha256,
        "view_metadata": str(metadata_path),
        "view_metadata_sha256": _sha256_file(metadata_path),
        "examples": int(labels.size),
        "same_class_pairs": int(targets.size),
        "weighted_binary_prevalence": prevalence,
        "radius_global_spearman": global_spearman if np.isfinite(global_spearman) else None,
        "radius_within_class_residual_spearman": (
            residual_spearman if np.isfinite(residual_spearman) else None
        ),
        "analysis_error": analysis_error,
        **analysis,
        "gates": gates,
        "passes_gate_1": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
