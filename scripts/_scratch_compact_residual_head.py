#!/usr/bin/env python3
"""Fail-fast nonlinear compact-head experiment on authenticated feature archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import (
    CompactMetricConfig,
    _blockwise_hard_negatives,
    _positive_rows,
    _score_compact_metric_codes,
)
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import fit_centered_pca
from sfora.teacher_anchored_distillation import (
    class_balanced_anchor_schedule,
    positive_coverage_hard_negative_loss,
)


class ResidualCompactHead(torch.nn.Module):
    """PCA affine head plus a zero-initialized, low-rank nonlinear residual."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor,
        hidden_dimensions: int,
        seed: int,
    ) -> None:
        super().__init__()
        if (
            weight.ndim != 2
            or bias.shape != (weight.shape[0],)
            or hidden_dimensions < 1
            or seed < 0
        ):
            raise ValueError("residual compact head authority differs")
        self.primary = torch.nn.Linear(weight.shape[1], weight.shape[0])
        self.residual_input = torch.nn.Linear(weight.shape[1], hidden_dimensions, bias=False)
        self.residual_output = torch.nn.Linear(hidden_dimensions, weight.shape[0], bias=False)
        with torch.no_grad():
            self.primary.weight.copy_(weight)
            self.primary.bias.copy_(bias)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.nn.init.kaiming_uniform_(
                self.residual_input.weight,
                a=math.sqrt(5.0),
                generator=generator,
            )
            self.residual_output.weight.zero_()

    @property
    def residual_parameter_count(self) -> int:
        """Return the exact nonlinear-branch parameter count."""

        return self.residual_input.weight.numel() + self.residual_output.weight.numel()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        residual = self.residual_output(torch.nn.functional.gelu(self.residual_input(embeddings)))
        return torch.nn.functional.normalize(self.primary(embeddings) + residual, dim=1)


def geometry_anchor_loss(current: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    """Penalize angular drift from the frozen PCA representation."""

    if current.shape != initial.shape or current.ndim != 2:
        raise ValueError("geometry anchor authority differs")
    return (1.0 - torch.einsum("bd,bd->b", current, initial.detach())).mean()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initial_projection(normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pca = fit_centered_pca(normalized, dimensions=128)
    weight = pca.components.float().contiguous()
    bias = (-(pca.components.double() @ pca.mean.double())).float().contiguous()
    return weight, bias


def _linear_head(weight: torch.Tensor, bias: torch.Tensor) -> torch.nn.Linear:
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    with torch.no_grad():
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    return head


def _head_output(head: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    value = head(rows)
    if isinstance(head, ResidualCompactHead):
        return value
    return torch.nn.functional.normalize(value, dim=1)


def train_arm(
    *,
    normalized: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    nonlinear: bool,
    anchor_weight: float,
    config: CompactMetricConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, tuple[float, ...], str]:
    label_array = labels.numpy()
    groups = {
        int(label): np.flatnonzero(label_array == int(label)).astype(np.int64)
        for label in np.unique(label_array)
    }
    eligible = {
        label: rows for label, rows in groups.items() if len(rows) >= config.rows_per_class
    }
    classes_per_update = min(config.classes_per_update, len(eligible))
    eligible_rows = sum(len(rows) for rows in eligible.values())
    updates_per_cycle = math.ceil(
        config.anchor_epochs_per_cycle
        * eligible_rows
        / (classes_per_update * config.rows_per_class)
    )
    schedule = class_balanced_anchor_schedule(
        label_array,
        seed=config.seed,
        updates=updates_per_cycle,
        classes_per_update=classes_per_update,
        rows_per_class=config.rows_per_class,
    )
    head: torch.nn.Module
    if nonlinear:
        head = ResidualCompactHead(
            weight=weight,
            bias=bias,
            hidden_dimensions=32,
            seed=config.seed,
        )
    else:
        head = _linear_head(weight, bias)
    head = head.to(device)
    bank = normalized.to(device)
    label_device = labels.to(device)
    with torch.inference_mode():
        start = _head_output(head, bank).detach().contiguous()
    maximum_group = max(len(groups[label]) for label in eligible)
    hard_negatives = min(config.hard_negatives, len(labels) - maximum_group)
    frozen_negatives = _blockwise_hard_negatives(
        start,
        label_device,
        k=hard_negatives,
        block_size=config.mining_block_size,
    )
    optimizer = torch.optim.Adam(
        head.parameters(),
        lr=config.learning_rate * math.sqrt(config.output_dimensions / normalized.shape[1]),
    )
    losses: list[float] = []
    scheduled_rows = tuple(row.copy() for row in schedule.row_indexes)
    for _cycle in range(config.cycles):
        for anchor_numpy in scheduled_rows:
            anchors = torch.from_numpy(anchor_numpy).to(device)
            positive_rows, positive_mask = _positive_rows(anchor_numpy, label_array, eligible)
            positives = torch.from_numpy(positive_rows).to(device)
            mask = torch.from_numpy(positive_mask).to(device).contiguous()
            negatives = frozen_negatives[anchors]
            anchor_codes = _head_output(head, bank[anchors])
            positive_codes = _head_output(head, bank[positives.reshape(-1)]).reshape(
                len(anchors), positives.shape[1], -1
            )
            negative_codes = _head_output(head, bank[negatives.reshape(-1)]).reshape(
                len(anchors), negatives.shape[1], -1
            )
            loss = positive_coverage_hard_negative_loss(
                torch.einsum("bd,bpd->bp", anchor_codes, positive_codes).contiguous(),
                mask,
                torch.einsum("bd,bnd->bn", anchor_codes, negative_codes).contiguous(),
                torch.einsum("bd,bd->b", anchor_codes, start[anchors]).contiguous(),
                temperature=config.temperature,
                margin=config.margin,
                anchor_weight=anchor_weight,
                positive_aggregation="mean_logit",
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
    return head, tuple(losses), schedule.sha256


def score_arm(
    head: torch.nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, object]:
    with torch.inference_mode():
        normalized = torch.nn.functional.normalize(embeddings.to(device), dim=1)
        projected = _head_output(head, normalized)
        codes = torch.round(projected * 127.0).clamp(-127, 127).to(torch.int8).cpu()
    map_at_r, recall, average_precision, recall_rows = _score_compact_metric_codes(
        codes, labels, device=device
    )
    return {
        "map_at_r": map_at_r,
        "recall_at_1": recall,
        "average_precision": average_precision,
        "recall_rows": recall_rows,
    }


def paired_class_bootstrap(
    candidate: tuple[float, ...],
    control: tuple[float, ...],
    labels: np.ndarray,
) -> dict[str, float]:
    classes = np.unique(labels)
    candidate_values = np.asarray(candidate)
    control_values = np.asarray(control)
    differences = np.asarray(
        [
            candidate_values[labels == value].mean()
            - control_values[labels == value].mean()
            for value in classes
        ],
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.PCG64(17))
    samples = differences[
        generator.integers(0, len(differences), size=(10_000, len(differences)))
    ].mean(axis=1)
    lower, median, upper = np.quantile(samples, (0.025, 0.5, 0.975))
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.features) != args.features_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(Path(__file__)) != args.script_sha256
    ):
        raise ValueError("residual compact head authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration["schema"]
        != "sfora-compact-residual-head-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["features_sha256"] != args.features_sha256
        or preregistration["script_sha256"] != args.script_sha256
    ):
        raise ValueError("residual compact head authority differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    with np.load(args.features, allow_pickle=False) as archive:
        required = {
            "fit_embeddings",
            "fit_labels",
            "evaluation_embeddings",
            "evaluation_labels",
        }
        if set(archive.files) != required | {"metadata_json"}:
            raise ValueError("residual compact feature authority differs")
        fit = torch.from_numpy(np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32))
        fit_labels = torch.from_numpy(
            np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
        )
        evaluation = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
        )
        evaluation_labels = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
        )
    normalized = torch.nn.functional.normalize(fit, dim=1).contiguous()
    weight, bias = _initial_projection(normalized)
    config = CompactMetricConfig()
    definitions = {
        "affine": (False, 0.0),
        "affine_anchor": (False, config.anchor_weight),
        "residual": (True, 0.0),
        "residual_anchor": (True, config.anchor_weight),
    }
    started = time.monotonic()
    arms: dict[str, dict[str, object]] = {}
    schedule_sha256 = ""
    for name, (nonlinear, anchor_weight) in definitions.items():
        head, losses, schedule_digest = train_arm(
            normalized=normalized,
            labels=fit_labels,
            weight=weight,
            bias=bias,
            nonlinear=nonlinear,
            anchor_weight=anchor_weight,
            config=config,
            device=device,
        )
        if schedule_sha256 and schedule_digest != schedule_sha256:
            raise ValueError("residual compact schedule differs")
        schedule_sha256 = schedule_digest
        scored = score_arm(head, evaluation, evaluation_labels, device)
        arms[name] = {
            "map_at_r": scored["map_at_r"],
            "recall_at_1": scored["recall_at_1"],
            "average_precision": scored["average_precision"],
            "final_loss": losses[-1],
            "updates": len(losses),
        }
    candidate = arms["residual_anchor"]
    control_name = max(
        ("affine", "affine_anchor", "residual"),
        key=lambda name: (arms[name]["map_at_r"], arms[name]["recall_at_1"], name),
    )
    control = arms[control_name]
    interval = paired_class_bootstrap(
        candidate["average_precision"],  # type: ignore[arg-type]
        control["average_precision"],  # type: ignore[arg-type]
        evaluation_labels.numpy(),
    )
    map_delta = float(candidate["map_at_r"]) - float(control["map_at_r"])
    recall_delta = float(candidate["recall_at_1"]) - float(control["recall_at_1"])
    passed = map_delta >= 0.005 and interval["lower"] > 0.0 and recall_delta >= 0.0
    for row in arms.values():
        del row["average_precision"]
    result = {
        "schema": "sfora-compact-residual-head-result-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "features_sha256": args.features_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "script_sha256": args.script_sha256,
        "dataset": "cars196",
        "storage_bytes_per_item": 128,
        "hidden_dimensions": 32,
        "residual_parameter_count": 768 * 32 + 32 * 128,
        "schedule_sha256": schedule_sha256,
        "arms": arms,
        "candidate": "residual_anchor",
        "strongest_control": control_name,
        "map_at_r_delta": map_delta,
        "recall_at_1_delta": recall_delta,
        "paired_class_bootstrap_map_delta_95": interval,
        "gate": {
            "minimum_map_at_r_delta": 0.005,
            "require_positive_interval_lower": True,
            "minimum_recall_at_1_delta": 0.0,
            "passed": passed,
        },
        "next_action": "replicate-unchanged-on-cub" if passed else "close-family",
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
