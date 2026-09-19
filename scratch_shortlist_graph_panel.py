#!/usr/bin/env python3
"""Throwaway frozen shortlist-graph rule across untouched retrieval datasets."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scratch_cub_local_metric_oracle import file_sha256, fit_pca, project
from scratch_cub_shortlist_graph import (
    NEIGHBORS,
    SHORTLIST,
)


K = 1_000
BATCH = 256


def remap(labels: np.ndarray) -> np.ndarray:
    mapping = {value: index for index, value in enumerate(sorted(set(labels.tolist())))}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    archive = np.load(path, allow_pickle=False)
    if "features" in archive:
        values = archive["features"]
        labels = archive["labels"]
        if "metadata_json" in archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            name = str(metadata["dataset"])
        else:
            name = path.stem
    else:
        values = archive["evaluation_embeddings"]
        labels = archive["evaluation_labels"]
        name = path.stem
    return np.ascontiguousarray(values, dtype=np.float32), labels.astype(np.int64), name


def metrics_from_hits(
    hits: torch.Tensor, relevant: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    width = hits.shape[1]
    positions = torch.arange(1, width + 1, device=hits.device, dtype=torch.float64)
    precision = hits.cumsum(1).double() / positions
    mask = positions[None, :] <= relevant[:, None]
    ap = (precision * hits * mask).sum(1) / relevant
    return ap.cpu(), hits[:, 0].double().cpu()


def evaluate(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    fit_values: np.ndarray | None = None,
    preserve_head: bool = False,
) -> dict[str, object]:
    device = torch.device("cuda")
    normalized = F.normalize(torch.from_numpy(values).float(), dim=1).contiguous()
    fit = normalized
    if fit_values is not None:
        fit = F.normalize(torch.from_numpy(fit_values).float(), dim=1).contiguous()
    mean, components = fit_pca(fit)
    reduced = project(normalized, mean, components)
    codes = torch.round(reduced * 127.0).clamp(-127, 127).to(torch.int8)
    gallery = F.normalize(codes.float(), dim=1).to(device)
    label_tensor = torch.from_numpy(remap(labels)).to(device)
    relevant = torch.bincount(label_tensor)[label_tensor] - 1
    if bool((relevant <= 0).any()):
        raise ValueError("singleton class differs")
    metric_width = int(relevant.max())
    candidate_width = min(K, len(gallery) - 1)
    if candidate_width < SHORTLIST:
        raise ValueError("dataset is smaller than frozen shortlist")

    baseline_ap_parts: list[torch.Tensor] = []
    baseline_r1_parts: list[torch.Tensor] = []
    graph_ap_parts: list[torch.Tensor] = []
    graph_r1_parts: list[torch.Tensor] = []
    for start in range(0, len(gallery), BATCH):
        stop = min(start + BATCH, len(gallery))
        scores = gallery[start:stop] @ gallery.T
        local = torch.arange(stop - start, device=device)
        scores[local, torch.arange(start, stop, device=device)] = -torch.inf
        order = torch.topk(
            scores, k=candidate_width, dim=1, largest=True, sorted=True
        ).indices
        baseline_hits = label_tensor[order[:, :metric_width]].eq(
            label_tensor[start:stop, None]
        )
        ap, r1 = metrics_from_hits(baseline_hits, relevant[start:stop])
        baseline_ap_parts.append(ap)
        baseline_r1_parts.append(r1)

        shortlist = order[:, :SHORTLIST]
        candidates = gallery[shortlist]
        gram = candidates @ candidates.transpose(1, 2)
        diagonal = torch.arange(SHORTLIST, device=device)
        gram[:, diagonal, diagonal] = -torch.inf
        neighbors = torch.argsort(
            gram, dim=2, descending=True, stable=True
        )[:, :, :NEIGHBORS]
        base = scores.gather(1, shortlist)
        propagated = base.gather(1, neighbors.reshape(stop - start, -1)).reshape(
            stop - start, SHORTLIST, NEIGHBORS
        ).mean(2)
        graph_scores = 0.5 * (base + propagated)
        if preserve_head:
            relative_tail = torch.argsort(
                graph_scores[:, 1:], dim=1, descending=True, stable=True
            ) + 1
            relative = torch.cat(
                (
                    torch.zeros(
                        (stop - start, 1), dtype=torch.int64, device=device
                    ),
                    relative_tail,
                ),
                dim=1,
            )
        else:
            relative = torch.argsort(
                graph_scores, dim=1, descending=True, stable=True
            )
        graph_order = torch.cat(
            (shortlist.gather(1, relative), order[:, SHORTLIST:]), dim=1
        )
        graph_hits = label_tensor[graph_order[:, :metric_width]].eq(
            label_tensor[start:stop, None]
        )
        ap, r1 = metrics_from_hits(graph_hits, relevant[start:stop])
        graph_ap_parts.append(ap)
        graph_r1_parts.append(r1)

    baseline_ap = torch.cat(baseline_ap_parts)
    baseline_r1 = torch.cat(baseline_r1_parts)
    graph_ap = torch.cat(graph_ap_parts)
    graph_r1 = torch.cat(graph_r1_parts)
    delta = graph_ap - baseline_ap
    classes = torch.unique(label_tensor.cpu(), sorted=True)
    class_means = torch.stack(
        [delta[label_tensor.cpu() == label].mean() for label in classes]
    )
    return {
        "rows": len(labels),
        "classes": len(classes),
        "baseline": {
            "map_at_r": float(baseline_ap.mean()),
            "recall_at_1": float(baseline_r1.mean()),
        },
        "graph": {
            "map_at_r": float(graph_ap.mean()),
            "recall_at_1": float(graph_r1.mean()),
            "map_delta": float(delta.mean()),
            "class_delta_p10": float(torch.quantile(class_means, 0.10)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    datasets: dict[str, object] = {}
    deltas: list[float] = []
    r1_losses: list[float] = []
    for path in args.datasets:
        values, labels, name = load_dataset(path)
        result = evaluate(values, labels)
        result["artifact_sha256"] = file_sha256(path)
        datasets[name] = result
        deltas.append(result["graph"]["map_delta"])
        r1_losses.append(
            result["graph"]["recall_at_1"] - result["baseline"]["recall_at_1"]
        )
        print(json.dumps({"dataset_complete": name, "map_delta": deltas[-1]}), flush=True)

    decision = {
        "macro_map_delta": float(np.mean(deltas)),
        "positive_datasets": sum(delta > 0.0 for delta in deltas),
        "dataset_count": len(deltas),
        "worst_map_delta": min(deltas),
        "worst_recall_at_1_delta": min(r1_losses),
    }
    decision["passed"] = (
        decision["macro_map_delta"] >= 0.010
        and decision["positive_datasets"] >= len(deltas) - 1
        and decision["worst_map_delta"] >= -0.002
        and decision["worst_recall_at_1_delta"] >= -0.002
    )
    payload = {
        "schema": "scratch-shortlist-graph-panel-v1",
        "claim_eligible": False,
        "representation": "dataset-fit-pca128-int8",
        "shortlist": SHORTLIST,
        "neighbors": NEIGHBORS,
        "mix": 0.5,
        "datasets": datasets,
        "decision": decision,
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    main()
