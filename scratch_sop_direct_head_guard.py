#!/usr/bin/env python3
"""Compose learned direct 768->128 heads with rank-one-safe shortlist smoothing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scratch_cub_local_metric_oracle import file_sha256
from scratch_shortlist_graph_panel import metrics_from_hits, remap


BATCH = 256
SHORTLIST = 128
NEIGHBORS = 5
PRIOR_EIGENVALUE_FLOOR = 1.0e-4
LOCAL_TRACE_FLOOR = 1.0e-8
SCORE_STD_FLOOR = 1.0e-6


def normalized_within_class_covariance(
    codes: torch.Tensor, labels: np.ndarray
) -> torch.Tensor:
    """Return a trace-normalized covariance of training within-class residuals."""

    unit = F.normalize(codes.float().cuda(), dim=1)
    label_tensor = torch.from_numpy(remap(labels)).cuda()
    class_count = int(label_tensor.max()) + 1
    sums = torch.zeros(class_count, unit.shape[1], device=unit.device)
    sums.index_add_(0, label_tensor, unit)
    counts = torch.bincount(label_tensor, minlength=class_count).clamp_min(1)
    means = sums / counts[:, None]
    residuals = unit - means[label_tensor]
    covariance = residuals.T @ residuals / len(residuals)
    covariance = covariance * (covariance.shape[0] / covariance.trace())
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    covariance = (eigenvectors * eigenvalues.clamp_min(PRIOR_EIGENVALUE_FLOOR)) @ eigenvectors.T
    return covariance * (covariance.shape[0] / covariance.trace())


def lexicographic_candidates(scores: torch.Tensor, width: int) -> torch.Tensor:
    """Select score-descending, global-ordinal-ascending candidates exactly."""

    retained = min(width + 1, scores.shape[1])
    values, indexes = torch.topk(
        scores, k=retained, dim=1, largest=True, sorted=False
    )
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    score_order = torch.argsort(values, dim=1, descending=True, stable=True)
    indexes = indexes.gather(1, score_order)
    values = values.gather(1, score_order)
    if retained > width:
        ambiguous = values[:, width - 1] == values[:, width]
        for row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
            boundary = values[row, width - 1]
            candidates = torch.nonzero(
                scores[row] >= boundary, as_tuple=False
            ).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = scores[row, candidates]
            order = torch.argsort(candidate_values, descending=True, stable=True)
            indexes[row, :width] = candidates[order[:width]]
    return indexes[:, :width]


def band_order(scores: torch.Tensor, protected_cutoffs: tuple[int, ...]) -> torch.Tensor:
    boundaries = (0, *protected_cutoffs, SHORTLIST)
    return torch.cat(
        [
            torch.argsort(
                scores[:, lower:upper], dim=1, descending=True, stable=True
            )
            + lower
            for lower, upper in zip(boundaries, boundaries[1:])
            if lower != upper
        ],
        dim=1,
    )


def evaluate_codes(
    codes: torch.Tensor,
    labels: np.ndarray,
    *,
    protected_cutoffs: tuple[int, ...],
    covariance_prior: torch.Tensor | None = None,
) -> dict[str, object]:
    gallery = codes.float().cuda()
    inverse_norms = torch.linalg.vector_norm(codes.float(), dim=1).reciprocal()
    inverse_norms = inverse_norms.to(torch.float16).cuda().float()
    label_tensor = torch.from_numpy(remap(labels)).cuda()
    relevant = torch.bincount(label_tensor)[label_tensor] - 1
    width = int(relevant.max())
    candidate_width = min(len(gallery) - 1, max(SHORTLIST, width))
    baseline_ap: list[torch.Tensor] = []
    baseline_r1: list[torch.Tensor] = []
    guarded_ap: list[torch.Tensor] = []
    guarded_r1: list[torch.Tensor] = []
    dba_ap: list[torch.Tensor] = []
    dba_r1: list[torch.Tensor] = []
    cld_ap: list[torch.Tensor] = []
    cld_r1: list[torch.Tensor] = []
    cld_dba_ap: list[torch.Tensor] = []
    cld_dba_r1: list[torch.Tensor] = []
    positive_ap: list[torch.Tensor] = []
    positive_r1: list[torch.Tensor] = []
    prior_dba_ap: list[torch.Tensor] = []
    prior_dba_r1: list[torch.Tensor] = []
    oracle_ap: list[torch.Tensor] = []
    oracle_r1: list[torch.Tensor] = []
    cutoffs = tuple(sorted(set((*protected_cutoffs, 1, 10, 100))))
    recalls: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {cutoff: [] for cutoff in cutoffs}
        for name in (
            "baseline",
            "contextual",
            "normalized_dba",
            "cld",
            "cld_dba",
            "positive_only",
            "prior_dba",
            "oracle",
        )
    }
    fallback_count = 0
    prior_cholesky = (
        None
        if covariance_prior is None
        else torch.linalg.cholesky(covariance_prior)
    )
    for start in range(0, len(gallery), BATCH):
        stop = min(start + BATCH, len(gallery))
        count = stop - start
        scores = (
            (gallery[start:stop] @ gallery.T)
            * inverse_norms[start:stop, None]
            * inverse_norms[None, :]
        )
        scores[
            torch.arange(count, device=gallery.device),
            torch.arange(start, stop, device=gallery.device),
        ] = -torch.inf
        order = lexicographic_candidates(scores, candidate_width)
        shortlist = order[:, :SHORTLIST]
        candidates = gallery[shortlist]
        candidate_norms = inverse_norms[shortlist]
        gram = (
            (candidates @ candidates.transpose(1, 2))
            * candidate_norms[:, :, None]
            * candidate_norms[:, None, :]
        )
        diagonal = torch.arange(SHORTLIST, device=gallery.device)
        gram[:, diagonal, diagonal] = -torch.inf
        neighbors = torch.argsort(
            gram, dim=2, descending=True, stable=True
        )[:, :, :NEIGHBORS]
        base = scores.gather(1, shortlist)
        propagated = base.gather(1, neighbors.reshape(count, -1)).reshape(
            count, SHORTLIST, NEIGHBORS
        ).mean(2)
        graph_scores = 0.5 * (base + propagated)
        contextual_relative = band_order(graph_scores, protected_cutoffs)
        unit_candidates = candidates * candidate_norms[:, :, None]
        batch_indexes = torch.arange(count, device=gallery.device)[:, None, None]
        neighbor_vectors = unit_candidates[batch_indexes, neighbors]
        augmented = F.normalize(
            0.5 * (unit_candidates + neighbor_vectors.mean(2)), dim=2
        )
        query_unit = gallery[start:stop] * inverse_norms[start:stop, None]
        dba_scores = torch.einsum("bd,bkd->bk", query_unit, augmented)
        dba_relative = band_order(dba_scores, protected_cutoffs)
        positive_mean = 0.5 * (query_unit + unit_candidates[:, 0])
        negatives = unit_candidates[:, 40:SHORTLIST]
        negative_mean = negatives.mean(1)
        centered_negatives = negatives - negative_mean[:, None, :]
        covariance = torch.einsum(
            "bnd,bne->bde", centered_negatives, centered_negatives
        ) / centered_negatives.shape[1]
        ridge = covariance.diagonal(dim1=1, dim2=2).sum(1) / covariance.shape[1]
        prior = (
            torch.eye(covariance.shape[1], device=gallery.device)
            if covariance_prior is None
            else covariance_prior
        )
        active = ridge > LOCAL_TRACE_FLOOR
        fallback_count += int((~active).sum())
        cld_scores = dba_scores.clone()
        if bool(active.any()):
            discriminant = torch.linalg.solve(
                covariance[active] + ridge[active, None, None] * prior,
                (positive_mean[active] - negative_mean[active])[:, :, None],
            ).squeeze(2)
            cld_scores[active] = torch.einsum(
                "bkd,bd->bk", unit_candidates[active], discriminant
            )
        cld_relative = band_order(cld_scores, protected_cutoffs)
        standardized_cld = (cld_scores - cld_scores.mean(1, keepdim=True)) / cld_scores.std(
            1, keepdim=True, correction=0
        ).clamp_min(SCORE_STD_FLOOR)
        standardized_dba = (dba_scores - dba_scores.mean(1, keepdim=True)) / dba_scores.std(
            1, keepdim=True, correction=0
        ).clamp_min(SCORE_STD_FLOOR)
        cld_dba_scores = standardized_cld + standardized_dba
        cld_dba_relative = band_order(cld_dba_scores, protected_cutoffs)
        positive_scores = torch.einsum("bkd,bd->bk", unit_candidates, positive_mean)
        positive_relative = band_order(positive_scores, protected_cutoffs)
        if prior_cholesky is None:
            prior_discriminant = positive_mean - negative_mean
        else:
            prior_discriminant = torch.cholesky_solve(
                (positive_mean - negative_mean).T, prior_cholesky
            ).T
        prior_scores = torch.einsum(
            "bkd,bd->bk", unit_candidates, prior_discriminant
        )
        standardized_prior = (
            prior_scores - prior_scores.mean(1, keepdim=True)
        ) / prior_scores.std(1, keepdim=True, correction=0)
        prior_dba_scores = standardized_prior + standardized_dba
        prior_dba_relative = band_order(prior_dba_scores, protected_cutoffs)
        guarded_order = torch.cat(
            (shortlist.gather(1, contextual_relative), order[:, SHORTLIST:]), dim=1
        )
        dba_order = torch.cat(
            (shortlist.gather(1, dba_relative), order[:, SHORTLIST:]), dim=1
        )
        cld_order = torch.cat(
            (shortlist.gather(1, cld_relative), order[:, SHORTLIST:]), dim=1
        )
        cld_dba_order = torch.cat(
            (shortlist.gather(1, cld_dba_relative), order[:, SHORTLIST:]), dim=1
        )
        positive_order = torch.cat(
            (shortlist.gather(1, positive_relative), order[:, SHORTLIST:]), dim=1
        )
        prior_dba_order = torch.cat(
            (shortlist.gather(1, prior_dba_relative), order[:, SHORTLIST:]), dim=1
        )
        shortlist_hits = label_tensor[shortlist].eq(label_tensor[start:stop, None])
        boundaries = (0, *protected_cutoffs, SHORTLIST)
        oracle_relative = torch.cat(
            [
                torch.argsort(
                    shortlist_hits[:, lower:upper].to(torch.int8),
                    dim=1,
                    descending=True,
                    stable=True,
                )
                + lower
                for lower, upper in zip(boundaries, boundaries[1:])
                if lower != upper
            ],
            dim=1,
        )
        oracle_order = torch.cat(
            (shortlist.gather(1, oracle_relative), order[:, SHORTLIST:]), dim=1
        )
        for name, ranked_order, ap_parts, r1_parts in (
            ("baseline", order, baseline_ap, baseline_r1),
            ("contextual", guarded_order, guarded_ap, guarded_r1),
            ("normalized_dba", dba_order, dba_ap, dba_r1),
            ("cld", cld_order, cld_ap, cld_r1),
            ("cld_dba", cld_dba_order, cld_dba_ap, cld_dba_r1),
            ("positive_only", positive_order, positive_ap, positive_r1),
            ("prior_dba", prior_dba_order, prior_dba_ap, prior_dba_r1),
            ("oracle", oracle_order, oracle_ap, oracle_r1),
        ):
            hits = label_tensor[ranked_order[:, :width]].eq(
                label_tensor[start:stop, None]
            )
            ap, r1 = metrics_from_hits(hits, relevant[start:stop])
            ap_parts.append(ap)
            r1_parts.append(r1)
            for cutoff in cutoffs:
                cutoff_hits = label_tensor[ranked_order[:, :cutoff]].eq(
                    label_tensor[start:stop, None]
                )
                recalls[name][cutoff].append(cutoff_hits.any(1).double().cpu())
    baseline_ap_value = float(torch.cat(baseline_ap).mean())
    baseline_r1_value = float(torch.cat(baseline_r1).mean())
    guarded_ap_value = float(torch.cat(guarded_ap).mean())
    guarded_r1_value = float(torch.cat(guarded_r1).mean())
    dba_ap_value = float(torch.cat(dba_ap).mean())
    dba_r1_value = float(torch.cat(dba_r1).mean())
    cld_ap_value = float(torch.cat(cld_ap).mean())
    cld_r1_value = float(torch.cat(cld_r1).mean())
    cld_dba_ap_value = float(torch.cat(cld_dba_ap).mean())
    cld_dba_r1_value = float(torch.cat(cld_dba_r1).mean())
    positive_ap_value = float(torch.cat(positive_ap).mean())
    positive_r1_value = float(torch.cat(positive_r1).mean())
    prior_dba_ap_value = float(torch.cat(prior_dba_ap).mean())
    prior_dba_r1_value = float(torch.cat(prior_dba_r1).mean())
    oracle_ap_value = float(torch.cat(oracle_ap).mean())
    result: dict[str, object] = {
        "baseline_map_at_r": baseline_ap_value,
        "baseline_recall_at_1": baseline_r1_value,
        "guarded_map_at_r": guarded_ap_value,
        "guarded_recall_at_1": guarded_r1_value,
        "map_delta": guarded_ap_value - baseline_ap_value,
        "recall_at_1_delta": guarded_r1_value - baseline_r1_value,
        "normalized_dba_map_at_r": dba_ap_value,
        "normalized_dba_recall_at_1": dba_r1_value,
        "contextual_minus_normalized_dba_map_at_r": guarded_ap_value - dba_ap_value,
        "cld_map_at_r": cld_ap_value,
        "cld_recall_at_1": cld_r1_value,
        "cld_minus_normalized_dba_map_at_r": cld_ap_value - dba_ap_value,
        "cld_dba_map_at_r": cld_dba_ap_value,
        "cld_dba_recall_at_1": cld_dba_r1_value,
        "cld_dba_minus_normalized_dba_map_at_r": cld_dba_ap_value - dba_ap_value,
        "positive_only_map_at_r": positive_ap_value,
        "positive_only_recall_at_1": positive_r1_value,
        "prior_dba_map_at_r": prior_dba_ap_value,
        "prior_dba_recall_at_1": prior_dba_r1_value,
        "prior_dba_minus_normalized_dba_map_at_r": prior_dba_ap_value
        - dba_ap_value,
        "numerical_fallback_count": fallback_count,
        "oracle_map_at_r": oracle_ap_value,
        "oracle_minus_normalized_dba_map_at_r": oracle_ap_value - dba_ap_value,
    }
    for name, by_cutoff in recalls.items():
        result[f"{name}_recall"] = {
            str(cutoff): float(torch.cat(parts).mean())
            for cutoff, parts in by_cutoff.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--protected-cutoffs", type=int, nargs="+", default=[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protected_cutoffs = tuple(args.protected_cutoffs)
    if (
        tuple(sorted(set(protected_cutoffs))) != protected_cutoffs
        or protected_cutoffs[0] < 1
        or protected_cutoffs[-1] >= SHORTLIST
    ):
        raise ValueError("protected cutoffs must be unique, sorted, and inside 1..127")
    started = time.monotonic()
    archive = np.load(args.features, allow_pickle=False)
    embeddings = F.normalize(
        torch.from_numpy(
            np.ascontiguousarray(archive["test_embeddings"], dtype=np.float32)
        ),
        dim=1,
    ).contiguous()
    labels = archive["test_labels"].astype(np.int64)
    train_embeddings = F.normalize(
        torch.from_numpy(
            np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        ),
        dim=1,
    ).contiguous()
    train_labels = archive["train_labels"].astype(np.int64)
    results: dict[str, object] = {}
    for checkpoint in args.checkpoints:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        projected = F.normalize(
            embeddings @ state["weight"].T + state["bias"], dim=1
        )
        codes = torch.round(projected * 127.0).clamp(-127, 127).to(torch.int8)
        train_projected = F.normalize(
            train_embeddings @ state["weight"].T + state["bias"], dim=1
        )
        train_codes = (
            torch.round(train_projected * 127.0).clamp(-127, 127).to(torch.int8)
        )
        covariance_prior = normalized_within_class_covariance(
            train_codes, train_labels
        )
        result = evaluate_codes(
            codes,
            labels,
            protected_cutoffs=protected_cutoffs,
            covariance_prior=covariance_prior,
        )
        result["checkpoint_sha256"] = file_sha256(checkpoint)
        results[checkpoint.stem] = result
        print(json.dumps({"checkpoint_complete": checkpoint.stem, **result}), flush=True)
    guarded_maps = [float(result["guarded_map_at_r"]) for result in results.values()]
    guarded_r1s = [float(result["guarded_recall_at_1"]) for result in results.values()]
    payload = {
        "schema": "scratch-sop-direct-head-guard-v1",
        "claim_eligible": False,
        "protocol": "already-observed-sop-official-test",
        "features_sha256": file_sha256(args.features),
        "fixed_rule": {
            "shortlist": SHORTLIST,
            "neighbors": NEIGHBORS,
            "mix": 0.5,
            "protected_cutoffs": list(protected_cutoffs),
        },
        "results": results,
        "aggregate": {
            "mean_guarded_map_at_r": float(np.mean(guarded_maps)),
            "minimum_guarded_map_at_r": min(guarded_maps),
            "mean_guarded_recall_at_1": float(np.mean(guarded_r1s)),
            "all_map_target_0_496": all(value >= 0.496 for value in guarded_maps),
            "all_r1_exactly_preserved": all(
                float(result["recall_at_1_delta"]) == 0.0
                for result in results.values()
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n"
    args.output.write_text(wire)
    print(wire, end="")


if __name__ == "__main__":
    main()
