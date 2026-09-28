#!/usr/bin/env python3
"""One fixed256D main-supervision capacity screen with128D folded serving."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, schedule, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from probe_inshop_source_classifier import prototype_scores, unused_gradient
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import initialize_head_and_classifier, member_bank_positive_ordinals

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.representation_ceiling import fit_centered_pca
from sfora.unicom_training import sharded_mask_arcface_loss


def fold_uncentered_head(head, components):
    """Fold a zero-bias linear output map into an affine projected head."""
    result = nn.Linear(
        head.in_features, components.shape[0], dtype=head.weight.dtype, device=head.weight.device
    )
    with torch.no_grad():
        result.weight.copy_(components @ head.weight)
        result.bias.copy_(components @ head.bias)
    return result


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "source-cache", "source-diagnostic", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.source_cache) != SOURCE_CACHE_SHA
        or sha256(args.source_diagnostic)
        != "ace3f13e92bc0357c195b307da3fde128780f33a9f7a9a8c12af7b0b8200f29f"
    ):
        raise ValueError("wide main-head cache authority differs")
    previous = json.loads(args.source_diagnostic.read_text())
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("wide main-head fit authority differs")
    names = sorted({labels[row] for row in fit})
    lookup = {name: index for index, name in enumerate(names)}
    target = torch.tensor([lookup[labels[row]] for row in fit])
    query_rows = previous["query_train_rows"]
    if (
        len(query_rows) != 512
        or not set(query_rows).issubset(fit)
        or len({labels[row] for row in query_rows}) != 512
    ):
        raise ValueError("wide main-head frozen panel differs")
    positions = torch.tensor([fit.index(row) for row in query_rows])
    qtarget = target[positions]
    values = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if values.shape != (25882, 1024) or values.dtype != np.float32:
        raise ValueError("wide main-head cache geometry differs")
    source = torch.from_numpy(values[list(fit)].copy())
    unit = F.normalize(source, dim=1)
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    narrow, narrow_classifier, pca_sha = initialize_head_and_classifier(
        source, tuple(target.tolist()), allow_singletons=True
    )
    if pca_sha != "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8":
        raise ValueError("wide main-head narrow PCA differs")
    pca = fit_centered_pca(unit, dimensions=256)
    if not torch.equal(pca.components[:128], narrow.weight.detach()):
        raise ValueError("wide PCA changes narrow source basis")
    wide = nn.Linear(1024, 256)
    with torch.no_grad():
        wide.weight.copy_(pca.components)
        wide.bias.copy_(-(pca.components @ pca.mean))
        projected = pca.apply(unit)
        sums = torch.zeros(len(names), 256)
        for row, label in enumerate(target.tolist()):
            sums[label] += projected[row]
        wide_classifier = nn.Parameter(F.normalize(sums, dim=1))
        wide_fit = F.normalize(wide(unit), dim=1)
        _, singular, right = torch.linalg.svd(wide_fit.double(), full_matrices=False)
        if not torch.isfinite(singular).all() or singular[127] <= 1e-10:
            raise ValueError("uncentered compactor rank differs")
        components = right[:128].clone()
        for row in components:
            if row[row.abs().argmax()] < 0:
                row.neg_()
        components = components.float().contiguous()
    folded = fold_uncentered_head(wide, components)
    hits = {}
    with torch.no_grad():
        two_stage = F.normalize(wide_fit @ components.T, dim=1)
        folded_fit = F.normalize(folded(unit), dim=1)
        fold_error = float((two_stage - folded_fit).abs().max())
        for name, head in (("narrow128", narrow), ("wide256", wide), ("folded128", folded)):
            fit_values = F.normalize(head(unit), dim=1)
            scores = prototype_scores(fit_values[positions], fit_values, target, positions, qtarget)
            hits[name] = (scores.argmax(1) == qtarget).int().numpy()
    if hits["narrow128"].tolist() != previous["per_query_compact_hit"]:
        raise ValueError("wide main-head previous narrow hit replay differs")
    qlabels = np.asarray([labels[row] for row in query_rows])
    delta = hits["wide256"] - hits["narrow128"]
    lower = bootstrap_lower(delta, qlabels)
    upper = -bootstrap_lower(-delta, qlabels)
    banks = {
        name: F.normalize(head(unit).detach(), dim=1)
        for name, head in (("narrow", narrow), ("wide", wide))
    }
    positives = member_bank_positive_ordinals(target.numpy(), allow_singletons=True)
    counts = torch.bincount(target)
    batches = schedule(tuple(labels[row] for row in fit), 1000, seed=179024)
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    if schedule_sha != "c12def923f604ec0c72fe52f498dda80985905760428d4c8ad8485972a5996b9":
        raise ValueError("wide main-head schedule differs")
    fractions, ratios, residuals, losses = [], [], [], []
    for rows in batches[:17]:
        indexes = torch.tensor(rows)
        gradients, objectives = {}, {}
        query = source[indexes].detach().clone().requires_grad_()
        for name, head, classifier in (
            ("narrow", narrow, narrow_classifier),
            ("wide", wide, wide_classifier),
        ):
            compact = head(F.normalize(query, dim=1))
            loss = sharded_mask_arcface_loss(
                compact,
                classifier,
                target[indexes],
                torch.arange(head.out_features).reshape(1, -1),
                margin=0.3,
                scale=64,
            )
            if bool((counts[target[indexes]] > 1).all()):
                positive = positives[indexes]
                width = int((positive >= 0).sum(1).max())
                loss = loss + 8 * smooth_ap_bank_loss(
                    F.normalize(compact.float(), dim=1), banks[name], positive[:, :width], indexes
                )
            gradients[name] = torch.autograd.grad(loss, query)[0]
            objectives[name] = float(loss.detach())
        unit_query = F.normalize(query.detach(), dim=1)
        wide_unused = unused_gradient(gradients["wide"], narrow.weight.detach(), unit_query)
        narrow_unused = unused_gradient(gradients["narrow"], narrow.weight.detach(), unit_query)
        norms = {name: value.double().norm(dim=1) for name, value in gradients.items()}
        if not all(torch.isfinite(value).all() and (value > 0).all() for value in norms.values()):
            raise ValueError("wide main-head gradient undefined")
        fractions.extend((wide_unused.double().norm(dim=1) / norms["wide"]).tolist())
        ratios.extend((norms["wide"] / norms["narrow"]).tolist())
        residuals.extend((narrow_unused.double().norm(dim=1) / norms["narrow"]).tolist())
        losses.append(objectives)
    wall = time.perf_counter() - started
    orthogonal = torch.allclose(components @ components.T, torch.eye(128), rtol=0, atol=2e-5)
    criteria = {
        "wide_prototype_gain": float(delta.mean()) >= 0.01,
        "wide_prototype_lower": lower > 0,
        "folded_gross_floor": float((hits["folded128"] - hits["narrow128"]).mean()) >= -0.01,
        "new_gradient_route": float(np.median(fractions)) >= 0.2,
        "bounded_gradient_pressure": 0.25 <= float(np.median(ratios)) <= 4,
        "finite_1088_rows": len(fractions) == 1088
        and np.isfinite(fractions).all()
        and np.isfinite(ratios).all(),
        "narrow_residual": max(residuals) <= 1e-5,
        "orthogonality_fold": bool(orthogonal) and fold_error <= 1e-5,
        "cpu_budget": wall <= 120,
    }
    receipt = {
        "schema": "sfora-inshop-wide-main-head-cache-v1",
        "claim_eligible": False,
        "split": "official TRAIN fit only;512 leave-query-out prototypes, no held reads",
        "decision": "GO_BOUNDED_ENCODER_DESIGN" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "query_train_rows": query_rows,
        "per_query_hits": {name: value.tolist() for name, value in hits.items()},
        "prototype_accuracy": {name: float(value.mean()) for name, value in hits.items()},
        "wide_minus_narrow": float(delta.mean()),
        "wide_minus_narrow_95": [lower, upper],
        "median_wide_unused_gradient_fraction": float(np.median(fractions)),
        "median_wide_narrow_gradient_norm_ratio": float(np.median(ratios)),
        "narrow_unused_gradient_fraction_max": max(residuals),
        "wide_unused_gradient_fractions": fractions,
        "wide_narrow_gradient_norm_ratios": ratios,
        "losses": losses,
        "folded_normalized_float_max_error": fold_error,
        "cpu_main_wall_seconds": wall,
        "source_sha256": sha256(Path(__file__)),
        "source_cache_sha256": SOURCE_CACHE_SHA,
        "partition_sha256": PARTITION_SHA,
        "fit_sha256": digest_rows(fit),
        "narrow_pca_sha256": pca_sha,
        "schedule_sha256": schedule_sha,
        "compactor_sha256": hashlib.sha256(components.numpy().tobytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "decision",
                    "criteria",
                    "prototype_accuracy",
                    "wide_minus_narrow_95",
                    "median_wide_unused_gradient_fraction",
                    "median_wide_narrow_gradient_norm_ratio",
                    "cpu_main_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
