#!/usr/bin/env python3
"""Frozen TRAIN-fit cached GELU/half-linear activation-placement filter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from preflight_inshop_siglip2_unseen_gallery import (
    PARTITION_SHA,
    digest_rows,
    schedule,
    split,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_training import sharded_mask_arcface_loss
from train_sop_siglip2_compact import (
    initialize_head_and_classifier,
    member_bank_positive_ordinals,
    member_bank_rank_loss,
    member_bank_refresh_rows,
    member_bank_refresh_values,
    score_packed_full_gallery,
)

CACHE_SHA = "f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885"
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
SEEDS = (17, 23, 29)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Residual(nn.Module):
    def __init__(self, primary, source, seed, nonlinear):
        super().__init__()
        self.primary = copy.deepcopy(primary)
        self.register_buffer("center", F.normalize(source, dim=1).mean(0))
        self.down = nn.Linear(1024, 32, bias=False)
        self.up = nn.Linear(32, 128, bias=False)
        self.nonlinear = nonlinear
        generator = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=5**0.5, generator=generator)
            z = self.down(F.normalize(source, dim=1) - self.center)
            self.scale = float(z.std(unbiased=False))
            if not np.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("residual scale undefined")
            self.down.weight.div_(self.scale)
            self.up.weight.zero_()
        assert sum(p.numel() for p in self.parameters()) == 168064

    def residual(self, unit):
        z = self.down(unit - self.center)
        return self.up(F.gelu(z) if self.nonlinear else 0.5 * z)

    def forward(self, source):
        unit = F.normalize(source.float(), dim=1)
        return self.primary(unit) + self.residual(unit)


def self_test():
    torch.manual_seed(1)
    x = torch.randn(32, 1024)
    primary = nn.Linear(1024, 128)
    a, b = Residual(primary, x, 17, True), Residual(primary, x, 17, False)
    expected = primary(F.normalize(x, dim=1))
    assert torch.equal(a(x), b(x)) and torch.equal(a(x), expected)
    wa, wb = (
        pack_int8_unit_embeddings(F.normalize(a(x), dim=1)),
        pack_int8_unit_embeddings(F.normalize(b(x), dim=1)),
    )
    assert wa.to_bytes() == wb.to_bytes()
    labels = torch.arange(8).repeat_interleave(4)
    classifier = nn.Parameter(torch.randn(8, 128))
    masks = torch.arange(128).unsqueeze(0)

    bank = F.normalize(expected, dim=1).detach()
    positives = member_bank_positive_ordinals(labels.numpy())

    def objective(raw, head):
        return sharded_mask_arcface_loss(
            raw, classifier, labels, masks, margin=0.3, scale=64
        ) + 8 * member_bank_rank_loss(
            raw, bank, head, positives, torch.arange(len(x)), live_head=False
        )

    def loss(head, query):
        return objective(head(query), head)

    baseline_x = x.clone().requires_grad_()
    baseline_loss = objective(primary(F.normalize(baseline_x, dim=1)), primary)
    baseline_grad = torch.autograd.grad(baseline_loss, (baseline_x, primary.weight, primary.bias))
    for head in (a, b):
        query = x.clone().requires_grad_()
        current = loss(head, query)
        assert torch.equal(current, baseline_loss)
        grads = torch.autograd.grad(current, (query, head.primary.weight, head.primary.bias))
        for actual, reference in zip(grads, baseline_grad, strict=True):
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.05)
        for step in range(2):
            opt.zero_grad()
            loss(head, x).backward()
            assert head.up.weight.grad.norm() > 0
            assert (
                (head.down.weight.grad.norm() == 0)
                if step == 0
                else (head.down.weight.grad.norm() > 0)
            )
            opt.step()
        dead = copy.deepcopy(head)
        with torch.no_grad():
            dead.down.weight.zero_()
            dead.up.weight.zero_()
        loss(dead, x).backward()
        assert dead.down.weight.grad.norm() == 0 and dead.up.weight.grad.norm() == 0
    rows, positions = member_bank_refresh_rows((3, 0, 3, 1))
    assert rows == (0, 1, 3) and positions == (1, 3, 2)
    table = member_bank_positive_ordinals(
        np.array([0, 0, 1], dtype=np.int64), allow_singletons=True
    )
    assert table.tolist() == [[1], [0], [-1]]
    low = {
        "recall_at_1": 0.0,
        "map_at_r": 0.0,
        "per_query_r1": [0.0] * 3,
        "per_query_ap": [0.0] * 3,
    }
    high = {
        "recall_at_1": 1.0,
        "map_at_r": 1.0,
        "per_query_r1": [1.0] * 3,
        "per_query_ap": [1.0] * 3,
    }
    fixture = {"base": low, "arms": {"control": {"quality": low}, "nonlinear": {"quality": high}}}
    combined = aggregate([fixture] * 3, np.array([0, 0, 1]))
    assert combined["advance_image_proposal"] and combined["r1_product_bootstrap_95ci"] == [1, 1]
    fixture["arms"]["nonlinear"]["quality"] = low
    assert not aggregate([fixture] * 3, np.array([0, 0, 1]))["advance_image_proposal"]
    print("PASS full-objective parity; live/dead gradients; duplicate/singleton bank contracts")


def load_inputs(partition, cache):
    if sha(partition) != PARTITION_SHA or sha(cache) != CACHE_SHA:
        raise ValueError("input authority differs")
    labels = tuple(
        line.split()[1]
        for line in partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    )
    fit, _ = split(labels)
    assert len(fit) == 13283 and digest_rows(fit) == FIT_SHA
    products = sorted(
        {labels[i] for i in fit},
        key=lambda p: (hashlib.sha256(b"sfora-nonlinear-v1\0" + p.encode("utf-8")).digest(), p),
    )
    assert len(products) == 2004
    train_names = set(products[:1002])
    train_rows = tuple(i for i in fit if labels[i] in train_names)
    val_rows = tuple(i for i in fit if labels[i] not in train_names)
    assert set(train_rows).isdisjoint(val_rows) and set(train_rows) | set(val_rows) == set(fit)
    names = sorted(train_names)
    codec = {name: i for i, name in enumerate(names)}
    target = torch.tensor([codec[labels[i]] for i in train_rows])
    values = np.load(cache, mmap_mode="r", allow_pickle=False)
    assert values.shape == (25882, 1024) and values.dtype == np.float32
    train = torch.from_numpy(values[list(train_rows)].copy())
    val = torch.from_numpy(values[list(val_rows)].copy())
    assert torch.isfinite(train).all() and torch.isfinite(val).all()
    val_names = sorted({labels[i] for i in val_rows})
    val_codec = {name: i for i, name in enumerate(val_names)}
    val_labels = torch.tensor([val_codec[labels[i]] for i in val_rows])
    counts = torch.bincount(val_labels)
    queries = torch.nonzero(counts[val_labels] > 1).flatten()
    primary, classifier, pca_sha = initialize_head_and_classifier(
        train, tuple(target.tolist()), allow_singletons=True
    )
    batches = {
        seed: schedule(tuple(labels[i] for i in train_rows), 1000, seed=seed) for seed in SEEDS
    }
    models = {seed: Residual(primary, train, seed, True) for seed in SEEDS}
    manifest = {
        "partition_sha256": PARTITION_SHA,
        "cache_sha256": CACHE_SHA,
        "train_rows_sha256": digest_rows(train_rows),
        "validation_rows_sha256": digest_rows(val_rows),
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "eligible_queries": len(queries),
        "validation_singletons": int((counts == 1).sum()),
        "pca_sha256": pca_sha,
        "classifier_sha256": digest(classifier),
        "center_sha256": digest(models[17].center),
        "seeds": {
            str(seed): {
                "down_sha256": digest(models[seed].down.weight),
                "scale": models[seed].scale,
                "schedule_sha256": hashlib.sha256(
                    np.asarray(batches[seed], dtype="<i4").tobytes()
                ).hexdigest(),
            }
            for seed in SEEDS
        },
        "script_sha256": sha(__file__),
        "helper_sha256": sha(Path(__file__).with_name("train_sop_siglip2_compact.py")),
        "dependencies_sha256": {
            name: sha(module.__file__)
            for name, module in sorted(sys.modules.items())
            if (
                name.startswith("sfora.")
                or name in ("preflight_inshop_siglip2_unseen_gallery", "train_sop_siglip2_compact")
            )
            and getattr(module, "__file__", None)
        },
        "torch": torch.__version__,
        "num_threads": torch.get_num_threads(),
    }
    return train, target, val, val_labels, queries, primary, classifier, batches, models, manifest


def score(head, values, labels, queries):
    with torch.no_grad():
        features = F.normalize(head(values), dim=1)
        packed = pack_int8_unit_embeddings(features)
    return score_packed_full_gallery(
        packed.codes.float(), packed.inverse_norms, labels, queries, device=torch.device("cpu")
    )


def train_arm(head, classifier, values, target, batches):
    started = time.perf_counter()
    classifier = nn.Parameter(classifier.detach().clone())
    params = list(head.parameters()) + [classifier]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.05)
    with torch.no_grad():
        bank = F.normalize(head(values), dim=1).detach()
    positives = member_bank_positive_ordinals(target.numpy(), allow_singletons=True)
    counts = torch.bincount(target)
    losses, rank_updates = [], 0
    for batch in batches:
        index = torch.tensor(batch)
        optimizer.zero_grad(set_to_none=True)
        raw = head(values[index])
        loss = sharded_mask_arcface_loss(
            raw, classifier, target[index], torch.arange(128).unsqueeze(0), margin=0.3, scale=64
        )
        if bool((counts[target[index]] > 1).all()):
            table = positives[index]
            width = int((table >= 0).sum(1).max())
            loss = loss + 8 * member_bank_rank_loss(
                raw, bank, head, table[:, :width], index, live_head=False
            )
            rank_updates += 1
        if not bool(torch.isfinite(loss)):
            raise ValueError("cached loss nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        optimizer.step()
        rows, positions = member_bank_refresh_rows(tuple(batch))
        bank[torch.tensor(rows)] = member_bank_refresh_values(
            values[index], raw, torch.tensor(positions), live_head=False
        )
        if any(not torch.isfinite(p).all() for p in params):
            raise ValueError("cached parameter nonfinite")
        losses.append(float(loss.detach()))
    return {
        "wall_seconds": time.perf_counter() - started,
        "losses": losses,
        "rank_updates": rank_updates,
    }


def curvature(head, train, val):
    with torch.no_grad():
        unit_train, unit_val = F.normalize(train, dim=1), F.normalize(val, dim=1)
        design = torch.cat((unit_train, torch.ones(len(train), 1)), 1).double()
        residual = head.residual(unit_train).double()
        fit = torch.linalg.lstsq(design, residual, driver="gelsd").solution

        def fraction(unit):
            residual = head.residual(unit).double()
            energy = residual.square().sum()
            if energy <= 0:
                return None
            design = torch.cat((unit, torch.ones(len(unit), 1)), 1).double()
            return float((residual - design @ fit).square().sum() / energy)

        return {"train": fraction(unit_train), "validation": fraction(unit_val)}


def aggregate(results, labels):
    deltas = {}
    for metric, key in (("r1", "per_query_r1"), ("map", "per_query_ap")):
        deltas[metric] = np.mean(
            [
                np.asarray(r["arms"]["nonlinear"]["quality"][key])
                - np.asarray(r["arms"]["control"]["quality"][key])
                for r in results
            ],
            axis=0,
        )
    clusters = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    sums = np.array([deltas["r1"][rows].sum() for rows in clusters])
    counts = np.array([len(rows) for rows in clusters])
    rng = np.random.Generator(np.random.PCG64(179031))
    samples = []
    for _ in range(5000):
        chosen = rng.integers(len(clusters), size=len(clusters))
        samples.append(float(sums[chosen].sum() / counts[chosen].sum()))
    lower, upper = np.quantile(samples, [0.025, 0.975]).tolist()
    nonnegative = all(
        r["arms"]["nonlinear"]["quality"][metric] >= r["base"][metric]
        for r in results
        for metric in ("recall_at_1", "map_at_r")
    ) and all(
        r["arms"]["nonlinear"]["quality"]["recall_at_1"]
        >= r["arms"]["control"]["quality"]["recall_at_1"]
        for r in results
    )
    return {
        "mean_paired_r1_gain": float(deltas["r1"].mean()),
        "r1_product_bootstrap_95ci": [lower, upper],
        "mean_paired_map_gain": float(deltas["map"].mean()),
        "each_seed_nonnegative": nonnegative,
        "advance_image_proposal": bool(
            deltas["r1"].mean() >= 0.0025
            and lower > 0
            and deltas["map"].mean() >= 0.003
            and nonnegative
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("cached filter requires CUDA hidden")
    torch.set_num_threads(8)
    if args.self_test:
        self_test()
        return
    started = time.perf_counter()
    inputs = load_inputs(args.partition, args.cache)
    train, target, val, val_labels, queries, primary, classifier, batches, models, manifest = inputs
    if args.seed is None:
        save(args.preflight, manifest)
        print(json.dumps(manifest))
        return
    if json.loads(args.preflight.read_text()) != manifest:
        raise ValueError("preflight authority differs")
    previous = [json.loads(p.read_text()) for p in args.previous]
    assert [r["seed"] for r in previous] == list(SEEDS[: SEEDS.index(args.seed)])
    assert all(r["manifest"] == manifest for r in previous)
    if previous:
        assert previous[0]["advance_seed17"] is True
    nonlinear = models[args.seed]
    control = copy.deepcopy(nonlinear)
    control.nonlinear = False
    assert torch.equal(nonlinear(train), control(train))
    base = score(lambda x: primary(F.normalize(x, dim=1)), val, val_labels, queries)
    reports = {}
    # One frozen arm order; later seeds reverse it to expose process-order effects.
    for name, head in (
        (("control", control), ("nonlinear", nonlinear))
        if args.seed != 23
        else (("nonlinear", nonlinear), ("control", control))
    ):
        reports[name] = train_arm(head, classifier, train, target, batches[args.seed])
        reports[name]["quality"] = score(head, val, val_labels, queries)
    weights = {
        f"{arm}.{name}": value.detach().numpy()
        for arm, model in (("control", control), ("nonlinear", nonlinear))
        for name, value in model.state_dict().items()
    }
    weights_path = args.output.with_suffix(".weights.npz")
    with weights_path.open("xb") as stream:
        np.savez(stream, **weights)
    curve = curvature(nonlinear, train, val)
    n, c = (reports[name]["quality"] for name in ("nonlinear", "control"))
    first_gate = (
        n["recall_at_1"] > c["recall_at_1"]
        and n["recall_at_1"] > base["recall_at_1"]
        and n["map_at_r"] - c["map_at_r"] >= 0.002
        and n["map_at_r"] >= base["map_at_r"]
    )
    combined = (
        aggregate(previous + [{"arms": reports, "base": base}], val_labels[queries].numpy())
        if args.seed == 29
        else None
    )
    if time.perf_counter() - started > 180:
        raise TimeoutError("whole-process seed-pair cap exceeded")
    save(
        args.output,
        {
            "manifest": manifest,
            "seed": args.seed,
            "base": base,
            "arms": reports,
            "curvature": curve,
            "advance_seed17": first_gate if args.seed == 17 else None,
            "weights_sha256": sha(weights_path),
            "whole_wall_seconds": time.perf_counter() - started,
            "claim_eligible": False,
            "aggregate": combined,
        },
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "base_r1": base["recall_at_1"],
                "nonlinear_r1": n["recall_at_1"],
                "control_r1": c["recall_at_1"],
                "advance_seed17": first_gate,
                "curvature": curve,
            }
        )
    )


if __name__ == "__main__":
    main()
