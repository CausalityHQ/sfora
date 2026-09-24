"""Matched frozen-B16 SOP head screen for in-batch versus fit-bank negatives.

Exploratory train-only control. This cannot establish a full-backbone method gain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_evaluation import score_symmetric
from sfora.unicom_rank_finish import identity_balanced_batches

ARCHIVE_SHA = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
STEPS = 256
MARGIN = 0.05
SCALE = 16.0
LR = 1e-4


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(model: nn.Module, rows: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    with torch.inference_mode():
        values = F.normalize(model(rows), dim=1).float().contiguous()
        float_score = score_symmetric(values, labels)
        packed = pack_int8_unit_embeddings(values)
        packed_score = score_symmetric(
            packed.codes.float(), labels, inverse_norms=packed.inverse_norms
        )
    return {
        arm: {
            "recall_at_1": score["recall_at_1"],
            "map_at_r": score["map_at_r"],
            "per_query_r1": score["per_query_r1"],
            "per_query_ap": score["per_query_ap"],
        }
        for arm, score in (("float", float_score), ("packed", packed_score))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or digest(args.source_archive) != ARCHIVE_SHA:
        raise ValueError("SOP source or output authority differs")
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    with np.load(args.source_archive, allow_pickle=False) as archive:
        source = np.asarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    if source.shape != (59_551, 768) or labels.shape != (59_551,):
        raise ValueError("SOP pretrained source inventory differs")
    source = np.ascontiguousarray(source / np.linalg.norm(source, axis=1, keepdims=True))
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit_ids = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    holdout_ids = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit_ids) != 53_700 or len(holdout_ids) != 5_851:
        raise ValueError("SOP fit/holdout inventory differs")
    fit = torch.from_numpy(np.ascontiguousarray(source[fit_ids]))
    held = torch.from_numpy(np.ascontiguousarray(source[holdout_ids]))
    fit_labels = labels[fit_ids]
    held_labels = torch.from_numpy(np.ascontiguousarray(labels[holdout_ids]))
    schedule = identity_balanced_batches(
        tuple(map(str, fit_labels)),
        batch_size=64,
        images_per_identity=4,
        seed=SEED,
        epoch=1,
        steps=STEPS,
    )
    schedule_array = np.asarray(schedule, dtype=np.int64)
    schedule_sha = hashlib.sha256(schedule_array.astype("<i4").tobytes()).hexdigest()
    fit_labels_t = torch.from_numpy(np.ascontiguousarray(fit_labels))
    positive_rows: list[torch.Tensor] = []
    inbatch_negatives: list[torch.Tensor] = []
    bank_negatives: list[torch.Tensor] = []
    mining_started = time.perf_counter()
    for batch_tuple in schedule:
        batch = torch.tensor(batch_tuple, dtype=torch.int64)
        batch_labels = fit_labels_t[batch]
        batch_scores = fit[batch] @ fit[batch].T
        same = batch_labels[:, None] == batch_labels[None, :]
        distinct = batch[:, None] != batch[None, :]
        if not bool((same & distinct).any(dim=1).all()):
            raise ValueError("SOP batch positive inventory differs")
        positive_rows.append(
            batch[(batch_scores.masked_fill(~(same & distinct), torch.inf)).argmin(dim=1)]
        )
        inbatch_negatives.append(batch[(batch_scores.masked_fill(same, -torch.inf)).argmax(dim=1)])
        scores = fit[batch] @ fit.T
        scores.masked_fill_(batch_labels[:, None] == fit_labels_t[None, :], -torch.inf)
        bank_negatives.append(scores.argmax(dim=1))
    mining_seconds = time.perf_counter() - mining_started
    del source

    def fresh_head() -> nn.Linear:
        head = nn.Linear(768, 768, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.eye(768))
        return head

    baseline = evaluate(fresh_head(), held, held_labels)
    results: dict[str, object] = {}
    for name, negatives in (("inbatch", inbatch_negatives), ("fit_bank", bank_negatives)):
        head = fresh_head()
        optimizer = torch.optim.Adam(head.parameters(), lr=LR)
        arm_started = time.perf_counter()
        last_loss = 0.0
        for step, batch_tuple in enumerate(schedule):
            anchor = torch.tensor(batch_tuple, dtype=torch.int64)
            q = F.normalize(head(fit[anchor]), dim=1)
            pos = F.normalize(head(fit[positive_rows[step]]), dim=1)
            neg = F.normalize(head(fit[negatives[step]]), dim=1)
            pos_sim = (q * pos).sum(dim=1)
            neg_sim = (q * neg).sum(dim=1)
            loss = F.softplus((neg_sim - pos_sim + MARGIN) * SCALE).mean() / SCALE
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
        training_seconds = time.perf_counter() - arm_started
        results[name] = {
            "score": evaluate(head, held, held_labels),
            "training_seconds": training_seconds,
            "last_loss": last_loss,
            "head_weight_sha256": hashlib.sha256(
                head.weight.detach().numpy().astype("<f4").tobytes()
            ).hexdigest(),
        }
        print(
            name,
            json.dumps({"last_loss": last_loss, "training_seconds": training_seconds}),
            flush=True,
        )
    receipt = {
        "schema": "sfora-sop-frozen-b16-bank-negative-head-screen-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": (
            "official train; fit products for training, disjoint held-out products for selection"
        ),
        "source_archive_sha256": ARCHIVE_SHA,
        "script_sha256": digest(Path(__file__)),
        "seed": SEED,
        "steps": STEPS,
        "schedule_sha256": schedule_sha,
        "architecture": (
            "frozen pretrained UNICOM B16@224; trainable identity-initialized 768x768 linear head"
        ),
        "loss": f"softplus({SCALE}*(cos(negative)-cos(positive)+{MARGIN}))/{SCALE}",
        "positive_mining": (
            "source-feature least-similar distinct same-product row within each scheduled batch"
        ),
        "negative_mining": (
            "source-feature highest cosine wrong-product row; in batch or entire fit bank"
        ),
        "learning_rate": LR,
        "optimizer": "Adam, zero weight decay, gradient norm clip 1",
        "fit_rows": len(fit_ids),
        "holdout_rows": len(holdout_ids),
        "gallery_bytes_per_item": 770,
        "mining_seconds": mining_seconds,
        "total_seconds": time.perf_counter() - started,
        "baseline": baseline,
        "arms": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps({"output": str(args.output), "total_seconds": receipt["total_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
