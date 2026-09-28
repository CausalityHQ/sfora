#!/usr/bin/env python3
"""Fixed three-seed centroid initializer confirmation; one serial GPU campaign."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from score_inshop_crop_view_pair import GALLERY_SHA, QUERY_SHA, bootstrap_lower, roles, sha256
from train_inshop_siglip2_unseen_gallery import source_manifest

from sfora.unicom_inshop import parse_inshop_partition

SEEDS = (179028, 179029, 179030)
QUALIFIED_SHA = "d6f0d09de37c9121f6358a95514fb312f04d56800d719045b6e09132c508c214"
PARITY = {
    "same_parent_vision",
    "same_folded_head",
    "exact_codes",
    "exact_inverse_norms",
    "exact_live_codes",
    "exact_live_inverse_norms",
}
PAIRED = (
    "source_sha256",
    "source_files_sha256",
    "fit_rows_sha256",
    "held_rows_sha256",
    "executed_schedule_sha256",
    "first_input_batch_sha256",
    "features_sha256",
    "preflight_sha256",
    "model_file_sha256",
    "export_batch_size",
)


def validate_history(receipt, seed):
    if receipt["seed"] != seed or receipt["updates"] != 1000:
        raise ValueError("confirmation seed/update differs")
    for key in ("all_step_losses", "step_seconds", "preclip_grad_norms"):
        values = np.asarray(receipt[key])
        if values.shape != (1000,) or not np.isfinite(values).all():
            raise ValueError(f"confirmation {key} inventory differs")
    if len(receipt["first_input_batch_sha256"]) != 1000 or any(
        len(value) != 64 for value in receipt["first_input_batch_sha256"]
    ):
        raise ValueError("confirmation pixel inventory differs")
    history = receipt["width_history"]
    if len(history) != 1000 or any(
        set(row["group_gradient_norms"]) != {"vision", "head", "classifier"}
        or not all(
            np.isfinite(value) and value > 0 for value in row["group_gradient_norms"].values()
        )
        for row in history
    ):
        raise ValueError("confirmation gradient history differs")
    parity = receipt["matched_public32_parity"]
    if not parity.keys() >= PARITY or not all(parity.values()):
        raise ValueError("confirmation live parity differs")


def aggregate(pairs, labels):
    if len(pairs) != 3:
        raise ValueError("confirmation aggregate requires all three seeds")
    report = {}
    for metric, vector in (("r1", "per_query_r1"), ("map", "per_query_ap")):
        deltas = np.asarray(
            [
                np.asarray(pair["products"][vector]) - np.asarray(pair["control"][vector])
                for pair in pairs
            ]
        )
        if deltas.shape != (3, len(labels)) or not np.isfinite(deltas).all():
            raise ValueError("confirmation aligned quality vectors differ")
        effects = deltas.mean(axis=1)
        mean = float(effects.mean())
        radius = 4.3026527299 * float(effects.std(ddof=1)) / np.sqrt(3)
        pooled = deltas.mean(axis=0)
        report[metric] = {
            "seed_effects": effects.tolist(),
            "mean": mean,
            "seed_t95": [mean - radius, mean + radius],
            "conditional_product95": [
                bootstrap_lower(pooled, labels),
                -bootstrap_lower(-pooled, labels),
            ],
        }
    report["go"] = bool(
        all(
            min(report[key]["seed_effects"]) >= 0 and report[key]["conditional_product95"][0] > 0
            for key in ("r1", "map")
        )
        and report["map"]["mean"] >= 0.005
        and report["map"]["seed_t95"][0] > 0
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "dataset-root",
        "model-snapshot",
        "features-dir",
        "preflight-dir",
        "qualification",
        "cached-gate",
        "native-control",
        "gate",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.output_dir.exists() or sha256(args.qualification) != QUALIFIED_SHA:
        raise ValueError("confirmation authority/output differs")
    preflights = {seed: args.preflight_dir / f"{seed}.json" for seed in SEEDS}
    for seed, path in preflights.items():
        if json.loads(path.read_text())["seed"] != seed:
            raise ValueError("confirmation preflight seed differs")
    training_sources = source_manifest()
    sources = dict(training_sources)
    for filename in (
        __file__,
        "run_inshop_centroid_pca_smoke.py",
        "compare_inshop_sop_warmstart_100.py",
        "score_inshop_crop_view_pair.py",
    ):
        path = Path(__file__).with_name(Path(filename).name).resolve()
        sources[str(path)] = sha256(path)
    frozen = {
        "preflight_sha256": {str(seed): sha256(path) for seed, path in preflights.items()},
        "gate_sha256": sha256(args.gate),
        "qualification_sha256": QUALIFIED_SHA,
        "source_sha256": sources,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "freeze.json").write_text(json.dumps(frozen, sort_keys=True) + "\n")
    result = {
        "schema": "sfora-inshop-centroid-pca-confirmation-v1",
        "claim_eligible": False,
        "decision": "RUNNING",
        "seeds": {str(seed): {"status": "unrun", "arms": {}, "quality": {}} for seed in SEEDS},
        "freeze": frozen,
    }
    started = time.perf_counter()

    def save():
        result["campaign_wall_seconds"] = time.perf_counter() - started
        (args.output_dir / "receipt.json").write_text(
            json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
        )

    def run(command, name, timeout):
        with (args.output_dir / f"{name}.log").open("wb") as log:
            subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=min(timeout, max(0.1, 7000 - (time.perf_counter() - started))),
            )

    base = []
    for name in ("dataset-root", "model-snapshot", "features-dir"):
        base += [f"--{name}", str(getattr(args, name.replace("-", "_")))]
    active = "smoke"
    save()
    try:
        smoke_dir = args.output_dir / "smoke"
        run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_inshop_centroid_pca_smoke.py")),
                *base,
                "--preflight",
                str(preflights[179028]),
                "--cached-gate",
                str(args.cached_gate),
                "--native-control",
                str(args.native_control),
                "--confirmation",
                str(args.qualification),
                "--output-dir",
                str(smoke_dir),
            ],
            "smoke",
            125,
        )
        smoke = json.loads((smoke_dir / "receipt.json").read_text())
        result["smoke_sha256"] = sha256(smoke_dir / "receipt.json")
        if smoke["decision"] != "GO_FROZEN_SEED_GATE":
            raise ValueError("fresh smoke failed")
        train = tuple(
            row for row in parse_inshop_partition(args.dataset_root) if row.split == "train"
        )
        _, held = split(tuple(row.label for row in train))
        if digest_rows(held) != "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b":
            raise ValueError("confirmation held authority differs")
        labels = tuple(train[row].label for row in held)
        query, gallery = roles(
            labels, tuple(train[row].image_path for row in held), args.dataset_root
        )
        if (
            len(query) != 6354
            or len(gallery) != 6245
            or any(
                hashlib.sha256(np.asarray(rows, dtype="<i4").tobytes()).hexdigest() != expected
                for rows, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA))
            )
        ):
            raise ValueError("confirmation packed roles differ")
        qlabels = np.asarray([labels[row] for row in query])
        for seed in SEEDS:
            state = result["seeds"][str(seed)]
            state["status"] = "running"
            for name in ("products", "control") if seed == 179029 else ("control", "products"):
                active = f"{seed}-{name}"
                destination = args.output_dir / str(seed) / name
                destination.parent.mkdir(exist_ok=True)
                torch.cuda.empty_cache()
                run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py")),
                        *base,
                        "--preflight",
                        str(preflights[seed]),
                        "--preflight-sha256",
                        frozen["preflight_sha256"][str(seed)],
                        "--arm",
                        "freeze_emb",
                        "--updates",
                        "1000",
                        "--seed",
                        str(seed),
                        "--centroid-pca-smoke",
                        name,
                        "--centroid-pca-confirmation",
                        str(args.qualification),
                        "--output-dir",
                        str(destination),
                    ],
                    active,
                    1100,
                )
                receipt_path = destination / "receipt.json"
                receipt = json.loads(receipt_path.read_text())
                state["arms"][name] = receipt
                state.setdefault("receipt_sha256", {})[name] = sha256(receipt_path)
                save()
                validate_history(receipt, seed)
                if (
                    receipt["source_files_sha256"] != training_sources
                    or receipt["preflight_sha256"] != frozen["preflight_sha256"][str(seed)]
                ):
                    raise ValueError("confirmation frozen source/preflight differs")
                if seed == 179028:
                    reference = smoke["arms"][name]
                    if receipt["first_input_batch_sha256"][:17] != reference[
                        "first_input_batch_sha256"
                    ] or not np.allclose(
                        receipt["all_step_losses"][:17],
                        reference["all_step_losses"],
                        rtol=0,
                        atol=1e-5,
                    ):
                        raise ValueError("fresh smoke trajectory differs")
                values_path = destination / "held_values.npy"
                if (
                    sha256(values_path) != receipt["held_values_sha256"]
                    or sha256(destination / "checkpoint.pt") != receipt["checkpoint_sha256"]
                ):
                    raise ValueError("confirmation checkpoint/export authority differs")
                values = np.load(values_path, allow_pickle=False)
                if values.shape != (12599, 128) or not np.isfinite(values).all():
                    raise ValueError("confirmation export inventory differs")
                state["quality"][name] = packed_quality(values, labels, query, gallery)
                save()
                if name == "control" and (
                    state["quality"][name]["recall_at_1"] < 0.952470884482216
                    or state["quality"][name]["map_at_r"] < 0.7798550374654079
                ):
                    raise ValueError("invalid full-budget native control")
                print(
                    json.dumps(
                        {
                            "terminal": active,
                            "r1": state["quality"][name]["recall_at_1"],
                            "map": state["quality"][name]["map_at_r"],
                        }
                    ),
                    flush=True,
                )
            control, product = (state["arms"][name] for name in ("control", "products"))
            if any(control[key] != product[key] for key in PAIRED):
                raise ValueError("confirmation paired inputs differ")
            for key in ("recall_at_1", "map_at_r"):
                if state["quality"]["products"][key] < state["quality"]["control"][key]:
                    raise ValueError("negative paired seed quality")
            criteria = {
                "train": product["training_wall_including_member_bank_init_seconds"]
                <= 1.05 * control["training_wall_including_member_bank_init_seconds"],
                "step": np.median(product["step_seconds"][1:])
                <= 1.05 * np.median(control["step_seconds"][1:]),
                "cuda": product["training_peak_cuda_allocated_bytes"]
                <= 1.005 * control["training_peak_cuda_allocated_bytes"],
                "wall": product["whole_arm_wall_seconds"]
                <= 1.10 * control["whole_arm_wall_seconds"],
                "geometry": all(
                    arm["width_terminal_geometry"][key]
                    >= fraction * arm["width_initial_geometry"][key]
                    for arm in (control, product)
                    for key, fraction in (("variance", 0.5), ("effective_rank", 0.8))
                ),
            }
            state["criteria"] = {key: bool(value) for key, value in criteria.items()}
            if not all(criteria.values()):
                raise ValueError("confirmation resource/geometry gate failed")
            state["status"] = "complete"
            save()
        if (
            any(sha256(Path(path)) != digest for path, digest in sources.items())
            or sha256(args.gate) != frozen["gate_sha256"]
        ):
            raise ValueError("confirmation frozen sources changed")
        result["aggregate"] = aggregate(
            [result["seeds"][str(seed)]["quality"] for seed in SEEDS], qlabels
        )
        if time.perf_counter() - started > 7000:
            raise ValueError("confirmation campaign budget exceeded")
        result["decision"] = (
            "GO_PRODUCTION_QUALIFICATION_DESIGN" if result["aggregate"]["go"] else "KILL"
        )
    except Exception as error:
        result.update(
            decision="KILL", failed_stage=active, error=f"{type(error).__name__}: {error}"
        )
        for state in result["seeds"].values():
            if state["status"] == "running":
                state["status"] = "failed"
        save()
        raise
    save()
    print(
        json.dumps(
            {key: result[key] for key in ("decision", "campaign_wall_seconds", "aggregate")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
