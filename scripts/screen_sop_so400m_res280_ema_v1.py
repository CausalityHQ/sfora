#!/usr/bin/env python3
"""Score the EMA shadow of an SOP So400m@280 run against the same run's final weights (TRAIN-held only).

Pre-registered with train_sop_so400m_res280_ema_pilot.py (decision rule in its docstring).
  validity: the final checkpoint re-encoded on all TRAIN rows must reproduce the receipt's per-query R@1 exactly
            (as in the resolution and gallery-flip screens), else INVALID; bitwise equality with the run's saved
            train_embeddings.npy is reported as information.
  primary : EMA vs final, held 5,851 vs full 59,551 TRAIN gallery, packed int8 + native top-10 (evaluate_packed),
            paired product bootstrap 5000 seed 179019 on per-query R@1 and AP.
  report  : final (this seed) vs res280 seed 179024 receipt 9e4a05db, paired per query (seed replicate, not gating).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_res280_pilot as base  # noqa: E402

pair = base.pair
SEED179024_RECEIPT = "9e4a05db43879acffce11dcee99fc768bde08106f36d23311cf0c117c99b7be9"


def product_interval(d, products):
    _, inv = np.unique(products, return_inverse=True)
    s, n, k, rng = np.bincount(inv, d), np.bincount(inv), inv.max() + 1, np.random.default_rng(179019)
    draws = [(c * s).sum() / (c * n).sum() for c in (np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(5000))]
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def paired(a: dict, b: dict, products) -> dict:
    out = {}
    for key, metric in (("per_query_r1", "dR1"), ("per_query_ap", "dmAP")):
        d = np.asarray(a[key], float) - np.asarray(b[key], float)
        out[metric] = {"mean": float(d.mean()), "product_95": product_interval(d, products)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("run-dir", "seed179024-run-dir", "model-snapshot", "archive", "dataset-root", "native-library", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists() and torch.cuda.is_available()
    receipt = json.loads((a.run_dir / "receipt.json").read_text())
    sidecar = json.loads((a.run_dir / "ema_receipt.json").read_text())
    assert sidecar["training_receipt_sha256"] == pair.sha256(a.run_dir / "receipt.json")
    assert sidecar["ema_state_sha256"] == pair.sha256(a.run_dir / "ema_state.pt")
    assert pair.sha256(a.run_dir / "checkpoint.pt") == receipt["checkpoint_sha256"] == sidecar["checkpoint_sha256"]
    assert receipt["seed"] == 179025 and receipt["updates"] == 1000 and receipt["substrate_config"]["crop"] == 280
    assert pair.sha256(a.seed179024_run_dir / "receipt.json") == SEED179024_RECEIPT
    other = json.loads((a.seed179024_run_dir / "receipt.json").read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(a.model_snapshot, local_files_only=True, backend="torchvision")
    assert processor.size == {"height": 280, "width": 280}
    full = AutoModel.from_pretrained(a.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16)
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1152, 128).cuda().eval()
    final = torch.load(a.run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    vision.load_state_dict(final["vision"], strict=True)
    head.load_state_dict(final["head"], strict=True)
    with np.load(a.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    paths = pair.paths_from_archive(a.dataset_root, relatives)
    part = pair.deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=pair.FIT_SEED)
    fit = np.asarray(part.fit_row_indexes, dtype=np.int64)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    assert (len(fit), len(held)) == (53_700, 5_851)
    probe = pair.export_all(vision, head, paths, tuple(map(int, labels)), processor, workers=4, batch_size=64)
    probe_q = pair.evaluate_packed(probe, labels, fit, held, a.native_library)
    saved = np.load(a.run_dir / "train_embeddings.npy", mmap_mode="r")
    reproduced = probe_q["per_query_r1"] == receipt["quality"]["per_query_r1"]
    result = {"schema": "sop-so400m-res280-ema-screen-v1", "official_read": False,
              "ema": {k: sidecar[k] for k in ("decay", "steps", "ema_parameters", "ema_elements", "ema_state_sha256")},
              "validity_final_per_query_r1_reproduced": bool(reproduced),
              "info_final_embeddings_bitwise_equal": bool(np.array_equal(probe.numpy(), saved))}
    if not reproduced:
        result["decision"] = "INVALID: final re-encode does not reproduce the receipt per-query R@1"
    else:
        ema = torch.load(a.run_dir / "ema_state.pt", map_location="cpu", weights_only=True)
        vision.load_state_dict(ema["vision"], strict=True)
        head.load_state_dict(ema["head"], strict=True)
        values = pair.export_all(vision, head, paths, tuple(map(int, labels)), processor, workers=4, batch_size=64)
        q = pair.evaluate_packed(values, labels, fit, held, a.native_library)
        products = labels[held]
        result["final"] = {k: receipt["quality"][k] for k in ("recall_at_1", "map_at_r")}
        result["ema_quality"] = {k: q[k] for k in ("recall_at_1", "map_at_r", "native_top10_exact")}
        result["ema_minus_final"] = paired(q, receipt["quality"], products)
        result["seed_replicate_final179025_minus_final179024"] = paired(receipt["quality"], other["quality"], products)
        result["seed179024_final"] = {k: other["quality"][k] for k in ("recall_at_1", "map_at_r")}
        d = result["ema_minus_final"]
        advance = d["dR1"]["product_95"][0] > 0 and d["dmAP"]["product_95"][1] > 0
        result["decision"] = "ADVANCE EMA" if advance else "KILL EMA"
    a.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
