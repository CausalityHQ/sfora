#!/usr/bin/env python3
"""Zero-training SOP substrate screen: pretrained SigLIP2 giant-opt/16@256 vs So400m/14@384.

Pre-registered (committed before any giant SOP number exists):
  rows/split: SOP official TRAIN (59,551), product-disjoint seed 179019 fit 0.9 (53,700 / 5,851), as
              screen_sop_so400m_pretrained_v1.py.
  arms      : So400m = existing export sfora-sop-so400m-screen-v1-a2 (features 13e75098...);
              giant  = fresh export of google/siglip2-giant-opt-patch16-256 @ 46d7129e here (same export code).
  scoring   : unchanged score_corpus (fit-only PCA128, signed-int8 packing), both arms.
  metric    : packed_pca128 full_train_gallery R@1 (held queries vs all 59,551 TRAIN images).
  ADVANCE iff giant - So400m >= +1.0 pp on that metric AND packed holdout_only R@1 also higher.
              Rationale: pretrained Large->So400m +2.17 became +1.47 official after fine-tuning; a +1.0
              pretrained gap is the smallest that plausibly survives fine-tuning at a cost of ~2x encode.
  ADVANCE => a fine-tuned giant vs So400m TRAIN pair (separately designed); else the giant lane closes.
  No official TEST read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from export_sop_siglip2_train import UNICOM_TRAIN_ARCHIVE_SHA256, authenticated_train_rows, sha256
from score_sop_pretrained_substrate import score_corpus
from screen_sop_so400m_pretrained_v1 import export_so400 as export_features_fp16
from sfora.representation_ceiling import deterministic_class_partition

GIANT_REVISION = "46d7129e1aa1527cc5a44d86fb35250df3abf0aa"
SO400_FEATURES_SHA256 = "13e750983bd7b375dddbe5984ae4fa69201073a1628ddcaf58bd1779a07c5790"
ADVANCE_PP = 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-train-archive", "dataset-root", "giant-snapshot", "so400-features", "output-dir"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    assert torch.cuda.is_available() and not a.output_dir.exists()
    assert a.giant_snapshot.resolve().name == GIANT_REVISION
    assert sha256(a.unicom_train_archive) == UNICOM_TRAIN_ARCHIVE_SHA256
    assert sha256(a.so400_features) == SO400_FEATURES_SHA256
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(a.unicom_train_archive, allow_pickle=False) as archive:
        rows = authenticated_train_rows(archive, a.dataset_root)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    assert len(rows) == 59_551
    part = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
    fit = np.asarray(part.fit_row_indexes, dtype=np.int64)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    assert (len(fit), len(held)) == (53_700, 5_851)
    a.output_dir.mkdir(parents=True)
    export = export_features_fp16(rows, a.giant_snapshot, a.output_dir / "giant_train_features.npy", a.batch_size)
    print(json.dumps({"export": export}), flush=True)
    feats = {"so400": np.load(a.so400_features, allow_pickle=False),
             "giant": np.load(a.output_dir / "giant_train_features.npy", allow_pickle=False)}
    result = {"schema": "sop-giant-pretrained-screen-v1", "official_read": False, "export": export,
              "ordered_rows_sha256": hashlib.sha256("\n".join(
                  f"{r.image_id}\0{r.label}\0{r.relative_path}" for r in rows).encode()).hexdigest()}
    for name, f in feats.items():
        arm = score_corpus(name, f, labels, fit, held)
        result[name] = {k: {g: arm["score"][k][g]["recall_at_1"] for g in arm["score"][k]} for k in arm["score"]}
    full = (result["giant"]["packed_pca128"]["full_train_gallery"] - result["so400"]["packed_pca128"]["full_train_gallery"]) * 100
    hold = (result["giant"]["packed_pca128"]["holdout_only"] - result["so400"]["packed_pca128"]["holdout_only"]) * 100
    result["delta_pp"] = {"packed_full_train_gallery_r1": full, "packed_holdout_r1": hold}
    result["decision"] = "ADVANCE" if full >= ADVANCE_PP and hold > 0 else "CLOSE"
    (a.output_dir / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
