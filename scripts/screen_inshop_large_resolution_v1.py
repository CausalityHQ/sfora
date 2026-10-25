#!/usr/bin/env python3
"""Zero-training In-Shop screen: pretrained SigLIP2 Large/16 at 384 px vs its native 256 px.

Why: at matched 256 px, fine-tuned So400m and Large tie on In-Shop official (96.244 vs 96.230,
native256-full-v1). The So400m win came from 384 px (screen 447bddc4: +5.67 pretrained; official
+0.56/+0.61 vs So400m@256). Large@384 (576 tokens, 24 x 1024) costs about 0.55x the So400m@384
encoder FLOPs. If resolution transfers to Large, a fine-tuned Large@384 could be a roughly 1.7x
faster point near the So400m@384 quality.

Pre-registered (committed before any number exists):
  rows/split: identical to screen_inshop_so400m_resolution_v1 (In-Shop TRAIN only, ordered-rows sha
              2d43ad5c..., singleton products always fit, fit 0.5 seed 179019); no query/gallery.
  arms      : google/siglip2-large-patch16-256 @ 787800c8, fp16, pooled features; r256 (native),
              r384 (processor 384, interpolated position embeddings; at 256 interpolation is identity).
  metric    : unchanged score_corpus packed_pca128 full_train_gallery R@1.
  ADVANCE iff r384 - r256 >= +1.0 pp AND packed holdout_only R@1 is also higher.
              => a fine-tuned In-Shop Large@384 512-D pilot vs the existing So400m@384 pilot.
  Reported, not gating: Large r384 vs So400m r384 (83.13 / 87.03), encode img/s of each arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from score_sop_pretrained_substrate import score_corpus
from screen_inshop_so400m_resolution_v1 import export_arm
from sfora.representation_ceiling import deterministic_class_partition
from sfora.unicom_inshop import parse_inshop_partition

LARGE_REVISION = '787800c8990e6f058423089178e718139608408c'
ORDERED_ROWS_SHA256 = '2d43ad5cee694b2f38d7e3d806c5f983b9161ede48b54bd9cc7841b31d53ee5c'
ADVANCE_PP = 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--large-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    assert torch.cuda.is_available() and not args.output_dir.exists()
    assert args.large_snapshot.resolve().name == LARGE_REVISION
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [r for r in parse_inshop_partition(args.dataset_root) if r.split == 'train']
    ordered = hashlib.sha256('\n'.join(
        f'{r.image_path.relative_to(args.dataset_root)}\0{r.label}' for r in rows).encode()).hexdigest()
    assert ordered == ORDERED_ROWS_SHA256
    ids = {label: i for i, label in enumerate(sorted({r.label for r in rows}))}
    labels = np.asarray([ids[r.label] for r in rows], dtype=np.int64)
    counts = np.bincount(labels)
    multi = np.flatnonzero(counts[labels] >= 2)
    part = deterministic_class_partition(tuple(map(int, labels[multi])), fit_fraction=0.5, seed=179019)
    held = multi[np.asarray(part.validation_row_indexes, dtype=np.int64)]
    fit = np.setdiff1d(np.arange(len(rows), dtype=np.int64), held)
    assert (len(fit), len(held)) == (12_761, 13_121)
    args.output_dir.mkdir(parents=True)
    result = {'schema': 'inshop-large-resolution-screen-v1', 'official_read': False, 'ordered_rows_sha256': ordered,
              'fit_rows': len(fit), 'held_rows': len(held)}
    for name, crop in (('r256', 256), ('r384', 384)):
        out = args.output_dir / f'large_{name}_train_features.npy'
        result[f'{name}_export'] = export_arm(rows, args.large_snapshot, crop, out, args.batch_size)
        arm = score_corpus(name, np.load(out, allow_pickle=False), labels, fit, held)
        result[name] = {k: {g: arm['score'][k][g]['recall_at_1'] for g in arm['score'][k]} for k in arm['score']}
        print(json.dumps({name: result[name], 'export': result[f'{name}_export']}), flush=True)
    full = (result['r384']['packed_pca128']['full_train_gallery'] - result['r256']['packed_pca128']['full_train_gallery']) * 100
    hold = (result['r384']['packed_pca128']['holdout_only'] - result['r256']['packed_pca128']['holdout_only']) * 100
    result['delta_pp'] = {'packed_full_train_gallery_r1': full, 'packed_holdout_r1': hold,
                          'large384_minus_so400m384_full_train_gallery': result['r384']['packed_pca128']['full_train_gallery'] * 100 - 83.13390747656428}
    result['decision'] = 'ADVANCE' if full >= ADVANCE_PP and hold > 0 else 'CLOSE'
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


if __name__ == '__main__':
    sys.exit(main())
