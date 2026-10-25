#!/usr/bin/env python3
"""Zero-training In-Shop screen: pretrained So400m/14 at 448 and 512 px vs its native 384.

Pre-registered before any number (quality lever: resolution beyond 384).
- Rows/split: identical to screen_inshop_so400m_resolution_v1 (TRAIN only, ordered sha 2d43ad5c...).
- Arms: r384 (identity interpolation), r448, r512. Position embeddings are interpolated.
- Metric: unchanged score_corpus packed_pca128 full_train_gallery R@1.
- ADVANCE iff max(r448, r512) - r384 >= +2.0 pp, AND that arm's packed holdout_only R@1 is also higher.
  Fine-tuning kept ~10% of the 256->384 pretrained gain (+5.67 -> +0.59 held).
  ADVANCE => fine-tuned In-Shop pilot at the winning resolution vs the 384 pilot.
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

from score_sop_pretrained_substrate import score_corpus
from screen_inshop_so400m_resolution_v1 import export_arm
from sfora.representation_ceiling import deterministic_class_partition
from sfora.unicom_inshop import parse_inshop_partition

ORDERED = '2d43ad5cee694b2f38d7e3d806c5f983b9161ede48b54bd9cc7841b31d53ee5c'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--so400-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    assert torch.cuda.is_available() and not a.output_dir.exists()
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [r for r in parse_inshop_partition(a.dataset_root) if r.split == 'train']
    assert hashlib.sha256('\n'.join(f'{r.image_path.relative_to(a.dataset_root)}\0{r.label}' for r in rows).encode()).hexdigest() == ORDERED
    ids = {l: i for i, l in enumerate(sorted({r.label for r in rows}))}
    labels = np.asarray([ids[r.label] for r in rows], dtype=np.int64)
    counts = np.bincount(labels)
    multi = np.flatnonzero(counts[labels] >= 2)
    part = deterministic_class_partition(tuple(map(int, labels[multi])), fit_fraction=0.5, seed=179019)
    held = multi[np.asarray(part.validation_row_indexes, dtype=np.int64)]
    fit = np.setdiff1d(np.arange(len(rows), dtype=np.int64), held)
    a.output_dir.mkdir(parents=True)
    res = {'schema': 'inshop-so400m-highres-screen-v1', 'official_read': False}
    for crop in (384, 448, 512):
        out = a.output_dir / f'r{crop}_train_features.npy'
        res[f'r{crop}_export'] = export_arm(rows, a.so400_snapshot, crop, out, 16)
        arm = score_corpus(f'r{crop}', np.load(out), labels, fit, held)
        res[f'r{crop}'] = {k: arm['score'][k][g]['recall_at_1'] for k in ('packed_pca128',) for g in ('full_train_gallery',)} | {
            'holdout': arm['score']['packed_pca128']['holdout_only']['recall_at_1']}
        print(json.dumps({crop: res[f'r{crop}'], 'img_s': res[f'r{crop}_export']['images_per_second']}), flush=True)
    base = res['r384']
    best = max((448, 512), key=lambda c: res[f'r{c}']['packed_pca128'])
    d = (res[f'r{best}']['packed_pca128'] - base['packed_pca128']) * 100
    res['best'], res['delta_pp'] = best, d
    res['decision'] = 'ADVANCE' if d >= 2.0 and res[f'r{best}']['holdout'] > base['holdout'] else 'CLOSE'
    (a.output_dir / 'result.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ('best', 'delta_pp', 'decision')}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
