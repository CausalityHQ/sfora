#!/usr/bin/env python3
"""Zero-training SOP substrate screen: pretrained SigLIP2 So400m/14@384 vs Large/16@256.

Pre-registered (committed before any So400m SOP number exists):
  rows/split: authenticated SOP official TRAIN (59,551 rows), product-disjoint fit/held
              partition seed 179019 fit 0.9 (53,700 / 5,851), as score_sop_pretrained_substrate.
  arms      : Large = existing export sfora-siglip2-train-28e19203 (features d15f76e4...);
              So400m = fresh export of google/siglip2-so400m-patch14-384 @ e8e48729 here.
  scoring   : unchanged score_corpus (fit-only PCA128, signed-int8 packing), both arms.
  metric    : packed_pca128 full_train_gallery R@1 (held queries vs all 59,551 TRAIN images).
  ADVANCE iff So400m - Large >= +2.0 pp on that metric AND packed holdout_only R@1 is also
              higher. Rationale: fine-tuning kept ~half of the pretrained B->L gap on SOP, and
              the SOP fine-tuned incumbent's best recipe gain was +0.56 pp.
  ADVANCE => a separately reviewed fine-tuned SOP So400m vs Large pair; else the SOP
             substrate lane is closed. No official TEST read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from export_sop_siglip2_train import UNICOM_TRAIN_ARCHIVE_SHA256, authenticated_train_rows, export_features, sha256
from score_sop_pretrained_substrate import score_corpus
from sfora.representation_ceiling import deterministic_class_partition

SO400_REVISION = 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'
LARGE_FEATURES_SHA256 = 'd15f76e459b90836df2807260a34d06692e2f2e5b8ba17b3449ed56500b8357a'
ADVANCE_PP = 2.0


def export_so400(rows, snapshot: Path, output: Path, batch_size: int) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True,
                                      dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def encode(batch):
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert('RGB'))
        inputs = processor(images=images, return_tensors='pt')
        out = model.get_image_features(**{k: v.cuda() for k, v in inputs.items() if torch.is_tensor(v)})
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        return out.float().cpu().numpy()

    started = time.perf_counter()
    width = int(encode(rows[:batch_size]).shape[1])
    export_features(rows, encode, output, width=width, batch_size=batch_size)
    torch.cuda.synchronize()
    return {'width': width, 'encode_wall_seconds': time.perf_counter() - started,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'features_sha256': sha256(output)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--unicom-train-archive', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--so400-snapshot', type=Path, required=True)
    p.add_argument('--large-features', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    assert torch.cuda.is_available() and not args.output_dir.exists()
    assert args.so400_snapshot.resolve().name == SO400_REVISION
    assert sha256(args.unicom_train_archive) == UNICOM_TRAIN_ARCHIVE_SHA256
    assert sha256(args.large_features) == LARGE_FEATURES_SHA256
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.unicom_train_archive, allow_pickle=False) as archive:
        rows = authenticated_train_rows(archive, args.dataset_root)
        labels = np.asarray(archive['train_labels'], dtype=np.int64)
    assert len(rows) == 59_551
    part = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
    fit = np.asarray(part.fit_row_indexes, dtype=np.int64)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    assert (len(fit), len(held)) == (53_700, 5_851)
    args.output_dir.mkdir(parents=True)
    export = export_so400(rows, args.so400_snapshot, args.output_dir / 'so400_train_features.npy', args.batch_size)
    print(json.dumps({'export': export}), flush=True)
    large = np.load(args.large_features, allow_pickle=False)
    so400 = np.load(args.output_dir / 'so400_train_features.npy', allow_pickle=False)
    result = {'schema': 'sop-so400m-pretrained-screen-v1', 'official_read': False, 'export': export,
              'ordered_rows_sha256': hashlib.sha256('\n'.join(
                  f'{r.image_id}\0{r.label}\0{r.relative_path}' for r in rows).encode()).hexdigest()}
    for name, feats in (('large', large), ('so400', so400)):
        arm = score_corpus(name, feats, labels, fit, held)
        result[name] = {k: {g: arm['score'][k][g]['recall_at_1'] for g in arm['score'][k]} for k in arm['score']}
    full = (result['so400']['packed_pca128']['full_train_gallery'] - result['large']['packed_pca128']['full_train_gallery']) * 100
    hold = (result['so400']['packed_pca128']['holdout_only'] - result['large']['packed_pca128']['holdout_only']) * 100
    result['delta_pp'] = {'packed_full_train_gallery_r1': full, 'packed_holdout_r1': hold}
    result['decision'] = 'ADVANCE' if full >= ADVANCE_PP and hold > 0 else 'CLOSE'
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
