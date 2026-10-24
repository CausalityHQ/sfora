#!/usr/bin/env python3
"""Zero-training In-Shop resolution screen: pretrained SigLIP2 So400m/14 at 256 px vs native 384 px.

Why: every In-Shop So400m run (native256 campaign, official 96.244) fed 256x256 pixels to a
patch14-384 model (interpolated position embeddings). On SOP, training/eval 280 -> 384 gave
+0.41 held / +0.49 official R@1 (7e12c180, 21102577), and the UNICOM In-Shop reference upsamples
256 px images to 336. The 2026-09-30 "288 px only supersamples" rejection was a premise, never measured.

Pre-registered (committed before any number exists):
  rows/split: In-Shop official TRAIN only (25,882 rows, 3,997 products; query/gallery never read),
              product-disjoint fit/held partition seed 179019 fit 0.5.
  arms      : google/siglip2-so400m-patch14-384 @ e8e48729, fp16, pooled image features;
              r256 = processor size 256 + interpolated pos-encoding (as the native256 campaign);
              r384 = unchanged processor/model (native).
  scoring   : unchanged score_corpus (fit-only PCA128, signed-int8 packing), both arms.
  metric    : packed_pca128 full_train_gallery R@1 (held queries vs all 25,882 TRAIN images).
  ADVANCE iff r384 - r256 >= +1.0 pp on that metric AND packed holdout_only R@1 is also higher.
              Rationale: fine-tuning kept ~half of pretrained gaps on SOP; +0.5 fine-tuned would
              close the 0.39 pp gap to the dated 96.7 reference.
  ADVANCE => a fine-tuned In-Shop So400m@384 TRAIN pilot; else In-Shop resolution lane closed.
  No official query/gallery read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from export_sop_siglip2_train import export_features, sha256
from score_sop_pretrained_substrate import score_corpus
from sfora.representation_ceiling import deterministic_class_partition
from sfora.unicom_inshop import parse_inshop_partition

SO400_REVISION = 'e8e487298228002f3d8a82e0cd5c8ea9c567f57f'
ADVANCE_PP = 1.0


def export_arm(rows, snapshot: Path, crop: int | None, output: Path, batch_size: int) -> dict:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True)
    if crop is not None:
        processor.size = {'height': crop, 'width': crop}
    model = AutoModel.from_pretrained(snapshot, local_files_only=True, use_safetensors=True,
                                      dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def encode(batch):
        images = []
        for row in batch:
            with Image.open(row.image_path) as image:
                images.append(image.convert('RGB'))
        inputs = processor(images=images, return_tensors='pt')
        out = model.get_image_features(pixel_values=inputs['pixel_values'].cuda().half(),
                                       interpolate_pos_encoding=crop is not None)
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        return out.float().cpu().numpy()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    width = int(encode(rows[:batch_size]).shape[1])
    export_features(rows, encode, output, width=width, batch_size=batch_size)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    del model
    torch.cuda.empty_cache()
    return {'width': width, 'encode_wall_seconds': seconds, 'images_per_second': len(rows) / seconds,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'features_sha256': sha256(output),
            'pixels': crop or 384}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--so400-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=32)
    args = p.parse_args()
    assert torch.cuda.is_available() and not args.output_dir.exists()
    assert args.so400_snapshot.resolve().name == SO400_REVISION
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = [r for r in parse_inshop_partition(args.dataset_root) if r.split == 'train']
    assert len(rows) == 25_882
    ids = {label: i for i, label in enumerate(sorted({r.label for r in rows}))}
    labels = np.asarray([ids[r.label] for r in rows], dtype=np.int64)
    part = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.5, seed=179019)
    fit = np.asarray(part.fit_row_indexes, dtype=np.int64)
    held = np.asarray(part.validation_row_indexes, dtype=np.int64)
    args.output_dir.mkdir(parents=True)
    result = {'schema': 'inshop-so400m-resolution-screen-v1', 'official_read': False,
              'rows': len(rows), 'products': len(ids), 'fit_rows': len(fit), 'held_rows': len(held),
              'ordered_rows_sha256': hashlib.sha256('\n'.join(
                  f'{r.image_path.relative_to(args.dataset_root)}\0{r.label}' for r in rows).encode()).hexdigest()}
    for name, crop in (('r256', 256), ('r384', None)):
        out = args.output_dir / f'{name}_train_features.npy'
        result[f'{name}_export'] = export_arm(rows, args.so400_snapshot, crop, out, args.batch_size)
        print(json.dumps({name: result[f'{name}_export']}), flush=True)
        arm = score_corpus(name, np.load(out, allow_pickle=False), labels, fit, held)
        result[name] = {k: {g: arm['score'][k][g]['recall_at_1'] for g in arm['score'][k]} for k in arm['score']}
        print(json.dumps({name: result[name]}), flush=True)
    full = (result['r384']['packed_pca128']['full_train_gallery'] - result['r256']['packed_pca128']['full_train_gallery']) * 100
    hold = (result['r384']['packed_pca128']['holdout_only'] - result['r256']['packed_pca128']['holdout_only']) * 100
    result['delta_pp'] = {'packed_full_train_gallery_r1': full, 'packed_holdout_r1': hold}
    result['decision'] = 'ADVANCE' if full >= ADVANCE_PP and hold > 0 else 'CLOSE'
    (args.output_dir / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps(result, indent=1, sort_keys=True), flush=True)


if __name__ == '__main__':
    sys.exit(main())
