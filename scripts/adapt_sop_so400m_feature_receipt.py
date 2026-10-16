#!/usr/bin/env python3
"""Publish the SOP So400m screen features as a sfora-sop-siglip2-train-feature-export-v1 candidate dir.

Every receipt field is recomputed or copied from the verified screen output (no new encode):
features are hard-linked, rehashed and checked against the screen result; model files are hashed;
row identity is re-derived from the authoritative archive. `derived_from` names the screen result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from export_sop_siglip2_train import UNICOM_TRAIN_ARCHIVE_SHA256, sha256

MODEL_FILES = ('config.json', 'preprocessor_config.json', 'model.safetensors')


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--screen-dir', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--model-snapshot', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    result = json.loads((a.screen_dir / 'result.json').read_text())
    features = a.screen_dir / 'so400_train_features.npy'
    assert not a.output_dir.exists() and sha256(a.archive) == UNICOM_TRAIN_ARCHIVE_SHA256
    assert result['schema'] == 'sop-so400m-pretrained-screen-v1'
    assert sha256(features) == result['export']['features_sha256']
    with np.load(a.archive, allow_pickle=False) as archive:
        rows = hashlib.sha256('\n'.join(f'{int(i)}\0{int(y)}\0{r}' for i, y, r in zip(
            archive['train_image_ids'], archive['train_labels'], archive['train_relative_paths'].astype(str),
            strict=True)).encode()).hexdigest()
    assert rows == result['ordered_rows_sha256']
    shape = np.load(features, mmap_mode='r').shape
    assert shape == (59_551, 1152)
    a.output_dir.mkdir()
    os.link(features, a.output_dir / 'train_features.npy')
    receipt = {
        'schema': 'sfora-sop-siglip2-train-feature-export-v1', 'claim_eligible': False,
        'split': 'SOP official TRAIN rows only', 'full_train': True, 'rows': shape[0], 'dimension': shape[1],
        'model_id': 'google/siglip2-so400m-patch14-384', 'model_revision': a.model_snapshot.resolve().name,
        'model_file_sha256': {n: sha256(a.model_snapshot / n) for n in MODEL_FILES},
        'unicom_train_archive_sha256': UNICOM_TRAIN_ARCHIVE_SHA256, 'ordered_rows_sha256': rows,
        'features_sha256': sha256(a.output_dir / 'train_features.npy'),
        'encode_wall_seconds': result['export']['encode_wall_seconds'],
        'derived_from': {'screen_result_sha256': sha256(a.screen_dir / 'result.json'),
                         'screen_dir': str(a.screen_dir), 'script': 'screen_sop_so400m_pretrained_v1.py'},
    }
    (a.output_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
