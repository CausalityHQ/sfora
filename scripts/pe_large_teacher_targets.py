#!/usr/bin/env python3
"""Authenticate the completed TRAIN-fit teacher table before student use."""
import argparse
import copy
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import export_pe_large_teacher_fit as export

ROOT = Path('/home/riomus/runs/sfora-large-teacher-fit-targets-v1')
RECEIPT_SHA = '277f9b774470355f76c16675a9c7808abe602c83612d83f4d90e4d37cb5507fe'
CPU_SHA = '4731d7a4a4a3fc645e22dd67cd58ebbb39479df550ed031983d159fd7afcccb8'
EXECUTION_SHA = '39736900038d368b8153748041da23dbe3b732246ddb09242c399674288e0a45'


def rows(receipt, frozen):
    assert receipt['fit_manifest'] == frozen['fit_manifest'] and receipt['target_products'] == frozen['target'], 'teacher row authority differs'


def load(frozen, code):
    assert export.pair.sha(ROOT / 'receipt.json') == RECEIPT_SHA
    receipt = json.loads((ROOT / 'receipt.json').read_text())
    assert export.pair.sha(export.CPU) == CPU_SHA
    cpu = json.loads(export.CPU.read_text())
    assert receipt['cpu_authority_sha256'] == CPU_SHA and receipt['code'] == cpu['code'] == code
    for key in ('teacher_checkpoint_sha256', 'teacher_whole_sha256', 'teacher_verification_prefix_sha256', 'teacher_head_sha256', 'environment', 'first_two_fit_pixels_sha256'):
        assert receipt[key] == cpu[key]
    assert receipt['teacher_checkpoint_sha256'] == export.TEACHER_SHA
    assert receipt['read_only'] and receipt['optimizer_updates'] == receipt['held_images'] == 0 and not receipt['quality_read']
    assert all(receipt[k] for k in ('strict400_native_head_reload_and_direct_whole_calibration_exact', 'independent_two_block_suffix_head_packed_exact', 'cpu_cuda_rng_unchanged', 'prefix_data_mutation_rejected_at_exit', 'cuda'))
    assert receipt['fit_images'] == 13283 and receipt['peak_cuda_allocated_bytes'] < 10_000_000_000
    rows(receipt, frozen)
    arrays = []
    for name, key in (('teacher.fit.npy', 'fit_sha256'), ('teacher.reference-fit.npy', 'reference_fit_sha256')):
        assert export.pair.sha(ROOT / name) == receipt[key], 'teacher cache hash differs'
        values = np.load(ROOT / name, allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (13283, 128) and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        arrays.append(values)
    assert np.array_equal(*arrays)
    targets = torch.from_numpy(arrays[0].copy())
    assert targets.device.type == 'cpu' and not targets.requires_grad and targets.grad_fn is None
    return targets, receipt


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--consumer-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    assert export.pair.sha(Path(__file__)) == args.consumer_sha256
    _, frozen, code = export.startup(Path(__file__).resolve().parent, EXECUTION_SHA)
    targets, receipt = load(frozen, code)
    original = export.pair.sha
    with patch.object(export.pair, 'sha', lambda p: 'altered' if Path(p) == ROOT / 'teacher.fit.npy' else original(p)):
        try:
            load(frozen, code)
        except AssertionError as error:
            assert str(error) == 'teacher cache hash differs'
        else:
            raise AssertionError('changed teacher cache accepted')
    for key in ('fit_manifest', 'target_products'):
        changed = copy.deepcopy(receipt)
        changed[key][0] = None
        try:
            rows(changed, frozen)
        except AssertionError as error:
            assert str(error) == 'teacher row authority differs'
        else:
            raise AssertionError('changed teacher row accepted')
    export.pair.smoke.save(args.output, {'consumer_sha256': args.consumer_sha256, 'export_receipt_sha256': RECEIPT_SHA, 'cpu_authority_sha256': CPU_SHA, 'fit_sha256': receipt['fit_sha256'], 'shape': list(targets.shape), 'cache_hash_and_image_product_row_rejection': True, 'constant_cpu_targets': True, 'optimizer_updates': 0, 'held_images': 0, 'quality_read': False})
    print('PASS complete TRAIN-fit teacher cache, image/product order, constant targets and changed-cache/row rejection; no CUDA/training/held/quality')


if __name__ == '__main__':
    main()
