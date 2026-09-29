#!/usr/bin/env python3
"""CPU replay of complete TRAIN vectors and fixed final-block pilot decision."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_pe_large_final_block_pilot as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--auditor-sha256', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    pair = driver.pair
    assert pair.sha(Path(__file__)) == args.auditor_sha256
    root = Path(__file__).resolve().parent
    _, frozen, prior, cpu, initial, _, code = driver.startup(root, args.execution_sha256)
    receipt_path = args.output / 'receipt.json'
    assert pair.sha(receipt_path) == args.receipt_sha256
    r = json.loads(receipt_path.read_text())
    assert r['execution_sha256'] == args.execution_sha256 and r['preflight_sha256'] == driver.GPU_SHA
    assert r['updates'] == 100 and r['quality_read'] == 'In-Shop TRAIN-held only'
    assert r['official_read'] is False and r['claim_eligible'] is False
    assert r['full_held_independent_whole_encoder_exact'] is False
    assert all(r[k] is True for k in ('updated_gpu_strict_reload_exact', 'updated_whole_native_B64_calibration_exact', 'full_held_strict_loaded_whole_and_independent_live_suffix_exact', 'packed_per_query_parity', 'full_held_head_reload_exact', 'frozen_complement_unchanged', 'full_live_loaded_native_state_exact'))
    assert r['held_images'] == 12599 and r['held_scope_count'] == 394 and 0 < r['peak_cuda_allocated_bytes'] < 10_000_000_000
    assert pair.sha(args.output / 'pe.pt') == r['checkpoint_sha256']
    assert pair.sha(args.output / 'training.json') == r['training_sha256']
    t = json.loads((args.output / 'training.json').read_text())
    assert t['execution_sha256'] == args.execution_sha256 and t['frozen_sha256'] == cpu['frozen_complement_sha256']
    assert t['backward_nodes'] == initial['backward_nodes'] and t['caller_cpu_rng_unchanged'] and t['one_pending_cpu_batch']
    assert len(t['step_seconds']) == len(t['losses']) == len(t['scales']) == len(t['worker_input_seconds']) == 100
    assert t['training_wall_seconds_including_fill_drain'] >= sum(t['step_seconds'])
    assert np.median(t['step_seconds'][2:]) == r['median_step_3_100_seconds'] <= 0.71769696
    assert t['scales'] == [128] * 100 and t['rgb_sha256'] == prior['rgb_sha256']
    previous = json.loads((driver.MECHANICS_DIR / 'training.json').read_text())
    assert t['losses'][:17] == previous['losses'] and t['scales'][:17] == previous['scales']
    assert [d['step'] for d in t['diagnostics']] == [1, 100]
    assert all(len(d['native_gradient_norms']) == 16 and all(np.isfinite(v) and v > 0 for v in (*d['native_gradient_norms'].values(), *d['head_proxy_gradient_norms'])) for d in t['diagnostics'])
    arrays = []
    for name, key in (('pe.held.npy', 'held_sha256'), ('pe.live-held.npy', 'live_held_sha256')):
        path = args.output / name
        assert pair.sha(path) == r[key]
        values = np.load(path, allow_pickle=False)
        assert values.shape == (12599, 128) and values.dtype == np.float32 and np.isfinite(values).all()
        assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
        arrays.append(values)
    assert np.array_equal(*arrays)
    labels = tuple(row['product'] for row in frozen['held_manifest'])
    quality = pair.packed_quality(arrays[0], labels, frozen['query'], frozen['gallery'], device=torch.device('cpu'))
    assert all(np.max(np.abs(np.asarray(v) - np.asarray(r['quality'][k]))) < 1e-6 for k, v in quality.items())
    dense = json.loads((driver.base.INIT / 'receipt.json').read_text())['arms']['pe']
    products = np.asarray(labels)[frozen['query']]
    intervals = {}
    for k in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(quality[k]) - np.asarray(dense['quality'][k])
        intervals[k] = {'mean_delta': float(np.mean(delta))}
        for kind, groups in (('product', products), ('query', np.arange(len(delta)))):
            intervals[k][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            intervals[k][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
        assert all(abs(v - r['paired_dense_pe_intervals'][k][name]) < 1e-6 for name, v in intervals[k].items())
    floors = bool(quality['recall_at_1'] >= 0.951720176 and quality['map_at_r'] >= 0.776237120 and all(v['product_lower95'] > 0 for v in intervals.values()))
    assert floors == r['advance']
    assert pair.sha(receipt_path) == args.receipt_sha256 and pair.sha(Path(__file__)) == args.auditor_sha256
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output / 'cpu-audit.json', {'pass': True, 'advance': floors, 'quality': quality, 'paired_dense_pe_intervals': intervals, 'receipt_sha256': args.receipt_sha256, 'execution_sha256': args.execution_sha256, 'auditor_sha256': args.auditor_sha256, 'claim_eligible': False, 'quality_read': 'TRAIN-held saved-vector CPU replay only'})
    print('PASS authenticated complete TRAIN vectors, CPU packed per-query replay, intervals and fixed decision')


if __name__ == '__main__':
    main()
