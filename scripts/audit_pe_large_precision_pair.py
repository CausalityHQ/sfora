#!/usr/bin/env python3
"""Independent raw ordered-pair latency pilot replay; no GPU or new quality read."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--driver-sha256', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sha(Path(__file__)) == args.driver_sha256 and not args.output.exists()
    path = args.root / 'latency.json'
    assert sha(path) == args.receipt_sha256
    receipt = json.loads(path.read_text())
    assert sha(args.root / 'precision-pair-execution.json') == receipt['execution_sha256']
    assert receipt['code'] == json.loads((args.root / 'precision-pair-execution.json').read_text())
    assert all(sha(args.root / n) == h for n,h in receipt['code'].items())
    assert receipt['source_head_rng_environment_code_library_preserved'] and receipt['all_repeated_native_ordinals_score_bits_exact']
    assert receipt['gallery_images'] == 6245 and receipt['threads'] == 8 and receipt['warmups'] == 5
    assert receipt['optimizer_updates'] == 0 and not receipt['official_read'] and not receipt['p99_certified'] and not receipt['claim_eligible']
    rng = np.random.default_rng(179032)
    decisions = []
    for size in ('1', '32'):
        result = receipt['timing'][size]
        pairs = result['pairs']
        assert len(pairs) == result['calls_per_mode'] == 100 and not result['p99_certified']
        for sample in pairs:
            modes = ['fp32_autocast', 'fp16_native']
            assert sample['order'] == (modes if rng.integers(2) == 0 else list(reversed(modes)))
        control = np.asarray([s['fp32_autocast_ms'] for s in pairs])
        candidate = np.asarray([s['fp16_native_ms'] for s in pairs])
        assert np.isfinite(control).all() and np.isfinite(candidate).all() and min(control.min(),candidate.min()) > 0
        for mode, values in (('fp32_autocast',control),('fp16_native',candidate)):
            for quantile in (.5,.95):
                assert abs(float(np.quantile(values,quantile)) - result[mode + '_p' + str(int(100*quantile)) + '_ms']) < 1e-12
        delta = candidate - control
        bootstrap = np.random.default_rng(179032).choice(delta,size=(5000,100),replace=True).mean(axis=1)
        assert abs(float(delta.mean()) - result['paired_mean_candidate_minus_control_ms']) < 1e-12
        for name, quantile in (('lower',.025),('upper',.975)):
            assert abs(float(np.quantile(bootstrap,quantile)) - result['paired_mean_delta_' + name + '95_ms']) < 1e-12
        decision = bool(np.quantile(bootstrap,.975) < 0)
        assert decision == result['pilot_mean_faster']
        decisions.append(decision)
    assert all(decisions) == receipt['pilot_go'] and sha(Path(__file__)) == args.driver_sha256 and sha(path) == args.receipt_sha256
    args.output.write_text(json.dumps({'pass':True,'pilot_go':receipt['pilot_go'],'receipt_sha256':args.receipt_sha256,'auditor_sha256':args.driver_sha256,'pair_order_seed':179032,'bootstrap_seed':179032,'bootstrap_draws':5000,'all_raw_pair_quantiles_intervals_decision_exact':True,'claim_eligible':False,'p99_certified':False},indent=2)+'\n')
    print('PASS independent raw ordered-pair quantiles/intervals/pilot decision replay')


if __name__ == '__main__':
    main()
