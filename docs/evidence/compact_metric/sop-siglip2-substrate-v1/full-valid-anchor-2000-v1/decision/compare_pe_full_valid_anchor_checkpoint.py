#!/usr/bin/env python3
"""Paired fixed2000 full-control/corrected-candidate decision from audited wires."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import qualify_pe_full_valid_anchor_checkpoint as qualified

ROOT = Path('/home/riomus/runs/sfora-full-valid-anchor-checkpoint-v1')
CODE = '4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375'
TRAIN = 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
CONTROL = Path('/home/riomus/runs/sfora-large-optimization-full-official-b32-v1')
CONTROL_SHA = '33c00bcec5a4ed82d734d71546aba2e4bd9bad84798eafb528240fa13e238388'
CONTROL_AUDIT = '8fbe0043f148dde48e04a67dc270b8e2f61c175d9a75142628a05de10989453f'
CANDIDATE = Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--decision-sha256', required=True)
    p.add_argument('--candidate-sha256', required=True)
    p.add_argument('--candidate-audit-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    assert Path(qualified.__file__).resolve().parent == ROOT
    pair = qualified.pair
    assert pair.sha(Path(__file__)) == args.decision_sha256
    torch.set_num_threads(8)
    _, _, _, _, _, terminal, protocol, _, code, cost = qualified.authority(ROOT, CODE, TRAIN)
    assert pair.sha(CONTROL / 'receipt.json') == CONTROL_SHA and pair.sha(CONTROL / 'cpu-audit.json') == CONTROL_AUDIT
    assert pair.sha(CANDIDATE / 'receipt.json') == args.candidate_sha256
    assert pair.sha(CANDIDATE / 'cpu-audit.json') == args.candidate_audit_sha256
    receipts, audits = [], []
    for directory, expected in ((CONTROL, CONTROL_SHA), (CANDIDATE, args.candidate_sha256)):
        measured = json.loads((directory / 'receipt.json').read_text())
        audit = json.loads((directory / 'cpu-audit.json').read_text())
        assert audit['pass'] and audit['receipt_sha256'] == expected and measured['arm'] == 'full'
        assert measured['batch'] == 32 and measured['query_images'] == 14218 and measured['gallery_images'] == 12612
        assert measured['native_all_query_top10_ordinal_score_bits_exact'] and measured['source_state_rng_environment_code_library_preserved']
        assert measured['independent_updated_native_original_preprocessor_first32_each_role_packed_exact'] and measured['peak_cuda_allocated_bytes'] < 10_000_000_000
        for role in ('query', 'gallery'):
            qualified.qualified.selected.fp16.load_packed(directory, measured, role)
        for metric in ('per_query_r1', 'per_query_ap'):
            assert np.max(np.abs(np.asarray(measured['quality'][metric]) - np.asarray(audit['quality'][metric]))) < 1e-6
        assert all(abs(value - audit['quality_intervals'][metric][kind]) < 1e-6 for metric, interval in measured['quality_intervals'].items() for kind, value in interval.items())
        receipts.append(measured); audits.append(audit)
    candidate = receipts[1]
    assert candidate['execution_sha256'] == CODE and candidate['code'] == code and candidate['training_receipt_sha256'] == TRAIN
    proof = ROOT / 'updated-cpu-proof.json'
    assert candidate['cpu_authority_sha256'] == pair.sha(proof)
    cpu = json.loads(proof.read_text())
    assert cpu['pass'] and cpu['strict_updated_native_CPU_B2_whole_head_packed_exact'] and cpu['changed_driver_rejected']
    assert candidate['checkpoint_sha256'] == cpu['checkpoint_sha256']
    export_root = Path('/home/riomus/runs/sfora-full-valid-anchor-native2000-v1')
    exported = json.loads((export_root / 'receipt.json').read_text())
    assert pair.sha(export_root / 'native.pt') == exported['checkpoint_sha256'] == candidate['checkpoint_sha256']
    assert exported['training_receipt_sha256'] == TRAIN and exported['source_resume_sha256'] == terminal['checkpoint_sha256']
    labels = np.asarray([row['product'] for row in protocol['protocol']['query']])
    assert len(labels) == 14218
    comparison, quality = {}, {}
    for metric in ('per_query_r1', 'per_query_ap'):
        a, b = (np.asarray(r['quality'][metric], dtype=np.float64) for r in audits)
        assert a.shape == b.shape == (14218,) and np.isfinite(a).all() and np.isfinite(b).all()
        delta = b - a
        comparison[metric] = {'mean_difference': float(delta.mean())}
        quality[metric] = {'control': float(a.mean()), 'candidate': float(b.mean())}
        for kind, groups in (('product', labels), ('query', np.arange(len(labels)))):
            comparison[metric][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            comparison[metric][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    necessary = quality['per_query_r1']['candidate'] > .967
    assert necessary == candidate['advance_minimum_dated_reference_screen'] == audits[1]['advance_minimum_dated_reference_screen']
    paired = all(row['product_lower95'] > 0 for row in comparison.values())
    baseline_training = json.loads((qualified.cpu.FULL_CONTROL / 'receipt.json').read_text())
    assert pair.sha(qualified.cpu.FULL_CONTROL / 'receipt.json') == qualified.cpu.FULL_CONTROL_SHA
    control_cost = sum(json.loads((Path(f'/home/riomus/runs/sfora-large-optimization-full-{end}-v1') / 'receipt.json').read_text())['training_wall_seconds'] for end in range(100, 2001, 100))
    assert baseline_training['completed_step'] == 2000 and cost / control_cost <= 1.10
    for mode in ('export', 'source', 'official', 'audit'):
        log = (ROOT / f'full-valid-anchor-{mode}-v1.log').read_text()
        assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
    assert all(pair.sha(ROOT / n) == h for n, h in code.items()) and pair.sha(Path(__file__)) == args.decision_sha256
    result = {'pass': True, 'decision': 'GO' if necessary and paired else 'KILL', 'scope': 'fixed full-valid-anchor2000 procedure only', 'dataset': 'DeepFashion In-Shop', 'split': 'previously observed official query/gallery', 'query_images': 14218, 'gallery_images': 12612, 'quality': quality, 'comparison': comparison, 'necessary_dated_reference_screen': necessary, 'paired_product_lower95_positive': paired, 'dated_reference_r1': .967, 'remaining_dated_r1_gap_pp': 100 * (.967 - quality['per_query_r1']['candidate']), 'training_seconds': {'control': control_cost, 'candidate': cost}, 'training_images_per_second': {'control': 128000 / control_cost, 'candidate': 128000 / cost}, 'training_cost_ratio': cost / control_cost, 'recovered_rank_updates': 375, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019, 'uncertainty_conditional_on_fixed_checkpoints': True, 'control_receipt_sha256': CONTROL_SHA, 'control_audit_sha256': CONTROL_AUDIT, 'candidate_receipt_sha256': args.candidate_sha256, 'candidate_audit_sha256': args.candidate_audit_sha256, 'candidate_training_sha256': TRAIN, 'qualification_execution_sha256': CODE, 'decision_source_sha256': args.decision_sha256, 'integrity': 'PASS', 'resource': 'PASS', 'candidate_artifacts_preserved': True, 'current_strongest_external_reference_gate': 'OPEN', 'public_latency_measured': False, 'claim_eligible': False, 'prior_official_benchmark_exposure': True, 'production_joint_goal_met': False}
    pair.smoke.save(args.output, result)
    print(json.dumps({k: result[k] for k in ('decision', 'quality', 'comparison', 'training_cost_ratio', 'remaining_dated_r1_gap_pp')}), flush=True)


if __name__ == '__main__':
    main()
