#!/usr/bin/env python3
"""Frozen two-seed paired TRAIN decision, only after all four CPU audits."""
import argparse
import hashlib
import json
from pathlib import Path


def go(seeds, pooled, costs):
    return bool(
        set(seeds) == {179041, 179042}
        and set(costs) == set(seeds)
        and all(row[k] > 0 for row in seeds.values() for k in ('r1_delta', 'map_delta'))
        and pooled['r1']['mean_difference'] >= .002
        and pooled['ap']['mean_difference'] >= .005
        and all(pooled[k]['product_lower95'] > 0 for k in ('r1', 'ap'))
        and all(row[k] <= 1.10 for row in costs.values() for k in ('training_ratio', 'median_ratio'))
    )


def check():
    seeds = {179041: {'r1_delta': .002, 'map_delta': .005}, 179042: {'r1_delta': .002, 'map_delta': .005}}
    pooled = {'r1': {'mean_difference': .002, 'product_lower95': .001}, 'ap': {'mean_difference': .005, 'product_lower95': .001}}
    costs = {s: {'training_ratio': 1.10, 'median_ratio': 1.10} for s in seeds}
    assert go(seeds, pooled, costs)
    assert not go({179041: seeds[179041]}, pooled, costs)
    assert not go(seeds, pooled, {179041: costs[179041]})
    assert not go({**seeds, 179042: {'r1_delta': 0, 'map_delta': .005}}, pooled, costs)
    assert not go(seeds, {**pooled, 'r1': {**pooled['r1'], 'product_lower95': 0}}, costs)
    assert not go(seeds, {**pooled, 'ap': {**pooled['ap'], 'mean_difference': .004999}}, costs)
    assert not go(seeds, pooled, {**costs, 179042: {'training_ratio': 1.100001, 'median_ratio': 1}})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--execution-sha256')
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    check()
    if args.check:
        print('PASS frozen paired decision boundary; no Torch or outcome reads')
        return
    assert args.execution_sha256 and args.inputs and args.inputs_sha256 and args.output and not args.output.exists()
    root = Path(__file__).resolve().parent
    manifest = root / 'native-valid-anchor-decision-execution.json'
    assert sha(manifest) == args.execution_sha256
    code = json.loads(manifest.read_text())
    assert all(sha(root / n) == h for n, h in code.items())
    old = json.loads((root / 'native-valid-anchor-held-execution.json').read_text())
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    assert sha(args.inputs) == args.inputs_sha256
    entries = json.loads(args.inputs.read_text())
    assert {(v['seed'], v['arm']) for v in entries} == {(s, a) for s in (179041, 179042) for a in ('control', 'treatment')} and len(entries) == 4
    import numpy as np
    import torch
    import train_inshop_pe_pair as pair
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    records = {}
    roles = None
    for entry in entries:
        seed, arm = entry['seed'], entry['arm']
        directory = Path(f'/home/riomus/runs/sfora-native-valid-anchor-held-{seed}-{arm}-v1')
        assert sha(directory / 'receipt.json') == entry['receipt_sha256'] and sha(directory / 'cpu-audit.json') == entry['audit_sha256']
        receipt = json.loads((directory / 'receipt.json').read_text())
        audit = json.loads((directory / 'cpu-audit.json').read_text())
        assert audit['pass'] and audit['receipt_sha256'] == entry['receipt_sha256']
        assert receipt['code'] == old and receipt['execution_sha256'] == sha(root / 'native-valid-anchor-held-execution.json')
        assert receipt['official_read'] is False and receipt['optimizer_updates'] == 0 and receipt['claim_eligible'] is False
        assert receipt['source_head_rng_environment_unchanged'] and receipt['independent_two_block_suffix_head_packed_exact']
        assert receipt['held_images'] == 12599 and len(receipt['query']) == 6354 and len(receipt['gallery']) == 6245
        this_roles = {k: receipt[k] for k in ('held_manifest', 'query', 'gallery')}
        assert roles is None or roles == this_roles
        roles = this_roles
        source = Path('/home/riomus/runs/sfora-native-valid-anchor-held-v3') / f'source-{seed}-{arm}-proof.json'
        assert sha(source) == receipt['cpu_authority_sha256'] == entry['source_sha256']
        proof = json.loads(source.read_text())
        assert proof['native_training_completed_updates'] == 100 and proof['native_training_seed'] == seed and proof['native_training_arm'] == arm
        assert proof['teacher_checkpoint_sha256'] == receipt['teacher_checkpoint_sha256']
        train_dir = Path(f'/home/riomus/runs/sfora-native-valid-anchor-{seed}-{arm}-100-v1')
        assert sha(train_dir / 'receipt.json') == proof['native_training_receipt_sha256']
        train = json.loads((train_dir / 'receipt.json').read_text())
        assert train['pass'] and train['updates'] == 100 and train['peak_cuda_allocated_bytes'] < 10_000_000_000
        quality = audit['quality']
        assert all(np.max(np.abs(np.asarray(quality[k]) - np.asarray(receipt['quality'][k]))) < 1e-6 for k in quality)
        assert sha(directory / 'large.held.npy') == receipt['held_sha256'] and sha(directory / 'large.reference-held.npy') == receipt['reference_held_sha256']
        records[(seed, arm)] = {'receipt': receipt, 'quality': quality, 'training': train}
    products = np.asarray([r['product'] for r in roles['held_manifest']])[roles['query']]
    per_seed, costs, deltas = {}, {}, {'r1': [], 'ap': []}
    for seed in (179041, 179042):
        control, treatment = (records[(seed, a)] for a in ('control', 'treatment'))
        c, t = control['training'], treatment['training']
        assert c['initial_state_sha256'] == t['initial_state_sha256'] and c['schedule_sha256'] == t['schedule_sha256']
        assert all(a['pixels_sha256'] == b['pixels_sha256'] and a['rgb_sha256'] == b['rgb_sha256'] for a, b in zip(c['steps'], t['steps'], strict=True))
        costs[seed] = {'training_ratio': t['training_wall_seconds'] / c['training_wall_seconds'], 'median_ratio': t['median_step_seconds'] / c['median_step_seconds'], 'control_seconds': c['training_wall_seconds'], 'treatment_seconds': t['training_wall_seconds']}
        per_seed[seed] = {'control': {k: control['quality'][k] for k in ('recall_at_1', 'map_at_r')}, 'treatment': {k: treatment['quality'][k] for k in ('recall_at_1', 'map_at_r')}}
        for short, metric, point in (('r1', 'per_query_r1', 'r1_delta'), ('ap', 'per_query_ap', 'map_delta')):
            delta = np.asarray(treatment['quality'][metric]) - np.asarray(control['quality'][metric])
            assert delta.shape == (6354,) and np.isfinite(delta).all()
            deltas[short].append(delta)
            per_seed[seed][point] = float(delta.mean())
            per_seed[seed][short + '_product_lower95'] = pair.bootstrap_lower(delta, products)
            per_seed[seed][short + '_product_upper95'] = -pair.bootstrap_lower(-delta, products)
    pooled = {}
    for short, arrays in deltas.items():
        mean = np.mean(np.stack(arrays), axis=0)
        assert mean.shape == (6354,)  # Average paired seeds first; never concatenate queries.
        pooled[short] = {'mean_difference': float(mean.mean()), 'product_lower95': pair.bootstrap_lower(mean, products), 'product_upper95': -pair.bootstrap_lower(-mean, products)}
    decision = 'GO' if go(per_seed, pooled, costs) else 'KILL'
    assert all(sha(root / n) == h for n, h in code.items()) and sha(args.inputs) == args.inputs_sha256
    pair.smoke.save(args.output, {'decision': decision, 'execution_sha256': args.execution_sha256, 'inputs_sha256': args.inputs_sha256, 'inputs': entries, 'dataset': 'InShop official TRAIN product-disjoint previously observed holdout', 'query_images': 6354, 'gallery_images': 6245, 'per_seed': per_seed, 'costs': costs, 'pooled_seed_averaged_deltas': pooled, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019, 'intervals_condition_on_two_checkpoints': True, 'official_read': False, 'claim_eligible': False})
    print(decision, json.dumps({'per_seed': per_seed, 'costs': costs, 'pooled': pooled}), flush=True)


if __name__ == '__main__':
    main()
