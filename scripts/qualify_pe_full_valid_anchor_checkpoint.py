#!/usr/bin/env python3
"""Honest corrected2000 authority/export adapter for unchanged native qualification."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch
import torch
import train_pe_full_valid_anchor as training
import qualify_pe_large_coverage_checkpoint as qualified

cpu, driver, pair = training.cpu, training.driver, training.pair
TRAIN_CODE = '0da63376f73c9bc55c4c5f6e8c0f3f80fab7efe211ef6234a72af2ce4ed3ac36'
MECHANICS = Path('/home/riomus/runs/sfora-full-valid-anchor-mechanics-v1')
MECHANICS_SHA = '92118c1b50159b98d192b37e2c25041085c95aa9d0dcf93644890f8b39fdedc7'


def authority(root, execution, training_sha):
    manifest = root / 'full-valid-anchor-checkpoint-execution.json'
    assert pair.sha(manifest) == execution
    code = json.loads(manifest.read_text())
    previous = json.loads((root / 'full-valid-anchor-training-execution.json').read_text())
    assert pair.sha(root / 'full-valid-anchor-training-execution.json') == TRAIN_CODE
    assert len(code) == len(previous) + 1 and all(code[n] == h for n, h in previous.items())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'coverage checkpoint code differs'
    helpers = cpu.previous.qualified.confirmation.selected.helpers
    with patch.object(cpu.previous.qualified.confirmation.selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, old, baseline = training.startup(root, TRAIN_CODE)
    assert old == previous and pair.sha(MECHANICS / 'receipt.json') == MECHANICS_SHA
    mechanics = json.loads((MECHANICS / 'receipt.json').read_text())
    assert mechanics['pass'] and mechanics['training_state_discarded'] and mechanics['native_17_equals_serialized8_plus9_exact']
    last_sha = None
    last_checkpoint = None
    total_cost = 0.
    recovered = 0
    initial = None
    for end in range(100, 2001, 100):
        run = Path(f'/home/riomus/runs/sfora-full-valid-anchor-{end}-v1')
        value = json.loads((run / 'receipt.json').read_text())
        assert value['pass'] and value['completed_step'] == end and value['chunk_start'] == end - 100 and value['total_updates'] == 2000
        assert value['arm'] == 'full' and value['seed'] == 179032 and value['intervention'] == 'full-valid-anchor-v1'
        assert value['execution_sha256'] == TRAIN_CODE and value['mechanics_sha256'] == MECHANICS_SHA and value['cpu_authority_sha256'] == training.CPU_SHA
        assert value['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA
        assert value['frozen_source_code_environment_rng_preserved'] and value['all_input_hashes_equal_archived_control'] and not value['quality_read']
        assert value['previous_receipt_sha256'] == last_sha and value['cost_ratio_vs_archived_control'] <= 1.10 and value['peak_cuda_allocated_bytes'] < 10_000_000_000
        assert value['previous_checkpoint_sha256'] == last_checkpoint and value['control_training_receipt_sha256'] == cpu.FULL_CONTROL_SHA
        assert [s['step'] for s in value['steps']] == list(range(end - 99, end + 1))
        assert all(s['rgb_sha256'] == b['rgb_sha256'] and s['pixels_sha256'] == b['pixels_sha256'] and s['rank_active_before'] == b['rank_active'] for s, b in zip(value['steps'], baseline[end]['steps'], strict=True))
        initial = value['initial_state_sha256'] if initial is None else initial
        assert value['initial_state_sha256'] == initial == mechanics['initial_state_sha256'] and value['schedule_sha256'] == mechanics['schedule_sha256']
        log = (root / f'full-valid-anchor-{end}.log').read_text()
        assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
        total_cost += value['training_wall_seconds']; recovered += value['recovered_rank_updates']
        last_sha = pair.sha(run / 'receipt.json')
        last_checkpoint = value['checkpoint_sha256']
    assert last_sha == training_sha and recovered == 375
    assert pair.sha(run / 'resume.pt') == value['checkpoint_sha256']
    assert total_cost / sum(v['training_wall_seconds'] for v in baseline.values()) <= 1.10
    protocol = json.loads(qualified.PROTOCOL.read_text())
    assert pair.sha(qualified.PROTOCOL) == qualified.PROTOCOL_SHA and protocol['pass'] and protocol['prior_official_benchmark_exposure']
    assert protocol['train_query_gallery_ids_disjoint'] and pair.sha(control.dataset_root / 'Eval/list_eval_partition.txt') == protocol['partition_sha256'] == qualified.selected.PARTITION_SHA
    official = helpers(root, code)
    return control, source, prior, proof, run, value, protocol, official, code, total_cost


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--training-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--export-cpu', action='store_true')
    p.add_argument('--qualify-cpu', action='store_true')
    p.add_argument('--cpu-proof', type=Path)
    p.add_argument('--cpu-sha256')
    p.add_argument('--audit-cpu', action='store_true')
    p.add_argument('--receipt-sha256')
    args = p.parse_args()
    assert sum((args.export_cpu, args.qualify_cpu, args.audit_cpu)) <= 1
    root = Path(__file__).resolve().parent
    control, source, prior, proof, run, terminal, protocol, official, code, total = authority(root, args.execution_sha256, args.training_sha256)
    weights = Path('/home/riomus/runs/sfora-full-valid-anchor-native2000-v1')
    if args.export_cpu:
        assert not torch.cuda.is_available() and args.output == weights and not weights.exists()
        torch.set_num_threads(8)
        disk = torch.load(run / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
        identity = disk['identity']
        assert identity['global_step'] == identity['total_updates'] == 2000 and identity['intervention'] == 'full-valid-anchor-v1'
        assert identity['execution_sha256'] == TRAIN_CODE and identity['seed'] == 179032 and identity['arm'] == 'full'
        assert identity['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA and identity['schedule_sha256'] == terminal['schedule_sha256']
        assert identity['initializers_sha256'] == proof['arms']['full']['initializers_sha256'] and identity['frozen_prefix_sha256'] == proof['frozen_prefix_sha256']
        assert len(disk['vision']) == 400 and driver.fingerprint(disk['buffers']) == identity['buffers_sha256']
        assert len(disk['optimizer']['state']) == len(identity['parameter_names']) and all(int(s['step']) == 2000 for s in disk['optimizer']['state'].values())
        assert [{k: v for k, v in g.items() if k != 'params'} for g in disk['optimizer']['param_groups']] == identity['optimizer_groups']
        assert identity['precision'] == 'cuda_fp16' and disk['scaler'] and len(disk['cuda_rng']) == 1
        values = {**disk['vision'], **disk['head'], 'classifier': disk['classifier'], 'bank': disk['bank'], **{'runtime.' + n: v for n, v in disk['buffers'].items()}}
        assert all(torch.isfinite(v).all() for v in values.values())
        assert all(torch.isfinite(v).all() for state in disk['optimizer']['state'].values() for v in state.values() if isinstance(v, torch.Tensor))
        weights.mkdir(exist_ok=False)
        checkpoint = weights / 'native.pt'
        torch.save({'vision': disk['vision'], 'head': disk['head']}, checkpoint)
        whole = pair.smoke.digest({**disk['vision'], **{'runtime.' + n: v for n, v in disk['buffers'].items()}})
        head = pair.smoke.digest(disk['head'])
        assert all(pair.sha(root / n) == h for n, h in code.items())
        pair.smoke.save(weights / 'receipt.json', {'pass': True, 'arm': 'full', 'intervention': identity['intervention'], 'completed_step': 2000, 'execution_sha256': args.execution_sha256, 'training_receipt_sha256': args.training_sha256, 'source_resume_sha256': terminal['checkpoint_sha256'], 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'checkpoint_sha256': pair.sha(checkpoint), 'updated_whole_sha256': whole, 'updated_head_sha256': head, 'aggregate_training_wall_seconds': total, 'recovered_rank_updates': 375, 'strict400_terminal_vision_head_export': True, 'optimizer_updates': 0, 'quality_read': False})
        return
    exported = json.loads((weights / 'receipt.json').read_text())
    assert exported['pass'] and exported['training_receipt_sha256'] == args.training_sha256 and exported['source_resume_sha256'] == terminal['checkpoint_sha256']
    assert exported['execution_sha256'] == args.execution_sha256 and exported['completed_step'] == 2000 and exported['intervention'] == 'full-valid-anchor-v1'
    assert pair.sha(weights / 'native.pt') == exported['checkpoint_sha256']

    def startup(r, execution, arm, receipt_sha):
        assert r == root and execution == args.execution_sha256 and arm == 'full' and receipt_sha == args.training_sha256
        assert all(pair.sha(root / n) == h for n, h in code.items()), 'coverage checkpoint code differs'
        assert pair.sha(weights / 'native.pt') == exported['checkpoint_sha256']
        return control, source, prior, proof, weights, exported, protocol, official, code

    argv = ['full-corrected-qualification', '--execution-sha256', args.execution_sha256, '--arm', 'full', '--training-sha256', args.training_sha256, '--output', str(args.output)]
    if args.qualify_cpu:
        argv += ['--qualify-cpu']
    else:
        assert args.cpu_proof and args.cpu_sha256
        argv += ['--cpu-proof', str(args.cpu_proof), '--cpu-sha256', args.cpu_sha256]
        if args.audit_cpu:
            assert args.receipt_sha256
            argv += ['--audit-cpu', '--receipt-sha256', args.receipt_sha256]
    # Bind corrected checkpoint honestly; old source/official implementation stays unchanged.
    with patch.object(qualified, 'startup', startup), patch.object(sys, 'argv', argv):
        qualified.main()


if __name__ == '__main__':
    main()
