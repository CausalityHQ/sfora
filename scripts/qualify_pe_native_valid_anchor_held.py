#!/usr/bin/env python3
"""Honest fixed100 source adapter for unchanged native TRAIN export/audit."""
import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import torch
import train_pe_native_valid_anchor as training
import qualify_pe_large_trained_candidate as trained

cpu, driver = training.cpu, training.driver
TRAIN_CODE = 'eeed3caf006b6f05d3b1ebc3abb5444f221b2a35a5db26193b9d71fe54865a3b'
MECHANICS = Path('/home/riomus/runs/sfora-native-valid-anchor-mechanics-v1')
MECHANICS_SHA = '812113620cda621afad08276d3fb73ba62eb9e10c3c5acf9a642c7bb971a5450'


def authority(root, execution, seed, arm, receipt_sha):
    path = root / 'native-valid-anchor-held-execution.json'
    assert driver.pair.sha(path) == execution
    code = json.loads(path.read_text())
    assert all(driver.pair.sha(root / n) == h for n, h in code.items()), 'native held source differs'
    previous = json.loads((root / 'native-valid-anchor-training-execution.json').read_text())
    assert driver.pair.sha(root / 'native-valid-anchor-training-execution.json') == TRAIN_CODE
    assert len(code) == len(previous) + 1 and all(code[n] == h for n, h in previous.items())
    helpers = cpu.qualified.confirmation.selected.helpers
    with patch.object(cpu.qualified.confirmation.selected, 'helpers', lambda r, _: helpers(r, code)):
        control, _, prior, native, old = training.startup(root, TRAIN_CODE)
    assert old == previous and driver.pair.sha(MECHANICS / 'receipt.json') == MECHANICS_SHA
    # Close all four fixed training arms before any CPU model/export contention.
    runs = {}
    for declared_seed in (179041, 179042):
        runs[declared_seed] = {}
        for declared_arm in ('control', 'treatment'):
            name = f'native-valid-anchor-{declared_seed}-{declared_arm}-100-v1'
            run = Path('/home/riomus/runs/sfora-' + name)
            value = json.loads((run / 'receipt.json').read_text())
            assert value['pass'] and value['seed'] == declared_seed and value['arm'] == declared_arm and value['updates'] == 100
            assert value['execution_sha256'] == TRAIN_CODE and value['cpu_authority_sha256'] == training.CPU_SHA
            assert value['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA
            assert not value['quality_read'] and not value['training_state_discarded']
            assert value['strict400_reload_whole_head_packed_exact'] and value['source_state_rng_environment_code_preserved']
            assert [s['step'] for s in value['steps']] == list(range(1, 101))
            assert value['peak_cuda_allocated_bytes'] < 10_000_000_000
            assert driver.pair.sha(run / 'native.pt') == value['checkpoint_sha256']
            log = (root / (name + '.log')).read_text()
            assert 'Finished with result: success' in log and 'Main processes terminated with: code=exited/status=0' in log and 'Memory swap peak: 0B' in log
            runs[declared_seed][declared_arm] = (run, value)
        a, b = (runs[declared_seed][n][1] for n in ('control', 'treatment'))
        assert a['initial_state_sha256'] == b['initial_state_sha256'] and a['schedule_sha256'] == b['schedule_sha256']
        assert all(x['rgb_sha256'] == y['rgb_sha256'] and x['pixels_sha256'] == y['pixels_sha256'] for x, y in zip(a['steps'], b['steps'], strict=True))
        assert b['training_wall_seconds'] / a['training_wall_seconds'] <= 1.10 and b['median_step_seconds'] / a['median_step_seconds'] <= 1.10
        assert b['recovered_rank_updates'] == sum(not s['rank_active_before'] for s in a['steps']) > 0
    run, value = runs[seed][arm]
    assert driver.pair.sha(run / 'receipt.json') == receipt_sha
    _, frozen, _ = trained.teacher.startup(root, trained.TEACHER_CODE_SHA)
    assert frozen['fit_manifest'] == native['arms']['half']['rows']
    assert len(frozen['held_manifest']) == 12599 and len(frozen['query']) == 6354 and len(frozen['gallery']) == 6245
    assert {r['product'] for r in frozen['held_manifest']}.isdisjoint(r['product'] for r in frozen['fit_manifest'])
    return control, frozen, prior, run, value, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--seed', type=int, choices=(179041, 179042), required=True)
    parser.add_argument('--arm', choices=('control', 'treatment'), required=True)
    parser.add_argument('--training-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-cpu', action='store_true')
    parser.add_argument('--cpu-proof', type=Path)
    parser.add_argument('--cpu-sha256')
    parser.add_argument('--audit-cpu', action='store_true')
    parser.add_argument('--check-startup-only', action='store_true')
    parser.add_argument('--receipt-sha256')
    args = parser.parse_args()
    assert not (args.qualify_cpu and args.audit_cpu)
    root = Path(__file__).resolve().parent
    control, frozen, prior, run, terminal, code = authority(root, args.execution_sha256, args.seed, args.arm, args.training_sha256)
    binding = {'native_training_receipt_sha256': args.training_sha256, 'native_training_seed': args.seed, 'native_training_arm': args.arm, 'native_training_completed_updates': 100, 'native_training_execution_sha256': TRAIN_CODE}
    if args.check_startup_only:
        assert not torch.cuda.is_available() and not args.output.exists() and not args.qualify_cpu and not args.audit_cpu
        real_sha = driver.pair.sha
        with patch.object(driver.pair, 'sha', lambda p: '0' * 64 if Path(p).resolve() == Path(__file__).resolve() else real_sha(p)):
            try:
                authority(root, args.execution_sha256, args.seed, args.arm, args.training_sha256)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed held driver accepted')
        driver.pair.smoke.save(args.output, {'pass': True, **binding, 'execution_sha256': args.execution_sha256, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False})
        return
    checkpoint = run / 'native.pt'
    def source_startup(r, ex):
        assert r == root and ex == args.execution_sha256
        return control, frozen, code
    def held_startup(r, ex):
        assert r == root and ex == args.execution_sha256
        return control, frozen, proof, fit, code
    with ExitStack() as scope:
        scope.enter_context(patch.object(trained.teacher, 'TEACHER', checkpoint))
        scope.enter_context(patch.object(trained.teacher, 'TEACHER_SHA', terminal['checkpoint_sha256']))
        if args.qualify_cpu:
            assert not torch.cuda.is_available()
            scope.enter_context(patch.object(trained.teacher, 'startup', source_startup))
            argv = ['source-cpu', '--execution-sha256', args.execution_sha256, '--output', str(args.output), '--qualify-cpu']
            scope.enter_context(patch.object(sys, 'argv', argv))
            trained.teacher.main()
            proof = json.loads(args.output.read_text())
            assert proof['teacher_checkpoint_sha256'] == terminal['checkpoint_sha256'] and proof['teacher_whole_sha256'] == terminal['updated_whole_sha256'] and proof['teacher_head_sha256'] == terminal['updated_head_sha256']
            assert proof['read_only'] and proof['optimizer_updates'] == 0 and not proof['quality_read']
            assert all(driver.pair.sha(root / n) == h for n, h in code.items())
            driver.pair.smoke.save(args.output, {**proof, **binding})
            return
        assert args.cpu_proof and args.cpu_sha256 and driver.pair.sha(args.cpu_proof) == args.cpu_sha256
        proof = json.loads(args.cpu_proof.read_text())
        assert all(proof[k] == v for k, v in binding.items()) and proof['code'] == code
        assert proof['teacher_checkpoint_sha256'] == terminal['checkpoint_sha256'] and proof['teacher_whole_sha256'] == terminal['updated_whole_sha256'] and proof['teacher_head_sha256'] == terminal['updated_head_sha256']
        assert proof['strict400_native_head_reload_and_direct_whole_calibration_exact'] and proof['prefix_data_mutation_rejected_at_exit'] and proof['cpu_cuda_rng_unchanged']
        fit = {**proof, 'numerical_flags': prior['numerical_flags']}
        scope.enter_context(patch.object(trained, 'startup', held_startup))
        scope.enter_context(patch.object(trained, 'CPU_SHA', args.cpu_sha256))
        scope.enter_context(patch.object(trained, 'FIT_SHA', args.cpu_sha256))
        argv = ['native-train-held', '--execution-sha256', args.execution_sha256, '--output', str(args.output)]
        if args.audit_cpu:
            assert args.receipt_sha256
            argv += ['--audit-cpu', '--receipt-sha256', args.receipt_sha256]
        scope.enter_context(patch.object(sys, 'argv', argv))
        trained.main()


if __name__ == '__main__':
    main()
