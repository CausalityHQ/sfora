#!/usr/bin/env python3
"""Apply the TRAIN-surviving fixed average to corrected full native training."""
if not __debug__:
    raise SystemExit('Qualification requires Python assertions; optimized mode is forbidden')

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import qualify_pe_full_valid_anchor_checkpoint as endpoint
import qualify_pe_native_trajectory_average as average

pair, qualified = endpoint.pair, endpoint.qualified
OLD_CODE = '4dcc6227059da973ff4b9c400264b8a7a4e6e5af03b7f40fa5df157d4c4d6375'
TRAIN = 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
TRAIN_GO = Path('/home/riomus/runs/sfora-native-trajectory-average-decision-v1/decision.json')
TRAIN_GO_SHA = '51487110ff2c6f225e54e3ed1e41a2240d32afd2197b5a4b39143cb1e972c509'
WEIGHTS = Path('/home/riomus/runs/sfora-full-native-trajectory-average-weights-v1')
BASELINE = Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1')
BASELINE_SHA = 'fde8508c057057bec0edc515e7c25c28b53686b34ec31c157dad227cdebbed84'
BASELINE_AUDIT = 'e865e1514ceb801f6748e1e7461516a588e5d28f1c679f95060d79598b8d05fd'


def require_train_go(value):
    assert value['pass'] and value['decision'] == 'GO' and value['integrity_resource_pass']
    assert value['additional_training_updates'] == 0 and average.survives(value['comparison'])
    assert value['execution_sha256'] == '765a5a01fd2163ec2a22230620c937646c7a0cdd3a1f38dedaa16971fcd64027'
    assert value['query_images'] == 6354 and value['gallery_images'] == 6245
    assert value['bootstrap_draws'] == 5000 and value['bootstrap_seed'] == 179019 and not value['official_read']


def authority(root, execution):
    manifest = root / 'full-native-average-execution.json'
    assert pair.sha(manifest) == execution
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'full average source differs'
    old_manifest = root / 'full-valid-anchor-checkpoint-execution.json'
    assert pair.sha(old_manifest) == OLD_CODE
    old = json.loads(old_manifest.read_text())
    assert all(code[n] == h == pair.sha(root / n) for n, h in old.items())
    assert set(code) - set(old) == {'qualify_pe_full_native_trajectory_average.py', 'qualify_pe_native_trajectory_average.py', 'evaluate_unicom_checkpoint_soup.py', 'test_pe_full_native_trajectory_average.py', 'full-fixed-average-plan.md'}
    assert pair.sha(TRAIN_GO) == TRAIN_GO_SHA
    go = json.loads(TRAIN_GO.read_text()); require_train_go(go)
    for value in go['matched_receipts']:
        run = Path('/home/riomus/runs/sfora-native-trajectory-average-' + value['variant'] + '-held-v1')
        assert pair.sha(run / 'receipt.json') == value['receipt_sha256'] and pair.sha(run / 'cpu-audit.json') == value['audit_sha256']
    helpers = qualified.selected.helpers
    # Close historical authority before binding the extended imported source inventory.
    with patch.object(qualified.selected, 'helpers', lambda r, _: helpers(r, code)):
        result = endpoint.authority(root, OLD_CODE, TRAIN)
    assert result[-2] == old
    return (*result[:-2], code, result[-1])


def export(root, args):
    assert not torch.cuda.is_available() and args.output == WEIGHTS and not args.output.exists()
    started = time.perf_counter()
    result = authority(root, args.execution_sha256)
    _, _, _, proof, _, terminal, _, _, code, _ = result
    real_sha = pair.sha
    with patch.object(pair, 'sha', lambda p: '0' * 64 if Path(p).resolve() == Path(__file__).resolve() else real_sha(p)):
        try:
            authority(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'full average source differs'
        else:
            raise AssertionError('changed full average driver accepted')
    states, inputs = [], []
    initial = None
    for step in average.STEPS:
        run = Path(f'/home/riomus/runs/sfora-full-valid-anchor-{step}-v1')
        value = json.loads((run / 'receipt.json').read_text())
        assert value['pass'] and value['completed_step'] == step and value['execution_sha256'] == endpoint.TRAIN_CODE
        assert pair.sha(run / 'resume.pt') == value['checkpoint_sha256']
        state = torch.load(run / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
        identity = state['identity']
        assert identity['global_step'] == step and identity['total_updates'] == 2000 and identity['arm'] == 'full' and identity['seed'] == 179032
        assert identity['intervention'] == 'full-valid-anchor-v1' and identity['execution_sha256'] == endpoint.TRAIN_CODE
        assert identity['source_checkpoint_sha256'] == endpoint.driver.coverage.teacher.TEACHER_SHA and identity['schedule_sha256'] == terminal['schedule_sha256']
        assert identity['frozen_prefix_sha256'] == proof['frozen_prefix_sha256'] and len(state['vision']) == 400
        assert endpoint.driver.fingerprint(state['buffers']) == identity['buffers_sha256']
        assert identity['parameter_names'] == [n for n, train in identity['model_roles'] if train] + ['compact_head.' + n for n in state['head']] + ['classifier']
        current = {k: v for k, v in identity.items() if k != 'global_step'}
        initial = current if initial is None else initial
        assert current == initial
        states.append(state); inputs.append({'step': step, 'receipt_sha256': pair.sha(run / 'receipt.json'), 'resume_sha256': value['checkpoint_sha256']})
    serving = average.average_native(tuple(states))
    assert any(not average.same_bytes(serving['vision'][n], states[-1]['vision'][n]) for n, train in initial['model_roles'] if train)
    assert any(not average.same_bytes(v, states[-1]['head'][n]) for n, v in serving['head'].items())
    args.output.mkdir(exist_ok=False)
    torch.save(serving, args.output / 'native.pt')
    whole = {**serving['vision'], **{'runtime.' + n: v for n, v in states[-1]['buffers'].items()}}
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output / 'receipt.json', {'pass': True, 'arm': 'full', 'intervention': 'fixed-native-trajectory-average-v1', 'completed_step': 2000,
        'execution_sha256': args.execution_sha256, 'training_receipt_sha256': TRAIN, 'train_gate_sha256': TRAIN_GO_SHA,
        'source_checkpoint_sha256': endpoint.driver.coverage.teacher.TEACHER_SHA, 'inputs': inputs, 'steps': average.STEPS,
        'checkpoint_sha256': pair.sha(args.output / 'native.pt'), 'updated_whole_sha256': pair.smoke.digest(whole), 'updated_head_sha256': pair.smoke.digest(serving['head']),
        'frozen_runtime_buffers_exact': True, 'trainable_vision_and_head_fp64_uniform_mean': True, 'changed_driver_rejected': True,
        'optimizer_updates': 0, 'quality_read': False, 'training_state_resumable': False, 'export_wall_seconds': time.perf_counter() - started,
        'aggregate_archived_training_wall_seconds': result[-1]})
    print('PASS exact TRAIN-surviving full corrected trajectory average; no quality read')


def decision(root, args):
    assert not torch.cuda.is_available() and not args.output.exists()
    _, _, _, _, _, _, protocol, _, code, _ = authority(root, args.execution_sha256)
    assert pair.sha(BASELINE / 'receipt.json') == BASELINE_SHA and pair.sha(BASELINE / 'cpu-audit.json') == BASELINE_AUDIT
    assert args.candidate and args.receipt_sha256 and args.audit_sha256
    assert pair.sha(args.candidate / 'receipt.json') == args.receipt_sha256 and pair.sha(args.candidate / 'cpu-audit.json') == args.audit_sha256
    base, candidate = (json.loads((r / 'receipt.json').read_text()) for r in (BASELINE, args.candidate))
    a, b = (json.loads((r / 'cpu-audit.json').read_text()) for r in (BASELINE, args.candidate))
    assert a['pass'] and b['pass'] and a['receipt_sha256'] == BASELINE_SHA and b['receipt_sha256'] == args.receipt_sha256
    assert base['code'] == json.loads((root / 'full-valid-anchor-checkpoint-execution.json').read_text()) and candidate['code'] == code
    assert base['arm'] == candidate['arm'] == 'full' and base['training_receipt_sha256'] == candidate['training_receipt_sha256'] == TRAIN
    assert candidate['execution_sha256'] == args.execution_sha256 and candidate['checkpoint_sha256'] == json.loads((WEIGHTS / 'receipt.json').read_text())['checkpoint_sha256']
    for value in (base, candidate):
        assert value['query_images'] == 14218 and value['gallery_images'] == 12612 and value['batch'] == 32
        assert value['native_all_query_top10_ordinal_score_bits_exact'] and value['source_state_rng_environment_code_library_preserved']
    for run, value in ((BASELINE, base), (args.candidate, candidate)):
        for role in ('query', 'gallery'):
            for field in ('codes', 'inverse'):
                assert pair.sha(run / (role + '.' + field + '.npy')) == value[role + '_' + field + '_sha256']
    for phase in ('average-export', 'updated-cpu', 'official-public', 'official-audit'):
        log = (root / (phase + '.log')).read_text()
        assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
    labels = np.asarray([r['product'] for r in protocol['protocol']['query']])
    effect = {}
    for metric in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(b['quality'][metric]) - np.asarray(a['quality'][metric])
        assert delta.shape == (14218,) and np.isfinite(delta).all()
        effect[metric] = {'mean_difference': float(delta.mean())}
        for kind, groups in (('product', labels), ('query', np.arange(len(delta)))):
            effect[metric][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            effect[metric][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    next_stage = bool(b['quality']['recall_at_1'] > .967 and all(v['product_lower95'] > 0 for v in effect.values()))
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'decision.json', {'pass': True, 'decision': 'GO' if next_stage else 'KILL', 'procedure_only': True,
        'train_gate_sha256': TRAIN_GO_SHA, 'execution_sha256': args.execution_sha256, 'code': code,
        'baseline_receipt_sha256': BASELINE_SHA, 'baseline_audit_sha256': BASELINE_AUDIT, 'candidate_receipt_sha256': args.receipt_sha256, 'candidate_audit_sha256': args.audit_sha256,
        'endpoint_quality': a['quality'], 'average_quality': b['quality'], 'comparison': effect,
        'protocol_sha256': qualified.PROTOCOL_SHA, 'dataset_split': 'previously observed In-Shop official14218q12612g',
        'dated_reference_screen_pass': b['quality']['recall_at_1'] > .967, 'current_strongest_reference_qualified': False,
        'additional_training_updates': 0, 'public_latency_measured': False, 'independent_untouched_confirmation': False,
        'bootstrap_draws': 5000, 'bootstrap_seed': 179019, 'full_production_quality_speed_goal_achieved': False})
    print(('GO' if next_stage else 'KILL') + ' fixed full averaging procedure; preserve valid candidates and overall production goal')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--phase', choices=('export', 'cpu', 'public', 'audit', 'decision'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-proof', type=Path)
    p.add_argument('--cpu-sha256')
    p.add_argument('--receipt-sha256')
    p.add_argument('--audit-sha256')
    p.add_argument('--candidate', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8); torch.manual_seed(pair.SEED)
    if args.phase == 'export':
        export(root, args); return
    if args.phase == 'decision':
        decision(root, args); return
    control, source, prior, proof, _, _, protocol, official, code, _ = authority(root, args.execution_sha256)
    value = json.loads((WEIGHTS / 'receipt.json').read_text())
    assert value['pass'] and value['execution_sha256'] == args.execution_sha256 and value['train_gate_sha256'] == TRAIN_GO_SHA
    assert value['training_receipt_sha256'] == TRAIN and value['steps'] == list(average.STEPS) and value['changed_driver_rejected']
    assert pair.sha(WEIGHTS / 'native.pt') == value['checkpoint_sha256']
    def startup(r, execution, arm, training):
        assert r == root and execution == args.execution_sha256 and arm == 'full' and training == TRAIN
        assert all(pair.sha(root / n) == h for n, h in code.items()), 'coverage checkpoint code differs'
        assert pair.sha(WEIGHTS / 'native.pt') == value['checkpoint_sha256']
        return control, source, prior, proof, WEIGHTS, value, protocol, official, code
    updated, pixels = qualified.updated, pair.pixels
    processors = []
    def original_updated(*values):
        result = updated(*values); processors.append(result[2]); return result
    def original_pixels(supplied, images, arm):
        assert len(processors) == 1
        # Public encoder calls its own processor; native sentinel uses its distinct original.
        if args.phase == 'public':
            assert supplied is not processors[0]
        return pixels(processors[0], images, arm)
    argv = ['full-averaged-native', '--execution-sha256', args.execution_sha256, '--arm', 'full', '--training-sha256', TRAIN, '--output', str(args.output)]
    if args.phase == 'cpu':
        argv += ['--qualify-cpu']
    else:
        assert args.cpu_proof and args.cpu_sha256
        argv += ['--cpu-proof', str(args.cpu_proof), '--cpu-sha256', args.cpu_sha256]
        if args.phase == 'audit':
            assert args.receipt_sha256
            argv += ['--audit-cpu', '--receipt-sha256', args.receipt_sha256]
    with ExitStack() as scope:
        scope.enter_context(patch.object(qualified, 'startup', startup))
        scope.enter_context(patch.object(qualified, 'updated', original_updated))
        scope.enter_context(patch.object(pair, 'pixels', original_pixels))
        scope.enter_context(patch.object(sys, 'argv', argv))
        qualified.main()


if __name__ == '__main__':
    main()
