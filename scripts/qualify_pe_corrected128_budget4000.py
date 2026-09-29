#!/usr/bin/env python3
"""Terminal actual public128/official qualification for one fixed4000 endpoint."""
if not __debug__:
    raise SystemExit('optimized mode is forbidden')

import argparse
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import train_pe_corrected128_budget4000 as continuation
import qualify_pe_large_coverage_checkpoint as qualified

pair, driver = continuation.pair, continuation.driver
SOURCE = '4f0483cc6c45f2049118fe945eb8836388b53fe00914b5c979208562afe069f9'
CPU_PROOF = Path('/home/riomus/runs/sfora-corrected128-budget4000-cpu-v2/receipt.json')
CPU_SHA = 'b470f4e19cac120f20c2c6c89edd3353e8e9a6959b899c66697e2d3d107e5ba6'
CONTROLLER = Path('/home/riomus/runs/sfora-corrected128-budget4000-controller-v2')
CONTROLLER_SHA = '3689589c93343d16f19f7b61d91ba5c5f1d30bd6b8e4da4eb8edea15087e914e'
WEIGHTS = Path('/home/riomus/runs/sfora-corrected128-budget4000-weights-v1')
PUBLIC = Path('/home/riomus/runs/sfora-corrected128-budget4000-official-v1')
BASELINE = Path('/home/riomus/runs/sfora-full-valid-anchor-official-b32-v1')
BASELINE_SHA = 'fde8508c057057bec0edc515e7c25c28b53686b34ec31c157dad227cdebbed84'
BASELINE_AUDIT = 'e865e1514ceb801f6748e1e7461516a588e5d28f1c679f95060d79598b8d05fd'


def survives(candidate, baseline, comparison):
    values = (candidate['recall_at_1'], candidate['map_at_r'], baseline['map_at_r'],
              comparison['per_query_r1']['product_lower95'])
    return bool(all(math.isfinite(x) for x in values) and candidate['recall_at_1'] > .967
                and candidate['map_at_r'] >= baseline['map_at_r']
                and comparison['per_query_r1']['product_lower95'] > 0)


def run_path(end):
    return Path(f'/home/riomus/runs/sfora-corrected128-budget4000-{end}-v2')


def authority(root, expected):
    manifest = root / 'corrected128-budget4000-qualification-execution.json'
    assert pair.sha(manifest) == expected
    code = json.loads(manifest.read_text())
    oldpath = root / 'corrected128-budget4000-execution.json'
    assert pair.sha(oldpath) == SOURCE
    old = json.loads(oldpath.read_text())
    assert len(old) == 104 and all(code[n] == h for n, h in old.items())
    assert set(code) - set(old) == {'qualify_pe_corrected128_budget4000.py',
                                    'test_pe_corrected128_budget4000_qualification.py'}
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'budget4000 qualification source differs'
    selected = continuation.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, 'helpers', lambda r, _: helpers(r, code)):
        control, source, prior, proof, inherited, parent = continuation.authority(root, SOURCE)
    assert inherited == old and pair.sha(CONTROLLER / 'run_pe_corrected128_budget4000.py') == CONTROLLER_SHA
    assert pair.sha(CPU_PROOF) == CPU_SHA
    admission = json.loads(CPU_PROOF.read_text())
    assert admission['pass'] and admission['execution_sha256'] == SOURCE and admission['total_updates'] == 4000
    state = json.loads((CONTROLLER / 'state.json').read_text())
    assert state['status'] == 'complete' and state['current_end'] == 4000 and len(state['completed']) == 20
    assert state['source_sha256'] == SOURCE and state['cpu_sha256'] == CPU_SHA and not state['quality_read']
    assert all(x in (CONTROLLER / 'controller.log').read_text() for x in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    previous_sha = continuation.PARENT_RECEIPT
    receipts = []
    for end in range(2100, 4001, 100):
        run = run_path(end); path = run / 'receipt.json'; value = json.loads(path.read_text())
        assert value['pass'] and value['intervention'] == continuation.METHOD and value['arm'] == 'full'
        assert value['execution_sha256'] == SOURCE and value['completed_step'] == end and value['chunk_start'] == end - 100
        assert value['total_updates'] == 4000 and value['previous_receipt_sha256'] == previous_sha
        assert value['parent_receipt_sha256'] == continuation.PARENT_RECEIPT and value['parent_checkpoint_sha256'] == continuation.PARENT_CHECKPOINT
        assert value['schedule_sha256'] == admission['schedule_sha256']
        assert value['peak_cuda_allocated_bytes'] < 10_000_000_000 and not value['quality_read']
        assert [s['step'] for s in value['steps']] == list(range(end - 99, end + 1))
        assert all(x in (root / f'sfora-corrected128-budget4000-{end}-v2.log').read_text() for x in
                   ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
        previous_sha = pair.sha(path)
        assert any(r['end'] == end and r['receipt_sha256'] == previous_sha for r in state['completed'])
        receipts.append(value)
    assert pair.sha(run_path(4000) / 'resume.pt') == receipts[-1]['checkpoint_sha256']
    assert all(r['schedule_sha256'] == receipts[0]['schedule_sha256'] for r in receipts)
    assert pair.sha(qualified.PROTOCOL) == qualified.PROTOCOL_SHA
    protocol = json.loads(qualified.PROTOCOL.read_text())
    assert protocol['pass'] and protocol['train_query_gallery_ids_disjoint'] and protocol['prior_official_benchmark_exposure']
    assert pair.sha(control.dataset_root / 'Eval/list_eval_partition.txt') == protocol['partition_sha256'] == qualified.selected.PARTITION_SHA
    official = helpers(root, code)
    return control, source, prior, proof, protocol, official, code, receipts, parent


def export(root, args, proof, code, receipts):
    assert not torch.cuda.is_available() and args.output == WEIGHTS and not WEIGHTS.exists()
    terminal = receipts[-1]; disk = torch.load(run_path(4000) / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    identity = disk['identity']
    assert identity['global_step'] == identity['total_updates'] == 4000 and identity['intervention'] == continuation.METHOD
    assert identity['execution_sha256'] == SOURCE and identity['seed'] == 179032 and identity['arm'] == 'full'
    assert identity['schedule_sha256'] == terminal['schedule_sha256'] and identity['source_checkpoint_sha256'] == driver.coverage.teacher.TEACHER_SHA
    assert identity['initializers_sha256'] == proof['arms']['full']['initializers_sha256'] and identity['frozen_prefix_sha256'] == proof['frozen_prefix_sha256']
    assert len(disk['vision']) == 400 and disk['head']['weight'].shape == (128, 1024)
    assert disk['bank'].shape == (25882, 128) and disk['classifier'].shape == (3997, 128)
    assert driver.fingerprint(disk['buffers']) == identity['buffers_sha256']
    assert len(disk['optimizer']['state']) == len(identity['parameter_names']) and all(int(s['step']) == 4000 for s in disk['optimizer']['state'].values())
    assert identity['precision'] == 'cuda_fp16' and disk['scaler'] and len(disk['cuda_rng']) == 1
    assert all(torch.isfinite(v).all() for role in ('vision', 'head', 'buffers') for v in disk[role].values())
    assert torch.isfinite(disk['bank']).all() and torch.isfinite(disk['classifier']).all()
    assert all(torch.isfinite(v).all() for s in disk['optimizer']['state'].values() for v in s.values() if isinstance(v, torch.Tensor))
    WEIGHTS.mkdir(exist_ok=False)
    torch.save({'vision': disk['vision'], 'head': disk['head']}, WEIGHTS / 'native.pt')
    whole = pair.smoke.digest({**disk['vision'], **{'runtime.' + n: v for n, v in disk['buffers'].items()}})
    head = pair.smoke.digest(disk['head'])
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(WEIGHTS / 'receipt.json', {'pass': True, 'intervention': continuation.METHOD, 'arm': 'full',
        'completed_step': 4000, 'execution_sha256': args.execution_sha256,
        'training_receipt_sha256': pair.sha(run_path(4000) / 'receipt.json'), 'source_resume_sha256': terminal['checkpoint_sha256'],
        'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'checkpoint_sha256': pair.sha(WEIGHTS / 'native.pt'),
        'updated_whole_sha256': whole, 'updated_head_sha256': head,
        'aggregate_continuation_training_wall_seconds': sum(r['training_wall_seconds'] for r in receipts),
        'strict400_terminal_vision_head_export': True, 'optimizer_updates': 0, 'quality_read': False})
    print('PASS fixed4000 CPU native export; no official quality')


def public(root, args, control, source, prior, proof, protocol, official, code, receipts):
    assert pair.sha(WEIGHTS / 'receipt.json') == args.weights_sha256
    value = json.loads((WEIGHTS / 'receipt.json').read_text())
    assert value['pass'] and value['execution_sha256'] == args.execution_sha256
    assert value['training_receipt_sha256'] == pair.sha(run_path(4000) / 'receipt.json')
    assert pair.sha(WEIGHTS / 'native.pt') == value['checkpoint_sha256']

    def startup(r, execution, arm, training):
        assert r == root and execution == args.execution_sha256 and arm == 'full' and training == value['training_receipt_sha256']
        assert all(pair.sha(root / n) == h for n, h in code.items()), 'coverage checkpoint code differs'
        assert pair.sha(WEIGHTS / 'native.pt') == value['checkpoint_sha256']
        return control, source, prior, proof, WEIGHTS, value, protocol, official, code

    updated, pixels = qualified.updated, pair.pixels
    processors = []

    def original_updated(*values):
        result = updated(*values); processors.append(result[2]); return result

    def original_pixels(supplied, images, arm):
        assert len(processors) == 1
        if args.phase == 'public':
            assert supplied is not processors[0]
        return pixels(processors[0], images, arm)

    argv = ['corrected128-budget4000', '--execution-sha256', args.execution_sha256, '--arm', 'full',
            '--training-sha256', value['training_receipt_sha256'], '--output', str(args.output)]
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


def decision(root, args, protocol, code, receipts):
    assert not torch.cuda.is_available() and not args.output.exists()
    assert pair.sha(BASELINE / 'receipt.json') == BASELINE_SHA and pair.sha(BASELINE / 'cpu-audit.json') == BASELINE_AUDIT
    assert args.receipt_sha256 and args.audit_sha256
    assert pair.sha(PUBLIC / 'receipt.json') == args.receipt_sha256 and pair.sha(PUBLIC / 'cpu-audit.json') == args.audit_sha256
    old, new = (json.loads((p / 'receipt.json').read_text()) for p in (BASELINE, PUBLIC))
    a, b = (json.loads((p / 'cpu-audit.json').read_text()) for p in (BASELINE, PUBLIC))
    assert a['pass'] and b['pass'] and a['receipt_sha256'] == BASELINE_SHA and b['receipt_sha256'] == args.receipt_sha256
    assert old['arm'] == new['arm'] == 'full' and new['code'] == code and new['execution_sha256'] == args.execution_sha256
    assert new['checkpoint_sha256'] == json.loads((WEIGHTS / 'receipt.json').read_text())['checkpoint_sha256']
    assert new['query_images'] == old['query_images'] == 14218 and new['gallery_images'] == old['gallery_images'] == 12612
    assert new['batch'] == old['batch'] == 32 and new['native_all_query_top10_ordinal_score_bits_exact']
    assert new['source_state_rng_environment_code_library_preserved'] and new['independent_updated_native_original_preprocessor_first32_each_role_packed_exact']
    for path, value in ((BASELINE, old), (PUBLIC, new)):
        for role in ('query', 'gallery'):
            for field in ('codes', 'inverse'):
                assert pair.sha(path / (role + '.' + field + '.npy')) == value[role + '_' + field + '_sha256']
    for phase in ('export', 'cpu', 'public', 'audit'):
        log = (root / ('budget4000-' + phase + '-v1.log')).read_text()
        assert all(s in log for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    labels = np.asarray([r['product'] for r in protocol['protocol']['query']]); effect = {}
    for metric in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(b['quality'][metric]) - np.asarray(a['quality'][metric])
        assert delta.shape == (14218,) and np.isfinite(delta).all()
        effect[metric] = {'mean_difference': float(delta.mean())}
        for kind, groups in (('product', labels), ('query', np.arange(len(delta)))):
            effect[metric][kind + '_lower95'] = pair.bootstrap_lower(delta, groups)
            effect[metric][kind + '_upper95'] = -pair.bootstrap_lower(-delta, groups)
    outcome = survives(b['quality'], a['quality'], effect)
    parent_cost = 0.
    for end in range(100, 2001, 100):
        value = json.loads((Path(f'/home/riomus/runs/sfora-full-valid-anchor-{end}-v1') / 'receipt.json').read_text())
        assert value['pass'] and value['completed_step'] == end and value['intervention'] == 'full-valid-anchor-v1'
        parent_cost += value['training_wall_seconds']
    new_cost = sum(r['training_wall_seconds'] for r in receipts)
    assert math.isclose(parent_cost, 2441.790976, rel_tol=0, abs_tol=.001)
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / 'decision.json', {'pass': True, 'decision': 'GO' if outcome else 'KILL',
        'procedure_only': True, 'execution_sha256': args.execution_sha256, 'code': code,
        'baseline_receipt_sha256': BASELINE_SHA, 'baseline_audit_sha256': BASELINE_AUDIT,
        'candidate_receipt_sha256': args.receipt_sha256, 'candidate_audit_sha256': args.audit_sha256,
        'baseline_quality': a['quality'], 'candidate_quality': b['quality'], 'comparison': effect,
        'archived_first2000_training_wall_seconds': parent_cost, 'additional2000_training_wall_seconds': new_cost,
        'total4000_training_wall_seconds': parent_cost + new_cost, 'additional_training_updates': 2000,
        'dataset_split': 'previously observed DeepFashion In-Shop official14218q12612g',
        'dated_reference_screen_pass': b['quality']['recall_at_1'] > .967,
        'independent_untouched_confirmation': False, 'current_strongest_reference_qualified': False,
        'public_latency_measured': False, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019,
        'full_production_quality_speed_goal_achieved': False})
    print(('GO' if outcome else 'KILL') + ' fixed corrected128 budget4000 procedure; overall product goal remains active')


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--phase', choices=('export', 'cpu', 'public', 'audit', 'decision'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--weights-sha256'); p.add_argument('--cpu-proof', type=Path); p.add_argument('--cpu-sha256')
    p.add_argument('--receipt-sha256'); p.add_argument('--audit-sha256')
    args = p.parse_args(); root = Path(__file__).resolve().parent
    torch.set_num_threads(8); torch.manual_seed(pair.SEED)
    control, source, prior, proof, protocol, official, code, receipts, _ = authority(root, args.execution_sha256)
    if args.phase == 'export':
        export(root, args, proof, code, receipts)
    elif args.phase == 'decision':
        decision(root, args, protocol, code, receipts)
    else:
        public(root, args, control, source, prior, proof, protocol, official, code, receipts)


if __name__ == '__main__':
    main()
