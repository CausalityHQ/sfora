#!/usr/bin/env python3
"""Independent CPU official packed comparison of the four fixed complete roles.

Freeze unchanged104 plus this driver as105. Parent owns both admission locks
and 8GiB/no-swap limits; startup120 seconds, complete score300 seconds.
"""
import argparse
import inspect
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import export_large_dense_retained_official as export

if not __debug__:
    raise SystemExit('Qualification requires assertions')

OUTPUT = Path('/home/riomus/runs/sfora-dense-retained-official-score-v1/decision.json')


def authenticate(root, execution, export_execution, export_startup_sha, receipts):
    assert Path(inspect.getfile(export)).resolve() == root / export.DRIVER
    assert export.sha(root / export.MANIFEST) == export_execution == export.sha(export.SOURCE / export.MANIFEST)
    retained, inherited, control, frozen, prior, cpus, protocol, decision, code, exported, official = export.startup(root, execution, score=True)
    assert export.sha(root / export.MANIFEST) == export_execution == export.sha(export.SOURCE / export.MANIFEST)
    bound = export.binding(retained, cpus, exported, export_execution)
    export.admission(root, export_startup_sha, bound)
    wires, records = {}, {}
    # Authenticate all four units before the first official quality call.
    for arm, role in export.ORDER:
        packed, record = export.load_role(arm, role, receipts[arm, role], bound, export_startup_sha, protocol)
        assert record['environment'] == cpus[arm]['environment']
        assert record['numerical_flags'] == prior['numerical_flags']
        if role == 'query':
            assert record['gallery_receipt_sha256'] == receipts[arm, 'gallery']
        else:
            assert record['gallery_receipt_sha256'] is None
        wires[arm, role], records[arm, role] = packed, record
    export.loaded_authority(root, code)
    inherited.old.pair.executing_authority(root, code)
    return retained, inherited, frozen, cpus, protocol, decision, code, official, wires, records


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--export-execution-sha256', required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--export-startup-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check-startup-only', action='store_true')
    p.add_argument('--startup-sha256')
    for arm, role in export.ORDER:
        p.add_argument(f'--{arm}-{role}-sha256', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    assert root == export.SCORE_SOURCE and args.protocol_sha256 == export.PROTOCOL_SHA
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1')
    assert args.output == (root / 'startup.json' if args.check_startup_only else OUTPUT)
    assert not args.output.exists() and not args.output.is_symlink() and args.output.parent.resolve() == args.output.parent
    assert args.check_startup_only == (args.startup_sha256 is None)
    started = time.perf_counter()
    cap = 120 if args.check_startup_only else 300
    export.alarm(cap)
    receipts = {(arm, role): getattr(args, f'{arm}_{role}_sha256') for arm, role in export.ORDER}
    values = authenticate(root, args.execution_sha256, args.export_execution_sha256, args.export_startup_sha256, receipts)
    retained, inherited, frozen, cpus, protocol, decision, code, official, wires, records = values
    import torch
    import numpy as np
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    bound = {**export.binding(retained, cpus, code, args.execution_sha256),
        'export_execution_sha256': args.export_execution_sha256,
        'export_startup_receipt_sha256': args.export_startup_sha256,
        'role_receipts': {f'{arm}_{role}': digest for (arm, role), digest in receipts.items()},
        'role_wires': {f'{arm}_{role}': {
            'codes_sha256': record['codes_sha256'], 'inverse_sha256': record['inverse_sha256'],
            'normal_exit_log_sha256': export.sha(export.SOURCE / f'sfora-dense-retained-official-{arm}-{role}-v1.log')}
            for (arm, role), record in records.items()}}
    if args.check_startup_only:
        export.replay_train(inherited, frozen, decision, official)
        original_sha = export.sha
        with patch.object(export, 'sha', lambda f: 'changed' if Path(f).resolve() == Path(__file__).resolve() else original_sha(f)):
            try:
                export.source_authority(root, args.execution_sha256, score=True)
            except AssertionError:
                pass
            else:
                raise AssertionError('changed official scorer accepted')
        # Imported after the first authority pass, then bound by a fresh guard.
        from sfora.siglip2_compact_serving import Siglip2CompactEncoder
        assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / 'src/sfora/siglip2_compact_serving.py'
        authenticate(root, args.execution_sha256, args.export_execution_sha256, args.export_startup_sha256, receipts)
        export.save(args.output, {**bound, 'pass': True, 'read_only': True,
            'changed_driver_rejected': True, 'late_import_checked': True,
            'actual_TRAIN_candidate_control_per_query_exact': True, 'all_four_complete_role_authority': True,
            'model_loaded': False, 'official_images_decoded': 0, 'official_quality_read': False, **export.usage(started, cap)})
        print('PASS independent105 CPU startup; all four roles authenticated, no official quality', flush=True)
        return
    admission = export.read(root / 'startup.json', args.startup_sha256)
    assert all(admission[k] == v for k, v in bound.items())
    assert admission['pass'] and admission['read_only'] and admission['changed_driver_rejected'] and admission['late_import_checked']
    assert admission['actual_TRAIN_candidate_control_per_query_exact'] and admission['all_four_complete_role_authority']
    assert not admission['model_loaded'] and admission['official_images_decoded'] == 0 and not admission['official_quality_read']
    export.resource_receipt(admission, 120)
    qlabels, glabels = ([r['product'] for r in protocol['protocol'][role]] for role in ('query', 'gallery'))
    quality = {arm: export.packed_score(official, wires[arm, 'query'], wires[arm, 'gallery'], qlabels, glabels) for arm in export.ARMS}
    quality_intervals = {}
    for arm in export.ARMS:
        quality_intervals[arm] = {}
        for name in ('per_query_r1', 'per_query_ap'):
            raw = np.asarray(quality[arm][name])
            row = {'mean': float(raw.mean())}
            for kind, groups in (('product', np.asarray(qlabels)), ('query', np.arange(len(raw)))):
                row[kind + '_lower95'] = inherited.old.pair.bootstrap_lower(raw, groups)
                row[kind + '_upper95'] = -inherited.old.pair.bootstrap_lower(-raw, groups)
            quality_intervals[arm][name] = row
    intervals = {}
    for name in ('per_query_r1', 'per_query_ap'):
        delta = np.asarray(quality['candidate'][name]) - np.asarray(quality['control'][name])
        intervals[name] = {'mean_delta': float(delta.mean()), 'per_query_delta': delta.tolist()}
        for kind, groups in (('product', np.asarray(qlabels)), ('query', np.arange(len(delta)))):
            intervals[name][kind + '_lower95'] = inherited.old.pair.bootstrap_lower(delta, groups)
            intervals[name][kind + '_upper95'] = -inherited.old.pair.bootstrap_lower(-delta, groups)
    go = (quality['candidate']['recall_at_1'] > .967 and
        quality['candidate']['map_at_r'] >= quality['control']['map_at_r'] and
        intervals['per_query_r1']['product_lower95'] > 0)
    # Reauthenticate sources, proofs, full wires, metadata and images at exit.
    authenticate(root, args.execution_sha256, args.export_execution_sha256, args.export_startup_sha256, receipts)
    assert export.sha(root / 'startup.json') == args.startup_sha256
    result = {**bound, 'pass': True, 'decision': 'GO' if go else 'KILL',
        'procedure': 'fixed exploratory retained179032 versus matched100-update control; dated necessary screen only',
        'startup_receipt_sha256': args.startup_sha256, 'quality': quality, 'quality_intervals': quality_intervals, 'paired_intervals': intervals,
        'query_images': 14218, 'gallery_images': 12612, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019,
        'minimum_dated_reference_r1': .967, 'advance_minimum_dated_reference_screen': bool(go),
        'strongest_current_reference_verified': False, 'external_target_verified': False,
        'metric': 'original score_asymmetric packed R@1/mAP@R fractions, stable gallery order',
        'interval_scope': 'conditional on frozen weights; paired product/query resampling',
        'official_read': True, 'quality_read': True, 'public_speed_confirmed': False, 'p99_certified': False,
        'fixed_qualification_only': True, 'valid_checkpoints_retained': True,
        'all_four_complete_role_authority': True, 'source_state_proofs_preserved': True,
        **export.usage(started, cap)}
    export.save(args.output, result)
    print('GO necessary dated screen only' if go else 'KILL fixed retained official qualification only', flush=True)


if __name__ == '__main__':
    main()
