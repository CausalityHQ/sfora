#!/usr/bin/env python3
"""Actual full-TRAIN native valid-anchor authority and gradient qualification."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import qualify_pe_native_valid_anchor_cpu as previous

lane, driver = previous.lane, previous.driver
CONFIRMATION = Path('/home/riomus/runs/sfora-native-valid-anchor-confirmation-v1/decision.json')
CONFIRMATION_SHA = '3b350839a0f88edd5ea472628a914abee685d323fea15e91f213ffa55cfde56a'
FULL_CONTROL = Path('/home/riomus/runs/sfora-large-optimization-full-2000-v1')
FULL_CONTROL_SHA = previous.FULL_TRAIN


def code_guard(root, expected):
    manifest = root / 'full-valid-anchor-cpu-execution.json'
    assert driver.pair.sha(manifest) == expected
    code = json.loads(manifest.read_text())
    old = json.loads((root / 'native-valid-anchor-confirmation-execution.json').read_text())
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    assert all(driver.pair.sha(root / n) == h for n, h in code.items()), 'full valid-anchor source differs'
    return code


def authority(root, expected, allowed=None):
    code = code_guard(root, expected)
    if allowed is not None:
        assert len(allowed) == len(code) + 1 and all(allowed[n] == h for n, h in code.items())
        assert all(driver.pair.sha(root / n) == h for n, h in allowed.items())
    assert driver.pair.SEED == 179032 and driver.TOTAL_UPDATES == 2000
    assert driver.pair.sha(CONFIRMATION) == CONFIRMATION_SHA
    result = json.loads(CONFIRMATION.read_text())
    assert result['decision'] == 'KILL' and result['prior_official_benchmark_exposure'] and not result['claim_eligible']
    assert driver.pair.sha(previous.FULL_OFFICIAL / 'cpu-audit.json') == previous.FULL_AUDIT
    audit = json.loads((previous.FULL_OFFICIAL / 'cpu-audit.json').read_text())
    assert audit['pass'] and not audit['matched_pool_comparison']['survivor']
    assert driver.pair.sha(previous.FULL_OFFICIAL / 'receipt.json') == audit['receipt_sha256']
    helpers = previous.qualified.confirmation.selected.helpers
    with patch.object(previous.qualified.confirmation.selected, 'helpers', lambda r, _: helpers(r, code if allowed is None else allowed)):
        control, source, prior, proof, _, terminal, *_ = previous.qualified.authority(root, previous.PREVIOUS_CODE, 'full', FULL_CONTROL_SHA)
    assert driver.pair.sha(FULL_CONTROL / 'receipt.json') == FULL_CONTROL_SHA and terminal['completed_step'] == 2000
    arm = proof['arms']['full']
    assert len(arm['rows']) == 25882 and len(arm['classes']) == 3997
    inputs = driver.initializers(proof, 'full')
    target = inputs['target'].numpy()
    batches = driver.schedule(target)
    rank = np.bincount(target)[target[batches]] > 1
    assert int((~rank.all(axis=1)).sum()) == 375
    return control, source, prior, proof, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(driver.pair.SEED)
    root = Path(__file__).resolve().parent
    control, source, _, proof, code = authority(root, args.execution_sha256)
    real_sha = driver.pair.sha
    with patch.object(driver.pair, 'sha', lambda p: 'altered' if Path(p) == Path(__file__) else real_sha(p)):
        try:
            code_guard(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'full valid-anchor source differs'
        else:
            raise AssertionError('changed full driver accepted')
    state = driver.fresh(control, source, proof, 'full', 'cpu')
    assert len(state['target']) == 25882 and len(state['classifier']) == 3997
    before = driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    eligible = (state['positive'] >= 0).any(dim=1)
    index = torch.tensor([int(eligible.nonzero()[0]), int((~eligible).nonzero()[0])])
    assert len(set(state['target'][index].tolist())) == 2
    images, rgb = driver.pair.augmented_images(control.dataset_root, proof['arms']['full']['rows'], tuple(index.tolist()), 1)
    pixels = driver.pair.pixels(state['processor'], images, 'large')
    rng = torch.random.get_rng_state().clone()
    pooled = state['model'](pixel_values=pixels).pooler_output.float()
    values = (state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index)
    ce, zero, raw = lane.original_terms(pooled, *values, False)
    new_ce, rank, new_raw = lane.terms(pooled, *values, False)
    assert zero == 0 and torch.equal(ce, new_ce) and torch.equal(raw, new_raw)
    raw.retain_grad()
    rank = lane.valid_rank(raw, state['bank'], state['head'], state['positive'][index], index)
    individual = driver.pair.smoke.member_bank_rank_loss(raw[:1], state['bank'], state['head'], state['positive'][index[:1]], index[:1], live_head=False)
    assert torch.equal(rank, individual * .5)
    micro = sum(lane.valid_rank(raw[i:i+1], state['bank'], state['head'], state['positive'][index[i:i+1]], index[i:i+1]) * .5 for i in range(2))
    assert torch.equal(rank, micro)
    invalid = lane.valid_rank(raw[1:], state['bank'], state['head'], state['positive'][index[1:]], index[1:])
    assert invalid == 0 and invalid.requires_grad
    a = lane.original_terms(pooled[:1], state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index[:1], True)
    b = lane.terms(pooled[:1], state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index[:1], True)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    rank.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0].norm() > 0 and raw.grad[1].count_nonzero() == 0
    gradients = {}
    for name, param in state['model'].named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() if param.requires_grad else param.grad is None
    for i in range(12, 24):
        gradients[str(i)] = sum(float(p.grad.double().norm()) for p in state['model'].encoder.layers[i].parameters())
        assert gradients[str(i)] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in state['head'].parameters())
    assert state['classifier'].grad is None and not state['bank'].requires_grad and not state['optimizer'].state and state['counter'] == 0
    assert before == driver.fingerprint({k: state[k].state_dict() if k in ('model', 'head') else state[k] for k in ('model', 'head', 'classifier', 'bank')})
    assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
    assert torch.equal(rng, torch.random.get_rng_state())
    code_guard(root, args.execution_sha256)
    assert driver.pair.sha(driver.coverage.teacher.TEACHER) == driver.coverage.teacher.TEACHER_SHA
    driver.pair.smoke.save(args.output, {'pass': True, 'arm': 'full', 'execution_sha256': args.execution_sha256, 'code': code, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'control_training_receipt_sha256': FULL_CONTROL_SHA, 'confirmation_decision_sha256': CONFIRMATION_SHA, 'fit_rows': 25882, 'classes': 3997, 'rank_omitted_control_updates': 375, 'eligible_row': int(index[0]), 'singleton_row': int(index[1]), 'rgb_sha256': rgb, 'pixels_sha256': driver.pair.smoke.digest({'pixels': pixels}), 'same_CE_raw_active_rank_exact': True, 'individual_and_micro_denominator_exact': True, 'singleton_rank_gradient_zero_valid_native_gradient_nonzero': True, 'all_invalid_microbatch_graph_zero': True, 'native_gradient_norms': gradients, 'frozen_whole_head_bank_classifier_preserved': True, 'CPU_rng_preserved': True, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False, 'GPU_training_qualified': False, 'claim_eligible': False})
    print('PASS actual full25882/3997 native valid-anchor gradients/source/normalization; zero updates or quality')


if __name__ == '__main__':
    main()
