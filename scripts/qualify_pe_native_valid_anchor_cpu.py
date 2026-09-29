#!/usr/bin/env python3
"""Actual native fit-pixel gradient and denominator qualification; no updates."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import torch
import pe_native_valid_anchor as lane
import qualify_pe_large_optimization_checkpoint as qualified

driver = lane.driver
PREVIOUS_CODE = '8c6771b3fc63de4967120c6ae1a388b8e5311a454bfd060716488859fac6aa0c'
FULL_TRAIN = 'dc42d59c4c14a8dfc53b524fd1afbf50a4944d4e5818a1bba3c3f28aefa58412'
FULL_AUDIT = '8fbe0043f148dde48e04a67dc270b8e2f61c175d9a75142628a05de10989453f'
FULL_OFFICIAL = Path('/home/riomus/runs/sfora-large-optimization-full-official-b32-v1')


def code_guard(root, expected):
    manifest = root / 'native-valid-anchor-execution.json'
    assert driver.pair.sha(manifest) == expected
    code = json.loads(manifest.read_text())
    assert all(driver.pair.sha(root / n) == h for n, h in code.items()), 'native valid-anchor code differs'
    assert driver.pair.sha(root / 'large-optimization-checkpoint-execution.json') == PREVIOUS_CODE
    prior = json.loads((root / 'large-optimization-checkpoint-execution.json').read_text())
    assert len(code) == len(prior) + 2 and all(code[n] == h for n, h in prior.items())
    return code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(driver.pair.SEED)
    root = Path(__file__).resolve().parent
    code = code_guard(root, args.execution_sha256)
    real_sha = driver.pair.sha
    with patch.object(driver.pair, 'sha', lambda p: '0' * 64 if Path(p) == Path(__file__).resolve() else real_sha(p)):
        try:
            code_guard(root, args.execution_sha256)
        except AssertionError:
            pass
        else:
            raise AssertionError('changed driver accepted')
    assert real_sha(FULL_OFFICIAL / 'cpu-audit.json') == FULL_AUDIT
    audit = json.loads((FULL_OFFICIAL / 'cpu-audit.json').read_text())
    assert audit['pass'] and not audit['matched_pool_comparison']['survivor']
    assert real_sha(FULL_OFFICIAL / 'receipt.json') == audit['receipt_sha256']
    control, source, _, proof, *_ = qualified.authority(root, PREVIOUS_CODE, 'full', FULL_TRAIN)
    state = driver.fresh(control, source, proof, 'half', 'cpu')
    before = driver.fingerprint({'vision': state['model'].state_dict(), 'head': state['head'].state_dict(), 'bank': state['bank']})
    target = state['target']
    eligible = (state['positive'] >= 0).any(dim=1)
    valid_row, singleton = int(eligible.nonzero()[0]), int((~eligible).nonzero()[0])
    index = torch.tensor([valid_row, singleton])
    assert len(set(target[index].tolist())) == 2
    arm = proof['arms']['half']
    images, rgb = driver.pair.augmented_images(control.dataset_root, arm['rows'], tuple(index.tolist()), 1)
    pixels = driver.pair.pixels(state['processor'], images, 'large')
    pooled = state['model'](pixel_values=pixels).pooler_output.float()
    values = (state['head'], state['classifier'], state['bank'], target, state['positive'], index)
    ce, zero, raw = lane.original_terms(pooled, *values, False)
    new_ce, rank, new_raw = lane.terms(pooled, *values, False)
    assert zero == 0 and torch.equal(ce, new_ce) and torch.equal(raw, new_raw)
    raw.retain_grad()
    # Use the same actual raw tensor for rank-only gradient observation.
    rank = lane.valid_rank(raw, state['bank'], state['head'], state['positive'][index], index)
    individual = driver.pair.smoke.member_bank_rank_loss(raw[:1], state['bank'], state['head'], state['positive'][index[:1]], index[:1], live_head=False)
    assert torch.equal(rank, individual * .5)
    micro = sum(lane.valid_rank(raw[i:i+1], state['bank'], state['head'], state['positive'][index[i:i+1]], index[i:i+1]) * .5 for i in range(2))
    assert torch.equal(rank, micro)
    invalid_zero = lane.valid_rank(raw[1:], state['bank'], state['head'], state['positive'][index[1:]], index[1:])
    assert invalid_zero == 0 and invalid_zero.requires_grad
    a = lane.original_terms(pooled[:1], state['head'], state['classifier'], state['bank'], target, state['positive'], index[:1], True)
    b = lane.terms(pooled[:1], state['head'], state['classifier'], state['bank'], target, state['positive'], index[:1], True)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    rank.backward()
    assert torch.isfinite(raw.grad).all() and raw.grad[0].norm() > 0 and raw.grad[1].count_nonzero() == 0
    gradients = {}
    for name, param in state['model'].named_parameters():
        if param.requires_grad:
            assert param.grad is not None and torch.isfinite(param.grad).all()
        else:
            assert param.grad is None
    for i in range(12, 24):
        gradients[str(i)] = sum(float(p.grad.double().norm()) for p in state['model'].encoder.layers[i].parameters())
        assert gradients[str(i)] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in state['head'].parameters())
    assert state['classifier'].grad is None and not state['bank'].requires_grad and not state['optimizer'].state
    assert before == driver.fingerprint({'vision': state['model'].state_dict(), 'head': state['head'].state_dict(), 'bank': state['bank']})
    assert driver.coverage.frozen_digest(state['model'], state['inventory']) == proof['frozen_prefix_sha256']
    code_guard(root, args.execution_sha256)
    driver.pair.smoke.save(args.output, {'pass': True, 'execution_sha256': args.execution_sha256, 'code': code, 'source_checkpoint_sha256': driver.coverage.teacher.TEACHER_SHA, 'fit_rows': len(arm['rows']), 'eligible_row': valid_row, 'singleton_row': singleton, 'rgb_sha256': rgb, 'pixels_sha256': driver.pair.smoke.digest({'pixels': pixels}), 'same_CE_raw_active_rank_exact': True, 'individual_and_micro_denominator_exact': True, 'singleton_rank_gradient_zero_valid_native_gradient_nonzero': True, 'all_invalid_microbatch_graph_zero': True, 'native_gradient_norms': gradients, 'frozen_whole_head_bank_preserved': True, 'changed_driver_rejected': True, 'optimizer_updates': 0, 'quality_read': False, 'GPU_training_qualified': False, 'claim_eligible': False})
    print('PASS actual native F5 valid-anchor gradients and denominator; no training or quality')


if __name__ == '__main__':
    main()
