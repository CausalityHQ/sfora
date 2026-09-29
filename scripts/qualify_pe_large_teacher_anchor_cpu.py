#!/usr/bin/env python3
"""Actual native CPU gradient proof for the fixed teacher-anchor objective."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import pe_large_teacher_targets as cache
from pe_large_teacher_anchor import anchor, check

native, base, pair = cache.export.native, cache.export.base, cache.export.pair
CACHE_CPU = Path('/home/riomus/runs/sfora-large-teacher-fit-v1/teacher-cache-cpu-v1.json')
CACHE_CPU_SHA = '94d39956c4ede8a128e58a3ae7278cd9ed0d3f8fc45f266abf36166b322d18be'


def startup(root, execution_sha):
    path = root / 'teacher-anchor-execution.json'
    assert pair.sha(path) == execution_sha
    code = json.loads(path.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'anchor execution code differs'
    control, frozen, old = cache.export.startup(root, cache.EXECUTION_SHA)
    assert all(code[n] == h for n, h in old.items())
    assert pair.sha(CACHE_CPU) == CACHE_CPU_SHA
    qualified = json.loads(CACHE_CPU.read_text())
    assert qualified['consumer_sha256'] == code['pe_large_teacher_targets.py']
    assert qualified['cache_hash_and_image_product_row_rejection'] and qualified['constant_cpu_targets']
    assert qualified['export_receipt_sha256'] == cache.RECEIPT_SHA
    targets, receipt = cache.load(frozen, old)
    return control, frozen, targets, receipt, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    check()
    root = Path(__file__).resolve().parent
    control, frozen, targets, receipt, code = startup(root, args.execution_sha256)
    _, _, _, cpu, _ = cache.export.qualified.startup(root, cache.export.QUAL_SHA)
    model, processor = pair.smoke.load_arm(control, 'large')
    inventory = native.freeze(model)
    assert inventory == {k: tuple(v) for k, v in cpu['inventory'].items()}
    assert pair.smoke.digest(base.whole_state(model)) == cpu['whole_original_source_sha256']
    assert json.loads(json.dumps(native.environment(model, processor))) == cpu['environment']
    head = nn.Linear(1024, 128)
    assert pair.sha(base.INIT / 'initializers.npz') == 'd0118caf6a87d779e892e70a3e3ad5f1026bfa783f661d0e3a53c5147ce880a6'
    with np.load(base.INIT / 'initializers.npz', allow_pickle=False) as init:
        head.load_state_dict({k: torch.from_numpy(init['large.head.' + k]) for k in ('weight', 'bias')})
        classifier = nn.Parameter(torch.from_numpy(init['large.classifier'].copy()))
        bank = torch.from_numpy(init['large.bank'].copy())
    head_sha = pair.smoke.digest(head.state_dict())
    proxy_sha = pair.smoke.digest({'classifier': classifier})
    targets_sha = pair.smoke.digest({'targets': targets})
    active = [p for p in model.parameters() if p.requires_grad]
    members = active + list(head.parameters()) + [classifier]
    optimizer = torch.optim.AdamW([{'params': active, 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-4}, {'params': [classifier], 'lr': 1e-4}], weight_decay=0.05)
    assert len(active) == 32 and len(members) == len({id(p) for p in members}) == 35 and not optimizer.state
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in members}
    images, _ = pair.augmented_images(control.dataset_root, frozen['fit_manifest'], (0, 1), None)
    pixels = pair.pixels(processor, images, 'large')
    rng = torch.random.get_rng_state().clone()
    seen = {}

    def observe(module, inputs):
        value = inputs[0]
        if module is model.encoder.layers[-2]:
            base.assert_frozen_tokens((value,))
            key = 'block22'
        else:
            assert value.requires_grad and value.grad_fn is not None and value.dtype == torch.float32
            key = 'block23' if module is model.encoder.layers[-1] else 'pool'
        seen[key] = list(value.shape)

    hooks = [m.register_forward_pre_hook(observe) for m in (model.encoder.layers[-2], model.encoder.layers[-1], model.head)]
    try:
        source = model(pixel_values=pixels).pooler_output
    finally:
        for hook in hooks:
            hook.remove()
    assert seen == {k: [2, 256, 1024] for k in ('block22', 'block23', 'pool')}
    raw = pair.smoke.compact_head_features(source, head)
    index = torch.tensor((0, 1))
    positives = pair.smoke.member_bank_positive_ordinals(np.asarray(frozen['target'], dtype=np.int64), allow_singletons=True)
    ce = pair.smoke.sharded_mask_arcface_loss(raw, classifier, torch.tensor(frozen['target'])[index], torch.arange(128).unsqueeze(0), margin=0.3, scale=64)
    rank = pair.smoke.member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False)
    geometry = anchor(raw, targets[index])
    anchor_grad = torch.autograd.grad(geometry, members[:-1], retain_graph=True)
    assert len(anchor_grad) == 34 and all(torch.isfinite(g).all() and g.norm() > 0 for g in anchor_grad)
    loss = ce + 8 * rank + geometry
    assert torch.isfinite(source).all() and torch.isfinite(raw).all() and torch.isfinite(loss) and geometry >= 0
    loss.backward()
    assert all((p.grad is None) == (not p.requires_grad) for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in members)
    assert torch.equal(rng, torch.random.get_rng_state()) and not optimizer.state
    assert pair.smoke.digest(base.whole_state(model)) == cpu['whole_original_source_sha256']
    assert pair.smoke.digest(native.frozen_state(model)) == cpu['frozen_complement_sha256']
    assert pair.smoke.digest(head.state_dict()) == head_sha and pair.smoke.digest({'classifier': classifier}) == proxy_sha
    assert pair.smoke.digest({'targets': targets}) == targets_sha and targets.grad is None
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output, {'execution_sha256': args.execution_sha256, 'code': code, 'teacher_export_sha256': cache.RECEIPT_SHA, 'cache_cpu_authority_sha256': CACHE_CPU_SHA, 'native_cpu_authority_sha256': cache.export.qualified.CPU_SHA, 'environment': cpu['environment'], 'inventory': inventory, 'whole_original_source_sha256': cpu['whole_original_source_sha256'], 'frozen_complement_sha256': cpu['frozen_complement_sha256'], 'objective': 'CE + 8*rank + 32*mean(sum((normalize(raw)-teacher_unit)**2))', 'losses': {'ce': float(ce.detach()), 'rank': float(rank.detach()), 'anchor': float(geometry.detach()), 'total': float(loss.detach())}, 'anchor_only_gradient_norms': [float(g.norm()) for g in anchor_grad], 'combined_gradient_norms': [float(p.grad.norm()) for p in members], 'graph': seen, 'teacher_constant_content_preserved': True, 'original_model_head_proxy_rng_preserved': True, 'optimizer_updates': 0, 'held_images': 0, 'quality_read': False, 'cuda': False})
    print('PASS actual CPU original-native anchor-only34/combined35 finite positive gradients, frozen source/head/proxy/teacher/RNG and zero updates/held/quality')


if __name__ == '__main__':
    main()
