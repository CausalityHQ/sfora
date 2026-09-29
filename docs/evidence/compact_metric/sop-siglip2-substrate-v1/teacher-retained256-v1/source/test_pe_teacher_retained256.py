#!/usr/bin/env python3
"""One focused expansion check; run Torch only on DGX."""
import subprocess
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from train_pe_teacher_retained256 import expand, terms, driver

torch.set_num_threads(8)
torch.manual_seed(179032)
vision, head = nn.Linear(1024, 1024), nn.Linear(1024, 128)
classifier = nn.Parameter(torch.randn(3, 128))
state = {'model': vision, 'head': head, 'classifier': classifier,
         'bank': F.normalize(torch.randn(4, 128), dim=1),
         'target': torch.tensor([0, 1, 0, 2]),
         'positive': torch.tensor([[2], [-1], [0], [-1]])}
state['params'] = driver.coverage.parameters(vision, head, classifier)
state['optimizer'] = torch.optim.AdamW([
    {'params': vision.parameters(), 'lr': 1e-5},
    {'params': head.parameters(), 'lr': 1e-4},
    {'params': [classifier], 'lr': 1e-4}], weight_decay=.05)
old = {k: v.detach().clone() for k, v in head.state_dict().items()}
old_bank, old_proxy = state['bank'].clone(), classifier.detach().clone()
rng = torch.random.get_rng_state().clone()
tail_sha = expand(state)
assert len(tail_sha) == 64 and torch.equal(rng, torch.random.get_rng_state())
assert all(torch.equal(v[:128], old[k]) and v[128:].count_nonzero() == 0
           for k, v in state['head'].state_dict().items())
assert torch.equal(state['bank'][:, :128], old_bank) and state['bank'][:, 128:].count_nonzero() == 0
assert torch.equal(state['classifier'][:, :128], old_proxy)
assert torch.allclose(state['classifier'][:, 128:].norm(dim=1), .001 * old_proxy.norm(dim=1), rtol=1e-6, atol=0)
assert [id(p) for g in state['optimizer'].param_groups for p in g['params']] == [id(p) for _, p in state['params']]
source = torch.randn(2, 1024)
index = torch.tensor([0, 1])
ce, rank, raw = terms(source, state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index, False)
expected = F.linear(F.normalize(source, dim=1), old['weight'], old['bias'])
assert torch.equal(raw[:, :128], expected) and raw[:, 128:].count_nonzero() == 0
ce.backward()
assert torch.isfinite(state['head'].weight.grad).all() and state['head'].weight.grad[128:].norm() > 0
assert torch.isfinite(state['classifier'].grad).all() and state['classifier'].grad[:, 128:].norm() > 0
state['optimizer'].zero_grad(set_to_none=True)
with torch.no_grad():
    state['classifier'][:, 128:].zero_()
dead, _, _ = terms(source, state['head'], state['classifier'], state['bank'], state['target'], state['positive'], index, False)
dead.backward()
assert state['head'].weight.grad[128:].count_nonzero() == 0
entry = str(Path(__file__).with_name('train_pe_teacher_retained256.py'))
for flags in (['-O'], ['-OO']):
    result = subprocess.run([sys.executable, *flags, entry, '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr
print('PASS teacher prefixes/RNG/registration/full-width CE/new-head gradients/dead-tail negative/optimized rejection')
