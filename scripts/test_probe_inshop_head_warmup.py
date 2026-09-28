import torch
from probe_inshop_head_warmup import warm_head


def test_head_warmup_detaches_source_and_updates_downstream_parameters():
    torch.manual_seed(3)
    source = torch.randn(6, 6, requires_grad=True)
    head = torch.nn.Linear(6, 2)
    classifier = torch.nn.Parameter(torch.randn(3, 2))
    target = torch.tensor([0, 0, 1, 1, 2, 2])
    before = head.weight.detach().clone()
    losses = warm_head(source, head, classifier, target, [(0, 1, 2, 3, 4, 5)] * 100)
    assert len(losses) == 100 and all(torch.isfinite(torch.tensor(losses)))
    assert source.grad is None and not torch.equal(before, head.weight)
