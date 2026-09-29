#!/usr/bin/env python3
"""Actual F5 native CPU admission for the fixed lower-stack adapter pilot."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assert not args.output.exists()
    assert sha(root / "large-lower-execution.json") == args.execution_sha256
    current = json.loads((root / "large-lower-execution.json").read_text())
    old = json.loads((root / "large-coverage-execution.json").read_text())
    assert set(current) == set(old) | {"large_lower_adapter.py", "qualify_large_lower_adapter_cpu.py", "check_large_lower_adapter.py"}
    assert all(current[k] == v for k, v in old.items())
    assert all(sha(root / k) == v for k, v in current.items())

    import torch
    from torch import nn
    import pe_large_coverage as coverage
    from large_lower_adapter import install, merge

    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(coverage.pair.SEED)
    control, frozen, cpu, _, _ = coverage.startup(root, sha(root / "large-coverage-execution.json"))
    pools = coverage.inputs(control, frozen)
    fit = pools["half"]
    model, head, processor, inventory = coverage.load_native(control, cpu)
    assert len(model.state_dict()) == 400
    before = coverage.pair.smoke.digest(coverage.trained.base.whole_state(model))
    frozen_before = coverage.frozen_digest(model, inventory)
    images, rgb = coverage.pair.augmented_images(control.dataset_root, fit["rows"], (0, 1), None)
    pixels = coverage.pair.pixels(processor, images, "large")
    pixel_sha = coverage.pair.smoke.digest({"pixels": pixels})
    assert pixel_sha == cpu["first_two_fit_pixels_sha256"]
    with torch.no_grad():
        baseline = model(pixel_values=pixels).pooler_output.detach().clone()
    rng = torch.random.get_rng_state().clone()
    sites = install(model, rank=8, seed=179032)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert len(sites) == 24
    assert all(module.parametrizations.weight.original.shape == (1024, 1024) for module, _ in sites)
    with torch.no_grad():
        assert torch.equal(model(pixel_values=pixels).pooler_output, baseline)

    classifier = nn.Parameter(fit["classifier"].clone())
    bank = fit["bank"].clone()
    factors = [p for module, _ in sites for p in module.parametrizations.weight[0].parameters()]
    factor_ids = {id(p) for p in factors}
    dense = [p for p in model.parameters() if p.requires_grad and id(p) not in factor_ids]
    members = dense + factors + list(head.parameters()) + [classifier]
    assert len(factors) == 48 and len({id(p) for p in members}) == len(members)
    assert {id(p) for p in members} == {id(p) for _, p in coverage.parameters(model, head, classifier)}
    optimizer = torch.optim.AdamW([{"params": dense, "lr": 1e-5},
                                   {"params": factors, "lr": 1e-4},
                                   {"params": list(head.parameters()) + [classifier], "lr": 1e-4}], weight_decay=.05)
    assert [id(p) for group in optimizer.param_groups for p in group["params"]] == [id(p) for p in members]
    positive = coverage.pair.smoke.member_bank_positive_ordinals(fit["target"].numpy(), allow_singletons=True)
    index = torch.tensor((0, 1))
    pooled = model(pixel_values=pixels).pooler_output
    ce, rank, _ = coverage.terms(pooled, head, classifier, bank, fit["target"], positive, index, True)
    loss = ce + 8 * rank
    assert torch.isfinite(loss)
    loss.backward()
    factor_b_norms = [float(module.parametrizations.weight[0].B.grad.norm()) for module, _ in sites]
    assert all(v > 0 for v in factor_b_norms)
    assert all(not module.parametrizations.weight[0].A.grad.any() for module, _ in sites)
    assert all(module.parametrizations.weight.original.grad is None for module, _ in sites)
    upper = [sum(float(p.grad.norm()) for p in model.encoder.layers[i].parameters() if p.grad is not None) for i in range(12, 24)]
    assert all(v > 0 for v in upper)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert classifier.grad is not None and classifier.grad.norm() > 0
    assert all(p.grad is None for p in model.embeddings.parameters())
    merge(sites)
    assert len(model.state_dict()) == 400
    assert coverage.pair.smoke.digest(coverage.trained.base.whole_state(model)) == before
    assert coverage.frozen_digest(model, inventory) == frozen_before
    assert torch.equal(rng, torch.random.get_rng_state())
    result = {"pass": True, "execution_sha256": args.execution_sha256, "source_whole_sha256": before,
              "frozen_prefix_sha256": frozen_before, "pixels_sha256": pixel_sha, "rgb_sha256": rgb,
              "site_count": len(sites), "factor_b_gradient_norms": factor_b_norms,
              "upper_gradient_norms": upper, "optimizer_member_count": len(members),
              "loss": float(loss.detach()), "quality_read": False,
              "gpu_training_qualified": False}
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("PASS actual F5 native CPU source, lower adapter identity/gradients/merge; no GPU or quality")


if __name__ == "__main__":
    main()
