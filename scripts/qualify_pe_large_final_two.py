#!/usr/bin/env python3
"""Actual native CPU final-two-block gradients and strict whole/suffix parity."""

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_final_two as native


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not torch.cuda.is_available()
    torch.set_num_threads(8)
    base, root = native.base, Path(__file__).resolve().parent
    pair = base.pair
    torch.manual_seed(pair.SEED)
    control, frozen, prior = base.control(root)
    old_cpu = Path("/home/riomus/runs/sfora-large-native-pool-cpu-v3/preflight.json")
    assert pair.sha(old_cpu) == "8def5e7b45595d6d6831f691fc34d46afb255fa834efc38bff7b5d478c7d0754"
    old = json.loads(old_cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in old["code"].items())
    for path, digest in old["environment"]["native_files"].items():
        assert pair.sha(Path(path)) == digest
    code = {**old["code"], **{n: pair.sha(root / n) for n in ("pe_large_final_two.py", "qualify_pe_large_final_two.py", "check_pe_large_final_two.py", "pe_immutable_amp.py")}}
    model, processor = pair.smoke.load_arm(control, "large")
    inventory = native.freeze(model)
    assert len(model.state_dict()) == 400 and sum(p.numel() for p in model.parameters()) == 315956224
    assert len(inventory["trainable"]) == 32 and len(inventory["frozen"]) == 368
    original = pair.smoke.digest(base.whole_state(model))
    assert original == old["whole_original_source_sha256"]
    foreign = pair.smoke.digest(native.frozen_state(model))
    assert json.loads(json.dumps(base.environment(model, processor))) == old["environment"]
    environment = native.environment(model, processor)
    runtime = base.runtime_identity(model)
    assert pair.sha(base.INIT / "initializers.npz") == old["initializers_sha256"]
    head = nn.Linear(1024, 128)
    with np.load(base.INIT / "initializers.npz", allow_pickle=False) as init:
        head.load_state_dict({k: torch.from_numpy(init["large.head." + k]) for k in ("weight", "bias")})
        classifier = nn.Parameter(torch.from_numpy(init["large.classifier"].copy()))
        bank = torch.from_numpy(init["large.bank"].copy())
    images, rgb = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1)
    first = pair.pixels(processor, images, "large")
    assert rgb == prior["rgb_sha256"][0]
    assert pair.smoke.digest({"pixels": first}) == frozen["initializers"]["large"]["first_pixels_sha256"]
    del first, images
    images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = pair.pixels(processor, images, "large")
    with torch.no_grad():
        baseline = model(pixel_values=pixels).pooler_output
    seen = {}

    def block_input(module, inputs):
        assert module is model.encoder.layers[-2] and len(inputs) == 2 and inputs[1] is None
        base.assert_frozen_tokens((inputs[0],))
        seen["block_input"] = list(inputs[0].shape)

    def second_block_input(module, inputs):
        assert module is model.encoder.layers[-1] and len(inputs) == 2 and inputs[1] is None
        assert inputs[0].requires_grad and inputs[0].grad_fn is not None and inputs[0].dtype == torch.float32
        seen["second_block_input"] = list(inputs[0].shape)

    def pool_input(module, inputs):
        assert module is model.head and len(inputs) == 1
        assert inputs[0].requires_grad and inputs[0].grad_fn is not None and inputs[0].dtype == torch.float32
        seen["pool_input"] = list(inputs[0].shape)

    hooks = [model.encoder.layers[-2].register_forward_pre_hook(block_input), model.encoder.layers[-1].register_forward_pre_hook(second_block_input), model.head.register_forward_pre_hook(pool_input)]
    try:
        source = model(pixel_values=pixels).pooler_output
    finally:
        for hook in hooks:
            hook.remove()
    assert torch.equal(source, baseline) and seen == {"block_input": [2, 256, 1024], "second_block_input": [2, 256, 1024], "pool_input": [2, 256, 1024]}
    raw = pair.smoke.compact_head_features(source, head)
    index = torch.tensor((0, 1))
    positives = pair.smoke.member_bank_positive_ordinals(np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True)
    loss = pair.smoke.sharded_mask_arcface_loss(raw, classifier, torch.tensor(frozen["target"])[index], torch.arange(128).unsqueeze(0), margin=0.3, scale=64)
    loss += 8 * pair.smoke.member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False)
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        assert (parameter.grad is None) == (not parameter.requires_grad), name
        if parameter.requires_grad:
            assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0, name
            gradients[name] = float(parameter.grad.norm())
    assert set(gradients) == set(inventory["trainable"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in [*head.parameters(), classifier])
    active = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": active, "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-4}, {"params": [classifier], "lr": 1e-4}], weight_decay=0.05)
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert not optimizer.state and len(members) == len({id(p) for p in members}) == 35
    assert {id(p) for p in members} == {id(p) for p in active + list(head.parameters()) + [classifier]}
    assert pair.smoke.digest(base.whole_state(model)) == original
    model.zero_grad(set_to_none=True)
    parameters = (model.encoder.layers[-2].self_attn.q_proj.bias, model.encoder.layers[-1].mlp.fc2.bias)
    saved_parameters = [p.detach().clone() for p in parameters]
    negatives = []
    try:
        with torch.no_grad():
            parameters[0][0].add_(0.01)
            attention_changed = model(pixel_values=pixels).pooler_output
            assert not torch.equal(attention_changed, baseline)
            parameters[1][0].add_(0.01)
            changed = model(pixel_values=pixels).pooler_output
            assert not torch.equal(changed, attention_changed)
        with TemporaryDirectory(dir=root, prefix="final-two-block-roundtrip-") as tmp:
            path = Path(tmp) / "native.pt"
            torch.save({"vision": model.state_dict(), "head": head.state_dict()}, path)
            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            loaded = type(model)(copy.deepcopy(model.config)).float().eval()
            loaded.load_state_dict(state["vision"], strict=True)
            native.freeze(loaded)
            loaded_head = nn.Linear(1024, 128).eval()
            loaded_head.load_state_dict(state["head"], strict=True)
            assert pair.smoke.digest(native.frozen_state(loaded)) == foreign
            with torch.no_grad():
                whole = loaded(pixel_values=pixels).pooler_output
            assert torch.equal(whole, changed)
            with native.verify_export(loaded, model) as (encode, observations):
                a, b = encode(pixels)
                assert torch.equal(a, whole) and torch.equal(a, b)
                va = F.normalize(pair.smoke.compact_head_features(a, loaded_head), dim=1)
                vb = F.normalize(pair.smoke.compact_head_features(b, head), dim=1)
                assert torch.isfinite(va).all() and torch.isfinite(vb).all() and torch.equal(va, vb)
            assert len(observations) == 1 and all(observations[0][k] is False for k in ("loaded_amp_inside", "captured_block_amp_inside", "between_encoder_amp_enabled", "live_amp_inside"))
            for name, parameter in (("final_attention", parameters[0]), ("final_mlp", parameters[1]), ("frozen_pool", model.head.probe), ("frozen_prefix", model.encoder.layers[0].mlp.fc2.bias)):
                before = parameter.detach().clone()
                try:
                    with native.verify_export(loaded, model) as (encode, _):
                        with torch.no_grad():
                            parameter.flatten()[0].add_(0.01)
                        encode(pixels)
                except AssertionError as error:
                    assert str(error) == "native pair changed"
                    negatives.append({"name": name, "rejected_by": str(error)})
                else:
                    raise AssertionError(f"mutated {name} accepted")
                finally:
                    with torch.no_grad():
                        parameter.copy_(before)
                assert not any(m._forward_hooks or m._forward_pre_hooks for m in (*loaded.modules(), *model.modules()))
            for name, parameter, expected in (("data_first_block", parameters[0], "native first adapted block differs"), ("data_final_block", parameters[1], "native final block differs"), ("data_pool", model.head.probe, "native suffix differs"), ("data_prefix_exit", model.encoder.layers[0].mlp.fc2.bias, "native pair state changed")):
                before, version = parameter.detach().clone(), parameter._version
                try:
                    with native.verify_export(loaded, model) as (encode, _):
                        parameter.data.flatten()[0].add_(0.01)
                        assert parameter._version == version, "negative failed to bypass version guard"
                        encode(pixels)
                except AssertionError as error:
                    assert str(error) == expected, (name, str(error))
                    negatives.append({"name": name, "rejected_by": str(error)})
                else:
                    raise AssertionError(f"data mutation {name} accepted")
                finally:
                    with torch.no_grad():
                        parameter.copy_(before)
                assert not any(m._forward_hooks or m._forward_pre_hooks for m in (*loaded.modules(), *model.modules()))
            assert base.runtime_identity(model) == base.runtime_identity(loaded) == runtime
    finally:
        with torch.no_grad():
            for p, original_parameter in zip(parameters, saved_parameters, strict=True):
                p.copy_(original_parameter)
    assert pair.smoke.digest(base.whole_state(model)) == original
    assert pair.smoke.digest(native.frozen_state(model)) == foreign
    assert native.environment(model, processor) == environment
    with torch.no_grad():
        assert torch.equal(model(pixel_values=pixels).pooler_output, baseline)
    assert all(pair.sha(root / n) == h for n, h in code.items())
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / "preflight.json", {"code": code, "original_native_cpu_sha256": pair.sha(old_cpu), "environment": environment, "runtime_identity": runtime, "inventory": inventory, "whole_original_source_sha256": original, "frozen_complement_sha256": foreign, "actual_graph": seen, "actual_native_gradient_norms": gradients, "actual_head_proxy_gradient_norms": [float(p.grad.norm()) for p in [*head.parameters(), classifier]], "actual_loss": float(loss.detach()), "native_trainable_tensors": len(gradients), "native_trainable_parameters": sum(p.numel() for p in active), "optimizer_tensors": len(members), "strict400_key_updated_reload_exact": True, "whole_native_calibration_exact": True, "independent_final_suffix_source_head_exact": True, "scope_observations": observations, "negatives_rejected": negatives, "observer_cleanup_and_original_state_output_restored": True, "optimizer_updates": 0, "held_images": 0, "quality_read": False, "cuda": False})
    print("PASS actual native final-two-block-only gradients/frozen pool, strict whole calibration + independent suffix/head parity, mutation rejection and restoration; no updates/held/quality")


if __name__ == "__main__":
    main()
