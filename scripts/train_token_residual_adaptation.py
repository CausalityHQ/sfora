#!/usr/bin/env python3
"""W-aware CPU startup, one discarded candidate17, or one fresh matched TRAIN100."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")
import time
STARTED = time.perf_counter()
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

METHOD = "token-quadrant-residual-v1"
ADDED = {"train_token_residual_adaptation.py", "test_token_residual_adaptation.py"}
CPU_EXECUTION = "96377b69e3c6fc1730bdffa674884977150ec4a4c35febab154aa42151ead408"
CPU_PROOF = "67c7c2452d211f4755770c703147d970a588cddace63d6e1550ed2728ecec06b"
CPU_LOG = "cff9c3de737c1b486501abda47c591486f0d449f2268441611df3955029cb6aa"
SOURCE = "cc58377f0e9aa90be529bf9a3eca1746a2dc467f765dd9681a3b9b690e324566"
INITIAL = "77c26114a74472682f7f511d732dfac2679d99bfe120ee52d3f78c032a9d9867"
SIGLIP = "274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31"
RESUME_KEYS = frozenset(("identity", "vision", "buffers", "head", "residual", "classifier",
                         "bank", "optimizer", "scaler", "cpu_rng", "cuda_rng"))
# Historical control receipts bind inputs only; every new endpoint measures its own cost/state.
INPUT_AUTHORITIES = {
    179032: "1ec76986fb13916165d510935a9e7fb4508cd243a8cce0b7ad8211a75fb04331",
    179041: "739337d9b6ac6b653fc8c42e00f60ebb92cbd6061ba3393f25a731b6fd46b70f"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_authority(path, expected, log=False):
    assert path and expected and sha(path) == expected, "authority digest differs"
    return path.read_text() if log else json.loads(path.read_text())


def resources(value, cap, gpu):
    assert value["unit_invocation_id"] and Path(value["unit_cgroup"]).name.endswith(".service")
    assert value["unit_memory_max_bytes"] == 8 * 1024**3 and value["unit_memory_swap_max_bytes"] == 0
    assert math.isfinite(value["total_seconds"]) and 0 < value["total_seconds"] < cap
    assert 0 < value["host_max_rss_kib"] <= 8 * 1024 * 1024 and value["host_swap_kib"] == 0
    peak = value["peak_cuda_allocated_bytes"]
    assert 0 < peak < 10_000_000_000 if gpu else peak == 0


def validate_log(text, value, cap):
    assert all(s in text for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    unit = Path(value["unit_cgroup"]).name
    assert f"Running as unit: {unit}; invocation ID: {value['unit_invocation_id']}" in text
    matches = re.findall(r"Service runtime: (?:(\d+)min )?([\d.]+)s", text)
    assert len(matches) == 1 and 0 < int(matches[0][0] or 0) * 60 + float(matches[0][1]) < cap
    rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", text)
    assert rss and 0 < value["host_max_rss_kib"] <= int(rss[0]) <= 8 * 1024 * 1024
    assert "flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock" in text
    assert "flock -n /home/riomus/.sfora-siglip2-gpu.lock" in text


def validate_cpu(value, execution, code, previous):
    assert value["schema"] == "token-residual-cpu-v1" and value["pass"] is True
    assert execution == value["execution_sha256"] == CPU_EXECUTION and value["code"] == previous
    assert len(previous) == 118 and len(code) == 120 and set(code) - set(previous) == ADDED
    assert all(code[n] == h for n, h in previous.items())
    assert value["boundary"] == 12 and value["optimizer_members"] == {"control":208,"candidate":209}
    assert value["source_checkpoint_sha256"] == SOURCE and value["initial_state_sha256"] == INITIAL
    assert value["module_sha256"] == SIGLIP and value["token_shape"] == [2,256,1024]
    assert value["token_dtype"] == "torch.float32"
    assert all(value[k] is True for k in ("same_call_tokens_and_layout_exact", "independent_quadrants_exact",
        "zero_raw_and_packed_reload_exact", "nonzero_W_gradient", "nonzero_W_live_token_witness",
        "shared_head_zero_gradient_exact", "RNG_preserved", "source_and_W_negatives_rejected"))
    assert value["optimizer_updates"] == 0 and value["quality_read"] is False
    resources(value,120,False)


def closure(root, execution, cpu_execution):
    import qualify_token_residual_cpu as cpu
    code = read_authority(root / "token-residual-train-execution.json", execution)
    previous = cpu.closure(root, cpu_execution)
    assert len(code) == 120 and set(code) - set(previous) == ADDED
    assert all(code[n] == h for n,h in previous.items())
    assert all(sha(root / n) == h for n,h in code.items()), "execution source changed"
    return code, previous


def validate_startup(value, args, cpu):
    assert value["schema"] == "token-residual-train-v1" and value["intervention"] == METHOD
    assert value["phase"] == "startup" and value["pass"] is True and value["boundary"] == 12
    assert value["execution_sha256"] == args.execution_sha256
    assert value["cpu_authority_sha256"] == args.cpu_sha256 and value["cpu_log_sha256"] == args.cpu_log_sha256
    assert value["cpu_execution_sha256"] == args.cpu_execution_sha256
    assert value["source_checkpoint_sha256"] == cpu["source_checkpoint_sha256"] == SOURCE
    assert value["initial_state_sha256"] == cpu["initial_state_sha256"] == INITIAL
    assert value["optimizer_members"] == {"control":208,"candidate":209}
    assert value["optimizer_updates"] == 0 and value["quality_read"] is False
    assert all(value[k] is True for k in ("both_arms_complete_state_exact", "resume_negatives_rejected",
        "optimizer_roles_options_exact", "changed_module_rejected", "RNG_preserved"))
    assert set(value["schedules"]) == {"179032","179041"}
    for seed in INPUT_AUTHORITIES:
        assert value["schedules"][str(seed)]["input_authority_sha256"] == INPUT_AUTHORITIES[seed]
    assert set(value["resume_identities"]) == {"control","candidate"}
    for arm,base in value["resume_identities"].items():
        assert base["schema"] == "token-residual-complete-resume-v1" and base["intervention"] == METHOD
        assert base["residual_arm"] == arm and base["residual_role"] == (arm == "candidate")
        assert base["precision"] == "cpu_float32" and base["boundary"] == 12 and base["width"] == 128
        assert base["module_sha256"] == value["code"]["token_residual_readout.py"]
        assert base["residual_shape"] == [128,4096] and base["residual_dtype"] == "torch.float32"
        assert base["grid"] == [16,16] and base["quadrant_order"] == ["NW","NE","SW","SE"]
        assert base["pooling"] == "fixed_8x8_mean_fp32" and base["unit_normalization"] == "concatenated_quadrants_l2_before_W"
        assert base["execution_sha256"] == args.execution_sha256 and base["source_checkpoint_sha256"] == SOURCE
        assert base["initial_state_sha256"] == INITIAL and base["cpu_authority_sha256"] == args.cpu_sha256
        assert base["cpu_log_sha256"] == args.cpu_log_sha256 and base["cpu_execution_sha256"] == args.cpu_execution_sha256
        assert len(base["parameter_names"]) == 208 + (arm == "candidate")
        assert len(base["optimizer_groups"]) == 3 + (arm == "candidate")
    resources(value,120,False)


def validate_mechanics(value, args, cpu, startup):
    validate_startup({**startup},args,cpu)
    assert value["schema"] == "token-residual-train-v1" and value["pass"] is True and value["intervention"] == METHOD
    assert value["phase"] == "mechanics" and value["arm"] == "candidate" and value["seed"] == 179032
    for key, expected in (("execution_sha256",args.execution_sha256), ("cpu_authority_sha256",args.cpu_sha256),
        ("cpu_execution_sha256",args.cpu_execution_sha256), ("cpu_log_sha256",args.cpu_log_sha256),
        ("startup_authority_sha256",args.startup_sha256), ("startup_log_sha256",args.startup_log_sha256),
        ("initial_state_sha256",INITIAL), ("source_checkpoint_sha256",SOURCE)):
        assert value[key] == expected
    assert value["updates"] == value["completed_step"] == 17 and value["boundary"] == 12
    assert value["checkpoint_sha256"] is None and value["quality_read"] is False
    assert all(value[k] is True for k in ("training_state_discarded", "native_17_equals_serialized8_plus9_exact",
        "strict400_head_W_whole_packed_exact", "frozen_named_state_buffers_rng_preserved"))
    projection = value["chunk100_admission_seconds"]
    assert math.isfinite(projection) and 0 < projection <= 269
    expected = startup["schedules"]["179032"]
    assert value["schedule_sha256"] == expected["schedule_sha256"]
    base = value["resume_identity"]
    assert base["residual_arm"] == "candidate" and base["intervention"] == METHOD
    assert base["schedule_sha256"] == value["schedule_sha256"]
    assert base["class_sequence_sha256"] == expected["class_sequence_sha256"]
    assert base["module_sha256"] == value["code"]["token_residual_readout.py"]
    assert value["code"] == startup["code"] and len(base["parameter_names"]) == 209
    assert len(value["steps"]) == 17
    for step,row in enumerate(value["steps"],1):
        assert row["step"] == row["optimizer_counter"] == step and row["augmentation_step"] == 1000+step
        assert len(row["image_ids"]) == len(set(row["image_ids"])) == 64
        assert all(type(i) is int and 0 <= i < 13283 for i in row["image_ids"])
        assert math.isfinite(row["W_gradient_norm"]) and row["W_gradient_norm"] > 0
        assert row["token_shape"] == [16,256,1024] and row["token_dtype"] in ("torch.float16","torch.float32")
        assert row["pooled_shape"] == [16,1024] and row["pooled_dtype"] in ("torch.float16","torch.float32")
        assert row["encoder_calls"] == 4 and row["zero_W_raw_packed_same_pass_exact"] is (step == 1)
    resources(value,120,True)


def identity(state, args, initial, schedule_sha, class_sha, code):
    base = native.identity(state, SimpleNamespace(**vars(args),boundary=12),initial,schedule_sha)
    base.update(schema="token-residual-complete-resume-v1", intervention=METHOD,
        residual_arm=args.arm, residual_role=args.arm == "candidate", module_sha256=code["token_residual_readout.py"],
        grid=[16,16], quadrant_order=["NW","NE","SW","SE"], pooling="fixed_8x8_mean_fp32",
        unit_normalization="concatenated_quadrants_l2_before_W", residual_shape=[128,4096],
        residual_dtype="torch.float32", class_sequence_sha256=class_sha,
        input_authorities={str(k):v for k,v in INPUT_AUTHORITIES.items()},
        startup_authority_sha256=args.startup_sha256, startup_log_sha256=args.startup_log_sha256,
        initial_rng_sha256=old.fingerprint({"cpu":torch.random.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if state["scaler"] else []}),
        precision="native_float32_fp16_autocast" if state["scaler"] else "cpu_float32",
        numerical_flags=old.coverage.teacher.qualified.numerical_flags())
    base.update(head_roles=[(n,p.requires_grad) for n,p in state["head"].named_parameters()],
                classifier_role=state["classifier"].requires_grad)
    return base


def verify(state, base):
    assert base["residual_arm"] in ("control","candidate") and base["residual_role"] == (base["residual_arm"] == "candidate")
    assert base["module_sha256"] == sha(Path(residual.__file__)) and base["intervention"] == residual.METHOD == METHOD
    common = {**base, "parameter_names":base["parameter_names"][:208], "optimizer_groups":base["optimizer_groups"][:3]}
    cpu_module.verify(state,common,base["residual_arm"])
    residual._validate_members(state,base["residual_arm"])
    assert [(n,p.requires_grad) for n,p in state["head"].named_parameters()] == base["head_roles"]
    assert state["classifier"].requires_grad == base["classifier_role"] is True
    assert all(p.requires_grad for p in state["head"].parameters())
    assert [n for n,_ in state["params"]] == base["parameter_names"]
    names = {id(p):n for n,p in state["params"]}
    assert [{"parameter_names":[names[id(p)] for p in g["params"]],
             "options":{k:v for k,v in g.items() if k != "params"}}
            for g in state["optimizer"].param_groups] == base["optimizer_groups"]


def payload(state, base):
    return {"identity":{**base,"global_step":state["counter"]}, "vision":state["model"].state_dict(),
        "buffers":dict(state["model"].named_buffers()), "head":state["head"].state_dict(),
        "residual":state["residual"].detach(), "classifier":state["classifier"].detach(), "bank":state["bank"],
        "optimizer":state["optimizer"].state_dict(), "scaler":state["scaler"].state_dict() if state["scaler"] else None,
        "cpu_rng":torch.random.get_rng_state(), "cuda_rng":torch.cuda.get_rng_state_all() if state["scaler"] else []}


def tensor_layout(value, template):
    assert isinstance(value,torch.Tensor) and value.shape == template.shape and value.dtype == template.dtype, "tensor shape/dtype differs"
    assert bool(torch.isfinite(value).all()), "nonfinite resume tensor"


def validate_resume(saved, base, step, groups, template, members):
    """Reject every layout/member/authority mismatch before copying any saved tensor."""
    assert set(saved) == RESUME_KEYS, "complete W resume required"
    assert type(step) is int and 0 <= step <= 100
    assert saved["identity"] == {**base,"global_step":step}, "foreign resume identity/counter"
    residual.validate_weight(saved["residual"],base["residual_arm"])
    assert len(saved["vision"]) == 400
    optimizer = saved["optimizer"]
    assert set(optimizer) == {"param_groups","state"} and optimizer["param_groups"] == groups, "optimizer order/options differ"
    ids = [i for group in groups for i in group["params"]]
    assert len(ids) == len(set(ids)) == len(base["parameter_names"]) == len(members)
    assert set(optimizer["state"]) == (set(ids) if step else set()), "optimizer member state differs"
    for i,(_,parameter) in zip(ids,members,strict=True):
        if not step: continue
        moment = optimizer["state"][i]
        assert set(moment) == {"step","exp_avg","exp_avg_sq"} and float(moment["step"]) == step, "optimizer step/moments differ"
        assert isinstance(moment["step"],torch.Tensor) and moment["step"].numel() == 1
        tensor_layout(moment["exp_avg"],parameter); tensor_layout(moment["exp_avg_sq"],parameter)
    cpu = base["precision"] == "cpu_float32"
    if cpu:
        assert saved["scaler"] is None and saved["cuda_rng"] == []
    else:
        assert set(saved["scaler"]) == set(template["scaler"]) and len(saved["cuda_rng"]) == 1
        assert math.isfinite(saved["scaler"]["scale"]) and saved["scaler"]["scale"] >= 128
        assert all(math.isfinite(float(v)) for v in saved["scaler"].values())
        tensor_layout(saved["cuda_rng"][0],template["cuda_rng"][0])
    tensor_layout(saved["cpu_rng"],template["cpu_rng"])
    for key in ("vision","buffers","head"):
        assert saved[key].keys() == template[key].keys(), "strict state keys differ"
        for name,value in saved[key].items(): tensor_layout(value,template[key][name])
    for key in ("residual","classifier","bank"): tensor_layout(saved[key],template[key])
    assert old.fingerprint(saved["buffers"]) == base["buffers_sha256"]
    assert all(torch.equal(saved["buffers"][n],saved["vision"][n]) for n in saved["buffers"] if n in saved["vision"])
    named_state = {**saved["vision"],**saved["buffers"]}
    assert old.fingerprint({n:named_state[n] for n in base["frozen_names"]}) == base["frozen_sha256"]


def save(state, base, path):
    state["optimizer"].zero_grad(set_to_none=True)
    verify(state,base)
    value = payload(state,base)
    validate_resume(value,base,state["counter"],value["optimizer"]["param_groups"],value,state["params"])
    fingerprint = old.fingerprint(value)
    native.atomic_write(path,lambda stream:torch.save(value,stream))
    disk = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    assert old.fingerprint(disk) == fingerprint, "serialized full W state differs"
    return sha(path),fingerprint


def restore(state, base, path, expected_sha, fingerprint, step):
    verify(state,base)
    assert sha(path) == expected_sha
    saved = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    template = payload(state,base)
    validate_resume(saved,base,step,template["optimizer"]["param_groups"],template,state["params"])
    assert old.fingerprint(saved) == fingerprint
    state["model"].load_state_dict(saved["vision"],strict=True)
    state["head"].load_state_dict(saved["head"],strict=True)
    with torch.no_grad():
        for key in ("residual","classifier","bank"): state[key].copy_(saved[key])
        for n,value in state["model"].named_buffers(): value.copy_(saved["buffers"][n])
    state["optimizer"].load_state_dict(saved["optimizer"])
    if state["scaler"]:
        state["scaler"].load_state_dict(saved["scaler"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
    torch.random.set_rng_state(saved["cpu_rng"])
    state["counter"] = step
    verify(state,base)
    assert old.fingerprint(payload(state,base)) == fingerprint, "restored full W state differs"


def startup_checks(state, base, directory):
    """CPU real-tensor state validation; synthetic moments never enter the live optimizer."""
    verify(state,base)
    path = directory / (base["residual_arm"] + ".pt")
    saved_sha,fingerprint = save(state,base,path)
    restore(state,base,path,saved_sha,fingerprint,0)
    value = payload(state,base)
    groups = value["optimizer"]["param_groups"]
    def check(v,step=0): validate_resume(v,base,step,groups,value,state["params"])
    check(value)
    for key in RESUME_KEYS:
        bad = {**value}; del bad[key]
        qualified.rejects(lambda:check(bad))
    for key in ("residual_arm","residual_role","module_sha256","grid","unit_normalization",
                "source_checkpoint_sha256","execution_sha256","cpu_log_sha256","schedule_sha256"):
        bad = {**value,"identity":{**value["identity"],key:"foreign"}}
        qualified.rejects(lambda:check(bad))
    for weight in (value["residual"][:,:4095],value["residual"].double(),value["residual"]+float("nan")):
        qualified.rejects(lambda:check({**value,"residual":weight}))
    for key in ("vision","head","buffers"):
        name = next(iter(value[key]))
        bad = {**value,key:{**value[key],name:torch.zeros(1,dtype=torch.float64)}}
        qualified.rejects(lambda:check(bad))
    if base["residual_arm"] == "control":
        qualified.rejects(lambda:check({**value,"residual":value["residual"]+1}))
    # Exercise all member moment shapes/ordering using existing parameter storage, without updates or CUDA state.
    ids = [i for g in groups for i in g["params"]]
    moments = {i:{"step":torch.tensor(1.),"exp_avg":p.detach(),"exp_avg_sq":p.detach()}
               for i,(_,p) in zip(ids,state["params"],strict=True)}
    synthetic = {**value,"identity":{**base,"global_step":1},"optimizer":{"param_groups":groups,"state":moments}}
    check(synthetic,1)
    for case in ("member","moment","step","shape","dtype","order","options"):
        bad = {**synthetic,"optimizer":{"param_groups":copy.deepcopy(groups),"state":dict(moments)}}
        if case == "member": del bad["optimizer"]["state"][ids[-1]]
        elif case in ("moment","step","shape","dtype"):
            bad["optimizer"]["state"][ids[-1]] = dict(moments[ids[-1]])
            moment = bad["optimizer"]["state"][ids[-1]]
            if case == "moment": del moment["exp_avg"]
            elif case == "step": moment["step"] = torch.tensor(1.5)
            elif case == "shape": moment["exp_avg"] = torch.zeros(1)
            else: moment["exp_avg"] = moment["exp_avg"].double()
        elif case == "order": bad["optimizer"]["param_groups"][0]["params"].reverse()
        else: bad["optimizer"]["param_groups"][-1]["lr"] = 1e-3
        qualified.rejects(lambda:check(bad,1))
    members = state["optimizer"].param_groups[0]["params"]
    member = members.pop(); qualified.rejects(lambda:verify(state,base)); members.append(member)
    members.append(member); qualified.rejects(lambda:verify(state,base)); members.pop()
    state["residual"].requires_grad_(not base["residual_role"])
    qualified.rejects(lambda:verify(state,base)); state["residual"].requires_grad_(base["residual_role"])
    for tensor in (next(iter(qualified.late.frozen_state(state["model"],12).values())),
                   next(iter(dict(state["model"].named_buffers()).values()))):
        before = tensor.detach().clone()
        with torch.no_grad(): tensor.add_(1)
        qualified.rejects(lambda:verify(state,base))
        with torch.no_grad(): tensor.copy_(before)
    verify(state,base)
    assert state["counter"] == 0 and not state["optimizer"].state


def candidate_step(state, pixels, ids, active):
    """Scoped substitution at the existing forward/terms boundary; no hooks or replay."""
    pending = []
    audit = {}
    ratios = []
    calls = 0
    zero_start = state["counter"] == 0
    fp16 = old.previous.training.fp16
    clipping = torch.nn.utils.clip_grad_norm_
    def forward(model,x):
        nonlocal calls
        assert model is state["model"] and not pending
        calls += 1
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            result = model(pixel_values=x)
        tokens = result.last_hidden_state
        assert tokens.requires_grad and tokens.shape == (16,256,1024)
        facts = {"token_shape":list(tokens.shape),"token_dtype":str(tokens.dtype),
                 "pooled_shape":list(result.pooler_output.shape),"pooled_dtype":str(result.pooler_output.dtype)}
        assert not audit or all(audit[k] == v for k,v in facts.items())
        audit.update(facts); pending.append(tokens)
        return result.pooler_output.float()
    def terms(pooled,head,classifier,bank,target,positive,index,rank_active):
        tokens = pending.pop()
        raw = residual.raw_features(pooled,tokens,head,state["residual"])
        ce = pair.smoke.sharded_mask_arcface_loss(raw,classifier,target[index],
            torch.arange(128,device=pooled.device).unsqueeze(0),margin=.3,scale=64)
        rank = native.objective.lane.valid_rank(raw,bank,head,positive[index],index)
        with torch.no_grad():
            global_raw = pair.smoke.compact_head_features(pooled,head)
            if state["counter"] == 0:
                assert torch.equal(raw,global_raw), "same-pass zero-W raw differs"
                old.previous.training.packed_equal(torch.nn.functional.normalize(raw,dim=1),
                    torch.nn.functional.normalize(global_raw,dim=1))
            ratios.append(float((raw-global_raw).norm(dim=1).mean()/global_raw.norm(dim=1).mean()))
        return ce,rank,raw
    def preclip(parameters,*a,**kw):
        assert [id(p) for p in parameters] == [id(p) for _,p in state["params"]]
        weight = state["residual"]
        assert weight.grad is not None and bool(torch.isfinite(weight.grad).all())
        audit["W_gradient_norm"] = float(weight.grad.double().norm())
        assert math.isfinite(audit["W_gradient_norm"]) and audit["W_gradient_norm"] > 0
        return clipping(parameters,*a,**kw)
    try:
        with patch.object(old.previous.training,"fp16",forward), patch.object(old.coverage,"terms",terms), \
             patch.object(torch.nn.utils,"clip_grad_norm_",preclip):
            row = old.step(state,pixels,ids,active,micro=16)
        assert not pending and calls == 4 and old.previous.training.fp16 is fp16
    finally:
        pending.clear()
    row.update(**audit,encoder_calls=calls,zero_W_raw_packed_same_pass_exact=zero_start,
               W_norm=float(state["residual"].detach().norm()),
               residual_global_raw_norm_ratio=statistics.mean(ratios))
    return row


def strict_reload(state, path, pixels, base):
    """Independent strict400/head/W reload and independent normalized quadrant equation."""
    disk = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    state["model"].eval(); state["head"].eval()
    assert bool(torch.count_nonzero(state["residual"])), "updated nonzero W required"
    with torch.random.fork_rng(devices=[0]):
        model = type(state["model"])(copy.deepcopy(state["model"].config)).float().eval()
        head = torch.nn.Linear(1024,128).eval()
        weight = residual.new_weight("cpu",False)
        assert len(disk["vision"]) == 400
        model.load_state_dict(disk["vision"],strict=True); head.load_state_dict(disk["head"],strict=True)
        buffers = dict(model.named_buffers()); assert buffers.keys() == disk["buffers"].keys()
        with torch.no_grad():
            weight.copy_(disk["residual"])
            for n,value in buffers.items(): value.copy_(disk["buffers"][n])
        for key,values in (("vision",model.state_dict()),("head",head.state_dict()),("residual",weight)):
            assert old.fingerprint(values) == old.fingerprint(disk[key])
        assert old.fingerprint(buffers) == base["buffers_sha256"]
        assert old.fingerprint(qualified.late.frozen_state(model,12)) == base["frozen_sha256"]
        del disk; gc.collect()
        model.cuda(); head.cuda(); weight = weight.cuda()
        with torch.no_grad():
            with torch.autocast(device_type="cuda",dtype=torch.float16):
                a = state["model"](pixel_values=pixels); b = model(pixel_values=pixels)
            assert torch.equal(a.pooler_output,b.pooler_output) and torch.equal(a.last_hidden_state,b.last_hidden_state)
            va = residual.raw_features(a.pooler_output.float(),a.last_hidden_state,state["head"],state["residual"])
            grid = b.last_hidden_state.float().reshape(len(pixels),16,16,1024)
            quadrant = torch.cat([grid[:,r:r+8,c:c+8].mean(dim=(1,2)) for r,c in ((0,0),(0,8),(8,0),(8,8))],dim=1)
            vb = pair.smoke.compact_head_features(b.pooler_output.float(),head) + torch.nn.functional.linear(
                torch.nn.functional.normalize(quadrant,dim=1),weight)
            assert torch.equal(va,vb)
            va,vb = (torch.nn.functional.normalize(v,dim=1) for v in (va,vb))
            old.previous.training.packed_equal(va,vb)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--phase",choices=("startup","mechanics","train"),required=True)
    p.add_argument("--arm",choices=("control","candidate"),required=True)
    p.add_argument("--seed",type=int,choices=(179032,179041),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execution-sha256",required=True)
    p.add_argument("--cpu-execution-sha256",required=True)
    for name in ("cpu","startup","mechanics"):
        for suffix in ("proof","log"): p.add_argument(f"--{name}-{suffix}",type=Path,required=name == "cpu")
        p.add_argument(f"--{name}-sha256",required=name == "cpu")
        p.add_argument(f"--{name}-log-sha256",required=name == "cpu")
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    assert args.cpu_execution_sha256 == CPU_EXECUTION and args.cpu_sha256 == CPU_PROOF and args.cpu_log_sha256 == CPU_LOG
    root = Path(__file__).resolve().parent
    code,previous = closure(root,args.execution_sha256,args.cpu_execution_sha256)
    cpu = read_authority(args.cpu_proof,args.cpu_sha256)
    validate_cpu(cpu,args.cpu_execution_sha256,code,previous)
    validate_log(read_authority(args.cpu_log,args.cpu_log_sha256,log=True),cpu,120)
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::",1)[1].strip().lstrip("/")
    assert int((cgroup / "memory.max").read_text()) == 8*1024**3 and int((cgroup / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (cgroup / "cgroup.procs").read_text().split()
    invocation = os.environ["INVOCATION_ID"]
    startup = mechanics = None
    if args.phase == "startup":
        assert not any((args.startup_proof,args.startup_sha256,args.startup_log,args.startup_log_sha256,
                        args.mechanics_proof,args.mechanics_sha256,args.mechanics_log,args.mechanics_log_sha256))
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    else:
        startup = read_authority(args.startup_proof,args.startup_sha256)
        validate_startup(startup,args,cpu)
        assert startup["code"] == code
        validate_log(read_authority(args.startup_log,args.startup_log_sha256,log=True),startup,120)
        if args.phase == "mechanics":
            assert (args.arm,args.seed) == ("candidate",179032)
            assert not any((args.mechanics_proof,args.mechanics_sha256,args.mechanics_log,args.mechanics_log_sha256))
        else:
            mechanics = read_authority(args.mechanics_proof,args.mechanics_sha256)
            validate_mechanics(mechanics,args,cpu,startup)
            validate_log(read_authority(args.mechanics_log,args.mechanics_log_sha256,log=True),mechanics,120)
    inputs = {}
    for seed,digest in INPUT_AUTHORITIES.items():
        path = Path(f"/home/riomus/runs/sfora-late-dense-control-{seed}-v1/receipt.json")
        history = read_authority(path,digest)
        assert history["pass"] and history["phase"] == "train" and history["boundary"] == 12 and history["seed"] == seed
        assert history["execution_sha256"] == cpu_module_original() and history["source_checkpoint_sha256"] == SOURCE
        assert history["initial_state_sha256"] == INITIAL and len(history["steps"]) == 100 and history["quality_read"] is False
        inputs[seed] = history
    global torch,native,qualified,old,pair,residual,cpu_module
    import torch
    import train_late_dense_adaptation as native
    import token_residual_readout as residual
    import qualify_token_residual_cpu as cpu_module
    qualified,old,pair = native.qualification,native.old,native.pair
    pair.executing_authority(root,code)
    selected = qualified.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected,"helpers",lambda r,_:helpers(r,code)):
        control,source,prior,proof,_,_ = qualified.startup(root,cpu_module.ORIGINAL)
    pair.executing_authority(root,code)
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    if args.phase == "startup":
        assert not torch.cuda.is_available()
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
        torch.cuda.reset_peak_memory_stats()
    facts = {"schema":"token-residual-train-v1","pass":True,"intervention":METHOD,"code":code,
        "phase":args.phase,"arm":args.arm,"seed":args.seed,"boundary":12,"execution_sha256":args.execution_sha256,
        "cpu_authority_sha256":args.cpu_sha256,"cpu_log_sha256":args.cpu_log_sha256,
        "cpu_execution_sha256":args.cpu_execution_sha256,"source_checkpoint_sha256":SOURCE,"quality_read":False,
        "claim_eligible":False,"startup_authority_sha256":args.startup_sha256,"startup_log_sha256":args.startup_log_sha256,
        "mechanics_sha256":args.mechanics_sha256,"mechanics_log_sha256":args.mechanics_log_sha256}
    def fresh(arm,device):
        state,initial = qualified.fresh(control,source,proof,12,device)
        assert initial == INITIAL and not state["optimizer"].state and state["counter"] == 0
        assert state["scaler"] is None or state["scaler"].get_scale() == 128
        rng = old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if state["scaler"] else []})
        residual.attach(state,arm)
        assert old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if state["scaler"] else []}) == rng
        return state,initial
    def schedules(state):
        target = state["target"].cpu().tolist()
        result = {}
        for seed in INPUT_AUTHORITIES:
            batches = old.coverage.schedule(target,seed=seed).tolist()
            schedule_sha = pair.smoke.digest({"batches":torch.tensor(batches)})
            assert schedule_sha == inputs[seed]["schedule_sha256"]
            class_sha = hashlib.sha256(json.dumps([[target[i] for i in b] for b in batches],separators=(",",":")).encode()).hexdigest()
            result[str(seed)] = {"schedule_sha256":schedule_sha,"class_sequence_sha256":class_sha,
                "input_authority_sha256":INPUT_AUTHORITIES[seed]}
        return result
    checkpoint_sha = None
    if args.phase == "startup":
        with TemporaryDirectory(prefix="discard-token-startup-") as temporary:
            bases = {}
            for arm in ("control","candidate"):
                state,initial = fresh(arm,"cpu")
                expected = schedules(state)
                option = SimpleNamespace(**{**vars(args),"arm":arm})
                chosen = expected[str(args.seed)]
                base = identity(state,option,initial,chosen["schedule_sha256"],chosen["class_sequence_sha256"],code)
                rng = torch.random.get_rng_state().clone()
                startup_checks(state,base,Path(temporary))
                assert torch.equal(rng,torch.random.get_rng_state())
                bases[arm] = base
                del state; gc.collect()
        original_sha = sha
        with patch(__name__+".sha",lambda p:"changed" if p.resolve() == Path(residual.__file__).resolve() else original_sha(p)):
            qualified.rejects(lambda:closure(root,args.execution_sha256,args.cpu_execution_sha256))
        facts.update(initial_state_sha256=INITIAL,optimizer_members={"control":208,"candidate":209},
            optimizer_updates=0,schedules=expected,resume_identities=bases,both_arms_complete_state_exact=True,
            resume_negatives_rejected=True,optimizer_roles_options_exact=True,changed_module_rejected=True,RNG_preserved=True,
            synthetic_resume_validation_only=True,checkpoint_sha256=None)
    else:
        state,initial = fresh(args.arm,"cuda")
        expected = schedules(state); assert expected == startup["schedules"]
        target = state["target"].cpu().tolist()
        batches = old.coverage.schedule(target,seed=args.seed).tolist()
        chosen = expected[str(args.seed)]
        base = identity(state,args,initial,chosen["schedule_sha256"],chosen["class_sequence_sha256"],code)
        verify(state,base)
        for key in ("parameter_names","optimizer_groups","head_roles","classifier_role","frozen_names",
                    "frozen_sha256","buffers_sha256","target_positive_sha256","module_sha256","residual_role",
                    "grid","quadrant_order","pooling","unit_normalization","residual_shape","residual_dtype"):
            # Receipt JSON converts role tuples to lists; compare canonical JSON values.
            assert json.loads(json.dumps(base[key])) == startup["resume_identities"][args.arm][key]
        rng = old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()})
        if mechanics:
            batches032 = old.coverage.schedule(target,seed=179032).tolist()
            assert [r["image_ids"] for r in mechanics["steps"]] == batches032[:17]
        counts = old.coverage.np.bincount(target)
        def update(s,step):
            counter,augmentation = native.counters(step)
            assert s["counter"] == counter-1
            torch.cuda.synchronize(); tick = time.perf_counter()
            ids = tuple(batches[step-1])
            with patch.object(pair,"SEED",args.seed):
                images,rgb = pair.augmented_images(control.dataset_root,proof["arms"]["half"]["rows"],ids,augmentation)
            pixels = pair.pixels(s["processor"],images,"large")
            assert pixels.shape == (64,3,256,256)
            pixel_sha = pair.smoke.digest({"pixels":pixels})
            history = inputs[args.seed]["steps"][step-1]
            assert history["augmentation_step"] == augmentation and history["rgb_sha256"] == rgb and history["pixels_sha256"] == pixel_sha
            active = all(counts[target[i]] > 1 for i in ids)
            assert active == history["rank_active_before"]
            if args.arm == "candidate": row = candidate_step(s,pixels,ids,active)
            else:
                with patch.object(old.coverage,"terms",native.objective.terms): row = old.step(s,pixels,ids,active,micro=16)
                assert not bool(torch.count_nonzero(s["residual"]))
                row.update(W_norm=0.,residual_global_raw_norm_ratio=0.)
            assert s["counter"] == counter
            row.update(optimizer_counter=counter,augmentation_step=augmentation,image_ids=list(ids),
                rgb_sha256=rgb,pixels_sha256=pixel_sha,rank_active_before=active)
            if mechanics and args.arm == "candidate" and args.seed == 179032 and step <= 17:
                assert native.diagnostic(row) == native.diagnostic(mechanics["steps"][step-1])
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize(); row["seconds"] = time.perf_counter()-tick
            print(json.dumps(row),flush=True)
            return row
        serialization = 0.
        if args.phase == "mechanics":
            with TemporaryDirectory(prefix="discard-token-mechanics-") as temporary:
                path = Path(temporary) / "step8.pt"
                rows = []
                for step in range(1,18):
                    rows.append(update(state,step))
                    if step == 8:
                        tick = time.perf_counter(); saved_sha,saved_fingerprint = save(state,base,path)
                        serialization += time.perf_counter()-tick
                terminal = old.fingerprint(payload(state,base))
                del state; gc.collect(); torch.cuda.empty_cache()
                state,again = fresh(args.arm,"cuda"); assert again == initial
                assert identity(state,args,initial,chosen["schedule_sha256"],chosen["class_sequence_sha256"],code) == base
                restore(state,base,path,saved_sha,saved_fingerprint,8)
                resumed = [update(state,step) for step in range(9,18)]
                assert [native.diagnostic(r) for r in resumed] == [native.diagnostic(r) for r in rows[8:]]
                assert old.fingerprint(payload(state,base)) == terminal
                path17 = Path(temporary) / "step17.pt"
                tick = time.perf_counter(); _,fingerprint = save(state,base,path17)
                serialization += time.perf_counter()-tick
                assert fingerprint == terminal
                del state["optimizer"]; gc.collect(); torch.cuda.empty_cache()
                with patch.object(pair,"SEED",args.seed):
                    images,_ = pair.augmented_images(control.dataset_root,proof["arms"]["half"]["rows"],tuple(batches[0][:2]),1001)
                strict_reload(state,path17,pair.pixels(state["processor"],images,"large").cuda(),base)
            overhead = time.perf_counter()-STARTED-sum(r["seconds"] for r in rows+resumed)
            admission = 100*max(statistics.median(r["seconds"] for r in rows[2:]),statistics.mean(r["seconds"] for r in rows))+max(30.,overhead)
            assert admission <= 269
            facts.update(training_state_discarded=True,native_17_equals_serialized8_plus9_exact=True,
                strict400_head_W_whole_packed_exact=True,resumed_steps=resumed,
                chunk100_admission_seconds=admission,measured_unit_overhead_seconds=overhead)
        else:
            tick = time.perf_counter(); rows = [update(state,step) for step in range(1,101)]
            wall = time.perf_counter()-tick
            args.output.mkdir(exist_ok=False)
            tick = time.perf_counter(); checkpoint_sha,terminal = save(state,base,args.output / "resume.pt")
            serialization = time.perf_counter()-tick
            facts.update(training_state_discarded=False,training_wall_seconds=wall,images_per_second=6400/wall)
        assert old.fingerprint(qualified.late.frozen_state(state["model"],12)) == base["frozen_sha256"]
        assert old.fingerprint(dict(state["model"].named_buffers())) == base["buffers_sha256"]
        assert old.fingerprint({"cpu":torch.random.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}) == rng
        assert old.coverage.teacher.qualified.numerical_flags() == base["numerical_flags"]
        assert json.loads(json.dumps(old.coverage.trained.native.environment(state["model"],state["processor"]))) == source["environment"]
        assert sha(qualified.RESUME) == SOURCE
        facts.update(initial_state_sha256=initial,schedule_sha256=chosen["schedule_sha256"],resume_identity=base,
            updates=len(rows),completed_step=len(rows),steps=rows,checkpoint_sha256=checkpoint_sha,
            terminal_state_fingerprint=terminal,frozen_named_state_buffers_rng_preserved=True,serialization_seconds=serialization,
            median_step_3_end_seconds=statistics.median(r["seconds"] for r in rows[2:]))
    assert closure(root,args.execution_sha256,args.cpu_execution_sha256) == (code,previous)
    swap = int(next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmSwap:")).split()[1])
    facts.update(unit_invocation_id=invocation,unit_cgroup=str(cgroup),unit_memory_max_bytes=8*1024**3,
        unit_memory_swap_max_bytes=0,total_seconds=time.perf_counter()-STARTED,host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        host_swap_kib=swap,peak_cuda_allocated_bytes=0 if args.phase == "startup" else torch.cuda.max_memory_allocated())
    resources(facts,300 if args.phase == "train" else 120,args.phase != "startup")
    if args.phase != "train": args.output.mkdir(exist_ok=False)
    native.atomic_write(args.output / "receipt.json",lambda stream:stream.write((json.dumps(facts,indent=2)+"\n").encode()))
    print("PASS token residual " + args.phase + "; no quality read",flush=True)


def cpu_module_original():
    import qualify_token_residual_cpu as cpu
    return cpu.ORIGINAL


if __name__ == "__main__":
    main()
