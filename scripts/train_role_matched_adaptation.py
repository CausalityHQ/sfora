#!/usr/bin/env python3
"""Parent-owned role-matched native128 continuation, no quality reads.

Freeze role-matched-train-execution.json: exact qualified v3 118-file manifest
plus ONLY this file and test_role_matched_adaptation.py (120 total). Keep the
114-file late-dense-execution.json and role-matched-cpu-execution.json intact.
Run --phase startup --arm control --seed 179032 on CPU (CUDA_VISIBLE_DEVICES=''),
then --phase mechanics --arm role_matched --seed 179032 on one CUDA device,
then one --phase train --arm control|role_matched --seed 179032|179041 per unit.
Endpoint order, four-unit cost/quality admission, locks and native jobs belong
to the parent. No endpoint is selected here and historical cost is never used.

Every phase needs --output NEW_DIRECTORY --execution-sha256 TRAIN120_SHA,
--cpu-execution-sha256 CPU118_SHA --cpu-proof CPU_JSON --cpu-sha256 CPU_JSON_SHA
--cpu-log ORIGINAL_CPU_LOG --cpu-log-sha256 CPU_LOG_SHA. Mechanics/train also
need --startup-proof DIR/receipt.json --startup-sha256 SHA --startup-log LOG
--startup-log-sha256 SHA; train additionally needs the four --mechanics-* flags.
CPU authority is pinned to the actual v3 result, not interchangeable evidence.
--boundary accepts only 12. Use both established flock locks, systemd caps
RuntimeMaxSec=120 (startup/mechanics) or 300 (train), MemoryMax=8G, SwapMax=0,
CUBLAS_WORKSPACE_CONFIG=:4096:8 and the frozen source PYTHONPATH.

Output receipt.json schema role-matched-train-v1 binds code, CPU/startup/
mechanics proofs and original logs, source/initial state, source-derived mask,
all schedules, complete resume identity, per-step inputs/losses/gradients,
whole-unit resources and non-quality assertions. Train additionally publishes
resume.pt with EXACT ten keys: identity, vision, buffers, head, classifier,
bank, optimizer, scaler, cpu_rng, cuda_rng. Immutable role/positive sidecars
are metadata-derived, fingerprint-bound and never replace original positive.
Startup checks both fresh arms without updates. Mechanics discards its actual
17-step state after exact uninterrupted17 vs serialized8+9 and independent
updated400/head/nonpersistent-buffer/packed reload. Projection<=269 is only a
TRAIN100 admission estimate. Every train starts source1000 with fresh AdamW,
scaler128/counter0; candidate179032 alone replays the first17 diagnostics.

Import and --help use stdlib only. Local check: python3 -B -S
scripts/test_role_matched_adaptation.py. Native execution is parent-owned.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions')

import time
STARTED = time.perf_counter()
import argparse
import copy
import gc
import hashlib
import inspect
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

METHOD = 'role-matched-smoothap-v1'
SCHEMA = 'role-matched-train-v1'
RESUME_SCHEMA = 'role-matched-complete-resume-v1'
ADDED = {'train_role_matched_adaptation.py', 'test_role_matched_adaptation.py'}
CPU_ADDED = {'role_matched_bank_rank.py', 'test_role_matched_bank_rank.py',
             'qualify_role_matched_cpu.py', 'test_role_matched_cpu.py'}
CPU_EXECUTION = 'f5042997e2e94fbd3f8a3bfc357d042d1d015ae889fcd84023deb44718151db8'
CPU_PROOF = '451222ed2f041d3836ee17692cec145e55bc6a47791c7cb1e9fb8bd521ca8a0c'
CPU_LOG = 'f851e651ea83ea4aa5cf1b80c5e4c7e2f50a53746bdee4b5c19cc1ada1035c6e'
ORIGINAL = 'c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3'
SOURCE = 'cc58377f0e9aa90be529bf9a3eca1746a2dc467f765dd9681a3b9b690e324566'
INITIAL = '77c26114a74472682f7f511d732dfac2679d99bfe120ee52d3f78c032a9d9867'
SIGLIP = '274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31'
MASK = '64e34f8a40c7d8de9cb766f2170a6bf324357604dba7e1166e73772e0b9eb99d'
RESUME_KEYS = frozenset(('identity', 'vision', 'buffers', 'head', 'classifier',
                         'bank', 'optimizer', 'scaler', 'cpu_rng', 'cuda_rng'))
# Same INPUT_AUTHORITIES as token-residual trainer; input/loss/gradient witnesses only.
INPUT_AUTHORITIES = {
    179032: '1ec76986fb13916165d510935a9e7fb4508cd243a8cce0b7ad8211a75fb04331',
    179041: '739337d9b6ac6b653fc8c42e00f60ebb92cbd6061ba3393f25a731b6fd46b70f'}
CPU_FACTS = ('native400_whole_head_packed_reload_exact', 'control_loss_gradient_exact',
             'rank_live_source_head_layers_gradient', 'bank_detached_unchanged',
             'source_roles_complete_state_exact', 'RNG_preserved')
STARTUP_FACTS = ('both_arms_complete_state_exact', 'resume_negatives_rejected',
                 'optimizer_roles_options_exact', 'changed_module_rejected', 'RNG_preserved')
MECHANICS_FACTS = ('training_state_discarded', 'native_17_equals_serialized8_plus9_exact',
                   'strict400_reload_whole_head_packed_exact', 'frozen_named_state_buffers_rng_preserved')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_authority(path, expected, log=False):
    assert path and expected and sha(path) == expected, 'authority digest differs'
    text = Path(path).read_text()
    return text if log else json.loads(text)


def atomic_write(path, writer):
    """Exclusive atomic publication; delegate to the frozen implementation at runtime."""
    temporary = path.with_name(path.name + '.part')
    assert not path.exists() and not path.is_symlink()
    with temporary.open('xb') as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def resources(value, cap, gpu):
    assert value['unit_invocation_id'] and Path(value['unit_cgroup']).name.endswith('.service')
    assert value['unit_memory_max_bytes'] == 8 * 1024**3 and value['unit_memory_swap_max_bytes'] == 0
    assert math.isfinite(value['total_seconds']) and 0 < value['total_seconds'] < cap
    assert 0 < value['host_max_rss_kib'] <= 8 * 1024 * 1024 and value['host_swap_kib'] == 0
    peak = value['peak_cuda_allocated_bytes']
    assert 0 < peak < 10_000_000_000 if gpu else peak == 0


def validate_log(text, value, cap):
    assert all(s in text for s in ('Finished with result: success', 'code=exited/status=0', 'Memory swap peak: 0B'))
    unit = Path(value['unit_cgroup']).name
    assert f"Running as unit: {unit}; invocation ID: {value['unit_invocation_id']}" in text
    runtimes = re.findall(r'Service runtime: (?:(\d+)min )?([\d.]+)s', text)
    assert len(runtimes) == 1 and 0 < int(runtimes[0][0] or 0)*60 + float(runtimes[0][1]) < cap
    rss = re.findall(r'Maximum resident set size \(kbytes\): (\d+)', text)
    assert rss and 0 < value['host_max_rss_kib'] <= int(rss[0]) <= 8*1024*1024
    for lock in ('/home/riomus/runs/.sfora-siglip2-gpu.lock', '/home/riomus/.sfora-siglip2-gpu.lock'):
        assert 'flock -n ' + lock in text


def validate_cpu(value, execution, code, previous):
    assert value['schema'] == 'role-matched-cpu-v1' and value['pass'] is True
    assert execution == value['execution_sha256'] == CPU_EXECUTION and value['code'] == previous
    assert len(previous) == 118 and len(code) == 120 and set(code)-set(previous) == ADDED
    assert all(code[n] == h for n,h in previous.items()), 'qualified CPU closure changed'
    assert value['source_checkpoint_sha256'] == SOURCE and value['initial_state_sha256'] == INITIAL
    assert value['boundary'] == 12 and value['optimizer_members'] == 208
    assert value['role_positive_sha256'] == MASK and value['module_sha256'] == previous['role_matched_bank_rank.py']
    assert value['siglip_module_sha256'] == SIGLIP
    assert (value['fit_query_images'], value['fit_gallery_images']) == (6673, 6610)
    assert all(value[k] is True for k in CPU_FACTS)
    facts = value['witnesses']
    assert (facts['held_rows'], facts['query'], facts['gallery']) == (12599, 6354, 6245)
    assert all(facts[k] is True for k in ('exact_role_hashes', 'fit_disjointness_checked',
        'independent_oracle_loss_gradient', 'no_mask_exact_loss_gradient_parity', 'control_exact_parity',
        'detached_unchanged_bank', 'fullbatch_allinvalid', 'fp32_autocast_off', 'non_diagonal_self'))
    assert value['optimizer_updates'] == 0 and value['quality_read'] is False
    resources(value, 120, False)


def closure(root, execution, cpu_execution):
    assert cpu_execution == CPU_EXECUTION
    code = read_authority(root/'role-matched-train-execution.json', execution)
    previous = read_authority(root/'role-matched-cpu-execution.json', CPU_EXECUTION)
    original = read_authority(root/'late-dense-execution.json', ORIGINAL)
    assert len(original) == 114 and len(previous) == 118 and len(code) == 120
    assert set(previous)-set(original) == CPU_ADDED and set(code)-set(previous) == ADDED
    assert all(previous[n] == h for n,h in original.items())
    assert all(code[n] == h for n,h in previous.items())
    assert all(sha(root/n) == h for n,h in code.items()), 'execution source changed'
    return code, previous


def validate_arguments(args):
    assert args.phase in ('startup', 'mechanics', 'train') and args.arm in ('control', 'role_matched')
    assert args.seed in INPUT_AUTHORITIES and args.boundary == 12
    assert (args.cpu_execution_sha256, args.cpu_sha256, args.cpu_log_sha256) == (CPU_EXECUTION, CPU_PROOF, CPU_LOG)
    startup = [getattr(args, 'startup_'+k) for k in ('proof', 'sha256', 'log', 'log_sha256')]
    mechanics = [getattr(args, 'mechanics_'+k) for k in ('proof', 'sha256', 'log', 'log_sha256')]
    if args.phase == 'startup':
        assert (args.arm, args.seed) == ('control', 179032) and not any(startup + mechanics)
    else:
        assert all(startup)
        if args.phase == 'mechanics':
            assert (args.arm, args.seed) == ('role_matched', 179032) and not any(mechanics)
        else:
            assert all(mechanics)


def validate_startup(value, args, cpu):
    assert value['schema'] == SCHEMA and value['intervention'] == METHOD and value['pass'] is True
    assert (value['phase'], value['arm'], value['seed'], value['boundary']) == ('startup', 'control', 179032, 12)
    for k,v in (('execution_sha256',args.execution_sha256), ('cpu_authority_sha256',args.cpu_sha256),
                ('cpu_log_sha256',args.cpu_log_sha256), ('cpu_execution_sha256',CPU_EXECUTION),
                ('source_checkpoint_sha256',SOURCE), ('initial_state_sha256',INITIAL), ('role_positive_sha256',MASK)):
        assert value[k] == v
    assert cpu['initial_state_sha256'] == INITIAL and value['optimizer_members'] == {'control':208, 'role_matched':208}
    assert value['optimizer_updates'] == 0 and value['quality_read'] is False and value['checkpoint_sha256'] is None
    assert all(value[k] is True for k in STARTUP_FACTS)
    assert set(value['schedules']) == {str(k) for k in INPUT_AUTHORITIES}
    assert set(value['resume_identities']) == {'control', 'role_matched'}
    for seed,digest in INPUT_AUTHORITIES.items():
        assert value['schedules'][str(seed)]['input_authority_sha256'] == digest
    for arm,base in value['resume_identities'].items():
        assert base['schema'] == RESUME_SCHEMA and base['intervention'] == METHOD and base['arm'] == arm
        assert base['precision'] == 'cpu_float32' and base['boundary'] == 12 and base['width'] == 128
        assert base['objective'] == objective_name(arm) and base['role_positive_sha256'] == MASK
        assert base['code'] == value['code'] and base['module_sha256'] == value['code']['role_matched_bank_rank.py']
        assert base['execution_sha256'] == args.execution_sha256 and base['source_checkpoint_sha256'] == SOURCE
        assert base['cpu_authority_sha256'] == CPU_PROOF and base['cpu_log_sha256'] == CPU_LOG
        assert base['cpu_execution_sha256'] == CPU_EXECUTION and base['initial_state_sha256'] == INITIAL
        assert len(base['parameter_names']) == 208 and len(base['optimizer_groups']) == 3
    resources(value, 120, False)


def validate_mechanics(value, args, cpu, startup):
    validate_startup(startup, args, cpu)
    assert value['schema'] == SCHEMA and value['pass'] is True and value['intervention'] == METHOD
    assert (value['phase'], value['arm'], value['seed'], value['boundary']) == ('mechanics', 'role_matched', 179032, 12)
    for k,v in (('execution_sha256',args.execution_sha256), ('cpu_authority_sha256',CPU_PROOF),
                ('cpu_log_sha256',CPU_LOG), ('cpu_execution_sha256',CPU_EXECUTION),
                ('startup_authority_sha256',args.startup_sha256), ('startup_log_sha256',args.startup_log_sha256),
                ('source_checkpoint_sha256',SOURCE), ('initial_state_sha256',INITIAL), ('role_positive_sha256',MASK)):
        assert value[k] == v
    assert value['updates'] == value['completed_step'] == 17 and value['checkpoint_sha256'] is None
    assert value['quality_read'] is False and all(value[k] is True for k in MECHANICS_FACTS)
    projection = value['chunk100_admission_seconds']
    assert math.isfinite(projection) and 0 < projection <= 269
    expected = startup['schedules']['179032']
    base = value['resume_identity']
    assert base['schema'] == RESUME_SCHEMA and base['arm'] == 'role_matched' and base['objective'] == METHOD
    assert value['schedule_sha256'] == base['schedule_sha256'] == expected['schedule_sha256']
    assert base['class_sequence_sha256'] == expected['class_sequence_sha256']
    assert value['code'] == base['code'] == startup['code'] and base['role_positive_sha256'] == MASK
    assert len(base['parameter_names']) == 208 and value['optimizer_members'] == 208
    assert len(value['steps']) == 17 and len(value['resumed_steps']) == 9
    for step,row in enumerate(value['steps'], 1):
        validate_row(row, step)
    assert [diagnostic(r) for r in value['steps'][8:]] == [diagnostic(r) for r in value['resumed_steps']]
    resources(value, 120, True)


def diagnostic(row):
    return {k:v for k,v in row.items() if k != 'seconds'}


def validate_row(row, step):
    assert row['step'] == row['optimizer_counter'] == step and row['augmentation_step'] == 1000+step
    assert len(row['image_ids']) == len(set(row['image_ids'])) == 64
    assert all(type(i) is int and 0 <= i < 13283 for i in row['image_ids'])
    assert all(math.isfinite(row[k]) for k in ('ce', 'rank', 'loss', 'preclip_norm', 'seconds'))
    assert row['loss'] == row['ce'] + 8*row['rank'] and row['scale'] == 128
    assert set(row['gradient_norms']) == {str(i) for i in range(12,24)}
    assert all(math.isfinite(v) and v > 0 for v in row['gradient_norms'].values())


def objective_name(arm):
    assert arm in ('control', 'role_matched')
    return METHOD if arm == 'role_matched' else 'corrected-original-smoothap-v1'


def attach_roles(state, rows):
    """FIT metadata sidecars; original target/positive stay under native.verify."""
    assert len(rows) == len(state['target']) == 13283
    before = old.fingerprint({k:state[k] for k in ('target','positive')})
    rng = rng_fingerprint(state)
    roles, positives = role.build_role_inventory(tuple(r['product'] for r in rows), tuple(r['relative_path'] for r in rows))
    device = state['bank'].device
    state['role_roles'] = torch.tensor(roles, dtype=torch.long, device=device)
    state['role_positive'] = torch.tensor(positives, dtype=torch.long, device=device)
    assert old.fingerprint({'roles':state['role_roles'], 'positive':state['role_positive']}) == MASK
    assert old.fingerprint({k:state[k] for k in ('target','positive')}) == before
    assert rng_fingerprint(state) == rng
    state['role_versions'] = (state['role_roles']._version, state['role_positive']._version)


def rng_fingerprint(state):
    return old.fingerprint({'cpu':torch.random.get_rng_state(),
        'cuda':torch.cuda.get_rng_state_all() if state['scaler'] else []})


def identity(state, args, initial, chosen, code, *, initial_rng_sha256=None):
    base = native.identity(state, args, initial, chosen['schedule_sha256'])
    base.update(schema=RESUME_SCHEMA, intervention=METHOD, arm=args.arm, dataset_arm='half',
        objective=objective_name(args.arm), rank_temperature=.01, rank_coefficient=8,
        mask_rule='sha256-path-gallery-round-half-opposite-role-all-positives-v1',
        role_positive_sha256=MASK, module_sha256=code['role_matched_bank_rank.py'], code=code,
        objective_sha256=old.fingerprint({n:code[n] for n in ('role_matched_bank_rank.py',
            'train_pe_teacher_retained256.py', 'pe_native_valid_anchor.py', 'src/sfora/deployed_code_rank.py')}),
        class_sequence_sha256=chosen['class_sequence_sha256'],
        input_authorities={str(k):v for k,v in INPUT_AUTHORITIES.items()},
        startup_authority_sha256=args.startup_sha256, startup_log_sha256=args.startup_log_sha256,
        initial_rng_sha256=initial_rng_sha256 if initial_rng_sha256 is not None else rng_fingerprint(state),
        precision='native_float32_fp16_autocast' if state['scaler'] else 'cpu_float32',
        numerical_flags=old.coverage.teacher.qualified.numerical_flags(),
        head_roles=[(n,p.requires_grad) for n,p in state['head'].named_parameters()],
        classifier_role=state['classifier'].requires_grad)
    return base


def verify(state, base):
    assert base['schema'] == RESUME_SCHEMA and base['intervention'] == METHOD
    assert base['objective'] == objective_name(base['arm']) and base['boundary'] == 12
    assert base['source_checkpoint_sha256'] == SOURCE and base['initial_state_sha256'] == INITIAL
    assert base['module_sha256'] == sha(Path(role.__file__)) == base['code']['role_matched_bank_rank.py']
    native.verify(state, base)
    assert len(state['params']) == 208 and len(state['optimizer'].param_groups) == 3
    assert [(n,p.requires_grad) for n,p in state['head'].named_parameters()] == base['head_roles']
    assert all(p.requires_grad for p in state['head'].parameters()) and state['classifier'].requires_grad == base['classifier_role'] is True
    assert state['role_versions'] == (state['role_roles']._version, state['role_positive']._version)
    assert old.fingerprint({'roles':state['role_roles'], 'positive':state['role_positive']}) == base['role_positive_sha256'] == MASK


def payload(state, base):
    value = old.payload(state, base)
    assert set(value) == RESUME_KEYS
    return value


def tensor_layout(value, template):
    assert isinstance(value, torch.Tensor) and value.shape == template.shape and value.dtype == template.dtype, 'tensor shape/dtype differs'
    assert bool(torch.isfinite(value).all()), 'nonfinite resume tensor'


def validate_resume(saved, base, step, groups, template, members):
    """Complete ten-key native state, validated before any live copy/load."""
    assert set(saved) == RESUME_KEYS, 'complete native resume required'
    assert type(step) is int and 0 <= step <= 100
    assert saved['identity'] == {**base,'global_step':step}, 'foreign resume identity/counter'
    assert base['schema'] == RESUME_SCHEMA and base['intervention'] == METHOD
    assert base['objective'] == objective_name(base['arm']) and base['role_positive_sha256'] == MASK
    assert len(saved['vision']) == 400
    optimizer = saved['optimizer']
    assert set(optimizer) == {'param_groups','state'} and optimizer['param_groups'] == groups, 'optimizer order/options differ'
    ids = [i for g in groups for i in g['params']]
    assert len(ids) == len(set(ids)) == len(base['parameter_names']) == len(members) == 208
    assert set(optimizer['state']) == (set(ids) if step else set()), 'optimizer member state differs'
    for i,(_,parameter) in zip(ids,members,strict=True):
        if not step: continue
        moment = optimizer['state'][i]
        assert set(moment) == {'step','exp_avg','exp_avg_sq'}
        scalar = moment['step']
        assert isinstance(scalar,torch.Tensor) and scalar.numel() == 1 and scalar.dtype == torch.float32
        assert float(scalar) == step, 'optimizer step differs'
        tensor_layout(moment['exp_avg'],parameter); tensor_layout(moment['exp_avg_sq'],parameter)
    if base['precision'] == 'cpu_float32':
        assert saved['scaler'] is None and saved['cuda_rng'] == []
    else:
        assert base['precision'] == 'native_float32_fp16_autocast'
        assert saved['scaler'] == {**template['scaler'], 'scale':128., '_growth_tracker':step}
        assert len(saved['cuda_rng']) == 1
        tensor_layout(saved['cuda_rng'][0],template['cuda_rng'][0])
    tensor_layout(saved['cpu_rng'],template['cpu_rng'])
    for key in ('vision','buffers','head'):
        assert saved[key].keys() == template[key].keys(), 'strict state keys differ'
        for name,value in saved[key].items(): tensor_layout(value,template[key][name])
    for key in ('classifier','bank'): tensor_layout(saved[key],template[key])
    assert saved['classifier'].shape == (2004,128) and saved['bank'].shape == (13283,128)
    assert old.fingerprint(saved['buffers']) == base['buffers_sha256']
    assert all(torch.equal(saved['buffers'][n],saved['vision'][n]) for n in saved['buffers'] if n in saved['vision'])
    named = {**saved['vision'], **saved['buffers']}
    assert old.fingerprint({n:named[n] for n in base['frozen_names']}) == base['frozen_sha256']


def save(state, base, path):
    state['optimizer'].zero_grad(set_to_none=True)
    verify(state,base)
    value = payload(state,base)
    validate_resume(value,base,state['counter'],value['optimizer']['param_groups'],value,state['params'])
    fingerprint = old.fingerprint(value)
    native.atomic_write(path,lambda stream:torch.save(value,stream))
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert old.fingerprint(disk) == fingerprint, 'serialized complete state differs'
    del disk, value
    gc.collect()
    return sha(path),fingerprint


def own_optimizer_steps(optimizer):
    # Non-fused AdamW retains CPU steps verbatim: clone only these scalars.
    # Moment arrays are copied to CUDA by load_state_dict, never host-cloned.
    for moment in optimizer['state'].values():
        moment['step'] = moment['step'].clone()


def restore(state, base, path, expected_sha, fingerprint, step):
    verify(state,base)
    assert sha(path) == expected_sha
    saved = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    template = payload(state,base)
    validate_resume(saved,base,step,template['optimizer']['param_groups'],template,state['params'])
    assert old.fingerprint(saved) == fingerprint
    state['model'].load_state_dict(saved['vision'],strict=True)
    state['head'].load_state_dict(saved['head'],strict=True)
    with torch.no_grad():
        state['classifier'].copy_(saved['classifier']); state['bank'].copy_(saved['bank'])
        for n,value in state['model'].named_buffers(): value.copy_(saved['buffers'][n])
    own_optimizer_steps(saved['optimizer'])
    state['optimizer'].load_state_dict(saved['optimizer'])
    if state['scaler']:
        state['scaler'].load_state_dict(saved['scaler'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    torch.random.set_rng_state(saved['cpu_rng'])
    state['counter'] = step
    del saved, template
    gc.collect()
    verify(state,base)
    assert old.fingerprint(payload(state,base)) == fingerprint, 'restored complete state differs'


def startup_checks(state, base, directory):
    path = directory/(base['arm']+'.pt')
    saved_sha,fingerprint = save(state,base,path)
    restore(state,base,path,saved_sha,fingerprint,0)
    value = payload(state,base)
    groups = value['optimizer']['param_groups']
    def check(v): validate_resume(v,base,0,groups,value,state['params'])
    for key in RESUME_KEYS:
        bad = dict(value); del bad[key]
        qualified.rejects(lambda:check(bad))
    for key in ('arm','objective','objective_sha256','role_positive_sha256','source_checkpoint_sha256',
                'initial_state_sha256','execution_sha256','cpu_authority_sha256','cpu_log_sha256','initial_rng_sha256'):
        qualified.rejects(lambda:check({**value,'identity':{**value['identity'],key:'foreign'}}))
    qualified.rejects(lambda:check({**value,'identity':{**value['identity'],'schema':'late-dense-complete-resume-v1'}}))
    for key in ('vision','head','buffers'):
        bad = {**value,key:dict(value[key])}; del bad[key][next(iter(bad[key]))]
        qualified.rejects(lambda:check(bad))
    qualified.rejects(lambda:check({**value,'optimizer':{**value['optimizer'],'state':{0:{}}}}))
    for case in ('order','options'):
        bad_groups = copy.deepcopy(groups)
        if case == 'order': bad_groups[0]['params'].reverse()
        else: bad_groups[-1]['lr'] = 1e-3
        qualified.rejects(lambda:check({**value,'optimizer':{**value['optimizer'],'param_groups':bad_groups}}))
    members = state['optimizer'].param_groups[0]['params']
    removed = members.pop(); qualified.rejects(lambda:verify(state,base)); members.append(removed)
    members.append(removed); qualified.rejects(lambda:verify(state,base)); members.pop()
    qualified.rejects(lambda:restore(state,base,path,'foreign',fingerprint,0))
    verify(state,base)
    assert state['counter'] == 0 and not state['optimizer'].state
    path.unlink()


def corrected_step(state, pixels, ids, active, arm):
    """Actual corrected terms always calls valid_rank, including rank_active=True."""
    original = native.objective.lane.valid_rank
    def rank(raw, bank, head, positives, index):
        if arm == 'control':
            return original(raw,bank,head,positives,index)
        assert arm == 'role_matched' and bank is state['bank'] and head is state['head']
        return role.bank_rank_loss(raw,bank,state['role_positive'][index],index,
            arm=arm,labels=state['target'],roles=state['role_roles'])
    with patch.object(old.coverage,'terms',native.objective.terms), patch.object(native.objective.lane,'valid_rank',rank):
        return old.step(state,pixels,ids,active,micro=16)


def strict_reload(state, path, pixels, base):
    """Independent CUDA construction before mmap; release mapping before live checks."""
    state['model'].eval(); state['head'].eval()
    rng = rng_fingerprint(state)
    with torch.random.fork_rng(devices=[0]):
        with torch.device('cuda'):
            model = type(state['model'])(copy.deepcopy(state['model'].config)).float().eval()
            head = torch.nn.Linear(1024,128).float().eval()
        disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert len(disk['vision']) == 400
        model.load_state_dict(disk['vision'],strict=True); head.load_state_dict(disk['head'],strict=True)
        buffers = dict(model.named_buffers()); assert buffers.keys() == disk['buffers'].keys()
        with torch.no_grad():
            for n,value in buffers.items(): value.copy_(disk['buffers'][n])
        expected = {k:old.fingerprint(disk[k]) for k in ('vision','head','buffers')}
        del disk
        gc.collect()
        for key,values in (('vision',model.state_dict()),('head',head.state_dict()),('buffers',buffers)):
            assert old.fingerprint(values) == expected[key]
        assert old.fingerprint(model.state_dict()) == old.fingerprint(state['model'].state_dict())
        assert old.fingerprint(head.state_dict()) == old.fingerprint(state['head'].state_dict())
        assert old.fingerprint(buffers) == base['buffers_sha256']
        assert old.fingerprint(qualified.late.frozen_state(model,12)) == base['frozen_sha256']
        with torch.no_grad():
            a,b = (old.previous.training.fp16(m,pixels) for m in (state['model'],model))
            assert torch.equal(a,b)
            ra,rb = (pair.smoke.compact_head_features(v,h) for v,h in ((a,state['head']),(b,head)))
            assert torch.equal(ra,rb)
            va,vb = (torch.nn.functional.normalize(v,dim=1) for v in (ra,rb))
            assert torch.equal(va,vb)
            old.previous.training.packed_equal(va,vb)
    assert rng_fingerprint(state) == rng


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase',choices=('startup','mechanics','train'),required=True)
    parser.add_argument('--arm',choices=('control','role_matched'),required=True)
    parser.add_argument('--boundary',type=int,choices=(12,),default=12)
    parser.add_argument('--seed',type=int,choices=tuple(INPUT_AUTHORITIES),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-sha256',required=True)
    parser.add_argument('--cpu-execution-sha256',required=True)
    for name in ('cpu','startup','mechanics'):
        for suffix in ('proof','log'): parser.add_argument(f'--{name}-{suffix}',type=Path,required=name == 'cpu')
        parser.add_argument(f'--{name}-sha256',required=name == 'cpu')
        parser.add_argument(f'--{name}-log-sha256',required=name == 'cpu')
    args = parser.parse_args()
    validate_arguments(args)
    assert not args.output.exists() and not args.output.is_symlink()
    root = Path(__file__).resolve().parent
    code,previous = closure(root,args.execution_sha256,args.cpu_execution_sha256)
    cpu = read_authority(args.cpu_proof,args.cpu_sha256)
    validate_cpu(cpu,args.cpu_execution_sha256,code,previous)
    validate_log(read_authority(args.cpu_log,args.cpu_log_sha256,log=True),cpu,120)
    cgroup = Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().split('0::',1)[1].strip().lstrip('/')
    assert int((cgroup/'memory.max').read_text()) == 8*1024**3 and int((cgroup/'memory.swap.max').read_text()) == 0
    assert str(os.getpid()) in (cgroup/'cgroup.procs').read_text().split()
    invocation = os.environ['INVOCATION_ID']
    startup = mechanics = None
    if args.phase == 'startup':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    else:
        startup = read_authority(args.startup_proof,args.startup_sha256)
        validate_startup(startup,args,cpu)
        assert startup['code'] == code
        validate_log(read_authority(args.startup_log,args.startup_log_sha256,log=True),startup,120)
        if args.phase == 'train':
            mechanics = read_authority(args.mechanics_proof,args.mechanics_sha256)
            validate_mechanics(mechanics,args,cpu,startup)
            validate_log(read_authority(args.mechanics_log,args.mechanics_log_sha256,log=True),mechanics,120)
    inputs = {}
    for seed,digest in INPUT_AUTHORITIES.items():
        history = read_authority(Path(f'/home/riomus/runs/sfora-late-dense-control-{seed}-v1/receipt.json'),digest)
        assert history['pass'] is True and history['phase'] == 'train' and history['boundary'] == 12 and history['seed'] == seed
        assert history['execution_sha256'] == ORIGINAL and history['source_checkpoint_sha256'] == SOURCE
        assert history['initial_state_sha256'] == INITIAL and len(history['steps']) == 100 and history['quality_read'] is False
        inputs[seed] = history
    global torch,native,qualified,old,pair,role
    import torch
    import train_late_dense_adaptation as native
    import role_matched_bank_rank as role
    from sfora.deployed_code_rank import smooth_ap_bank_loss
    qualified,old,pair = native.qualification,native.old,native.pair
    pair.executing_authority(root,code)
    assert sha(Path(inspect.getfile(smooth_ap_bank_loss))) == code['src/sfora/deployed_code_rank.py']
    selected = qualified.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected,'helpers',lambda r,_:helpers(r,code)):
        control,source,prior,proof,_,_ = qualified.startup(root,ORIGINAL)
    pair.executing_authority(root,code)
    torch.set_num_threads(8); torch.manual_seed(args.seed)
    if args.phase == 'startup':
        assert not torch.cuda.is_available()
    else:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        assert old.coverage.teacher.qualified.numerical_flags() == prior['numerical_flags']
        # Whole-unit peak starts BEFORE the first factory and is never reset again.
        torch.cuda.reset_peak_memory_stats()
    facts = {'schema':SCHEMA,'pass':True,'intervention':METHOD,'code':code,
        'phase':args.phase,'arm':args.arm,'seed':args.seed,'boundary':12,'execution_sha256':args.execution_sha256,
        'cpu_authority_sha256':CPU_PROOF,'cpu_log_sha256':CPU_LOG,'cpu_execution_sha256':CPU_EXECUTION,
        'source_checkpoint_sha256':SOURCE,'role_positive_sha256':MASK,'quality_read':False,'claim_eligible':False,
        'startup_authority_sha256':args.startup_sha256,'startup_log_sha256':args.startup_log_sha256,
        'mechanics_sha256':args.mechanics_sha256,'mechanics_log_sha256':args.mechanics_log_sha256}
    def fresh(device):
        state,initial = qualified.fresh(control,source,proof,12,device)
        assert initial == INITIAL and state['counter'] == 0 and not state['optimizer'].state
        assert state['scaler'] is None or state['scaler'].get_scale() == 128
        assert sha(Path(inspect.getfile(type(state['model'])))) == SIGLIP
        attach_roles(state,proof['arms']['half']['rows'])
        return state,initial
    def schedules(state):
        target = state['target'].cpu().tolist()
        result = {}
        for seed in INPUT_AUTHORITIES:
            batches = old.coverage.schedule(target,seed=seed)
            digest = pair.smoke.digest({'batches':torch.from_numpy(batches)})
            assert digest == inputs[seed]['schedule_sha256']
            classes = [[int(target[int(i)]) for i in batch] for batch in batches]
            result[str(seed)] = {'schedule_sha256':digest,
                'class_sequence_sha256':hashlib.sha256(json.dumps(classes,separators=(',',':')).encode()).hexdigest(),
                'input_authority_sha256':INPUT_AUTHORITIES[seed]}
        return result
    checkpoint_sha = None
    if args.phase == 'startup':
        with TemporaryDirectory(prefix='discard-role-startup-') as temporary:
            bases = {}
            for arm in ('control','role_matched'):
                state,initial = fresh('cpu')
                expected = schedules(state)
                option = SimpleNamespace(**{**vars(args),'arm':arm})
                base = identity(state,option,initial,expected[str(args.seed)],code)
                rng = rng_fingerprint(state)
                startup_checks(state,base,Path(temporary))
                assert rng_fingerprint(state) == rng
                bases[arm] = base
                del state
                gc.collect()
        original_sha = sha
        with patch(__name__+'.sha',lambda p:'changed' if Path(p).resolve() == Path(role.__file__).resolve() else original_sha(p)):
            qualified.rejects(lambda:closure(root,args.execution_sha256,CPU_EXECUTION))
        facts.update(initial_state_sha256=INITIAL,optimizer_members={'control':208,'role_matched':208},
            optimizer_updates=0,schedules=expected,resume_identities=bases,checkpoint_sha256=None,
            both_arms_complete_state_exact=True,resume_negatives_rejected=True,optimizer_roles_options_exact=True,
            changed_module_rejected=True,RNG_preserved=True)
    else:
        state,initial = fresh('cuda')
        expected = schedules(state); assert expected == startup['schedules']
        chosen = expected[str(args.seed)]
        batches = old.coverage.schedule(state['target'].cpu().tolist(),seed=args.seed).tolist()
        base = identity(state,args,initial,chosen,code)
        verify(state,base)
        for key in ('parameter_names','optimizer_groups','head_roles','classifier_role','frozen_names',
                    'frozen_sha256','buffers_sha256','target_positive_sha256','role_positive_sha256',
                    'module_sha256','objective_sha256','objective','mask_rule'):
            assert json.loads(json.dumps(base[key])) == startup['resume_identities'][args.arm][key]
        rng = rng_fingerprint(state)
        original_terms,original_rank = old.coverage.terms,native.objective.lane.valid_rank
        target = state['target'].cpu().tolist()
        counts = old.coverage.np.bincount(target)
        def update(s,step):
            counter,augmentation = native.counters(step)
            assert s['counter'] == counter-1
            torch.cuda.synchronize(); tick = time.perf_counter()
            ids = tuple(int(i) for i in batches[step-1])
            with patch.object(pair,'SEED',args.seed):
                images,rgb = pair.augmented_images(control.dataset_root,proof['arms']['half']['rows'],ids,augmentation)
            pixels = pair.pixels(s['processor'],images,'large')
            assert pixels.shape == (64,3,256,256)
            pixel_sha = pair.smoke.digest({'pixels':pixels})
            history = inputs[args.seed]['steps'][step-1]
            assert history['augmentation_step'] == augmentation and history['rgb_sha256'] == rgb and history['pixels_sha256'] == pixel_sha
            active = all(counts[target[i]] > 1 for i in ids)
            assert active == history['rank_active_before']
            row = corrected_step(s,pixels,ids,active,args.arm)
            assert old.coverage.terms is original_terms and native.objective.lane.valid_rank is original_rank
            assert s['counter'] == counter
            if args.arm == 'control':
                for key in ('ce','rank','loss','scale','preclip_norm','gradient_norms'):
                    assert row[key] == history[key], 'fresh original control loss/gradient differs'
            row.update(optimizer_counter=counter,augmentation_step=augmentation,image_ids=list(ids),
                rgb_sha256=rgb,pixels_sha256=pixel_sha,rank_active_before=active)
            if mechanics and args.arm == 'role_matched' and args.seed == 179032 and step <= 17:
                assert diagnostic(row) == diagnostic(mechanics['steps'][step-1]), 'candidate mechanics replay differs'
            assert s['role_versions'] == (s['role_roles']._version,s['role_positive']._version)
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            torch.cuda.synchronize(); row['seconds'] = time.perf_counter()-tick
            validate_row(row,step)
            print(json.dumps(row,allow_nan=False),flush=True)
            return row
        serialization = 0.
        if args.phase == 'mechanics':
            with TemporaryDirectory(prefix='discard-role-mechanics-') as temporary:
                path = Path(temporary)/'step8.pt'
                rows = []
                for step in range(1,18):
                    rows.append(update(state,step))
                    if step == 8:
                        tick = time.perf_counter(); saved_sha,saved_fingerprint = save(state,base,path)
                        serialization += time.perf_counter()-tick
                terminal = old.fingerprint(payload(state,base))
                del state
                gc.collect(); torch.cuda.empty_cache()
                state,again = fresh('cuda'); assert again == initial
                assert identity(state,args,initial,chosen,code,initial_rng_sha256=base['initial_rng_sha256']) == base
                restore(state,base,path,saved_sha,saved_fingerprint,8)
                resumed = [update(state,step) for step in range(9,18)]
                assert [diagnostic(r) for r in resumed] == [diagnostic(r) for r in rows[8:]]
                assert old.fingerprint(payload(state,base)) == terminal
                path17 = Path(temporary)/'step17.pt'
                tick = time.perf_counter(); _,fingerprint = save(state,base,path17)
                serialization += time.perf_counter()-tick
                assert fingerprint == terminal
                verify(state,base)
                del state['optimizer']
                gc.collect(); torch.cuda.empty_cache()
                with patch.object(pair,'SEED',args.seed):
                    images,_ = pair.augmented_images(control.dataset_root,proof['arms']['half']['rows'],tuple(int(i) for i in batches[0][:2]),1001)
                strict_reload(state,path17,pair.pixels(state['processor'],images,'large').cuda(),base)
            overhead = time.perf_counter()-STARTED-sum(r['seconds'] for r in rows+resumed)
            admission = 100*max(statistics.median(r['seconds'] for r in rows[2:]),statistics.mean(r['seconds'] for r in rows))+max(30.,overhead)
            assert math.isfinite(admission) and admission <= 269
            facts.update(training_state_discarded=True,native_17_equals_serialized8_plus9_exact=True,
                strict400_reload_whole_head_packed_exact=True,resumed_steps=resumed,
                chunk100_admission_seconds=admission,measured_unit_overhead_seconds=overhead)
        else:
            tick = time.perf_counter(); rows = [update(state,step) for step in range(1,101)]
            wall = time.perf_counter()-tick
            args.output.mkdir(exist_ok=False)
            tick = time.perf_counter(); checkpoint_sha,terminal = save(state,base,args.output/'resume.pt')
            serialization = time.perf_counter()-tick
            facts.update(training_state_discarded=False,training_wall_seconds=wall,images_per_second=6400/wall,
                original_control_100_loss_gradient_exact=args.arm == 'control')
        assert old.fingerprint(qualified.late.frozen_state(state['model'],12)) == base['frozen_sha256']
        assert old.fingerprint(dict(state['model'].named_buffers())) == base['buffers_sha256']
        assert old.fingerprint({'roles':state['role_roles'],'positive':state['role_positive']}) == MASK
        assert rng_fingerprint(state) == rng
        assert old.coverage.teacher.qualified.numerical_flags() == base['numerical_flags']
        assert old.coverage.terms is original_terms and native.objective.lane.valid_rank is original_rank
        assert json.loads(json.dumps(old.coverage.trained.native.environment(state['model'],state['processor']))) == source['environment']
        assert sha(qualified.RESUME) == SOURCE
        facts.update(initial_state_sha256=initial,schedule_sha256=chosen['schedule_sha256'],resume_identity=base,
            optimizer_members=208,updates=len(rows),completed_step=len(rows),steps=rows,checkpoint_sha256=checkpoint_sha,
            terminal_state_fingerprint=terminal,frozen_named_state_buffers_rng_preserved=True,serialization_seconds=serialization,
            median_step_3_end_seconds=statistics.median(r['seconds'] for r in rows[2:]))
    assert closure(root,args.execution_sha256,CPU_EXECUTION) == (code,previous)
    swap = int(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmSwap:')).split()[1])
    facts.update(unit_invocation_id=invocation,unit_cgroup=str(cgroup),unit_memory_max_bytes=8*1024**3,
        unit_memory_swap_max_bytes=0,total_seconds=time.perf_counter()-STARTED,host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        host_swap_kib=swap,peak_cuda_allocated_bytes=0 if args.phase == 'startup' else torch.cuda.max_memory_allocated())
    resources(facts,300 if args.phase == 'train' else 120,args.phase != 'startup')
    if args.phase != 'train': args.output.mkdir(exist_ok=False)
    native.atomic_write(args.output/'receipt.json',lambda stream:stream.write((json.dumps(facts,indent=2,allow_nan=False)+'\n').encode()))
    print('PASS role matched '+args.phase+'; no quality read',flush=True)


if __name__ == '__main__':
    main()
