#!/usr/bin/env python3
"""Stage1 fixed TRAIN vector measurement; native qualification is UNRUN.

Root supplies actual hashes, never future placeholders. CLI --execution-sha256
SHA --authority FILE --authority-sha256 SHA --output NEWDIR. execution.json is
exactly FILES. connected-gradient-decomposition-launch-v1 has exactly LAUNCH_KEYS.
FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}.
CODE={root:canonical separate directory,execution_sha256:SHA,code:exact own2}.
UNIT={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
native_peak_rss_kib,both_locks_held:true}; genuine normal-exit admission remains.
training is accepted source-v6 CODE; evaluator is the original authenticated
bootstrap/terminal-reader CODE, never a quality-reader invocation. bootstrap is
the original control061 TRAIN launch FILE. candidates are exactly061 then069,
each {seed,launch:FILE,terminal:UNIT,checkpoint:FILE,terminal_state_sha256:SHA}.
python and workspace_source are explicit FILEs. workspace_source exposes the
unchanged capture_workspace_owner(torch,{guards,source_cpu,warm}) interface;
both records retain their original genuine CPU/warm roles. No direct private
CUDA cleanup, extra empty_cache, peak reset, helper adapter or global rebinding.
The original release's existing allocator flush remains unchanged.

Four sequential fresh states:061 init/accepted128,069 init/accepted128. First
original B64, canonical then augmented, micro16; inherited canonical S ONLY,
complete6355 gallery, unchanged loss_terms and full-B64 K63/K64 denominators.
Actual MSE/ranking/total gradients for fourMLP+A+C, FP64 bounded chunk scalars,
zero-norm cosine=null, rtol1e-5/atol1e-6 correspondence is numerical only.
Three joint-view and three per-view accumulators plus one returned gradient set;
no vector serialization. Per-view accumulators die before the next view.
No optimizer step, new galleries, forks, held metrics, causal/GO or stale-gallery
attribution. B/U and causal interventions remain future. Proposed3000s/8GiB,
events0/swap0/CUDA<10GB/both lifetime locks/<=8sets/512MiB extra are UNMEASURED.
The parent alone freezes and launches native authority, captures normal exit,
and decides anything scientific. Source tests confer no native eligibility.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace
import weakref

if not __debug__:
    raise SystemExit('unoptimized original assertions required')

HERE = Path(__file__).absolute().parent
FILES = {'qualify_connected_gradient_decomposition.py','test_connected_gradient_decomposition.py'}
LAUNCH_KEYS = {'schema','execution_sha256','python','training','evaluator','bootstrap','candidates',
               'workspace_source','output','resource_policy','both_locks_held'}
POLICY = {'seconds':3000,'host_bytes':8*1024**3,'swap_bytes':0,
          'cuda_allocated_bytes_exclusive':10_000_000_000,
          'gradient_sets_max':8,'extra_tensor_bytes':512*1024**2}
SEEDS, VIEWS, TERMS = (179061,179069), ('canonical','augmented'), ('regression','ranking','total')
MLP = tuple('encoder.layers.26.mlp.'+layer+'.'+field for layer in ('fc1','fc2') for field in ('weight','bias'))
NAMES = (*MLP,'A','C')
VECTOR_BYTES, CHUNK, RTOL, ATOL = 40_359_232, 65536, 1e-5, 1e-6
TRAIN_EXECUTION = 'a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c'
TRAIN_CODE = {'train_siglip2_connected_mlp.py':'79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b',
              'test_siglip2_connected_mlp.py':'8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25'}
INITIAL = {179061:('8b89897fdaaad94770e40bd2e69422734709e6d2d73710aa496610bdb2882454',
                   '4a09ad011dfc4bc1897482b96ed69f882efe0c0962833a280f58a3cbd5adeaee'),
           179069:('1747e22ecbf0e27b7058fa1ec40e83c411423b66a51175a9ca1abe660b95f2f7',
                   'd1a20da69e22b11aab38e963db3cdd096a253568766c98d8a1e6324c3dd7111e')}
ACCEPTED = {179061:('cc20ff1b808fabd158d5545e2f270fd0e7d30e12eefc1d2e5ced1b1859a0a8b3',
                    '13b8bb5d1007afc8bd8c7142b2835b25250f94c83a45847669577fa175b1a540'),
            179069:('85c2503e5399451197c589f5b154eba914200f029de8a846812bbb9fe89aaf71',
                    'bb90967e1bfdf360d0da57dec21548ac8782d7e7652552a28da4a28076f99372')}
NATIVE = {'torch','numpy','PIL','transformers','safetensors','torchvision','sfora'}


def require(value, message):
    if not value:
        raise ValueError(message)


def file_fact(fact):
    require(type(fact) is dict and fact.keys() == {'path','sha256'} and
            type(fact['path']) is str and type(fact['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}',fact['sha256']), 'exact actual FILE required')
    path = Path(fact['path'])
    require(path.is_absolute() and str(path) == fact['path'], 'canonical absolute FILE required')
    return path


def file_bytes(fact, guards, *, keep=False):
    path = file_fact(fact)
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    before,digest,raw = path.stat(),hashlib.sha256(),None
    with path.open('rb') as stream:
        if keep:
            raw = stream.read(64*1024**2+1)
            require(len(raw) <= 64*1024**2, 'source/metadata FILE exceeds64MiB')
            digest.update(raw)
        else:
            while block := stream.read(1024**2):
                digest.update(block)
        after = os.fstat(stream.fileno())
    require(before == after == path.stat() and digest.hexdigest() == fact['sha256'], 'current FILE bytes differ')
    require(guards.setdefault(str(path),fact['sha256']) == fact['sha256'], 'conflicting FILE authority')
    return raw if keep else path


def read_json(fact, guards):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(file_bytes(fact,guards,keep=True),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'nonfinite JSON'))


def check_unit(unit):
    require(type(unit) is dict and unit.keys() == {'receipt','log','unit','invocation_id',
            'service_seconds','native_peak_rss_kib','both_locks_held'} and unit['both_locks_held'] is True,
            'complete original UNIT required')
    for key in ('receipt','log'):
        file_fact(unit[key])
    require(type(unit['unit']) is str and re.fullmatch('[A-Za-z0-9_.@-]+',unit['unit']) and
            type(unit['invocation_id']) is str and re.fullmatch('[0-9a-f]{32}',unit['invocation_id']) and
            all(type(unit[k]) in (int,float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds','native_peak_rss_kib')), 'actual UNIT scalars required')


def closure(fact, names, guards):
    require(type(fact) is dict and fact.keys() == {'root','execution_sha256','code'} and
            type(fact['root']) is str and type(fact['code']) is dict and fact['code'].keys() == names,
            'exact original CODE required')
    root = Path(fact['root'])
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical CODE root required')
    require(read_json({'path':str(root/'execution.json'),'sha256':fact['execution_sha256']},guards) == fact['code'],
            'actual execution closure differs')
    for name,digest in fact['code'].items():
        require(Path(name).name == name, 'flat source filename required')
        file_bytes({'path':str(root/name),'sha256':digest},guards)
    return root


def load_module(name, fact, guards):
    path = file_fact(fact)
    raw = file_bytes(fact,guards,keep=True)
    require(name not in sys.modules, 'fresh source module namespace required')
    spec = importlib.util.spec_from_file_location(name,path)
    require(spec is not None and spec.loader is not None, 'original module spec required')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw,str(path),'exec',dont_inherit=True),vars(module))
    return module


def validate_launch(launch, args):
    require(type(launch) is dict and launch.keys() == LAUNCH_KEYS and
            launch['schema'] == 'connected-gradient-decomposition-launch-v1' and
            launch['execution_sha256'] == args.execution_sha256 and launch['output'] == str(args.output) and
            launch['both_locks_held'] is True and launch['resource_policy'] == POLICY and
            all(type(v) is int for v in launch['resource_policy'].values()), 'exact prospective launch/policy required')
    for name in ('python','bootstrap','workspace_source'):
        file_fact(launch[name])
    require(Path(launch['workspace_source']['path']).name == 'observe_connected_control_batch_execution.py',
            'explicit original workspace FILE interface required')
    require(type(launch['training']) is dict and launch['training'].keys() == {'root','execution_sha256','code'} and
            launch['training']['execution_sha256'] == TRAIN_EXECUTION and launch['training']['code'] == TRAIN_CODE,
            'unchanged accepted source-v6 required')
    require(type(launch['evaluator']) is dict and launch['evaluator'].keys() == {'root','execution_sha256','code'} and
            type(launch['evaluator']['code']) is dict and launch['evaluator']['code'].keys() ==
            {'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'}, 'exact evaluator helper CODE required')
    require(type(launch['candidates']) is list and len(launch['candidates']) == 2, 'two accepted candidates required')
    for seed,endpoint in zip(SEEDS,launch['candidates'],strict=True):
        require(type(endpoint) is dict and endpoint.keys() == {'seed','launch','terminal','checkpoint','terminal_state_sha256'} and
                type(endpoint['seed']) is int and endpoint['seed'] == seed, 'exact seed-ordered candidate required')
        for key in ('launch','checkpoint'):
            file_fact(endpoint[key])
        check_unit(endpoint['terminal'])
        require((endpoint['checkpoint']['sha256'],endpoint['terminal_state_sha256']) == ACCEPTED[seed],
                'original accepted step128 checkpoint/typed identity required')


def prepare(args, started):
    require(not any(name.split('.')[0] in NATIVE for name in sys.modules), 'native import preceded admission')
    guards = {}
    authority = {'path':str(args.authority),'sha256':args.authority_sha256}
    launch = read_json(authority,guards)
    validate_launch(launch,args)
    output = args.output
    require(HERE.resolve() == HERE and output.is_absolute() and output.parent.resolve() == output.parent and
            output.parent.is_dir() and not output.exists() and not output.is_symlink(), 'exclusive canonical output required')
    code = read_json({'path':str(HERE/'execution.json'),'sha256':args.execution_sha256},guards)
    require(type(code) is dict and code.keys() == FILES, 'prospective own exact2 required')
    for name,digest in code.items():
        file_bytes({'path':str(HERE/name),'sha256':digest},guards)
    require(launch['python']['path'] == str(Path(sys.executable).resolve()), 'prospective interpreter path differs')
    file_bytes(launch['python'],guards)
    training = closure(launch['training'],set(TRAIN_CODE),guards)
    evaluation = closure(launch['evaluator'],{'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'},guards)
    require(len({HERE,training,evaluation}) == 3 and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (HERE,training,evaluation)), 'separate source/output lifetimes required')
    connected = load_module('_gradient_connected_original',
        {'path':str(training/'train_siglip2_connected_mlp.py'),'sha256':TRAIN_CODE['train_siglip2_connected_mlp.py']},guards)
    evaluator = load_module('_gradient_original_endpoint_helper',
        {'path':str(evaluation/'evaluate_siglip2_connected_mlp.py'),
         'sha256':launch['evaluator']['code']['evaluate_siglip2_connected_mlp.py']},guards)
    require(evaluator.TRAINING == launch['training'], 'evaluator original training source role differs')
    workspace = load_module('_gradient_original_workspace_owner',launch['workspace_source'],guards)
    helper_context = {'trainer':connected,'guards':guards,'code':launch['evaluator']['code']}
    historical = SimpleNamespace(execution_sha256=TRAIN_EXECUTION,authority=Path(launch['bootstrap']['path']),
        authority_sha256=launch['bootstrap']['sha256'],output=output,phase='train',arm='control',seed=SEEDS[0])
    # These are unchanged genuine helper APIs, including their existing inverses.
    bootstrap,bootstrap_guard = evaluator.load_bootstrap_authority(helper_context,historical)
    context,reader = bootstrap()
    require(helper_context['training_context'] is context, 'original bootstrap ownership differs')
    context['started'] = context['fit_context']['unit_started'] = started
    for path,digest in guards.items():
        require(context['guards'].setdefault(path,digest) == digest, 'prospective/original guard conflict')
    # Each unchanged helper retains its captured guard dictionary, with equal facts.
    for path,digest in context['guards'].items():
        require(guards.setdefault(path,digest) == digest, 'original/prospective guard conflict')
    admit = reader
    records = {}
    for endpoint in launch['candidates']:
        seed = endpoint['seed']
        selected = connected.read_json(endpoint['launch'],context['guards'])
        connected.check_launch(selected,SimpleNamespace(execution_sha256=TRAIN_EXECUTION,phase='train',arm='candidate',seed=seed))
        require(connected.method(selected) == connected.method(context['connected_launch']) and
                selected['selected_cpu'] == context['connected_launch']['selected_cpu'], 'candidate original method/CPU differs')
        if seed != SEEDS[0]:
            for arm in connected.ARMS:
                admit(selected['selected_mechanics'][arm],'mechanics',arm,seed)
        control = admit(selected['fresh_control'],'train','control',seed)
        if seed == SEEDS[0]:
            require(control['authority'] == launch['bootstrap'], 'control061 bootstrap FILE role differs')
        candidate = admit(endpoint['terminal'],'train','candidate',seed)
        evaluator.check_paired_initialization(connected,control,candidate)
        require(candidate['launch'] == selected and candidate['authority'] == endpoint['launch'] and
                candidate['result']['checkpoint'] == endpoint['checkpoint'] and
                candidate['result']['terminal_state_sha256'] == endpoint['terminal_state_sha256'], 'accepted complete candidate endpoint differs')
        for arm,record in (('control',control),('candidate',candidate)):
            mechanics = context['connected_terminals'][f'mechanics:{seed}:{arm}']
            require(record['result']['identity'] == mechanics['result']['identity'] and
                    [connected.diagnostic(row) for row in record['result']['steps'][:17]] ==
                    [connected.diagnostic(row) for row in mechanics['result']['steps']], 'original first17 mechanics replay differs')
        records[seed] = candidate
    prior = context['original_cpu_record']['invocation']
    require(launch['python'] == {'path':prior['python'],'sha256':prior['python_sha256']} and
            sys.version == prior['python_version'], 'original complete interpreter differs')
    for seed in SEEDS:
        qualification = connected.select_initializer(context['original_cpu_record'],seed)
        require((qualification['checkpoint']['sha256'],qualification['terminal_state_sha256']) == INITIAL[seed],
                'original accepted seed-specific initializer differs')
    for path,digest in context['guards'].items():
        require(guards.setdefault(path,digest) == digest, 'complete merged source guard differs')
    def guard():
        bootstrap_guard(helper_context)
    guard()
    return launch,code,connected,evaluator,workspace,context,records,guard


def accumulate_gradients(torch, accumulator, gradients, members):
    require(len(accumulator) == len(gradients) == len(members) == 6, 'six-member gradient inventory required')
    for total, gradient, parameter in zip(accumulator,gradients,members,strict=True):
        require(gradient is not None and gradient.shape == parameter.shape and
                gradient.dtype == torch.float32 and gradient.device.type == parameter.device.type and
                torch.isfinite(gradient).all().item(), 'missing/nonfinite/wrong-shape gradient')
        total.add_(gradient.detach())


def first_batch(trainer, state, expected):
    batch = state['schedules'][str(state['seed'])][0].tolist()
    require(state['seed'] in SEEDS and len(batch) == len(set(batch)) == 64 and batch == expected,
            'original unique first B64 schedule differs')
    full = trainer.ranking_membership(state['ranking_bank'],batch)
    require(full['valid'] == (63 if state['seed'] == SEEDS[0] else 64) and
            trainer.loss_denominators(full['valid']) == (128,2*full['valid']), 'original full-B64 denominators differ')
    return batch,full


def measure_terms(torch, trainer, context, state, raw, anchors, full_valid, members, vectors, released):
    mse = rank = term = gradients = None
    try:
        mse,rank,membership = trainer.loss_terms(context,state,raw,anchors,full_valid)
        for label,term in (('regression',mse),('ranking',rank),('total',mse+rank)):
            gradients = torch.autograd.grad(term,members,retain_graph=label != 'total',allow_unused=True)
            released.extend(weakref.ref(g) for g in gradients if g is not None)
            accumulate_gradients(torch,vectors[label],gradients,members)
            gradients = None
        return {'mse':float(mse.detach()),'rank':float(rank.detach()),'membership':membership}
    finally:
        mse = rank = term = gradients = None


def vector_summary(torch, vectors, *, chunk=CHUNK):
    require(vectors.keys() == set(TERMS) and all(len(v) == 6 for v in vectors.values()), 'exact gradient terms required')
    require(type(chunk) is int and 0 < chunk <= CHUNK, 'bounded FP64 chunk required')
    scalars = []
    for regression, ranking, total in zip(*(vectors[t] for t in TERMS),strict=True):
        require(regression.shape == ranking.shape == total.shape and
                all(v.dtype == torch.float32 for v in (regression,ranking,total)), 'FP32 gradient vectors required')
        row = {k:[] for k in ('regression_sq','ranking_sq','total_sq','dot')}
        max_error = 0.
        for offset in range(0,total.numel(),chunk):
            r,k,t = (v.reshape(-1)[offset:offset+chunk].double() for v in (regression,ranking,total))
            require(all(torch.isfinite(v).all().item() for v in (r,k,t)), 'nonfinite FP64 gradient chunk')
            require(torch.allclose(t,r+k,rtol=RTOL,atol=ATOL), 'total=MSE+rank numerical correspondence failed')
            max_error = max(max_error,float((t-r-k).abs().max().item()))
            for key,value in (('regression_sq',(r*r).sum()),('ranking_sq',(k*k).sum()),
                              ('total_sq',(t*t).sum()),('dot',(r*k).sum())):
                row[key].append(float(value.item()))
            del r,k,t,value
        scalars.append({**{key:math.fsum(values) for key,values in row.items()},'max_error':max_error})
    groups = {name:(i,) for i,name in enumerate(NAMES)}
    groups.update(fc1=(0,1),fc2=(2,3),allfour=(0,1,2,3),AC=(4,5),all6=tuple(range(6)))
    result = {}
    for name,indices in groups.items():
        norm = {t:math.sqrt(math.fsum(scalars[i][t+'_sq'] for i in indices)) for t in TERMS}
        dot = math.fsum(scalars[i]['dot'] for i in indices)
        cosine = dot/(norm['regression']*norm['ranking']) if norm['regression'] and norm['ranking'] else None
        require(all(math.isfinite(n) for n in (*norm.values(),dot)) and
                (cosine is None or math.isfinite(cosine) and abs(cosine) <= 1+1e-12), 'nonfinite vector summary')
        result[name] = {**{t+'_norm':n for t,n in norm.items()},'regression_ranking_dot':dot,
                        'regression_ranking_cosine':cosine,'zero_norm_cosine_undefined':cosine is None,
                        'total_equals_sum':True,'max_abs_correspondence_error':max(scalars[i]['max_error'] for i in indices)}
    return result


def finish_cleanup(error, callbacks):
    failures = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as failure:
            traceback.print_exception(failure,file=sys.stderr)
            failures.append(failure.with_traceback(None))
    if error is not None:
        for failure in failures:
            error.add_note('cleanup also failed: '+repr(failure))
        raise error
    if failures:
        for failure in failures[1:]:
            failures[0].add_note('cleanup also failed: '+repr(failure))
        raise failures[0]


def build_state(torch, context, connected, seed, step, record):
    if step == 0:
        return connected.fresh(context,'candidate',seed,'cuda')
    require(step == 128 and record['seed'] == seed and record['arm'] == 'candidate', 'accepted state role differs')
    qualification = connected.select_initializer(context['original_cpu_record'],seed)
    cpu,identity = connected.load_initializer(context,qualification)
    context['trainer'].release(context,cpu)
    cpu = identity = None
    fact,digest = record['result']['checkpoint'],record['result']['terminal_state_sha256']
    path = connected.bound_file(context['guards'],fact['path'],fact['sha256'])
    disk = pages = None
    try:
        disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        require(disk.keys() == connected.PAYLOAD_KEYS and json.loads(json.dumps(disk['identity'],allow_nan=False)) ==
                record['result']['identity'], 'complete original typed checkpoint/receipt identity differs')
        identity = copy.deepcopy(disk['identity'])
        connected.check_payload(context,disk,identity,128)
        with path.open('rb') as stream:
            pages = context['legacy']['original'].CheckpointPages(stream)
            require(connected.fingerprint(context,disk,consumed=pages.consume) == digest,
                    'complete original typed accepted payload differs')
    finally:
        disk = pages = None
        gc.collect()
        connected.mapping_absent(path)
    return connected.restore(context,path,fact['sha256'],digest,identity,128)


def state_digest(context, connected, state):
    return connected.fingerprint(context,connected.payload(context,state,state['identity']))


def state_refs(context, state):
    trainer = context['trainer']
    references = trainer.tensor_weakrefs(context,{k:state[k] for k in trainer.STATIC_KEYS+('A','C')})
    references += trainer.tensor_weakrefs(context,state['optimizer_object'].state_dict())
    references += [weakref.ref(state[k]) for k in ('model','processor_object','head_object','optimizer_object','scaler_object')]
    references += [weakref.ref(p) for p in (*state['model'].parameters(),*state['model'].buffers(),
                                           *state['head_object'].parameters(),*state['head_object'].buffers())]
    return references


def measure_view(torch, context, connected, state, batch, full, members, joint, view):
    from torch.nn import functional as F
    trainer = context['trainer']
    vectors = {term:[torch.zeros_like(p) for p in members] for term in TERMS}
    refs = [weakref.ref(g) for values in vectors.values() for g in values]
    temporary,rows,images,released,error = {},[],[],[],None
    result = None
    try:
        for offset in range(0,64,16):
            check_resources(torch,context)
            anchors = batch[offset:offset+16]
            temporary['cpu_pixels'],facts = context['witness'].pixels_for(trainer,context,state,state['processor_object'],anchors,view)
            images.extend(facts)
            with torch.autocast('cuda',enabled=False):
                temporary['pixels'] = temporary['cpu_pixels'].to('cuda')
                temporary['features'] = F.normalize(state['model'](pixel_values=temporary['pixels']).pooler_output.float(),dim=1)
                require(temporary['features'].shape == (16,1152) and temporary['features'].dtype == torch.float32 and
                        temporary['features'].requires_grad, 'original live FP32 micro16 features required')
                temporary['raw'] = context['connected'].raw_features(temporary['features'],state['head_object'],state['A'],
                    state['means'],state['C'],state['mu_train'],context['legacy']['quadratic'],trainer.helper_guard(context))
                rows.append(measure_terms(torch,trainer,context,state,temporary['raw'],anchors,full['valid'],members,vectors,released))
            released.extend(weakref.ref(value) for value in temporary.values())
            temporary.clear()
            gc.collect()
            require(all(ref() is None for ref in released), 'micro graph/gradient/pixels lifetime survived release')
            released.clear()
        torch.cuda.synchronize()
        result = {'view':view,'mse':math.fsum(r['mse'] for r in rows),'rank':math.fsum(r['rank'] for r in rows),
                  'micro_membership':[{'anchors':batch[i*16:(i+1)*16],**r['membership']} for i,r in enumerate(rows)],
                  'images':images,'vectors':vector_summary(torch,vectors)}
        result['loss'] = result['mse']+result['rank']
        for term in TERMS:
            for accumulator,gradient in zip(joint[term],vectors[term],strict=True):
                accumulator.add_(gradient)
        accumulator = gradient = None
    except BaseException as failure:
        traceback.print_exception(failure,file=sys.stderr)
        error = failure.with_traceback(None)
    finally:
        accumulator = gradient = None
        temporary.clear()
        vectors.clear()
        gc.collect()
        finish_cleanup(error,[torch.cuda.synchronize,
            lambda:require(all(ref() is None for ref in (*refs,*released)), 'view graph/vector lifetime survived release')])
    return result


def measure_state(torch, context, connected, state, expected):
    trainer = context['trainer']
    batch,full = first_batch(trainer,state,expected)
    parameters = dict(state['model'].named_parameters())
    members = tuple(parameters[n] for n in MLP)+tuple(state[n] for n in ('A','C'))
    require(len(parameters) == 448 and sum(p.numel()*p.element_size() for p in members) == VECTOR_BYTES and
            all(p.dtype == torch.float32 and p.requires_grad and p.grad is None for p in members) and
            state['views']['canonical'].shape == (6355,1152) and state['teachers']['T'].shape == (6355,128) and
            len(state['ranking_bank']['target']) == 6355, 'original all448/six-member/complete S gallery required')
    # Six accumulators + one returned set; FP64 scratch is bounded independently.
    extra_bound = 7*VECTOR_BYTES+16*CHUNK*8
    require(7 <= POLICY['gradient_sets_max'] and extra_bound <= POLICY['extra_tensor_bytes'], 'diagnostic tensor bound exceeded')
    joint = {term:[torch.zeros_like(p) for p in members] for term in TERMS}
    refs = [weakref.ref(g) for values in joint.values() for g in values]
    result,error = None,None
    try:
        views = [measure_view(torch,context,connected,state,batch,full,members,joint,view) for view in VIEWS]
        result = {'batch':batch,'full_membership':full,'gallery':'inherited canonical S only',
            'regression_denominator_rows':128,'ranking_denominator':2*full['valid'],
            'views':views,'both_views':vector_summary(torch,joint),
            'mse':math.fsum(v['mse'] for v in views),'rank':math.fsum(v['rank'] for v in views),
            'gradient_sets_max':7,'extra_tensor_bytes_bound':extra_bound,
            'correspondence_tolerances':{'rtol':RTOL,'atol':ATOL,'numerical_only':True}}
        result['loss'] = result['mse']+result['rank']
    except BaseException as failure:
        traceback.print_exception(failure,file=sys.stderr)
        error = failure.with_traceback(None)
    finally:
        joint.clear()
        parameters = members = None
        gc.collect()
        finish_cleanup(error,[lambda:require(all(ref() is None for ref in refs), 'joint vector lifetime survived release')])
    return result


def state_measurement(torch, context, connected, seed, step, record, guard):
    state,result,before,refs,error = None,None,None,[],None
    try:
        guard()
        state = build_state(torch,context,connected,seed,step,record)
        refs = state_refs(context,state)
        connected.integrity(context,state,state['identity'])
        before = state_digest(context,connected,state)
        qualification = connected.select_initializer(context['original_cpu_record'],seed)
        result = measure_state(torch,context,connected,state,qualification['scope_schedule'][0])
        result.update(seed=seed,step=step,complete_payload_sha256=before)
    except BaseException as failure:
        traceback.print_exception(failure,file=sys.stderr)
        error = failure.with_traceback(None)
    finally:
        def integrity():
            if state is not None:
                connected.integrity(context,state,state['identity'])
                require(before is None or state_digest(context,connected,state) == before, 'complete model/state/RNG bytes changed')
        def release():
            nonlocal state
            if state is not None:
                try:
                    connected.release(context,state)
                finally:
                    state = None
                    context['initial'].clear()
                    gc.collect()
        finish_cleanup(error,[integrity,release,torch.cuda.synchronize,
            lambda:require(all(ref() is None for ref in refs), 'state/model/processor lifetime survived release'),
            lambda:context['trainer'].require_no_training(context),guard])
    return result


def check_resources(torch, context):
    cgroup = context['legacy']['source_driver'].cgroup_memory()
    context['old'].zero_events(cgroup)
    unit = Path(cgroup['path']).name.removesuffix('.service')
    context['legacy']['selected']['genuine']['reference'].admit_cgroup(cgroup,unit)
    require(time.perf_counter()-context['started'] < POLICY['seconds'] and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= POLICY['host_bytes'] and
            (not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated() < POLICY['cuda_allocated_bytes_exclusive']),
            'whole proposed diagnostic resource envelope exceeded')
    return cgroup


def bind_workspace(evaluator, workspace, fact, context, torch):
    guard = evaluator.source_live_guard(workspace,fact['sha256'],context['guards'],
        names={'capture_workspace_owner','authenticated','require'})
    guard()
    roles = {'guards':context['guards'],'source_cpu':context['legacy']['selected']['source_cpu'],
             'warm':context['legacy']['warm_record']}
    callback = workspace.capture_workspace_owner(torch,roles)
    require(type(callback) is FunctionType and callback.__globals__ is vars(workspace) and
            callback.__module__ == workspace.__name__ and callback.__name__ == 'dispose' and
            callback.__defaults__ is None and callback.__kwdefaults__ is None, 'original workspace callback binding differs')
    code,closure = callback.__code__,callback.__closure__
    used = False
    def dispose():
        nonlocal used
        require(not used, 'workspace disposal already attempted')
        used = True
        guard()
        require(callback.__code__ is code and callback.__closure__ is closure and
                callback.__globals__ is vars(workspace) and callback.__defaults__ is None and
                callback.__kwdefaults__ is None, 'captured workspace callback changed')
        try:
            callback()
        finally:
            guard()
    return dispose


def final_resources(torch, context, before, rng, flags, dispose, resources):
    error = None
    try:
        if dispose is not None and torch.cuda.is_initialized():
            dispose()
    except BaseException as failure:
        traceback.print_exception(failure,file=sys.stderr)
        error = failure.with_traceback(None)
    def cuda_and_rng():
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            require(torch.cuda.memory_allocated() == 0, 'final CUDA allocated bytes must be zero')
        require(context['legacy']['source_driver'].numerical_flags() == flags and
                torch.equal(rng[0],torch.random.get_rng_state()) and
                all(torch.equal(a,b) for a,b in zip(rng[1],torch.cuda.get_rng_state_all(),strict=True)),
                'final original numerical flags/RNG changed')
    def cgroup():
        after = check_resources(torch,context)
        require(after['path'] == before['path'], 'whole enclosing cgroup changed')
        resources.update(cgroup_before=before,cgroup_after=after,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),final_cuda_allocated_bytes=torch.cuda.memory_allocated())
    finish_cleanup(error,[cuda_and_rng,cgroup])


def run(args):
    started = time.perf_counter()
    require(not sys.flags.optimize and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and
            re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')), 'original fresh CUDA0 enclosing UNIT required')
    launch,code,connected,evaluator,workspace,context,records,guard = prepare(args,started)
    source,trainer,legacy = context['legacy']['source_driver'],context['trainer'],context['legacy']
    packages = source.package_origins(legacy['prior'])
    require(packages == legacy['selected']['packages'], 'original package origins differ')
    legacy['prior']['packages'] = packages
    import torch
    require(not torch.cuda.is_initialized() and torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            'genuine original admission must precede CUDA')
    flags = context['original_cpu_record']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original FP32 numerical flags differ')
    context['flags'] = legacy['flags'] = flags
    packing = legacy['selected']['launch']['helpers']['packing']
    legacy['packing'] = legacy['genuine'].load_helper('_quadratic_packing',packing['path'],packing['sha256'],context['guards'])
    witness = context['connected_launch']['witness']
    path = Path(witness['root'])/'connected_residual_readout.py'
    context['connected'] = connected.load_authenticated('_gradient_live_connected_readout',path,
        witness['files']['connected_residual_readout.py'],context['guards'])
    context['connected_function'] = (context['connected'].raw_features,context['connected'].raw_features.__code__)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one original genuine CUDA encoder required')
    before = check_resources(torch,context)
    rng = (torch.random.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])
    dispose = bind_workspace(evaluator,workspace,launch['workspace_source'],context,torch)
    results,resources,error = [],{},None
    try:
        for seed in SEEDS:
            for step in (0,128):
                results.append(state_measurement(torch,context,connected,seed,step,records[seed],guard))
                check_resources(torch,context)
    except BaseException as failure:
        traceback.print_exception(failure,file=sys.stderr)
        error = failure.with_traceback(None)
    finally:
        def restore_rng():
            torch.random.set_rng_state(rng[0])
            torch.cuda.set_rng_state_all(rng[1])
        def exit_integrity():
            trainer.require_no_training(context)
            trainer.exit_rehash(context)  # Complete genuine fresh uncached exit reader.
            trainer.audit_origin_diagnostics(context,context['nearest'].native_source_api(context),
                admission=legacy['original'].FlatAdmission(),require_exact=True)
            for path,digest in tuple(context['guards'].items()):
                file_bytes({'path':path,'sha256':digest},{})
            require(read_json({'path':str(HERE/'execution.json'),'sha256':args.execution_sha256},{}) == code,
                    'own exact2 uncached final closure differs')
            guard()
        finish_cleanup(error,[restore_rng,
            lambda:final_resources(torch,context,before,rng,flags,dispose,resources),exit_integrity,
            lambda:check_resources(torch,context)])
    require([(r['seed'],r['step']) for r in results] == [(s,k) for s in SEEDS for k in (0,128)] and
            all([v['view'] for v in r['views']] == list(VIEWS) for r in results), 'all four states/both views required')
    receipt = {'schema':'connected-gradient-decomposition-v1','pass':True,'engineering_only':True,
        'model_fit_qualified':False,'cost_qualified':False,'quality_read':False,'state_reuse_eligible':False,
        'causal_intervention':False,'stale_gallery_attribution':False,'scientific_decision':'HOLD_NEW_TRAINING_ARM',
        'training_updates':0,'shadow_updates':0,'new_gallery_banks':0,'states':results,
        'native_result_requires_parent_terminal':True,'terminal_exit_and_both_locks_require_parent_receipt':True,
        'scope':'original CONTROL1008 TRAIN6355','source':copy.deepcopy(context['source']),
        'authority':{'path':str(args.authority),'sha256':args.authority_sha256},'launch':launch,'code':code,
        'execution_sha256':args.execution_sha256,'input_guards':dict(context['guards']),
        'resource_policy':POLICY,'numerical_flags':flags,'whole_seconds':time.perf_counter()-started,
        'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'exit_rehash_pass':True,'all_graphs_vectors_models_released':True,
        'workspace_ownership_limit':'original source-pinned admitted builtin binding; no C-function-pointer authentication',
        'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),
            'python_sha256':launch['python']['sha256'],'python_version':sys.version,'optimize':sys.flags.optimize,
            'invocation_id':os.environ['INVOCATION_ID'],'pid':os.getpid(),
            'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']},
        **resources}
    args.output.mkdir()
    with (args.output/'receipt.json').open('x') as stream:
        json.dump(receipt,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')
    check_resources(torch,context)
    print(json.dumps({'pass':True,'engineering_only':True,'whole_seconds':time.perf_counter()-started}),flush=True)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    result.add_argument('--execution-sha256',required=True)
    result.add_argument('--authority',type=Path,required=True)
    result.add_argument('--authority-sha256',required=True)
    result.add_argument('--output',type=Path,required=True)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
