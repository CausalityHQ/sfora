#!/usr/bin/env python3
"""Stdlib source/admission falsifiers. No native qualification is claimed.

Run: python3 -B scripts/test_siglip2_connected_probe.py --source-only
The parent runs this once under timeout120 and address-space1GiB after repairs.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import CodeType, FunctionType, ModuleType, SimpleNamespace
from functools import lru_cache
import gc
import weakref
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'train_siglip2_connected_probe.py'


def rejects(call, text):
    try:
        call()
    except (ValueError, TypeError, KeyError) as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError('accepted mutant: ' + text)


def function(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_contract(d):
    tree = ast.parse(DRIVER.read_text())
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name.split('.')[0] for n in top_imports for a in n.names] + [
        n.module.split('.')[0] for n in top_imports if isinstance(n, ast.ImportFrom)]
    assert not set(names) & (d.NATIVE | {'qualify_actual_objective_encoder_gradients',
                                        'train_siglip2_identity_diversity'})
    assert d.FILES == {'train_siglip2_connected_probe.py', 'test_siglip2_connected_probe.py'}
    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/actual-objective-gradient-v2-freeze'
    admitted = json.loads((evidence/'authority.json').read_text())
    assert d.WITNESS_FILES == admitted['files'].keys() and len(d.WITNESS_FILES) == 7
    assert 'test_connected_encoder_gradients.py' not in d.WITNESS_FILES
    for name,digest in admitted['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == digest
    assert d.PROBE == ('head.probe',) and d.PROBE_SHAPES == [[1,1,1152]]
    assert len(d.HISTORICAL_MLP) == 4 and sum(__import__('math').prod(s) for s in d.HISTORICAL_MLP_SHAPES) == 9921872
    assert d.parameter_roles('control') == (['A', 'C'], [[128,160],[128,1152]], 167936)
    assert d.parameter_roles('candidate') == (['A','C',*d.PROBE], [[128,160],[128,1152],*d.PROBE_SHAPES], 169088)
    rejects(lambda: d.parameter_roles('CONTROL2016'), 'arm')
    assert d.policy('cpu')['seconds'] == 600
    assert d.policy('mechanics')['seconds'] == 1200
    assert d.policy('train')['seconds'] == 3000
    assert all(d.policy(p)['host_bytes'] == 8*1024**3 and d.policy(p)['swap_bytes'] == 0
               for p in ('cpu','mechanics','train'))
    assert d.ADAM['lr'] == 1e-4 and d.RECIPE['encoder_lr'] == 1e-5
    assert d.ADAM['weight_decay'] == .05 and d.ADAM['betas'] == (.9,.999)
    assert all(d.ADAM[k] is False for k in ('amsgrad','maximize','foreach','capturable','differentiable','fused'))
    original = ast.parse((HERE/'train_siglip2_identity_diversity.py').read_text())
    static = next(n.value for n in original.body if isinstance(n,ast.Assign) and
                  any(isinstance(t,ast.Name) and t.id == 'STATIC_KEYS' for t in n.targets))
    assert d.STATIC_KEYS == ast.literal_eval(static)
    assert d.PAYLOAD_KEYS == {'schema','identity','source','A','C',*d.STATIC_KEYS,'optimizer','scaler',
                             'counter','cpu_rng','cuda_rng','numerical_flags','base_vision','encoder','vision_sha256'}


def initializer_selection(d):
    records = [{'arm': a, 'seed': s, 'identity': {'device': dev, 'scope': {'arm': a}},
                'checkpoint': {'path': '/fixture/'+a+str(s), 'sha256': 'a'*64},
                'terminal_state_sha256': 'b'*64}
               for a in ('control','candidate') for s in d.SEEDS for dev in ('cpu','cuda')]
    record = {'qualifications': records}
    for seed in d.SEEDS:
        chosen = d.select_initializer(record, seed)
        assert chosen['arm'] == chosen['identity']['scope']['arm'] == 'control'
        assert chosen['identity']['device'] == 'cpu' and chosen['seed'] == seed
        rejects(lambda: d.select_initializer({'qualifications': records+[chosen]}, seed), 'unique')
        rejects(lambda: d.select_initializer({'qualifications': [r for r in records if r is not chosen]}, seed), 'unique')


def authority_falsifiers(d):
    file = {'path':'/fixture/authority.json', 'sha256':'a'*64}
    unit = {'receipt':file, 'log':file, 'unit':'fixture', 'invocation_id':'a'*32,
            'service_seconds':1., 'native_peak_rss_kib':1, 'both_locks_held':True}
    args = SimpleNamespace(execution_sha256='b'*64, phase='cpu', arm='control', seed=d.SEEDS[0])
    launch = {'schema':d.AUTHORITY_SCHEMA, 'execution_sha256':args.execution_sha256,
              'phase':'cpu', 'arm':'control', 'seed':args.seed, 'recipe':copy.deepcopy(d.RECIPE),
              'resource_policy':d.policy('cpu'), 'both_locks_held':True,
              'original_cpu':{'authority':file, 'terminal':unit},
              'historical_mlp_actual_gradient':{'authority':file, 'terminal':unit},
              'witness':{'root':'/fixture/witness', 'files':{n:'a'*64 for n in d.WITNESS_FILES}},
              'selected_cpu':None, 'selected_mechanics':None, 'fresh_control':None}
    d.check_launch(launch, args)
    for key, value in [('schema','wrong'), ('seed',True), ('arm','candidate'),
                       ('resource_policy', {**d.policy('cpu'),'seconds':300}),
                       ('resource_policy', {**d.policy('cpu'),'seconds':500}),
                       ('resource_policy', {**d.policy('cpu'),'seconds':601}),
                       ('both_locks_held',False), ('selected_cpu',unit)]:
        rejects(lambda k=key,v=value: d.check_launch({**launch,k:v},args), 'launch')
    for key, value in [('classes', {'control':1008,'candidate':2016}), ('microbatch',32),
                       ('clip',2.), ('updates',17), ('encoder_lr',1e-4)]:
        changed = copy.deepcopy(launch)
        changed['recipe'][key] = value
        rejects(lambda: d.check_launch(changed,args), 'launch')
    for value in [{**file,'extra':1}, {**file,'path':'relative'}, {**file,'sha256':'future'}]:
        rejects(lambda: d.file_fact(value), 'FILE')
    for value in [{**unit,'both_locks_held':False}, {**unit,'invocation_id':'bad'},
                  {**unit,'service_seconds':float('nan')}]:
        rejects(lambda: d.check_unit(value), 'UNIT')
    rejects(lambda: d.strict_json('{"x":1,"x":2}'), 'duplicate')
    rejects(lambda: d.strict_json('{"x":NaN}'), 'nonfinite')
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'bytes'
        path.write_bytes(b'original')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        d.bound_file({}, path, digest)
        path.write_bytes(b'mutant')
        rejects(lambda: d.bound_file({},path,digest), 'bytes')


def initializer_runtime_falsifiers(d):
    """Real historical bodies and private bindings; tensor/I/O seams stay stdlib."""
    from contextlib import nullcontext
    import builtins

    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    path = evidence/'export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py'
    sha = '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8'
    original = d.load_authenticated('_initializer_original_test',path,sha,{})
    spec = importlib.util.spec_from_file_location('_initializer_historical_tests',HERE/'test_siglip2_identity_diversity.py')
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    original_members = dict(vars(original))
    names = ('integrity','restore','canonical_initial_witness')
    original_codes = {n:getattr(original,n).__code__ for n in names}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            f = historical.canonical_fixture()
            context,state,ident = f.context,f.state,f.ident
            clock,peak,events = [0.],[0],[]
            context.update(trainer=original,guards={str(path):sha},started=123.,
                           args=SimpleNamespace(phase='cpu'),connected_args=SimpleNamespace(phase='cpu'))
            context['flags'] = {'threads':1}
            head = state['head_object']
            head._forward_hooks = head._forward_pre_hooks = head._backward_hooks = {}
            state.update(target=historical.CanonicalTensor([0,0]),original_rows=historical.CanonicalTensor([1,2]),
                         target_list=[0,0],row_list=[1,2])
            state['ranking_bank'] = original.ranking_bank(state['target_list'],state['row_list'])
            optimizer = state['optimizer_object']
            optimizer.param_groups = [{'params':[state['A'],state['C']]}]
            optimizer.defaults = {'fixture':True}
            ident.update(parameter_names=['A','C'],optimizer_defaults=dict(optimizer.defaults),
                         arm='control',seed=d.SEEDS[0],device='cpu')
            helper = SimpleNamespace(check_weight=lambda *a:events.append('weight'))
            context['legacy']['quadratic'] = SimpleNamespace(_check_base=lambda *a:events.append('base'))
            context['nearest'] = SimpleNamespace(require_no_model=lambda *a:events.append('no_model'))
            membership = root/'membership'
            membership.write_text('0::/fixture\n')
            cgroup = root/'cgroup'/'fixture'
            cgroup.mkdir(parents=True)
            values = {'memory.max':str(8*1024**3),'memory.current':'1','memory.peak':'1',
                      'memory.swap.current':'0','memory.swap.peak':'0','memory.swap.max':'0',
                      'memory.events':'oom 0\noom_kill 0\nmax 0'}
            for n,v in values.items(): (cgroup/n).write_text(v)
            node = function(ast.parse((HERE/'qualify_siglip2_substrate_cpu.py').read_bytes()),'cgroup_memory')
            cgroup_ns = {'require':d.require,'POLICY':d.policy('cpu'),
                         'Path':lambda p:membership if p == '/proc/self/cgroup' else root/'cgroup'}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<real cgroup seam>','exec'),cgroup_ns)
            context['legacy']['source_driver'] = SimpleNamespace(numerical_flags=lambda:dict(context['flags']),
                                                                 cgroup_memory=cgroup_ns['cgroup_memory'])
            f.torch.cuda = SimpleNamespace(max_memory_allocated=lambda:peak[0],set_rng_state_all=lambda *a:None)
            f.torch.random = SimpleNamespace(set_rng_state=lambda *a:events.append('rng'))
            checkpoint = root/'checkpoint'
            checkpoint.write_bytes(b'stdlib restore seam')
            disk = {'optimizer':{'state':{},'param_groups':[]},'scaler':{},
                    'cpu_rng':SimpleNamespace(clone=lambda:None)}
            f.torch.load = lambda *a,**kw:disk
            optimizer.load_state_dict = lambda *a:events.append('optimizer')
            state['scaler_object'].load_state_dict = lambda *a:events.append('scaler')
            context['legacy']['original'] = SimpleNamespace(CheckpointPages=lambda stream:SimpleNamespace(consume=lambda *a:None))
            seams = {k:f.ns[k] for k in ('fingerprint','payload','canonical_copy_check','tensor_weakrefs','fullfeature_raw_features')}
            seams.update(time=SimpleNamespace(perf_counter=lambda:clock[0]),
                         helper_guard=lambda *a:events.append('helper') or helper,
                         own_A=lambda *a,**kw:events.append('owner'),
                         check_payload=lambda *a:events.append('payload'),
                         require_no_training=lambda *a:events.append('no_training'),
                         timed=lambda *a:nullcontext(),bound_file=lambda *a:checkpoint,
                         fresh=lambda *a,**kw:state,identity=lambda *a:ident)

            def run(api, name):
                # Patch only the private test namespace's tensor/I/O boundaries.
                # All integrity predicates and restore/canonical bodies are real.
                namespace = api.integrity.__globals__
                assert all(getattr(api,n).__globals__ is namespace for n in names)
                assert namespace['integrity'] is api.integrity
                with patch.dict(namespace,seams), patch.dict(sys.modules,{'torch':f.torch}):
                    if name == 'restore':
                        with patch.dict(namespace,fingerprint=lambda *a,**kw:'digest',payload=lambda *a:disk):
                            assert api.restore(context,checkpoint,'file','digest',ident,0) is state
                    elif name == 'canonical_initial_witness':
                        result = api.canonical_initial_witness(context,state,ident)
                        assert result['live_unchanged'] and result['temporary_references_released']
                    else:
                        api.integrity(context,state,ident)

            # Clone exact original code solely to attach stdlib test seams; the
            # live authenticated historical module is never patched or rebound.
            old_ns = dict(vars(original))
            for n in names:
                fn = getattr(original,n)
                old_ns[n] = FunctionType(fn.__code__,old_ns,n,fn.__defaults__)
                old_ns[n].__kwdefaults__ = copy.deepcopy(fn.__kwdefaults__)
            old = SimpleNamespace(**{n:old_ns[n] for n in names})
            for phase,ceiling in (('cpu',600),('mechanics',1200),('train',3000)):
                context['connected_args'].phase = phase
                for elapsed in (501.,ceiling-.001):
                    clock[0] = context['started']+elapsed
                    for n in names:
                        rejects(lambda n=n:run(old,n),'whole-unit deadline')
                api = d._initializer_runtime(context)
                for elapsed in (501.,ceiling-.001):
                    clock[0] = context['started']+elapsed
                    for n in names: run(api,n)
                for elapsed in (ceiling,ceiling+.001):
                    clock[0] = context['started']+elapsed
                    for n in names:
                        rejects(lambda n=n:run(api,n),'whole-unit deadline')
                assert context['started'] == 123. and context['args'].phase == 'cpu'
            clock[0] = context['started']+499.999
            for n in names: run(old,n)
            clock[0] = context['started']+500.
            for n in names: rejects(lambda n=n:run(old,n),'whole-unit deadline')
            clock[0] = context['started']+501.
            assert original.policy('cpu')['seconds'] == 500
            assert {'helper','owner','weight','base','no_model','payload','no_training','optimizer','scaler','rng'} <= set(events)

            # Retained integrity branches and helper boundaries still reject.
            for key,value,text in [('ranking_bank',{},'ranking bank'),('target_list',[],'ranking bank'),
                                   ('row_list',[],'ranking bank')]:
                with patch.dict(state,{key:value}): rejects(lambda:run(api,'integrity'),text)
            for owner,key,value in [(state['A'],'grad',object()),(state['C'],'grad',object()),
                                    (state['C'],'requires_grad',False),(head.weight,'requires_grad',True),
                                    (head,'training',False),(head,'_forward_hooks',{'hook':1}),
                                    (head,'_forward_pre_hooks',{'hook':1}),(head,'_backward_hooks',{'hook':1}),
                                    (optimizer,'defaults',{}),(optimizer,'param_groups',[])]:
                with patch.object(owner,key,value): rejects(lambda:run(api,'integrity'),'roles/hooks')
            with patch.dict(state,device='cuda'):
                rejects(lambda:run(api,'integrity'),'roles/hooks')
            with patch.object(optimizer,'param_groups',[{'params':[state['C'],state['A']]}]):
                rejects(lambda:run(api,'integrity'),'roles/hooks')
            optimizer.state = {state[n]:{k:SimpleNamespace(device=SimpleNamespace(type='cuda'))
                               for k in ('exp_avg','exp_avg_sq')} for n in ('A','C')}
            with patch.dict(state,counter=1): rejects(lambda:run(api,'integrity'),'moments')
            with patch.object(context['legacy']['source_driver'],'numerical_flags',lambda:{}):
                rejects(lambda:run(api,'integrity'),'numerical flags')
            for key,bad in [('memory.max','1'),('memory.peak',str(8*1024**3+1)),
                            ('memory.swap.current','1'),('memory.swap.peak','1'),('memory.swap.max','1'),
                            ('memory.events','oom 1\noom_kill 0\nmax 0')]:
                (cgroup/key).write_text(bad)
                rejects(lambda:run(api,'integrity'),'cgroup memory')
                (cgroup/key).write_text(values[key])
            for name in ('helper_guard','own_A','check_payload'):
                def fail(*a,**kw): raise ValueError('retained '+name)
                with patch.dict(seams,{name:fail}): rejects(lambda:run(api,'integrity'),'retained '+name)
            with patch.object(context['nearest'],'require_no_model',lambda *a:d.require(False,'retained model')):
                rejects(lambda:run(api,'integrity'),'retained model')
            # CUDA ceiling executes without a CUDA import or a peak reset.
            with patch.dict(state,device='cuda'), patch.object(state['A'].device,'type','cuda'):
                for t in head.parameters(): t.device.type = 'cuda'
                peak[0] = 9_999_999_999
                run(api,'integrity')
                peak[0] = 10_000_000_000
                rejects(lambda:run(api,'integrity'),'CUDA peak')
                for t in head.parameters(): t.device.type = 'cpu'

            # Factory rejects changed live functions, origins, guards and bytes.
            @contextmanager
            def changed_function(fn, key, value):
                saved = getattr(fn,key)
                setattr(fn,key,value)
                try:
                    yield
                finally:
                    setattr(fn,key,saved)

            for name in names:
                fn = getattr(original,name)
                with patch.object(original,name,lambda *a:None):
                    rejects(lambda:d._initializer_runtime(context),'live function')
                with changed_function(fn,'__code__',(lambda *a:None).__code__):
                    rejects(lambda:d._initializer_runtime(context),'live function')
                foreign = FunctionType(fn.__code__,dict(vars(original)),name,fn.__defaults__)
                with patch.object(original,name,foreign):
                    rejects(lambda:d._initializer_runtime(context),'live function')
                with changed_function(fn,'__defaults__',(None,)):
                    rejects(lambda:d._initializer_runtime(context),'live function')
            for defaults in ({'compare_native':True},{'compare_native':0},{'compare_native':0.},
                             {},{'compare_native':False,'extra':False},None):
                with changed_function(original.canonical_initial_witness,'__kwdefaults__',defaults):
                    rejects(lambda:d._initializer_runtime(context),'live function')
            with patch.object(original.__spec__,'origin','/foreign.py'):
                rejects(lambda:d._initializer_runtime(context),'origin')
            with patch.dict(sys.modules,{original.__name__:ModuleType(original.__name__)}):
                rejects(lambda:d._initializer_runtime(context),'origin')
            with patch.dict(context['guards'],{str(path):'0'*64}):
                rejects(lambda:d._initializer_runtime(context),'source guard')
            with patch.dict(vars(original),_connected_policy=d.policy):
                rejects(lambda:d._initializer_runtime(context),'private namespace')
            foreign = root/'foreign.py'
            foreign.write_bytes(path.read_bytes()+b'\n')
            with patch.object(original,'__file__',str(foreign)), patch.object(original.__spec__,'origin',str(foreign)), \
                    patch.dict(context['guards'],{str(foreign):sha}):
                rejects(lambda:d._initializer_runtime(context),'bytes')
            with patch.object(Path,'read_bytes',lambda p:path.read_text().encode()+b'\n'):
                rejects(lambda:d._initializer_runtime(context),'changed before compilation')
            assert vars(original).keys() == original_members.keys()
            assert all(vars(original)[n] is value for n,value in original_members.items())
            assert all(getattr(original,n).__code__ is original_codes[n] for n in names)
            assert vars(original)['__builtins__'] is vars(builtins)
    finally:
        sys.modules.pop(original.__name__,None)
    print('PASS initializer runtime: real restore/direct/canonical RED500 -> phase GREEN; unchanged predicates/origins')


def runtime_envelope_inverse(raw):
    """Invert only the pinned adapter block and exact prospective edits."""
    raw = probe_inverse(raw)
    start,end = raw.index(b'def _initializer_runtime('),raw.index(b'def load_initializer(')
    assert hashlib.sha256(raw[start:end]).hexdigest() == \
           '8a793e1517da90f9fcacd507b3fbf7df2b5025db07df89b6f9b2bf8fcfa025e0', 'initializer adapter bytes differ'
    raw = raw[:start]+raw[end:]
    replacements = (
        (b"diagnostic(mechanics['result']['steps'][step-1])",
         b"diagnostic(mechanics['steps'][step-1])"),
        (b'selects NEW mechanics1200', b'selects NEW mechanics300'),
        (b'Gates: ownCPU600/mechanics1200/TRAIN3000', b'Gates: ownCPU600/mechanics300/TRAIN600'),
        (b"'seconds':600 if phase == 'cpu' else 1200 if phase == 'mechanics' else 3000",
         b"'seconds':300 if phase == 'mechanics' else 600"),
        (b"(reader,control,control_unit,policy('train')['seconds'],guards)",
         b'(reader,control,control_unit,600,guards)'),
        (b"    runtime = _initializer_runtime(context)\n    state = runtime.restore(context,path,fact['sha256'],digest,ident,0)",
         b"    state = trainer.restore(context,path,fact['sha256'],digest,ident,0)"),
        (b'    runtime.integrity(context,state,ident)', b'    trainer.integrity(context,state,ident)'),
        (b'    canonical = runtime.canonical_initial_witness(context,state,ident)',
         b'    canonical = trainer.canonical_initial_witness(context,state,ident)'),
    )
    for new,old in replacements:
        assert raw.count(new) == 1, 'exact prospective runtime envelope differs'
        raw = raw.replace(new,old)
    assert hashlib.sha256(raw).hexdigest() == \
           '55935d5a7e7299d1a11f14617cd5ef07f4232afd32148abf942c74edbc636e99', 'runtime inverse bytes differ'
    assert hashlib.sha256(ast.dump(ast.parse(raw),include_attributes=False).encode()).hexdigest() == \
           '4b67697a6d907b240e8dfe4527f1ca24b8f382ec91f0e7a1c61dd060fb9b6c31', 'runtime inverse AST differs'
    return raw


def historical_inverse_contract(d):
    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
    assert hashlib.sha256((evidence/'connected-mlp-cpu-v3-freeze/test_siglip2_connected_mlp.py').read_bytes()).hexdigest() == \
           '35daffe103bc52fe3fbbc4af33a0348494b9477ce49052db7f5d0cf5be083cc1'
    path = HERE/'test_siglip2_identity_diversity.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
           '617c25bd705d4ee15182c9f85c2ac6bba4f2a12c44934f1e3e73ff66cfa868c8'
    spec = importlib.util.spec_from_file_location('_historical_inverse_tests',path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    # Keep the existing source/test inverse chain and its historical hashes.
    old.HistoricalGradientOracle().test_exact_source_and_test_inverse_keeps_all_historical_hashes()
    old.CanonicalSourceContract().test_exact_pre_edit_source_bytes_ast_and_all_historical_tests_preserved()
    restored = old.scan_batch_inverse(old.PATH.read_bytes())
    assert hashlib.sha256(restored).hexdigest() == \
           '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8'
    assert hashlib.sha256(ast.dump(ast.parse(restored),include_attributes=False).encode()).hexdigest() == \
           'f36a0e472318190c06808527d2ce97fd9ebd20d9e9a94232763343f48b22486c'
    raw = DRIVER.read_bytes()
    for before,after in ((b'context[\'started\']',b'context[\'reset_clock\']'),
                         (b'<= 1.50',b'<= 1.51'),
                         (b"_connected_policy(context['connected_args'].phase)",b"_connected_policy('train')")):
        assert before in raw
        try:
            runtime_envelope_inverse(raw.replace(before,after))
        except AssertionError:
            pass
        else:
            raise AssertionError('runtime inverse accepted unrelated mutation')
    print('PASS exact runtime/CPU/batch/historical source and historical test inverses')


def cpu_envelope_inverse(raw):
    replacements = (
        (b'one NEW CPU600 UNIT', b'one NEW CPU300 UNIT'),
        (b'Gates: ownCPU600/mechanics300/TRAIN600', b'Gates: ownCPU300/mechanics300/TRAIN600'),
        (b"'seconds':300 if phase == 'mechanics' else 600", b"'seconds':600 if phase == 'train' else 300"),
    )
    for new, old in replacements:
        assert raw.count(new) == 1, 'exact prospective CPU envelope differs'
        raw = raw.replace(new, old)
    return raw


def actual_gradient_scan_falsifier(d):
    """Real fresh reads at the extracted callsite; no native/speed qualification."""
    import os
    import threading
    from collections import Counter

    raw = cpu_envelope_inverse(runtime_envelope_inverse(DRIVER.read_bytes()))
    old = (b"    for path,digest in record['input_guards'].items():\n"
           b"        bound_file(context['guards'],path,digest)\n")
    new = b"    batch_bound_files(context['guards'],record['input_guards'].items())\n"
    assert raw.count(old) + raw.count(new) == 1
    original = raw.replace(new, old)
    # Exact inverse pins ALL production bytes/AST, including helpers and the
    # surrounding receipt, gradient, witness-closure and terminal predicates.
    assert hashlib.sha256(original).hexdigest() == \
           'cd68f9b109ca48dfc488a248b66c6e2f4a1e58e884aa3c598f4325d4c912edb2'
    baseline, candidate = ast.parse(original), ast.parse(raw)
    assert hashlib.sha256(ast.dump(baseline,include_attributes=False).encode()).hexdigest() == \
           '1578ada3f1ab342af6f83ca060acdb24af9755808d762d43b1f8d35bfed2a1fb'
    body = function(baseline,'admit_actual_gradient').body
    index = next(i for i,n in enumerate(body) if isinstance(n,ast.For) and
                 ast.unparse(n.iter) == "record['input_guards'].items()")
    serial, batched = body[index], function(candidate,'admit_actual_gradient').body[index]
    namespace = dict(vars(d))
    helpers = [function(candidate,n) for n in ('require','file_fact','bound_file','batch_bound_files')]
    exec(compile(ast.Module(body=helpers,type_ignores=[]),str(DRIVER),'exec'),namespace)
    real_bound, real_open = namespace['bound_file'], Path.open

    def run(statement, items, owner, overlap=False):
        lock, barrier = threading.Lock(), threading.Barrier(2,timeout=1.)
        calls, reads, threads, edges = [], [], set(), ['before']
        active = peak = 0
        overlapped = False

        def bound(guards, path, digest):
            with lock:
                calls.append((str(path),digest))
                threads.add(threading.current_thread())
            return real_bound(guards,path,digest)

        @contextmanager
        def opened(path, *args, **kwargs):
            nonlocal active, peak, overlapped
            with real_open(path,*args,**kwargs) as stream:
                assert args == ('rb',) and not kwargs
                with lock:
                    reads.append(str(path))
                    edges.append('read')
                    active += 1
                    peak = max(peak,active)
                    waits = overlap and len(reads) <= 2
                try:
                    if waits:
                        try:
                            barrier.wait()
                            overlapped = True
                        except threading.BrokenBarrierError:
                            pass  # Serial RED finishes after a bounded wait.
                    yield stream
                finally:
                    with lock:
                        active -= 1

        code = compile(ast.Module(body=[statement],type_ignores=[]),str(DRIVER),'exec')
        # A receipt dict has unique keys; this items facade also exercises the
        # helper's required per-occurrence contract for repeated descriptors.
        inputs = SimpleNamespace(items=lambda:iter(items))
        error = None
        with patch.dict(namespace,bound_file=bound), patch.object(Path,'open',opened):
            try:
                exec(code,{**namespace,'record':{'input_guards':inputs},'context':{'guards':owner}})
                edges.append('after')
            except ValueError as exc:
                error = str(exc)
        assert active == 0 and peak <= 4
        assert all(t is threading.current_thread() or not t.is_alive() for t in threads), 'workers not joined'
        assert edges[0] == 'before' and edges[1:] == ['read']*len(reads) + ([] if error else ['after'])
        return SimpleNamespace(error=error,calls=calls,reads=reads,overlapped=overlapped,peak=peak)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        files = [root/f'member{i}' for i in range(5)]
        for i,path in enumerate(files):
            path.write_bytes(b'member %d\n' % i)
        items = [(str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
        repeated = items + [items[0],items[2]]
        initial = dict([items[3],items[0]])
        serial_owner, owner = dict(initial), dict(initial)
        red = run(serial,repeated,serial_owner)
        green = run(batched,repeated,owner,overlap=True)
        assert red.error is green.error is None
        assert red.calls == repeated and red.reads == [p for p,_ in repeated]
        assert red.peak == 1 and not red.overlapped
        assert green.overlapped and 2 <= green.peak <= 4, 'witness scan still serial'
        assert Counter(green.calls) == Counter(repeated)
        assert Counter(green.reads) == Counter(p for p,_ in repeated), 'fresh occurrence missing'
        assert list(owner.items()) == list(serial_owner.items()) == list({**initial,**dict(items)}.items())

        def failure(entries, expected, guards=initial, overlap=False):
            owner = dict(guards)
            result = run(batched,entries,owner,overlap)
            assert result.error and expected in result.error, (expected,result.error)
            assert list(owner.items()) == list(guards.items()), 'failed scan published owner guards'
            return result

        # Same size and restored mtime cannot turn an existing guard into a cache.
        victim = files[0]
        stamp, content = victim.stat(), victim.read_bytes()
        victim.write_bytes(b'changed!\n')
        os.utime(victim,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
        assert (victim.stat().st_size,victim.stat().st_mtime_ns) == (stamp.st_size,stamp.st_mtime_ns)
        mutated = failure(repeated,'current FILE bytes',overlap=True)
        assert mutated.overlapped and Counter(mutated.reads) == Counter(p for p,_ in repeated)
        victim.write_bytes(content)

        link = root/'link'
        link.symlink_to(files[0])
        for path in (str(link),str(root),str(root/'missing'),str(root/'..'/root.name/files[0].name)):
            failure([items[1],(path,items[0][1])],'canonical regular FILE')
        failure([items[1],(items[0][0],'0'*64)],'current FILE bytes')
        conflict = {items[0][0]:'0'*64}
        conflicted = failure(repeated,'conflicting FILE authority',conflict)
        assert Counter(conflicted.reads) == Counter(p for p,_ in repeated)
        failure([items[0],(items[0][0],'0'*64)],'current FILE bytes')
        for malformed in (('relative',items[0][1]),(items[0][0],'A'*64),(items[0][0],None)):
            rejected = failure([items[1],malformed],'exact FILE')
            assert rejected.calls == rejected.reads == []

        # Accepted differences: descriptor prevalidation precedes byte errors;
        # owner conflicts follow all byte results; serial publishes its prefix.
        faults = [(items[0][0],'0'*64),('relative',items[1][1])]
        assert 'current FILE bytes' in run(serial,faults,{}).error
        assert failure(faults,'exact FILE',{}).calls == []
        faults = [items[0],(items[1][0],'0'*64)]
        assert 'conflicting FILE authority' in run(serial,faults,dict(conflict)).error
        failure(faults,'current FILE bytes',conflict)
        serial_prefix = {}
        assert run(serial,[items[0],(items[1][0],'0'*64)],serial_prefix).error
        assert list(serial_prefix.items()) == [items[0]]
        failure([items[0],(items[1][0],'0'*64)],'current FILE bytes',{})
    print('PASS witness scan: inverse source/AST, fresh occurrences, ordered owner, bounded/joined workers, negatives')


def lifetime_restore_math(d):
    tree = ast.parse(DRIVER.read_text())
    text = lambda name: ast.unparse(function(tree,name))
    restore = text('restore')
    assert restore.index('require_no_training') < restore.index('torch.load')
    assert "'cpu' if k == 'step'" in restore and 'pages.copy' in restore
    assert restore.index('construct_encoder') < restore.index('bind_optimizer')
    assert restore.index('bind_optimizer') < restore.index('set_rng_state')
    assert 'mapping_absent' in restore and 'del disk' in restore
    assert 'frozen=' not in restore and 'frozen_cache' not in DRIVER.read_text()
    assert 'canonical_initial_witness' in text('load_initializer')
    initializer = text('load_initializer')
    assert 'runtime = _initializer_runtime(context)' in initializer
    assert 'trainer = context[\'trainer\']' in initializer
    for name in ('restore','integrity','canonical_initial_witness'):
        assert initializer.count('runtime.'+name+'(') == 1
        assert 'trainer.'+name+'(' not in initializer
    assert 'trainer.check_payload' in text('check_payload')
    assert 'copy.deepcopy' in text('load_initializer') and "disk['identity']" in text('load_initializer')
    assert '.eval()' in text('construct_encoder')
    assert text('load_inference').index('admit_bundle') < text('load_inference').index('import torch')
    assert 'trainer.exit_rehash' in text('run')
    assert 'loss_terms' in text('update') and 'pixels_for' in text('update')
    assert 'connected.raw_features' in text('update')
    update = text('update')
    assert update.index('unscale_') < update.index('clip_grad_norm_') < update.index('scaler.step')
    assert 'enabled=False' in update and 'requires_grad' in update
    assert 'ranking' in update and 'autograd.grad' in update
    assert 'max_memory_allocated' in text('resource_check')
    assert 'reset_peak' not in DRIVER.read_text()
    assert '447' in text('encoder_facts') and 'nonpersistent' in text('encoder_facts')
    assert 'vision_sha256' in text('inference_outputs') and 'fingerprint' in text('inference_outputs')
    assert 'construct_encoder' in text('load_inference') and "disk['encoder']" in text('load_inference')
    assert 'apply_overlay' in text('construct_encoder') and 'load_vision' in text('construct_encoder')
    assert 'deny_training_dependencies' in text('qualify_bundle')
    assert 'train_siglip2_identity_diversity' not in text('load_inference')
    assert 'qualify_actual_objective' not in text('load_inference')
    assert 'range(1, 9)' in text('arm_run') and 'range(9, 18)' in text('arm_run')
    assert 'diagnostic' in text('arm_run') and 'payload' in text('arm_run')
    old = ast.parse((HERE/'train_siglip2_identity_diversity.py').read_text())
    # Native inference math is the original value-equivalent formula, including
    # the detached residual (serving has no gradients). Training uses the witness.
    assert ast.dump(function(tree,'fullfeature_raw_features'),include_attributes=False) == \
           ast.dump(function(old,'fullfeature_raw_features'),include_attributes=False)
    witness = ast.parse((HERE/'qualify_actual_objective_encoder_gradients.py').read_text())
    assert ast.dump(function(tree,'_processor_cache'),include_attributes=False) == \
           ast.dump(function(witness,'_processor_cache'),include_attributes=False)


class Tensor:
    """Metadata-only stand-in: no numeric/native training is simulated."""
    def __init__(self, shape=(), value=0., dtype='float32', device='cpu'):
        self.shape,self.value,self.dtype = tuple(shape),value,dtype
        self.device = SimpleNamespace(type=device)
        self.requires_grad,self.grad_fn,self.grad,self.is_leaf = False,None,None,True
    def detach(self): return Tensor(self.shape,self.value,self.dtype,self.device.type)
    def to(self, device='cpu', *, copy=False, **kwargs):
        assert copy, 'same-device copies must own storage'
        return Tensor(self.shape,self.value,self.dtype,device)
    def copy_(self, value): self.value = value.value; return self
    def requires_grad_(self, value): self.requires_grad = value; return self
    def __float__(self): return float(self.value)


@contextmanager
def tensor_seam():
    old = {name:sys.modules.get(name) for name in ('torch','torch.nn','torch.nn.functional')}
    torch = ModuleType('torch')
    torch.Tensor,torch.float32 = Tensor,'float32'
    torch.isfinite = lambda v: SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:v.value != float('inf')))
    @contextmanager
    def no_grad(): yield
    torch.no_grad = no_grad
    nn = ModuleType('torch.nn')
    nn.LayerNorm = type('LayerNorm',(),{})
    torch.nn = nn
    nn.functional = ModuleType('torch.nn.functional')
    sys.modules.update({'torch':torch,'torch.nn':nn,'torch.nn.functional':nn.functional})
    try:
        yield torch
    finally:
        for name,module in old.items():
            if module is None: sys.modules.pop(name,None)
            else: sys.modules[name] = module


def overlay_optimizer_seams(d):
    with tensor_seam():
        params = {n:Tensor(shape) for n,shape in zip(d.PROBE,d.PROBE_SHAPES,strict=True)}
        params.update({f'frozen.{i}':Tensor((1,)) for i in range(447)})
        model = SimpleNamespace(named_parameters=lambda:params.items(),state_dict=lambda:params)
        overlay = {n:Tensor(shape,i+1.) for i,(n,shape) in enumerate(zip(d.PROBE,d.PROBE_SHAPES,strict=True))}
        identities = {n:id(p) for n,p in params.items()}
        d.apply_overlay(model,overlay)
        assert {n:id(p) for n,p in params.items()} == identities
        assert [params[n].value for n in d.PROBE] == [1.]
        assert all(p.value == 0 for n,p in params.items() if n not in d.PROBE)
        rejects(lambda:d.apply_overlay(model,{**overlay,'head.mlp.fc2.bias':Tensor((1152,))}), 'names')
        rejects(lambda:d.apply_overlay(model,{n:v for n,v in overlay.items() if n != d.PROBE[0]}), 'names')
        for bad in (Tensor((1152,)),Tensor(d.PROBE_SHAPES[0],dtype='float16'),Tensor(d.PROBE_SHAPES[0],float('inf'))):
            rejects(lambda:d.apply_overlay(model,{**overlay,d.PROBE[0]:bad}), 'shape/dtype/role')
        overlay[d.PROBE[0]].requires_grad = True
        rejects(lambda:d.apply_overlay(model,overlay), 'shape/dtype/role')
        overlay[d.PROBE[0]].requires_grad = False
        for arm in d.ARMS:
            names,shapes,_ = d.parameter_roles(arm)
            groups = [{**d.ADAM,'params':[0,1]}]
            if arm == 'candidate': groups.append({**d.ADAM,'lr':1e-5,'params':[2]})
            identity = {'arm':arm,'parameter_names':names,'parameter_shapes':shapes,
                        'optimizer_defaults':copy.deepcopy(d.ADAM),
                        'optimizer_groups':[{k:v for k,v in g.items() if k != 'params'} for g in groups]}
            saved = {'optimizer':{'state':{i:{'step':Tensor((),8.),'exp_avg':Tensor(s,1.),'exp_avg_sq':Tensor(s,2.)}
                                            for i,s in enumerate(shapes)},'param_groups':groups}}
            d.check_optimizer(saved,identity,8)
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['params'].reverse()
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'groups')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['param_groups'][0]['maximize'] = 0
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'groups')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'].pop(len(names)-1)
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'ownership')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][0]['step'].device.type = 'cuda'
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'CPU step')
            wrong = copy.deepcopy(saved)
            wrong['optimizer']['state'][len(names)-1]['exp_avg'].shape = (1,)
            rejects(lambda:d.check_optimizer(wrong,identity,8), 'moment')
            changed_identity = copy.deepcopy(identity)
            changed_identity['optimizer_groups'][0]['lr'] = 1e-5
            rejects(lambda:d.check_optimizer(saved,changed_identity,8), 'defaults/rates')
        source = ast.parse((HERE/'train_siglip2_substrate_adaptation.py').read_text())
        pages = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name == 'CheckpointPages')
        method = next(n for n in pages.body if isinstance(n,ast.FunctionDef) and n.name == 'copy')
        ns = {}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<original pages copy>','exec'),ns)
        consumed = []
        owner = SimpleNamespace(consume=lambda v:consumed.append(v))
        owner.copy = lambda value,device='cpu':ns['copy'](owner,value,device)
        step,moment = Tensor((),8.),Tensor((3,),1.)
        copied = d.owned_copy({'step':step,'moments':[moment]},owner,'cpu')
        assert copied['step'] is not step and copied['moments'][0] is not moment and consumed == [step,moment]
        copied['step'].value = 9.
        assert step.value == 8.


def owned_loader_admission(d):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        code = {}
        for name in d.FILES | d.SERVING_FILES | {'joint_relational_compaction.py'}:
            raw = DRIVER.read_bytes() if name == DRIVER.name else b'# owned source\n'
            (root/name).write_bytes(raw)
            code[name] = hashlib.sha256(raw).hexdigest()
        files = {}
        for name in ('vision.pt','endpoint.pt','processor.json'):
            (root/name).write_bytes(b'owned artifact')
            files[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
        vendor = root/'vendor.py'
        vendor.write_bytes(b'installed source')
        env = {'packages':{name:{'root':str(root/'packages'/name)} for name in d.NATIVE-{'sfora'}},
               'files':{str(vendor):hashlib.sha256(vendor.read_bytes()).hexdigest()},
               'native_files':{str(vendor):hashlib.sha256(vendor.read_bytes()).hexdigest()},'vision_constructor':str(vendor)}
        manifest = {'schema':d.BUNDLE_SCHEMA,'code':code,'files':files,'endpoint_state_sha256':'a'*64,
                    'environment':env,'encoder_identity':{},'base_vision_sha256':'b'*64,'vision_sha256':'c'*64,
                    'scope':{'arm':'control','manifest_sha256':d.SCOPE_SHA256,'arm_sha256':d.CONTROL_SHA256}}
        def write(value):
            p = root/'bundle.json'
            p.write_text(json.dumps(value))
            return hashlib.sha256(p.read_bytes()).hexdigest()
        digest = write(manifest)
        actual,_ = d.admit_bundle(root,digest)
        assert actual == manifest
        (root/'endpoint.pt').write_bytes(b'updated overlay substitution')
        rejects(lambda:d.admit_bundle(root,digest), 'bytes')
        # Actual public loader must reject before importing native packages.
        before = set(sys.modules)
        rejects(lambda:d.load_inference(root,digest,'cpu'), 'bytes')
        assert not {n.split('.')[0] for n in set(sys.modules)-before} & d.NATIVE
        (root/'endpoint.pt').write_bytes(b'owned artifact')
        bad = copy.deepcopy(manifest)
        bad['code']['train_siglip2_identity_diversity.py'] = 'a'*64
        rejects(lambda:d.admit_bundle(root,write(bad)), 'schema/closure')
        bad = copy.deepcopy(manifest)
        bad['environment']['files']['/historical/teacher.npy'] = 'a'*64
        rejects(lambda:d.admit_bundle(root,write(bad)), 'TRAIN data')
        import os
        os.link(root/'vision.pt',root/'alias')
        rejects(lambda:d.admit_bundle(root,write(manifest)), 'single-link')


def restore_seam(d):
    """Execute the actual restore ordering/CPU-step ownership seam with fake I/O.

    Payload/native validators are exercised separately. This check addresses
    the original failure mode: Adam preserves a CPU step alias by reference.
    """
    originals = {n:getattr(d,n) for n in ('bound_file','check_payload','fingerprint','construct_encoder',
        'bind_optimizer','encoder_facts','integrity','payload','mapping_absent')}
    events,consumed = [],[]
    with tensor_seam() as torch, tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'saved.pt'
        path.write_bytes(b'seam')
        disk = {'optimizer':{'param_groups':[],'state':{0:{'step':Tensor((),8.),
                   'exp_avg':Tensor((3,),2.),'exp_avg_sq':Tensor((3,),3.)}}},'scaler':{},'config':{},'buffers':{},
                'provenance':{'encoder':{'source_proof':{'runtime':{}}}},'encoder':{},
                'vision_sha256':'updated','cpu_rng':Tensor((3,),4.,'uint8'),'cuda_rng':[]}
        weak_step = weakref.ref(disk['optimizer']['state'][0]['step'])
        torch.load = lambda *a,**kw:disk
        torch.random = SimpleNamespace(set_rng_state=lambda value:events.append(('rng',value)))
        torch.cuda = SimpleNamespace(set_rng_state_all=lambda values:None)
        class Pages:
            def __init__(self, stream): events.append('pages')
            def consume(self, value): consumed.append(value)
            def copy(self, value, device='cpu'):
                result = value.to(device,copy=True)
                self.consume(value)
                return result
        class Optimizer:
            def load_state_dict(self, value): self.state = value['state']; events.append('optimizer_loaded')
        class Model:
            def named_parameters(self): return iter([(d.PROBE[0],Tensor((1,)))])
            def to(self,device): events.append('transfer'); return self
            def eval(self): return self
        state = {}
        def model_free(context, arm, seed, device, initial):
            events.append('static_copied')
            return state
        trainer = SimpleNamespace(require_no_training=lambda context:events.append('release_first'),fresh=model_free,
                                  own_A=lambda *a,**kw:events.append('owners'))
        context = {'trainer':trainer,'guards':{},'legacy':{'original':SimpleNamespace(CheckpointPages=Pages),
            'source_driver':None,'prior':{'entry':{'input':{'preprocessor':{'path':'processor.json'}}}},'selected':{'packages':{}}}}
        identity = {'seed':179061,'device':'cpu','arm':'candidate','base_vision':{},'encoder_identity':{'runtime':{}}}
        def construct(*a,**kw): events.append('construct'); return Model(),Model(),None,{},{}
        def bind(state):
            events.append('bind')
            state['optimizer_object'] = Optimizer()
            state['scaler_object'] = SimpleNamespace(load_state_dict=lambda value:events.append('scaler'))
        d.bound_file = lambda guards,path,digest:Path(path)
        d.check_payload = lambda *a:events.append('validate')
        d.fingerprint = lambda *a,**kw:'digest'
        d.construct_encoder,d.bind_optimizer = construct,bind
        d.encoder_facts = lambda *a,**kw:{'vision_sha256':'updated'}
        d.integrity = lambda *a:events.append('integrity')
        d.payload = lambda *a:{}
        d.mapping_absent = lambda path:events.append('mapping_absent')
        try:
            restored = d.restore(context,path,'file','digest',identity,8)
            steps = restored['optimizer_object'].state
            assert steps[0]['step'] is not disk['optimizer']['state'][0]['step']
            steps[0]['step'].value = 9.
            assert disk['optimizer']['state'][0]['step'].value == 8.
            assert events.index('release_first') < events.index('validate') < events.index('construct')
            assert events.index('transfer') < events.index('bind') < events.index('optimizer_loaded')
            rng = next(v for v in events if isinstance(v,tuple) and v[0] == 'rng')
            assert rng[1] is not disk['cpu_rng']
            assert events.index('optimizer_loaded') < events.index(rng) < events.index('mapping_absent') < events.index('integrity')
            del disk
            consumed.clear()
            gc.collect()
            assert weak_step() is None, 'CPU step must not pin the serialized owner'
        finally:
            for name,value in originals.items(): setattr(d,name,value)


def current_encoder_seam(d):
    """Execute real 448/frozen447/runtime/role and public pre-forward falsifiers."""
    def fp(value):
        if isinstance(value,Tensor): return repr((value.shape,value.dtype,value.value))
        if isinstance(value,dict): return repr([(k,fp(v)) for k,v in sorted(value.items())])
        return repr(value)
    class Module:
        def __init__(self):
            self.training = False
            self._forward_hooks,self._forward_pre_hooks,self._backward_hooks = {},{},{}
            self._non_persistent_buffers_set = set()
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.embeddings = Module()
            self.embeddings._non_persistent_buffers_set = {'position_ids'}
            self.config = SimpleNamespace(_attn_implementation='sdpa',layer_norm_eps=1e-6)
            self.config.to_dict = lambda:{'attention':self.config._attn_implementation}
            self.params = {n:Tensor(shape,1.) for n,shape in zip(d.PROBE,d.PROBE_SHAPES,strict=True)}
            self.params.update({f'frozen.{i}':Tensor((1,)) for i in range(447)})
            self.position = Tensor((1,256),0.,'int64')
        def named_parameters(self): return self.params.items()
        def state_dict(self): return self.params
        def named_modules(self): return [('',self),('embeddings',self.embeddings)]
        def named_buffers(self): return [('embeddings.position_ids',self.position)]
    class Processor:
        backend = 'torchvision'
        def to_json_string(self): return '{}'
        def __call__(self,**kw): raise AssertionError('mutant reached serving math')
    class Head(Module):
        def __init__(self): super().__init__(); self.training = True; self.weight = Tensor((128,1152),1.)
        def parameters(self): return [self.weight]
        def modules(self): return [self]
        def state_dict(self): return {'weight':self.weight}
    original_cache = d._processor_cache
    original_probe_source = d.probe_source
    d.probe_source = lambda *a:None
    with tensor_seam(),tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        original,source = SimpleNamespace(fingerprint=fp),SimpleNamespace(module_origin=lambda cls,packages:{'class':cls.__name__},
                                                                            numerical_flags=lambda:{})
        guards = {}
        for i,module in enumerate((original,source)):
            path = root/f'helper{i}.py'
            path.write_text('# admitted helper')
            module.__file__ = str(path)
            guards[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        model,processor = Model(),Processor()
        identity = {'inventory':{n:list(p.shape) for n,p in model.named_parameters()},
                    'nonpersistent':{'embeddings':['position_ids']},'buffers_sha256':fp(dict(model.named_buffers())),
                    'frozen_sha256':fp({n:p for n,p in model.named_parameters() if n not in d.PROBE}),
                    'runtime':d.model_structure(model,source,{})}
        state = {'model':model,'encoder_identity':identity,'device':'cpu','arm':'candidate',
                 'processor_object':processor,'processor':{'config':{},'backend':'torchvision','origin':{'class':'Processor'}},
                 'processor_cache':'cache','guards':guards}
        d._processor_cache = lambda *a,**kw:'cache'
        try:
            facts = d.encoder_facts(state,original,source,{},serving=True)
            endpoint = {**state,'modules':{'train_siglip2_substrate_adaptation.py':original,'qualify_siglip2_substrate_cpu.py':source},
                'head_object':Head(),'A':Tensor((128,160),1.),'C':Tensor((128,1152),1.),'means':{'concat':Tensor((160,),0.)},
                'mu_train':Tensor((1152,),0.),'mu_train_provenance':{},'scope':{},'common_statistics':{},'flags':{},
                'vision_sha256':facts['vision_sha256'],
                'manifest':{'environment':{'packages':{}},'encoder_identity':copy.deepcopy(identity),'vision_sha256':facts['vision_sha256']}}
            endpoint['A'].requires_grad = endpoint['C'].requires_grad = True
            endpoint['readout_sha256'] = fp(d.inference_readout_tree(endpoint))
            for name in d.PROBE:
                model.params[name].value = 0. # Exact original-four member substituted.
                rejects(lambda:d.inference_outputs(endpoint,[object()]), 'current .data')
                model.params[name].value = 1.
            frozen = model.params['frozen.0']
            frozen.value = 2.
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'frozen447')
            frozen.value = 0.
            model.params[d.PROBE[0]].requires_grad = True
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'role/gradient')
            model.params[d.PROBE[0]].requires_grad = False
            model.embeddings._non_persistent_buffers_set.clear()
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'nonpersistent')
            model.embeddings._non_persistent_buffers_set.add('position_ids')
            model.position.value = 1.
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'buffers')
            model.position.value = 0.
            model.config._attn_implementation = 'eager'
            rejects(lambda:d.encoder_facts(state,original,source,{},serving=True), 'SDPA')
            model.config._attn_implementation = 'sdpa'
            endpoint['head_object'].weight.requires_grad = True
            rejects(lambda:d.inference_outputs(endpoint,[object()]), 'frozen head roles')
            endpoint['head_object'].weight.requires_grad = False
            endpoint['head_object'].weight.value = 2.
            rejects(lambda:d.inference_outputs(endpoint,[object()]), 'current .data')
        finally:
            d._processor_cache = original_cache
            d.probe_source = original_probe_source


def processor_release_seam(d):
    """Exercise NEW actual release paths with the admitted backend's real LRU code."""
    evidence = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/actual-objective-processor-cache-source'
    raw = (evidence/'original-image_processing_backends.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == d.BACKEND_SHA
    method = '_fuse_mean_std_and_rescale_factor'
    names = ('transformers.image_processing_backends','transformers.models.siglip.image_processing_siglip')
    previous = {name:sys.modules.get(name) for name in names}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'backend.py'
        path.write_bytes(raw)
        backend = ModuleType(names[0])
        backend.__file__ = str(path)
        backend.torch = SimpleNamespace(tensor=lambda x,device:x,float32='float32')
        code = compile(raw,str(path),'exec',dont_inherit=True)
        cls_code = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == 'TorchvisionBackend')
        function_code = next(c for c in cls_code.co_consts if isinstance(c,CodeType) and c.co_name == method)
        fn = FunctionType(function_code,vars(backend))
        fn.__defaults__ = (None,)*6
        wrapper = lru_cache(maxsize=10)(fn)
        backend.TorchvisionBackend = type('TorchvisionBackend',(),{'__module__':names[0],method:wrapper})
        siglip = ModuleType(names[1])
        siglip.SiglipImageProcessor = type('SiglipImageProcessor',(backend.TorchvisionBackend,),{'__module__':names[1]})
        sys.modules.update(dict(zip(names,(backend,siglip))))
        guards = {str(path):d.BACKEND_SHA}
        class Owner:
            def __init__(self): self.values = []
            def parameters(self): return iter(self.values)
            def buffers(self): return iter(())
        try:
            processor = siglip.SiglipImageProcessor()
            cache = d._processor_cache(processor,guards,empty=True)
            # No preprocessing is simulated: the real memoized method with all
            # optional values None returns (None,None,None) and retains self.
            assert getattr(processor,method)() == (None,None,None)
            ref = weakref.ref(processor)
            endpoint = {'processor_object':processor,'processor_cache':cache,'guards':guards,'modules':{},
                        'model':Owner(),'head_object':Owner(),'A':Owner(),'C':Owner(),'mu_train':Owner()}
            del processor
            gc.collect()
            assert ref() is not None and cache.cache_info().currsize == 1
            d.release_inference(endpoint)
            assert ref() is None and endpoint == {} and cache.cache_info().currsize == 0
            processor = siglip.SiglipImageProcessor()
            getattr(processor,method)()
            refs = []
            state = {k:Owner() for k in d.STATIC_KEYS+('A','C')}
            state.update(processor_object=processor,processor_cache=wrapper,model=Owner(),head_object=Owner(),
                         optimizer_object=SimpleNamespace(state_dict=lambda:{}),scaler_object=Owner())
            # SimpleNamespace cannot be weak-referenced; use the actual owned
            # optimizer seam shape with a weak-referenceable object instead.
            class Optimizer(Owner):
                def state_dict(self): return {}
            state['optimizer_object'] = Optimizer()
            trainer = SimpleNamespace(tensor_weakrefs=lambda context,value:[],release=lambda context,state:state.clear())
            ref = weakref.ref(processor)
            del processor
            d.release({'trainer':trainer,'guards':guards},state)
            assert ref() is None and state == {} and wrapper.cache_info().currsize == 0
        finally:
            wrapper.cache_clear()
            for name,module in previous.items():
                if module is None: sys.modules.pop(name,None)
                else: sys.modules[name] = module


def first17_mechanics_record_seam(d, record):
    """Execute the real producer wrapper and TRAIN lookup on a full receipt."""
    tree = ast.parse(DRIVER.read_text())
    producer = next(n for n in ast.walk(function(tree,'run')) if isinstance(n,ast.Assign) and
                    isinstance(n.value,ast.Dict) and
                    any(isinstance(k,ast.Constant) and k.value == 'result' for k in n.value.keys))
    namespace = {'arm':record['result']}
    exec(compile(ast.Module(body=[producer],type_ignores=[]),'<actual arm wrapper>','exec'),namespace)
    record = {**record,**namespace['result']}
    lookup = next(n for n in ast.walk(function(tree,'arm_run')) if isinstance(n,ast.If) and
                  any(isinstance(c,ast.Name) and c.id == 'step' for c in ast.walk(n.test)))
    code = compile(ast.Module(body=[lookup],type_ignores=[]),'<actual first17 lookup>','exec')
    seed,arm = record['seed'],record['arm']
    def replay(receipt, step):
        row = {**record['result']['steps'][step-1],'core_seconds':99.,'seconds':100.}
        exec(code,{**vars(d),'args':SimpleNamespace(phase='train'),'step':step,'row':row,
                   'seed':seed,'arm':arm,
                   'context':{'connected_terminals':{f'mechanics:{seed}:{arm}':receipt}}})
    assert 'steps' not in record and 'qualifications' not in record
    for step in range(1,18):
        replay(record,step)  # Real diagnostic excludes only update timing.
    for step in (1,17):
        wrong = copy.deepcopy(record)
        wrong['result']['steps'][step-1]['step'] = -1
        # A top-level decoy must never hide nested diagnostic tampering.
        wrong['steps'] = record['result']['steps']
        rejects(lambda:replay(wrong,step),'fresh first17 mechanics replay differs')
    for wrong,key in (({k:v for k,v in record.items() if k != 'result'},'result'),
                      ({**record,'result':{k:v for k,v in record['result'].items() if k != 'steps'}},'steps')):
        rejects(lambda:replay(wrong,1),key)
    short = {**record,'result':{**record['result'],'steps':record['result']['steps'][:-1]}}
    try:
        replay(short,17)
    except IndexError:
        pass
    else:
        raise AssertionError('accepted missing mechanics step17')
    exec(code,{**vars(d),'args':SimpleNamespace(phase='train'),'step':18,'context':{}})
    exec(code,{**vars(d),'args':SimpleNamespace(phase='mechanics'),'step':1,'context':{}})
    print('PASS executed producer-shaped full mechanics receipt: first17/steps/missing/tamper negatives')


def cost_terminal_falsifiers(d):
    """Actual timing/receipt predicates, with native work replaced by a clock."""
    events = []
    def connected(): pass
    def entry(*args):
        events.append('integrity')
        raise ValueError('entry reached')
    with tensor_seam(), patch.multiple(d,
            time=SimpleNamespace(perf_counter=lambda:events.append('tick') or 0.),
            bound_file=lambda *a:events.append('helper'), integrity=entry):
        context = {'trainer':None,'connected':SimpleNamespace(raw_features=connected,__file__='/helper'),
                   'connected_function':(connected,connected.__code__),'guards':{'/helper':'sha'}}
        rejects(lambda:d.update(context,{'device':'cpu','counter':0},{},1), 'entry reached')
    assert events == ['tick','helper','integrity'], events

    def timed_arm(phase, arm, core):
        clock = [0.]
        identity = {'arm':arm,'seed':d.SEEDS[0]}
        def update(context, state, identity, step):
            seconds = core/(128 if phase == 'train' else 34 if phase == 'mechanics' else 1)
            clock[0] += seconds
            return {'step':step,'core_seconds':seconds,'seconds':seconds}
        def qualify(*args):
            clock[0] += 100.  # Common construction/reload/bundle work cannot dilute core.
            return {}
        mechanics = {'result':{'steps':[{'step':s} for s in range(1,18)]}}
        context = {'connected_args':SimpleNamespace(phase=phase,output=Path('/fixture')),
                   'connected_terminals':{f'mechanics:{d.SEEDS[0]}:{arm}':mechanics}}
        with patch.multiple(d, time=SimpleNamespace(perf_counter=lambda:clock[0]),
                fresh=lambda *a:{'identity':identity}, update=update,
                tamper_witness=lambda *a:None, image_witness=lambda *a:{},
                save=lambda *a:('file','digest'), release=lambda c,s:s.clear(),
                restore=lambda *a:{'identity':identity}, payload=lambda *a:{},
                fingerprint=lambda *a:'digest', inference_members=lambda *a:{},
                export_bundle=lambda *a:{'sha256':'bundle'}, qualify_bundle=qualify):
            result = d.arm_run(context,arm,d.SEEDS[0],'cpu' if phase == 'cpu' else 'cuda',
                               discarded_update=phase == 'cpu')
        assert result['total_training_core_seconds'] == core
        assert result['arm_work_seconds'] == core+100.
        assert len(result['steps']) == (128 if phase == 'train' else 17 if phase == 'mechanics' else 1)
        assert len(result['replay_steps']) == (17 if phase == 'mechanics' else 0)
        return result
    control = timed_arm('train','control',100.)
    candidate = timed_arm('train','candidate',180.)
    timed_arm('mechanics','candidate',34.)
    cpu = timed_arm('cpu','candidate',1.)
    trainer = SimpleNamespace(tensor_weakrefs=lambda *a:[],release=lambda c,s:s.clear())
    with patch.multiple(d, arm_run=lambda *a,**kw:cpu,
            select_initializer=lambda *a:{'checkpoint':{},'canonical_initial':{}},
            load_initializer=lambda *a:({k:None for k in d.STATIC_KEYS+('A','C')},{}), resource_check=lambda *a:None):
        assert d.cpu_run({'trainer':trainer,'original_cpu_record':{}})['total_training_core_seconds'] == 1.
    assert candidate['arm_work_seconds']/control['arm_work_seconds'] == 1.4
    tree = ast.parse(DRIVER.read_text())
    gate = next(n for n in function(tree,'run').body if isinstance(n,ast.If) and
                'fresh_control_record' in ast.unparse(n))
    code = compile(ast.Module(body=[gate],type_ignores=[]),'<actual in-process cost gate>','exec')
    context = {'fresh_control_record':{**control,'wall_seconds':200.},
               'connected_launch':{'fresh_control':{'service_seconds':260.}}}
    namespace = {**vars(d),'args':SimpleNamespace(phase='train',arm='candidate'),
                 'context':context,'result':candidate,'wall':280.}
    rejects(lambda:exec(code,namespace), 'core/whole cost')
    namespace.update(result={'total_training_core_seconds':140.},wall=300.)
    exec(code,namespace)  # Exact <=1.50 boundary, using matching wall scopes.
    namespace['wall'] = 310.  # Would pass against the longer control service time.
    rejects(lambda:exec(code,namespace), 'core/whole cost')

    file = {'path':'/fixture/authority.json','sha256':'a'*64}
    unit = {'receipt':file,'log':file,'unit':'fixture','invocation_id':'a'*32,
            'service_seconds':260.,'native_peak_rss_kib':1,'both_locks_held':True}
    initializer = {'checkpoint':file,'canonical_initial':{}}
    context = {'connected_args':SimpleNamespace(execution_sha256='b'*64),
               'connected_code':{},'source':{},'original_cpu_record':{'numerical_flags':{}}}
    for phase in ('cpu','mechanics','train'):
        launch = {'schema':d.AUTHORITY_SCHEMA,'execution_sha256':'b'*64,'phase':phase,
                  'arm':'control','seed':d.SEEDS[0],'recipe':copy.deepcopy(d.RECIPE),
                  'resource_policy':d.policy(phase),'both_locks_held':True,
                  'original_cpu':{'authority':file,'terminal':unit},
                  'historical_mlp_actual_gradient':{'authority':file,'terminal':unit},
                  'witness':{'root':'/fixture/witness','files':{n:'a'*64 for n in d.WITNESS_FILES}},
                  'selected_cpu':None if phase == 'cpu' else unit,
                  'selected_mechanics':{a:unit for a in d.ARMS} if phase == 'train' else None,
                  'fresh_control':None}
        context['connected_launch'] = launch
        active = 'candidate' if phase == 'cpu' else 'control'
        count = 1 if phase == 'cpu' else 17 if phase == 'mechanics' else 128
        identity = {'arm':active,'seed':d.SEEDS[0],'device':'cpu' if phase == 'cpu' else 'cuda',
                    'method':d.method(launch),'scope':{'arm':'control','manifest_sha256':d.SCOPE_SHA256,
                    'arm_sha256':d.CONTROL_SHA256},'parameter_names':d.parameter_roles(active)[0],
                    'parameter_shapes':d.parameter_roles(active)[1]}
        rows = [{'step':s,'core_seconds':1.,'seconds':1.} for s in range(1,count+1)]
        replay = copy.deepcopy(rows) if phase == 'mechanics' else []
        total = float(len(rows)+len(replay))
        result = {'identity':identity,'training_updates':count,'steps':rows,'replay_steps':replay,
                  'independent_17_vs_8_plus_9':phase == 'mechanics','fresh_first17_replay':True,
                  'total_training_core_seconds':total,'arm_work_seconds':total+100.}
        record = {'schema':d.SCHEMA,'phase':phase,'arm':'control','seed':d.SEEDS[0],
                  'launch':launch,'execution_sha256':'b'*64,'code':{},'source':{},
                  'resource_policy':d.policy(phase),'quality_read':False,'state_reuse_eligible':False,
                  'cost_qualified':False,'model_fit_qualified':phase == 'train',
                  'total_training_core_seconds':total,'wall_seconds':total+110.,'process_peak_rss_kib':1,
                  'peak_cuda_allocated_bytes':0,'numerical_flags':{},
                  'invocation':{'optimize':0,'cuda_visible_devices':'' if phase == 'cpu' else '0',
                                'cublas_workspace_config':None if phase == 'cpu' else ':4096:8'},
                  **({'qualifications':[result],
                      'accepted_initializers':[{'seed':s,**initializer} for s in d.SEEDS]}
                     if phase == 'cpu' else {'result':result})}
        for key in ('historical_mlp_prerequisite_only','pass','exit_rehash_pass','sequential_model_ownership','strict_reload_exact',
                    'native_training_inference_exact','inference_artifact_independent',
                    'bundle_original_dependencies_denied','updated_source_mutants_rejected',
                    'both_locks_held_in_parent_authority','cost_requires_parent_normal_exit_units',
                    'terminal_exit_and_both_locks_require_parent_receipt'):
            record[key] = True
        with patch.multiple(d, check_steps=lambda *a:None, select_initializer=lambda *a:initializer):
            d.check_terminal(context,record,phase,'control',d.SEEDS[0])
            if phase == 'mechanics':
                first17_mechanics_record_seam(d,record)
            ceiling = d.policy(phase)['seconds']
            d.check_terminal(context,{**record,'wall_seconds':ceiling-.001},phase,'control',d.SEEDS[0])
            for elapsed in (ceiling,ceiling+.001):
                rejects(lambda:d.check_terminal(context,{**record,'wall_seconds':elapsed},
                                                phase,'control',d.SEEDS[0]),'terminal')
            for key,value in [('peak_cuda_allocated_bytes',10_000_000_000),
                              ('peak_cuda_allocated_bytes',-1),('peak_cuda_allocated_bytes',False),
                              ('peak_cuda_allocated_bytes',0.),('peak_cuda_allocated_bytes',None),
                              ('model_fit_qualified',phase != 'train'),('model_fit_qualified',int(phase == 'train')),
                              ('wall_seconds',float('inf')),('total_training_core_seconds',True),
                              ('process_peak_rss_kib',True),('pass',1),
                              ('cost_requires_parent_normal_exit_units',1)]:
                rejects(lambda k=key,v=value:d.check_terminal(context,{**record,k:v},phase,'control',d.SEEDS[0]), 'terminal')
            for key,value in [('cuda_visible_devices','0' if phase == 'cpu' else ''),('optimize',False)]:
                wrong = copy.deepcopy(record)
                wrong['invocation'][key] = value
                rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            for key,value in [('steps',rows[:-1]),('total_training_core_seconds',True),
                              ('arm_work_seconds',float('inf')),('training_updates',float(count))]:
                wrong = copy.deepcopy(record)
                (wrong['qualifications'][0] if phase == 'cpu' else wrong['result'])[key] = value
                rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            wrong = copy.deepcopy(record)
            (wrong['qualifications'][0] if phase == 'cpu' else wrong['result'])['steps'][0]['core_seconds'] = True
            rejects(lambda:d.check_terminal(context,wrong,phase,'control',d.SEEDS[0]), 'terminal')
            if phase != 'cpu':
                record['peak_cuda_allocated_bytes'] = 9_999_999_999
                d.check_terminal(context,record,phase,'control',d.SEEDS[0])
                record['invocation']['cublas_workspace_config'] = None
                rejects(lambda:d.check_terminal(context,record,phase,'control',d.SEEDS[0]), 'terminal')


def original_terminal_binding(d):
    """Catch unbound terminal readers using the real signed AST and cgroup API."""
    modules = {}
    try:
        for name in ('train_siglip2_substrate_adaptation', 'fit_siglip2_prototype_residual',
                     'export_siglip2_substrate_fit', 'train_siglip2_nearest_ranking',
                     'train_siglip2_quadratic_readout'):
            spec = importlib.util.spec_from_file_location('_terminal_test_'+name, HERE/(name+'.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module  # The fitter's dataclass needs its module.
            modules[name] = module
            spec.loader.exec_module(module)
        original, fitter, init, nearest, old = modules.values()
        assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == fitter.TERMINAL_SOURCE_SHA
        assert hashlib.sha256(Path(init.__file__).read_bytes()).hexdigest() == \
               '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8'
        admission = original.FlatAdmission()
        admission.init = init  # Actual quadratic bootstrap contract.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(path, value):
                path.write_text(value if isinstance(value,str) else json.dumps(value))
                return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            guards = {m.__file__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules.values()}
            source_guards = dict(guards)
            launch = {'witness':{'root':str(root/'witness'),'files':{}},'fresh_control':None}
            auth = write(root/'authority.json',launch)
            execution = write(root/'execution.json',{})
            launch_fact = {'python':'/python','python_sha256':'b'*64,'python_version':'3'}
            legacy = {'original':original,'admission':admission,'selected':{'genuine':{'reference':init}},
                      'invocations':set()}
            context = {'trainer':None,'guards':guards,'legacy':legacy,'nearest':nearest,'fitter':fitter,'old':old,
                       'fit_context':{'legacy':legacy,'guards':guards},'connected_root':root,
                       'connected_args':SimpleNamespace(execution_sha256=execution['sha256']),
                       'connected_code':{},'connected_launch':launch,
                       'original_cpu_record':{'invocation':launch_fact},'witness':SimpleNamespace(POLICY={'seconds':300})}
            def fixture(unit_name, identity, arm, seconds, core, phase='train', runtime=None):
                values = {'memory.max':str(8*1024**3),'memory.current':'1','memory.peak':'3',
                          'memory.swap.current':'0','memory.swap.peak':'0','memory.swap.max':'0',
                          'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0'}
                cgroup = {'path':'/sys/fs/cgroup/'+unit_name+'.service','values':values}
                final = {**copy.deepcopy(cgroup),'invocation_id':identity,'command_exit_status':0}
                log = '\n'.join([f'Running as unit: {unit_name}.service; invocation ID: {identity}',
                    '\tExit status: 0','Finished with result: success',
                    'Main processes terminated with: code=exited/status=0','\tSwaps: 0','Memory swap peak: 0B',
                    '\tMaximum resident set size (kbytes): 2','Service runtime: '+(runtime or f'{seconds}s'),
                    'FINAL_CGROUP '+json.dumps(final)])+'\n'
                unit = {'receipt':{'path':str(root/(arm+'.json')),'sha256':'a'*64},
                        'log':write(root/(arm+'.log'),log),'unit':unit_name,'invocation_id':identity,
                        'service_seconds':seconds,'native_peak_rss_kib':2,'both_locks_held':True}
                record = {'authority':copy.deepcopy(auth),'authority_sha256':auth['sha256'],'launch':copy.deepcopy(launch),
                          'invocation':{**launch_fact,'invocation_id':identity,'optimize':0,
                          'argv':d.cli(root,auth['path'],auth['sha256'],execution['sha256'],
                                       phase,arm,d.SEEDS[0],root)},
                          'input_guards':{execution['path']:execution['sha256']},'wall_seconds':seconds-.25,
                          'whole_seconds':seconds-.25,'process_peak_rss_kib':1,'total_training_core_seconds':core,
                          'cgroup_before':copy.deepcopy(cgroup),'cgroup_after':copy.deepcopy(cgroup)}
                unit['receipt'] = write(Path(unit['receipt']['path']),record)
                return record,unit,log,final
            _,control_unit,control_log,_ = fixture('control','c'*32,'control',10.,5.)
            launch['fresh_control'] = control_unit
            auth.update(write(Path(auth['path']),launch))
            candidate,unit,log,final = fixture('candidate','d'*32,'candidate',14.,7.)
            # Receipt model/state predicates have their own falsifiers above;
            # all file/authority/CLI/source/log/lock/cgroup/cost predicates run here.
            def admit():
                legacy['invocations'].clear()
                guards.clear(); guards.update(source_guards)
                return d.admit_terminal(context,unit,'train','candidate',d.SEEDS[0])
            with patch.object(d,'check_terminal',lambda *a:None):
                assert admit() == candidate  # RED: actual signed reader reaches missing self.init.
                assert context['connected_terminal_cgroups'][f'train:{d.SEEDS[0]}:candidate'] == final
                # The actual-gradient projection/call statements, unchanged from
                # the NEW driver, exercise its separate reader construction.
                node = function(ast.parse(DRIVER.read_text()),'admit_historical_mlp_actual_gradient')
                seam = [n for n in node.body if isinstance(n,ast.Assign) and
                        isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('projected','final')]
                namespace = {**vars(d),'context':context,'record':{**candidate,'invocation_id':unit['invocation_id']},
                             'selected':{'terminal':unit}}
                exec(compile(ast.Module(body=seam,type_ignores=[]),'<actual gradient terminal seam>','exec'),namespace)
                assert namespace['final'] == final
                reader1,reader2 = d.fresh_terminal_reader(context),d.fresh_terminal_reader(context)
                assert reader1 is not reader2 and reader1 is not admission and reader1.init is reader2.init is init
                assert reader1.entries == reader2.entries == {} and reader1.verified == reader2.verified == set()
                for wrong in (None,SimpleNamespace(admit_cgroup=lambda *a:None)):
                    admission.init = wrong
                    rejects(admit,'genuine terminal initializer')
                del admission.init
                rejects(admit,'genuine terminal initializer')
                admission.init = init
                prior = source_guards.pop(init.__file__)
                rejects(admit,init.__file__)
                source_guards[init.__file__] = '0'*64
                rejects(admit,'bytes')
                source_guards[init.__file__] = prior
                # Fresh reader must reject changed bytes even after success.
                Path(unit['log']['path']).write_text(log+'changed\n')
                rejects(admit,'SHA256')
                Path(unit['log']['path']).write_text(log)
                required = log.splitlines()[:-1]
                for line in required:
                    for mutated in (log.replace(line+'\n',''),log+line+'\n'):
                        unit['log'] = write(Path(unit['log']['path']),mutated)
                        rejects(admit,'original')
                unit['log'] = write(Path(unit['log']['path']),log)
                for change,text in [({'both_locks_held':False},'UNIT'),({'service_seconds':3001.},'caps'),
                                    ({'native_peak_rss_kib':8*1024**2+1},'caps')]:
                    saved = dict(unit)
                    unit.update(change)
                    rejects(admit,text)
                    unit.clear(); unit.update(saved)
                for stage in ('cgroup_before','cgroup_after'):
                    saved = copy.deepcopy(candidate[stage])
                    for value,text in [({},'path'),({**saved,'path':'relative/candidate.service'},'enclosing unit'),
                                       ({**saved,'values':{**saved['values'],'memory.swap.max':'1'}},'caps'),
                                       ({**saved,'values':{**saved['values'],'memory.events':
                                                          'low 1\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0'}},
                                        'disallowed cgroup event')]:
                        candidate[stage] = value
                        unit['receipt'] = write(Path(unit['receipt']['path']),candidate)
                        rejects(admit,text)
                    candidate[stage] = saved
                unit['receipt'] = write(Path(unit['receipt']['path']),candidate)
                for mutated,text in [(None,'footer'),({**final,'invocation_id':'e'*32},'footer'),
                                     ({**final,'path':'/wrong.service'},'enclosing cgroup'),
                                     ({**final,'values':{**final['values'],'memory.peak':'2'}},'whole-unit peak')]:
                    footer = '' if mutated is None else 'FINAL_CGROUP '+json.dumps(mutated)+'\n'
                    unit['log'] = write(Path(unit['log']['path']),log[:log.index('FINAL_CGROUP ')]+footer)
                    rejects(admit,text)
                for key,value in [('memory.max','1'),('memory.current','0'),('memory.peak',str(8*1024**3+1)),
                                  ('memory.swap.current','1'),('memory.swap.peak','1'),('memory.swap.max','1'),
                                  ('memory.events','max 1\noom 0\noom_kill 0')]:
                    mutated = copy.deepcopy(final)
                    mutated['values'][key] = value
                    unit['log'] = write(Path(unit['log']['path']),log[:log.index('FINAL_CGROUP ')]+
                                        'FINAL_CGROUP '+json.dumps(mutated)+'\n')
                    rejects(admit,'caps' if key != 'memory.events' else 'failure events')
                unit['log'] = write(Path(unit['log']['path']),log)
                unit['log'] = write(Path(unit['log']['path']),log+log[log.index('FINAL_CGROUP '):])
                rejects(admit,'footer')
                unit['log'] = write(Path(unit['log']['path']),log)
                # The separate live-control cost reader must also keep the cgroup
                # and normal-exit predicates; the candidate's valid log cannot cover it.
                control_unit['log'] = write(Path(control_unit['log']['path']),control_log.replace('\tExit status: 0','\tExit status: 1'))
                candidate['launch']['fresh_control'] = control_unit
                unit['receipt'] = write(Path(unit['receipt']['path']),candidate)
                launch['fresh_control'] = control_unit
                auth.update(write(Path(auth['path']),launch))
                candidate['authority'] = copy.deepcopy(auth)
                candidate['authority_sha256'] = auth['sha256']
                candidate['invocation']['argv'] = d.cli(root,auth['path'],auth['sha256'],execution['sha256'],
                                                       'train','candidate',d.SEEDS[0],root)
                unit['receipt'] = write(Path(unit['receipt']['path']),candidate)
                rejects(admit,'normal-exit')

                def duration_case(phase, seconds, runtime, *, control_seconds=None,
                                  control_runtime=None, core=100., control_core=100., broken=None):
                    nonlocal auth
                    launch['fresh_control'] = None
                    if control_seconds is not None:
                        auth = write(root/'long-control-authority.json',launch)
                        _,control,control_log,_ = fixture('long-control','1'*32,'control',control_seconds,
                                                          control_core,runtime=control_runtime)
                        if broken is not None:
                            control['log'] = write(Path(control['log']['path']),control_log.replace(
                                'Finished with result: success','Finished with result: '+broken))
                        launch['fresh_control'] = control
                    arm = 'control' if control_seconds is None else 'candidate'
                    auth = write(root/'long-authority.json',launch)
                    record,unit,_,_ = fixture('long-'+arm,'2'*32,arm,seconds,core,phase,runtime)
                    legacy['invocations'].clear()
                    guards.clear(); guards.update(source_guards)
                    return d.admit_terminal(context,unit,phase,arm,d.SEEDS[0]),record

                # Wall/core/service times are consistent in every longer fixture.
                # The original reader admits service <= cap; check_terminal's
                # strict wall < cap is exercised separately above.
                for phase,ceiling,spelling in (('cpu',600,'10min 0s'),
                                              ('mechanics',1200,'20min 0s'),
                                              ('train',3000,'50min 0s')):
                    admitted,record = duration_case(phase,ceiling-.1,f'{ceiling/60-1:g}min 59.9s')
                    assert admitted == record
                    admitted,record = duration_case(phase,ceiling,spelling)
                    assert admitted == record
                    rejects(lambda:duration_case(phase,ceiling+.001,f'{ceiling/60:g}min 1ms'),'caps')
                admitted,record = duration_case('mechanics',600.075,'10min 75ms')
                assert admitted == record  # > old mechanics300
                for control_seconds,control_runtime,seconds,runtime in (
                        (1200.075,'20min 75ms',1700.,'28min 20s'),
                        (2999.9,'49min 59.9s',2999.9,'49min 59.9s')):
                    admitted,record = duration_case('train',seconds,runtime,
                        control_seconds=control_seconds,control_runtime=control_runtime)
                    assert admitted == record  # Both arms > old TRAIN600.
                rejects(lambda:duration_case('train',2999.9,'49min 59.9s',
                    control_seconds=3000.001,control_runtime='50min 1ms'),'caps')
                # Independent live core and whole-service gates retain <=1.50.
                admitted,record = duration_case('train',1800.,'30min 0s',control_seconds=1200.,
                                                control_runtime='20min 0s',core=150.)
                assert admitted == record
                rejects(lambda:duration_case('train',1800.,'30min 0s',control_seconds=1200.,
                    control_runtime='20min 0s',core=150.001),'cost ratio')
                rejects(lambda:duration_case('train',1800.001,'30min 1ms',control_seconds=1200.,
                    control_runtime='20min 0s',core=150.),'cost ratio')
                for status in ('timeout','signal','exit-code'):
                    rejects(lambda:duration_case('train',1700.,'28min 20s',control_seconds=1200.075,
                        control_runtime='20min 75ms',broken=status),'normal-exit')
        assert not {n.split('.')[0] for n in sys.modules} & d.NATIVE
        print('PASS real original terminal AST: connected/gradient/control seams and binding/log/cgroup falsifiers')
    finally:
        for module in modules.values():
            sys.modules.pop(module.__name__,None)


# Exact finite production inverse, before the unchanged historical inverse chain.
PROBE_INVERSE = ((b'train_siglip2_connected_mlp.py', b'train_siglip2_connected_probe.py', 5), (b'test_siglip2_connected_mlp.py', b'test_siglip2_connected_probe.py', 1), (b'siglip2-connected-mlp', b'siglip2-connected-probe', 4), (b'MLP', b'PROBE', 39), (b"PROBE = tuple('encoder.layers.26.mlp.'+layer+'.'+field for layer in ('fc1','fc2') for field in ('weight','bias'))\nPROBE_SHAPES = [[4304,1152],[4304],[1152,4304],[1152]]", b"HISTORICAL_MLP = tuple('encoder.layers.26.mlp.'+layer+'.'+field for layer in ('fc1','fc2') for field in ('weight','bias'))\nHISTORICAL_MLP_SHAPES = [[4304,1152],[4304],[1152,4304],[1152]]\nPROBE = ('head.probe',)\nPROBE_SHAPES = [[1,1,1152]]\nPROBE_SOURCE_SHA = '274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31'", 1), (b"'actual_gradient'", b"'historical_mlp_actual_gradient'", 4), (b'admit_actual_gradient', b'admit_historical_mlp_actual_gradient', 2), (b"def admit_historical_mlp_actual_gradient(context):\n    launch,trainer = context['connected_launch'],context['trainer']\n    selected = launch['historical_mlp_actual_gradient']\n    record = read_json(selected['terminal']['receipt'],context['guards'])\n    auth = read_json(selected['authority'],context['guards'])\n    require(record['schema'] == 'actual-objective-encoder-gradients-v1' and record['pass'] is True and\n            record['engineering_only'] is True and record['training_updates'] == 0 and\n            all(record[k] is False for k in ('model_fit_qualified','cost_qualified','quality_read','state_reuse','state_reuse_eligible')) and\n            all(record[k] is True for k in ('exit_rehash_pass','all_temporary_references_released')) and\n            record['frozen_encoder_tensors'] == 444 and record['unchanged_encoder_tensors'] == 448 and\n            record['gradient_parameters'] == list(PROBE) and record['resource_policy'] == context['witness'].POLICY and\n            selected['terminal']['invocation_id'] == record['invocation_id'] == '1a0ef321f35748ec9b88abf7115d7fba' and\n            selected['terminal']['receipt']['sha256'] == 'e7da19ba546f9d5375697d3d0347b3c96f9cfe0ee5183f9341d473114461d896' and\n            selected['terminal']['log']['sha256'] == '3dd0843123c576c91869286f1f69320ae2f4fa50c32aef7c714bc51e1f305b1a' and\n            record['authority_sha256'] == selected['authority']['sha256'] and\n            auth['schema'] == context['witness'].AUTHORITY_SCHEMA and auth['files'] == launch['witness']['files'] and\n            auth['both_locks_held'] is True and auth['resource_policy'] == context['witness'].POLICY,\n            'original actual-gradient engineering admission differs')\n    require([v['view'] for v in record['views']] == list(VIEWS), 'both actual pixel views required')\n    for view in record['views']:\n        require(all(view[k] is True for k in ('detached_forward_identical','detached_encoder_gradients_absent',\n                'detached_A_C_gradients_match','GPU_readout_unchanged','global_RNG_unchanged','temporary_references_released')) and\n                view['frozen_encoder_gradients_absent'] == 444, 'actual gradient/lifetime/RNG predicates differ')\n        for term in ('ranking','total'):\n            require(all(view['gradients'][term][n]['norm'] > 0 and view['gradients'][term][n]['nonzero'] > 0\n                        for n in PROBE), 'all-four actual ranking/total gradients required')\n    batch_bound_files(context['guards'],record['input_guards'].items())\n    require(all(record['input_guards'].get(str(Path(launch['witness']['root'])/n)) == h\n                for n,h in launch['witness']['files'].items()), 'admitted witness closure differs')\n    # Only field names differ in the discarded witness receipt. The original\n    # terminal reader retains every log, cgroup, normal-exit and lock predicate.\n    projected = {**record,'invocation':{'invocation_id':record['invocation_id'],'optimize':0},\n                 'wall_seconds':record['whole_seconds'],'process_peak_rss_kib':selected['terminal']['native_peak_rss_kib']}\n    final = context['fitter'].original_terminal_reader(context['fit_context'])(\n        fresh_terminal_reader(context),projected,selected['terminal'],context['witness'].POLICY['seconds'],context['guards'])\n    for cgroup in (record['cgroup_before'],record['cgroup_after'],final):\n        context['old'].zero_events(cgroup)\n\n\n", b"def admit_historical_mlp_actual_gradient(context):\n    launch,trainer = context['connected_launch'],context['trainer']\n    selected = launch['historical_mlp_actual_gradient']\n    record = read_json(selected['terminal']['receipt'],context['guards'])\n    auth = read_json(selected['authority'],context['guards'])\n    require(record['schema'] == 'actual-objective-encoder-gradients-v1' and record['pass'] is True and\n            record['engineering_only'] is True and record['training_updates'] == 0 and\n            all(record[k] is False for k in ('model_fit_qualified','cost_qualified','quality_read','state_reuse','state_reuse_eligible')) and\n            all(record[k] is True for k in ('exit_rehash_pass','all_temporary_references_released')) and\n            record['frozen_encoder_tensors'] == 444 and record['unchanged_encoder_tensors'] == 448 and\n            record['gradient_parameters'] == list(HISTORICAL_MLP) and record['resource_policy'] == context['witness'].POLICY and\n            selected['terminal']['invocation_id'] == record['invocation_id'] == '1a0ef321f35748ec9b88abf7115d7fba' and\n            selected['terminal']['receipt']['sha256'] == 'e7da19ba546f9d5375697d3d0347b3c96f9cfe0ee5183f9341d473114461d896' and\n            selected['terminal']['log']['sha256'] == '3dd0843123c576c91869286f1f69320ae2f4fa50c32aef7c714bc51e1f305b1a' and\n            record['authority_sha256'] == selected['authority']['sha256'] and\n            auth['schema'] == context['witness'].AUTHORITY_SCHEMA and auth['files'] == launch['witness']['files'] and\n            auth['both_locks_held'] is True and auth['resource_policy'] == context['witness'].POLICY,\n            'original actual-gradient engineering admission differs')\n    require([v['view'] for v in record['views']] == list(VIEWS), 'both actual pixel views required')\n    for view in record['views']:\n        require(all(view[k] is True for k in ('detached_forward_identical','detached_encoder_gradients_absent',\n                'detached_A_C_gradients_match','GPU_readout_unchanged','global_RNG_unchanged','temporary_references_released')) and\n                view['frozen_encoder_gradients_absent'] == 444, 'actual gradient/lifetime/RNG predicates differ')\n        for term in ('ranking','total'):\n            require(all(view['gradients'][term][n]['norm'] > 0 and view['gradients'][term][n]['nonzero'] > 0\n                        for n in HISTORICAL_MLP), 'all-four actual ranking/total gradients required')\n    batch_bound_files(context['guards'],record['input_guards'].items())\n    require(all(record['input_guards'].get(str(Path(launch['witness']['root'])/n)) == h\n                for n,h in launch['witness']['files'].items()), 'admitted witness closure differs')\n    # Only field names differ in the discarded witness receipt. The original\n    # terminal reader retains every log, cgroup, normal-exit and lock predicate.\n    projected = {**record,'invocation':{'invocation_id':record['invocation_id'],'optimize':0},\n                 'wall_seconds':record['whole_seconds'],'process_peak_rss_kib':selected['terminal']['native_peak_rss_kib']}\n    final = context['fitter'].original_terminal_reader(context['fit_context'])(\n        fresh_terminal_reader(context),projected,selected['terminal'],context['witness'].POLICY['seconds'],context['guards'])\n    for cgroup in (record['cgroup_before'],record['cgroup_after'],final):\n        context['old'].zero_events(cgroup)\n\n\n", 1), (b'444', b'447', 5), (b"def admit_historical_mlp_actual_gradient(context):\n    launch,trainer = context['connected_launch'],context['trainer']\n    selected = launch['historical_mlp_actual_gradient']\n    record = read_json(selected['terminal']['receipt'],context['guards'])\n    auth = read_json(selected['authority'],context['guards'])\n    require(record['schema'] == 'actual-objective-encoder-gradients-v1' and record['pass'] is True and\n            record['engineering_only'] is True and record['training_updates'] == 0 and\n            all(record[k] is False for k in ('model_fit_qualified','cost_qualified','quality_read','state_reuse','state_reuse_eligible')) and\n            all(record[k] is True for k in ('exit_rehash_pass','all_temporary_references_released')) and\n            record['frozen_encoder_tensors'] == 447 and record['unchanged_encoder_tensors'] == 448 and\n            record['gradient_parameters'] == list(HISTORICAL_MLP) and record['resource_policy'] == context['witness'].POLICY and\n            selected['terminal']['invocation_id'] == record['invocation_id'] == '1a0ef321f35748ec9b88abf7115d7fba' and\n            selected['terminal']['receipt']['sha256'] == 'e7da19ba546f9d5375697d3d0347b3c96f9cfe0ee5183f9341d473114461d896' and\n            selected['terminal']['log']['sha256'] == '3dd0843123c576c91869286f1f69320ae2f4fa50c32aef7c714bc51e1f305b1a' and\n            record['authority_sha256'] == selected['authority']['sha256'] and\n            auth['schema'] == context['witness'].AUTHORITY_SCHEMA and auth['files'] == launch['witness']['files'] and\n            auth['both_locks_held'] is True and auth['resource_policy'] == context['witness'].POLICY,\n            'original actual-gradient engineering admission differs')\n    require([v['view'] for v in record['views']] == list(VIEWS), 'both actual pixel views required')\n    for view in record['views']:\n        require(all(view[k] is True for k in ('detached_forward_identical','detached_encoder_gradients_absent',\n                'detached_A_C_gradients_match','GPU_readout_unchanged','global_RNG_unchanged','temporary_references_released')) and\n                view['frozen_encoder_gradients_absent'] == 447, 'actual gradient/lifetime/RNG predicates differ')\n        for term in ('ranking','total'):\n            require(all(view['gradients'][term][n]['norm'] > 0 and view['gradients'][term][n]['nonzero'] > 0\n                        for n in HISTORICAL_MLP), 'all-four actual ranking/total gradients required')\n    batch_bound_files(context['guards'],record['input_guards'].items())\n    require(all(record['input_guards'].get(str(Path(launch['witness']['root'])/n)) == h\n                for n,h in launch['witness']['files'].items()), 'admitted witness closure differs')\n    # Only field names differ in the discarded witness receipt. The original\n    # terminal reader retains every log, cgroup, normal-exit and lock predicate.\n    projected = {**record,'invocation':{'invocation_id':record['invocation_id'],'optimize':0},\n                 'wall_seconds':record['whole_seconds'],'process_peak_rss_kib':selected['terminal']['native_peak_rss_kib']}\n    final = context['fitter'].original_terminal_reader(context['fit_context'])(\n        fresh_terminal_reader(context),projected,selected['terminal'],context['witness'].POLICY['seconds'],context['guards'])\n    for cgroup in (record['cgroup_before'],record['cgroup_after'],final):\n        context['old'].zero_events(cgroup)\n\n\n", b"def admit_historical_mlp_actual_gradient(context):\n    launch,trainer = context['connected_launch'],context['trainer']\n    selected = launch['historical_mlp_actual_gradient']\n    record = read_json(selected['terminal']['receipt'],context['guards'])\n    auth = read_json(selected['authority'],context['guards'])\n    require(record['schema'] == 'actual-objective-encoder-gradients-v1' and record['pass'] is True and\n            record['engineering_only'] is True and record['training_updates'] == 0 and\n            all(record[k] is False for k in ('model_fit_qualified','cost_qualified','quality_read','state_reuse','state_reuse_eligible')) and\n            all(record[k] is True for k in ('exit_rehash_pass','all_temporary_references_released')) and\n            record['frozen_encoder_tensors'] == 444 and record['unchanged_encoder_tensors'] == 448 and\n            record['gradient_parameters'] == list(HISTORICAL_MLP) and record['resource_policy'] == context['witness'].POLICY and\n            selected['terminal']['invocation_id'] == record['invocation_id'] == '1a0ef321f35748ec9b88abf7115d7fba' and\n            selected['terminal']['receipt']['sha256'] == 'e7da19ba546f9d5375697d3d0347b3c96f9cfe0ee5183f9341d473114461d896' and\n            selected['terminal']['log']['sha256'] == '3dd0843123c576c91869286f1f69320ae2f4fa50c32aef7c714bc51e1f305b1a' and\n            record['authority_sha256'] == selected['authority']['sha256'] and\n            auth['schema'] == context['witness'].AUTHORITY_SCHEMA and auth['files'] == launch['witness']['files'] and\n            auth['both_locks_held'] is True and auth['resource_policy'] == context['witness'].POLICY,\n            'original actual-gradient engineering admission differs')\n    require([v['view'] for v in record['views']] == list(VIEWS), 'both actual pixel views required')\n    for view in record['views']:\n        require(all(view[k] is True for k in ('detached_forward_identical','detached_encoder_gradients_absent',\n                'detached_A_C_gradients_match','GPU_readout_unchanged','global_RNG_unchanged','temporary_references_released')) and\n                view['frozen_encoder_gradients_absent'] == 444, 'actual gradient/lifetime/RNG predicates differ')\n        for term in ('ranking','total'):\n            require(all(view['gradients'][term][n]['norm'] > 0 and view['gradients'][term][n]['nonzero'] > 0\n                        for n in HISTORICAL_MLP), 'all-four actual ranking/total gradients required')\n    batch_bound_files(context['guards'],record['input_guards'].items())\n    require(all(record['input_guards'].get(str(Path(launch['witness']['root'])/n)) == h\n                for n,h in launch['witness']['files'].items()), 'admitted witness closure differs')\n    # Only field names differ in the discarded witness receipt. The original\n    # terminal reader retains every log, cgroup, normal-exit and lock predicate.\n    projected = {**record,'invocation':{'invocation_id':record['invocation_id'],'optimize':0},\n                 'wall_seconds':record['whole_seconds'],'process_peak_rss_kib':selected['terminal']['native_peak_rss_kib']}\n    final = context['fitter'].original_terminal_reader(context['fit_context'])(\n        fresh_terminal_reader(context),projected,selected['terminal'],context['witness'].POLICY['seconds'],context['guards'])\n    for cgroup in (record['cgroup_before'],record['cgroup_after'],final):\n        context['old'].zero_events(cgroup)\n\n\n", 1), (b'initial_four_sha256', b'initial_probe_sha256', 2), (b'list(range(2,6))', b'list(range(2,3))', 1), (b'Frozen connected last-PROBE method', b'Connected existing single pooling probe method', 1), (b'actual_gradient={authority:FILE,terminal:UNIT} is admitted actual-v2 CUDA300.', b'historical_mlp_actual_gradient={authority:FILE,terminal:UNIT} retains actual-v2 CUDA300.\n It proves only HISTORICAL_MLP (four tensors), never the new probe. Own CPU\n and both fresh mechanics are the sole actual-objective probe qualification.', 1), (b'nonzero-four public bundle', b'nonzero-single-probe public bundle', 1), (b'Only candidate adds four absolute layer26 PROBE parameters', b'Only candidate adapts original head.probe[1,1,1152]', 1), (b'exact four absolute FP32 encoder tensors', b'one absolute FP32 head.probe tensor', 1), (b'exact four overlay names', b'exact single probe overlay names', 1), (b'exact four/frozen447', b'exact single probe/frozen447', 1), (b'exact four/full448', b'exact single probe/full448', 1), (b'four typed payload', b'single probe typed payload', 1), (b'Exact original-four substitution', b'Exact original-probe substitution', 1), (b'        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None', b'        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None\n        view_total = [torch.zeros_like(p) for p in members] if step == 1 else None', 1), (b'                    del accumulator,gradient,gradients\n                loss = mse+rank', b"                    del accumulator,gradient,gradients\n                    gradients = torch.autograd.grad(mse+rank,members,retain_graph=True,allow_unused=True)\n                    for accumulator,gradient in zip(view_total,gradients,strict=True):\n                        require(gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all().item(),\n                                'actual total member gradient absent/nonfinite')\n                        accumulator.add_(gradient.detach())\n                    del accumulator,gradient,gradients\n                loss = mse+rank", 1), (b"            view_gradients.append({'view':view,'ranking_gradient_norms':norms})", b"            total_norms = {n:float(g.double().norm()) for n,g in zip(names,view_total,strict=True)}\n            require(all(math.isfinite(v) and v > 0 for v in total_norms.values()),\n                    'both-view actual total gradients must be nonzero')\n            require(all(p.grad is None for n,p in state['model'].named_parameters() if n not in PROBE),\n                    'frozen447 actual gradients must be absent')\n            view_gradients.append({'view':view,'ranking_gradient_norms':norms,'total_gradient_norms':total_norms})\n            del view_total", 1), (b"                 all(v['ranking_gradient_norms'].keys() == set(identity['parameter_names']) and\n                     all(math.isfinite(n) and n > 0 for n in v['ranking_gradient_norms'].values()) for v in row['view_gradients']))", b"                 all(all(v[term].keys() == set(identity['parameter_names']) and\n                     all(math.isfinite(n) and n > 0 for n in v[term].values())\n                     for term in ('ranking_gradient_norms','total_gradient_norms')) for v in row['view_gradients']))", 1), (b"    # A pooling-head member can share the nominated fc2.bias shape; strict names\n    # and updated complete identity must reject the substitution as well.\n    nominated = PROBE[-1]\n    wrong = next(p for n,p in params.items() if n.startswith('head.') and p.shape == params[nominated].shape)\n    backup = params[nominated].detach().clone()\n    try:\n        params[nominated].data.copy_(wrong)\n        expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'same-shaped pooling-head substitution accepted')\n    finally:\n        params[nominated].data.copy_(backup)\n", b"    # A distinct same-numel native leaf prevents the old self-copy no-op.\n    nominated = PROBE[0]\n    wrong = next(p for n,p in params.items() if n != nominated and p.is_leaf and\n                 p.numel() == params[nominated].numel() and\n                 not torch.equal(p.detach().reshape(params[nominated].shape),params[nominated].detach()))\n    backup = params[nominated].detach().clone()\n    before = original.fingerprint(params[nominated])\n    try:\n        params[nominated].data.copy_(wrong.detach().reshape(params[nominated].shape))\n        require(original.fingerprint(params[nominated]) != before, 'same-numel substitution must change actual bytes')\n        expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'same-numel distinct-leaf substitution accepted')\n    finally:\n        params[nominated].data.copy_(backup)\n    require(original.fingerprint(params[nominated]) == before, 'same-numel substitution restoration differs')\n", 1), (b'def model_structure(model, source, packages):', b'def _probe_decorators(packages, guards):\n    """Exact executable decorator sources already guarded by original CPU-v5."""\n    descriptors = (\n        (\'transformers.utils.generic\',\'252a17b73d020df90157e4033e2db51dcac8c11c9ffb799c33d5302a71c4964e\',\n         (\'merge_with_config_defaults\',)),\n        (\'transformers.utils.output_capturing\',\'65fa93bcfd2314d2a680c08c3699773f2ffea44340e85cd0e6392698b83b37bb\',\n         (\'capture_outputs\',)),\n        (\'transformers.utils.auto_docstring\',\'1c807048db9d45b45af9a4960802af5186bbe8f817059ac163abea49c187f660\',\n         (\'auto_docstring\',\'auto_method_docstring\')),\n    )\n    admitted = {}\n    for name,sha,names in descriptors:\n        module = sys.modules.get(name)\n        require(module is not None, \'original probe decorator module absent\')\n        path = Path(module.__file__)\n        require(path.is_relative_to(Path(packages[\'transformers\'][\'root\'])) and guards.get(str(path)) == sha,\n                \'CPU-v5 probe decorator source guard differs\')\n        raw = bound_file(guards,path,sha).read_bytes()\n        require(hashlib.sha256(raw).hexdigest() == sha, \'probe decorator source changed before compilation\')\n        code = compile(raw,str(path),\'exec\',dont_inherit=True)\n        for member in names:\n            expected = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == member)\n            fn = vars(module).get(member)\n            require(type(fn) is FunctionType and fn.__globals__ is vars(module) and fn.__code__ == expected and\n                    fn.__closure__ is None and \'__wrapped__\' not in vars(fn), \'live probe decorator factory differs\')\n            admitted[member] = (fn,expected,module)\n    return admitted\n\n\ndef _probe_vision_forward(fn, expected, module, decorators):\n    """Exact two-wrapper circuit from bound sources, including callable closures."""\n    merge,merge_code,generic = decorators[\'merge_with_config_defaults\']\n    capture,capture_code,outputs = decorators[\'capture_outputs\']\n    auto,_,_ = decorators[\'auto_docstring\']\n    require(module.merge_with_config_defaults is merge and module.capture_outputs is capture and\n            module.auto_docstring is auto, \'model probe decorator bindings differ\')\n    outer = next(c for c in merge_code.co_consts if isinstance(c,CodeType) and c.co_name == \'wrapper\')\n    factory = next(c for c in capture_code.co_consts if isinstance(c,CodeType) and c.co_name == \'wrapped_fn\')\n    inner = next(c for c in factory.co_consts if isinstance(c,CodeType) and c.co_name == \'wrapper\')\n    # auto_method_docstring returns the same function: no third executable wrapper.\n    for code,owner,flags in ((outer,generic,{}),(inner,outputs,{\'tie_last_hidden_states\':False})):\n        require(type(fn) is FunctionType and fn.__code__ == code and fn.__globals__ is vars(owner) and\n                fn.__defaults__ is None and fn.__kwdefaults__ is None and fn.__closure__ is not None and\n                len(fn.__closure__) == len(code.co_freevars) and set(code.co_freevars) == {\'func\',*flags},\n                \'genuine live probe circuit differs: wrapper\')\n        cells = dict(zip(code.co_freevars,fn.__closure__,strict=True))\n        require(all(cells[key].cell_contents is value for key,value in flags.items()),\n                \'genuine live probe wrapper flags differ\')\n        target = cells[\'func\'].cell_contents\n        require(vars(fn).get(\'__wrapped__\') is target, \'probe wrapper actual callable closure differs\')\n        fn = target\n    require(type(fn) is FunctionType and fn.__code__ == expected and fn.__globals__ is vars(module) and\n            fn.__closure__ is None and \'__wrapped__\' not in vars(fn) and type(fn.__defaults__) is tuple and\n            len(fn.__defaults__) == 1 and fn.__defaults__[0] is False and fn.__kwdefaults__ is None,\n            \'genuine live probe circuit differs: base forward\')\n\n\ndef probe_source(model, packages, guards):\n    """Bind genuine classes/live methods to the CPU-v5 observed Siglip source."""\n    module = sys.modules.get(\'transformers.models.siglip.modeling_siglip\')\n    require(module is not None and type(model) is getattr(module,\'SiglipVisionModel\',None) and\n            type(model.head) is getattr(module,\'SiglipMultiheadAttentionPoolingHead\',None) and\n            model.use_head is True and model.config.hidden_size == 1152 and\n            list(model.head.probe.shape) == PROBE_SHAPES[0] and\n            dict(model.named_parameters()).get(PROBE[0]) is model.head.probe,\n            \'genuine existing single pooling probe differs\')\n    path = Path(module.__file__)\n    require(path.is_relative_to(Path(packages[\'transformers\'][\'root\'])) and\n            guards.get(str(path)) == PROBE_SOURCE_SHA, \'CPU-v5 probe source guard differs\')\n    raw = bound_file(guards,path,PROBE_SOURCE_SHA).read_bytes()\n    require(hashlib.sha256(raw).hexdigest() == PROBE_SOURCE_SHA, \'probe source changed before compilation\')\n    code = compile(raw,str(path),\'exec\',dont_inherit=True)\n    for instance in (model,model.head):\n        cls = type(instance)\n        cls_code = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == cls.__name__)\n        for name in (\'__init__\',\'forward\'):\n            fn = vars(cls)[name]\n            expected = next(c for c in cls_code.co_consts if isinstance(c,CodeType) and c.co_name == name)\n            require(name not in vars(instance), \'genuine live probe circuit differs: instance\')\n            if cls is type(model) and name == \'forward\':\n                _probe_vision_forward(fn,expected,module,_probe_decorators(packages,guards))\n                continue\n            require(type(fn) is FunctionType and fn.__globals__ is vars(module) and\n                    fn.__code__ == expected and \'__wrapped__\' not in vars(fn) and\n                    fn.__defaults__ is None and fn.__kwdefaults__ is None and\n                    (fn.__closure__ is None if not expected.co_freevars else\n                     expected.co_freevars == (\'__class__\',) and fn.__closure__ is not None and\n                     len(fn.__closure__) == 1 and fn.__closure__[0].cell_contents is cls),\n                    \'genuine live probe circuit differs: \'+cls.__name__+\'.\'+name)\n\n\ndef model_structure(model, source, packages):', 1), (b'    model = source.construct(config,construct_context).eval()', b"    model = source.construct(config,construct_context).eval()\n    probe_source(model,construct_context['packages'],guards)", 1), (b"    ident = state['encoder_identity']", b"    ident = state['encoder_identity']\n    probe_source(model,packages,state['guards'])", 1), (b"            record['quality_read'] is False", b"            record['historical_mlp_prerequisite_only'] is True and\n            record['quality_read'] is False", 1), (b"'pass':True,'quality_read':False,", b"'pass':True,'quality_read':False,\n        'historical_mlp_prerequisite_only':True,", 1))


def probe_inverse(raw):
    for before,after,count in reversed(PROBE_INVERSE):
        assert raw.count(after) == count, 'finite probe inverse occurrence differs'
        raw = raw.replace(after,before)
    assert hashlib.sha256(raw).hexdigest() == '79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b', 'finite probe inverse bytes differ'
    return raw


def probe_contract_falsifiers(d):
    original = (HERE/'train_siglip2_connected_mlp.py').read_bytes()
    assert probe_inverse(DRIVER.read_bytes()) == original
    assert hashlib.sha256((HERE/'test_siglip2_connected_mlp.py').read_bytes()).hexdigest() == '8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25'
    tree = ast.parse(DRIVER.read_bytes())
    historical = function(tree,'admit_historical_mlp_actual_gradient')
    old = function(ast.parse(original),'admit_actual_gradient')
    projected = copy.deepcopy(historical)
    projected.name = 'admit_actual_gradient'
    for node in ast.walk(projected):
        if isinstance(node,ast.Name) and node.id == 'HISTORICAL_MLP': node.id = 'MLP'
        if isinstance(node,ast.Constant) and node.value == 'historical_mlp_actual_gradient': node.value = 'actual_gradient'
    assert ast.dump(projected,include_attributes=False) == ast.dump(old,include_attributes=False), 'historical admission predicate changed'
    assert 'PROBE' not in {n.id for n in ast.walk(historical) if isinstance(n,ast.Name)}
    assert d.PROBE_SOURCE_SHA == '274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31'
    update = function(tree,'update')
    assert "total_gradient_norms" in ast.unparse(update)
    assert "frozen447 actual gradients must be absent" in ast.unparse(update)
    assert '25%' not in DRIVER.read_text() and 'norm10' not in DRIVER.read_text()
    for before,after in ((b'<= 1.50',b'<= 1.51'),(b"for view in VIEWS:",b"for view in VIEWS[:1]:"),
                         (b'features.requires_grad',b'True'),(b'p.grad is None for n,p',b'True for n,p')):
        assert before in DRIVER.read_bytes()
        try: probe_inverse(DRIVER.read_bytes().replace(before,after))
        except AssertionError: pass
        else: raise AssertionError('inverse accepted unrelated mutation')
    probe_native_source_seam(d)
    probe_gradient_seam(d)
    probe_substitution_seam(d)
    print('PASS probe finite inverse/historical-only admission/genuine source/actual two-view gradient/substitution seams')


def probe_native_source_seam(d):
    # Real compiled original head forward; symbolic tensor operations, no native numerics.
    import types
    path = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-pooling-probe-source-v1/modeling_siglip.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == d.PROBE_SOURCE_SHA
    name = 'transformers.models.siglip.modeling_siglip'
    module = ModuleType(name)
    module.__file__ = str(path)
    code = compile(raw,str(path),'exec',dont_inherit=True)
    classes = {}
    for cname in ('SiglipVisionModel','SiglipMultiheadAttentionPoolingHead'):
        cc = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == cname)
        cell = types.CellType()
        members = {}
        for n in ('__init__','forward'):
            fc = next(c for c in cc.co_consts if isinstance(c,CodeType) and c.co_name == n)
            members[n] = FunctionType(fc,vars(module),closure=(cell,) if fc.co_freevars else None)
        classes[cname] = type(cname,(),members)
        cell.cell_contents = classes[cname]
        setattr(module,cname,classes[cname])
    model_cls = classes['SiglipVisionModel']
    model_cls.forward.__defaults__ = (False,)
    decorators = probe_decorator_fixture(module,model_cls)
    probe_decorator_admission_seam(d,decorators)
    model = object.__new__(model_cls)
    model.head = object.__new__(classes['SiglipMultiheadAttentionPoolingHead'])
    model.use_head = True
    model.config = SimpleNamespace(hidden_size=1152)
    class Query:
        shape = (1,1,1152)
        def repeat(self,batch,x,y): return ('live-original-query',self,batch)
    query = Query()
    model.head.probe = query
    model.named_parameters = lambda:iter([('head.probe',query)])
    guards = {str(path):d.PROBE_SOURCE_SHA}
    packages = {'transformers':{'root':str(path.parent)}}
    with patch.dict(sys.modules,{name:module}), patch.object(d,'_probe_decorators',lambda *a:decorators):
        d.probe_source(model,packages,guards)
        class Hidden:
            shape = (2,256,1152)
            def __add__(self,other): assert other is self; return self
            def __getitem__(self,key): assert key == (slice(None),0); return 'token0'
        hidden = Hidden()
        def attention(q,k,v):
            assert q == ('live-original-query',query,2) and k is v is hidden
            return (hidden,None)
        model.head.attention = attention
        model.head.layernorm = lambda h:h
        model.head.mlp = lambda h:h
        assert model.head.forward(hidden) == 'token0'
        for cls in classes.values():
            for method in ('__init__','forward'):
                with patch.object(cls,method,lambda *a:None):
                    rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
        probe_wrapper_mutants(d,model,packages,guards,decorators)
        with patch.object(model.head,'forward',lambda *a:None):
            rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
        with patch.object(query,'shape',(1,4,1152)):
            rejects(lambda:d.probe_source(model,packages,guards),'single pooling probe')
        with patch.dict(guards,{str(path):'0'*64}):
            rejects(lambda:d.probe_source(model,packages,guards),'source guard')
        with patch.object(Query,'repeat',lambda *a:('detached-query',)):
            try: model.head.forward(hidden)
            except AssertionError: pass
            else: raise AssertionError('detached query reached original attention')


def probe_gradient_seam(d):
    # Execute the actual first-update view aggregation guards, with gradients at the seam.
    tree = ast.parse(DRIVER.read_bytes())
    update = function(tree,'update')
    view_if = next(n for n in ast.walk(update) if isinstance(n,ast.If) and
                   any(isinstance(c,ast.Name) and c.id == 'total_norms' for c in ast.walk(n)))
    class Grad:
        def __init__(self,value): self.value = value
        def double(self): return self
        def norm(self): return self.value
        def add_(self,other): self.value += other.value
    names = ['A','C','head.probe']
    frozen = SimpleNamespace(grad=None)
    state = {'model':SimpleNamespace(named_parameters=lambda:iter([('frozen',frozen)]))}
    ns = {**vars(d),'names':names,'state':state,'view_gradients':[],
          'ranking_total':[Grad(0) for _ in names],'step':1}
    code = compile(ast.Module(body=[view_if],type_ignores=[]),'<actual per-view gradient guards>','exec')
    def run(view,rank=1.,total=2.):
        ns.update(view=view,ranking=[Grad(rank) for _ in names],view_total=[Grad(total) for _ in names])
        exec(code,ns)
    for view in d.VIEWS: run(view)
    assert [v['view'] for v in ns['view_gradients']] == list(d.VIEWS)
    assert all(v['ranking_gradient_norms']['head.probe'] > 0 and v['total_gradient_norms']['head.probe'] > 0
               for v in ns['view_gradients'])
    rejects(lambda:run('canonical',rank=0.),'SmoothAP gradients')
    rejects(lambda:run('augmented',total=0.),'total gradients')
    rejects(lambda:run('augmented',total=float('nan')),'total gradients')
    frozen.grad = object()
    rejects(lambda:run('canonical'),'frozen447 actual gradients')
    # The full guard belongs before scaler clipping/step, inside each first view.
    assert ast.unparse(update).index('frozen447 actual gradients') < ast.unparse(update).index('scaler.unscale_')


def probe_substitution_seam(d):
    # Execute the actual same-numel mutation block, including finally restoration.
    node = function(ast.parse(DRIVER.read_bytes()),'portable_mutants')
    start = next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and
                 isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'nominated')
    body = node.body[start:-1]
    class Value:
        shape = (1,1,1152)
        is_leaf = True
        def __init__(self,value): self.value = value; self.data = self
        def numel(self): return 1152
        def detach(self): return self
        def reshape(self,shape): assert shape == self.shape; return self
        def clone(self): return Value(self.value)
        def copy_(self,other): self.value = other.value
    probe,wrong = Value(3.),Value(9.)
    params = {'head.probe':probe,'head.layernorm.weight':wrong}
    calls=[]
    def inference(endpoint,images):
        calls.append(probe.value)
        d.require(probe.value == 3.,'updated identity')
    ns = {**vars(d),'params':params,'original':SimpleNamespace(fingerprint=lambda v:str(v.value)),
          'torch':SimpleNamespace(equal=lambda a,b:a.value == b.value),
          'portable':SimpleNamespace(inference_outputs=inference),'endpoint':{},'images':[]}
    exec(compile(ast.Module(body=body,type_ignores=[]),'<actual distinct leaf mutant>','exec'),ns)
    assert calls == [9.] and probe.value == 3. and wrong.value == 9.
    with patch.dict(ns,portable=SimpleNamespace(inference_outputs=lambda *a:None)):
        rejects(lambda:exec(compile(ast.Module(body=body,type_ignores=[]),'<accepted mutant>','exec'),ns),'substitution accepted')
    assert probe.value == 3., 'failed mutant must restore'

DECORATOR_FIXTURES = {'generic.py': {'merge_with_config_defaults': (968, 'def merge_with_config_defaults(func):\n    """\n    Decorator using config field (if they exist) as default value for some args and kwargs. Precedence is always\n    given to the args/kwargs that are explicitly passed.\n    """\n\n    @wraps(func)\n    def wrapper(self, *args, **kwargs):\n        args_with_config_defaults = [\n            "use_cache",\n            "vision_feature_layer",\n            "vision_feature_select_strategy",\n            "vision_aspect_ratio",\n        ]\n        for arg_name in args_with_config_defaults:\n            arg_index = None\n            if arg_name in func.__code__.co_varnames:\n                arg_index = func.__code__.co_varnames.index(arg_name) - 1  # -1 for self\n\n            if arg_index is not None and len(args) > arg_index and args[arg_index] is not None:\n                arg_value = args[arg_index]\n            elif kwargs.get(arg_name) is not None:\n                arg_value = kwargs[arg_name]\n            else:\n                arg_value = getattr(self.config, arg_name, None)\n\n            if arg_value is not None:\n                # Arg-specific handling\n                if arg_name == "use_cache":\n                    if getattr(self, "gradient_checkpointing", False) and self.training and arg_value:\n                        logger.warning_once(\n                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."\n                        )\n                        arg_value = False\n                elif arg_name == "vision_feature_select_strategy":\n                    valid_strategies = ["default", "full"]\n                    if arg_value not in valid_strategies:\n                        raise ValueError(\n                            f"`Unexpected select feature strategy: {arg_value}. Please select from {valid_strategies}."\n                        )\n\n                if arg_index is not None and len(args) > arg_index:\n                    args = list(args)\n                    args[arg_index] = arg_value\n                    args = tuple(args)\n                else:\n                    kwargs[arg_name] = arg_value\n\n        # Maybe temporarily overwrite config value to create the correct mask - kwarg takes precedence\n        is_causal = kwargs.get("is_causal", getattr(self.config, "is_causal", None))\n        if is_causal is not None:\n            is_causal_in_config = hasattr(self.config, "is_causal")\n            if is_causal_in_config:\n                is_causal_original_value = self.config.is_causal\n            # Set it to both config and kwargs (it\'s needed in both, and can come from only 1 of the sources)\n            self.config.is_causal = is_causal\n            kwargs["is_causal"] = is_causal\n\n        # Call the original forward with the updated kwargs/config\n        try:\n            if kwargs.get("debug_io", False):\n                from ..model_debugging_utils import model_addition_debugger_context\n\n                with model_addition_debugger_context(\n                    self, kwargs.get("debug_io_dir", "model_debug"), kwargs.get("prune_layers")\n                ):\n                    output = func(self, *args, **kwargs)\n            else:\n                output = func(self, *args, **kwargs)\n        # Restore original config value\n        finally:\n            if is_causal is not None:\n                if is_causal_in_config:\n                    self.config.is_causal = is_causal_original_value\n                else:\n                    del self.config.is_causal\n\n        return output\n\n    return wrapper\n', '6bc988528d5780cccdb6d16981e3eb9f6f7e4f004adfd34f0da959b8cb086de1')}, 'output_capturing.py': {'capture_outputs': (205, 'def capture_outputs(func=None, *, tie_last_hidden_states=True):\n    """\n    Decorator to intercept specific layer outputs through hooks. The hooks are installed only once and lazily,\n    the first time output capture is requested with the `output_xxx` kwargs/config.\n    The implementation is fully context/thread safe, except when using `torch.compile`, as dynamo is unable to trace\n    through `ContextVar` methods.\n\n    Args:\n        tie_last_hidden_states (`bool`, *optional*, defaults to `True`):\n            Whether to overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`.\n            This is true for all language models and should be toggled off only if\n            `out.hidden_states[-1]` has to be the hidden state before last layer norm, which\n            is needed for some vision models (e.g. CLIP, SigLIP)\n    """\n\n    def wrapped_fn(func):\n        @wraps(func)\n        def wrapper(self, *args, **kwargs):\n            # Pop it so that internal modules always return a dict even if False is requested\n            return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))\n\n            # _can_record_outputs is None by default\n            capturable_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}\n            recordable_keys = {\n                f"output_{k}": kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))\n                for k in capturable_flags\n            }\n            # For BC as cross-attentions used to be captured with `output_attentions`\n            if "cross_attentions" in capturable_flags:\n                recordable_keys["output_cross_attentions"] = kwargs.get(\n                    "output_attentions", getattr(self.config, "output_attentions", False)\n                )\n            # The sam model variants need this annoying exception as well...\n            if "mask_decoder_attentions" in capturable_flags:\n                recordable_keys["output_mask_decoder_attentions"] = kwargs.get(\n                    "output_attentions", getattr(self.config, "output_attentions", False)\n                )\n\n            collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}\n            # Make sure hooks are installed if we need to collect outputs\n            if len(collected_outputs) > 0:\n                maybe_install_capturing_hooks(self)\n            # Let\'s activate the output collector hooks if needed!\n            output_token = _active_collector.set(collected_outputs)\n\n            # Run the forward\n            try:\n                outputs = func(self, *args, **kwargs)\n            # Reset the states\n            finally:\n                _active_collector.reset(output_token)\n\n            # Inject collected outputs into model output (return everything as tuples for BC)\n            for key in collected_outputs:\n                if key == "hidden_states":\n                    if not tie_last_hidden_states:\n                        pass\n                    elif hasattr(outputs, "vision_hidden_states"):\n                        collected_outputs[key] = collected_outputs[key][:-1]\n                        collected_outputs[key].append(outputs.vision_hidden_states)\n                    elif hasattr(outputs, "last_hidden_state"):\n                        collected_outputs[key] = collected_outputs[key][:-1]\n                        collected_outputs[key].append(outputs.last_hidden_state)\n\n                    outputs[key] = tuple(collected_outputs[key])\n                elif key == "attentions":\n                    # In this case, the second item are cross attentions\n                    if isinstance(capturable_flags[key], list) and len(capturable_flags[key]) == 2:\n                        outputs[key] = tuple(collected_outputs[key][0::2])\n                        outputs["cross_" + key] = tuple(collected_outputs[key][1::2])\n                    else:\n                        outputs[key] = tuple(collected_outputs[key])\n                else:\n                    outputs[key] = tuple(collected_outputs[key])\n\n            if return_dict is False:\n                outputs = outputs.to_tuple()\n\n            return outputs\n\n        return wrapper\n\n    if func is not None:\n        return wrapped_fn(func)\n    return wrapped_fn\n', 'c869baf015ad9a39d2baa0b6568b719109906135cbe5dbe1ee8e4fab78025e9e')}, 'auto_docstring.py': {'auto_docstring': (4364, 'def auto_docstring(obj=None, *, custom_intro=None, custom_args=None, checkpoint=None):\n    r"""\n    Automatically generates comprehensive docstrings for model classes and methods in the Transformers library.\n\n    This decorator reduces boilerplate by automatically including standard argument descriptions while allowing\n    overrides to add new or custom arguments. It inspects function signatures, retrieves predefined docstrings\n    for common arguments (like `input_ids`, `attention_mask`, etc.), and generates complete documentation\n    including examples and return value descriptions.\n\n    For complete documentation and examples, read this [guide](https://huggingface.co/docs/transformers/auto_docstring).\n\n    Examples of usage:\n\n        Basic usage (no parameters):\n        ```python\n        @auto_docstring\n        class MyAwesomeModel(PreTrainedModel):\n            def __init__(self, config, custom_parameter: int = 10):\n                r\'\'\'\n                custom_parameter (`int`, *optional*, defaults to 10):\n                    Description of the custom parameter for MyAwesomeModel.\n                \'\'\'\n                super().__init__(config)\n                self.custom_parameter = custom_parameter\n        ```\n\n        Using `custom_intro` with a class:\n        ```python\n        @auto_docstring(\n            custom_intro="This model implements a novel attention mechanism for improved performance."\n        )\n        class MySpecialModel(PreTrainedModel):\n            def __init__(self, config, attention_type: str = "standard"):\n                r\'\'\'\n                attention_type (`str`, *optional*, defaults to "standard"):\n                    Type of attention mechanism to use.\n                \'\'\'\n                super().__init__(config)\n        ```\n\n        Using `custom_intro` with a method, and specify custom arguments and example directly in the docstring:\n        ```python\n        @auto_docstring(\n            custom_intro="Performs forward pass with enhanced attention computation."\n        )\n        def forward(\n            self,\n            input_ids: Optional[torch.Tensor] = None,\n            attention_mask: Optional[torch.Tensor] = None,\n        ):\n            r\'\'\'\n            custom_parameter (`int`, *optional*, defaults to 10):\n                Description of the custom parameter for MyAwesomeModel.\n\n            Example:\n\n            ```python\n            >>> model = MyAwesomeModel(config)\n            >>> model.forward(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.tensor([1, 1, 1]))\n            ```\n            \'\'\'\n        ```\n\n        Using `custom_args` to define reusable arguments:\n        ```python\n        VISION_ARGS = r\'\'\'\n        pixel_values (`torch.FloatTensor`, *optional*):\n            Pixel values of the input images.\n        image_features (`torch.FloatTensor`, *optional*):\n            Pre-computed image features for efficient processing.\n        \'\'\'\n\n        @auto_docstring(custom_args=VISION_ARGS)\n        def encode_images(self, pixel_values=None, image_features=None):\n            # ... method implementation\n        ```\n\n        Combining `custom_intro` and `custom_args`:\n        ```python\n        MULTIMODAL_ARGS = r\'\'\'\n        vision_features (`torch.FloatTensor`, *optional*):\n            Pre-extracted vision features from the vision encoder.\n        fusion_strategy (`str`, *optional*, defaults to "concat"):\n            Strategy for fusing text and vision modalities.\n        \'\'\'\n\n        @auto_docstring(\n            custom_intro="Processes multimodal inputs combining text and vision.",\n            custom_args=MULTIMODAL_ARGS\n        )\n        def forward(\n            self,\n            input_ids,\n            attention_mask=None,\n            vision_features=None,\n            fusion_strategy="concat"\n        ):\n            # ... multimodal processing\n        ```\n\n        Using with ModelOutput classes:\n        ```python\n        @dataclass\n        @auto_docstring(\n            custom_intro="Custom model outputs with additional fields."\n        )\n        class MyModelOutput(ImageClassifierOutput):\n            r\'\'\'\n            loss (`torch.FloatTensor`, *optional*):\n                The loss of the model.\n            custom_field (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):\n                A custom output field specific to this model.\n            \'\'\'\n\n            # Standard fields like hidden_states, logits, attentions etc. can be automatically documented\n            # However, given that the loss docstring is often different per model, you should document it above\n            loss: Optional[torch.FloatTensor] = None\n            logits: Optional[torch.FloatTensor] = None\n            hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None\n            attentions: Optional[tuple[torch.FloatTensor, ...]] = None\n            custom_field: Optional[torch.FloatTensor] = None\n        ```\n\n    Args:\n        custom_intro (`str`, *optional*):\n            Custom introduction text to add to the docstring. This replaces the default\n            introduction text generated by the decorator before the Args section. Use this to describe what\n            makes your model or method special.\n        custom_args (`str`, *optional*):\n            Custom argument documentation in docstring format. This allows you to define\n            argument descriptions once and reuse them across multiple methods. The format should follow the\n            standard docstring convention: `arg_name (`type`, *optional*, defaults to `value`): Description.`\n        checkpoint (`str`, *optional*):\n            Checkpoint name to use in examples within the docstring. This is typically\n            automatically inferred from the model configuration class, but can be overridden if needed for\n            custom examples.\n\n    Note:\n        - Standard arguments (`input_ids`, `attention_mask`, `pixel_values`, etc.) are automatically documented\n          from predefined descriptions and should not be redefined unless their behavior differs in your model.\n        - New or custom arguments should be documented in the method\'s docstring using the `r\'\'\' \'\'\'` block\n          or passed via the `custom_args` parameter.\n        - For model classes, the decorator derives parameter descriptions from the `__init__` method\'s signature\n          and docstring.\n        - Return value documentation is automatically generated for methods that return ModelOutput subclasses.\n    """\n\n    def auto_docstring_decorator(obj):\n        if len(obj.__qualname__.split(".")) > 1:\n            return auto_method_docstring(\n                obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint\n            )\n        else:\n            return auto_class_docstring(obj, custom_args=custom_args, custom_intro=custom_intro, checkpoint=checkpoint)\n\n    if obj:\n        return auto_docstring_decorator(obj)\n\n    return auto_docstring_decorator\n', '5d4648d39bafc41ae8cc640227293868fff30ae13af86928a4eddb9845d25d5f'), 'auto_method_docstring': (4100, 'def auto_method_docstring(\n    func,\n    parent_class=None,\n    custom_intro=None,\n    custom_args=None,\n    checkpoint=None,\n    source_args_dict=None,\n    allowed_params=None,\n):\n    """\n    Wrapper that automatically generates docstring.\n    """\n\n    # Use inspect to retrieve the method\'s signature\n    sig = inspect.signature(func)\n    indent_level = get_indent_level(func) if not parent_class else get_indent_level(parent_class)\n\n    # Get model information\n    model_name_lowercase, class_name, config_class = _get_model_info(func, parent_class)\n    func_documentation = func.__doc__\n\n    if custom_args is not None and func_documentation is not None:\n        func_documentation = "\\n" + set_min_indent(custom_args.strip("\\n"), 0) + "\\n" + func_documentation\n    elif custom_args is not None:\n        func_documentation = "\\n" + set_min_indent(custom_args.strip("\\n"), 0)\n\n    # Add intro to the docstring before args description if needed\n    if custom_intro is not None:\n        docstring = set_min_indent(custom_intro, indent_level + 4)\n        if not docstring.strip().endswith("\\n"):\n            docstring += "\\n"\n    else:\n        docstring = add_intro_docstring(func, class_name=class_name, indent_level=indent_level)\n\n    # Process Parameters section\n    docstring += _process_parameters_section(\n        func_documentation,\n        sig,\n        func,\n        class_name,\n        model_name_lowercase,\n        parent_class,\n        indent_level,\n        source_args_dict,\n        allowed_params,\n    )\n\n    # Process Returns section\n    return_docstring, func_documentation = _process_returns_section(\n        func_documentation, sig, config_class, indent_level\n    )\n    docstring += return_docstring\n\n    # Process Example section\n    example_docstring = _process_example_section(\n        func_documentation,\n        func,\n        parent_class,\n        class_name,\n        model_name_lowercase,\n        config_class,\n        checkpoint,\n        indent_level,\n    )\n    docstring += example_docstring\n\n    # Format the docstring with the placeholders\n    docstring = format_args_docstring(docstring, model_name_lowercase)\n\n    # Assign the dynamically generated docstring to the wrapper function\n    func.__doc__ = docstring\n    return func\n', '301b344d6592fe2542db50dffd92236bdca73ac4b8fe2193c78ebfe4e15148b8')}}


def probe_decorator_fixture(model_module, model_cls):
    """Compile only exact captured function spans, then instantiate real wrapper factories."""
    from functools import wraps
    decorators = {}
    for filename,functions in DECORATOR_FIXTURES.items():
        name = 'transformers.utils.'+filename.removesuffix('.py')
        module = ModuleType(name)
        module.__file__ = '/fixture/'+filename
        module.wraps = wraps
        for member,(line,span,sha) in functions.items():
            assert hashlib.sha256(span.encode()).hexdigest() == sha
            code = compile('\n'*(line-1)+span,module.__file__,'exec',dont_inherit=True)
            expected = next(c for c in code.co_consts if isinstance(c,CodeType) and c.co_name == member)
            fn = FunctionType(expected,vars(module))
            setattr(module,member,fn)
            decorators[member] = (fn,expected,module)
    merge = decorators['merge_with_config_defaults'][0]
    capture = decorators['capture_outputs'][0]
    capture.__defaults__ = (None,)
    capture.__kwdefaults__ = {'tie_last_hidden_states':True}
    model_module.merge_with_config_defaults = merge
    model_module.capture_outputs = capture
    model_module.auto_docstring = decorators['auto_docstring'][0]
    # auto_method_docstring only changes __doc__ and returns its original func;
    # no native/doc generation runs in this test. Two runtime wrappers are real.
    auto_tree = ast.parse(DECORATOR_FIXTURES['auto_docstring.py']['auto_method_docstring'][1])
    returns = [n for n in ast.walk(auto_tree) if isinstance(n,ast.Return)]
    assert len(returns) == 1 and isinstance(returns[0].value,ast.Name) and returns[0].value.id == 'func'
    model_cls.forward = merge(capture(tie_last_hidden_states=False)(model_cls.forward))
    return decorators


def probe_wrapper_mutants(d,model,packages,guards,decorators):
    """Real probe_source rejects executable forgeries despite valid __wrapped__ chains."""
    from functools import wraps
    for cls in (type(model),type(model.head)):
        original = cls.forward
        @wraps(original)
        def forged(*a,**kw):
            raise AssertionError('forged outer forward executed')
        with patch.object(cls,'forward',forged):
            rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
    outer = type(model).forward
    inner = outer.__wrapped__
    raw = inner.__wrapped__
    # A valid wrapper's real closure can disagree with its harmless __wrapped__ decoy.
    for wrapper in (outer,inner):
        cells = dict(zip(wrapper.__code__.co_freevars,wrapper.__closure__,strict=True))
        cell = cells['func']
        original = cell.cell_contents
        @wraps(original)
        def forged_target(*a,**kw): return None
        try:
            cell.cell_contents = forged_target
            rejects(lambda:d.probe_source(model,packages,guards),'actual callable closure')
        finally:
            cell.cell_contents = original
        foreign = FunctionType(wrapper.__code__,dict(wrapper.__globals__),closure=wrapper.__closure__)
        wraps(wrapper.__wrapped__)(foreign)
        if wrapper is outer:
            with patch.object(type(model),'forward',foreign):
                rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
        else:
            cell = outer.__closure__[0]
            try:
                cell.cell_contents = foreign
                with patch.object(outer,'__wrapped__',foreign):
                    rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
            finally:
                cell.cell_contents = inner
    cells = dict(zip(inner.__code__.co_freevars,inner.__closure__,strict=True))
    flag = cells['tie_last_hidden_states']
    for bad in (True,0,0.,None):
        try:
            flag.cell_contents = bad
            rejects(lambda:d.probe_source(model,packages,guards),'wrapper flags')
        finally:
            flag.cell_contents = False
    with patch.object(type(model),'forward',raw):
        rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
    @wraps(raw)
    def forged_inner(*a,**kw): return None
    cell = outer.__closure__[0]
    try:
        cell.cell_contents = forged_inner
        with patch.object(outer,'__wrapped__',forged_inner):
            rejects(lambda:d.probe_source(model,packages,guards),'live probe circuit')
    finally:
        cell.cell_contents = inner
    d.probe_source(model,packages,guards)
    print('PASS genuine decorated vision/direct head: wraps forgeries, actual closure/flags/globals/base negatives')


def probe_decorator_admission_seam(d,decorators):
    """Execute actual factory predicates; source descriptors match original CPU-v5."""
    node = function(ast.parse(DRIVER.read_bytes()),'_probe_decorators')
    descriptors = ast.literal_eval(next(n.value for n in node.body if isinstance(n,ast.Assign) and
                                  isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'descriptors'))
    receipt = json.loads((HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/identity-diversity-v1/cpu-v5/receipt.json').read_text())
    for name,sha,_ in descriptors:
        suffix = name.replace('.','/')+'.py'
        guards = [value for path,value in receipt['input_guards'].items() if path.endswith('/'+suffix)]
        assert guards == [sha], 'decorator source not in original CPU-v5 guards'
    loop = next(n for n in ast.walk(node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == 'member')
    code = compile(ast.Module(body=[loop],type_ignores=[]),'<actual decorator factory predicates>','exec')
    from functools import wraps
    for member,(fn,expected,module) in decorators.items():
        ns = {**vars(d),'names':[member],'code':SimpleNamespace(co_consts=(expected,)),
              'module':module,'admitted':{}}
        exec(code,ns)
        assert ns['admitted'][member] == (fn,expected,module)
        @wraps(fn)
        def forged(*a,**kw): return None
        for bad in (forged,FunctionType(expected,dict(vars(module)))):
            with patch.object(module,member,bad):
                rejects(lambda:exec(code,ns),'live probe decorator factory')
    print('PASS actual decorator factory/code/globals predicates and original CPU-v5 full-source guards')

def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-only', action='store_true', required=True)
    p.parse_args()
    assert DRIVER.exists(), 'new trainer missing'
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location('_connected_source_test', DRIVER)
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    assert not {n.split('.')[0] for n in set(sys.modules)-before} & d.NATIVE
    probe_contract_falsifiers(d)
    source_contract(d)
    initializer_selection(d)
    initializer_runtime_falsifiers(d)
    authority_falsifiers(d)
    actual_gradient_scan_falsifier(d)
    historical_inverse_contract(d)
    lifetime_restore_math(d)
    overlay_optimizer_seams(d)
    owned_loader_admission(d)
    restore_seam(d)
    current_encoder_seam(d)
    processor_release_seam(d)
    cost_terminal_falsifiers(d)
    original_terminal_binding(d)
    print('PASS source-only connected single probe contracts/falsifiers; native UNRUN')


if __name__ == '__main__':
    main()
