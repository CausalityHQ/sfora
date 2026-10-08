#!/usr/bin/env python3
"""Owned connected last-MLP evaluation; source-only status is UNQUALIFIED.

FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}.
UNIT={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
      native_peak_rss_kib,both_locks_held:true}; normal exit is independently read.
CODE={root:canonical separate directory,execution_sha256:SHA,code:exact hashes}.
CLI: --execution-sha256 SHA --authority FILE --authority-sha256 SHA
     --phase cpu|export|score [--seed 179061|179069 --arm control|candidate]
     --output NEWDIR. execution.json contains exactly FILES. LAUNCH_KEYS is exact.
Endpoint={seed,arm,launch:FILE,terminal:UNIT,checkpoint:FILE,
          terminal_state_sha256,bundle:FILE,inference_state_sha256}.
TRAIN artifacts are result.checkpoint/result.parity.bundle; inference identity
is manifest.endpoint_state_sha256. Original base proof is never updated448 proof.
Complete CPU600 + paired same-seed mechanics1200 + fresh TRAIN128 (3000s) normal exits
and core/whole <=1.50 precede native/held reads. First061 CONTINUE precedes069;
selection same-four GO precedes unchanged VAL. No qualification is inferred.
Exports are complete B32 query/gallery (including tails), TWO sequential owned
public loads, raw/unit/codes/inverse/wire readbacks and updated-encoder oracle.
Parent must freeze actual TRAINING descriptor before native execution. No
historical evaluator qualification substitutes for this evaluator's CPU gate.
"""
import argparse
import ast
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time
from types import FunctionType, SimpleNamespace
import weakref

if not __debug__:
    raise SystemExit('qualification requires assertions')

UNIT_STARTED = time.perf_counter()
SCHEMA = 'siglip2-connected-mlp-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-connected-mlp-evaluation-launch-v1'
FILES = {'evaluate_siglip2_connected_mlp.py','test_connected_mlp_evaluation.py'}
TRAIN_FILES = {'train_siglip2_connected_mlp.py','test_siglip2_connected_mlp.py'}
# Parent supplied actual current freeze; future corrections require a new freeze.
TRAINING = {'root':'/home/riomus/runs/sfora-connected-mlp-train-source-v6',
    'execution_sha256':'a3d4e1e4ea76084a4036a1c7626b00353401adc3833838375974509aed2f2e8c',
    'code':{'train_siglip2_connected_mlp.py':'79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b',
            'test_siglip2_connected_mlp.py':'8391b3dd38a682dd327c0d0a3f0a62a5e699f7e934a6e39959600ed7f045bc25'}}
TRAINING_CPU = {'both_locks_held':True,'invocation_id':'230a11e4f1334336b6d195f60bc0bd7e',
    'log':{'path':'/home/riomus/runs/sfora-connected-mlp-train-source-v6/cpu-v6-original.log',
        'sha256':'a5c767cee5a689c5d0e0c29b4e355488e8350286033b77b2199e665d816dae14'},
    'native_peak_rss_kib':6426592,
    'receipt':{'path':'/home/riomus/runs/sfora-connected-mlp-cpu-v6/receipt.json',
        'sha256':'4ecd63f75a09ff1757a9a1bdf1e80c29865c5bfb75483c63f3a3e09bc9aeeaa8'},
    'service_seconds':433.509,'unit':'sfora-connected-mlp-cpu-v6'}
EVALUATOR_PINS = {
    'evaluate_siglip2_identity_diversity.py':'95cb8823236408537e04108fd63727fd3a323671f04eb33a6349b23a51ce638d',
    'test_identity_diversity_evaluation.py':'1cdbd7fef94f9f6812534e03cde9d009d5108b2a57d97cf09d9394d6046715e0'}
CONTROL_SHA256 = '1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280'
COST_POLICY = {'whole_service_ratio_max':1.50,'total_training_core_ratio_max':1.50,
    'core':'actual summed update windows: guards, decode/preprocess, complete gallery, both views, optimizer, integrity',
    'denominator':'fresh same-seed live frozen-encoder CONTROL TRAIN128 normal-exit UNIT',
    'phase_timings':'inclusive observations retained; no subtraction',
    'shared_export_in_ratios':False,'optimization_throughput_is_image_training_throughput':False}
LAUNCH_KEYS = {'schema','execution_sha256','training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference',
    'phase','arm','seed','stage','panel','endpoints','selected_cpu','exports','first_selection','selection_go',
    'resource_policies','cost_policy','both_locks_held','selection_previously_exposed','scope'}
READINESS = ('training_units','matched_costs','cpu_qualification','source_replay','concat_replay',
    'updated_state','wire_readbacks','bundle_portability')
MEMBERS = ('config','buffers','processor','head','A','means','C','mu_train','mu_train_provenance',
    'scope','common_statistics','base_vision','encoder','encoder_identity','arm')


# BEGIN ENDPOINT READER AUTHENTICATION
# Capture the interpreter bindings at evaluator import, before helper admission.
_SOURCE_BUILTINS = tuple(vars(__import__('builtins')).items())

FIRST_SELECTION_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v5',
    'execution_sha256':'a4ca55fadf9d0dd5a87d4c4163c374e88a5f21abc8a4553588434c5e8273bf6a',
    'code':{'evaluate_siglip2_connected_mlp.py':'919a05d0f3de2eeb3b99e4a8da9519992082881eb2b84ddc4a257e75c3ca1b69',
            'test_connected_mlp_evaluation.py':'df1e233279e04bacd64b6bbd47361d43d350740fd496434e79de7b53b9f66ccb'}}
FIRST_SELECTION_UNIT = {'both_locks_held':True,'invocation_id':'c47869c3b2d24e70a2213545e371265e',
    'log':{'path':FIRST_SELECTION_OWNER['root']+'/first-selection-score-v2-original.log',
        'sha256':'1db5215b0f7700b56283d76266edd9cba7b1e7a234f4af0e2da2e293548812f3'},
    'native_peak_rss_kib':1090416,
    'receipt':{'path':'/home/riomus/runs/sfora-connected-mlp-evaluation-first-selection-score-v2/receipt.json',
        'sha256':'bc80bef471cd05c7ab842258a9b86f3e8813f5dabc5235e2202399f5c607028d'},
    'service_seconds':617.915,'unit':'sfora-connected-mlp-evaluation-first-selection-score-v2'}


def _capture_source_builtins(function):
    """Keep the import-time baseline out of the mutable helper-admission globals."""
    canonical = _SOURCE_BUILTINS
    error,code = ValueError,function.__code__
    def source_live_guard(module, digest, guards, names=None, class_name=None):
        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:
            raise error('authenticated builtin baseline/source binding changed')
        return function(canonical,module,digest,guards,names,class_name,None)
    def initializer_live_guard(context):
        if _SOURCE_BUILTINS is not canonical or function.__code__ is not code:
            raise error('authenticated builtin baseline/source binding changed')
        t = context['training_context']
        return function(canonical,t['legacy']['selected']['genuine']['reference'],
            '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8',
            t['guards'],{'admit_cgroup','require'},None,context)
    source_live_guard.initializer = initializer_live_guard
    return source_live_guard


@_capture_source_builtins
def source_live_guard(baseline, module, digest, guards, names, class_name, initializer_owner):
    """Independent lexical source/runtime binding, including genuine class methods."""
    canonical = {key:value for key,value in baseline}
    builtin_namespace = canonical['__import__']('builtins').__dict__
    namespaces = [module.__dict__,canonical['globals']()]
    error = canonical['ValueError']
    def builtin_guard():
        # No global/builtin calls: even all/any/type/ValueError may have changed.
        if _SOURCE_BUILTINS is not baseline:
            raise error('authenticated builtin baseline binding changed')
        for key,value in canonical.items():
            if builtin_namespace.get(key) is not value:
                raise error('authenticated builtin binding changed: '+key)
            if key not in ('__name__','__doc__','__package__','__loader__','__spec__'):
                for namespace in namespaces:
                    if key in namespace:
                        raise error('authenticated builtin shadow: '+key)
    def require(condition, message):
        if not condition:
            raise error(message)
    builtin_guard()
    import builtins
    from types import ModuleType
    require(type(module) is ModuleType and vars(module).get('__builtins__') is vars(builtins),
        'authenticated module/builtins required')
    path = Path(module.__file__); spec = module.__spec__
    if initializer_owner is None:
        def registry_guard():
            return sys.modules.get(module.__name__) is module
    else:
        # The original exporter never registers this one module. Its actual
        # holder objects, captured independently of mutable snapshots, own it.
        from importlib.machinery import SourceFileLoader
        t = initializer_owner['training_context']; legacy = t['legacy']
        selected = legacy['selected']; genuine = selected['genuine']; admission = legacy['admission']
        fit_context = t['fit_context']; original = legacy['original']; Flat = original.FlatAdmission
        require(module.__name__ == '_genuine_fit_reference' and
            path == Path('/home/riomus/runs/sfora-native256-fit-export-source-v1/export_siglip2_substrate_fit.py') and
            spec is not None and type(spec.loader) is SourceFileLoader and module.__loader__ is spec.loader and
            spec.loader.name == module.__name__ and spec.loader.path == str(path) and
            type(admission) is Flat, 'authenticated initializer original source/loader differs')
        loader_name,loader_path = spec.loader.name,spec.loader.path
        def registry_guard():
            require(initializer_owner['training_context'] is t and t['legacy'] is legacy and
                t['fit_context'] is fit_context and fit_context['legacy'] is legacy and
                legacy['selected'] is selected and selected['genuine'] is genuine and
                genuine['reference'] is module and legacy['admission'] is admission and admission.init is module and
                legacy['original'] is original and original.FlatAdmission is Flat and type(admission) is Flat,
                'authenticated initializer owner binding changed')
            require(t['guards'] is guards and guards.get(str(path)) == digest,
                'authenticated initializer source digest changed')
            require(module.__loader__ is spec.loader and vars(spec.loader) == {'name':loader_name,'path':loader_path},
                'authenticated initializer loader changed')
            return module.__name__ not in sys.modules and all(value is not module for value in sys.modules.values())
    require(spec is not None and spec.loader is not None and spec.name == module.__name__ and
        Path(spec.origin) == path and registry_guard(),
        'authenticated module registry/origin differs')
    raw = bound_file(guards,path,digest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'authenticated source changed before compilation')
    tree = ast.parse(raw,filename=str(path)); compiled = compile(raw,str(path),'exec',dont_inherit=True)
    if initializer_owner is not None:
        # This pinned initializer has only stdlib imports and constant assignments.
        expected_globals = {}
        nodes = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign))]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',dont_inherit=True),expected_globals)
        expected_names = set(expected_globals) | {n.name for n in tree.body if isinstance(n,ast.FunctionDef)} | {
            '__name__','__doc__','__package__','__loader__','__spec__','__file__','__cached__'}
        require(vars(module).keys() == expected_names and all(type(vars(module)[k]) is type(v) and
            vars(module)[k] == v for k,v in expected_globals.items()), 'authenticated initializer source globals differ')
    values = dict(vars(module)); literals = {k:copy.deepcopy(v) for k,v in values.items()
        if k != '__builtins__' and isinstance(v,(dict,list,tuple,set,frozenset))}
    loader,name = spec.loader,module.__name__; functions = []; classes = []
    scopes = [(module,tree.body,compiled)]
    if class_name is not None:
        node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == class_name)
        cls = getattr(module,class_name)
        require(type(cls) is type and cls.__module__ == name and cls.__qualname__ == class_name and
            cls.__bases__ == (object,), 'genuine reader class differs')
        # Compile the definition alone in a minimal namespace to include the
        # interpreter's own class metadata (including Python 3.13 attributes).
        template_namespace = {'__name__':name}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',dont_inherit=True),template_namespace)
        template_class = template_namespace[class_name]
        require(vars(cls).keys() == vars(template_class).keys() and cls.__doc__ == template_class.__doc__ and
            all(type(vars(cls)[k]) is type(v) for k,v in vars(template_class).items()) and
            all(getattr(cls,k,None) == getattr(template_class,k,None)
                for k in ('__firstlineno__','__static_attributes__')), 'genuine reader class inventory differs')
        for key,value in vars(template_class).items():
            if isinstance(value,property):
                require(vars(cls)[key].fset is None and vars(cls)[key].fdel is None,
                    'genuine reader property descriptor differs')
        classes.append((cls,dict(vars(cls)),cls.__module__,cls.__qualname__,cls.__bases__))
        scopes.append((cls,node.body,next(c for c in compiled.co_consts if getattr(c,'co_name',None) == class_name)))
    for owner,nodes,code in scopes:
        for node in nodes:
            if not isinstance(node,ast.FunctionDef) or (owner is module and names is not None and node.name not in names):
                continue
            fn = getattr(owner,node.name); wrapper = None
            if owner is module and node.decorator_list:
                require([ast.unparse(d) for d in node.decorator_list] == ['contextmanager'],
                    'unexpected authenticated function decorator')
                wrapper,fn = fn,fn.__wrapped__
                namespaces.append(wrapper.__globals__)
                template = contextmanager(fn)
                require(wrapper.__code__ is template.__code__ and wrapper.__globals__ is template.__globals__ and
                    wrapper.__builtins__ is vars(builtins),
                    'authenticated contextmanager wrapper differs')
            if isinstance(fn,property):
                fn = fn.fget
            expected = next(c for c in code.co_consts if getattr(c,'co_name',None) == node.name)
            evaluate = lambda n: eval(compile(ast.Expression(n),str(path),'eval'),{'__builtins__':{}},vars(module))
            defaults = tuple(evaluate(n) for n in node.args.defaults) or None
            kw = {a.arg:evaluate(n) for a,n in zip(node.args.kwonlyargs,node.args.kw_defaults) if n is not None} or None
            qualified = node.name if owner is module else class_name+'.'+node.name
            require(type(fn) is FunctionType and fn.__globals__ is vars(module) and fn.__code__ == expected and
                fn.__code__.co_filename == str(path) and fn.__module__ == name and fn.__closure__ is None and
                fn.__builtins__ is vars(builtins) and
                fn.__name__ == node.name and fn.__qualname__ == qualified and
                fn.__defaults__ == defaults and fn.__kwdefaults__ == kw,
                'authenticated source function/defaults differ: '+node.name)
            functions.append((owner,node.name,fn,fn.__code__,copy.deepcopy(defaults),copy.deepcopy(kw),
                qualified,wrapper,wrapper.__code__ if wrapper is not None else None))
    def guard():
        builtin_guard()
        require(module.__name__ == name and registry_guard() and module.__spec__ is spec and
            spec.name == name and spec.loader is loader and Path(spec.origin) == Path(module.__file__) == path and
            vars(module).keys() == values.keys() and all(vars(module)[k] is v for k,v in values.items()) and
            all(vars(module)[k] == v for k,v in literals.items()), 'authenticated module/global binding changed')
        for cls,members,owner_name,qualified,bases in classes:
            require(vars(cls).keys() == members.keys() and all(vars(cls)[k] is v for k,v in members.items()),
                'authenticated reader class binding changed')
            require((cls.__module__,cls.__qualname__,cls.__bases__) == (owner_name,qualified,bases),
                'authenticated reader class metadata changed')
        for owner,key,fn,code,defaults,kw,qualified,wrapper,wrapper_code in functions:
            actual = getattr(owner,key)
            if wrapper is not None:
                require(actual is wrapper and wrapper.__code__ is wrapper_code and wrapper.__wrapped__ is fn and
                    wrapper.__globals__ is contextmanager(fn).__globals__ and
                    wrapper.__builtins__ is vars(builtins) and
                    (wrapper.__module__,wrapper.__name__,wrapper.__qualname__) == (name,key,qualified) and
                    wrapper.__defaults__ is None and wrapper.__kwdefaults__ is None and
                    len(wrapper.__closure__) == 1 and wrapper.__closure__[0].cell_contents is fn,
                    'authenticated function wrapper changed')
                actual = wrapper.__wrapped__
            if isinstance(actual,property):
                actual = actual.fget
            require(actual is fn and fn.__code__ is code and fn.__globals__ is vars(module) and
                fn.__builtins__ is vars(builtins) and
                (fn.__module__,fn.__name__,fn.__qualname__) == (name,key,qualified) and
                fn.__defaults__ == defaults and fn.__kwdefaults__ == kw,
                'authenticated live function/defaults changed: '+key)
        bound_file({},path,digest)
    guard()
    return guard


def batch_terminal_files(Flat, reader, guards, items):
    """Fresh private original reader per occurrence; join, then publish in order.

    Deliberately stronger than verified-set skips: reread even cached JSON paths.
    Failure leaves all owner state unchanged; existing json_bytes is never touched.
    """
    require(type(reader) is Flat, 'genuine terminal reader required')
    require(vars(reader).keys() <= {'entries','verified','json_bytes','init'} and
        type(reader.entries) is dict and type(reader.verified) is set and type(reader.json_bytes) is dict,
        'genuine terminal reader instance state required')
    def one(item):
        fresh = Flat()
        path = fresh.bound_file({},*item)
        return str(path),fresh.entries[str(path)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one,item) for item in items]
        results = [future.result() for future in futures]
    entries,verified,staged = dict(reader.entries),set(reader.verified),dict(guards)
    for path,fact in results:
        require(entries.setdefault(path,fact) == fact, 'conflicting file SHA256/size authority')
        require(staged.setdefault(path,fact[0]) == fact[0], 'conflicting stage file authority')
        verified.add(path)
    reader.entries.update(entries); reader.verified.update(verified); guards.update(staged)


def load_endpoint_reader(context):
    """Own only the frozen v6 terminal input loop; retain every genuine callback."""
    import builtins
    trainer,t = context['trainer'],context['training_context']
    original = t['legacy']['original']; Flat = original.FlatAdmission
    source_guard = source_live_guard(trainer,TRAINING['code']['train_siglip2_connected_mlp.py'],context['guards'])
    flat_guard = source_live_guard(original,'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543',
        t['guards'],class_name='FlatAdmission')
    nearest = t['nearest']
    nearest_guard = source_live_guard(nearest,t['guards'][nearest.__file__],t['guards'],
        names={'read_json','file_fact','bound_file','strict_json','require'})
    # Authenticate the terminal callback's genuine source modules, without replaying
    # native-origin payload reads. Original native/exit guards still run unchanged.
    initializer = t['legacy']['selected']['genuine']['reference']
    terminal_guards = tuple(source_live_guard(m,t['guards'][m.__file__],t['guards'],names=names)
        for m,names in ((t['trainer'],{'check_steps','check_ranking_bank','ranking_membership',
                                      'ranking_bank','json_sha256','require'}),
                       (t['fitter'],None),(t['old'],{'zero_events','require'}))) + (source_live_guard.initializer(context),)
    node = next(n for n in ast.parse(Path(trainer.__file__).read_bytes()).body
        if isinstance(n,ast.FunctionDef) and n.name == 'admit_terminal')
    dump = lambda n: ast.dump(n,include_attributes=False)
    require(hashlib.sha256(dump(node).encode()).hexdigest() ==
        '1761ee02b913176ebc7f249f509801ac0ed527f1ddc480755bd0bef521f61f8b', 'frozen terminal AST differs')
    old = copy.deepcopy(node)
    loop = ast.parse("for path,digest in record['input_guards'].items():\n    reader.bound_file(guards,path,digest)").body[0]
    indices = [i for i,n in enumerate(node.body) if dump(n) == dump(loop)]
    require(len(indices) == 1, 'exact terminal input loop required')
    index = indices[0]
    node.body[index:index+1] = ast.parse("_batch_terminal_files(_Flat,reader,guards,record['input_guards'].items())").body
    node.name = 'connected_endpoint_terminal'
    restored = copy.deepcopy(node); restored.name = old.name; restored.body[index] = old.body[index]
    require(dump(restored) == dump(old), 'terminal derivative changed predicates')
    callbacks = {name:getattr(trainer,name) for name in
        ('check_unit','fresh_terminal_reader','check_terminal','require','cli','policy','read_json')}
    namespace = {'__name__':__name__,'Path':Path,**callbacks,'_batch_terminal_files':batch_terminal_files,'_Flat':Flat}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),namespace)
    derivative = namespace['connected_endpoint_terminal']; code = derivative.__code__; bindings = dict(namespace)
    owned_names = ('batch_terminal_files','require','bound_file')
    owned = (batch_terminal_files,require,bound_file)
    evaluator_path = Path(__file__)
    raw = bound_file({},evaluator_path,context['code']['evaluate_siglip2_connected_mlp.py']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == context['code']['evaluate_siglip2_connected_mlp.py'],
        'endpoint evaluator source changed before compilation')
    compiled = compile(raw,__file__,'exec',dont_inherit=True)
    for name,fn in zip(owned_names,owned):
        require(type(fn) is FunctionType and fn.__globals__ is globals() and fn.__builtins__ is vars(builtins) and
            (fn.__module__,fn.__name__,fn.__qualname__) == (__name__,name,name) and fn.__closure__ is None and
            fn.__code__ == next(c for c in compiled.co_consts if getattr(c,'co_name',None) == name) and
            fn.__defaults__ is None and fn.__kwdefaults__ is None,
            'endpoint owned callback source differs')
    owned_codes = tuple(fn.__code__ for fn in owned)
    owned_globals = {k:globals()[k] for k in (*owned_names,'ThreadPoolExecutor','Path','hashlib','os','__name__','__file__','__builtins__')}
    dependencies = {key:t[key] for key in ('trainer','nearest','fitter','old','fit_context','legacy')}
    def guard(current):
        source_guard()
        if (any(fn.__code__ is not c or fn.__defaults__ is not None or fn.__kwdefaults__ is not None or
                fn.__builtins__ is not vars(builtins) or (fn.__module__,fn.__name__,fn.__qualname__) != (__name__,name,name)
                for name,fn,c in zip(owned_names,owned,owned_codes)) or any(globals()[k] is not v for k,v in owned_globals.items())):
            raise ValueError('endpoint owned callback/global changed')
        require(current['trainer'] is trainer and current['training_context'] is t and
            all(t[k] is v for k,v in dependencies.items()) and t['legacy']['original'] is original and
            original.FlatAdmission is Flat and t['legacy']['selected']['genuine']['reference'] is initializer and
            t['legacy']['admission'].init is initializer,
            'endpoint context/source binding changed')
        flat_guard(); nearest_guard()
        for check in terminal_guards:
            check()
        require(derivative.__code__ is code and derivative.__defaults__ is None and derivative.__kwdefaults__ is None and
            derivative.__name__ == derivative.__qualname__ == 'connected_endpoint_terminal' and
            derivative.__module__ == __name__ and derivative.__code__.co_filename == __file__ and
            derivative.__builtins__ is vars(builtins) and
            derivative.__globals__ is namespace and namespace.keys() == bindings.keys() and
            all(namespace[k] is v for k,v in bindings.items()), 'endpoint derivative binding changed')
    def admit(unit,phase,arm,seed):
        guard(context)
        try:
            return derivative(t,unit,phase,arm,seed)
        finally:
            guard(context)
    guard(context)
    return admit,guard


def load_first_selection(context):
    """Only the exact source-v5 first CONTINUE prerequisite, once per invocation."""
    fact,unit = copy.deepcopy(FIRST_SELECTION_OWNER),copy.deepcopy(FIRST_SELECTION_UNIT)
    require(context['launch']['stage'] == 'full' and context['launch']['first_selection'] == unit,
        'exact original first-selection prerequisite required')
    require(closure(fact['root'],fact['execution_sha256'],FILES,context['guards']) == fact['code'],
        'first-selection owner exact2 differs')
    original = load_authenticated('_connected_first_owner_v5',Path(fact['root'])/'evaluate_siglip2_connected_mlp.py',
        fact['code']['evaluate_siglip2_connected_mlp.py'],context['guards'])
    live_guard = source_live_guard(original,fact['code']['evaluate_siglip2_connected_mlp.py'],context['guards'])
    context['first_evaluator'] = original
    guard_helpers(context)
    authenticated = tuple((m,p,s,dict(v),tuple((fn,c,copy.deepcopy(d),copy.deepcopy(kw)) for fn,c,d,kw in f),copy.deepcopy(l))
        for m,p,s,v,f,l in context['helper_snapshots'])
    used = False
    def guard(current):
        live_guard()
        actual = current['helper_snapshots']
        require(FIRST_SELECTION_OWNER == fact and FIRST_SELECTION_UNIT == unit and current['first_evaluator'] is original and
            len(actual) == len(authenticated) and all(a[0] is b[0] and a[1] == b[1] and a[2] is b[2] and
                a[3].keys() == b[3].keys() and all(a[3][k] is v for k,v in b[3].items()) and
                tuple(a[4]) == b[4] and a[5] == b[5] for a,b in zip(actual,authenticated,strict=True)),
            'first-selection owner/snapshot binding changed')
        guard_helpers(current)
        require(closure(fact['root'],fact['execution_sha256'],FILES,{}) == fact['code'],
            'first-selection fresh owner closure differs')
    def owner_context(current):
        common = current['common_guards']
        remove = {str(current['root']/'execution.json'):current['args'].execution_sha256,
            **{str(current['root']/n):h for n,h in current['code'].items()}}
        insert = {str(Path(fact['root'])/'execution.json'):fact['execution_sha256'],
            **{str(Path(fact['root'])/n):h for n,h in fact['code'].items()}}
        require(current['code'].keys() == FILES and len(remove) == len(insert) == 3 and
            remove.keys().isdisjoint(insert) and common.keys().isdisjoint(insert) and
            all(common.get(p) == h for p,h in remove.items()), 'first-selection exact three closure swap required')
        return {**current,'root':Path(fact['root']),
            'args':SimpleNamespace(**{**vars(current['args']),'execution_sha256':fact['execution_sha256']}),
            'code':fact['code'],'common_guards':{**{p:h for p,h in common.items() if p not in remove},**insert}}
    def admit(current, selected, phase, *, stage, panel):
        nonlocal used
        guard(current)
        require(not used and current is context and current['launch']['stage'] == 'full' and
            current['launch']['first_selection'] == selected == unit and
            (phase,stage,panel) == ('score','first','selection'), 'first-selection once-only prerequisite role required')
        used = True
        owner = owner_context(current)
        try:
            record = original.accept_unit(owner,selected,'score',stage='first',panel='selection')
            expected = owner_context(current); specific = {'root','args','code','common_guards'}
            require(owner.keys() == current.keys() and all(owner[k] is current[k] for k in owner.keys()-specific) and
                all(owner[k] == expected[k] for k in specific-{'args'}) and vars(owner['args']) == vars(expected['args']),
                'first-selection shared owner context changed')
            require(record['decision'] == 'CONTINUE' and record['launch']['endpoints'] == current['launch']['endpoints'][:2],
                'first061 KILL prohibits069 and VAL')
            return record
        finally:
            guard(current)
    guard(context)
    return admit,guard


def admission_exit_guard(*guards):
    def guard(context):
        for check in guards:
            if check is not None:
                check(context)
    return guard
# END ENDPOINT READER AUTHENTICATION


# BEGIN ORIGINAL EXPORT OWNER
ORIGINAL_EXPORT_OWNER = {'root':'/home/riomus/runs/sfora-connected-mlp-evaluation-source-v3',
    'execution_sha256':'c76330fe0bb0cefa5f3ec7d3a4e82785ffc7620bae6583ddf55af67b6aca651f',
    'code':{'evaluate_siglip2_connected_mlp.py':'bce0c43bae6d24f50ab8ce7c60f8410abd307697ec81d91825e0a043a20efe08',
            'test_connected_mlp_evaluation.py':'7a1eef346f0999c277d0d6c934141d24db693bb0516af985547efb97ee7cd5e9'}}
ORIGINAL_SCORE_AUTHORITY = {'path':ORIGINAL_EXPORT_OWNER['root']+'/authority-first-selection-score-v1.json',
    'sha256':'d25da1234f661f79437f6708769a4de21c61116f7b5639c540bc26c7032cb160'}
ORIGINAL_EXPORT_UNITS = {'candidate-179061': {'both_locks_held': True,
                      'invocation_id': '06071c350b134c0991b12f45ba1550c8',
                      'log': {'path': '/home/riomus/runs/sfora-connected-mlp-evaluation-source-v3/export-candidate-179061-v1-original.log',
                              'sha256': 'f00017484e2e0dc57b40afe821f228d4cf8485076e9fd6371dfdc1af37852d9a'},
                      'native_peak_rss_kib': 4206904,
                      'receipt': {'path': '/home/riomus/runs/sfora-connected-mlp-evaluation-export-candidate-179061-v1/receipt.json',
                                  'sha256': 'd43ef0d5ad435a90ab584f91ef9efa31e002a04ecb2aef3253a96062e7833bf0'},
                      'service_seconds': 1289.351,
                      'unit': 'sfora-connected-mlp-evaluation-export-candidate-179061-v1'},
 'control-179061': {'both_locks_held': True,
                    'invocation_id': 'c8a9ba3f025744519d3e6411e3809161',
                    'log': {'path': '/home/riomus/runs/sfora-connected-mlp-evaluation-source-v3/export-control-179061-v2-original.log',
                            'sha256': 'a917e632f87b25cdca6ad5ed3d4b11dfc0fe98dffa0baf2b749d7a54bc2cd6fe'},
                    'native_peak_rss_kib': 4207316,
                    'receipt': {'path': '/home/riomus/runs/sfora-connected-mlp-evaluation-export-control-179061-v2/receipt.json',
                                'sha256': 'dea4527b76ce6c1c12f5e1d64dd2e5b1cace371a432a114ca1e7f18d1cc4f264'},
                    'service_seconds': 1264.694,
                    'unit': 'sfora-connected-mlp-evaluation-export-control-179061-v2'}}


def load_original_owner(context):
    """Authenticate only the pinned historical inputs, never the failed score."""
    fact = ORIGINAL_EXPORT_OWNER; guards = context['guards']
    require(closure(fact['root'],fact['execution_sha256'],FILES,guards) == fact['code'],
        'original export owner exact2 differs')
    original = load_authenticated('_connected_export_owner_v3',Path(fact['root'])/'evaluate_siglip2_connected_mlp.py',
        fact['code']['evaluate_siglip2_connected_mlp.py'],guards)
    launch = read_json(ORIGINAL_SCORE_AUTHORITY,guards)
    original.check_launch(launch,SimpleNamespace(execution_sha256=fact['execution_sha256'],phase='score',arm=None,seed=None))
    require(launch['stage'] == 'first' and launch['panel'] == 'selection' and
        launch['exports'] == ORIGINAL_EXPORT_UNITS and launch == {**context['launch'],
            'execution_sha256':fact['execution_sha256'],'selected_cpu':launch['selected_cpu'],
            'resource_policies':{**context['launch']['resource_policies'],
                'score':{**context['launch']['resource_policies']['score'],'seconds':500}}},
        'original selection endpoint/procedure differs')
    context.update(original_evaluator=original,original_launch=launch)
    guard_helpers(context)
    # The authority holds this closure locally, outside replaceable context slots.
    authenticated = tuple((m,p,s,dict(v),tuple((fn,c,copy.deepcopy(d),copy.deepcopy(kw)) for fn,c,d,kw in f),copy.deepcopy(l))
        for m,p,s,v,f,l in context['helper_snapshots'])
    def original_guard(current):
        actual = current['helper_snapshots']
        require(current['original_evaluator'] is original and len(actual) == len(authenticated) and
            all(a[0] is b[0] and a[1] == b[1] and a[2] is b[2] and a[3].keys() == b[3].keys() and
                all(a[3][k] is v for k,v in b[3].items()) and tuple(a[4]) == b[4] and a[5] == b[5]
                for a,b in zip(actual,authenticated,strict=True)), 'authenticated owner/snapshot binding changed')
        guard_helpers(current)
    return original_guard


def original_owner_context(context, original_guard):
    """Six owner-specific members; preserve the authenticated shared snapshot/ledger."""
    original_guard(context)
    require(context['original_launch'] == read_json(ORIGINAL_SCORE_AUTHORITY,context['guards']),
        'original score input authority changed')
    fact = ORIGINAL_EXPORT_OWNER; common = context['common_guards']
    current = {str(context['root']/'execution.json'):context['args'].execution_sha256,
        **{str(context['root']/n):h for n,h in context['code'].items()}}
    historical = {str(Path(fact['root'])/'execution.json'):fact['execution_sha256'],
        **{str(Path(fact['root'])/n):h for n,h in fact['code'].items()}}
    require(len(current) == len(historical) == 3 and current.keys().isdisjoint(historical) and
        all(common.get(p) == h for p,h in current.items()) and common.keys().isdisjoint(historical),
        'exact three owner closure snapshot entries required')
    owner_common = {p:h for p,h in common.items() if p not in current}
    merge_guards(owner_common,historical)
    owner = {**context,'root':Path(fact['root']),
        'args':SimpleNamespace(execution_sha256=fact['execution_sha256'],authority=Path(ORIGINAL_SCORE_AUTHORITY['path']),
            authority_sha256=ORIGINAL_SCORE_AUTHORITY['sha256'],phase='score',arm=None,seed=None,output=context['args'].output),
        'code':fact['code'],'launch':context['original_launch'],'common_guards':owner_common}
    context['original_cpu'] = owner['cpu'] = context['original_evaluator'].accept_unit(
        owner,owner['launch']['selected_cpu'],'cpu',panel='selection')
    owner['original_cpu'] = context['original_cpu']
    return owner


def check_original_owner(context, owner, original_guard):
    original_guard(context)
    original = context['original_evaluator']; fact = ORIGINAL_EXPORT_OWNER
    require(original is sys.modules.get('_connected_export_owner_v3') and
        any(snapshot[0] is original for snapshot in context['helper_snapshots']),
        'dispatch original reader differs from authenticated snapshot')
    specific = {'root','args','code','launch','common_guards','cpu'}
    expected_common = {p:h for p,h in context['common_guards'].items()
        if p not in {str(context['root']/'execution.json'),*(str(context['root']/n) for n in FILES)}}
    merge_guards(expected_common,{str(Path(fact['root'])/'execution.json'):fact['execution_sha256'],
        **{str(Path(fact['root'])/n):h for n,h in fact['code'].items()}})
    require(owner.keys() == context.keys() and all(owner[k] is context[k] for k in owner.keys()-specific) and
        owner['root'] == Path(fact['root']) and owner['code'] == fact['code'] and
        owner['launch'] is context['original_launch'] and owner['cpu'] is context['original_cpu'] and
        owner['common_guards'] == expected_common and vars(owner['args']) ==
        dict(execution_sha256=fact['execution_sha256'],authority=Path(ORIGINAL_SCORE_AUTHORITY['path']),
            authority_sha256=ORIGINAL_SCORE_AUTHORITY['sha256'],phase='score',arm=None,seed=None,output=context['args'].output) and
        read_json(ORIGINAL_SCORE_AUTHORITY,context['guards']) == owner['launch'],
        'original owner six-member/shared context binding differs')


def admit_export(context, owner, endpoint, original_guard):
    key = label(endpoint); unit = context['launch']['exports'][key]
    if owner is not None:
        check_original_owner(context,owner,original_guard)
    if owner is not None and unit == owner['launch']['exports'].get(key):
        require(context['args'].phase == 'score' and context['launch']['stage'] == 'first' and
            context['launch']['panel'] == 'selection' and
            endpoint in owner['launch']['endpoints'], 'original exports have first-selection-only endpoint authority')
        record = context['original_evaluator'].accept_unit(owner,unit,'export',endpoint['arm'],endpoint['seed'])
        require(record['payload_facts'] == context['cpu']['payload_facts'][key],
            'original export differs from current independently admitted CPU payload')
        return record
    return accept_unit(context,unit,'export',endpoint['arm'],endpoint['seed'])
# END ORIGINAL EXPORT OWNER


def check_endpoint(endpoint):
    require(isinstance(endpoint,dict) and endpoint.keys() == {'seed','arm','launch','terminal','checkpoint',
        'terminal_state_sha256','bundle','inference_state_sha256'} and endpoint['arm'] in ARMS and
        type(endpoint['seed']) is int and endpoint['seed'] in SEEDS and sha(endpoint['terminal_state_sha256']) and
        sha(endpoint['inference_state_sha256']), 'complete connected endpoint required')
    for key in ('launch','checkpoint','bundle'):
        check_file(endpoint[key])
    check_unit(endpoint['terminal'])
    root = Path(endpoint['terminal']['receipt']['path']).parent
    require(Path(endpoint['checkpoint']['path']) == root/(label(endpoint)+'-terminal.pt') and
        Path(endpoint['bundle']['path']) == root/(label(endpoint)+'-bundle')/'bundle.json',
        'connected endpoint artifact roles differ')


def check_endpoint_binding(endpoint, record, manifest, training):
    """Pure metadata boundary: actual nested trainer result, never a legacy projection."""
    result = record['result']; identity = result['identity']; parity = result['parity']
    require(record['phase'] == 'train' and record['seed'] == identity['seed'] == endpoint['seed'] and
        record['arm'] == identity['arm'] == endpoint['arm'] and identity['device'] == 'cuda' and
        record['authority'] == endpoint['launch'] and result['checkpoint'] == endpoint['checkpoint'] and
        type(result['training_updates']) is int and result['training_updates'] == 128 and
        result['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and parity['bundle'] == endpoint['bundle'] and
        manifest['schema'] == 'siglip2-connected-mlp-bundle-v1' and
        manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and
        manifest['base_vision_sha256'] == identity['base_vision']['sha256'] and
        manifest['files']['vision.pt'] == identity['base_vision']['checkpoint']['sha256'] and
        manifest['encoder_identity'] == identity['encoder_identity'] and
        manifest['scope'] == identity['scope'] ==
            {'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256} and
        all(manifest['code'][name] == training['code'][name] for name in TRAIN_FILES) and
        all(parity[key] is True for key in ('strict_reload_exact','native_training_inference_exact',
            'inference_artifact_independent','bundle_original_dependencies_denied','updated_source_mutants_rejected')),
        'complete nested TRAIN128 endpoint/bundle binding differs')
    require(sha(manifest['vision_sha256']) and manifest['vision_sha256'] == parity['vision_sha256'] and
        ((manifest['vision_sha256'] == manifest['base_vision_sha256']) if endpoint['arm'] == 'control' else
         (manifest['vision_sha256'] != manifest['base_vision_sha256'])),
        'updated encoder differs from authenticated TRAIN parity/base role')


SCOPE_SHA256 = '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726'

ARMS = ('control', 'candidate')

SEEDS = (179061, 179069)

ORDER = ((179061,'control'),(179061,'candidate'),(179069,'candidate'),(179069,'control'))

METRICS = ('per_query_r1','per_query_ap')

PANELS = {'selection':(3449,1734,1715,498),'validation':(3479,1749,1730,498)}

NATIVE = {'torch','numpy','PIL','transformers','safetensors','torchvision','sfora'}

NEAREST_EVALUATOR = {'root':'/home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1',
 'execution_sha256':'5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59',
 'code':{'evaluate_siglip2_nearest_ranking.py':'73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be',
         'test_nearest_ranking_evaluation.py':'a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e'}}

GENUINE_PINS = {'evaluate_siglip2_genuine_views.py':'85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa',
                'test_siglip2_genuine_view_evaluation.py':'ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35'}

REFERENCE = {'root': '/home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2',
 'execution_sha256': 'c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970',
 'code': {'evaluate_siglip2_prototype_residual.py': 'e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb',
          'test_siglip2_prototype_residual_evaluation.py': '5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49'}}

GENUINE_EXECUTION_SHA = '82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: '+v))

def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def check_file(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and Path(value['path']).is_absolute() and sha(value['sha256']),
            'exact actual FILE required')

def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and sha(expected), 'canonical FILE/SHA required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell()-len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'file SHA256 differs: '+str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    return path

def batch_bound_files(guards, items):
    """Fresh evaluator reads per occurrence; publish only after joined success."""
    items = list(items)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bound_file, {}, path, expected) for path, expected in items]
        paths = [future.result() for future in futures]
    staged = dict(guards)
    for path, (_, expected) in zip(paths, items):
        require(staged.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    guards.update(staged)
    return paths

def read_json(value, guards):
    check_file(value)
    path = bound_file(guards, value['path'], value['sha256'])
    with path.open('rb') as stream:
        raw = stream.read(64*1024**2+1)
    require(len(raw) <= 64*1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)

def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json({'path':str(root/'execution.json'), 'sha256':expected}, guards)
    require(code.keys() == set(names) and all(sha(v) for v in code.values()), 'exact execution closure required')
    for name, digest in code.items():
        bound_file(guards, root/name, digest)
    return code

def merge_guards(target, values):
    for p, h in values.items():
        require(target.setdefault(p,h) == h, 'conflicting source authority: '+p)

def load_authenticated(name, path, digest, guards):
    require(name not in sys.modules, 'preloaded helper forbidden')
    raw = bound_file(guards, path, digest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper bytes changed before execution')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'helper origin differs')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == Path(module.__spec__.origin) == path, 'loaded helper origin differs')
    return module

def policy(phase):
    require(phase in ('cpu','export','score'), 'fixed evaluation phase required')
    return {'seconds': 1500 if phase == 'export' else 700 if phase == 'score' else 500, 'host_bytes':8*1024**3,
            'swap_bytes':0, 'cuda_visible_devices':'0' if phase == 'export' else ''}

def check_unit(unit):
    require(isinstance(unit,dict) and unit.keys() == {'receipt','log','unit','invocation_id',
        'service_seconds','native_peak_rss_kib','both_locks_held'} and unit['both_locks_held'] is True and
        re.fullmatch('[A-Za-z0-9_.@-]+',unit['unit']) and re.fullmatch('[0-9a-f]{32}',unit['invocation_id']) and
        all(type(unit[k]) in (int,float) and math.isfinite(unit[k]) and unit[k]>0
            for k in ('service_seconds','native_peak_rss_kib')), 'complete actual UNIT required')
    check_file(unit['receipt']); check_file(unit['log'])

def batch_sizes(count):
    require(type(count) is int and count>0,'positive image population required')
    return [min(32,count-start) for start in range(0,count,32)]

def check_resource_facts(record,phase):
    require(record['resource_policy'] == policy(phase) and
        type(record['wall_seconds']) in (int,float) and math.isfinite(record['wall_seconds']) and
        0 < record['wall_seconds'] < policy(phase)['seconds'] and
        type(record['process_peak_rss_kib']) in (int,float) and math.isfinite(record['process_peak_rss_kib']) and
        0 < record['process_peak_rss_kib'] <= 8*1024**2 and
        type(record['peak_cuda_allocated_bytes']) is int and
        (0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000 if phase == 'export' else
         record['peak_cuda_allocated_bytes'] == 0) and record['cuda_initialized'] is (phase == 'export'),
        'whole-unit endpoint resources/partial native receipt differ')

def json_form(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))

def seeds(stage):
    require(stage in ('first','full'), 'fixed prospective stage required')
    return SEEDS[:1] if stage == 'first' else SEEDS

def endpoint_order(stage):
    seeds(stage)
    return ORDER[:2] if stage == 'first' else ORDER

def label(endpoint):
    return endpoint['arm']+'-'+str(endpoint['seed'])

def check_code(value, names, pins=None):
    require(isinstance(value,dict) and value.keys() == {'root','execution_sha256','code'} and
        isinstance(value['root'],str) and Path(value['root']).is_absolute() and sha(value['execution_sha256']) and
        value['code'].keys() == set(names) and all(sha(h) for h in value['code'].values()) and
        (pins is None or value['code'] == pins), 'actual exact separate code descriptor required')

def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execution-sha256',required=True); p.add_argument('--authority',type=Path,required=True)
    p.add_argument('--authority-sha256',required=True); p.add_argument('--phase',choices=('cpu','export','score'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--output',type=Path,required=True)
    return p

def cli(args):
    result = [str(Path(__file__).absolute()),'--execution-sha256',args.execution_sha256,
        '--authority',str(args.authority),'--authority-sha256',args.authority_sha256,'--phase',args.phase]
    if args.seed is not None:
        result += ['--seed',str(args.seed),'--arm',args.arm]
    return result+['--output',str(args.output)]

def resources(context,before):
    import resource
    import torch
    args=context['args']; legacy=context['training_context']['legacy']; source=legacy['source_driver']
    after=source.cgroup_memory(); unit=Path(after['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(after,unit)
    for value in (before,after):
        context['helper'].zero_events(value)
    peak=torch.cuda.max_memory_allocated() if args.phase == 'export' else 0
    wall=time.perf_counter()-UNIT_STARTED; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap=next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmSwap:'))
    require(before['path'] == after['path'] and wall<policy(args.phase)['seconds'] and 0<rss<=8*1024**2 and
        int(swap.split()[1]) == 0 and peak<10_000_000_000 and
        (args.phase == 'export' or not torch.cuda.is_initialized()), 'whole-unit resource/zero swap cap differs')
    return {'resource_policy':policy(args.phase),'wall_seconds':wall,'process_peak_rss_kib':rss,
        'peak_cuda_allocated_bytes':peak,'cgroup_before':before,'cgroup_after':after,
        'both_locks_held_in_parent_authority':True,'terminal_exit_and_both_locks_require_parent_receipt':True}


def check_launch(launch,args):
    require(isinstance(launch,dict) and launch.keys() == LAUNCH_KEYS and
        launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
        launch['phase'] == args.phase and launch['arm'] == args.arm and launch['seed'] == args.seed and
        launch['both_locks_held'] is True and launch['selection_previously_exposed'] is True and
        launch['resource_policies'] == {p:policy(p) for p in ('cpu','export','score')} and
        launch['cost_policy'] == COST_POLICY and launch['nearest_evaluator'] == NEAREST_EVALUATOR and launch['reference'] == REFERENCE and
        launch['panel'] in PANELS, 'frozen connected evaluator launch differs')
    check_code(launch['training'],TRAIN_FILES)
    require(TRAINING is not None and launch['training'] == TRAINING, 'actual parent-frozen trainer2 differs')
    check_file(launch['scope'])
    require(launch['scope']['sha256'] == SCOPE_SHA256, 'frozen complete scope FILE differs')
    check_code(launch['evaluator_reference'],EVALUATOR_PINS,EVALUATOR_PINS)
    check_code(launch['genuine_evaluator'],GENUINE_PINS,GENUINE_PINS)
    require(launch['genuine_evaluator']['root'] == '/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4' and
        launch['genuine_evaluator']['execution_sha256'] == GENUINE_EXECUTION_SHA,
        'original paired-seed evaluator origin differs')
    require([(e['seed'],e['arm']) for e in launch['endpoints']] == list(endpoint_order(launch['stage'])),
        'prospective ordered endpoints required')
    for endpoint in launch['endpoints']:
        check_endpoint(endpoint)
    require((args.phase == 'export') == (args.arm in ARMS and type(args.seed) is int and
        args.seed in seeds(launch['stage'])) and (args.phase == 'export' or args.arm is args.seed is None),
        'export owns exactly one admitted seed/arm')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
        (args.phase != 'cpu' or launch['panel'] == 'selection'), 'fresh stage CPU metadata qualification required')
    require(launch['exports'].keys() == ({label(e) for e in launch['endpoints']} if args.phase == 'score' else set()),
        'complete admitted export set required before score')
    require((launch['first_selection'] is None) == (launch['stage'] == 'first') and
        (launch['selection_go'] is None) == (launch['panel'] == 'selection') and
        (launch['panel'] != 'validation' or launch['stage'] == 'full'), 'first continuation/full selection GO required')
    for u in (launch['selected_cpu'],launch['first_selection'],launch['selection_go'],*launch['exports'].values()):
        if u is not None:
            check_unit(u)
    units = [e['terminal'] for e in launch['endpoints']]+list(launch['exports'].values())
    for k in ('selected_cpu','first_selection','selection_go'):
        if launch[k] is not None:
            units.append(launch[k])
    require(len({u['unit'] for u in units}) == len(units) and
        len({u['invocation_id'] for u in units}) == len(units), 'distinct complete evaluation/training units required')

def binding(context):
    return {k:context['launch'][k] for k in ('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference',
        'stage','panel','endpoints','cost_policy','both_locks_held','selection_previously_exposed','scope')}

def guard_helpers(context):
    snapshots = context.setdefault('helper_snapshots',[])
    modules = [context[key] for key in ('trainer','evaluator_reference','nearest_evaluator','math','reference','helper','baseline')]
    if 'original_evaluator' in context:
        modules.append(context['original_evaluator'])
    if 'first_evaluator' in context:
        modules.append(context['first_evaluator'])
    if not snapshots:
        for module in modules:
            values = dict(vars(module))
            functions = [(f,f.__code__,f.__defaults__,copy.deepcopy(f.__kwdefaults__))
                         for f in values.values() if isinstance(f,FunctionType)]
            literals = {k:copy.deepcopy(v) for k,v in values.items() if k != '__builtins__' and
                        isinstance(v,(dict,list,tuple,set,frozenset))}
            # Mutable owned runtime caches are checked by the trainer's authenticated API.
            snapshots.append((module,Path(module.__file__),module.__spec__,values,functions,literals))
    require(len(snapshots) == len(modules) and all(snapshot[0] is module
        for snapshot,module in zip(snapshots,modules,strict=True)), 'complete context-bound helper snapshot inventory required')
    for module,path,spec,values,functions,literals in snapshots:
        require(module.__spec__ is spec and Path(spec.origin) == Path(module.__file__) == path and
            sys.modules.get(module.__name__) is module and vars(module).keys() == values.keys() and
            all(vars(module)[k] is v for k,v in values.items()) and
            all(f.__code__ is code and f.__defaults__ == defaults and f.__kwdefaults__ == kw
                for f,code,defaults,kw in functions) and all(vars(module)[k] == v for k,v in literals.items()),
            'authenticated helper live source/global binding changed')
        bound_file({},path,context['guards'][str(path)])


def check_paired_initialization(trainer, control, candidate):
    c,a = (r['result']['identity'] for r in (control,candidate))
    for ident,arm in ((c,'control'),(a,'candidate')):
        require(ident['arm'] == arm and ident['device'] == 'cuda' and
            (ident['parameter_names'],ident['parameter_shapes']) == trainer.parameter_roles(arm)[:2] and
            ident['scope'] == {'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256},
            'same CONTROL scope with exact connected parameter roles required')
    require(c['seed'] == a['seed'] and all(c[k] == a[k] for k in
        ('method','source','scope','initializer','validator_identity','encoder_identity','base_vision',
         'optimizer_defaults','initial_scaler','numerical_flags')),
        'same-seed initializer/static/schedule/RNG/encoder identity differs')
    require(control['launch']['selected_cpu'] == candidate['launch']['selected_cpu'] and
        control['launch']['selected_mechanics'] == candidate['launch']['selected_mechanics'],
        'same-seed complete CPU/paired mechanics admission differs')


def admit_training_unit(context, unit, phase, arm, seed, endpoint_reader):
    """Deduplicate only exact already-admitted UNITs, preserving original readers."""
    t,trainer = context['training_context'],context['trainer']
    key = f'{phase}:{seed}:{arm}'
    known = context['training_units']
    if key in known:
        require(known[key] == unit, 'same phase/seed/arm changed original UNIT')
        return t['connected_terminals'][key]
    record = endpoint_reader(unit,phase,arm,seed)
    known[key] = unit
    return record


def admit_endpoints(context, endpoints, endpoint_reader):
    trainer,t,launch = context['trainer'],context['training_context'],context['launch']
    by_pair = {(e['seed'],e['arm']):e for e in launch['endpoints']}
    for endpoint in endpoints:
        seed,arm = endpoint['seed'],endpoint['arm']
        el = read_json(endpoint['launch'],context['guards'])
        trainer.check_launch(el,SimpleNamespace(execution_sha256=launch['training']['execution_sha256'],
            phase='train',arm=arm,seed=seed))
        require(trainer.method(el) == trainer.method(t['connected_launch']) and
            el['selected_cpu'] == t['connected_launch']['selected_cpu'],
            'same frozen connected source/method/complete CPU required')
        if arm == 'candidate':
            require(el['fresh_control'] == by_pair[seed,'control']['terminal'],
                'candidate fresh_control must be the paired same-seed control UNIT')
        for a in ARMS:
            record = admit_training_unit(context,el['selected_mechanics'][a],'mechanics',a,seed,endpoint_reader)
            require(record['launch']['selected_cpu'] == el['selected_cpu'], 'mechanics changes selected CPU')
        record = admit_training_unit(context,endpoint['terminal'],'train',arm,seed,endpoint_reader)
        require(record['launch'] == el and record['result']['identity'] ==
            t['connected_terminals'][f'mechanics:{seed}:{arm}']['result']['identity'] and
            [trainer.diagnostic(r) for r in record['result']['steps'][:17]] ==
            [trainer.diagnostic(r) for r in t['connected_terminals'][f'mechanics:{seed}:{arm}']['result']['steps']],
            'fresh TRAIN128 changes seed-specific mechanics identity/first17')
        bound_file(context['guards'],endpoint['checkpoint']['path'],endpoint['checkpoint']['sha256'])
        manifest,guards = trainer.admit_bundle(Path(endpoint['bundle']['path']).parent,endpoint['bundle']['sha256'])
        check_endpoint_binding(endpoint,record,manifest,launch['training'])
        merge_guards(context['guards'],guards)
        context['manifests'][seed,arm] = manifest
        context['records'][seed,arm] = record
    for seed in sorted({e['seed'] for e in endpoints}):
        check_paired_initialization(trainer,context['records'][seed,'control'],context['records'][seed,'candidate'])
    stage = 'full' if len(context['records']) == 4 else 'first'
    costs = context['evaluator_reference'].paired_cost({key:{
        'service_seconds':by_pair[key]['terminal']['service_seconds'],
        'total_training_core_seconds':record['total_training_core_seconds']}
        for key,record in context['records'].items()},stage)
    require(all(v['pass'] for v in costs.values()), 'fresh core/whole cost gate failed before held quality')
    context['costs'] = costs
    require(len({e['checkpoint']['sha256'] for e in by_pair.values()}) == len(by_pair),
        'trained checkpoint reused between endpoints')
    merge_guards(context['guards'],t['guards'])


def preparation_costs(context):
    """Retain actual original preparation and each charged phase; no subtraction."""
    t = context['training_context']; selected = t['legacy']['selected']
    unit = selected['source']['preparation_terminal']; proof = selected['export_record']
    context['evaluator_reference'].check_preparation_unit(unit)
    require(read_json(unit['proof'],context['guards']) == proof, 'original CONTROL preparation proof differs')
    startup = read_json(proof['startup_terminal'],context['guards'])
    context['evaluator_reference'].check_preparation_unit(startup)
    for u in (startup,unit):
        bound_file(context['guards'],u['log']['path'],u['log']['sha256'])
    return {'shared_control_startup':startup,'shared_control_export':unit,
        'total_preparation_service_seconds':startup['service_seconds']+unit['service_seconds'],
        'preparation_excluded_from_training_ratios':True,'qualification_separate':True}


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded source admission')
    root,guards = Path(__file__).absolute().parent,{}
    code = closure(root,args.execution_sha256,FILES,guards)
    launch = read_json({'path':str(args.authority),'sha256':args.authority_sha256},guards)
    check_launch(launch,args)
    keys = ('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference')
    original_active = (args.phase == 'score' and launch['stage'] == 'first' and launch['panel'] == 'selection' and
        launch['first_selection'] is None and launch['selection_go'] is None and launch['exports'] == ORIGINAL_EXPORT_UNITS)
    roots = [root]+([Path(ORIGINAL_EXPORT_OWNER['root'])] if original_active else [])+[Path(launch[k]['root']) for k in keys]
    if launch['stage'] == 'full':
        roots.append(Path(FIRST_SELECTION_OWNER['root']))
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
        not args.output.exists() and not args.output.is_symlink() and
        all(not a.is_relative_to(b) and not b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]) and
        all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in roots),
        'separate immutable closures and exclusive output required')
    loaded = {}
    for key,filename in zip(keys,('train_siglip2_connected_mlp.py','evaluate_siglip2_identity_diversity.py',
            'evaluate_siglip2_nearest_ranking.py','evaluate_siglip2_genuine_views.py',
            'evaluate_siglip2_prototype_residual.py'),strict=True):
        fact = launch[key]
        require(closure(fact['root'],fact['execution_sha256'],fact['code'],guards) == fact['code'],
            'complete authenticated source closure differs: '+key)
        loaded[key] = load_authenticated('_connected_eval_'+key,Path(fact['root'])/filename,fact['code'][filename],guards)
    trainer,e,native,math_helper,reference = (loaded[k] for k in keys)
    require(trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and trainer.SEEDS == SEEDS and
        trainer.SCHEMA == 'siglip2-connected-mlp-v1' and trainer.AUTHORITY_SCHEMA == 'siglip2-connected-mlp-launch-v1' and
        trainer.INFERENCE_SCHEMA == 'siglip2-connected-mlp-inference-v1' and
        trainer.BUNDLE_SCHEMA == 'siglip2-connected-mlp-bundle-v1' and
        trainer.RECIPE['classes'] == dict.fromkeys(ARMS,1008) and trainer.RECIPE['rows'] == 6355 and
        trainer.policy('cpu')['seconds'] == 600 and trainer.policy('train')['seconds'] == 3000 and
        trainer.policy('mechanics')['seconds'] == 1200 and
        native.REFERENCE == REFERENCE == e.REFERENCE and e.NEAREST_EVALUATOR == NEAREST_EVALUATOR and
        e.GENUINE_PINS == GENUINE_PINS and e.GENUINE_EXECUTION_SHA == GENUINE_EXECUTION_SHA and
        math_helper.ORDER == ORDER and math_helper.METRICS == METRICS and math_helper.PANELS == PANELS,
        'connected trainer/pinned scientific predicates differ')
    first = launch['endpoints'][0]; training = launch['training']
    require(read_json(first['launch'],guards)['selected_cpu'] == TRAINING_CPU,
        'parent-frozen complete CPUv6 UNIT required; no mechanics/TRAIN qualification inferred')
    targs = SimpleNamespace(execution_sha256=training['execution_sha256'],authority=Path(first['launch']['path']),
        authority_sha256=first['launch']['sha256'],phase='train',arm='control',seed=SEEDS[0],output=args.output)
    t = trainer.authority(targs)  # Original CPU, actual gradient, CPU600 and mechanics061 normal exits.
    require(t['launch']['scope'] == launch['scope'], 'original CONTROL scope FILE differs')
    common_guards = {p:h for p,h in {**guards,**t['guards']}.items()
        if p not in (str(args.authority),first['launch']['path'])}
    units = {'cpu:179061:control':t['connected_launch']['selected_cpu'],
        **{f'mechanics:179061:{a}':u for a,u in t['connected_launch']['selected_mechanics'].items()}}
    context = {'args':args,'root':root,'guards':guards,'code':code,'launch':launch,'trainer':trainer,
        'evaluator_reference':e,'training_context':t,'nearest_evaluator':native,'math':math_helper,'reference':reference,
        'records':{},'manifests':{},'training_units':units,'accepted_units':[],'common_guards':common_guards}
    endpoint_reader,endpoint_guard = load_endpoint_reader(context)
    admit_endpoints(context,launch['endpoints'][:2],endpoint_reader)
    archived = read_json(native.CONCAT_TERMINAL['receipt'],guards)
    require(archived['invocation']['invocation_id'] == native.CONCAT_TERMINAL['invocation_id'] and
        archived['schema'] == reference.SCHEMA and archived['phase'] == 'score' and archived['pass'] is True and
        archived['source_code'] == launch['reference']['code'] and
        archived['execution_sha256'] == launch['reference']['execution_sha256'] and
        archived['quality']['concat']['recall_at_1'] == 0.9648212226066898 and
        archived['quality']['concat']['map_at_r'] == 0.8177754035543956 and
        archived['spec']['panel'] == 'selection' and archived['validation_quality_exposed'] is False,
        'preserved accepted concat source/floors differ')
    spec = archived['spec']
    for item,pins in ((spec['original_evaluator'],reference.EVALUATOR_PINS),
        (reference.EVALUATION_REFERENCE,reference.REFERENCE_PINS),(reference.ORIGINAL_REFERENCE,reference.ORIGINAL_PINS)):
        require(closure(item['root'],item['execution_sha256'],pins,guards) == pins, 'immutable scoring/source closure differs')
    baseline = load_authenticated('_compact_eval_baseline',Path(spec['original_evaluator']['root'])/'evaluate_siglip2_quadratic_readout.py',
        reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'],guards)
    helper = load_authenticated('_compact_eval_helper',Path(reference.REFERENCE_ROOT)/'export_siglip2_substrate_adaptation.py',
        reference.REFERENCE_PINS['export_siglip2_substrate_adaptation.py'],guards)
    legacy = t['legacy']; terminal_reader = reference.original_terminal_reader(t['fit_context'])
    final = terminal_reader(legacy['admission'],archived,native.CONCAT_TERMINAL,500,guards)
    for value in (archived['cgroup_before'],archived['cgroup_after'],final):
        helper.zero_events(value)
    batch_bound_files(guards,archived['input_guards'].items())
    for name,h in archived['files'].items():
        bound_file(guards,Path(archived['output'])/name,h)
    merge_guards(guards,t['guards'])
    s = {'args':args,'guards':guards,'spec':spec,'selected':legacy,'admission':legacy['admission'],
        'helper':helper,'baseline':baseline,'fit':legacy['prior']['fit'],'partition':legacy['selected']['partition'],
        'origin_records':[archived],'terminals':[native.CONCAT_TERMINAL]}
    require(s['partition'] == read_json(spec['partition'],guards) and
        spec['partition'] == t['fit_context']['launch']['partition'] and
        s['partition']['original_cache']['sha256'] == reference.FIT_SHA, 'original ordered panel partition differs')
    reference.source_selection_adapter(baseline,t['fit_context'])(s)
    source_record = s['source_record']
    require(source_record['source_code'] == launch['genuine_evaluator']['code'] and
        source_record['execution_sha256'] == launch['genuine_evaluator']['execution_sha256'], 'original paired-seed math source differs')
    context.update(score_context=s,concat_record=archived,terminal_reader=terminal_reader,
        helper=helper,baseline=baseline)
    context['preparation_costs'] = preparation_costs(context)
    original_guard = load_original_owner(context) if original_active else None
    first_reader,first_guard = load_first_selection(context) if launch['stage'] == 'full' else (None,None)
    guard_helpers(context)
    if launch['stage'] == 'full':
        first_receipt = first_reader(context,launch['first_selection'],'score',stage='first',panel='selection')
        require(first_receipt['decision'] == 'CONTINUE' and
            first_receipt['launch']['endpoints'] == launch['endpoints'][:2],
            'first061 KILL prohibits069 and VAL')
        context['first_receipt'] = first_receipt
        admit_endpoints(context,launch['endpoints'][2:],endpoint_reader)
    if launch['panel'] == 'validation':
        selection = accept_unit(context,launch['selection_go'],'score',stage='full',panel='selection')
        require(selection['decision'] == 'GO' and selection['selection_go_admits_validation_only'] is True and
            selection['launch']['endpoints'] == launch['endpoints'], 'same-four selection GO required before VAL')
    context['required_guards'] = dict(guards)
    if args.phase != 'cpu':
        context['cpu'] = accept_unit(context,launch['selected_cpu'],'cpu',panel='selection')
    if args.phase == 'score':
        owner = original_owner_context(context,original_guard) if original_active else None
        context['export_records'] = {label(e):admit_export(context,owner,e,original_guard) for e in launch['endpoints']}
    original_guard = admission_exit_guard(endpoint_guard,first_guard,original_guard)
    return context,original_guard


def native_start(context):
    args,t=context['args'],context['training_context']; legacy=t['legacy']; source=legacy['source_driver']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == policy(args.phase)['cuda_visible_devices'] and
        os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and sys.flags.optimize == 0 and
        re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')), 'deterministic complete-unit environment required')
    prior=legacy['selected']['source_cpu']['invocation']; python=Path(sys.executable).resolve()
    require(str(python) == prior['python'] and bound_file(context['guards'],python,prior['python_sha256']) == python and
        sys.version == prior['python_version'], 'qualified original interpreter required')
    before=source.cgroup_memory(); unit=Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before,unit); context['helper'].zero_events(before)
    units=[e['terminal'] for e in context['launch']['endpoints']]+context['accepted_units']+context['score_context']['terminals']
    require(unit not in {u['unit'] for u in units} and os.environ['INVOCATION_ID'] not in
        {u['invocation_id'] for u in units}|legacy['invocations'], 'new distinct complete unit required')
    import torch
    require(not torch.cuda.is_initialized(), 'initial source admission must precede CUDA')
    flags=legacy['selected']['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'source numerical flags differ')
    if args.phase in ('cpu','export') or context['launch']['panel'] == 'validation':
        t['trainer'].prepare_native(t)
    else:
        t['fitter'].prepare_original(t['fit_context']); t['flags']=legacy['flags']
    context['score_context']['packing']=legacy['packing']; context['flags']=flags
    t['trainer'].helper_guard(t); guard_helpers(context)
    merge_guards(context['guards'],t['guards'])
    if args.phase == 'export':
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one admitted CUDA device required')
    else:
        require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU phase must hide CUDA')
    return before

def fullfeature_oracle(context,state,features):
    """Independent accepted concat plus explicit centered residual; never the owned wrapper."""
    import torch
    from torch.nn import functional as F
    trainer,t=context['trainer'],context['training_context'];arm=state['arm']
    require(arm in ARMS, 'fullfeature oracle arm differs')
    base=t['trainer'].helper_guard(t).raw_features(features,state['head_object'],
        torch.nn.Parameter(state['A'].detach().clone()),state['means'],'concat',t['legacy']['quadratic'])
    C=state['C'].detach().clone();mu=state['mu_train'].detach().clone()
    zero=torch.count_nonzero(C).item() == 0
    require(not zero, 'updated both-arm C must be nonzero')
    raw=base+F.linear(features-mu,C)
    output=packed_outputs(context,raw)
    proof={'C_exact_zero':zero,'residual_nonzero_witness':False,
        'omitted_C_mutant_rejected':False,'wrong_mu_mutant_rejected':False}
    omitted=packed_outputs(context,base)
    column=int(C.abs().sum(dim=0).argmax().item())
    wrong_mu=mu.clone();wrong_mu[column]+=1./float(C[:,column].abs().max().item())
    wrong=packed_outputs(context,base+F.linear(features-wrong_mu,C))
    digest=trainer.fingerprint(t,output)
    proof['residual_nonzero_witness']=proof['omitted_C_mutant_rejected']=(
        digest != trainer.fingerprint(t,omitted))
    proof['wrong_mu_mutant_rejected']=digest != trainer.fingerprint(t,wrong)
    context['evaluator_reference'].check_residual_oracle(proof,arm)
    return output,proof

def cpu_calibration(context):
    """Updated bundle readouts versus the accepted helper, synthetic features only."""
    import torch
    from torch.nn import functional as F
    t=context['training_context'];trainer=context['trainer'];legacy=t['legacy'];facts={}
    oracles={}
    features=F.normalize(torch.linspace(-1,1,32*1152,dtype=torch.float32).reshape(32,1152),dim=1)
    for endpoint in context['launch']['endpoints']:
        directory=Path(endpoint['bundle']['path']).parent
        manifest,guards=trainer.admit_bundle(directory,endpoint['bundle']['sha256'])
        merge_guards(context['guards'],guards)
        disk=torch.load(directory/'endpoint.pt',map_location='cpu',weights_only=True,mmap=True)
        check_inference_members(trainer,disk)
        name='_compact_cpu_readout_'+label(endpoint).replace('-','_')
        serving=load_authenticated(name,directory/'prototype_residual_readout.py',
            manifest['code']['prototype_residual_readout.py'],context['guards'])
        head=legacy['selected']['cached'].head_from('control',tensors=disk['head']).requires_grad_(False).train()
        A=torch.nn.Parameter(disk['A'].clone())
        with torch.no_grad(),torch.autocast('cpu',enabled=False):
            raw=trainer.fullfeature_raw_features(features,head,A,disk['means'],disk['C'],disk['mu_train'],
                disk['arm'],legacy['quadratic'],serving)
            first=packed_outputs(context,raw)
            second,proof=fullfeature_oracle(context,{'head_object':head,'A':A,**disk},features.clone())
            context['helper'].exact(tuple_outputs(first),tuple_outputs(second))
            require(trainer.fingerprint(t,first) == trainer.fingerprint(t,second), 'synthetic updated raw/unit/pack/wire parity differs')
        facts[label(endpoint)]=trainer.fingerprint(t,first)
        oracles[label(endpoint)]=proof
        require(sys.modules.pop(name,None) is serving, 'synthetic helper registry changed')
        del head,A,raw,first,second,disk,serving;gc.collect()
        trainer.mapping_absent(directory/'endpoint.pt')
    del features
    return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,
        'raw_unit_packed_exact':True,'outputs_sha256':facts,'residual_oracles':oracles}

def packed_outputs(context,raw):
    return context['training_context']['old'].packed_outputs(context['training_context']['legacy'],raw)

def tuple_outputs(values):
    return tuple(values[k] for k in ('raw','unit','codes','inverse_norms'))

def endpoint_facts(context,state):
    trainer,t=context['trainer'],context['training_context']
    config=state['model'].config.to_dict();labels=t['initial']['config'].get('id2label')
    require(type(config) is dict and type(labels) is dict and labels and
        type(config.get('id2label')) is dict and len(config['id2label']) == len(labels) and
        all(type(k) is str and re.fullmatch(r'0|[1-9][0-9]*',k) and type(v) is str for k,v in labels.items()) and
        all(type(k) is int and str(k) in labels and type(v) is str and v == labels[str(k)]
            for k,v in config['id2label'].items()), 'live id2label differs from frozen config')
    config={**config,'id2label':{str(k):v for k,v in config['id2label'].items()}}
    current = trainer.encoder_facts(state,t['legacy']['original'],t['legacy']['source_driver'],
        state['manifest']['environment']['packages'],serving=True)
    return {'vision_sha256':current['vision_sha256'],
        'encoder_identity_sha256':trainer.fingerprint(t,state['encoder_identity']),
        'encoder_sha256':trainer.fingerprint(t,{n:dict(state['model'].named_parameters())[n] for n in trainer.MLP}),
        'members':{k:trainer.fingerprint(t,config if k == 'config' else
            dict(state['model'].named_buffers()) if k == 'buffers' else
            json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else
            dict(state['head_object'].state_dict()) if k == 'head' else state[k])
            for k in ('config','buffers','processor_config','head','A','means','C','mu_train','mu_train_provenance','arm',
                'scope','common_statistics')}}

def images_outputs(context,state,rows,*,oracle=False):
    """Canonical original images; true B32 serving order, no held cache path."""
    import torch
    from PIL import Image
    from torch.nn import functional as F
    trainer,t=context['trainer'],context['training_context']
    require(0<len(rows)<=32, 'actual native B32 image boundary required')
    images=[]; rgb=hashlib.sha256(); rng=torch.random.get_rng_state().clone()
    try:
        for row in rows:
            path=bound_file(context['guards'],row['path'],row['image_sha256'])
            with Image.open(path) as opened:
                image=opened.convert('RGB')
            images.append(image); rgb.update(str(image.size).encode()); rgb.update(image.tobytes())
        portable,endpoint=context['portable_entry']
        with context['evaluator_reference'].bundle_reads_only(context,endpoint):
            values=portable.inference_outputs(state,images)
        if oracle:
            with context['evaluator_reference'].bundle_reads_only(context,endpoint):
                trainer.portable_mutants(portable,state,images,Path(endpoint['bundle']['path']).parent)
        pixels=state['processor_object'](images=images,return_tensors='pt')['pixel_values']
        require(pixels.dtype == torch.float32 and pixels.shape == (len(rows),3,256,256) and
            torch.isfinite(pixels).all().item() and torch.equal(rng,torch.random.get_rng_state()), 'native canonical pixels/RNG differ')
        if oracle:
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.float16):
                    pooled=state['model'](pixel_values=pixels.to('cuda')).pooler_output
                with torch.autocast('cuda',enabled=False):
                    features=F.normalize(pooled.float(),dim=1)
                    expected,proof=fullfeature_oracle(context,state,features)
            context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))
            require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')
            del pooled,features,expected
        fact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),
            'outputs_sha256':trainer.fingerprint(t,values),'residual_oracle':proof if oracle else None}
        del pixels
        return values,fact
    finally:
        for image in images:
            image.close()

def export_pass(context,state,rows,mapping):
    import torch
    t=context['training_context']; raw=torch.empty((len(rows),128),dtype=torch.float32); unit=torch.empty_like(raw)
    images=[]; sizes={}
    for role in ('query','gallery'):
        indices=mapping[role]; sizes[role]=[]
        for start in range(0,len(indices),32):
            batch=indices[start:start+32]
            print(json.dumps({'event':'COMPACT_TIMING','stage':'images_outputs','boundary':'begin','role':role,'batch_start':start,'batch_size':len(batch),'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
            values,fact=images_outputs(context,state,[rows[i] for i in batch],oracle=start == 0 or start+32 >= len(indices))
            print(json.dumps({'event':'COMPACT_TIMING','stage':'images_outputs','boundary':'end','role':role,'batch_start':start,'batch_size':len(batch),'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
            raw[batch]=values['raw']; unit[batch]=values['unit']; fact['role']=role
            images.append(fact); sizes[role].append(len(batch)); del values
    require(sizes == {r:batch_sizes(len(mapping[r])) for r in ('query','gallery')}, 'complete actual B32 roles/tails differ')
    packed=t['legacy']['packing'].pack_int8_unit_embeddings(unit)
    return (raw,unit,packed.codes,packed.inverse_norms),images,sizes

def preserved_validation(context,fixed):
    import torch
    t=context['training_context']; s=context['score_context']; trainer=context['trainer']
    panel=s['partition']['panels']['validation']; cache=s['baseline'].cache_rows(s,panel['original_rows'])
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    head=t['legacy']['selected']['cached'].head_from('control',tensors=t['common']['head']).requires_grad_(False).train()
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        source=tuple_outputs(packed_outputs(context,head(cache)))
        raw=t['trainer'].helper_guard(t).raw_features(cache,head,torch.nn.Parameter(t['common']['A'].clone()),
            t['common']['means'],'concat',t['legacy']['quadratic'])
        concat=tuple_outputs(packed_outputs(context,raw))
    result=[fixed.packed_quality(v[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu')) for v in (source,concat)]
    del head,cache,source,concat,raw; gc.collect()
    return result


@contextmanager
def endpoint_scope(context, endpoint):
    """Use the real original initializer API, release it before any public model."""
    import torch
    trainer,t = context['trainer'],context['training_context']
    cpu = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    state = None
    try:
        qualification = trainer.select_initializer(t['original_cpu_record'],endpoint['seed'])
        state,_ = trainer.load_initializer(t,qualification)
        t['initial']['config'] = copy.deepcopy(state['config'])
        t['trainer'].release(t,state)
        torch.random.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)
        yield
        require(torch.equal(cpu,torch.random.get_rng_state()) and
            all(torch.equal(a,b) for a,b in zip(cuda,
                torch.cuda.get_rng_state_all() if cuda else [],strict=True)), 'endpoint evaluation RNG changed')
    finally:
        if state:
            t['trainer'].release(t,state)
        t.pop('initial',{}).clear()
        torch.random.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


def check_inference_members(trainer, disk):
    require(disk.keys() == trainer.INFERENCE_KEYS and disk['schema'] == trainer.INFERENCE_SCHEMA,
        'complete connected inference schema differs')


def authenticate_payloads(context, endpoint):
    """Full optimizer/RNG/static checkpoint admission followed by the owned endpoint."""
    import torch
    trainer,t = context['trainer'],context['training_context']
    path = bound_file(context['guards'],endpoint['checkpoint']['path'],endpoint['checkpoint']['sha256'])
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        ident = copy.deepcopy(disk['identity'])
        trainer.check_payload(t,disk,ident,128)
        require(json_form(ident) == context['records'][endpoint['seed'],endpoint['arm']]['result']['identity'],
            'complete terminal identity differs')
        with path.open('rb') as stream:
            pages = t['legacy']['original'].CheckpointPages(stream)
            require(trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['terminal_state_sha256'],
                'complete TRAIN128 typed current bytes differ')
            del pages
        members = {k:trainer.fingerprint(t,ident[k] if k in ('arm','encoder_identity') else disk[k]) for k in MEMBERS}
        vision = disk['vision_sha256']
    finally:
        del disk
        gc.collect(); trainer.mapping_absent(path)
    directory = Path(endpoint['bundle']['path']).parent
    manifest,guards = trainer.admit_bundle(directory,endpoint['bundle']['sha256'])
    check_endpoint_binding(endpoint,context['records'][endpoint['seed'],endpoint['arm']],manifest,context['launch']['training'])
    merge_guards(context['guards'],guards)
    path = directory/'endpoint.pt'
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        check_inference_members(trainer,disk)
        require(trainer.fingerprint(t,disk) == manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and
            trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and
            all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and
            disk['source'] == ident['source'] == t['source'] and disk['numerical_flags'] == t['flags'] and
            disk['vision_sha256'] == vision == manifest['vision_sha256'] and
            disk['base_vision'] == ident['base_vision'] and disk['encoder_identity'] == ident['encoder_identity'] and
            disk['scope'] == t['initial']['scope'] and
            disk['base_vision']['checkpoint'] == t['initial']['provenance']['encoder']['checkpoint'] and
            manifest['files']['vision.pt'] == disk['base_vision']['checkpoint']['sha256'],
            'bundle substitutes trained four/config/head/A/C/means/source/current bytes')
        for name,shape in (('C',(128,1152)),('mu_train',(1152,))):
            t['legacy']['quadratic']._check_tensor(disk[name],shape,'cpu',frozen=True)
        require(torch.count_nonzero(disk['C']).item() > 0, 'updated both-arm C must be nonzero')
        t['old'].finite_tree(disk)
        trainer.expect_rejection(lambda:check_inference_members(trainer,{**disk,'schema':'foreign'}),
            'malformed inference accepted')
        facts = {'identity':json_form(ident),'members':members,'vision_sha256':vision,
            'base_vision_sha256':manifest['base_vision_sha256'],'fixed_sha256':disk['fixed_sha256'],
            'processor_config_sha256':trainer.fingerprint(t,disk['processor']['config']),
            'terminal_state_sha256':endpoint['terminal_state_sha256'],
            'inference_state_sha256':endpoint['inference_state_sha256'],'bundle':endpoint['bundle']}
        check_payload_facts(context,facts,endpoint)
        return facts
    finally:
        del disk
        gc.collect(); trainer.mapping_absent(path)


def check_payload_facts(context, facts, endpoint):
    manifest = context['manifests'][endpoint['seed'],endpoint['arm']]
    require(facts.keys() == {'identity','members','vision_sha256','base_vision_sha256','fixed_sha256',
            'processor_config_sha256','terminal_state_sha256','inference_state_sha256','bundle'} and
        facts['identity'] == context['records'][endpoint['seed'],endpoint['arm']]['result']['identity'] and
        facts['members'].keys() == set(MEMBERS) and all(sha(v) for v in facts['members'].values()) and
        all(sha(facts[k]) for k in ('vision_sha256','base_vision_sha256','fixed_sha256','processor_config_sha256')) and
        facts['vision_sha256'] == manifest['vision_sha256'] and
        facts['base_vision_sha256'] == manifest['base_vision_sha256'] and
        facts['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
        facts['inference_state_sha256'] == endpoint['inference_state_sha256'] and facts['bundle'] == endpoint['bundle'],
        'complete connected payload facts differ')


def cpu_qualification(context):
    facts = {}
    for endpoint in context['launch']['endpoints']:
        with endpoint_scope(context,endpoint):
            first = authenticate_payloads(context,endpoint)
            second = authenticate_payloads(context,endpoint)
            require(first == second, 'independent full checkpoint/bundle authentication differs')
            facts[label(endpoint)] = first
    proof = context['reference'].qualify_bootstrap(context['score_context'])
    context['training_context']['trainer'].helper_guard(context['training_context'])
    guard_helpers(context)
    return {'payload_facts':facts,'synthetic_bootstrap':proof,'updated_payloads_authenticated':True,
        'malformed_inference_rejected':True,'metadata_only':True,'calibration':cpu_calibration(context),
        'files':{},'quality_read':False}


def native_export(context):
    trainer,t,args = context['trainer'],context['training_context'],context['args']
    endpoint = next(e for e in context['launch']['endpoints'] if (e['seed'],e['arm']) == (args.seed,args.arm))
    with endpoint_scope(context,endpoint):
        facts = authenticate_payloads(context,endpoint)
        key,directory = label(endpoint),Path(endpoint['bundle']['path']).parent
        require(facts == context['cpu']['payload_facts'][key], 'CPU-qualified updated endpoint differs')
        rows,mapping = context['nearest_evaluator'].image_rows(context,context['launch']['panel'])
        first = None
        for index in range(2):
            name = '_connected_export_entry_'+str(index)+'_'+key.replace('-','_')
            portable = state = None
            try:
                with context['evaluator_reference'].bundle_reads_only(context,endpoint):
                    portable = load_authenticated(name,directory/'train_siglip2_connected_mlp.py',
                        context['launch']['training']['code']['train_siglip2_connected_mlp.py'],context['guards'])
                    state = portable.load_inference(directory,endpoint['bundle']['sha256'],'cuda')
                context['portable_entry'] = (portable,endpoint)
                t['live_model'] = weakref.ref(state['model'])
                model_facts = endpoint_facts(context,state)
                require(model_facts['vision_sha256'] == facts['vision_sha256'] and
                    model_facts['encoder_identity_sha256'] == facts['members']['encoder_identity'] and
                    model_facts['encoder_sha256'] == facts['members']['encoder'] and
                    all(model_facts['members'][k] == facts['members'][k] for k in
                        ('config','buffers','head','A','means','C','mu_train','mu_train_provenance','arm','scope','common_statistics')) and
                    model_facts['members']['processor_config'] == facts['processor_config_sha256'],
                    'complete live updated448/absolute4/readout roles differ')
                values,images,sizes = export_pass(context,state,rows,mapping)
                require(endpoint_facts(context,state) == model_facts, 'forward mutated full live endpoint')
                t['nearest'].native_source_api(t).audit_origins(t['legacy'],require_exact=True,
                    admission=t['legacy']['original'].FlatAdmission())
                if index == 0:
                    first = (values,images,sizes,model_facts)
                    files = context['baseline'].write_wires(context['score_context'],key,values)
                else:
                    context['helper'].exact(first[0],values)
                    require(trainer.fingerprint(t,first[0]) == trainer.fingerprint(t,values) and
                        (images,sizes,model_facts) == first[1:], 'independent complete query/gallery B32/tails differ')
                    context['baseline'].readback_wires(context['score_context'],key,files,values)
                    panel_facts = context['baseline'].value_facts(context['score_context'],values)
                del values
            finally:
                if state is not None:
                    portable.release_inference(state)
                t['trainer'].require_no_training(t)
                if portable is not None:
                    require(sys.modules.pop(name,None) is portable, 'owned loader registry changed')
                context.pop('portable_entry',None)
        del first
        gc.collect()
    return {'payload_facts':facts,'inference_state_sha256':endpoint['inference_state_sha256'],
        'panel_facts':panel_facts,'images':images,'ordered_images_sha256':context['reference'].json_digest(rows),
        'batch_sizes':sizes,'strict_independent_reload_exact':True,'full_updated_state_exact':True,
        'raw_unit_packed_readback_exact':True,'bundle_dependency_boundary_enforced':True,
        'same_role_oracle_exact':True,'updated_source_mutants_rejected':True,
        'native_exact_four_post_calibration':True,'files':files,'quality_read':False}


def score_exports(context):
    import numpy as np
    import torch
    s=context['score_context']; launch=context['launch']; panel_name=launch['panel']; panel=s['partition']['panels'][panel_name]
    e=context['evaluator_reference']; native=context['nearest_evaluator']; fixed=s['baseline'].scoring_math(s)
    # Both archived panels are replayed with exact original per-query arrays first.
    source,concat=native.archived_replay(context,fixed)
    if panel_name == 'validation':
        require(launch['selection_go'] is not None, 'sealed validation requires same-four selection GO')
        source,concat=preserved_validation(context,fixed)
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    rows,_=native.image_rows(context,panel_name); held={}
    for endpoint in launch['endpoints']:
        key=label(endpoint); record=context['export_records'][key]
        require(record['payload_facts'] == context['cpu']['payload_facts'][key] and
            record['ordered_images_sha256'] == context['reference'].json_digest(rows), 'CPU/bundle/original ordered-image mapping differs')
        values=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
        require(s['baseline'].value_facts(s,values) == record['panel_facts'], 'full export wire facts differ before any candidate metric')
        # Independent readback for ALL arms finishes before the first candidate metric.
        repeated=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
        context['helper'].exact(values,repeated); held[key]=values; del repeated
    readiness=dict.fromkeys(READINESS,True)
    def compute_quality():
        quality={}
        for endpoint in launch['endpoints']:
            key=label(endpoint); record=context['export_records'][key]; values=held.pop(key)
            first=fixed.packed_quality(values[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            second=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
            context['helper'].exact(values,second)
            replay=fixed.packed_quality(second[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            context['reference'].replay_equal(first,replay)
            quality.setdefault(str(endpoint['seed']),{})[endpoint['arm']]=first
            del values,second
        return quality
    quality=quality_after_readiness(readiness,compute_quality)
    _,average=context['math'].averaged_deltas(quality,launch['stage'],panel_name)
    immediate=all(e.immediate_quality_pass(quality[str(seed)],source,concat,PANELS[panel_name][1]) for seed in seeds(launch['stage'])) and e.selection_floors_pass(quality,panel_name)
    intervals={}
    if launch['stage'] == 'full' and immediate:
        # Equal-seed deltas first; original helper resets SAME PCG64 draws5000/179019
        # for every metric and both product/query interval signs.
        intervals=context['reference'].paired_intervals(fixed,average,np.asarray(labels)[panel['query']])
    decision=e.decide(context['math'],quality,source,concat,launch['stage'],panel_name,intervals,context['costs'])
    return {**decision,'readiness':readiness,'quality':quality,'source_quality':source,'concat_quality':concat,
        'paired_seed_average_intervals':intervals,'bootstrap_seed':179019,
        'bootstrap_draws':5000 if launch['stage'] == 'full' and immediate else 0,'quality_read':True,'files':{},
        'source_archived_perquery_exact':True,'concat_archived_perquery_exact':True,
        'all_export_wires_readback_before_quality':True,'persisted_wire_scoring_replay_exact':True,
        'updated_descriptors_from_image_encoders':True,'query_images':PANELS[panel_name][1],
        'gallery_images':PANELS[panel_name][2],'products':PANELS[panel_name][3],
        'metric_units':'fractions; multiply by100 for percentage points',
        'interval_scope':'equal-seed paired product/query deltas; shared5000/179019 conditional draws',
        'selection_previously_exposed':True,'validation_quality_exposed':panel_name == 'validation',
        'public_encoder_qualified':False,'optimization_throughput_is_image_training_throughput':False,
        'preparation_costs':context['preparation_costs']}

def quality_after_readiness(readiness,score):
    require(readiness.keys() == set(READINESS) and all(v is True for v in readiness.values()),
            'complete qualification/source/cost/wire admission must precede candidate quality')
    return score()

def check_receipt(context,record,phase,arm=None,seed=None,stage=None,panel=None):
    stage = stage or context['launch']['stage']; panel = panel or context['launch']['panel']
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and record['seed'] == seed and
        record['stage'] == stage and record['panel'] == panel and record['execution_sha256'] == context['args'].execution_sha256 and
        record['source_code'] == context['code'] and record['source'] == context['training_context']['source'] and
        record['cost_policy'] == COST_POLICY and all(record[k] is True for k in
            ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
             'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority')) and
        all(record[k] is False for k in ('official_read','global_production_goal_met','public_latency_measured','product_go')),
        'complete source/resource evaluator receipt required')
    require(record['preparation_costs'] == context['preparation_costs'],
        'complete original shared CONTROL preparation costs differ')
    check_resource_facts(record,phase)
    rargs = SimpleNamespace(execution_sha256=record['execution_sha256'],authority=Path(record['authority']['path']),
        authority_sha256=record['authority']['sha256'],phase=phase,arm=arm,seed=seed,output=Path(record['output']))
    check_launch(record['launch'],rargs)
    require(read_json(record['authority'],context['guards']) == record['launch'] and
        record['launch']['stage'] == stage and record['launch']['panel'] == panel and
        record['numerical_flags'] == context['training_context']['legacy']['selected']['source_cpu']['numerical_flags'] and
        record['invocation']['cublas_workspace_config'] == ':4096:8' and
        record['authority_sha256'] == record['authority']['sha256'] and
        record['binding'] == binding({'launch':record['launch']}) and record['invocation']['argv'] == cli(rargs) and
        record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == policy(phase)['cuda_visible_devices'],
        'actual launch/CLI/binding differs')
    for k in ('training','evaluator_reference','nearest_evaluator','genuine_evaluator','reference','cost_policy','selection_previously_exposed','scope'):
        require(record['launch'][k] == context['launch'][k], 'receipt changes frozen procedure: '+k)
    endpoints = context['launch']['endpoints'][:2] if stage == 'first' else context['launch']['endpoints']
    require(record['launch']['endpoints'] == endpoints and
        record['cost'] == {str(s):context['costs'][str(s)] for s in seeds(stage)}, 'receipt changes endpoint/cost binding')
    if stage == context['launch']['stage']:
        require(record['launch']['first_selection'] == context['launch']['first_selection'], 'same eligible first selection required')
        if phase != 'cpu':
            require(record['launch']['selected_cpu'] == context['launch']['selected_cpu'], 'same complete stage CPU required')
    if panel == 'validation':
        require(record['launch']['selection_go'] == context['launch']['selection_go'], 'same sealed selection GO required')
    if phase == 'cpu':
        context['reference'].check_synthetic_bootstrap(record['synthetic_bootstrap'])
        require(record['files'] == {} and record['quality_read'] is False and record['metadata_only'] is True and
            record['updated_payloads_authenticated'] is True and record['malformed_inference_rejected'] is True and
            record['payload_facts'].keys() == {label(e) for e in endpoints} and
            record['calibration']['same_role_forward_exact'] is True and record['calibration']['raw_unit_packed_exact'] is True,
            'CPU full-payload/bundle/synthetic metadata qualification differs')
        require(record['calibration']['residual_oracles'].keys() == {label(e) for e in endpoints},
            'complete synthetic endpoint residual proofs required')
        for endpoint in endpoints:
            context['evaluator_reference'].check_residual_oracle(record['calibration']['residual_oracles'][label(endpoint)],endpoint['arm'])
            check_payload_facts(context,record['payload_facts'][label(endpoint)],endpoint)
    elif phase == 'export':
        endpoint = next(e for e in endpoints if (e['seed'],e['arm']) == (seed,arm)); key=label(endpoint)
        require(record['launch']['selected_cpu'] == context['launch']['selected_cpu'] and
            record['files'].keys() == {key+s for s in ('.raw.npy','.unit.npy','.packed.bin')} and
            record['batch_sizes'] == {r:batch_sizes(PANELS[panel][i]) for r,i in (('query',1),('gallery',2))} and
            record['payload_facts'] == context['cpu']['payload_facts'][key] and
            record['inference_state_sha256'] == endpoint['inference_state_sha256'] and
            all(record[k] is True for k in ('strict_independent_reload_exact','full_updated_state_exact',
                'raw_unit_packed_readback_exact','updated_source_mutants_rejected','bundle_dependency_boundary_enforced',
                'same_role_oracle_exact','native_exact_four_post_calibration')),
            'two complete B32 bundle-only export/readback witnesses differ')
        context['reference'].check_value_facts(record['panel_facts'],PANELS[panel][0])
        check_export_images(context,record,endpoint,panel)
    else:
        decision = context['evaluator_reference'].decide(context['math'],record['quality'],record['source_quality'],record['concat_quality'],stage,panel,
            record['paired_seed_average_intervals'],record['cost'])
        require(all(record[k] == v for k,v in decision.items()) and
            record['readiness'] == dict.fromkeys(READINESS,True) and record['quality_read'] is True and record['files'] == {} and
            record['bootstrap_seed'] == 179019 and record['bootstrap_draws'] ==
            (5000 if stage == 'full' and decision['immediate_quality_pass'] else 0) and
            all(record[k] is True for k in ('source_archived_perquery_exact','concat_archived_perquery_exact',
                'all_export_wires_readback_before_quality','persisted_wire_scoring_replay_exact')),
            'first/full paired scoring gate/readiness differs')
        if panel == 'selection':
            context['reference'].replay_equal(context['score_context']['source_record']['quality']['179061']['control'],record['source_quality'])
            context['reference'].replay_equal(context['concat_record']['quality']['concat'],record['concat_quality'])

def accept_unit(context,unit,phase,arm=None,seed=None,stage=None,panel=None):
    check_unit(unit)
    record = read_json(unit['receipt'],context['guards'])
    check_receipt(context,record,phase,arm,seed,stage,panel)
    require(Path(unit['receipt']['path']) == Path(record['output'])/'receipt.json', 'complete receipt output role differs')
    t=context['training_context']; legacy=t['legacy']
    final=context['terminal_reader'](legacy['admission'],record,unit,policy(phase)['seconds'],context['guards'])
    for value in (record['cgroup_before'],record['cgroup_after'],final):
        context['helper'].zero_events(value)
    require(unit['invocation_id'] not in legacy['invocations'], 'reused terminal invocation')
    legacy['invocations'].add(unit['invocation_id'])
    batch_bound_files(context['guards'],record['input_guards'].items())
    for n,h in record['files'].items():
        bound_file(context['guards'],Path(record['output'])/n,h)
    require(all(record['input_guards'].get(p) == h for p,h in context['common_guards'].items()),
        'foreign complete receipt omits frozen original source guards')
    t['nearest'].native_source_api(t).audit_origins(legacy,initial=True)
    require(record['origins']['packages'] == legacy['selected']['packages'] and
        all(record['input_guards'].get(p) == h for p,h in record['origins']['files'].items()), 'receipt original origins differ')
    if phase == 'export':
        known=set(legacy['selected']['source_cpu']['origins']['files'])|set(legacy['warm_record']['origins']['files'])
        actual=set(record['origins']['files'])-known
        authority=read_json(t['launch']['native_authority'],context['guards'])
        require(authority['proof']['sha256'] == t['nearest'].NATIVE_PROOF_PINS['proof'], 'native proof pin differs')
        proof=read_json(authority['proof'],context['guards'])
        site=Path(proof['authority']['installed_site_root'])
        expected={str(site/name):fact['sha256'] for name,fact in proof['comparison']['selected_members'].items()}
        require(actual == set(expected) and actual <= set(record['origins']['native_files']) and
            all(record['origins']['files'][p] == h for p,h in expected.items()),
            'export receipt observed supplemental membership/bytes must be exact original four')
    prior=legacy['selected']['source_cpu']['invocation']
    require(all(record['invocation'][k] == prior[k] for k in ('python','python_sha256','python_version')),
        'qualified original interpreter differs')
    context['accepted_units'].append(unit)
    return record

def exit_rehash(context, original_guard):
    if original_guard is not None:
        original_guard(context)
    trainer,t=context['trainer'],context['training_context']
    t['trainer'].require_no_training(t); t['trainer'].helper_guard(t); guard_helpers(context)
    api=t['nearest'].native_source_api(t)
    api.audit_origins(t['legacy'],require_exact=context['args'].phase == 'export')
    api.exit_rehash(t['fit_context'])
    merge_guards(context['guards'],t['guards']); merge_guards(context['guards'],t['legacy']['guards'])
    # Fresh uncached complete source/payload/bundle files, including restored-mtime mutations.
    trainer.batch_bound_files({},context['guards'].items())
    for descriptor,names,pins in (({'root':str(context['root']),'execution_sha256':context['args'].execution_sha256},FILES,context['code']),
        *(((ORIGINAL_EXPORT_OWNER,FILES,ORIGINAL_EXPORT_OWNER['code']),) if 'original_evaluator' in context else ()),
        (context['launch']['training'],TRAIN_FILES,context['launch']['training']['code']),
        (context['launch']['evaluator_reference'],EVALUATOR_PINS,EVALUATOR_PINS),
        (context['launch']['nearest_evaluator'],NEAREST_EVALUATOR['code'],NEAREST_EVALUATOR['code']),
        (context['launch']['genuine_evaluator'],GENUINE_PINS,GENUINE_PINS),
        (context['launch']['reference'],context['launch']['reference']['code'],context['launch']['reference']['code'])):
        require(closure(descriptor['root'],descriptor['execution_sha256'],names,{}) == pins,
            'fresh complete source closure changed at exit')
    for endpoint in context['launch']['endpoints']:
        _,guards=trainer.admit_bundle(Path(endpoint['bundle']['path']).parent,endpoint['bundle']['sha256'])
        merge_guards(context['guards'],guards)
    guard_helpers(context)
    api.audit_origins(t['legacy'],require_exact=context['args'].phase == 'export')
    merge_guards(context['guards'],t['legacy']['origins']['files'])
    if original_guard is not None:
        original_guard(context)
    return t['legacy']['origins']

def run(args):
    require(sys.argv == cli(args), 'fixed canonical CLI order required')
    context,original_guard=authority(args)
    context['training_context']['fit_context']['unit_started']=UNIT_STARTED
    before=native_start(context)
    import torch
    t=context['training_context']; source=t['legacy']['source_driver']; seed=args.seed or SEEDS[0]
    torch.random.default_generator.manual_seed(seed)
    if args.phase == 'export':
        torch.cuda.manual_seed_all(seed)
    rng=torch.random.get_rng_state().clone(); flags=source.numerical_flags()
    cuda_rng=torch.cuda.get_rng_state_all() if args.phase == 'export' else []
    args.output.mkdir()
    print(json.dumps({'progress':'admitted','phase':args.phase,'stage':context['launch']['stage'],
        'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    result=cpu_qualification(context) if args.phase == 'cpu' else native_export(context) if args.phase == 'export' else score_exports(context)
    require(torch.equal(rng,torch.random.get_rng_state()) and source.numerical_flags() == flags,
        'whole-unit CPU RNG/numerical flags differ')
    require(args.phase != 'export' or all(torch.equal(a,b) for a,b in
        zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit CUDA RNG differs')
    print(json.dumps({'progress':'exit_rehash','seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'begin','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    origins=exit_rehash(context,original_guard)
    print(json.dumps({'event':'COMPACT_TIMING','stage':'exit_rehash','boundary':'end','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    prior=t['legacy']['selected']['source_cpu']['invocation']
    record={'schema':SCHEMA,'phase':args.phase,'arm':args.arm,'seed':args.seed,'stage':context['launch']['stage'],
        'panel':context['launch']['panel'],'binding':binding(context),'source_code':context['code'],
        'execution_sha256':args.execution_sha256,'source':t['source'],'launch':context['launch'],
        'authority':{'path':str(args.authority),'sha256':args.authority_sha256},'authority_sha256':args.authority_sha256,
        'output':str(args.output),'cost':context['costs'],'cost_policy':COST_POLICY,'pass':True,
        'preparation_costs':context['preparation_costs'],
        'engineering_admission_pass':True,'integrity_pass':True,'resources_pass':True,'exit_rehash_pass':True,
        'sequential_model_ownership':True,'rng_flags_preserved':True,'cuda_initialized':torch.cuda.is_initialized(),
        'official_read':False,'global_production_goal_met':False,'public_latency_measured':False,'product_go':False,
        'numerical_flags':flags,'origins':origins,'input_guards':context['guards'],
        'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
            'python_version':sys.version,'optimize':sys.flags.optimize,'pid':os.getpid(),
            'invocation_id':os.environ['INVOCATION_ID'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},**result,**resources(context,before)}
    check_receipt(context,record,args.phase,args.arm,args.seed)
    context['helper'].publish(args.output/'receipt.json',record)
    require(time.perf_counter()-UNIT_STARTED<policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
    return record

def main():
    args=parser().parse_args()
    try:
        result=run(args)
    except (OSError,ValueError,ImportError,KeyError,TypeError,AttributeError,RuntimeError,SyntaxError) as error:
        raise SystemExit('Connected-MLP evaluation rejected: '+str(error)) from error
    print(json.dumps({'schema':SCHEMA,'phase':args.phase,'output':str(args.output),'decision':result.get('decision')}),flush=True)


def check_export_images(context, record, endpoint, panel):
    """Both complete roles and tails, not a one-batch surrogate export."""
    mapping = context['score_context']['partition']['panels'][panel]
    images = record['images']
    expected = [(role,indices[start:start+32]) for role in ('query','gallery')
        for indices in (mapping[role],) for start in range(0,len(indices),32)]
    require(len(images) == len(expected) and record['quality_read'] is False,
        'complete unscored image export required')
    for fact,(role,indices) in zip(images,expected,strict=True):
        require(fact.keys() == {'rows','rgb_sha256','pixels_sha256','outputs_sha256','residual_oracle','role'} and
            fact['role'] == role and [r['panel_ordinal'] for r in fact['rows']] == indices and
            all(r['role'] == role and r['original_row'] == mapping['original_rows'][i]
                for i,r in zip(indices,fact['rows'],strict=True)) and
            all(sha(fact[k]) for k in ('rgb_sha256','pixels_sha256','outputs_sha256')),
            'complete original B32 image-role witness differs')
        if indices[0] == mapping[role][0] or indices[-1] == mapping[role][-1]:
            context['evaluator_reference'].check_residual_oracle(fact['residual_oracle'],endpoint['arm'])
        else:
            require(fact['residual_oracle'] is None, 'unexpected image oracle role')


if __name__ == '__main__':
    main()
