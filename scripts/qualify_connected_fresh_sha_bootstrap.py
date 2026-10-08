#!/usr/bin/env python3
"""Prospective discarded fresh-SHA paired native falsifier, bootstrap v2; source alone is UNQUALIFIED.

CLI: python -B qualify_connected_fresh_sha_bootstrap.py --authority FILE --authority-sha256 SHA
--proposal FILE --proposal-sha256 SHA --output NEWDIR, in exactly that argv order.
FILE={path:canonical_absolute_regular_file,sha256:actual64hex}. --authority is the ORIGINAL
installed control-serving v2 authority, unchanged. --proposal is strict JSON with exactly
schema connected-fresh-sha-bootstrap-proposal-v2, driver, test, proposed{runtime,ledger,bridge}
FILEs and commit (40 hex, recorded only). The driver and test sit beside the original
source-v2 files; no proposed pin is hardcoded here. The frozen v1 driver/test are untouched.

run() is the original qualify_connected_control_serving.run verbatim except the finite
SEAMS; read_proposal proves that inverse on the SHA-authenticated bytes of both files
before any native import. Original admission, genuine ConnectedCompactIndex factory,
owner release, full uncached source exit, both locks, body300/whole1500/8GiB/zero swap/
CUDA<10GB are unchanged. falsify replaces only the 22-call request body: one genuine
owner whose FIRST act is one unprofiled public B1 request of the first admitted TRAIN image
through the unchanged observer.measure_request, followed immediately by the original native
api.audit_origins(require_exact=True) (v1 died only on that exact-four exit predicate after a
zero-forward body); a failure there stops all hash work. Then the proposed CUDA SHA pipeline
vs the original serial fingerprint on the same full448 state and frozen444 tree, exact
digests first, then mutants, then three alternating timing pairs that exclude the bootstrap.
Engineering evidence only: no speed, p99, quality or state-reuse claim; any failed predicate
raises and publishes nothing. The terminal unit/footer reader stays root-owned.
"""
import argparse
import ast
import concurrent.futures
import gc
import hashlib
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys
import threading
import time
import weakref
from types import SimpleNamespace

import qualify_connected_control_serving as control
import qualify_connected_serving_requests as requests

RECEIPT_SCHEMA = 'connected-fresh-sha-bootstrap-falsifier-diagnostic-v2'
PROPOSAL_SCHEMA = 'connected-fresh-sha-bootstrap-proposal-v2'
DRIVER_NAME = 'qualify_connected_fresh_sha_bootstrap.py'
TEST_NAME = 'test_connected_fresh_sha_bootstrap.py'
PROPOSAL_KEYS = {'schema','driver','test','proposed','commit'}
CONTROL,KEYS,SCHEMA,SOURCES = control.CONTROL,control.KEYS,control.SCHEMA,control.SOURCES
prepare_observation,admit_control = control.prepare_observation,control.admit_control
BUDGET = 96*1024**2
NATIVE = {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'}
FLAGS = ('quality_read','quality_eligible','qualification_eligible','state_reuse_eligible',
         'optimization_eligible','product_go','speed_go')
OVER_MESSAGE = 'CUDA fingerprint snapshot exceeds byte budget'
CPU_MESSAGE = 'CUDA fingerprint requires CUDA tensor leaves'
PEAK_SEMANTICS = ('cumulative whole-unit guard peaks at phase boundaries, never reset; '
                  'an isolated transient host peak is not attributable')
STARTED = time.perf_counter()
# Finite inverse of run(): (id, original statements, driver statements), col-0 source text.
SEAMS = (
    ('S1',
     '''require(sys.argv == cli(authority_fact,str(args.output),sources['control_driver']['path']), 'fixed canonical control CLI order required')
''',
     '''proposal_fact = {'path':str(args.proposal),'sha256':args.proposal_sha256}
proposal = read_proposal(proposal_fact,sources)
require(sys.argv == fresh_cli(authority_fact,proposal_fact,str(args.output),proposal['driver']['path']), 'fixed canonical fresh-sha CLI order required')
'''),
    ('S2',
     '''self_source = requests.Source(sys.modules[__name__],sources['control_driver'])
''',
     '''control_source = requests.Source(control,sources['control_driver'])
self_source = requests.Source(sys.modules[__name__],proposal['driver'])
'''),
    ('S3',
     '''locks = requests.Locks(authority['locks'])
''',
     '''locks = requests.Locks(authority['locks'])
proposed_source = requests.Source.load(proposal['proposed']['runtime']); owned.append(proposed_source)
require_no_native()
'''),
    ('S4',
     '''frozen = [authority_fact,authority['observation'],authority['native_runtime'],authority['evaluation_authority'],
    *sources.values(),*observation['sources'].values(),observation['bundle']['manifest'],observation['control_export_receipt'],
    observation['gallery']['file'],observation['native'],*observation['train_images']]
''',
     '''frozen = [authority_fact,authority['observation'],authority['native_runtime'],authority['evaluation_authority'],
    *sources.values(),*observation['sources'].values(),observation['bundle']['manifest'],observation['control_export_receipt'],
    observation['gallery']['file'],observation['native'],*observation['train_images'],
    proposal_fact,*proposal_files(proposal)]
'''),
    ('S5',
     '''for source in (self_source,request_source,observer_source,native_source,evaluator_source,
    bridge_source,wrapper_source,packed_source,packing_source): source.check()
''',
     '''for source in (self_source,request_source,observer_source,native_source,evaluator_source,
    bridge_source,wrapper_source,packed_source,packing_source,control_source,proposed_source): source.check()
'''),
    ('S6',
     '''diagnostic = requests.request_body(factory,observer,read_images,torch.cuda.synchronize,
    [Path(f['path']) for f in observation['train_images']],observation['sources'],guard)
''',
     '''diagnostic = falsify(factory,guard,torch,proposed_source,observer,read_images,
    [Path(observation['train_images'][0]['path'])],api,context['training_context']['legacy'])
'''),
    ('S7',
     '''record = {'schema':'connected-control-serving-diagnostic-v1','status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
    'authority':authority_fact,'sources':sources,'control':CONTROL,'control_export':authority['control_export'],
    'observation':authority['observation'],'native_runtime':authority['native_runtime'],'output':str(output),
    'ties':ties,**diagnostic,'quality_read':False,'quality_eligible':False,'qualification_eligible':False,
    'state_reuse_eligible':False,'optimization_eligible':False,'product_go':False,'resource_policy':policy,
    'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
    'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
        'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
        'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
''',
     '''record = {'schema':RECEIPT_SCHEMA,'status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
    'authority':authority_fact,'proposal':proposal_fact,'proposed':proposal['proposed'],'proposal_commit':proposal['commit'],
    'sources':sources,'control':CONTROL,'control_export':authority['control_export'],
    'observation':authority['observation'],'native_runtime':authority['native_runtime'],'output':str(output),
    'ties':ties,**diagnostic,'quality_read':False,'quality_eligible':False,'qualification_eligible':False,
    'state_reuse_eligible':False,'optimization_eligible':False,'product_go':False,'speed_go':False,'resource_policy':policy,
    'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
    'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
        'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
        'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
'''),
    ('S8',
     '''validate_receipt(record,authority,authority_fact)
''',
     '''validate_fresh_receipt(record,authority,authority_fact,proposal,proposal_fact,observation)
'''),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_no_native():
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded control admission')


def native_imports(source):
    """Imports executed at module load (outside function bodies) that would reach native packages."""
    found = []
    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)): continue
            if isinstance(child,ast.Import): found.extend(a.name for a in child.names)
            elif isinstance(child,ast.ImportFrom): found.append(child.module or '.' if child.level == 0 else '.relative')
            walk(child)
    walk(ast.parse(source))
    return [n for n in found if n.split('.')[0] in NATIVE or n == '.relative']


def fresh_cli(authority, proposal, output, driver):
    return [driver,'--authority',authority['path'],'--authority-sha256',authority['sha256'],
            '--proposal',proposal['path'],'--proposal-sha256',proposal['sha256'],'--output',output]


def proposal_files(proposal):
    return [proposal['driver'],proposal['test'],*proposal['proposed'].values()]


def seam_dump(source, texts):
    """ast.dump of run() with each seam's statement sequence replaced; exactly one match each."""
    runs = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'run']
    require(len(runs) == 1, 'exactly one module run() required')
    nodes = list(ast.walk(runs[0]))
    for index,text in enumerate(texts):
        pattern,hits = [ast.dump(s) for s in ast.parse(text).body],0
        for node in nodes:
            for field in ('body','orelse','finalbody'):
                stmts = getattr(node,field,None)
                if not (isinstance(stmts,list) and stmts and isinstance(stmts[0],ast.stmt)): continue
                i = 0
                while i+len(pattern) <= len(stmts):
                    if [ast.dump(s) for s in stmts[i:i+len(pattern)]] == pattern:
                        stmts[i:i+len(pattern)] = [ast.Expr(ast.Name('__SEAM_%d__' % index,ast.Load()))]
                        hits += 1
                    i += 1
        require(hits == 1, 'seam %d must match exactly once' % index)
    return ast.dump(runs[0])


def check_seams(original, driver):
    require(seam_dump(original,[old for _,old,_ in SEAMS]) == seam_dump(driver,[new for _,_,new in SEAMS]),
            'driver run() differs from the original outside the declared seams')


def read_proposal(fact, sources):
    """Authenticate every proposed FILE and the seam inverse before any native execution."""
    proposal = requests.strict_json(requests.read_file(fact))
    require(type(proposal) is dict and proposal.keys() == PROPOSAL_KEYS and proposal['schema'] == PROPOSAL_SCHEMA and
        type(proposal['proposed']) is dict and proposal['proposed'].keys() == {'runtime','ledger','bridge'} and
        type(proposal['commit']) is str and re.fullmatch('[0-9a-f]{40}',proposal['commit']), 'exact fresh-sha proposal required')
    here = Path(__file__).absolute()
    require(here.name == DRIVER_NAME and proposal['driver']['path'] == str(here) and
        proposal['test']['path'] == str(here.with_name(TEST_NAME)), 'current driver/test FILE required')
    raw = {name:requests.read_file(f) for name,f in
        (('driver',proposal['driver']),('test',proposal['test']),*proposal['proposed'].items())}
    paths = [fact['path'],*(f['path'] for f in proposal_files(proposal))]
    require(len(set(paths)) == len(paths) and not set(paths) & {f['path'] for f in sources.values()},
        'distinct proposal/driver/test/proposed FILEs required')
    runtime,ledger = proposal['proposed']['runtime'],proposal['proposed']['ledger']
    require(('RUNTIME_SHA256 = "%s"' % runtime['sha256']).encode() in raw['ledger'] and
        ('"%s"' % ledger['sha256']).encode() in raw['bridge'], 'proposed ledger/bridge closure literals differ')
    require(not native_imports(raw['runtime']), 'proposed runtime must stay stdlib-only at module load')
    def fingerprint_ast(source):
        return [ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == 'fingerprint']
    original = fingerprint_ast(requests.read_file(sources['runtime']))
    require(len(original) == 1 and fingerprint_ast(raw['runtime']) == original, 'original fingerprint() must be AST unchanged')
    check_seams(requests.read_file(sources['control_driver']),raw['driver'])
    return proposal


def tree_bytes(tree):
    return sum(p.numel()*p.element_size() for p in tree.values())


def same_bytes(torch, a, b):
    return torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8))


def capture_error(call, tree):
    """Exception with only the callee's own frames retained; this frame never keeps the tree."""
    try: call(tree)
    except Exception as error:
        call = tree = None
        return error
    raise ValueError('expected failure did not occur')


def mutants(torch, pair, params, frozen, full, base, mlp):
    """Unchanged-version .data byte flips (MLP leaf, frozen leaf) and strided-alias mutation; restored exactly in finally."""
    (_,t_name),(_,w_name),(_,m_name) = (min((p.numel(),n) for n,p in frozen.items() if p.dim() == 1 and p.numel() >= 32),
        min((p.numel(),n) for n,p in frozen.items() if p.dim() == 2 and min(p.shape) >= 4),
        min((p.numel(),n) for n,p in params.items() if n in mlp and p.dim() == 1))
    T,W,M = params[t_name],params[w_name],params[m_name]
    saved_t,saved_w,saved_m = T.detach().clone(),W.detach().clone(),M.detach().clone()
    version_t,version_w,version_m = T._version,W._version,M._version
    original_seen = {k:[base[k]['original']] for k in ('frozen444','full448')}
    def distinct(name, step):
        require(step[name]['original'] == step[name]['proposed'], 'mutated original/proposed digests differ')
        require(step[name]['original'] not in original_seen[name], 'mutation did not change a fresh digest')
        original_seen[name].append(step[name]['original'])
    steps,failures = [],[]
    try:
        require(same_bytes(torch,T,saved_t) and same_bytes(torch,W,saved_w) and same_bytes(torch,M,saved_m), 'mutant baselines differ')
        flips = M.data.view(torch.uint8)
        flips[0] ^= 0x01
        flips = None
        require(M._version == version_m and not same_bytes(torch,M,saved_m), 'MLP byte flip precondition differs')
        step = {'frozen444':pair(frozen),'full448':pair(full)}
        require(step['frozen444'] == base['frozen444'], 'MLP byte flip must leave frozen444 at baseline')
        distinct('full448',step)
        mlp_byte = {'leaf':m_name,'dtype':str(M.dtype),'shape':list(M.shape),'version':version_m,'byte':0,'mask':1,**step}
        M.data.copy_(saved_m)
        require(same_bytes(torch,M,saved_m) and M._version == version_m, 'MLP byte flip was not restored')
        flips = T.data.view(torch.uint8)
        for byte,mask in ((0,0x01),(8,0x40),(17,0x04)):
            flips[byte] ^= mask
            require(T._version == version_t, 'unchanged-version mutation precondition differs')
            step = {'byte':byte,'mask':mask,'frozen444':pair(frozen),'full448':pair(full)}
            distinct('frozen444',step); distinct('full448',step)
            steps.append(step)
        flips = None
        data = W.data
        views = {'a_t':data.t(),'a_off':data.reshape(-1)[3:],'a_str':data[1:,::2]}
        require(not views['a_t'].is_contiguous() and not views['a_str'].is_contiguous() and
            views['a_off'].storage_offset() == data.storage_offset()+3 and
            views['a_str'].storage_offset() == data.storage_offset()+data.stride(0), 'native alias geometry differs')
        facts = [{'name':k,'shape':list(v.shape),'stride':list(v.stride()),'storage_offset':v.storage_offset()}
                 for k,v in views.items()]
        before = pair(views)
        require(before['original'] == before['proposed'], 'alias tree original/proposed digests differ')
        views['a_str'][0,0] += 1.0
        require(not same_bytes(torch,W,saved_w) and W._version == version_w, 'strided alias did not mutate the base bytes')
        after = pair(views)
        step = {'frozen444':pair(frozen),'full448':pair(full)}
        distinct('frozen444',step); distinct('full448',step)
        require(after['original'] == after['proposed'] and after['original'] != before['original'], 'alias mutation digests differ')
        alias = {'leaf':w_name,'shape':list(W.shape),'views':facts,'tree_before':before,'mutation':
            {'through':'a_str','index':[0,0],'delta':1.0,'base_bytes_changed':True,'version':version_w,
             'tree_after':after,**step}}
        views = data = None
    finally:
        for leaf,saved in ((M,saved_m),(T,saved_t),(W,saved_w)):
            try: leaf.data.copy_(saved)
            except BaseException as error: failures.append(error)
        if failures: raise failures[0]
    restored = {'frozen444':pair(frozen),'full448':pair(full),
        'leaf_bytes_equal':same_bytes(torch,T,saved_t) and same_bytes(torch,W,saved_w) and same_bytes(torch,M,saved_m),
        'versions_unchanged':T._version == version_t and W._version == version_w and M._version == version_m}
    require(restored['frozen444'] == base['frozen444'] and restored['full448'] == base['full448'] and
        restored['leaf_bytes_equal'] and restored['versions_unchanged'], 'restored digests/bytes/versions differ')
    unchanged = {'leaf':t_name,'dtype':str(T.dtype),'shape':list(T.shape),'version':version_t,'steps':steps}
    return mlp_byte,unchanged,alias,restored


def measure(index, torch, proposed_source, bounded, budget=BUDGET):
    index._check_current()
    endpoint,original,proposed = index._endpoint,index._module,proposed_source.module
    require(endpoint['modules'] == {'runtime':original} and proposed.MLP == original.MLP and
        vars(proposed)['hashlib'] is hashlib and
        vars(proposed)['ThreadPoolExecutor'] is concurrent.futures.ThreadPoolExecutor and
        callable(proposed._fingerprint_cuda_dict), 'admitted original/proposed runtime identity differs')
    threads = threading.active_count()
    def original_hash(tree):
        return original.fingerprint(tree)
    def proposed_hash(tree):
        before = threading.active_count()
        proposed_source.check()
        value = proposed._fingerprint_cuda_dict(tree)
        proposed_source.check()
        require(threading.active_count() == before, 'proposed SHA workers were not joined')
        return value
    def pair(tree):
        return {'original':original_hash(tree),'proposed':proposed_hash(tree)}
    def equal(value, message):
        require(value['original'] == value['proposed'], message)
        return value
    def peaks():
        resources = bounded()
        return {'process_peak_rss_kib':resources['process_peak_rss_kib'],
                'peak_cuda_allocated_bytes':resources['peak_cuda_allocated_bytes']}
    ident = endpoint['encoder_identity']
    params = dict(endpoint['model'].named_parameters())
    frozen = {n:p for n,p in params.items() if n not in original.MLP}
    full = endpoint['model'].state_dict()
    require(len(params) == 448 and len(frozen) == 444 and len(full) == 448 and params.keys() == full.keys(),
        'exact full448/frozen444 trees required')
    base = {'frozen444':equal(pair(frozen),'baseline frozen444 digests differ'),
            'full448':equal(pair(full),'baseline full448 digests differ')}
    require(base['frozen444']['original'] == ident['frozen_sha256'] and
        base['full448']['original'] == endpoint['vision_sha256'], 'complete typed hashes differ from encoder identity')
    bounded()
    edge = {'empty':torch.zeros((0,3),dtype=torch.float32,device='cuda'),
            'scalar':torch.tensor(1.25,dtype=torch.float32,device='cuda')}
    edge_digests = equal(pair(edge),'empty/scalar digests differ')
    edge = None
    bounded()
    rows = budget//(4*4096)
    require(rows*4*4096 == budget, 'near-limit geometry requires budget multiple of 16KiB')
    dense = {'dense':torch.arange(budget//4,dtype=torch.float32,device='cuda').reshape(rows,4096).t()}
    start = peaks()
    near_original,near_proposed = original_hash(dense),proposed_hash(dense)
    near = {'shape':list(dense['dense'].shape),'stride':list(dense['dense'].stride()),'bytes':budget,'budget_bytes':budget,
        'digests':equal({'original':near_original,'proposed':near_proposed},'near-limit dense transpose digests differ'),
        'whole_unit_peaks':{'before':start,'after':peaks()},'peak_semantics':PEAK_SEMANTICS}
    dense = None
    over = {'over':torch.zeros(budget//4+1,dtype=torch.float32,device='cuda')}
    start = peaks()
    try: proposed_hash(over)
    except ValueError as rejected: rejection = str(rejected)
    else: raise ValueError('over-limit tree must be rejected')
    require(rejection == OVER_MESSAGE, 'over-limit rejection must be the exact genuine budget error')
    over_limit = {'bytes':budget+4,'error_type':'ValueError','error':rejection,'original_digest':original_hash(over),
        'whole_unit_peaks':{'before':start,'after':peaks()},'peak_semantics':PEAK_SEMANTICS}
    over = None
    bounded()
    good = torch.zeros(budget//4096,dtype=torch.float32,device='cuda')
    bad = {'a_cuda':good,'b_cpu':torch.zeros(4,dtype=torch.float32)}
    refs = [weakref.ref(good),weakref.ref(bad['b_cpu'])]
    proposed_source.check()
    error = capture_error(proposed._fingerprint_cuda_dict,bad)
    proposed_source.check()
    good = bad = None
    gc.collect()
    follow = equal(pair({'ok':torch.arange(1024,dtype=torch.float32,device='cuda')}),'post-error digests differ')
    require(type(error) is ValueError and str(error) == CPU_MESSAGE, 'retained failure must be the exact genuine CPU-leaf error')
    require(all(ref() is None for ref in refs) and threading.active_count() == threads,
        'retained error kept tensors or workers alive')
    retained = {'error_type':type(error).__name__,'error':str(error),'threads_before':threads,
        'threads_after':threading.active_count(),'leaf_weakrefs_dead':True,'followup':follow}
    error = refs = None
    bounded()
    mlp_byte,unchanged,alias,restored = mutants(torch,pair,params,frozen,full,base,original.MLP)
    index._check_current()
    bounded()
    expected,trees,pairs = [base['frozen444']['original'],base['full448']['original']],[frozen,full],[]
    for order in ('OP','PO','OP'):
        row = {'order':order}
        for who in order:
            name,call = ('original',original.fingerprint) if who == 'O' else ('proposed',proposed._fingerprint_cuda_dict)
            proposed_source.check()
            torch.cuda.synchronize()
            started = time.perf_counter()
            digests = [call(tree) for tree in trees]
            torch.cuda.synchronize()
            row[name+'_seconds'] = time.perf_counter()-started
            proposed_source.check()
            require(digests == expected, 'timed digests differ')
        row['ratio_proposed_over_original'] = row['proposed_seconds']/row['original_seconds']
        pairs.append(row)
        bounded()
    timings = {'unit':'frozen444+full448','pairs':pairs,'median_ratio':statistics.median(
        r['ratio_proposed_over_original'] for r in pairs),
        'semantics':'3 alternating bare-call pairs; Source checks outside the clock; engineering evidence only, not p99 or confidence'}
    falsifier = {'trees':{'frozen444':{'leaves':444,'bytes':tree_bytes(frozen)},'full448':{'leaves':448,'bytes':tree_bytes(full)}},
        'identity':{'frozen_sha256':ident['frozen_sha256'],'vision_sha256':endpoint['vision_sha256']},'baseline':base,
        'edge_leaves':{'leaves':['empty','scalar'],'digests':edge_digests},'near_limit_transpose':near,
        'over_limit':over_limit,'retained_error':retained,'mlp_byte':mlp_byte,'unchanged_version_byte':unchanged,'aliases':alias,'restored':restored}
    require(threading.active_count() == threads, 'driver leaked threads')
    return {'falsifier':falsifier,'timings':timings}


def seconds(value):
    require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'finite nonnegative seconds required')
    return value


def check_bootstrap(value):
    """Exact bootstrap record: one unprofiled B1 request, typed native bytes, exact-four audit passed."""
    timing = 'seconds read_decode_seconds public_call_seconds completion_sync_seconds native_capture_seconds image_cleanup_seconds'
    require(type(value) is dict and value.keys() == set((timing+' batch kind image native instrumented qualification_eligible '
        'audit_seconds audit_exact_four ended_body_seconds').split()), 'exact bootstrap evidence required')
    for key in (*timing.split(),'audit_seconds','ended_body_seconds'): seconds(value[key])
    require(type(value['batch']) is int and value['batch'] == 1 and value['kind'] == 'bootstrap' and
        type(value['image']) is str and value['image'] and value['instrumented'] is False and
        value['qualification_eligible'] is False and value['audit_exact_four'] is True and value['seconds'] > 0 and
        abs(value['seconds']-value['read_decode_seconds']-value['public_call_seconds']-value['completion_sync_seconds']) < 1e-6,
        'complete bootstrap request timing differs')
    native = value['native']
    require(type(native) is list and len(native) == 2, 'complete bootstrap native ID/score witness required')
    for row,fmt,width in zip(native,('q','f'),(8,4),strict=True):
        formats = ('q','l') if fmt == 'q' and struct.calcsize('l') == 8 else (fmt,)
        require(type(row) is dict and row.keys() == {'format','shape','hex'} and row['format'] in formats and
            row['shape'] == [1,10] and all(type(v) is int for v in row['shape']) and type(row['hex']) is str and
            re.fullmatch('[0-9a-f]+',row['hex']) and len(row['hex']) == 10*width*2, 'typed bootstrap native ID/score bytes differ')
    require(all(v[0] >= 0 for v in struct.iter_unpack('<q',bytes.fromhex(native[0]['hex']))) and
        all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',bytes.fromhex(native[1]['hex']))),
        'finite bootstrap native score/nonnegative ID required')


def bootstrap(index, observer, read_images, synchronize, paths, api, legacy, bounded, started):
    """One genuine unprofiled public B1 request, then the original exact-four audit; nothing else precedes hashing."""
    require(type(paths) is list and len(paths) == 1, 'exactly one admitted TRAIN image required')
    bounded()
    result,row = observer.measure_request(index,read_images,synchronize,paths)
    result = None  # no native result or tensor alias survives; images were closed by the genuine request cleanup
    audited = time.perf_counter()
    api.audit_origins(legacy,require_exact=True)
    audit = time.perf_counter()-audited
    bounded()
    record = {**row,'batch':1,'kind':'bootstrap','image':str(paths[0]),'audit_seconds':audit,'audit_exact_four':True,
        'ended_body_seconds':time.perf_counter()-started}
    check_bootstrap(record)
    return record


def falsify(factory, guard, torch, proposed_source, observer, read_images, paths, api, legacy):
    started = time.perf_counter()
    admission,charges = started-STARTED,{}
    def bounded():
        resources = guard()
        require(time.perf_counter()-started < 300, 'fresh-sha body300 cap exceeded')
        return resources
    with requests.owner(factory,bounded,charges) as index:
        boot = bootstrap(index,observer,read_images,torch.cuda.synchronize,paths,api,legacy,bounded,started)
        hash_started = time.perf_counter()-started
        phases = measure(index,torch,proposed_source,bounded)
    index = None
    bounded()
    phases['bootstrap'],phases['hash_started_body_seconds'] = boot,hash_started
    phases['admission'] = {'driver_seconds_before_body':admission,'owner_admission_seconds':charges['admission_seconds'],
        'owner_release_seconds':charges['release_seconds']}
    return {'phases':phases,'body_seconds':time.perf_counter()-started,'forward_calls':1,
        'public_b1_b32_parity_required_later':True,
        'product_p99':'UNQUALIFIED; requires full public B1/B32 parity and10000interleaved paired calls'}


def validate_fresh_receipt(record, authority, authority_fact, proposal, proposal_fact, observation):
    def hex64(value):
        require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'SHA256 required')
        return value
    def pair(value):
        require(type(value) is dict and value.keys() == {'original','proposed'} and
            hex64(value['original']) == hex64(value['proposed']), 'equal original/proposed digests required')
        return value['original']
    def exact(value, keys):
        require(type(value) is dict and value.keys() == set(keys.split()), 'exact measurement evidence required')
    require(record['schema'] == RECEIPT_SCHEMA and record['status'] == 'DISCARDED_DIAGNOSTIC' and
        record['engineering_only'] is True and record['authority'] == authority_fact and
        record['proposal'] == proposal_fact and record['proposed'] == proposal['proposed'] and
        record['proposal_commit'] == proposal['commit'] and record['sources'] == authority['sources'] and
        record['control'] == CONTROL and record['control_export'] == authority['control_export'] and
        all(record[k] is False for k in FLAGS) and record['full_uncached_exit_pass'] is True and
        record['normal_terminal_required'] is True and record['owned_cleanup_requires_terminal'] is True and
        record['forward_calls'] == 1 and record['public_b1_b32_parity_required_later'] is True,
        'complete discarded fresh-sha receipt required')
    require(record['invocation']['argv'] == fresh_cli(authority_fact,proposal_fact,record['output'],proposal['driver']['path']) and
        record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == '0' and
        record['invocation']['cublas_workspace_config'] == ':4096:8', 'exact fresh-sha CLI/environment required')
    require(Path(proposal['driver']['path']).name == DRIVER_NAME and Path(proposal['test']['path']).name == TEST_NAME and
        all(record['input_guards'].get(f['path']) == f['sha256'] for f in proposal_files(proposal)),
        'receipt must bind the exact new driver/test/proposed FILEs')
    phases = record['phases']
    exact(phases,'admission bootstrap falsifier timings hash_started_body_seconds')
    exact(phases['admission'],'driver_seconds_before_body owner_admission_seconds owner_release_seconds')
    boot = phases['bootstrap']
    check_bootstrap(boot)
    first = observation['train_images'][0]
    requests.read_file(first)  # fresh SHA256 authentication of the admitted first TRAIN image
    require(record['observation'] == authority['observation'] and boot['image'] == first['path'] and
        record['input_guards'].get(first['path']) == first['sha256'],
        'bootstrap must be the first SHA-authenticated admitted TRAIN image')
    charged = (sum(seconds(v) for v in phases['admission'].values())-phases['admission']['driver_seconds_before_body'] +
        boot['seconds']+boot['native_capture_seconds']+boot['image_cleanup_seconds']+boot['audit_seconds'])
    require(0 < seconds(record['body_seconds']) <= 300 and charged <= record['body_seconds']+1e-6 and
        record['whole_process_seconds'] >= phases['admission']['driver_seconds_before_body']+record['body_seconds'],
        'body300 and phase accounting required')
    require(0 < boot['ended_body_seconds'] <= seconds(phases['hash_started_body_seconds']) <= record['body_seconds'],
        'bootstrap must precede all hash work')
    f = phases['falsifier']
    exact(f,'trees identity baseline edge_leaves near_limit_transpose over_limit retained_error mlp_byte unchanged_version_byte aliases restored')
    exact(f['trees'],'frozen444 full448')
    require(f['trees']['frozen444']['leaves'] == 444 and f['trees']['full448']['leaves'] == 448, 'exact tree inventory required')
    exact(f['baseline'],'frozen444 full448')
    require(pair(f['baseline']['frozen444']) == hex64(f['identity']['frozen_sha256']) and
        pair(f['baseline']['full448']) == hex64(f['identity']['vision_sha256']), 'baseline differs from encoder identity')
    pair(f['edge_leaves']['digests']); pair(f['near_limit_transpose']['digests'])
    near = f['near_limit_transpose']
    require(near['bytes'] == near['budget_bytes'] == BUDGET and near['peak_semantics'] == PEAK_SEMANTICS, 'near-limit dense transpose evidence required')
    over = f['over_limit']
    require(over['bytes'] == BUDGET+4 and over['error_type'] == 'ValueError' and over['error'] == OVER_MESSAGE and hex64(over['original_digest']) and
        over['peak_semantics'] == PEAK_SEMANTICS, 'over-limit rejection evidence required')
    for peaks in (near['whole_unit_peaks'],over['whole_unit_peaks']):
        exact(peaks,'before after')
        for row in peaks.values():
            exact(row,'process_peak_rss_kib peak_cuda_allocated_bytes')
            require(all(type(v) in (int,float) and math.isfinite(v) and 0 <= v for v in row.values()), 'finite nonnegative cumulative peaks required')
    retained = f['retained_error']
    require(retained['error_type'] == 'ValueError' and retained['error'] == CPU_MESSAGE and retained['leaf_weakrefs_dead'] is True and
        retained['threads_before'] == retained['threads_after'], 'retained-error lifecycle required')
    pair(retained['followup'])
    seen = {k:{pair(f['baseline'][k])} for k in ('frozen444','full448')}
    mlp = f['mlp_byte']
    require(mlp['byte'] == 0 and mlp['mask'] == 1 and pair(mlp['frozen444']) == pair(f['baseline']['frozen444']) and
        pair(mlp['full448']) != pair(f['baseline']['full448']), 'MLP byte flip evidence required')
    seen['full448'].add(pair(mlp['full448']))
    require(len(f['unchanged_version_byte']['steps']) == 3, 'three unchanged-version mutations required')
    for step in [*f['unchanged_version_byte']['steps'],f['aliases']['mutation']]:
        for k in seen:
            digest = pair(step[k])
            require(digest not in seen[k], 'mutation digests must be fresh and distinct')
            seen[k].add(digest)
    alias = f['aliases']
    require(alias['mutation']['base_bytes_changed'] is True and len(alias['views']) == 3 and
        pair(alias['tree_before']) != pair(alias['mutation']['tree_after']), 'strided alias mutation evidence required')
    restored = f['restored']
    require(restored['leaf_bytes_equal'] is True and restored['versions_unchanged'] is True and
        restored['frozen444'] == f['baseline']['frozen444'] and restored['full448'] == f['baseline']['full448'],
        'exact restored digests required')
    timings = phases['timings']
    exact(timings,'unit pairs median_ratio semantics')
    require([r['order'] for r in timings['pairs']] == ['OP','PO','OP'], 'three alternating pairs required')
    for r in timings['pairs']:
        exact(r,'order original_seconds proposed_seconds ratio_proposed_over_original')
        require(0 < seconds(r['original_seconds']) and 0 < seconds(r['proposed_seconds']) and
            math.isclose(r['ratio_proposed_over_original'],r['proposed_seconds']/r['original_seconds'],rel_tol=1e-9),
            'recomputed pair ratio differs')
    require(math.isclose(timings['median_ratio'],statistics.median(r['ratio_proposed_over_original'] for r in timings['pairs']),
        rel_tol=1e-9), 'recomputed median ratio differs')


def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
        'unoptimized -B unprofiled startup required')
    require(not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'} for n in sys.modules),
        'native import preceded control admission')
    authority_fact = {'path':str(args.authority),'sha256':args.authority_sha256}
    authority = requests.strict_json(requests.read_file(authority_fact))
    require(type(authority) is dict and authority.keys() == KEYS and authority['schema'] == SCHEMA and
        authority['control'] == CONTROL and type(authority['control']['seed']) is int, 'exact separate control authority required')
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() == SOURCES, 'complete actual control source pins required')
    for role,name in (('control_driver','qualify_connected_control_serving.py'),
        ('control_native','connected_control_native_authority.py'),('control_test','test_connected_control_serving.py'),
        ('request_driver','qualify_connected_serving_requests.py'),('request_test','test_connected_serving_requests.py'),
        ('observer','observe_connected_serving.py'),('observer_test','test_observe_connected_serving.py')):
        require(sources[role]['path'] == str(Path(__file__).with_name(name).absolute()), 'current source role differs: '+role)
    proposal_fact = {'path':str(args.proposal),'sha256':args.proposal_sha256}
    proposal = read_proposal(proposal_fact,sources)
    require(sys.argv == fresh_cli(authority_fact,proposal_fact,str(args.output),proposal['driver']['path']), 'fixed canonical fresh-sha CLI order required')
    request_source = requests.Source(requests,sources['request_driver'])
    control_source = requests.Source(control,sources['control_driver'])
    self_source = requests.Source(sys.modules[__name__],proposal['driver'])
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink(),
        'exclusive canonical new output required')
    owned,failures,context,exit_guard,api,before,guard = [],[],None,None,None,None,None
    try:
        locks = requests.Locks(authority['locks'])
        proposed_source = requests.Source.load(proposal['proposed']['runtime']); owned.append(proposed_source)
        require_no_native()
        observer_source = requests.Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        observation = prepare_observation(authority['observation'],observer)
        require(observation['sources']['observer'] == sources['observer'] and
            observation['sources']['test'] == sources['observer_test'] and observation['sources']['bridge'] == sources['bridge'] and
            all(observation['sources'][role] == sources[role] for role in ('runtime','ledger','packed')),
            'control observation source binding differs')
        for file in sources.values(): observer.file_bytes(file)
        native_source = requests.Source.load(sources['control_native']); owned.append(native_source)
        runtime = requests.strict_json(observer.file_bytes(authority['native_runtime'],keep=True))
        require(runtime['library'] == observation['native'], 'same frozen control native FILE required')
        native_source.module.validate_runtime_compiler(runtime,observer)
        fact = authority['evaluator']
        evaluator_source = native_source.module.load_evaluator_source(
            {'path':str(Path(fact['root'])/'evaluate_siglip2_connected_mlp.py'),
             'sha256':fact['code']['evaluate_siglip2_connected_mlp.py']},requests); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        evaluator.check_code(fact,evaluator.FILES)
        require(evaluator.closure(fact['root'],fact['execution_sha256'],evaluator.FILES,{}) == fact['code'],
            'complete original evaluator CODE differs')
        eargs = SimpleNamespace(execution_sha256=fact['execution_sha256'],
            authority=Path(authority['evaluation_authority']['path']),authority_sha256=authority['evaluation_authority']['sha256'],
            phase='export',arm='control',seed=179061,output=output)
        context,exit_guard = evaluator.authority(eargs)
        context['training_context']['fit_context']['unit_started'] = STARTED
        endpoint,exported = admit_control(evaluator,context,authority,observation)
        frozen = [authority_fact,authority['observation'],authority['native_runtime'],authority['evaluation_authority'],
            *sources.values(),*observation['sources'].values(),observation['bundle']['manifest'],observation['control_export_receipt'],
            observation['gallery']['file'],observation['native'],*observation['train_images'],
            proposal_fact,*proposal_files(proposal)]
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in frozen})
        runtime_authority = native_source.module.CombinedAuthority(context['training_context'],authority['native_runtime'],observer,requests)
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in runtime_authority.provenance_facts()})
        api = runtime_authority.install(evaluator_source,context)
        policy = observation['resource_policy']
        require(policy['whole_process_seconds'] <= evaluator.policy('export')['seconds'] and
            time.perf_counter()-STARTED+300+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
            'insufficient frozen admission/body/exit headroom')
        before = evaluator.native_start(context)
        import torch
        from PIL import Image
        from sfora import cutile_int8,joint_relational_compaction,packed_int8
        wrapper_source = requests.Source(cutile_int8,sources['native_wrapper'])
        packed_source = requests.Source(packed_int8,sources['packed'])
        packing_source = requests.Source(joint_relational_compaction,sources['packing'],packed_source=packed_source)
        bridge_source = requests.Source.load(sources['bridge']); owned.append(bridge_source)
        torch.random.default_generator.manual_seed(179061); torch.cuda.manual_seed_all(179061)
        rng,cuda_rng = torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
        def guard(*,reserve=True):
            locks.check()
            for source in (self_source,request_source,observer_source,native_source,evaluator_source,
                bridge_source,wrapper_source,packed_source,packing_source,control_source,proposed_source): source.check()
            for file in frozen: observer.file_bytes(file)
            observer.check_runtime_sources(observation['sources'],observation['bundle'])
            api.authenticate()
            evaluator.guard_helpers(context)
            resources = evaluator.resources(context,before)
            resources['wall_seconds'] = time.perf_counter()-STARTED
            requests.check_resources(resources,policy,reserve=reserve)
            require(torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in
                zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit RNG changed')
            return resources
        guard()
        ties = requests.native_ties(joint_relational_compaction.PackedInt8Embeddings,
            cutile_int8.CutilePackedInt8Gallery,observation['native'],observer)
        api.audit_origins(context['training_context']['legacy'])
        guard()
        def read_images(paths): return requests.decode_images(observer,Image,observation['train_images'],paths)
        with evaluator.endpoint_scope(context,endpoint):
            payloads = evaluator.authenticate_payloads(context,endpoint)
            require(payloads == context['cpu']['payload_facts'][evaluator.label(endpoint)], 'CPU-qualified control payload differs')
            rows,mapping = context['nearest_evaluator'].image_rows(context,'selection')
            selected = {r['path']:r for r in rows}
            require(all(f['path'] in selected and selected[f['path']]['image_sha256'] == f['sha256']
                for f in observation['train_images']), 'ordered TRAIN image membership/bytes differ')
            name = evaluator.label(endpoint)+'.packed.bin'
            wire = requests.read_file({'path':str(Path(exported['output'])/name),'sha256':exported['files'][name]})
            gallery_wire = b''.join(wire[i*130:(i+1)*130] for i in mapping['gallery'])
            require(len(wire) == len(rows)*130 and observation['gallery']['count'] == len(mapping['gallery']) and
                hashlib.sha256(gallery_wire).hexdigest() == observation['gallery']['file']['sha256'],
                'original control TRAIN gallery row/wire binding differs')
            wire = gallery_wire = None
            def factory():
                return bridge_source.module.ConnectedCompactIndex.from_bundle(
                    bundle_dir=Path(observation['bundle']['directory']),expected_bundle_sha256=endpoint['bundle']['sha256'],
                    gallery_path=Path(observation['gallery']['file']['path']),expected_gallery_sha256=observation['gallery']['file']['sha256'],
                    gallery_count=observation['gallery']['count'],native_library_path=Path(observation['native']['path']),
                    expected_native_library_sha256=observation['native']['sha256'])
            require(time.perf_counter()-STARTED+300+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
                'insufficient whole-process body/exit headroom')
            diagnostic = falsify(factory,guard,torch,proposed_source,observer,read_images,
                [Path(observation['train_images'][0]['path'])],api,context['training_context']['legacy'])
        guard()
        prior = context['training_context']['legacy']['selected']['source_cpu']['invocation']
        record = {'schema':RECEIPT_SCHEMA,'status':'DISCARDED_DIAGNOSTIC','engineering_only':True,
            'authority':authority_fact,'proposal':proposal_fact,'proposed':proposal['proposed'],'proposal_commit':proposal['commit'],
            'sources':sources,'control':CONTROL,'control_export':authority['control_export'],
            'observation':authority['observation'],'native_runtime':authority['native_runtime'],'output':str(output),
            'ties':ties,**diagnostic,'quality_read':False,'quality_eligible':False,'qualification_eligible':False,
            'state_reuse_eligible':False,'optimization_eligible':False,'product_go':False,'speed_go':False,'resource_policy':policy,
            'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
            'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
                'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
                'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    except BaseException as error:
        failures.append(error)
    finally:
        if context is not None:
            try:
                if api is None: evaluator.exit_rehash(context,exit_guard)
                else: api.evaluator_exit(context,exit_guard)
                if guard is not None: final_resources = guard(reserve=False)
            except BaseException as error: failures.append(error)
        if not failures:
            try:
                record['combined_native'] = api.evidence()
                final_resources = guard(reserve=False)
                record['full_uncached_exit_pass'] = True
                record['resources'] = final_resources
                record['input_guards'] = dict(context['guards'])
                record['whole_process_seconds'] = time.perf_counter()-STARTED
                validate_fresh_receipt(record,authority,authority_fact,proposal,proposal_fact,observation)
                requests.check_resources({**final_resources,'wall_seconds':record['whole_process_seconds']},policy,reserve=False)
                locks.check()
                output.mkdir()
                context['helper'].publish(output/'receipt.json',record)
            except BaseException as error: failures.append(error)
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module: del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned source registry changed at exit'))
        if not failures:
            try:
                locks.check()
                require(time.perf_counter()-STARTED < policy['whole_process_seconds'], 'publication/cleanup included whole-process cap exceeded')
            except BaseException as error: failures.append(error)
        if failures: requests.raise_failures(failures)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--authority-sha256',required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--proposal-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
