#!/usr/bin/env python3
"""Source-only same-terminal-readout SS/SF/FS/FF diagnostic; native gate UNRUN.

Root freezes exactly this file and test_connected_gallery_freshness.py in
execution.json, then supplies connected-gallery-freshness-launch-v1. CLI:
--execution-sha256 SHA --authority FILE --authority-sha256 SHA --output NEWDIR.
FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}.
Authority keys are LAUNCH_KEYS below; sources are exactly SOURCE_NAMES, each
a FILE in the accepted full score's input_guards. runtime has exactly
native_authority/source_cpu/warm FILEs in their original authenticated roles.
score_terminal is the original UNIT, not a fabricated proof-to-receipt role.

Original CPU/warm origins and the existing exact vendor supplement remain the
runtime authority. Reuse the original imported_origins, bind_native_authority,
inverse-checked private audit adaptation, source_live_guard, FlatAdmission and
terminal/cgroup predicates. Never construct a broad legacy/fitter native owner,
replace original globals, or traverse inherited guards. Only explicit metadata,
descriptor/cache/endpoint bytes and admitted source/runtime files may be read.

All four stored wires and accepted per-query R1/AP replay precede new readouts.
Both complete terminal controls must reproduce raw/unit BITS and packed bits,
including original role B32/tails6/19. An instrument difference stops the job.
S uses CUDA FP32 readout/normalization, with direct original FIT cache rows;
F is the same candidate's accepted fresh descriptor. The scorer stays original
CPU. Independent reloads and omitted-C/wrong-mu negatives precede results.
700s/8GiB/noSwap/CUDA<10GB/bothlocks remain fixed. Root owns launch and terminal
verification. Outputs describe exposed selection only; KILL stays unchanged.
No survivor, training, VAL/official, new images, causal-null or speed claim.
"""
if not __debug__:
    raise SystemExit('optimized mode forbidden; original assertions required')

import __future__
import argparse
import ast
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
import struct
import sys
import time
from types import FunctionType

FILES = {'diagnose_connected_gallery_freshness.py', 'test_connected_gallery_freshness.py'}
SOURCE_NAMES = {'evaluator','connected','identity','original','cached','quadratic','quadratic_owner','readout',
                'packing','source_driver','extract','nearest','fitter','initializer','baseline','scorer'}
SOURCE_FILES = dict(zip(('evaluator','connected','identity','original','cached','quadratic','quadratic_owner','readout',
                        'packing','source_driver','extract','nearest','fitter','initializer','baseline','scorer'),
    ('evaluate_siglip2_connected_mlp.py','train_siglip2_connected_mlp.py','train_siglip2_identity_diversity.py',
     'train_siglip2_substrate_adaptation.py','train_siglip2_cached_readout.py','quadratic_readout.py',
     'train_siglip2_quadratic_readout.py','prototype_residual_readout.py','joint_relational_compaction.py',
     'qualify_siglip2_substrate_cpu.py','extract_siglip2_vision_source.py','train_siglip2_nearest_ranking.py',
     'fit_siglip2_prototype_residual.py','export_siglip2_substrate_fit.py','evaluate_siglip2_quadratic_readout.py',
     'reference_compare_inshop_sop_warmstart_100.py'),strict=True))
LAUNCH_KEYS = {'schema','execution_sha256','score','score_verification','score_terminal','partition',
               'fit','original_cache','scope','sources','runtime','resource_policy','both_locks_held',
               'candidate_status','qualification_eligible','state_reuse_eligible'}
LIMITS = {'seconds':700,'host_bytes':8*1024**3,'swap_bytes':0,'cuda_visible_devices':'0',
          'cuda_allocated_bytes_exclusive':10_000_000_000}
ORDER = ((179061,'control'),(179061,'candidate'),(179069,'candidate'),(179069,'control'))
SCORE_SHA = '01ae023cb89b828c029817582cdb48204ff8177e76047ccec0773e5d90aafbfa'
IDENTITY_SHA = '840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8'
SCORER_SHA = '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250'
SCORER_AST = '717489188a008ceba1b930d9e7dff32b90cd5347302835538eb171fe23f4b33e'


def require(condition, message):
    if not condition: raise ValueError(message)


def file_fact(fact):
    require(type(fact) is dict and fact.keys() == {'path','sha256'} and
            type(fact['path']) is str and type(fact['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}',fact['sha256']), 'actual exact FILE required')
    path=Path(fact['path'])
    require(path.is_absolute() and str(path)==fact['path'] and path.resolve()==path and
            path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    require(path.suffix.lower() not in {'.jpg','.jpeg','.png','.webp','.bmp','.tiff'} and
            not (path.suffix=='.pt' and path.name!='endpoint.pt'), 'forbidden images/model/checkpoint FILE')
    return path


def file_bytes(fact, guards, *, keep=False):
    path=file_fact(fact);before=path.stat(); digest=hashlib.sha256();raw=None
    with path.open('rb') as stream:
        if keep:
            raw=stream.read(64*1024**2+1)
            require(len(raw)<=64*1024**2,'metadata/source/wire FILE exceeds64MiB')
            digest.update(raw)
        else:
            while block:=stream.read(1024**2): digest.update(block)
        after=os.fstat(stream.fileno())
    require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
            (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns) and
            path.stat()==after,'FILE changed while reading')
    require(digest.hexdigest()==fact['sha256'],'current FILE SHA differs: '+str(path))
    require(guards.setdefault(str(path),fact['sha256'])==fact['sha256'],'conflicting FILE authority')
    return raw if keep else path


def read_json(fact, guards):
    def pairs(items):
        value=dict(items);require(len(value)==len(items),'duplicate JSON key');return value
    return json.loads(file_bytes(fact,guards,keep=True),object_pairs_hook=pairs,
                      parse_constant=lambda value:require(False,'nonfinite JSON'))


def rehash_files(guards):
    for path,digest in tuple(guards.items()):file_bytes({'path':path,'sha256':digest},{})


def npy_header(fact, shape, guards):
    path=file_bytes(fact,guards)
    with path.open('rb') as stream:
        magic=stream.read(8);require(magic in (b'\x93NUMPY\x01\x00',b'\x93NUMPY\x02\x00'),'NPY version differs')
        size=2 if magic[-2]==1 else 4;length=int.from_bytes(stream.read(size),'little')
        require(0<length<=65536,'NPY header size differs')
        header=ast.literal_eval(stream.read(length).decode('latin1'))
        require(type(header) is dict and header.keys()=={'descr','fortran_order','shape'} and
                header['descr']=='<f4' and header['fortran_order'] is False and
                type(header['shape']) is tuple and header['shape']==shape,'NPY shape/dtype/endian/layout differs')
        require(path.stat().st_size==8+size+length+math.prod(shape)*4,'NPY payload size differs')
    return header


def roles(panel, count):
    query,gallery=panel['query'],panel['gallery']
    require(type(query) is list and type(gallery) is list and query and gallery and
            all(type(i) is int and 0<=i<count for i in query+gallery) and
            len(set(query+gallery))==len(query)+len(gallery)==count,'complete unique query/gallery mapping required')
    return query,gallery


def panel_mapping(partition, fit, scope, *, counts=(3449,1734,1715,498),train_counts=(6355,1008)):
    panel=partition['panels']['selection'];rows=panel['original_rows'];query,gallery=roles(panel,counts[0])
    require(len(rows)==counts[0] and len(set(rows))==len(rows) and
            all(type(r) is int and 0<=r<len(fit['targets']) for r in rows) and
            (len(query),len(gallery),len(panel['original_class_ids']))==counts[1:],'selection direct FIT panel differs')
    labels=tuple(fit['class_names'][fit['targets'][r]] for r in rows)
    require(len(set(labels))==counts[3] and set(labels[i] for i in query)==set(labels[i] for i in gallery),
            'selection positive/product inventory differs')
    active=scope['control'];train=[r['original_fit_index'] for r in active['rows']]
    require(train==partition['panels']['train']['original_rows'] and len(train)==train_counts[0] and
            len(set(train))==len(train) and len(active['class_names'])==train_counts[1] and
            not set(train).intersection(rows) and not set(active['class_names']).intersection(labels),
            'actual control FIT scope/selection disjointness differs')
    return labels


def decode_wire(wire, count):
    require(type(wire) is bytes and len(wire)==count*130,'stored wire layout/length differs')
    codes=[];inverse=[]
    for start in range(0,len(wire),130):
        codes.append(struct.unpack_from('<128b',wire,start));v=struct.unpack_from('<e',wire,start+128)[0]
        require(math.isfinite(v) and v>0,'finite positive wire inverse required');inverse.append(v)
    return codes,inverse


def exact_wire(actual, expected):
    require(actual==expected,'exact stored wire/endian bytes differ')


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def wire_quality(wire, labels, query, gallery):
    """Small stdlib reference; production uses decoded integer CPU tensors below."""
    codes,inverse=decode_wire(wire,len(labels));hits=[];aps=[]
    for q in query:
        scores=[f32(f32(sum(a*b for a,b in zip(codes[q],codes[g]))*inverse[q])*inverse[g]) for g in gallery]
        order=sorted(range(len(gallery)),key=lambda i:-scores[i]);r=sum(labels[q]==labels[g] for g in gallery)
        require(r>0,'positive inventory required');positive=0;ap=0.
        for rank,i in enumerate(order[:r],1):
            if labels[q]==labels[gallery[i]]:positive+=1;ap=f32(ap+f32(positive/rank))
        hits.append(int(labels[q]==labels[gallery[order[0]]]))
        aps.append(f32(ap/r))
    return {'per_query_r1':hits,'per_query_ap':aps,'recall_at_1':sum(hits)/len(hits),'map_at_r':sum(aps)/len(aps)}


def replay(expected, actual):
    require(expected['per_query_r1']==actual['per_query_r1'],'per-query R1 replay differs')
    require(expected['per_query_ap']==actual['per_query_ap'],'per-query AP replay differs')


def compose(stale, fresh, query, gallery, cell):
    require(cell in ('SS','SF','FS','FF'),'fixed cell required')
    if isinstance(stale,list):
        values=copy.deepcopy(stale)
        for indices,side in ((query,cell[0]),(gallery,cell[1])):
            if side=='F':
                for i in indices:values[i]=copy.deepcopy(fresh[i])
    else:
        values=stale.clone()
        if cell[0]=='F':values[query]=fresh[query]
        if cell[1]=='F':values[gallery]=fresh[gallery]
    return values


def effects(cells):
    return {metric:{
        'query_at_stale_gallery':[a-b for a,b in zip(cells['FS'][metric],cells['SS'][metric],strict=True)],
        'gallery_at_stale_query':[a-b for a,b in zip(cells['SF'][metric],cells['SS'][metric],strict=True)],
        'interaction':[ff-fs-sf+ss for ff,fs,sf,ss in zip(*(cells[k][metric] for k in ('FF','FS','SF','SS')),strict=True)]}
        for metric in ('per_query_r1','per_query_ap')}


def byte_differences(actual, expected):
    result={}
    for name in ('raw','unit','codes','inverse_norms','wire'):
        a,b=actual[name],expected[name]
        result[name]={'exact':a==b,'different_bytes':sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b))}
        if name in ('raw','unit') and len(a)==len(b) and len(a)%4==0:
            result[name]['max_absolute_difference']=max((abs(x[0]-y[0]) for x,y in
                zip(struct.iter_unpack('<f',a),struct.iter_unpack('<f',b))),default=0.)
    return result


def control_byte_differences(actual, expected, panel, fit):
    """Localize retained control bytes; tail confinement never establishes cause."""
    result=byte_differences(actual,expected);mapping={};count=len(panel['original_rows']);fit_count=len(fit['targets'])
    query,gallery=roles(panel,count)
    require(len(fit['rows'])==fit_count and all(type(r) is int and 0<=r<fit_count for r in panel['original_rows']),
            'original FIT row mapping required')
    for role,indices in (('query',query),('gallery',gallery)):
        for index,i in enumerate(indices):
            original=panel['original_rows'][i];start=index//32*32;cache_start=original//32*32
            size=min(32,len(indices)-start)
            mapping[i]={'mapping_status':'AVAILABLE','role':role,'role_index':index,
                'live_encoder_batch_index':index//32,'live_encoder_batch_size':size,
                'live_encoder_batch_row':index%32,'live_encoder_tail':size<32,
                'original_fit_index':original,'official_train_row':fit['rows'][original]['train_row'],
                'original_cache_batch_index':original//32,'original_cache_batch_start':cache_start,
                'original_cache_batch_size':min(32,fit_count-cache_start),'original_cache_batch_row':original%32}
    for name,width in (('raw',512),('unit',512),('codes',128),('inverse_norms',2),('wire',130)):
        a,b=actual[name],expected[name];rows=[]
        for start in range(0,max(len(a),len(b)),width):
            left,right=a[start:start+width],b[start:start+width]
            different=sum(x!=y for x,y in zip(left,right))+abs(len(left)-len(right))
            if different:
                i=start//width
                rows.append({'selection_row':i,'different_bytes':different,'actual_bytes':len(left),
                    'expected_bytes':len(right),**mapping.get(i,{'mapping_status':'UNAVAILABLE'})})
        complete=len(a)==len(b)==count*width
        hypothesis='FALSIFIED' if any(r.get('live_encoder_tail') is False for r in rows) else (
            'UNAVAILABLE' if not complete else 'NOT_FALSIFIED' if rows else 'NO_MISMATCH')
        result[name]['row_localization']={'status':'AVAILABLE' if all(r['mapping_status']=='AVAILABLE' for r in rows)
            else 'UNAVAILABLE','row_bytes':width,'actual_bytes':len(a),'expected_bytes':len(b),'rows':rows,
            'tail_only_hypothesis':hypothesis,'causation_established':False}
    return result


def require_control_tap(differences):
    require(all(v['exact'] for v in differences.values()),'control tap differs; counterfactual stopped')


class Budget:
    def __init__(self,started=None,clock=time.perf_counter):
        self.clock=clock;self.started=clock() if started is None else started
    def check(self):
        require(self.clock()-self.started<700,'whole700-second diagnostic cap reached')


def cleanup_error(error, callbacks):
    failures=[]
    for callback in callbacks:
        try:callback()
        except BaseException as failure:failures.append(failure)
    if error is not None:
        for failure in failures:error.add_note('cleanup: '+repr(failure))
        raise error
    if failures:
        for failure in failures[1:]:failures[0].add_note('cleanup: '+repr(failure))
        raise failures[0]


def prepare(args):
    """Current bytes, exact roles and narrow metadata; no native imports."""
    guards={};root=Path(__file__).absolute().parent
    code=read_json({'path':str(root/'execution.json'),'sha256':args.execution_sha256},guards)
    require(type(code) is dict and code.keys()==FILES,'exact2 diagnostic closure required')
    for name,digest in code.items():file_bytes({'path':str(root/name),'sha256':digest},guards)
    launch=read_json({'path':str(args.authority),'sha256':args.authority_sha256},guards)
    require(type(launch) is dict and launch.keys()==LAUNCH_KEYS and
            launch['schema']=='connected-gallery-freshness-launch-v1' and
            launch['execution_sha256']==args.execution_sha256 and launch['resource_policy']==LIMITS and
            launch['both_locks_held'] is True and launch['candidate_status']=='KILL' and
            launch['qualification_eligible'] is False and launch['state_reuse_eligible'] is False,
            'exact diagnostic-only KILL authority/caps required')
    require(launch['score']['sha256']==SCORE_SHA and launch['score_terminal']['receipt']==launch['score'],
            'original full score FILE/UNIT required')
    score=read_json(launch['score'],guards);verification=read_json(launch['score_verification'],guards)
    require(score['schema']=='siglip2-connected-mlp-evaluation-v1' and score['phase']=='score' and
            score['stage']=='full' and score['panel']=='selection' and score['decision']=='KILL' and
            all(score[k] is True for k in ('pass','integrity_pass','exit_rehash_pass','resources_pass',
                'rng_flags_preserved','persisted_wire_scoring_replay_exact','all_export_wires_readback_before_quality')) and
            score['official_read'] is False and score['validation_quality_exposed'] is False,
            'accepted original complete selection score required')
    require(verification['pass'] is True and verification['decision']=='KILL' and
            verification['terminal']==launch['score_terminal'] and
            verification['independently_authenticated_currentCPU_and_four_current_full_exports_metadata_pass'] is True and
            verification['original_normal_terminal_duration_rss_and_cgroup_predicates_checked'] is True,
            'original score parent verification/UNIT differs')
    historical=score['input_guards']
    def original(fact):
        require(historical.get(fact['path'])==fact['sha256'],'FILE absent from original full-score authority')
        return read_json(fact,guards)
    partition=original(launch['partition']);fit=original(launch['fit']);scope=original(launch['scope'])
    require(launch['partition']['sha256']==score['source']['partition_sha256'] and
            launch['fit']==partition['original_fit'] and launch['original_cache']==partition['original_cache'] and
            launch['scope']==score['launch']['scope'],'original FIT/cache/partition/scope bindings differ')
    require(historical.get(launch['original_cache']['path'])==launch['original_cache']['sha256'],
            'original full cache pin differs')
    npy_header(launch['original_cache'],(13283,1152),guards)
    labels=panel_mapping(partition,fit,scope)
    sources=launch['sources'];require(type(sources) is dict and sources.keys()==SOURCE_NAMES,'exact source FILE roles required')
    for name,fact in sources.items():
        require(Path(fact['path']).name==SOURCE_FILES[name],'original source FILE role differs')
        require(historical.get(fact['path'])==fact['sha256'],'source FILE absent from original authority')
        file_bytes(fact,guards)
    require(sources['identity']['sha256']==IDENTITY_SHA and sources['scorer']['sha256']==SCORER_SHA and
            sources['initializer']['sha256']=='163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8' and
            sources['fitter']['sha256']=='95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b' and
            sources['connected']['sha256']==score['launch']['training']['code']['train_siglip2_connected_mlp.py'] and
            sources['evaluator']['sha256']==score['source_code']['evaluate_siglip2_connected_mlp.py'],
            'actual source-v5 readout/scorer/connected/evaluator required')
    runtime=launch['runtime'];require(type(runtime) is dict and runtime.keys()=={'native_authority','source_cpu','warm'},
                                    'exact runtime authority roles required')
    require(runtime['native_authority']==score['source']['native_authority'] and
            runtime['source_cpu']==score['source']['native_source']['source_cpu']['proof'] and
            runtime['warm']==score['source']['warm_start']['terminal']['receipt'],'original runtime proof roles differ')
    source_cpu=original(runtime['source_cpu']);warm=original(runtime['warm'])
    original(runtime['native_authority'])
    cpu=original(score['launch']['selected_cpu']['receipt']);exports={};manifests={};endpoints=score['launch']['endpoints']
    require([(e['seed'],e['arm']) for e in endpoints]==list(ORDER),'exact original four endpoints required')
    require(cpu['phase']=='cpu' and cpu['stage']=='full' and cpu['panel']=='selection' and
            cpu['launch']['endpoints']==endpoints and cpu['payload_facts'].keys()==score['launch']['exports'].keys() and
            cpu['updated_payloads_authenticated'] is True,'accepted full CPU four-member binding differs')
    panel=partition['panels']['selection']
    for endpoint in endpoints:
        key=label(endpoint);record=original(score['launch']['exports'][key]['receipt']);manifest=original(endpoint['bundle'])
        require((record['phase'],record['stage'],record['panel'],record['seed'],record['arm'])==
                ('export','full','selection',endpoint['seed'],endpoint['arm']) and
                record['launch']['selected_cpu']==score['launch']['selected_cpu'] and
                record['payload_facts']==cpu['payload_facts'][key] and
                record['inference_state_sha256']==endpoint['inference_state_sha256'] and
                record['quality_read'] is False and all(record[k] is True for k in (
                    'strict_independent_reload_exact','full_updated_state_exact','raw_unit_packed_readback_exact',
                    'same_role_oracle_exact','native_exact_four_post_calibration')),
                'original full export endpoint/CPU witness differs')
        require(manifest['schema']=='siglip2-connected-mlp-bundle-v1' and
                manifest['endpoint_state_sha256']==endpoint['inference_state_sha256'] and
                manifest['code']['train_siglip2_connected_mlp.py']==sources['connected']['sha256'],
                'original TRAIN bundle typed-state/code binding differs')
        path=Path(endpoint['bundle']['path']).parent/'endpoint.pt'
        payload={'path':str(path),'sha256':manifest['files']['endpoint.pt']}
        require(historical.get(str(path))==payload['sha256'] and record['input_guards'].get(str(path))==payload['sha256'],
                'original endpoint member FILE differs')
        file_bytes(payload,guards)
        require(record['files'].keys()=={key+s for s in ('.raw.npy','.unit.npy','.packed.bin')},'exact descriptor files required')
        for name,digest in record['files'].items():
            fact={'path':str(Path(record['output'])/name),'sha256':digest}
            require(historical.get(fact['path'])==digest,'descriptor absent from original score authority')
            if name.endswith('.npy'):npy_header(fact,(3449,128),guards)
            else:decode_wire(file_bytes(fact,guards,keep=True),3449)
        check_export_mapping(record,panel,fit)
        exports[key]=record;manifests[key]=manifest
    output=Path(args.output)
    require(output.is_absolute() and output.parent.resolve()==output.parent and not output.exists(),
            'exclusive canonical output directory required')
    return dict(args=args,launch=launch,score=score,cpu=cpu,exports=exports,manifests=manifests,
                partition=partition,fit=fit,scope=scope,labels=labels,guards=guards,source_cpu=source_cpu,warm=warm)


def label(endpoint):return endpoint['arm']+'-'+str(endpoint['seed'])


def check_export_mapping(record, panel, fit):
    expected=[(role,indices[start:start+32]) for role in ('query','gallery')
              for indices in (panel[role],) for start in range(0,len(indices),32)]
    require(len(record['images'])==len(expected),'complete original role batch count differs')
    for batch,(role,indices) in zip(record['images'],expected,strict=True):
        require(batch['role']==role and len(batch['rows'])==len(indices),'original role B32/tail differs')
        for row,i in zip(batch['rows'],indices,strict=True):
            r=panel['original_rows'][i];original=fit['rows'][r]
            require(row['panel_ordinal']==i and row['original_row']==r and row['role']==role and
                    row['target']==fit['targets'][r] and row['product']==fit['class_names'][fit['targets'][r]] and
                    all(row[k]==original[k] for k in ('train_row','relative_path','image_sha256')),
                    'original panel/FIT/export row metadata differs')
    require(record['batch_sizes']=={r:[len(indices[s:s+32]) for s in range(0,len(indices),32)]
                                   for r in ('query','gallery') for indices in (panel[r],)},'original role sizes differ')


class Sources:
    """Genuine modules and the accepted evaluator's existing live source guards."""
    def __init__(self,context):
        self.context=context;self.modules={};self.checks=[];self.error=None
    def load(self,name):
        fact=self.context['launch']['sources'][name];raw=file_bytes(fact,self.context['guards'],keep=True)
        module_name='_gallery_freshness_'+name
        require(module_name not in sys.modules,'source namespace already owned')
        spec=importlib.util.spec_from_file_location(module_name,fact['path']);module=importlib.util.module_from_spec(spec)
        sys.modules[module_name]=module;self.modules[name]=module
        # Compile the authenticated bytes rather than trusting a stale pyc.
        exec(compile(raw,fact['path'],'exec',dont_inherit=True),vars(module))
        return module
    def guard(self):
        for check in self.checks:check()
    def close(self):
        def remove(module):
            require(sys.modules.get(module.__name__) is module,'source registry ownership changed')
            del sys.modules[module.__name__]
        callbacks=[lambda module=module:remove(module) for module in reversed(tuple(self.modules.values()))]
        callbacks += [self.modules.clear,self.checks.clear,gc.collect]
        cleanup_error(None,callbacks)
    def admit(self):
        evaluator=self.load('evaluator')
        # The validator's builtin baseline contains identity-only builtin objects.
        # Its own deepcopy-literal guard is unsuitable for guarding that module.
        bindings=dict(vars(evaluator));functions=[];seen=set()
        def capture(fn):
            if type(fn) is not FunctionType or id(fn) in seen:return
            seen.add(id(fn));cells=tuple((c,c.cell_contents) for c in fn.__closure__ or ())
            functions.append((fn,fn.__code__,fn.__defaults__,copy.deepcopy(fn.__kwdefaults__),fn.__globals__,cells))
            for _,value in cells:capture(value)
        for value in bindings.values():capture(value)
        literals={k:copy.deepcopy(v) for k,v in bindings.items() if k not in ('__builtins__','_SOURCE_BUILTINS') and
                  isinstance(v,(dict,list,tuple,set,frozenset))}
        def evaluator_guard():
            require(sys.modules.get(evaluator.__name__) is evaluator and vars(evaluator).keys()==bindings.keys() and
                    all(vars(evaluator)[k] is v for k,v in bindings.items()) and
                    all(vars(evaluator)[k]==v for k,v in literals.items()),'original evaluator namespace changed')
            for fn,code,defaults,kw,namespace,cells in functions:
                require(fn.__code__ is code and fn.__defaults__==defaults and fn.__kwdefaults__==kw and
                        fn.__globals__ is namespace and all(c.cell_contents is v for c,v in cells),
                        'original evaluator function/closure changed')
            file_bytes(self.context['launch']['sources']['evaluator'],{})
        self.checks.append(evaluator_guard)
        for name in SOURCE_NAMES-{'evaluator','scorer','packing'}:
            module=self.load(name)
            self.checks.append(evaluator.source_live_guard(module,self.context['launch']['sources'][name]['sha256'],
                self.context['guards'],class_name='FlatAdmission' if name=='original' else None))
        self.guard()


def origin_audit(context, sources):
    """Only the original existing exact-supplement audit adaptation; no broad API."""
    modules=sources.modules;nearest=modules['nearest'];old=modules['quadratic_owner'];guards=context['guards']
    owner={'guards':guards,'launch':{'native_authority':context['launch']['runtime']['native_authority']},'source':{}}
    supplement,_,_=nearest.bind_native_authority(owner)
    # This is the original origin primitive's actual narrow context, not a fitter.
    legacy={'source_driver':modules['source_driver'],'extract':modules['extract'],'prior':{'guards':guards},
            'selected':{'packages':context['source_cpu']['origins']['packages'],'source_cpu':context['source_cpu']},
            'warm_record':context['warm'],'guards':guards}
    require(context['cpu']['origins']['packages']==legacy['selected']['packages'] and
            all(r['origins']['packages']==legacy['selected']['packages'] for r in context['exports'].values()),
            'original installed package origin roles differ')
    raw=file_bytes(context['launch']['sources']['quadratic_owner'],guards,keep=True)
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='audit_origins')
    original=copy.deepcopy(node);before=ast.parse("(expected, context['warm_record']['origins'])",mode='eval').body
    after=ast.parse("(expected, context['warm_record']['origins'], _nearest_supplement)",mode='eval').body
    dump=lambda n:ast.dump(n,include_attributes=False)
    class Substitute(ast.NodeTransformer):
        def __init__(self,old,new):self.old,self.new,self.count=old,new,0
        def visit(self,value):
            if dump(value)==dump(self.old):self.count+=1;return ast.copy_location(copy.deepcopy(self.new),value)
            return super().visit(value)
    forward=Substitute(before,after);node=forward.visit(node)
    inverse=Substitute(after,before);restored=inverse.visit(copy.deepcopy(node))
    require(forward.count==inverse.count==1 and dump(restored)==dump(original),'existing supplemental origin adapter predicates differ')
    # Confirm the exact preexisting adapter expressions in the actual nearest source.
    nearest_raw=file_bytes(context['launch']['sources']['nearest'],guards,keep=True)
    require(b'"(expected, context[\'warm_record\'][\'origins\'])"' in nearest_raw and
            b'"(expected, context[\'warm_record\'][\'origins\'], _nearest_supplement)"' in nearest_raw,
            'original native supplement adapter source differs')
    namespace={**vars(old),'_nearest_supplement':supplement}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),old.__file__,'exec'),namespace)
    fn=namespace['audit_origins'];code=fn.__code__;bindings=dict(namespace)
    references=(legacy['source_driver'],legacy['extract'],legacy['selected'],legacy['warm_record'],legacy['prior'])
    known={};known_modules={}
    for origins in (context['source_cpu']['origins'],context['warm']['origins'],supplement):
        for p,h in origins['files'].items():require(known.setdefault(p,h)==h,'conflicting original origin bytes')
        for n,p in origins['modules'].items():require(known_modules.setdefault(n,p)==p,'conflicting original origin modules')
    for record in (context['score'],context['cpu'],*context['exports'].values()):
        require(all(known.get(p)==h for p,h in record['origins']['files'].items()) and
                all(known_modules.get(n)==p for n,p in record['origins']['modules'].items()),
                'full score/CPU/export origins exceed original authority')
    def audit():
        sources.guard();require(fn.__code__ is code and namespace.keys()==bindings.keys() and
            all(namespace[k] is v for k,v in bindings.items()) and
            all(legacy[k] is v for k,v in zip(('source_driver','extract','selected','warm_record','prior'),references)),
            'private original origin source/context binding changed')
        # Unchanged authority primitive reopens its exact original vendor proof.
        current,_,_=nearest.bind_native_authority(owner)
        require(current==supplement,'original vendor supplement changed')
        fn(legacy,admission=modules['original'].FlatAdmission())
        return legacy['origins']
    return audit


def endpoint_payload(context, sources, endpoint):
    import torch
    key=label(endpoint);m=sources.modules;manifest=context['manifests'][key];facts=context['cpu']['payload_facts'][key]
    path=Path(endpoint['bundle']['path']).parent/'endpoint.pt'
    file_bytes({'path':str(path),'sha256':manifest['files']['endpoint.pt']},context['guards'])
    disk=torch.load(path,map_location='cpu',weights_only=True)
    try:
        check_endpoint_payload(disk,endpoint,manifest,facts,context['cpu']['source'],context['cpu']['numerical_flags'],m)
        head=m['cached'].head_from('control',tensors=disk['head']).requires_grad_(False).to('cuda').train()
        A=torch.nn.Parameter(disk['A'].detach().clone().to('cuda'))
        values={'head':head,'A':A,'C':disk['C'].detach().clone().to('cuda'),
                'mu_train':disk['mu_train'].detach().clone().to('cuda'),
                'means':{k:v.detach().clone().to('cuda') for k,v in disk['means'].items()},'arm':disk['arm']}
        return values
    finally:
        del disk;gc.collect();m['connected'].mapping_absent(path)


def check_endpoint_payload(disk,endpoint,manifest,facts,source,flags,modules):
    import torch
    original=modules['original'];modules['evaluator'].check_inference_members(modules['connected'],disk)
    fingerprint=original.fingerprint;ident=facts['identity']
    require(fingerprint(disk)==manifest['endpoint_state_sha256']==endpoint['inference_state_sha256']==facts['inference_state_sha256'] and
            fingerprint({k:v for k,v in disk.items() if k!='fixed_sha256'})==disk['fixed_sha256']==facts['fixed_sha256'] and
            facts['terminal_state_sha256']==endpoint['terminal_state_sha256'] and facts['bundle']==endpoint['bundle'] and
            set(facts['members'])==set(modules['evaluator'].MEMBERS) and
            all(fingerprint(disk[k])==h for k,h in facts['members'].items()),'complete typed endpoint/member/current binding differs')
    require(disk['arm']==endpoint['arm']==ident['arm'] and ident['seed']==endpoint['seed'] and
            disk['source']==source==ident['source'] and disk['numerical_flags']==flags and
            modules['identity'].scope_identity(disk['scope'])==ident['scope']==manifest['scope'] and
            disk['base_vision']==ident['base_vision'] and
            disk['encoder_identity']==ident['encoder_identity']==manifest['encoder_identity'] and
            manifest['files']['vision.pt']==disk['base_vision']['checkpoint']['sha256'] and
            disk['vision_sha256']==facts['vision_sha256']==manifest['vision_sha256'] and
            disk['base_vision']['sha256']==facts['base_vision_sha256']==manifest['base_vision_sha256'] and
            fingerprint(disk['processor']['config'])==facts['processor_config_sha256'],
            'original endpoint source/numerical/static provenance differs')
    for name,shape in (('A',(128,160)),('C',(128,1152)),('mu_train',(1152,))):
        modules['quadratic']._check_tensor(disk[name],shape,'cpu',frozen=True)
        require(torch.isfinite(disk[name]).all().item(),'finite readout payload required')
    require(torch.count_nonzero(disk['C']).item()>0,'nonzero complete terminal C required')
    modules['readout'].check_means(disk['means'],'cpu',modules['quadratic'])
    modules['identity'].check_mu_train_provenance(disk['mu_train_provenance'])


def scorer(sources, context):
    import numpy as np
    import torch
    path=context['launch']['sources']['scorer']['path'];raw=file_bytes(context['launch']['sources']['scorer'],context['guards'],keep=True)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='packed_quality']
    tree=ast.Module(body=nodes,type_ignores=[])
    require(len(nodes)==1 and hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==SCORER_AST,
            'unchanged original packed scorer AST differs')
    namespace={'np':np,'torch':torch,'pack_int8_unit_embeddings':sources.modules['packing'].pack_int8_unit_embeddings}
    exec(compile(tree,path,'exec',flags=__future__.annotations.compiler_flag,dont_inherit=True),namespace)
    return namespace['packed_quality']


def native_wire_quality(wire,labels,query,gallery):
    """Independent persisted-wire arithmetic: signed integer dot, two FP32 multiplies."""
    import torch
    codes,inverse=decode_wire(wire,len(labels));code=torch.tensor(codes,dtype=torch.int32)
    inv=torch.tensor(inverse,dtype=torch.float32);classes={s:i for i,s in enumerate(sorted(set(labels)))}
    ids=torch.tensor([classes[s] for s in labels]);counts=torch.bincount(ids[gallery])[ids[query]]
    require(int(counts.min())>0,'stored-wire positive inventory differs')
    width=int(counts.max());ranks=torch.arange(1,width+1);hits=[];aps=[]
    for start in range(0,len(query),128):
        rows=query[start:start+128];scores=(code[rows]@code[gallery].T).float()*inv[rows,None]*inv[None,gallery]
        order=torch.argsort(scores,dim=1,descending=True,stable=True)[:,:width]
        matches=ids[gallery][order]==ids[rows,None];r=counts[start:start+len(rows)]
        ap=((matches.cumsum(dim=1)/ranks[None,:])*matches*(ranks[None,:]<=r[:,None])).sum(dim=1)/r
        hits.extend(int(v) for v in matches[:,0].tolist());aps.extend(float(v) for v in ap.tolist())
    return {'per_query_r1':hits,'per_query_ap':aps,'recall_at_1':sum(hits)/len(hits),'map_at_r':sum(aps)/len(aps)}


def output_bytes(values):
    return {k:values[k] if k=='wire' else memoryview(values[k].contiguous().reshape(-1).view(
        sys.modules['torch'].uint8).numpy()).tobytes() for k in ('raw','unit','codes','inverse_norms','wire')}


def fresh_values(context,sources,endpoint):
    import numpy as np
    import torch
    key=label(endpoint);record=context['exports'][key];values={}
    for name in ('raw','unit'):
        filename=key+'.'+name+'.npy';fact={'path':str(Path(record['output'])/filename),'sha256':record['files'][filename]}
        npy_header(fact,(3449,128),context['guards']);array=np.load(fact['path'],allow_pickle=False)
        require(np.isfinite(array).all(),'finite fresh descriptor required');values[name]=torch.from_numpy(array)
    require((values['raw'].norm(dim=1)>0).all().item() and
            torch.allclose(values['unit'].norm(dim=1),torch.ones(3449),atol=1e-5,rtol=0),'nonzero/raw unit descriptor required')
    packed=sources.modules['packing'].pack_int8_unit_embeddings(values['unit'])
    values.update(codes=packed.codes,inverse_norms=packed.inverse_norms,wire=packed.to_bytes())
    filename=key+'.packed.bin';wire=file_bytes({'path':str(Path(record['output'])/filename),
        'sha256':record['files'][filename]},context['guards'],keep=True)
    exact_wire(values['wire'],wire)
    require({k:sources.modules['source_driver'].tensor_fact(values[k]) for k in ('raw','unit','codes','inverse_norms')}==
            record['panel_facts'],'original raw/unit/code/inverse payload facts differ')
    return values


def stale_values(context,sources,endpoint,features,budget,*,mutant=None):
    import torch
    state=None;raw=None;values=None
    try:
        state=endpoint_payload(context,sources,endpoint);panel=context['partition']['panels']['selection']
        raw=torch.empty((3449,128),dtype=torch.float32)
        for role in ('query','gallery'):
            for start in range(0,len(panel[role]),32):
                budget.check();sources.guard();indices=panel[role][start:start+32];batch=features[indices].to('cuda')
                C=torch.zeros_like(state['C']) if mutant=='omitted_C' else state['C']
                mu=torch.zeros_like(state['mu_train']) if mutant=='wrong_mu' else state['mu_train']
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    batch_raw=sources.modules['identity'].fullfeature_raw_features(batch,state['head'],state['A'],
                        state['means'],C,mu,state['arm'],sources.modules['quadratic'],sources.modules['readout'])
                    # Normalization must happen on CUDA in the same original role batch.
                    outputs=sources.modules['quadratic_owner'].packed_outputs({'packing':sources.modules['packing']},batch_raw)
                raw[indices]=outputs['raw']
                if values is None:
                    values={k:torch.empty((3449,128) if k in ('unit','codes') else (3449,),dtype=outputs[k].dtype)
                            for k in ('unit','codes','inverse_norms')}
                for k in values:values[k][indices]=outputs[k]
                require(torch.cuda.max_memory_allocated()<LIMITS['cuda_allocated_bytes_exclusive'],'CUDA allocation cap reached')
                del batch,C,mu,batch_raw,outputs
        values['raw']=raw;packed=sources.modules['packing'].pack_int8_unit_embeddings(values['unit'])
        require(torch.equal(packed.codes,values['codes']) and torch.equal(packed.inverse_norms,values['inverse_norms']),
                'same-role packed outputs differ from full-panel packing')
        values['wire']=packed.to_bytes();return values
    finally:
        state=None;gc.collect()


def terminal_admission(context,sources):
    """Original reader, original UNITs; no inherited input-guard loop."""
    m=sources.modules;reader=m['original'].FlatAdmission();reader.init=m['initializer']
    t={'legacy':{'original':m['original'],'admission':reader},'guards':context['guards']}
    terminal=m['fitter'].original_terminal_reader(t)
    units=[(context['score'],context['launch']['score_terminal'],700),
           (context['cpu'],context['score']['launch']['selected_cpu'],700)]
    units += [(context['exports'][key],unit,1500) for key,unit in context['score']['launch']['exports'].items()]
    for record,unit,seconds in units:
        m['evaluator'].check_unit(unit)
        terminal(reader,record,unit,seconds,context['guards'])
    return {unit['invocation_id'] for _,unit,_ in units}


def run(args):
    budget=Budget();context=None;sources=None;audit=None;error=None;receipt=None;before_rng=None;before_flags=None
    fresh={};features=None;first=second=mutated=cell_values=None;final_check=None
    require(not {'torch','numpy','PIL','transformers','torchvision','sfora'}.intersection(sys.modules),
            'native imports must follow explicit admission')
    try:
        context=prepare(args);budget.check();sources=Sources(context);sources.admit()
        require(Path(context['launch']['sources']['quadratic']['path']).name=='quadratic_readout.py' and
                Path(context['launch']['sources']['quadratic_owner']['path']).name=='train_siglip2_quadratic_readout.py',
                'separate genuine quadratic primitive/owner FILEs required')
        invocations=terminal_admission(context,sources);m=sources.modules;source=m['source_driver']
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and
                re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')) and
                os.environ['INVOCATION_ID'] not in invocations,'original single-CUDA/fresh enclosing unit required')
        prior=context['score']['invocation'];python=Path(sys.executable).resolve()
        require(str(python)==prior['python'] and m['extract'].sha(python)==prior['python_sha256'] and
                sys.version==prior['python_version'],'original admitted interpreter required')
        before=source.cgroup_memory();unit=Path(before['path']).name.removesuffix('.service')
        initializer=m['initializer']
        final_check=lambda:final_state(budget,source,initializer,before,before_rng,before_flags,receipt)
        initializer.admit_cgroup(before,unit)
        audit=origin_audit(context,sources)
        import torch
        require(not torch.cuda.is_initialized(),'CUDA initialization must follow admission')
        flags=context['cpu']['numerical_flags'];torch.set_num_threads(flags['threads'])
        if torch.get_num_interop_threads()!=flags['interop_threads']:torch.set_num_interop_threads(flags['interop_threads'])
        before_flags=copy.deepcopy(flags)
        require(source.numerical_flags()==flags,'original numerical flags differ')
        before_rng=(torch.random.get_rng_state().clone(),[v.clone() for v in torch.cuda.get_rng_state_all()])
        require(torch.cuda.device_count()==1 and torch.cuda.max_memory_allocated()<LIMITS['cuda_allocated_bytes_exclusive'],
                'one original CUDA device/early allocation cap required')
        packing=sources.load('packing')
        sources.checks.append(m['evaluator'].source_live_guard(packing,
            context['launch']['sources']['packing']['sha256'],context['guards']))
        fixed=scorer(sources,context);audit();args.output.mkdir()
        panel=context['partition']['panels']['selection'];labels=context['labels'];query,gallery=panel['query'],panel['gallery']
        # Finish ALL fresh artifact checks before any new readout or released metric.
        for endpoint in context['score']['launch']['endpoints']:
            budget.check();key=label(endpoint);fresh[key]=fresh_values(context,sources,endpoint)
            quality=fixed(fresh[key]['unit'].numpy(),labels,query,gallery,device=torch.device('cpu'))
            replay(context['score']['quality'][str(endpoint['seed'])][endpoint['arm']],quality)
            replay(quality,native_wire_quality(fresh[key]['wire'],labels,query,gallery))
        features=m['baseline'].cache_rows({'partition':context['partition'],'guards':context['guards']},panel['original_rows'])
        require(features.shape==(3449,1152) and features.dtype==torch.float32,'direct original selection cache tap differs')
        m['connected'].mapping_absent(context['launch']['original_cache']['path'])
        instruments={};cells={}
        for endpoint in context['score']['launch']['endpoints']:
            if endpoint['arm']!='control':continue
            key=label(endpoint);first=stale_values(context,sources,endpoint,features,budget)
            differences=control_byte_differences(output_bytes(first),output_bytes(fresh[key]),panel,context['fit']);instruments[key]=differences
            write_json(args.output/(key+'-tap.json'),differences)
            require_control_tap(differences)
            replay(context['score']['quality'][str(endpoint['seed'])]['control'],
                   fixed(first['unit'].numpy(),labels,query,gallery,device=torch.device('cpu')))
            second=stale_values(context,sources,endpoint,features,budget)
            require_control_tap(byte_differences(output_bytes(first),output_bytes(second)))
            for mutant in ('omitted_C','wrong_mu'):
                mutated=stale_values(context,sources,endpoint,features,budget,mutant=mutant)
                require(output_bytes(first)['raw']!=output_bytes(mutated)['raw'],mutant+' readout negative failed')
                mutated=None
            first=second=None;gc.collect()
        for endpoint in context['score']['launch']['endpoints']:
            if endpoint['arm']!='candidate':continue
            key=label(endpoint);first=stale_values(context,sources,endpoint,features,budget)
            second=stale_values(context,sources,endpoint,features,budget)
            require_control_tap(byte_differences(output_bytes(first),output_bytes(second)));second=None
            for mutant in ('omitted_C','wrong_mu'):
                mutated=stale_values(context,sources,endpoint,features,budget,mutant=mutant)
                require(output_bytes(first)['raw']!=output_bytes(mutated)['raw'],mutant+' readout negative failed');mutated=None
            values={}
            for cell in ('SS','SF','FS','FF'):
                budget.check();cell_values=compose(first['unit'],fresh[key]['unit'],query,gallery,cell)
                values[cell]=fixed(cell_values.numpy(),labels,query,gallery,device=torch.device('cpu'))
                packed=packing.pack_int8_unit_embeddings(cell_values)
                replay(values[cell],native_wire_quality(packed.to_bytes(),labels,query,gallery))
                del packed;cell_values=None
            replay(context['score']['quality'][str(endpoint['seed'])]['candidate'],values['FF'])
            cells[str(endpoint['seed'])]={'cells':values,'effects':effects(values)}
            first=None;gc.collect()
        features=None;fresh.clear();gc.collect();torch.cuda.synchronize()
        require(torch.cuda.memory_allocated()==0,'readout CUDA tensor lifetime survived cleanup')
        require(source.numerical_flags()==before_flags and torch.equal(torch.random.get_rng_state(),before_rng[0]) and
                len(torch.cuda.get_rng_state_all())==len(before_rng[1]) and
                all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),before_rng[1],strict=True)),
                'original RNG/numerical flags changed')
        budget.check();after=source.cgroup_memory();m['initializer'].admit_cgroup(after,unit)
        require(after['path']==before['path'],'enclosing diagnostic unit changed')
        receipt={'schema':'connected-gallery-freshness-diagnostic-v1','candidate_status':'KILL unchanged',
            'qualification_eligible':False,'state_reuse_eligible':False,'official_read':False,'validation_read':False,
            'descriptive_only':True,'causal_null_claim':False,'controls':instruments,'seeds':cells,
            'authority':{'path':str(args.authority),'sha256':args.authority_sha256},'launch':context['launch'],
            'cgroup_before':before,'cgroup_after':after,'invocation':{'invocation_id':os.environ['INVOCATION_ID'],'optimize':0},
            'numerical_flags':before_flags,'rng_flags_preserved':True,'resource_policy':LIMITS,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'terminal_exit_and_both_locks_require_parent_receipt':True,
            'remaining_confound':'FIT export batch composition/preprocessing; control parity does not prove all candidate arithmetic equivalence'}
    except BaseException as failure:error=failure
    finally:
        features=first=second=mutated=cell_values=None;fresh.clear();gc.collect()
        callbacks=[]
        if audit is not None:callbacks.append(audit)
        if context is not None and sources is not None and 'connected' in sources.modules:
            callbacks.append(lambda:[sources.modules['connected'].mapping_absent(Path(e['bundle']['path']).parent/'endpoint.pt')
                                     for e in context['score']['launch']['endpoints']])
        if context is not None:callbacks.append(lambda:rehash_files(context['guards']))
        callbacks.append(budget.check)
        if sources is not None:callbacks.append(sources.close)
        if final_check is not None:callbacks.append(final_check)
        cleanup_error(error,callbacks)
    receipt.update(exit_rehash_pass=True,integrity_pass=True,cleanup_pass=True,wall_seconds=budget.clock()-budget.started)
    write_json(args.output/'receipt.json',receipt)
    budget.check()
    return receipt


def final_state(budget,source,initializer,before,before_rng,flags,receipt):
    """After audit, full rehash and source disposal; this peak covers mandatory exit."""
    budget.check()
    torch=sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()
        require(torch.cuda.max_memory_allocated()<LIMITS['cuda_allocated_bytes_exclusive'],'final whole-lifetime CUDA cap differs')
    if flags is not None:require(source.numerical_flags()==flags,'final numerical flags changed')
    if before_rng is not None:
        require(torch.equal(torch.random.get_rng_state(),before_rng[0]) and
                len(torch.cuda.get_rng_state_all())==len(before_rng[1]) and
                all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),before_rng[1],strict=True)),
                'final RNG changed')
    after=source.cgroup_memory();unit=Path(before['path']).name.removesuffix('.service')
    initializer.admit_cgroup(after,unit)
    require(after['path']==before['path'],'final enclosing cgroup changed')
    if receipt is not None:
        require(torch.cuda.memory_allocated()==0,'final CUDA tensor cleanup differs')
        receipt.update(cgroup_after=after,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    budget.check()


def write_json(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-sha256',required=True)
    p.add_argument('--authority',required=True,type=Path)
    p.add_argument('--authority-sha256',required=True)
    p.add_argument('--output',required=True,type=Path)
    return p


if __name__=='__main__':
    run(parser().parse_args())
