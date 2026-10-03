#!/usr/bin/env python3
"""Bounded stdlib/source falsifiers; no Torch, images, corpus or native work."""
import ast
import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().with_name('evaluate_siglip2_compact_ranking.py')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value

e=module('_compact_evaluation_tests',PATH)
math_helper=module('_compact_math_tests',PATH.with_name('evaluate_siglip2_genuine_views.py'))


def quality(r1,ap):
    return {'recall_at_1':sum(r1)/len(r1),'map_at_r':sum(ap)/len(ap),'per_query_r1':r1,'per_query_ap':ap}


def descriptor(path,digest='a'*64):
    return {'path':str(path),'sha256':digest}


def unit(number):
    return {'receipt':descriptor('/tmp/u'+str(number)+'/receipt.json'),'log':descriptor('/tmp/u'+str(number)+'.log'),
        'unit':'u'+str(number),'invocation_id':format(number,'032x'),'service_seconds':10,
        'native_peak_rss_kib':100,'both_locks_held':True}


def launch(stage='first',phase='cpu',panel='selection'):
    endpoints=[]
    for i,(seed,arm) in enumerate(e.endpoint_order(stage),1):
        endpoints.append({'seed':seed,'arm':arm,'launch':descriptor('/tmp/l'+str(i)),'terminal':unit(i),
            'checkpoint':descriptor('/tmp/u'+str(i)+'/resume.pt',format(i,'064x')),
            'terminal_state_sha256':'b'*64,'bundle':descriptor('/tmp/u'+str(i)+'/bundle/bundle.json'),
            'inference_state_sha256':'c'*64})
    args=SimpleNamespace(execution_sha256='d'*64,phase=phase,arm=None,seed=None)
    if phase == 'export':args.seed,args.arm=e.endpoint_order(stage)[-1]
    result={'schema':e.AUTHORITY_SCHEMA,'execution_sha256':args.execution_sha256,
        'training':copy.deepcopy(e.TRAINING),
        'nearest_evaluator':e.NEAREST_EVALUATOR,'genuine_evaluator':{'root':'/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4',
            'execution_sha256':e.GENUINE_EXECUTION_SHA,'code':e.GENUINE_PINS},'reference':e.REFERENCE,
        'phase':phase,'arm':args.arm,'seed':args.seed,'stage':stage,'panel':panel,'endpoints':endpoints,
        'selected_cpu':None if phase=='cpu' else unit(10),'exports':{},
        'first_selection':None if stage=='first' else unit(11),
        'selection_go':None if panel=='selection' else unit(12),'both_locks_held':True,
        'selection_previously_exposed':True,'cost_policy':e.COST_POLICY,
        'resource_policies':{p:e.policy(p) for p in ('cpu','export','score')}}
    if phase=='score':result['exports']={e.label(v):unit(i+20) for i,v in enumerate(endpoints)}
    return result,args


def controls(stage):
    return {(s,a):{'service_seconds':100,'total_training_core_seconds':60} for s,a in e.endpoint_order(stage)}


def panels(stage):
    n=e.PANELS['selection'][1]
    left=[0 if i<12 else 1 for i in range(n)]; right=left.copy();right[:6]=[1]*6
    source=quality(left,[.8]*n);concat=quality(left,[.817]*n)
    q={str(s):{'control':quality(left,[.818]*n),'candidate':quality(right,[.822]*n)} for s in e.seeds(stage)}
    return q,source,concat


def intervals(q):
    _,avg=math_helper.averaged_deltas(q,'full','selection')
    return {m:{'mean_delta':sum(avg[m])/len(avg[m]),'product_lower95':.001,'product_upper95':.1,
               'query_lower95':-.001,'query_upper95':.1} for m in e.METRICS}


class SourceAdmissionFixture:
    """Run the real context-composition block and pinned adapter, without native work.

    The old authority's actual dictionary expression supplies the pre-native
    context. Only external receipts/logs/wires are disconnected stand-ins.
    """
    def __init__(self, root):
        def pinned(name, filename, digest):
            if name not in sys.modules:
                e.load_authenticated(name, PATH.with_name(filename), digest, {})
            value=sys.modules[name]
            e.bound_file({},value.__file__,digest)
            return value
        reference=pinned('_compact_context_reference_tests','evaluate_siglip2_prototype_residual.py',
            e.REFERENCE['code']['evaluate_siglip2_prototype_residual.py'])
        self.baseline=pinned('_compact_context_baseline_tests','evaluate_siglip2_quadratic_readout.py',
            reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'])
        b=self.baseline
        self.partition={'original_cache':descriptor(root/'cache.npy',reference.FIT_SHA),
            'panels':{'selection':{'original_rows':[8,3,5]},'validation':{'original_rows':[9,4]}}}
        self.path=root/'partition.json';self.path.write_text(json.dumps(self.partition))
        part=descriptor(self.path,hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.spec={'partition':part,'source_selection':{'inventory':descriptor(root/'inventory.json',b.SOURCE_INVENTORY_SHA)}}
        fit={'class_names':['one','two'],'targets':[0,1]}
        selected={'partition':copy.deepcopy(self.partition),'source':{'standin':'authenticated source'},
            'genuine':{'prior':{'fit':fit}},'source_driver':None,'original':None,'extract':None,
            'source_cpu':{'invocation':{'invocation_id':'0'*32}}}
        self.calls=[]
        def wire_file(guards,path,digest):
            fact=next(v for v in b.SOURCE_INVENTORY['files'].values() if v['path']==str(path))
            e.require(digest==fact['sha256'],'wire descriptor differs')
            guards[str(path)]=digest;self.calls.append(('wire',str(path)))
            return SimpleNamespace(stat=lambda:SimpleNamespace(st_size=fact['bytes']))
        admission=SimpleNamespace(bound_file=wire_file)
        # Evaluate the authenticated authority's REAL context expression: no
        # synthetic top-level partition, initial state or ownership checks.
        original=PATH.with_name('train_siglip2_quadratic_readout.py')
        e.bound_file({},original,reference.ORIGINAL_PINS[original.name])
        node=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        expression=next(n.value for n in node.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='context' for t in n.targets))
        self.legacy=eval(compile(ast.Expression(expression),str(original),'eval'),
            {'args':None,'root':root,'code':{},'launch':{'partition':part},'guards':{},
             'genuine':None,'selected':selected,'admission':admission,'record':{}})
        self.training={'legacy':self.legacy,'fit_context':{'launch':{'partition':part},'guards':{}},'guards':{}}
        self.archived={'cgroup_before':{},'cgroup_after':{},'input_guards':{},'files':{},'output':str(root)}
        inventory=b.SOURCE_INVENTORY
        self.record={'schema':'siglip2-genuine-view-evaluation-v1','phase':'score','pass':True,
            'engineering_admission_pass':True,'source':selected['source'],
            'spec':{'panel':'selection','stage':'first','endpoints':[inventory['baseline_endpoint']],
                'partition':inventory['partition'],'evaluation_reference':{'root':b.REFERENCE_ROOT,
                    'execution_sha256':b.REFERENCE_EXECUTION_SHA}},
            'resource_policy':b.policy('score'),'query_images':1734,'gallery_images':1715,'panel_products':498,
            'peak_cuda_allocated_bytes':0,'files':{n:v['sha256'] for n,v in inventory['files'].items()},
            'output':str(Path(next(iter(inventory['files'].values()))['path']).parent),
            'quality':{'179061':{'control':{'recall_at_1':inventory['control_quality']['recall_at_1'],
                'map_at_r':inventory['control_quality']['map_at_r'],'per_query_r1':[1]*1670+[0]*64,
                'per_query_ap':[inventory['control_quality']['map_at_r']]*1734}}},
            'input_guards':{},'cgroup_before':{},'cgroup_after':{}}
        for k in ('strict_independent_head_reload_exact','train_raw_unit_cpu_packed_exact','rng_flags_preserved',
            'exit_rehash_pass','full_panel_raw_unit_packed_replay_exact','per_query_replay_exact'):self.record[k]=True
        for k in ('official_read','public_latency_measured','global_production_goal_met','cuda_initialized'):self.record[k]=False
        def reader(admitted,record,terminal,seconds,guards):
            e.require(admitted is admission,'original admission object differs')
            self.calls.append(('terminal',seconds));return {}
        # Copy the adapter's namespace; ORIGINAL module globals stay untouched.
        adapter_factory=FunctionType(reference.source_selection_adapter.__code__,
            {**vars(reference),'original_terminal_reader':lambda _:reader})
        def source_adapter(baseline,context):
            adapted=adapter_factory(baseline,context)  # Full-byte/live-code/AST authentication stays real.
            def receipt(value,guards):
                self.calls.append(('source_json',value['path']))
                if value==self.spec['source_selection']['inventory']:return inventory
                e.require(value==inventory['receipt'],'source receipt descriptor differs')
                return self.record
            adapted.__globals__['read_json']=receipt
            return adapted
        self.reference=SimpleNamespace(FIT_SHA=reference.FIT_SHA,original_terminal_reader=lambda _:reader,
            source_selection_adapter=source_adapter)
        self.originals=[(m,dict(vars(m))) for m in (reference,b)]

    def compose(self):
        node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='authority')
        def assigns(n,name):
            return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
        start=next(i for i,n in enumerate(node.body) if assigns(n,'legacy'))
        stop=next(i for i,n in enumerate(node.body) if assigns(n,'source_record'))+1
        namespace={**vars(e),'t':self.training,'reference':self.reference,'spec':self.spec,'args':None,
            'guards':{},'archived':self.archived,'baseline':self.baseline,
            'helper':SimpleNamespace(zero_events=lambda _:None),'native':SimpleNamespace(CONCAT_TERMINAL=unit(99))}
        exec(compile(ast.Module(body=node.body[start:stop],type_ignores=[]),str(PATH),'exec'),namespace)
        return namespace['s']


class EvaluationTests(unittest.TestCase):
    def test_exact_source_stage_seed_and_roles(self):
        for stage in ('first','full'):
            for phase in ('cpu','export','score'):
                value,args=launch(stage,phase);e.check_launch(value,args)
                for k,v in (('selection_previously_exposed',False),('both_locks_held',False),
                    ('stage','other'),('reference',{}),('endpoints',value['endpoints'][::-1])):
                    with self.subTest(stage=stage,phase=phase,k=k),self.assertRaises((ValueError,KeyError)):
                        e.check_launch({**value,k:v},args)
        value,args=launch()
        for k,v in (('first_selection',unit(11)),('selected_cpu',unit(10)),('panel','validation'),('seed',179069),
                    ('arm','control'),('exports',{'control-179061':unit(20)})):
            with self.subTest(k=k),self.assertRaises(ValueError):e.check_launch({**value,k:v},args)
        value,args=launch('full','score','validation');e.check_launch(value,args)
        for k in ('selection_go','first_selection','selected_cpu'):
            with self.assertRaises(ValueError):e.check_launch({**value,k:None},args)
        value,args=launch('full');value['training']['code']['foreign.py']='0'*64
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch();value['endpoints'][0]['seed']=True
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch('full');value['first_selection']=value['endpoints'][0]['terminal']
        with self.assertRaises(ValueError):e.check_launch(value,args)

    def test_pre_native_partition_and_pinned_source_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            f=SourceAdmissionFixture(Path(directory))
            self.assertTrue({'partition','partition_check','initial'}.isdisjoint(f.legacy))
            s=f.compose()
            self.assertEqual(s['partition']['panels']['selection']['original_rows'],[8,3,5])
            self.assertIs(s['fit'],f.legacy['prior']['fit']);self.assertIs(s['selected'],f.legacy)
            self.assertIs(s['source_record'],f.record);self.assertIs(s['origin_records'][-1],f.record)
            self.assertEqual(s['terminals'][-1],f.baseline.SOURCE_SCORE_TERMINAL)
            self.assertEqual(s['source_guards'][str(f.spec['source_selection']['inventory']['path'])],f.baseline.SOURCE_INVENTORY_SHA)
            self.assertEqual(s['guards'][str(f.path)],f.spec['partition']['sha256'])
            self.assertEqual([n for kind,n in f.calls if kind=='terminal'],[500,300])
            self.assertEqual(sum(kind=='wire' for kind,_ in f.calls),3)
            for module,values in f.originals:
                self.assertEqual(vars(module).keys(),values.keys())
                self.assertTrue(all(vars(module)[k] is v for k,v in values.items()))
            self.assertFalse(any(n.split('.')[0] in e.NATIVE for n in sys.modules))

    def test_partition_order_descriptor_cache_and_hash_rejected_before_source(self):
        for mutation in ('order','descriptor','cache','hash'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                f=SourceAdmissionFixture(Path(directory))
                if mutation=='order':f.legacy['selected']['partition']['panels']['selection']['original_rows'].reverse()
                elif mutation=='descriptor':f.training['fit_context']['launch']['partition']=descriptor(f.path,'0'*64)
                elif mutation=='cache':
                    f.partition['original_cache']['sha256']='0'*64
                    f.path.write_text(json.dumps(f.partition))
                    f.spec['partition']['sha256']=hashlib.sha256(f.path.read_bytes()).hexdigest()
                    f.legacy['selected']['partition']=copy.deepcopy(f.partition)
                else:f.path.write_text(json.dumps(f.partition)+' ')
                with self.assertRaisesRegex(ValueError,'ordered panel partition|SHA256'):f.compose()
                self.assertFalse(any(kind=='source_json' for kind,_ in f.calls))

    def test_first_screen_never_bootstraps(self):
        q,source,concat=panels('first');costs=e.paired_cost(controls('first'),'first')
        result=e.decide(math_helper,q,source,concat,'first','selection',{},costs)
        self.assertEqual(result['decision'],'CONTINUE');self.assertFalse(result['selection_go_admits_validation_only'])
        for replacement in (q['179061']['control'],quality(q['179061']['candidate']['per_query_r1'],[.7]*1734)):
            bad=copy.deepcopy(q);bad['179061']['candidate']=replacement
            self.assertEqual(e.decide(math_helper,bad,source,concat,'first','selection',{},costs)['decision'],'KILL')
            with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'first','selection',{'ci':{}},costs)
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'first','selection',{'ci':{}},costs)
        records=controls('first');records[179061,'candidate']['service_seconds']=151
        self.assertEqual(e.decide(math_helper,q,source,concat,'first','selection',{},e.paired_cost(records,'first'))['decision'],'KILL')

    def test_full_equal_seed_floor_sign_and_intervals(self):
        q,source,concat=panels('full');costs=e.paired_cost(controls('full'),'full');bounds=intervals(q)
        result=e.decide(math_helper,q,source,concat,'full','selection',bounds,costs)
        self.assertEqual(result['decision'],'GO');self.assertTrue(result['selection_go_admits_validation_only'])
        self.assertFalse(result['global_production_goal_met']);self.assertFalse(result['product_go'])
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['product_lower95']=0
        self.assertEqual(e.decide(math_helper,q,source,concat,'full','selection',bad,costs)['decision'],'KILL')
        bad=copy.deepcopy(bounds);bad[e.METRICS[0]]['mean_delta']=.7
        with self.assertRaises(ValueError):e.decide(math_helper,q,source,concat,'full','selection',bad,costs)
        bad=copy.deepcopy(q);bad['179069']['candidate']=copy.deepcopy(bad['179069']['control'])
        self.assertEqual(e.decide(math_helper,bad,source,concat,'full','selection',{},costs)['decision'],'KILL')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)
        high=quality([1]*1734,[.9]*1734)
        self.assertEqual(e.decide(math_helper,q,high,concat,'full','selection',{},costs)['decision'],'KILL')
        self.assertEqual(e.decide(math_helper,q,source,high,'full','selection',{},costs)['decision'],'KILL')
        bad=copy.deepcopy(q);bad['179069']['candidate']['per_query_ap'][0]=float('nan')
        with self.assertRaises(ValueError):e.decide(math_helper,bad,source,concat,'full','selection',bounds,costs)

    def test_each_seed_core_and_whole_cost_boundary(self):
        records=controls('full');records[179069,'candidate'].update(service_seconds=150,total_training_core_seconds=90)
        self.assertTrue(all(v['pass'] for v in e.paired_cost(records,'full').values()))
        for key,value in (('total_training_core_seconds',90.01),('service_seconds',150.01)):
            bad=copy.deepcopy(records);bad[179069,'candidate'][key]=value
            self.assertFalse(e.paired_cost(bad,'full')['179069']['pass'])
        for bad_value in (0,True,float('nan'),float('inf'),101):
            bad=copy.deepcopy(records);bad[179061,'control']['total_training_core_seconds']=bad_value
            with self.assertRaises(ValueError):e.paired_cost(bad,'full')
        with self.assertRaises(ValueError):e.paired_cost(controls('first'),'full')

    def test_no_quality_before_readiness(self):
        calls=[]
        for key in e.READINESS:
            ready=dict.fromkeys(e.READINESS,True);ready[key]=False
            with self.assertRaises(ValueError):e.quality_after_readiness(ready,lambda:calls.append('quality'))
        self.assertEqual(calls,[])
        e.quality_after_readiness(dict.fromkeys(e.READINESS,True),lambda:calls.append('quality'))
        self.assertEqual(calls,['quality'])

    def test_diagnostic_and_native_roles(self):
        value=e.diagnostic_result([.1,-.1],[.09,-.2])
        self.assertFalse(value['mechanism_demonstrated']);self.assertFalse(value['utility_veto']);self.assertTrue(value['diagnostic_only'])
        self.assertEqual(e.batch_sizes(1734),[32]*54+[6]);self.assertEqual(e.batch_sizes(1715),[32]*53+[19])
        self.assertEqual(e.batch_sizes(1749)[-1],21);self.assertEqual(e.batch_sizes(1730)[-1],2)
        for invalid in (0,True,-1):
            with self.assertRaises(ValueError):e.batch_sizes(invalid)
        with self.assertRaises(ValueError):e.diagnostic_result([float('nan')],[0])

    def test_strict_files_closure_and_restored_mtime(self):
        for raw in ('{"a":1,"a":2}','{"a":NaN}'):
            with self.assertRaises(ValueError):e.strict_json(raw)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'worker.py';path.write_bytes(b'first')
            digest=hashlib.sha256(path.read_bytes()).hexdigest();code={'worker.py':digest}
            manifest=root/'execution.json';manifest.write_text(json.dumps(code));manifest_sha=hashlib.sha256(manifest.read_bytes()).hexdigest()
            self.assertEqual(e.closure(root,manifest_sha,code,{}),code)
            prior=path.stat();path.write_bytes(b'other');os.utime(path,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            with self.assertRaises(ValueError):e.bound_file({},path,digest)
            path.write_bytes(b'first');link=root/'link';link.symlink_to(path)
            with self.assertRaises(ValueError):e.bound_file({},link,digest)
            fifo=root/'fifo';os.mkfifo(fifo)
            with self.assertRaises(ValueError):e.bound_file({},fifo,digest)
            with self.assertRaises(ValueError):e.closure(root,manifest_sha,{'worker.py','extra'}, {})

    def test_bundle_boundary_denies_train_warm_optimizer_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir();owned=bundle/'vision.pt';owned.write_bytes(b'owned')
            outside=root/'resume.pt';outside.write_bytes(b'forbidden')
            trainer=SimpleNamespace(admit_bundle=lambda *_:({'environment':{'packages':{},'files':{}}},{}))
            endpoint={'bundle':descriptor(bundle/'bundle.json')}
            with e.bundle_reads_only({'trainer':trainer},endpoint):
                self.assertEqual(owned.read_bytes(),b'owned')
                for path in (outside,root/'warm.pt',root/'optimizer.pt',root/'teachers.npy'):
                    with self.assertRaisesRegex(ValueError,'external dependency'):path.read_bytes()
                with self.assertRaisesRegex(ValueError,'attempted write'):owned.write_bytes(b'changed')
            self.assertEqual(outside.read_bytes(),b'forbidden');self.assertEqual(owned.read_bytes(),b'owned')

    def test_partial_metadata_receipt_and_foreign_bindings_rejected(self):
        value,args=launch();flags={'threads':1};source={'actual':'source'}
        args.authority=Path('/tmp/launch');args.authority_sha256='a'*64;args.output=Path('/tmp/output')
        context={'args':args,'launch':value,'code':dict.fromkeys(e.FILES,'b'*64),
            'training_context':{'source':source,'legacy':{'selected':{'source_cpu':{'numerical_flags':flags}}}},
            'costs':e.paired_cost(controls('first'),'first'),'reference':SimpleNamespace(check_synthetic_bootstrap=lambda _:None),
            'guards':{}}
        record={'schema':e.SCHEMA,'phase':'cpu','arm':None,'seed':None,'stage':'first','panel':'selection',
            'execution_sha256':args.execution_sha256,'source_code':context['code'],'source':source,
            'cost_policy':e.COST_POLICY,'launch':value,'authority':descriptor(args.authority),
            'authority_sha256':args.authority_sha256,'binding':e.binding(context),'numerical_flags':flags,
            'output':str(args.output),'cost':context['costs'],'resource_policy':e.policy('cpu'),
            'wall_seconds':1,'process_peak_rss_kib':100,'peak_cuda_allocated_bytes':0,'cuda_initialized':False,
            'invocation':{'argv':e.cli(args),'optimize':0,'cuda_visible_devices':'','cublas_workspace_config':':4096:8'},
            'synthetic_bootstrap':{},'files':{},'quality_read':False,'metadata_only':True,
            'updated_payloads_authenticated':True,'malformed_inference_rejected':True,
            'payload_facts':{e.label(v):{} for v in value['endpoints']},
            'calibration':{'same_role_forward_exact':True,'raw_unit_packed_exact':True}}
        for key in ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
            'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority'):record[key]=True
        for key in ('official_read','global_production_goal_met','public_latency_measured','product_go'):record[key]=False
        with patch.object(e,'read_json',return_value=value):
            e.check_receipt(context,record,'cpu')
            for key,bad in (('metadata_only',False),('updated_payloads_authenticated',False),('payload_facts',{}),
                ('quality_read',True),('source_code',{}),('stage','full'),('panel','validation'),('numerical_flags',{})):
                with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                    e.check_receipt(context,{**record,key:bad},'cpu')

    def test_whole_unit_resource_caps(self):
        for phase in ('cpu','export','score'):
            record={'resource_policy':e.policy(phase),'wall_seconds':1,'process_peak_rss_kib':100,
                'peak_cuda_allocated_bytes':1 if phase=='export' else 0,'cuda_initialized':phase=='export'}
            e.check_resource_facts(record,phase)
            for key,value in (('wall_seconds',e.policy(phase)['seconds']),('process_peak_rss_kib',8*1024**2+1),
                ('peak_cuda_allocated_bytes',10_000_000_000),('cuda_initialized',phase!='export')):
                with self.assertRaises(ValueError):e.check_resource_facts({**record,key:value},phase)

    def test_owned_native_api_exit_and_exact_membership(self):
        from test_siglip2_nearest_ranking import NativeAdmissionFixture,driver
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);f=NativeAdmissionFixture(root)
            class ExitContext(dict):
                def __getitem__(self,key):
                    accessed.add(key);return super().__getitem__(key)
            accessed=set()
            fit=f.context['fit_context']=ExitContext(f.context['fit_context'])
            phases=fit['phase_seconds'];fit.pop('unit_started')  # Authority does not supply the run clock.
            api=f.admit()
            def code_descriptor(name,names):
                target=root/name;target.mkdir()
                for n in names:(target/n).write_bytes(PATH.with_name(n).read_bytes())
                code={n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in names}
                manifest=target/'execution.json';manifest.write_text(json.dumps(code))
                return {'root':str(target),'code':code,'execution_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
            own=code_descriptor('own',e.FILES);train=code_descriptor('train',e.TRAIN_FILES) if all(PATH.with_name(n).exists() for n in e.TRAIN_FILES) else own
            refs=[code_descriptor('ref'+str(i),pins) for i,pins in enumerate((e.NEAREST_EVALUATOR['code'],e.GENUINE_PINS,e.REFERENCE['code']))]
            # No model exists; use the REAL native API, with only disconnected trainer/source snapshots stubbed.
            trainer=SimpleNamespace(require_no_training=lambda _:None,helper_guard=lambda _:None,
                admit_bundle=lambda *_:({},{}))
            f.context['nearest']=driver
            context={'trainer':trainer,'training_context':f.context,'args':SimpleNamespace(phase='export',execution_sha256=own['execution_sha256']),
                'root':Path(own['root']),'code':own['code'],'guards':{},'launch':{'training':train,
                    'nearest_evaluator':refs[0],'genuine_evaluator':refs[1],'reference':refs[2],'endpoints':[]}}
            def action():
                phases.clear();return e.exit_rehash(context)
            with patch.object(e,'guard_helpers'),patch.object(e,'NEAREST_EVALUATOR',refs[0]), \
                patch.object(e,'GENUINE_PINS',refs[1]['code']),patch.object(e,'TRAIN_FILES',train['code']),redirect_stdout(io.StringIO()):
                # Execute the actual run prefix through native_start, stopping
                # before the Torch import. The fixture used to mask this bug
                # by supplying its own unrelated fitter start timestamp.
                run_node=next(n for n in ast.parse(PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
                stop=next(i for i,n in enumerate(run_node.body) if isinstance(n,ast.Import) and
                    any(a.name=='torch' for a in n.names))
                prefix=copy.deepcopy(run_node);prefix.body=prefix.body[:stop]
                def start(value):
                    self.assertIs(value,context)
                    self.assertIs(action(),f.legacy['origins'])
                    self.assertIs(fit['unit_started'],e.UNIT_STARTED,'inherited exit must use the original evaluator start')
                    self.assertIs(fit['phase_seconds'],phases)
                def run_prefix(node):
                    namespace={**vars(e),'cli':lambda _:sys.argv,'authority':lambda _:context,'native_start':start}
                    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(PATH),'exec'),namespace)
                    namespace['run'](context['args'])
                for phase in ('cpu','export','score'):
                    context['args'].phase=phase;fit.pop('unit_started',None)
                    before=e.time.perf_counter()-e.UNIT_STARTED
                    run_prefix(prefix)
                    after=e.time.perf_counter()-e.UNIT_STARTED
                    self.assertEqual(list(phases),['old_exit_begin','old_exit_end','own_union_begin',
                        'own_union_end','closure_begin','closure_end'])
                    self.assertTrue(all(before<=v<=after for v in phases.values()))
                    self.assertEqual(list(phases.values()),sorted(phases.values()))
                required={'legacy','old','root','args','code','guards','readout','phase_seconds','unit_started'}
                self.assertTrue(required<=accessed,'real private exit context fields were not exercised')
                # Each inherited required field is consumed by the authentic
                # private adapter; neither a gate nor a source predicate is stubbed.
                for key in sorted(required):
                    saved=fit.pop(key)
                    try:
                        phases.clear()
                        with self.subTest(missing=key),self.assertRaises(KeyError) as error:action()
                        self.assertEqual(error.exception.args,(key,))
                    finally:fit[key]=saved
                target=ast.parse("context['training_context']['fit_context']['unit_started']=None").body[0].targets[0]
                for replacement in (None,'time.perf_counter()',"context['training_context']['nearest'].UNIT_STARTED"):
                    class ChangeStart(ast.NodeTransformer):
                        count=0
                        def visit_Assign(self,node):
                            if any(ast.dump(t,include_attributes=False)==ast.dump(target,include_attributes=False) for t in node.targets):
                                self.count+=1
                                if replacement is None:return None
                                node.value=ast.parse(replacement,mode='eval').body
                            return self.generic_visit(node)
                    change=ChangeStart();mutant=change.visit(copy.deepcopy(prefix))
                    self.assertEqual(change.count,1)
                    fit.pop('unit_started',None)
                    expected=KeyError if replacement is None else AssertionError
                    with self.subTest(start=replacement),self.assertRaises(expected):run_prefix(mutant)
                    if replacement is None:
                        with self.assertRaisesRegex(KeyError,'unit_started'):action()
                context['args'].phase='export';run_prefix(prefix)
                self.assertIs(action(),f.legacy['origins'])
                f.set_origins({p:h for i,(p,h) in enumerate(f.files.items()) if i})
                with self.assertRaisesRegex(ValueError,'exact four'):action()
                f.set_origins(f.files)
                with self.assertRaisesRegex(ValueError,'owned legacy'):api.audit_origins(dict(f.legacy))
                private=next(c.cell_contents for c in api.audit_origins.__closure__ if isinstance(c.cell_contents,driver.FunctionType)
                    and c.cell_contents.__name__=='audit_origins')
                with patch.dict(private.__globals__,{'_nearest_supplement':{'files':{},'modules':{}}}),self.assertRaisesRegex(ValueError,'global binding changed'):action()
                manifest=Path(own['root'])/'execution.json';raw=manifest.read_bytes();manifest.write_bytes(raw+b' ')
                with self.assertRaisesRegex(ValueError,'SHA256'):action()
                manifest.write_bytes(raw);f.unchanged_originals(self)

    def test_source_order_bundle_loader_and_no_global_rebinding(self):
        source=PATH.read_text();tree=ast.parse(source);nodes={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        text=lambda name:ast.get_source_segment(source,nodes[name])
        score=text('score_exports')
        self.assertLess(score.index('archived_replay('),score.index('quality_after_readiness('))
        self.assertLess(score.index('repeated=native.read_wires('),score.index('quality_after_readiness('))
        self.assertLess(score.index("launch['stage'] == 'full' and immediate"),score.index('paired_intervals('))
        export=text('native_export');self.assertIn('for pass_index in range(2)',export)
        self.assertIn("portable.load_inference(directory,endpoint['bundle']['sha256'],'cuda')",export)
        self.assertIn('with bundle_reads_only(context,endpoint)',export);self.assertIn('require_exact=True',export)
        self.assertNotIn('native_raw',export);self.assertNotIn('cache_rows',export)
        exit_source=text('exit_rehash');self.assertIn("api.exit_rehash(t['fit_context'])",exit_source)
        self.assertIn("require_exact=context['args'].phase == 'export'",exit_source)
        for forbidden in ('reset_peak','globals()[','if False','torch.load('):
            if forbidden != 'torch.load(':self.assertNotIn(forbidden,source)
        for node in tree.body:
            if isinstance(node,ast.Import):self.assertTrue(all(n.name.split('.')[0] not in e.NATIVE for n in node.names))
        self.assertEqual(e.TRAIN_FILES,{'train_siglip2_compact_ranking.py','test_siglip2_compact_ranking.py'})


if __name__ == '__main__':
    unittest.main()
