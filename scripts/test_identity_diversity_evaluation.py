#!/usr/bin/env python3
"""Stdlib source/contract falsifiers only; no native performance qualification."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from types import FunctionType, SimpleNamespace
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().with_name('evaluate_siglip2_identity_diversity.py')
TRAINER_ROOT=Path(os.environ.get('SFORA_IDENTITY_DIVERSITY_TRAINER_ROOT',str(PATH.parent)))


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


e=module('_diversity_evaluation_tests',PATH)
legacy=module('_diversity_historical_tests',PATH.with_name('test_compact_ranking_evaluation.py'))
math_helper=legacy.math_helper


def descriptor(path,digest='a'*64):
    return {'path':str(path),'sha256':digest}


def unit(number,*,proof=False):
    result=legacy.unit(number)
    if proof:result['proof']=result.pop('receipt')
    return result


def launch(stage='first',phase='cpu',panel='selection'):
    result,args=legacy.launch(stage,phase,panel)
    result.update(schema=e.AUTHORITY_SCHEMA,training=copy.deepcopy(e.TRAINING),cost_policy=e.COST_POLICY,
        scope=descriptor('/tmp/frozen-scope.json',e.SCOPE_SHA256),candidate_cache={
            'exporter':{'root':'/tmp/new-exporter','execution_sha256':'e'*64,
                'code':dict.fromkeys(e.EXPORTER_FILES,'f'*64)},
            'authority':descriptor('/tmp/candidate-cache-launch.json'),'terminal':unit(30,proof=True)})
    return result,args


def scope(arm):
    return {'arm':arm,'manifest_sha256':e.SCOPE_SHA256,'arm_sha256':e.ARM_SHA256[arm]}


def pair():
    records=[]
    for index,arm in enumerate(e.ARMS):
        common={k:'a'*64 for k in ('initial_A_sha256','initial_C_sha256','mu_train_sha256','mu_train_provenance_sha256',
            'common_statistics_sha256')}
        ident={**common,'arm':arm,'seed':179061,'scope':scope(arm),'source':{},'numerical_flags':{},
            'parameter_names':['A','C'],'parameter_shapes':[[128,160],[128,1152]],
            'common_initial_sha256':'b'*64,'static_sha256':str(index+1)*64,
            'schedule_provenance_sha256':str(index+3)*64,'initial_cpu_rng_sha256':'c'*64,'initial_cuda_rng_sha256':'d'*64}
        records.append({**common,'arm':arm,'seed':179061,'source':{},'numerical_flags':{},'identity':ident,
            'common_input_raw_unit_packed_sha256':'e'*64,'initial_raw_unit_packed_sha256':str(index+5)*64,
            'steps':[{'batch':[index],'gradients':[index]}]})
    return records


def payload_facts(endpoint,record):
    return {'identity':copy.deepcopy(record['identity']),
        'members':dict.fromkeys(('config','buffers','processor','head','A','means','C','mu_train',
            'mu_train_provenance','scope','common_statistics','arm'),'a'*64),
        'vision_sha256':'b'*64,'fixed_sha256':'c'*64,'processor_config_sha256':'d'*64,
        'terminal_state_sha256':endpoint['terminal_state_sha256'],
        'inference_state_sha256':endpoint['inference_state_sha256'],'bundle':endpoint['bundle']}


class DiversityTests(unittest.TestCase):
    def test_actual_trainer_exact2_descriptor_and_public_apis(self):
        self.assertEqual(e.TRAINING['execution_sha256'],'a12a1aa5624c032c08616298b3211d35e8f0ff3c2fcfc95e6a170187dc36fd76')
        self.assertEqual(e.FILES,{'evaluate_siglip2_identity_diversity.py','test_identity_diversity_evaluation.py'})
        self.assertEqual(e.TRAIN_FILES,{'train_siglip2_identity_diversity.py','test_siglip2_identity_diversity.py'})
        guards={}
        self.assertEqual(e.closure(TRAINER_ROOT,e.TRAINING['execution_sha256'],e.TRAIN_FILES,guards),e.TRAINING['code'])
        trainer=e.load_authenticated('_diversity_actual_trainer',TRAINER_ROOT/'train_siglip2_identity_diversity.py',
            e.TRAINING['code']['train_siglip2_identity_diversity.py'],guards)
        try:
            self.assertEqual(trainer.FILES,e.TRAIN_FILES)
            for name in ('load_inference','inference_outputs','release_inference','admit_bundle','inference_readout_tree',
                    'prepare_scope','release_scope','canonical_row','check_payload','check_scope_schedule_fact'):
                self.assertTrue(callable(getattr(trainer,name)))
            self.assertEqual((trainer.SCHEMA,trainer.AUTHORITY_SCHEMA,trainer.INFERENCE_SCHEMA,trainer.BUNDLE_SCHEMA),
                ('siglip2-identity-diversity-v1','siglip2-identity-diversity-launch-v1',
                 'siglip2-identity-diversity-inference-v1','siglip2-identity-diversity-bundle-v1'))
            self.assertEqual(trainer.SCOPE_SHA256,e.SCOPE_SHA256)
            self.assertEqual(trainer.ARM_SHA256,e.ARM_SHA256)
            self.assertEqual(trainer.RECIPE['classes'],{'control':1008,'candidate':2016})
            self.assertNotIn('hinge',trainer.RECIPE)
            value,args=launch()
            trainlaunch={'schema':trainer.AUTHORITY_SCHEMA,'execution_sha256':e.TRAINING['execution_sha256'],
                'phase':'cpu','arm':'control','seed':179061,'nearest':trainer.NEAREST,'fitter':trainer.FITTER,
                'accepted':trainer.ACCEPTED,'readout':trainer.READOUT,'recipe':trainer.RECIPE,
                'resource_policy':trainer.policy('cpu'),'both_locks_held':True,'selected_cpu':None,'selected_mechanics':None,
                'native_authority':descriptor('/tmp/native-authority'),'scope':value['scope'],
                'candidate_cache':value['candidate_cache']}
            trainer.check_launch(trainlaunch,SimpleNamespace(execution_sha256=e.TRAINING['execution_sha256'],phase='cpu',arm='control',seed=179061))
            self.assertTrue({'scope','common_statistics','cache_provenance','masks','schedule_provenance'} <= set(trainer.STATIC_KEYS))
            self.assertTrue({'scope','common_statistics','C','mu_train','mu_train_provenance'} <= trainer.INFERENCE_KEYS)
            for bad in ({k:None for k in trainer.INFERENCE_KEYS if k!='C'},
                    dict.fromkeys(trainer.INFERENCE_KEYS|{'extra'}),{**dict.fromkeys(trainer.INFERENCE_KEYS),'schema':'foreign'}):
                with self.assertRaises(ValueError):e.check_inference_members(trainer,bad)
        finally:
            sys.modules.pop('_diversity_actual_trainer')
        self.assertFalse(any(n.split('.')[0] in e.NATIVE for n in sys.modules))

    def test_launch_file_unit_scope_cache_and_prerequisite_mutants(self):
        for stage in ('first','full'):
            for phase in ('cpu','export','score'):
                value,args=launch(stage,phase);e.check_launch(value,args)
                self.assertIn('scope',e.binding({'launch':value}))
                self.assertIn('candidate_cache',e.binding({'launch':value}))
                for field,bad in (('scope',descriptor('/tmp/scope','0'*64)),('scope',{'path':'relative','sha256':e.SCOPE_SHA256}),
                        ('scope',{**value['scope'],'extra':True}),('schema',legacy.e.AUTHORITY_SCHEMA),
                        ('training',legacy.e.TRAINING),('both_locks_held',False),('selection_previously_exposed',False),
                        ('cost_policy',legacy.e.COST_POLICY),('endpoints',value['endpoints'][::-1]),('stage','search')):
                    with self.subTest(stage=stage,phase=phase,field=field),self.assertRaises((ValueError,KeyError)):
                        e.check_launch({**value,field:bad},args)
                for field,bad in (('exporter',{}),('authority',descriptor('relative')),('terminal',unit(30)),
                        ('terminal',{**unit(30,proof=True),'both_locks_held':False})):
                    changed=copy.deepcopy(value);changed['candidate_cache'][field]=bad
                    with self.subTest(field=field),self.assertRaises((ValueError,KeyError)):
                        e.check_launch(changed,args)
        value,args=launch('full','score','validation');e.check_launch(value,args)
        for field in ('selected_cpu','first_selection','selection_go'):
            with self.assertRaises(ValueError):e.check_launch({**value,field:None},args)
        value,args=launch('full');value['first_selection']=value['endpoints'][0]['terminal']
        with self.assertRaises(ValueError):e.check_launch(value,args)
        value,args=launch();value['endpoints'][0]['seed']=True
        with self.assertRaises(ValueError):e.check_launch(value,args)
        for malformed in ({'path':'relative','sha256':'a'*64},descriptor('/tmp/file','A'*64),
                {**descriptor('/tmp/file'),'extra':True}):
            with self.assertRaises(ValueError):e.check_file(malformed)
        for field,bad in (('service_seconds',True),('native_peak_rss_kib',float('nan')),
                ('invocation_id',''),('both_locks_held',False)):
            with self.assertRaises(ValueError):e.check_unit({**unit(1),field:bad})

    def test_common_input_and_state_exact_with_legitimately_different_scopes(self):
        c,a=pair();e.check_paired_initialization(c,a)
        for location,fields in (('record',('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',
                'mu_train_provenance_sha256','common_input_raw_unit_packed_sha256','common_statistics_sha256','numerical_flags')),
                ('identity',('source','initial_A_sha256','initial_C_sha256','mu_train_sha256','mu_train_provenance_sha256',
                 'common_initial_sha256','common_statistics_sha256','numerical_flags','initial_cpu_rng_sha256','initial_cuda_rng_sha256'))):
            for field in fields:
                changed=copy.deepcopy(a);target=changed if location=='record' else changed['identity']
                target[field]='f'*64
                with self.subTest(location=location,field=field),self.assertRaises(ValueError):e.check_paired_initialization(c,changed)
        for field,bad in (('parameter_names',['C','A']),('parameter_shapes',[[128,160],[128,1151]]),
                ('parameter_shapes',[[128.,160],[128,1152]]),('scope',scope('control')),('schedule_provenance_sha256','bad')):
            changed=copy.deepcopy(a);changed['identity'][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):e.check_paired_initialization(c,changed)
        changed=copy.deepcopy(a);changed['steps']=[{'batch':[42],'gradients':[1,2,3]}]
        changed['initial_raw_unit_packed_sha256']='9'*64
        e.check_paired_initialization(c,changed)

    def test_scope_lifetime_release_before_next_endpoint_and_on_error(self):
        events=[];t={};records={}
        def prepare(context,arm):
            self.assertNotIn('initial',context);events.append(('prepare',arm))
            context['initial']={'scope':scope(arm)};return context['initial']
        def release(context):events.append(('release',context.pop('initial')['scope']['arm']))
        trainer=SimpleNamespace(prepare_scope=prepare,release_scope=release,scope_identity=lambda value:value)
        context={'trainer':trainer,'training_context':t,'records':records}
        for arm in e.ARMS:records[179061,arm]={'identity':{'scope':scope(arm)}}
        for arm in e.ARMS:
            with e.endpoint_scope(context,{'seed':179061,'arm':arm}):self.assertEqual(t['initial']['scope'],scope(arm))
        with self.assertRaisesRegex(RuntimeError,'forward'):
            with e.endpoint_scope(context,{'seed':179061,'arm':'candidate'}):raise RuntimeError('forward')
        records[179061,'candidate']['identity']['scope']=scope('control')
        with self.assertRaises(ValueError):
            with e.endpoint_scope(context,{'seed':179061,'arm':'candidate'}):self.fail('scope swap reached forward')
        self.assertNotIn('initial',t)
        self.assertEqual(events,[('prepare','control'),('release','control')]+[('prepare','candidate'),('release','candidate')]*3)

    def test_each_train_scope_schedule_matches_its_complete_cpu_qualification(self):
        records={};qualifications=[];calls=[]
        for seed in e.SEEDS:
            for arm,original in zip(e.ARMS,pair(),strict=True):
                record=copy.deepcopy(original);record['seed']=seed;record['identity']['seed']=seed
                record['identity']['ranking_bank_sha256']=('1' if arm=='control' else '2')*64
                record.update(scope_schedule=[[0,1] if arm=='control' else [2,3]],scope_masks=[[True,False]],
                    schedule_provenance={'seed':seed,'classes':1008 if arm=='control' else 2016})
                records[seed,arm]=record;qualifications.append(copy.deepcopy(record))
        cpu={'qualifications':qualifications,'common_input_raw_unit_packed_sha256':'e'*64,'common_statistics_sha256':'a'*64}
        t={'terminals':{'cpu:179061:control':cpu}}
        trainer=SimpleNamespace(check_scope_schedule_fact=lambda record:calls.append((record['seed'],record['arm'])))
        e.check_training_qualification(trainer,t,records,'full')
        self.assertEqual(calls,list(e.ORDER))
        for field,bad in (('scope_schedule',[[2,1]]),('scope_masks',[[False,True]]),
                ('schedule_provenance',{}),('common_input_raw_unit_packed_sha256','f'*64)):
            changed=copy.deepcopy(records);changed[179069,'control'][field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):e.check_training_qualification(trainer,t,changed,'full')
        for field in ('static_sha256','schedule_provenance_sha256','scope','common_initial_sha256','ranking_bank_sha256'):
            changed=copy.deepcopy(records);changed[179069,'candidate']['identity'][field]='foreign'
            with self.subTest(field=field),self.assertRaises(ValueError):e.check_training_qualification(trainer,t,changed,'full')

    def test_archived_perquery_replay_and_all_wire_readbacks_precede_candidate_metrics(self):
        q,source,concat=legacy.panels('first');launch_value,_=launch('first','score');events=[]
        rows=[{'id':0},{'id':1}]
        digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        class Embedding:
            def __init__(self,arm):self.arm=arm
            def numpy(self):return self.arm
        values={arm:(arm,Embedding(arm),'codes','norms') for arm in e.ARMS}
        def read(context,output,key,files,count):events.append(('read',key));return values[key.split('-')[0]]
        def replay(context,fixed):events.extend([('archive','source'),('archive','concat')]);return source,concat
        def quality(arm,*args,**kwargs):events.append(('metric',arm));return q['179061'][arm]
        def exact(left,right):
            if left != right:raise ValueError('wire changed before metric')
        fixed=SimpleNamespace(packed_quality=quality)
        baseline=SimpleNamespace(scoring_math=lambda _:fixed,value_facts=lambda context,value:{'whole':'facts'})
        exports={}
        for endpoint in launch_value['endpoints']:
            arm=endpoint['arm'];triples=[[0,1,2]] if arm=='control' else [[3,4,5]]
            exports[e.label(endpoint)]={'payload_facts':{'arm':arm},'ordered_images_sha256':digest(rows),
                'output':'/tmp/export-'+arm,'files':{},'panel_facts':{'whole':'facts'},
                'diagnostic':{'triples':triples,'triples_sha256':digest(triples),'margins':[.03],
                    'diagnostic_only':True,'utility_veto':False,'remine':False}}
        context={'launch':launch_value,'export_records':exports,'cpu':{'payload_facts':{
                e.label(v):{'arm':v['arm']} for v in launch_value['endpoints']}},
            'score_context':{'baseline':baseline,'fit':{'class_names':['x','y'],'targets':[0,1]},
                'partition':{'panels':{'selection':{'original_rows':[0,1],'query':[0],'gallery':[1]}}}},
            'nearest_evaluator':SimpleNamespace(archived_replay=replay,image_rows=lambda *_:(rows,{}),read_wires=read),
            'reference':SimpleNamespace(json_digest=digest,replay_equal=exact),'helper':SimpleNamespace(exact=exact),
            'math':math_helper,'costs':e.paired_cost(legacy.controls('first'),'first'),'preparation_costs':{},
            'records':{(179061,a):{'identity':{'scope':scope(a)}} for a in e.ARMS}}
        builtins={**vars(sys.modules['builtins'])}
        fake_modules={'numpy':SimpleNamespace(),'torch':SimpleNamespace(device=lambda name:name)}
        builtins['__import__']=lambda name,*args,**kwargs:fake_modules[name]
        execute=FunctionType(e.score_exports.__code__,{**vars(e),'__builtins__':builtins})
        result=execute(context)
        self.assertEqual(result['decision'],'CONTINUE');self.assertEqual(result['bootstrap_draws'],0)
        first_metric=next(i for i,event in enumerate(events) if event[0]=='metric')
        self.assertEqual(events[:first_metric],[('archive','source'),('archive','concat'),
            ('read','control-179061'),('read','control-179061'),('read','candidate-179061'),('read','candidate-179061')])
        self.assertEqual(set(result['diagnostic']['179061']),set(e.ARMS))
        events.clear();reads=[0]
        def corrupt(*args):
            reads[0]+=1;value=read(*args)
            return ('mutant',*value[1:]) if reads[0]==4 else value
        context['nearest_evaluator'].read_wires=corrupt
        with self.assertRaises(ValueError):execute(context)
        self.assertFalse(any(event[0]=='metric' for event in events))

    def test_scope_payload_facts_and_official_train_micro16_namespaces(self):
        value,_=launch();records={}
        for endpoint,record in zip(value['endpoints'],pair(),strict=True):records[endpoint['seed'],endpoint['arm']]=record
        context={'records':records}
        for endpoint in value['endpoints']:
            record=records[endpoint['seed'],endpoint['arm']];facts=payload_facts(endpoint,record)
            e.check_payload_facts(context,facts,endpoint)
            for field,bad in (('identity',records[179061,'candidate' if endpoint['arm']=='control' else 'control']['identity']),
                    ('bundle',descriptor('/tmp/foreign')),('terminal_state_sha256','f'*64),
                    ('members',{k:v for k,v in facts['members'].items() if k!='C'})):
                with self.subTest(field=field),self.assertRaises(ValueError):e.check_payload_facts(context,{**facts,field:bad},endpoint)
        for arm in e.ARMS:
            ids=list(reversed(range(16)))
            rows=[{'original_train_row':1000+i*3,'scoped_target':i,'relative_path':'Img/img/'+str(i),
                'image_sha256':format(i,'064x')} for i in range(16)]
            record={'identity':{'scope':scope(arm)},'scope_schedule':[ids]}
            initial={'scope':{'arm':arm},'original_rows':[r['original_train_row'] for r in rows]}
            def canonical(context,state,ordinal):
                row=rows[ordinal];return row,Path('/tmp/dataset')/row['relative_path'],{'official_train_row':row['original_train_row'],'scope':scope(arm)}
            trainer=SimpleNamespace(canonical_row=canonical)
            context={'trainer':trainer,'training_context':{'initial':initial,'scope_manifest':{arm:{'rows':rows}},
                'legacy':{'prior':{'fit':{'dataset_root':'/tmp/dataset'}}}},'records':{(179061,arm):record}}
            output=e.train_rows(context,ids)
            witness={'batch':ids,'rows':output,'residual_oracle':{'C_exact_zero':False,'residual_nonzero_witness':True,
                'omitted_C_mutant_rejected':True,'wrong_mu_mutant_rejected':True}}
            endpoint={'seed':179061,'arm':arm};e.check_train_witness(context,witness,endpoint)
            for field,bad in (('original_row',0),('scope',scope('candidate' if arm=='control' else 'control')),
                    ('path','/tmp/wrong-image'),('scoped_target',999),('image_sha256','f'*64)):
                changed=copy.deepcopy(witness);changed['rows'][0][field]=bad
                with self.subTest(arm=arm,field=field),self.assertRaises(ValueError):e.check_train_witness(context,changed,endpoint)
            with self.assertRaises(ValueError):e.check_train_witness(context,{**witness,'batch':list(range(16))},endpoint)

    def test_actual_control_and_new_candidate_preparation_costs_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);documents={}
            def write(name,value):
                path=root/name;path.write_text(json.dumps(value));fact=descriptor(path,hashlib.sha256(path.read_bytes()).hexdigest())
                documents[path.name]=value;return fact
            def prepare(number,seconds):
                startup=unit(number,proof=True);startup['service_seconds']=number+.125
                export=unit(number+1,proof=True);export['service_seconds']=seconds
                for item in (startup,export):
                    item['log']=write(str(number)+'-'+item['unit']+'.log',{'original':'complete log standin'})
                proof={'startup_terminal':write(str(number)+'-startup.json',startup),'full':'admitted export standin'}
                export['proof']=write(str(number)+'-export.json',proof)
                return export,proof
            control,cp=prepare(10,291.75);candidate,ap=prepare(30,310.25)
            context={'guards':{},'launch':{'candidate_cache':{'terminal':candidate}},'training_context':{
                'legacy':{'selected':{'source':{'preparation_terminal':control},'export_record':cp}},
                'candidate_preparation':{'source':{'terminal':candidate},'record':ap}}}
            result=e.preparation_costs(context)
            self.assertEqual(result['control']['export_service_seconds'],291.75)
            self.assertEqual(result['candidate']['export_service_seconds'],310.25)
            self.assertEqual(result['total_preparation_service_seconds'],642.25)
            self.assertEqual(result['candidate']['export'],candidate)
            self.assertNotIn('shared_export_seconds',e.COST_POLICY)
            changed=copy.deepcopy(context);changed['training_context']['candidate_preparation']['record']['full']='foreign'
            with self.assertRaises(ValueError):e.preparation_costs(changed)
            log=Path(candidate['log']['path']);prior=log.stat();log.write_bytes(b'changed')
            os.utime(log,ns=(prior.st_atime_ns,prior.st_mtime_ns))
            with self.assertRaises(ValueError):e.preparation_costs(context)

    def test_first_explicit_retained_concat_floors_and_full_original_gates(self):
        q,source,concat=legacy.panels('first');cost=e.paired_cost(legacy.controls('first'),'first')
        self.assertEqual(e.decide(math_helper,q,source,concat,'first','selection',{},cost)['decision'],'CONTINUE')
        for r1,ap in (([0]*1734,[.82]*1734),([1]*1734,[.8177754035543955]*1734)):
            low_source=legacy.quality([0]*1734,[.5]*1734);low_concat=legacy.quality([0]*1734,[.6]*1734)
            bad=copy.deepcopy(q);bad['179061']['control']=legacy.quality([0]*1734,[.7]*1734)
            bad['179061']['candidate']=legacy.quality(r1,ap)
            # Preserve the intended one-ulp-below-floor aggregate: repeated sum
            # in the historical fixture rounds this synthetic mean to the floor.
            bad['179061']['candidate']['map_at_r']=statistics.mean(ap)
            self.assertEqual(e.decide(math_helper,bad,low_source,low_concat,'first','selection',{},cost)['decision'],'KILL')
        at=e.RETAINED_CONCAT_FLOORS
        self.assertFalse(e.selection_floors_pass({'179061':{'candidate':at}},'selection'))
        self.assertTrue(e.selection_floors_pass({'179061':{'candidate':{**at,'recall_at_1':at['recall_at_1']+1e-12}}},'selection'))
        self.assertTrue(e.selection_floors_pass({'179061':{'candidate':at}},'validation'))
        q,source,concat=legacy.panels('full');cost=e.paired_cost(legacy.controls('full'),'full');bounds=legacy.intervals(q)
        for metric in e.METRICS:
            bad=copy.deepcopy(bounds);bad[metric]['product_lower95']=0
            self.assertEqual(e.decide(math_helper,q,source,concat,'full','selection',bad,cost)['decision'],'KILL')
        weak=copy.deepcopy(q)
        for seed in weak:
            weak[seed]['candidate']=legacy.quality(weak[seed]['candidate']['per_query_r1'],[.819]*1734)
        self.assertEqual(e.decide(math_helper,weak,source,concat,'full','selection',legacy.intervals(weak),cost)['decision'],'KILL')

    def test_runtime_and_448_vision_readout_scope_fingerprints(self):
        namespace=retained_namespace()
        fixture=namespace['EndpointFactsFixture']()
        fixture.state.update(scope={'arm':'candidate','payload':{'rows':['official']}},
            common_statistics={'control_e0':.125,'common_input':{'features_sha256':'a'*64}})
        facts=e.endpoint_facts(fixture.context,fixture.state)
        self.assertEqual(facts['vision_sha256'],fixture.fingerprint(fixture.vision))
        for field in ('scope','common_statistics','C','mu_train','means'):
            old=copy.deepcopy(fixture.state[field]);fixture.state[field]={'substituted':True}
            self.assertNotEqual(e.endpoint_facts(fixture.context,fixture.state)['members'][field],facts['members'][field])
            fixture.state[field]=old
        tree=ast.parse(PATH.read_text());functions={n.name:ast.get_source_segment(PATH.read_text(),n)
            for n in tree.body if isinstance(n,ast.FunctionDef)}
        export=functions['scoped_native_export'];payload=functions['authenticate_payloads'];authority=functions['authority']
        for name in ('load_inference','inference_outputs','release_inference','admit_bundle'):
            self.assertIn(name,PATH.read_text())
        self.assertIn('for pass_index in range(2)',export)
        self.assertIn('portable.release_inference(state)',export)
        self.assertIn('readback_wires(',export);self.assertIn('require_exact=True',export)
        self.assertIn("trainer.check_payload(t,disk,ident,128)",payload)
        self.assertIn("manifest['vision_inventory'] == t['initial']['provenance']['encoder']['inventory']",payload)
        for term in ('scope','common_statistics','C','mu_train','mu_train_provenance'):
            self.assertIn("'"+term+"'",payload)
        self.assertIn('trainer.admit_terminal(',authority)
        self.assertIn("record['completed_step'] == 128",authority)
        self.assertLess(authority.index("all(v['pass'] for v in costs.values())"),authority.index('archived = read_json('))
        score=functions['score_exports']
        self.assertLess(score.index('archived_replay('),score.index('quality_after_readiness('))
        self.assertLess(score.index('repeated=native.read_wires('),score.index('quality_after_readiness('))
        self.assertIn("launch['stage'] == 'full' and immediate",score)
        self.assertNotIn("c['triples'] == a['triples']",score)
        self.assertNotIn('283.636',PATH.read_text())
        self.assertNotIn('globals()[',PATH.read_text())
        self.assertNotIn('reset_peak',PATH.read_text())

    def test_complete_source_delta_and_unchanged_native_math_quality_seams(self):
        original=PATH.with_name('evaluate_siglip2_compact_ranking.py').read_text()
        self.assertEqual(hashlib.sha256(original.encode()).hexdigest(),HISTORICAL_SHA256)
        current=PATH.read_text();lines=current.splitlines(keepends=True)
        for start,expected,replacement in reversed(SOURCE_EDITS):
            count=len(expected.splitlines(keepends=True))
            self.assertEqual(''.join(lines[start:start+count]),expected)
            lines[start:start+count]=replacement.splitlines(keepends=True)
        self.assertEqual(''.join(lines),original)
        self.assertEqual(ast.dump(ast.parse(''.join(lines)),include_attributes=False),
            ast.dump(ast.parse(original),include_attributes=False))
        oldtree=ast.parse(original);newtree=ast.parse(current)
        old={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
        new={n.name:n for n in newtree.body if isinstance(n,ast.FunctionDef)}
        for name in ('strict_json','check_file','bound_file','closure','load_authenticated','guard_helpers',
                'check_resource_facts','check_unit','paired_cost','metric_deltas','immediate_quality_pass',
                'quality_after_readiness','accept_unit','native_start','fullfeature_oracle','cpu_calibration',
                'images_outputs','native_train_witness','diagnostic_triples','train_diagnostic','export_pass',
                'exit_rehash','resources','bundle_reads_only','lazy_runtime_files','distribution_identity_files',
                'original_native_map_witness','grouped_md_origin'):
            self.assertEqual(ast.dump(new[name],include_attributes=False),ast.dump(old[name],include_attributes=False),name)
        for n in newtree.body:
            if isinstance(n,ast.Import):self.assertTrue(all(a.name.split('.')[0] not in e.NATIVE for a in n.names))


def retained_namespace():
    """Reuse actual historical stdlib fixtures in a private definition namespace."""
    namespace={**vars(legacy),'e':e,'PATH':PATH}
    source=PATH.with_name('test_compact_ranking_evaluation.py').read_text()
    classes=[n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in
        {'EndpointFactsFixture','PortableRuntimeFixture','GroupedMdFixture','SourceAdmissionFixture'}]
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(PATH),'exec'),namespace)
    return namespace


# Existing behavior tests run against the new evaluator; no production helper globals change.
class RetainedSeamsTests(unittest.TestCase):
    pass


_retained=retained_namespace()
for _name in (
    'test_first_screen_never_bootstraps','test_full_equal_seed_floor_sign_and_intervals',
    'test_each_seed_core_and_whole_cost_boundary','test_no_quality_before_readiness',
    'test_diagnostic_and_native_roles','test_strict_files_closure_and_restored_mtime','test_whole_unit_resource_caps',
    'test_pre_native_partition_and_pinned_source_adapter','test_partition_order_descriptor_cache_and_hash_rejected_before_source',
    'test_bundle_boundary_denies_train_warm_optimizer_reads',
    'test_bundle_boundary_self_maps_reads_and_proc_denials_fresh_and_cached',
    'test_bundle_boundary_imports_record_pinned_source_without_cached_code',
    'test_bundle_boundary_native_grants_require_record_guard_and_original_cpu_origin',
    'test_bundle_boundary_grouped_md_metadata_fresh_cached_and_source',
    'test_bundle_boundary_grouped_md_map_denials_before_hashing',
    'test_bundle_boundary_grouped_md_identity_record_and_origin_denials',
    'test_numpy_missing_origin_rejects_new_files_symlinks_and_record_rows'):
    _fn=getattr(legacy.EvaluationTests,_name)
    setattr(RetainedSeamsTests,_name,FunctionType(_fn.__code__,_retained,_name,_fn.__defaults__,_fn.__closure__))

# SOURCE_EDITS and HISTORICAL_SHA256 follow: exact reviewed prospective source delta.
HISTORICAL_SHA256 = '8ecaa206d0343dde117cec97fa92a0179cb8a5991127d4c18dc7917a7c535563'
SOURCE_EDITS = (
    (1, '"""Independent identity-diversity CPU admission, native bundle export and paired scoring.\n', '"""Independent compact-ranking CPU admission, native bundle export and paired scoring.\n'),
    (37, "SCHEMA = 'siglip2-identity-diversity-evaluation-v1'\nAUTHORITY_SCHEMA = 'siglip2-identity-diversity-evaluation-launch-v1'\nFILES = {'evaluate_siglip2_identity_diversity.py', 'test_identity_diversity_evaluation.py'}\nTRAIN_FILES = {'train_siglip2_identity_diversity.py', 'test_siglip2_identity_diversity.py'}\nTRAINING = {'code': {'test_siglip2_identity_diversity.py': '986ced3e9a0234b7aa2c26ae4688fb4e12810a42c8f017ed6d7007e655027d8f', 'train_siglip2_identity_diversity.py': 'c71100e09db22625905dffeee8368a7c068457ecea5f5068a234630e51ed4be8'}, 'execution_sha256': 'a12a1aa5624c032c08616298b3211d35e8f0ff3c2fcfc95e6a170187dc36fd76', 'root': '/home/riomus/runs/sfora-so400-identity-diversity-train-source-v2'}\nSCOPE_SHA256 = '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726'\nARM_SHA256 = {'control':'1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280',\n              'candidate':'c12e557afa89b53dc37b984ef0c1d369c4dbff42186c2fc63f4adc331ad31882'}\nEXPORTER_FILES = {'export_siglip2_identity_diversity_views.py','test_siglip2_identity_diversity_views.py'}\n", "SCHEMA = 'siglip2-compact-live-top1-evaluation-v1'\nAUTHORITY_SCHEMA = 'siglip2-compact-live-top1-evaluation-launch-v1'\nFILES = {'evaluate_siglip2_compact_ranking.py', 'test_compact_ranking_evaluation.py'}\nTRAIN_FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}\nTRAINING = {'root': '/home/riomus/runs/sfora-so400-live-top1-train-source-v2', 'execution_sha256': '3a7e7a42866e44ada935287fb15025fd0ab3469acea1a4579ca90e3ab8584f88', 'code': {'train_siglip2_compact_ranking.py': '80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218', 'test_siglip2_compact_ranking.py': '70f86f22065888d569f834fc1b4b70f47e3a5d10331b013e158e220c2892a92e'}}\n"),
    (51, "RETAINED_CONCAT_FLOORS = {'recall_at_1':0.9648212226066898,'map_at_r':0.8177754035543956}\n", ''),
    (640, " 'preparation':'actual original control and fresh candidate startup/export UNITs separately authenticated',\n 'shared_export_in_ratios':False,\n", " 'shared_export_seconds':283.636,'shared_export_in_ratios':False,\n"),
    (645, " 'resource_policies','cost_policy','both_locks_held','selection_previously_exposed','scope','candidate_cache'}\n", " 'resource_policies','cost_policy','both_locks_held','selection_previously_exposed'}\n"),
    (757, "\ndef selection_floors_pass(quality,panel):\n    return panel != 'selection' or all(q['candidate']['recall_at_1'] > RETAINED_CONCAT_FLOORS['recall_at_1'] and\n        q['candidate']['map_at_r'] >= RETAINED_CONCAT_FLOORS['map_at_r'] for q in quality.values())\n\n", ''),
    (835, "    check_file(launch['scope'])\n    require(launch['scope']['sha256'] == SCOPE_SHA256, 'frozen complete scope FILE differs')\n    cache = launch['candidate_cache']\n    require(isinstance(cache,dict) and cache.keys() == {'exporter','authority','terminal'},\n        'fresh candidate preparation authority required')\n    check_code(cache['exporter'],EXPORTER_FILES)\n    check_file(cache['authority'])\n    check_preparation_unit(cache['terminal'])\n", ''),
    (888, 'def check_preparation_unit(value):\n    require(isinstance(value,dict) and \'proof\' in value and \'receipt\' not in value,\n        \'exporter original proof UNIT required\')\n    check_unit({(\'receipt\' if k == \'proof\' else k):v for k,v in value.items()})\n\n\ndef preparation_costs(context):\n    """Costs from already fully admitted original and prospective source units."""\n    t=context[\'training_context\']; selected=t[\'legacy\'][\'selected\']\n    candidate=t[\'candidate_preparation\']; result={}\n    for arm,unit,proof in ((\'control\',selected[\'source\'][\'preparation_terminal\'],selected[\'export_record\']),\n            (\'candidate\',candidate[\'source\'][\'terminal\'],candidate[\'record\'])):\n        check_preparation_unit(unit)\n        require(read_json(unit[\'proof\'],context[\'guards\']) == proof, \'actual preparation proof differs\')\n        startup=read_json(proof[\'startup_terminal\'],context[\'guards\'])\n        check_preparation_unit(startup)\n        for u in (startup,unit):\n            bound_file(context[\'guards\'],u[\'log\'][\'path\'],u[\'log\'][\'sha256\'])\n        result[arm]={\'startup\':startup,\'export\':unit,\'startup_service_seconds\':startup[\'service_seconds\'],\n            \'export_service_seconds\':unit[\'service_seconds\'],\n            \'total_service_seconds\':startup[\'service_seconds\']+unit[\'service_seconds\']}\n    require(result[\'candidate\'][\'export\'] == context[\'launch\'][\'candidate_cache\'][\'terminal\'] and\n        result[\'control\'][\'export\'] != result[\'candidate\'][\'export\'], \'distinct actual scope preparation required\')\n    return {**result,\'total_preparation_service_seconds\':sum(v[\'total_service_seconds\'] for v in result.values()),\n        \'preparation_excluded_from_training_ratios\':True,\'bundle_preparation_separate\':True,\n        \'qualification_separate\':True,\'both_cache_target_preparation_charged_to_core\':True}\n\n\n', ''),
    (921, '    immediate = all(immediate_quality_pass(quality[str(s)],source,concat,count) for s in seeds(stage)) and selection_floors_pass(quality,panel)\n', '    immediate = all(immediate_quality_pass(quality[str(s)],source,concat,count) for s in seeds(stage))\n'),
    (972, "        'stage','panel','endpoints','cost_policy','both_locks_held','selection_previously_exposed','scope','candidate_cache')}\n", "        'stage','panel','endpoints','cost_policy','both_locks_held','selection_previously_exposed')}\n"),
    (1004, "        ident=record['identity']\n        require(ident['scope'] == {'arm':arm,'manifest_sha256':SCOPE_SHA256,'arm_sha256':ARM_SHA256[arm]} and\n            record['seed'] == ident['seed'] and type(ident['seed']) is int and ident['seed'] in SEEDS and\n            all(record[k] == ident[k] for k in ('source','numerical_flags','initial_A_sha256','initial_C_sha256',\n                'mu_train_sha256','mu_train_provenance_sha256')) and\n            ident['common_statistics_sha256'] == record['common_statistics_sha256'] and\n            all(sha(ident[k]) for k in ('static_sha256','schedule_provenance_sha256','common_initial_sha256',\n                'common_statistics_sha256')) and sha(record['common_input_raw_unit_packed_sha256']),\n            'separate complete scope/statistics/schedule initialization required')\n    require(c['seed'] == a['seed'] and all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','common_input_raw_unit_packed_sha256','common_statistics_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('common_initial_sha256','common_statistics_sha256','initial_A_sha256','initial_C_sha256',\n", "    require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_C_sha256','mu_train_sha256',\n            'mu_train_provenance_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and\n        all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','initial_C_sha256',\n"),
    (1017, "            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')),\n        'fresh same-seed shared initialization/common-input/RNG differs')\n", "            'initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and\n        [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],\n        'fresh same-seed complete initialization/RNG/schedule differs')\n"),
    (1042, "    for key,filename in (('training','train_siglip2_identity_diversity.py'),\n", "    for key,filename in (('training','train_siglip2_compact_ranking.py'),\n"),
    (1052, "        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-identity-diversity-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-identity-diversity-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-identity-diversity-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-identity-diversity-bundle-v1' and math_helper.ORDER == ORDER and\n", "        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-live-top1-v1' and\n        trainer.AUTHORITY_SCHEMA == 'siglip2-compact-live-top1-launch-v1' and\n        trainer.INFERENCE_SCHEMA == 'siglip2-compact-live-top1-inference-v1' and\n        trainer.BUNDLE_SCHEMA == 'siglip2-compact-live-top1-bundle-v1' and math_helper.ORDER == ORDER and\n"),
    (1061, "    require(t['launch']['scope'] == launch['scope'] and t['launch']['candidate_cache'] == launch['candidate_cache'],\n        'evaluator scope/candidate preparation differs from trainer admission')\n", ''),
    (1087, "            manifest['scope'] == record['identity']['scope'] and\n            manifest['common_statistics_sha256'] == record['identity']['common_statistics_sha256'] and\n", ''),
    (1095, "    check_training_qualification(trainer,t,records,launch['stage'])\n", ''),
    (1101, "    preparation=preparation_costs({'training_context':t,'launch':launch,'guards':guards})\n", ''),
    (1141, "        'preparation_costs':preparation,\n", ''),
    (1159, "\n\ndef check_training_qualification(trainer,t,records,stage):\n    cpu=t['terminals']['cpu:'+str(SEEDS[0])+':control']\n    qualified={(q['seed'],q['arm']):q for q in cpu['qualifications']}\n    require(qualified.keys() == {(s,a) for s in SEEDS for a in ARMS}, 'complete CPU four-scope qualification required')\n    for seed,arm in endpoint_order(stage):\n        record=records[seed,arm]; q=qualified[seed,arm]\n        trainer.check_scope_schedule_fact(record)\n        require(record['common_input_raw_unit_packed_sha256'] == cpu['common_input_raw_unit_packed_sha256'] ==\n                q['common_input_raw_unit_packed_sha256'] and\n            record['common_statistics_sha256'] == cpu['common_statistics_sha256'] == q['common_statistics_sha256'] and\n            all(record['identity'][k] == q['identity'][k] for k in ('scope','static_sha256','schedule_provenance_sha256',\n                'common_initial_sha256','common_statistics_sha256','ranking_bank_sha256','initial_A_sha256','initial_C_sha256',\n                'mu_train_sha256','mu_train_provenance_sha256')) and\n            record['scope_schedule'] == q['scope_schedule'] and record['scope_masks'] == q['scope_masks'] and\n            record['schedule_provenance'] == q['schedule_provenance'],\n            'TRAIN endpoint changes CPU-qualified scope/common-input/static/schedule authority')\n", ''),
    (1187, "    require(record['preparation_costs'] == context['preparation_costs'],\n        'complete actual original-control/new-candidate preparation costs differ')\n", ''),
    (1201, "    for k in ('training','nearest_evaluator','genuine_evaluator','reference','cost_policy','selection_previously_exposed','scope','candidate_cache'):\n", "    for k in ('training','nearest_evaluator','genuine_evaluator','reference','cost_policy','selection_previously_exposed'):\n"),
    (1223, "            check_payload_facts(context,record['payload_facts'][label(endpoint)],endpoint)\n", ''),
    (1236, "        check_train_witness(context,record['train_witness'],endpoint)\n", "        require(len(record['train_witness']['batch']) == 16, 'actual TRAIN micro16 witness required')\n        check_residual_oracle(record['train_witness']['residual_oracle'],arm)\n"),
    (1337, "            'C','mu_train','mu_train_provenance','scope','common_statistics')}\n", "            'C','mu_train','mu_train_provenance')}\n"),
    (1358, "    require(trainer.scope_identity(disk['scope']) == ident['scope'] == manifest['scope'] == disk['source']['scope'] and\n        disk['scope'] == t['initial']['scope'] and\n        trainer.fingerprint(t,disk['common_statistics']) == ident['common_statistics_sha256'] ==\n            manifest['common_statistics_sha256'] == disk['source']['common_statistics_sha256'],\n        'bundle complete scope/common statistics substitution rejected')\n", ''),
    (1388, "\n\ndef check_payload_facts(context,facts,endpoint):\n    require(facts.keys() == {'identity','members','vision_sha256','fixed_sha256','processor_config_sha256',\n            'terminal_state_sha256','inference_state_sha256','bundle'} and\n        facts['identity'] == json_form(context['records'][endpoint['seed'],endpoint['arm']]['identity']) and\n        facts['members'].keys() == {'config','buffers','processor','head','A','means','C','mu_train',\n            'mu_train_provenance','scope','common_statistics','arm'} and\n        all(sha(v) for v in facts['members'].values()) and\n        all(sha(facts[k]) for k in ('vision_sha256','fixed_sha256','processor_config_sha256')) and\n        facts['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and\n        facts['inference_state_sha256'] == endpoint['inference_state_sha256'] and facts['bundle'] == endpoint['bundle'],\n        'complete qualified endpoint/scope/current-byte payload facts differ')\n", ''),
    (1467, "        with endpoint_scope(context,e):\n            first=authenticate_payloads(context,e); second=authenticate_payloads(context,e)\n            require(first == second, 'independent full payload/bundle current-byte authentication differs')\n            facts[label(e)]=first\n", "        first=authenticate_payloads(context,e); second=authenticate_payloads(context,e)\n        require(first == second, 'independent full payload/bundle current-byte authentication differs')\n        facts[label(e)]=first\n"),
    (1476, "\n\n@contextmanager\ndef endpoint_scope(context,endpoint):\n    trainer,t=context['trainer'],context['training_context']\n    initial=trainer.prepare_scope(t,endpoint['arm'])\n    del initial\n    try:\n        require(trainer.scope_identity(t['initial']['scope']) == context['records'][endpoint['seed'],endpoint['arm']]['identity']['scope'],\n            'endpoint active scope differs')\n        yield\n    finally:\n        trainer.release_scope(t)\n", ''),
    (1768, "            for k in ('config','buffers','processor_config','head','A','means','C','mu_train','mu_train_provenance','arm',\n                'scope','common_statistics')}}\n", "            for k in ('config','buffers','processor_config','head','A','means','C','mu_train','mu_train_provenance','arm')}}\n"),
    (1812, "    t=context['training_context']; trainer=context['trainer']; rows=[]\n", "    t=context['training_context']; legacy=t['legacy']; fit=legacy['prior']['fit']; rows=[]\n"),
    (1814, "        row,path,identity=trainer.canonical_row(t,t['initial'],ordinal)\n        rows.append({'train_ordinal':ordinal,'original_row':identity['official_train_row'],\n            'scope':identity['scope'],'scoped_target':row['scoped_target'],\n", "        row,path,_=t['nearest'].canonical_row(t,t['initial'],ordinal)\n        rows.append({'train_ordinal':ordinal,'original_row':int(t['initial']['original_rows'][ordinal]),\n"),
    (1819, "\n\ndef check_train_witness(context,witness,endpoint):\n    t=context['training_context']; arm,seed=endpoint['arm'],endpoint['seed']\n    record=context['records'][seed,arm]; scope=t['scope_manifest'][arm]\n    ids=record['scope_schedule'][0][:16]\n    require(witness['batch'] == ids and len(witness['rows']) == len(ids) == 16,\n        'actual scope first TRAINmicro16 schedule required')\n    root=Path(t['legacy']['prior']['fit']['dataset_root'])\n    for ordinal,row in zip(ids,witness['rows'],strict=True):\n        selected=scope['rows'][ordinal]\n        require(row == {'train_ordinal':ordinal,'original_row':selected['original_train_row'],\n            'scope':record['identity']['scope'],'scoped_target':selected['scoped_target'],\n            'path':str((root/selected['relative_path']).resolve()),'image_sha256':selected['image_sha256']},\n            'TRAIN witness substitutes scope/FIT/official image row namespace')\n    check_residual_oracle(witness['residual_oracle'],arm)\n", ''),
    (1904, "    endpoint=next(e for e in context['launch']['endpoints'] if\n        (e['seed'],e['arm']) == (context['args'].seed,context['args'].arm))\n    with endpoint_scope(context,endpoint):\n        return scoped_native_export(context)\n\n\ndef scoped_native_export(context):\n", ''),
    (1924, "            portable=load_authenticated(portable_name,directory/'train_siglip2_identity_diversity.py',\n                context['launch']['training']['code']['train_siglip2_identity_diversity.py'],context['guards'])\n", "            portable=load_authenticated(portable_name,directory/'train_siglip2_compact_ranking.py',\n                context['launch']['training']['code']['train_siglip2_compact_ranking.py'],context['guards'])\n"),
    (1936, "                    ('config','buffers','head','A','means','C','mu_train','mu_train_provenance','arm','scope','common_statistics')),\n", "                    ('config','buffers','head','A','means','C','mu_train','mu_train_provenance','arm')),\n"),
    (1983, "    head=t['legacy']['selected']['cached'].head_from('control',tensors=t['common']['head']).requires_grad_(False).train()\n", "    head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()\n"),
    (1986, "        raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(t['common']['A'].clone()),\n            t['common']['means'],'concat',t['legacy']['quadratic'])\n", "        raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(t['initial']['A'].clone()),\n            t['initial']['means'],'concat',t['legacy']['quadratic'])\n"),
    (2017, "        require(all(v['diagnostic_only'] is True and v['utility_veto'] is False and v['remine'] is False and\n            v['triples_sha256'] == context['reference'].json_digest(v['triples']) and\n            len(v['triples']) == len(v['margins']) <= 128 and\n            all(type(x) in (int,float) and math.isfinite(x) for x in v['margins']) for v in (c,a)),\n            'separate fixed scoped TRAIN diagnostics cannot remine or veto utility')\n        diagnostics[str(seed)]={arm:{'scope':context['records'][seed,arm]['identity']['scope'],\n            'triples':len(v['triples']),'violations':sum(x<.05 for x in v['margins']),\n            'diagnostic_only':True,'utility_veto':False,'remine':False} for arm,v in zip(ARMS,(c,a),strict=True)}\n", "        require(c['triples'] == a['triples'] and c['triples_sha256'] == a['triples_sha256'] and\n            all(v['diagnostic_only'] is True and v['utility_veto'] is False and v['remine'] is False for v in (c,a)),\n            'fixed TRAIN diagnostic cannot remine or veto utility')\n        diagnostics[str(seed)]=diagnostic_result(c['margins'],a['margins'])\n"),
    (2040, "    immediate=all(immediate_quality_pass(quality[str(seed)],source,concat,PANELS[panel_name][1]) for seed in seeds(launch['stage'])) and selection_floors_pass(quality,panel_name)\n", "    immediate=all(immediate_quality_pass(quality[str(seed)],source,concat,PANELS[panel_name][1]) for seed in seeds(launch['stage']))\n"),
    (2058, "        'preparation_costs':context['preparation_costs']}\n", "        'preparation_costs':{'shared_genuine_export_seconds':283.636,'shared_bundle_preparation_separate':True,\n            'shared_qualification_separate':True,'both_cache_target_preparation_charged_to_core':True}}\n"),
    (2136, "        'preparation_costs':context['preparation_costs'],\n", ''),
)


if __name__ == '__main__':
    unittest.main()
