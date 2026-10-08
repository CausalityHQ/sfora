#!/usr/bin/env python3
"""Bounded stdlib falsifiers only; CUDA, images and original native gates UNRUN."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
import unittest

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'diagnose_connected_gallery_freshness.py'
EVIDENCE = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'


def load_driver():
    assert DRIVER.is_file(), 'freshness diagnostic implementation missing'
    spec = importlib.util.spec_from_file_location('_freshness_test', DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def packed(rows):
    return b''.join(struct.pack('<128be', *(list(codes) + [0] * (128-len(codes))), inv)
                    for codes, inv in rows)


def source_facts(d):
    paths={name:HERE/file for name,file in d.SOURCE_FILES.items() if name not in ('scorer','packing')}
    paths['identity']=EVIDENCE/'export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py'
    return {name:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in paths.items()}


class PayloadTensor:
    """Native byte/device boundary standin; original typed hashing runs unchanged."""
    dtype='torch.float32';device='cpu';layout='torch.strided';requires_grad=False;grad_fn=None;_version=0
    def __init__(self,shape,value=.05):
        self.shape=shape;self.raw=struct.pack('<f',value)*math.prod(shape)
    def data_ptr(self):return id(self.raw)
    def detach(self):return self
    def cpu(self):return self
    def contiguous(self):return self
    def reshape(self,*a):return self
    def view(self,*a):return self
    def numpy(self):return self.raw


class ScoreTensor:
    """Tiny FP32 array standin for the actual original packed scorer body."""
    def __init__(self,values):self.v=copy.deepcopy(values)
    def copy(self):return ScoreTensor(self.v)
    def tolist(self):return copy.deepcopy(self.v)
    def float(self):return self
    def to(self,*a):return self
    def cpu(self):return self
    def min(self):return min(self.v)
    def max(self):return max(self.v)
    @property
    def T(self):return ScoreTensor([list(row) for row in zip(*self.v,strict=True)])
    def __getitem__(self,index):
        if isinstance(index,ScoreTensor):return ScoreTensor([[self.v[i] for i in row] if isinstance(row,list)
                                                           else self.v[row] for row in index.v])
        if not isinstance(index,tuple):return ScoreTensor(self.v[index] if isinstance(index,slice) else
            [self.v[i] for i in index] if isinstance(index,list) else self.v[index])
        row,col=index
        if row is None:return ScoreTensor([self[col].v])
        selected=self[row].v
        if col is None:return ScoreTensor([[v] for v in selected])
        return ScoreTensor([v[col] for v in selected])
    def __matmul__(self,other):
        return ScoreTensor([[float(sum(a*b for a,b in zip(row,col,strict=True)))
                             for col in zip(*other.v,strict=True)] for row in self.v])
    def binary(self,other,fn):
        a,b=self.v,other.v if isinstance(other,ScoreTensor) else other
        def apply(x,y):
            if not isinstance(x,list) and not isinstance(y,list):return fn(x,y)
            xx=x if isinstance(x,list) else [x];yy=y if isinstance(y,list) else [y]
            width=max(len(xx),len(yy));assert len(xx) in (1,width) and len(yy) in (1,width)
            return [apply(xx[0 if len(xx)==1 else i],yy[0 if len(yy)==1 else i]) for i in range(width)]
        return ScoreTensor(apply(a,b))
    def __mul__(self,other):return self.binary(other,lambda a,b:struct.unpack('<f',struct.pack('<f',a*b))[0])
    def __truediv__(self,other):return self.binary(other,lambda a,b:struct.unpack('<f',struct.pack('<f',a/b))[0])
    def __eq__(self,other):return self.binary(other,lambda a,b:a==b)
    def __le__(self,other):return self.binary(other,lambda a,b:a<=b)
    def cumsum(self,dim):
        assert dim==1
        result=[]
        for row in self.v:
            n=0;out=[]
            for v in row:n+=v;out.append(n)
            result.append(out)
        return ScoreTensor(result)
    def sum(self,dim):
        assert dim==1
        return ScoreTensor([struct.unpack('<f',struct.pack('<f',sum(row)))[0] for row in self.v])


@contextmanager
def metadata_fixture(d):
    """Only I/O stands in: original score/CPU/export/verification bytes stay pinned."""
    blobs={};objects={};logs={}
    def actual(fact,path):
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==fact['sha256']
        blobs[fact['path']]=(fact['sha256'],raw)
        return json.loads(raw)
    folder=EVIDENCE/'connected-mlp-evaluation-full-selection-score-v1'
    score_raw=(folder/'receipt.json').read_bytes();assert hashlib.sha256(score_raw).hexdigest()==d.SCORE_SHA
    score=json.loads(score_raw);verification_raw=(folder/'verification.json').read_bytes()
    verification=json.loads(verification_raw);score_unit=verification['terminal']
    actual(score_unit['receipt'],folder/'receipt.json')
    verification_fact={'path':'/fixture/verification.json','sha256':hashlib.sha256(verification_raw).hexdigest()}
    blobs[verification_fact['path']]=(verification_fact['sha256'],verification_raw)
    cpu=actual(score['launch']['selected_cpu']['receipt'],EVIDENCE/'connected-mlp-evaluation-full-cpu-v5/receipt.json')
    exports={}
    for key,unit in score['launch']['exports'].items():
        exports[key]=actual(unit['receipt'],EVIDENCE/('connected-mlp-evaluation-full-export-'+key+'-v2')/'receipt.json')
    for unit in [score_unit,score['launch']['selected_cpu'],*score['launch']['exports'].values()]:
        path=EVIDENCE/Path(unit['receipt']['path']).parent.name.removeprefix('sfora-')/'original.log'
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==unit['log']['sha256']
        logs[unit['log']['path']]=(unit['log']['sha256'],path)
    historical=score['input_guards']
    def pin(name,digest=None):
        p,h=next((p,h) for p,h in historical.items() if Path(p).name==name and (digest is None or h==digest))
        return {'path':p,'sha256':h}
    local=source_facts(d)
    sources={name:pin(file,local[name]['sha256'] if name not in ('packing','scorer') else
                      d.SCORER_SHA if name=='scorer' else None) for name,file in d.SOURCE_FILES.items()}
    scope_fact=score['launch']['scope'];scope=actual(scope_fact,EVIDENCE/'identity-diversity-v1/scope.json')
    rows=sorted((r for b in exports['control-179061']['images'] for r in b['rows']),key=lambda r:r['panel_ordinal'])
    targets=[0]*13283;fitrows=[{} for _ in targets];classes=['unused-'+str(i) for i in range(max(r['target'] for r in rows)+1)]
    for row in rows:
        targets[row['original_row']]=row['target'];fitrows[row['original_row']]=row
        classes[row['target']]=row['product']
    fit={'targets':targets,'rows':fitrows,'class_names':classes}
    fit_fact=pin('fit.json');cache_fact=pin('fit.npy','c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716')
    panel={'original_rows':[r['original_row'] for r in rows],
           'query':[i for i,r in enumerate(rows) if r['role']=='query'],
           'gallery':[i for i,r in enumerate(rows) if r['role']=='gallery'],
           'original_class_ids':sorted(set(r['target'] for r in rows))}
    partition={'original_fit':fit_fact,'original_cache':cache_fact,'panels':{'selection':panel,
        'train':{'original_rows':[r['original_fit_index'] for r in scope['control']['rows']]}}}
    partition_fact=next({'path':p,'sha256':h} for p,h in historical.items() if h==score['source']['partition_sha256'])
    runtime={'native_authority':score['source']['native_authority'],
             'source_cpu':score['source']['native_source']['source_cpu']['proof'],
             'warm':score['source']['warm_start']['terminal']['receipt']}
    objects.update({fit_fact['path']:fit,partition_fact['path']:partition})
    objects.update({v['path']:{} for v in runtime.values()})
    for endpoint in score['launch']['endpoints']:
        key=d.label(endpoint);ident=cpu['payload_facts'][key]['identity'];record=exports[key]
        member=str(Path(endpoint['bundle']['path']).parent/'endpoint.pt')
        objects[endpoint['bundle']['path']]={'schema':'siglip2-connected-mlp-bundle-v1',
            'endpoint_state_sha256':endpoint['inference_state_sha256'],
            'code':{'train_siglip2_connected_mlp.py':sources['connected']['sha256']},
            'files':{'endpoint.pt':historical[member]},'scope':ident['scope']}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        def npy(name,shape):
            p=tmp/name;header=repr({'descr':'<f4','fortran_order':False,'shape':shape}).encode()+b'\n'
            with p.open('wb') as stream:
                stream.write(b'\x93NUMPY\x01\x00'+len(header).to_bytes(2,'little')+header)
                stream.truncate(10+len(header)+4*shape[0]*shape[1])
            return p
        cache=npy('cache.npy',(13283,1152));array=npy('panel.npy',(3449,128));wire=packed([([1],1)]*3449)
        code={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in d.FILES}
        code_raw=json.dumps(code).encode();execution=hashlib.sha256(code_raw).hexdigest()
        launch={'schema':'connected-gallery-freshness-launch-v1','execution_sha256':execution,
            'score':score_unit['receipt'],'score_verification':verification_fact,'score_terminal':score_unit,
            'partition':partition_fact,'fit':fit_fact,'original_cache':cache_fact,'scope':scope_fact,
            'sources':sources,'runtime':runtime,'resource_policy':d.LIMITS,'both_locks_held':True,
            'candidate_status':'KILL','qualification_eligible':False,'state_reuse_eligible':False}
        args=SimpleNamespace(execution_sha256=execution,authority=tmp/'authority.json',
                             authority_sha256='a'*64,output=tmp/'output')
        objects[str(args.authority)]=launch
        blobs[str(HERE/'execution.json')]=(execution,code_raw)
        blobs.update({str(HERE/name):(digest,(HERE/name).read_bytes()) for name,digest in code.items()})
        reads=[];mutations={}
        def reader(fact,guards,*,keep=False):
            p,h=fact['path'],fact['sha256'];reads.append(p)
            assert Path(p).suffix.lower() not in {'.jpg','.png'} and not (p.endswith('.pt') and Path(p).name!='endpoint.pt')
            assert guards.setdefault(p,h)==h
            if p in blobs:
                expected,raw=blobs[p];assert expected==h and hashlib.sha256(raw).hexdigest()==h
            elif p in objects:raw=json.dumps(objects[p]).encode()
            elif p.endswith('.packed.bin'):raw=wire
            elif p in historical:
                assert historical[p]==h
                if not keep:return cache if p==cache_fact['path'] else array
                raw=b'fixture endpoint/source bytes'
            else:raise AssertionError('undeclared fixture read: '+p)
            if p in mutations:
                value=json.loads(raw);mutations[p](value);raw=json.dumps(value).encode()
            return raw if keep else tmp/'unused'
        yield SimpleNamespace(args=args,launch=launch,score=score,cpu=cpu,exports=exports,
            verification=verification,objects=objects,reader=reader,mutations=mutations,reads=reads,logs=logs,code=code)


class FreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load_driver() if DRIVER.is_file() else None

    def setUp(self):
        if self.d is None and self._testMethodName != 'test_implementation_exists':
            self.skipTest('implementation absent')

    def test_implementation_exists(self):
        self.assertTrue(DRIVER.is_file(), 'freshness diagnostic implementation missing')

    def test_signed_ties_unequal_inverses_interleaved_roles_and_ap(self):
        rows=[([1],1),([1],3),([-1],1),([2],1),([1],2),([1],1)]
        labels=('A','B','A','A','B','A');query=[0,2];gallery=[1,3,4,5]
        q=self.d.wire_quality(packed(rows),labels,query,gallery)
        self.assertEqual(q['per_query_r1'],[0,1]);self.assertEqual(q['per_query_ap'],[0.25,1.0])
        rows[4]=([1],2.5)
        lower=self.d.wire_quality(packed(rows),labels,query,gallery)
        self.assertEqual(lower['per_query_r1'],q['per_query_r1']);self.assertEqual(lower['per_query_ap'],[0.,1.])
        with self.assertRaisesRegex(ValueError,'AP'):
            self.d.replay(q, lower)

    def test_wire_layout_endian_nonfinite_and_exact_bytes(self):
        wire=packed([([-127,126],0.5), ([1,-1],2)])
        codes, inverse=self.d.decode_wire(wire,2)
        self.assertEqual(codes[0][:2],(-127,126))
        self.assertEqual(inverse,[0.5,2])
        for changed in (wire[:-1],wire+b'x',wire[:128]+b'\x00\x00'+wire[130:]):
            with self.assertRaises(ValueError): self.d.decode_wire(changed,2)
        with self.assertRaisesRegex(ValueError,'wire'):
            self.d.exact_wire(wire,wire[:128]+wire[128:130][::-1]+wire[130:])

    def test_perquery_replay_does_not_accept_equal_averages(self):
        a={'per_query_r1':[0,1],'per_query_ap':[0.25,0.5]}
        for b in ({'per_query_r1':[1,0],'per_query_ap':[0.25,0.5]},
                  {'per_query_r1':[0,1],'per_query_ap':[0.5,0.25]}):
            with self.assertRaises(ValueError):self.d.replay(a,b)

    def test_actual_original_packed_scorer_signed_ties_and_ap_mutation(self):
        rows=[([1],1),([1],3),([-1],1),([2],1),([1],2),([1],1)]
        labels=('A','B','A','A','B','A');query=[0,2];gallery=[1,3,4,5]
        def bins(ids):return ScoreTensor([ids.v.count(i) for i in range(max(ids.v)+1)])
        def argsort(values,**kw):
            self.assertEqual(kw,dict(dim=1,descending=True,stable=True))
            return ScoreTensor([sorted(range(len(row)),key=lambda i:-row[i]) for row in values.v])
        torch=SimpleNamespace(inference_mode=lambda:lambda fn:fn,from_numpy=lambda a:a,
            device=lambda name:name,tensor=lambda v,**kw:ScoreTensor(v),bincount=bins,
            arange=lambda start,end,**kw:ScoreTensor(list(range(start,end))),argsort=argsort,int32='int32',float32='float32')
        path=HERE/'compare_inshop_sop_warmstart_100.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.d.SCORER_SHA)
        context={'launch':{'sources':{'scorer':{'path':str(path),'sha256':self.d.SCORER_SHA}}},'guards':{}}
        def pack(value):
            codes,inverse=self.d.decode_wire(packed(rows),len(rows))
            self.assertEqual(value.v,[[i] for i in range(6)])
            return SimpleNamespace(codes=ScoreTensor([list(row) for row in codes]),inverse_norms=ScoreTensor(inverse))
        sources=SimpleNamespace(modules={'packing':SimpleNamespace(pack_int8_unit_embeddings=pack)})
        with patch.dict(sys.modules,{'torch':torch,'numpy':SimpleNamespace(mean=lambda v:sum(v)/len(v))}):
            original=self.d.scorer(sources,context)
            first=original(ScoreTensor([[i] for i in range(6)]),labels,query,gallery,device='cpu')
            self.assertEqual(first['per_query_r1'],[0,1]);self.assertEqual(first['per_query_ap'],[.25,1.])
            self.d.replay(first,self.d.native_wire_quality(packed(rows),labels,query,gallery))
            self.d.replay(first,self.d.wire_quality(packed(rows),labels,query,gallery))
            rows[4]=([1],2.5)
            changed=original(ScoreTensor([[i] for i in range(6)]),labels,query,gallery,device='cpu')
            self.assertEqual(changed['per_query_r1'],first['per_query_r1'])
            self.assertEqual(changed['per_query_ap'],[0.,1.])
            with self.assertRaisesRegex(ValueError,'AP'):self.d.replay(first,changed)

    def test_complete_typed_endpoint_state_and_member_mutants(self):
        sources=self.d.Sources({'launch':{'sources':source_facts(self.d)},'guards':{}})
        finite=SimpleNamespace(all=lambda:SimpleNamespace(item=lambda:True))
        torch=SimpleNamespace(Tensor=PayloadTensor,uint8='uint8',isfinite=lambda v:finite,
            count_nonzero=lambda v:SimpleNamespace(item=lambda:int(any(v.raw))))
        try:
            sources.admit();m=sources.modules
            disk={k:{} for k in m['connected'].INFERENCE_KEYS}
            provenance={'domain':'actual CPU-renormalized FP32 features','rows':6355,'view':'canonical',
                'reduction':'torch FP32 mean(dim=0) canonical ordinal order','accepted_checkpoint':m['identity'].ACCEPTED['checkpoint'],
                'canonical_cache':{'normalized':True,'raw_pooled_cache':False,'shape':[6355,1152],'dtype':'float32'},
                **{k:'f'*64 for k in ('canonical_features_sha256','original_rows_sha256','target_sha256','partition_sha256')}}
            disk.update(schema=m['connected'].INFERENCE_SCHEMA,arm='control',source={'accepted':'source'},
                numerical_flags={'threads':8},A=PayloadTensor((128,160)),C=PayloadTensor((128,1152)),
                mu_train=PayloadTensor((1152,)),means={'linear':PayloadTensor((32,)),'concat':PayloadTensor((160,))},
                mu_train_provenance=provenance,base_vision={'sha256':'b'*64,'checkpoint':{'sha256':'v'*64}},
                processor={'config':{'kind':'original'}},vision_sha256='v'*64,encoder_identity={'inventory':{}},scope={'control':6355})
            with patch.dict(sys.modules,{'torch':torch}):
                fingerprint=m['original'].fingerprint
                def binding(value):
                    value['fixed_sha256']=fingerprint({k:v for k,v in value.items() if k!='fixed_sha256'})
                    h=fingerprint(value);bundle={'path':'/fixture/bundle.json','sha256':'d'*64}
                    endpoint={'arm':'control','seed':179061,'bundle':bundle,'inference_state_sha256':h,'terminal_state_sha256':'t'*64}
                    manifest={'endpoint_state_sha256':h,'encoder_identity':value['encoder_identity'],'scope':value['scope'],
                              'files':{'vision.pt':'v'*64},'vision_sha256':'v'*64,'base_vision_sha256':'b'*64}
                    facts={'inference_state_sha256':h,'fixed_sha256':value['fixed_sha256'],'terminal_state_sha256':'t'*64,
                        'bundle':bundle,'members':{k:fingerprint(value[k]) for k in m['evaluator'].MEMBERS},
                        'identity':{'arm':'control','seed':179061,**{k:value[k] for k in ('source','scope','base_vision','encoder_identity')}},
                        'vision_sha256':'v'*64,'base_vision_sha256':'b'*64,'processor_config_sha256':fingerprint(value['processor']['config'])}
                    return endpoint,manifest,facts
                endpoint,manifest,facts=binding(disk)
                check=lambda v,e=endpoint,b=manifest,f=facts:self.d.check_endpoint_payload(v,e,b,f,disk['source'],disk['numerical_flags'],m)
                check(disk)
                for name,mutate in (
                    ('missing member',lambda v:v.pop('encoder')),
                    ('typed member',lambda v:v.__setitem__('config',[])),
                    ('wrong mu',lambda v:v.__setitem__('mu_train',PayloadTensor((1152,),.5))),
                    ('omitted C',lambda v:v.__setitem__('C',PayloadTensor((128,1152),0.))),
                    ('extra member',lambda v:v.__setitem__('unknown',{}))):
                    with self.subTest(name=name):
                        changed=copy.deepcopy(disk);mutate(changed)
                        with self.assertRaises(ValueError):check(changed)
                for name,mutate,reason in (
                    ('wrong shape',lambda v:v.__setitem__('A',PayloadTensor((160,128))),'shape'),
                    ('wrong dtype',lambda v:setattr(v['mu_train'],'dtype','torch.float64'),'dtype'),
                    ('zero C',lambda v:v.__setitem__('C',PayloadTensor((128,1152),0.)),'nonzero'),
                    ('TRAIN provenance',lambda v:v['mu_train_provenance'].__setitem__('rows',6354),'provenance')):
                    with self.subTest(name=name):
                        changed=copy.deepcopy(disk);mutate(changed);e,b,f=binding(changed)
                        with self.assertRaisesRegex(ValueError,reason):check(changed,e,b,f)
            sources.guard()
        finally:sources.close()

    def test_npy_rejects_object_big_endian_fortran_trailing_and_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'array.npy'
            def write(descr='<f4',order=False,shape=(2,128),extra=b''):
                h=repr({'descr':descr,'fortran_order':order,'shape':shape}).encode()+b'\n'
                p.write_bytes(b'\x93NUMPY\x01\x00'+len(h).to_bytes(2,'little')+h+b'\0'*(2*128*4)+extra)
                return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            self.d.npy_header(write(),(2,128),{})
            for kw in ({'descr':'>f4'},{'descr':'|O'},{'order':True},{'shape':(128,2)},{'extra':b'x'}):
                with self.assertRaises(ValueError):self.d.npy_header(write(**kw),(2,128),{})

    def test_explicit_file_reads_and_uncached_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.json';p.write_text('{}')
            fact={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}; guards={}
            self.d.read_json(fact,guards)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'SHA'):
                self.d.rehash_files(guards)
            for name in ('sample.jpg','vision.pt','resume.pt','control-terminal.pt'):
                bad=Path(tmp)/name;bad.write_bytes(b'forbidden')
                with self.assertRaisesRegex(ValueError,'forbidden'):
                    self.d.file_bytes({'path':str(bad),'sha256':hashlib.sha256(bad.read_bytes()).hexdigest()},{})

    def test_panel_role_mutations_and_fit_scope_not_official_ordinals(self):
        panel={'original_rows':[7,2,9,4],'query':[2,0],'gallery':[3,1],
               'original_class_ids':[0,1]}
        fit={'class_names':['A','B','T'],'targets':[2]*10,'rows':[{} for _ in range(10)]}
        for r,t in ((7,0),(2,0),(9,1),(4,1)):fit['targets'][r]=t
        scope={'control':{'rows':[{'original_fit_index':0},{'original_fit_index':1}],
                          'class_names':['T'],'original_rows':[7,9]}}
        partition={'panels':{'selection':panel,'train':{'original_rows':[0,1]}}}
        labels=self.d.panel_mapping(partition,fit,scope,counts=(4,2,2,2),train_counts=(2,1))
        self.assertEqual(labels,('A','A','B','B'))
        for field,value in (('query',[0,0]),('gallery',[3,0]),('original_rows',[7,7,9,4])):
            bad=copy.deepcopy(partition);bad['panels']['selection'][field]=value
            with self.assertRaises(ValueError):self.d.panel_mapping(bad,fit,scope,counts=(4,2,2,2),train_counts=(2,1))
        scope['control']['rows'][0]['original_fit_index']=7
        with self.assertRaises(ValueError):self.d.panel_mapping(partition,fit,scope,counts=(4,2,2,2),train_counts=(2,1))

    def test_role_cell_composition_and_descriptive_interaction(self):
        stale=[[10],[20],[30],[40]];fresh=[[1],[2],[3],[4]]
        self.assertEqual(self.d.compose(stale,fresh,[2,0],[3,1],'SF'),[[10],[2],[30],[4]])
        self.assertEqual(self.d.compose(stale,fresh,[2,0],[3,1],'FS'),[[1],[20],[3],[40]])
        cells={k:{'per_query_r1':[v],'per_query_ap':[v/2]} for k,v in {'SS':0,'SF':1,'FS':1,'FF':1}.items()}
        facts=self.d.effects(cells)
        self.assertEqual(facts['per_query_r1']['interaction'],[-1])

    def test_failed_control_tap_stops_and_reports_differences(self):
        a={k:b'abc' for k in ('raw','unit','codes','inverse_norms','wire')}
        b={**a,'raw':b'abd'}
        differences=self.d.byte_differences(a,b)
        self.assertEqual(differences['raw']['different_bytes'],1)
        with self.assertRaisesRegex(ValueError,'control tap'):
            self.d.require_control_tap(differences)

    def test_whole_cap_and_cleanup_preserve_original_error(self):
        b=self.d.Budget(started=0,clock=lambda:700)
        with self.assertRaisesRegex(ValueError,'700'):b.check()
        error=ValueError('original tap failure');calls=[]
        with self.assertRaisesRegex(ValueError,'original tap failure') as caught:
            self.d.cleanup_error(error,[lambda:calls.append('first'),lambda:(_ for _ in ()).throw(ValueError('exit failure')),lambda:calls.append('last')])
        self.assertIs(caught.exception,error);self.assertEqual(calls,['first','last'])
        self.assertTrue(any('exit failure' in n for n in error.__notes__))

    def test_original_readout_seam_uses_terminal_c_and_mu(self):
        # Reuse the repository's stdlib standins; run the actual frozen function.
        spec=importlib.util.spec_from_file_location('_freshness_original_test',HERE/'test_siglip2_identity_diversity.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        _,probe=fixture.oracle_fixture('candidate',179061)
        c={n:v.cell_contents for n,v in zip(probe.__code__.co_freevars,probe.__closure__,strict=True)}
        Tensor=type(c['canonical']);state=c['state'];torch=c['torch'];F=c['functional']
        frozen=HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py'
        raw=frozen.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),self.d.IDENTITY_SHA)
        nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('fullfeature_raw_features','parameter_roles','require')]
        ns={'ARMS':('control','candidate')};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(frozen),'exec'),ns)
        snapshot=dict(ns);features=c['canonical'];C=Tensor(.03,'C',(128,1152));mu=Tensor([.6,.4],shape=(1152,))
        def independently_loaded():
            return c['Parameter'](Tensor(.05,'A',(128,160)),0),C.clone(),mu.clone()
        one=independently_loaded();two=independently_loaded()
        self.assertTrue(all(a is not b for a,b in zip(one,two,strict=True)))
        with patch.dict(sys.modules,{'torch':torch,'torch.nn':SimpleNamespace(functional=F)}):
            def call(tensors):
                A,residual,mean=tensors
                return ns['fullfeature_raw_features'](features,None,A,None,residual,mean,'control',c['primitive'],c['readout'])
            actual=call(one);reloaded=call(two)
            expected=c['readout'].raw_features(features,None,one[0],None,'concat',c['primitive'])+F.linear(features.detach().float()-mu,C)
            numbers=lambda value:[[float(v) for v in row] for row in value.tolist()]
            self.assertEqual(numbers(actual),numbers(expected));self.assertEqual(numbers(actual),numbers(reloaded))
            omitted=call((one[0],Tensor(0.,'C',(128,1152)),mu))
            wrong=call((one[0],C,Tensor([0.,0.],shape=(1152,))))
            self.assertNotEqual(numbers(actual),numbers(omitted));self.assertNotEqual(numbers(actual),numbers(wrong))
        self.assertEqual(ns.keys(),snapshot.keys());self.assertTrue(all(ns[k] is v for k,v in snapshot.items()))

    def test_registry_tamper_and_partial_admission_cleanup(self):
        from types import ModuleType
        owner=ModuleType('_freshness_owned_test');victim=ModuleType('_freshness_foreign_test');foreign=ModuleType(victim.__name__)
        sources=self.d.Sources({});sources.modules={'one':owner,'two':victim};sources.checks=[lambda:None]
        with patch.dict(sys.modules,{owner.__name__:owner,victim.__name__:foreign}):
            with self.assertRaisesRegex(ValueError,'registry'):sources.close()
            self.assertIs(sys.modules[victim.__name__],foreign);self.assertNotIn(owner.__name__,sys.modules)
            self.assertEqual(sources.modules,{});self.assertEqual(sources.checks,[])
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'broken.py';p.write_text("raise ValueError('partial admission failed')\n")
            fact={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            sources=self.d.Sources({'launch':{'sources':{'bad':fact}},'guards':{}})
            with self.assertRaisesRegex(ValueError,'partial admission'):sources.load('bad')
            sources.close();self.assertNotIn('_gallery_freshness_bad',sys.modules)

    def test_final_whole_cap_includes_exit_cleanup(self):
        clock=[0.];budget=self.d.Budget(started=0,clock=lambda:clock[0])
        budget.check();clock[0]=700.
        with self.assertRaisesRegex(ValueError,'700'):
            self.d.final_state(budget,None,None,None,None,None,{})

    def test_genuine_final_cgroup_cuda_rng_and_flags_after_disposal(self):
        sources=self.d.Sources({'launch':{'sources':source_facts(self.d)},'guards':{}})
        sources.admit();initializer=sources.modules['initializer'];sources.close()
        before={'path':'/sys/fs/cgroup/test.service','values':{'memory.max':str(8*1024**3),
            'memory.current':'1','memory.peak':'1','memory.swap.current':'0','memory.swap.peak':'0',
            'memory.swap.max':'0','memory.events':'max 0\noom 0\noom_kill 0'}}
        after=copy.deepcopy(before);peak=[1];rng=[1];flags={'threads':8};budget=self.d.Budget()
        source=SimpleNamespace(cgroup_memory=lambda:after,numerical_flags=lambda:flags)
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,synchronize=lambda:None,
            max_memory_allocated=lambda:peak[0],memory_allocated=lambda:0,get_rng_state_all=lambda:[rng[0]]),
            random=SimpleNamespace(get_rng_state=lambda:1),equal=lambda a,b:a==b)
        with patch.dict(sys.modules,{'torch':torch}):
            receipt={};self.d.final_state(budget,source,initializer,before,(1,[1]),{'threads':8},receipt)
            self.assertIs(receipt['cgroup_after'],after)
            after['values']['memory.peak']=str(8*1024**3+1)
            with self.assertRaisesRegex(ValueError,'cgroup caps'):
                self.d.final_state(budget,source,initializer,before,(1,[1]),{'threads':8},{})
            after['values']['memory.peak']='1';peak[0]=10_000_000_000
            with self.assertRaisesRegex(ValueError,'CUDA cap'):
                self.d.final_state(budget,source,initializer,before,(1,[1]),{'threads':8},{})
            peak[0]=1;rng[0]=2
            with self.assertRaisesRegex(ValueError,'RNG'):
                self.d.final_state(budget,source,initializer,before,(1,[1]),{'threads':8},{})
            rng[0]=1;flags['threads']=4
            with self.assertRaisesRegex(ValueError,'flags'):
                self.d.final_state(budget,source,initializer,before,(1,[1]),{'threads':8},{})

    def test_actual_original_source_guards_and_module_disposal(self):
        sources=self.d.Sources({'launch':{'sources':source_facts(self.d)},'guards':{}})
        try:
            sources.admit();sources.guard()
            self.assertEqual(set(sources.modules),self.d.SOURCE_NAMES-{'scorer','packing'})
            original=sources.modules['original'];fn=original.fingerprint;prior=fn.__code__
            # Mutate only the test-owned module; restore before its genuine guard/cleanup.
            try:
                fn.__code__=(lambda value:None).__code__
                with self.assertRaisesRegex(ValueError,'source|function'):sources.guard()
            finally:fn.__code__=prior
            sources.guard()
        finally:sources.close()
        self.assertFalse(any(n.startswith('_gallery_freshness_') for n in sys.modules))

    def test_complete_prepare_original_receipts_and_terminal_admission(self):
        import builtins
        importer=builtins.__import__;native={'torch','numpy','PIL','transformers','torchvision','sfora'};attempts=[]
        def stdlib_only(name,*a,**kw):
            if name.split('.')[0] in native:attempts.append(name);raise AssertionError('forbidden native import')
            return importer(name,*a,**kw)
        with metadata_fixture(self.d) as f,patch.object(self.d,'file_bytes',f.reader),patch('builtins.__import__',stdlib_only):
            context=self.d.prepare(f.args)
            self.assertEqual(len(context['labels']),3449);self.assertEqual(len(context['exports']),4)
            self.assertEqual(context['launch']['score']['sha256'],self.d.SCORE_SHA)
            mutations=[
                ('source role',str(f.args.authority),lambda v:v['sources'].__setitem__('initializer',v['sources']['extract'])),
                ('historical fitter',str(f.args.authority),lambda v:v['sources'].__setitem__('fitter',next(
                    {'path':p,'sha256':h} for p,h in f.score['input_guards'].items()
                    if Path(p).name=='fit_siglip2_prototype_residual.py' and h.startswith('cc3ff')))),
                ('UNIT',str(f.args.authority),lambda v:v['score_terminal'].__setitem__('invocation_id','0'*32)),
                ('fingerprint',f.score['launch']['exports']['control-179061']['receipt']['path'],
                    lambda v:v.__setitem__('inference_state_sha256','0'*64)),
                ('CPU facts',f.score['launch']['selected_cpu']['receipt']['path'],
                    lambda v:v['payload_facts']['control-179061'].__setitem__('fixed_sha256','0'*64)),
                ('endpoint member',f.score['launch']['endpoints'][0]['bundle']['path'],
                    lambda v:v['files'].__setitem__('endpoint.pt','0'*64)),
                ('partition',f.launch['partition']['path'],lambda v:v['panels']['selection']['query'].__setitem__(0,1)),
                ('extra closure',str(HERE/'execution.json'),lambda v:v.__setitem__('extra.py','0'*64))]
            for name,p,mutate in mutations:
                with self.subTest(name=name):
                    f.mutations[p]=mutate
                    with self.assertRaises((ValueError,AssertionError)):self.d.prepare(f.args)
                    f.mutations.clear()
            self.assertEqual(attempts,[])
        # The genuine reader and duration-only original adapter execute all six
        # authenticated logs; its file reader alone maps unavailable remote paths.
        sources=self.d.Sources({'launch':{'sources':source_facts(self.d)},'guards':{}})
        try:
            sources.admit();m=sources.modules;reader=m['original'].FlatAdmission();reader.init=m['initializer']
            terminal=m['fitter'].original_terminal_reader({'legacy':{'original':m['original'],'admission':reader},
                                                         'guards':sources.context['guards']})
            with metadata_fixture(self.d) as f:
                def log_reader(guards,path,digest):
                    expected,p=f.logs[path];self.assertEqual(digest,expected)
                    self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),digest);guards[path]=digest;return p
                reader.bound_file=log_reader
                units=[(f.score,f.launch['score_terminal'],700),(f.cpu,f.score['launch']['selected_cpu'],700)]
                units += [(f.exports[key],unit,1500) for key,unit in f.score['launch']['exports'].items()]
                for record,unit,cap in units:
                    terminal(reader,record,unit,cap,sources.context['guards'])
                    bad=copy.deepcopy(unit);bad['invocation_id']='0'*32
                    with self.assertRaisesRegex(ValueError,'invocation'):
                        terminal(reader,record,bad,cap,sources.context['guards'])
                    bad=copy.deepcopy(unit);bad['service_seconds']=cap+1
                    with self.assertRaisesRegex(ValueError,'duration'):
                        terminal(reader,record,bad,cap,sources.context['guards'])
                sources.guard()
        finally:sources.close()
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError,'native imports'):
                self.d.run(SimpleNamespace())

    def test_cli_help_and_optimized_rejection_without_native_imports(self):
        help_result=subprocess.run([sys.executable,'-B',str(DRIVER),'--help'],capture_output=True,text=True)
        self.assertEqual(help_result.returncode,0,help_result.stderr)
        result=subprocess.run([sys.executable,'-B','-O',str(DRIVER),'--help'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('optimized',result.stderr.lower())
        self.assertFalse({'torch','numpy','PIL','transformers','torchvision'}.intersection(sys.modules))


if __name__=='__main__':
    if not __debug__:raise SystemExit('optimized test mode forbidden')
    unittest.main()
