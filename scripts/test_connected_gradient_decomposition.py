#!/usr/bin/env python3
"""Stdlib source falsifiers; native gradient/resource qualification is UNRUN.

Narrow checks: --test NAME under timeout15/address-space1GiB.
The root owns the final --source-only gate under timeout120/address-space1GiB.
"""
import argparse
import ast
from contextlib import nullcontext, redirect_stderr
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'qualify_connected_gradient_decomposition.py'


class Vector:
    def __init__(self, values, dtype='float32'):
        self.values, self.dtype = list(values), dtype
        self.shape, self.grad = (len(self.values),), None
        self.device = SimpleNamespace(type='cuda')

    def numel(self): return len(self.values)
    def reshape(self, *shape): return self
    def detach(self): return Vector(self.values, self.dtype)
    def double(self): return Vector(self.values, 'float64')
    def __getitem__(self, key): return Vector(self.values[key], self.dtype)
    def __add__(self, other): return Vector([a+b for a,b in zip(self.values, other.values, strict=True)], self.dtype)
    def __sub__(self, other): return Vector([a-b for a,b in zip(self.values, other.values, strict=True)], self.dtype)
    def __mul__(self, other): return Vector([a*b for a,b in zip(self.values, other.values, strict=True)], self.dtype)
    def sum(self): return Vector([math.fsum(self.values)], self.dtype)
    def abs(self): return Vector([abs(v) for v in self.values], self.dtype)
    def max(self): return Vector([max(self.values)], self.dtype)
    def item(self):
        assert len(self.values) == 1
        return self.values[0]
    def all(self): return Vector([all(self.values)])
    def add_(self, other):
        self.values = (self+other).values
        return self


def fake_vectors():
    return SimpleNamespace(float32='float32',
        zeros_like=lambda p: Vector([0.]*p.numel()),
        isfinite=lambda p: Vector([math.isfinite(v) for v in p.values]),
        allclose=lambda a,b,rtol,atol: all(abs(x-y) <= atol+rtol*abs(y)
            for x,y in zip(a.values,b.values,strict=True)))


def original_functions(names, **namespace):
    raw = (HERE/'train_siglip2_identity_diversity.py').read_bytes()
    tree = ast.parse(raw)
    selected = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == set(names)
    pins = {'loss_denominators':'ce86de123238eee428db8c62ff8ed4c09c2cde5ad6fbe52a85139623802a78e5',
        'json_sha256':'e6afa190880b91cfdb1e70946e6618bc5792b89e4b5e2bb8d72e8e549b865d98',
        'ranking_bank':'bc2b9bb03c0e4437e68e2b3e67b34dd0a8bbef6b7c2fbeee7ee613197e14905c',
        'ranking_membership':'c56dd67441cc81a47e8fcfb71032e4c751b7d5b70f2b6df19ca6236b9301813f',
        'smooth_ap_terms':'a90d81810d4295691ff1829215a07b9ab1793cd040ad92530d6bffaefa1eb698',
        'ranking_gallery':'80a6ab2cf778da8b8f6c78ccb183cd877085c4b62d1ac83655aa6f86ba69e3bd',
        'loss_terms':'d9cdceea0ac03e072e6a33f86053823a786e12a3cf180e645eebe002edcffc26'}
    assert all(hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()==pins[n.name] for n in selected)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<original complete loss>', 'exec'),namespace)
    return SimpleNamespace(**namespace)


class Dual:
    def __init__(self, value, derivative=(0.,)*6, reach=()):
        self.value,self.derivative,self.reach = float(value),tuple(derivative),frozenset(reach)
    @staticmethod
    def of(value): return value if isinstance(value,Dual) else Dual(value)
    def __add__(self, other):
        b = Dual.of(other)
        return Dual(self.value+b.value,[a+c for a,c in zip(self.derivative,b.derivative)],self.reach|b.reach)
    __radd__ = __add__
    def __neg__(self): return Dual(-self.value,[-d for d in self.derivative],self.reach)
    def __sub__(self, other): return self+-Dual.of(other)
    def __rsub__(self, other): return Dual.of(other)+-self
    def __mul__(self, other):
        b = Dual.of(other)
        return Dual(self.value*b.value,[a*b.value+c*self.value for a,c in zip(self.derivative,b.derivative)],self.reach|b.reach)
    __rmul__ = __mul__
    def __truediv__(self, other):
        b = Dual.of(other)
        return self*Dual(1/b.value,[-d/(b.value*b.value) for d in b.derivative],b.reach)
    def sigmoid(self):
        value = 1/(1+math.exp(-self.value)) if self.value >= 0 else math.exp(self.value)/(1+math.exp(self.value))
        return Dual(value,[value*(1-value)*d for d in self.derivative],self.reach)
    def sqrt(self):
        value = math.sqrt(self.value)
        return Dual(value,[d/(2*value) for d in self.derivative],self.reach)


class Tensor:
    """Small dual/autograd standin; executes extracted original Torch expressions."""
    dtype = 'float32'
    grad = None
    device = SimpleNamespace(type='cuda')
    def __init__(self, data, shape=None):
        self.data = [Dual.of(x) for x in data]
        self.shape = tuple(shape if shape is not None else (len(self.data),))
        assert math.prod(self.shape)==len(self.data)
    @property
    def requires_grad(self): return any(x.reach for x in self.data)
    @property
    def grad_fn(self): return 'dual' if self.requires_grad else None
    def detach(self): return Tensor([x.value for x in self.data],self.shape)
    def to(self, *args, **kwargs): return self
    def float(self): return self
    def numel(self): return len(self.data)
    def item(self):
        assert len(self.data)==1
        return self.data[0].value
    def __float__(self): return self.item()
    def __getitem__(self, index):
        if isinstance(index,tuple):
            assert len(self.shape)==1 and len(index)==2 and None in index
            return Tensor(self.data,(1,len(self.data)) if index[0] is None else (len(self.data),1))
        if isinstance(index,Tensor): index = [int(x.value) for x in index.data]
        if isinstance(index,int):
            if len(self.shape)==1: return Tensor([self.data[index]],())
            n=self.shape[1]; return Tensor(self.data[index*n:(index+1)*n],(n,))
        indices = list(range(self.shape[0]))[index] if isinstance(index,slice) else index
        if len(self.shape)==1: return Tensor([self.data[i] for i in indices])
        n=self.shape[1]; return Tensor([x for i in indices for x in self.data[i*n:(i+1)*n]],(len(indices),n))
    def binary(self, other, operation):
        other = other if isinstance(other,Tensor) else Tensor([other],())
        rank = max(len(self.shape),len(other.shape))
        a=(1,)*(rank-len(self.shape))+self.shape; b=(1,)*(rank-len(other.shape))+other.shape
        shape=tuple(max(x,y) for x,y in zip(a,b))
        assert all(x==y or x==1 or y==1 for x,y in zip(a,b))
        result=[]
        for i in range(math.prod(shape)):
            coordinates=[(i//math.prod(shape[j+1:]))%shape[j] for j in range(rank)]
            position=lambda dims:sum((c if n>1 else 0)*math.prod(dims[j+1:]) for j,(c,n) in enumerate(zip(coordinates,dims)))
            result.append(operation(self.data[position(a)],other.data[position(b)]))
        return Tensor(result,shape)
    def __add__(self,b): return self.binary(b,lambda a,b:a+b)
    __radd__=__add__
    def __sub__(self,b): return self.binary(b,lambda a,b:a-b)
    def __rsub__(self,b): return self.binary(b,lambda a,b:b-a)
    def __mul__(self,b): return self.binary(b,lambda a,b:a*b)
    __rmul__=__mul__
    def __truediv__(self,b): return self.binary(b,lambda a,b:a/b)
    def __ne__(self,b): return self.binary(b,lambda a,b:a.value!=b.value)
    def __gt__(self,b): return self.binary(b,lambda a,b:a.value>b.value)
    def __matmul__(self,b):
        rows,n=self.shape; columns=b.shape[1]
        assert n==b.shape[0]
        return Tensor([sum((self.data[i*n+k]*b.data[k*columns+j] for k in range(n)),Dual(0))
                       for i in range(rows) for j in range(columns)],(rows,columns))
    @property
    def T(self):
        rows,columns=self.shape
        return Tensor([self.data[i*columns+j] for j in range(columns) for i in range(rows)],(columns,rows))
    def square(self): return self*self
    def sum(self, dim=None):
        if dim is None:return Tensor([sum(self.data,Dual(0))],())
        assert dim==1 and len(self.shape)==2
        rows,n=self.shape
        return Tensor([sum(self.data[i*n:(i+1)*n],Dual(0)) for i in range(rows)])
    def mean(self): return self.sum()/len(self.data)
    def sigmoid(self): return Tensor([x.sigmoid() for x in self.data],self.shape)
    def norm(self, dim):
        return Tensor([x.sqrt() for x in self.square().sum(dim).data])
    def all(self): return Tensor([all(x.value for x in self.data)],())


def dual_torch():
    torch = ModuleType('torch')
    functional = ModuleType('torch.nn.functional')
    def normalize(tensor,dim):
        assert dim==1
        norms=tensor.norm(dim)
        return tensor/Tensor(norms.data,(tensor.shape[0],1))
    functional.normalize = normalize
    nn = ModuleType('torch.nn'); nn.functional = functional
    torch.nn,torch.float32 = nn,'float32'
    torch.tensor = lambda values,device:Tensor(values)
    torch.autocast = lambda *a,**kw:nullcontext()
    torch.isfinite = lambda t: Vector([math.isfinite(x.value) for x in t.data]) if isinstance(t,Tensor) else fake_vectors().isfinite(t)
    def grad(term,members,retain_graph,allow_unused):
        assert allow_unused
        scalar=term.data[0]
        return tuple(Vector([scalar.derivative[i]]) if i in scalar.reach else None for i in range(len(members)))
    torch.autograd=SimpleNamespace(grad=grad)
    return torch,{'torch':torch,'torch.nn':nn,'torch.nn.functional':functional}


def dual_fixture(d, *, count=64):
    torch,modules=dual_torch()
    parameters=[Dual(.05*(i+1),[float(i==j) for j in range(6)],(i,)) for i in range(6)]
    members=tuple(Vector([p.value]) for p in parameters)
    features=Tensor([v for i in range(count) for v in (1.+(i%7)*.07,.7+(i%5)*.13)],(count,2))
    targets=[i//2 for i in range(count)]
    if count==64: targets[-4:] = [30,30,30,31]
    trainer=original_functions(('ranking_bank','json_sha256','ranking_membership','loss_denominators',
            'smooth_ap_terms','ranking_gallery','loss_terms'),require=d.require,hashlib=hashlib,json=json,ARMS=('control','candidate'))
    def readout(context,state,values): return values*Tensor([parameters[4]+2,parameters[5]+1])
    trainer.loss_terms.__globals__['raw_features']=readout
    bank=trainer.ranking_bank(targets,list(range(count)))
    state={'arm':'candidate','device':'cuda','ranking_bank':bank,'target':Tensor(targets),
           'views':{'canonical':features},'teachers':{'T':features,'P':Tensor([.4,.6]*(max(targets)+1),(max(targets)+1,2)),'e0':2.}}
    def raw(anchors):
        x=features[anchors]
        varying=Tensor([parameters[0]+parameters[1]*.3+1,parameters[2]+parameters[3]*.2+1])
        return readout({},state,x*varying)
    return torch,modules,trainer,state,members,raw


class DecompositionTests(unittest.TestCase):
    def driver(self):
        self.assertTrue(DRIVER.is_file(), 'missing Stage1 gradient decomposition')
        spec = importlib.util.spec_from_file_location('_gradient_source_test', DRIVER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_source_exists(self):
        self.assertTrue(DRIVER.is_file(), 'missing Stage1 gradient decomposition')

    def test_parallel_opposed_orthogonal(self):
        d, torch = self.driver(), fake_vectors()
        for regression, ranking, cosine in (([1.,2.],[2.,4.],1.),
                ([1.,2.],[-2.,-4.],-1.), ([1.,0.],[0.,1.],0.),
                ([0.,0.],[1.,0.],None)):
            r,k = Vector(regression),Vector(ranking)
            result = d.vector_summary(torch, {'regression':(r,)*6,'ranking':(k,)*6,'total':(r+k,)*6}, chunk=1)
            self.assertEqual(result['all6']['regression_ranking_cosine'], cosine)
            self.assertAlmostEqual(result['all6']['regression_norm'], math.sqrt(6*sum(x*x for x in regression)))
            self.assertAlmostEqual(result['all6']['regression_ranking_dot'], 6*sum(x*y for x,y in zip(regression,ranking)))
            self.assertEqual(set(result), {*d.NAMES,'fc1','fc2','allfour','AC','all6'})
        bad = {'regression':(Vector([1]),)*6,'ranking':(Vector([1]),)*6,'total':(Vector([1]),)*6}
        with self.assertRaisesRegex(ValueError, 'correspondence'): d.vector_summary(torch,bad)
        for term in d.TERMS:
            mutant = dict(bad); mutant.pop(term)
            with self.assertRaisesRegex(ValueError, 'terms'): d.vector_summary(torch,mutant)

    def test_accumulation_missing_and_nonfinite(self):
        d, torch = self.driver(), fake_vectors()
        members = tuple(Vector([1.,2.]) for _ in range(6))
        accumulator = [torch.zeros_like(p) for p in members]
        d.accumulate_gradients(torch,accumulator,tuple(Vector([2.,3.]) for _ in members),members)
        d.accumulate_gradients(torch,accumulator,tuple(Vector([-1.,4.]) for _ in members),members)
        self.assertEqual([v.values for v in accumulator], [[1.,7.]]*6)
        for values in ((None,)*6, (Vector([math.nan,1.]),)*6, (Vector([1.]),)*6):
            with self.assertRaisesRegex(ValueError, 'gradient'): d.accumulate_gradients(torch,accumulator,values,members)

    def test_original_schedule_and_full_denominators(self):
        d = self.driver()
        self.assertTrue(hasattr(d,'first_batch'), 'missing original B64 contract')
        trainer = original_functions(('ranking_bank','json_sha256','ranking_membership','loss_denominators'),
            require=d.require, hashlib=hashlib,json=json)
        root = HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
        receipt = json.loads((root/'identity-diversity-v1/cpu-v5/receipt.json').read_text())
        class Batch(list):
            def tolist(self): return list(self)
        for seed,valid in ((179061,63),(179069,64)):
            q = next(q for q in receipt['qualifications'] if q['arm']=='control' and q['seed']==seed)
            expected = q['scope_schedule'][0]
            state = {'seed':seed,'schedules':{str(seed):[Batch(expected)]},'ranking_bank':q['ranking_bank']}
            batch,full = d.first_batch(trainer,state,expected)
            self.assertEqual(batch,expected)
            self.assertEqual(full['valid'],valid)
            self.assertEqual(trainer.loss_denominators(valid),(128,2*valid))
            for bad in (expected[:16],expected[:-1]+[expected[0]],list(reversed(expected))):
                state['schedules'][str(seed)][0] = Batch(bad)
                with self.assertRaisesRegex(ValueError,'B64'): d.first_batch(trainer,state,expected)
            state['schedules'][str(seed)][0] = Batch(expected)
            state['ranking_bank']['sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError,'membership'): d.first_batch(trainer,state,expected)

    def test_strict_file_unit_and_uncached_reads(self):
        d = self.driver()
        self.assertTrue(hasattr(d,'read_json'), 'missing exact FILE/UNIT admission')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve()/'record.json'
            path.write_text('{"a":1}')
            fact = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            guards = {}
            self.assertEqual(d.read_json(fact,guards),{'a':1})
            path.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError,'bytes'): d.read_json(fact,guards)
            for text in ('{"a":1,"a":2}','{"x":NaN}'):
                path.write_text(text)
                fact['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError): d.read_json(fact,{})
            unit = {'receipt':fact,'log':fact,'unit':'fixed.service','invocation_id':'a'*32,
                    'service_seconds':1.,'native_peak_rss_kib':1,'both_locks_held':True}
            d.check_unit(unit)
            for key in unit:
                bad = dict(unit); bad.pop(key)
                with self.assertRaises(ValueError): d.check_unit(bad)
            for field,value in (('both_locks_held',1),('service_seconds',True),('native_peak_rss_kib',math.inf)):
                with self.assertRaises(ValueError): d.check_unit({**unit,field:value})
            with self.assertRaises(ValueError): d.file_fact({**fact,'future':True})
            with self.assertRaises(ValueError): d.file_fact({**fact,'sha256':'A'*64})

    def test_real_loss_and_micro_full_scale(self):
        d=self.driver()
        self.assertTrue(hasattr(d,'measure_terms'),'missing actual loss/gradient accumulation seam')
        torch,modules,trainer,state,members,raw=dual_fixture(d)
        with patch.dict(sys.modules,modules):
            full={term:[Vector([0.]) for _ in members] for term in d.TERMS}
            scalar=d.measure_terms(torch,trainer,{},state,raw(list(range(64))),list(range(64)),63,members,full,[])
            micro={term:[Vector([0.]) for _ in members] for term in d.TERMS}
            losses=[0.,0.]
            for offset in range(0,64,16):
                anchors=list(range(offset,offset+16))
                result=d.measure_terms(torch,trainer,{},state,raw(anchors),anchors,63,members,micro,[])
                losses[0]+=result['mse'];losses[1]+=result['rank']
            self.assertAlmostEqual(losses[0],scalar['mse'],places=12)
            self.assertAlmostEqual(losses[1],scalar['rank'],places=12)
            for term in d.TERMS:
                for a,b in zip(full[term],micro[term],strict=True): self.assertAlmostEqual(a.item(),b.item(),places=11)
            for r,k,t in zip(*(micro[n] for n in d.TERMS),strict=True):self.assertAlmostEqual(t.item(),r.item()+k.item(),places=11)
            for k in (64,16):
                scaled={term:[Vector([0.]) for _ in members] for term in d.TERMS}
                result=d.measure_terms(torch,trainer,{},state,raw(list(range(16))),list(range(16)),k,members,scaled,[])
                self.assertAlmostEqual(result['rank'],trainer.loss_terms({},state,raw(list(range(16))),list(range(16)),63)[1].item()*63/k)
            # Both views use the same denominators: dropping either loses half this fixed fixture's vector.
            both={term:[Vector([0.]) for _ in members] for term in d.TERMS}
            for _ in d.VIEWS:
                d.measure_terms(torch,trainer,{},state,raw(list(range(64))),list(range(64)),63,members,both,[])
            for term in d.TERMS:
                for a,b in zip(full[term],both[term]):self.assertAlmostEqual(2*a.item(),b.item())

    def test_full_gallery_ties_self_singleton_and_mutants(self):
        d=self.driver()
        self.assertTrue(hasattr(d,'measure_terms'),'missing actual loss/gradient accumulation seam')
        torch,modules,trainer,state,members,raw=dual_fixture(d,count=6355)
        # One positive and6353 tied impostors: historical sigmoid(0)=.5, not hard-rank ties.
        trainer.loss_terms.__globals__['raw_features']=lambda c,s,x:x*Tensor([Dual(1,reach=(4,)),Dual(1,reach=(5,))])
        state['views']['canonical']=Tensor([1.,1.]*6355,(6355,2))
        with patch.dict(sys.modules,modules):
            query=Tensor([Dual(1,reach=(0,1,2,3,4,5)),Dual(1,reach=(0,1,2,3,4,5))],(1,2))
            mse,rank,facts=trainer.loss_terms({},state,query,[0],63)
            self.assertEqual(facts['positive'],[[1]])
            self.assertEqual(facts['eligible_counts'],[6354])
            self.assertAlmostEqual(rank.item(),(1-1/(1+.5*6353))/126)
            _,singleton,facts=trainer.loss_terms({},state,query,[6354],63)
            self.assertEqual(facts['positive'],[[]]);self.assertEqual(singleton.item(),0.)
        # Original loss source runs with real dual gradients; detachment mutants cannot pass.
        torch,modules,trainer,state,members,raw=dual_fixture(d,count=8)
        namespace=trainer.loss_terms.__globals__
        original=ast.parse((HERE/'train_siglip2_identity_diversity.py').read_text())
        loss=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='loss_terms')
        text=ast.unparse(loss)
        anchors=list(range(8))
        with patch.dict(sys.modules,modules):
            baseline={t:[Vector([0.]) for _ in members] for t in d.TERMS}
            d.measure_terms(torch,trainer,{},state,raw(anchors),anchors,64,members,baseline,[])
            for before,after in (('(raw - target)','(raw.detach() - target)'),
                    ('F.normalize(raw, dim=1)','F.normalize(raw.detach(), dim=1)'),
                    ('ranking_gallery(context, state).T','ranking_gallery(context, state).detach().T')):
                self.assertIn(before,text)
                altered=dict(namespace)
                exec(compile(text.replace(before,after),'<loss mutant>','exec'),altered)
                mutant=SimpleNamespace(loss_terms=altered['loss_terms'])
                vectors={t:[Vector([0.]) for _ in members] for t in d.TERMS}
                try:d.measure_terms(torch,mutant,{},state,raw(anchors),anchors,64,members,vectors,[])
                except ValueError:continue
                self.assertTrue(any(abs(a.item()-b.item())>1e-8 for term in d.TERMS for a,b in zip(baseline[term],vectors[term])))

    def test_launch_exact_roles_and_omissions(self):
        d=self.driver()
        self.assertTrue(hasattr(d,'validate_launch'),'missing Stage1 launch contract')
        fact={'path':'/fixture/source.py','sha256':'a'*64}
        unit={'receipt':fact,'log':fact,'unit':'fixed','invocation_id':'a'*32,
              'service_seconds':1.,'native_peak_rss_kib':1,'both_locks_held':True}
        launch={'schema':'connected-gradient-decomposition-launch-v1','execution_sha256':'b'*64,
            'python':fact,'training':{'root':'/fixture/train','execution_sha256':d.TRAIN_EXECUTION,'code':d.TRAIN_CODE},
            'evaluator':{'root':'/fixture/eval','execution_sha256':'c'*64,
                'code':{'evaluate_siglip2_connected_mlp.py':'a'*64,'test_connected_mlp_evaluation.py':'a'*64}},
            'bootstrap':fact,'workspace_source':{**fact,'path':'/fixture/observe_connected_control_batch_execution.py'},
            'candidates':[{'seed':s,'launch':fact,'terminal':unit,
                'checkpoint':{**fact,'sha256':d.ACCEPTED[s][0]},'terminal_state_sha256':d.ACCEPTED[s][1]} for s in d.SEEDS],
            'output':'/fixture/output','resource_policy':dict(d.POLICY),'both_locks_held':True}
        args=SimpleNamespace(execution_sha256='b'*64,output=Path('/fixture/output'))
        d.validate_launch(launch,args)
        for key in launch:
            mutant=copy.deepcopy(launch);mutant.pop(key)
            with self.assertRaises(ValueError):d.validate_launch(mutant,args)
        for key,value in (('candidates',launch['candidates'][:1]),('candidates',list(reversed(launch['candidates']))),
                ('both_locks_held',1),('resource_policy',{**d.POLICY,'seconds':True}),
                ('workspace_source',{**fact,'path':'/fixture/unknown_cleanup.py'})):
            with self.assertRaises(ValueError):d.validate_launch({**launch,key:value},args)
        for field in ('launch','terminal','checkpoint','terminal_state_sha256','seed'):
            mutant=copy.deepcopy(launch);mutant['candidates'][0].pop(field)
            with self.assertRaises(ValueError):d.validate_launch(mutant,args)
        mutant=copy.deepcopy(launch);mutant['candidates'][0]['checkpoint']['sha256']='0'*64
        with self.assertRaises(ValueError):d.validate_launch(mutant,args)
        mutant=copy.deepcopy(launch);mutant['training']['code']['extra.py']='a'*64
        with self.assertRaises(ValueError):d.validate_launch(mutant,args)

    def test_state_ownership_success_rejection_and_alias(self):
        d=self.driver()
        self.assertTrue(hasattr(d,'state_measurement'),'missing sequential state/cleanup owner')
        class Owned:pass
        events=[]
        context={'initial':{},'original_cpu_record':{},'trainer':SimpleNamespace(require_no_training=lambda c:events.append('no_training'))}
        live=[]
        def build(*a):
            owner=Owned();state={'owner':owner,'identity':{'seed':179061},'counter':0}
            live.append(weakref.ref(owner));return state
        def release(c,state): events.append('release');state.clear()
        connected=SimpleNamespace(integrity=lambda *a:events.append('integrity'),release=release,
            select_initializer=lambda *a:{'scope_schedule':[[]]})
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
        for reject in (False,True):
            events.clear()
            def measure(*a):
                events.append('view')
                if reject:raise ValueError('original measurement failure')
                return {'view':'stub'}
            with redirect_stderr(io.StringIO()),patch.object(d,'build_state',build),patch.object(d,'measure_state',measure),\
                 patch.object(d,'state_digest',lambda *a:'unchanged'),patch.object(d,'state_refs',lambda c,s:[weakref.ref(s['owner'])]):
                if reject:
                    with self.assertRaisesRegex(ValueError,'original measurement failure'):
                        d.state_measurement(torch,context,connected,179061,0,{},lambda:None)
                else:d.state_measurement(torch,context,connected,179061,0,{},lambda:None)
            self.assertEqual(events.count('release'),1)
            self.assertTrue(all(ref() is None for ref in live))
            self.assertIn('no_training',events)
        leaked=[]
        def keep_alias(*a):
            state=build();leaked.append(state['owner']);return state
        with redirect_stderr(io.StringIO()),patch.object(d,'build_state',keep_alias),patch.object(d,'measure_state',lambda *a:{}),\
             patch.object(d,'state_digest',lambda *a:'unchanged'),patch.object(d,'state_refs',lambda c,s:[weakref.ref(s['owner'])]):
            with self.assertRaisesRegex(ValueError,'state.*lifetime'):
                d.state_measurement(torch,context,connected,179061,0,{},lambda:None)
        leaked.clear()

    def test_cleanup_keeps_original_failure_and_runs_all(self):
        d=self.driver();events=[]
        original=ValueError('original')
        def bad():events.append('bad');raise ValueError('cleanup')
        with redirect_stderr(io.StringIO()),self.assertRaisesRegex(ValueError,'original') as caught:
            d.finish_cleanup(original,[bad,lambda:events.append('last')])
        self.assertIs(caught.exception,original)
        self.assertEqual(events,['bad','last'])
        self.assertTrue(any('cleanup' in note for note in original.__notes__))

    def test_bound_workspace_source_rejects_function_replacement(self):
        d=self.driver()
        self.assertTrue(hasattr(d,'bind_workspace'),'missing explicit source-pinned workspace interface')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp).resolve()/'observe_connected_control_batch_execution.py'
            path.write_text('def require(ok, message):\n if not ok: raise ValueError(message)\n'
                'def authenticated(fact, guards): pass\n'
                'def capture_workspace_owner(torch, context):\n'
                ' def dispose(): torch.events.append("dispose")\n return dispose\n')
            fact={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            owner=d.load_module('_test_gradient_workspace',fact,{})
            evaluator_path=HERE/'evaluate_siglip2_connected_mlp.py'
            evaluator=d.load_module('_test_gradient_evaluator_live_guard',
                {'path':str(evaluator_path),'sha256':hashlib.sha256(evaluator_path.read_bytes()).hexdigest()},{})
            context={'guards':{str(path):fact['sha256']},'legacy':{'selected':{'source_cpu':{}},'warm_record':{}},'events':[]}
            torch=SimpleNamespace(events=[])
            dispose=d.bind_workspace(evaluator,owner,fact,context,torch)
            dispose()
            self.assertEqual(torch.events,['dispose'])
            with self.assertRaisesRegex(ValueError,'already attempted'):dispose()
            # Actual helper's source guard runs; only workspace behavior is a stdlib standin.
            with patch.object(owner,'capture_workspace_owner',owner.capture_workspace_owner):
                original=owner.capture_workspace_owner
                # Constructor standin gets the same original role mapping as the native function.
                context['legacy']['selected']['source_cpu']={'origins':{'files':{}}}
                context['legacy']['warm_record']={'origins':{'files':{}}}
                with self.assertRaisesRegex(ValueError,'source|function|binding'):
                    owner.capture_workspace_owner=lambda *args:lambda:None
                    d.bind_workspace(evaluator,owner,fact,context,SimpleNamespace())
                owner.capture_workspace_owner=original
            sys.modules.pop(owner.__name__,None);sys.modules.pop(evaluator.__name__,None)

    def test_final_allocated_zero_and_cleanup_error(self):
        d=self.driver();events=[]
        self.assertTrue(hasattr(d,'final_resources'),'missing final zero-allocation/RNG/cgroup gate')
        memory={'bytes':32}
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:True,
            synchronize=lambda:events.append('sync'),memory_allocated=lambda:memory['bytes'],
            max_memory_allocated=lambda:64,get_rng_state_all=lambda:[2]),
            random=SimpleNamespace(get_rng_state=lambda:1),equal=lambda a,b:a==b)
        context={'legacy':{'source_driver':SimpleNamespace(numerical_flags=lambda:{'original':True})}}
        def resource_check(*a):events.append('cgroup');return {'path':'/same'}
        for leftover in (0,1):
            result={};memory['bytes']=32
            def dispose():events.append('dispose');memory['bytes']=leftover
            with redirect_stderr(io.StringIO()),patch.object(d,'check_resources',resource_check):
                if leftover:
                    with self.assertRaisesRegex(ValueError,'allocated bytes must be zero'):
                        d.final_resources(torch,context,{'path':'/same'},(1,[2]),{'original':True},dispose,result)
                else:d.final_resources(torch,context,{'path':'/same'},(1,[2]),{'original':True},dispose,result)
            self.assertIn('cgroup_after',result)
        def broken():raise ValueError('original disposer failure')
        with redirect_stderr(io.StringIO()),patch.object(d,'check_resources',resource_check):
            with self.assertRaisesRegex(ValueError,'original disposer failure'):
                d.final_resources(torch,context,{'path':'/same'},(1,[2]),{'original':True},broken,{})

    def test_source_contract_and_complete_math_correspondence(self):
        d=self.driver();tree=ast.parse(DRIVER.read_text())
        imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        top={a.name.split('.')[0] for n in imports for a in n.names}
        top|={n.module.split('.')[0] for n in imports if isinstance(n,ast.ImportFrom)}
        self.assertFalse(top&d.NATIVE)
        calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
        for forbidden in ('empty_cache','reset_peak_memory_stats','load_partial','backward','step','save','quality','fork'):
            self.assertFalse(any(c.split('.')[-1]==forbidden for c in calls),forbidden)
        self.assertFalse({'connected.update','trainer.update'}&set(calls))
        self.assertFalse(any('_cuda_clearCublas' in c for c in calls))
        self.assertEqual(d.FILES,{DRIVER.name,Path(__file__).name})
        self.assertEqual(d.POLICY['gradient_sets_max'],8)
        self.assertLess(7*d.VECTOR_BYTES+16*d.CHUNK*8,d.POLICY['extra_tensor_bytes'])
        evidence=HERE.parent/'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
        archived=json.loads((evidence/'connected-mlp-cpu-v6-freeze/execution.json').read_text())
        self.assertEqual(d.TRAIN_CODE,archived)
        for name,digest in archived.items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        witness=json.loads((evidence/'actual-objective-gradient-v2-freeze/authority.json').read_text())
        for name,digest in witness['files'].items():self.assertEqual(hashlib.sha256((HERE/name).read_bytes()).hexdigest(),digest)
        functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        view=ast.unparse(functions['measure_view'])
        self.assertIn('range(0, 64, 16)',view)
        self.assertIn('enabled=False',view)
        self.assertIn('F.normalize',view)
        self.assertIn('context[\'connected\'].raw_features',view)
        self.assertNotIn('features\'].detach()',view)
        terms=ast.unparse(functions['measure_terms'])
        self.assertEqual(terms.count('trainer.loss_terms('),1)
        self.assertIn('full_valid',terms)
        self.assertIn('torch.autograd.grad',terms)
        self.assertIn('mse + rank',terms)
        self.assertIn('accumulate_gradients',terms)
        loader=ast.unparse(functions['build_state'])
        for required in ('weights_only=True','mmap=True',"map_location='cpu'",'consumed=pages.consume',
                         'connected.check_payload','connected.restore','copy.deepcopy(disk[\'identity\'])'):
            self.assertIn(required,loader)
        run=ast.unparse(functions['run'])
        self.assertIn('for step in (0, 128)',run)
        self.assertIn('trainer.exit_rehash',run)
        self.assertIn('require_exact=True',run)
        self.assertIn('file_bytes',run)
        self.assertNotIn('ast.NodeTransformer',DRIVER.read_text())
        for seed in d.SEEDS:
            receipt=json.loads((evidence/f'connected-mlp-train-candidate-{seed}-v1/receipt.json').read_text())
            self.assertEqual((receipt['result']['checkpoint']['sha256'],receipt['result']['terminal_state_sha256']),d.ACCEPTED[seed])
        self.assertFalse(d.NATIVE&sys.modules.keys())

    def test_actual_joint_view_loop_and_scalar_norm(self):
        d=self.driver();torch=fake_vectors()
        parameters={n:Vector([1.]) for n in d.MLP}
        parameters.update({f'frozen{i}':Vector([0.]) for i in range(444)})
        members=tuple(parameters[n] for n in d.MLP)+(Vector([1.]),Vector([1.]))
        for p in members:p.requires_grad=True;p.element_size=lambda:4
        trainer=SimpleNamespace()
        state={'model':SimpleNamespace(named_parameters=lambda:parameters.items()),'A':members[4],'C':members[5],
            'views':{'canonical':SimpleNamespace(shape=(6355,1152))},'teachers':{'T':SimpleNamespace(shape=(6355,128))},
            'ranking_bank':{'target':[0]*6355}}
        def view(torch,context,connected,state,batch,full,members,joint,name):
            for term,value in (('regression',1.),('ranking',2.),('total',3.)):
                for member in joint[term]:member.add_(Vector([value]))
            return {'view':name,'mse':.2,'rank':.3}
        with patch.object(d,'VECTOR_BYTES',24),patch.object(d,'first_batch',lambda *a:(list(range(64)),{'valid':63})),\
             patch.object(d,'measure_view',view):
            result=d.measure_state(torch,{'trainer':trainer},SimpleNamespace(),state,[])
        self.assertEqual([r['view'] for r in result['views']],['canonical','augmented'])
        self.assertAlmostEqual(result['both_views']['all6']['total_norm'],6*math.sqrt(6))
        self.assertAlmostEqual(result['both_views']['AC']['regression_ranking_dot'],16.)
        self.assertEqual(result['loss'],1.)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--source-only',action='store_true')
    parser.add_argument('--test',action='append')
    args=parser.parse_args()
    if args.test:
        suite=unittest.TestSuite(DecompositionTests(name) for name in args.test)
    else:
        assert args.source_only,'explicit --source-only final gate or --test narrow check required'
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(DecompositionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    assert not {'torch','numpy','transformers','PIL','safetensors','torchvision','sfora'}&sys.modules.keys()
    if result.wasSuccessful():
        print('PASS stdlib selected checks; native UNRUN; scientific HOLD unchanged.' if args.test else
              'PASS stdlib Stage1 exact source/authority, actual extracted loss and gradient accumulation, '
              'full-B64 scaling, complete gallery/self/ties/singletons, detachment/omission mutants, '
              'sequential ownership and final zero-allocation seams. Native UNRUN; scientific HOLD unchanged.')
    raise SystemExit(0 if result.wasSuccessful() else 1)
