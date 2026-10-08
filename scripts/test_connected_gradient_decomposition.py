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


def consumed_page_inverse(raw):
    if b'release_consumed_pages' not in raw:return raw
    seams = (
        (b'*, keep=False, release_consumed_pages=False', b'*, keep=False'),
        (b"    require(type(release_consumed_pages) is bool, 'consumed-page release boolean required')\n"
         b"    require(not (release_consumed_pages and keep), 'consumed-page release requires streaming')\n", b''),
        (b"        if release_consumed_pages:\n"
         b"            page_size = os.sysconf('SC_PAGESIZE')\n"
         b"            require(type(page_size) is int and page_size > 0, 'native page size differs')\n"
         b"            fd,consumed,released = stream.fileno(),0,0\n", b''),
        (b"                if release_consumed_pages:\n"
         b"                    consumed += len(block)\n"
         b"                    completed = consumed//page_size*page_size\n"
         b"                    if completed > released:\n"
         b"                        os.posix_fadvise(fd,released,completed-released,os.POSIX_FADV_DONTNEED)\n"
         b"                        released = completed\n", b''),
        (b"file_bytes({'path':path,'sha256':digest},{},release_consumed_pages=True)",
         b"file_bytes({'path':path,'sha256':digest},{})"))
    for current,original in seams:
        if raw.count(current) != 1:
            raise ValueError('consumed-page exact source seam differs')
        raw = raw.replace(current,original,1)
    return raw


def consumed_page_test_inverse(raw):
    tree = ast.parse(raw)
    names = {'consumed_page_inverse','consumed_page_test_inverse','consumed_page_fixture',
        'test_consumed_page_stream_hash_before_aligned_advice',
        'test_consumed_page_short_reads_native_sizes_and_tails',
        'test_consumed_page_default_keep_and_strict_mode',
        'test_consumed_page_sysconf_and_io_failures_are_fatal',
        'test_consumed_page_original_identity_hash_and_guard_checks',
        'test_consumed_page_advice_failure_keeps_genuine_exit_cleanup',
        'test_consumed_page_inverse_preserves_whole_original_source_and_tests'}
    removed,entries = [],[]
    class Restore(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in names:
                removed.append(node.name)
                return None
            expected = {'memory_observer_inverse': 'ast.parse(consumed_page_inverse(raw))',
                'memory_observer_test_inverse': 'consumed_page_test_inverse(raw)',
                'test_memory_observer_inverse_keeps_complete_original_predicates':
                    'consumed_page_inverse(DRIVER.read_bytes())'}
            if node.name in expected:
                if ast.unparse(node.body[0].value) != expected[node.name]:
                    raise ValueError('consumed-page inverse entry seam differs')
                original = 'DRIVER.read_bytes()' if node.name.startswith('test_') else 'ast.parse(raw)'
                node.body[0].value = ast.parse(original,mode='eval').body
                entries.append(node.name)
            return self.generic_visit(node)
    tree = Restore().visit(tree)
    if sorted(removed) != sorted(names) or sorted(entries) != sorted((
            'memory_observer_inverse','memory_observer_test_inverse',
            'test_memory_observer_inverse_keeps_complete_original_predicates')):
        raise ValueError('consumed-page inverse exact test seams differ')
    return tree


def consumed_page_fixture(d, path, data, short_read=None):
    path.write_bytes(data)
    trace = SimpleNamespace(events=[],streams=[],read_bytes=0,hashed_bytes=0,error=None)
    opener,sha256 = Path.open,hashlib.sha256
    trace.fact = {'path':str(path),'sha256':sha256(data).hexdigest()}
    class Reader:
        def __init__(self, stream):self.stream = stream
        def __enter__(self):return self
        def __exit__(self, *args):self.stream.close()
        def fileno(self):return self.stream.fileno()
        def read(self, size):
            if trace.error == 'read':raise OSError('read failed')
            if trace.error == 'next_read' and trace.read_bytes:raise OSError('next_read failed')
            block = self.stream.read(min(size,short_read) if short_read else size)
            trace.read_bytes += len(block)
            trace.events.append(('read',size,len(block)))
            return block
    def open_file(owner, mode):
        stream = opener(owner,mode)
        if mode != 'rb':return stream
        trace.streams.append(stream)
        trace.fd = stream.fileno()
        return Reader(stream)
    class Digest:
        def __init__(self):self.digest = sha256()
        def update(self, block):
            if trace.error == 'update':raise OSError('hash update failed')
            self.digest.update(block)
            trace.hashed_bytes += len(block)
            trace.events.append(('hash',len(block)))
        def hexdigest(self):
            if trace.error == 'hexdigest':raise OSError('hash hexdigest failed')
            return self.digest.hexdigest()
    def advice(fd, offset, length, mode):
        trace.events.append(('advice',fd,offset,length,mode,trace.hashed_bytes,trace.read_bytes))
        if trace.error == 'advice':raise OSError('advice failed')
    trace.open,trace.sha256,trace.advice = open_file,Digest,advice
    return trace


def memory_observer_inverse(raw):
    tree = ast.parse(consumed_page_inverse(raw))
    pins = {'memory_text':'384b371934a232fa16e548a1354cae7d31a2518f54b0499286d2b45f6f75cb1d',
        'observe_memory':'d4eec123d5016a46741e1f22479a1798cd76543b5ddb86da08dc11f9044765ae',
        'raise_memory_cancellation':'1b7916f540b5e2c81e73e74c50ac76640e543d2db807337765d20553fa1ec7dc',
        'require_memory_observation':'48710d6891ebad33b9aea2b53a68d96802c21fc7cffb050ba57040b09e189ed7'}
    expected = {f"observe_memory(context, 'exit:{phase}:{edge}')" for phase in ('genuine','origin','union','closure') for edge in ('begin','end')}
    expected.update(f"observe_memory(context, f'state:{{seed}}:{{step}}:{phase}:{edge}')" for phase in ('build','measure','release') for edge in ('begin','end'))
    expected.update({"observe_memory(context, 'check_resources:before')","observe_memory(context, 'final_cleanup:begin')",
        "observe_memory(context, 'final_cleanup:end')","observe_memory(memory_context, 'prepare:begin')",
        "observe_memory(memory_context, 'prepare:end')"})
    removed,calls,assignments,cancellations = [],[],[],[]
    wraps,gates,receipt = 0,0,0
    def observation(node):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id == 'observe_memory':
            text = ast.unparse(node)
            if text not in expected:
                raise ValueError('memory observer call seam differs')
            calls.append(text)
            return True
        return False
    def cancellation(node):
        text = ast.unparse(node)
        if text in ('raise_memory_cancellation(context)','raise_memory_cancellation(memory_context)'):
            cancellations.append(text)
            return True
        return False
    class Restore(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in pins:
                if hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest() != pins[node.name]:
                    raise ValueError('memory observer definition seam differs')
                removed.append(node.name)
                return None
            return self.generic_visit(node)
        def visit_Try(self, node):
            nonlocal wraps
            if len(node.finalbody) == 1 and isinstance(node.finalbody[0],ast.Expr) and observation(node.finalbody[0].value):
                if node.handlers or node.orelse:
                    raise ValueError('memory observer wrapper seam differs')
                wraps += 1
                node.finalbody = []
                return self.generic_visit(node).body
            return self.generic_visit(node)
        def visit_Expr(self, node):
            nonlocal gates
            if observation(node.value) or cancellation(node.value):return None
            if ast.unparse(node) == 'require_memory_observation(context, resources)':
                gates += 1
                return None
            return self.generic_visit(node)
        def visit_Assign(self, node):
            text = ast.unparse(node)
            if text in ('memory_context = {}',"context['memory_observation'] = memory_context['memory_observation']"):
                assignments.append(text)
                return None
            return self.generic_visit(node)
        def visit_List(self, node):
            nonlocal gates
            kept = []
            for value in node.elts:
                if isinstance(value,ast.Lambda) and (observation(value.body) or cancellation(value.body)):continue
                if ast.unparse(value) == 'lambda: require_memory_observation(context, resources)':
                    gates += 1
                    continue
                kept.append(value)
            node.elts = kept
            return self.generic_visit(node)
        def visit_Dict(self, node):
            nonlocal receipt
            kept = []
            for key,value in zip(node.keys,node.values,strict=True):
                if isinstance(key,ast.Constant) and key.value == 'phase_memory_observation':
                    if ast.unparse(value) != "context['memory_observation']":
                        raise ValueError('memory observer receipt seam differs')
                    receipt += 1
                else:kept.append((key,value))
            node.keys = [key for key,value in kept];node.values = [value for key,value in kept]
            return self.generic_visit(node)
    tree = Restore().visit(tree)
    if (sorted(removed) != sorted(pins) or sorted(calls) != sorted(expected) or wraps != 8 or
            gates != 2 or receipt != 1 or len(assignments) != 2 or len(set(assignments)) != 2 or
            cancellations.count('raise_memory_cancellation(context)') != 8 or
            cancellations.count('raise_memory_cancellation(memory_context)') != 2):
        raise ValueError('memory observer exact seam counts differ')
    return tree


def memory_observer_test_inverse(raw):
    tree = consumed_page_test_inverse(raw)
    names = {'memory_observer_inverse','memory_observer_test_inverse','memory_fixture',
        'test_memory_observer_records_pressure_before_original_guard_rejects',
        'test_memory_observer_bounds_and_failures_keep_original_cleanup',
        'test_memory_observer_build_failure_keeps_release_sync_and_guards',
        'test_memory_observer_cancellation_preserves_release_and_stops_measurement',
        'test_memory_observer_cancellation_preserves_final_cleanup_and_primary',
        'test_memory_observer_complete_receipt_requires_every_phase',
        'test_memory_observer_inverse_keeps_complete_original_predicates'}
    removed,restored = [],[]
    class Restore(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in names:
                removed.append(node.name)
                return None
            if node.name in ('gradient_workspace_inverse','gradient_workspace_test_inverse'):
                expected = 'memory_observer_inverse(raw)' if node.name == 'gradient_workspace_inverse' else 'memory_observer_test_inverse(raw)'
                if ast.unparse(node.body[0].value) != expected:
                    raise ValueError('memory observer test entry seam differs')
                node.body[0].value = ast.parse('ast.parse(raw)',mode='eval').body
                restored.append(node.name)
            return self.generic_visit(node)
    tree = Restore().visit(tree)
    if sorted(removed) != sorted(names) or sorted(restored) != ['gradient_workspace_inverse','gradient_workspace_test_inverse']:
        raise ValueError('memory observer test seam counts differ')
    return tree


def gradient_workspace_inverse(raw):
    tree = memory_observer_inverse(raw)
    current = ("workspace_source exposes capture_workspace_owner(torch,{guards,source_cpu,warm},\n"
               "single_forward_witness=False): zero allocation after one clear, without the\n"
               "batch default's exact32MiB witness. Both records retain their original genuine\n"
               "CPU/warm roles. No direct private\n")
    original = ("workspace_source exposes the\n"
                "unchanged capture_workspace_owner(torch,{guards,source_cpu,warm}) interface;\n"
                "both records retain their original genuine CPU/warm roles. No direct private\n")
    doc = tree.body[0].value if isinstance(tree.body[0],ast.Expr) else None
    if not isinstance(doc,ast.Constant) or doc.value.count(current) != 1:
        raise ValueError('gradient workspace documentation seam differs')
    doc.value = doc.value.replace(current,original,1)
    count = 0
    class Restore(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal count
            if ast.unparse(node) == 'workspace.capture_workspace_owner(torch, roles, single_forward_witness=False)':
                count += 1
                node.keywords = []
            return self.generic_visit(node)
    Restore().visit(tree)
    if count != 1:
        raise ValueError('gradient workspace call seam count differs')
    return tree


def gradient_workspace_test_inverse(raw):
    tree = memory_observer_test_inverse(raw)
    names = {'gradient_workspace_inverse','gradient_workspace_test_inverse',
        'test_workspace_contract_selects_gradient_and_keeps_callback_guards',
        'test_workspace_contract_failure_runs_original_rng_and_cgroup',
        'test_workspace_contract_inverse_preserves_whole_gradient_and_tests'}
    removed = []
    count = 0
    class Restore(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name in names:
                removed.append(node.name)
                return None
            return self.generic_visit(node)
        def visit_Constant(self, node):
            nonlocal count
            current = 'def capture_workspace_owner(torch, context, *, single_forward_witness):\n'
            if isinstance(node.value,str) and current in node.value:
                if node.value.count(current) != 1:
                    raise ValueError('gradient workspace fixture seam differs')
                count += 1
                node.value = node.value.replace(current,'def capture_workspace_owner(torch, context):\n',1)
            return node
    tree = Restore().visit(tree)
    if sorted(removed) != sorted(names) or count != 1:
        raise ValueError('gradient workspace test seam counts differ')
    return tree


def memory_fixture(tmp):
    root = Path(tmp)/'sys/fixture.service'
    root.mkdir(parents=True)
    proc = Path(tmp)/'cgroup'; proc.write_text('0::/fixture.service\n')
    (root/'memory.current').write_text('1000\n')
    (root/'memory.peak').write_text('2000\n')
    (root/'memory.events').write_text('low 0\nhigh 0\nmax 17588\noom 0\noom_kill 0\noom_group_kill 0\n')
    fields = ('anon','file','file_mapped','kernel','slab','slab_reclaimable','active_file','inactive_file',
              'pgscan_direct','pgsteal_direct','workingset_refault_file')
    (root/'memory.stat').write_text(''.join(f'{name} {value}\n' for value,name in enumerate(fields,11)))
    def paths(value):
        if value == '/proc/self/cgroup':return proc
        if value == '/sys/fs/cgroup':return root.parent
        return Path(value)
    return root,proc,paths


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

    def test_consumed_page_stream_hash_before_aligned_advice(self):
        d = self.driver();page = d.os.sysconf('SC_PAGESIZE')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stream.bin';data = b'abc123'*(2*1024**2//6+1)+b'tail'
            fact = {'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}
            for _ in range(2):
                trace = consumed_page_fixture(d,path,data);guards = {}
                with patch.object(Path,'open',trace.open),patch.object(d.hashlib,'sha256',trace.sha256),\
                     patch.object(d.os,'posix_fadvise',trace.advice):
                    self.assertEqual(d.file_bytes(fact,guards,release_consumed_pages=True),path)
                self.assertEqual(guards,{str(path):fact['sha256']})
                self.assertEqual(trace.read_bytes,len(data));self.assertEqual(trace.hashed_bytes,len(data))
                self.assertTrue(all(stream.closed for stream in trace.streams))
                released = 0
                for index,event in enumerate(trace.events):
                    if event[0] == 'read':self.assertEqual(event[1],1024**2)
                    if event[0] != 'advice':continue
                    _,fd,offset,length,mode,hashed,read = event
                    self.assertEqual(trace.events[index-1][0],'hash')
                    self.assertEqual((offset,length,mode),(released,read//page*page-released,d.os.POSIX_FADV_DONTNEED))
                    self.assertEqual(hashed,read);self.assertGreater(length,0)
                    self.assertEqual(offset%page,0);self.assertEqual(length%page,0)
                    self.assertEqual(fd,trace.fd)
                    released += length
                self.assertEqual(released,len(data)//page*page)
            path.write_bytes(b'z'*len(data))
            with patch.object(d.os,'posix_fadvise',trace.advice),self.assertRaisesRegex(ValueError,'bytes'):
                d.file_bytes(fact,{},release_consumed_pages=True)

    def test_consumed_page_short_reads_native_sizes_and_tails(self):
        d = self.driver()
        cases = ((4096,0,None,()),(4096,4095,None,()),(4096,4096,None,((0,4096),)),
                 (4096,3*4096+19,3001,((0,4096),(4096,4096),(8192,4096))),
                 (65536,2*65536+7,17001,((0,65536),(65536,65536))),
                 (8192,8192+1,4097,((0,8192),)),
                 (3*1024**2,3*1024**2+1,None,((0,3*1024**2),)))
        with tempfile.TemporaryDirectory() as tmp:
            for page,size,short,want in cases:
                with self.subTest(page=page,size=size,short=short):
                    path = Path(tmp)/'edge.bin';trace = consumed_page_fixture(d,path,b'x'*size,short)
                    def advice(fd,offset,length,mode):
                        self.assertEqual(d.os.fstat(fd).st_ino,path.stat().st_ino)
                        self.assertEqual(d.os.fstat(fd).st_dev,path.stat().st_dev)
                        trace.advice(fd,offset,length,mode)
                    with patch.object(Path,'open',trace.open),patch.object(d.hashlib,'sha256',trace.sha256),\
                         patch.object(d.os,'sysconf',return_value=page) as sysconf,patch.object(d.os,'posix_fadvise',advice):
                        self.assertEqual(d.file_bytes(trace.fact,{},release_consumed_pages=True),path)
                    sysconf.assert_called_once_with('SC_PAGESIZE')
                    self.assertEqual(tuple((e[2],e[3]) for e in trace.events if e[0]=='advice'),want)
                    self.assertEqual(trace.hashed_bytes,size);self.assertEqual(trace.read_bytes,size)
                    self.assertTrue(all(stream.closed for stream in trace.streams))

    def test_consumed_page_default_keep_and_strict_mode(self):
        d = self.driver()
        self.assertEqual(d.file_bytes.__kwdefaults__,{'keep':False,'release_consumed_pages':False})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'record.json';path.write_bytes(b'{"a":1}')
            fact = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            with patch.object(d.os,'posix_fadvise',side_effect=AssertionError('unexpected advice')),\
                 patch.object(d.os,'sysconf',side_effect=AssertionError('unexpected sysconf')):
                self.assertEqual(d.file_bytes(fact,{}),path)
                self.assertEqual(d.file_bytes(fact,{},release_consumed_pages=False),path)
                self.assertEqual(d.file_bytes(fact,{},keep=True),b'{"a":1}')
                self.assertEqual(d.read_json(fact,{}),{'a':1})
                for mode in (0,1,None,'yes',[],object()):
                    with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'boolean'):
                        d.file_bytes(fact,{},release_consumed_pages=mode)
                with self.assertRaisesRegex(ValueError,'streaming'):
                    d.file_bytes(fact,{},keep=True,release_consumed_pages=True)
                with self.assertRaises(TypeError):d.file_bytes(fact,{},False,True)

    def test_consumed_page_sysconf_and_io_failures_are_fatal(self):
        d = self.driver()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fail.bin'
            for page in (0,-1,True,4096.,'4096',None,OSError('sysconf failed')):
                with self.subTest(page=page):
                    trace = consumed_page_fixture(d,path,b'x'*8192)
                    options = {'side_effect':page} if isinstance(page,OSError) else {'return_value':page}
                    with patch.object(Path,'open',trace.open),patch.object(d.os,'sysconf',**options),\
                         patch.object(d.os,'posix_fadvise',trace.advice):
                        with self.assertRaises(OSError if isinstance(page,OSError) else ValueError):
                            d.file_bytes(trace.fact,{},release_consumed_pages=True)
                    self.assertEqual(trace.events,[]);self.assertTrue(trace.streams[0].closed)
            for error in ('read','next_read','sha256','update','hexdigest','advice','fstat','stat'):
                with self.subTest(error=error):
                    trace = consumed_page_fixture(d,path,b'x'*8192);trace.error = error;guards = {}
                    stat = Path.stat
                    def path_stat(owner, *args, **kwargs):
                        if owner == path and error == 'stat' and trace.streams and trace.streams[0].closed:
                            raise OSError('stat failed')
                        return stat(owner,*args,**kwargs)
                    with patch.object(Path,'open',trace.open),patch.object(Path,'stat',path_stat),\
                         patch.object(d.hashlib,'sha256',side_effect=OSError('sha256 failed') if error=='sha256' else trace.sha256),\
                         patch.object(d.os,'posix_fadvise',trace.advice),\
                         (patch.object(d.os,'fstat',side_effect=OSError('fstat failed')) if error=='fstat' else nullcontext()):
                        with self.assertRaisesRegex(OSError,error):d.file_bytes(trace.fact,guards,release_consumed_pages=True)
                    self.assertEqual(guards,{});self.assertTrue(all(s.closed for s in trace.streams))
                    if error == 'next_read':self.assertTrue(any(e[0]=='advice' for e in trace.events))
                    if error in ('read','sha256','update'):self.assertFalse(any(e[0]=='advice' for e in trace.events))

    def test_consumed_page_original_identity_hash_and_guard_checks(self):
        d = self.driver()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'current.bin'
            for mutation in ('hash','guard','rewrite','replace','fstat'):
                with self.subTest(mutation=mutation):
                    trace = consumed_page_fixture(d,path,b'x'*8192)
                    fact = dict(trace.fact);guards = {}
                    if mutation == 'hash':fact['sha256'] = '0'*64
                    if mutation == 'guard':guards[str(path)] = '0'*64
                    original_stat = path.stat()
                    def advice(fd,offset,length,mode):
                        trace.advice(fd,offset,length,mode)
                        if mutation == 'rewrite':
                            path.write_bytes(b'y'*8192)
                            d.os.utime(path,ns=(original_stat.st_atime_ns,original_stat.st_mtime_ns+1_000_000_000))
                        if mutation == 'replace':path.unlink();path.write_bytes(b'x'*8192)
                    changed = list(original_stat);changed[1] += 1
                    with patch.object(Path,'open',trace.open),patch.object(d.os,'posix_fadvise',advice),\
                         (patch.object(d.os,'fstat',return_value=d.os.stat_result(changed)) if mutation=='fstat' else nullcontext()):
                        with self.assertRaisesRegex(ValueError,'authority' if mutation=='guard' else 'bytes'):
                            d.file_bytes(fact,guards,release_consumed_pages=True)
                    self.assertTrue(trace.streams[0].closed)
                    if mutation != 'guard':self.assertEqual(guards,{})
            class BrokenGuard(dict):
                def setdefault(self, *args):raise OSError('guard failed')
            with patch.object(d.os,'posix_fadvise',trace.advice),self.assertRaisesRegex(OSError,'guard'):
                d.file_bytes(trace.fact,BrokenGuard(),release_consumed_pages=True)
            for bad in (path.parent/'absent',path.parent/'link'):
                if bad.name == 'link':bad.symlink_to(path)
                with self.assertRaisesRegex(ValueError,'regular'):
                    d.file_bytes({'path':str(bad),'sha256':trace.fact['sha256']},{},release_consumed_pages=True)

    def test_consumed_page_advice_failure_keeps_genuine_exit_cleanup(self):
        d = self.driver();events = []
        tree = ast.parse(DRIVER.read_bytes())
        exit_node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='exit_integrity')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'union.bin';trace = consumed_page_fixture(d,path,b'x'*8192);trace.error = 'advice'
            def reject():events.append('resource');raise ValueError('cgroup memory failure event')
            trainer = SimpleNamespace(require_no_training=lambda c:events.append('no_training'),
                exit_rehash=lambda c:events.append('genuine'),audit_origin_diagnostics=lambda *a,**k:events.append('origin'))
            legacy = {'original':SimpleNamespace(FlatAdmission=object),'source_driver':SimpleNamespace(cgroup_memory=reject)}
            context = {'guards':{str(path):trace.fact['sha256']},'legacy':legacy,
                'nearest':SimpleNamespace(native_source_api=lambda c:None)}
            def observe(c,phase):events.append(phase)
            namespace = {**d.__dict__,'trainer':trainer,'legacy':legacy,'context':context,'observe_memory':observe,
                'args':SimpleNamespace(execution_sha256='0'*64),'code':{},'read_json':lambda *a:{},
                'guard':lambda:events.append('closure_guard')}
            exec(compile(ast.Module(body=[exit_node],type_ignores=[]),str(DRIVER),'exec'),namespace)
            for primary in (None,RuntimeError('original measurement failure')):
                events.clear()
                with patch.object(Path,'open',trace.open),patch.object(d.os,'posix_fadvise',trace.advice),\
                     patch.object(d,'observe_memory',observe),redirect_stderr(io.StringIO()):
                    with self.assertRaises(OSError if primary is None else RuntimeError) as caught:
                        d.finish_cleanup(primary,[namespace['exit_integrity'],lambda:d.check_resources(SimpleNamespace(),context),
                            lambda:events.append('remaining_cleanup')])
                self.assertEqual(events,['no_training','exit:genuine:begin','genuine','exit:genuine:end',
                    'exit:origin:begin','origin','exit:origin:end','exit:union:begin','exit:union:end',
                    'check_resources:before','resource','remaining_cleanup'])
                self.assertTrue(any('cgroup memory failure event' in n for n in caught.exception.__notes__))
                if primary is not None:
                    self.assertIs(caught.exception,primary)
                    self.assertTrue(any('advice failed' in n for n in primary.__notes__))
                self.assertTrue(all(s.closed for s in trace.streams))

    def test_consumed_page_inverse_preserves_whole_original_source_and_tests(self):
        raw = DRIVER.read_bytes();original = consumed_page_inverse(raw)
        self.assertEqual(hashlib.sha256(ast.dump(ast.parse(original),include_attributes=False).encode()).hexdigest(),
                         'fcd77b68ebe1375c471f22fbbb2d65f0a15dbdf3b2dcdcd3f91ec3a717082b07')
        tests = consumed_page_test_inverse(Path(__file__).read_bytes())
        self.assertEqual(hashlib.sha256(ast.dump(tests,include_attributes=False).encode()).hexdigest(),
                         'a25957750731ecf708fb410c58fbdb672d58b4dd096d8c1bd8c64a913f42d539')
        for before,after in ((b'completed = consumed//page_size*page_size',b'completed = consumed'),
                (b'POSIX_FADV_DONTNEED',b'POSIX_FADV_NORMAL'),
                (b'release_consumed_pages=True',b'release_consumed_pages=False'),
                (b"os.sysconf('SC_PAGESIZE')",b"os.sysconf('SC_PHYS_PAGES')")):
            with self.assertRaisesRegex(ValueError,'seam'):consumed_page_inverse(raw.replace(before,after,1))
        for before,after in ((b'digest.update(block)',b'digest.update(b"x")'),
                (b'before == after == path.stat()',b'True'),
                (b"digest.hexdigest() == fact['sha256']",b'True'),
                (b"guards.setdefault(str(path),fact['sha256']) == fact['sha256']",b'True'),
                (b'trainer.exit_rehash(context)',b'trainer.require_no_training(context)'),
                (b'require_exact=True',b'require_exact=False'),(b'zero_events(cgroup)',b'zero_events({})')):
            restored = consumed_page_inverse(raw.replace(before,after,1))
            self.assertNotEqual(ast.dump(ast.parse(restored),include_attributes=False),
                                ast.dump(ast.parse(original),include_attributes=False))

    def test_memory_observer_records_pressure_before_original_guard_rejects(self):
        d = self.driver();events=[]
        with tempfile.TemporaryDirectory() as tmp:
            root,proc,paths = memory_fixture(tmp)
            def reject():events.append('original_guard');raise ValueError('cgroup memory failure event')
            context = {'legacy':{'source_driver':SimpleNamespace(cgroup_memory=reject)}}
            with patch.object(d,'Path',paths), patch.dict(d.os.environ,{'INVOCATION_ID':'a'*32}), redirect_stderr(stream := io.StringIO()):
                with self.assertRaisesRegex(ValueError,'cgroup memory failure event'):d.check_resources(SimpleNamespace(),context)
            record = json.loads(stream.getvalue())
            self.assertEqual(events,['original_guard'])
            self.assertEqual(record['cgroup_path'],str(root))
            self.assertEqual(record['memory_stat']['anon'],11)
            self.assertEqual(record['memory_stat']['file'],12)
            self.assertEqual(record['memory_events']['max'],17588)
            self.assertEqual(record['memory_current'],1000)
            self.assertEqual(record['memory_peak'],2000)
            self.assertEqual(record['invocation_id'],'a'*32)
            self.assertGreater(record['timestamp_ns'],0)
            self.assertEqual(record['phase'],'check_resources:before')

    def test_memory_observer_bounds_and_failures_keep_original_cleanup(self):
        d = self.driver()
        for mutant in ('missing','oversize','duplicate','negative','membership','invocation','limit'):
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as tmp:
                root,proc,paths = memory_fixture(tmp)
                if mutant == 'missing':(root/'memory.stat').unlink()
                if mutant == 'oversize':(root/'memory.stat').write_bytes(b'x'*16385)
                if mutant == 'duplicate':(root/'memory.events').write_text('max 0\nmax 1\n')
                if mutant == 'negative':(root/'memory.current').write_text('-1\n')
                if mutant == 'membership':proc.write_text('0::/../fixture.service\n')
                context,events = {},[]
                original = ValueError('original primary failure')
                with patch.object(d,'Path',paths), patch.dict(d.os.environ,{'INVOCATION_ID':'bad' if mutant == 'invocation' else 'a'*32}), redirect_stderr(stream := io.StringIO()):
                    if mutant == 'limit':
                        for _ in range(129):d.observe_memory(context,'check_resources:before')
                    else:d.observe_memory(context,'check_resources:before')
                    with self.assertRaisesRegex(ValueError,'original primary failure') as caught:
                        d.finish_cleanup(original,[lambda:d.observe_memory(context,'final_cleanup:end'),
                            lambda:events.append('remaining_cleanup'),lambda:d.require_memory_observation(context,{})])
                self.assertIs(caught.exception,original)
                self.assertEqual(events,['remaining_cleanup'])
                self.assertTrue(any('memory observation' in n for n in caught.exception.__notes__))
                records = [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith('{')]
                self.assertLessEqual(len(records),128)
                self.assertTrue(any(record['status']=='FAIL_UNACCEPTED' for record in records))
                self.assertTrue(context['memory_observation']['failed'])

    def test_memory_observer_build_failure_keeps_release_sync_and_guards(self):
        d = self.driver();events=[]
        original = ValueError('original build failure')
        def build(*args):events.append('build');raise original
        context = {'initial':{},'trainer':SimpleNamespace(require_no_training=lambda c:events.append('no_training'))}
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
        with tempfile.TemporaryDirectory() as tmp:
            root,proc,paths = memory_fixture(tmp);(root/'memory.stat').unlink()
            with patch.object(d,'Path',paths), patch.dict(d.os.environ,{'INVOCATION_ID':'a'*32}), \
                 patch.object(d,'build_state',build), redirect_stderr(stream := io.StringIO()):
                with self.assertRaisesRegex(ValueError,'original build failure') as caught:
                    d.state_measurement(torch,context,SimpleNamespace(),179061,0,{},lambda:events.append('guard'))
            self.assertIs(caught.exception,original)
            self.assertEqual(events,['guard','build','sync','no_training','guard'])
            records = [json.loads(line) for line in stream.getvalue().splitlines() if line.startswith('{')]
            self.assertEqual([r['phase'] for r in records],['state:179061:0:build:begin','state:179061:0:build:end',
                'state:179061:0:release:begin','state:179061:0:release:end'])

    def test_memory_observer_cancellation_preserves_release_and_stops_measurement(self):
        d = self.driver()
        class Owned:pass
        cases = [('release:begin',original) for original in (None,ValueError('original'),KeyboardInterrupt('original'))]
        cases += [('build:end',None),('measure:begin',None)]
        for stage,original in cases:
            for kind in (KeyboardInterrupt,SystemExit):
                for boundary in ('reader','sink'):
                    with self.subTest(stage=stage,original=original,kind=kind,boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                        root,proc,paths = memory_fixture(tmp)
                        events,refs,current = [],[],{}
                        context = {'initial':{},'original_cpu_record':{},
                            'trainer':SimpleNamespace(require_no_training=lambda c:events.append('no_training'))}
                        def build(*args):
                            owner = Owned();refs.append(weakref.ref(owner))
                            return {'owner':owner,'identity':{}}
                        def measure(*args):
                            events.append('measure')
                            if original is not None:raise original
                            return {}
                        def release(c,state):events.append('release');state.clear()
                        connected = SimpleNamespace(integrity=lambda *a:None,release=release,
                            select_initializer=lambda *a:{'scope_schedule':[[]]})
                        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:events.append('sync')))
                        real_observe,real_read = d.observe_memory,d.memory_text
                        def observe(c,phase):current['phase']=phase;real_observe(c,phase)
                        def read(path):
                            if boundary == 'reader' and current['phase'].endswith(stage):raise kind('observer cancellation')
                            return real_read(path)
                        def sink(*args,**kwargs):
                            if boundary == 'sink' and current['phase'].endswith(stage):raise kind('observer cancellation')
                        with patch.object(d,'Path',paths),patch.dict(d.os.environ,{'INVOCATION_ID':'a'*32}), \
                             patch.object(d,'observe_memory',observe),patch.object(d,'memory_text',read), \
                             patch.object(d,'print',sink,create=True),patch.object(d,'build_state',build), \
                             patch.object(d,'measure_state',measure),patch.object(d,'state_digest',lambda *a:'same'), \
                             patch.object(d,'state_refs',lambda c,s:[weakref.ref(s['owner'])]),redirect_stderr(io.StringIO()):
                            with self.assertRaises(BaseException) as caught:
                                d.state_measurement(torch,context,connected,179061,0,{},lambda:events.append('guard'))
                        if original is not None:self.assertIs(caught.exception,original)
                        else:
                            self.assertIs(type(caught.exception),kind)
                            self.assertEqual(str(caught.exception),'observer cancellation')
                        self.assertEqual(events.count('release'),1)
                        self.assertIn('sync',events);self.assertIn('no_training',events)
                        self.assertEqual(events.count('guard'),2)
                        self.assertEqual(events.count('measure'),int(stage=='release:begin'))
                        self.assertTrue(all(ref() is None for ref in refs),'observer retained model/frame')
                        json.dumps(context['memory_observation'],allow_nan=False)

    def test_memory_observer_cancellation_preserves_final_cleanup_and_primary(self):
        d = self.driver();tree=ast.parse(DRIVER.read_bytes())
        run = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        final = next(n for n in run.body if isinstance(n,ast.Try) and any(
            isinstance(v,ast.FunctionDef) and v.name=='exit_integrity' for v in n.finalbody)).finalbody
        seam = [n for n in final if isinstance(n,ast.Expr)]
        self.assertEqual(len(seam),2)
        for kind in (KeyboardInterrupt,SystemExit):
            for boundary in ('reader','sink'):
                for original in (None,ValueError('original'),KeyboardInterrupt('original')):
                    with self.subTest(kind=kind,boundary=boundary,original=original), tempfile.TemporaryDirectory() as tmp:
                        root,proc,paths = memory_fixture(tmp);events,context = [],{}
                        def read(path):raise kind('observer cancellation')
                        def sink(*a,**k):raise kind('observer cancellation')
                        namespace = {**vars(d),'context':context,'error':original,'resources':{},
                            'restore_rng':lambda:events.append('rng'),'torch':None,'before':None,'rng':None,
                            'flags':None,'dispose':None,'final_resources':lambda *a:events.append('final'),
                            'exit_integrity':lambda:events.append('exit'),'check_resources':lambda *a:events.append('resource')}
                        with patch.object(d,'Path',paths),patch.dict(d.os.environ,{'INVOCATION_ID':'a'*32}), \
                             patch.object(d,'memory_text',read if boundary=='reader' else d.memory_text), \
                             patch.object(d,'print',sink if boundary=='sink' else print,create=True),redirect_stderr(io.StringIO()):
                            with self.assertRaises(BaseException) as caught:
                                exec(compile(ast.Module(body=seam,type_ignores=[]),str(DRIVER),'exec'),namespace)
                        if original is not None:self.assertIs(caught.exception,original)
                        else:self.assertIs(type(caught.exception),kind)
                        self.assertEqual(events,['rng','final','exit','resource'])
                        json.dumps(context['memory_observation'],allow_nan=False)

    def test_memory_observer_complete_receipt_requires_every_phase(self):
        d = self.driver()
        phases = ['prepare:begin','prepare:end','check_resources:before','final_cleanup:begin','final_cleanup:end']
        phases += [f'state:{seed}:{step}:{phase}:{edge}' for seed in (179061,179069) for step in (0,128)
                   for phase in ('build','measure','release') for edge in ('begin','end')]
        phases += [f'exit:{phase}:{edge}' for phase in ('genuine','origin','union','closure') for edge in ('begin','end')]
        with tempfile.TemporaryDirectory() as tmp:
            root,proc,paths = memory_fixture(tmp)
            context = {};resources = {'cgroup_before':{'path':str(root)},'cgroup_after':{'path':str(root)}}
            with patch.object(d,'Path',paths), patch.dict(d.os.environ,{'INVOCATION_ID':'a'*32}), redirect_stderr(io.StringIO()):
                for phase in phases:d.observe_memory(context,phase)
            d.require_memory_observation(context,resources)
            self.assertEqual(context['memory_observation']['records'],37)
            for phase in phases:
                count = context['memory_observation']['phases'].pop(phase)
                with self.assertRaisesRegex(ValueError,'phases'):d.require_memory_observation(context,resources)
                context['memory_observation']['phases'][phase] = count
            resources['cgroup_after']['path'] = '/other'
            with self.assertRaisesRegex(ValueError,'cgroup'):d.require_memory_observation(context,resources)
            with self.assertRaisesRegex(ValueError,'memory observation'):d.require_memory_observation({},resources)

    def test_memory_observer_inverse_keeps_complete_original_predicates(self):
        raw = consumed_page_inverse(DRIVER.read_bytes())
        restored = gradient_workspace_inverse(raw)
        self.assertEqual(hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest(),
                         '5b0f91d6422804cf7ae39e31d5111dbae268d5556e203febc71444f744b59c81')
        original = gradient_workspace_test_inverse(Path(__file__).read_bytes())
        self.assertEqual(hashlib.sha256(ast.dump(original,include_attributes=False).encode()).hexdigest(),
                         '952451a53e8230642a1d3c0d8750b1eee2088b55bcb88582644f0ae1ffd311b5')
        for before,after in ((b"observe_memory(context,'check_resources:before')",b"observe_memory(context,'check_resources:after')"),
                (b"observation['attempts'] >= 128",b"observation['attempts'] >= 129"),
                (b"lambda:require_memory_observation(context,resources)",b"lambda:require_memory_observation(context,{})")):
            self.assertEqual(raw.count(before),1)
            with self.assertRaisesRegex(ValueError,'seam'):memory_observer_inverse(raw.replace(before,after,1))
        for before,after in ((b"zero_events(cgroup)",b"zero_events({})"),
                (b"resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= POLICY['host_bytes']",b'True'),
                (b'require_exact=True',b'require_exact=False'),
                (b"before == after == path.stat()",b'True'),
                (b"file_bytes({'path':path,'sha256':digest},{})",b"file_bytes({'path':path,'sha256':digest},context['guards'])")):
            self.assertIn(before,raw)
            changed = gradient_workspace_inverse(raw.replace(before,after,1))
            self.assertNotEqual(ast.dump(changed,include_attributes=False),ast.dump(restored,include_attributes=False))

    def test_workspace_contract_selects_gradient_and_keeps_callback_guards(self):
        d = self.driver()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'workspace.py'
            path.write_text('def require(ok, message):\n if not ok: raise ValueError(message)\n'
                'def authenticated(fact, guards): pass\n'
                'def capture_workspace_owner(torch, context, *, single_forward_witness):\n'
                ' require(single_forward_witness is False, "gradient selector differs")\n'
                ' require(context["source_cpu"] is torch.source_cpu and context["warm"] is torch.warm, "roles differ")\n'
                ' def dispose(): torch.events.append("clear")\n return dispose\n')
            fact = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            owner = d.load_module('_gradient_workspace_selector',fact,{})
            evaluator_path = HERE/'evaluate_siglip2_connected_mlp.py'
            evaluator = d.load_module('_gradient_workspace_selector_guard',
                {'path':str(evaluator_path),'sha256':hashlib.sha256(evaluator_path.read_bytes()).hexdigest()},{})
            source_cpu,warm = {},{}
            torch = SimpleNamespace(events=[],source_cpu=source_cpu,warm=warm)
            context = {'guards':{str(path):fact['sha256']},
                'legacy':{'selected':{'source_cpu':source_cpu},'warm_record':warm}}
            try:
                dispose = d.bind_workspace(evaluator,owner,fact,context,torch)
                dispose()
                self.assertEqual(torch.events,['clear'])
                with self.assertRaisesRegex(ValueError,'already attempted'):dispose()
                dispose = d.bind_workspace(evaluator,owner,fact,context,torch)
                callback = dict(zip(dispose.__code__.co_freevars,dispose.__closure__,strict=True))['callback'].cell_contents
                def changed():return torch
                callback.__code__ = changed.__code__
                with self.assertRaisesRegex(ValueError,'callback changed'):dispose()
                with self.assertRaisesRegex(ValueError,'already attempted'):dispose()
                self.assertEqual(torch.events,['clear'])
            finally:
                sys.modules.pop(owner.__name__,None);sys.modules.pop(evaluator.__name__,None)

    def test_workspace_contract_failure_runs_original_rng_and_cgroup(self):
        from test_connected_control_batch_execution import load, workspace_contract_runtime
        d = self.driver()
        owner = load(HERE/'observe_connected_control_batch_execution.py','_gradient_workspace_final_owner')
        try:
            for mutant in ('residual','clear','rng','cgroup'):
                with self.subTest(mutant=mutant), workspace_contract_runtime(owner) as f:
                    f.torch.random = SimpleNamespace(get_rng_state=lambda:(f.events.append('rng') or 1))
                    f.torch.cuda.get_rng_state_all = lambda:[2]
                    f.torch.equal = lambda a,b:a==b
                    context = {'legacy':{'source_driver':SimpleNamespace(numerical_flags=lambda:{'original':True})}}
                    failure = ValueError('clear failure after release')
                    if mutant == 'residual':f.memory['after'] = 1
                    else:f.memory['clear_error'] = failure
                    if mutant == 'rng':f.torch.random.get_rng_state = lambda:(f.events.append('rng') or 3)
                    def cgroup(*args):
                        f.events.append('cgroup')
                        if mutant == 'cgroup':raise ValueError('cgroup memory failure event')
                        return {'path':'/same'}
                    dispose = f.capture(f.torch,f.context,single_forward_witness=False)
                    with patch.object(d,'check_resources',cgroup), self.assertRaises(ValueError) as caught:
                        d.final_resources(f.torch,context,{'path':'/same'},(1,[2]),{'original':True},dispose,{})
                    self.assertEqual(f.events.count('clear'),1)
                    self.assertIn('cgroup',f.events)
                    if mutant == 'residual':
                        self.assertEqual(str(caught.exception),'workspace after scalars differ')
                        self.assertTrue(any('allocated bytes must be zero' in n for n in caught.exception.__notes__))
                    else:
                        self.assertIs(caught.exception,failure)
                        self.assertIn('rng',f.events)
                    if mutant in ('rng','cgroup'):
                        self.assertTrue(any(('RNG' if mutant == 'rng' else 'cgroup memory failure') in n
                                            for n in caught.exception.__notes__))
                    with self.assertRaisesRegex(ValueError,'already attempted'):dispose()
        finally:sys.modules.pop(owner.__name__,None)

    def test_workspace_contract_inverse_preserves_whole_gradient_and_tests(self):
        raw = DRIVER.read_bytes()
        restored = gradient_workspace_inverse(raw)
        self.assertEqual(hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest(),
                         '5b0f91d6422804cf7ae39e31d5111dbae268d5556e203febc71444f744b59c81')
        tests = gradient_workspace_test_inverse(Path(__file__).read_bytes())
        self.assertEqual(hashlib.sha256(ast.dump(tests,include_attributes=False).encode()).hexdigest(),
                         '952451a53e8230642a1d3c0d8750b1eee2088b55bcb88582644f0ae1ffd311b5')
        with self.assertRaisesRegex(ValueError,'call seam'):
            gradient_workspace_inverse(raw.replace(b'workspace.capture_workspace_owner(torch,roles,single_forward_witness=False)',
                b'workspace.capture_workspace_owner(torch,roles,single_forward_witness=True)',1))
        for before,after in ((b'        guard()',b'        guard(None)'),
                (b'require(not used,',b'require(True,'),
                (b'callback.__code__ is code',b'True'),
                (b'torch.cuda.memory_allocated() == 0',b'torch.cuda.memory_allocated() <= 1'),
                (b'trainer.exit_rehash(context)',b'trainer.require_no_training(context)')):
            self.assertIn(before,raw)
            changed = gradient_workspace_inverse(raw.replace(before,after,1))
            self.assertNotEqual(ast.dump(changed,include_attributes=False),ast.dump(restored,include_attributes=False))

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
                'def capture_workspace_owner(torch, context, *, single_forward_witness):\n'
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
