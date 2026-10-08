"""Stdlib-only, <=15s/AS1GiB: request-driver control-driver packed-source joint-source."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
from types import FunctionType, ModuleType
from unittest.mock import patch

resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
signal.alarm(15)
assert __debug__ and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
denied = {'torch','numpy','sfora','PIL','transformers','torchvision','safetensors','ctypes','cupy','triton'}
assert not any(n.split('.')[0] in denied for n in sys.modules)
class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in denied: raise AssertionError('thirdparty execution denied: '+fullname)
sys.meta_path.insert(0,NoNative())

def fact(path):
    assert path.is_absolute() and path.resolve() == path and path.is_file() and not path.is_symlink()
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def load(name,path):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

rejections = 0
def reject(call,fragment):
    global rejections
    try: call()
    except ValueError as error:
        assert fragment in str(error), str(error)
        rejections += 1
    else: raise AssertionError('mutation admitted: '+fragment)

@contextmanager
def attribute(value,name,replacement):
    original = getattr(value,name)
    try:
        setattr(value,name,replacement)
        yield
    finally: setattr(value,name,original)

assert len(sys.argv) == 5
request_path,control_path,packed_path,joint_path = [Path(p) for p in sys.argv[1:]]
facts = {name:fact(path) for name,path in zip(('request','control','packed','joint'),
    (request_path,control_path,packed_path,joint_path),strict=True)}
driver = load('_deployed_packing_alias_driver',request_path)
nn = ModuleType('torch.nn'); nn.Module = object; nn.functional = ModuleType('torch.nn.functional')
stubs = {'sfora':ModuleType('sfora'),'torch':ModuleType('torch'),'numpy':ModuleType('numpy'),
    'torch.nn':nn,'torch.nn.functional':nn.functional}
with patch.dict(sys.modules,stubs):
    packed = load('sfora.packed_int8',packed_path)
    joint = load('sfora.joint_relational_compaction',joint_path)
    owner = driver.Source(packed,facts['packed'])
    reject(lambda:driver.Source(joint,facts['joint']),'live source code differs')
    compose = lambda:driver.Source(joint,facts['joint'],packed_source=owner)
    source = compose(); source.check()
    for path in (request_path,control_path):
        run = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,ast.FunctionDef) and n.name == 'run')
        assignments = sorted((n for n in ast.walk(run) if isinstance(n,ast.Assign) and
            isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('packed_source','packing_source')),
            key=lambda n:n.lineno)
        values = {'Source':driver.Source,'requests':driver,'packed_int8':packed,'joint_relational_compaction':joint,
            'sources':{'packed':facts['packed'],'packing':facts['joint']}}
        exec(compile(ast.Module(body=assignments,type_ignores=[]),str(path),'exec'),values)
        assert [n.targets[0].id for n in assignments] == ['packed_source','packing_source']
        values['packing_source'].check()
        guard = next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name == 'guard')
        loop = next(n for n in ast.walk(guard) if isinstance(n,ast.For) and
            isinstance(n.target,ast.Name) and n.target.id == 'source')
        order = [n.id for n in loop.iter.elts]
        assert order.index('packed_source') < order.index('packing_source')
    for fn in (joint._validate_basis,joint.RelationalLinearEncoder.forward,packed._unit_rows,
            packed.fixed_int8_unit_codes,packed.pack_int8_unit_embeddings,
            packed.PackedInt8Embeddings.from_bytes.__func__,packed.PackedInt8Embeddings.bytes_per_vector.fget):
        with attribute(fn,'__code__',(lambda *a:None).__code__):
            reject(source.check,'live source'); reject(compose,'live source')
    for name in ('_PACKED_INT8_ARTIFACT_MAGIC','_SHA256_BYTES','_unit_rows','PackedInt8Embeddings',
            'fixed_int8_unit_codes','pack_int8_unit_embeddings'):
        with attribute(joint,name,object()):
            reject(source.check,'live source'); reject(compose,'packing reexports')
    for module,name in ((joint,'math'),(packed,'torch')):
        with attribute(module,name,object()): reject(source.check,'live source')
    for value in (packed._unit_rows,packed.PackedInt8Embeddings):
        for name in ('__module__','__name__','__qualname__'):
            with attribute(value,name,'foreign'):
                reject(source.check,'live source'); reject(compose,'live source')
    fn = packed._unit_rows
    foreign = FunctionType(fn.__code__,dict(fn.__globals__),fn.__name__)
    foreign.__module__,foreign.__qualname__ = fn.__module__,fn.__qualname__
    with attribute(joint,'_unit_rows',foreign): reject(compose,'packing reexports')
    with attribute(packed,'_unit_rows',foreign):
        reject(source.check,'live source')
        reject(lambda:driver.Source(packed,facts['packed']),'live source')
    wrong = driver.Source(driver,facts['request'])
    for invalid in (object(),wrong):
        reject(lambda:driver.Source(joint,facts['joint'],packed_source=invalid),'packing source composition')
    with patch.object(driver,'read_file',wraps=driver.read_file) as reads:
        source.check(); source.check()
        assert [c.args[0]['path'] for c in reads.call_args_list] == [str(packed_path),str(joint_path)]*2
    source.check()
assert not any(n.split('.')[0] in denied for n in sys.modules)
print(json.dumps({'status':'PASS','python':sys.version,'facts':facts,'rejections':rejections,
    'native_execution':False,'both_driver_startup_and_guard_order':True,'AS_bytes':1024**3,'deadline_seconds':15},sort_keys=True))
