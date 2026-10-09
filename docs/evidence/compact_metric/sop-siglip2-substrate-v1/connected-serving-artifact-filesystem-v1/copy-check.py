import ast, hashlib, os, tempfile
from pathlib import Path
from types import SimpleNamespace
source=Path(__file__).with_name('frozen-source.py').read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='sorted')
loop=compile(ast.Module(body=[node],type_ignores=[]),'<actual assembly copy loop>','exec')
stamp_code=compile(ast.Module(body=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stamp')],type_ignores=[]),'<actual stamp>','exec')
import stat
for changed in (False,True):
    with tempfile.TemporaryDirectory() as root:
        root=Path(root);origin=root/'source';origin.write_bytes(b'opaque bytes')
        output=root/'out';output.mkdir()
        proxy=SimpleNamespace(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
        def read(fd,count):
            block=os.read(fd,count)
            if changed and block:
                origin.write_bytes(b'changed byte')
            return block
        proxy.read=read
        namespace={'os':proxy,'Path':Path,'hashlib':hashlib,'stat':stat,'facts':{'opaque':{'bytes':12,'sha256':hashlib.sha256(b'opaque bytes').hexdigest()}},'settings':{'sources':{'opaque':str(origin)}},'output':output}
        exec(stamp_code,namespace)
        try: exec(loop,namespace)
        except AssertionError:
            assert changed
        else:
            assert not changed and (output/'opaque').read_bytes()==b'opaque bytes'
print('actual streamed-copy loop: unchanged PASS; concurrent same-size mutation rejected')
