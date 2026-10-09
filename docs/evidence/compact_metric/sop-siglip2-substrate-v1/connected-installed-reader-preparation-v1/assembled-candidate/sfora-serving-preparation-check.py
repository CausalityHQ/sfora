"""Actual bridge snapshots and genuine helper modules; no native imports."""
import ast, hashlib, importlib.util, sys, tempfile, types
from pathlib import Path
root=Path('/home/rb/worktrees/sfora-positive-causality')
owned=[]
with tempfile.TemporaryDirectory() as tmp:
    folder=Path(tmp)
    def load(name,filename,raw):
        path=folder/filename;path.write_bytes(raw)
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;owned.append((name,module))
        exec(compile(raw,str(path),'exec',dont_inherit=True),vars(module))
        return module,path,hashlib.sha256(raw).hexdigest()
    try:
        package=types.ModuleType('sfora');package.__path__=[];sys.modules['sfora']=package;owned.append(('sfora',package))
        bridge,_,_=load('sfora.connected_compact_serving','connected_compact_serving.py',(root/'src/sfora/connected_compact_serving.py').read_bytes())
        raw=(root/'src/sfora/connected_inference.py').read_bytes()+b'\n'+Path(__file__).with_name('sfora-serving-metadata-prototype.py').read_bytes()+b'\ndef _serving_metadata(path, expected): return read_serving_metadata(path, expected, require)\n'+Path(__file__).with_name('sfora-installed-reader-core.py').read_bytes()
        runtime,runtime_path,runtime_sha=load('_draft_reader_runtime','connected_inference.py',raw)
        paths=[]
        for filename in ('_connected_inference_authority.py','packed_int8.py'):
            p=folder/filename;data=(root/'src/sfora'/filename).read_bytes();p.write_bytes(data);paths.append((str(p),hashlib.sha256(data).hexdigest()))
        ledger=ast.parse((folder/'_connected_inference_authority.py').read_text())
        historical=next(ast.literal_eval(n.value) for n in ledger.body if isinstance(n,ast.Assign) and n.targets[0].id=='HISTORICAL_CODE')
        runtime._bind_runtime(historical,((str(runtime_path),runtime_sha),*paths))
        private=types.ModuleType('_draft_helpers');private.__path__=[];sys.modules[private.__name__]=private;owned.append((private.__name__,private))
        helpers=[];guards=[];raws=[]
        for name in ('connected_gallery_provenance','connected_serving_artifact','connected_serving_admission','connected_artifact_identity','connected_installed_environment'):
            data=(root/'src/sfora'/(name+'.py')).read_bytes();module,path,sha=load(private.__name__+'.'+name,name+'.py',data)
            helpers.append(module);guards.append((str(path),sha));raws.append(data)
        owner=bridge.ConnectedCompactIndex();owner._module=runtime
        owner._owned={module.__name__:module for module in (runtime,private,*helpers)}
        owner._guards=tuple((Path(path),sha,False) for path,sha in ((str(runtime_path),runtime_sha),*paths,*guards))
        owner._snapshot(runtime,raw)
        owner._snapshot(bridge,(folder/'connected_compact_serving.py').read_bytes())
        for module,data in zip(helpers,raws,strict=True):owner._snapshot(module,data)
        binding=(tuple(helpers),tuple(guards),owner._check_current)
        assert runtime._serving_helper_binding(binding)==binding
        try: runtime._serving_helper_binding((tuple(helpers),tuple(guards),lambda:None))
        except ValueError:pass
        else:raise AssertionError('foreign callback accepted')
        exec(compile(Path(__file__).with_name('sfora-serving-prepare-fixture.py').read_bytes(), '/tmp/sfora-serving-prepare-fixture.py', 'exec'), globals())
        original_code = owner._check_current.__func__.__code__
        owner._check_current.__func__.__code__ = (lambda self: None).__code__
        try:
            try: runtime._serving_helper_binding(binding)
            except ValueError: pass
            else: raise AssertionError('checker code changed to no-op was accepted')
        finally:
            owner._check_current.__func__.__code__ = original_code
        fn = owner._check_current.__func__
        for attribute, changed in (('__defaults__', (None,)), ('__kwdefaults__', {'foreign': None})):
            saved = getattr(fn, attribute)
            setattr(fn, attribute, changed)
            try:
                try: runtime._serving_helper_binding(binding)
                except ValueError: pass
                else: raise AssertionError(attribute + ' mutation accepted')
            finally:
                setattr(fn, attribute, saved)
        rows = owner._callables
        owner._callables = rows + [next(row for row in rows if row[0] is fn)]
        try:
            try: runtime._serving_helper_binding(binding)
            except ValueError: pass
            else: raise AssertionError('duplicate checker snapshot accepted')
        finally:
            owner._callables = rows
        assert runtime._serving_helper_binding(binding) == binding
        helpers[0]._SCOPE['arm']='candidate'
        try: runtime._serving_helper_binding(binding)
        except ValueError:pass
        else:raise AssertionError('live helper literal mutation accepted')
        assert not any(n.split('.')[0] in {'torch','numpy','PIL','transformers'} for n in sys.modules)
        print('PASS: genuine binding; foreign callback, checker code/defaults/kwdefaults, duplicate snapshot, helper literal mutations reject; restored binding passes; native UNRUN')
    finally:
        for name,module in reversed(owned):
            if sys.modules.get(name) is module:del sys.modules[name]
