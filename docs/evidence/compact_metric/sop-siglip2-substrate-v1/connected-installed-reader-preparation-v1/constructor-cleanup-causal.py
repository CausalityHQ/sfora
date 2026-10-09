import ast, gc, hashlib, json, sys, tempfile, types
from pathlib import Path
from unittest.mock import patch
raw=Path("src/sfora/connected_inference.py").read_text()
node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=="construct_encoder")
segment=ast.get_source_segment(raw,node)
primary=RuntimeError("primary load sentinel")
cleanup=ValueError("mapping cleanup sentinel")
class Model:
    def eval(self): return self
class Pages:
    def __init__(self, stream): pass
    def consume(self, value): pass
class Processor: pass
torch=types.ModuleType("torch")
transformers=types.ModuleType("transformers")
transformers.AutoImageProcessor=Processor
config={}
def load(*args, **kwargs):
    return {"vision":dict.fromkeys(range(448)),"buffers":{},"config":config,"runtime":{},"cpu_rng":None}
torch.load=load
def require(ok,message):
    if not ok: raise ValueError(message)
def fail_load(*args): raise primary
checks=[]
def map_check(path):
    checks.append(str(path))
    if mask: raise cleanup
ns={"require":require,"bound_file":lambda guards,path,sha:Path(path),"construct":lambda *args:Model(),"CheckpointPages":Pages,"fingerprint":lambda *args,**kwargs:"h","load_vision":fail_load,"gc":gc,"mapping_absent":map_check}
exec(compile(ast.Module(body=[node],type_ignores=[]),"source-extracted-construct_encoder.py","exec"),ns)
rows=[]
with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,{"torch":torch,"transformers":transformers}):
    file=Path(directory)/"opaque.pt";file.write_bytes(b"opaque")
    for mask in (False,True):
        try: ns["construct_encoder"]({"guards":{}},config,{},file,{"checkpoint":{"path":str(file),"sha256":"h"},"sha256":"h"})
        except BaseException as error:
            rows.append({"cleanup_failed":mask,"primary_identity_preserved":error is primary,"cleanup_identity_surfaced":error is cleanup,"original_primary_in_context":error.__context__ is primary})
            if not mask: assert error is primary
            else: assert error is cleanup and error.__context__ is primary
        else: raise AssertionError("failure became success")
assert len(checks)==2
result={"schema":"reader-constructor-cleanup-causal-v1","source_sha256":hashlib.sha256(raw.encode()).hexdigest(),"function_segment_sha256":hashlib.sha256(segment.encode()).hexdigest(),"rows":rows,"mapping_checks_attempted":len(checks),"source_only":True,"native_root_cause_claim":False,"native_modules_imported":[]}
print(json.dumps(result,indent=2,sort_keys=True))
