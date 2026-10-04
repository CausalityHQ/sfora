import hashlib,importlib.util,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
started=time.perf_counter();root=Path(__file__).resolve().parent
authority=json.loads((root/'import-only-authority-v1.json').read_text())
def bound(spec):
 p=Path(spec['path']);assert p.is_absolute() and p.resolve()==p and p.is_file()
 assert hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'];return p
for spec in authority['inputs'].values():bound(spec)
def load(name,spec):
 p=bound(spec);s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
e=load('compact_import_diagnostic',authority['inputs']['evaluator']);trainer=load('compact_import_trainer',authority['inputs']['trainer'])
record=json.loads(bound(authority['inputs']['cpu_receipt']).read_text());proof=json.loads(bound(authority['inputs']['source_proof']).read_text())
manifest=json.loads(bound(authority['inputs']['bundle']).read_text());guards={}
context={'trainer':trainer,'guards':guards,'required_guards':record['input_guards'],'training_context':{'legacy':{'selected':{'source_cpu':{'origins':proof['origins']}}}}}
endpoint={'bundle':authority['inputs']['bundle']}
trace=[];trace_enabled=[False]
def audit(event,args):
 if trace_enabled[0] and event in ('open','os.listdir','os.scandir') and args and isinstance(args[0],(str,bytes,os.PathLike)):
  assert len(trace)<50000,'bounded trace exceeded'
  trace.append({'event':event,'path':str(Path(os.fsdecode(args[0])).resolve()),'mode':str(args[1]) if len(args)>1 else None})
sys.addaudithook(audit)
def maps():
 return sorted({x.split(maxsplit=5)[5] for x in Path('/proc/self/maps').read_text().splitlines() if len(x.split(maxsplit=5))==6 and x.split(maxsplit=5)[5].startswith('/') and '.so' in x.split(maxsplit=5)[5]})
def modules():
 return {n:{'file':getattr(m,'__file__',None),'origin':getattr(getattr(m,'__spec__',None),'origin',None)} for n,m in tuple(sys.modules.items())}
# Match the actual export's admitted tensor/vision package prefix; no late
# Transformer/model/processor import is moved outside the tested boundary.
import torch,numpy,PIL,torchvision
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
before=maps();before_modules=modules();error=None
try:
 trace_enabled[0]=True
 with e.bundle_reads_only(context,endpoint):
  from transformers import SiglipVisionConfig,SiglipVisionModel,AutoImageProcessor
  from transformers.models.siglip.image_processing_siglip import SiglipImageProcessor
 trace_enabled[0]=False
except Exception as exc:
 trace_enabled[0]=False;error={'type':type(exc).__name__,'message':str(exc)}
after=maps();assert not torch.cuda.is_initialized()
assert time.perf_counter()-started<300
print(json.dumps({'schema':'compact-lazy-import-diagnostic-v1','pass':error is None,'error':error,'qualification_eligible':False,'state_reuse_eligible':False,'model_constructed':False,'images':0,'quality_read':False,'seconds':time.perf_counter()-started,'before_modules':before_modules,'after_modules':modules(),'native_before':before,'native_after':after,'new_native':sorted(set(after)-set(before)),'native_not_in_original_proof':sorted(set(after)-set(proof['origins']['native_files'])),'trace':trace,'runtime_guards':guards},sort_keys=True))
if error:raise SystemExit(1)
