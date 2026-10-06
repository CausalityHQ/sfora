"""Engineering-only fresh validation timing; no native imports, model or quality."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
started=time.perf_counter();authority=Path(sys.argv[1]);expected=sys.argv[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
require(not sys.flags.optimize and os.environ.get('CUDA_VISIBLE_DEVICES')=='','fixed CPU-only invocation')
require(sha(authority)==expected,'authority differs');a=json.loads(authority.read_text());root=authority.parent
for n,h in a['files'].items():require(sha(root/n)==h,'source differs')
require(sha(a['receipt']['path'])==a['receipt']['sha256'],'original export receipt differs')
r=json.loads(Path(a['receipt']['path']).read_text());require(r['phase']=='export' and r['integrity_pass'] and not r['quality_read'],'original accepted export required')
items=list(r['input_guards'].items());m=runpy.run_path(str(root/'train_siglip2_identity_diversity.py'));rows=[]
for round in range(3):
 for mode in (('serial','batch') if round%2==0 else ('batch','serial')):
  begin=time.perf_counter()
  if mode=='serial':
   for p,h in items:m['bound_file']({},p,h)
  else:m['batch_bound_files']({},items)
  row={'round':round,'mode':mode,'seconds':time.perf_counter()-begin,'files':len(items)};rows.append(row);print(json.dumps(row),flush=True)
# Real same-size mutation, with restored mtime, must fail both procedures.
probe=root/'mutation-probe.bin'
with probe.open('xb') as f:f.write(b'original')
stat=probe.stat();digest=sha(probe);probe.write_bytes(b'mutated!');os.utime(probe,ns=(stat.st_atime_ns,stat.st_mtime_ns))
for mode in ('serial','batch'):
 try:
  if mode=='serial':m['bound_file']({},probe,digest)
  else:m['batch_bound_files']({},[(str(probe),digest)])
 except ValueError:pass
 else:raise ValueError('mutation accepted '+mode)
probe.unlink()
for n,h in a['files'].items():require(sha(root/n)==h,'exit source differs')
require(sha(authority)==expected and sha(a['receipt']['path'])==a['receipt']['sha256'],'exit inputs differ')
result={'schema':'export-exit-scan-ab-v1','pass':True,'engineering_only':True,'model_fit_qualified':False,'quality_read':False,'state_reuse_eligible':False,'rows':rows,'mutation_rejected':True,'wall_seconds':time.perf_counter()-started,'invocation_id':os.environ['INVOCATION_ID']}
require(result['wall_seconds']<300,'prospective whole unit deadline')
with Path(a['output']).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
