import hashlib,json,os,subprocess
from pathlib import Path
from parse_cutile_jit_timing import parse
root=Path('/home/riomus/runs/sfora-cutile-threads-v1')
hashes=json.loads((root/'binaries.json').read_text())
summary={'claim_eligible':False,'targets':[]}
for name,digest in hashes.items(): assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
for name in hashes:
    if name.endswith('.so'): continue
    result=subprocess.run([str(root/name),'--nocapture','--test-threads=1'],timeout=60)
    summary['targets'].append({'binary':name,'sha256':hashes[name],'exit':result.returncode})
    assert result.returncode==0
print('THREAD_GATE_BEGIN',flush=True)
env=os.environ.copy(); env['PYTHONPATH']='/home/riomus/sfora-siglip2-deployed-batch-v1/src'
with (root/'threads.log').open('x') as log:
    result=subprocess.run(['/home/riomus/group-learning/.venv/bin/python',str(root/'probe_cutile_threads.py'),'--library',str(root/'candidate.so'),'--output',str(root/'threads.json')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
assert result.returncode==0
report=json.loads((root/'threads.json').read_text())
assert report['library_sha256']==hashes['candidate.so']
assert report['source_sha256']==hashlib.sha256((root/'probe_cutile_threads.py').read_bytes()).hexdigest()
assert len(report['records'])==8 and len({r['native_id'] for r in report['records']})==2
text=(root/'threads.log').read_text()
phases=[f'THREAD_{thread}_B{batch}_{phase}' for thread in [1,2] for batch in [1,32] for phase in ['first','repeat']]
counts=[]
for i,phase in enumerate(phases):
    assert text.count(phase)==1
    part=text.split(phase,1)[1]
    if i+1<len(phases): part=part.split(phases[i+1],1)[0]
    count=part.count('CUTILE_JIT_TIMING '); counts.append(count)
    if phase.endswith('repeat'): assert count==0, 'repeat cache miss'
assert counts==[4,0,2,0,4,0,2,0], 'thread-local cache prediction failed'
summary.update(thread_exit=result.returncode,phase_misses=counts,native_records=parse(text),exact_searches=8,distinct_threads=2)
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({'gate':'PASS','exact_searches':8,'phase_misses':counts}),flush=True)
