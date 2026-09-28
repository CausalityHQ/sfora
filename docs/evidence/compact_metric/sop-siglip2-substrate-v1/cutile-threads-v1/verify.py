"""Replay thread outputs against the previously archived scalar-bit oracle."""
import hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]/'scripts'))
from parse_cutile_jit_timing import parse
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for name,digest in receipt['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
summary=json.loads((root/'summary.json').read_text())
report=json.loads((root/'threads.json').read_text())
reference=json.loads((root.parent/'cutile-native-compare-v1/b32-0-original.json').read_text())
assert report['library_sha256']==json.loads((root/'binaries.json').read_text())['candidate.so']
assert report['source_sha256']==hashlib.sha256((root/'probe_cutile_threads.py').read_bytes()).hexdigest()
assert (root/'probe_cutile_runtime_bounds.py').read_bytes()==(root.parent/'cutile-native-compare-v1/probe.py').read_bytes()
records=report['records']
assert len(records)==8 and len({r['native_id'] for r in records})==2
assert [(r['thread'],r['batch'],r['phase']) for r in records]==[(t,b,p) for t in [1,2] for b in [1,32] for p in ['first','repeat']]
text=(root/'threads.log').read_text()
counts=[]
for i,r in enumerate(records):
    assert r['ordinals']==reference['ordinals'][:r['batch']]
    assert r['score_bits']==reference['score_bits'][:r['batch']]
    assert r['search_ns']>0
    marker=f"THREAD_{r['thread']}_B{r['batch']}_{r['phase']}"
    part=text.split(marker,1)[1]
    if i<7:
        n=records[i+1]; part=part.split(f"THREAD_{n['thread']}_B{n['batch']}_{n['phase']}",1)[0]
    counts.append(part.count('CUTILE_JIT_TIMING '))
assert counts==summary['phase_misses']==[4,0,2,0,4,0,2,0]
parsed=parse(text)
assert len(parsed)==12 and parsed==summary['native_records']
assert len({(r['module'],r['function'],r['key']) for r in parsed})==6
assert all(t['exit']==0 for t in summary['targets'])
assert [int(n) for n in re.findall(r'test result: ok\. (\d+) passed; 0 failed',(root/'gpu.log').read_text())]==[11,1,1,2]
for name in ['cpu-time.txt','gpu-time.txt']:
    assert 'Exit status: 0' in (root/name).read_text()
print('PASS all15 Rust tests, eight scalar-bit outputs and two-thread native cache behavior')
