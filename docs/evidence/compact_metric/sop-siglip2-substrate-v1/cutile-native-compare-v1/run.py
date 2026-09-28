import hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
root=Path('/home/riomus/runs/sfora-cutile-native-compare-v1')
hashes=json.loads((root/'library-hashes.json').read_text())
probe_hash=hashlib.sha256((root/'probe.py').read_bytes()).hexdigest()
summary={'claim_eligible':False,'processes':[],'gate_pass':False}
env=os.environ.copy()
env.update(PYTHONPATH='/home/riomus/sfora-siglip2-deployed-batch-v1/src',CUTILE_JIT_TIMING='1')
for batch in [1,32]:
    for i, arm in enumerate(['original','candidate','candidate','original']):
        stem=f'b{batch}-{i}-{arm}'
        assert hashlib.sha256((root/(arm+'.so')).read_bytes()).hexdigest()==hashes[arm]
        assert hashlib.sha256((root/'probe.py').read_bytes()).hexdigest()==probe_hash
        start=time.monotonic()
        with (root/(stem+'.log')).open('x') as log:
            result=subprocess.run(['/usr/bin/time','-v','-o',str(root/(stem+'-time.txt')),'/home/riomus/group-learning/.venv/bin/python',str(root/'probe.py'),'--library',str(root/(arm+'.so')),'--batch',str(batch),'--output',str(root/(stem+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=75)
        record={'stem':stem,'arm':arm,'batch':batch,'exit':result.returncode,'wall_seconds':time.monotonic()-start}
        summary['processes'].append(record)
        (root/'controller-progress.json').write_text(json.dumps(summary,indent=2)+'\n')
        assert result.returncode==0, record
        text=(root/(stem+'.log')).read_text()
        assert text.count('PHASE_HOT')==1 and text.count('PHASE_TIES')==1
        assert 'CUTILE_JIT_TIMING ' not in text.split('PHASE_HOT',1)[1].split('PHASE_TIES',1)[0], 'hot JIT miss'
        report=json.loads((root/(stem+'.json')).read_text())
        assert report['checked_searches']==57 and len(report['raw_search_ns'])==50 and len(report['checked_main_outputs'])==56
        assert report['source_sha256']==probe_hash and report['library_sha256']==hashes[arm]
        for ordinals,bits in report['checked_main_outputs']:
            assert ordinals==report['ordinals'] and bits==report['score_bits']
        record['create_ms']=report['create_ns']/1e6
        record['first_search_ms']=report['first_search_ns']/1e6
        print(json.dumps(record),flush=True)
summary['batches']={}
for batch in [1,32]:
    reports={arm:[json.loads((root/(p['stem']+'.json')).read_text()) for p in summary['processes'] if p['batch']==batch and p['arm']==arm] for arm in ['original','candidate']}
    reference=reports['original'][0]
    for arm in reports:
        for r in reports[arm]:
            for key in ['fixture_sha256','ordinals','score_bits','tie_ordinals','tie_score_bits']:
                assert r[key]==reference[key], key
    values={arm:np.quantile([v/1e6 for r in rs for v in r['raw_search_ns']],[.5,.95]).tolist() for arm,rs in reports.items()}
    ratios=(np.array(values['candidate'])/np.array(values['original'])).tolist()
    summary['batches'][str(batch)]={'p50_p95_ms':values,'candidate_over_original':ratios,'pass':ratios[0]<=1.05 and ratios[1]<=1.10}
summary['gate_pass']=all(b['pass'] for b in summary['batches'].values())
(root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary),flush=True)
assert summary['gate_pass'], 'frozen hot latency gate failed; no retry'
