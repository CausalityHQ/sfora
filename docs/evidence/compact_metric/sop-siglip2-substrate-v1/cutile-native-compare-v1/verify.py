"""Replay archived exact outputs, native misses and frozen hot quantiles."""
import hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / 'scripts'))
from parse_cutile_jit_timing import parse
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for name,digest in receipt['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
summary=json.loads((root/'summary.json').read_text())
assert summary['gate_pass'] and len(summary['processes'])==8
native=[]
reports={}
for batch in [1,32]:
    rows=[p for p in summary['processes'] if p['batch']==batch]
    assert [p['arm'] for p in rows]==['original','candidate','candidate','original']
    for p in rows:
        stem=p['stem']; r=json.loads((root/(stem+'.json')).read_text())
        reports[stem]=r
        assert p['exit']==0 and r['checked_searches']==57
        assert len(r['checked_main_outputs'])==56 and len(r['raw_search_ns'])==50
        assert all(isinstance(n,int) and n>0 for n in r['raw_search_ns'])
        assert r['source_sha256']==hashlib.sha256((root/'probe.py').read_bytes()).hexdigest()
        assert r['library_sha256']==receipt['libraries_sha256'][p['arm']]
        for ordinals,bits in r['checked_main_outputs']:
            assert ordinals==r['ordinals'] and bits==r['score_bits']
        assert r['tie_ordinals']==[list(range(10)) for _ in range(batch)]
        text=(root/(stem+'.log')).read_text()
        assert 'CUTILE_JIT_TIMING ' not in text.split('PHASE_HOT')[1].split('PHASE_TIES')[0]
        records=parse(text)
        assert len(records)==(5 if p['arm']=='original' else 4)
        assert sum(rec['function']=='merge_topk' for rec in records)==(2 if p['arm']=='original' else 1)
        native.append({'stem':stem,'misses':len(records),'stage_ms':[sum(rec[key] for rec in records) for key in ['stage1_ms','stage2_ms','stage3_ms']]})
        assert 'Exit status: 0' in (root/(stem+'-time.txt')).read_text()
    reference=reports[rows[0]['stem']]
    for p in rows:
        for key in ['fixture_sha256','ordinals','score_bits','tie_ordinals','tie_score_bits']:
            assert reports[p['stem']][key]==reference[key]
    for arm in ['original','candidate']:
        values=sorted(n/1e6 for p in rows if p['arm']==arm for n in reports[p['stem']]['raw_search_ns'])
        assert len(values)==100
        for index,fraction in enumerate([.5,.95]):
            position=(len(values)-1)*fraction; low=int(position)
            observed=values[low]+(values[low+1]-values[low])*(position-low)
            assert math.isclose(observed,summary['batches'][str(batch)]['p50_p95_ms'][arm][index],abs_tol=1e-12)
    ratios=summary['batches'][str(batch)]['candidate_over_original']
    assert ratios[0]<=1.05 and ratios[1]<=1.10
assert 'Exit status: 0' in (root/'controller-time.txt').read_text()
print(json.dumps({'verified_searches':456,'native':native,'gate':'PASS'},indent=2))
