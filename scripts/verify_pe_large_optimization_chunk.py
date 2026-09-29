#!/usr/bin/env python3
"""Read-only terminal native-chunk verification; no model or GPU work."""
import hashlib
import json
import math
import re
import sys
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    arm,end=sys.argv[1],int(sys.argv[2])
    assert arm in ('half','full') and 100<=end<=2000 and end%100==0
    root=Path('/home/riomus/runs/sfora-large-optimization-chunks-v1')
    output=Path(f'/home/riomus/runs/sfora-large-optimization-{arm}-{end}-v1')
    receipt=output/'receipt.json'
    value=json.loads(receipt.read_text())
    assert value['advance'] and value['arm']==arm and value['completed_step']==end and value['chunk_start']==end-100
    assert value['updates']==100 and value['total_updates']==2000
    assert not value['quality_read'] and not value['claim_eligible'] and value['frozen_source_code_environment_rng_preserved']
    code=root/'large-optimization-chunk-execution.json'
    assert sha(code)==value['execution_sha256']=='a7c27140c1cd53d1aee3b41f6e1c7bfb43173a467cab3855cfc2c92cac41bf7f'
    assert all(sha(root/n)==h for n,h in json.loads(code.read_text()).items())
    assert value['source_checkpoint_sha256']=='f5fbf0e8eb3492274b6febeb6fd06a4b0d8b0fce4c7170069f3fa0d7b0a8df4d'
    assert value['cpu_authority_sha256']=='c1e3e15a06ff6723a83713543cbdc07d8e9c78d3178ea7bd5c042fd91b23b143'
    assert value['mechanics_sha256']=='41cdd66db34b6e55d7aca72945133b6e25d0cea10cf18aae2a666d4ff4d1be1d'
    assert value['actual_gradient_input_identities']==(2004 if arm=='half' else 3997)
    assert value['minimum_id_exposure']>=1 and value['peak_cuda_allocated_bytes']<10_000_000_000
    assert [r['step'] for r in value['steps']]==list(range(end-99,end+1))
    for row in value['steps']:
        assert all(math.isfinite(row[k]) for k in ('seconds','ce','rank','loss','preclip_norm','scale'))
        assert len(row['gradient_norms'])==12 and all(math.isfinite(v) and v>0 for v in row['gradient_norms'].values())
    assert value['images_per_second']==6400/value['training_wall_seconds']
    assert sha(output/'resume.pt')==value['checkpoint_sha256']
    if end>100:
        previous=Path(f'/home/riomus/runs/sfora-large-optimization-{arm}-{end-100}-v1')
        assert sha(previous/'receipt.json')==value['previous_receipt_sha256']
        old=json.loads((previous/'receipt.json').read_text())
        assert old['checkpoint_sha256']==value['previous_checkpoint_sha256'] and old['completed_step']==end-100
        assert old['schedule_sha256']==value['schedule_sha256']
    else:
        assert value['previous_receipt_sha256'] is None and value['previous_checkpoint_sha256'] is None
    time=(root/f'{arm}-{end}-time.txt').read_text()
    assert re.search(r'Exit status: 0\s*$',time)
    assert re.search(r'Swaps: 0\s',time)
    rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',time)[1])
    assert rss*1024<8*1024**3
    wall=re.search(r'Elapsed \(wall clock\) time .*: (\d+:\d+(?::\d+)?(?:\.\d+)?)',time)[1]
    seconds=0.
    for part in wall.split(':'):
        seconds=seconds*60+float(part)
    assert seconds<300
    # Transient unit properties disappear at exit; use the original wait receipt.
    log=(root/f'{arm}-{end}.log').read_text()
    assert 'Finished with result: success' in log and 'Main processes terminated with: code=exited/status=0' in log
    assert 'Memory swap peak: 0B' in log
    runtime=re.search(r'Service runtime: (?:(\d+)min )?([\d.]+)s',log)
    unit_seconds=int(runtime[1] or 0)*60+float(runtime[2])
    assert unit_seconds<300
    assert sha(root/'run_pe_large_optimization_chunk.sh')=='9f80b8490cf62cf2931ff795078a4be8be2e8d66672fd5bdbafa35d3d0985308'
    print(json.dumps({'verified':True,'arm':arm,'completed_step':end,'receipt_sha256':sha(receipt),'checkpoint_sha256':value['checkpoint_sha256'],'training_seconds':value['training_wall_seconds'],'images_per_second':value['images_per_second'],'unit_seconds':unit_seconds,'python_seconds':seconds,'peak_cuda_allocated_bytes':value['peak_cuda_allocated_bytes'],'peak_RSS_KiB':rss,'unit_memory_peak_reported':re.search(r'Memory peak: (\S+)',log)[1],'enforced_resource_limits':'299+1s/8GiB/noSwap/twoGPUlocks; launcher SHA verified','no_swap':True,'quality_read':False}))


if __name__=='__main__':
    main()
