"""Replay CPU source authority and the deferred review decision, no GPU claim."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for name,digest in receipt['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
preflight=json.loads((root/'cpu-preflight.json').read_text())
assert preflight['decision']=='PASS_CPU_AUTHORITY_ONLY'
assert preflight['fit_rows']==13283 and preflight['fit_products']==2004
assert preflight['classifier_shape']==[2004,128] and preflight['cuda_visible_devices']==''
assert preflight['source_sha256']==hashlib.sha256((root/'preflight.py').read_bytes()).hexdigest()
assert preflight['expected_feature_sha256']=='ea78db9abaf5200bc9952caef8fecd5cd7c38dd29f8cd33ad55de00bd8e1371b'
for name in ['cpu-time.txt','writer-time.txt']:
    assert 'Exit status: 0' in (root/name).read_text()
assert 'positive-tail self-test passed' in (root/'writer.log').read_text()
review=json.loads((root/'dual-result.json').read_text())
assert review['status']=='completed' and len(review['reviewers'])==2
assert all(r['status']=='completed' and r['exit_code']==0 for r in review['reviewers'])
assert receipt['decision']=='DEFER_RECOVERY_NO_GPU_LAUNCH'
assert receipt['recovery_gpu_launched'] is False
print('PASS CPU authority, unshipped writer check and split review; no recovered tensor claim')
