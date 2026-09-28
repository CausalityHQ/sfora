"""Replay the bounded FFI regression and complete Rust test receipts."""
import hashlib,json,re
from pathlib import Path
root=Path(__file__).resolve().parent
receipt=json.loads((root/'receipt.json').read_text())
for name,digest in receipt['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest, name
red=json.loads((root/'red-receipt.json').read_text())
green=json.loads((root/'green-receipt.json').read_text())
assert red['test_returncode']==-6 and green['test_returncode']==0
assert red['address_space_bytes']==green['address_space_bytes']==1073741824
assert red['cuda_visible_devices']==green['cuda_visible_devices']==''
assert 'memory allocation of 274877890688 bytes failed' in (root/'red.log').read_text()
assert '1 passed; 0 failed' in (root/'green.log').read_text()
assert hashlib.sha256((root/'ffi.rs.txt').read_bytes()).hexdigest()==green['ffi_sha256']
assert 'stddef.h' in (root/'full-gpu.log').read_text()
assert 'Exit status: 101' in (root/'full-gpu-time.txt').read_text()
records=json.loads((root/'full-tests-receipt.json').read_text())
assert len(records)==4 and all(r['exit']==0 for r in records)
assert records[0]['sha256']==green['binary_sha256']
counts=[int(n) for n in re.findall(r'test result: ok\. (\d+) passed; 0 failed', (root/'full-gpu-v2.log').read_text())]
assert counts==[11,1,1,2] and sum(counts)==15
for name in ['red-time.txt','green-time.txt','full-build-time.txt','full-gpu-v2-time.txt']:
    assert 'Exit status: 0' in (root/name).read_text()
# Analytical boundary equivalence, not execution on a 32-bit target.
maximum=(2**31-1)//128*128
for rows in [9,10,128,59551,maximum-1,maximum,maximum+1,2**31-1]:
    padded=(rows+127)//128*128
    assert (10<=rows<=maximum)==(rows>=10 and padded<=2**31-1)
for bits in [32,64]:
    limit=2**(bits-1)-1
    rows=min(maximum,limit//128)
    assert rows*128<=limit and rows*2<=limit
print('PASS bounded red/green, source hash, 15 Rust tests and boundary arithmetic')
