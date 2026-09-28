import hashlib,json,os,subprocess
from pathlib import Path
root=Path('/home/riomus/runs/sfora-cutile-ffi-size-v1')
base=Path('/home/riomus/cutile-rs-c299d449/target/release/deps')
files=['sfora_cutile_int8_score-31028a1d4def638e','rc4_profile-26552007ceae0576','sfora_cutile_int8_score-3bcad7c2897388a4','merge_boundary-16a4d0078bee43e4']
assert hashlib.sha256((base/files[0]).read_bytes()).hexdigest()==json.loads((root/'green-receipt.json').read_text())['binary_sha256']
records=[]
for name in files:
    src=base/name; dst=root/name
    dst.write_bytes(src.read_bytes()); dst.chmod(0o755)
    digest=hashlib.sha256(dst.read_bytes()).hexdigest()
    result=subprocess.run([str(dst),'--nocapture','--test-threads=1'],timeout=250)
    records.append({'binary':name,'sha256':digest,'exit':result.returncode})
    (root/'full-tests-receipt.json').write_text(json.dumps(records,indent=2)+'\n')
    assert result.returncode==0
print('PASS all four compiled Rust test targets',flush=True)
