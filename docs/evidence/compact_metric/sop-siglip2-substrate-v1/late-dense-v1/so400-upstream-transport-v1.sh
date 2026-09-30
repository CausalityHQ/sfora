#!/bin/bash
set -euo pipefail
out=/home/riomus/runs/sfora-so400-native256-upstream-v1
base=https://huggingface.co/google/siglip2-so400m-patch16-256/resolve/e8708ab72d125807e45b36fb7d4e0aacbb59f379
mkdir "$out"
for name in config.json preprocessor_config.json model.safetensors; do
  curl --fail --location --silent --show-error --max-time 260 "$base/$name" --output "$out/$name.partial"
done
/usr/bin/python3 - "$out" <<'PY'
import hashlib,json,os,sys,time
from pathlib import Path
root=Path(sys.argv[1]); start=time.monotonic()
pins={'config.json':('13ee943037e446415ff6e7e406ecf79fd955f482f4ba5b437d3bc10fba7f1362',537),'preprocessor_config.json':('d14ba2ee3fd816f3de8abaddc31953565128eaf37c73ad4bed32101a98465aff',394),'model.safetensors':('810ddc85ac019e6b2738b9130ae3602e41eb093a848a297a0ad296ca1f39dc67',4542792928)}
records={}
for name,(expected,size) in pins.items():
 p=root/(name+'.partial'); assert p.stat().st_size==size
 h=hashlib.sha256(); offset=0
 with p.open('rb',buffering=0) as f:
  buf=bytearray(1048576)
  while count:=f.readinto(buf):
   h.update(memoryview(buf)[:count]); os.posix_fadvise(f.fileno(),offset,count,os.POSIX_FADV_DONTNEED); offset+=count
 assert h.hexdigest()==expected,(name,h.hexdigest())
 records[name]={'sha256':h.hexdigest(),'bytes':size}
for name in pins: (root/(name+'.partial')).rename(root/name)
receipt={'schema':'siglip2-upstream-transport-v1','model':'google/siglip2-so400m-patch16-256','revision':'e8708ab72d125807e45b36fb7d4e0aacbb59f379','files':records,'native_executed':False,'model_qualified':False,'quality_read':False,'verification_seconds':time.monotonic()-start}
with (root/'receipt.json').open('x') as f: json.dump(receipt,f,sort_keys=True,indent=2)
print(json.dumps(receipt,sort_keys=True))
PY
