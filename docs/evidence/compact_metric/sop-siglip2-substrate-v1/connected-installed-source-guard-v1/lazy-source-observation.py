settings={'/home/riomus/runs/sfora-connected-installed-site-v2/site-packages/transformers/__init__.py': {'sha256': '135ddabaeed9a78e2a40f5fd5ec948361b59650c96ce9222cba011826ee17632', 'bytes': 41198}, '/home/riomus/runs/sfora-connected-installed-site-v2/site-packages/transformers/utils/import_utils.py': {'sha256': '854f6a998a194c12f2e813f745b602a9e7aa7816b0e3f4153da19cfb708b6046', 'bytes': 116503}}
import hashlib,json,resource,signal,stat
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824));signal.alarm(120)
result={}
for name,row in settings.items():
 p=Path(name);assert p.resolve()==p and stat.S_ISREG(p.lstat().st_mode)
 raw=p.read_bytes();assert len(raw)==row["bytes"] and hashlib.sha256(raw).hexdigest()==row["sha256"]
 result[name]={**row,"source":raw.decode()}
print(json.dumps(result,sort_keys=True))
