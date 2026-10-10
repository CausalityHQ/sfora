import hashlib,json,resource,signal,stat,sys,time
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824));signal.alarm(120);tick=time.monotonic()
python=Path('/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13')
assert Path(sys.executable).resolve()==python and hashlib.sha256(python.read_bytes()).hexdigest()=='9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'
path=Path('/home/riomus/runs/sfora-connected-control-serving-native-authority-v5/native-authority.json')
assert path.resolve()==path and stat.S_ISREG(path.lstat().st_mode) and path.stat().st_size<=1048576
raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()=='cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1'
authority=json.loads(raw);assert authority['schema']=='connected-control-native-authority-v2'
assert path.read_bytes()==raw
print(json.dumps({'schema':'original-supplemental-native-path-preflight-v1','authority_path':str(path),'authority_sha256':hashlib.sha256(raw).hexdigest(),'authority':authority,'elapsed_seconds':time.monotonic()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'native_library_bytes_read':False,'target_modified':False,'native_imports':False,'product_go':False},sort_keys=True))
