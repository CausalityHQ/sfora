settings={'wheel_sha256': '73525cee710db1235662409a3befc4e47102ab76f0196b9d815cacd6b6a2ae13', 'python': '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13', 'python_sha256': '9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b', 'directory': '/home/riomus/runs/sfora-connected-current-wheel-v3', 'wheel_path': '/home/riomus/runs/sfora-connected-current-wheel-v2/sfora-0.3.0rc4-py3-none-any.whl'}
import base64,csv,hashlib,importlib.abc,io,json,os,resource,signal,subprocess,sys,time,zipfile
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824));signal.alarm(120);tick=time.monotonic()
assert str(Path(sys.executable).resolve())==settings['python'] and hashlib.sha256(Path(settings['python']).read_bytes()).hexdigest()==settings['python_sha256']
root=Path(settings['directory']);root.mkdir(mode=0o700,exist_ok=False);wheel=root/'sfora-0.3.0rc4-py3-none-any.whl';raw=Path(settings['wheel_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==settings['wheel_sha256']
with wheel.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
pip=Path('/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/ensurepip/_bundled/pip-25.2-py3-none-any.whl');pip_sha='6d67a2b4e7f14d8b31b8b52648866fa717f45a1eb70e83002f4331d07e953717';assert pip.resolve()==pip and hashlib.sha256(pip.read_bytes()).hexdigest()==pip_sha
code='import sys;sys.path.insert(0,'+repr(str(pip))+');from pip._internal.cli.main import main;raise SystemExit(main(sys.argv[1:]))'
cmd=[settings['python'],'-I','-S','-B','-c',code,'--isolated','install','--disable-pip-version-check','--no-index','--no-deps','--no-compile','--no-cache-dir','--target',str(root/'installed'),str(wheel)]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=110);assert len(r.stdout)<=1048576
with (root/'installer.log').open('xb') as f:f.write(r.stdout);f.flush();os.fsync(f.fileno())
assert r.returncode==0,r.stdout.decode(errors='replace')
assert hashlib.sha256(pip.read_bytes()).hexdigest()==pip_sha
with zipfile.ZipFile(wheel) as archive:
 names=archive.namelist();assert len(names)==len(set(names)) and not any(n.endswith('.pyc') or n.endswith('.pth') for n in names)
 record=next(n for n in names if n.endswith('.dist-info/RECORD'))
 for name,encoded,size in csv.reader(io.StringIO(archive.read(record).decode()),strict=True):
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  if name==record:assert encoded==size=='';continue
  content=archive.read(name);assert encoded=='sha256='+base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip('=') and len(content)==int(size)
  if name.startswith('sfora/'):assert (root/'installed'/name).read_bytes()==content
class BlockML(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','numpy','PIL','transformers','safetensors','torchvision'}:raise AssertionError('forbidden ML import '+name)
sys.meta_path.insert(0,BlockML());sys.path.insert(0,str(root/'installed'))
import sfora.connected_compact_serving as bridge
assert bridge.__file__==str(root/'installed/sfora/connected_compact_serving.py')
assert callable(bridge.ConnectedCompactIndex.from_serving_artifact)
index=bridge.ConnectedCompactIndex();index.close();index.close();assert index._closed
assert not list((root/'installed').rglob('*.pyc'))
assert not any(n.split('.')[0] in {'torch','numpy','PIL','transformers','safetensors','torchvision'} for n in sys.modules)
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==settings['wheel_sha256']
print(json.dumps({'schema':'actual-dgx-current-wheel-source-import-v1','directory':str(root),'python_sha256':settings['python_sha256'],'wheel_sha256':settings['wheel_sha256'],'wheel_bytes':len(raw),'canonical_bridge_import':True,'source_member_count':sum(n.startswith('sfora/') for n in names),'installer_kind':'bundled-pip-25.2','installer_path':str(pip),'installer_sha256':pip_sha,'installer_exit':r.returncode,'installer_log_sha256':hashlib.sha256(r.stdout).hexdigest(),'elapsed_seconds':time.monotonic()-tick,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'native_ml_imports':False,'native_qualified':False,'model_requests_run':0},sort_keys=True))
