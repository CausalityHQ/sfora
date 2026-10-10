import base64,csv,hashlib,importlib.abc,io,json,stat,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
wheel=Path('/data/target/sfora-installed-native-wheel-d3c17a98/dist/sfora-0.3.0rc4-py3-none-any.whl')
installed=Path('/data/target/sfora-installed-native-wheel-d3c17a98/installed')
with zipfile.ZipFile(wheel) as archive:
 names=archive.namelist();assert len(names)==len(set(names)) and not any(n.endswith('.pyc') for n in names)
 assert not any('/rust/' in n or n.endswith('.rs') for n in names)
 record=next(n for n in names if n.endswith('.dist-info/RECORD'))
 rows=list(csv.reader(io.StringIO(archive.read(record).decode()),strict=True));assert len(rows)==len(names)
 for name,encoded,size in rows:
  assert not Path(name).is_absolute() and '..' not in Path(name).parts
  if name==record:assert encoded==size=='';continue
  raw=archive.read(name);expected='sha256='+base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')
  assert encoded==expected and int(size)==len(raw)
  if name.startswith('sfora/'):
   committed=subprocess.check_output(['git','show','d3c17a98:src/'+name],cwd=ROOT)
   assert raw==committed and (installed/name).read_bytes()==raw and stat.S_ISREG((installed/name).lstat().st_mode)
source_names=[n for n in names if n.startswith('sfora/')]
class BlockML(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','numpy','PIL','transformers','safetensors','torchvision'}:raise AssertionError('ML import '+name)
sys.meta_path.insert(0,BlockML());sys.path.insert(0,str(installed))
import sfora.connected_compact_serving as bridge
assert Path(bridge.__file__).resolve()==installed/'sfora/connected_compact_serving.py'
assert callable(bridge.ConnectedCompactIndex.from_serving_artifact)
index=bridge.ConnectedCompactIndex();index.close();index.close();assert index._closed
assert not any(n.split('.')[0] in {'torch','numpy','PIL','transformers','safetensors','torchvision'} for n in sys.modules)
assert not list(installed.rglob('*.pyc'))
print(json.dumps({'schema':'current-wheel-source-import-v1','wheel':str(wheel),'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest(),'wheel_bytes':wheel.stat().st_size,'source_members':len(source_names),'committed_source_match':True,'canonical_installed_bridge_import':True,'native_ml_imports':False,'native_qualified':False,'model_requests_run':0},sort_keys=True))
