#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=
export RAYON_NUM_THREADS=1
export UV_CONCURRENT_BUILDS=1
export UV_CONCURRENT_INSTALLS=1
stage=/home/riomus/runs/sfora-connected-installed-wheel-stage-v1
target=/home/riomus/runs/sfora-connected-installed-control-serving-wheel-v1
sha256sum -c <<'HASHES'
f0b0e0f06371526b474450abef98e0d33c1cb06598e7d1b8a09dbe7e013840e7  /snap/astral-uv/1779/bin/uv
9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b  /home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13
a1ff1b26f4d1d696e016344e03099ae2618621624929de70395dba81e5671eb6  /home/riomus/runs/sfora-connected-installed-wheel-stage-v1/sfora-0.3.0rc4-py3-none-any.whl
HASHES
test ! -e "$target"
mkdir "$target"
/usr/bin/time -v /usr/bin/timeout 30 /usr/bin/prlimit --as=1073741824 /snap/astral-uv/1779/bin/uv pip install --offline --no-deps --no-cache --python /home/riomus/group-learning/.venv/bin/python --target "$target/site-packages" "$stage/sfora-0.3.0rc4-py3-none-any.whl"
/home/riomus/group-learning/.venv/bin/python -I -S -B - <<'PY'
from pathlib import Path
import hashlib,json,zipfile,sys
root=Path('/home/riomus/runs/sfora-connected-installed-control-serving-wheel-v1')
site=root/'site-packages'
wheel=Path('/home/riomus/runs/sfora-connected-installed-wheel-stage-v1/sfora-0.3.0rc4-py3-none-any.whl')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(wheel.read_bytes())=='a1ff1b26f4d1d696e016344e03099ae2618621624929de70395dba81e5671eb6'
files={}
with zipfile.ZipFile(wheel) as archive:
    names=[name for name in archive.namelist() if name.startswith('sfora/') and not name.endswith('/')]
    assert len(names)==110 and len(names)==len(set(names))
    for name in names:
        path=site/name
        assert path.resolve()==path.absolute() and not path.is_symlink()
        data=path.read_bytes()
        assert data==archive.read(name),name
        files[name]={'sha256':sha(data),'bytes':len(data)}
assert sorted(str(p.relative_to(site)) for p in (site/'sfora').rglob('*') if p.is_file())==sorted(files)
metadata={str(p.relative_to(site)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(site.glob('*.dist-info/*')) if p.is_file()}
record={'schema':'sfora-installed-wheel-source-v1','wheel_sha256':sha(wheel.read_bytes()),'python':sys.version,'site_packages':str(site),'files':files,'metadata':metadata,'source_matches_wheel':True,'dependencies_installed':False,'native_qualified':False}
with (root/'installed-source-manifest.json').open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps({'decision':'PASS_WHEEL_INSTALL_SOURCE_BYTES_ONLY','files':len(files),'metadata_files':len(metadata),'manifest_sha256':sha((root/'installed-source-manifest.json').read_bytes()),'native_qualified':False},sort_keys=True))
PY
