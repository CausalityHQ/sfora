import hashlib, json, os, shutil, subprocess, tarfile
from pathlib import Path
root=Path('/home/riomus/runs/sfora-cutile-native-compare-v1')
assert hashlib.sha256((root/'sources.tar').read_bytes()).hexdigest()=='1934cab287826c0a80f265c2014536889764c0abe33b589027454b90e4e210a2'
with tarfile.open(root/'sources.tar') as archive: archive.extractall(root, filter='data')
manifest=json.loads((root/'manifest.json').read_text())
for arm, files in manifest.items():
    for name, digest in files.items(): assert hashlib.sha256((root/arm/name).read_bytes()).hexdigest()==digest
assert [name for name in manifest['original'] if manifest['original'][name]!=manifest['candidate'][name]]==['rust/sfora-cutile-int8-score/src/topk.rs']
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='',CUDA_TOOLKIT_PATH='/usr/local/cuda-13.0',CARGO_TARGET_DIR='/home/riomus/cutile-rs-c299d449/target',BINDGEN_EXTRA_CLANG_ARGS='-I/usr/lib/gcc/aarch64-linux-gnu/13/include')
env['PATH']='/home/riomus/.cargo/bin:/usr/local/cuda-13.0/bin:'+env['PATH']
for arm in ['candidate']:
    subprocess.run(['cargo','clean','--release','-p','sfora-cutile-int8-score'],cwd=root/arm,env=env,check=True)
    subprocess.run(['cargo','build','--offline','--locked','--release','--lib','-p','sfora-cutile-int8-score'],cwd=root/arm,env=env,check=True)
    shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/libsfora_cutile_int8_score.so',root/(arm+'.so'))
(root/'library-hashes.json').write_text(json.dumps({arm:hashlib.sha256((root/(arm+'.so')).read_bytes()).hexdigest() for arm in ['original','candidate']},indent=2)+'\n')
assert (root/'original.so').read_bytes() != (root/'candidate.so').read_bytes()
print('PASS candidate actually rebuilt; distinct library hashes',flush=True)
