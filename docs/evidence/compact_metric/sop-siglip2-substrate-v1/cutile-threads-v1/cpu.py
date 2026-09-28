import hashlib,json,os,shutil,subprocess
from pathlib import Path
root=Path('/home/riomus/runs/sfora-cutile-threads-v1')
env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',CUDA_TOOLKIT_PATH='/usr/local/cuda-13.0',CARGO_TARGET_DIR='/home/riomus/cutile-rs-c299d449/target',BINDGEN_EXTRA_CLANG_ARGS='-I/usr/lib/gcc/aarch64-linux-gnu/13/include',PYTHONPATH='/home/riomus/sfora-siglip2-deployed-batch-v1/src')
env['PATH']='/home/riomus/.cargo/bin:/usr/local/cuda-13.0/bin:'+env['PATH']
subprocess.run(['/home/riomus/group-learning/.venv/bin/python',str(root/'probe_cutile_threads.py'),'--self-test'],env=env,check=True)
manifest=json.loads(Path('/home/riomus/runs/sfora-cutile-native-compare-v1/manifest.json').read_text())['candidate']
manifest['rust/sfora-cutile-int8-score/src/ffi.rs']='654901b53ee41f3c1c4a2d8e55910e6f2241e89e896fbb018d5b17f921bcc809'
for name,digest in manifest.items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest, name
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['cargo','clean','--release','-p','sfora-cutile-int8-score'],cwd=root/'source',env=env,check=True)
subprocess.run(['cargo','build','--offline','--locked','--release','--lib'],cwd=root/'source',env=env,check=True)
shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/libsfora_cutile_int8_score.so',root/'candidate.so')
subprocess.run(['cargo','test','--offline','--locked','--release','--no-run'],cwd=root/'source',env=env,check=True)
files=['sfora_cutile_int8_score-31028a1d4def638e','rc4_profile-26552007ceae0576','sfora_cutile_int8_score-3bcad7c2897388a4','merge_boundary-16a4d0078bee43e4']
for name in files: shutil.copyfile(Path(env['CARGO_TARGET_DIR'])/'release/deps'/name,root/name); (root/name).chmod(0o755)
(root/'binaries.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files+['candidate.so']},indent=2)+'\n')
print('PASS CPU thread self-check, source hashes and isolated candidate compile',flush=True)
