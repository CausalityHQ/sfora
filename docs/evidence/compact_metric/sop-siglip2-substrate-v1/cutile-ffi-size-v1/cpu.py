import hashlib,json,os,resource,subprocess,sys
from pathlib import Path
root=Path('/home/riomus/runs/sfora-cutile-ffi-size-v1')
phase=sys.argv[1]; assert phase in ['red','green']
assert not (root/(phase+'-receipt.json')).exists()
env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES='',CUDA_TOOLKIT_PATH='/usr/local/cuda-13.0',CARGO_TARGET_DIR='/home/riomus/cutile-rs-c299d449/target',BINDGEN_EXTRA_CLANG_ARGS='-I/usr/lib/gcc/aarch64-linux-gnu/13/include')
env['PATH']='/home/riomus/.cargo/bin:/usr/local/cuda-13.0/bin:'+env['PATH']
subprocess.run(['cargo','clean','--release','-p','sfora-cutile-int8-score'],cwd=root,env=env,check=True)
compiled=subprocess.run(['cargo','test','--offline','--locked','--release','--lib','--no-run','--message-format=json','-p','sfora-cutile-int8-score'],cwd=root,env=env,stdout=subprocess.PIPE,text=True,check=True)
artifacts=[json.loads(line) for line in compiled.stdout.splitlines() if line.startswith('{')]
binaries=[a['executable'] for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a.get('profile',{}).get('test')]
assert len(binaries)==1
binary=Path(binaries[0]); (root/(phase+'-tests')).write_bytes(binary.read_bytes()); (root/(phase+'-tests')).chmod(0o755)
def cap():
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
result=subprocess.run([str(root/(phase+'-tests')),'ffi::tests::ffi_rejects_unsupported_gallery_sizes_before_buffer_access','--exact','--nocapture','--test-threads=1'],env=env,preexec_fn=cap,timeout=20)
receipt={'phase':phase,'test_returncode':result.returncode,'ffi_sha256':hashlib.sha256((root/'rust/sfora-cutile-int8-score/src/ffi.rs').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'address_space_bytes':1024**3,'cuda_visible_devices':''}
(root/(phase+'-receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt),flush=True)
if phase=='green': assert result.returncode==0
else: assert result.returncode==-6, 'expected bounded allocation abort from unguarded baseline'
