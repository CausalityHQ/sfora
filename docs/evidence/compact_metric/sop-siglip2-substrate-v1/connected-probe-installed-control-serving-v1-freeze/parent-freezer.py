from pathlib import Path
import ast,hashlib,json,re,shutil,subprocess,sys,importlib.util
repo=Path('/home/rb/worktrees/sfora-positive-causality')
out=Path('/tmp/sfora-probe-installed-control-serving-v1-freeze');out.mkdir()
root='/home/riomus/runs/sfora-connected-probe-installed-control-serving-source-v1'
output='/home/riomus/runs/sfora-connected-probe-installed-control-serving-diagnostic-v1'
unit='sfora-connected-probe-installed-control-serving-diagnostic-v1'
site='/home/riomus/runs/sfora-connected-probe-installed-serving-wheel-v1/site-packages'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def write(n,d):(out/n).write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
roles={'probe_driver':'qualify_connected_probe_serving.py','probe_test':'test_connected_probe_serving.py','request_driver':'qualify_connected_serving_requests.py','request_test':'test_connected_serving_requests.py','observer':'observe_connected_serving.py','observer_test':'test_observe_connected_serving.py','control_native':'connected_control_native_authority.py'}
sources={}
for role,name in roles.items():
 shutil.copyfile(repo/'scripts'/name,out/name);sources[role]={'path':root+'/'+name,'sha256':sha(out/name)}
assert sources['probe_driver']['sha256']=='bf8d63d4ba0a9e852f0da33fe6cf14b844e89c3d79a8c3824a9ec3ed77b71c94'
assert sources['probe_test']['sha256']=='a864c91328d871710adf45fbabdde903b140b96c329a310ebfccbc8151782a9b'
installed=load(repo/'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-installed-wheel-source-v1/installed-source-manifest.json')
assert installed['site_packages']==site and installed['source_matches_wheel'] and not installed['native_qualified']
for role,name in {'bridge':'connected_compact_serving.py','runtime':'connected_probe_inference.py','ledger':'_connected_probe_inference_authority.py','packed':'packed_int8.py','native_wrapper':'cutile_int8.py','packing':'joint_relational_compaction.py'}.items():
 sources[role]={'path':site+'/sfora/'+name,'sha256':installed['files']['sfora/'+name]['sha256']}
exportroot=Path('/tmp/sfora-probe-export-control-179061-v1-freeze')
launch=load(exportroot/'authority-export-control-179061-v1.json')
endpoint=launch['endpoints'][0];assert endpoint['arm']=='control' and endpoint['seed']==179061
manifest=load('/tmp/sfora-probe-control-train-v1-result/bundle.json');assert sha('/tmp/sfora-probe-control-train-v1-result/bundle.json')==endpoint['bundle']['sha256']
bdirectory=str(Path(endpoint['bundle']['path']).parent)
bundle={'directory':bdirectory,'manifest':endpoint['bundle'],'code':{n:{'path':bdirectory+'/'+n,'sha256':h} for n,h in manifest['code'].items()},'files':{n:{'path':bdirectory+'/'+n,'sha256':h} for n,h in manifest['files'].items()}}
assets=load('/tmp/sfora-probe-control-serving-assets-v1/metadata.json')
assert assets['source_export']==load('/tmp/sfora-probe-export-control-179061-v1-result/accepted-unit.json')
old=load('/tmp/sfora-installed-control-serving-freeze-v2/authority.json')
dist='sfora-0.3.0rc4.dist-info'
metadata=installed['metadata']
wheel={'site_root':site,'distribution':'sfora','version':'0.3.0rc4','record':{'path':site+'/'+dist+'/RECORD','sha256':metadata[dist+'/RECORD']['sha256']},'direct_url':{'path':site+'/'+dist+'/direct_url.json','sha256':metadata[dist+'/direct_url.json']['sha256']}}
policy={'body_seconds':300,'host_bytes':8589934592,'swap_bytes':0,'cuda_allocated_bytes_exclusive':10000000000,'whole_process_seconds':1500,'exit_reserve_seconds':300}
authority={'schema':'connected-probe-serving-authority-v1','sources':sources,'evaluator':{'root':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1','execution_sha256':launch['execution_sha256'],'code':{'evaluate_siglip2_connected_probe.py':'6b60df9162a5cdfc253c61535dac09681604ba1868a6b2135a75fd65b8ffdd1d','test_connected_probe_evaluation.py':'c101bc5517acc720f66084a0d7859630f3d344a02831a671e2e790820fa68890'}},'evaluation_authority':{'path':'/home/riomus/runs/sfora-connected-probe-evaluation-source-v1/authority-export-control-179061-v1.json','sha256':sha(exportroot/'authority-export-control-179061-v1.json')},'endpoint':{'arm':'control','seed':179061,'stage':'first','panel':'selection'},'train_export':assets['source_export'],'bundle':bundle,'gallery':assets['gallery'],'images':assets['images'],'native':{'authority':old['native_runtime'],'library':{'path':'/home/riomus/runs/sfora-cutile-threads-v1/candidate.so','sha256':'3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526'},'archived_control_binary_ack':True},'wheel':wheel,'locks':old['locks'],'resource_policy':policy}
assert authority['native']['authority']=={'path':'/home/riomus/runs/sfora-connected-control-serving-native-authority-v5/native-authority.json','sha256':'cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1'}
sys.path.insert(0,str(repo/'scripts'))
spec=importlib.util.spec_from_file_location('_sfora_probe_freeze_shape',repo/'scripts/qualify_connected_probe_serving.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.check_shape(authority)
assert not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers'} for n in sys.modules)
write('authority.json',authority)
# Outer hashes include every installed source+metadata file. Large model and archive inputs remain owned by the unchanged original admission and fresh exit.
pins={root+'/'+n:sha(out/n) for n in [*roles.values(),'authority.json']}
pins.update({site+'/'+n:v['sha256'] for n,v in {**installed['files'],**metadata}.items()})
pins.update({authority['native']['authority']['path']:authority['native']['authority']['sha256'],authority['evaluation_authority']['path']:authority['evaluation_authority']['sha256']})
oldcommand=Path('/tmp/sfora-installed-control-serving-freeze-v2/command.sh').read_text()
oldblocks=re.findall("sha256sum -c <<'HASHES'\n(.*?)\nHASHES",oldcommand,re.S);assert len(oldblocks)==2 and oldblocks[0]==oldblocks[1]
oldrows=dict((p,h) for h,p in (row.split('  ',1) for row in oldblocks[0].splitlines()))
for p in ['/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13','/home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py','/home/riomus/runs/sfora-native256-stop-cgroup-footer-v1.py','/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras']:pins[p]=oldrows[p]
block='\n'.join(h+'  '+p for p,h in sorted(pins.items()))
command=oldcommand.replace(oldblocks[0],block)
command=command.replace('/home/riomus/runs/sfora-connected-installed-control-serving-source-v2',root).replace('/home/riomus/runs/sfora-connected-installed-control-serving-diagnostic-v2',output).replace('/home/riomus/runs/sfora-connected-installed-control-serving-wheel-v1/site-packages',site).replace('qualify_connected_control_serving.py','qualify_connected_probe_serving.py').replace('8d917eb2bc24db5b36dba74e1557da4b76aa1f8917f88adf10e462a3169fd4b3',sha(out/'authority.json'))
# Replacement occurs only outside frozen fresh rows; assert their exact identity.
assert re.findall("sha256sum -c <<'HASHES'\n(.*?)\nHASHES",command,re.S)==[block,block]
assert '--authority-sha256 '+sha(out/'authority.json') in command
(out/'command.sh').write_text(command)
launcher=Path('/tmp/sfora-installed-control-serving-freeze-v2/launch.sh').read_text().replace('/home/riomus/runs/sfora-connected-installed-control-serving-source-v2',root).replace('sfora-connected-installed-control-serving-diagnostic-v2',unit).replace('97b9dea7bed947a3b23b615d8fbdd14830e9ae30a23efdafab1c89e1fabac3ca',sha(out/'command.sh'))
assert sha(out/'command.sh') in launcher
(out/'launch.sh').write_text(launcher)
(out/'controller.sh').write_text('#!/bin/bash\nset -u\nset -o noclobber\n/bin/bash '+root+'/launch.sh > '+root+'/original.log 2>&1\nsfora_launch_status=$?\nprintf \'%s\\n\' "$sfora_launch_status" > '+root+'/terminal-status.txt\nexit "$sfora_launch_status"\n')
for n in ['command.sh','launch.sh','controller.sh']:subprocess.run(['bash','-n',str(out/n)],check=True)
freeze={'schema':'connected-probe-installed-control-serving-freeze-v1','root':root,'output':output,'unit':unit,'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'source_files':roles,'files':{p.name:sha(p) for p in sorted(out.iterdir())},'resource_policy':policy,'source_only_shape_check':True,'native_qualified':False,'launches':0,'quality_decision':'candidate pooling-probe FIRST KILL; only accepted control serving parity eligible','stop_rule':'ONE engineering-only original job; any parity/source/ownership/lifetime/body300/whole1500/8GiB/zeroSwap/CUDA<10GB/full exit/normal terminal failure rejects; no duplicate or automatic cap increase; no quality or speed claim.'}
write('freeze.json',freeze)
print(json.dumps({'freeze_sha256':sha(out/'freeze.json'),'authority_sha256':sha(out/'authority.json'),'source_files':len(roles),'outer_fresh_occurrences':len(pins),'native_qualified':False},sort_keys=True))
