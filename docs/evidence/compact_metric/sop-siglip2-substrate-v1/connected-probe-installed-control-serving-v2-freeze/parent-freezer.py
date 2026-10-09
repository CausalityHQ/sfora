from pathlib import Path
import ast,hashlib,json,re,shutil,subprocess
old=Path('/tmp/sfora-probe-installed-control-serving-v1-freeze');out=Path('/tmp/sfora-probe-installed-control-serving-v2-freeze');out.mkdir()
f=json.loads((old/'freeze.json').read_text());root=f['root'].removesuffix('v1')+'v2';output=f['output'].removesuffix('v1')+'v2';unit=f['unit'].removesuffix('v1')+'v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(out/n).write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
auth=json.loads((old/'authority.json').read_text())
for role,name in f['source_files'].items():
 shutil.copyfile(Path('scripts')/name,out/name);auth['sources'][role]={'path':root+'/'+name,'sha256':sha(out/name)}
assert sha(out/'qualify_connected_probe_serving.py')=='149808c99e9216afdf7aacdc7e4610ea810598ea2dee8d1b963243ac394253dd'
assert sha(out/'test_connected_probe_serving.py')=='37e209bf12e823571a828136d85bb8e9dd5da28cbc2309f4414bd7d17e35db84'
for role in auth['sources'].keys()-f['source_files'].keys():assert auth['sources'][role]==json.loads((old/'authority.json').read_text())['sources'][role]
write('authority.json',auth)
command=(old/'command.sh').read_text();blocks=re.findall("sha256sum -c <<'HASHES'\n(.*?)\nHASHES",command,re.S);assert len(blocks)==2 and blocks[0]==blocks[1]
newrows=[]
for row in blocks[0].splitlines():
 h,p=row.split('  ',1)
 if p.startswith(f['root']+'/'):
  n=p[len(f['root'])+1:];p=root+'/'+n;h=sha(out/n)
 newrows.append(h+'  '+p)
newblock='\n'.join(newrows);command=command.replace(blocks[0],newblock).replace(f['root'],root).replace(f['output'],output).replace(sha(old/'authority.json'),sha(out/'authority.json'))
assert re.findall("sha256sum -c <<'HASHES'\n(.*?)\nHASHES",command,re.S)==[newblock,newblock]
(out/'command.sh').write_text(command)
launch=(old/'launch.sh').read_text().replace(f['root'],root).replace(f['unit'],unit).replace(sha(old/'command.sh'),sha(out/'command.sh'))
assert '--unit='+unit+' ' in launch and f['unit'] not in launch and root+'/command.sh' in launch
(out/'launch.sh').write_text(launch)
controller=(old/'controller.sh').read_text().replace(f['root'],root);(out/'controller.sh').write_text(controller)
for n in ['command.sh','launch.sh','controller.sh']:subprocess.run(['bash','-n',str(out/n)],check=True)
write('freeze.json',{**f,'schema':'connected-probe-installed-control-serving-freeze-v2','root':root,'output':output,'unit':unit,'source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'files':{p.name:sha(p) for p in out.iterdir()},'source_worker_commit':'8892a8841293e66584f1fbbf5d3d9246b65cb703','prior_v1':'original SSH71350 invcddb55d04747481a8afc6822d46fcb67 FAIL startup shared-runs checkout inference preserved','root_review':'Exactly three call arguments .parent.parent -> .parent, full production byte inverse PASS, all98 original tests AST identical; genuine helper predicates intact. Original102 suite PASS3.32s/56176KiB/AS1GiB, suite bytes equal final commit; unchanged helper suites not repeated.','stop_rule':'ONE new engineering-only normal-terminal job AFTER original live fresh-SHA SSH35033 collected and idle/locks/source preflight. Original parity/mutants/lifetimes/full exit/body300/whole1500/8GiB/zeroSwap/CUDA<10GB unchanged; no duplicate, auto cap escalation, quality/speed/product GO.','launches':0})
print(json.dumps({'authority_sha256':sha(out/'authority.json'),'freeze_sha256':sha(out/'freeze.json'),'unit':unit,'launch_held_for':'original SSH35033 terminal collection'},sort_keys=True))
