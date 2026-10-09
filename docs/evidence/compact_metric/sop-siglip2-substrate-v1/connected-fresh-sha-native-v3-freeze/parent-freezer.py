from pathlib import Path
import json,hashlib,re,subprocess,shutil
base=Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-fresh-sha-native-v2-freeze')
out=Path('/tmp/sfora-connected-fresh-sha-native-v3-freeze');out.mkdir()
oldroot='/home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v2';root='/home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v3'
oldoutput='/home/riomus/runs/sfora-connected-fresh-sha-native-v2';output='/home/riomus/runs/sfora-connected-fresh-sha-native-v3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,d):(out/n).write_text(json.dumps(d,sort_keys=True,indent=2)+'\n')
proposal=json.loads((base/'proposal.json').read_text());before=dict(proposal)
proposal['schema']='connected-fresh-sha-bootstrap-proposal-v2'
assert {**proposal,'schema':before['schema']}==before
assert sha(Path('scripts/qualify_connected_fresh_sha_bootstrap.py'))==proposal['driver']['sha256']
assert sha(Path('scripts/test_connected_fresh_sha_bootstrap.py'))==proposal['test']['sha256']
write('proposal.json',proposal)
oldsha=sha(base/'proposal.json');newsha=sha(out/'proposal.json')
command=(base/'command.sh').read_text().replace(oldroot,root).replace(oldoutput,output).replace(oldsha,newsha)
assert command.replace(root,oldroot).replace(output,oldoutput).replace(newsha,oldsha)==(base/'command.sh').read_text()
blocks=re.findall("sha256sum -c <<'HASHES'\n(.*?)\nHASHES",command,re.S);assert len(blocks)==2 and blocks[0]==blocks[1]
assert '--proposal-sha256 '+newsha in command
(out/'command.sh').write_text(command)
launch=(base/'launch.sh').read_text().replace(oldroot,root).replace(oldoutput,output).replace(sha(base/'command.sh'),sha(out/'command.sh'))
(out/'launch.sh').write_text(launch)
(out/'controller.sh').write_text('#!/bin/bash\nset -u\nset -o noclobber\n/bin/bash '+root+'/launch.sh > '+root+'/original.log 2>&1\nsfora_status=$?\nprintf \'%s\\n\' "$sfora_status" > '+root+'/terminal-status.txt\nexit "$sfora_status"\n')
for n in ['command.sh','launch.sh','controller.sh']:subprocess.run(['bash','-n',str(out/n)],check=True)
write('freeze.json',{'schema':'connected-fresh-sha-root-freeze-v3','source_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'root':root,'output':output,'unit':'sfora-connected-fresh-sha-native-v3','files':{p.name:sha(p) for p in out.iterdir()},'change':'ONLY proposal schema now equals authenticated driver PROPOSAL_SCHEMA; canonical new proposal/log/output/unit paths and exact corresponding hashes. Driver/test/proposed three source files/original authority/all resource and science policies unchanged.','prior_failures':'v1 exact-four FAIL and v2 proposal-schema FAIL preserved; no native result from either','basis':'Actual installed serving v2 instrumented warmup attributes tensor SHA ~1.377s versus B1 median1.689s; bounds/parity/lifetime qualification of already source-reviewed fresh SHA helper precedes adoption. No latency-removability or speed GO assumed.','policy':{'body_seconds':300,'whole_seconds':1500,'host_bytes':8589934592,'swap_bytes':0,'cuda_exclusive_bytes':10000000000},'stop_rule':'ONE new engineering-only job after actual stdlib read_proposal accepts frozen files. Any bootstrap/authority/parity/mutation/lifetime/resource/full exit/normal-terminal failure rejects v3; no repeat/automatic cap change. No quality/product/speed GO, no training or current pooling-arm continuation.','launches':0,'native_qualified':False})
print(json.dumps({'root':root,'proposal_sha256':newsha,'command_sha256':sha(out/'command.sh'),'freeze_sha256':sha(out/'freeze.json')},sort_keys=True))
