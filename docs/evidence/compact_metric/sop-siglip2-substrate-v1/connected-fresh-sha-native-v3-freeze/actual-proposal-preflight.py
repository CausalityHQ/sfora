from pathlib import Path
import json,hashlib,sys,runpy
root=Path('/home/riomus/runs/sfora-connected-installed-control-serving-source-v2')
sys.path.insert(0,str(root))
proposal=Path('/home/riomus/runs/sfora-connected-fresh-sha-native-freeze-v3/proposal.json')
p=json.loads(proposal.read_bytes())
assert hashlib.sha256(proposal.read_bytes()).hexdigest()=='4f334611a1440260168a3f583a7cf21d9e7d62dec3f268bd128358cd1015644b'
assert hashlib.sha256(Path(p['driver']['path']).read_bytes()).hexdigest()==p['driver']['sha256']
a=json.loads((root/'authority.json').read_bytes())
d=runpy.run_path(p['driver']['path'],run_name='_sfora_fresh_proposal_preflight')
assert d['read_proposal']({'path':str(proposal),'sha256':hashlib.sha256(proposal.read_bytes()).hexdigest()},a['sources'])==p
assert not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'} for n in sys.modules)
print(json.dumps({'schema':'fresh-sha-actual-proposal-source-check-v3','actual_read_proposal_pass':True,'actual_seam_inverse_pass':True,'native_modules':[],'native_qualified':False},sort_keys=True))
