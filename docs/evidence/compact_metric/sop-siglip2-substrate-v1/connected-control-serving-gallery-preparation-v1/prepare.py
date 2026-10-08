import hashlib,json,os,pathlib,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(268435456,268435456))
p=pathlib.Path('/home/riomus/runs/sfora-connected-control-serving-gallery-v1/inputs.json')
raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='e7ae306ac5282c81e6e9c26bc32844db08b5f264049931df19caa2eaffdfa017'
j=json.loads(raw)
def checked(f):
 p=pathlib.Path(f['path']);assert p.is_absolute() and p.resolve()==p and not p.is_symlink()
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==f['sha256'];return b
partition=json.loads(checked(j['partition']))['panels']['selection']
assert partition['gallery']==j['gallery_ordinals'] and len(partition['original_rows'])==j['rows']
wire=checked(j['source']);assert len(wire)==j['rows']*130
ordinals=j['gallery_ordinals'];assert len(ordinals)==j['count']==1715 and len(set(ordinals))==1715
assert all(type(i) is int and 0<=i<j['rows'] for i in ordinals)
gallery=b''.join(wire[i*130:(i+1)*130] for i in ordinals);assert len(gallery)==1715*130
out=pathlib.Path(j['output']);assert out.parent.resolve()==out.parent and not out.exists()
with out.open('xb') as f:f.write(gallery);f.flush();os.fsync(f.fileno())
assert out.read_bytes()==gallery and checked(j['source'])==wire
record={'schema':'connected-control-gallery-preparation-v1','inputs_sha256':'e7ae306ac5282c81e6e9c26bc32844db08b5f264049931df19caa2eaffdfa017','source':j['source'],'partition':j['partition'],'gallery':{'path':str(out),'sha256':hashlib.sha256(gallery).hexdigest()},'count':1715,'rows':3449,'wire_row_bytes':130,'native_imports':[],'quality_read':False,'native_qualified':False}
receipt=out.parent/'preparation.json'
with receipt.open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(record,sort_keys=True))
