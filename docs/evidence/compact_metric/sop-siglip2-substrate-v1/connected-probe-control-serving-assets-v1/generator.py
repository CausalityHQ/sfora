import pathlib,json,hashlib,sys
u=json.load(sys.stdin);p=pathlib.Path(u['receipt']['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==u['receipt']['sha256'];r=json.loads(raw)
assert r['arm']=='control' and r['seed']==179061 and r['phase']=='export'
rows={x['role']:[row for batch in r['images'] if batch['role']==x['role'] for row in batch['rows']] for x in r['images']}
assert len(rows['query'])==1734 and len(rows['gallery'])==1715
allrows=rows['query']+rows['gallery'];assert {x['panel_ordinal'] for x in allrows}==set(range(3449))
wirepath=pathlib.Path(r['output'])/'control-179061.packed.bin';wire=wirepath.read_bytes();assert len(wire)==3449*130 and hashlib.sha256(wire).hexdigest()==r['files'][wirepath.name]
gallery=b''.join(wire[x['panel_ordinal']*130:(x['panel_ordinal']+1)*130] for x in rows['gallery'])
root=pathlib.Path('/home/riomus/runs/sfora-connected-probe-control-serving-gallery-v1');root.mkdir();target=root/'gallery.bin'
with target.open('xb') as stream:stream.write(gallery);stream.flush()
assert target.read_bytes()==gallery
fact=lambda x:{'path':x['path'],'sha256':x['image_sha256']}
tail=next(x['rows'] for x in reversed(r['images']) if x['role']=='query');assert len(tail)==6
out={'schema':'connected-probe-serving-assets-v1','source_export':u,'wire_sha256':r['files'][wirepath.name],'gallery':{'file':{'path':str(target),'sha256':hashlib.sha256(gallery).hexdigest()},'count':1715},'gallery_ordinals':[x['panel_ordinal'] for x in rows['gallery']],'images':{'fixed32':[fact(x) for x in rows['query'][:32]],'tail':{'role':'query','files':[fact(x) for x in tail]}},'quality_or_speed_qualified':False}
(root/'metadata.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
