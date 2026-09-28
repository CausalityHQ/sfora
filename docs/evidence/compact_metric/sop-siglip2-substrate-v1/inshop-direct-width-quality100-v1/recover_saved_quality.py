import hashlib,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import torch
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

root=Path(sys.argv[1]); started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=json.loads((root/'receipt.json').read_text())
partition=Path('/tmp/sfora-inshop-partition-replay.txt')
assert sha(partition)=='cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'
rows=[line.split() for line in partition.read_text().splitlines()[2:] if line.split()[2]=='train']
counts=Counter(x[1] for x in rows)
eligible=sorted((x for x,n in counts.items() if n>1),key=lambda x:(hashlib.sha256(b'inshop-unseen-gallery-v1\0'+x.encode()).digest(),x))
names=set(eligible[:(len(eligible)+1)//2])
held=[i for i,x in enumerate(rows) if x[1] in names]
labels=np.array([rows[i][1] for i in held])
grouped=defaultdict(list)
for i,label in enumerate(labels):grouped[label].append(i)
query=[];gallery=[]
for label in sorted(grouped):
    ordered=sorted(grouped[label],key=lambda i:hashlib.sha256(('Img/'+rows[held[i]][0]).encode()).digest())
    n=max(1,min(len(ordered)-1,round(len(ordered)/2)))
    gallery.extend(ordered[:n]);query.extend(ordered[n:])
query=sorted(query);gallery=sorted(gallery)
assert (len(query),len(gallery),len(held))==(6354,6245,12599)
from compare_inshop_sop_warmstart_100 import packed_quality
from score_inshop_crop_view_pair import bootstrap_lower
values=np.load(root/'256/held_values.npy',allow_pickle=False)
assert values.shape==(12599,256) and values.dtype==np.float32 and np.isfinite(values).all()
packed=pack_int8_unit_embeddings(torch.from_numpy(values))
for name,array in (('codes',packed.codes.numpy()),('inverse_norms',packed.inverse_norms.numpy())):np.save(root/'256'/f'held_{name}.npy',array)
quality=packed_quality(values,tuple(labels),query,gallery,device=torch.device('cpu'))
arm={'held_values_sha256':sha(root/'256/held_values.npy'),'held_packed_sha256':{name:sha(root/'256'/f'held_{name}.npy') for name in ('codes','inverse_norms')},'asymmetric_quality':quality}
r['arms']['256']=arm
r['diagnostic_only']=True
r['original_gate_decision']='KILL_DIRECT_WIDTH_EXECUTION'
r['original_receipt_sha256']=sha(root/'receipt.json')
r['diagnostic_seconds']=time.monotonic()-started
r['deltas']={}
for name,field,floor in (('recall','per_query_r1',.005),('map','per_query_ap',.01)):
    delta=np.array(quality[field])-np.array(r['arms']['128']['asymmetric_quality'][field])
    lo=bootstrap_lower(delta,labels[query]);hi=-bootstrap_lower(-delta,labels[query])
    r['deltas'][name]={'point':float(delta.mean()),'lower95':lo,'upper95':hi,'quality_floor_pass':bool(delta.mean()>=floor and lo>0)}
dest=root/'diagnostic';dest.mkdir(exist_ok=False)
for width in ('128','256'):(dest/width).symlink_to(root/width,target_is_directory=True)
(dest/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'diagnostic_seconds':r['diagnostic_seconds'],'native':r['arms']['128']['asymmetric_quality']['recall_at_1'],'candidate':quality['recall_at_1'],'native_map':r['arms']['128']['asymmetric_quality']['map_at_r'],'candidate_map':quality['map_at_r'],'deltas':r['deltas']}))
