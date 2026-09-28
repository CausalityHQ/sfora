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
out={'receipt_sha256':sha(root/'receipt.json'),'arms':{},'method':'All queries independently scored by NumPy float32 integer-exact dots, declared float32 norm multiplication order, stable sorting; compare every R1/AP and repack saved embeddings.'}
for width,arm in r['arms'].items():
    directory=root/width
    if 'asymmetric_quality' not in arm:continue
    values=np.load(directory/'held_values.npy',allow_pickle=False)
    codes=np.load(directory/'held_codes.npy',allow_pickle=False)
    norms=np.load(directory/'held_inverse_norms.npy',allow_pickle=False)
    assert sha(directory/'held_values.npy')==arm['held_values_sha256']
    for name in ('codes','inverse_norms'):assert sha(directory/f'held_{name}.npy')==arm['held_packed_sha256'][name]
    repacked=pack_int8_unit_embeddings(torch.from_numpy(values))
    assert np.array_equal(codes,repacked.codes.numpy()) and np.array_equal(norms,repacked.inverse_norms.numpy())
    qlabel=labels[query];glabel=labels[gallery]
    counts=np.array([np.sum(glabel==label) for label in qlabel])
    ranks=np.arange(1,int(counts.max())+1,dtype=np.float32)
    gc=codes[gallery].astype(np.float32);gn=norms[gallery].astype(np.float32)
    hits=[];aps=[]
    for start in range(0,len(query),128):
        q=query[start:start+128]
        scores=(codes[q].astype(np.float32)@gc.T)*norms[q,None].astype(np.float32)*gn[None,:]
        order=np.argsort(-scores,axis=1,kind='stable')[:,:len(ranks)]
        match=glabel[order]==labels[q,None]
        n=counts[start:start+len(q)]
        precision=np.cumsum(match,axis=1,dtype=np.float32)/ranks
        ap=(precision*match*(ranks[None,:]<=n[:,None])).sum(axis=1)/n
        hits.extend(match[:,0].astype(int));aps.extend(ap)
    quality=arm['asymmetric_quality']
    assert np.array_equal(hits,quality['per_query_r1'])
    error=float(np.max(np.abs(np.asarray(aps)-quality['per_query_ap'])))
    assert error<2e-7,error
    assert np.mean(hits)==quality['recall_at_1']
    out['arms'][width]={'queries_checked':len(query),'recall_at_1':float(np.mean(hits)),'map_at_r':float(np.mean(aps)),'max_ap_absolute_error':error,'repacked_codes_and_norms_exact':True}
out['seconds']=time.monotonic()-started
(root/'independent-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
