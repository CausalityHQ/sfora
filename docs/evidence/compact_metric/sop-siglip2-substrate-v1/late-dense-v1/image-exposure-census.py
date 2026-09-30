import ast,collections,hashlib,json,pathlib
import numpy as np
root=pathlib.Path('/home/riomus/runs/sfora-late-dense-source-v4')
manifest=root/'late-dense-execution.json'; code=json.loads(manifest.read_text()); source=root/'pe_large_coverage.py'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3'
assert hashlib.sha256(source.read_bytes()).hexdigest()==code[source.name]
p=pathlib.Path('/home/riomus/runs/sfora-large-teacher-fit-targets-v1/receipt.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='277f9b774470355f76c16675a9c7808abe602c83612d83f4d90e4d37cb5507fe'
rows=json.loads(p.read_text())['fit_manifest'];assert len(rows)==13283
names=sorted({r['product'] for r in rows});assert len(names)==2004
lookup={v:i for i,v in enumerate(names)}; target=np.array([lookup[r['product']] for r in rows])
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='schedule')
node.args.defaults=[ast.Constant(179032)];ns={'np':np};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),ns)
out={'schema':'fit-image-exposure-census-v1','source_execution_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'sampler_source_sha256':code[source.name],'fit_receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fit_images':13283,'fit_products':2004,'quality_read':False,'model_execution':False,'seeds':{}}
for seed in (179032,179041):
 b=ns['schedule'](target,seed);flat=b.ravel();uses=collections.Counter(map(int,target[flat]));members=collections.Counter(map(int,target))
 unique=len(set(map(int,flat))); ceiling=sum(min(n,members[c]) for c,n in uses.items())
 out['seeds'][str(seed)]={'slots':int(b.size),'unique_images':unique,'repeated_slots':int(b.size)-unique,'max_unique_same_class_order':ceiling,'avoidable_repeat_slots':ceiling-unique,'identities_exposed':len(uses),'class_slot_counts':dict(collections.Counter(uses.values())),'schedule_int64_bytes_sha256':hashlib.sha256(b.astype('<i8').tobytes()).hexdigest()}
print(json.dumps(out,indent=2))
