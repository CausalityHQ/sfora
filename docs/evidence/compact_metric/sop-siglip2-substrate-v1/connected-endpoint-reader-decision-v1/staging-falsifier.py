"""Stdlib feasibility only; does not authenticate deployed modules or qualify native code."""
import ast, hashlib, json, os, re, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
source=Path(__file__).resolve().parents[5]/'scripts/train_siglip2_substrate_adaptation.py'
tree=ast.parse(source.read_bytes())
ns=dict(Path=Path,hashlib=hashlib,json=json,os=os,re=re)
names={'require','strict_json','bound_file','FlatAdmission'}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in names],type_ignores=[]),str(source),'exec'),ns)
Flat=ns['FlatAdmission']; require=ns['require']
def staged(reader,guards,items):
    items=list(items)
    def one(item):
        fresh=Flat(); private={}
        path=fresh.bound_file(private,*item)
        return path,fresh.entries[str(path)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(one,items))
    entries=dict(reader.entries); verified=set(reader.verified); owner=dict(guards)
    for path,fact in results:
        key=str(path)
        require(entries.setdefault(key,fact)==fact,'reader size/digest conflict')
        require(owner.setdefault(key,fact[0])==fact[0],'owner digest conflict')
        verified.add(key)
    reader.entries.update(entries); reader.verified.update(verified); guards.update(owner)
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'a.json'; p.write_bytes(b'{"x":1}')
    q=Path(d)/'b'; q.write_bytes(b'hello')
    hp=hashlib.sha256(p.read_bytes()).hexdigest(); hq=hashlib.sha256(q.read_bytes()).hexdigest()
    serial=Flat(); parallel=Flat(); sg={}; pg={}
    for r,g in ((serial,sg),(parallel,pg)): assert r.read_json(p,hp,g)=={'x':1}
    items=[(p,hp),(q,hq),(q,hq)]
    for path,h in items: serial.bound_file(sg,path,h)
    staged(parallel,pg,items)
    assert (serial.entries,serial.verified,serial.json_bytes,sg)==(parallel.entries,parallel.verified,parallel.json_bytes,pg)
    snapshot=(dict(parallel.entries),set(parallel.verified),dict(parallel.json_bytes),dict(pg))
    q.write_bytes(b'jello')
    try: staged(parallel,pg,items)
    except ValueError: pass
    else: raise AssertionError('fresh current-byte mutation accepted')
    assert snapshot==(parallel.entries,parallel.verified,parallel.json_bytes,pg)
    q.write_bytes(b'hello'); parallel.entries[str(q)]=(hq,999)
    try: staged(parallel,pg,items)
    except ValueError: pass
    else: raise AssertionError('reader size conflict accepted')
print('PASS: actual extracted class success-state equality, JSON-cache preservation, duplicate fresh reads, restored-size mutation rejection and atomic failure; module authentication/native speed UNTESTED')
