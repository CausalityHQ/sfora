import ast, hashlib, json, os, tempfile, types
from pathlib import Path, PurePosixPath
results=[]
for filename,name,args in [('connected_artifact_identity.py','_installed_sha',()),('connected_installed_environment.py','_read_file',('0'*64,None))]:
 raw=Path('src/sfora',filename).read_bytes();tree=ast.parse(raw)
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
 with tempfile.TemporaryDirectory() as tmp:
  leaf=Path(tmp,'tiny');leaf.write_bytes(b'x')
  opened=[];foreign=[];closed=[];primary=RuntimeError('directory-close injection')
  def opening(*a,**kw):
   fd=os.open(*a,**kw);opened.append(fd);return fd
  def closing(fd):
   os.close(fd);closed.append(fd)
   if len(closed)==1:
    replacement=os.open(leaf,os.O_RDONLY);foreign.append(replacement)
    assert replacement==fd
    raise primary
  proxy=types.SimpleNamespace(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
  proxy.open=opening;proxy.close=closing
  ns={'os':proxy,'PurePosixPath':PurePosixPath}
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(filename),'exec'),ns)
  caught=None
  try:ns[name](PurePosixPath(leaf),*args)
  except BaseException as error:caught=error
  def live(fd):
   try:os.fstat(fd);return True
   except OSError:return False
  leaked=[fd for fd in opened if live(fd)]
  row={'source':filename,'source_sha256':hashlib.sha256(raw).hexdigest(),'function':name,'primary_preserved':caught is primary,'foreign_descriptor_closed':not live(foreign[0]),'leaked_child_descriptors':len(leaked)}
  assert row['primary_preserved'] and row['foreign_descriptor_closed'] and row['leaked_child_descriptors']==1
  results.append(row)
  for fd in set(opened+foreign):
   if live(fd):os.close(fd)
print(json.dumps({'schema':'directory-handoff-source-falsifier-v1','synthetic_files_only':True,'results':results},indent=2))
