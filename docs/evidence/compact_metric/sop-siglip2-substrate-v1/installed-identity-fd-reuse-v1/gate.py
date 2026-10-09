import hashlib,json,pathlib,resource,subprocess,time
root=pathlib.Path.cwd()
out=root/"docs/evidence/compact_metric/sop-siglip2-substrate-v1/installed-identity-fd-reuse-v1"
scratch=pathlib.Path("/tmp/sfora-identity-fd-reuse-static")
scratch.mkdir(exist_ok=True)
source=root/"src/sfora/connected_artifact_identity.py"
standalone=scratch/source.name
standalone.write_bytes(source.read_bytes())
cmds={
"tests-py312":["/home/rb/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12","-I","-B","scripts/test_connected_artifact_identity.py","-v"],
"tests-py314":["/usr/bin/python3.14","-I","-B","scripts/test_connected_artifact_identity.py","-v"],
"ruff-check":["/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff","check","--no-cache",str(source),"scripts/test_connected_artifact_identity.py"],
"ruff-format":["/data/cache/uv/archive-v0/FGM38RAsHizKx-KP/bin/ruff","format","--check","--no-cache",str(source),"scripts/test_connected_artifact_identity.py"],
"mypy-source":["/data/cache/uv/archive-v0/A5hAhU0uya3PTOGc/bin/mypy","--config-file",str(root/"pyproject.toml"),"--cache-dir",str(scratch/"mypy"),str(standalone)]}
results={}
for name,cmd in cmds.items():
 start=time.monotonic()
 with (out/(name+".log")).open("wb") as log:
  result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=30)
 (out/(name+".exit")).write_text(str(result.returncode)+"\n")
 results[name]={"argv":cmd,"exit":result.returncode,"seconds":time.monotonic()-start}
 print(name,result.returncode,flush=True)
report={"schema":"identity-fd-reuse-source-gate-v1","results":results,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"test_sha256":hashlib.sha256((root/"scripts/test_connected_artifact_identity.py").read_bytes()).hexdigest(),"limits":{"outer_seconds":120,"address_bytes":1073741824},"native_qualified":False,"product_go":False}
(out/"final-gate.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
raise SystemExit(any(r["exit"] for r in results.values()))
