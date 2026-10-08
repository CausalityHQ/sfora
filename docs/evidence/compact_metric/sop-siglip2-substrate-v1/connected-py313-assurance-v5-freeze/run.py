import hashlib,json,pathlib,sys,runpy
layer_only = sys.argv[1:] == ['--layer']
root=pathlib.Path(__file__).resolve().parent
assert sys.version.split()[0]=='3.13.9'
assert hashlib.sha256(pathlib.Path(sys.executable).read_bytes()).hexdigest()=='9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'
for name,fact in json.loads((root/'inventory.json').read_bytes()).items():
 p=root/name
 assert not p.is_symlink() and p.is_file() and p.stat().st_size==fact['size'],name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==fact['sha256'],name
sys.path.insert(0,str(root/'scripts'))
sys.argv=[str(root/'scripts/test_connected_mlp_evaluation.py'),'--source-only']
if layer_only:
 layer=runpy.run_path(sys.argv[0],run_name='_bootstrap_failing_layer')
 layer['bootstrap_prerequisite_contract']()
 print('REPAIRED_BOOTSTRAP_LAYER_PASS',flush=True)
 assert not {'torch','numpy','PIL','transformers','sfora'} & sys.modules.keys()
 sys.exit(0)
runpy.run_path(sys.argv[0],run_name='__main__')
assert not {'torch','numpy','PIL','transformers','sfora'} & sys.modules.keys()
print('PINNED_INTERPRETER_SOURCE_ONLY_PASS',flush=True)
