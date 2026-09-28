import json, os, subprocess, sys, tempfile
from pathlib import Path
root=Path('/home/riomus/runs/sfora-pe-augmented-pair-v1')
args=['--root',str(root),'--cache','/home/riomus/runs/sfora-pe-fullfit-cache-v3',
'--output','/home/riomus/runs/sfora-pe-augmented-100-v2','--dataset-root','/home/riomus/datasets/inshop_official_standard',
'--large-snapshot','/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c',
'--mechanics-dir','/home/riomus/runs/sfora-pe-fp16-smoke-v1',
'--preflight-sha256','41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293','--check-startup-only']
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
command=[sys.executable,str(root/'train_inshop_pe_pair.py')]+args
good=subprocess.run(command,capture_output=True,text=True,timeout=40)
assert good.returncode==0,good.stderr
with tempfile.TemporaryDirectory() as directory:
 copy=Path(directory)/'trainer-copy.py'
 copy.write_bytes((root/'train_inshop_pe_pair.py').read_bytes())
 bad=subprocess.run([sys.executable,str(copy)]+args,capture_output=True,text=True,timeout=40)
 assert bad.returncode!=0 and 'Path(__file__).resolve()' in bad.stderr,bad.stderr
 assert 'torch.cuda.is_available()' not in bad.stderr
print(json.dumps({'valid_root_startup_pass':True,'copied_executing_trainer_rejected_before_cuda':True,'quality_read':False},indent=2))
