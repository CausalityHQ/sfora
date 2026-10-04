"""One read-only source-path observation; never a qualification receipt."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path('/home/riomus/runs/sfora-so400-identity-diversity-view-source-v1')
SOURCE_SHA = 'c7bfb5041e889682a37198d60a3d2adb93bc51db146b5167bc8bb6f1199de9ce'
path = ROOT / 'export_siglip2_identity_diversity_views.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA
spec = importlib.util.spec_from_file_location('_diversity_path_observation', path)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
args = driver.parser().parse_args(['--execution-sha256', '58df68f86640821f410268d76b1f1978024c4403df7129e9afa86a5b244eaeeb', '--authority', str(ROOT / 'authority.json'), '--authority-sha256', '890f4035eebe4ff1cb4ecf04155583254d4fcab0c80a05e347f4e6d3d1300cd9', '--arm', 'candidate', '--phase', 'startup', '--output', '/home/riomus/runs/sfora-identity-diversity-path-observation-only-v1'])

def trace(frame, event, arg):
    if event == 'call':
        return trace if frame.f_code.co_name == 'canonical' else None
    if event == 'exception' and frame.f_code.co_name == 'canonical':
        value = frame.f_locals.get('path')
        print(json.dumps({'schema': 'diversity-canonical-path-observation-v1', 'path': str(value), 'exception': str(arg[1]), 'file': frame.f_code.co_filename, 'line': frame.f_lineno, 'stack': [{'file': x.filename, 'line': x.lineno, 'name': x.name} for x in traceback.extract_stack(frame)], 'qualification_eligible': False}), flush=True)
    return trace

sys.settrace(trace)
try:
    context = driver.authority(args)
    print(json.dumps({'authority_completed': True, 'qualification_eligible': False}), flush=True)
finally:
    sys.settrace(None)
