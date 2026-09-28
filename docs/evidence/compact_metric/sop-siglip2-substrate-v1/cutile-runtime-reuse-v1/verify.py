"""Replay the native JIT receipt without running GPU work."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[4]
sys.path.insert(0, str(repo / 'scripts'))
from parse_cutile_jit_timing import parse

d = json.loads((root / 'receipt.json').read_text())
for name, digest in d['files_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
assert hashlib.sha256((repo / 'scripts/parse_cutile_jit_timing.py').read_bytes()).hexdigest() == d['parser_sha256']
records = parse((root / 'reuse.log').read_text())
assert records == json.loads((root / 'native-jit-records.json').read_text())
assert len({(x['module'], x['function'], x['key']) for x in records}) == len(records)
assert Counter(x['function'] for x in records) == d['counts']
assert Counter(x['generics'] for x in records if x['function'] == 'score_block_topk') == {'1,128,32,128,16,10': 5, '32,128,32,128,16,10': 5}
assert sum(x['function'] == 'score_block_topk' for x in records) <= 10
for stage, total in d['summed_stage_ms'].items():
    assert sum(x[stage] for x in records) == total
assert '1 passed; 0 failed; 0 ignored' in (root / 'reuse.log').read_text()
assert 'Exit status: 0' in (root / 'reuse-time.txt').read_text()
assert d['advance'] and not d['claim_eligible'] and d['wall_seconds'] < 60
print('PASS raw hashes, native cache misses, exact test result and scoped advance')
