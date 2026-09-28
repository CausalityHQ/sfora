import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(20)
torch.backends.cuda.matmul.allow_tf32 = False
out = Path('/home/riomus/runs/sfora-pixel-transfer-v1')
assert not (out / 'receipt.json').exists()
records = []
for count in (1, 32):
    # Exact pinned processor output levels, with all 256 levels in every channel.
    x = ((torch.arange(count * 3 * 256 * 256) % 256).float() - 127.5) / 127.5
    x = x.reshape(count, 3, 256, 256)
    def transfer(arm):
        if arm == 'original':
            return x.to('cuda:0').to(torch.float16)
        return x.to(device='cuda:0', dtype=torch.float16)
    a, b = transfer('original'), transfer('candidate')
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    del a, b
    samples = {'original': [], 'candidate': []}
    peaks = {}
    for arm in samples:
        for _ in range(5):
            y = transfer(arm)
            torch.cuda.synchronize()
            del y
        torch.cuda.reset_peak_memory_stats()
        y = transfer(arm)
        torch.cuda.synchronize()
        peaks[arm] = torch.cuda.max_memory_allocated()
        del y
    for block in range(25):
        order = ['original', 'candidate', 'candidate', 'original']
        if block % 2:
            order = ['candidate', 'original', 'original', 'candidate']
        for arm in order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            y = transfer(arm)
            torch.cuda.synchronize()
            samples[arm].append((time.perf_counter_ns() - start) / 1e6)
            assert torch.equal(y.view(torch.int16).cpu(), x.half().view(torch.int16))
            del y
    stats = {arm: {'p50_ms': float(np.quantile(v, .5)), 'p95_ms': float(np.quantile(v, .95))} for arm, v in samples.items()}
    passed = stats['candidate']['p50_ms'] <= .95 * stats['original']['p50_ms'] and stats['candidate']['p95_ms'] <= stats['original']['p95_ms'] and peaks['candidate'] <= peaks['original']
    records.append({'batch': count, 'samples_ms': samples, 'stats': stats, 'peak_allocated_bytes': peaks, 'gate': passed, 'pixel_bits_exact': True})
r = {'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(), 'records': records, 'advance': all(x['gate'] for x in records), 'claim_eligible': False, 'scope': 'synthetic pinned-processor pixel levels; no image decode/encoder/search/full-call timing'}
(out / 'receipt.json').write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({k: v for k, v in r.items() if k != 'records'}))
for row in records:
    print(json.dumps({k: v for k, v in row.items() if k != 'samples_ms'}))
