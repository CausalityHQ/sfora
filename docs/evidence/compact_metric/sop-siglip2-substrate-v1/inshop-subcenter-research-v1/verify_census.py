"""Replay the exploratory census; this does not identify a causal mechanism."""
import hashlib
import json
import math
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
partition = Path(sys.argv[1])
raw = root.parent / 'inshop-fit-positive-tail-v1/receipt.json'
assert hashlib.sha256(partition.read_bytes()).hexdigest() == 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c'
assert hashlib.sha256(raw.read_bytes()).hexdigest() == '953be9e6eab846f22e3e187f043eebfaf3161af0bbe4a752f6c68987ffb7fb49'
d = json.loads(raw.read_text())
rows = [x.split() for x in partition.read_text().splitlines()[2:] if x.split()[2] == 'train']
counts = Counter(x[1] for x in rows)
eligible = sorted((x for x, n in counts.items() if n > 1), key=lambda x: (hashlib.sha256(b'inshop-unseen-gallery-v1\0' + x.encode()).digest(), x))
held = set(eligible[:math.ceil(len(eligible) / 2)])
fit = [i for i, x in enumerate(rows) if x[1] not in held]
assert len(rows) == 25882 and len(fit) == 13283
assert hashlib.sha256(b''.join(struct.pack('<q', i) for i in fit)).hexdigest() == d['fit_rows_sha256']
series = defaultdict(set)
for i in fit:
    series[rows[i][1]].add(Path(rows[i][0]).stem.split('_')[0])
assert set(d['query_ordinals']) == {i for i, r in enumerate(fit) if counts[rows[r][1]] > 1}
assert len(d['query_ordinals']) == len(d['margins']) == 13271
assert all(math.isfinite(m) for m in d['margins'])
bins = defaultdict(list)
for i, m in zip(d['query_ordinals'], d['margins'], strict=True):
    name = rows[fit[i]][1]
    bins[(counts[name], len(series[name]))].append(m)
assert sum(m > .03 for v in bins.values() for m in v) == d['tail_count'] == 2506

def summarize(v):
    return {'anchors': len(v), 'tail': sum(x > .03 for x in v), 'fraction': sum(x > .03 for x in v) / len(v), 'median': statistics.median(v)}

report = {
    'schema': 'sfora-subcenter-exploratory-census-v1',
    'claim_eligible': False,
    'partition_sha256': hashlib.sha256(partition.read_bytes()).hexdigest(),
    'receipt_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
    'fit_rows_sha256': d['fit_rows_sha256'],
    'strata': [{'images': n, 'series': s, **summarize(v)} for (n, s), v in sorted(bins.items())],
    'ten_plus_images_three_plus_series': summarize([m for (n, s), v in bins.items() if n >= 10 and s >= 3 for m in v]),
    'nine_ten_images_two_series': summarize([m for (n, s), v in bins.items() if n in (9, 10) and s == 2 for m in v]),
    'limitations': 'Worst-positive order statistic depends on product size; filename series is a metadata proxy; no gradient conflict or modality demonstrated; no feature tensors rescored.',
}
print(json.dumps(report, indent=2, allow_nan=False))
