"""Parse native cuTile0.1.1 cache-miss timing records."""
import json
import math
import sys
from pathlib import Path


def parse(text):
    records = []
    for line in text.splitlines():
        if 'CUTILE_JIT_TIMING ' not in line:
            continue
        fields = dict(part.split('=', 1) for part in line.split('CUTILE_JIT_TIMING ', 1)[1].split())
        assert set(fields) == {'module', 'function', 'key', 'stage1_ms', 'stage2_ms', 'stage3_ms', 'generics'}
        for name in ['stage1_ms', 'stage2_ms', 'stage3_ms']:
            fields[name] = float(fields[name])
            assert math.isfinite(fields[name]) and fields[name] >= 0
        records.append(fields)
    assert records, 'No cache-miss timing records'
    return records


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        sample = 'test x ... CUTILE_JIT_TIMING module=m function=f key=k stage1_ms=1.200 stage2_ms=2.300 stage3_ms=0.400 generics=1,128'
        assert parse(sample)[0]['stage2_ms'] == 2.3
        try:
            parse(sample.replace('2.300', 'nan'))
        except AssertionError:
            pass
        else:
            raise AssertionError('Nonfinite timing accepted')
        print('PASS native timing parser self-check')
    else:
        print(json.dumps(parse(Path(sys.argv[1]).read_text()), indent=2, allow_nan=False))
