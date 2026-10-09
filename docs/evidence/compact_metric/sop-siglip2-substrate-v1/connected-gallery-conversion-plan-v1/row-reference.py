"""Read-only opaque-row reference; no artifact or native qualification."""
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

WIDTH = 130

def row_digest(fd, ordinals):
    if type(ordinals) is not list or any(type(p) is not int or p < 0 for p in ordinals):
        raise ValueError('exact nonnegative ordinals required')
    if ordinals != sorted(set(ordinals)):
        raise ValueError('ordered unique ordinals required')
    digest = hashlib.sha256()
    for p in ordinals:
        row = os.pread(fd, WIDTH, p * WIDTH)
        if len(row) != WIDTH:
            raise ValueError('complete opaque row required')
        digest.update(row)
    return digest.hexdigest()

def self_test():
    wire = b''.join(bytes([i])*WIDTH for i in range(7))
    with tempfile.TemporaryFile() as f:
        f.write(wire); f.flush()
        expected = hashlib.sha256(bytes([1])*WIDTH+bytes([3])*WIDTH+bytes([5])*WIDTH).hexdigest()
        assert row_digest(f.fileno(), [1,3,5]) == expected
        assert hashlib.sha256(wire[-3*WIDTH:]).hexdigest() != expected
        for invalid in ([True], [3,1], [1,1], [7]):
            try: row_digest(f.fileno(), invalid)
            except ValueError: pass
            else: raise AssertionError('invalid row list accepted')

def observe():
    root = Path('/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-control-179061-v2')
    receipt = root/'receipt.json'; raw = receipt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407'
    r = json.loads(raw)
    assert r['phase']=='export' and r['stage']=='full' and r['arm']=='control' and r['seed']==179061
    batches = [b for b in r['images'] if b['role']=='gallery']
    rows = [row for batch in batches for row in batch['rows']]
    ordinals = [row['panel_ordinal'] for row in rows]
    all_ordinals = [row['panel_ordinal'] for batch in r['images'] for row in batch['rows']]
    assert len(rows)==1715 and [len(b['rows']) for b in batches]==[32]*53+[19]
    assert sorted(all_ordinals)==list(range(3449))
    ids = [row['relative_path'] for row in rows]
    assert all(type(i) is str and i for i in ids) and len(set(ids))==1715
    wire = root/'control-179061.packed.bin'
    assert wire.resolve()==wire and not wire.is_symlink()
    fd = os.open(wire, os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        assert stat.S_ISREG(before.st_mode) and before.st_size==448370
        original = hashlib.sha256()
        for offset in range(0, before.st_size, 65536): original.update(os.pread(fd, 65536, offset))
        assert original.hexdigest()=='c2260ca93cedf8ae5c0840034d2ed06d10860539d33df562fe82a7eac30b1558'
        selected = row_digest(fd, ordinals)
        wrong_suffix = hashlib.sha256(os.pread(fd,222950,448370-222950)).hexdigest()
        assert selected != wrong_suffix
        after=os.fstat(fd); current=wire.stat()
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        assert fields(before)==fields(after)==fields(current)
    finally: os.close(fd)
    assert receipt.read_bytes()==raw
    print(json.dumps({'schema':'connected-gallery-row-reference-v1','original_receipt_sha256':hashlib.sha256(raw).hexdigest(),'combined_wire_sha256':original.hexdigest(),'gallery_count':1715,'gallery_bytes':222950,'gallery_wire_sha256':selected,'consumer_relative_path_ids_sha256':hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest(),'suffix_mutant_rejected':True,'artifact_written':False,'native_execution':False,'serving_accepted':False},sort_keys=True))

if __name__=='__main__':
    self_test()
    if os.environ.get('SFORA_ROW_REFERENCE_REMOTE')=='1': observe()
    else: print('opaque-row reference self-test PASS; no original data read')
