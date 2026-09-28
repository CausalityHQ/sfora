"""Matched native-search diagnostic; synthetic geometry, no dataset quality."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sfora.cutile_int8 import CutilePackedInt8Gallery


def norms(codes):
    squared = np.square(codes.astype(np.int32)).sum(axis=1)
    assert (squared > 0).all()
    return np.ascontiguousarray((1 / np.sqrt(squared)).astype('<f2'))


def fixture(rows, batch):
    q, c = np.indices((32, 128))
    queries = (((q * 47 + c * 73 + (q ^ c) * 11) % 255) - 127).astype('i1')
    queries[0] = np.where(np.arange(128) % 2, -127, 127)
    r, c = np.indices((rows, 128))
    gallery = (((r * 29 + c * 31 + (r ^ c) * 11) % 255) - 127).astype('i1')
    for start in [0, 16384, 32768, 49152, rows - 32]:
        if 0 <= start and start + 32 <= rows:
            gallery[start:start + 32] = queries
    if rows > 501:
        gallery[500], gallery[501] = -queries[0], np.ones(128, dtype='i1')
    queries = np.ascontiguousarray(queries[:batch])
    return gallery, norms(gallery), queries, norms(queries)


def oracle(gallery, gn, queries, qn):
    scores = (queries.astype(np.int32) @ gallery.astype(np.int32).T).astype('f4')
    scores *= qn.astype('f4')[:, None]
    scores *= gn.astype('f4')[None, :]
    order = np.argsort(-scores, axis=1, kind='stable')[:, :10]
    return order, np.take_along_axis(scores, order, axis=1).view('u4')


def check(observed, expected):
    ordinals, scores = observed
    assert np.array_equal(ordinals, expected[0]), 'ordinal parity failed'
    assert np.array_equal(scores.view('u4'), expected[1]), 'score-bit parity failed'


def main():
    if sys.argv[1:] == ['--self-test']:
        g, gn, q, qn = fixture(64, 32)
        expected = oracle(g, gn, q, qn)
        assert expected[0][0, :2].tolist() == [0, 32]
        check((expected[0], expected[1].view('f4')), expected)
        corrupt = expected[1].copy(); corrupt[0, 0] ^= 1
        try:
            check((expected[0], corrupt.view('f4')), expected)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt score bits accepted')
        tied = np.ones_like(g)
        tie = oracle(tied, norms(tied), q, qn)
        assert np.array_equal(tie[0], np.broadcast_to(np.arange(10), (32, 10)))
        assert len({row.tobytes() for row in q}) == 32
        g, gn, q, qn = fixture(59551, 32)
        full = oracle(g, gn, q, qn)
        for query in range(32):
            assert full[0][query, :5].tolist() == [query, 16384 + query,
                32768 + query, 49152 + query, 59519 + query]
        dots = q[0].astype('i4') @ g[[500, 501]].astype('i4').T
        assert dots[0] < 0 and dots[1] == 0
        print('PASS distinct queries, stable ties and score-bit rejection')
        return
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=[1, 32], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (a.output.exists() or a.output.is_symlink() or not a.library.is_absolute()
            or not a.library.is_file()):
        raise ValueError('library/output authority differs')
    g, gn, q, qn = fixture(59551, a.batch)
    expected = oracle(g, gn, q, qn)
    report = {'schema': 'sfora-runtime-bounds-native-v1', 'claim_eligible': False,
              'workload': 'synthetic SOP-TRAIN-sized geometry; no images or quality',
              'gallery_rows': len(g), 'batch': a.batch, 'warmups': 5,
              'fixture_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest()
                                 for k, v in [('gallery', g), ('gallery_norms', gn), ('queries', q), ('query_norms', qn)]},
              'library_sha256': hashlib.sha256(a.library.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    print('PHASE_CREATE', file=sys.stderr, flush=True)
    start = time.perf_counter_ns()
    with CutilePackedInt8Gallery.open(a.library, g, gn) as prepared:
        report['create_ns'] = time.perf_counter_ns() - start
        print('PHASE_FIRST', file=sys.stderr, flush=True)
        start = time.perf_counter_ns(); result = prepared.search(q, qn)
        report['first_search_ns'] = time.perf_counter_ns() - start
        check(result, expected)
        checked = [[result[0].tolist(), result[1].view('u4').tolist()]]
        print('PHASE_HOT', file=sys.stderr, flush=True)
        samples = []
        for i in range(55):
            start = time.perf_counter_ns(); result = prepared.search(q, qn)
            elapsed = time.perf_counter_ns() - start
            check(result, expected)
            checked.append([result[0].tolist(), result[1].view('u4').tolist()])
            if i >= 5:
                samples.append(elapsed)
        report['raw_search_ns'] = samples
        report['checked_main_outputs'] = checked
    print('PHASE_TIES', file=sys.stderr, flush=True)
    tied = np.ones_like(g); tn = norms(tied)
    tie = oracle(tied, tn, q, qn)
    with CutilePackedInt8Gallery.open(a.library, tied, tn) as prepared:
        check(prepared.search(q, qn), tie)
    report.update(checked_searches=57, ordinals=expected[0].tolist(), score_bits=expected[1].tolist(),
                  tie_ordinals=tie[0].tolist(), tie_score_bits=tie[1].tolist())
    with a.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'batch': a.batch, 'checked_searches': 57, 'exact': True}), flush=True)


if __name__ == '__main__':
    main()
