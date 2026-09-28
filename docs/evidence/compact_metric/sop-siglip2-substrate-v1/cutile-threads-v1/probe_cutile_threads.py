"""Sequential same-handle thread coverage; synthetic geometry, no speed claim."""
import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

from probe_cutile_runtime_bounds import fixture, oracle, check
from sfora.cutile_int8 import CutilePackedInt8Gallery


def run_threads(prepared, q, qn, expected):
    records, errors = [], []
    first_done, release = threading.Event(), threading.Event()

    def worker(index):
        try:
            for batch in [1, 32]:
                for phase in ['first', 'repeat']:
                    print(f'THREAD_{index}_B{batch}_{phase}', file=sys.stderr, flush=True)
                    start = time.perf_counter_ns()
                    result = prepared.search(q[:batch], qn[:batch])
                    elapsed = time.perf_counter_ns() - start
                    check(result, (expected[0][:batch], expected[1][:batch]))
                    records.append({'thread': index, 'native_id': threading.get_native_id(),
                                    'batch': batch, 'phase': phase, 'search_ns': elapsed,
                                    'ordinals': result[0].tolist(),
                                    'score_bits': result[1].view('u4').tolist()})
        except BaseException as error:
            errors.append(error)
        finally:
            if index == 1:
                first_done.set()
                release.wait(30)

    first = threading.Thread(target=worker, args=(1,))
    second = threading.Thread(target=worker, args=(2,))
    first.start()
    try:
        assert first_done.wait(30), 'first worker deadline'
        if errors:
            raise errors[0]
        second.start(); second.join(30)
        assert not second.is_alive(), 'second worker deadline'
        if errors:
            raise errors[0]
    finally:
        release.set(); first.join(30)
    assert len(records) == 8
    assert len({r['native_id'] for r in records}) == 2
    return records


def main():
    if sys.argv[1:] == ['--self-test']:
        g, gn, q, qn = fixture(64, 32)
        expected = oracle(g, gn, q, qn)
        class Fake:
            def search(self, q, qn):
                order, bits = oracle(g, gn, q, qn)
                return order, bits.view('f4')
        assert len(run_threads(Fake(), q, qn, expected)) == 8
        class Broken:
            def search(self, q, qn):
                raise ValueError('deliberate worker failure')
        try:
            run_threads(Broken(), q, qn, expected)
        except ValueError as error:
            assert str(error) == 'deliberate worker failure'
        else:
            raise AssertionError('worker error swallowed')
        print('PASS two distinct threads, sequential checks and retained outputs')
        return
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.library.is_absolute() and args.library.is_file()
    assert not args.output.exists() and not args.output.is_symlink()
    g, gn, q, qn = fixture(59551, 32)
    expected = oracle(g, gn, q, qn)
    with CutilePackedInt8Gallery.open(args.library, g, gn) as prepared:
        records = run_threads(prepared, q, qn, expected)
    report = {'claim_eligible': False, 'gallery_rows': len(g),
              'library_sha256': hashlib.sha256(args.library.read_bytes()).hexdigest(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'records': records}
    with args.output.open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False); output.write('\n')
    print('PASS eight exact searches on two distinct successive threads', flush=True)


if __name__ == '__main__':
    main()
