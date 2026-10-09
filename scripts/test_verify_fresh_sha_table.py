#!/usr/bin/env python3
"""Stdlib-only gate: <=55s, AS<=1GiB; real advice only on tiny tempfiles."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

HELPER = Path(__file__).with_name('verify_fresh_sha_table.py')
PAGE = os.sysconf('SC_PAGE_SIZE')
CHUNK = 1024 * 1024


def row(path, data):
    return (hashlib.sha256(data).hexdigest() + '  ' + str(path) + '\n').encode('utf-8')


def load_helper():
    spec = importlib.util.spec_from_file_location('verify_fresh_sha_table', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, table, *args):
        return subprocess.run([sys.executable, '-I', '-B', '-S', str(HELPER), *args],
                              input=table, capture_output=True, timeout=5)

    def file(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_cli_matches_sha256sum_and_preserves_ordered_duplicates(self):
        files = [(self.file('empty', b''), b''),
                 (self.file('partial page', b'p' * (PAGE + 7)), b'p' * (PAGE + 7)),
                 (self.file('multichunk', b'm' * (2 * CHUNK + 17)), b'm' * (2 * CHUNK + 17))]
        ordered = [files[2], files[0], files[2], files[1], files[0]]
        table = b''.join(row(path, data) for path, data in ordered)
        original = subprocess.run(['sha256sum', '-c'], input=table, capture_output=True,
                                  timeout=5, check=True)
        result = self.cli(table)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, b'')
        self.assertEqual(result.stdout, original.stdout)

    def test_cli_utf8_literal_path_and_final_row_without_lf(self):
        path = self.file('café with spaces', b'hello')
        result = self.cli(row(path, b'hello').rstrip(b'\n'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, (str(path) + ': OK\n').encode())

    def test_cli_mismatch_stops_after_prior_success(self):
        first = self.file('first', b'good')
        bad = self.file('bad', b'changed')
        last = self.file('last', b'last')
        result = self.cli(row(first, b'good') + row(bad, b'expected') + row(last, b'last'))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, (str(first) + ': OK\n').encode())
        self.assertIn(b'mismatch', result.stderr)

    def test_cli_missing_unreadable_and_directory_fail(self):
        unreadable = self.file('unreadable', b'x')
        unreadable.chmod(0)
        self.addCleanup(unreadable.chmod, 0o600)
        for path in (self.root / 'missing', unreadable, self.root):
            with self.subTest(path=path):
                result = self.cli(row(path, b'x'))
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b'')
                self.assertTrue(result.stderr)

    def test_cli_rejects_arguments_empty_input_and_malformed_records(self):
        path = self.file('good', b'x')
        good = row(path, b'x')
        for table in (b'', b'\n', good[:64].upper() + good[64:], good[:64] + b' *' + good[66:],
                      good[:64] + b' ' + good[66:], good[:64] + b'  relative\n',
                      b'g' + good[1:], good[:-1] + b'\r\n', good[:-1] + b'\x00\n',
                      good[:64] + b'  /bad\xff\n', b'0' * 64 + b'  /' + b'x' * 5000 + b'\n'):
            with self.subTest(table=table[:80]):
                result = self.cli(table)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stdout, b'')
                self.assertTrue(result.stderr)
        result = self.cli(good, '--anything')
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b'')
        self.assertTrue(result.stderr)

    def test_record_error_keeps_prior_success_and_never_opens_later_record(self):
        helper = load_helper()
        first = self.file('first', b'good')
        output = io.StringIO()
        with self.assertRaises(ValueError):
            helper.verify_table(io.BytesIO(row(first, b'good') + b'malformed\n' + row('/missing', b'')), output)
        self.assertEqual(output.getvalue(), str(first) + ': OK\n')

    def test_duplicate_occurrence_reopens_after_same_size_restored_mtime_mutation(self):
        helper = load_helper()
        path = self.file('mutable', b'old!')
        stat = path.stat()

        class MutatingOutput(io.StringIO):
            def flush(self):
                path.write_bytes(b'new!')
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        output = MutatingOutput()
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            helper.verify_table(io.BytesIO(row(path, b'old!') * 2), output)
        self.assertEqual(output.getvalue(), str(path) + ': OK\n')
        self.assertEqual(path.stat().st_size, stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)

    def test_short_reads_hash_every_byte_and_advise_only_consumed_full_pages(self):
        helper = load_helper()
        total = 16 * CHUNK + PAGE + 31

        class ShortReader:
            position = 0
            calls = 0
            buffers = set()
            eof = False
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def fileno(self):
                return 42
            def readinto(self, buffer):
                self.buffers.add(id(buffer))
                self.calls += 1
                self_test.assertEqual(len(buffer), CHUNK)
                sizes = (17, PAGE - 22, 5, PAGE - 6, 999, CHUNK - 17)
                count = min(sizes[(self.calls - 1) % len(sizes)], total - self.position)
                self.eof = count == 0
                buffer[:count] = b'x' * count
                self.position += count
                return count

        reader = ShortReader()
        self_test = self
        advised = []
        def advice(fd, start, length, flag):
            self.assertEqual((fd, flag), (42, os.POSIX_FADV_DONTNEED))
            self.assertEqual(start, sum(advised))
            self.assertGreater(length, 0)
            self.assertEqual(start % PAGE, 0)
            self.assertEqual(length % PAGE, 0)
            self.assertLessEqual(start + length, reader.position)
            advised.append(length)

        expected = hashlib.sha256()
        for size in [CHUNK] * 16 + [PAGE + 31]:
            expected.update(b'x' * size)
        tracemalloc.start()
        try:
            with patch('builtins.open', return_value=reader), patch.object(helper.os, 'posix_fadvise', side_effect=advice):
                actual = helper.hash_file('/fake', bytearray(CHUNK), PAGE)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertEqual(actual, expected.hexdigest())
        self.assertEqual(reader.position, total)
        self.assertTrue(reader.eof)
        self.assertEqual(sum(advised), total - total % PAGE)
        self.assertEqual(len(reader.buffers), 1)
        self.assertLess(peak, 4 * CHUNK)

    def test_empty_and_partial_page_files_do_not_advise_unread_tail(self):
        helper = load_helper()
        for data in (b'', b'a' * (PAGE - 1), b'a' * PAGE, b'a' * (PAGE + 1)):
            with self.subTest(size=len(data)):
                path = self.file('tail', data)
                with patch.object(helper.os, 'posix_fadvise') as advice:
                    self.assertEqual(helper.hash_file(str(path), bytearray(CHUNK), PAGE), hashlib.sha256(data).hexdigest())
                ranges = [(call.args[1], call.args[2]) for call in advice.call_args_list]
                self.assertEqual(ranges, [(0, PAGE)] if len(data) >= PAGE else [])

    def test_all_occurrences_share_buffer_but_hash_afresh(self):
        helper = load_helper()
        path = self.file('repeated', b'x')
        original = helper.hash_file
        buffers = []
        def hash_file(name, buffer, page_size):
            buffers.append(buffer)
            return original(name, buffer, page_size)
        with patch.object(helper, 'hash_file', side_effect=hash_file):
            helper.verify_table(io.BytesIO(row(path, b'x') * 3), io.StringIO())
        self.assertEqual(len(buffers), 3)
        self.assertTrue(all(buffer is buffers[0] for buffer in buffers))
        self.assertEqual(len(buffers[0]), CHUNK)

    def test_open_read_and_advice_errors_propagate_without_success(self):
        helper = load_helper()
        path = self.file('page', b'x' * PAGE)
        for operation in ('open', 'read', 'advice'):
            with self.subTest(operation=operation):
                output = io.StringIO()
                if operation == 'open':
                    context = patch('builtins.open', side_effect=PermissionError('denied'))
                elif operation == 'read':
                    stream = unittest.mock.MagicMock()
                    stream.__enter__.return_value = stream
                    stream.readinto.side_effect = OSError('read failure')
                    context = patch('builtins.open', return_value=stream)
                else:
                    context = patch.object(helper.os, 'posix_fadvise', side_effect=OSError('advice failure'))
                with context, self.assertRaises(OSError):
                    helper.verify_table(io.BytesIO(row(path, b'x' * PAGE)), output)
                self.assertEqual(output.getvalue(), '')

    def test_parser_bounds_readline_and_rejects_oversized_before_open(self):
        helper = load_helper()
        class Input:
            def readline(self, limit):
                self_test.assertGreater(limit, 0)
                self_test.assertLessEqual(limit, 4164)
                return b'x' * limit
        self_test = self
        with patch('builtins.open', side_effect=AssertionError('opened malformed row')):
            with self.assertRaises(ValueError):
                helper.verify_table(Input(), io.StringIO())

    def test_stdin_and_stdout_errors_propagate(self):
        helper = load_helper()
        path = self.file('good', b'x')
        stream = unittest.mock.Mock()
        stream.readline.side_effect = OSError('stdin failure')
        with self.assertRaises(OSError):
            helper.verify_table(stream, io.StringIO())
        output = unittest.mock.Mock()
        output.write.side_effect = OSError('stdout failure')
        with self.assertRaises(OSError):
            helper.verify_table(io.BytesIO(row(path, b'x')), output)

    def test_main_reports_read_and_advice_errors_and_preserves_prior_success(self):
        helper = load_helper()
        first = self.file('first', b'good')
        path = self.file('page', b'x' * PAGE)
        for operation in ('read', 'advice'):
            with self.subTest(operation=operation):
                output_bytes = io.BytesIO()
                output = io.TextIOWrapper(output_bytes, encoding='utf-8')
                errors = io.StringIO()
                source = unittest.mock.Mock(buffer=io.BytesIO(row(first, b'good') + row(path, b'x' * PAGE)))
                if operation == 'read':
                    original_open = open
                    def open_file(name, *args, **kwargs):
                        if name == str(first):
                            return original_open(name, *args, **kwargs)
                        stream = unittest.mock.MagicMock()
                        stream.__enter__.return_value = stream
                        stream.readinto.side_effect = OSError('read failure')
                        return stream
                    context = patch('builtins.open', side_effect=open_file)
                else:
                    context = patch.object(helper.os, 'posix_fadvise', side_effect=OSError('advice failure'))
                with context, patch.object(helper.sys, 'argv', [str(HELPER)]), \
                     patch.object(helper.sys, 'stdin', source), patch.object(helper.sys, 'stdout', output), \
                     patch.object(helper.sys, 'stderr', errors):
                    self.assertEqual(helper.main(), 1)
                self.assertEqual(output_bytes.getvalue(), (str(first) + ': OK\n').encode())
                self.assertIn(operation + ' failure', errors.getvalue())
                output.close()


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
    signal.alarm(55)
    unittest.main()
