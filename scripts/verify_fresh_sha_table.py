#!/usr/bin/env python3
"""Fresh ordered checks: python3 -I -B -S verify_fresh_sha_table.py < table.

Rows are lowercase SHA256, two spaces, an absolute UTF-8 path, and optional
final LF; paths are literal (no escaped-name decoding), <=4096 bytes, without
NUL/CR/LF. Rows are bounded to 4163 bytes including LF. Stop on the first error;
flush each successful ``path: OK`` in order, including duplicate occurrences.
The launcher owns independent helper authentication and all resource gates.
"""
import hashlib
import os
import re
import sys

BUFFER_BYTES = 1024 * 1024
MAX_ROW_BYTES = 64 + 2 + 4096 + 1
ROW = re.compile(rb'[0-9a-f]{64}  /[^\x00\r\n]*')


def hash_file(path, buffer, page_size):
    digest = hashlib.sha256()
    view = memoryview(buffer)
    consumed = advised = 0
    with open(path, 'rb', buffering=0) as stream:
        while True:
            count = stream.readinto(buffer)
            if count == 0:
                break
            digest.update(view[:count])
            consumed += count
            aligned = consumed - consumed % page_size
            if aligned > advised:
                # Partial pages stay until fully consumed, including across short reads.
                os.posix_fadvise(stream.fileno(), advised, aligned - advised, os.POSIX_FADV_DONTNEED)
                advised = aligned
    return digest.hexdigest()


def verify_table(source, output):
    buffer = bytearray(BUFFER_BYTES)
    page_size = os.sysconf('SC_PAGE_SIZE')
    records = 0
    while True:
        raw = source.readline(MAX_ROW_BYTES + 1)
        if not raw:
            if not records:
                raise ValueError('empty SHA table')
            return
        records += 1
        if len(raw) > MAX_ROW_BYTES:
            raise ValueError(f'row {records}: oversized SHA record')
        raw = raw.removesuffix(b'\n')
        if not ROW.fullmatch(raw) or len(raw[66:]) > 4096:
            raise ValueError(f'row {records}: malformed SHA record')
        path = raw[66:].decode('utf-8', errors='strict')
        if hash_file(path, buffer, page_size) != raw[:64].decode('ascii'):
            raise ValueError(f'{path}: SHA256 mismatch')
        print(f'{path}: OK', file=output, flush=True)


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError('usage: python3 -I -B -S verify_fresh_sha_table.py < table')
        sys.stdout.reconfigure(encoding='utf-8', errors='strict')
        verify_table(sys.stdin.buffer, sys.stdout)
        return 0
    except (OSError, ValueError) as error:
        print(f'verify_fresh_sha_table: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
