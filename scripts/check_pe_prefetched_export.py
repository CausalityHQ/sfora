#!/usr/bin/env python3
"""Exercise the real writer's order, final short batch and worker failure."""

import tempfile
from pathlib import Path
import numpy as np
from pe_prefetched_export import export_prefetched


def main():
    rows = tuple(range(67))
    calls = []

    def encode(batch, prepared):
        assert tuple(batch) == prepared
        calls.append(prepared)
        return np.repeat(np.asarray(batch, dtype=np.float32)[:, None], 2, axis=1)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "features.npy"
        export_prefetched(rows, tuple, encode, path, width=2)
        assert [len(c) for c in calls] == [32, 32, 3]
        assert np.array_equal(np.load(path)[:, 0], rows)
        assert not path.with_name(path.name + ".partial").exists()

        def fail(batch):
            if batch[0] == 32:
                raise RuntimeError("worker failure")
            return tuple(batch)

        failed = Path(directory) / "failed.npy"
        try:
            export_prefetched(rows, fail, encode, failed, width=2)
        except RuntimeError as error:
            assert str(error) == "worker failure"
        else:
            raise AssertionError("worker exception accepted")
        assert not failed.exists() and not failed.with_name(failed.name + ".partial").exists()
    print("PASS original atomic writer with one-pending input, row order, short batch and worker failure cleanup")


if __name__ == "__main__":
    main()
