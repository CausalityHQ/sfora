#!/usr/bin/env python3
"""Reject changed L14 cache authority before CUDA; check saved feature invariants."""

import tempfile
from pathlib import Path

import numpy as np

from export_pe_l14_fit import check_features, startup
from pe_core_authority import sha


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        authority = root / "preflight.json"
        authority.write_text("{}")
        expected = sha(authority)
        authority.write_text('{"altered": true}')
        try:
            startup(authority, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("changed manifest accepted")
        output = root / "features.npy"
        values = np.zeros((4, 1024), dtype=np.float32)
        values[:, 0] = 1
        np.save(output, values)
        assert check_features(output, 4) < 1e-5
        for bad in (
            values.astype(np.float16),
            values[:, :512],
            values * 2,
            values * np.nan,
        ):
            np.save(output, bad)
            try:
                check_features(output, 4)
            except AssertionError:
                pass
            else:
                raise AssertionError("malformed descriptors accepted")
    print(
        "PASS changed manifest rejected before CUDA; shape/dtype/finite/unit descriptors"
    )


if __name__ == "__main__":
    main()
