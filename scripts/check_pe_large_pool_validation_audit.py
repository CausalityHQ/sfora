#!/usr/bin/env python3
"""Reject partial, unauthenticated, nonfinite and unequal saved validation vectors."""

import copy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import audit_pe_large_pool_checkpoint_validation as audit


def main():
    with TemporaryDirectory() as tmp:
        out = Path(tmp)
        values = np.zeros((12599, 128), dtype=np.float32)
        values[:, 0] = 1
        for name in ("loaded-held.npy", "reference-held.npy"):
            np.save(out / name, values, allow_pickle=False)
        receipt = {
            "loaded_held_sha256": hashlib.sha256((out / "loaded-held.npy").read_bytes()).hexdigest(),
            "reference_held_sha256": hashlib.sha256((out / "reference-held.npy").read_bytes()).hexdigest(),
        }
        a, b = audit.load_vectors(out, receipt)
        assert np.array_equal(a, values) and np.array_equal(b, values)
        for case in ("hash", "partial", "unequal", "nonfinite", "norm", "shape", "dtype"):
            bad = copy.deepcopy(receipt)
            changed = values.copy()
            if case == "hash":
                bad["reference_held_sha256"] = "unauthenticated"
            elif case == "partial":
                (out / "reference-held.npy").rename(out / "reference-held.npy.partial")
            else:
                if case == "unequal":
                    changed[0, :2] = (0, 1)
                elif case == "nonfinite":
                    changed[0, 0] = np.nan
                elif case == "norm":
                    changed[0, 0] = 2
                elif case == "shape":
                    changed = changed[:-1]
                elif case == "dtype":
                    changed = changed.astype(np.float64)
                np.save(out / "reference-held.npy", changed, allow_pickle=False)
                bad["reference_held_sha256"] = hashlib.sha256((out / "reference-held.npy").read_bytes()).hexdigest()
            try:
                audit.load_vectors(out, bad)
            except (AssertionError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"accepted {case} vectors")
            np.save(out / "reference-held.npy", values, allow_pickle=False)
    print("PASS complete saved-vector authentication and seven negative guards; synthetic CPU fixture, no model quality read")


if __name__ == "__main__":
    main()
