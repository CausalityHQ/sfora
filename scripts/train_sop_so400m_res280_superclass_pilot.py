#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@280 recipe with super-category-homogeneous batches.

Identical to train_sop_so400m_res280_pilot.py (d0568eee) except the batch schedule: each 64-image batch
(16 products x 4) is drawn from ONE of the 12 SOP super-categories (path prefix, e.g. kettle_final), so in-batch
negatives are same-category. Per super-category, the unchanged identity_balanced_batches (coverage-first) builds
its sub-schedule with steps allocated by product count (largest remainder); sub-schedules are interleaved by a
seeded permutation. Everything else (objective, member bank, optimizer, seed, 1000 updates) is unchanged.

Hypothesis (perspectives: retrieval researcher predicts gain from harder in-batch negatives on fine-grained
same-category confusions; skeptic predicts ~0 because the full-fit member bank already sees all negatives).
Pre-registered decision: held R@1 (5,851 vs full 59,551 TRAIN gallery, packed int8) paired product-bootstrap
lower95 vs the res280 13-frozen run (receipt 9e4a05db, seed 179024) > 0 => ONE official read at 280. Else KILL.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_res280_pilot as base  # noqa: E402

pair = base.pair
_original = pair.identity_balanced_batches
SUPER: dict[str, str] = {}


def superclass_batches(labels, *, batch_size, images_per_identity, seed, epoch, steps, coverage_first=False):
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(SUPER[label], []).append(index)
    names = sorted(groups)
    products = np.array([len({labels[i] for i in groups[n]}) for n in names], dtype=float)
    share = steps * products / products.sum()
    alloc = np.floor(share).astype(int)
    alloc[np.argsort(-(share - alloc), kind="stable")[: steps - alloc.sum()]] += 1
    subs = {}
    for name, n in zip(names, alloc, strict=True):
        rows = groups[name]
        sub = _original(tuple(labels[i] for i in rows), batch_size=batch_size, images_per_identity=images_per_identity,
                        seed=seed, epoch=epoch, steps=int(n), coverage_first=coverage_first) if n else ()
        subs[name] = [tuple(rows[i] for i in batch) for batch in sub]
    order = np.random.default_rng((seed, epoch, 0x5C)).permutation(np.repeat(np.arange(len(names)), alloc))
    cursor = dict.fromkeys(names, 0)
    batches = []
    for k in order:
        name = names[k]
        batches.append(subs[name][cursor[name]])
        cursor[name] += 1
    assert len(batches) == steps and all(len({SUPER[labels[i]] for i in b}) == 1 for b in batches)
    return tuple(batches)


def main():
    archive = Path(sys.argv[sys.argv.index("--unicom-l14-archive") + 1])
    with np.load(archive, allow_pickle=False) as a:
        for label, rel in zip(a["train_labels"].tolist(), a["train_relative_paths"].astype(str), strict=True):
            prefix = rel.split("/")[0]
            assert SUPER.setdefault(str(label), prefix) == prefix
    assert len(set(SUPER.values())) == 12
    pair.identity_balanced_batches = superclass_batches
    pair.main()


if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400_280"
    main()
