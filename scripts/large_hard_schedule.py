"""Fixed fit-only anchor reweighting; the loss already sees all fit negatives."""
from collections import deque

import numpy as np


def margins(target, bank):
    assert target.dtype == np.int64 and target.ndim == 1
    assert bank.dtype == np.float32 and bank.ndim == 2 and len(bank) == len(target)
    assert np.isfinite(bank).all() and np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-5, rtol=0)
    result = np.empty(len(target), dtype=np.float32)
    for start in range(0, len(target), 256):
        end = min(start + 256, len(target))
        scores = bank[start:end] @ bank.T
        same = target[start:end, None] == target[None, :]
        scores[np.arange(end - start), np.arange(start, end)] = -np.inf
        positive = np.where(same, scores, -np.inf).max(axis=1)
        negative = np.where(~same, scores, -np.inf).max(axis=1)
        result[start:end] = np.where(np.isfinite(positive), positive - negative, np.inf)
    return result


def schedule(target, margin, seed, *, hard_slots=32):
    assert hard_slots in (0, 32)
    assert target.dtype == np.int64 and target.ndim == 1 and margin.shape == target.shape
    names = sorted(set(target.tolist()))
    assert names == list(range(len(names))) and 64 <= len(names) <= 3200
    rng = np.random.default_rng(seed)
    order = rng.permutation(names)
    members = {c: np.flatnonzero(target == c) for c in names}
    pool = np.flatnonzero(margin < 0)
    if hard_slots and (len(pool) < 320 or len(set(target[pool])) < 64):
        raise ValueError('hard pool cannot support32 distinct slots and10-repeat ceiling')
    # Keep the archived zero-treatment RNG consumption exactly unchanged.
    queue = deque(map(int, rng.permutation(pool))) if hard_slots else deque()
    exposure = np.zeros(len(target), dtype=np.int64)
    batches = []
    width = 64 - hard_slots
    for step in range(100):
        classes = order[(step * width + np.arange(width)) % len(order)]
        batch = [int(rng.choice(members[int(c)])) for c in classes]
        chosen = set(map(int, classes))
        for _ in range(hard_slots):
            for _ in range(len(queue)):
                row = queue.popleft()
                if exposure[row] < 10:
                    queue.append(row)
                    if int(target[row]) not in chosen:
                        batch.append(row)
                        chosen.add(int(target[row]))
                        exposure[row] += 1
                        break
            else:
                raise ValueError('distinct hard pool depleted under frozen repeat ceiling')
        batches.append(batch)
    batches = np.asarray(batches, dtype=np.int64)
    assert batches.shape == (100, 64) and all(len(set(target[b])) == 64 for b in batches)
    assert set(target[batches.ravel()]) == set(names)
    return batches
