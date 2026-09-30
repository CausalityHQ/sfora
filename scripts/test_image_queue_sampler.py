"""Synthetic stdlib checks; actual sampler/native admission belongs to the parent."""

import copy
import random
import unittest
from collections import Counter

from image_queue_sampler import queue_schedule


def fixture():
    # 65 classes: singleton, multi-cycle queues, and a queue longer than its use.
    sizes = [3, 1, 103] + [2 + c % 6 for c in range(3, 65)]
    target = list(range(65))
    for c, size in enumerate(sizes):
        target.extend([c] * (size - 1))
    batches = [[(13 * step + 7 * slot) % 65 for slot in range(64)]
               for step in range(100)]
    return target, batches


class QueueScheduleTests(unittest.TestCase):
    def test_determinism_plain_lists_and_unchanged_inputs(self):
        target, batches = fixture()
        before = copy.deepcopy((target, batches))
        result = queue_schedule(target, batches, 179032)
        self.assertEqual(result, queue_schedule(target, batches, 179032))
        self.assertNotEqual(result, queue_schedule(target, batches, 179041))
        self.assertEqual(before, (target, batches))
        self.assertIs(type(result), list)
        for batch in result:
            self.assertIs(type(batch), list)
            for row in batch:
                self.assertIs(type(row), int)
        self.assertEqual(result, queue_schedule(tuple(target),
                         tuple(map(tuple, batches)), 179032))

    def test_exact_class_slot_sequence(self):
        target, batches = fixture()
        result = queue_schedule(target, batches, 179032)
        self.assertEqual(len(result), 100)
        for original, candidate in zip(batches, result):
            self.assertEqual(len(candidate), 64)
            self.assertEqual([target[r] for r in original],
                             [target[r] for r in candidate])
            self.assertEqual(len({target[r] for r in candidate}), 64)

    def test_canonical_hash_order_and_cycle_exhaustion(self):
        target, batches = fixture()
        result = queue_schedule(target, batches, 179032)
        uses = [[] for _ in range(65)]
        for batch in result:
            for row in batch:
                uses[target[row]].append(row)
        # Independently inspected SHA256 ranks for rows 0, 65, 66, cycles 0/1.
        self.assertEqual(uses[0][:6], [66, 0, 65, 65, 0, 66])
        for c, rows in enumerate(uses):
            members = {r for r, label in enumerate(target) if label == c}
            size = len(members)
            for start in range(0, len(rows), size):
                cycle = rows[start:start + size]
                self.assertEqual(len(cycle), len(set(cycle)))
                self.assertTrue(set(cycle) <= members)
                if len(cycle) == size:
                    self.assertEqual(set(cycle), members)

    def test_maximum_unique_count_at_fixed_class_slots(self):
        target, batches = fixture()
        slots = Counter(target[r] for batch in batches for r in batch)
        members = Counter(target)
        ceiling = sum(min(slots[c], members[c]) for c in members)
        for seed in (179032, 179041, -1):
            result = queue_schedule(target, batches, seed)
            self.assertEqual(len({r for batch in result for r in batch}), ceiling)

    def test_global_rng_preserved_on_success_and_rejection(self):
        target, batches = fixture()
        state = random.getstate()
        queue_schedule(target, batches, 179032)
        self.assertEqual(state, random.getstate())
        with self.assertRaises(ValueError):
            queue_schedule(target, batches[:-1], 179032)
        self.assertEqual(state, random.getstate())

    def test_malformed_targets(self):
        target, batches = fixture()
        cases = [None, 1, [], [0] * len(target),
                 [c + 1 for c in target], [c * 2 for c in target]]
        for invalid in (-1, 1.0, "0", True, None, [0]):
            changed = target.copy()
            changed[0] = invalid
            cases.append(changed)
        for bad in cases:
            with self.subTest(target=repr(bad)[:80]):
                with self.assertRaises(ValueError):
                    queue_schedule(bad, batches, 179032)

    def test_malformed_batches(self):
        target, batches = fixture()
        cases = [None, 1, [], batches[:-1], batches + [batches[0]],
                 [list(range(64)) for _ in range(100)]]  # Missing class 64.
        for bad_batch in (None, 1, batches[0][:-1], batches[0] + [64]):
            changed = copy.deepcopy(batches)
            changed[0] = bad_batch
            cases.append(changed)
        for invalid in (-1, len(target), 1.0, "0", True, None, [0],
                        batches[0][1],  # Repeated row.
                        target.index(7, 65)):  # Different row, same class (7).
            changed = copy.deepcopy(batches)
            changed[0][0] = invalid
            cases.append(changed)
        for index, bad in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    queue_schedule(target, bad, 179032)

    def test_malformed_seeds(self):
        target, batches = fixture()
        for seed in (None, True, "179032", 179032.0, []):
            with self.subTest(seed=seed):
                with self.assertRaises(ValueError):
                    queue_schedule(target, batches, seed)


if __name__ == "__main__":
    unittest.main()
