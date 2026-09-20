from __future__ import annotations

import unittest

from _scratch_profile_packed_int8_1m import _latency_summary


class PackedInt8MillionProfileTests(unittest.TestCase):
    def test_latency_summary_uses_nearest_rank_and_measured_throughput(self) -> None:
        observed = _latency_summary([10, 20, 30, 40, 50], query_batch=2)

        self.assertEqual(observed["p50_ns"], 30)
        self.assertEqual(observed["p99_ns"], 50)
        self.assertEqual(observed["mean_ns"], 30.0)
        self.assertEqual(observed["queries_per_second"], 2e9 / 30.0)


if __name__ == "__main__":
    unittest.main()
