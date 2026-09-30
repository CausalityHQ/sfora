"""Authority and resource rejection checks without Torch or model execution."""
import copy
import json
from pathlib import Path
import unittest

import train_image_queue_adaptation as driver

EVIDENCE = Path(__file__).resolve().parents[1] / "docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1"


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.cpu = json.loads((EVIDENCE / "image-queue-cpu-proof-v2.json").read_text())
        self.previous = self.cpu["code"]
        self.code = {**self.previous, **{n: "new" for n in driver.ADDED}}

    def test_actual_cpu_authority_and_foreign_closure_rejection(self):
        execution = self.cpu["execution_sha256"]
        driver.validate_cpu(self.cpu, execution, self.code, self.previous)
        for key, value in (("schema", "foreign"), ("pass", False), ("boundary", 10),
                           ("optimizer_members", 240), ("quality_read", True), ("optimizer_updates", 1)):
            changed = copy.deepcopy(self.cpu); changed[key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                driver.validate_cpu(changed, execution, self.code, self.previous)
        changed = dict(self.code); changed["pe_large_coverage.py"] = "foreign"
        with self.assertRaises(AssertionError): driver.validate_cpu(self.cpu, execution, changed, self.previous)
        changed = dict(self.code); changed.pop(next(iter(driver.ADDED)))
        with self.assertRaises(AssertionError): driver.validate_cpu(self.cpu, execution, changed, self.previous)

    def test_resource_caps_reject_real_limits(self):
        driver.resources(self.cpu, 120, False)
        for key, value in (("host_swap_kib", 1), ("host_max_rss_kib", 8388609),
                           ("total_seconds", 120), ("total_seconds", float("nan")),
                           ("unit_memory_swap_max_bytes", 1), ("peak_cuda_allocated_bytes", 1)):
            changed = {**self.cpu, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(AssertionError):
                driver.resources(changed, 120, False)
        gpu = {**self.cpu, "peak_cuda_allocated_bytes": 6_000_000_000}
        driver.resources(gpu, 120, True)
        with self.assertRaises(AssertionError): driver.resources({**gpu, "peak_cuda_allocated_bytes": 10_000_000_000}, 120, True)

    def test_original_log_binds_invocation_rss_cap_and_locks(self):
        text = (EVIDENCE / "image-queue-cpu-v2.log").read_text()
        driver.validate_log(text, self.cpu, 120)
        for altered in (text.replace(self.cpu["unit_invocation_id"], "foreign"),
                        text.replace("Service runtime: 37.327s", "Service runtime: 120s"),
                        text.replace("5156656", "5156657"),
                        text.replace("flock -n /home/riomus/.sfora-siglip2-gpu.lock", "missing-lock")):
            with self.assertRaises(AssertionError): driver.validate_log(altered, self.cpu, 120)


if __name__ == "__main__":
    unittest.main()
