"""Independent stdlib replay of the fixed CPU screen receipt and stop rule."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
r = json.loads((root / "receipt.json").read_text())
freeze = json.loads((root / "freeze-and-invocation.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert r["source_sha256"] == freeze["source_sha256"] == sha("scripts/probe_sop_direct_batch32.py")
assert r["serving_sha256"] == freeze["serving_sha256"] == sha("src/sfora/siglip2_compact_serving.py")
assert freeze["gate_sha256"] == sha("docs/sop_direct_batch32_cpu_gate_2026-09-28.md")
assert r["batches_checked"] == 32 and r["synthetic_exact"]
assert len(r["image_sha256"]) == len(set(r["image_sha256"])) == 1024
assert not r["cuda_used"] and not r["quality_measured"] and not r["full_latency_measured"]
assert r["torch"] == "2.12.1+cu130" and r["threads"] == 20
def quantile(values, q):
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    return (values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)) / 1e6
stats = {}
for arm, samples in r["raw_ns"].items():
    assert len(samples) == 64 and all(type(v) is int and v > 0 for v in samples)
    stats[arm] = {"p50_ms": quantile(samples, .50), "p95_ms": quantile(samples, .95)}
    assert all(abs(stats[arm][k] - r["timing"][arm][k]) < 1e-12 for k in stats[arm])
checks = {"exact": r["batches_checked"] == 32,
          "p50": stats["direct"]["p50_ms"] <= .70 * stats["control"]["p50_ms"],
          "p95": stats["direct"]["p95_ms"] <= .80 * stats["control"]["p95_ms"],
          "rss": r["peak_rss_bytes"] < 2_000_000_000, "budget": r["whole_seconds"] <= 120}
assert checks == r["gates"] and r["decision"] == "KILL_DIRECT_BATCH32_CPU" and not all(checks.values())
out = {"receipt_sha256": sha(root / "receipt.json"), "source_and_gate_exact": True,
       "quantiles_and_decision_replayed": True, "criteria": checks,
       "p50_ratio": stats["direct"]["p50_ms"] / stats["control"]["p50_ms"],
       "p95_ratio": stats["direct"]["p95_ms"] / stats["control"]["p95_ms"],
       "scope": "Every recorded timing/role-count/source/decision checked; pixel equality attested by original run, no independent pixel replay."}
(root / "verification.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
