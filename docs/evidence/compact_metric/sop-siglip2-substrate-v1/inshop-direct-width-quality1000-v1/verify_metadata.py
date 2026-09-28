"""Replay the frozen direct-width1000 decision without training or GPU work."""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
source = Path(sys.argv[2])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
r = json.loads((root / "receipt.json").read_text())
assert sha(root / "receipt.json") == "8a25d14a58e9914d93a34809879edca71925ec09d6d834a4150d702bff9213cc"
partition = Path(sys.argv[3])
assert sha(partition) == "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
rows = [line.split() for line in partition.read_text().splitlines()[2:] if line.split()[2] == "train"]
counts = Counter(row[1] for row in rows)
eligible = sorted((x for x, n in counts.items() if n > 1), key=lambda x: (hashlib.sha256(b"inshop-unseen-gallery-v1\0" + x.encode()).digest(), x))
held_names = set(eligible[:(len(eligible) + 1) // 2])
held = [row for row in rows if row[1] in held_names]
groups = defaultdict(list)
for i, row in enumerate(held):
    groups[row[1]].append(i)
query, gallery = [], []
for label in sorted(groups):
    ordered = sorted(groups[label], key=lambda i: hashlib.sha256(("Img/" + held[i][0]).encode()).digest())
    n = max(1, min(len(ordered) - 1, round(len(ordered) / 2)))
    gallery.extend(ordered[:n])
    query.extend(ordered[n:])
query.sort()
gallery.sort()
assert (len(held), len(query), len(gallery), len(groups)) == (12599, 6354, 6245, 1993)
assert hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() == "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
assert hashlib.sha256(np.asarray(gallery, dtype="<i4").tobytes()).hexdigest() == "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"
a, b = r["arms"]["128"], r["arms"]["256"]
checks = {
    "same_pixels": a["first_input_batch_sha256"] == b["first_input_batch_sha256"] and len(a["first_input_batch_sha256"]) == 1000,
    "same_pca_rows": a["direct_initial_rows_sha256"] == b["direct_initial_rows_sha256"],
    "median_step": np.median(b["step_seconds"]) <= 1.1 * np.median(a["step_seconds"]),
    "peak_training_cuda": b["training_peak_cuda_allocated_bytes"] <= a["training_peak_cuda_allocated_bytes"] + 2**30,
    "whole_budget": r["whole_seconds"] <= 2200,
    "train_wall": b["training_wall_including_member_bank_init_seconds"] <= 1.1 * a["training_wall_including_member_bank_init_seconds"],
    "same_protocol": all(a[k] == b[k] for k in ("source_files_sha256", "fit_rows_sha256", "held_rows_sha256", "preflight_sha256", "model_file_sha256", "executed_schedule_sha256", "features_sha256", "seed", "arm", "updates", "vision_lr", "batch_size", "rank_coefficient")),
}
for width, arm in r["arms"].items():
    assert arm["seed"] == 179024 and arm["updates"] == 1000 and arm["batch_size"] == 64
    assert arm["export_batch_size"] == 32 and arm["width_fold"] is None
    assert arm["direct_width_confirmation_sha256"] == "c86c47c64d49df087b3db5103637009ac8d5ef22ff382bbe65035c2422f06a61"
    assert arm["whole_arm_wall_seconds"] <= 1100
    assert arm["training_wall_including_member_bank_init_seconds"] == arm["training_wall_seconds"] + arm["member_bank_init_seconds"]
    assert arm["frozen_encoder_blocks"] == list(range(12)) and arm["frozen_embeddings"]
    assert arm["training_width"] == int(width) and (arm["fit_rows"], arm["held_rows"]) == (13283, 12599)
    for path, digest in arm["source_files_sha256"].items():
        assert sha(source / path.split("/source/")[1]) == digest, path
    durable = json.loads((root / width / "direct_training_checkpoint.json").read_text())
    assert all(arm[k] == value for k, value in durable.items())
    child = json.loads((root / width / "receipt.json").read_text())
    assert all(arm[k] == value for k, value in child.items())
    checks[width + "_frozen"] = arm["mechanics_frozen_sha256"] == arm["mechanics_terminal_frozen_sha256"]
    checks[width + "_trained"] = arm["mechanics_initial_trainable_sha256"] != arm["mechanics_terminal_trainable_sha256"]
    checks[width + "_geometry"] = all(arm["width_terminal_geometry"][k] >= .5 * arm["width_initial_geometry"][k] for k in ("variance", "effective_rank"))
    checks[width + "_reload"] = all(arm["private_native_fp16_reload"]["checks"].values())
    checks[width + "_stable"] = len(arm["all_step_losses"]) == 1000 and bool(np.isfinite(arm["all_step_losses"]).all()) and len(arm["step_seconds"]) == 1000
    grads = np.asarray([[step["group_gradient_norms"][name] for name in ("vision", "head", "classifier")] for step in arm["width_history"]])
    checks[width + "_gradients"] = grads.shape == (1000, 3) and bool(np.isfinite(grads).all() and (grads > 0).all())
    checks[width + "_checkpoint"] = sha(root / width / "checkpoint.pt") == arm["checkpoint_sha256"] == arm["private_native_fp16_reload"]["checkpoint_sha256"]
    checks[width + "_fixture"] = sha(root / width / "native_fp16_fit_fixture.pt") == arm["private_native_fp16_reload"]["fixture_sha256"]
assert a["asymmetric_quality"]["recall_at_1"] >= .970
labels = np.asarray([held[i][1] for i in query])
_, inverse = np.unique(labels, return_inverse=True)
counts = np.bincount(inverse)
intervals = {}
for name, field, floor in (("recall", "per_query_r1", .005), ("map", "per_query_ap", .01)):
    delta = np.asarray(b["asymmetric_quality"][field]) - np.asarray(a["asymmetric_quality"][field])
    totals = np.bincount(inverse, weights=delta)
    rng = np.random.default_rng(179019)
    draws = np.empty(5000)
    for start in range(0, 5000, 512):
        picked = rng.integers(0, len(counts), size=(min(512, 5000 - start), len(counts)))
        draws[start:start + len(picked)] = totals[picked].sum(axis=1) / counts[picked].sum(axis=1)
    lower, upper = np.quantile(draws, [.025, .975])
    expected = r["deltas"][name]
    assert np.allclose([delta.mean(), lower, upper], [expected["point"], expected["lower95"], expected["upper95"]], atol=1e-14, rtol=0)
    intervals[name] = {"point": float(delta.mean()), "lower95": float(lower), "upper95": float(upper)}
    checks[name] = delta.mean() >= floor and lower > 0
checks = {k: bool(v) for k, v in checks.items()}
assert checks == r["criteria"]
assert r["decision"] == "KILL_DIRECT_WIDTH_QUALITY1000" and not checks["recall"] and not checks["map"]
hits_a, hits_b = (np.asarray(arm["asymmetric_quality"]["per_query_r1"]) for arm in (a, b))
out = {"receipt_sha256": sha(root / "receipt.json"), "criteria": checks, "deltas": intervals,
       "rescues": int(((hits_b == 1) & (hits_a == 0)).sum()), "regressions": int(((hits_b == 0) & (hits_a == 1)).sum()),
       "all_source_files_exact": True, "durable_and_child_receipts_exact": True,
       "checkpoint_and_fixture_hashes_exact": True, "decision": r["decision"],
       "scope": "Receipt histories/guard attestations and hashes replayed; no new checkpoint forward or tensor-state audit."}
(root / "metadata-audit.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
