#!/usr/bin/env python3
"""Run with python3 -B -S; no numerical/native imports or archived quality read."""
if not __debug__:
    raise SystemExit("Checks require assertions")

import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import struct
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("consensus", ROOT / "score_inshop_archived_consensus.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def rejects(fn):
    try:
        fn()
    except (AssertionError, ValueError, KeyError, FileExistsError):
        return
    raise AssertionError("invalid input accepted")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class Matrix:
    """Tiny float32 tensor stand-in executes the actual scorer without Torch."""
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        if isinstance(key, list):
            return Matrix([self.rows[i] for i in key])
        row, col = key
        if col is None:
            return Matrix([[self.rows[i][0]] for i in row])
        if row is None:
            return Matrix([[self.rows[i][0] for i in col]])
        return Matrix([r[col] for r in self.rows[row]])

    @property
    def T(self):
        return Matrix([list(row) for row in zip(*self.rows)])

    def __matmul__(self, other):
        return Matrix([[f32(sum(a * b for a, b in zip(row, col))) for col in zip(*other.rows)] for row in self.rows])

    def __mul__(self, other):
        return Matrix([[f32(v * other.rows[i % len(other.rows)][j % len(other.rows[0])])
                        for j, v in enumerate(row)] for i, row in enumerate(self.rows)])

    def add_(self, other):
        self.rows = [[f32(a + b) for a, b in zip(x, y)] for x, y in zip(self.rows, other.rows)]
        return self

    def div_(self, value):
        self.rows = [[f32(v / value) for v in row] for row in self.rows]
        return self

    def tolist(self):
        return self.rows


def argsort(matrix, *, dim, descending, stable):
    assert dim == 1 and descending is True and stable is True
    return Matrix([sorted(range(len(row)), key=lambda i: -row[i]) for row in matrix.rows])


def main():
    # Tied scores preserve gallery ordinal; rows with smaller R ignore padding.
    gallery = [0, 1, 1, 2]
    orders = [sorted(range(4), key=lambda j: -v[j]) for v in
              ([1, 1, 0, 0], [0, 2, 1, 3], [0, 2, 1, 0])]
    quality = score.reduce_rankings([row[:2] for row in orders], [0, 1, 1], gallery)
    assert quality["per_query_r1"] == [1, 0, 1]
    assert quality["per_query_ap"] == [1, .25, 1]
    assert quality["map_at_r"] == .75
    rejects(lambda: score.reduce_rankings([[0]], [9], gallery))
    rejects(lambda: score.reduce_rankings([[0]], [1], gallery))
    rejects(lambda: score.reduce_rankings([[1, 1]], [1], gallery))

    torch = SimpleNamespace(inference_mode=nullcontext, argsort=argsort)
    code = Matrix([[2, 0], [0, 2], [2, 0], [0, 1], [1, 0], [0, 2]])
    inverse = Matrix([[.5], [.5], [.5], [1.], [1.], [.5]])
    pair = code, inverse
    # CPU float32 scoring has gallery ties; same arithmetic path for single/three.
    expected = score.reduce_rankings([[0, 2], [1, 3]], [0, 1], [0, 1, 0, 2])
    for pairs in ([pair], [pair, pair, pair]):
        assert score.packed_quality(pairs, [0, 1], [2, 3, 4, 5], [0, 1], [0, 1, 0, 2], torch) == expected
    rejects(lambda: score.packed_quality([pair, pair], [0], [2, 3, 4, 5], [0], [0, 1, 0, 2], torch))
    # Cross the 128-query block boundary through the production loop.
    repeated = score.packed_quality([pair] * 3, [0] * 129, [2, 3, 4, 5], [0] * 129, [0, 1, 0, 2], torch)
    assert repeated["per_query_ap"] == [1.] * 129

    deltas = {s: {m: [.002] * 4 for m in score.METRICS} for s in score.SEEDS}
    intervals = {m: dict(mean_delta=.002, product_lower95=.0001, product_upper95=.01,
                        query_lower95=-.1, query_upper95=.1) for m in score.METRICS}
    assert score.quality_gate(deltas, intervals) == (True, True)
    for metric in score.METRICS:
        bad = copy.deepcopy(intervals); bad[metric]["product_lower95"] = 0
        assert score.quality_gate(deltas, bad) == (True, False)
        bad = copy.deepcopy(intervals); bad[metric]["query_lower95"] = math.nan
        assert score.quality_gate(deltas, bad) == (True, False)
    bad = copy.deepcopy(deltas); bad[179041]["per_query_ap"] = [-.001] * 4
    rejects(lambda: score.quality_gate(bad, intervals))  # mean must bind arrays
    bad_intervals = copy.deepcopy(intervals); bad_intervals["per_query_ap"]["mean_delta"] = .0005
    assert score.quality_gate(bad, bad_intervals) == (False, False)
    bad = copy.deepcopy(deltas)
    bad[179041]["per_query_r1"] = [0] * 4
    bad_intervals = copy.deepcopy(intervals); bad_intervals["per_query_r1"]["mean_delta"] = .001
    assert score.quality_gate(bad, bad_intervals) == (False, False)

    inventory = {"schema": "archived-consensus-v1", "execution": dict.fromkeys(score.EXECUTION, "hash"),
                 "source_manifests": [{}, {}], "decisions": [{}, {}], "endpoints": []}
    for seed, arm in score.ORDER:
        run = f"/original/{seed}/{arm}"
        cpu = dict(receipt=run + "/cpu/proof.json", log=run + "/cpu/log", unit=f"cpu-{seed}-{arm}",
                   invocation_id=f"cpu-id-{seed}-{arm}")
        inventory["endpoints"].append(dict(seed=seed, arm=arm, run=run, receipt=run + "/receipt.json",
            log=run + "/log", unit=f"wire-{seed}-{arm}", invocation_id=f"wire-id-{seed}-{arm}",
            checkpoint_path=run + "/training/resume.pt", cpu=cpu))
    score.validate_inventory(inventory)
    for mutation in ("schema", "order", "extra_execution", "missing_execution", "duplicate_unit", "relative_cpu", "receipt", "checkpoint"):
        bad = copy.deepcopy(inventory)
        if mutation == "schema": bad["schema"] = "other"
        elif mutation == "order": bad["endpoints"].reverse()
        elif mutation == "extra_execution": bad["execution"]["extra.py"] = "hash"
        elif mutation == "missing_execution": bad["execution"].pop(next(iter(score.EXECUTION)))
        elif mutation == "duplicate_unit": bad["endpoints"][1]["cpu"]["unit"] = bad["endpoints"][0]["unit"]
        elif mutation == "relative_cpu": bad["endpoints"][0]["cpu"]["receipt"] = "proof.json"
        elif mutation == "receipt": bad["endpoints"][0]["receipt"] = "/other/receipt.json"
        else: bad["endpoints"][0]["checkpoint_path"] = "/original/weights.pt"
        rejects(lambda: score.validate_inventory(bad))

    roles = {"held_manifest": [{"product": str(i % 1993), "relative_path": str(i)} for i in range(12599)],
             "query": list(range(6354)), "gallery": list(range(6354, 12599))}
    digest = hashlib.sha256(json.dumps(roles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    score.validate_roles(roles, digest)
    rejects(lambda: score.validate_roles(roles, "0" * 64))
    bad = copy.deepcopy(roles); bad["gallery"][0] = 0
    bad_digest = hashlib.sha256(json.dumps(bad, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rejects(lambda: score.validate_roles(bad, bad_digest))

    endpoint = dict(seed=179032, arm="dense", receipt_sha256="wire", checkpoint_sha256="checkpoint")
    files = {n: "hash" for n in score.FILES}
    receipt = dict(pass_=True, seed=179032, arm="candidate", authority_sha256="authority",
                   execution_sha256="source", source_code={"helper.py": "hash"},
                   checkpoint_sha256="checkpoint", files=files, batch=32, width=128,
                   precision="private_native_fp16", optimizer_updates=0, quality_read=False,
                   official_read=False, claim_eligible=False, public_serving_qualified=False,
                   public_latency_measured=False, full_held_independent_whole_head_packed_exact=True,
                   source_head_rng_flags_preserved=True, held_manifest=[{"product": "a"}],
                   query=[0], gallery=[1])
    receipt["pass"] = receipt.pop("pass_")
    decision = {"pass": True, "decision": "KILL", "authority_sha256": "authority", "execution_sha256": "source",
                "inputs": [dict(seed=179032, arm="candidate", receipt_sha256="wire",
                                checkpoint_sha256="checkpoint", files=files)]}
    frozen = {k: receipt[k] for k in ("held_manifest", "query", "gallery")}
    validate = lambda r, e=endpoint, d=decision: score.validate_wire(r, e, d, receipt["source_code"], frozen)
    validate(receipt)  # KILL is a valid archived integrity result.
    for key, value in (("pass", False), ("seed", 179041), ("arm", "queue"),
                       ("authority_sha256", "other"), ("execution_sha256", "other"),
                       ("checkpoint_sha256", "other"), ("optimizer_updates", 1),
                       ("quality_read", True), ("batch", 64), ("precision", "float32"),
                       ("full_held_independent_whole_head_packed_exact", False),
                       ("source_code", {}), ("files", {}), ("query", [1])):
        bad = copy.deepcopy(receipt); bad[key] = value
        rejects(lambda: validate(bad))
    bad = copy.deepcopy(decision); bad["inputs"] *= 2
    rejects(lambda: validate(receipt, d=bad))

    endpoint["cpu"] = dict(receipt="proof", receipt_sha256="cpu-hash", log_sha256="cpu-log")
    receipt.update(cpu_proof="proof", cpu_authority_sha256="cpu-hash", cpu_log_sha256="cpu-log")
    for k in ("boundary", "training_receipt_sha256", "terminal_state_fingerprint", "whole_sha256", "f16_whole_sha256", "head_sha256"):
        receipt[k] = "binding"
    cpu = dict(receipt, strict400_whole_head_fit_packed_reload_exact=True,
               frozen_roles_complete_state_exact=True, cpu_rng_preserved=True)
    score.validate_cpu(cpu, receipt, endpoint, receipt["source_code"])
    for key in ("checkpoint_sha256", "head_sha256", "whole_sha256", "cpu_rng_preserved", "quality_read", "source_code"):
        bad = copy.deepcopy(cpu); bad[key] = "other"
        rejects(lambda: score.validate_cpu(bad, receipt, endpoint, receipt["source_code"]))

    same = {m: [1, 0] for m in score.METRICS}
    assert score.validate_replay(same, same) == {m: 0 for m in score.METRICS}
    bad = copy.deepcopy(same); bad["per_query_ap"][1] = 5e-7
    assert score.validate_replay(bad, same)["per_query_ap"] == 5e-7
    bad["per_query_ap"][1] = 2e-6
    rejects(lambda: score.validate_replay(bad, same))
    bad = copy.deepcopy(same); bad["per_query_r1"][1] = 1
    rejects(lambda: score.validate_replay(bad, same))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        score.publish(path, {"pass": True})
        rejects(lambda: score.publish(path, {}))
        link = Path(tmp) / "link"; link.symlink_to(Path(tmp) / "missing")
        rejects(lambda: score.publish(link, {}))
        rejects(lambda: score.read_json(path, "0" * 64))
        digest = score.sha(path)
        assert score.read_json(path, digest) == {"pass": True}
        score.check_code(Path(tmp), {path.name: digest})
        rejects(lambda: score.check_code(Path(tmp), {path.name: "0" * 64}))
        rejects(lambda: score.check_code(Path(tmp), {"../receipt.json": digest}))
        rejects(lambda: score.check_code(Path(tmp), {str(path): digest}))
    assert subprocess.run([sys.executable, "-B", "-S", "-O", str(ROOT / "score_inshop_archived_consensus.py")],
                          capture_output=True).returncode != 0
    assert "torch" not in sys.modules and "numpy" not in sys.modules
    print("PASS stdlib rankings, ties/padding, paired gates, authority/replay rejection, exclusive output")


if __name__ == "__main__":
    main()
