#!/usr/bin/env python3
"""Official SOP TEST score of the two-seed So400m@384/512 embedding ensemble from saved official test embeddings.

Pre-registered (3428f85e): new SOP incumbent iff paired official packed R@1 (ensemble − seed 179024) product-bootstrap
lower95 > 0. Inputs are the two evaluators' test_embeddings.npy (unit 512-D, pinned by their receipts); ensemble =
concat/sqrt(2) (cosine mean), packed int8 (1026 B/row), scored with the official score_symmetric. No image encode here.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_evaluation import score_symmetric
import train_sop_siglip2_substrate_pair as pair

def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in ("run-a", "run-b", "archive", "output"): p.add_argument(f"--{n}", type=Path, required=True)
    a = p.parse_args(); assert not a.output.exists()
    rec = {}
    for k, d in (("a", a.run_a), ("b", a.run_b)):
        r = json.loads((d / "receipt.json").read_text()); assert pair.sha256(d / "test_embeddings.npy") == r["test_embeddings_sha256"]; rec[k] = r
    assert rec["a"]["seed"] == 179024 and rec["b"]["seed"] == 179025
    ea, eb = (torch.from_numpy(np.load(d / "test_embeddings.npy")) for d in (a.run_a, a.run_b))
    ens = torch.cat([ea, eb], 1) / np.sqrt(2)
    labels = torch.from_numpy(np.asarray(np.load(a.archive, allow_pickle=False)["test_labels"], dtype=np.int64)).cuda()
    out = {"schema": "sop-official-seed-ensemble-v1", "claim_eligible": False}
    q = {}
    for name, e in (("seed_179024", ea), ("seed_179025", eb), ("ensemble", ens)):
        pk = pack_int8_unit_embeddings(e)
        q[name] = score_symmetric(pk.codes.float().cuda(), labels, inverse_norms=pk.inverse_norms.cuda())
        out[name] = {k: v for k, v in q[name].items() if not isinstance(v, list)}
    assert q["seed_179024"]["recall_at_1"] == rec["a"]["packed_quality"]["recall_at_1"]
    lab = labels.cpu().numpy(); u, inv = np.unique(lab, return_inverse=True); N = np.bincount(inv); rng = np.random.default_rng(179019)
    for key, m in (("per_query_r1", "dR1"), ("per_query_ap", "dmAP")):
        d = np.array(q["ensemble"][key], float) - np.array(q["seed_179024"][key], float); S = np.bincount(inv, d); bs = []
        for _ in range(5000):
            c = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)); bs.append((c * S).sum() / (c * N).sum())
        out[m] = {"mean": float(d.mean()), "product_95": [float(v) for v in np.percentile(bs, [2.5, 97.5])]}
    out["decision"] = "NEW INCUMBENT" if out["dR1"]["product_95"][0] > 0 else "KILL"
    a.output.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n"); print(json.dumps(out, sort_keys=True))

if __name__ == "__main__": main()
