# SOP official TEST: fine-tuned So400m vs Large pair (seed 179024) — PROMOTE So400m (SOP quality Pareto candidate)

Rule (pre-registered in scripts/evaluate_sop_substrate_pair_official.py, e88fffcc, committed before the read):
So400m promoted iff official packed R@1 > Large pair control AND > prior SOP official best 91.7419.
One read per arm, authority = TRAIN ADVANCE ../sop-so400m-pair-179024-v3/RESULT.md. Driver + log: driver-and-log.txt.
Full receipts (with per-query lists) on DGX: /home/riomus/runs/sfora-sop-pair-official-{large,so400}-179024-v1/receipt.json
(sha 41b9a0c0… / 491d90fa…); summary.json holds every scalar.

Protocol: SOP official TEST, 60,502 symmetric query/gallery, 11,316 products, self excluded; pinned image manifest
28a3ec05…; 128-D signed-int8 packed gallery (130 B/row), native top-10 exact vs packed oracle, per-query R@1 equal.

| arm | packed R@1 | packed mAP@R | R@10 | export img/s (GB10, fp16, b64) |
|---|---|---|---|---|
| Large/16@256 pair control | 91.660 | 76.757 | 98.167 | 190.6 |
| So400m/14@384 | **93.131** | **80.351** | 98.53 | 46.9 |

Paired so400 − large, product bootstrap (5000, seed 179019):
- dR@1 +1.471 pp, 95% [+1.289, +1.654]
- dmAP@R +3.594 pp, 95% [+3.388, +3.793]

vs prior SOP official best 91.7419 (exploratory): +1.39 pp. TRAIN-held gain (+1.28) transferred in full (+1.47),
unlike In-Shop. Single seed; claim_eligible false (protocol previously read).
Cost: encode 4.1x slower (46.9 vs 190.6 img/s), training 3.8x wall, 2.9x CUDA. Gallery bytes unchanged.
Next: efficiency for the So400m SOP candidate (encode cost), retaining quality.
