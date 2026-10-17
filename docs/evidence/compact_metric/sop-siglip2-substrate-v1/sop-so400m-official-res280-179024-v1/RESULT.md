# SOP official TEST: So400m trained+evaluated at 280 — RETAINED (dominates the @336 point)

Rule (ff998c71, pre-registered): retain iff official packed R@1 > Large pair official 91.660 AND > prior best 91.7419.
Receipt on DGX: /home/riomus/runs/sfora-sop-so400m-official-res280-179024-v1/receipt.json (sha 68a49976…).

| point | packed R@1 | mAP@R | R@10 | encode img/s (with decode) |
|---|---|---|---|---|
| Large pair @256 | 91.660 | 76.757 | 98.167 | 190.6 |
| **So400m trained+tested 280** | **92.594** | **78.928** | 98.407 | **86.0** |
| So400m trained 384, tested 336 | 92.594 | 78.879 | 98.410 | 60.1 |
| So400m @384 | 93.131 | 80.351 | 98.53 | 46.9 |

vs Large official, paired product bootstrap: dR@1 +0.934 [+0.755, +1.112]; dmAP@R +2.170 [+1.971, +2.361].
Decision: RETAINED. Same R@1 count as @336 (56,021/60,502; coincidental — different checkpoints/embeddings,
1,846 per-query disagreements), higher mAP, 1.43x faster → @336 point is dominated.
SOP Pareto frontier now: @384 (quality, 46.9 img/s), res280 (92.59, 86.0 img/s), Large (91.66, 190.6 img/s).
Single seed; claim_eligible false.
