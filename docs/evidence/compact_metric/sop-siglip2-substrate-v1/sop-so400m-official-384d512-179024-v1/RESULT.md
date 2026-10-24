# SOP official TEST: So400m@384 with 512-D head — NEW SOP QUALITY LEADER (93.967)

Rule (7e12c180, pre-registered): new best iff official packed R@1 > 93.480. One read. Receipt sha 3ec99d1e… (DGX
/home/riomus/runs/sfora-sop-so400m-official-384d512-179024-v1). Packed int8 512-D scored exactly in torch.

SOP Pareto frontier (official TEST, packed int8, single seed 179024):
| point | R@1 | mAP@R | R@10 | encode img/s | gallery B/row |
|---|---|---|---|---|---|
| **So400m@384, 512-D** | **93.967** | **82.694** | 98.889 | 46.3 | 514 |
| So400m@280, 512-D | 93.480 | 81.317 | 98.750 | 85.7 | 514 |
| So400m@384, 128-D | 93.131 | 80.351 | 98.53 | 46.9 | 130 (dominated by 280/512 except bytes) |
| So400m@280, 128-D | 92.594 | 78.928 | 98.407 | 86.0 | 130 |
| Large@256, 128-D | 91.660 | 76.757 | 98.167 | 190.6 | 130 |
vs 280/512-D: dR@1 +0.488 [+0.355, +0.620]; dmAP +1.377 [+1.254, +1.495] (paired product bootstrap 5000).
vs prior exploratory SOP official best 91.7419: +2.23 pp. claim_eligible false (protocol previously read).
