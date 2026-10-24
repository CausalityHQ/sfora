# SOP official TEST: So400m@280 with 512-D head — NEW SOP QUALITY LEADER (93.480)

Rule (8537a321, pre-registered): retained as the SOP best iff official packed R@1 > 93.131. One read.
Receipt on DGX: /home/riomus/runs/sfora-sop-so400m-official-res280d512-179024-v1/receipt.json (sha 188a8273…).
Packed int8 512-D (514 B/row) scored exactly in torch; native 128-only kernel not applicable.

| point | packed R@1 | mAP@R | R@10 | encode img/s | gallery B/row |
|---|---|---|---|---|---|
| **So400m@280, 512-D** | **93.480** | **81.317** | 98.750 | 85.7 | 514 |
| So400m@384, 128-D | 93.131 | 80.351 | 98.53 | 46.9 | 130 |
| So400m@280, 128-D | 92.594 | 78.928 | 98.407 | 86.0 | 130 |
| Large@256, 128-D | 91.660 | 76.757 | 98.167 | 190.6 | 130 |

Paired product bootstrap (5000): vs @384/128-D dR@1 +0.349 [+0.194, +0.502], dmAP +0.966 [+0.798, +1.134];
vs @280/128-D dR@1 +0.886 [+0.748, +1.023], dmAP +2.389 [+2.240, +2.534].
Dominates @384/128-D on quality AND encode speed (1.83x); costs 4x gallery bytes. Single seed (recipe seed-stable:
179025 replicate −0.02 held). claim_eligible false. Serving follow-up: 512-D native int8 kernel.
