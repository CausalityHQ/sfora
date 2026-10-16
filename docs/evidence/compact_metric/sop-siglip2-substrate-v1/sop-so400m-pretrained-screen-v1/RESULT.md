# SOP pretrained So400m vs Large substrate screen: ADVANCE (pre-registered c7602797)

Zero-training, SOP official TRAIN only (59,551 rows; product-disjoint fit/held seed 179019,
53,700 / 5,851). Unchanged score_corpus (fit-only PCA128, signed-int8 packing). No TEST read.

| packed PCA128 R@1 | Large/16@256 | So400m/14@384 | delta |
|---|---|---|---|
| held vs full TRAIN gallery | 78.346 | 80.516 | **+2.171 pp** |
| held-only | 90.292 | 91.523 | +1.231 pp |
| float full-width held-only | 91.061 | 92.070 | +1.008 pp |

Rule: ADVANCE iff full-gallery delta >= +2.0 pp and held-only higher -> both met.
Large reproduces its prior pretrained measurement (78.3456%). So400m export: 1138 s for 59,551 images
(vs Large 509 s), 3.0 GB CUDA peak. Attempt 1 (sfora-sop-so400m-screen-v1.log) was killed at 79 s by
user@1000 exit after the last ssh session closed (bare systemd-run, no driver); a2 ran under a
detached session driver.

Next (pre-registered consequence): a separately reviewed fine-tuned SOP So400m vs Large paired run
with the incumbent true-freeze recipe. Expected effect if ~half of the pretrained gap survives
fine-tuning: ~+1 pp official R@1 over 91.74.
