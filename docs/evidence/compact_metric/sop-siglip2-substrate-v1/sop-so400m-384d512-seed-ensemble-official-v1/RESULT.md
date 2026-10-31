# SOP official TEST: two-seed So400m@384/512 embedding ensemble — NEW SOP INCUMBENT (94.129)

Rule (3428f85e / 4ccf42be, pre-registered): new incumbent iff paired official packed R@1 (ensemble − seed 179024)
product-bootstrap lower95 > 0. Seed-179025 official export receipt 1838339c…; ensemble result aee46bcf….
| arm | packed R@1 | mAP@R | gallery B/row | encode |
|---|---|---|---|---|
| seed 179024 (previous incumbent) | 93.967 | 82.694 | 514 | 1x |
| seed 179025 | 93.990 | 82.866 | 514 | 1x |
| **ensemble (concat/√2, int8)** | **94.129** | **83.109** | 1026 | 2x |
Paired: dR@1 +0.162 [+0.088, +0.236]; dmAP +0.416 [+0.356, +0.478]. TRAIN-held predicted +0.34 (transfer 0.47x).
Quality-first policy: retained. Cost doubles encode and gallery bytes. Single official read.
