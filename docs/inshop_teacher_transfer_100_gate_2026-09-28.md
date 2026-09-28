# Frozen 100-update teacher-transfer retrieval gate

Same teacher-transfer hypothesis only; frozen Base source remains KILL.
Successful wiring/stability receipt SHA
4a3d01143a2036575c19d71fda93bf7524b9b6b8bf42627f2ae60494b6083fb3
authorizes this next bounded gate. No new review or hypothesis search.

Fit13283 images/2004 official TRAIN products; held12599 images/1993 disjoint
products, fixed6354 query/6245 gallery roles. **Held has previously been
observed: this is exploratory, not independent confirmation or SOTA.**
Export only Base fit images for PCA128/proxies/bank; held cache positions are
zero placeholders and never used for initialization. Native Base FP32
parameters, FP16-autocast cache/export and BF16 training. Original canonical
processor, crop0.8–1/flip, seed179024/1000-update coverage-first schedule prefix,
100×64 updates, all12 Base blocks trainable, same AdamW/clip/ArcFace+8bank.

Run serial main-only Base control and true teacher first. True adds0.1
leave-self-out relational KL at0.20. Pinned seed179026 Large teacher uses
FP32-exact checkpoint load, frozen/eval, ONLINE BF16 same augmented tensors.
This measures real augmentation KD cost, unlike the fixed-pixel cached smoke.
One pair-preserving sham follows only if true-versus-control passes. Rotate
whole product groups only within equal batch multiplicity, preserving every
same-product relation; log all100 input hashes/losses/teacher costs.

Use existing packed ordinal stable-tie scorer for R@1 and mAP@R and existing
5000-draw product-cluster bootstrap. Verify source/cache/schedule/PCA and all
100 input hashes match between arms, native model and teacher authorities,
finite100 updates/no skips, pinned feature/checkpoint/held-array hashes.

GO to a separately frozen paired-seed/full-budget gate only if:

- True-minus-main packed R@1≥+0.30pp and paired95 lower>0.
- True-minus-main mAP@R≥0 and paired95 lower≥−0.002.
- True packed R@1≥92.1999% (within3pp of archived matched100-update Large
  baseline95.1999%); no extrapolation of an early-budget quality result.
- True training wall≤113.864s (1.5× archived Large75.909s), peak CUDA<32GiB.
  Archived Large timing is a dated budget guard, not a new matched-speed claim.
- If these pass, true-minus-sham R@1 point>0 and mAP@R point≥0. If sham wins
  or ties, KILL teacher-specific mechanism; no changing coefficient/temperature.

Any failure stops before full-budget/paired-seed/official/serving work.
Hard aggregate external900s limit for fit-cache + all runs/evaluation, one
global DGX lock. Each child≤240s. Existing Large reference stays archived;
no expensive repeated baseline until student evidence warrants it.
Report source acquisition, teacher original training (historical seed179026
745.063s), online target cost, whole-arm/export/scoring costs separately.
No claim of faster total training from student-only step cost. Production
serving remains unchanged until packed held qualification justifies Base API.
