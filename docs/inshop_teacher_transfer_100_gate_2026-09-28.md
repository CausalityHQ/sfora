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

## Terminal: KILL

Original DGX unit `sfora-inshop-teacher-transfer-100-v1`, invocation
`32be6ea4b7364ed6a0c2a34022383f90`, exited0. Both arms completed100 stable
updates and all100 augmented input hashes matched. The executor stopped
before sham because the paired quality gates failed. No full-budget run,
official TEST read, serving gate or production promotion follows.

| Reused official TRAIN held6354 queries /6245 gallery | Base main-only control | Base + trained Large teacher | Archived Large100-update reference |
|---|---:|---:|---:|
| Packed R@1, % |93.641800|93.673277|95.1999|
| Packed mAP@R, fraction |0.736656811|0.734615344|0.779876|
| TRAIN100×64 wall, seconds |53.030560|85.850298|75.909 (dated)|
| Images / measured TRAIN second |120.6851|74.5484|Not remeasured|
| TRAIN peak allocated CUDA, bytes |9,332,096,512|10,596,121,088|12,939,458,560 (dated)|
| Whole arm including setup/export, seconds |98.503515|133.704515|Not remeasured|
| Held export / symmetric score wall, seconds |35.962875 /0.162003|36.005213 /0.157832|66.320 export (dated)|
| Image-to-top-k p50/p95/p99/QPS |Not measured|Not measured|Not remeasured|

True-minus-control R@1 was **+0.031476pp**,95% paired product-bootstrap
**[-0.276896,+0.322386]pp** (48 rescues/46 regressions): fails+0.30pp and
positive-lower floors. mAP@R delta **-0.002041467**,95%
**[-0.004388191,+0.000165944]**, fails nonnegative point/lower guard.
These are one-seed exploratory results, not generalization confirmation.
True stayed above the frozen gross Large floor and passed cost/resource
guards; those passes cannot rescue failed quality. Archived Large numbers
are same-panel budget references, not a newly measured external speed win.

Base fit-only acquisition encoded13283 images/768 coordinates in49.240358s,
peak1,760,986,624B. Cache file SHA
dc47ca0e6c058361ab314a0ec7cb9aa6fe6c8b82b0d7e6a7d0b3dc0c430bc3e0;
held cache rows stayed zero and did not initialize any trained component.
Online teacher intervals totaled32.856474s within the85.850298s treatment
training wall. Teacher historical original training745.063s is an additional
cost; no total-training efficiency claim is supported. Entire executor
main wall305.870132s. Raw student checkpoints/held arrays remain on DGX.

Aggregate receipt SHA
**383570eee205b379d25dd618eb8f7dee7b83990e714305dc3efa03d320cb686f**.
Committed child receipts/logs/packed flags bind original artifacts. Independent
CPU replay verified all6354 per-arm packed hits exactly, AP within5e-7,
partition/fit/query/gallery/array/training-receipt hashes and both bootstrap
bounds. Focused19 tests and Ruff pass. DGX/consultations are idle. Production
API/checkpoint selection remains the qualified Large path; keep this closed
experimental trainer only for reproducibility. No coefficient, temperature,
target-layer, update-budget or panel search follows this negative result.

The next cheap solver audit is the existing reference ArcFace derivative
contract: `sharded_mask_arcface_logits` changes the target under `no_grad`,
retaining the original cosine derivative. That faithfully reproduces UNICOM's
surrogate; it is **not a newly discovered implementation bug**. Determine
whether an analytically differentiated target materially changes the main
gradient while preserving forward logits and remaining finite at endpoints,
before considering ONE distinct training-algorithm gate. Reuse an existing
helper if present; no new GPU training without a bounded fit-only CPU pass.

## Bounded published-reference check

The [2025 LoCoRe paper, Table2](https://arxiv.org/html/2503.21772v1)
reports up to83.8% SOP and89.4% In-Shop R@1 across its variants on its
chosen global-descriptor panel, with100-image local-token re-ranking. These
points do not supersede the dated UNICOM reference, and do not certify the
latest global frontier. Its re-ranking gains do not transfer automatically
to this stronger packed encoder/scorer. No re-ranking experiment was launched.
