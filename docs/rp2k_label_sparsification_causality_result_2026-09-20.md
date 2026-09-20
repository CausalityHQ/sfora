# RP2K label-sparsification causality result

## Result

The preregistered diagnostic supports positive-label coverage as a causal
limitation of the current supervised compact projection.

| Arm | positive-eligible rows | updates/cycle | mMP@5 | R@1 |
|---|---:|---:|---:|---:|
| dense-label reference | 15,264 | 640 | 0.969804 | 0.984230 |
| sparse native schedule | 3,214 | 135 | 0.964311 | 0.980273 |
| sparse dense-matched updates | 3,214 | 640 | 0.958448 | 0.977771 |

The deterministic intervention changed only fit labels after the third row of
each class, leaving embeddings, validation, byte width, initialization,
negative bank, scorer, and tie behavior unchanged.  It reduced positive-
eligible fit coverage from 99.99% to 21.05%, close to MET-small's measured
20.51%.

The native sparse arm lost `0.005493` mMP@5.  Matching the dense arm's update
count did not restore the loss; it increased the drop to `0.011356`.  The frozen
classification is therefore `positive_coverage_supported`, not
`optimizer_exposure_supported`.

Post-result identification correction: the intervention assigned fresh
singleton labels to removed classmates while hard-negative eligibility uses
label inequality. It therefore both removes positive access and permits true
classmates to act as false negatives. The frozen classification describes the
registered intervention outcome, but it does **not** isolate positive coverage
as the sole cause. A clean successor must preserve original-class negative
exclusion while restricting positive access and matching anchor exposure.

## Consequence

The current supervised objective is useful when genuine repeated positives
cover the fit corpus, but training longer cannot make it generic on singleton-
dominated data.  A successor must turn unlabeled singleton rows into positive
information through query-independent teacher-neighborhood or augmentation
consistency.  It must also be tested against OPQ at the same total byte budget;
MET's strong OPQ result shows that a 64-dimensional rank bottleneck is not
automatically the best use of 512 bits.

This diagnostic reused already-observed RP2K validation and is claim-ineligible.
It selects a mechanism to investigate; it does not promote a new method.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_rp2k_label_sparsification_causality_v1.json`
- result SHA-256:
  `8a23ddbe446b168ec35d772b5cfc6f65d60cf59d7dc355672b3596ffcbf7e0ce`
- native sparse checkpoint SHA-256:
  `538042bcfae71f6f93eec52dac0e60b005602ffcd269726dd0288adf477d2799`
- matched-update checkpoint SHA-256:
  `c489c5803ef769688e883d1545e7cb61edfb97fa93e7cd9e089ddde3b195db07`
- driver SHA-256:
  `9158a3e85a191d06edd4f440d81a6468c92d38474dc659e404e4e7f63c6be22a`
- runtime: 20.83 seconds on the registered DGX environment.
