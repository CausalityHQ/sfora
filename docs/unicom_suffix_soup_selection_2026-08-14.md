# UniCOM suffix-soup selection freeze

This note freezes the train-only selection search before the seed-0 epoch-16
checkpoint or any soup holdout score is observed.

- Training source: commit `53662d5306a91577819047b12c3b01078cb923ea`.
- Screen: official UniCOM ViT-L/14@336 In-Shop recipe, seed 0, epochs 16,
  checkpoints 4/8/12/16, global batch 128, FP32, official eight-mask
  PartialFC-equivalent objective and official 512-coordinate evaluation.
- Selection data: the already frozen 20% train-identity holdout, seed 0. Test
  query/gallery data are not consulted during selection.
- Candidate windows: every suffix of `[4, 8, 12, 16]`.
- WiSE interpolation alphas: `[0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 1.0]`.
- Every candidate receives a fresh full optimization-split BatchNorm
  recalibration. Selection is by holdout mAP, then holdout Recall@1, then the
  selector's frozen deterministic tie-break. The selected state is recalibrated
  again before atomic publication.
- Endpoint control: window `[16]`, alpha `1.0`, evaluated by the same path.
- Seed-0 promotion gate: selected-minus-endpoint holdout mAP at least `0.003`
  absolute, with Recall@1 no worse by more than `0.00125` absolute.
- If promoted, freeze the selected window and alpha without further tuning and
  run paired training seeds 1, 2, 3, 4, 5, and 6. Seed 0 is selection-only and
  is excluded from every confirmatory statistic because its winner is selected
  from the 28-candidate grid. The six fixed replications are the only
  training-seed sample used for the claim; there is no outcome-adaptive seed
  extension.
- The primary method-level effect is selected-minus-endpoint holdout mAP for
  each training seed. Report the six deltas, their mean, sample standard
  deviation (`ddof=1`), and the two-sided 95% paired Student-t interval
  `mean +/- 2.5705818356363146 * sample_sd / sqrt(6)` (five degrees of
  freedom). This interval estimates uncertainty across training seeds. It is
  distinct from every within-seed query bootstrap interval.
- A quality claim requires all six mAP deltas strictly positive, the paired
  Student-t lower bound strictly above zero, the exact two-sided sign-test
  p-value at most 0.05 (six positive replication signs give
  `2 / 64 = 0.03125`), and every
  paired Recall@1 delta at least `-0.00125`. The sample standard deviation must
  also be strictly positive: duplicated or relabelled evidence and an exactly
  zero-variance six-run sample cannot support the claim. Each seed's selected
  and endpoint checkpoint paths must be internally valid; the summary hashes
  the actual checkpoint bytes and requires every seed's digest set to be
  disjoint from every other seed. Renaming or copying one run cannot create an
  independent replication.
- The summary command consumes the seed-0 screen JSON and the six fixed-report
  JSON paths, computes the screen file's SHA-256 itself, and requires every
  fixed report to bind that exact digest and the same frozen training protocol.
  It does not accept a caller-supplied digest or in-memory report mapping.
- A quality claim also requires measured training time, inference latency, and
  storage cost. Four runs are not used for a
  confirmatory claim because even four positive signs have exact two-sided
  p-value `2 / 16 = 0.125`.
- If the gate fails, close checkpoint averaging for this trajectory and test
  the already researched next candidate: step-level EMA with class-mean
  classifier initialization. Do not tune this soup grid again on the same
  holdout.

The combined 28-candidate grid is deliberately frozen in one pass. It includes
both the original coarse grid and the independently proposed near-endpoint
alphas without allowing a second holdout-adaptive search.
