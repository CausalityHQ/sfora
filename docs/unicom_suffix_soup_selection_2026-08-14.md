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
  run paired training seeds 1, 2, and 3. A quality claim requires all four
  paired deltas positive and a paired 95% confidence interval above zero, plus
  measured training time, inference latency, and storage cost.
- If the gate fails, close checkpoint averaging for this trajectory and test
  the already researched next candidate: step-level EMA with class-mean
  classifier initialization. Do not tune this soup grid again on the same
  holdout.

The combined 28-candidate grid is deliberately frozen in one pass. It includes
both the original coarse grid and the independently proposed near-endpoint
alphas without allowing a second holdout-adaptive search.
