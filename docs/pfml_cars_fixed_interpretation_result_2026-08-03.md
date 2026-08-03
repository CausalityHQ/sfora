# Cars196 PFML fixed-interpretation result

Date: 2026-08-03. Verdict: **FAILED BOTH PREREGISTERED METRIC GATES; NO
TUNING, FOLLOW-UP SEED, OR CANDIDATE DERIVATION.**

## Result

The repaired local interpretation completed 200 epochs / 16,200 optimizer
steps on the pinned standard Cars196 first-98/last-98 class split. Its primary
final-state R@1 is **0.793137**. The raw best-over-training, selected from 20
fixed-cadence test evaluations, is **0.836305 at epoch 70**. The locked gates
were final at least 0.890 and raw best at least 0.895, so the misses are
**-9.686 points final** and **-5.869 points raw best**. All scalar states were
finite.

This falsifies the prospectively disclosed local reading only. It does not
falsify the CVPR 2025 PFML paper because the authors did not publish executable
code and the primary sources leave augmentation, sampling, proxy normalization,
gradient clipping, head initialization, pretrained-weight digest, and the
sum/mean implementation under coupled Adam incompletely identified. The main
paper and supplement also disagree on the base learning rate.

## Artifact and corpus verification

- report SHA-256: `e1574317fce7242fd02b20d6b6a4c9ddb024382562458ae26c613c6b8c09998f`
- final checkpoint SHA-256: `e8daec24afc80ac4d314f00efc032d80b6bb6c41b69425cc003a7288f1ab0274`
- final test pack SHA-256: `771ce5c8d3ceb61a200df9c5b34c54324ec92105e338c992a52c0f31ba390283`
- scalar audit SHA-256: `0a00d83b0f0d3e3a2e8860459fa4bb61dba70d5e0393a87f93d08fb9c8d4090f`

The checkpoint is explicitly `final_training_state`, step 16,200, student
source, and its full training config matches the report. Independent reload used
Hugging Face revision `9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`: 8,054 train images / 98 classes
and 8,131 test images / 98 disjoint classes. There is zero cross-split identity,
example-ID, decoded-content, or source-reference overlap. The audit found 30
within-train and 9 within-test repeated decoded-content rows; they are retained
as corpus properties and do not cross the zero-shot partition.

Re-encoding the final checkpoint reproduces production R@1 exactly at
**0.793137**. A stable full ordering gives **0.792768**, three queries lower.
The difference comes from rounded squared-distance ties: the benchmark scorer
uses partial `argpartition` followed by a stable ordering of the selected set,
whereas the diagnostic orders the full gallery stably. This is a **0.0369-point
tie sensitivity**, far too small to affect either failed gate. An earlier
verifier mistakenly renormalized the already torch-normalized vectors and then
used a global argmin; that false discrepancy was rejected and the regression
test now binds exported-coordinate squared-L2 and the production top-k rule.

## Mechanism and authorization

The curve peaks at epoch 70 and ends 4.317 points lower, but the run preserves
only scalar curves and one final checkpoint; it cannot support sample-level
trajectory claims. The raw loss is dominated by force-free constant potential
terms and is not interpretable as active force. Because the reference failed,
the controller correctly skipped the train-pack export, proxy occupancy, and
field census. No PFML hyperparameter tuning or method proposal is authorized
from this ambiguous failed baseline.
