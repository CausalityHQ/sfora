# OAPF pre-data Claude audit and adjudication

This audit was completed on 2026-08-02 before generating any OAPF diagnostic
embeddings or outcomes. Claude Opus was used as an adversarial reviewer; its
claims are treated as criticism, not evidence. The frozen operating artifacts
are:

- checkpoint SHA-256 `31307c9e0ce816397e3d3b3ff3f0084dc84b3ef47e8e9847ecbc71fa3b97fcbd`;
- training-report SHA-256 `e84aa1b7a0e3ee052b5bd4ce13a6a8e77396cb4f4738797a83853c4f4ded92cc`.

## Accepted implementation corrections

The first helper implementation could materialise several `[pairs, 6, 512]`
float64 arrays, did not enforce six L2-normalised views on the radius path, and
did not test the registered 9/12 versus 10/12 outcome boundary. Those are real
defects and must be repaired before data. The final script must also state that
the five held-out folds test the diagnostic regression across classes; the
epoch-10 encoder itself was trained on every training class. Training-only kNN
and nearest-negative computations may use the entire frozen training set and
are therefore transductive features of this diagnostic, not an independent
encoder-generalisation test.

## Rejected claims of fatal circularity

The audit called pack-A RMS dispersion and pack-A q90 radius circular because
they summarise the same six-view cloud. That is not the registered inference.
RMS is deliberately the adjacent ordinary-dispersion control; the incremental
question is whether tail extent adds information beyond it. Sharing draws
removes an avoidable pack-level nuisance. A pass licenses only that narrow
incremental statement, not a new latent construct.

The audit also objected that held-out compatibility necessarily changes with a
held-out augmentation displacement. That dependence is the phenomenon under
test, not label leakage: pack B never defines the predictor, and the target is
not thresholded by a pack-B radius. The registered positive direction is
deliberately difficult because crop instability can mechanically reduce
compatibility. If that adverse effect dominates, the proposed mechanism—large
orbit means a larger zero-force plateau because less attraction is needed—is
wrong and OAPF dies. The result must not be broadened into a claim that orbit
response contains no information at all.

Likewise, framing and pose are not nuisance variables for this candidate: they
are candidate sources of per-image tolerance. Within-class derangement asks
whether assigning the measured endpoint response to its own image matters;
it does not claim causal identification of an acquisition-independent factor.

## Limitations retained prospectively

- The direction test is partly redundant with incremental AUC when the positive
  constraint is active. It remains a sign falsification, not independent
  evidence.
- Six samples make q90 a noisy tail estimate. The independent-pack Spearman gate
  is explicitly the cheap test of whether that estimate is reliable enough.
- The official `scale=(0.08, 1.0)` crop can omit the garment. That is intentionally
  the exact augmentation distribution encountered by the reference training
  recipe; a pass concerns robustness to that recipe, not abstract semantic
  transformations.
- `rho_i = 0.2 r_i / median(r)` fixes scale but does not bound its spread. If the
  diagnostic passes, the observed prospective spread must be reported before
  training. It cannot be clipped or tuned after inspection.

These adjudications change implementation safeguards and the strength of the
allowed interpretation. They do not change any numerical gate or permit OAPF
training. A pass still only advances the candidate to faithful fixed-bandwidth
PFML reproduction.
