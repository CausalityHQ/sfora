# ALSP frozen falsifier: KILL

The train-only Amortized Local-Scale Potential (ALSP) fails its prospectively
frozen In-Shop seed-0 gate. The scalar potential is strongly predictable, but
subtracting it from cosine similarity makes retrieval worse.

## Result

- Raw Recall@1: `0.9137009424672949`.
- ALSP Recall@1: `0.9126459417639612`.
- ALSP gain: `-0.0010550007033337527` (80 wrong-to-right, 95
  right-to-wrong; exact paired `p=0.2898925237752064`).
- True gallery-density oracle Recall@1: `0.9161626107750739`.
- Oracle gain: `0.0024616683077789414` (110 wrong-to-right, 75
  right-to-wrong; exact paired `p=0.01222063657146658`).
- Predicted-vs-observed density: Pearson `0.7414744057355482`, Spearman
  `0.6264383103235855`, MSE `0.0017908569348296767`.
- Fixed scale sensitivity gains: `-0.00014066676044444115` at `0.5`,
  `-0.0010550007033337527` at `1.0`, and `-0.005415670277113427` at `2.0`.
- Observed-on-predicted calibration: slope `0.8052681998173947`, intercept
  `0.09179728125836972`.
- Selected normalized ridge lambda: `0.0001`.

Only correlation and both empirical-null predicates pass. Absolute gain,
oracle recovery, paired significance, and the target-permutation control fail.
The complete decision is therefore `passes_falsifier=false`.

## Interpretation

The cross-dataset density defect is real: the true gallery-local correction
again improves Recall@1, and a train-only linear head predicts much of the
gallery density variation. What fails is the stronger ALSP claim that the
predicted unary value can be inserted directly as a gallery-side score bias.
Even halving its fixed coefficient remains harmful, so this is not rescued by
the preregistered scale diagnostic.

This closes ALSP as a learning candidate and blocks its conditional GPU gate.
It does not close query-conditioned or pair-conditioned corrections, nor does
it establish a new SOTA method. The next candidate must explain why the true
test-graph density is useful while its accurately predicted unary surrogate is
not; it needs a new prospective design rather than tuning on this outcome.

## Provenance and verification

- Original design commit: `5062fb4`.
- Prospective review amendment: `7ed0912`.
- Frozen evaluator source commit: `4db0568`.
- Result JSON SHA-256:
  `a30100b25d2b62c0731232a5b3493dabb6162d0902c507beb8897ef80d26c40d`.
- Input SHA-256 values: train `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`,
  query `ef5278fd9aae7a6398a6c74133e6acc0ded05e39647087bdf78459223b9eb761`,
  gallery `6eb89ff57e7a6002f2ba71f9659e04dabd0cafdb1996be3d85f5211731ba861a`.
- The persisted v2 schema passed the production validator. A separate
  blockwise implementation reproduced the ALSP, oracle, and all three scale
  sensitivity rankings, recalls, paired transitions, and exact p-values.
- The result is a regular mode-0600 file with no sibling temporary artifact.

ALSP is adjacent to local scaling, Mutual Proximity, CSLS, and cohort
Z/T/S/AS-Norm. Its only defensible intended distinction was amortizing a
gallery-side potential without a test graph; this falsifier rejects that
distinction as an effective retrieval correction on the frozen pair.
