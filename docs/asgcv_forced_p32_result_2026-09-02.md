# ASG-CV forced-verdict P32 result

Date: 2026-09-02  
Status: **TRAIN-ONLY DIAGNOSTIC PASS — claim ineligible**

## Result

The committed forced-verdict pilot at source
`58780461b97c94b2b2d26aa7a6a7ba13a75ca035` evaluated 32 balanced Cars196
training-band pairs with Qwen3-VL-8B revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.  It compared the teacher-forced
prefix likelihoods of `SAME` and `DIFFERENT`; it did not sample or generate a
completion.

The canonical result is 33,851 bytes with SHA-256
`d239fee6ce4dd2a2804c5cfd8da58786aa534253dc938793b577086098097266`.
All 32 candidate receipts and the result are retained on the DGX under
`/home/riomus/sfora-asgcv-forced-results-58780461b97c94b2b2d26aa7a6a7ba13a75ca035`.

| Metric | Result | Frozen gate | Outcome |
|---|---:|---:|---|
| pair accuracy | 875,000 ppm | 625,000 ppm | pass |
| ROC AUC | 992,188 ppm | 700,000 ppm | pass |
| SAME recall | 750,000 ppm | 500,000 ppm | pass |
| DIFFERENT recall | 1,000,000 ppm | 500,000 ppm | pass |

Every gradient was finite and nonzero.  Gradient norms ranged from
`1.7825445188995622e-05` to `16.803231592566092`.  Repeated ordinals 0 and 31
reproduced their exact gradient SHA-256 digests.  Peak recorded CUDA reserved
memory was 22,873,636,864 bytes and peak process RSS was 18,571,636,736 bytes.
Memory PSI full avg10 was 0.00 at terminal.  No official-test artifact was
opened, `official_test_access=false`, `generated_tokens=0`, and
`claim_eligible=false` are enforced by the canonical result schema.

## Interpretation

The earlier sampled eight-completion pilot produced seed-invariant verdicts and
no mixed-verdict groups.  That failure does not show that the dataset lacks
semantic relation signal.  It shows that stochastic free-form generation is a
bad gradient-acquisition boundary for this model/prompt.  Removing the terminal
EOS target and scoring only the demanded verdict prefixes changes the balanced
P32 diagnostic from chance-level accuracy to 87.5% with near-perfect ranking.

This is evidence for a cheap amortized supervisor, not a retrieval result.  The
next gate captures relation-correct forced gradients on the predictor-training
class band, trains the existing rank-16 patch-gradient predictor, and evaluates
gradient agreement once on a disjoint held-out training-class band.  Only a positive
gradient-transfer gate warrants integrating the student into metric learning.

## Forced-gradient distillation outcome

That next gate is now terminal and **negative**.  Source
`635ac8fe64d623b8ab2469b0a3eff39ce003fd04` captured 128 balanced predictor-training
pairs and 32 disjoint `e1_optimization` pairs, then trained the frozen rank-16
student for 20 ordinal-order epochs.  The canonical result is 1,793 bytes with
SHA-256 `46f131cf5c3dd67ca00ebb5b98e253c3abeabdb1f4c913d288f66a95c4d4d84a`;
the canonical predictor state SHA-256 is
`f3328c5eb8687e7f4d4461b79d4b88b4f936a25ed54c3183fcb20aa5620b8d82`.

| Metric | Result | Frozen gate | Outcome |
|---|---:|---:|---|
| median dense cosine | -50 ppm | 500,000 ppm | fail |
| positive-cosine rate | 468,750 ppm | 750,000 ppm | fail |
| nonzero-prediction rate | 1,000,000 ppm | 1,000,000 ppm | pass |

All 128 train triples and 32 optimization triples reopened successfully; no partial
files remained, the terminal process cleared, and memory PSI full avg10 was 0.00.
The earlier all-zero prediction defect is therefore fixed, but the live student is
directionally uninformative.  Its median training-pair cosine is only
`8.02e-05`, so this is not a held-out-only generalization failure.

Post-result analysis on the now-burned train/optimization bands localizes the
failure.  Exact gradients span more than six orders of magnitude in norm
(`8.43e-06` to `15.24` on the training band), while the frozen objective normalizes
each example independently.  More importantly, the target fields themselves are
compressible but their coefficients are not recovered by the current input/model:

- individual rank-16 matrix approximations retain roughly 0.89--0.99 cosine on
  sampled targets;
- a rank-16 channel subspace fitted on train-only gradients has held-out median
  projection cosine 0.542, and rank 64 raises it to 0.665;
- a scale-stable closed-form patch-to-rank-64 coefficient probe reaches only 0.066
  median cosine on its training pairs and 0.046 on the burned optimization pairs.

The scientific conclusion is narrow: reject this patch-token-to-dense-gradient
student, not the forced-verdict teacher signal.  The next method should distill the
teacher's strongly ranked scalar verdict/margin into the retrieval objective (or a
small pair scorer), instead of reconstructing a 2,097,152-value gradient field.
No official-test artifact was opened and none of these diagnostics is claim eligible.
