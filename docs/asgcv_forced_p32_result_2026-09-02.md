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
gradient agreement once on the disjoint validation class band.  Only a positive
gradient-transfer gate warrants integrating the student into metric learning.

