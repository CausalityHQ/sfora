# UniCOM EMA × imprint mechanism predictions

## Timing and scope

This note was frozen after the seed-0 imprinted arm had emitted its epoch-4
instrument row and before it emitted epoch 8, 12, or 16. The observed epoch-4
instrument values were mAP@R `0.8967302587153507` and Recall@1
`0.986198243412798`; the observed epoch-5 and epoch-6 training losses were
`4.306671477252652` and `4.0146084764729375`. These values motivated the
predictions below but do not change any registered gate, threshold, seed, arm,
checkpoint, or evaluation rule.

The predictions are explanatory tests, not additional promotion conditions.
The strict factorial report remains the sole seed-0 decision authority.

## Mechanism hypothesis

Class-mean imprinting initializes each normalized classifier row near the
ArcFace objective's class-mean fixed point. Relative to a random head, this
should reduce early gradients that distort pretrained open-set features during
the high-learning-rate portion of the OneCycle schedule. The parameter EMA is
initialized from the pretrained state, so early EMA checkpoints should behave
partly as a pretrained-to-finetuned interpolation. Imprinting and EMA are
therefore predicted to be partly substitutable rather than additive at the
endpoint.

## Frozen predictions

1. **Endpoint advantage:** the hardened seed-0 `imprinted_raw` epoch-16 mAP@R
   exceeds `random_raw` epoch 16 by at least the already-registered promotion
   margin `0.003`.
2. **Shrinking imprint gap:** the hardened `imprinted_raw - random_raw` mAP@R
   gap is positive at epochs 4, 8, 12, and 16 and decreases monotonically over
   those checkpoints.
3. **Sub-additive EMA endpoint:** at epoch 16,
   `abs(imprinted_ema - imprinted_raw) < 0.0015`, and that absolute EMA effect
   is smaller than `abs(random_ema - random_raw)`.
4. **No imprint leakage into the backbone:** before optimization, the imprinted
   arm's backbone embedding bytes equal the initial pretrained backbone's
   embedding bytes for the same ordered holdout inputs. Any mismatch invalidates
   the mechanism interpretation and requires a pipeline audit.

All four predictions will be reported as written even when the factorial gate
closes. No failed prediction may be repaired by changing EMA decay, imprint
norm, epochs, thresholds, or checkpoint selection.

## Prior-art boundary

Class-mean weight imprinting and linear-probe-then-fine-tune are established
ideas; this experiment does not claim to invent them. Its contribution, if the
paired evidence confirms it, is a controlled measurement of their interaction
with parameter EMA and open-set retrieval under the fixed UniCOM recipe.
