# Reference training-batch fidelity audit 240

Date: 2026-08-03.

## Defect

The pinned Proxy Anchor implementation constructs its ordinary shuffled
training loader with `drop_last=True`. SFORA's registered BN-Inception Proxy
Anchor recipes inherited a loader with `drop_last=False`, and their epoch
schedule used the ceiling of image count divided by batch size. This was found
from the live corrected-SOP step count, not from a score: 19,860 instead of the
upstream 19,800.

The mismatch adds one smaller optimizer batch per epoch and shifts the number of
updates in warm-up and each scheduler interval. For official SOP the difference
is 331 versus 330 updates per epoch. For official In-Shop it is 144 versus 143.
It is not honest to assume the score effect is zero merely because the update
count difference is small.

## Scope of old claims

Historical In-Shop Proxy Anchor and derived arms remain internally paired under
the same SFORA batch policy, so this discovery alone does not mathematically
erase their within-harness deltas. They must, however, be labeled modified
reproductions rather than exact upstream-recipe results until repeated with the
correct loader. Their absolute comparison with the published 0.9035 baseline is
not artifact-grade evidence of faithful reproduction. No old In-Shop value is
silently relabeled as corrected.

The earlier SOP results were already retracted for the larger official-split
defect. Both attempted corrected-SOP launches were stopped before producing a
report: the first used the wrong backbone and the second exposed this batch
policy mismatch.

## Repair and regression condition

`ImageEndToEndConfig.drop_last_train_batch` now controls both `DataLoader` and
epoch-to-step resolution. It defaults false to preserve project protocols and
archived CUB/Cars recipe digests, but the registered BN-Inception Proxy Anchor
recipes explicitly set it true, matching upstream. Unit tests lock the SOP
schedule to 330 updates per epoch / 19,800 total and retain existing CUB/Cars
recipe digests. The corrected run must print 19,800 total steps before its score
is admissible.

## Independent warm-up defect found before restart

The same command audit found that SFORA's warm-up exclusion recognized the new
ResNet head only by `fc.*`. BN-Inception calls it `model.embedding.*`, so prior
BN-Inception runs froze the embedding head together with the pretrained
backbone during epoch 1 and trained only the proxies. Upstream explicitly keeps
both `model.embedding.parameters()` and the Proxy Anchor criterion parameters
trainable. This is a mechanism-level mismatch, not merely a step-count issue.

Historical In-Shop BN-Inception comparisons therefore share two deviations:
the incomplete batch was kept and the new embedding head was incorrectly
frozen during warm-up. Their paired deltas remain observations from the same
modified harness, but neither their absolute baselines nor claims of exact
official-recipe evaluation are valid. The repair excludes both `fc.*` and
`model.embedding.*` from backbone freezing and locks that behavior in a unit
test.
