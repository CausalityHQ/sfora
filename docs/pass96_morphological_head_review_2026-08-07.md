# Pass 96 — morphological/max–min head (2026-08-07)

## Dead at Gate 2

The positive-side transfer deficit suggests that average pooling may erase the
single local evidence that identifies a nearest positive.  A morphological
head would replace the final pooling with learned max/min (dilation/erosion)
aggregates and train Proxy Anchor on the resulting descriptor.

Primary retrieval work already occupies the mechanism: global pooling studies
find max pooling and generalized pooling to be strong fine-grained retrieval
operators, and adaptive-region pooling explicitly learns region aggregation for
fine-grained image retrieval.  A differentiable max/min or morphology layer is
therefore a pooling substitution, not a new supervision signal; its expected
effect is indistinguishable from existing max/attention/region-pooling controls.
No implementation or GPU run occurred.
