# Pass 68 — CIS Gate-2 and algebra review

Verdict: **DEAD; no GPU run.**

## Algebraic failure

For `Z_ic = <f_i,p_c>` and a bundle `S`, the proposed logit is
`u_c = <sum_i f_i,p_c> = sum_i Z_ic`.  The nonlinear loss does create a
cross-sample Hessian, but it does not create ownership-aware supervision.  For
an unnormalised hard-positive BCE (and likewise any active monotone positive
term), `dL/dZ_ic = g_c` is identical for every member of `S`.  A positive label
for class `c` therefore pushes the non-owner images toward proxy `p_c` with the
same sign as the owner.  This directly rewards the cross-class interference
the mechanism claims to remove.  Saturation makes the term vacuous; there is
no active ownership-selective regime.

Normalising the bundle changes the gradient through the shared norm
`r^2=m+2 sum_{i<j}<f_i,f_j>`, but that only adds an uncontrolled angular
coupling.  It does not identify which member owns a proxy.  The negative
aggregate also permits cancellation: it constrains `sum_i Z_ic`, not the
per-image maximum or positive tail.  Thus it supplies no bound on the
per-query collision that matters at retrieval.

For `m=2`, the normalised sum has exactly the same direction as a 1/2 embedding
mixup vector; hard `(1,1)` targets merely add a uniform wrong-direction pull
relative to mixup's `(0.5,0.5)` targets.  An ownership-aware cross-member
penalty or a max/log-sum-exp over non-owner logits would be a different method,
not CIS, and would need a new Gate-2 search.

## Prior-art checks

The construction is also too close to established compositional and
multi-label proxy training to claim novelty.  Deep Compositional Metric
Learning (Zheng et al., CVPR 2021) trains losses on composites of sub-
embeddings; Compositional Embeddings for Multi-Label One-Shot Learning (Li et
al., WACV 2021) trains set-valued label compositions; and Xing et al.,
“Multimorbidity Content-Based Medical Image Retrieval Using Proxies”
(arXiv:2211.12185) explicitly assigns one image to multiple class proxies for
retrieval.  Metrix (ICLR 2022, “It Takes Two to Tango: Mixup for Deep Metric
Learning”) supplies the direct mixed-embedding/mixed-label control.  None needs
to be identical for CIS to fail: the gradient defect alone means the proposed
signal is not causally aligned with the Gate-1 between-class failure.

## Gate outcome

Pass 67 supplied the 51.9% between-class provenance, but CIS cannot act on that
quantity with its symmetric hard-positive decoder.  The frozen 0.9115 In-Shop
forecast is therefore void; no implementation, screen, or selection-corrected
number is authorized.
