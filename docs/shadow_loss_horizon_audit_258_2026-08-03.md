# Shadow Loss horizon and algebra audit (258)

Date: 2026-08-03. Primary manuscript checked before any implementation or GPU.

## Why it was checked

The online speed horizon surfaced Khan et al., *Shadow Loss: Memory-Linear Deep
Metric Learning with Anchor Projection* (ICLR 2026 withdrawn submission; earlier
arXiv:2311.14012). It claims roughly 1.5--2x faster convergence and evaluates CUB,
Cars196, SOP, and In-Shop, making it an apparent occupant or inspiration for the
standing faster-or-better objective.

## Algebraic audit

The manuscript defines, for anchor `a`, positive `p`, and negative `n`,

`pi_a(x) = a·x / ||a||`, `delta_x = | ||a|| - pi_a(x) |`, and
`max(delta_p - delta_n + margin, 0)`.

Its analysis and experiments require L2-normalized embeddings. Then `||a||=1` and
`a·x` is in `[-1,1]`, so the absolute value is redundant:

`delta_p = 1 - cos(a,p)` and `delta_n = 1 - cos(a,n)`.

Therefore the inner loss is exactly

`cos(a,n) - cos(a,p) + margin`.

Squared-Euclidean triplet loss on the same unit vectors is

`2[cos(a,n) - cos(a,p)] + margin`.

The two have identical ordering and gradients up to a constant factor and margin
rescaling. “Projection onto the anchor axis” does not define a new similarity-learning
operator; it is the standard cosine form of normalized triplet loss.

## Evidence quality and benchmark relevance

The OpenReview record labels the ICLR 2026 submission **withdrawn**. The manuscript's
large-scale results are SOP R@1 **69.94** and In-Shop R@1 **72.33**, compared only with
triplet-family baselines. Those are far below the upstream Proxy Anchor references
(about 79.1 SOP and 91.9 In-Shop) and cannot occupy the quality horizon. The claimed
loss-buffer saving also leaves encoder activations and the batch similarity/mining work
intact; it is not evidence of end-to-end 1.5--2x wall-clock savings against a faithful
proxy baseline.

## Verdict

**DEAD at Gate 2; no implementation or GPU.** Shadow Loss is normalized cosine
triplet under a new geometric description, and its weak/withdrawn benchmark evidence
does not justify replacing Proxy Anchor. It is useful as an adversarial lesson: a
cross-field projection story must be reduced algebraically before being treated as a
new mechanism.

Primary sources:

- Khan et al., arXiv:2311.14012, equations 8--10 and 16--19,
  https://arxiv.org/abs/2311.14012.
- Withdrawn OpenReview record, https://openreview.net/forum?id=3fx0Kz6Zfl.
