# Pass 124 — SRC implementation-level distinction audit

**Status: LIVE-NARROW; no GPU authorization yet.**

This audit is the required second look before the preregistered SRC screen.
It is deliberately conservative: a difference in notation or a new weight is
not treated as a new method.

## Mechanism under review

For a training-only bundle of distinct-class images (x_1,ldots,x_m), SRC
forms a normalized real-image sum and supervises it against the union of the
member class proxies.  For each omitted member (j), it then forms the
leave-one-out sum of the other real images and supervises that residual against
the **complementary proxy set** (all bundle labels except (j)).  The deployed
descriptor remains the ordinary 512-D single-image embedding.  The decisive
claim is that each omission supplies a different target, so the model is
trained to encode which class contribution is missing, not merely that an
arbitrary mixture is a valid multi-label sample.

## Adversarial prior-art comparison

* **Deep Compositional Metric Learning (Zheng et al., CVPR 2021)** learns
  sub-embeddings and learned compositors, then applies metric losses to
  composites.  The checked paper does not form a sum of *real images* and does
  not train leave-one-out composites against complementary class-proxy sets.
  SRC would be dead if its implementation reduced to learned sub-embedding
  composition; the current code has neither sub-embedding branches nor a
  compositor.

* **HSE / mixed-sample metric learning (Yang et al., ICCV 2023)** creates
  additional mixed samples and labels.  Mixing or union supervision alone is
  the CIS control and is not claimed as SRC.  The residual mode adds one
  complementary target per omitted member, which must beat the union-only and
  per-image complementary controls to establish that this extra object matters.

* **Your Dissimilarities Define You: Complementary Learning Exploiting Class
  Diversities (Katsikas et al., CVPR 2026)** assigns per-image distributions
  over non-target classes.  This is the closest warning: it occupies
  complementary non-target supervision for a single image.  It does not, in
  the checked primary paper, construct a multi-image coalition or a separate
  leave-one-out complementary target for every omitted member.  SRC therefore
  remains live-narrow only, with a mandatory per-image complementary-target
  control; any claim that ignores this comparison is invalid.

* **Proxy Synthesis (Gu et al., AAAI 2021)** creates synthetic classes and
  proxies to improve unseen-class boundaries.  SRC does not synthesize a new
  class or proxy and instead uses real-image sums plus omission-specific
  targets.  This is adjacent, not mechanism-identical.

## CPU evidence

The implementation has finite loss/gradient, bundle permutation-invariance,
changed-target-under-omission, and non-equivalence-to-single-image tests.  The
focused objective/recipe check currently passes 3 coalition tests (the full
focused suite previously passed 65 tests).  These tests establish only that
the proposed object is implemented; they are not benchmark evidence.

## Decision

SRC is **not called novel** and is not queued while Pass119 is active.  If the
primary controller and its corrected random control finish cleanly, the next
authorized action is one matched-compute In-Shop seed with the preregistered
prediction 0.9192, falsifier 0.9180, and mandatory union/no-residual and
per-image-complementary controls.  Raw best-over-training and final/frozen
values must both be reported; no second dataset follows unless the mechanism
survives those controls.

## Primary sources

- Zheng et al., *Deep Compositional Metric Learning*, CVPR 2021:
  https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Deep_Compositional_Metric_Learning_CVPR_2021_paper.html
- Katsikas et al., *Your Dissimilarities Define You: Complementary Learning
  Exploiting Class Diversities*, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Katsikas_Your_Dissimilarities_Define_You_Complementary_Learning_Exploiting_Class_Diversities_CVPR_2026_paper.html
- Gu et al., *Proxy Synthesis: Learning with Synthetic Classes for Deep Metric
  Learning*, AAAI 2021:
  https://ojs.aaai.org/index.php/AAAI/article/view/16236
