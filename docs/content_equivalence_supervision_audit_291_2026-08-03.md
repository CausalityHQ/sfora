# Content-equivalence supervision audit 291

Date: 2026-08-03. Gate 1/2 audit only; no implementation or GPU.

## Measured provenance

Audit 288 found 690 exact-file duplicate groups in official SOP train, but only
12 groups (24 of 59,551 rows, **0.0403%**) cross product labels. A proposed
supervision change would hash canonical file content and change a negative pair
to unknown whenever two different labels have identical bytes. Within-label
duplicates could be sampled once per epoch to save redundant backbone passes.

The causal ceiling is too small for a meaningful quality result: even perfect
handling directly changes supervision for only 24 training rows. Removing all
duplicate rows could reduce unique SOP image decoding/forward exposure by about
1.7%, but ordinary exact deduplication—not a new learning algorithm—provides that
gain.

## Prior art

The learning mechanism is occupied even though exact hashing is a particularly
high-precision detector:

- Liu et al., *Noise-resistant Deep Metric Learning with Ranking-based Instance
  Selection* (CVPR 2021), introduce PRISM to identify and suppress noisy DML
  instances using similarity and memory:
  https://arxiv.org/abs/2103.16047.
- Yu et al., *Enhancing Sample Utilization in Noise-Robust Deep Metric Learning
  With Subgroup-Based Positive-Pair Selection* (2025), explicitly select reliable
  positive pairs/prototypes for suspicious DML samples rather than trusting the
  original binary relation: https://arxiv.org/abs/2501.11063.
- Huynh et al., *Boosting Contrastive Self-Supervised Learning With False
  Negative Cancellation* (WACV 2022), identify false negatives and either remove
  or attract them:
  https://openaccess.thecvf.com/content/WACV2022/html/Huynh_Boosting_Contrastive_Self-Supervised_Learning_With_False_Negative_Cancellation_WACV_2022_paper.html.
- Balmaseda et al., *Discovering Global False Negatives On the Fly for
  Self-supervised Contrastive Learning* (ICML 2025), learn global per-anchor
  false-negative thresholds with dataset-size-independent per-step cost:
  https://proceedings.mlr.press/v267/balmaseda25a.html.

## Verdict

**DEAD at Gate 1 and independently Gate 2.** The measured conflicting set is too
small to support the expected benchmark effect, and negative-to-unknown handling
of suspected label conflicts is established noise-robust/false-negative learning.
Replacing learned suspicion with an exact content hash improves detector precision
but does not create a new supervision mechanism. Hash deduplication remains a
valid data hygiene and modest throughput policy, not a novel similarity method.
