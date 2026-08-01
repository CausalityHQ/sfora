# Relational-distillation composition audit

## Question

The only intervention in the corrected matrix that replicated across CUB and
Cars is EMA relational distillation. Does its row-wise neighbourhood target
mainly transfer same-class soft positives, or graded relations to other classes?

## Diagnostic

For each of five saved final CUB HERD training packs, I sampled 500 deterministic
45-class × 4-image batches, matching the registered batch composition. I applied
the implemented τ=0.1 row-softmax to pairwise cosine similarities with the
diagonal removed, then measured the target mass assigned to the other three
same-class images.

| seed | mean same-class mass | median | 5th--95th percentile | nearest is same class |
|---:|---:|---:|---:|---:|
| 0 | 0.1110 | 0.1039 | 0.0500--0.1980 | 0.9507 |
| 1 | 0.0798 | 0.0753 | 0.0376--0.1377 | 0.8930 |
| 2 | 0.1042 | 0.0985 | 0.0479--0.1803 | 0.9397 |
| 3 | 0.0781 | 0.0737 | 0.0372--0.1341 | 0.8923 |
| 4 | 0.0935 | 0.0871 | 0.0431--0.1654 | 0.9182 |

Thus the closest relation is usually same-class, yet **88.9--92.2%** of the
softmax mass lies on different-class samples. The target is not merely a soft
same-class attraction; its composition is dominated by graded cross-class dark
relations. The uniform-composition baseline would allocate only 3/179 = 1.68%
to same-class samples, so the result also shows strong same-class enrichment.

## Scope and caveat

These packs contain final HERD embeddings. They do not expose the EMA teacher at
every training step, so this is a structural audit of the implemented target on
the converged representation, not a causal ablation of which mass produced the
gain. A deciding decomposition would require separately distilling conditional
same- and different-class distributions. That experiment is not warranted as a
novel method because Gate 2 is already occupied (candidate 49).

## Cross-seed stability of the dark geometry

All five packs share 5,864 image IDs. On 400,000 deterministic random image
pairs (395,829 different-class and 4,102 same-class after excluding self-pairs),
the ten independent seed-pair comparisons give:

| relation | mean Pearson similarity correlation | mean Spearman correlation |
|---|---:|---:|
| same class | 0.9098 | 0.9083 |
| different class | **0.8127** | **0.7936** |

The cross-class dark geometry is therefore substantially reproducible across
independent runs, not arbitrary seed noise. This still does not show which part
caused the retrieval gain, and using replica agreement as a target-reliability
filter is occupied multi-teacher agreement distillation (candidate 50).

## Class-pair versus image-level residual

On 500,000 deterministic cross-class pair draws, I decomposed each cosine into
the dot product of its two unnormalised class centroids plus an image-pair
residual. Class-pair means explain **52.57--58.90%** of raw cross-class variance.
Their cross-seed agreement is extremely high (mean Pearson **0.9037**, Spearman
**0.8921**), but the residual is also reproducible (Pearson **0.6980**, Spearman
**0.6756**). Thus stable dark geometry is not merely a relation among class
proxies: it retains which individual images depart from their classes' typical
relationship. Distilling that fixed-effect residual is nevertheless occupied
pairwise-difference relational distillation (candidate 51).
