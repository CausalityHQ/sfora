# Candidate 201: training-proxy response coordinates — Gate-2 audit

Date: 2026-08-02. No implementation or candidate GPU was performed.

## Provenance and construction

The repository measured **99.975%** proxy-to-own-centroid assignment but only
**70.303%** centroid-to-own-proxy assignment. That makes the learned proxy bank
a plausible stable relational coordinate system even though it does not justify
another reciprocal-training loss. A horizon re-scan also found Mou et al.,
*How Classification Baseline Works for Deep Metric Learning: A Perspective of
Metric Space* (ACML/PMLR 2025), which compares examples using a Cross-Entropy-
derived metric on their class-probability vectors.

Candidate 201 would represent an image embedding `x` by its similarities or
softmax responses to all **training-class** Proxy Anchor proxies and compare two
unseen-class images in that response space. It uses no external data, but must
still yield one 512-D descriptor.

## Exact linear death

Let the unit proxy rows form `P in R^(C x 512)` and let `s(x) = P x`. Cosine in
the response space is

`cos(Px, Py) = (x^T P^T P y) / sqrt((x^T P^T P x)(y^T P^T P y))`.

With `A = (P^T P)^(1/2)`, this is **exactly** `cos(Ax, Ay)`. The candidate is a
fixed PSD Mahalanobis remapping that can be folded into the existing linear
embedding head. It adds neither information nor supervision. On CUB/Cars its
rank is at most the 100/98 training classes; on In-Shop it can have full rank,
but it remains the same separable linear map. Compressing or learning another
map after the responses merely creates a second embedding head.

## Nonlinear response prior art

Softmax makes the coordinate map nonlinear, but not novel. The construction is
the classic dissimilarity representation: describe an object by its relations
to a prototype set. Pękalska, Duin, and Paclík explicitly define a representation
set of prototypes and classify from the resulting dissimilarity coordinates.
More directly, Torresani, Szummer, and Fitzgibbon's **Classemes** descriptor is
the output vector of a bank of category classifiers and is designed for compact
retrieval/classification of novel categories. Fixed-budget landmark selection,
projection, or coding changes its compression, not its mechanism.

Mou et al. do not fill an open-set gap: their experiments are CIFAR-10/100 with
the same class vocabulary at training and evaluation, and their output dimension
equals the class count. Once the axes are foreign training labels rather than
the evaluated labels, their vector becomes the established prototype/classifier-
response representation above.

## Verdict

**DEAD at Gate 2.** Linear responses reduce exactly to a fixed Mahalanobis head;
nonlinear responses are occupied dissimilarity/Classemes descriptors. The
99.975% ownership measurement motivates checking the bank, but does not create a
new relation, and no positive benchmark result could rescue the novelty claim.

Primary sources:

- Mou et al., PMLR 260 (2025):
  https://proceedings.mlr.press/v260/mou25a.html
- Torresani, Szummer, and Fitzgibbon, *Efficient Object Category Recognition
  Using Classemes*, ECCV 2010:
  https://www.microsoft.com/en-us/research/wp-content/uploads/2010/09/TorresaniSzummerFitzgibbon-classemes-eccv10.pdf
- Pękalska, Duin, and Paclík, *Prototype selection for dissimilarity-based
  classifiers*, Pattern Recognition 39(2), 2006:
  https://doi.org/10.1016/j.patcog.2005.06.012
