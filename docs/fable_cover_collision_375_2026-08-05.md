# Candidate 375: CoVeR (Compositional Vocabulary Retrieval)

Date: 2026-08-05. Status: **DEAD at Gates 1 and 2.** No diagnostic,
preregistration, implementation, or GPU.

## Frozen proposal and independent review

A strictly isolated Fable proposer returned CoVeR: turn each image into a
512-dimensional square-root histogram of sparse patch-to-vocabulary assignments,
then use Proxy Anchor with sparse nonnegative class proxies in that histogram
space. A balance term and a proposed per-atom reuse floor were intended to make
the vocabulary class-transcending. The frozen proposal forecast 0.795 CUB,
0.944 In-Shop, and 0.954 Cars196 with a ResNet-50/512-D single-view descriptor.

The exact proposer artifact is
`docs/fable_cover_proposal_375_2026-08-05.txt` (SHA-256
`ed051b0b6eba56f4127542561dad8b72788406ce78646c74e62ecc4a827efe40`).
A separate clean Fable session reviewed that immutable text and returned
**DEAD**. Its exact artifact is
`docs/fable_cover_review_375_2026-08-05.txt` (SHA-256
`e832ffe2d9786f7673ff99279716c80dedd307009712a2f613b839f5438e79ae`).

## Gate 1: no eligible repository provenance

No measurement in the verified evidence packet establishes CoVeR's proposed
causal premise: that class-template absorption is a material source of corrected
benchmark errors, that shared patch-word histograms preserve more unseen-identity
information than continuous pooled features, or that sparse proxy reuse repairs
those errors. The corrected In-Shop packet establishes persistent queries and
foreign-proxy/image confusion agreement, not a compositional-vocabulary deficit.

There is older contrary evidence, but it remains below the post-audit-321
reliability boundary and therefore cannot decide Gate 1: a frozen Cars probe
scored global pooling 0.8306 versus token MaxSim 0.8159 (-1.47 points), and the
CUB region arm remained below Proxy Anchor after its scorer was repaired.
Candidate 230 already recorded that no matched positive measurement supported a
token aggregator. These observations weaken the story but are not promoted to
verified premises. The decisive Gate-1 fact is absence of an eligible positive
measurement, not reuse of quarantined negatives.

The proposal's three substitute premises are not measurements of its mechanism.
The seen-to-unseen gap does not identify template absorption; the DINO/ViT versus
ResNet comparison changes initialization objective, pretraining, architecture,
and head simultaneously; and generic spectrum/generalization correlations do
not identify a sparse word histogram as the intervention.

## Gate 2a: the frozen objective is internally broken

For nonnegative class-code columns `A_c`, CoVeR defines

`u_j = sum_c A[j,c] / ||A_c||_1`

and penalizes `sum_j max(0, m - u_j)^2`, with
`m = ceil(0.6 C s / k)`. But `sum_j u_j = C` identically. On CUB
(`C=100, s=24, k=512`) the frozen floor is `m=3`, which would require
total usage at least `512*3=1536` although only `100` exists. On In-Shop
the frozen values require `512*38=19456` although only `3997` exists.
The constraint is unsatisfiable. Under the fixed total, its convex optimum is
uniform mass `u_j=C/k`; it duplicates the separate uniformity term plus a
constant rather than enforcing a minimum number of classes per atom.

A count repair is not cosmetic. Top-s softplus entries can be arbitrarily small,
so an atom can satisfy a positive-count rule with epsilon mass while remaining a
private template. A minimum-mass-per-class repair changes the mathematical
object and must be proposed and audited anew.

The claimed proxy "factorization" is also a misdescription. The patch dictionary
`D` is `2048 x 512`, but proxy `pi_c = normalize(sqrt(A_c))` is directly a vector
in the 512 histogram coordinates; the proposal never constructs `D A_c` or
otherwise factors a proxy through `D`. `A_c` is the proxy coordinate vector.
The genuine intervention is a sparse nonnegative proxy in a learned histogram
basis, not a new shared factorization.

## Gate 2b: the intended mechanism still admits the named shortcut

Even granting a repaired reuse term, the training-class template solution stays
inside the hypothesis class. With 512 atoms and 100 CUB classes, the network can
allocate effectively private or low-overlap atom supports. The upstream
25-million-parameter backbone can learn class-identifying patch detectors; sparse
histogram coordinates do not bound that capacity.

The geometry actively favors this solution. Both images and proxies lie in the
nonnegative orthant, so cosine is in `[0,1]`, while the frozen Proxy Anchor
negative margin asks for similarities at or below `-0.1`. That target is
unreachable. The negative loss therefore keeps pushing different classes toward
disjoint supports, directly opposing cross-class reuse. Balance terms act on
proxy mass, not image-side atom/gate usage, and held-out-seen-class tuning rewards
the private solution. The frozen object reparameterizes class templates rather
than excluding them.

## Gate 2c: mechanism-level prior art

The closest primary work is substantially nearer than the proposal states:

- Nauta et al., *PIP-Net: Patch-Based Intuitive Prototypes for Interpretable
  Image Classification*, CVPR 2023, learns class-shareable patch prototypes,
  represents an image by prototype-presence scores, and connects those scores to
  classes with a sparse nonnegative linear layer. This is the load-bearing
  patch-vocabulary plus sparse nonnegative class-code architecture. Replacing
  classification with Proxy Anchor, max with gated sum, and prototype presence
  with a square-root count changes loss and pooling estimators, not the
  supervision mechanism:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.html>.
- Kundu et al., *Subsidiary Prototype-space Alignment*, NeurIPS 2022, explicitly
  inserts a learned word-prototype vocabulary, soft-quantizes spatial features,
  globally pools the resulting word histogram, encourages sparse histograms,
  and argues that shared visual primitives represent unknown/private classes.
  This is CoVeR's stated transfer mechanism in another open-set setting:
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/bf121b033db3bac31c3193e8a0dcbf66-Paper-Conference.pdf>.
- Mairal, Bach, and Ponce, *Task-Driven Dictionary Learning*, already jointly
  learns sparse dictionary features for supervised tasks:
  <https://arxiv.org/abs/1009.5358>.
- SALAD supplies modern assignment aggregation with a learned reject/dustbin
  gate in a deployed single global retrieval descriptor:
  <https://openaccess.thecvf.com/content/CVPR2024/papers/Izquierdo_Optimal_Transport_Aggregation_for_Visual_Place_Recognition_CVPR_2024_paper.pdf>.

Classical bag-of-visual-words supplies the histogram descriptor, and its
square-root plus L2 normalization is the Hellinger/Bhattacharyya map rather than
a new retrieval geometry. Project candidate 264 already rejected sparse
nonnegative shared-atom proxies as compositional proxy dictionary
regularization. CoVeR's only proposed escape was making the same atom index the
deployed descriptor; PIP-Net and SPA close that escape.

## Forecast and cost audit

The `s log2(k)` bit count does not bound the class-specific functions the
backbone can implement, and no cited generalization theorem maps it to Recall@1.
The 90% intervals and crossing probabilities therefore have no empirical or
theoretical estimator. Reaching 0.795 CUB from the named Proxy Anchor base would
require roughly a ten-point head-only lift despite discarding continuous
residual information. The DINO/ViT contrast cannot attribute such a lift.

The cost estimate is favorable by construction: making ResNet conv5 stride one
quadruples that stage's spatial work and plausibly pushes the backbone overhead
past the claimed 25--30% before entmax. Exact wall time is secondary because the
candidate already fails provenance, algebra, and prior art.

## Mechanism recorded

CoVeR attempted to prevent train-class templates by forcing both images and
class proxies through a sparse shared patch-word histogram. It dies because the
verified repository does not measure the claimed deficit, the reuse constraint
is unsatisfiable, the nonnegative Proxy Anchor geometry favors private supports,
the backbone retains class-template capacity, and PIP-Net plus SPA already
occupy the joint architecture and transfer rationale. No GPU follows.
