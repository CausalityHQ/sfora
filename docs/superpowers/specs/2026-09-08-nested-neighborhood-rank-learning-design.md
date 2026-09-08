# Nested Neighborhood Rank Learning Design

## Status and objective

This document defines the next generic Sfora similarity-learning method after
the train-only SOP representation-ceiling diagnosis. The objective is one
end-to-end encoder that preserves retrieval quality at a 128-byte int8 serving
representation and exposes a 32-coordinate prefix for a cheap first-stage
scan. Dataset parsing and evidence publication remain outside the reusable
method module.

The evidence closes several cheaper alternatives. A train-only affine map from
the frozen SOP source representation toward the stronger teacher gained only
0.000869 mAP@R, with a one-sided original-class bootstrap lower bound of
-0.000390. Teacher PCA-128 lost 0.021380 mAP@R. Resident int4 decoding was
3.71 times slower than resident int8, while representation loss dominated
quantization loss. Frozen linear compression and int4 are therefore not the
next research path.

An independent Astra review proposed a nonlinear bottleneck rescue test. An
independent Fable review instead identified end-to-end metric learning as the
only route that can exceed the frozen teacher ceiling. Earlier Sfora evidence
also found only a modest nonlinear cached-head gain, while a deployment-geometry
Smooth-AP finish improved identity-disjoint In-Shop development mAP@R by
0.013147 without reducing recall. The design therefore combines end-to-end
metric separation, teacher-neighborhood preservation, nested widths, and a
late rank-sensitive finish.

## Method

The reusable method is **Nested Neighborhood Rank Learning (NNRL)**. An image,
text, or other modality encoder produces a dense feature `h`. A trainable
projection produces 128 ordered coordinates:

`z = normalize(W_skip h + alpha * W_2 GELU(LayerNorm(W_1 h)))`.

The residual branch starts at zero contribution. The first 32 coordinates and
all 128 coordinates are normalized independently. The method does not depend
on class names, image paths, or a specific dataset.

The following equations are normative. For a normalized student row `z_i^w`
and normalized prefix of a shared trainable 128-coordinate proxy `p_c^w`, let
`s_ic^w = dot(z_i^w,p_c^w)`. With `alpha=32`, `m=0.1`, batch-present
proxy set `C+`, and all training proxies `C`, the authoritative loss is
`mean_{c in C+} log(1 + sum_{i:y_i=c} exp(-alpha*(s_ic^w-m))) +
mean_{c in C} log(1 + sum_{i:y_i!=c} exp(alpha*(s_ic^w+m)))`.
The raw proxy table is deterministically initialized from `Normal(0,0.01)`.
Both widths share that table and normalize its 32- and 128-coordinate slices
independently.

For neighborhood anchoring, cosine similarities are computed only against
off-class batch members; the diagonal, repeated immutable sample IDs, and every
same-label member are masked. This preserves inter-class teacher topology
rather than duplicating Proxy Anchor's within-class supervision. Teacher
probabilities and student key rows are stop-gradient; gradients flow through
each student query row only. For the valid-key mask `M_i`,
`t_ij=softmax_{j in M_i}(cos(T_i,T_j)/tau)` and
`q_ij=softmax_{j in M_i}(dot(z_i,stopgrad(z_j))/tau)`; the loss is
`-mean_i sum_{j in M_i} t_ij log(q_ij)`.
Before GPU training, a deterministic 1,000-batch replay over train-only teacher
embeddings chooses the smallest temperature in `(0.05, 0.10, 0.20)` whose
median target effective support `exp(entropy)` is at least 4 and whose 5th
percentile is at least 2. If none passes, the neighborhood objective is
classified redundant and no neighborhood GPU arm runs. Smooth-AP uses cosine
distance, excludes the query and all repeated copies of its sample ID from
competitors, uses every other same-label row as a positive, temperature 0.01,
and averages AP surrogates over queries with at least one positive. For each
positive `p`, its soft rank is
`1 + sum_{j!=p} sigmoid((d_ip-d_ij)/tau)` and its positive-only soft rank uses
the same sum restricted to other positives. The per-query AP surrogate is the
mean positive-only-rank/full-rank ratio; `L_rank=1-mean(AP)`. Its scalar equation
and vectorized implementation must agree under mutation tests.

Training has two phases:

1. **Anchored representation phase.** Optimize the encoder and projection with
   a supervised Proxy-Anchor loss at widths 32 and 128 plus asymmetric
   neighborhood anchoring at width 128. For each batch, a frozen teacher defines
   a stop-gradient probability distribution over every non-self batch member.
   The student query distribution is trained toward that teacher distribution
   with cross-entropy. The student is never used to update the teacher target.
2. **Rank finish.** Continue the same encoder for a fixed short phase using
   differentiable AP in the 128-coordinate deployment geometry. Proxy-Anchor
   remains active at a fixed lower weight so rank optimization cannot erase
   global class separation.

The phase-one loss is

`L = L_PA128 + 0.25 L_PA32 + L_neighborhood128` for the combined arm.
The Proxy-Anchor-only arm omits the last term; the neighborhood-only arm omits
both Proxy-Anchor terms. Teacher and student use the same preflight-selected
temperature. The two-epoch finish uses
`L = 0.25 L_PA128 + L_rank128` with Smooth-AP temperature 0.01.

The experiment compares three capacity-matched phase-one arms: Proxy-Anchor only, neighborhood
anchoring only, and their combination. All arms use the same encoder,
projection, batches, epochs, optimizer steps, augmentation, and evaluation.
If the combined arm passes development gates, both it and its matched
Proxy-Anchor control receive the identical rank finish. Paired differences are
reported before and after finishing; extra epochs cannot be credited to the new
objective.

## Teacher and leakage boundary

Teacher embeddings are computed before student training from a frozen,
authenticated model and keyed by immutable sample identity. The source archive
may contain frozen evaluation inference, but the training process receives a
separately authenticated official-training-only snapshot containing all
official-training identities and no official-test rows. The trainer indexes
only the optimization identities for its seed; validation teacher rows are
available only to the evaluator and premise check. Neither official-test arrays
nor test records are opened or materialized. The student's train augmentation may differ from
the teacher's canonical view; this is intentional invariance supervision.

All model selection uses class-disjoint development identities drawn only from
the official training partition. The official test partition is unavailable to
training, mining, threshold selection, epoch selection, or arm selection. The
chosen recipe receives one fixed final readout. Repeating a changed recipe on
that readout requires a new dataset or is labeled exploratory.

Class-name embeddings are not part of the core method. SOP and In-Shop labels
are arbitrary product identities, so semantic names are unavailable and would
make the method dataset-dependent. A later optional semantic-proxy adapter may
be tested on CUB or Cars only after the generic method is fixed.

## First scientific experiment

The first experiment uses SOP because it exposes the verified frozen-source
failure. Initialize a UNICOM ViT-B/16 student and use the frozen UNICOM
ViT-L/14 teacher already authenticated by the representation-ceiling study.
The projection width is 128 with a 32-coordinate nested prefix. Use three fixed
outer class-disjoint splits with seeds 17, 1729, and 65537.

For each seed, call the committed `deterministic_class_partition` with
`fit_fraction=0.8`: it hashes the concatenation of the seed's unsigned
little-endian eight bytes and each signed little-endian eight-byte class ID,
orders by `(digest,class_id)`, and uses
`floor(class_count*0.8+0.5)` optimization identities. Membership must equal the
sealed representation-ceiling split. Every validation image is a
leave-one-out query against every other validation image, with exact
self-exclusion. Training batches contain eight deterministic anchor identities
and the three nearest distinct optimization identities to each anchor under
frozen teacher class-centroid cosine distance. A domain-separated PCG64 stream
permutes optimization identities and supplies anchors per step, reshuffling
only after exhaustion. At rollover, consume entries while skipping identities
already selected in the current batch until eight unique anchors are collected.
Reserve all eight anchors first;
process them in their sampled order and choose each neighbor from globally
unused identities, breaking distance ties by class ID. This yields
exactly 32 unique identities and four independently augmented images per
identity. The matched control uses exactly
the same batches. Training uses batch size 128, 224-pixel student inputs,
AdamW, backbone learning rate
1e-5, projection/proxy learning rate 1e-3, weight decay 1e-4, BF16 autocast,
and ten phase-one epochs. An epoch contains exactly
`max(1, floor(optimization_image_count / 128))` updates. The encoder feature
`h` is exactly the normalized output of the official `model(images)` call, the
same tap used by the authenticated source archive. Initialization, bicubic
random-resized-crop/flip/color augmentation parameters, worker count, sampler,
and RNG streams are serialized in the authenticated recipe. No early stopping
or checkpoint selection is permitted. The two finish epochs start from each
arm's epoch-10 checkpoint with fresh AdamW state and the same frozen schedule.

Run the three phase-one arms on seed 17. Promote the combination only if it is
better than the matched Proxy-Anchor control by at least 0.005 mAP@R, its
one-sided original-class bootstrap lower bound is positive, and Recall@1 is no
more than 0.001 worse. If promoted, run the same combination and control on
seeds 1729 and 65537, then apply the fixed rank finish. The official SOP test is
read once only if the mean confirmation delta remains at least 0.005 mAP@R and
all three seed deltas are nonnegative and every seed's Recall@1 delta is at
least -0.001. The same gates are recomputed after the matched finish.

The seed-17 receipt also performs a premise check against the sealed full-width
teacher on this exact validation protocol and includes a matched 768-coordinate
Proxy-Anchor control. If Proxy-Anchor-128 exceeds the frozen teacher, the
neighborhood arm is interpreted only as a weaker-teacher topology regularizer;
failure cannot reject distillation generally. The 768-coordinate control
separates width loss from objective loss.

The seed-17 S2SD-style control uses the primary 32/128 head. Its fixed objective
is `L_PA128 + 0.25*L_PA32 + L_self`, where `L_self` is the same masked off-class
cross-entropy but its stop-gradient target is the normalized current
768-coordinate encoder output `h` and its query distribution is the
128-coordinate head output. Temperature is the same preflight-bound value.
Detach the target distribution and student keys, while allowing query-path
gradients through the 128-coordinate head into encoder output `h`. The PA-768 control instead uses one
768-coordinate head and `L_PA768` only. Both use the same backbone, batches,
updates, augmentation, optimizer groups, and seed-17 manifest as primary arms.

The paired uncertainty calculation stores each class's AP-difference sum and
query count, then draws 10,000 class clusters with replacement using PCG64 seed
17. Each replicate is the sum of sampled class sums divided by the sum of their
query counts, matching query-weighted standard mAP@R. The lower bound is the
finite-sample 5th percentile selected by the fixed nearest-rank rule. Recall
deltas are recomputed from the same immutable ranked IDs rather than copied
from training logs.

The final compact-quality target is at least 0.50 mAP@R and 0.79 Recall@1 on
the standard SOP protocol at int8-128, without a float-vs-int8 loss greater than
0.003 mAP@R. These are feasibility gates chosen to exceed the frozen teacher,
not a state-of-the-art claim. The report also includes
float32-128 and teacher/full-width controls.

## Serving contract

The deployed database stores one signed int8 vector plus one float16 scale per
item. Queries retain float32 during scoring. A two-stage option scans the first
32 bytes of that same int8 code and derives its prefix norm during the scan; it
stores no duplicate fp16 prefix. It retains exactly
`min(4096, gallery_size)` candidates and reranks them with the full int8-128
code. No int4 code path is promoted.

The serving gate requires:

- exactly 128 int8 coordinates plus explicit scale metadata;
- at least 3.5 times less persistent vector memory than float32-128;
- deterministic scalar and optimized top-k identity equality;
- at least 1.5 times higher end-to-end throughput than exhaustive float32-128
  at the registered gallery size, or a documented rejection of the two-stage
  path in favor of resident float32;
- no p95 latency regression at the registered concurrency;
- two-stage mAP@R loss no greater than 0.003 and Recall@1 loss no greater than
  0.001 versus exhaustive int8, reported separately from scalar/optimized
  implementation equality;
- finite scores and exact no-clobber artifact publication.

The performance receipt fixes a one-million-row gallery, top-100 output,
single-query concurrency, 1,000 warm-ups, and at least 10,000 raw timed queries
on named hardware. Scientific SOP fidelity is measured separately on the
complete official gallery at both 4,096 candidates and the fraction-matched 248
candidates, and compared against both exhaustive int8 and float32 rankings.
The one-million-row performance gallery is generated deterministically by
cycling the real SOP int8 codes in seed-17 hash order and applying a
counter-based seed-17 integer perturbation in `{-1,0,1}` with saturation to
each repeated copy; its exact generator and resulting digest are receipt-bound.

Quality and serving gates are independent. A high-quality but slower int8 arm
does not become a performance claim; a fast arm that misses quality does not
become a model claim.

## Generalization

The core interfaces accept embeddings, labels, immutable sample IDs, and
optional teacher embeddings. SOP, In-Shop, CUB, Cars, and text datasets provide
adapters. After SOP, the exact recipe is repeated without tuning on In-Shop and
CUB. The current implementation milestone is a claim-ineligible SOP feasibility
screen. A generic publication claim additionally requires frozen adapters and
matched Proxy-Anchor controls on In-Shop and CUB: at least +0.005 paired mAP@R
with a positive one-sided class-bootstrap lower bound and Recall@1 delta at
least -0.001 on SOP and one transfer dataset, and mAP@R delta at least -0.003
and Recall@1 delta at least -0.001 on the third. Protocols and gates are frozen
before either transfer run. Text remains a later applicability study, not part
of this claim.

In-Shop retains its official query/gallery structure with identity-disjoint
training development. CUB and Cars use a frozen 50/50 class-disjoint bridge and
all validation images as leave-one-out queries; they never inherit SOP's split
fractions mechanically.

NNRL composes established Proxy Anchor, relational/S2SD-style similarity
distillation, Matryoshka-style nested coordinates, and Smooth-AP. Its testable
contribution is the off-class asymmetric external-teacher topology objective
and matched compact-serving protocol, not a claim that these ingredients are
individually new. Seed 17 includes an S2SD-style self-distillation control.

## Failure interpretation

- If neighborhood-only and combined arms fail against Proxy-Anchor, reject this
  teacher-neighborhood objective; do not infer that all distillation fails.
- If the combination passes before rank finish but loses after it, reject the
  fixed finish weights or schedule while retaining the anchored representation.
- If float128 passes but int8 fails, the representation is viable and the
  quantizer needs separate work.
- If all end-to-end arms remain below the frozen teacher, reject only this
  registered width/backbone/resolution/objective recipe. Attribution to
  backbone capacity requires matched-width and matched-resolution ablations;
  the result does not prove that every nonlinear frozen head is closed.
- If quality passes but serving fails, retain the model evidence and redesign
  the search kernel independently.

Every result is canonical, immutable, and `claim_eligible=false` until an
independent reproduction. Structural, resource, and scientific failures are
recorded separately.
