# PARC: Pareto-Adaptive Retrieval Cascade

**Status:** rejected at adversarial design review; do not implement

**Closure (2026-08-12):** PARC's adaptive-retrieval composition is occupied by
Matryoshka Adaptive Retrieval/AdANNS and ranking-distillation systems.  Its
prefix-only loss had no mechanism supporting the registered full-128-D quality
gain, the self-target could collapse information into the prefix, and the
Pareto claim mixed reference systems.  Review also found that the initial hard
set could contain surplus same-identity samples and push them down.  The design
is retained as negative research evidence.  Its useful controls (post-hoc SVD,
EMA averaging, matched ANN, content-bound caches, and profile-before-kernel)
must be carried into any successor.  None of the experiment stages below is
authorized.

**Target:** exceed the strongest reproducible compact-descriptor baseline while
reducing end-to-end retrieval cost.  The primary claim is a Pareto improvement,
not a loss-only improvement: higher retrieval quality, a cheaper encoder than a
UNICOM ViT-B/16 system, and a substantially cheaper million-item gallery scan.

## 1. Evidence that fixes the scope

The current controlled BN-Inception Proxy Anchor lane is approximately
91.4--91.8 In-Shop R@1.  MCPS-PG adds only about 0.2--0.3 point in the first two
paired seeds, while its proposed gradient-conflict mechanism fails its frozen
activation gate.  This is useful negative evidence: another small correction to
the same representation is not a credible route to the 94--97 point frontier.

The audited modern anchors are materially stronger:

- CRT reports 94.48 In-Shop R@1 from an ImageNet-1K MiT-B2 and a 128-D
  descriptor;
- supervised UNICOM reports 95.5 for ViT-B/16, 96.0 for ViT-L/14, and 96.7 for
  ViT-L/14 at 336 pixels, but releases no In-Shop-fine-tuned checkpoint; and
- the released UNICOM B/16 weights are a 74.6 zero-shot anchor, not a usable
  high-quality distillation teacher.

Systems arithmetic also closes the tempting wrong targets.  On In-Shop, the
3,997-class classifier and 12,612-item gallery scan are each below one tenth of
one percent of a ViT-B/16 forward.  Sampling 512 rather than 768 classifier
coordinates saves roughly 0.006% of the backbone-forward cost.  A native
PartialFC kernel cannot produce a meaningful end-to-end speedup here.  The
real cost centers are the encoder, descriptor width at deployment scale, and
the number of full-width gallery comparisons.

Therefore PARC starts from a modern efficient backbone and couples its compact
representation to an adaptive coarse-to-fine retrieval procedure.  Native
kernel work is conditional on profiling at realistic gallery scale; it is not
used to decorate negligible arithmetic.

## 2. Alternatives considered

### 2.1 UNICOM plus a fused/full-dimension PartialFC repair

This has the highest published quality ceiling, but the classifier repair is a
distributed-correctness control rather than a speed method.  ViT-B/16 is also
substantially more expensive than the current encoder, and the exact supervised
teacher is unavailable.  Keep UNICOM as a quality anchor and possible future
teacher, not the initial candidate.

### 2.2 Ordinary teacher-to-student distillation

This attacks encoder cost, but generic feature or pairwise EMA distillation is
already heavily occupied and the repository's own same-model EMA experiments
show that it can regress a well-regularized base.  It is also invalid until a
teacher demonstrably stronger than the student exists.  PARC instead uses an
offline, cross-model teacher only behind an explicit teacher-quality gate, and
distils the retrieval shortlist event rather than blindly matching features.

### 2.3 Quantization, PQ, or ANN over the current 512-D descriptor

This can accelerate a million-item service but cannot improve representation
quality and is irrelevant to the small In-Shop gallery.  It remains a systems
baseline.  PARC's representation is trained to make a low-dimensional exact
coarse scan safe before any approximate index is introduced.

### 2.4 PARC (selected)

Use a one-GPU modern efficient backbone with one ordered 128-D descriptor.  Its
32-D and 64-D prefixes are trained specifically to preserve the full-space
neighbour shortlist.  Retrieval first scans the 32-D prefix, then reranks only
the shortlist with all 128 dimensions.  A stronger frozen teacher, when one is
available, supplies offline neighbourhood targets without adding a teacher
forward to student training or inference.

The design combines established ingredients, but the candidate mechanism is
the coupling between prefix training and the deployed shortlist failure event.
The claim is not that Matryoshka embeddings, distillation, or two-stage search
are separately new.

## 3. Model and representation

The first matched lane uses the official CRT operating point where possible:

- MiT-B2 initialized from the same ImageNet-1K family;
- one 128-D linear retrieval head;
- unit normalization after selecting a descriptor width;
- one image and one global descriptor at inference; and
- the official dataset split, transforms, schedule, and retrieval metric for
  the reproduced baseline.

Let the unnormalized head output be `h(x) in R^128`.  For widths
`d in {32, 64, 128}`, define

```text
z_d(x) = normalize(h(x)[0:d]).
```

All widths are prefixes of the same tensor.  There are no separate projection
heads whose inference costs or parameters can be hidden.

The 128-D endpoint receives the matched supervised retrieval objective
`L_base`.  The initial implementation does not change the backbone, batch
sampler, augmentation, optimizer, or evaluation protocol while testing the
representation mechanism.

## 4. Shortlist-calibrated prefix objective

For an anchor `i`, construct a candidate set `C_i` from the current balanced
batch plus a detached FIFO memory of recent descriptors.  The memory contains
image IDs, labels, the unnormalized 128-D head output, and (when authorized)
the matching teacher embedding; an image may not compare with itself.  Prefixes
derived from memory are detached, so their stale values cannot receive a
gradient.  Candidate ordering is deterministic on CPU and stable by image ID
for ties.

Let `t_ij` be the detached full-space target similarity and `s^d_ij` the live
student-prefix similarity:

```text
t_ij   = <stopgrad(z_target(x_i)), stopgrad(z_target(x_j))>
s^d_ij = <z_d(x_i), z_d(x_j)>.
```

The normalized `z_target` may use the teacher's native dimension; only its
similarities are transferred.  It is selected in this strict order:

1. a frozen cross-model teacher if it passes the teacher gate in Section 7;
2. otherwise a stop-gradient copy of the live 128-D student endpoint.

For each anchor, first sort `C_i` by decreasing `t_ij`, with image ID as the
secondary key.  Let `P_i` be every same-identity candidate in that order.  Let
`T_i` be the first `K_train` unique candidates in the ordered concatenation of
`P_i` followed by the remaining teacher-ranked candidates.  Let `H_i` be the
next `K_hard` teacher-ranked candidates outside `T_i`.  An anchor is skipped if
either set is empty, and a batch is structurally invalid if every anchor is
skipped.  The prefix loss is a cutoff-aware pairwise ranking objective:

```text
L_short(d) = mean_i mean_(p,n in T_i x H_i)
             softplus((s^d_in - s^d_ip + m_short) / tau_short).
```

The loss directly penalizes candidates that would push a teacher-shortlisted
item below the coarse cutoff.  It does not require the prefix to reproduce the
entire teacher similarity matrix.

To prevent a prefix from preserving the shortlist while destroying ordering
inside it, add a bounded relational term over `T_i`:

```text
p_ij = softmax(t_ij / tau_teacher),  j in T_i
q^d_ij = softmax(s^d_ij / tau_prefix), j in T_i
L_rel(d) = mean_i KL(stopgrad(p_i) || q^d_i).
```

The complete objective is

```text
L_PARC = L_base(128)
       + lambda_32 * (L_short(32) + beta * L_rel(32))
       + lambda_64 * (L_short(64) + beta * L_rel(64)).
```

`K_train`, `K_hard`, the weights, and the temperatures are frozen by a
gradient-scale calibration on one
training batch before any retrieval result is inspected.  Calibration targets
each added prefix gradient norm at 10--25% of the base-head gradient norm; it
does not tune toward test R@1.

## 5. Optional offline cross-model teacher

Generic EMA self-distillation is not part of PARC.  A cross-model teacher is
enabled only when all of the following hold on an identity-disjoint validation
partition made solely from training identities:

1. its R@1 exceeds the reproduced 128-D student baseline by at least 1.0 point;
2. its mAP@R is no lower than the student baseline;
3. the gain is positive in every registered validation split or seed; and
4. its exact checkpoint, transform, input resolution, and descriptor
   normalization are recorded.

Teacher embeddings are computed offline for a fixed registered set of
augmentation seeds.  Student training regenerates the same view from the same
seed and loads the corresponding teacher embedding.  This avoids a teacher
forward in the student loop while keeping view identity exact.  Cache keys are
`(image_id, augmentation_seed, teacher_checkpoint_digest, transform_digest)`;
arm names are never cache keys.

If no teacher passes, PARC remains a self-targeted shortlist-calibrated nested
representation.  Failure to obtain a teacher is not permission to distil from
the 74.6 zero-shot UNICOM checkpoint.

## 6. Deployed retrieval

For every gallery item, store the full 128-D normalized descriptor.  The first
32 dimensions are exposed as an aligned view or separately packed array used
by the coarse scan.  If packed separately, those bytes count toward the storage
claim; they are not treated as free metadata.

For each query:

1. compute the model once and obtain `z_32` and `z_128`;
2. score all gallery `z_32` vectors and retain `K_deploy` candidates;
3. rerank those candidates using exact FP32 `z_128` cosine similarity; and
4. return results with stable gallery-index tie breaking.

`K_deploy` is selected on validation data before test evaluation as the
smallest value whose coarse candidate recall for the exact 128-D top-1 is at
least 99.9%.  Registered candidates are `{64, 128, 256, 512, 1024}`.  If 1024
misses the target, the 32-D cascade fails; the same rule is then evaluated for
64-D.  If both fail, PARC may retain its representation-quality result but has
no accelerated-retrieval claim.

The scientific quality metric is always recomputed with exact 128-D brute-force
retrieval as well as with the cascade.  ANN or quantization may not hide a
representation regression.

## 7. Evidence ladder and kill gates

### 7.1 Baseline and teacher gate

1. Finish the queued released-weight UNICOM export and reproduce its official
   zero-shot R@1 within the frozen tolerance.
2. Reproduce CRT/MiT-B2/128-D from official source.  If exact source is not
   runnable, document the incompatibility and use a named port; never compare a
   port as an exact reproduction.
3. Run an identity-disjoint frozen-feature probe for every potential teacher.
   A teacher that fails Section 5 is rejected before PARC training.

### 7.2 CPU and synthetic tests

- every width is the exact prefix of one 128-D tensor;
- normalization occurs after prefix selection;
- candidate/self exclusion and stable ordering are exact;
- moving a target item below the cutoff strictly increases `L_short`;
- permutations within the candidate set do not change the loss;
- removing the cutoff hard negatives makes a frozen adversarial fixture fail;
- cache keys bind content and transforms rather than arm names;
- cascade reranking matches exact 128-D ordering whenever the exact top set is
  contained in the shortlist; and
- report aggregation rejects missing, duplicated, or non-finite rows.

### 7.3 Eight-epoch, seed-0 smoke

Compare from identical initialization:

1. reproduced CRT baseline at 128-D;
2. ordinary multi-prefix classification/metric losses (Matryoshka control);
3. PARC self-targeted shortlist calibration; and
4. PARC with the offline teacher, only if authorized.

Continue only if a PARC arm satisfies all of:

- exact 128-D validation R@1 at least 0.3 point above the reproduced baseline;
- no mAP@R regression larger than 0.1 point;
- 32-D or 64-D candidate recall at `K_deploy <= 1024` at least 99.9%;
- student-training throughput at least 80% of the matched baseline;
- peak training memory at most 1.20 times baseline; and
- all losses and gradients finite.

### 7.4 Full matched experiment

Run three paired seeds on In-Shop and at least one second dataset reported by
CRT (SOP is preferred because of gallery scale).  PARC counts as a quality
improvement only if:

- the paired final R@1 delta is positive in at least two of three seeds;
- mean R@1 delta is at least 0.5 point and exceeds its paired standard error;
- mean mAP@R is not lower;
- it beats the ordinary Matryoshka and non-distilled controls; and
- it reaches or exceeds the strongest same-lane reproduced operating point.

The ambitious frontier target is In-Shop R@1 at least 95.5 with a 128-D
descriptor and a cheaper encoder than UNICOM ViT-B/16.  Falling short may still
establish a matched mechanism result, but it does not satisfy the SOTA/Pareto
claim.

## 8. Performance contract

Quality and performance are reported separately before any Pareto statement.

Encoder benchmarks use batch sizes 1, 32, and the largest non-OOM training
batch, with warmup and at least 1,000 timed iterations.  Report images/s,
median/p95 latency, peak device memory, parameter count, FLOPs from one named
counter, input resolution, precision, compiler mode, and hardware.

Retrieval benchmarks use exact synthetic galleries of 12,612, 100,000, and
1,000,000 descriptors plus the real datasets.  Compare:

- 128-D brute-force cosine;
- 32-D coarse plus exact 128-D reranking;
- FAISS exact inner product;
- one standard approximate-index control; and
- a custom native/Triton kernel only if profiling shows retrieval consumes at
  least 10% of end-to-end p95 latency at one million items.

A custom kernel must beat the strongest library baseline by at least 20% at the
same shortlist recall and numerical tolerance.  Otherwise it is deleted from
the claimed method.  Kernel work never substitutes for encoder benchmarking.

The full Pareto claim requires:

- quality: Section 7.4 passes;
- encoder: at least 1.5x lower batch-1 latency or 1.5x higher throughput than
  the matched UNICOM B/16 anchor;
- storage: at least 4x fewer descriptor bytes than 512-D FP32; and
- million-gallery retrieval: at least 2x lower p95 search latency than exact
  128-D brute force while retaining at least 99.9% of its top-1 results.

## 9. Fair baselines and ablations

The minimum comparison table contains:

1. official/reproduced CRT MiT-B2 128-D;
2. the same backbone and recipe with the simplest 128-D supervised head;
3. ordinary Matryoshka prefixes without shortlist calibration;
4. PARC without relational KL (`beta=0`);
5. PARC self-targeted;
6. PARC with teacher, if authorized;
7. teacher-only exact retrieval;
8. supervised UNICOM B/16 as a published and, if feasible, reproduced anchor;
9. exact 128-D search; and
10. FAISS exact and approximate search controls.

No score may be attributed to PARC when backbone, pretraining, image size,
epoch budget, sampler, or evaluator differs from its matched control.  Published
rows without released checkpoints or uncertainty remain operating points, not
paired evidence.

## 10. Failure modes and stopping rules

- **Teacher weaker than student:** reject teacher; do not lower the gate.
- **Prefix classification helps equally:** shortlist mechanism unsupported;
  report the ordinary Matryoshka result and close the novelty claim.
- **Quality rises only at 128-D while shortlist recall fails:** retain a
  quality-only result, but no retrieval-speed claim.
- **Cascade speeds only the 12K gallery by noise-level amounts:** report no
  deployment benefit; million-gallery evidence is required.
- **Compiler or native kernel creates the speedup:** label it systems-only and
  show the identical model with and without the optimization.
- **Student cannot clear CRT:** stop before teacher or kernel expansion.
- **Student improves CRT but remains below 95.5:** do not call it global SOTA;
  decide from second-dataset and cost evidence whether the Pareto result is
  still scientifically useful.
- **Any test-set-driven choice of width, shortlist, teacher, checkpoint, or
  cache:** invalidate that experiment and rerun only under a new prospective
  registration.

## 11. Implementation boundaries

The first implementation adds no CUDA/Triton code.  It introduces typed pure
functions for prefix construction, shortlist loss, stable candidate selection,
cascade reranking, and report aggregation, followed by integration behind an
explicit PARC configuration.  Existing recipes remain byte-for-byte behaviorally
unchanged when PARC is disabled.

Native acceleration is a later, separately attributable module guarded by the
Section 8 profile gate.  This ordering ensures that a learning result cannot be
confounded with a new numerical kernel and that a systems optimization targets
measured work rather than attractive but irrelevant arithmetic.
