# Pass209 evidence-conditioned method search

## Verdict

**NONE before new evidence.** A fresh cross-field search produced five trainable-
representation candidates, but every one collapsed to an occupied supervision
object or an already-closed local family under Gate 2. No implementation or GPU
training is authorized from this search.

The development decision surface remains fixed:

- Cars train classes `0..48` are the only optimization data;
- classes `49..81` are clean validation;
- classes `82..97` are burned descriptive evidence only;
- classes `98..195` are the official test classes, but they are already burned by
  the archived SigLIP-base result and therefore cannot support a fresh method
  claim;
- deployment is one pooled descriptor, one view, with no reranking;
- the development gate is a paired gain of at least `0.5` point over the
  same-backbone pooled control on `49..81`; the `97.4` target applies only to a
  sealed official `0..97`-train / `98..195`-test evaluation and must not be
  transferred to the smaller development gallery.

## Candidates rejected before implementation

### Gallery-Extrapolant Tail Supervision (GETS)

GETS would fit the upper tail of impostor similarities and optimize that tail so
batch-sized training extrapolates to a full retrieval gallery. Its executable
object is a combination of extreme-value or distributionally robust tail-risk
estimation and a listwise negative-pair loss. Tail extrapolation is not literally
General Pair Weighting over observed similarities, but neither ingredient nor
their conjunction supplies a new observation or trainable representation. The
adjacent occupied families are General Pair Weighting and the repository's
existential Recall@k/rank-hazard candidates.

### Interventional Class-Coarsening (ICC)

ICC would randomly merge confusable training classes and require separation of
the original identities inside each merged intervention. In executable form it
is an artificial multi-label or coarsened-proxy construction. The intervention
changes target policy, but supplies no new observation beyond labels and
embeddings; the relevant data-valuation and artificial-target families are
already occupied.

### Superposition-Decodable Embeddings (SDE)

SDE would require sums of two different-class descriptors to remain two-hot
decodable, borrowing a channel-decoding constraint from coding theory. This is a
synthetic relation/virtual-class objective and collides with the previously
rejected composite-class and recombination families. Coding language does not
change the supervision object.

### Trainable held-out-fold retrieval objective (HFRO-T)

Unfreezing the backbone does not repair HFRO's first-order reduction: a loss on a
held-out class fold after a virtual update produces the familiar cross-fold
gradient-alignment term. Disjoint-label meta-learning and gradient-alignment
families already occupy this object, and no Cars measurement currently funds the
extra bilevel cost.

### Native-Geometry Anchored Fine-tuning (NGAF)

NGAF would fine-tune using the checkpoint's native sigmoid geometry while
anchoring the representation to its frozen initialization. The native-objective
half is ordinary language-image fine-tuning; the anchor half is frozen-self
distillation or L2/weight interpolation. Their conjunction is recipe hygiene,
not a distinct metric-learning mechanism.

## Measurements required before another method is fundable

### M1 — same-backbone pooled control

Run the ordinary pooled Proxy Anchor control on the pinned
`google/siglip-so400m-patch14-384` substrate with three frozen seeds. Optimize
only on classes `0..48`, evaluate final-epoch checkpoints only on classes
`49..81`, and never instantiate Cars test examples. A separate GB10
memory/throughput smoke must precede the three scientific seeds. This control is
a measurement, not a claimable method. It supplies the denominator for the
required paired `+0.5`-point development gain. Its 33-class validation Recall@1
cannot be compared to the `97.4` official-test target.

### M2 — F-1 error taxonomy

Run the already-implemented evidence-only substrate probe once to reproduce the
sealed SigLIP-so400m result of exactly `1,242/1,345` correct and emit the ordered
103-error manifest. The original receipt contains only the aggregate, so the
manifest does not exist until this authenticated reproducing run succeeds. Then
render and classify all 103 errors. The taxonomy may choose the next hypothesis
family but may not tune a loss, threshold, schedule, or checkpoint. The exact
renderer is `sfora.error_contact_sheet.render_error_contact_sheets`.

### M3 — Cars transfer gap

On the already-burned classes `82..97`, measure how much within-training-class
contraction from the `0..48` pooled control transfers across the class boundary.
The measurement must compare the same initial substrate and final checkpoint and
must be reported per seed. It may choose whether the next hypothesis targets
transfer, capacity, or neither, but may not select a hyperparameter. Keeping this
hypothesis-generating measurement off `49..81` preserves clean validation for
the later paired method comparison.

## Decision rule

No new trainable method is admitted until M1--M3 exist. The next candidate must
be derived from the burned-band F-1 category distribution and transfer gap,
survive a new Gate-2 literature and repository occupancy audit, and then beat
the paired pooled control by at least `0.5` point on clean validation without
using `49..81` for hypothesis generation. Only after the candidate and recipe
are frozen may a separate three-seed model train on all official train classes
`0..97` and evaluate `98..195` once against the `97.4` target. Because historical
Cars test outcomes have already influenced this project, that final number is
comparative, claim-ineligible evidence; a fresh claim requires a genuinely
untouched dataset or preregistered holdout.

## Occupancy evidence

- GETS is adjacent to [Multi-Similarity Loss with General Pair Weighting
  (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.html)
  and the existential Recall@k/rank-hazard closures recorded in this repository.
- ICC is covered by the artificial multi-label, data-valuation, and proxy-policy
  closures in Pass202 and the method-search ledger; a causal label does not add a
  new observation.
- SDE is adjacent to [Embedding Expansion (CVPR
  2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.html)
  and the ledger's Composite Class Expansion/RECOMB entries.
- HFRO-T reduces to the cross-fold gradient-alignment/meta-learning object
  already audited in entries 228 and 61.
- NGAF combines the contrastive fine-tuning object of [FLYP (CVPR
  2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Goyal_Finetune_Like_You_Pretrain_Improved_Finetuning_of_Zero-Shot_Vision_Models_CVPR_2023_paper.html)
  with the frozen-self anchoring/distillation family.
