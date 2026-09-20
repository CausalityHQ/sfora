# Similarity-learning negative results ledger

This ledger prevents failed development hypotheses from being silently revived.
The canonical research checkout for the current line is
`/home/rb/worktrees/sfora-positive-causality`; the older dirty
`/home/rb/worktrees/sfora-emafactorial` checkout is not an authority for this
line. Unless stated otherwise, these are claim-ineligible development results.

## Matched OML SOP PQ/OPQ improves depth but is not quality-Pareto

The missing standard 128-byte baselines were fitted on all 59,551 official OML
SOP train rows with Faiss 1.12.0 and evaluated on the same 60,502-row official
test split.  Float-query asymmetric PQ128x8 reached `0.651683` mAP@R /
`0.864682` Recall@1; OPQ128_384,PQ128x8 reached `0.651739 / 0.864269`.
The exact float384 source is `0.654393 / 0.865575`, and learned256-int4 is
`0.649107 / 0.864451`.  OPQ therefore gains `+0.002632` mAP@R over learned
int4 but loses `0.000182` Recall@1.  Its paired source mAP delta is `-0.002654`
with 95% interval `[-0.003094,-0.002212]`, narrowly outside the frozen
`-0.003` noninferiority margin.  No arm is a quality-Pareto winner and the
unchanged In-Shop replication is not run.  Result SHA-256:
`79e2164daee89755e12c580ff1add6c4394fea5547b99aced325ac14bc98518d`.

## Ranking-aware int4 scale fitting destroys the strong OML SOP code

The frozen learned-scale diagnostic held the OML ViT-S/16 head, 256 dimensions,
128-byte gallery wire, float query, calibration rows, update count, and
optimizer fixed.  The established `0.999`-quantile scales scored `0.651362`
mAP@R / `0.865624` Recall@1.  Equal-update reconstruction-scale fitting fell to
`0.592345 / 0.838666`; hard positive-negative margin-scale fitting fell further
to `0.584095 / 0.835857`.  Margin minus quantile was `-0.067267` mAP@R with
paired 95% interval `[-0.068594,-0.065945]` and `-0.029768` Recall@1.  Margin
also lost `0.008250` mAP@R to the reconstruction control.  This exact
ranking-aware shared-scale family is killed without optimizer, weight, pair,
or learning-rate tuning.  Result SHA-256:
`f39c03ca4fc2d337edd0ad8ecab749396dc17afca403302216f47ce9d1558ebc`.

## OML SOP asymmetric int4 is a significant near-miss, not a promotion

On the frozen official OML ViT-S/16 SOP representation, leaving the transient
256-dimensional query in float while retaining the unchanged 128-byte int4
gallery raised mAP@R from `0.649105` to `0.651362` and Recall@1 from
`0.864451` to `0.865624`.  The paired per-query gains were `+0.002257` mAP@R
with 95% interval `[+0.001834,+0.002697]` and `+0.001174` Recall@1 with
interval `[+0.000430,+0.001934]`.  The remaining gap to float was `0.003021`
mAP@R, however, narrowly above the frozen `0.003000` ceiling.  The complete
gate therefore failed; the ceiling is not moved and the planned In-Shop
replication is not run.  The diagnostic attributes about 43% of the symmetric
int4 mAP loss to query quantization and leaves database-code error as the next
boundary.  Result SHA-256:
`2d5589414d595d78a0a10d2756f4c691ba8e78ac251d6b32d6ea30ed25bab3df`.

## Fit-only mixed precision is not a generic 1,024-bit improvement

The frozen unequal-bit probe trained the same 256-dimensional compact head and
allocated exactly 64 coordinates at eight bits, 128 at four bits, and dropped
64.  Allocation minimized fit-only positive-negative margin damage and was
compared with variance-ranked and fixed-random allocations at the identical
bit budget.  On Cars, the margin code reached `0.848305` mAP@R / `0.974788`
Recall@1, losing `0.003502` mAP@R to uniform learned256-int4.  On CUB it reached
`0.722579 / 0.911096`, gains of `0.002093 / 0.002450` over the better uniform
parent.  The registered rule required both datasets, so the family is killed
without tuning bit counts, clipping quantile, or the allocation objective.
Variance ranking was within `0.000365` mAP@R on Cars and `0.000206` on CUB,
providing no evidence that the more expensive margin allocator is materially
distinct.  Raw DGX log SHA-256:
`88be4b58712da700fb4dd016c02dac26c2151ab7e5e3bfedc455b09d47ec378a`;
checked-in summary:
`docs/evidence/compact_metric/mixed-precision-margin-probe-v1.json`.

## Top-2 DBA passes mAP but is not quality-Pareto on In-Shop

The frozen PCA128-int8 top-2 gallery augmentation gate used the official
disjoint In-Shop query/gallery split. It raised mAP@R from `0.777802` to
`0.794346`: delta `+0.016544`, paired per-query 95% interval
`[+0.014416, +0.018654]`. It therefore passed the preregistered primary
`+0.005` mAP gate. Recall@1 fell from `0.945773` to `0.938951`, however, a
`-0.006822` regression. The result is retained as a primary-gate pass but not
as Pareto-superior quality evidence. It authorizes at most one exact untouched
replication and does not authorize integration, 1M benchmarking, or kernels.
Full receipt SHA-256:
`ab91bf35c5ad0eac694dbb95c141856baacb1b9f3abae99aa12e930747119d11`.

The single untouched EuroSAT replication confirmed the same mechanism and
closed it. mAP@R rose from `0.430537` to `0.448966`, delta `+0.018429` with
paired 95% interval `[+0.018063, +0.018808]`, while Recall@1 fell from
`0.936420` to `0.931235`, delta `-0.005185`. The preregistered joint
quality-Pareto decision failed. No neighbour-count, mixture, rank-protection,
or scoring variant is permitted on this holdout. Full holdout receipt SHA-256:
`27a0388962facc2ad9c6b33eee95d93629072618907cef22713661e45e7afbf7`.

## Flip-view gate is invalid, not a quality result

The frozen Food-101/Oxford-IIIT Pet identity-plus-horizontal-flip gate did not
reach either flip arm.  Its sole bounded DGX process ran for about 19 minutes
and then rejected the authenticated Food-101 identity replay: maximum absolute
feature drift was `0.0023138634860515594`, above the preregistered `0.0002`
limit.  No flip score, bootstrap interval, result artifact, or promotion
decision exists.  This is an authority failure, not evidence for or against
database-view augmentation.

- Source commit: `e87d25da3831495b902a90d8ac897fc6d051b1ac`
- Runner SHA-256:
  `6a625544e51ce565ba13e1b76407ec36a0ff0f71d73b30e7b88c02c4c739f535`
- Preregistration SHA-256:
  `446e5639ec60946309146c6521b2a32690d9a1f2f7ffbf9ff5c66c73f0710bd7`
- Frozen UNICOM checkpoint SHA-256:
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`

The likely boundary is cross-runtime or cross-device FP16 feature replay: the
source revision, checkpoint, dataset identity, and official transform matched,
but the cached identity vectors were not reproduced within the frozen numeric
tolerance.  The run was not retried or retuned.  Any future retry must first
reconstruct the original feature-producer runtime and prove identity replay;
the current zero-training representation ladder takes priority.

The subsequently authorized same-process repair reached both view arms and
closed the method. Food-101 R@1 moved from `0.939120` to `0.940240`
(`+0.001120`), while Oxford-IIIT Pet moved from `0.958708` to `0.961355`
(`+0.002647`). The row-weighted pooled delta was only `+0.001320`, below the
frozen `+0.003` gate, and the class-clustered bootstrap lower 95% bound was
`-0.000139`. Mean identity-to-flip cosine was also too high on both datasets:
`0.986665` and `0.988384` versus the frozen `<0.98` requirement. All three
decision conditions failed. Horizontal-flip gallery aggregation is now closed
without crop, scale, weighting, or additional-view tuning. Full receipt
SHA-256: `5663fc06a670a5e52d52d7bdb4dbb9e84ced9bc95b8c5a0d7deec42eff6c1535`.

## First ladder receipt used the wrong int8 scoring surface

The first four-arm/five-dataset ladder process completed, but its independent
replay found that the two int8 arms had been restored and normalized into
float tensors before scoring.  The library's deployed path instead multiplies
integer dot products by the stored per-row f16 inverse norms.  This made the
receipt unmatched to the serving representation, even though the point
differences were small on the first checked dataset.  Full invalid receipt
SHA-256: `36a844af50d04b7803fd8338947debe0851f6034e762b6824dd76d994002dc58`.

No arm decision from that receipt is promoted.  The correction is
measurement-only: keep the frozen datasets, projections, training recipe,
bootstrap seed/count, and decision thresholds unchanged, and route int8 arms
through the exact `PackedInt8Embeddings` score.  A regression now compares the
ladder score to the library scorer query by query.  This is the sole authorized
rerun of the frozen ladder, not result-dependent tuning.

## The frozen width/learning ladder has no universal winner

The corrected exact-packed receipt evaluated PCA128-int8, PCA256-int4,
learned128-int8, and learned256-int4 on Cars, CUB, SOP, In-Shop, and Food-101.
The registered survival rule required at least `+0.005` mAP@R over PCA128-int8
and a paired per-query 95% interval excluding zero on every dataset.

- PCA256-int4 failed CUB (`-0.000085`, 95% CI
  `[-0.001084,+0.000927]`) and In-Shop (`+0.000287`,
  `[-0.001078,+0.001723]`).
- Learned128-int8 failed CUB (`+0.001367`,
  `[-0.000378,+0.003062]`).
- Learned256-int4 failed CUB (`+0.001075`,
  `[-0.000912,+0.003070]`).
- Learned256-int4 minus learned128-int8 was positive on Cars, SOP, and
  Food-101, but included zero on CUB and In-Shop.

All three candidates are killed as generic defaults.  The prospective
label-free width selector is also killed: it selected 256 correctly on
Food-101, but selected 128 on unseen Flowers-102 while 256 had the higher outer
mAP@R (`0.983412` versus `0.981970`).  No top-2 augmentation, 1M benchmark, or
CUDA optimization is authorized without a surviving representation.

Corrected full receipt SHA-256:
`149313594c2e5da36c3a1a8d7137bb0afb163da9e0209038039bed98bf69a188`;
details: `docs/same_teacher_zero_training_ladder_result_2026-09-20.md`.

## A single fixed compact projection is not domain universal

The target-domain supervised affine projection remains useful, but one frozen
projection trained on other domains did not transfer reliably.

- **CUB-200-2011, SHA-ordered 100 fit / 100 evaluation classes.** Learned
  int8-128 mAP@R `0.718743`, PCA int8-128 `0.719066`, delta `-0.000323`;
  learned Recall@1 `0.905495`, PCA `0.912146`.  The registered `+0.003` gate
  failed.  Receipt SHA-256:
  `b5ea9f7ad083f103b6764d87911facf07c232457a9a855e8b3f113c1e3685a9e`.
- **Six-domain frozen transfer.** A projection trained on 200 source classes
  from CUB and GPR was positive against PCA on only one of six evaluation
  domains.  Macro mAP@R delta was `-0.005361`; worst delta was Food101
  `-0.011495`.  The other deltas were CUB `+0.002042`, DTD `-0.007375`,
  Flowers102 `-0.002900`, held-out GPR `-0.001791`, and Oxford-IIIT Pet
  `-0.010648`.  Receipt SHA-256:
  `1bd145864f7182d5acd0156f8e2ac0a70e23d22069366980c9867aa797b4d917`.

Interpretation: the library fitting procedure is generic, but the resulting
metric currently requires authorized target-domain labels.  It is not a
universal pretrained projection.

## Aircraft does not reproduce the SOP information-access mechanism

On the frozen FGVC-Aircraft class-disjoint split, the same PCA initialization,
schedule, negatives, loss, and seed were used for three supervised arms:

| Arm | Trainable parameters | int8-128 mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| direct `768 -> 128` | 98,432 | 0.480614 | 0.740541 |
| factorized `128 -> 384 -> 128` | 98,432 | 0.477275 | 0.756757 |
| restricted `128 -> 128` | 16,384 | 0.479438 | 0.756757 |

Direct minus factorized mAP@R was only `+0.003339`, with paired class-bootstrap
95% interval `[-0.002201, +0.009218]`; direct minus restricted was `+0.001176`,
interval `[-0.004718, +0.007302]`.  The fixed information-access gate therefore
failed.  The large Aircraft gain over the unsupervised PCA baseline
(`0.439829`) is supervised metric adaptation, but this dataset does not show
that useful information had to come from outside the frozen 128-D subspace.

Full receipt SHA-256:
`09a56a958c384422802eb3b49de6f594623aa67598905b5b26845e6cd2af4c96`;
throwaway driver SHA-256:
`d697f2b76c0c0c5b610ffd568f9208d071e42e15337647579b97431513bcb093`.

## Candidate-local reranking is not the generic quality path

- The fixed contextual shortlist scorer improved over normalized DBA on
  Aircraft by `+0.002284` mAP@R (class-bootstrap 95% interval
  `[+0.000216, +0.004496]`), but on CUB its delta was only `+0.000762` with
  interval `[-0.000217, +0.001781]`; SOP and In-Shop point estimates were
  negative.  It is an engineered DBA/local-averaging variant, not a generic
  scientific improvement.
- A half-normalized DBA interpolation lost to the better parent on both CUB
  (`-0.000046`) and Aircraft (`-0.000417`).
- Gallery-density correction reduced unprotected DBA mAP@R by about `0.025`
  on both CUB and Aircraft and reduced Recall@1 by `0.0504` and `0.0402`
  respectively.
- Reciprocal reranking reduced In-Shop Recall@1 by `0.002813` on both frozen
  embedding pairs (`docs/inshop_reciprocal_reranking_result_2026-08-11.md`).
- A class-name semantic residual did not beat PCA/shuffled semantic controls
  on the tested domains.  Receipt SHA-256:
  `178ee3fff54335bc7bf7594d30eb27e9134b04d9275430b5eabc5596a7e060ba`.
- The local discriminant (CLD) line failed its fresh Stanford Dogs gate:
  normalized DBA mAP@R `0.658502`, CLD+DBA `0.651808`.  Global WCCN+DBA
  reached `0.668009`, but this is labeled same-domain transfer rather than a
  generic scorer result (`docs/evidence/stanford_dogs_cld/`).

These failures close additional scorer tuning. Train-time compact
representation learning with fit-only model selection and a safe PCA fallback
has now passed an untouched Oxford-IIIT Pet class-disjoint confirmation; see
`docs/compact_metric_selector_result_2026-09-19.md`. CUDA-Oxide/CuTile remain
implementation backends after the representation is selected.

## Exact 128-byte signed-int4 QAT does not repair the remaining SOP gap

A frozen quantization-aware continuation started from the authenticated OML
ViT-S/16 learned256 checkpoint.  It used the same 2,000-update schedule,
frozen hard negatives, learning rate, and initial state as a matched float
continuation.  The treatment alone applied the exact signed-int4 `[-7,7]`
dequantized forward with a straight-through gradient; scales were fixed from
official training rows before either continuation.

On the official Stanford Online Products test split, the starting int4 model
scored `0.649107 / 0.864451` mAP@R / Recall@1, the float-continuation control
scored `0.649800 / 0.865178`, and QAT scored `0.649725 / 0.865343`.  QAT minus
the matched control was `-0.000075` mAP@R with paired 95% interval
`[-0.000419,+0.000275]`.  QAT improved the starting model by only `+0.000618`
mAP@R, below the registered `+0.002` effect, and remained `-0.004669` behind
the float384 source with interval `[-0.005393,-0.003946]`.

The exact family is killed without tuning or In-Shop replication.  Result
SHA-256: `e525e76b51e8d1e4609915dd3624ca24b9e2e6cd00a82dd14b4d6ed8de09cb06`;
details: `docs/evidence/compact_metric/oml-vits16-sop-int4-qat-v1.json`.

## ScaNN anisotropic hashing improves its control but is not the 128-byte winner

The published ScaNN 1.4.2 score-aware anisotropic hashing mode was evaluated as
a matched standard implementation: 128 three-dimensional LUT256 blocks,
exactly 128 code bytes/item, dot-product search, no tree, and no exact rerank.
Both indexes fit only the unlabeled official SOP gallery.  The fixed threshold
was `0.2`; no parameter search was performed.

Isotropic ScaNN reached `0.650028 / 0.863029` mAP@R / Recall@1.  Anisotropic
ScaNN reached `0.651315 / 0.864352`, a paired `+0.001287` mAP gain with 95%
interval `[+0.000774,+0.001798]` and `+0.001322` Recall@1.  The mechanism is
therefore real under the compatible dot-product scorer, but its effect is below
the frozen `+0.002` gate.  Against the strongest matched Faiss OPQ128x8 arm it
traded `-0.000424` mAP for `+0.000083` Recall@1; against float384 it remained
`-0.003071 / -0.001223`, missing both noninferiority margins.

There is no Pareto winner and no In-Shop replication.  Result SHA-256:
`6087d16c96cafc4d67fe9015d5a12dd0b8fea11f6b8bc6cc26c631ab7587b123`;
details:
`docs/evidence/compact_metric/oml-vits16-sop-scann-anisotropic-128byte-v1.json`.

## A zero-initialized nonlinear residual does not improve the compact Cars head

The preregistered scratch probe held the authenticated UniCOM-L/14 teacher,
Cars-196 class-disjoint split, PCA-128 initialization, class-balanced schedule,
frozen hard negatives, positive-coverage objective, optimizer, update count,
int8-128 storage, and exact scorer fixed.  It changed only a zero-initialized
`768 -> 32 -> 128` GELU residual branch and the existing geometry anchor.

The anchored affine control reached `0.825656 / 0.972574` mAP@R / Recall@1.
The anchored residual reached `0.823730 / 0.973189`: mAP@R delta `-0.001926`
with paired class-bootstrap 95% interval `[-0.003448, -0.000272]`, despite a
small `+0.000615` Recall@1 delta.  It therefore fails the fixed `+0.005` mAP@R,
strictly-positive interval, and joint non-regression decision.  The unanchored
residual (`0.764694 / 0.967901`) also lost to its unanchored affine control
(`0.777091 / 0.967778`).

This exact 32-hidden residual family is closed without CUB replication or
hyperparameter tuning.  It does not establish that all nonlinear heads fail;
it shows that adding a small zero-initialized residual to the saturated compact
objective does not recover the teacher-dominated errors.  Runtime was 49.18 s
wall, maximum host RSS 2,500,984 KiB, peak CUDA allocation 404,395,520 bytes,
and zero swaps.  Receipt SHA-256:
`fb0a1a759d6c91dcd38f6a6cd8dfd15363b8a2087bda712dfd9777debac964fd`;
details: `docs/evidence/compact_metric/cars-compact-residual-head-v1.json`.

## UniCOM's pre-projection global vector does not recover the compact Cars gap

The frozen information gate re-encoded all 16,185 Cars196 images once with the
authenticated UniCOM ViT-L/14@336 checkpoint.  The model does not use a CLS
token: it flattens 576 normalized patch tokens into a learned 1,024-D global
vector, then applies its final `1,024 -> 768` projection.  The experiment
compared the existing fixed power-whitening compact method on that 1,024-D
pre-projection vector against both the authenticated final 768-D output and a
deterministic 1,024-D nonlinear expansion of the final output that contains no
new image information.  Every arm produced the same signed-int8 128-D plus
f16-inverse-norm 130-byte wire representation.

The final-output incumbent reproduced at `0.860662 / 0.974542` mAP@R /
Recall@1.  The derived-width control reached `0.858203 / 0.975157`.  The
pre-projection candidate reached `0.859131 / 0.974665`; relative to the
strongest control its mAP@R delta was `-0.001531`, with paired
evaluation-class bootstrap 95% interval `[-0.003298, -0.000012]`, and its
Recall@1 delta was only `+0.000123`.  The interval excludes zero in the wrong
direction, so the fixed `+0.005` joint gate fails decisively.

This closes the exact pre-projection readout family on Cars without another
layer, hyperparameter change, or unseen-dataset run.  It does not reject all
intermediate token representations, but it shows that the information removed
by UniCOM's final 1,024-to-768 projection is not the missing quality source
under the strongest compact method.  Full-cache reproduction cosine was at
least `0.9999995`; the one-shot run took 6:04.49 wall, used 10,156,848 KiB peak
host RSS, and performed no swapping.  Feature-cache SHA-256:
`a5e152d06be53fd968a74d5628bea95e8f77b6ea36f7dd2f57bd963c769f6ab1`;
result SHA-256:
`ffe324797a3693aa62cf09544de4332409a1092fce19c42c96c8365d9eaef9a7`;
details: `docs/evidence/compact_metric/cars-unicom-preprojection-v1.json`.

## Complementary DINO/UniCOM fusion does not survive full SOP evaluation

A frozen two-encoder screen concatenated 64-D PCA/int8 codes from DINOv2 and
UniCOM into the same 128-D int8 payload.  On six 512-query development panels
the compact fusion improved macro mAP@R over the better compact constituent by
`+0.017784`; five of six panels were positive and the worst-domain delta was
`-0.005902`.  The corresponding full-float fusion was also positive on all six
panels.  This made the mechanism worth one full-dataset falsification, but the
panel result was explicitly claim-ineligible.

On the official Stanford Online Products test (`60,502` queries/rows), compact
UniCOM alone reached `0.452324 / 0.723398` mAP@R / Recall@1, while the 64+64
fusion reached only `0.387317 / 0.670193`.  Its mAP@R delta versus the stronger
compact constituent was `-0.065007`, and it missed the fixed `0.496` mAP@R
target by `-0.108683`.  Full-float fusion (`0.422436 / 0.702968`) also remained
well below full-float UniCOM (`0.476328 / 0.745050`).

The SOP reversal closes this exact complementary-encoder fusion family.  It
does not justify doubling encoder cost, and no alternative width split or
dataset-specific fusion weight will be tuned.  Development-panel receipt
SHA-256: `866033d051e5fe4483dd9330d1dff138b9610c47b62dcb44a98ea94b0807039e`;
full-SOP receipt SHA-256:
`8b0f5b4bf76702e73b39beddfc1450c8ee88e209c9da4887db0afb994cb67e85`.
Details:
`docs/evidence/compact_metric/complementary-visual-fusion-v1.json` and
`docs/evidence/compact_metric/sop-complementary-visual-fusion-v1.json`.

## Final-block Smooth-AP rank finishing is positive but below the Cars gate

The frozen Cars196 class-disjoint probe updated only UniCOM transformer block
23 (`12,593,152` of `908,200,448` parameters) for four epochs with the existing
Smooth-AP rank-finishing loss.  It then refit the identical power-whitening
128-D head and evaluated the same signed-int8 128-D plus f16 inverse-norm
representation (`130` bytes/item).  Evaluation labels and metrics were not
consulted during training.

The authenticated baseline reproduced at `0.860660 / 0.974542` mAP@R /
Recall@1.  Rank finishing reached `0.865211 / 0.974542`, a `+0.004551` mAP@R
gain with paired evaluation-class bootstrap 95% interval
`[+0.003044,+0.006204]` and exactly zero Recall@1 change.  The effect is real,
but it misses the preregistered `+0.005` minimum practical effect by
`0.000449`.  The joint gate therefore fails.

This exact final-block method is closed without changing the learning rate,
epoch count, seed, augmentation, or loss and without CUB replication.  Runtime
was 2,374.75 seconds, peak CUDA allocation was 16,076,318,720 bytes, and the
run had no pressure stop.  The rejected 50-MB checkpoint was deleted; only the
canonical result remains.  Result SHA-256:
`623523d2c365f08dedacce508e5a930f4ec4a7345538bb1ada12d5902d7585c3`;
details:
`docs/evidence/compact_metric/cars-unicom-lastblock-rank-finish-v1.json`.
