# Similarity-learning negative results ledger

This ledger prevents failed development hypotheses from being silently revived.
The canonical research checkout for the current line is
`/home/rb/worktrees/sfora-positive-causality`; the older dirty
`/home/rb/worktrees/sfora-emafactorial` checkout is not an authority for this
line. Unless stated otherwise, these are claim-ineligible development results.

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
