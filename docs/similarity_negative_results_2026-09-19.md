# Similarity-learning negative results ledger

This ledger prevents failed development hypotheses from being silently revived.
The canonical research checkout for these results is
`/home/rb/worktrees/sfora-emafactorial-release`; the older dirty
`/home/rb/worktrees/sfora-emafactorial` checkout is not an authority for this
line. Unless stated otherwise, these are claim-ineligible development results.

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
