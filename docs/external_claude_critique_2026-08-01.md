# External Claude critique of the stopping audit

Date: 2026-08-01. The repository was read-only during both successful reviews;
no training or GPU work was authorized.

## Procedure

Claude Code 2.1.220 was used as an independent hostile reviewer. A first Opus
call that included web tools stalled in the external service and was terminated
without returning a result. Two bounded Sonnet calls then read
`docs/search_protocol.md`, `docs/search_stopping_audit_2026-08-01.md`,
`docs/method_search_verdict.md`, and `docs/results.md`. The prompt explicitly
rejected novelty based only on a descriptor, mask, weight, normalization, or
composition of known operators.

## First critique

The reviewer found three apparent taxonomy gaps and then rejected all three:

1. **Per-class routing between HIST and Proxy Anchor.** Motivated by the fused
   objective underperforming both task-specific bases, but ultimately either an
   already-failed shared-backbone loss mixture or ordinary model/loss selection.
2. **Community propagation over RSPG/ARCG graphs.** Motivated by RSPG densities
   0.6449 on CUB and 0.0866 on In-Shop and ARCG density 0.3631. The propagation
   operator is occupied by ProxyGML/STML, while the existing graph arms already
   measured positive-anchor self-erasure.
3. **Balanced block-design batching.** Motivated by the SOP setting that silently
   excluded 36% of classes and the CUB IPC4 loss of 2.74 points. This is an
   engineering form of class-balanced sampling/cross-batch memory rather than a
   new supervision relation.

Its only proposed reopening measurement was cross-replica stability of graph
communities.

## Adversarial response and second critique

The proposed reopening measurement was challenged with repository evidence:
candidate 2 already rejected cross-trajectory consensus; candidate 50 rejected
consensus-stable relational transfer as multi-teacher agreement; candidate 44
and ProxyGML/STML occupy contextual graph propagation. The reviewer conceded
that cross-replica communities compose a consensus mask with graph propagation
and are not a new operator. It also noted that only seed-0 In-Shop
operating-point embeddings exist, so obtaining the missing measurement would
require GPU warm-ups without a surviving novelty case.

Forced searches outside graph propagation, pair mining/weighting, proxy
splitting, teacher transfer, synthetic support, part alignment, and loss routing
produced only three renamings—survival-analysis IPSR, renormalized TIRD, and an
ecological mark-recapture acquisition estimator. The reviewer killed each as a
loss reformulation, normalization, or pair weighting and returned no surviving
candidate.

## Consequence

The external critique did not overturn the bounded stopping audit. Its useful
result is negative: the apparent community-stability gap is already the
composition of two occupied mechanisms. The search may continue by obtaining a
new repository observable or widening beyond fixed supervision operators, but
it should not spend GPU on any of these three proposals.
