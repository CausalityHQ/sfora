# Seventh blind proposer pass: invalid lane framing and repeated near-misses

Date: 2026-08-05. Blind prompt consultation `9706f93942cb4c34` was frozen and
committed before completion. Fable stopped after eight minutes; the harness
continued the same prompt under Claude Opus and returned a concatenation of two
`NONE` analyses. Exact artifacts:

- `docs/fable_blind_prompt_pass7_2026-08-05.txt`
- `docs/fable_blind_output_pass7_2026-08-05.txt`

No numbered candidate, diagnostic, implementation, preregistration, or GPU
follows.

## Prompt defect: capacity lanes were not separated precisely enough

The prompt preferred a ResNet-50/512-D/224px lane but then listed 0.766 CUB,
0.949 Cars, and 0.939 In-Shop as “comparable ResNet-50/global” horizons without
stating that the cited AdvRF and VAPNet deployed descriptors are
**2048-dimensional GAP features**, not 512-D heads. The primary-source audit in
`docs/open_set_fg_retrieval_horizon_2026-08-01.md` already records this. The
first output consequently treated those numbers as a 512-D frontier; the
fallback then incorrectly “corrected” AdvRF to 128-D. Both readings are wrong.

The relevant lanes are:

- published ResNet-50/2048-D/global: AdvRF 0.766 CUB / 0.949 Cars / 0.842 SOP,
  VAPNet 0.939 In-Shop;
- directly matched cheap ResNet-50/512-D reference: PA+DADA 0.729 CUB / 0.921
  Cars / 0.810 SOP / 0.930 In-Shop; and
- this repository's corrected BN-Inception/512-D/short-recipe In-Shop paired
  reference: final 0.9137009 and 0.9167956, which is a controlled research lane,
  not the general published frontier.

The general goal may still confront 2048-D published systems, but a method is
not allowed to claim that a 512-D baseline must cross a 2048-D frontier as a
matched-capacity test. Repeated `NONE` under that arithmetic is not evidence of
an impossibility. Future blind prompts must state descriptor dimension,
training machinery, and comparison purpose for every horizon separately.

## Near-miss 1: DS-NCA repeats candidate 373

The first output proposed applying unrolled Sinkhorn balancing to a batch
similarity matrix and maximizing total same-class transport mass. The only new
term is the column-marginal constraint; test inference remains cosine.

This is not a new candidate. Candidate 373 already audited the same causal and
mathematical core:

- the corrected evidence packet contains no positive measurement that hubness
  causes material retrieval errors;
- density-aware supervised DML already shapes embedding concentration;
- NeighborRetr computes training-sample centrality from a queue and adds a
  Sinkhorn uniform-marginal training loss to balance retrieval probability,
  then deploys ordinary similarity; and
- fixed-marginal Sinkhorn DML and candidate 175 already occupy batchwise
  optimal-transport allocation.

Removing candidate 373's held-out-class wrapper does not reopen its exact
Sinkhorn anti-hub operator. The output itself forecast no defensible crossing
without a new hub-error measurement and therefore returned `NONE`.

## Near-miss 2: nuisance transplantation repeats candidate 369

The fallback proposed tangent residuals around each class mean, parallel
transport to a common pole, and grafting one class's residual onto another
class mean while requiring recipient-class classification. It correctly found
its immediate zero-residual collapse: if every residual tends to zero, every
synthetic point becomes the recipient mean and maximizes the stated objective.

The deeper premise and operator were already rejected in candidate 369. A
prospective three-seed disjoint-identity test found nuisance-transfer ratios
`0.9312, 0.9287, 0.9345`, all below the registered `> 1.15` requirement; the
leading within-class subspace was not identity-light on new classes. Cross-class
offset transplantation is occupied by Delta-Encoder, DVML, Feature Space
Transfer, Meta Variance Transfer, and Embedding Expansion. Adding a variance
floor would be a substantive new regularizer and would not repair the failed
exchangeability premise.

## Rejected structural claim

The fallback asserted that *any* label-derived supervision can constrain at
most `C-1` descriptor directions and gives exactly zero gradient to all other
directions. The rank of between-class scatter does not prove this. Pairwise and
tuple losses use sample-dependent difference directions, learned proxies move
through descriptor space, normalization changes gradient directions, and a
nonlinear backbone can be constrained without its parameter gradients lying in
one fixed `C-1` output subspace. At most, a fixed linear proxy head gives a
local statement about a particular embedding-gradient span. The universal
impossibility conclusion and the claimed exhaustive list of within-class
signals do not follow and are not added to the evidence ledger.

## Verdict

**No candidate.** Both detailed near-misses are exact mechanism repeats with
failed or absent Gate-1 provenance, and the pass's frontier arithmetic was
invalidated by a dimension-lane ambiguity. The useful result is a correction
to the generation protocol, not another negative method count.
