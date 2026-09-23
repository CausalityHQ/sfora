# Quality and performance continuation

## Outcome

Raise official image-retrieval quality while retaining Sfora's 130-byte
int8-128 gallery wire and measured one-million-row packed-search latency.
Report encoder latency separately from search latency. A benchmark score is
not a global state-of-the-art claim unless the protocol and published
comparator actually support it.

## Existing evidence and decision

The rank-finished UNICOM ViT-L/14@336 source plus Sfora's compact head scores
0.800020 mAP@R and 0.954283 Recall@1 on official In-Shop query/gallery. A
matched full-rank affine head adds only 0.001612 mAP@R and 0.000141 Recall@1;
head width alone has little room. On SOP, a reproduced OML ViT-S/16 source
reaches 0.865575 Recall@1, but previous 128-byte compact probes lose quality
against that source. An equal-weight OML/UNICOM fusion on the first 512
queries of fit-only SOP fold 0 reduced mAP@R from 0.821019 to 0.757674 and
Recall@1 from 0.927734 to 0.919922. This fusion is closed.

The next candidate moves training into the deployed code geometry: continue
the rank-finished In-Shop backbone with a 768-to-128 affine head, normalize,
fake-quantize to signed int8 with a straight-through gradient, renormalize, and
apply the existing SmoothAP rank objective. The released search representation
remains int8-128 plus one f16 inverse norm; search code and storage do not
change. The model path adds no new encoder layer beyond the existing compact
head, so any encoder latency change must be measured, not assumed.

## Data boundary and gates

The rank-finished model was trained using the registered class-disjoint
In-Shop train identity holdout. A new head baseline must be refitted only on
that model's optimization identities. The stored production head was fitted
on all official train identities and cannot be used as a holdout comparator.
The fit-only screen scores three arms on identical held-out identities using
the exact packed cosine: the refitted original head; a SmoothAP continuation
on float-normalized 128-D vectors; and the same continuation with signed-int8
fake quantization inside the loss. Both continuation arms start from the same
head, use the same batch schedule and optimizer updates, and end in the same
130-byte wire. This separates a rank-loss gain from a gain caused by exposing
rounding during training. The screen reports mAP@R, Recall@1, per-query
differences, artifact hashes, resource use, and singleton-class exclusions.
Official query/gallery remains unopened by the screen.

The code-aware arm advances to a final-block joint probe only if it gains at
least 0.003 mAP@R over the original head, has no net Recall@1 query loss, and
gains at least 0.001 mAP@R over the matched float SmoothAP continuation. If
only the float continuation gains at least 0.003 mAP@R with no net Recall@1
loss, the next probe uses float rank training and scores the packed code at
deployment; the code-specific claim closes. If neither continuation clears
its gate, both head-only directions close, while any new backbone probe needs
its own rationale. Full-backbone confirmation and one official read require a
frozen plan and same-wire latency comparison. These are exploratory gates on
a previously used training holdout, not fresh publication evidence.

## Risks

Straight-through rounding may add no benefit; the causal contrast is the
deployed-code rank loss against the existing objective. The held-out query
count is much smaller than the official query/gallery count, so differences
must be paired and uncertainty reported. The official In-Shop test has been
observed previously; even a future gain there is product evidence, not an
untouched scientific confirmation.
