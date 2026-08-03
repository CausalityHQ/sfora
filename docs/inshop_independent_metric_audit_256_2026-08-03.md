# In-Shop independent-metric audit (256)

Date: 2026-08-03

The first final-state exporter called `image_query_gallery_retrieval_score`, the same
scoring function used inside the trainer. Re-encoding from a final checkpoint was an
independent checkpoint-state check, but agreement on R@1 could not detect a shared
query/gallery or ranking bug. Calling it an independent metric recomputation was too
strong.

The exporter now computes headline R@1 through a separate implementation: normalize
query and gallery rows, perform chunked cosine matrix products, take exactly one gallery
argmax per query, and compare identity labels. It does not import the benchmark scorer.
It also requires the official 25,882/3,997 training, 14,218/3,985 query, and
12,612/3,985 gallery counts; equal query/gallery identity sets; and zero overlap in both
example IDs and resolved source paths. Unit controls cover cosine scale invariance,
chunk boundaries, label matching, and zero-norm rejection.

This hardening does not validate the loader's parsing independently of its source
metadata, but it prevents the trainer and verifier from agreeing merely because they
execute the same retrieval function. The queued final report must expose the new
`independent_recall_at_1` and all three split-integrity fields before its number is used.
