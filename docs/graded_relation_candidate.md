# Candidate 4: graded within-class relation supervision

Status: gate 1 passed; prior-art audit required before implementation or GPU use.

## Gate 1 — provenance: PASS

Two failures in this repository point to the same defect in ordinary class-label
supervision:

1. Sub-center Proxy Anchor reached about **0.675 R@1** on CUB, roughly **1.7
   pt below** the corrected Proxy Anchor reference. Treating within-class
   variation as several discrete modes fragmented the class and hurt unseen-class
   transfer.
2. Tversky similarity reached **0.6758 R@1** on CUB, roughly **1.6 pt below**
   Proxy Anchor, and was **−4.63 pt** on the low-noise In-Shop screen. Its
   defining intervention was to give distant true positives more overlap-based
   attraction. The large negative indicates that many same-label images are
   distant for useful reasons; forcing every class-positive relation toward the
   same geometry destroys information.

The common supervision problem is that a binary class label says only
“same class,” while the retrieval embedding must preserve graded visual
relations within that class. Discrete sub-centers over-separate those relations;
uniform positive attraction over-collapses them.

Candidate 4 would add **pair-specific ordinal supervision** inside each training
class. A frozen, non-label feature source would identify which same-class pairs
share more local attributes, yielding constraints such as “within class c,
image a is more similar to b than to d.” Proxy Anchor would retain its ordinary
identity supervision, while the ordinal constraints preserve partial similarity
instead of declaring new identities or merely changing the similarity score.
This changes what supervision exists.

The proposal is measurement-derived: it predicts that preserving graded
within-class structure can avoid both measured failure modes. It is not yet a
method claim. No effect size, recipe, or GPU work is allowed unless a primary
literature audit finds that the supervision mechanism itself is unoccupied.

