# Blind corrected In-Shop outcome matrix 295

Date: 2026-08-03. Written while the corrected run was at epoch 9/60 and raw
best R@1 0.8208, before any mature score, report, checkpoint, final embedding,
or independent retrieval audit existed.

## Locked interpretation

| output | outcome | permitted conclusion | method consequence |
| --- | --- | --- | --- |
| raw best-test R@1 | inside registered `[0.904,0.934]` | official-recipe seed-0 reference is plausible | establish a control only |
| raw best-test R@1 | outside interval | fidelity prediction failed | diagnose; no method screening |
| independent frozen-final R@1 | exact agreement with report's declared scorer | usable untouched final-state reference | future arms may preregister against it |
| independent frozen-final R@1 | disagreement | scorer/artifact path broken | reject reference and repair |
| cosine or tie-aware R@1 | differs from canonical | quantify official duplicate/tie sensitivity | reporting caveat only |
| query/gallery content duplicates | locked 19 groups, 16 same-ID and three cross-ID | official benchmark contains trivial and contradictory exact matches | no duplicate-specific learning arm |
| raw-to-final gap | any value | trajectory description | no selection correction, EMA, averaging, or stopping claim |
| final artifact/config/hash audit | fails | inadmissible result | repair evidence, no scientific interpretation |

The same-identity exact query/gallery overlaps can directly affect at most 16 of
14,218 queries (**0.1125 point**) and the cross-identity overlaps at most three
(**0.0211 point**). Their realized scorer effect may be smaller. Neither bound is
large enough to motivate a benchmark-scale method, and train contains no
cross-identity exact duplicates.

## Search boundary

This run was registered to repair the control, not to generate hypotheses.
Filename-series and view-role structure, fragmentation, positive gating,
hierarchy, camera/view invariance, duplicates, and checkpoint stability already
map to occupied or falsified mechanisms. A result inside the interval does not
reopen them. A surprising raw/final gap or scorer difference is not a new
supervision object.

Candidate generation may reopen only if the final artifacts expose a genuinely
new, independently validated observable not listed above, followed by a
prospectively registered replication before any method design. Otherwise this
run supplies a corrected reference and more reliable stopping evidence, not an
excuse for post-hoc GPU work.
