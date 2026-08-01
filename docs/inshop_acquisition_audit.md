# In-Shop acquisition-metadata geometry audit

**CPU-only measurement recorded 2026-07-31.** No external model or new GPU run
was used. The official image paths embedded in the epoch-10 export contain a
garment category, an acquisition-group token, and an explicit view suffix such
as `front`, `side`, `back`, `full`, `flat`, or `additional`.

## The dominant factor is acquisition group, not view label

All 153,115 within-identity pairs were measured at the exact Proxy Anchor
epoch-10 operating point.

| within-item relation | pairs | mean cosine |
|---|---:|---:|
| same acquisition group | 41,312 | **0.8199** |
| different acquisition group | 111,803 | **0.6396** |
| same named view | 24,504 | 0.6714 |
| different named view | 128,611 | 0.6914 |

The apparent view reversal is explained by grouping: almost every same-group
pair has a different view suffix, yet remains very close. At leave-one-out top 1,
**90.90%** of neighbours share the query's acquisition token while only 7.34%
share its named view.

Ordinary training-set R@1 is `0.9382`. Forcing the candidate neighbour to have a
different named view raises it to `0.9502`, largely because this retains the many
easy same-group/different-view matches. Restricting candidates to the same named
view gives only `0.2967`. The `flat` view is a small hard stratum (329 images,
R@1 `0.6413`); the other named views range from `0.9102` to `0.9721`.

A corrected cross-**group** diagnostic is much harder. Among 14,705 images from
the 1,274 training identities that have another acquisition group, removing every
same-group gallery candidate lowers R@1 to **`0.5542`**. Even restricting to a
same-view candidate in another group gives only `0.5773` on the 13,936 queries
for which such a positive exists. The ordinary 0.9382 training number is therefore
dominated by acquisition-local retrieval.

## The official test partition partly rewards the shortcut

Parsing `Eval/list_eval_partition.txt` shows that 95.60% of the 14,218 official
queries have at least one same-acquisition-token gallery image, whereas only
57.28% have any relevant image with a different token. Across all relevant
query-gallery pairs, 27.42% share the token and 16.09% share the named view.

Consequently, In-Shop headline R@1 can be satisfied by an easy same-session match
for nearly every query. Removing acquisition evidence may improve genuine
cross-session identity retrieval while hurting the registered R@1 endpoint. A
method should not exploit this partition property, but neither should a lower R@1
be presented as evidence that session invariance failed without a cross-session
evaluation.

The acquisition effect also does **not** explain the separate Proxy Anchor
gradient-conflict finding. Same-group pairs have a 21.02% conflict rate versus
16.80% across groups, despite much higher mean gradient agreement. Cross-group
pairs are more often moderately misaligned, but not oppositional. Treating the
session token as the cause of gradient surgery's motivating signal would therefore
be another unsupported causal jump.

The shortcut is largely learned rather than inherited. In the existing one-step
export, same-group and cross-group cosines are `0.6947` and `0.6696`, a gap of
only `0.0251`; by epoch 10 the gap is `0.1804`. Training amplifies it **7.18×**
by increasing same-group cosine 0.1252 while decreasing cross-group cosine 0.0300.
See `docs/acquisition_drift_audit.md` for the temporal audit and its caveats.

## Interpretation and limitation

The backbone strongly encodes acquisition-session/model/background evidence
shared by a garment's photo series. The filename token is a real training-data
factor, but the audit alone cannot say which visual component causes the gap.
Any method using it must be described as acquisition-aware rather than claiming
that the token is a verified camera or wearer identity.

The measurement supplies excellent provenance for cross-group supervision, but
the mechanism is adjacent to camera-aware person re-identification and therefore
must pass that literature—not merely generic DML—before GPU use.
