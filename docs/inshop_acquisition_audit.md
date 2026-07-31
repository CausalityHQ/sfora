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

## Interpretation and limitation

The backbone strongly encodes acquisition-session/model/background evidence
shared by a garment's photo series. The filename token is a real training-data
factor, but the audit alone cannot say which visual component causes the gap.
Any method using it must be described as acquisition-aware rather than claiming
that the token is a verified camera or wearer identity.

The measurement supplies excellent provenance for cross-group supervision, but
the mechanism is adjacent to camera-aware person re-identification and therefore
must pass that literature—not merely generic DML—before GPU use.
