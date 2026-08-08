# Pass159 Stage-A result — norm-ranked cotangent transplant (2026-08-08)

## Verdict

**FAIL at Gate-1 Stage A; no Stage B, implementation, or GPU training is authorized.**

The frozen four-seed diagnostic retained all 1,984 eligible In-Shop training
identities in every seed (7,936 unique seed×identity rows). The candidate's pooled
alignment with the held-out smooth retrieval-margin gradient was `0.559383`. The
strongest frozen control was the receiver's own ordinary Proxy Anchor descent at
`0.592177`, so the primary candidate delta was **`-0.032794`**. The joint-identity
bootstrap 95% lower bound was `-0.033757`.

Every seed was negative against that same named control:

| seed | candidate mean | receiver-own mean | delta |
| ---: | ---: | ---: | ---: |
| 0 | 0.557518 | 0.590208 | **-0.032690** |
| 1 | 0.564806 | 0.599096 | **-0.034290** |
| 2 | 0.558275 | 0.590391 | **-0.032116** |
| 3 | 0.556932 | 0.589015 | **-0.032083** |

The candidate also lost to simple ambient projection of the same donor cotangent,
`0.559383` versus `0.561390` (**`-0.002007`**), so sphere parallel transport is not
the source of useful alignment. The fixed-hash, norm-permuted, and cosine-matched
donor controls all landed between `0.561978` and `0.562654`, slightly above the
high-norm candidate. Median residual outside the receiver-own direction was only
`0.112145`, inside the preregistered unresolved noncollapse interval and below the
`0.20` pass threshold.

Thus both prospective fail conditions hold: the pooled effect is nonpositive and all
four seed deltas are nonpositive. The mechanism-level result is stronger than “no
gain”: a high-norm same-identity example does **not** provide a better local angular
descent direction for a low-norm receiver. The receiver's own PA gradient is better,
ordinary donor choices are slightly better than norm ranking, and geodesic transport
slightly degrades the donor direction.

## Integrity

- Remote result: `/home/riomus/group-learning/reports/generated/pass159_stage_a_result.json`
- Result SHA-256: `fce9c2f6e690b1966670685cf6beff9705cd70c8d5043c40ea64db516300a371`
- Result size: 33,036,500 bytes
- Bootstrap: 1,000 joint identity resamples, PCG64 seed 159
- Official test use: immutable artifact binding only; no query/gallery value enters the
  candidate statistic
- Official final R@1 reproduced for all four digest-bound final checkpoints
- Train pre-head/head reconstruction maximum absolute difference: at most `1.35e-7`
- Eligible/excluded identities per seed: `1,984 / 0`

Two implementation/integrity defects stopped fail-closed attempts before a valid
pooled verdict: the legacy query pre-head export was batch-shape-dependent, and the
first geometry guard used float64-level unit tolerance on float32 descriptors. Both
were diagnosed from binding/exclusion metadata without reading a candidate effect,
fixed under regression tests, committed, and rerun with unchanged scientific
thresholds. Full audit:
`docs/pass159_prehead_batch_binding_defect_2026-08-08.md`.
