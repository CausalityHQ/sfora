# Top-2 DBA untouched EuroSAT holdout

The single preregistered holdout for the In-Shop top-2 DBA result used the
official 27,000-image EuroSAT RGB archive. Within every class, paths were
ordered by the frozen SHA-256 rule and assigned 40% to PCA fitting, 30% to
queries, and 30% to gallery. This produced 10,800 fit rows, 8,100 queries, and
8,100 disjoint gallery rows. Neither labels nor holdout results changed the
candidate formula.

## Verified result

| metric | PCA128-int8 | top-2 DBA | delta |
| --- | ---: | ---: | ---: |
| mAP@R | 0.430537 | 0.448966 | +0.018429 |
| Recall@1 | 0.936420 | 0.931235 | -0.005185 |

The paired per-query mAP@R 95% bootstrap interval was
`[+0.018063, +0.018808]` from 10,000 frozen draws. Thus the mAP effect replicated
cleanly and exceeded the `+0.005` gate. Recall@1 regressed, as it did on
In-Shop. The frozen joint decision therefore failed and closes this exact
top-2 DBA method without another dataset, neighbour count, mixture, protected
rank, or scoring formula.

This is useful negative evidence: offline top-2 DBA is a robust neighborhood
smoother for mAP@R, but it exchanges first-neighbour correctness for deeper
ranking quality. It is not quality-Pareto and does not proceed to library
integration, 1M serving measurements, matched-SOTA claims, or kernel work.

The one bounded DGX process completed in 1,287.12 seconds. Independent replay
verified per-query cardinalities and means, the exact paired mean, both frozen
decision predicates, zero self-neighbour selections, and terminal PID
clearance. External compute spend was zero.

## Authorities

- Source commit: `ad9644b708ffe2770ad903d6f24eaeef6325f790`
- EuroSAT archive MD5: `c8fa014336c82ac7804f0398fcb19387`
- Dataset manifest SHA-256:
  `6128eba85c886dbbc352252189461b9843ff64aa0e77b0f92cf13bd4cab414b2`
- UNICOM revision: `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
- UNICOM checkpoint SHA-256:
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`
- Runner SHA-256:
  `ecc9d64796af2e73bfd98af9df2370be9f6a5e87b48cb763685dc65e12e973f1`
- Preregistration SHA-256:
  `03813e30842afa74e416b9009e6941b2eee0235a9181a5ab1d5cdbde4cfe9658`
- Full receipt SHA-256:
  `27a0388962facc2ad9c6b33eee95d93629072618907cef22713661e45e7afbf7`
- Compact checked-in summary:
  `docs/evidence/top2_dba_eurosat_holdout_ad9644b7_summary.json`
