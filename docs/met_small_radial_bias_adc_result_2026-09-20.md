# MET-small radial-bias exact-ADC result

The frozen `alpha=0.5` radial-bias correction is **rejected**. It reduced both
primary metrics in every seed relative to ordinary exact squared-L2 ADC at
`alpha=1`.

## Authenticated run

- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- result SHA-256:
  `671551c84ac0edb62fc68e6e526e0cf2b1641ab148b8510accfc793017f05c9d`;
- geometry: 768 dimensions, PQ128x4, exactly 64 packed bytes per gallery row;
- fit population: 35,257 rows; powered primary: 3,050 pseudoqueries;
- shifted descriptive split: 129 queries;
- seeds: 50, 51, and 52; exact lookup ADC for both scorers;
- execution: NVIDIA GB10; final scientific process exited zero with no
  duplicate process.

## Powered primary result

| seed | alpha=1 mMP@5 / R@1 | alpha=0.5 mMP@5 / R@1 | mMP delta | R@1 delta |
|---:|---:|---:|---:|---:|
| 50 | 0.570426 / 0.502951 | 0.567525 / 0.497049 | -0.002902 | -0.005902 |
| 51 | 0.573421 / 0.502951 | 0.565831 / 0.490164 | -0.007590 | -0.012787 |
| 52 | 0.566011 / 0.504918 | 0.563574 / 0.497377 | -0.002437 | -0.007541 |

The three-seed mean deltas are **-0.004310 mMP@5** and **-0.008743
Recall@1**. The frozen 10,000-resample paired-query interval for the seed-mean
mMP@5 delta is **[-0.008989, +0.000614]**. The observed ordinary-ADC seed
spread is 0.007410 mMP@5. No promotion predicate passes.

## Shifted descriptive result

The 129-query split agrees in direction. Alpha=0.5 minus alpha=1 mMP@5 / R@1
deltas were:

- seed 50: -0.044057 / -0.054264;
- seed 51: -0.008786 / -0.023256;
- seed 52: -0.020930 / -0.007752.

## Decision

Close fixed `alpha=0.5` radial correction. Do not run the conditional OPQ64x8
follow-up and do not implement a specialized kernel. The negative result says
that a global coefficient change between decoded norm and query alignment does
not repair this codec. It does not rule out learned codebooks, a rank-aware
rotation, or a representation whose training objective directly preserves
local distance gaps.
