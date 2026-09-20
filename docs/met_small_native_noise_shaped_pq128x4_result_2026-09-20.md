# MET-small native fixed-codebook noise-shaped PQ128x4 result

## Result

The native paired-codebook experiment isolates a strong scorer interaction.
ScaNN-compatible noise-shaped assignment passes every powered-proxy gate under
dot product, but significantly degrades the same packed codes under Sfora's
current squared-L2 scoring geometry.

| scorer | mean pseudo mMP@5 delta | mean pseudo R@1 delta | query-bootstrap 95% | mechanism | seed floor |
|---|---:|---:|---:|---:|---:|
| dot product | **+0.009581** | **+0.002295** | **[+0.005240, +0.013984]** | pass | pass |
| squared L2 | **-0.005512** | **-0.015956** | **[-0.010124, -0.000832]** | fail | fail |

The dot-product gain is positive for every codebook seed (`+0.011142`,
`+0.009164`, `+0.008437` mMP@5) and is 4.1 times the isotropic seed spread
(`0.002361`). It reproduces about 78.7% of the external ScaNN screen's
`+0.012175` powered-proxy delta while using paired, shared codebooks and native
float32 exhaustive scoring.

Squared L2 reverses direction for every seed (`-0.007880`, `-0.005973`,
`-0.002683` mMP@5). This is consistent with its extra reconstruction-norm term:
for unit queries, decoded squared distance is
`1 + ||reconstruction||^2 - 2 * dot(query,reconstruction)`. Noise shaping
reduces parallel error but raises mean squared reconstruction error, so the
uncontrolled norm term can dominate the dot-product improvement.

The preregistered decisions are therefore:

- `public_codec_followup_supported=false`;
- `scorer_compatibility_diagnostic_supported=true`;
- `generic_supported=false`.

This result supports the assignment mechanism but does not promote the current
squared-L2 public codec. It licenses exactly one fixed normalization/scorer
attribution diagnostic, not a hyperparameter sweep.

## Secondary official-validation evidence

The 129-query official-validation split was preregistered as descriptive shift
evidence because it is underpowered to veto three codebook seeds. Across the
three seeds, dot product changes mMP@5 by mean `+0.010034` and Recall@1 by an
exact total of `-2` hits; squared L2 changes mMP@5 by mean `+0.002929` and
Recall@1 by `-1` total hit. Individual seed outcomes are noisy, including a
six-hit dot-product loss for seed 50 and two-hit gains for seeds 51 and 52.
These values do not override the powered 3,050-query decision.

## Codec and runtime evidence

- geometry: 768 dimensions, 128 blocks of width 6, 16 centers, low-nibble-first
  packing, exactly 64 bytes per gallery row;
- gallery: 35,257 rows; powered pseudo-queries: 3,050; official validation: 129;
- seeds: 50, 51, 52;
- public float32 and source-faithful float64 nearest-codeword initializers differ
  on exactly `0.0` assignments for every seed;
- noise shaping changes `33.48%` to `33.95%` of block assignments;
- per-seed conventional codebook fitting takes `4.14` to `4.16` seconds;
- source-faithful noise-shaped encoding takes `0.29` to `0.42` seconds;
- no custom CUDA, CuTile, or CUDA-Oxide kernel was used.

The scoring times are batch diagnostics over transiently decoded float32
vectors, not serving-latency measurements. The experiment determines which
similarity geometry preserves quality before any production optimization.

## Authority

- committed experiment/preregistration source:
  `bdb843b4576a53c71aa70f575cf4ecccdd3ae496`;
- driver SHA-256:
  `0c446cfd4d871d90ca113b2915e679b811cc4a13a4b0f1cc94c383b9164c9148`;
- feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- canonical result:
  `docs/evidence/rank_finished_l14_336_met_small_native_noise_shaped_pq128x4_v1.json`;
- result SHA-256:
  `436c4647b74a82872d7e7b8d092e73076ef9fc44606c38034725b8f7055799d3`;
- result bytes: `326,221`;
- remote process terminal status: `0`;
- external official test remained sealed.

## Next discriminator

Using the same fixed codebooks and packed assignments, compare exhaustive
rankings after L2-normalizing decoded gallery reconstructions. This removes only
the norm term and makes squared L2 ordering equivalent to cosine/dot ordering
for normalized queries. If normalization restores the dot-product gain, the
causal incompatibility is the decoded norm term and the next design question is
whether normalized reconstruction is acceptable in Sfora's public scorer. If
it does not, close this mechanism rather than tune thresholds or codebooks.
