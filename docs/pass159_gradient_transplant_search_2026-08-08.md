# Pass 159 — norm-ranked cotangent transplant (2026-08-08)

## Gate 1: provenance

Pass 158 found that identity-centered embedding norm predicts within-identity
correctness and margin (Pearson `+0.1417`, Spearman `+0.2097`), while using
norm directly in similarity or a margin is already occupied by quality-aware
metric learning.  This pass asked whether the signal could instead route
updates between same-identity examples: use a high-norm donor's first-order
Proxy-Anchor angular gradient, parallel-transport it to a low-norm receiver's
tangent space, and backpropagate the transported update.  Rival signatures and
augmentation response would be abstention checks only.

## Gate 2: prior art

The operator is not defensibly unoccupied.  If the transported update is
collinear with the receiver gradient it is pair/sample weighting (Multi-
Similarity loss with General Pair Weighting, Wang et al., CVPR 2019; DML-ALA,
Zheng et al., CVPR 2020).  If it is non-collinear it is direct gradient-field
manipulation (PCGrad, Yu et al., NeurIPS 2020; CAGrad, Liu et al., NeurIPS
2021).  A forward-loss rewrite using donor features is embedding-space
expansion/variation transfer (Ko et al., CVPR 2020; Meta Variance Transfer,
Park et al., AISTATS 2020).  Letting magnitude affect the objective returns to
MagFace (Meng et al., CVPR 2021) or AdaFace (Kim et al., CVPR 2022).  Using
response to choose augmentation strength is input-conditioned augmentation
(InstaAug, Miao et al., AISTATS 2023; AdaAug), while using it to admit or weight
pairs is mining/weighting.

## Verdict

**DEAD at Gate 2; no GPU run.**  Norm-ranked donor selection changes the
controller statistic, not the underlying operator.  The proposed mechanism
therefore collapses into occupied families under every formulation.

A CPU falsifier would have compared the first-order nearest-positive-minus-
nearest-foreign margin change against random-donor, norm-permuted, and
cosine-matched controls, but it cannot rescue the prior-art collision.

## Repaired Gate-2 re-audit (2026-08-08)

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.** The original judgement
treated every non-collinear gradient manipulation as mechanism-equivalent to
task-level PCGrad/CAGrad. Those methods combine distinct objective gradients; they do
not transport one example's cotangent into another example's sphere tangent space.
Scalar pair weighting and forward variation transfer likewise differ in both data
flow and decision point. The candidate is reopened only to a prospectively frozen
training-only causal diagnostic with random-donor, norm-permuted, cosine-matched,
receiver-own-gradient, and generic-surgery controls. No GPU is authorized by this
correction. Full re-audit:
`docs/repaired_gate2_reaudit_pass159_pass181_2026-08-08.md`.

## Gate-1 Stage-A preregistration after the repaired audit (2026-08-08)

Stage A is a necessary local-geometry screen and can kill the candidate, but cannot
by itself pass Gate 1 because an embedding-space direction may collapse or reverse
after the shared network Jacobian. It uses corrected final In-Shop PA seeds 0–3, the
matching final checkpoints/reports, and the retained 1,024-D pre-head train packs.
For each seed, reconstruct the 512-D pre-normalization **training** head output, its
norm, and its normalized descriptor. Training reconstruction must agree with the
digest-bound final train pack at `atol=rtol=2e-5`. Official query/gallery R@1 is
recomputed from the digest-bound final packs and must exactly equal both checkpoint
report and independent retrieval audit before those arrays are discarded; they serve
only artifact binding and never candidate selection or scoring. The legacy pre-head
query export has a documented batch-size-dependent tail and is descriptive only:
`docs/pass159_prehead_batch_binding_defect_2026-08-08.md`.

Because the pre-head packs contain labels but no example IDs or embedded digests, the
diagnostic must also load the independently exported, digest-bound final train/query/
gallery packs. It must verify exact label-row equality between each pre-head split and
its final pack, equality between reconstructed and exported normalized **training** embeddings,
the final packs' embedded checkpoint/report digests, and identical training example-ID
order across all four seeds. The immutable paths and SHA-256 values are frozen in
`docs/pass159_stage_a_manifest.json`; any mismatch fails closed before a statistic is
computed.

For every training identity with at least five rows, order images by the hexadecimal
SHA-256 digest of `pass159-stage-a-v1|<example_id>`. The first two rows are outcome
support and all remaining rows are controllers. Choose the controller with the
smallest pre-normalization norm as receiver and the one with the largest norm as
donor; equal-norm ties select the lowest hash at both ends. No outcome statistic
enters the partition or either selection.

Compute the exact **singleton-batch Proxy Anchor angular cotangent** at normalized
descriptor `z`, not the gradient with respect to the unnormalized head output. With
one normalized proxy `p_c` per class, `C` classes, `s_c=z·p_c`, and label `y`, freeze

`a = -alpha sigmoid(alpha(delta-s_y)) p_y
     + alpha/(C-1) sum_{c!=y} sigmoid(alpha(s_c+delta)) p_c`

and `g_z=(I-zz^T)a`. This is exact for the explicitly diagnostic singleton objective,
not a claim that it equals the sample's gradient inside the original size-120 batch.
The donor descent is `-g_z`. Project it once more into the donor tangent to suppress
floating-point drift, then parallel-transport it along the shortest sphere geodesic
into the receiver tangent space:

`PT(v) = v - ((v·z_r)/(1+z_d·z_r)) (z_d+z_r)`.

Pre-head norms, every cotangent norm, and every outcome-gradient norm must exceed
`1e-12`. A donor/receiver pair is excluded as geometrically unresolved when
`1+z_d·z_r <= 1e-6`; all exclusion counts and reasons are reported. No replacement
row may be selected after an exclusion.

The outcome is the training-only smooth retrieval margin

`m(z)=tau logsumexp_{s in S+}(z·s/tau)
      - tau logsumexp_{n in N32}(z·n/tau)`, with `tau=0.05`.

All supports are unit normalized. `S+` is the identity's two reserved supports.
`N32` is selected once, before any intervention, as the 32 foreign reserved-support
rows with largest receiver cosine; ties use the same fixed hash. The set is detached
and frozen. Thus the tangent outcome gradient is
`q_r=(I-z_r z_r^T)(sum_s softmax(s_r/tau)s - sum_n softmax(s_n/tau)n)`.
The primary score is cosine alignment `A=cos(q_r, PT(-g_d))`, so positive alignment
predicts a first-order margin increase.

All compared directions are normalized before scoring. Controls are:

1. the receiver's own PA descent;
2. the minimum-hash same-identity controller other than the receiver (candidate-blind
   fixed-hash donor; report how often it equals the candidate donor);
3. a norm-permuted donor: rotate actual controller norms by the integer encoded in the
   first eight hexadecimal digits of SHA-256(`pass159-norm-permute-v1|<label>`) modulo
   controller count, then select the largest assigned norm among non-receivers with
   hash tie-break;
4. among controllers excluding receiver and candidate donor, the donor minimizing
   squared standardized distance to the candidate donor in the two coordinates
   `(z_r·z_j, z_j·p_y)`. Standard deviations are frozen once per seed over every
   eligible non-receiver controller, floored at `1e-12`; ties use hash. Report both
   coordinate mismatches and the number of identities with only one alternative;
5. the candidate donor cotangent ambient-projected into the receiver tangent without
   geodesic transport; and
6. a proxy-only reference: compute the singleton angular cotangent at `p_y` and
   parallel-transport its descent from `p_y` to the receiver, using the same guards.

Candidate donor, receiver, and every controller stay outside outcome support. No
`q_r`, positive support, or foreign support quantity may enter donor matching or any
control choice.

For unit candidate direction `d` and unit receiver-own direction `u`, define the
noncollapse fraction exactly as `rho=||(I-uu^T)d||`. Its pooled median must be
`>=0.20`; `<0.10` fails and the interval is unresolved. Select one comparator `c*` as
the control with the largest pooled mean alignment across seed×identity rows (ties by
the numbered order above). Every reported seed delta is candidate mean minus that same
named `c*` mean. The bootstrap resamples eligible identity labels jointly across all
four seeds. Only complete identity labels with a valid row in all four seeds enter the
estimand; incomplete labels and their exclusion reasons are reported and never
replaced. In each of 1,000 fixed-seed (`159`) replicates it recomputes candidate mean
minus the maximum pooled control mean, giving a simultaneous strongest-control lower
bound rather than treating seed×identity rows as independent.

Stage A passes onward only if candidate alignment exceeds `c*` by `>=0.03`, the 95%
joint identity-bootstrap lower bound is positive, at least three of four seed deltas
against `c*` are `>=0.02`, and candidate mean strictly exceeds the no-transport
projection mean. The last condition is intentionally retained as a mechanism check
even though the strongest-control rule can make it redundant. It fails if the pooled
delta is nonpositive or at least three of four seed deltas are nonpositive; other
outcomes are unresolved.

A Stage-A pass authorizes only bounded Stage-B parameter-space VJPs. Stage B must show
that `J_r^T PT(-g_d)` retains a residual of at least `0.20` outside the receiver-own
update and improves alignment with a class-disjoint proxy-free retrieval gradient by
`>=0.02`, with positive clustered-bootstrap lower bound, all seeds positive, and a
small stateless step agreeing in sign and within 20% of the first-order forecast.
No training run is authorized before both stages clear.
