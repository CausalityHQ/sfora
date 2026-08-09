# Pass 200 — Receiver-Self Tangent Alignment (RSTA)

## Status

**LIVE-NARROW at Gate 2; Gate 1 unresolved.** This document freezes the
candidate, its necessary four-seed Stage-A diagnostic, and its eventual stochastic
training estimator before any RSTA statistic is computed. No benchmark training is
authorized until Stage A and a separately preregistered virtual-update Stage B pass.

## Gate 1 — measured provenance

The repaired Pass159 **singleton diagnostic on clean eval descriptors** found that
the receiver's own PA angular descent aligned `0.592177` with a held-out proxy-free
retrieval-margin direction,
whereas a transported high-norm same-class donor aligned only `0.559383` (difference
`-0.032794`, negative in all four corrected In-Shop seeds). The donor retained only
`0.112145` median residual outside the receiver-own direction. Pass181 independently
found that learned coordinate ownership is unstable (split-half rank stability
`0.0355`--`0.0864`) and nearly scalar (weight CV `0.0385`--`0.0407`). These
measurements reject that donor transfer and axiswise repair. They do not establish
the executed contextual B=180 PA cotangent. They instead motivate the missing
question tested here: does the shared network map an exact contextual full-batch
gradient into a worse receiver motion than its receiver-self component induces?

## Exact candidate

For one ordinary batch PA graph, let `z_i` be a normalized descriptor,
`J_i = partial z_i / partial theta` the Jacobian of the complete trainable encoder
(BN affine, backbone, and 1,024-to-512 head), and

`dbar_i = -partial L_PA(B) / partial z_i`

the **contextual** PA descent cotangent from that exact batch graph. Define the
Euclidean raw-gradient-flow receiver motion

`b_i = J_i sum_j J_j^T dbar_j`

and its receiver-self counterpart

`s_i = J_i J_i^T dbar_i`.

RSTA adds, for one deterministic receiver `r` per training batch,

`R = 1 - cos(b_r, stopgrad(s_r))
     + gamma [log((||b_r||+epsilon)/(||stopgrad(s_r)||+epsilon))]^2`

and trains `L_PA + lambda R`, with `lambda=0.10`, `gamma=0.10`, and
`epsilon=1e-8`. RSTA is disabled during the official 143-step head-only warmup and
starts when the full encoder becomes trainable. The receiver is the row with the
smallest SHA-256 digest of
`rsta-train-receiver-v1\0<seed>\0<global_step>\0<example_id>`, where seed and the
one-based executed step are unsigned canonical base-10 ASCII. The implementation
must add RSTA to `_requires_training_sample_indices`, bind every dataset row index to
its corrected training example ID before step 1, and fail closed on an out-of-range
or duplicate index. Normalized proxies are detached inside RSTA only; ordinary PA
continues to update them. The contextual
cotangents and `b` remain differentiable, while the complete `s` target is detached
in both terms. This is a second-order estimator, not a third-order one.

The method regularizes parameter regions in which the **future ordinary-PA raw
gradient field** preserves receiver-self motion. It does not claim that `b` is the
AdamW update or the current combined PA+RSTA velocity: Adam moments, clipping,
weight decay, proxy updates, and BN running-buffer evolution are outside this
decomposition. Inference remains one image, one 512-D descriptor, one view, and
cosine retrieval. The scalar penalty is invariant to a common orthogonal rotation
of descriptor, head, proxies, cotangents, and supports.

## Gate 2 — adversarial prior-art audit

The closest methods contain ingredients but not the same training object, data flow,
and decision point:

- [Gradient Agreement](https://arxiv.org/abs/1810.08178),
  [PCGrad](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf),
  and [CAGrad](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
  compare or manipulate task/component gradients in parameter space. They do not
  compare a total and self-induced velocity after both are mapped through a named
  receiver's output Jacobian.
- [OGD](https://proceedings.mlr.press/v108/farajtabar20a.html) projects new-task
  gradients away from historical protected outputs. Its object is continual-task
  retention, not simultaneous receiver-self preservation.
- [Fishr](https://proceedings.mlr.press/v162/rame22a.html) matches domain-level
  per-sample gradient variance, while ordinary Jacobian regularizers constrain
  norms, spectra, or smoothness. Neither conditions a full-batch functional update
  on the same example's self-induced output motion.
- [DML-ALA](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html)
  learns scalar sample weights from a validation objective. RSTA changes the
  empirical tangent field rather than sample eligibility or weight.
- [DoCL, AISTATS 2021](https://proceedings.mlr.press/v130/zhou21a.html) scores a
  receiver/sample using its raw residual/cotangent aligned with the total functional
  velocity and uses that score for curriculum sampling. It occupies the raw-cotangent
  target and eligibility mechanism, not RSTA's receiver-self target.
- [MGS, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/67b0579a7298d9cf39c59404d867bdd7-Abstract-Conference.html)
  explicitly derives self motion `K_ii d_i` and batch motion `sum_j K_ij d_j`, and
  regularizes global empirical-NTK trace/determinant. It does not align those two
  receiver-specific fields.
- NINT (arXiv:2511.15487) intends a score exactly `||b_i||^2` (up to sign and
  domain) and uses it for coordinate selection, without `s_i` or field alignment.
- NINT implementation disclosure: official commit `293bcd2` unpacks
  `torch.func.jvp` as `w_flat, _`, selecting the primal rather than the JVP; the
  paper's intended object is therefore treated here as conceptual prior art, not as
  evidence that the released implementation computes that object.
- Semantic Granularity Alignment (arXiv:2603.10785) analyzes self/off-diagonal NTK
  terms but executes hierarchical sampling and reweighting rather than a direct
  receiver-velocity intervention.
- Internal candidate 223 covers generic component-gradient projection, candidate
  267 uses gradient similarity to decide positive eligibility, Pass50 aligns
  class-disjoint last-layer parameter fields, and Passes186/197 use covariance
  preconditioning or stochastic backward routing. None has RSTA's receiver-specific
  `J_i`-mapped self target. Treating all non-collinear gradient methods as identical
  was the repaired Gate-2 error documented for Pass159.

Therefore the narrowest defensible proposition is the exact conjunction of a
contextual receiver velocity, a receiver-self target, and a differentiated
receiver-specific cosine-plus-norm penalty. It leaves sample eligibility unchanged.
Broader ingredients are occupied, and collision risk is high under any looser
combination standard. This exact conjunction remains **LIVE-NARROW**, not established
novel/SOTA. A singleton PA surrogate is forbidden: PA's proxy-level log-sum-exp and
present-proxy reductions make its cotangent differ from the executed contextual batch
cotangent.

## Gate 3 — Stage-A preregistration

### Immutable artifacts and data binding

Use only corrected final In-Shop PA seeds 0--3 from
`docs/pass159_stage_a_manifest.json`. Before any scientific statistic, verify every
frozen SHA-256, checkpoint/report/final-pack cross-digest, identical final-train
ID/label order across seeds, corrected source-image membership, and exact independent
query-to-gallery final R@1. Require In-Shop, BN-Inception, batch size `180`, dropped
last batch, trainable BN and BN affine, `proxy_anchor`, alpha `32`, delta `0.1`,
dimension `512`, and exactly one proxy for every train identity. Official query and
gallery arrays are binding-only and are discarded before candidate statistics.

Pass159's head reconstruction alone does not bind the backbone and transform source
executed by RSTA. Before statistics, load every checkpoint through the diagnostic's
current model factory and independently export **all train, query, and gallery rows**
in eval mode at batch size 128. Require exact labels/IDs and rowwise descriptor
agreement with every digest-bound final pack at `atol=rtol=2e-5`; recompute canonical
float64 query-to-gallery R@1 and require exact agreement with the report, retrieval
audit, and pack binding. Record source revision and source-file hashes. Then release
query/gallery arrays and instantiate the scientific scorer from a training-only
object with no query/gallery fields. A mismatch is INVALID, not a candidate result.

Let `H(domain,text)` be SHA-256 of `domain.encode("ascii") + b"\0" +
text.encode("utf-8")`. Integers inside text are unsigned canonical base-10 ASCII
without padding; digest-derived integers use the first eight bytes as unsigned
big-endian. Order by
`(digest, example_id)`. An identity is eligible with at least three corrected-train
images. Within each identity order rows by
`H("rsta-stage-a-v1|role|", example_id)`; ranks 0 and 1 are clean outcome supports
and rank 2 is the receiver. Order eligible identities by
`(H("rsta-stage-a-v1|identity|", canonical_int_label), label)` and take the first
64. Chunk them into eight groups of eight receivers.

Exclude every row of these 64 identities from distractors. Order all remaining rows
by `H("rsta-stage-a-v1|distractor|", example_id)` and allocate eight consecutive,
nonoverlapping blocks of 172. Each official-size batch is its eight receiver rows
plus 172 distractors, ordered by
`H("rsta-stage-a-v1|batch-order|<batch>|", example_id)`, where `<batch>` is the
zero-based canonical base-10 index. Batches and IDs are the
same in all seeds. Thus supports never enter a PA graph and same-identity peers do not
contaminate the receiver self/cross decomposition.

Apply the official training transform once per row and cache the tensor. Snapshot the
global Python `random`, legacy NumPy RandomState, and torch CPU RNG states. Derive
three independent unsigned big-endian seeds from
`H("rsta-stage-a-v1|augment-python|", example_id)`,
`H("rsta-stage-a-v1|augment-numpy|", example_id)`, and
`H("rsta-stage-a-v1|augment-torch|", example_id)`. Call `random.seed(seed_py)`,
`np.random.seed(seed_np % 2**32)`, and `torch.manual_seed(seed_torch)` immediately
before the official transform, then restore all three states immediately afterward.
Record transform/source hashes, tensor hashes, and the ordered batch ID matrix. These
are deterministic artificial diagnostic contexts of the official
size and transform distribution, not claims to reproduce historical shuffled
batches. The identical augmented receiver pixels are used across model seeds and
alternate contexts.

### Exact same-graph field

Clone each checkpoint and use `train()` mode. BN affine remains trainable. Suppress
only running-stat mutation on the diagnostic clone (`track_running_stats=False`) and
first prove that this matches an untouched train-mode forward before its buffer
update, including parameter gradients, and leaves the original buffer hashes
unchanged. Never use eval mode, microbatches, or a batch smaller than 180.

One full ordered-batch forward produces unit rows, then follows the production
double-normalization path exactly: `z=_normalize(model(images))` and the unmodified
`_proxy_anchor_loss(z, labels, proxy_embeddings=detached_proxies,
proxy_labels=proxy_labels, alpha=32, delta=.1, torch_module=torch)`
normalizes embeddings and proxies again internally. A similarity-only rewrite is
forbidden. Obtain contextual `dbar` from this exact loss with `create_graph=True`;
Stage A may detach measured values only after constructing the field.
The ordered parameter tuple contains every trainable encoder parameter, including BN
affine and the embedding head, and excludes proxies. Compute `g`, `b_i`, and `s_i`
with exact VJP/JVP composition on the same complete batch function. Parameter
directions may be normalized before a JVP for numerical stability only if the output
is rescaled and a dense-Jacobian fixture proves equality. Project `b`, `s`, and
`dbar` once into `T_zi S^511` only to remove numerical radial drift and report the
removed fraction. Stage A detaches measured values; this does not alter them.

Run Stage A in a fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before
CUDA initialization, `torch.use_deterministic_algorithms(True)` without warn-only,
cuDNN benchmark disabled, TF32 disabled for CUDA matmul and cuDNN, autocast disabled,
and FP32 model arithmetic except explicitly float64 reductions/bootstrap. Repeat the
first batch's `z,dbar,b,s` calculation and require bitwise equality before continuing.

Before the panel, a float64 dense-Jacobian fixture with batch 3, input dimension 2,
output dimension 3, and affine-plus-normalization must reproduce `b` and every `s_i`
at `atol=rtol=1e-8`; its central difference uses `epsilon=1e-5` and must match each
JVP at `atol=rtol=1e-6`. A two-sample train-BN fixture must match the bufferless
functional transform in outputs and trainable-parameter gradients at
`atol=rtol=1e-6`, without buffer mutation. On the first full batch of each seed, draw
`u` and `v` as C-order standard-normal arrays from separate fresh PCG64 streams
seeded by the first eight big-endian bytes of
`H("rsta-stage-a-v1|adjoint-u|", seed)` and
`H("rsta-stage-a-v1|adjoint-v|", seed)`, respectively. Require
`<Jv,u>=<v,J^Tu>` within relative error `5e-4`. No finite difference through the full ReLU network is an INVALID
switch. These are fixed integrity gates, not candidate controls.

### Proxy-free held-out direction

For each receiver use its two reserved, digest-bound clean eval-mode final-train
descriptors as positives. The foreign pool is rank-0 from every other eligible
identity, excluding any row in the current batch. Freeze the 32 largest receiver-view
cosines, tied by role digest and ID. With `tau=0.05`, define

`m_i = tau logmeanexp_p(z_i.p/tau) - tau logmeanexp_n32(z_i.n/tau)`

and tangent ascent `q_i=(I-z_i z_i^T) partial m_i/partial z_i`. No outcome quantity
enters roles, batches, receivers, cotangents, or controls.

### Statistics and controls

For every receiver record

- `A_self=cos(s_i,q_i)`, `A_batch=cos(b_i,q_i)`, and primary
  `Delta=A_self-A_batch`;
- `A_desc=cos(dbar_i,q_i)`,
  `rho=||(I-s_hat s_hat^T)b_hat||`,
  `log_ratio=log((||b||+1e-12)/(||s||+1e-12))`, and `cos(b,s)`;
- a deterministic PCG64 tangent-random target and a fixed cyclic derangement of `q`
  among the eight receivers by shifting identity-order position by +1 modulo 8, as
  negative controls. Seed a fresh random-target PCG64 stream from the first eight
  big-endian bytes of `H("rsta-stage-a-v1|random-target|", "<seed>\0<example_id>")`.
  Project each foreign `q` into the
  current receiver tangent and renormalize it; likewise project and renormalize the
  random vector before norm-matching it to `s`. A zero projected norm is INVALID;
- an analytic embedding-head-only control. If `x` is the captured pre-head feature,
  `h=Wx+b`, and `D_i=(I-z_i z_i^T)/||h_i||`, then
  `K^head_ij d_j=((x_i.x_j)+1)D_iD_jd_j`. Report `b_head`, `s_head`, and verify that
  `1-cos(s_head,dbar_i) <=1e-5`. This cannot kill or pass full RSTA because the
  backbone/BN kernel can change the result.

The already-frozen Stage-A receiver schema does not contain `cos(b_i,dbar_i)`, so it
is not added retrospectively. A matched raw-cotangent control is mandatory in Stage B
to compare RSTA's receiver-self target directly with DoCL's occupied target.

For an alternate-context check, take identity-order positions 0 and 1 **before final
batch-row hashing** from every primary group. Retain primary identity order and chunk
the resulting 16 labels as positions 0--7 and 8--15. Combine each with 172
new nonoverlapping distractors ordered by
`H("rsta-stage-a-v1|alternate-distractor|", ID)`, taking consecutive blocks. Exclude
all rows of the 16 selected identities, every rank-0/rank-1 support from every
eligible identity, and every primary distractor. Reuse exact receiver tensors and
deterministically augment new distractors. Recompute the same graph and statistics.

Generate a fixed dense orthogonal `Q` by float64 QR of a C-order PCG64(200) standard
normal 512x512 matrix. Multiply each Q column by `sign(R_kk)`, treating zero as +1,
then cast Q to the model dtype. On the first batch/receiver of each
seed, use the column-vector convention `W'=QW`, `b'=Qb`, and the stored-row
convention `proxy'=proxy Q^T`, `support'=support Q^T`. For each named vector `v` in
`z,dbar,b,s,q`, require
`||v_rot-Qv||_2/max(||v||_2,1e-12) <=5e-4`. Require absolute differences `<=2e-4`
for `A_self`, `A_batch`, `Delta`, `A_desc`, `rho`, `log_ratio`, and `cos(b,s)`;
otherwise the run is INVALID.

Fail the complete run as INVALID on any digest/config/ID mismatch, nonfinite tensor,
duplicate ID, missing gradient, norm `<=1e-12`, unit-row error `>2e-5`, radial
fraction `>1e-3`, or failed dense fixture, BN fixture, repeatability check, or
full-model adjoint identity. All 64 primary identities must be valid in all four
seeds and all 16 alternate identities must be valid in all four seeds; no row is
excluded, replaced, or silently dropped.

Aggregate equally by receiver within seed and equally by seed. Jointly bootstrap
complete identity labels across all four seeds with 10,000 NumPy PCG64(200)
resamples, retaining seed pairing. The primary 95% lower bound is the ordinary 2.5th
percentile of pooled `Delta`. Separately bootstrap `A_self-A_desc`.
The complete set is the intersection of valid selected identity labels across all
four seeds. Pooled means are the equal average of four within-seed means. Pooled
medians are the median of all complete seed-by-identity rows. Each replicate draws
`N_complete` indices with replacement once and applies the same index vector to all
four seeds, then recomputes within-seed means and their equal average. Hash the
float64 C-order replicate vector and record NumPy version.

### Frozen prediction and decision

The candidate predicts pooled `Delta >=0.03`: shared-network coupling should degrade
receiver-margin alignment by at least 0.03 cosine relative to receiver-self motion.
**PASS ONWARD** requires all of:

1. pooled `Delta >=0.03`, bootstrap lower bound `>0`, and at least three of four
   seed means `>=0.02`;
2. pooled `A_self-A_desc >0` with bootstrap lower bound `>0`; this is the load-bearing
   prior-art non-collapse gate distinguishing RSTA's receiver-self target from DoCL's
   occupied raw-cotangent target, rather than merely reusing that descriptor
   cotangent;
3. pooled median `rho >=0.20` and median `|log_ratio| >= log(1.10)`, identifying both
   directional and magnitude defects needed by the registered two-term penalty;
4. absolute deranged-q pooled Delta `<=0.01`; and
5. alternate-context pooled Delta `>0` with at least three of four alternate seed
   means positive.

**FAIL** takes precedence if pooled Delta `<=0`, at least three seed means are
nonpositive, median rho `<0.10`, alternate pooled Delta `<=0`, or at least three
alternate seed means are nonpositive. Every other outcome is **UNRESOLVED**. A
direction-only result that fails the magnitude condition does not silently delete
gamma; `gamma=0` would be a repaired candidate requiring a new preregistration.

Stage A authorizes only a separately preregistered virtual-update Stage B. Stage B
must compute the actual RSTA auxiliary parameter correction, norm-match it to
ordinary PA and exact-compute controls, and show improved class-disjoint proxy-free
margin alignment before any 8,580-step benchmark run.

## Theoretical cost and output contract

The main panel uses 32 full B=180 forward graphs across four seeds, 288 full-encoder
VJPs, and 288 JVPs. Alternate context adds 8 forwards, 72 VJPs, and 72 JVPs; rotation
adds 4 forwards, 8 VJPs, and 8 JVPs. Execute serially with one batch graph at peak.
No host wall-clock measurement is scientific evidence. Full-dataset/model work runs
on the DGX; local tests use tiny fixtures only.

Persist schema/spec/git/manifest hashes; environment/determinism settings; artifact
hashes/config/R1 binding; role, batch, transform, tensor, and source hashes; ordered
parameter names/count; every receiver/support/foreign/batch ID; every norm,
alignment, control, rho, and log ratio; rotation residuals; exclusions; seed/pooled
aggregates; bootstrap distribution hash/quantiles; every threshold Boolean; and the
first decisive verdict clause. Aggregates without receiver rows are insufficient.

The Euclidean tangent kernel is not invariant to arbitrary hidden-layer rescaling or
AdamW preconditioning. The registered rotation gate covers the deployed descriptor
gauge only; every RSTA result must state this scope limitation beside its headline.
