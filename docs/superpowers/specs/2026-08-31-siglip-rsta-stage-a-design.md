# SigLIP Receiver-Self Tangent Alignment Stage-A design

## Status and purpose

This is a prospective, claim-ineligible causal falsifier. It measures whether the
exact Receiver-Self Tangent Alignment (RSTA) mechanism survives in the pinned
SigLIP-so400m Cars control before any RSTA loss, benchmark training, clean-band
comparison, or SOTA claim is authorized.

The falsifier may execute only after the three ordinary pooled Proxy Anchor control
seeds `17`, `29`, and `43` have completed epoch 60 and published an authenticated
aggregate receipt. It consumes their sealed final checkpoints and optimization
classes `0..48` only. Clean classes `49..81`, burned classes `82..97`, and official
test classes `98..195` are forbidden inputs.

Before the scientific child is created, an authority-only controller authenticates
the complete final seed receipts, aggregate receipt, checkpoints, source, config,
dataset, and environment using the existing control implementation. It emits a
canonical `rsta-control-binding-v1` projection containing only source/config/dataset/
environment identities, the three checkpoint identities, one
`optimization_manifest_sha256` field, and the selected control
`microbatch_size`. The microbatch size is structural execution authority: the
controller requires the same authenticated `ControlRunAuthority` value for all
three seeds and the value must divide the fixed 120-row logical batch. It is not
selected or overridden by the scientific child. The child passes the binding to
the contextual replay, which uses only this bound value and the frozen control
score-disagreement tolerance `2e-5`; neither is a scientific CLI parameter.
`environment_sha256` is SHA-256 of canonical JSON over the outcome-blind
`ControlRunAuthority` environment projection: torch, transformers, and torchvision
versions, CUDA runtime, device name, steps per epoch, evaluation batch size, and
query block. The full-manifest digest is excluded from that environment preimage.
The projection is
outcome-blind: it contains no accuracy, loss, threshold, pass/fail, clean, burned,
or test evidence. After authenticating the control's complete image-free ordered
manifest, the authority phase also derives and binds a separate canonical
`optimization_manifest_sha256` over only optimization-band `(example_id,label)`
rows. The core RSTA module defines and validates its own immutable
`RstaControlBinding`; it does not import a script-private control authority type.
The scientific child receives only this binding, the three authenticated checkpoint
byte streams, the optimization-only manifest, and optimization-band pixels.
Inside the child's read-only image namespace, each optimization pixel object is a
flat file named
`SHA256("rsta-siglip-a-v1|image-path|" || NUL || example_id) + ".image"`.
The child requires that exact filename set and derives it from the manifest; paths
are never fields in the optimization manifest, and extra pixel objects are invalid.

## Current evidence and novelty boundary

Historical BN-Inception RSTA evidence found a strong batch-coupling defect but did
not validate the self target:

- pooled `A_self-A_batch = 0.26607445`, bootstrap lower bound `0.23252317`;
- pooled `A_self-A_desc = -0.05681438`, negative in all four seeds;
- median directional residual `rho = 0.916114`;
- median absolute log norm ratio `3.254504`;
- deranged control `-0.00949555`; and
- alternate-context delta `0.313704`, positive in all four seeds.

The old alternate context replaced almost the entire graph. The present alternate
panel retains receiver pixels while changing their class context, so `0.313704` is
not a like-for-like expected value. The conditions below bound only near-duplicate
class contexts; the alternate gate remains a sign test and `0.313704` is not a
target.

That result was `UNRESOLVED`, not a pass. The negative `A_self-A_desc` means that
the angular self target collapsed past the raw contextual cotangent on the old
backbone. The present falsifier does not reinterpret that evidence favorably; it
asks whether the vector-valued SigLIP tower changes the result.

RSTA's defensible novelty statement is deliberately narrow:

> RSTA aligns, in one receiver's output tangent space, the full contextual
> batch-induced motion to the stopped receiver-self mapped motion in angle and
> log-magnitude, and differentiates that penalty through the empirical kernel.

The following are occupied and must not be claimed as novel: per-sample
interference reduction, raw-cotangent harm reduction, functional-motion
objectives generally, NTK regularization generally, and component-gradient
projection. In particular, Avranas (arXiv:2607.16261, ICML 2026) changes the
training operator to reduce per-sample harm and therefore occupies the raw
cotangent decision point. DoCL occupies the same scalar harm observable for
curriculum decisions. RSTA remains distinct only if its vector receiver-self target
is materially different from the raw cotangent.

For the bias-free normalized linear projection `h=Wx`, `z=h/||h||`, and
`D=(I-zz^T)/||h||`, the projection-only self motion is

`s_i^W = ||x_i||^2 D_i^2 dbar_i`,

which is exactly parallel to the tangential component of `dbar_i`. A projection-only
RSTA operator is therefore structurally degenerate and forbidden. The primary
operator is the complete trainable tower plus projection. The final transformer
block plus projection is a preregistered descriptive secondary measurement only.

## Exact field

For normalized descriptor `z_i`, complete selected-parameter Jacobian `J_i`, and
the exact contextual Proxy Anchor descent cotangent

`dbar_i = -partial L_PA(B)/partial z_i`,

define

`g = sum_j J_j^T dbar_j`,

`b_i = J_i g`,

and

`s_i = J_i J_i^T dbar_i`.

The primary selected-parameter tuple contains every trainable vision-tower and
bias-free projection parameter and excludes proxies. The secondary tuple contains
the final complete transformer block, its layer normalizations, and the projection.
Parameter order is lexicographic by fully qualified name and is bound in the
receipt.

The SigLIP replay module contains no active dropout or training BatchNorm. Its rows
are therefore independent conditional on parameters. Logical-batch Proxy Anchor is
not row-separable, but the descriptor Jacobian is. The implementation must exploit
this exact structure:

1. use the authenticated logical-batch replay to obtain the complete `120x49`
   score matrix and `dL/dscores`;
2. compute `dbar = -(dL/dscores) @ normalize(proxies)` and prove it equals direct
   descriptor autograd on a dense fixture;
3. reuse the replayed, unclipped tower/projection gradient with sign reversed as
   `g`; and
4. compute receiver-only matrix-free JVP/VJP actions serially.

No dense Jacobian is materialized. A single full receiver Jacobian would require
approximately 877 GB in fp32. `torch.func.jvp`/`vjp` over `functional_call` is the
primary implementation. A preregistered double-backward JVP fallback is allowed
only if an exact pinned-module preflight shows forward-mode coverage failure. Tiny
dense-Jacobian and adjoint fixtures run in fp32 on CPU and require relative equality
within `1e-5`; the executed bf16 CUDA path has a separate bitwise repeatability
gate. The selected backend and preflight evidence are receipt fields. The module is
in `train()` mode, matching the control. `torch.compile`, active gradient
checkpointing during receiver transforms, and custom attention kernels are
forbidden in this diagnostic. Before disabling the control loader's non-reentrant
gradient checkpointing, a registered preflight must show descriptor and every
selected-parameter replay-gradient relative L2 agreement within `1e-5`; bind module
mode, checkpointing states, `use_reentrant`, and the comparison evidence. If neither
the primary nor fallback JVP covers the resulting pinned module, emit `INVALID`
before any scientific row is opened.

## Immutable data roles and sampling

Let `H(domain,text)` be SHA-256 of ASCII `domain`, one zero byte, and UTF-8 `text`.
Digest-derived integers use the first eight bytes as unsigned big-endian values;
integers in text are unsigned canonical base-10 ASCII.

All 49 optimization classes must have at least 15 distinct examples. Within each
class, order IDs by `(H("rsta-siglip-a-v1|role|", example_id), example_id)`:

- ranks 0 and 1 are clean eval-mode supports and never enter a PA graph;
- ranks 2, 3, and 4 are receivers;
- rank 5 is the ordinary same-class batch peer; and
- ranks 6 onward are eligible only as deterministic refill/alternate distractors.

Order classes by `(H("rsta-siglip-a-v1|class|", label), label)`.

Primary batch 0 contains the first 30 classes, each with ranks 2--5. Primary batch
1 contains the remaining 19 classes with ranks 2--5 plus the first 11 classes as
refills with ranks 6--9. Each batch is exactly 30 classes by four rows and matches
the control's executed logical-batch structure. Refill rows never become receivers.
Rows are ordered by `(H("rsta-siglip-a-v1|batch-order|<batch>|", example_id),
example_id)`.

The alternate panel orders classes using domain
`rsta-siglip-a-v1|alternate-class|`, retains the same receiver pixels, and uses
fresh, nonoverlapping refill/distractor rows chosen by domain
`rsta-siglip-a-v1|alternate-distractor|`. For every class, alternate candidates
exclude ranks 0--5 and every row used as a primary refill. Select the first
candidate as the alternate same-class peer. For a class also used as an alternate
refill, select the next four candidates as its refill rows, ordered by
`(H("rsta-siglip-a-v1|alternate-distractor|", example_id), example_id)`. No row is
selected as a peer or refill in more than one graph; only the registered receiver
rows are deliberately shared between primary and alternate panels. Every receiver
has a distinct same-class peer in both contexts, and no support enters either
graph. The minimum 15 rows per class covers six fixed roles, the possible four
primary refills, one alternate peer, and the possible four alternate refills. The
receipt binds every ID, label, role, row order, and tensor digest.

For each receiver, the alternate batch's foreign-class set must differ from its
primary foreign-class set. Because both panels draw 30-class batches from the same
49 classes, any two batches share at least 11 classes; under independent hash
orderings the shared foreign-class count has mean approximately 17.4 and standard
deviation approximately 1.7. The registered cap is therefore at most 22 shared
foreign classes, roughly three standard deviations above that mean. It excludes
only a near-duplicate context and is not a nonoverlap claim. The generated
hash-frozen partition is `INVALID` if either condition fails; it is not repaired or
resampled after inspection.

Apply the frozen training transform once per graph row. Snapshot and restore Python,
NumPy, and torch CPU RNG states around each transform. Per-row seeds are derived
independently with domains `augment-python`, `augment-numpy`, and `augment-torch`.
The identical receiver tensors are used across model seeds and contexts.

There are 147 receiver observations per model seed and 441 paired observations
overall.

## Proxy-free outcome direction

For receiver `i`, use its two reserved rank-0/1 eval-mode descriptors as positives.
The foreign pool is rank 0 from every other optimization class. Freeze the 32
largest receiver-view cosines with `(cosine, role digest, example ID)` ties. With
`tau=0.05`, define

`m_i = tau logmeanexp_p(z_i.p/tau) - tau logmeanexp_n32(z_i.n/tau)`

and tangent ascent

`q_i = (I-z_i z_i^T) partial m_i/partial z_i`.

Outcome data never select roles, batches, contexts, or thresholds. This is a
training-field mechanism test on memorized optimization identities, not a
generalization result.

## Statistics and controls

For every receiver record:

- `A_self=cos(s_i,q_i)`;
- `A_batch=cos(b_i,q_i)`;
- primary `Delta=A_self-A_batch`;
- `A_desc=cos(dbar_i,q_i)`;
- `cos(b_i,s_i)` and `rho=sqrt(max(0,1-cos(b_i,s_i)^2))`;
- `log_ratio=log((||b_i||+1e-12)/(||s_i||+1e-12))`; and
- `cos(b_i-s_i,q_i)` as a descriptive cross-contribution statistic.

Project `b`, `s`, and `dbar` once into the descriptor tangent and record the removed
radial fraction. Controls are a deterministic PCG64 tangent-random target and a
cyclic `+1` derangement of `q` within each batch, each projected and renormalized.
For receiver ID `r`, the random seed is the first eight bytes of
`H("rsta-siglip-a-v1|random-target|", r)` interpreted as unsigned big-endian;
PCG64 draws one float64 standard-normal vector, which is cast to the descriptor
dtype before projection. The derangement uses registered batch-row receiver order
and wraps the last receiver to the first. A zero projected norm is invalid.
The `1e-3` radial gate applies to the analytically tangent raw `b` and `s` fields
before their single projection, and to the residual of unit outcome/control
directions. The raw `dbar` cotangent is not analytically tangent: project it once
and record its radial fraction without applying that gate.

Aggregate receiver means within class, class means within seed, then average the
three seed means equally. Jointly bootstrap all 49 class labels across seeds with
10,000 NumPy `PCG64(200)` resamples. One class-index vector is applied to all three
seeds. Record the float64 C-order replicate-vector SHA-256 and NumPy version.

## Prospective decision

The four-to-three seed adaptation does not loosen either irreversible direction.
The old pass condition allowed at most one dissenting seed out of four; two of
three would loosen that fraction from 25% to 33%, so the new pass condition requires
unanimity. The old failure condition required at least three of four nonpositive
seeds; the new failure condition likewise requires all three. Requiring unanimity
for both directions further reduces power relative to the historical four-seed
experiment and deliberately moves ambiguous outcomes to `UNRESOLVED`.

`PASS_ONWARD` requires all of:

1. pooled `Delta >= 0.03`, bootstrap 95% lower bound `>0`, and every seed mean
   `>=0.02`;
2. pooled `A_self-A_desc >0` with bootstrap lower bound `>0`;
3. pooled median `rho >=0.20` and median `|log_ratio| >=log(1.10)`;
4. absolute pooled deranged-control Delta `<=0.01`; and
5. alternate pooled Delta `>0` with every alternate seed mean positive.

`FAIL` takes precedence if pooled Delta `<=0`, all three seed means are
nonpositive, median `rho <0.10`, alternate pooled Delta `<=0`, or all three
alternate seed means are nonpositive. Every other valid result is `UNRESOLVED`.
Passing only the direction term does not delete the registered magnitude term.

At 49 independent class units the bootstrap may be underpowered for a `0.03`
effect. That possibility is frozen now: a positive point estimate with a
nonpositive lower bound remains `UNRESOLVED`; no threshold repair or extra class
source is permitted afterward.

## Invalidity and determinism

Any authority mismatch, forbidden class access, duplicate/missing ID, insufficient
class cardinality, nonfinite tensor, missing parameter gradient, norm `<=1e-12`,
unit-row error `>2e-5`, gated tangent radial fraction `>1e-3`, or disagreement with
a registered fixture is `INVALID`, not a candidate result.

Run in a fresh process with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA init,
deterministic algorithms enabled without warn-only, cuDNN benchmark disabled, and
TF32 disabled for matmul and cuDNN. Match the executed control policy exactly:
bf16 CUDA autocast inside the pinned vision tower; fp32 projection, descriptor
normalization, Proxy Anchor loss, field scoring, and controls; float64 aggregation
and bootstrap. Bind `autocast_device_type`, `autocast_dtype`, and
`autocast_enabled`. Repeat the first full receiver calculation per seed and require
bitwise equality. Tiny fp32-CPU dense-Jacobian, adjoint, linear-head degeneracy,
rotation, row-separability, checkpointing-equivalence, and JVP-backend fixtures
must pass before scientific rows.

## Cost and resource envelope

Primary plus alternate panels require approximately 13,800 single-row forward
equivalents: six logical-batch replay backpropagations per panel across the three
seeds, 441 receiver VJPs, and 882 receiver JVPs per panel. This is approximately
the registered structural work count, not a measured epoch conversion. A
pre-science throughput preflight must project completion below the fixed one-DGX-hour
cap; otherwise the attempt is `INVALID` before scientific rows.

Load each checkpoint through a registered model-state-only path: validate schema,
`claim_eligible is false`, config and run-authority digests, seed, and
`completed_epoch == 60`, then load `model_state` strictly without constructing or
restoring an optimizer. Snapshot and restore Python, NumPy, torch CPU, and torch
CUDA RNG around checkpoint load. The diagnostic therefore has no AdamW moments and
adds one fp32 parameter-sized tangent. The row-at-a-time JVP path is estimated below
8 GiB. The process retains the control
campaign's 96 GiB unified-memory, zero-pressure-growth, and zero-swap-growth stops.
No new scientific run begins if the exact pinned-module JVP or throughput preflight
fails.

## Output and authorization boundary

The canonical newline-terminated result binds every input/checkpoint/config/source
digest, environment, class/row/tensor role, parameter tuple, backend preflight,
fixture, receiver row, control, aggregate, bootstrap hash, threshold Boolean, cost,
peak resource value, and first decisive clause. It always has
`claim_eligible=false`.

Receiver evidence stores the primitive per-panel cosines, motion/cotangent/control
norms, log norm ratio, and radial fractions. Serialization re-derives every Delta,
self-minus-descriptor value, rho, absolute log ratio, random-control Delta,
deranged-control Delta, alternate-panel Delta, aggregate, gate Boolean, verdict,
and first decisive clause; derived receiver statistics are never trusted inputs.
An authority or pre-science failure emits a canonical `INVALID` result with one
registered clause and no scientific metrics or receiver rows. A failure after the
first scientific row emits only the controller's terminal failure receipt and no
candidate result.

`PASS_ONWARD` authorizes only a separately preregistered virtual-update Stage B with
matched raw-cotangent and recent per-sample-harm controls. `FAIL`, `UNRESOLVED`, or
`INVALID` authorizes no RSTA training. No result from this falsifier is a benchmark,
novelty, or SOTA claim.
