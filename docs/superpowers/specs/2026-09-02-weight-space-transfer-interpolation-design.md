# Weight-space transfer interpolation diagnostic

Date: 2026-09-02  
Status: design; Cars evidence is claim-ineligible

## Decision

Before paying for another training objective, test whether SigLIP fine-tuning
gains unseen-class quality inefficiently because it moves the pretrained tower
past a more transferable point in weight space.

For every completed seed, evaluate a tower-only interpolation

```text
T(alpha) = (1 - alpha) * T_initial + alpha * T_trained
```

at the fixed grid `alpha = {0, 0.25, 0.5, 0.75, 1}`. The trained pooled
projection and trained proxy table are carried unchanged at every alpha. The
diagnostic uses only the already-burned Cars classes 82--97. It asks whether an
interior tower state consistently beats the trained endpoint while preserving
the learned projection.

An interior improvement funds a separately specified spectral shrinkage method.
A negative result declines funding in the current program but is only weak
evidence against spectral shrinkage in general. Scalar interpolation is a
causal control, not the novel method.

## Existing evidence and hypothesis

The frozen control baseline is 1242/1345, or 92.3420%, on the burned band.
Completed trained seeds score 1258/1345 (93.5316%) and 1248/1345 (92.7881%),
for a mean of 93.1599%. Training therefore improves the burned band; the issue
is not absolute forgetting. It is inefficient transfer: optimization gains are
much larger, raw 1152-dimensional and projected 512-dimensional errors are
effectively identical, and memorization-to-transfer ratios are about 0.13--0.17.

The control tests the stronger inverted-U hypothesis that a partially restored
tower can exceed both the pretrained and trained behaviors once paired with
the trained projection. A negative result does not prove that all layerwise or
spectral filters fail.

## Leakage and capability boundary

Classes 0--48 cannot be unseen validation folds because every checkpoint was
trained on all of them. Classes 82--97 are already burned and may be used only
for development. Classes 49--81 have also been evaluated by the existing
control; they are not a publication holdout, but this diagnostic receives no
additional capability for their images, labels, embeddings, or outcomes. The
official Cars test is unavailable. All Cars results remain claim-ineligible; a
publishable claim requires a fresh external benchmark.

The existing artifacts contain no burned-only pixel namespace. A one-time
authority preparer therefore loads the pinned local Cars dataset, verifies the
existing full manifest, and publishes only classes 82--97 into a flat
content-addressed image namespace with a sorted manifest of `(source ordinal,
example id, label, image SHA-256, byte length)`. The preparer has no model,
checkpoint, retrieval score, or selection capability. Although dataset loading
may encounter other bands in memory, it cannot serialize them. The scientific
diagnostic receives only the burned artifact and cannot load the dataset or use
network access.

## Authorities

Each seed binds:

- pinned model revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`;
- exact processor, torch/CUDA, autocast, and construction contract;
- epoch-60 checkpoint bytes, SHA-256, length, seed receipt, and run authority;
- seed-result bytes, SHA-256, and length;
- seed-result `model.initial_state_sha256`;
- checkpoint `initial_snapshot_sha256` as evaluation evidence only;
- pre-existing initial/final burned raw/projected counts, recalls, and margins;
- burned-only manifest and every image digest;
- this specification digest and the ordered alpha grid.

The seed-result loader validates the full registered receipt but normalizes
only initial-state and burned endpoint evidence into the child authority.
Clean-band values never appear in child arguments, output, logs, or exceptions.

Initial reconstruction follows the training order exactly: load the pinned
tower; set the seed on CPU and all CUDA generators; construct
`PooledProxyAnchorModel` (including `nn.Linear.reset_parameters` before the two
registered `kaiming_normal_` calls); move the complete model to the registered
evaluation device; then hash its fp32 state through CPU-contiguous bytes. The
digest must equal the authenticated
`initial_state_sha256`. `initial_snapshot_sha256` is not a model-state digest
and cannot substitute for it. RNG state is restored after reconstruction.

## State interpolation

Interpolation uses immutable CPU fp32 copies. Only floating `tower.*` tensors
are interpolated. `projection.weight` and `proxies` come exactly from the
trained checkpoint for every alpha. Non-floating tower tensors must be
byte-identical across endpoints. Names, shapes, dtypes, finite values, and the
complete trained state are validated before folding.

Alpha 0 is initial tower plus trained projection/proxies. Alpha 1 must equal the
complete trained state. Endpoint equality is scoped to this explicit surface;
it does not claim that alpha 0 equals the original random-projection model.
Repeated construction must produce the same folded-state digest. No weight is
cast to bf16; bf16 is used only by the registered forward autocast policy.

Each folded state is strictly loaded into a fresh model, evaluated with
gradient checkpointing disabled and no gradients, then released. No optimizer,
label, adaptive state, or outcome is available during state construction.

## Endpoint replay and evaluation

Before interpreting an interior row, the process evaluates the exact
reconstructed initial full model and exact trained full model. Their burned raw
and projected correct counts and denominators must exactly reproduce the seed
result. Recalls must equal count-derived values and margins must agree within
the existing `2e-5` replay tolerance. Any disagreement is an authority failure.

Every alpha is then evaluated on the complete burned band with the existing
leave-one-out Recall@1 authority and `(similarity, source ordinal)` tie rule.
Each seed emits five rows in ascending alpha order with:

- exact correct count, denominator, and recall ppm;
- nearest-positive, nearest-negative, and mean margin;
- folded-state digest and tower displacement norm;
- exact paired disagreements and McNemar p-value versus alpha 1;
- wall time and peak CUDA/RSS samples.

One common interior alpha passes only if:

1. mean Recall@1 exceeds alpha 1 by at least 0.30 percentage points;
2. no seed is worse than alpha 1 by more than one query;
3. mean nearest-class margin exceeds alpha 1;
4. endpoint, determinism, memory, and authority gates pass.

Selection order is `(mean recall descending, mean margin descending, alpha
descending)`. Per-seed alpha selection is forbidden. The 0.30-point threshold
is a funding gate, not a significance claim.

## Outcomes

The terminal class is exactly one of:

- `authority-failure`;
- `resource-failure`;
- `provisional-no-interior-benefit` for a negative two-seed result;
- `provisional-interior-benefit` for a positive two-seed result;
- `no-interior-benefit` for a negative three-seed result;
- `interior-benefit` for a positive three-seed result.

Neither provisional result funds or closes the route. Only
`interior-benefit` funds a spectral specification. It does not authorize a
publication claim or a fresh external evaluation by itself.

## Conditional spectral route

If the scalar control passes, a new design may replace global alpha with a
deterministic layerwise spectral filter selected without external holdout
outcomes, folded into the same state, and adding no inference branch. It must
beat the scalar control by at least 0.20 percentage points on the burned
three-seed mean before external evaluation is considered. The spectral method
is deliberately absent now to prevent a tuning garden.

## Resource and execution contract

Execution is serialized behind the existing DGX chain and never overlaps
another scientific GPU process. One seed and alpha is resident at a time. Stop
without restart at 96 GiB combined GPU memory, 110 GiB RSS, registered memory
pressure, or a progress timeout. A one-alpha preflight must project all three
seeds below 90 minutes. Scratch is content-addressed and removed after process
exit; only canonical claim-ineligible result bytes and SHA-256 are retained.
