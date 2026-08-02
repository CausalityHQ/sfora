# Post-RS@k candidate-203 audit

Date: 2026-08-02. Conducted after the source-faithful Cars196 RS@k result was
complete. No candidate implementation or GPU work followed.

## Available new evidence

The immutable artifact contains the complete 2,380-step loss trajectory, 35
test evaluations, final and selected R@k curves, and aggregate train/test GSI
interference summaries. It contains no checkpoint or embeddings.

Claude was asked to find one new supervision relation from those measurements
and to reduce it against all 202 recorded failures. Its bounded verdict was
**NONE**:

- train/test interference-ratio distributions cross, but the artifact omits
  their numerator and denominator. Seen-class compression can change the ratio
  without increasing axis-parallel variance, and nearest-axis selection shares
  the same embedding estimator. Interference penalties are already GSI/deep
  Fisher objectives; tail aggregation is group DRO.
- MAP@R versus R@1 and the R@k miss curve lead to Smooth-AP/FastAP, ERR/ListNet,
  or the already rejected first-hit hazard.
- changing weights on `{1,2,4,8,16}` is rank-discount weighting, and late
  trajectory stabilization returns to the falsified averaging line.

## Candidate 203 steelman and Gate-1 failure

The only apparently distinct proposal was a *resolvable co-occurrence design*:
because every RS@k batch contains four examples from each of all 98 classes,
construct the sequence of within-class four-image groups across batches to
balance which images co-occur. This would alter the finite set of relations
presented across training rather than reweight a relation inside one batch.

The suggested provenance diagnostic decomposed terminal-regime loss variance
within each 14-step epoch and attributed the residual to the random within-class
partition. That attribution is not identified. Random resized crops,
horizontal flips, the partition itself, parameter updates, and nondeterministic
CUDA kernels all vary within the epoch (`deterministic` was false). The artifact
stores only their scalar sum. Linear detrending cannot separate them, and a
large residual would demonstrate stochastic step noise rather than
co-occurrence sensitivity.

Therefore candidate 203 fails **Gate 1**: no repository measurement shows that
co-occurrence design carries an outcome-relevant signal. Computing the proposed
ratio would give a precise number for the wrong estimand, so it was not run.
Identifying this route would require a checkpoint and controlled reevaluation
under fixed transforms and alternate partitions, neither of which exists in the
completed artifact. Even a positive diagnostic would still require a Gate-2
audit against batch-design, tuple mining, memory-bank and experimental-design
prior art before implementation.

## Boundary update

The faithful RS@k reproduction establishes a stronger occupied Cars reference
but exposes no new supervision primitive in its aggregate artifact. The search
must obtain a genuinely identified measurement, not reinterpret stochastic
trajectory variance as a label relation.
