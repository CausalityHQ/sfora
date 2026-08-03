# Corrected-pixel RSPG post-mortem

Status: **DEAD at Gate 4; stopped prospectively at the epoch-40 refresh; seed 1
and all controls cancelled.**

## What ran

The corrected standard-256px In-Shop seed-0 arm used the locked RSPG recipe on
`/home/riomus/datasets/inshop_official_standard`.  Its independently rerun
epoch-10 graph retained 13,704 / 153,115 edges (density **0.0895**) and had
multi-component fraction **0.8660**.  The small difference from the separate
operating diagnostic (0.0835 / 0.8866) is an independent nondeterministic
trajectory, not a pixel or representation-stage mismatch; both passed the same
fixed feasibility thresholds.

Before RSPG activation, raw test R@1 reached **0.8769** at epoch 10.  One epoch
after replacing positive-to-proxy ownership with the registered sparse detached
pair attraction, R@1 fell to **0.7102** and the logged objective fell from about
2 to **0.0018**.  R@1 then declined to **0.5070** at epoch 40; the best observed
value remained the pre-activation **0.8769**, 4.06 points below the prospectively
registered 0.9175 per-seed falsifier.

At the registered epoch-40 refresh, the signature graph retained only **2,200 /
153,115 edges**, density **0.0144**, with multi-component fraction **0.9970**.
This is below RSPG's own 0.05 feasibility floor.  The arm was terminated at step
5,800 / 8,580, before a final report/checkpoint existed.  The 0.8769 value is
therefore explicitly a partial-run observed best, not a completed benchmark or
selection-corrected result.  It is sufficient only as part of the mechanism
kill: the live supervision object ceased satisfying the registered gate.

The epoch-10 graph also makes the failure quantitative: only **53.23%** of the
25,882 training images had any eligible neighbor, with mean degree **0.9884** and
median degree **1**. Thus the replacement removed positive ownership for nearly
half of samples before the later graph collapse. A post-hoc “masked ownership”
repair would be a different selective-proxy method, not a correction to this
registered replacement, and is occupied by Proxy-ISA and partial-positive proxy
families.

## Mechanism

RSPG does not accidentally treat an unknown same-class relation as negative.
`_proxy_anchor_negative_loss` excludes the own-class proxy by label, and a new
unit test proves its gradient is exactly zero while a foreign proxy receives a
gradient.  An external critique proposed own-proxy repulsion, but code inspection
and the test falsified that explanation.

The actual failure is self-erasing supervision.  The selected detached positive
edges are already close enough that their softplus attraction is nearly
satisfied, while removing the positive-to-proxy ownership term discards the
force that maintains class organization.  Representation quality collapses;
rival-signature agreement then becomes still rarer at refresh (8.95% to 1.44%),
which removes more attraction.  Thus the positive-to-unknown operator was
implemented, but the proposed replacement supervision cannot sustain the base.

## Decision

Seed 1, soft reweighting, distance gate, instance-neighbourhood gate, Cars and
CUB are cancelled.  The controls test which gate explains a successful arm;
they cannot rescue a full method whose graph becomes infeasible.  Candidate 18
remains dead on corrected official pixels.  Future graph candidates must retain
the base ownership force and demonstrate a non-trivial additive signal at the
operating margin; merely changing the source of a sparse eligibility graph
repeats this failure class.
