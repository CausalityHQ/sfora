# Pass 56 — Blind Proposal (recovered IFR specification)

**Method: IFR — Isothermal Fluctuation–Response Regularization.** Lane A:
ResNet-50, 512-D normalized descriptor, 224 px, single-view cosine retrieval,
200-epoch-equivalent budget. This file is a faithful recovery of the completed
Opus answer from retained terminal output. The shell harness omitted a middle
segment of the stream after completion; no scientific rerun was made. The
visible complete mathematics, controls, forecasts, and risks are preserved
below, and this retrieval limitation is itself non-authorizing until reviewed.

## Method and objective

Let `phi_omega` be a ResNet-50 ImageNet-1K backbone, `W` a bias-free
`512 x 2048` head, `z=W phi(x)`, and `zhat=z/||z||`. Learned normalized
proxies are used by a PFML base. A class-balanced batch has `C=32` classes,
`M=3` distinct images, and `V=2` augmentations (`B=192` views). Duplicated
SOP sources are masked from the fluctuation channel.

For image `i` in class `c`, the response displacement is
`dR_{c,i}=zhat_{c,i}^{(1)}-zhat_{c,i}^{(2)}`. The fluctuation channel contains
all cross-source, all-view differences within a class,
`dF_{c,(i,v),(j,v')}=zhat_{c,i}^{(v)}-zhat_{c,j}^{(v')}` for `i<j`.
Define low-rank second moments

```
SR_c = (1/M) sum_i dR dR^T
SF_c = (1/NF) sum dF dF^T
SbarX_-c = (1/(C-1)) sum_{c' != c} SX_c'.
```

For either channel, use leave-one-class-out isothermality

```
a_c = stopgrad(<S_c,Sbar_-c>/(||Sbar_-c||_F^2 + eps0))
Psi = sum_c ||S_c - a_c Sbar_-c||_F^2 /
      (stopgrad(sum_c ||S_c||_F^2) + C eps0),   eps0=1e-4.
```

The full objective is

```
L = L_PFML + gamma_R(t) Psi(SR) + gamma_F(t) Psi(SF)
    + beta/(CM) sum_{c,i} ||zhat_{c,i}^{(1)}-zhat_{c,i}^{(2)}||^2,
```

with `gamma_R=gamma_F=1`, `beta=0.25`, and a ramp from epoch-equivalent 5 to
20. The consistency term is explicitly a non-novel amplification guard and is
isolated by control C2. The Gram-domain implementation computes inner products
of second moments from `D D^T`; no 512x512 matrix is materialized. The proposed
overhead is about 0.12 GFLOP per step and ~0.6 MB.

The deployed descriptor is only `zhat`; training-only augmentation responses,
scatter operators, proxies, and leave-one-out statistics disappear at test.

The stated causal mode is class-gated nuisance cancellation: a proxy loss can
learn class-specific nuisance circuits rather than a class-generic invariant
circuit. On unseen identities those gates do not fire, increasing within-class
scatter. IFR claims that class-specific scatter shapes are unpredictable from
the leave-one-out consensus while a class-generic nuisance operator is
predictable, so it prefers the latter. The proposer cites Zhou et al.,
“Do Deep Networks Transfer Invariances Across Classes?” (arXiv:2203.09739 / ICLR
2022) as evidence for the disease; its generative remedy is excluded here.

The proposal acknowledges total collapse: `Psi` and consistency vanish, while
the repulsive PFML term diverges at inter-class collapse. It attacks common-mode
amplification by detaching the denominator and adding the consistency guard.
It claims the leave-one-out construction is the essential zero-shot simulation.

## Base recipe and controls

PFML is reproduced with its disclosed potential-field form, normalized proxies,
Adam, proxy learning rate 100x the network rate, 200 epochs, 15 proxies on
CUB/Cars and 2 on SOP/In-Shop. The proposer freezes its own completion of
undisclosed settings at `alpha=3`, `delta=0.2`, weight decay `1e-4`, cosine LR
decay, 5-epoch warmup, random resized crop and flip, and no head BN/bias; it
explicitly does not inherit PFML’s published means.

Controls: C0 PFML completion; C1 matched two-view batching without IFR; C2
consistency only; C3 response channel only; C4 fluctuation channel only; C5
full IFR; C6 no leave-one-out; C7 permuted class labels; C8 coding-rate/VICReg
alternatives; C9 IFR on unnormalized `z`; C10 head-only after epoch 20; C11
post-hoc projection of consensus eigendirections (WCCN control); C12 2x compute.
A held-out-training-class probe measures `Psi` and scatter, never test classes.

## Frozen forecasts and falsifiers

Five-seed forecasts over the proposer’s own PFML completion:

| dataset | base C0 | IFR C5 | delta |
|---|---:|---:|---:|
| CUB | 0.726 +/- 0.004 | 0.739 +/- 0.004 | +1.3 pp |
| Cars196 | 0.921 +/- 0.004 | 0.929 +/- 0.004 | +0.8 pp |
| SOP | 0.826 +/- 0.003 | 0.830 +/- 0.003 | +0.4 pp |
| In-Shop (unverified PFML extension) | 0.921 +/- 0.004 | 0.927 +/- 0.004 | +0.6 pp |

The proposal honestly says these do not decisively cross the published PFML
frontier (CUB only +0.5 pp nominally; Cars/SOP below one sigma; In-Shop below
PA+DADA). It instead claims a decisive paired improvement over its own base.
Falsifiers include CUB delta <0.4 pp, SOP delta >= CUB delta, consistency-only
explaining 70%, no held-out-class scatter reduction, reduced total scatter,
amplification >1.5x, post-hoc projection recovering 70%, permuted labels
recovering 70%, no-LOO matching full IFR, or gain only at 2x compute.

## Prior-art claims and risks

The proposer distinguishes IFR from Anti-Collapse/coding-rate, VICReg,
class-conditional invariance, augmentation-aware SSL, covariance-preserving
augmentation, WCCN, episodic/meta-learning, and fluctuation-dissipation work by
the specific leave-one-class-out prediction of class-conditional displacement
second-moment *shape*. It states that no prior work was found applying that
predictive statistic as DML supervision. Risks include thin per-class rank,
response/fluctuation redundancy, linear-head/WCCN reduction, recipe ambiguity,
and contamination inherited from ImageNet pretraining.

Sources named by the proposer include PFML (arXiv:2405.18560), Zhou et al.
(2203.09739), PA+DADA (2401.00617), Anti-Collapse Loss (2407.03106),
Rethinking Class-Collapsing (ICCV 2021), AdvRF (2507.21742), Rotating Spiders
(2106.04009), IAA (2211.16264), conditioned-projector augmentation SSL
(2306.06082), Threshold-Consistent Margin (2307.04047), and stochastic
fluctuation-dissipation work (2106.02220).
