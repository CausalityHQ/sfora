# Sixth isolated Fable pass: GES near-miss rejected

Date: 2026-08-05. The blind proposer was durable consultation
`055668d7dc704965`; it returned `NONE`, while exposing one fully specified
near-miss for audit: **Grassmannian Erasure Sufficiency (GES)**. Because the
proposer itself returned `NONE`, this is not a numbered candidate and does not
authorize a diagnostic, implementation, preregistration, or GPU run. A second
fresh reviewer was started as consultation `04061cd719f34ffe` before the local
verdict. Its final result is recorded below when available.

## Frozen mechanism

For normalized descriptor `z = g(x) / ||g(x)||`, GES lets an adversary choose
an arbitrary rank-`k` subspace `U`, projects it away with
`P_U = I - U U^T`, recomputes leave-one-out nonparametric class prototypes in
the projected representation, and minimizes the sum of the ordinary loss and
the worst projected loss. The claimed causal problem is *shortcut monopoly*:
a low-dimensional feature subspace allegedly becomes the sole carrier of seen
class identity, preventing other transferable evidence from being learned.
The coding-theory analogy says that identity evidence should survive erasure of
any `k`-dimensional subspace.

The proposer forecast only +2--4 CUB points, admitted that this would not cross
the audited 0.766 ResNet-tier frontier from its assumed 0.71 base, and returned
`NONE`. Its applicability argument also predicted less benefit on In-Shop,
where the comparable frontier gap is smaller. The proposal incorrectly left
the AdvRF capacity tier unverified; the repository's primary-source horizon
audit already verifies the relevant result and capacity lane.

## Gate 1: no measured shortcut monopoly

The corrected evidence packet measures highly stable In-Shop query errors,
cross-seed wrong-identity agreement, training-set saturation, and fragmented
within-class neighbourhoods. None identifies a low-rank subspace as the unique
carrier of seen-class information. Between-class rank being at most `C-1` is a
linear-algebra fact, not a repository measurement that this rank bottleneck
causes zero-shot errors. GES therefore lacks the required measurement-derived
provenance.

## Algebraic counterexample: arbitrary subspace erasure cannot certify the claim

Take any two nonidentical class prototypes `mu_a` and `mu_b`, and let
`v = mu_a - mu_b`. A rank-one adversary may choose

```
U = span(v).
```

Then

```
P_U (mu_a - mu_b) = 0,
```

so the two projected prototypes are identical before normalization. This is
true however widely the network has distributed the coordinates of `v`: every
pairwise difference is still one vector, and an arbitrary one-dimensional
subspace can erase its entire span. The erasure-code analogy therefore fails.
Coordinate erasures can be resisted by redundant coordinates; erasing an
adversarially chosen continuous subspace containing the completed codeword
difference cannot.

There is a second singularity. The adversary may choose a subspace containing a
sample or prototype vector, producing `P_U g = 0`; the stated normalization is
undefined. Adding an epsilon changes the result into epsilon-scale numerical
noise rather than a sufficiency certificate. Consequently, the minimax loss has
an unavoidable positive worst-case term even for an otherwise perfect
descriptor. Optimizing it can at most redistribute pairwise margins or flatten
the between-class spectrum; it cannot achieve the stated invariant.

## Gate 2: the remaining operation is occupied

Even without the fatal counterexample, GES is a worst-case feature-deletion
regularizer:

- Park et al., *Adversarial Dropout for Supervised and Semi-Supervised
  Learning* (AAAI 2018), chooses dropout masks that maximally damage the current
  supervision and trains the reconfigured network.
- Globerson and Roweis, *Nightmare at Test Time: Robust Learning by Feature
  Deletion* (ICML 2006), trains against worst-case deleted features.
- Ravfogel et al., *Linear Adversarial Concept Erasure* (ICML 2022), formulates
  identifying and removing a linear concept subspace as a constrained minimax
  game.

Changing an axis-aligned deletion mask to a Grassmann point is not enough when
the candidate's claimed advantage from that change is mathematically
impossible. Repository candidate 157 already rejected matches that must survive
independent evidence erasure as dropout/erasure consistency or hashing/ECOC;
candidates 246 and 250 separately close nested-dimension dropout and fixed
error-correcting class codes.

Primary sources:

- https://ojs.aaai.org/index.php/AAAI/article/view/11634
- https://icml.cc/Conferences/2006/proceedings.html
- https://proceedings.mlr.press/v162/ravfogel22a.html

## Verdict

**DEAD at Gates 1 and 2.** The local verdict does not depend on the proposer's
conservative effect forecast: GES has no verified causal provenance, and its
load-bearing certificate is impossible because a rank-one adversary can erase
any selected pairwise prototype difference exactly. No GPU follows.

## Independent frozen-proposal review

The fresh reviewer in durable consultation `04061cd719f34ffe` returned
**DEAD**. Fable stopped without completing after 11 minutes, so the configured
same-prompt fallback completed under Claude Opus. The useful independent
findings were:

- the objective can reward redundancy of an encoding without acquiring new
  transferable visual information;
- its cheap linear path is label-conditioned spectral redistribution, which
  undermines the claimed separation from spectral regularization and can trade
  away clean margin;
- no solver or value of `k` is specified for the load-bearing inner maximum;
- DiVA, BIER, adversarial dropout, and R-LACE are closer mechanism neighbours
  than the frozen proposal acknowledged; and
- the proposer was right to return `NONE`, although it should have verified
  AdvRF's tier before making tier-dependent arithmetic the stated reason.

The review also made two material errors that are rejected rather than imported
into the ledger. First, it claimed that 0.939 In-Shop was a mis-tiered CRT/
MiT-B2 number. The audited record distinguishes **VAPNet 0.939 with
ResNet-50/GAP** from **CRT 0.9448 with MiT-B2**. Second, it invoked Whitney's
generic-projection embedding result to claim that a nonlinear lift can preserve
a shortcut under *every* adversarial subspace. “Generic” or almost-everywhere
does not imply every subspace; choosing the rank-one span of a particular
secant erases that secant exactly. Neither error changes the DEAD verdict, but
both would corrupt the evidence ledger if repeated.
