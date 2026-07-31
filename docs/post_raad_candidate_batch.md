# Post-RAAD candidate batch: intervention-identified nuisance removal

**Gate 1 recorded 2026-07-31 before prior-art audit, implementation, or new
GPU work.**

## Provenance

ARCG showed that controlled spatial responses are stable and heterogeneous;
IPSR showed that ordering real neighbours by those responses changes In-Shop
R@1 by only +0.091 pt raw / +0.060 pt corrected. The response therefore behaves
like nuisance variation rather than identity relevance. RAAD then died on
direct instance-adaptive augmentation prior art.

The new question is not how to supervise the nuisance, but whether controlled
within-image differences can identify and remove it from the retrieval
representation.

## Candidate 1: IDNR — interventional differencing nuisance residualization

**External-science source:** fixed-effects estimation in econometrics removes
unit-specific signal by differencing repeated observations of the same unit;
the remaining within-unit change identifies the intervention-associated
component without outcome labels.

At the fixed epoch-10 checkpoint, retain the *vector* displacement
`z(intervention(x)) - z(center(x))` for every In-Shop training image and each
registered spatial intervention. Pool these within-image differences, estimate
their covariance, and define the nuisance subspace by a preregistered explained-
variance rule. During subsequent Proxy Anchor training and single-view
evaluation, project embeddings and proxies onto the orthogonal complement of
that fixed subspace before normalization and similarity. The ordinary label
objective remains unchanged; only directions empirically moved by controlled
spatial interventions are quotiented out.

The candidate is attractive because it uses paired differences to cancel
identity content, needs no nuisance labels or external model, adds one fixed
linear projection, and preserves one-view inference. It is risky under this
project's history because it changes representation geometry rather than adding
supervision. Gate 2 must therefore attack tangent distance/TangentProp,
augmentation tangent spaces, invariant subspace learning, nuisance attribute
projection, LEACE/INLP, and re-identification nuisance disentanglement before
any diagnostic.

## Candidate 2: local quotient similarity

Estimate a small tangent span separately for each image and compare two images
after removing the union of their local response spans. This is closer to the
measured per-image heterogeneity, but almost certainly collapses to classical
tangent distance and would require multiple stored vectors per gallery image.
It is lower priority and incompatible with the desired simple 512-d single-view
index unless a global approximation first succeeds.

## Candidate 3: intervention-orthogonal proxy updates

Project only the Proxy Anchor gradient—not the representation—away from the
global intervention subspace. This preserves full embeddings at inference but
is a form of constrained/gradient-projected optimization. It lacks direct
evidence that PA gradients occupy the nuisance span and is adjacent to
TangentProp and gradient surgery.

## Gate-1 decision

Advance **IDNR** to Gate 2. Its exact claim is a training-only, label-free
within-image intervention covariance used to construct one fixed global
quotient for supervised zero-shot retrieval. If classical tangent methods or
modern nuisance projection already implement that operator, kill it without
GPU.
