# Pass 121 fallback batch (research only)

This is a search memo, not a preregistration and not GPU authorization.  The
common provenance is Pass 67's corrected CUB decomposition: **51.9%** of
failed queries are in the between-class-centroid-overlap component.  CIS is
the deciding Pass 120 screen; these alternatives are held in reserve until
its controls resolve.

## 1. Stoichiometric residual coalition (SRC)

For a bundle of distinct classes, retain CIS's normalized sum but also form
each leave-one-out residual (the sum of all members except one).  Supervise
the sum with the union of member proxies and each residual with the union
minus its omitted class.  The training object is a *set of compositional
equations*, not a pairwise score or a reweighted positive.  It targets the
same measured between-class interference while giving the model a
counterfactual for which class contribution was removed.

Gate-2 warning: Deep Sets/Set Transformer, mixup/manifold mixup, multi-label
supervision, and Potential Field DML are adjacent.  SRC is only viable if a
primary-source audit finds no method that trains leave-one-out set residuals
against proxy-label complements.  If that distinction is cosmetic, kill it
before implementation.

## 2. Proxy-mass transport supervision (PMTS)

Treat a class bundle as a discrete mass distribution over its member proxies.
Train a permutation-invariant Sinkhorn transport objective between member
embeddings and the proxy mass, with the transport plan itself supplied as
supervision to the ordinary image descriptors.  Deployment remains one
descriptor; only the training set object is a distribution.  The falsifier
would be a no-transport multi-label control matching the result.

Gate-2 warning: Sinkhorn losses, SoftTriple/sub-centers, OT metric learning,
and class-distribution matching are close prior art.  This candidate is not
alive until the exact *bundle-to-proxy mass supervision* mechanism is checked
against those sources.

## 3. Proxy-conditioned gated residual head (PGRH)

Add a train-time head that predicts a low-rank residual for each image's
own-proxy direction and uses the residual to create a second, contrastive
label: images whose residuals agree receive a positive relation, while the
deployed path discards the head.  Provenance is the observed split between
proxy-centroid correctness and nearest-image correctness; the head would
supervise the currently unmodelled residual factor rather than alter cosine
scoring.

Gate-2 warning: teacher-view consensus, factorized metric routers, channel
rectifiers, and cross-image residual exchange are all adjacent or occupied.
This is the lowest-priority option and should be killed if it reduces to a
known factorized/teacher relation objective.

No number is preregistered here.  A surviving option must pass the full
protocol, including prior art before code, an In-Shop screen, out-of-sample
confirmation, raw plus selection-corrected reporting, and a second dataset.
