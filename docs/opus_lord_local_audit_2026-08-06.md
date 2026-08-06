# Pass 54 local evidence-aware audit: LORD

Date: 2026-08-06 UTC. Frozen proposal:
`docs/opus_lord_proposal_pass54_2026-08-06.md` (SHA-256
`65308915609a825de2baeb09fe3e61ee449c22e99e645367292c6b675c02b82a` is
noted in the freeze commit; the file was recovered from the completed Claude
fallback after its wrapper exited non-zero). No GPU work is authorised by
this audit.

## Gate 1 — provenance: FAIL

LORD is motivated by a claimed causal error mode: test-time diffusion or
manifold reranking supposedly repairs unseen-class impostors, and ordinary
label losses supposedly leave the relevant within-positive/within-negative
ordering unconstrained. The current evidence-reliability boundary does not
contain an audited training-only diffusion-headroom measurement, an impostor
reachability audit, or a measured relation between that headroom and official
query errors. The proposal's statement that a “large share” of CUB/Cars
impostors are manifold-far is an unverified premise, not repository
provenance. A future D1 diagnostic cannot retroactively establish Gate 1;
it would be an experiment conditioned on the proposal.

The proposal also treats the label-orthogonality claim as if selecting
positive and negative comparison sets did not itself use labels. Additive
shift invariance of a softmax within each set is true, but it does not make
the label-conditioned partition disappear from the objective. Thus the
claimed “zero information about the partition” is not a valid provenance
argument.

## Gate 2 — prior art and mathematical audit: FAIL

The repository already contains a primary-source audit of the direct
mechanism: Zeng et al., *Improving Deep Metric Learning via Self-Distillation
and Online Batch Diffusion Process*, Visual Intelligence 2024. OBD-SD freezes
a previous student, constructs a within-batch affinity/transition graph,
diffuses it, and distils the refined similarity into the current DML model;
it explicitly distinguishes training-time diffusion from test-time
reranking and evaluates CUB, Cars196, and SOP. LORD changes the graph scope,
truncates PageRank, restricts the KL to label-defined groups, and uses the
student's memory rather than an epoch teacher. Those are estimator,
stabilisation, and conditioning changes, not a new supervision action after
the claimed causal distinction fails. See
`docs/obd_sd_primary_prior_audit_2026-08-04.md`.

Additional occupied neighbours are already in the ledger: GCMT (IJCAI 2021)
for teacher similarity-graph consistency, contextual similarity distillation
(Wu et al., CVPR 2022), supervised contextual similarity optimisation (Liao
et al., arXiv:2210.01908), RKD/similarity-preserving distillation, XBM, and
S2SD. In particular, a graph-derived graded relation distilled into a
deployed global DML descriptor is not novel merely because the graph is
PPR, the teacher is nonparametric, or the KL is split into positive and
negative groups.

The frozen mathematical argument has independent failures:

1. At a collapsed memory, the graph is tied and the teacher target is
   uniform; both group KL gradients are zero. The proposed expansion factor
   rho is not a dynamical stability proof. Optimising a student to match a
   stop-gradient target gives a fixed target, then copying the student into
   memory reproduces its scale; it does not multiply perturbations by rho.
   Moreover the kNN graph is discontinuous at the all-ties collapse, so the
   claimed smooth Taylor expansion has no defined derivative there.
2. The proposal calls `g = -beta ||u_i-u_j||^2/2` a first-order perturbation,
   but it is second order in the perturbation. The asserted coefficient kappa
   is neither derived for the capped, symmetrised, tie-broken kNN graph nor
   shown positive for the stated normalised adjacency.
3. A truncated normalised-adjacency walk is not generally a row-stochastic
   PPR probability vector. Softmax can still consume nonnegative scores, but
   the claimed probabilistic “mass” and power-law interpretation do not
   establish a semantic teacher.
4. The “label-orthogonality” proposition only proves invariance to adding a
   constant separately within already label-selected sets. It does not prove
   that labels do not determine which pairs receive supervision; in fact the
   group restriction is explicitly label-conditioned.
5. `M[i] <- z_i` is an online cache, not an epoch-stable teacher. The proposal
   calls it one-epoch stale while the update rule refreshes sampled rows every
   step; no bound is supplied for drift, tie changes, or biased coverage.
6. The promised ≤1.07x cost is unsupported: each anchor performs a dynamic
   multi-hop kNN frontier, set construction, sparse graph symmetrisation,
   normalisation, and top-k selection over a full training memory. These are
   not “three sparse mat-vecs” alone, and no implementation benchmark exists.
7. The numerical forecast is explicitly below the audited Lane-A frontier
   (best LORD forecast 0.727 CUB, 0.909 Cars, 0.816 SOP versus PFML 0.734,
   0.927, 0.829). It therefore cannot meet the standing objective even if its
   unverified gains materialise.

The proposal's controls and thresholds are useful as a possible future
diagnostic, but they cannot repair Gate 1 or the occupied Gate 2 mechanism.
No preregistered GPU screen follows. A mandatory independent cold review is
still requested to preserve the protocol and to test whether any substantive
disagreement remains.

## Cold-review reconciliation

The independent review returned DEAD and agrees with the earliest Gate-1
decision. It independently confirms that the proposed provenance is only an
uncited diffusion-headroom assertion plus a generic label-loss argument. It
also confirms the circular temperature controller, the epsilon-driven
unregistered target gap, the missing dynamic-kNN cost, and the omitted
In-Shop forecast. Most importantly, its primary-source sweep identifies
Zeng et al.'s OBD-SD (arXiv:2211.07566; *Visual Intelligence* 2024) as the
same training-time graph-diffusion/self-distillation object and action on
ProxyAnchor, with plain-cosine deployment. The residual group-normalisation
property is accepted as a useful ablation, not a novel method. No material
disagreement remains and LORD is closed without GPU work.

Independent artifact: `docs/opus_lord_review_2026-08-06.md`.
