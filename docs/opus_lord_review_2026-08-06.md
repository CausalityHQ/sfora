# Pass 54 cold review: LORD

Review job `f7e0f5e2b64d4a21`, completed 2026-08-06 UTC. The reviewer
verified the frozen proposal SHA-256
`65308915609a825de2baeb09fe3e61ee449c22e99e645367292c6b675c02b82a` and
returned **DEAD**. Earliest failed gate: **Gate 1 (provenance)**. Gates 3 and
4 fail independently; no GPU is authorised.

## Gate 1 — FAIL

The proposal's central premise is an uncited claim that a large share of
unseen-class impostors are manifold-far and are repaired by diffusion, plus a
generic claim that label losses provide no graded local-geometry signal. No
fraction, run, artifact, or repository measurement establishes either claim.
The proposed D1 diffusion-headroom diagnostic is future work, so it cannot
retroactively establish provenance. The proposal's own text calls transfer to
unseen identities an assumption and its sources contain no measurement of the
error mode.

## Gate 2 — material defects

The group-softmax shift invariance and displayed KL gradient are correct, but
the collapse argument is not a mechanism certificate. The controller defines
`b=(tau_s/tau_t)*teacher_logits`, so the reported expansion ratio is
proportional to the temperature it controls; feedback drives the ratio to the
target by construction. It can amplify dispersion on a random graph just as
well as on a semantic graph, making the stability falsifier circular.

The epsilon guard is omitted from the frozen hyperparameter table even though
zero PPR entries become log(epsilon), creating a large, arbitrary cosine gap
for unreachable negatives and unspecified tie behaviour. The stated global
map `g_(n+1)=rho*g_n` is not derived for a cache whose rows are updated one at
a time; the kNN graph is discontinuous at all-ties collapse. Finally, the
cost estimate counts only three sparse mat-vecs and omits dynamic frontier
kNN construction over the full training memory. The reviewer notes that the
closest published implementation reports material overhead, above LORD's
own 1.15x kill threshold.

## Gate 3 — FAIL, occupied object/action

Zeng et al., *Self-distillation with Online Diffusion on Batch Manifolds
Improves Deep Metric Learning*, arXiv:2211.07566 (journal: *Visual
Intelligence*, 2024), freezes an earlier DML model, builds a normalized
affinity graph, applies restart diffusion, and distils the diffused
similarity into the current model with a supervised DML loss including
ProxyAnchor. It is training-time diffusion, plain-cosine deployment, and the
same graph-derived similarity-distillation object/action. LORD's memory
substrate, truncation, temperature controller, and label-defined group split
are implementation/conditioning variants, not a mechanism-level distinction
under this protocol. GCMT, contextual-similarity distillation/optimization,
RKD, XBM, and S2SD are additional occupied neighbours.

## Gate 4 — FAIL

The proposer explicitly forecasts no Lane-A frontier crossing: 0.727 CUB,
0.909 Cars, and 0.816 SOP versus audited PFML 0.734, 0.927, and 0.829. The
conditional PFML+LORD claim is expressly withdrawn and has no matched PFML
reproduction. The proposal also omits an In-Shop forecast row. Good protocol
hygiene (five seeds, held-out training-identity tuning, explicit controls and
source ambiguities) does not rescue a below-frontier candidate.

## Surviving distinction and disposition

The zero-sum, label-conditioned group gradient is a real checkable property,
but it is a normalization-domain refinement of an already occupied
graph-diffusion/relational-distillation action. It belongs, at most, as an
ablation inside an OBD-SD-style study, not as a novel standalone method.

Primary sources checked by the reviewer: OBD-SD arXiv:2211.07566 and journal
version DOI 10.1007/s44267-024-00051-0; STML CVPR 2022; and Liao et al.,
*Supervised Metric Learning to Rank for Retrieval via Contextual Similarity
Optimization*, ICML 2023 / arXiv:2210.01908. Review was read-only and no
files were edited by the reviewer.
