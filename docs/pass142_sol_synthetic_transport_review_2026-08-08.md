# Pass 142 — synthetic within-class transport review (Gate 2: NONE)

This read-only Sol review was focused on the remaining synthetic-within-class
supervision escape route. Fable and Claude were unavailable at their weekly
limit; that absence is not treated as corroboration. No files were edited by
the consultation and no GPU run is authorized.

## Provenance

The repository has a real transfer defect but not an identified transferable
factor: stable fragmentation affects 1,439/3,975 identities (cross-seed
partition agreement 0.94465), cardinality-matched unseen-minus-seen nearest-
positive similarity is -0.04968, and cross-fitted centroids exceed learned
proxies by roughly +0.661--+0.724 median margin. CIS and its paired Proxy
Anchor tied at raw-best In-Shop R@1 0.9170 in one seed, while CIS ended 0.9149
versus 0.9158 and has shared-gradient owner contamination.

## Candidate and fatal fork

The strongest remaining proposal transports a same-class donor variation on the
unit sphere. For donor pair `(a,b)`, let `delta_ab = log_{z_a}(z_b)` and for a
recipient `i` define `z_tilde_i = exp_{z_i}(T_{a->i} delta_ab)`, then supervise
`z_tilde_i` with the recipient class. This is exactly synthetic feature/embedding
augmentation. Changing the transport map, using spherical parallel transport,
or observing rather than sampling the variation changes an estimator, not the
supervision mechanism. Primary collisions are Delta-Encoder (NeurIPS 2018), Deep
Variational Metric Learning (ECCV 2018), Spherical Feature Transform (ECCV
2020), Meta Variance Transfer (AISTATS 2020), and SEE (IJCAI 2025).

If no virtual point is created, the remaining form is a relation target
`ell(<z_i,z_j>, stopgrad(r_hat_ij(D_train)))`: detached targets are pseudo-label
or relational distillation; live targets are message passing, graph
supervision, or pair reweighting. Online batch diffusion self-distillation,
intra-batch connections, and self-supervised intra-class ranking occupy those
mechanisms.

The CIS operator supplies no third route. Its normalized coalition gradient
gives every member the same proxy-gradient vector. Owner-safe linear leave-one-
out credit reduces to `tau p_c^T z_i`, ordinary single-image proxy supervision;
nonlinear coupling restores contamination or becomes compositional/routed
prior art.

## Verdict

**NONE at Gate 2.** No candidate survives the provenance/identifiability and
primary-art checks. The existing disjoint-identity CPU falsifier
(`rho_32 = 0.9312, 0.9287, 0.9345`, all below the locked 1.15 floor) further
rejects the load-bearing transferable-variation premise. No implementation,
preregistration, corrected In-Shop prediction, or GPU run is authorized from
this pass.
