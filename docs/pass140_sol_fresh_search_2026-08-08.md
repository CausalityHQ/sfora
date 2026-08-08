# Pass 140 — Codex Sol fresh mechanism search (2026-08-08)

## Result

**NONE at Gate 2.** Fable and its automatic Claude fallback were unavailable
because both providers hit the weekly limit. I therefore ran the same
read-only, repository-grounded review with Codex Sol (`gpt-5.6-sol`). It read
`docs/search_protocol.md`, `docs/method_search_verdict.md`, and the Pass 135–139
memos, and searched primary literature before proposing any implementation.
No candidate is authorized for CPU, implementation, or GPU work from this
pass.

The negative result is not an impossibility proof. It is a record that the
three strongest escape attempts below failed the project's novelty boundary.

## Attempt 1 — apical/local target credit assignment

The proposed mechanism was to generate hidden-layer targets with difference
target propagation,

\[
t_l = G_l(t_{l+1}) + h_l - G_l(h_{l+1}),
\]

and train each layer with a local target error. This is directly occupied by
Difference Target Propagation (Ernoult et al., ICML 2022), Deep Feedback
Control (Nøkland et al., NeurIPS 2021), and dendritic/error-feedback local
learning. It also has no repository measurement showing that backpropagation
credit assignment is the limiting defect. It therefore fails Gate 2.

## Attempt 2 — nonreciprocal proxy–encoder dynamics

An active-matter-inspired proposal let the encoder follow Proxy Anchor while
detached class proxies followed a different centroid/confusion flow. Written
as equations, this is an alternating proxy optimizer or stop-gradient proxy
update, not new supervision. It collides with stop-gradient softmax metric
learning and the ledger's proxy calibration, class-conditioned optimizer, and
preconditioning families. A CPU Jacobian decomposition would only show
changed coefficients on existing gradient atoms, so it fails Gate 2.

## Attempt 3 — group-tested coalition ownership

Signed multi-image codes and repeated coalition measurements could decode each
member's proxy ownership. Fixed codes add no information beyond deterministic
ECOC; learned codes become routed/codebook supervision. At bundle size two,
the ownership gradient is either shared or reduces to existing single-image or
compositional controls. It collides with CIS/SRC, Deep Compositional Metric
Learning, vector-symbolic binding, and the Pass 135 ownership-code audit.
It fails Gate 2.

## Search conclusion

The decisive unresolved boundary remains: identity equality alone does not
identify which within-class variation transfers to unseen identities. Proposed
signals so far reduce to geometry, weighting/mining, augmentation,
distillation, meta-learning, compositional mixtures, or established
credit-assignment rules. The current benchmark ceiling is also material: any
general claim must confront approximately 0.939 In-Shop, 0.766 CUB, or 0.949
Cars196 under the project's single-model/single-view protocol.

This pass consumed no GPU and produced no implementation. The live CIS
control sequence remains the next empirical evidence; its result may motivate
a narrower, measurement-grounded search, but this memo does not authorize a
new candidate by itself.
