# Pass 186 — Foreign-Gradient Fisher Preconditioning (DEAD at Gate 2)

The candidate would maintain a per-proxy covariance of gradients from foreign
negative terms only and precondition that proxy's update by its inverse square
root. Gate 1 provenance was the corrected epoch-10 acquisition-drift audit:
foreign-negative updates differ between cross-acquisition and same-group
pairs, while same-class gradient conflict is measured at 17.94%.

The distinction from generic K-FAC, Shampoo, AdaHessian, and Riemannian DML is
the foreign-negative-only covariance source. However, the repository's
independent Pass 147 review already audited the complete gradient/optimizer
escape class and found these measurements motivate occupied gradient-surgery,
gradient assessment, and preconditioning mechanisms. Proxy-ISA and
non-isotropy regularization are adjacent primary examples. A foreign-only
subcovariance is an optimizer estimator variant, not a new supervision object.
It is therefore DEAD at Gate 2; no CPU or GPU work was performed.

