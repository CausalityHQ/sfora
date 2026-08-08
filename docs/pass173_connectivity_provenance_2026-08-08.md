# Pass 173 — class-manifold connectivity provenance diagnostic

The stable fragmentation observations suggested that disconnected same-class
manifolds might be an information-bearing training referent. Before proposing
a bridge-preserving loss, this CPU diagnostic tests whether support-set
normalized-Laplacian algebraic connectivity predicts cross-fitted query R@1 on
the In-Shop training split beyond nearest-neighbour similarity.

The diagnostic is preregistered as a Gate-1 screen: connectivity must correlate
with class query accuracy at least `0.20` and exceed the nearest-neighbour
correlation by `0.05`. A failure means the fragmentation observation does not
identify a usable graph signal; no GPU is authorized.

## CPU result

On the corrected In-Shop training pack, the cross-fitted diagnostic retained
`1048` classes. Algebraic-connectivity correlation with query accuracy was
`+0.071725`, while nearest-neighbour similarity correlation was `+0.119417`.
The preregistered threshold failed in both directions. Connectivity therefore
does not supply Gate-1 provenance for a bridge-preserving method; no GPU run.
