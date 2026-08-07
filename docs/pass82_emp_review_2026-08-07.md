# Pass 82 EMP review (2026-08-07)

## Verdict: DEAD; no GPU authorized

The independent review rejects Excitable-Medium Pooling (EMP) at Gate 1 and
Gate 2, with additional protocol and frontier failures.

### Gate 1: no repository provenance

The proposal explicitly says it did not inspect the repository. Its motivating
number is a proposed future CPU preflight, not an existing audited measurement
from this project. A repository search found no EMP/FitzHugh--Nagumo result.
This violates the requirement that a candidate be motivated by a measurement
already made here.

### Gate 2: occupied mechanism

EMP is four tied residual steps combining a grid Laplacian, pointwise cubic
reaction, a recurrent state, and global average pooling. This is not a
materially new operator: cellular neural networks (Chua & Yang, 1988),
FitzHugh--Nagumo reaction-diffusion on image grids (Li et al., IEEE TIP 2015),
optimized reaction-diffusion (Chen et al., CVPR 2015), PDE-Net (Long et al.,
ICML 2018), GRAND (Chamberlain et al., ICML 2021), and GREAD (Choi et al.,
ICML 2023) cover the core construction. Scattering, R-MAC, NetVLAD, GeM,
spatial pyramids, bilinear CNNs, and attention-based DML cover the spatial
pooling endpoint. No exact FHN-to-retrieval paper was found, but a new name or
benchmark application is not enough to establish a new method.

### Independent failures

The proposal targets CUB first, contrary to the corrected protocol requiring
In-Shop first. It supplies no In-Shop prediction, paired control, or kill
threshold. Its optimistic CUB forecast (69.5 raw, 68.9 corrected) is about
3.9 points below the matched 512-D PFML frontier (73.4 +/- 0.3), so it does
not even forecast crossing the relevant frontier. The claimed <0.1% overhead
is also wrong: moving a 2048x512 projection before pooling over a 7x7 map adds
about 50.3M MACs, roughly 1.23% of a 4.09G ResNet-50 before dynamics.

The proposed CPU preflight applies the operator to raw backbone channels,
without the learned projection and tanh used by the deployed model, so a
passing AUC would not validate the candidate. Boundary conditions, stability,
pair weighting, and the permutation falsifier were underspecified; linear
diffusion followed by conservative global averaging collapses to scaled GAP.

The mandatory Fable review was unavailable because Fable and its Claude
fallback hit the weekly limit; this is procedural non-authorization, not
positive evidence. No implementation or GPU run occurred.

Full consultation result: independent Codex review `492e72bbead744a1`.
