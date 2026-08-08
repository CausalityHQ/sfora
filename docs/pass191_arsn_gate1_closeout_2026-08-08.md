# Pass191 ARSN Gate-1 closeout

ARSN (Amortized Rival-Statistics Normalization) is **DEAD at Gate 1** and
receives no GPU run.  Its premise was that a deterministic predictor could
infer the CE-BN leave-own-class-out moments from one image embedding, fixing
the hard/soft CE-BN train/evaluation mismatch.

On the existing trained In-Shop embedding pack, 500 random balanced batches
(18,000 held-out rows) produced the target moments.  A fixed linear predictor
from a 128-dimensional random projection of the full embedding gave aggregate
held-out R² `-0.0157` (mean targets `-0.0157`, log-variance `-0.0156`).  A
stronger two-hidden-layer MLP (128→256→128→128 outputs, eight epochs, train/test
batches kept disjoint) gave aggregate held-out R² `-0.0306` (mean `-0.0303`,
log-variance `-0.0321`).  Both are below the preregistered `R² >= 0.10`
amortizability requirement.  The target is batch-composition information, not
a deterministic property of an individual embedding.

This is an accuracy gate, not a CPU-time claim.  Concurrent wall-clock timing
was not used.  The mechanism is closed before implementation: no predictor can
recover the label-excluded moments from the deployment descriptor under this
test, so ARSN cannot repair CE-BN within the single-image constraint.

