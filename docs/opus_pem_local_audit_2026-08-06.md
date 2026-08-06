# Pass 37 local evidence-aware audit: PEM

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_pem_proposal_pass37_2026-08-06.md`  
Frozen proposal SHA-256: `aae89d8415ebad38ed1a41d1f96e8920d977e55d9268f9a85a1562a1ce6b718a`  
Proposer job: `2f8de7b0d24e42d6` (complete durable answer; runner exited 1 after streaming it)

This audit was written before launching or reading the mandatory independent
review.

## Provisional verdict

**DEAD at Gates 1 and 2, with the frozen executable specification internally
incomplete. No preregistration, implementation, or GPU is warranted.**

PEM integrates a class direction out of a vMF random-effects likelihood and
trains an embedding so the labelled batch partition beats local move, merge,
and split corruptions. The marginal likelihood is a coherent mathematical
object. The claimed causal provenance, absolute calibration, isotropy action,
and novelty do not follow from it, and the specified lookup cannot evaluate all
of its own hypotheses or train its only scalar parameter.

## Gate 0 and Gate 1: no verified proxy-absorption measurement

The proposal's causal premise is “proxy-absorbed calibration”: free seen-class
parameters allegedly let the backbone retain anisotropic, absolutely
miscalibrated class clouds that fail on unseen identities. The verified packet
contains no measurement that removes proxies at matched recipe and shows better
held-out-identity calibration, no transfer test of a learned absolute threshold,
and no evidence that unseen errors are caused by class-covariance anisotropy.

The closest repository evidence points away from the stated mechanism:

- the strongest audited Lane-A reference, PFML, uses *more* proxies (15 per
  class on CUB/Cars), not fewer, and PEM supplies no measured reason that its
  advantage should reverse that result;
- the project has repeatedly found that additional compactness or isotropy
  pressure is regularisation on a base already fitting rather than expanded
  supervision; and
- Pass 21 RIM and Pass 36 DARC found no verified causal link from a chosen
  within-class shape statistic to unseen-identity R@1.

The forecast `0.741/0.932/0.831` is an unsupported prior, not repository
provenance. It predicts only one conditional significance crossing and itself
declines the required corrected In-Shop frontier claim. Gate 1 therefore fails
before the mathematical defects below are considered.

## The frozen executable is incomplete in three independent places

### The lookup domain excludes move and merge states

Section 1.2 precomputes `phi(r)` only for `r in [0,K]`. But the specified
hypotheses require:

- `||S_b+z_i||` up to `K+1` for a move into a size-K class; and
- `||S_a+S_b||` up to `2K` for a merge.

Those are not boundary curiosities: coherent classes approach these upper
bounds. No extrapolation rule or larger table is specified. Consequently the
declared 4096-point spline cannot evaluate two of the three move families whose
logits define the loss.

### The learned concentration has no gradient path

The proposal declares `kappa=softplus(a)` learned at LR `1e-2`, but its custom
backward supplies only

```
d phi(R_A) / d z_i = kappa A_d(kappa R_A) S_A/R_A.
```

The table is “precomputed by quadrature” and rebuilt only whenever kappa changes
by more than one percent. That detached, thresholded rebuild supplies no
`d phi/d kappa`, hence no `dL/da`. If the table were instead built inside an
autograd graph every step, the stated caching procedure and custom backward
would not be the executable algorithm. The sole non-network learned object is
therefore not learnable as written.

### The schedule contradicts the clamp

Kappa is scheduled from 16 to 96 during epochs 1--20 while also being clamped
to `[32,256]`. Both cannot hold during the first part of training. It is also
unclear whether the schedule overrides `a`, initializes it, or adds a target to
it. This matters because kappa is the proposal's claimed absolute ruler, not a
harmless scale.

Any one repair changes the frozen executable and belongs to a new proposal.

## Kappa is an optimizer escape, not an identified absolute threshold

Even after supplying the missing derivative, kappa is not identified by the
model. For an already correctly clustered configuration, increasing kappa
makes coherent true clusters increasingly favoured over separated move/merge
alternatives. Thus the loss can reduce by driving kappa toward the arbitrary
upper clamp rather than by discovering a task-intrinsic calibration.

A standard-library evaluation of the proposal's exact
`phi(r)=log 0F1(;256;kappa^2 r^2/4)` at its own `R=6.8` illustrates this. For
two class means at 90 degrees, the merge log-evidence difference
`phi(sqrt(2)R)-2phi(R)` was:

| kappa | merge delta |
|---:|---:|
| 32 | -4.9097 |
| 64 | -35.1371 |
| 96 | -89.0807 |
| 128 | -158.5995 |
| 256 | -515.1654 |

For one perfectly coherent size-eight class, the split difference
`2phi(4)-phi(8)` similarly falls from `-26.7898` at kappa 32 to `-303.6963`
at 256. Nothing in the loss offsets this escape. The proposed “absolute” ruler
is therefore chiefly the chosen clamp endpoint.

The proposal's worked threshold is also numerically inaccurate beyond its
acknowledged roughness. Exact bisection at `kappa=96, R=6.8, d=512` puts the
equal-size merge zero at **74.610 degrees**, not about 68 degrees. Across
kappa `16/32/64/96/128/256`, the exact zeros are
`88.813/86.076/79.889/74.610/70.335/59.189` degrees. A threshold that moves by
nearly 30 degrees over the permitted scalar range is not dimension-calibrated
in the operational sense claimed.

Finally, the optimized posterior is not controlled by a single delta's zero
crossing. The loss sums roughly `B(P-1)` move hypotheses, `P(P-1)/2` merges,
and `P` splits. Small loss requires each family to overcome its multiplicity,
approximately a `log M` offset. The effective margin therefore depends on batch
composition and the arbitrarily enumerated corruption neighbourhood as well as
`(d,kappa)`.

## The split statistic does not identify isotropy

The integrated vMF evidence for a cluster depends only on its resultant norm
`R_A`. It contains no covariance matrix or eigenvalue functional. Many
anisotropic point configurations have the same resultant and exactly the same
evidence. Searching one top-eigenvector split may detect a two-mode elongation,
but rejecting that split does not imply isotropic covariance.

The claim that PEM “stays high until within-class scatter is near-isotropic” is
therefore false. The split action is more accurately ordinary supervised
unimodality/compactness pressure: make every labelled class hard to divide into
two higher-evidence groups. That may erase useful pose, viewpoint, sex, or
lifestage structure just as readily as it removes nuisance. A spherical vMF
likelihood assumes isotropy; fitting its evidence does not prove that isotropy
is the causal condition for unseen cosine retrieval.

The instance-memorisation argument also fails. A network can memorize training
images while arranging each labelled class as a compact nearly regular cloud
and class resultants far apart. That construction satisfies move, merge, and
split logits simultaneously. Memorisation is not “indifferent” to the split
term once the split term is part of the training objective.

## Gate 2: occupied supervision object and action

PEM's literal Bessel score may be a new wrapper. Its supervision object is not:
the labelled partition is required to outscore corrupted partitions under a
learned representation. That is the object and action of supervised metric
learning for partitioning and structured-clustering DML.

- Song et al., *Deep Metric Learning via Facility Location* (CVPR 2017), train
  the embedding through structured prediction so the ground-truth clustering
  beats alternative clusterings:
  https://openaccess.thecvf.com/content_cvpr_2017/html/Song_Deep_Metric_Learning_CVPR_2017_paper.html
- Lajugie, Arlot, and Bach, *Large-Margin Metric Learning for Partitioning
  Problems* (ICML 2014), learn a metric from labelled target partitions using a
  structured prediction objective:
  https://arxiv.org/abs/1303.1280
- Law, Urtasun, and Zemel, *Deep Spectral Clustering Learning* (ICML 2017),
  optimize representations so supplied similarity labels induce the desired
  partition:
  https://proceedings.mlr.press/v70/law17a.html
- DeepDPM uses marginal mixture modelling plus explicit split/merge operations
  while jointly learning a representation, albeit unsupervised:
  https://arxiv.org/abs/2203.14309
- Directional/vMF DML, probabilistic proxy DML, prototypical networks, and
  stochastic prototype embeddings occupy spherical or marginalized
  class-distribution scoring; NIR is already recorded in this repository:
  https://openaccess.thecvf.com/content/ICCV2021/papers/Scott_von_Mises-Fisher_Loss_An_Exploration_of_Embedding_Geometries_for_Supervised_Learning_ICCV_2021_paper.pdf
  and https://openreview.net/forum?id=rke2HRVYvH

The Bessel marginal likelihood changes the partition score, and the local
move/merge/split set changes the inference approximation. Neither changes what
supervision exists or what action it takes. This is exactly the distinction the
repository's stopping audit already closed under “global clustering is occupied
by facility-location DML.” Gate 2 fails even if no paper contains the literal
combination.

## Protocol and cost defects

The mandatory Gate-4 screen is corrected In-Shop. PEM assigns In-Shop `K=8`
without-replacement sampling, but the official training split has 25,882 images
over 3,997 identities—only 6.475 images per identity on average. The repository
sampler excludes every class with fewer than K examples. PEM would therefore
silently train on a filtered identity/image population rather than the corrected
full-corpus reference unless a new matched baseline and corpus accounting were
specified. It gives no such retained-class count.

Its forecast order instead targets five-seed CUB and Cars, where it predicts
small effects, and labels In-Shop a non-target below the reference. That is
incompatible with the registered In-Shop-first screen. The proposed nine
controls, K sweep, five seeds, and possible ten Cars seeds are also far more
than “1.02x training”; that figure describes one step, not the programme.

The primary CUB “crossing” depends on interpreting PFML's uncertainty one way
and uses an unreproduced recipe. If PFML's reported interval is SEM, PEM itself
calculates no crossing. Cars and SOP do not cross under its own arithmetic.

## Correct pieces worth preserving

- Integrating a uniform latent vMF direction gives the stated cluster evidence
  up to partition-independent factors.
- `phi'(r)=kappa A_d(kappa r)`, strict convexity, and the collapse penalty from
  merge hypotheses are mathematically sound.
- The proposal explicitly recognizes the quadratic small-argument regime,
  operational loss scale, PFML recipe uncertainty, and weak significance.
- Occam-off, moves-only, no-split, proxies-back, K-sweep, and wall-clock controls
  are useful diagnostics for any repaired structured-clustering experiment.
- Deployment legality is clean: one fixed 512-D descriptor and ordinary cosine
  retrieval.

These pieces do not establish repository provenance, a new supervision object,
an executable scalar path, isotropy, or an expected frontier crossing.

## Authorizing condition

There is none for frozen PEM. Expanding the lookup domain, defining a
differentiable kappa path or freezing kappa, repairing the schedule, replacing
the split claim, and supplying a mechanism-distinct supervision object would be
a substantive new proposal. The next protocol step is the one mandatory cold
review of this exact frozen artifact, not implementation or GPU work.
