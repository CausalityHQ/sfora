# Pass 27 local evidence-aware audit: FRAME

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_frame_proposal_pass27_2026-08-06.md`  
Independent-review prompt: `docs/opus_frame_review_prompt_2026-08-06.txt`  
Review prompt SHA-256: `92ac4341cb9e2c4725126a4bc9af04d0398d439a186d08f10e943d51def06955`  
Durable independent review: `6eec8709c3204dc4` (running when this audit was frozen)

This audit was written without reading the independent review result.

## Verdict

**DEAD at Gate 1 and occupied at Gate 2; the frozen collapse defence is also
false.** No preregistration, implementation, or GPU is warranted.

FRAME is more distinct than the recent tail and EMA returns. Its executable
object is nevertheless a class-wise covariance-decorrelation/common-principal-
components target with an auxiliary variance-profile information band. The
repository has a prospectively measured result against the shared-frame premise,
and benchmark-adjacent prior work already occupies both halves.

## Gate 1: the repository measures against the shared-frame premise

FRAME assumes that within-class nuisance factors from disjoint identities share
a useful global coordinate system: class covariance matrices should have common
eigenvectors, while their diagonal strengths may vary. Candidate 225 measured
the minimum linear transfer property prospectively on corrected In-Shop
epoch-10 embeddings. A leading within-class subspace learned from disjoint
source identities captured **less** target within-class energy than target
between-class energy:

| seed | source-to-target `rho_32` |
|---|---:|
| 0 | 0.9312 |
| 1 | 0.9287 |
| 2 | 0.9345 |

All three are below the locked `1.15` falsifier. The directions captured about
35--37% of target within-class energy and 38--40% of target between-class
energy. This does not prove that covariance operators cannot be made commuting
by retraining, but it directly contradicts the claimed provenance that a
class-exogenous linear nuisance frame is already a transferable structure worth
enforcing.

Pass 22 CINA independently failed Gate 1 for the same missing premise: no
verified measurement connects per-identity covariance alignment, common shape,
or second-moment homogenization to corrected official-query errors. Pass 19 CFK
likewise records that the evidence packet withholds support for a shared
cross-class nuisance basis. FRAME supplies mathematical motivation and invented
forecast ordering, not a new positive repository measurement.

The proxy-span observation is not sufficient provenance. The Euclidean gradient
of a score-only proxy loss with respect to a normalized descriptor lies in the
proxy span, but the gradient with respect to the pre-normalized descriptor is
projected by `I-zz^T`, and shared network parameters couple all output
directions. A large algebraic complement when `C<d` does not measure harmful
“nuisance parking,” nor show that conditional covariance decorrelation repairs
unseen retrieval. Its predicted CUB/Cars versus SOP/In-Shop ordering is a theory
prior, not repository evidence.

## Gate 2: both supervision objects are occupied

Term 1 is the exact class-wise decorrelation target of Choi and Rhee's cw-CR
(AAAI 2019): penalize off-diagonal entries of class-conditional covariance in
one shared representation coordinate system. A proxy-cosine objective leaves a
global rotational gauge, but that changes the interpretation of the chosen
frame, not the supervised statistic. An unbiased four-sample estimator changes
how the same functional is estimated under DML batch constraints.

The wider statistical object is classical **common principal components** and
approximate joint diagonalization: a family of covariance operators sharing
eigenvectors, equivalently commuting in the exact symmetric case. FRAME trains
the encoder rather than solving for a post-hoc basis, but “make learned
class-conditional covariance operators jointly diagonalizable” remains that
occupied target used as representation pressure.

Term 2 is class-conditional variance-profile preservation with upper and lower
anti-homogenization bounds. Its components are occupied by:

- cw-VR/class-wise variance regularization;
- Kirchhof et al.'s benchmark-matched non-isotropic probabilistic Proxy Anchor,
  which learns per-class coordinate anisotropy in a shared 512-D frame on CUB,
  Cars, and SOP;
- NIR, which explicitly preserves class-local non-isotropic structure for
  unseen-class proxy DML;
- nonlinear ICA with auxiliary class variables and variance modulation; and
- DVML/SFT and common-covariance representation targets on the neighboring
  shared-variation axis.

The two-sided information band and tetrad estimator may be a new conjunction.
Under this protocol, a new wrapper around occupied class-wise decorrelation and
variance-profile supervision is not a new supervision source—especially when
its shared-frame causal premise is prospectively adverse.

Primary sources already recorded in the repository:

- cw-CR/cw-VR: <https://arxiv.org/abs/1809.09307>
- Non-isotropic probabilistic proxy DML: <https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136860423.pdf>
- NIR: <https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf>
- DVML: <https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html>

## Frozen mathematical and operational failures

1. **Exact within-class collapse is stationary.** At `z_1=...=z_4`, every
   contrast is zero. The quartic U-statistic and its gradient are zero. The
   variance estimates are squared pairwise differences, so their gradients are
   also zero at equality. Although `I=0` makes the lower-band scalar penalty
   positive, its gradient through the zero pairwise differences is zero. The
   claimed two independent collapse blocks therefore provide no first-order
   escape from exact collapse.
2. **Detached normalization does not make the executed objective scale
   invariant.** The current numerator is divided by a mixture of current
   detached energy and a lagged EMA. Scaling current residuals changes the
   numerator quartically while part of the denominator remains historical. The
   proposal admits a 30%-weighted transient, but that transient is exactly a
   loss-reducing collapse channel during the moving training regime. At exact
   collapse the `epsilon` branch makes the term zero.
3. **Unbiased numerator does not imply an unbiased normalized gradient.** `T_c`
   is unbiased for off-diagonal energy under iid four-tuples. Dividing its
   gradient by a stochastic, detached, state-correlated denominator and adapting
   the encoder to repeated samples does not yield an unbiased estimator of the
   normalized population functional. Individual negative kernels can therefore
   affect optimization beyond “variance only.”
4. **The mutual-information band is sampler-relative.** `I` uses only the 30
   classes in the current batch and `log P`, not the full training identity set.
   Private variance coordinates for classes not jointly sampled are invisible
   to one another. Its meaning and thresholds change with `P`, so the stated
   information budget is a batch functional rather than a dataset-level bound
   on identity shortcuts.
5. **The null calibration assumes independent chi-square variance estimates.**
   EMA entries are produced by changing encoders, repeated/nonuniform class
   visits, normalized coordinates, and correlated image samples. A scalar
   effective degrees-of-freedom approximation does not establish the claimed
   `0.008`-nat null or the fourfold separation from the lower bound.
6. **Dead dimensions need not violate the lower band.** The information is
   computed after each class profile is normalized across coordinates. A subset
   of live coordinates can carry enough class modulation to keep `I>=iota_-`
   while hundreds of coordinates remain identically dead. Dead coordinates
   dilute the profiles but do not force the bound to fail when the surviving
   coordinates become more modulated.
7. **Diagonal covariance does not uniquely solve unseen ranking leakage.** For
   a fixed coordinate-wise diagonal, removing off-diagonal entries minimizes
   the stated worst-case eigenvalue bound. Training does not hold that diagonal,
   trace, class mean, normalized-sphere tangent plane, or positive/negative
   covariance fixed. The network can trade diagonal mass and mean separation,
   so Proposition 4 is not a causal theorem for the executed objective.
8. **Commuting covariances are not sufficient for shared physical factors.**
   Diagonal class-code noise, sensor artifacts, background coordinates, and
   arbitrary label-dependent variance profiles all commute. The information
   ceiling limits but does not identify semantic pose/illumination factors.
9. **The estimator excludes sparse classes from its own claimed datasets.** SOP
   and In-Shop identities often have fewer than four distinct images. Excluding
   them changes the class distribution seen by FRAME and makes its asserted
   large-class null effect partly an estimator-selection artifact.

## Forecast and control audit

The sampler-matched PA and PFML rows are unmeasured forecasts under changed
200-epoch and PK-sampler recipes. FRAME itself forecasts no confident two-
dataset frontier crossing: `FRAME+PFML` is only +0.4 CUB and +0.2 Cars over the
published references, with admitted crossing probabilities 0.45 and 0.25. The
method's strongest forecast is a matched-baseline mechanism gain, but Gate 1
already lacks evidence for that mechanism.

C1 conditional isotropy is necessary, as are the term ablations and cw-CR
comparison. However the frozen controls never implement **cw-CR with the same
four-sample estimator and no information band**, nor a generic class-wise
variance-profile band without commutativity. C5 Term-1-only is close but its
forecasted superadditivity has no locked falsification margin. A post-hoc AJD
rotation being retrieval-invariant proves only cosine's gauge; it does not show
that FRAME's training gain comes from commutativity rather than generic
conditional decorrelation or higher-order gradient noise.

## Mechanism lesson

FRAME is a useful negative because it reached a genuinely different
mathematical area and made its assumptions explicit. The failure is not that
covariance geometry is uninteresting. It is that the project already measured
the linear shared-nuisance transfer premise adversely, while class-wise
decorrelation, common principal components, and non-isotropic proxy DML occupy
the executed targets. A future covariance proposal needs a new positive
repository measurement that survives disjoint identities and must change what
supervision exists, not only provide a clever unbiased estimator for an old
functional.
