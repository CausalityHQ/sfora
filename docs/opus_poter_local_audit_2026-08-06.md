# Pass 33 local evidence-aware audit: POTER

Date: 2026-08-06 UTC  
Frozen proposal: docs/opus_poter_proposal_pass33_2026-08-06.md  
Independent-review prompt: docs/opus_poter_review_prompt_2026-08-06.txt  
Review prompt SHA-256: 98b75758826475b1ff4def34a3d0d2b616c17cd20085a0cb59689aa0fb40790e  
Durable independent review: 889bda09dfc04177 (running when this audit was frozen)

This audit was written without reading the independent review result or
partials.

## Verdict

**DEAD at Gate 1 and Gate 2, independently reinforced by an invalid
within-class excess object, false gradient mechanism, and a forecast that does
not contain a standalone frontier method. No preregistration, implementation,
or GPU.**

## Gate 1: no measured provenance

POTER assumes that corrected zero-shot errors are caused by a transportable
two-level extreme law, that proxy-score tails extrapolate from training
classes to unseen gallery classes, and that three sampled images identify how
the within-class maximum scales to as many as 82 images. No repository
measurement establishes any of these premises. Nothing saved here measures
GPD goodness of fit or threshold stability, tail transport across disjoint
identities, the joint proxy-score/excess law, or causal repair of official
query errors.

This is not a fresh unmeasured direction. Passes 9, 14, 15, 23, 26, and 31
already proposed the same gallery-depth argument through EVPC, RLM,
EGR-PFML, PORTAL, PORT, and XTail. Pass 31's independent calculations found
that a bulk-threshold GPD shape added only 0.0087 incremental R-squared beyond
simple tail summaries, while a constant coefficient predicted deep quantiles
better and live descent worsened the negative 75th percentile. POTER adds
another extrapolated layer but supplies no new prospective measurement that
reverses those findings.

The table of class counts and inverted R@1 probabilities is arithmetic, not
provenance. It cannot identify whether gallery multiplicity, representation
quality, positive variation, label noise, or class mixture causes a miss.

## The second-level excess is not the object claimed

POTER defines

    e_ic = max_{j:y_j=c} <z_i,z_j> - <z_i,p_c>

and asserts e_ic>=0. That inequality is false. A learned proxy may have higher
similarity to a query than every one of the three currently sampled class
images, so e_ic can be negative. The pooled empirical law is therefore not a
nonnegative peaks-over-threshold excess distribution.

Even when positive, e_ic mixes at least four changing objects: query
direction, class proxy error, which three images were sampled, and the
class-conditional image law. Pooling it over queries and classes does not
produce one exchangeable within-class distribution. Proxy score m_ic and
excess e_ic share the same query and class and are mechanically dependent,
while the proposal's factorization treats their marginal laws as separable.

The formal transform Q_n(p)=Q_K(p^(K/n)) is correct only if Q_K is the exact
quantile law of the maximum of K iid draws from one stationary base law.
POTER supplies a heterogeneous pooled law of max similarity minus a learned
proxy score, with duplicated and correlated product photos. Three draws cannot
validate the iid block-max premise. The positive correction invokes Q_(K-1)
without defining how its sample population is constructed; the earlier Q_K is
defined only for negative classes relative to their proxies. Thus the
positive-side term is not executable as frozen.

Using one mean images-per-class value further cannot represent a gallery whose
class sizes vary widely. Jensen effects make a maximum under the size
distribution different from a maximum at the mean size.

## The claimed dimension-expansion gradient is impossible on its stated path

The top-proxy scores are m_ic=z_i^T p_c. Every derivative of u_i, the top-k
exceedances, and sigma_i with respect to z_i is a linear combination of
proxies, hence lies in span(P). Reducing fitted proxy-score scale therefore
cannot “supply the missing gradient” in the proxy-orthogonal head directions;
it has exactly the same span restriction the proposal attributes to Proxy
Anchor. Sample-sample excess gradients may leave that span, but those are the
separate, invalid second-level term and do not prove D2.

The sign of dL/dsigma is also conditional on the threshold being above the
tail threshold and on support/clamp branches. It is not globally positive as
stated. Optimizing proxies can flatten their top scores without increasing
deployed descriptor dimension, giving an auxiliary-only shortcut analogous to
Pass 28 DOIR.

The collapse proof again confuses loss value with descriptor gradient. At
identical normalized descriptors and proxies, tangent cosine derivatives
vanish. The top-k spread collapses into ties, PWM denominators degenerate, and
hard order-statistic membership is undefined. Substituting sigma_min and
observing a high scalar loss does not show a first-order escape direction.

## Gate 2: seventh internal recurrence and occupied public object

POTER is the seventh repository recurrence of training-to-gallery tail
extrapolation. Its first level is XTail with all-class proxy scores instead of
sample negatives; its second level block-maxes sample-minus-proxy residuals.
That extra estimator does not change the supervision referent: penalize a
parametrically extrapolated extreme negative event at deployment scale.

Publicly, WEINCE, TriSim, Recall@k surrogates, top-k/CVaR and ranked-list
objectives occupy tail/rank training pressure. LDReg differentiates an
in-batch per-sample EVT/LID tail statistic into the representation. AnchorFace
and memory-bank methods address operating depth empirically; OpenMax, EVM, and
EVT similarity-search relevance prediction use the non-match upper-tail
object post hoc. A hierarchical wrapper and a train-count offset may be an
unpublished estimator, but do not create a new supervision object.

Primary sources already audited in this repository:

- WEINCE: https://arxiv.org/abs/2606.00262
- TriSim: https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf
- LDReg: https://arxiv.org/abs/2401.10474
- Recall@k surrogate: https://arxiv.org/abs/2108.11179
- AnchorFace: https://ojs.aaai.org/index.php/AAAI/article/view/20063

## Recipe, controls, and standing objective

P=40,K=3 requires replacement for sparse SOP/In-Shop identities, but the
proposal does not define it. Duplicate images create zero within-class
differences and alter the very block-max law being extrapolated. AMP fp16 is
also unsafe for cancellation-heavy PWM denominators unless the score matrix
and estimator are explicitly promoted to fp32.

C7 repeatedly evaluates fractions of the test gallery to validate mechanism.
That may be a report-only diagnostic after freezing, but it violates the
project's one-test-touch discipline if used to judge or iterate the method.
Class-disjoint training validation can perform an analogous gallery ladder
without touching test identities.

Most decisively, the frozen standalone PA+POTER method crosses no supplied
frontier. The only claimed crossing is PA+DADA+POTER on In-Shop, constructed by
assuming 60-percent additivity on top of a separately published method.
Neither the composite implementation nor matched DADA reproduction exists.
Even if every forecast landed exactly, the frozen standalone method would not
fulfill the standing objective.

## Mechanism lesson

Adding levels to an unvalidated extreme-value model does not create
provenance. Each level adds exchangeability, independence, and support
assumptions while retaining the same occupied supervision referent. A future
tail proposal must be rejected immediately unless a preregistered repository
measurement first demonstrates transport of the exact tail object and the
method's own standalone forecast can cross a matched frontier.
