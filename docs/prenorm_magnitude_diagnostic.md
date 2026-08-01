# Pre-normalisation magnitude diagnostic

Date preregistered: 2026-08-01

Every saved artifact L2-normalises embeddings, so the project has never measured
the raw head-output magnitude `||f(x)||_2`. No DML checkpoint was retained, so
an operating-point measurement requires a short trained model rather than an
ImageNet-initialisation proxy.

Train the official In-Shop `proxy_anchor` recipe at seed 0 for exactly 10 epochs
and save pre-normalisation magnitudes for train, query, and gallery. No
best-over-training selection is used. Using ordinary normalised retrieval,
compute (1) point-biserial Pearson correlation between query magnitude and R@1
correctness, (2) Spearman correlation between query magnitude and retrieval
margin (best same-class gallery cosine minus best different-class cosine), and
(3) within-identity magnitude ICC on train.

Prediction: absolute correctness correlation is at least **0.15** or absolute
margin Spearman is at least **0.20**. Both absolute correlations below **0.05**
falsify magnitude as a material missing observable. Intermediate values are
descriptive and do not justify a method run.

Magnitude-dependent margins, uncertainty weighting, spherical constraints, and
quality-aware face losses are established (MagFace, AdaFace, SEC, IDML). A
positive diagnostic is therefore not Gate-2 novelty. It may advance only if a
distinct supervision or comparison operator is stated and audited first.

## Instrumentation correction (before adjudication)

The first epoch-10 export completed at R@1 **0.8421**, but its alleged raw query
norms had mean `0.99999998` and standard deviation `3.10e-8`. Inspection showed
that official BN-Inception normalises inside its own `forward`, so observing the
model return value did not observe the embedding head. This artifact is invalid:
its near-zero correlations do not test the preregistration.

The exporter was corrected to hook the output of BN-Inception's final embedding
linear layer before the model's internal L2 normalisation. A regression test uses
a wrapper that normalises inside `forward` and proves that inputs with raw norms
5 and 13 are exported as 5 and 13. The frozen thresholds, dataset, seed, epoch,
and analysis above are unchanged. Because the saved checkpoint was produced
before the corrected hook existed and there is no checkpoint-only export path,
the same seed-matched 10-epoch measurement will be repeated. The official
recipe does not enable deterministic CUDA kernels, so fixed-seed replay may
differ slightly; the repository already measures that effect. This is an
instrumentation repair, not a second draw or threshold change.

## Corrected result

The corrected seed-0 replay reached epoch-10 In-Shop R@1 **0.84365** (no
checkpoint selection). Its raw head-output query magnitudes had mean **430.80**
and standard deviation **60.77**, proving that the hook now observes the
pre-normalisation head rather than the unit-normalised model return. The frozen
analysis measured:

| statistic | result | registered threshold |
| --- | ---: | ---: |
| Pearson, query magnitude vs R@1 correctness | **0.18675** | predicted absolute value >= 0.15 |
| Spearman, query magnitude vs retrieval margin | **0.32574** | predicted absolute value >= 0.20 |
| train identity ICC of magnitude | **0.57754** | descriptive |

Both predictions pass, and both are far outside the joint falsifier of absolute
correlation below 0.05. Pre-normalisation magnitude is a material missing
observable: larger magnitude is associated with correct retrieval and with a
wider positive-versus-negative cosine margin, while more than half of its train
variance under the one-way ICC is identity-structured.

This positive diagnostic does **not** advance a method to GPU. Quality-aware
face losses (MagFace, AdaFace), uncertainty-aware metric learning (IDML), SEC,
ESA's norm/confidence-driven principal-direction augmentation, and IAA's
class-covariance synthetic support occupy the direct actions. Using magnitude
to weight, mine, gate, adjust a margin, select augmentation strength, or alter
similarity would be an estimator substitution inside those established
operators. A new action would still need independent Gate-2 novelty.
