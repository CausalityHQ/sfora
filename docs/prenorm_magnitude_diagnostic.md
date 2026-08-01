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
