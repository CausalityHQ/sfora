# Pass 180 — Common-Axis Minimax Whitening (CAMW)

## Frozen proposal

Given unnormalised descriptors `h_i`, estimate each class covariance `S_c` and
between-class covariance `B`. Find an orthogonal common-principal-component
basis `Q` by approximate joint diagonalisation of all `S_c` plus `B`. For each
axis `j`, compute `b_j=q_j^T B q_j` and the 90th-percentile class variance
`v_j=Q_.9,c(q_j^T S_c q_j)`. Set a clipped robust Fisher weight
`m_j=((b_j+eps_b)/(v_j+eps_v))^alpha` and deploy
`normalize(diag(sqrt(m)) Q^T h)`. The carrier Proxy Anchor objective is
unchanged; the train-time version periodically recomputes `Q,m` from a
cross-fitted descriptor snapshot and applies the folded 512-D projection.

An orthogonal rotation alone cannot change cosine retrieval; the diagonal
robust weighting is the operative mechanism. Deployment remains one 512-D
descriptor and cosine nearest-neighbour search.

## Gate 1 CPU falsifier

On saved Proxy Anchor train/test embeddings, fit `Q,m` from one deterministic
half of each training class and evaluate held-out training covariance
diagonalisation plus official unseen retrieval. Keep only the top-64 pooled-PCA
subspace for CPU speed and leave the complement at unit weight. Kill before GPU
if held-out off-diagonal reduction is `<20%`, split-half axis-weight Spearman
correlation is `<0.60`, or retrieval is not positive on all three datasets (or
median gain is `<0.5` points). This is a necessary post-hoc diagnostic, not a
benchmark result for the train-time method.

## Gate 2 prior-art boundary

Flury, *Common Principal Components in k Groups* (JASA 1984), establishes the
statistical common-axis factorisation. Radenović et al., * CNN Image Retrieval*
(ECCV 2016), establishes supervised whitening for image retrieval. L-GM,
non-isotropy regularization, and discriminant-analysis metric learning are
adjacent. CAMW remains **LIVE-NARROW** only as the specific combination of
class-covariance joint diagonalisation with a worst-class quantile variance
denominator; if this is judged merely a reparameterized supervised whitening
or Fisher metric, it dies at Gate 2. No GPU is authorized yet.

## CPU result

The deterministic 158-class corrected In-Shop proxy diagnostic gave baseline
R@1 `0.913701` and CAMW `0.894852` (`−1.885` points), held-out off-diagonal
reduction `−0.008801`, and split-half axis stability `0.625897`. CAMW fails
Gate 1 and is closed without GPU.
