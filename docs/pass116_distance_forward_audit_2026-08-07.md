# Pass 116 — Local distance-forward Proxy Anchor (DEAD at Gate 2)

## Candidate

Replace end-to-end backpropagation in the BN-Inception/Proxy Anchor lane with a
layer-local distance-forward update: each block receives same-class and
different-class descriptor pairs and optimizes a local distance goodness, while
the final block is trained to emit the fixed 512-D cosine descriptor. The idea
was inspired by biological/local-energy learning and would have been a genuine
algorithmic departure from ordinary backpropagation.

## Gate 2 prior-art audit

The direction is already occupied before any repository implementation. The
primary-source **Distance-Forward** paper, *Advancing the forward-forward
algorithm towards high-performance deep local learning* (Neural Networks 2026,
doi:10.1016/j.neunet.2026.108765), explicitly “reformulate[s] FF using distance
metric learning” and proposes a distance-forward algorithm. HCL-FF (CVPR 2026,
https://openaccess.thecvf.com/content/CVPR2026/html/Yao_HCL-FF_Hierarchical_and_Contrastive_Learning_for_Forward-Forward_Algorithm_CVPR2026_paper.html)
also combines layer-local goodness with supervised contrastive alignment.
Applying either to Proxy Anchor, changing the backbone, or using a cosine
instead of Euclidean local distance changes the benchmark implementation, not
the mechanism-level novelty.

The candidate is **DEAD at Gate 2**. No GPU or CPU screen is authorized. This
search result is nevertheless useful: biological/local training is no longer a
novel escape class for this project unless the new object differs from both
distance-forward and supervised local-contrastive learning.
