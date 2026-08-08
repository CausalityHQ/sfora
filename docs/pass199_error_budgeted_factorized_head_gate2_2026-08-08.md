# Pass 199 — Error-budgeted factorized retrieval head

## Provenance

Corrected CUB HERD seed 0 has 1,813 top-1 failures split almost evenly between
local dispersion (873, 48.1%) and centroid overlap (940, 51.9%). Corrected In-Shop PA
seeds 0–3 instead have between/local failure ratios `3.05`, `3.11`, `3.22`, and
`3.15`, remaining `2.18:1` even in the strictest gallery-quality stratum.

The direct architecture response would divide one 512-D head into a locally
supervised pairwise-compactness subspace and a proxy/centroid-separation subspace,
allocate block width from the measured error ratio, concatenate them, and deploy the
single normalized descriptor.

## Verdict: DEAD at Gate 2

This is not an unoccupied mechanism. Divide-and-Conquer explicitly partitions one
deployed embedding into separately trained subspaces and concatenates them. Deep
Factorized Metric Learning factorizes network blocks and training signals with learned
routing. Multi-Head DML using Global and Local Representations is closer still: it
uses pairwise and proxy objectives in distinct local/global heads and concatenates the
retrieval descriptor.

The internal ledger also already closes error-conditioned local/global fusion
(Pass98), factorized routed embeddings (Pass127), partitioned subspace ensembles
(Pass179), backward routing (Pass197), and scalar pair/proxy adaptation. Setting block
width from a measured error ratio changes an allocation policy, not the architecture's
training object or data flow.

Primary collisions:

- Artem Sanakoyeu et al., *Divide and Conquer the Embedding Space for Metric
  Learning*, CVPR 2019,
  <https://openaccess.thecvf.com/content_CVPR_2019/papers/Sanakoyeu_Divide_and_Conquer_the_Embedding_Space_for_Metric_Learning_CVPR_2019_paper.pdf>.
- Wang et al., *Deep Factorized Metric Learning*, CVPR 2023,
  <https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.html>.
- Ebrahimpour et al., *Multi-Head Deep Metric Learning Using Global and Local
  Representations*, WACV 2022,
  <https://openaccess.thecvf.com/content/WACV2022/papers/Ebrahimpour_Multi-Head_Deep_Metric_Learning_Using_Global_and_Local_Representations_WACV_2022_paper.pdf>.

No implementation, CPU statistic, or GPU run is justified. The decomposition remains
useful evidence that the dominant editable error differs by dataset; it does not
supply a novel factorization operator.
