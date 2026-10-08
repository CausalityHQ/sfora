Primary-source screen, 2026-10-08. This is not an exhaustive SOTA survey or a protocol-matched baseline qualification.

| Reference | Dataset / split | Paper-reported Recall@1 | Decision |
|---|---|---:|---|
| [AdvRF, ICCV 2025](https://arxiv.org/html/2507.21742v1), Table 4 | SOP, official 60,502-image test | 84.2% | Below the existing 91.2% reference; no new floor. |
| [EviRank, August 2026 preprint](https://arxiv.org/html/2608.20886v1), Tables 4/15 | SOP, stated official 60,502-image test | 91.5% rounded; 91.46% in full table | Stronger reported quality lead; protocol and speed qualification unresolved. |

EviRank uses DINOv2 coarse retrieval, top-20 candidates, and Gemini-3-pro multimodal reranking for the reported pro result. This differs from a single packed-vector index. Its distilled variant reports 86.56% SOP Recall@1, not the pro result. The paper's Table 8 also differs from Table 4 in its AP value; do not silently merge those numbers. These are author-reported measurements, not independently reproduced results. No In-Shop result was identified in either paper.

Next: inspect evaluation code, query/gallery exclusions, training/test exposure, and complete request cost before declaring a matched reference or adopting a method. No new DGX experiment or scientific threshold is authorized by this screen.
