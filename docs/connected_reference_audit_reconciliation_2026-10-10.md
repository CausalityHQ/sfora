# Published-reference audit reconciliation — 10 October 2026

The original Fable consultation `c200cf839c814f68` completed normally (exit 0, 455 seconds, no fallback). It supplied a bounded literature/protocol review, not a reproduced benchmark or an exhaustive current-frontier certificate. No model, image, embedding, official evaluation or GPU experiment ran in that consultation.

Retain the dated published UNICOM R@1 screens: SOP **91.2%**, In-Shop **96.7%**, [arXiv2304.05884v1, Table4](https://arxiv.org/html/2304.05884v1). The root independently checked those table values. Backbone, pretraining, image processing, representation and scoring still differ from Sfora; matching the official split alone does not make a paired system comparison. Existing newer-paper access gaps in [the qualification ledger](published_reference_qualification_2026-09-28.md) remain open. The consultation's stronger frontier wording is not accepted as proof.

The review raises released-code TEST-max selection and preprocessing concerns. Fresh root fetches of the cited GitHub code failed; those code-specific claims remain consultation findings, not newly verified root facts. They cannot justify lowering either quality target. The archived local UNICOM In-Shop reproduction and Sfora's previously observed TEST history remain disclosed limitations, not replacement targets.

| Proposed decision | Root disposition | Next evidence required |
| --- | --- | --- |
| Lower the In-Shop screen to a weaker local reproduction | Reject | Surpass the defensible published protocol-matched target and disclose selection history. |
| Substitute a byte-matched embedding-only comparison for the production goal | Reject as completion | Useful auxiliary evidence cannot replace deployed image-to-top-k quality and matched full-pipeline speed. |
| Start another UNICOM training/reproduction run | Not selected or launched | Current installed-serving parity and the frozen TRAIN-only intervention keep priority; no duplicate reference experiment. |
| Claim current global SOTA from the bounded screen | Reject | Close primary-source access/protocol gaps and obtain independent confirmation. |

The historical SOP three-seed R@1 **91.7419%** and In-Shop **95.4823%** remain exploratory official results. The newer SO400 artifact is distinct and has no official result; do not relabel the historical backbone as the current artifact. Neither the In-Shop quality target nor matched end-to-end speed target is met. See the [verified decision table](sfora_results_checkpoint_2026-10-09.md).

The next serving gate remains one sequential original/installed control comparison on the exact B1/B2/B32 groups, complete typed outputs and native top-10. It is an engineering parity gate; passing it would authorize subsequent matched latency measurement, not a SOTA or p99 claim.
