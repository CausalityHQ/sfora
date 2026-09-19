# Stanford Dogs CLD falsifier evidence

`result-v1.json` is the unmodified canonical receipt from the one-shot fresh
gate preregistered in
`docs/stanford_dogs_cld_preregistration_2026-09-19.md`.

The gate failed. On 4,534 class-disjoint evaluation rows, normalized DBA scored
`0.6585023624` mAP@R, while the frozen training-covariance CLD+DBA scorer scored
`0.6518076965` (`-0.0066946659`). The global-prior and global-WCCN controls
scored `0.6660943763` and `0.6680092382`. Protected R@1/R@10/R@100 membership
was exactly preserved at `0.8767093075`, `0.9726510807`, and `0.9969122188`.

This result falsifies fixed ranks 41--128 as a generic negative set. Stanford
Dogs is development data after this receipt and cannot confirm a revised
scorer. The receipt SHA-256 is
`1398e7d99a96be27a119ac5cde1eb06a34b8ea3bf6bfe892da944255119481ea`.
