# Official SOP relational-linear evidence

This directory contains the sealed Stanford Online Products evaluation for the
generic Sfora relational-linear compressor. The fixed InShop recipe was reused
without SOP-result-driven tuning. Training consumed paired source/teacher
embeddings only; SOP labels and all test rows were evaluation-only.

## Frozen inputs

- Official SOP rows: 59,551 train and 60,502 test; 11,318 and 11,316 disjoint
  classes.
- UNICOM B/16 source archive SHA-256:
  `6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818`.
- UNICOM L/14@336 teacher archive SHA-256:
  `1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`.
- UNICOM revision: `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`.
- Exporter SHA-256:
  `8967844e48dc45bb5f0eff3692caa6079d40301e5e0905ffd17bf6f896cfec26`.
- Evaluator SHA-256:
  `0ebd8bd47f4ee9b1606d5ca06dbf799f8c95b2a9894d1ae1d06208c4e23f68cf`.
- Library SHA-256:
  `2a621a219c73801e54097891530e014462eb8ea66b665950f342952df2af84b5`.

## Retained outputs

- `sop-relational-linear-evaluation-v1.json`: 3,962,263 bytes, SHA-256
  `5125a8e0bfe242345257ca172ea62c92289d3585e7185e5b11d31bf92da2110c`.
- `sop-relational-linear-latency-v1.json`: 161,330 bytes, SHA-256
  `d18e6dedf55746f97271a320e1b7c9a87e498e147ea81c549b4a5229ee5a498a`.
- `sop-relational-linear-seed17.sfora-rl1`: 196,625 bytes, SHA-256
  `dad73cd0dd4662f3098899d862d5ef42dde0a2f2c81302cae3c08740f72ec100`.

All three relational seeds passed the preregistered quality gates. MAP@R gains
over PCA64 int8 were `0.015214`–`0.015350`; multiplicity-adjusted class-bootstrap
lower bounds were at least `0.013528` for MAP@R and `0.015234` for Recall@1.
The packed CPU p95 ratio was `1.002233` against the `1.10` gate. Persistent
storage remained exactly 66 bytes per item.

The receipts are claim-ineligible and do not establish absolute retrieval SOTA,
concurrent service latency, or generality beyond the two evaluated image domains.
