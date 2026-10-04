# Compact ranking convergence — 2026-10-04

Production SOP + InShop quality and deployed speed goal remains unmet.

| Dataset / split | Matched control / candidate R1 + mAP@R | Cost | Public latency | Decision / next decisive test |
|---|---|---|---|---|
| InShop TRAIN-only observed selection, 1,734 queries / 1,715 gallery / 498 products | New compact-ranking seed179061 pair: **unmeasured**. Archived source061 96.30911188% / 80.57229534%; accepted concat 96.48212226% / 81.77754036%, verified historical panel receipts | Fresh matched128 control307.394s whole /46.17883895s core; candidate305.380s /46.99553962s. Ratios0.99344815 whole /1.01768560 core, verified <=1.50. Shared preparation283.636s separately attributed | Unmeasured for new pair | CPUv6 PASS238.533s. Control exportv3 FAIL174.047s: boundary denied current-process maps used by checkpoint mmap validation. Correct only that read permission, fresh qualify, export same retained pair, then frozen first selection CONTINUE/KILL |
| InShop sealed TRAIN validation, 1,749 queries /1,730 gallery /498 products | Unread | Not run | Unmeasured | Requires full four-endpoint selection GO; no validation selection |
| SOP official TEST / InShop official query-gallery | Current candidate unmeasured | Not qualified | No matched speed win | Official confirmation follows TRAIN selection; historical exploratory results do not establish current joint SOTA |

The original CPUv6 receipt and footer verify host peak1,576,599,552 bytes, no memory events, zero swap and quality_read=false. Original control exportv3 terminated exit1, host peak3,620,134,912 bytes, zero events/swap. This failure is engineering permission integrity, not a quality KILL or timeout. Candidate exportv3 is staged but unlaunched; retained training checkpoints are unchanged.

Fixed quality thresholds remain unchanged: first seed requires positive R1 delta/nonnegative AP delta; full matched seeds require both positive R1/nonnegative AP, equal-seed means >=0.002 each and positive product bootstrap lower bounds with original shared5,000 draws. No official quality or product speed claim follows from CPU fixtures, import success, or cached training cost.
