# Raw patch correspondence screen

One hypothesis: global pooling and fixed quadrant averaging discard detailed
garment correspondences whose locations move between photographs. Test the
already trained, unprojected 256 x 1,024 final tokens with **symmetric mean
nearest-token cosine**: mean row maximum and mean column maximum, averaged.
No learned alignment, positional constraint, head projection, coefficient
search or preprocessing change. This is a training-supervision candidate,
not a proposed gallery format or a novelty claim.

The existing shared-head four-quadrant screen failed (50/151 rescues); this
changes representation granularity and matching before the head. The token
mean blend also failed, but did not retain local correspondences. The source
centroid configuration is closed: its useful-fit-direction screen did not
produce sufficiently strong encoder gradients. Do not reopen any of these
fixed configurations.

Run one read-only DGX screen on the archived 151 previously observed official
TRAIN-held miss triples (409 distinct images), pinned seed-179026 checkpoint
and original 256-pixel processor. Original packed triple margins must still
be negative and reproduce within 0.005 before accepting token results. The
baseline is the unchanged deployed global head, not a newly selected model.
Keep the exact archived checkpoint, miss, split, role and model hashes from
the existing spatial-part probe. No official query/gallery labels or new
quality selection panel is read.

Freeze GO at **at least 99/151 positive-over-impostor token wins AND median
token margin at least +0.02**, matching the earlier local-detail screen's
headroom bar. Whole process wall must be at most **60 seconds** and peak CUDA
allocation at most **4 GB**. Any source/replay/numerical/cost/quality failure
kills this exact raw-token matching configuration before training or a full
gallery export. A pass needs matched hit controls and a TRAIN-only bounded
gradient/cost gate before any learned objective. Triple rescue is not full
gallery Recall@1, does not count false-hit losses, and cannot prove SOTA.

Prior art: [MGA](https://arxiv.org/pdf/2302.08902) uses learned multiscale
local aggregation and cross-image attention; Table 1 reports 94.3% In-Shop
R@1, so it does not replace the dated UNICOM 96.7% reference. Its local-detail
motivation supports testing correspondence, not assuming the present score
works. A separate [2025 ESA paper](https://scholarworks.bwise.kr/cau/bitstream/2019.sw.cau/89458/1/Rethinking%20Metric%20Learning%20Enhancing%20Generalization%20to%20Unseen%20Classes.pdf)
focuses on unnormalized feature-radius shifts and virtual classes. Sfora's
source/head/classifier cosine normalization removes that raw-radius channel;
do not adopt its radius explanation without a distinct angular mechanism.
These primary-source checks do not certify the latest published frontier.

## Terminal decision: KILL

The original DGX Spark unit `sfora-inshop-raw-token-match-v1`, invocation
`4462068e430e49dcad0f59610a380e22`, exited 0 and GPU inspection is idle.
Only this one export ran; no training or full-gallery expansion followed.

| Official TRAIN previously observed miss panel | Result | Gate |
|---|---:|---|
| Raw-token positive-over-impostor triples | 53/151 (35.10%) vs global-head baseline 0/151 by panel construction | KILL: below 99/151 |
| Median raw-token margin | -0.0211086 | KILL: below +0.02 |
| Original packed maximum margin replay error | 0.00236139; every original triple still a miss | Authority passes |
| 409-image export and scoring wall | 7.423162 s | Resource check, not serving latency |
| Main-function wall, including authority/setup | 13.138305 s | External process timeout also enforced at 60 s |
| Peak allocated CUDA | 2,158,002,688 bytes | Below 4 GB |
| R@1, mAP@R, training cost, image-to-top-k p50/p95/p99/QPS | Not measured | No new claim |

Receipt `inshop-raw-token-match-v1/receipt.json` SHA-256
`ca96102c7f43c75da8a656af65d01086bcf6acf899e2b494ef2b637ea5bc7f05`.
Independent replay checked all 151 margins, count, median, numerical/resource
guards and decision. The raw field `whole_process_wall_seconds` actually
starts inside `main` after imports; it must be reported as main-function wall.
The external timeout governs the full child, and the preserved unit journal
records normal closure within that cap.

The first source inventory incorrectly resolved decorated `export_all` to
Torch's decorator file. The supplemental `export-helper-audit.json` resolves
the underlying archived module with `inspect.unwrap`, SHA-256
`ebc0986f112eb8ba72943595ac6ef93f5328c0bed105c2990cb8c70d7b3b0495`.
The source inventory now unwraps decorated helpers; this metadata fix did not
rerun or alter the original numerical experiment. The original script SHA
and receipt remain intact. Raw local nearest matching provides insufficient
headroom for this fixed route. Do not train it, change matching/pooling knobs
on this panel, or call these rescues full-gallery quality. Production package
and serving bytes are unchanged; the joint target remains unmet.
