# Prospective mechanics runtime amendment — approval pending

The corrected native Large mechanics procedure remains NO-GO under its frozen whole-unit 120-second limit. Original v4 invocation `9e61f1c141a84319aaf6a3ec7f721415` terminated at 120.738 seconds after 17 uninterrupted updates and two exactly matching replay rows. Independent step8 restoration took 21.181 seconds and completed at 114.421 seconds. The complete nine-row replay, second independent final reload/raw-packed calibration, and full uncached exit rehash did not finish.

Measured terminal peak was 6,390,661,120 bytes, all memory events zero and swap zero. The prior mapped-archive retention failure is corrected for the observed restore. These observations do not qualify whole mechanics or model quality.

The proposed change is **only** a prospective 300-second whole-unit cap for mechanics qualification, retaining:

- 8-GiB host memory, zero swap/events, CUDA allocation below 10 GB, both resource locks and complete-unit peaks.
- Every source/authority/CPU admission and serialized-state hash, independent model/optimizer/RNG restoration, exact 17 versus 8+9 replay, final whole/raw-packed reload and complete uncached exit guard.
- Discarded mechanics state, four fresh TRAIN100 endpoints only after both arms qualify, unchanged quality confidence and training-cost gates.
- Existing CPU120/FIT300/TRAIN300/held-export300 caps and the full SOP/InShop quality-and-speed objective.

Implementation would change mechanics `policy()` and its original terminal-duration admission together, update the narrow frozen-contract test pins, then freeze NEW immutable source/launch authorities. It would run one fresh Large mechanics job; So400 remains gated on a full Large pass. No failed v1–v4 invocation, receipt or state would be rescued or reclassified. Runtime beyond 120 seconds would be reported as measured preparation cost, not a training speed win.

No amendment has been applied or launched. The operator's existing instruction to keep the 120s/300s execution caps requires an explicit decision before this prospective change. If declined, the fixed procedure remains closed at its reproducible runtime blocker and the broader product goal stays open.

The exact proposed patch is preserved as `docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-runtime-amendment-proposed-v1.patch.txt` (SHA256 `4a0ecc62115a4af92f5eb599da05fa4d5e62325527bcd8aa0704360043b9c438`). It changes two production lines: mechanics policy duration and mechanics terminal-duration admission. Three test lines update the whole-contract AST pin, expected mechanics duration, and malformed resource-policy negative (301 seconds, since mechanics/train would share 300). The isolated proposal passed the full narrow stdlib check (original session6161 exit0); an independent AST diff confirms only policy/admit_mechanics production definitions change. Active production source remains unchanged, approval pending, native unrun.

## Authorization and prospective v5 boundary

Operator authorization delivered by devbox/main message immediate-1790882142067309118-1991058 permits ONE new Large engineering-only mechanics qualification up to300s. The reviewed patch is applied prospectively; all v1–v4 failures retain their verdicts. Root owns this trivial integration and DGX gate; no additional specialist is needed for the already-reviewed two-line correction. No TRAIN/quality read until complete17 plus independent8+9 and strict final reload pass. Any integrity/resource/parity/runtime failure kills this arm; no retry or cap escalation.

Frozen source root `/home/riomus/runs/sfora-native256-adaptation-source-v5`, execution SHA `d00d7d70ac2af4acb654d2054e28ffd6f0a0c286ad00d86d02869cfa5534b326`, Large authority SHA `5065f464218827ee96aca444af4e08ec8102cc725ae0fa0b5c251db917283fb1`. Original initialized CPU descriptors and all remaining inputs/limits unchanged. Output is new `/home/riomus/runs/sfora-native256-mechanics-large-v5`.

### v5 terminal outcome

Original session10687/invocatione3c02b979367415f8985c2eb69117440 exited0 successfully in189.303s whole service (driver187.290s). Full17 versus serialized8+independent9 and strict final whole/raw-packed reload, uncached exit rehash PASS; state discarded/no quality read. Driver host peak6,548,062,208B/events0/swap0, CUDA6,128,466,432B; median uninterrupted update2.085331s. Raw log and compact pinned receipt are committed beside prior failure evidence. Engineering mechanics GO only; product quality and speed remain unqualified. No second engineering job or TRAIN started under this one-job authorization.

### Prospective sequential So400 v5 gate

Authorization immediate-1790882776853633660-1991058 permits ONE sequential So400 mechanics-only300s gate after Large terminal authentication. Same exact v5 source execution; So400 authority SHA371cfbaa518b9b5df33934087f037bd9cac0d125635d171b0574ad3ba34fb3d7 binds unchanged original fullCPU proof. New output `/home/riomus/runs/sfora-native256-mechanics-so400-v5`. DGX compute/services idle and both nonblocking resource locks checked before launch. Full17+independent8+9/strict final reload/exit and all resource limits remain mandatory; any failure kills this arm. Report paired terminal decision before any freshTRAIN100 launch. No prior failure is reclassified.

### Paired mechanics decision

So400 original23754/invocationc9833e1754c84f5aaa3a69211d82ba5f normalexit0/service215.643s: complete17 versus8+9, strict independent final whole/raw-packed reload and full uncached exitPASS; state discarded/no quality. Terminal hostpeak7335288832B/events0/swap0, CUDA7286658048B. Both v5 mechanics arms are engineering GO under the prospective300s envelope; all v1–v4 failures remain unchanged. FreshTRAIN100 and held-quality remain unrun; this paired decision is reported before launching them.

### Fresh TRAIN100 terminal runtime NO-GO

First Large179032 original81064/invocation13508b928bbd47f2b7fd142052664873 timed out300.465s. All100 updates finished249.884s; strict final reload ended275.206s; full uncached exit rehash began275.346s but did not finish. Actual terminal peak6,476,185,600B/events0/swap0. No accepted training receipt or quality read. Remaining three endpoints are stopped before launch; partial checkpoint is unqualified and must not feed held scoring. This is a whole-unit engineering runtime failure, not model-quality evidence or an architecture KILL. No cap change/rescue. Next source-only correction must address demonstrated full-lifecycle integrity-read cost while preserving complete byte checks and300s cap.

### Prospective v6 source correction and requalification

Integratede4fd36c4 changes only trainer exit traversal: one fresh complete guard/origin byte pass, original FIT resolutions/aliases/import-origin/closure/error predicates retained. Original narrow stdlib55s checkPASS; parent exact name-keyed AST audit proves all native math/state/update/reload/caps unchanged. New root `/home/riomus/runs/sfora-native256-adaptation-source-v6`, exact5 execution `236327289f110ed2bf2012ea6e0d1ea48cf822b4e4094d9cf544081a2dc476f7`; original initializedCPU/FIT/PCA descriptors unchanged. One fresh Large mechanics300 gate precedes So400 and freshTRAIN; no failedstate reuse. All original resource/parity/discard checks mandatory; a failed gate stops this arm. Duplicate-read removal is verified by source/stdlib byte accounting, native elapsed saving remains unmeasured. Historicalv5mechanicsPASS and TRAINtimeout are preserved, never reclassified.

Largev6 original42130/5df7fca0d23748769bb5fa77fbf58085 terminalexit0/service170.814s, completeparity/reload/exitPASS. Exitrehash12.470384s versusv5 32.737546s (separate observedengineeringruns, notmatchedproductlatency). Terminalhost6540640256B/events0/swap0/CUDA6128466432B. State discarded/noquality; originalvalidCPU reused. So400v6requalification next underidenticallimits, freshTRAINonlyafterbothPASS.

So400v6 original26245/d7c9349d1d4b437aa001c4fdca40ac5f normalexit0/service194.051s, completeparity/reload/exitPASS, host7347122176B/events0/swap0/CUDA7286658048B. Exitrehash18.050490s versusv5 33.741376s, separateobservedengineeringruns. BOTHv6mechanicsqualified; state discarded/noquality. Reportpaireddecision thenfreshTRAIN100underunchanged300s; cap failure stopsremainingarms, noretroactivefailedstateadmission.

Freshv6TRAIN100 firstLarge032 original65330/1514237b262b4bb687f218b003d68859 runtimeNO-GO/timeout300.716s;100updates215.430404s,strictfinalreloadend291.897s,exitrehashbegin292.057unfinished. Terminalhost6,474,788,864B/events0/swap0, noacceptedreceipt/noquality. Remaining3endpointsnotlaunched; incompletecheckpointunqualified. v6exitcorrectionmechanicsgainremainsvalidbutwholeTRAINfitfails. No capextension/rescue/architecturequalityKILL. Nextboundedengineeringaxis exactcomplete-state fingerprint throughput, retaining identicaltypedSHA/nativecomputation/resources; no speculative runtimegainclaim.

### Prospective exact-digest fingerprint candidate

Read-only specialist `a0cbcce6694c4299` completed normally at source `97838823` (441 seconds). It supports implementing one bounded candidate, not a native runtime GO. Derived live-state byte counts are 1,993,693,488 bytes per Large update and 2,406,243,312 per So400 update. No SHA, transfer or synchronization timing partition was measured; these competing costs remain hypotheses.

The candidate uses two stdlib SHA256 workers, at most two outstanding tensor jobs and 64 MiB of aggregate staged buffers, including the next/current staging allocation. Only caller-owned CPU staging is permitted. Oversized tensors use the original serial path. Live cached-prefix fingerprints retain exact typed framing, leaf digests, traversal order, occurrence coverage and version audits. Mmap consumed-callback fingerprints remain serial with original hash-before-consume ordering. No science, precision, state, cap or integrity requirement changes.

Before native launch, root must verify the two-file source change and independent original-serializer parity checks, then freeze a new immutable code/authority closure. In that prospective mechanics gate, compare serial and candidate fingerprints on the same stationary payload at uninterrupted steps 3, 9 and 17, alternating execution order. Exact digest equality is mandatory. Median end-to-end saving must be at least 0.10 seconds per call; smaller savings, parity failure or resource violation kills this engineering candidate. Worker SHA duration alone is insufficient. Separate bounded diagnostic records must preserve update-row/replay schemas and add no synchronization solely for timing.

Complete Large mechanics precedes So400 under unchanged 300s/8GiB/no-swap/zero-events/<10GB CUDA/both-lock limits, including full 17 versus independent 8+9, strict final reload and uncached exit. Both must pass before any fresh TRAIN100. No native candidate has been launched, no failed checkpoint is reused, and no paired TRAIN fit or quality gain is claimed. This protocol decision stays root-owned; source implementation is delegated separately from the existing held-export worker.

Held-export/scoring source is integrated from worker `680b0ebb` as `2901d650`: original trainer5/future held10 authority, four accepted TRAIN100 endpoints, inference-only complete reload and unchanged paired quality/cost gates. Root stdlib tamper check passed; native parity, resource fit and quality remain unmeasured. Initial prospective limits are FIT CPU120, held export300 and final CUDA-hidden CPU score300 seconds, all with original memory/lock/integrity constraints. No native scorer attempt was changed or rescued.

| Dataset/split and comparison | Matched R1 / mAP@R | Verified training cost | Public latency | Remaining gap and next decisive gate |
| --- | --- | --- | --- | --- |
| InShop TRAIN-held: 6,354 queries / 6,245 gallery / 1,993 products; fresh native256 Large vs So400, FIT 13,283 images / 2,004 products | Both unmeasured; no accepted four-endpoint comparison | Failed Large seed179032 v6: 100 updates 215.430s, whole service timeout300.716s; So400 TRAIN unrun | Unmeasured for this comparison | Quality and matched speed unresolved. Qualify exact-digest fingerprint savings plus full mechanics, then fresh TRAIN100; only accepted endpoints admit held scoring. |

### Frozen prospective v7 mechanics boundary

Worker `8e4d5202` is integrated as `37ae7188`; its original final stdlib session86323 exited0 under55s. Parent source audit confirms byte-identical original serial fingerprint and unchanged native math/state/RNG/restore predicates/caps, except the exact reviewed diagnostic/screen inserts. Released memoryviews resolve the independently reproduced executor-retention race. These checks do not establish native saving or resource fit.

NEW immutable root `/home/riomus/runs/sfora-native256-adaptation-source-v7`, exact5 execution SHA256 `5199f5f9791489760ca05d847dd8350a0768bf163234dc0f175d58d1c7dd0460`. Large mechanics authority SHA256 `907de55cc3f72cd1b1b668ff1163e89fddd56fd0f315480e99a7f0670e9e88c1`; So400 `92b3fa255947e79163e80e14d0294ef19dcda223f8ff7294a920d4eb9f9a955e`. All original initializedCPU/FIT/PCA descriptors and the three reference/helper source hashes remain unchanged. First launch is ONE Large mechanics gate after idle/locks admission. Any saving/parity/integrity/resource/runtime failure stops this candidate; no So400 or TRAIN before the required preceding complete pass. Historical failures and unqualified checkpoints remain unchanged and ineligible. No native v7 measurement exists at this freeze.

### Original v7 terminal performance KILL

Original session73430/unit `sfora-native256-mechanics-large-v7`/invocation `b75becf0a16d43d9bf5fa0aa5a86c205` exited1 normally at157.788s at the declared saving screen. All17 uninterrupted updates completed; same-payload serial/candidate digests agreed at steps3/9/17. Serial times were1.130134/1.217865/1.163439s; candidate times4.203779/4.006806/4.105853s. Median saving was **-2.942413s**, failing the required +0.10s. Thus this bounded parallel SHA candidate is KILL; it is not a model-quality or universal architecture verdict.

Final candidate diagnostic measured3.781213s in caller staging,0.882304s summed worker SHA durations and0.295838s queue wait within4.105781s wall time. Worker sum is not elapsed time. Peak staged buffers33,885,720B/two outstanding jobs respected the bound. Whole-unit hostpeak6,286,626,816B/events0/swap0. CUDA ceiling was enforced during the completed updates; no final numeric CUDA peak receipt exists. The8+9 replay, final strict reload and uncached exit were not reached, so mechanics is unqualified. No accepted receipt, So400, TRAIN or quality read exists. Original raw log and compact terminal evidence are preserved; DGX is idle. No unchanged retry or cap change is permitted. Next intervention must address measured live snapshot staging cost while preserving exact digest/full-state checks and a usable TRAIN100 path.
