# SFORA 0.3.0rc3 release evidence

SFORA 0.3.0rc3 packages a fit-only compact-metric selector, its deterministic
PCA fallback, authenticated encoder persistence, and an optional persistent
cuTile exact scorer. The release has two deliberately separate status axes:

- **release-ready** means the library/API/package/system boundary passed its
  stated verification;
- **claim-eligible** means a scientific result may support a prospective
  publication claim.

A release-ready component can be backed by claim-ineligible panel evidence.
The latter remains useful for product and engineering decisions, but it must not
be relabelled as a prospective or universal SOTA result.

## One-table checkpoint

All numbers below are measured and receipt-backed. A dash means the quantity
was not measured or was not retained; it is not an estimate. Quality uses
mAP@R / Recall@1. `128 / 130 served` means 128 signed code bytes plus the
two-byte inverse norm used by the persistent exact scorer.

| Evidence | Dataset and split | Candidate vs matched baseline | Candidate quality | Baseline quality | Candidate minus baseline | Paired 95% mAP interval | Persistent bytes/item | Measured serving | Status and limit |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| prospective selector confirmation | Oxford-IIIT Pet, official trainval/test images split into 19 fit / 18 unseen classes by frozen SHA order | learned int8-128 vs PCA int8-128 | `0.862759 / 0.969595` | `0.847639 / 0.962275` | `+0.015121 / +0.007320` | fit-only selector: `[+0.005856, +0.016646]`; external delta CI not retained | `128 / 130 served` | see 1M row below | verified prospective selector pass; receipt remains `claim_eligible=false`, and this is not the standard same-class Pet protocol |
| standard equal-byte panel | Cars196, standard class-disjoint split, 8,054 fit / 8,131 evaluation | learned int8-128 vs strongest equal-byte Faiss OPQ mAP arm | `0.825656 / 0.972574` | `0.817630 / 0.973066` | `+0.008026 / -0.000492` | fit-only learned-vs-PCA selector: `[+0.023754, +0.043083]`; OPQ comparison CI not retained | `128 / 130 served` | see 1M row below | higher matched mAP per byte, but not Recall@1 dominance; historically exposed panel |
| frozen-width null | CUB-200-2011, same-teacher development split | learned256-int4 vs learned128-int8 | `0.720194 / 0.908645` | `0.720486 / 0.905495` | `-0.000292 / +0.003150` | `[-0.001432, +0.000832]` | `128` for either arm | — | width effect is null; production selector separately chose PCA and its external learned-minus-PCA mAP delta was `-0.000323` |
| official query/gallery panel | DeepFashion In-Shop, official identity-disjoint query/gallery, rank-finished UNICOM source | learned int8-128 vs matched-input float-768 source | `0.800020 / 0.954283` | `0.779547 / 0.945703` | `+0.020473 / +0.008580` | not retained | `128 / 130 served` vs `3,072` | see 1M row below | verified post-hoc composition; `claim_eligible=false` |
| official large-scale panel | Stanford Online Products, official 59,551-train / 60,502-test split | selected learned int8-128 vs frozen float-768 source | `0.511490 / 0.770173` | `0.476360 / 0.745099` | `+0.035130 / +0.025074` | not retained | `128 / 130 served` vs `3,072` | see 1M row below | verified standard panel; test was previously observed, so `claim_eligible=false` |
| exact packed serving | deterministic synthetic one-million-row gallery on NVIDIA GB10 | persistent cuTile packed int8 vs matched resident-float32 scorer | exact score bits and top-10 ordinals | exact score bits and top-10 ordinals | p99 speedup `1.67x` (B1), `1.64x` (B32) | — | `130` vs `512` | B1 p50/p99 `1.021/1.550 ms`, `954.6 q/s`; B32 `4.594/5.274 ms`, `6,870.4 q/s`; peak RSS `1.552 GB` | verified systems evidence; first JIT excluded; no descriptor-quality or universal SOTA claim |

The frozen same-teacher width ladder did **not** yield a universal winner:
CUB's learned-width interval crossed zero, as did In-Shop's; the policy tested
on unseen Flowers-102 did not generalize. SOP's width-256 improvement was
positive, not null, but the already observed split makes it claim-ineligible.
The release therefore keeps the fit-only learned/PCA selector and does not ship
a dataset-name rule or a universal width selector.

The fit path was functionally exercised at 59,551 SOP rows. Because that run did
not retain peak RSS, the documented resource-qualified boundary remains 25,882
fit rows; larger fits are not promised to stay within a generic memory budget.
At serving time the native backend is exact, fixed to 128 dimensions and top-10,
accepts query batches of 1 or 32, binds CUDA device 0, and serializes operations
per gallery handle.

## Authorities

- release assurance: `docs/evidence/release_assurance_v0_3_0_rc3.json`;
- compact selector: `docs/compact_metric_selector_result_2026-09-19.md`;
- frozen width ladder:
  `docs/evidence/same_teacher_zero_training_ladder_e87d25d_summary.json`;
- packed scorer: `docs/packed_int8_cutile_topk_result_2026-09-20.md`.

