# Pass 213 — Cars representation and protocol pivot

Date: 2026-09-01
Status: **EVIDENCE-CONDITIONED SYNTHESIS AND NEXT-ACTION REGISTRATION**

## Decision

Do not treat `97.4%` as a directly comparable target for the current SigLIP
development protocol. The published SAGA number uses a different model family,
embedding width, training surface, and evaluation gallery. A future method
claim requires a capacity-matched Qwen baseline and a separately frozen
holdout. The official Cars test surface remains burned and claim-ineligible.

Do not fund another open-ended SigLIP head, loss, or epoch sweep. First place
the current substrates and checkpoints on one common query/gallery protocol,
then resolve the already implemented bounded readout gates. Qwen feasibility is
the next model-family decision, not an assumed solution.

## Evidence that must remain separate

The authenticated seed-17 control receipt on the DGX reports these values:

| Control surface | Initial | Final | Change |
|---|---:|---:|---:|
| raw optimization | 88.8973% | 97.8299% | +8.9326 |
| projected optimization | 88.8468% | 97.9813% | +9.1345 |
| raw clean | 93.7363% | 94.5375% | +0.8012 |
| projected clean | 93.7000% | 94.5375% | +0.8376 |
| raw burned | 92.1933% | 93.5316% | +1.3383 |
| projected burned | 92.1933% | 93.5316% | +1.3383 |

Every row uses the same isolated-band leave-one-out scorer, but the bands contain
different identities and gallery sizes. A cross-band gain ratio is therefore a
descriptive single-seed diagnostic, not a mechanism bound. It must not be used
to extrapolate remaining headroom or subtract the clean value from the official
`97.4` comparison anchor.

A separate authenticated frozen SigLIP-so400m burned-band run reports
`1,242 / 1,345 = 92.3420%`, not the control's `92.1933%` initial value. Its error
manifest contains `103` errors. Under the class-name-derived equivalence groups
already registered for the band audit, `88` errors fall within a same-nameplate
group. Of those, `63` are the single Dodge Caliber Wagon 2012/2007 pair. The
collapsed score is `1,330 / 1,345 = 98.8848%`, but that is an easier task with
`11` equivalence classes instead of the original `16`; it is not an improved
strict-retrieval result.

A static execution trace narrows the `1,242`-versus-`1,240` discrepancy to
arithmetic/execution authority rather than dataset or model revision: the
control embeds through an eager `SiglipVisionModel` under CUDA bfloat16
autocast with evaluation batches of `32`, whereas the frozen substrate uses
`AutoModel.vision_model` in full float32 with batches of `8`. The common-protocol
audit must measure these variants before assigning causality; the code trace
alone does not establish which difference flips the two nearest-neighbor ties.

The defensible inference is therefore narrow: one model-year pair dominates
the frozen SigLIP-so400m error manifest. This may reflect missing
configuration evidence, label ambiguity, or both. Existing human review did
not clear its preregistered inter-rater reliability gate and therefore cannot
adjudicate the mechanism. No current evidence proves the data correct or
corrupt. The effect must be checked under the same protocol across frozen and
trained substrates before it drives architecture selection.

Plain local matching is already a negative control. On the same `1,345`-query
burned band, the existing SigLIP-base token-set screen reported pooled
`92.7881%`, MaxSim `85.5019%`, and hybrid `92.2677%`. A larger semantic model is
also not guaranteed to help: the in-repository frozen results are DINOv2-large
`88.9219%`, SigLIP2-so400m `91.2268%`, and SigLIP-so400m `92.3420%` on that
surface. Capacity and configuration sensitivity remain hypotheses.

## Registered sequence

The sole three-seed SigLIP control remains serialized on the DGX. After it
terminates:

1. Authenticate all three seed receipts and the aggregate terminal. Preserve
   raw and projected results as separate metrics.
2. Run the committed frozen band audit once. It evaluates the complete Cars
   training examples for labels `0..97` under the same conceptual isolated-band
   leave-one-out protocol. Before using it, require exact descriptor/count
   reconciliation with the control's raw initial evidence. The current
   `1,242`-versus-`1,240` burned discrepancy is an authority issue to explain,
   not a quantity to average away. Once reconciled, the audit provides an
   internally comparable three-band geometry table.
3. Reproduce error manifests for frozen DINOv2-large, SigLIP2-so400m, and
   SigLIP-so400m on the identical burned protocol, then produce the seed-17
   trained-checkpoint manifest with exactly the same preprocessing and tie
   rules. Record the Caliber `82↔83` contribution and the complete confusion
   distribution. A substrate-invariant Caliber failure raises the data/taxonomy
   posterior; a substrate-sensitive failure supports a representation pivot.
   Clean per-query errors remain unread.
4. Resolve the already committed optimization-only 27-depth
   intermediate-readout screen and RSTA Stage A. Together they have a hard
   budget of two DGX hours. A failed gate permanently closes its mechanism. A
   passing intermediate depth earns only the clean read specified by its
   existing sealed protocol.
5. Only after the control receipts, RSTA gate, repository assurance, and source
   identities are authenticated, prepare the SAGA GB10 feasibility inputs. The
   Qwen weights currently exist only in the DGX Hugging Face cache; they are not
   yet an immutable authenticated snapshot. Seal and hash them before use.
6. Run the existing SAGA feasibility diagnostic with synthetic images only and
   preserve its exact outcome: `FITS`, `MEMORY_FAIL`,
   `ATTENTION_UNAVAILABLE`, `TIME_BUDGET_FAIL`, `DETERMINISM_FAIL`,
   `BACKEND_INVALID`, or `AUTHORITY_INVALID`. ASG-CV is relevant only to a
   measured rollout/replay time bottleneck. It cannot repair missing attention,
   invalid authority, determinism failure, or an unfittable memory footprint.
7. `FITS` authorizes one Qwen seed with the already specified
   SFORA-substituted SAGA-style objective as a capacity diagnostic, trained only
   on classes `0..48`. The receipt must enumerate every substitution because
   the published Potential-Field objective is not sufficiently disclosed for
   an exact local implementation. Its comparison with SigLIP is descriptive
   capacity evidence, not a new-method result. Next reproduce one matched SAGA
   seed. Any new method must then beat the matched same-backbone SAGA baseline,
   not the SigLIP control.
8. A final clean-band difference must include paired per-query discordant counts
   and an exact McNemar/binomial test. A point estimate of `+0.5` alone is not a
   gate. A configuration that fails its one sealed clean read is not retuned on
   clean; later variants are exploratory until a fresh holdout is frozen.

## Candidate method after a positive capacity result

Only if the common-protocol and Qwen evidence support it, test a single deployed
descriptor with two trained factors:

- a coarse vehicle component preserving global geometry; and
- a configuration residual trained on within-nameplate hard negatives and
  stopped semantic attribute evidence from the frozen Qwen supervisor.

The residual must improve strict retrieval without materially reducing the
registered variant-collapsed diagnostic. Qwen language rollout/replay cost is
handled by the existing ASG-CV path only when the feasibility receipt identifies
that cost as load-bearing. No gallery fitting, transductive reranking,
official-test tuning, or generated evaluation labels are introduced.

## ETA and stops

- Remaining serialized SigLIP control: approximately `20–24` hours from the
  2026-09-01 checkpoint, dominated by the final seed.
- Common-protocol band and cross-substrate manifests plus the two existing
  bounded SigLIP gates: approximately `2–4` DGX hours.
- Qwen snapshot sealing: local I/O and hashing only, but not yet measured.
- SAGA feasibility: hard wall limit two hours after authenticated input sealing.
- First capacity-matched quality answer: determined by the feasibility receipt;
  no calendar duration is asserted before that measurement.

The earliest decisive protocol/representation verdict is roughly `22–28` hours
from the cited control checkpoint. A
release ETA is conditional: any authority, quality, memory, attention,
determinism, or time failure stops its branch instead of launching another
multi-day sweep.
