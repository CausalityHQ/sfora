# Pre-registration: is self-distillation a headroom-dependent regulariser?

**Written 2026-07-29, while queue v19 is still running and before any of its arms
have landed.** The point of writing it now is that every reading below is currently
falsifiable. Once the numbers arrive, whichever story fits them will feel obvious,
and this file exists to stop that.

---

## 1. The observation

Queue v18 finished the CUB Proxy Anchor pair at three seeds. It is the first arm in
this project to pass the pre-registered gate (≥ +0.50 pt and all seeds positive):

| CUB arm | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| `proxy_anchor` | 0.6825 | 0.6882 | 0.6921 | 0.6876 |
| `pa_distill` | 0.6916 | 0.6985 | 0.6994 | 0.6965 |

Paired: **+0.91 / +1.03 / +0.73 → +0.89 pt**, sd 0.153, t = 10.05, p = 0.0097.

This is in the **frozen-BatchNorm** setting, where the H3 teacher/student defect is
provably inert (`test_teacher_normalisation_fix_is_inert_when_batch_norm_is_frozen`).
Whatever this effect is, it is not that bug resurfacing.

## 2. The proposed explanation

Every distillation result we own, arranged by how much room the base had left. Define
**headroom** as the gap from that arm's base to the *best base on the same dataset* —
a crude proxy for "how far from this dataset's ceiling is this model".

| dataset | base | base R@1 | headroom | distillation Δ |
| --- | --- | ---: | ---: | ---: |
| CUB | Proxy Anchor | 0.6876 | **2.31 pt** | **+0.89** |
| CUB | HIST | 0.7107 | 0 | −0.03 |
| In-Shop | HIST | 0.9038 | 0 | +0.03 |
| In-Shop | Proxy Anchor | 0.9035 | 0.03 pt | −0.04 |

(In-Shop rows use the `_bnfix` arms, so the BatchNorm defect is excluded from all
four.)

The reading: **EMA self-distillation is a variance-reducing regulariser whose benefit
is proportional to the base's remaining headroom, and which is worth nothing at the
ceiling.** It buys generalisation for models that are underfitting and cannot help
models that are not.

That would be a modest but genuinely useful claim. Self-distillation as a regulariser
is old (Mean Teacher, Tarvainen & Valpola 2017); *a quantitative predictor of when it
pays* is not, and the field currently reports such gains as unconditional method wins.

## 3. Why this is much weaker evidence than the table makes it look

**Four rows, but only one is informative about the claim.** Three of them sit at
headroom ≈ 0 and all show |Δ| ≤ 0.04 — consistent with the hypothesis, but equally
consistent with "distillation does nothing, anywhere, and CUB PA is a fluke". A line
fitted to these four points has its slope determined **entirely by a single arm**.
This is n=1 for the parameter that matters, dressed up as n=4.

Stated honestly, what we have is: *one base with real headroom gained; three bases
without headroom did not.* That is suggestive and it is not a relation.

**Three further reasons for caution:**

1. The exact sign test floors at 0.25 for n=3. Assumption-free, +0.89 pt is not yet
   significant; the p = 0.0097 rests on a normality assumption three points cannot
   evidence.
2. The paired sd of 0.153 pt is far tighter than this harness's measured 1.08 pt
   fixed-seed spread would predict. Either Proxy Anchor runs are much more
   reproducible than the HIST runs that spread was measured on, or three seeds were
   lucky. Not currently known which.
3. A CUB screen has already produced a false positive here once (+0.52 pt at n=1,
   retracted). CUB's σ = 0.88 pt is exactly the scale of this effect.

**An alternative explanation with equal current support.** Our Proxy Anchor CUB
reproduction lands at 0.6876 against a published 0.697. `pa_distill` reaches 0.6965 —
which is the published number, near enough. So instead of "distillation adds
generalisation", the effect may be "distillation compensates for some unidentified
deficiency in our PA reproduction, and a faithful PA would show no gain". Nothing in
the current data separates these, and the second is unflattering enough that it
deserves stating first.

## 4. What v19 does and does not test

v19 runs seeds 3–5 of all four CUB arms. It buys:

- **Six paired seeds on the PA leg.** 6/6 positive gives an assumption-free
  p = 2⁻⁵ = 0.031, and re-measures the paired sd on a sample that did not generate
  the hypothesis. This settles *whether the effect is real*.
- **Six paired seeds on the HIST leg**, testing the dissociation. A clean split —
  PA holds ~+0.9, HIST stays flat — is a much better result than a bare improvement.

It does **not** test the headroom claim, and this is the part worth being clear about:
CUB HIST is a headroom-0 point, so three more seeds of it add precision to a point
the hypothesis already fits and contribute **nothing to the slope**. After v19 the
slope will still rest on one arm.

## 5. The experiment that would actually test it

The claim needs bases at *intermediate* headroom. The clean knob is
`embedding_dimensions` (512 → 128 → 64): it weakens Proxy Anchor monotonically while
changing nothing else about the recipe, and each weakened base is its own control.

Pre-registered predictions, in decreasing order of how much they would convince me:

1. **Monotonicity.** Distillation Δ increases monotonically as embedding dimension
   falls. Fails if any weaker base gains *less* than a stronger one.
2. **Proportionality.** Δ / headroom stays roughly constant across the family. The
   single point we have puts that ratio at 0.89 / 2.31 ≈ **0.39**; I will count the
   hypothesis as surviving if every arm lands in 0.2–0.6 and as refuted if the ratio
   varies by more than 3×.
3. **The ceiling.** A base at headroom 0 gains nothing, ±0.1 pt.

**What would falsify it outright:** a weakened base that gains *the same* +0.9 pt
regardless of how far below the ceiling it sits. That would mean distillation adds a
fixed increment rather than a headroom-proportional one — still a real effect, but
the explanation here would be wrong, and the "predictor of when it pays" framing
would have to go.

**What would kill it more quietly:** the PA leg failing to hold up at seeds 3–5. Then
there is nothing to explain and this document is the record of an idea that did not
survive its first confirmation, which is the outcome twelve of the previous thirteen
candidates had.

## 6. Outcome of v19/v20 — the effect is real, the explanation is not established

Both legs finished at six paired seeds on 2026-07-30.

| CUB leg | per-seed Δ | mean | sd | positive | paired t | exact sign p |
| --- | --- | ---: | ---: | :-: | ---: | ---: |
| `pa_distill` − `proxy_anchor` | +0.91 +1.03 +0.73 +0.32 +0.86 +0.10 | **+0.658** | 0.367 | **6/6** | +4.39 | **0.031** |
| `herd` − `hist` | −0.27 +0.34 −0.15 −0.39 +1.50 +0.76 | +0.298 | 0.729 | 3/6 | +1.00 | 1.000 |

**§4's question is answered: the PA effect is real.** Six of six positive gives the
assumption-free p = 0.031 that was set as the bar, and the paired t agrees.

**§2's explanation is not.** Three things go against it:

1. **The dissociation is not resolvable here.** The between-leg gap is +0.360 pt with
   SE 0.333 — t = 1.08, nowhere near significance. Reaching 80% power on a gap that
   size, given the pooled sd of 0.577, needs **≈40 seeds per arm**. That is ~120 GPU-h
   on CUB for one comparison, so this dataset cannot settle it. The same lesson as §4
   of [HANDOFF.md](HANDOFF.md), arriving from a new direction.
2. **The winner's curse was real, and it was large.** Splitting the PA leg by whether
   a seed generated the hypothesis: in-sample (0–2) **+0.890**, out-of-sample (3–5)
   **+0.427**. The honest effect size is *less than half* what the screening seeds
   showed. The pre-registered doubt about sd 0.153 was also correct — it is 0.367.
3. **On fresh seeds the ordering reverses.** HIST's out-of-sample mean is **+0.623**,
   *higher* than PA's +0.427. Both subsets are n=3 and far too noisy to conclude
   anything from, but the one cut of the data that is free of selection effects points
   the opposite way to the hypothesis rather than supporting it.

So: a real ~+0.43 pt gain from EMA self-distillation on Proxy Anchor/CUB, and no
evidence that headroom explains it.

### What this changes about the next experiment

The dissociation failed for a *structural* reason worth naming: it tried to resolve a
0.36 pt difference between two arms whose own noise is 0.37–0.73 pt. The lever was
smaller than the noise. Adding seeds to that design is throwing GPU at a 40-seed
requirement.

The `narrow128`/`narrow64` design does not have that problem, and this is the argument
for running it rather than abandoning the question. It sets headroom **by construction**
instead of measuring it, and it multiplies the effect rather than differencing two
similar ones: if the gain really scales with headroom, a 64-d base sitting several
points below the ceiling should show something in the region of +1.3 pt against a
per-seed sd of ~0.37 — resolvable at three seeds, not forty.

It also avoids the artifact flagged when `herd/seed4` came in at +1.50 against the
lowest HIST baseline: Δ = distilled − base is negatively correlated with base *by
construction*, so ranking measured baselines against their own deltas proves nothing.
Embedding width is fixed in the recipe and independent of any measurement, which is
exactly the property the test needs.

## 7. Standing caveat

Even in the best case this is not the novel *method* the project set out to find. It
is a conditional characterisation of a known technique. Worth having, worth writing
up honestly, and not worth inflating into anything more — the thirteen entries in
[results.md](results.md) are there because inflation was tried and did not survive
contact with a paired test.
