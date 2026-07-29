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

## 6. Standing caveat

Even in the best case this is not the novel *method* the project set out to find. It
is a conditional characterisation of a known technique. Worth having, worth writing
up honestly, and not worth inflating into anything more — the thirteen entries in
[results.md](results.md) are there because inflation was tried and did not survive
contact with a paired test.
