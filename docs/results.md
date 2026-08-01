# Benchmark results

All numbers are **CUB-200-2011**, ResNet-50 backbone, 512-dim embedding, the
standard zero-shot retrieval split (100 train / 100 disjoint test classes),
cosine **Recall@1**, reported as **best-over-training** (the protocol used by the
papers below — evaluate the held-out test classes every few epochs and take the
peak).

> **Single-model rank diagnostic (2026-08-01, CPU-only).** A preregistered
> train-fit/test-apply PCA sweep on CUB HERD seed 0 found unprojected R@1 0.6940,
> fitted centered rank-512 0.6818, rank-256 0.6794, and rank-128 0.6661. The
> fitted full-rank change (-1.215 points) violated the fixed 0.10-point validity
> tolerance, so this is a diagnostic failure rather than a compressed-method
> result. It shows that the cosine embedding's mean/origin is materially useful.

> **Status correction (2026-07-20).** Results below predate strict
> publication-backed method×dataset recipes and are retained as historical
> `modified_legacy` evidence. They must not be called official Proxy Anchor or HIST
> reproductions. Corrected reference/selected-extension experiments have been queued
> on the DGX; this document will add a separate table only after recipe IDs and
> digests validate. No legacy number is used to choose an unpublished recipe.

## Corrected reference-recipe evidence

The first corrected results have landed, and they are **negative for the method**.

### CUB-200 under the official recipes (seed 0 — single seed, screening only)

The comparison the headline claim never had. Both HIST arms carry official
LayerNorm; the arms differ only in the declared distillation delta.

| arm | R@1 | vs published | vs its base |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.6825 | reported 0.697 (**−1.45**) | — |
| PA + distillation | **0.6916** | — | **+0.91** |
| HIST | **0.7183** | reported 0.714 (**+0.43**) | — |
| HERD (HIST + distillation) | 0.7156 | — | **−0.27** |

Three things follow, and the first is the important one.

**1. The legacy "HERD beats HIST" result was the LayerNorm, not the
distillation.** The legacy pair was HIST 0.700 (control run *without* embedding
LayerNorm) → HERD 0.716, read as +1.6 for distillation. Corrected, with LayerNorm
held constant as official HIST specifies: HIST 0.7183 → HERD 0.7156, i.e.
**−0.27**. The entire historical gain is accounted for by the confound. This is
the single clearest outcome of the audit so far.

**2. HIST reproduces above its published number** (0.7183 vs 0.714). Proxy Anchor
lands 1.45 points below its own (0.6825 vs 0.697).

*Caveat on that second half, added after checking.* Calling this a "reproduction
gap" was premature at **one seed**: CUB seed noise in this harness has historically
been σ ≈ 0.6 pt, so 1.45 points is ≈ 1.4σ — unremarkable. A line-by-line fidelity
audit against the official repository also found nothing to blame it on:

| checked | official Proxy Anchor | ours |
| --- | --- | --- |
| CUB ResNet-50 command | `--batch-size 120 --lr 1e-4 --warm 5 --bn-freeze 1 --lr-decay-step 5` | identical |
| warmup semantics | freeze backbone, train embedding + proxies only | identical (`backbone_warmup_parameters`) |
| train transform | `RandomResizedCrop(224)`, torchvision default scale | identical |
| eval transform | `Resize(256)` + `CenterCrop(224)` | identical |
| proxy LR multiplier | `lr * 100` | identical |

(The audit was prompted by a specific suspicion — that we used torchvision's
default crop scale where the authors set `scale=(0.16, 1)`. The authors do **not**;
they call `RandomResizedCrop(224)` bare. Hypothesis wrong, fidelity confirmed.)

So the honest statement is that PA's single-seed number sits low but within noise,
with no identified fidelity defect. Seeds 1 and 2 settle it.

### The cognitive-science candidates: both fail, decisively

Two methods imported from outside ML, both replacing the **similarity function**
rather than adding a loss term, both differing from official Proxy Anchor only in
their declared recipe delta. Settled on In-Shop, where σ = 0.12 pt makes a single
seed conclusive.

| method | In-Shop R@1 | vs PA 0.9035 | CUB (3 seeds) |
| --- | ---: | ---: | ---: |
| Shepard exponential kernel | 0.8999 / 0.8998 | **−0.36 pt** (~3σ) | 0.6743 (−0.82, unreadable) |
| Tversky contrast similarity | 0.8600 / 0.8543 | **−4.63 pt** | 0.6758 (−0.67, unreadable) |

Both CUB columns are reported only to show they *are* unreadable: at σ = 0.88 pt a
0.7 pt gap is 0.8σ, and Shepard's city-block variant (`shepard_l1`, 0.6778) lands
inside its own Euclidean sibling's spread — so Shepard's integral/separable
distinction makes no measurable difference here either. The verdicts come from
In-Shop.

**Shepard.** The mechanism was real and the derivation holds: cosine-softmax *is*
secretly Gaussian — on unit vectors `cos = 1 − d²/2`, so `exp(cos/T) ∝ exp(−d²/2T)` —
and Shepard's `exp(−d)` genuinely has a fatter tail (measured 2.9× more mass on the
far neighbour at T = 0.05, which is Proxy Anchor's operating point). It simply does
not help. The hypothesis was that fatter tails would keep gradient flowing to
moderately-distant true positives and rescue orphans; the result suggests those
positives are distant *because they should be*, and weighting them more costs more
than it recovers. A small, real, resolvable negative.

**Tversky.** Much worse, and the size is informative. The bounded ratio form is
correct and its asymmetry is genuine (tests pin both), but discarding the *magnitude*
of agreement in favour of set-membership contrast throws away most of what a dense
embedding encodes. `x·f_k > 0` is a hard threshold: two embeddings agreeing strongly
and two agreeing barely both count as "sharing the feature". On sparse binary
fingerprints — where Tanimoto earns its keep in cheminformatics — that is lossless.
On dense CNN embeddings it is not.

Its two seeds also spread 0.57 pt — nearly 5× the 0.12 pt this dataset shows for
every other arm (Shepard's two seeds differ by 0.01 pt). A similarity that is both
much worse *and* much less stable is behaving like a poorly-conditioned objective,
not like a method that needs tuning. Neither seed comes within 4 pt of the baseline.

Worth recording as the general lesson from both: **a similarity function that is
better-motivated as a model of *human judgement* is not thereby better as a *retrieval
score*.** Tversky's model is descriptively correct about people; retrieval is not
asking the same question.

### The earlier new-method candidates: both clean negatives

| arm | seeds | mean | vs baseline |
| --- | --- | ---: | ---: |
| `region_pa` (multi-vector, MaxSim retrieval) | 0.6442 / 0.6453 / 0.6502 | **0.6466** | **−3.6 pt** vs PA 0.6825 |
| `local_nca` (L_in, memory positives) | 0.6590 / 0.5370 / 0.5240 | **0.5733** | **−13.7 pt** vs HIST 0.7107 |

**`region_pa` — decisive, and the evaluation lesson is the keeper.** Its first run
scored 0.5775 because the loss optimised a soft maximum over regions while retrieval
scored cosine on the *concatenated* region vector. Wiring in MaxSim retrieval was
worth **+6.7 pt** (0.5775 → 0.6442) — a real, reusable finding about evaluating
region-based models. The reason is sharp: MaxSim matches each query region to its best
gallery region wherever it sits, so the same object photographed off-centre scores
1.00; concatenated cosine compares slot-to-slot and scores it **0.00**. Even so, the
method sits 3.6 pt below Proxy Anchor with σ ≈ 0.03 pt across seeds, and peaks around
epoch 11–17 of 60 before decaying. Not noise, not rescuable by the metric.

**`local_nca` — failed, and the diagnostic refuted the failure I predicted.** I built
`local_nca_effective_positives` (exp of the entropy of the softmax over an anchor's
positives) expecting the "one buddy per anchor" collapse that Khosla et al.'s
objection to L_in predicts: ~1.0 would mean a single partner carries each anchor.
Measured:

| seed | best | peak epoch | effective positives | available |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.6590 | 23 | 31.1 | 39.9 |
| 1 | 0.5370 | **1** | 39.6 | 39.9 |
| 2 | 0.5240 | **1** | 39.6 | 39.9 |

The opposite happened. Effective positives ≈ *all* available, i.e. the positive
softmax is essentially **uniform** — every same-class instance weighted equally, which
is L_out's behaviour, not the selective behaviour L_in was chosen for. A flat positive
distribution at τ = 0.1 means all within-class similarities sit within ~0.1 of each
other: the representation is not discriminating inside a class at all. Two of three
seeds peak at **epoch 1** and never recover.

So the mechanism never engaged. Whether that is the memory queue supplying ~40
stale positives per anchor (drowning the selectivity), the temperature, or the
inherited HIST schedule is untested — but the honest statement is that this
implementation of L_in did not behave like L_in, and the negative result is about the
implementation as much as the idea.

### The properly paired CUB picture at SIX seeds (supersedes the 3-seed tables below)

Both CUB legs were taken to six paired seeds on 2026-07-30. This is the authoritative
CUB record; the 3-seed tables that follow are kept because they document how the
conclusions changed, not because their numbers should be quoted.

| arm | s0 | s1 | s2 | s3 | s4 | s5 | mean | sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proxy_anchor` | 0.6825 | 0.6882 | 0.6921 | 0.6924 | 0.6980 | 0.6982 | 0.6919 | 0.0060 |
| `pa_distill` | 0.6916 | 0.6985 | 0.6994 | 0.6956 | 0.7066 | 0.6992 | **0.6985** | 0.0050 |
| `hist` | 0.7183 | 0.7010 | 0.7127 | 0.7090 | 0.7009 | 0.7073 | 0.7082 | 0.0067 |
| `herd` | 0.7156 | 0.7044 | 0.7112 | 0.7051 | 0.7159 | 0.7149 | **0.7112** | 0.0053 |

| paired leg | per-seed Δ | mean | sd | positive | t | p_t | exact sign p |
| --- | --- | ---: | ---: | :-: | ---: | ---: | ---: |
| `pa_distill` − `proxy_anchor` | +0.91 +1.03 +0.73 +0.32 +0.86 +0.10 | **+0.658** | 0.367 | **6/6** | +4.39 | 0.0070 | **0.031** |
| `herd` − `hist` | −0.27 +0.34 −0.15 −0.39 +1.50 +0.76 | +0.298 | 0.729 | 3/6 | +1.00 | 0.36 | 1.000 |

**One real effect, and one lesson about how it was nearly overstated.**

EMA self-distillation on Proxy Anchor/CUB is **real**: six of six seeds positive gives
an assumption-free sign-test p = 0.031, the bar set in advance in
[headroom_hypothesis.md](headroom_hypothesis.md). But the size shrank badly once seeds
that had not generated the hypothesis arrived — in-sample (0–2) **+0.890**,
out-of-sample (3–5) **+0.427**. Quote **+0.43 pt**, not +0.89. The screening estimate
was inflated by more than a factor of two, and the paired sd it was computed against
(0.153) was luck; the honest figure is 0.367.

`herd` − `hist` is +0.298 with 3/6 positive — indistinguishable from zero, consistent
with the retraction recorded below. It is also **not** distinguishable from the PA leg:
the between-leg gap is +0.360 pt with SE 0.333 (t = 1.08), and separating them at 80%
power would need ≈40 seeds per arm. So *"distillation helps the underfitting base and
not the one at the ceiling"* remains an unsupported story, not a finding.

**The explanation was tested and refuted.** `narrow128` / `narrow64` weaken the Proxy
Anchor embedding from the official 512, setting headroom *by construction* rather than
measuring it (full detail in [headroom_hypothesis.md](headroom_hypothesis.md)):

| width | base | headroom | mean Δ | headroom predicted |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 0.6919 | 1.63 pt | +0.66 | +0.30 |
| 128 | 0.6584 | 4.98 pt | −0.07 | +0.92 |
| 64 | 0.6317 | 7.66 pt | +0.48 | +1.42 |

Headroom varies nearly 5× and Δ does not track it. A second explanation — distillation
as a regulariser that hurts a capacity-starved model — was registered in advance and
died in one run (it predicted Δ(64) < −0.89; observed +0.42 / +0.54). So the honest
statement is **a fixed increment of ~+0.4 pt, roughly independent of embedding width,
with no established mechanism.** Fourteenth and fifteenth candidate explanations, both
pre-registered, both dead.

### What the EMA teacher actually supplies: a 2×2, and a failed additivity prediction

An EMA teacher does two separable things — it is a **distillation target**, and it is an
**averaged copy of the weights**. `pa_distill` conflates them: it distils toward the
teacher but evaluates the *student*. These arms separate the roles. Momentum is held at
0.99 across the three non-base cells so a difference between cells cannot be the momentum.

| CUB arm | momentum | distil | evaluate teacher | seed 0 | seed 1 | mean Δ |
| --- | ---: | :-: | :-: | ---: | ---: | ---: |
| `pa_distill` | 0.999 | ✓ | — | +0.91 | +1.03 | **+0.658** (6 seeds) |
| `pa_ema_avg` | 0.999 | — | ✓ | +0.07 | +0.05 | **+0.06** |
| `pa_ema_avg_fast` | 0.99 | — | ✓ | +0.46 | +0.37 | **+0.41** |
| `pa_distill_fast` | 0.99 | ✓ | — | +0.30 | — | **+0.30** (n=1) |
| `pa_distill_avg` | 0.99 | ✓ | ✓ | +0.52 | — | **+0.52** (n=1) |

**Momentum dominates the difference between the two averaging arms.** At 0.999 over CUB's 2940 steps the average still
retains 0.999²⁹⁴⁰ = **5.3% of its initialisation** — a pretrained backbone but a *randomly
initialised* embedding head. That contamination is worth 0.35 pt: +0.06 against +0.41.
The bracket was built from that arithmetic *before* any GPU was spent, and it earned its
keep — the natural single arm to build is the one matching `pa_distill` at 0.999, and it
would have measured +0.06 and retired weight averaging as useless on this benchmark.

**The matched-momentum 2×2 does not reduce to either proposed binary explanation.**
At momentum 0.99 on seed 0, distillation-only is **+0.30**, averaging-only is
**+0.46**, and both is **+0.52**. The pre-registered alternatives were approximately
+0.5 (momentum carries the 0.99 cells) or approximately zero (averaging carries them
entirely). The observed +0.30 is intermediate: faster momentum improves the
distillation target, while evaluating the averaged weights carries the larger share
of the observed gain. Adding distillation on top of averaging buys only another
+0.06 on this seed, not the +0.30 an additive model predicts.

This resolves the mechanism qualitatively, not the effect size statistically:
`pa_distill_fast` and `pa_distill_avg` are each at **n=1**, while
`pa_ema_avg_fast` is at n=2. Queue v24 is completing all three cells to three seeds;
until then the 2×2 is a matched-seed mechanistic result, not a publishable estimate.

**The additivity prediction failed.** If the two roles were separable and complementary,
their matched-momentum seed-0 effects (+0.30 and +0.46) predict about +0.76 together.
Observed is **+0.52** — only +0.06 above averaging alone. The two roles therefore capture
mostly overlapping signal, not complementary signal.

**Caution on the tight sds.** `pa_ema_avg_fast` shows sd 0.064 at n=2, implying an
implausible 0.1 seeds for 80% power. An sd from two runs has *one* degree of freedom and
is nearly worthless: `pa_distill`'s own 3-seed sd was 0.153 and became **0.367** at six
seeds. Do not plan on it.

### The protocol systematically under-credits weight averaging

Averaging the weights smooths the *evaluated* model's test curve. Best-over-training is a
maximum over that curve, so a smoother curve collects a smaller selection bonus — the
protocol pays the noisier arm more. That was a prediction; it is now measured.

| CUB arm | evaluates | selection bonus |
| --- | --- | ---: |
| `proxy_anchor` | student | **0.769 pt** |
| `pa_distill` | student | **0.836 pt** |
| `pa_ema_avg_fast` | averaged weights | 0.306 pt |
| `pa_distill_avg` | averaged weights | 0.122 pt |
| `pa_ema_avg` | averaged weights | **0.074 pt** |

Every arm that evaluates the averaged weights collects **2.5–10× less** selection bonus
than one evaluating the student. Removing each arm's own bonus reverses the ranking:

| paired comparison | reported | corrected | protocol's effect |
| --- | ---: | ---: | ---: |
| `pa_ema_avg_fast` − `proxy_anchor` | +0.414 | **+0.732** | understates by 0.32 |
| `pa_ema_avg` − `proxy_anchor` | +0.059 | **+0.610** | understates by **0.55** |
| `pa_distill` − `proxy_anchor` | +0.658 | +0.592 | overstates by 0.07 |

Three consequences.

1. **Weight averaging is the strongest single intervention measured in this project**
   (+0.73 corrected), ahead of distillation (+0.59) — the opposite of the reported
   ordering.
2. **The 0.999 arm was never useless.** Reported at +0.06 it looked dead; corrected it is
   **+0.61**. Its apparent failure was almost entirely the protocol, not the method — the
   averaged model is so stable it earns almost no bonus while the noisy baseline earns
   +0.77. The initialisation-contamination story explains the *reported* gap between the
   two momenta; it does not survive the correction.
3. **The additivity failure looks different too.** `pa_distill_avg` collects only 0.122 pt
   of bonus, so its reported +0.52 is much closer to its true value than `pa_distill`'s
   reported +0.658 is to its +0.592.

**Caveats, stated plainly.** The averaging arms are at n=2. The correction is one specific
estimator (leave-one-out neighbour mean), and corrected values are *not* the benchmark
metric — the field reports best-over-training, so a paper claiming these numbers would
have to argue the protocol, not just quote them. But the direction is mechanically
predicted rather than fitted, and the magnitude (0.32–0.55 pt) is larger than most
published DML gains.

**A methodological correction that matters more than either.** This project has been
quoting CUB σ ≈ 0.88 pt and "+0.5 pt needs 12–37 seeds". Both came from three seeds.
At six:

| quantity | 3-seed figure | 6-seed figure |
| --- | ---: | ---: |
| per-arm across-seed σ | 0.88 pt | **0.57 pt** (0.50–0.68 across four arms) |
| paired sd, PA leg | 0.153 pt | **0.367 pt** |
| paired sd, HIST leg | 0.322 pt | **0.729 pt** |
| seeds for 80% power on +0.5 pt, PA leg | 12–37 | **5** |
| seeds for 80% power on +0.5 pt, HIST leg | 12–37 | **17** |

Two things follow. σ estimated from three seeds is worthless in *either* direction —
it was 55% too high for the arms and 2–4× too low for the paired differences. And the
seeds-required figure is **not a property of the dataset**: it depends on the arm,
because pairing removes most of the seed variance on the PA leg (0.60 → 0.37) and much
less on the HIST leg (0.68 → 0.73, i.e. none). Always report the paired sd of the
specific comparison; a dataset-level σ cannot power a paired test.

### The earlier 3-seed picture (retained for the audit trail)

With HIST finally run at seeds 1 and 2, the comparisons become valid — and almost
nothing survives.

| arm | seed 0 | seed 1 | seed 2 | mean | σ |
| --- | ---: | ---: | ---: | ---: | ---: |
| HIST | 0.7183 | 0.7010 | 0.7127 | **0.7107** | 0.0088 |
| HERD | 0.7156 | 0.7044 | — | 0.7100 | 0.0079 |
| herd_hg_incidence | 0.7235 | 0.7031 | 0.7070 | **0.7112** | 0.0108 |

Paired per-seed deltas against HIST:

| comparison | per-seed Δ | mean | paired t | p |
| --- | --- | ---: | ---: | ---: |
| herd_hg_incidence − HIST | +0.52, +0.20, −0.57 | **+0.05** | +0.16 (df=2) | 0.89 |
| HERD − HIST | −0.27, +0.34 | **+0.03** | +0.11 (df=1) | 0.93 |

**Two more of my own claims die here.**

1. **The "+0.52 win" evaporates completely.** Paired properly it is **+0.05 pt**,
   p = 0.89. Seed 0 gave +0.52, seed 2 gave −0.57. Pure noise.
2. **"Distillation hurts HIST (−0.27)" was also noise.** Paired over two seeds it is
   **+0.03 pt**, p = 0.93. The corrected-matrix headline that the legacy HERD gain
   was a LayerNorm confound still stands — the legacy +1.6 does not reproduce — but
   the replacement claim that distillation actively *hurts* HIST is not supported
   either. The honest statement is that on CUB, at this noise level, **HERD and HIST
   are indistinguishable**.

3. **HIST's reproduction number changes.** The 3-seed mean is **0.7107**, not the
   0.7183 of seed 0 — so HIST reproduces essentially *exactly* on its published
   0.714 (−0.33), rather than above it as this document previously claimed. Seed 0
   was the high draw.

Single-seed results below the noise floor, retained only to be explicit that they
are uninterpretable: `herd_hg` −0.08, `herd_hg_prototype` +0.00,
`hist_shot_geometric` −1.08, `hist_ipc4` −2.73. Only the last is plausibly outside
noise.

### ⚠️ Retraction: the "+0.52 win" was seed noise

Recorded prominently because it invalidates the section below it, and because it is
the most useful thing measured all day.

`herd_hg_incidence` scored **0.7235** on CUB seed 0 — apparently +0.52 over HIST and
the first arm to clear the screening gate. Seeds 1 and 2 came back **0.7031** and
**0.7070**.

| seed | 0 | 1 | 2 | mean | sd |
| --- | ---: | ---: | ---: | ---: | ---: |
| herd_hg_incidence | 0.7235 | 0.7031 | 0.7070 | **0.7112** | **0.0108** |

**CUB seed noise in this harness is σ ≈ 1.1 pt** — larger than the historical
σ ≈ 0.6, and far larger than every effect measured in this document. Seed 0 was a
lucky draw.

Two consequences, both serious:

1. **Every single-seed conclusion on CUB is uninterpretable**, including the
   ordering that motivated the follow-up work (`herd_hg` −0.09 vs
   `herd_hg_incidence` +0.52 vs `hist_ipc4` −2.74). Differences under ~2 pt at n=1
   are noise.
2. **Those comparisons were not even paired.** We hold only HIST *seed 0*, so
   "+0.52" subtracted a single-seed baseline from what later became a three-seed
   mean. That is not a valid comparison in either direction.

The queue has been reprioritised accordingly: **HIST seeds 1 and 2 run first**, no
further variants are screened at n=1, and the In-Shop arms are promoted because
there the effect (−1.39 pt) genuinely exceeds that dataset's seed noise (σ = 0.0012)
and is therefore measurable at all.

Also landed before the reprioritisation, and subject to the same caveat:

| arm (CUB seed 0) | R@1 | vs HIST seed 0 |
| --- | ---: | ---: |
| `herd_hg_prototype` (full-catalogue affinity) | 0.7183 | +0.00 |
| `hist_ipc4` (balanced sampling, IPC=4) | 0.6909 | −2.74 |

`hist_ipc4` is the one result here plausibly larger than noise: forcing 8 classes ×
4 samples per batch appears to *hurt* substantially, presumably because it collapses
the number of distinct classes — and therefore hyperedges — a batch can contain. The
hypothesis that HIST's hypergraph is under-used with random batches is not supported.

### The arm that appeared to beat HIST (superseded by the retraction above)

| arm (CUB seed 0) | R@1 | vs HIST | peak epoch | decay |
| --- | ---: | ---: | ---: | ---: |
| HIST | 0.7183 | — | 27 / 41 | −0.16 |
| HERD — pairwise EMA distillation | 0.7156 | −0.27 | 19 | −1.05 |
| herd_hg — propagated HGNN logits | 0.7174 | −0.09 | 26 | −0.13 |
| **herd_hg_incidence — Mahalanobis prototype affinity** | **0.7235** | **+0.52** | 22 | −0.61 |

**This inverts the argument that motivated the work, and the inversion is the
finding.** `herd_hg_incidence` was built as an *ablation control*, not a candidate:
`test_only_the_propagated_target_is_a_genuine_n_ary_quantity` proves its target
`H_i` depends solely on sample `i`'s own distances to the class Gaussians, is
invariant to the rest of the batch, and is therefore ordinary dark-knowledge KD over
Mahalanobis-proxy logits — **expressible without any hypergraph and carrying no
novelty claim**. The arm that *does* carry the novelty claim, the propagated HGNN
logits whose normalisation depends on the whole batch's hyperedge population, scored
−0.09.

So on this evidence the n-ary property is not what helps. The plausible reading,
consistent with §2b: the propagated target mixes the entire batch and is
correspondingly noisier, while the per-sample prototype affinity is a cleaner,
lower-variance target. Its curve supports that — it is far ahead early (0.6931 at
25% of the run against HIST's 0.6487) and peaks at 54%.

Caveats that matter more than the number:

* **One seed, and +0.52 is below CUB's σ ≈ 0.6.** This is a screening signal, not a
  result. Seeds 1 and 2 are running; a claim needs ≥ 6 seeds (see the p-value floor
  in `research_reset_plan.md` Phase 4).
* If it survives, the honest framing is **not** "hypergraph-native distillation
  works". It is "distilling an EMA teacher's class-prototype affinities helps a
  strong HIST base" — closer to Hinton-style KD than to anything novel, and the
  novelty question would have to be re-asked from scratch.

**2b. What the training curves say: the distillation is a regulariser.** The
best-over-training numbers hide the mechanism; the curves show it plainly.

| arm | best R@1 | peak epoch | position in run | decay after peak |
| --- | ---: | ---: | ---: | ---: |
| Proxy Anchor | 0.6825 | 12 / 60 | **20%** | **−1.35 pt** |
| PA + distillation | 0.6916 | 20 / 60 | 33% | −0.95 pt |
| HIST | 0.7183 | 27 / 41 | **66%** | **−0.16 pt** |
| HERD (pairwise distill) | 0.7156 | 19 / 41 | 46% | −1.05 pt |
| herd_hg (hypergraph distill) | 0.7174 | 26 / 41 | 63% | −0.13 pt |

Read down the "position in run" column:

* **Proxy Anchor overfits hard** — it peaks a fifth of the way through and then
  sheds 1.35 points. Distillation pushes the peak out to a third of the run and
  lifts it by +0.91. That is textbook regularisation.
* **HIST barely overfits** — it peaks two thirds through and sheds 0.16. It has no
  use for another regulariser, and the pairwise target actively *over*-regularises
  it: the peak is pulled forward to 46% and the decay triples to −1.05.
* **The hypergraph target is gentler** — it leaves HIST's curve shape essentially
  intact (63%, −0.13) and lands −0.09 instead of −0.27. Directionally better than
  pairwise, still not an improvement.

This single mechanism accounts for every distillation result we have, including
In-Shop, and it resolves the apparent contradiction in §2: the effect tracks how
much headroom the base has, not the dataset or the loss family. It also says
plainly what beating HIST requires — **added structure, not added regularisation.**

**3. Distillation helps the weaker base and not the stronger one.** On CUB it adds
+0.91 to Proxy Anchor (0.6825, underfitting) and subtracts 0.27 from HIST (0.7183).
Combined with In-Shop, where it costs both bases, the picture is:

| base | CUB (BatchNorm frozen) | In-Shop (BatchNorm trainable) |
| --- | ---: | ---: |
| Proxy Anchor | **+0.91** | −0.41 |
| HIST | −0.27 | −1.39 |

Two effects appear to be superimposed: a roughly −1.2 pt shift moving from CUB to
In-Shop for *both* bases (consistent with the BatchNorm teacher/student mismatch,
or simply with dataset size), and within each dataset a benefit that accrues only
to the weaker, underfitting base (consistent with a variance-reducing regularizer).
The queued `_bnfix` runs separate the first effect from the second.

*Resolved below (H3): the first effect was the BatchNorm teacher/student mismatch,
not dataset size. Fixing it recovers the In-Shop column on both legs.*

Single seed — CUB seed noise has historically been σ ≈ 0.6 pt, so −0.27 is well
inside noise while +0.91 is roughly 1.5σ. Neither is a result yet.

**DeepFashion In-Shop**, best-over-training R@1, 3 seeds. Proxy Anchor rows use
the official `reference` recipe; HIST rows use a frozen `selected_extension`
(HIST published no In-Shop recipe, so it was selected from SOP using
**training-split-only** scoring — no test leakage). Within each base, the plain
and distilled arms differ in `ema_distill_weight`/`ema_momentum`/`ema_distill_tau`
and nothing else (`derive_recipe`); `embedding_layer_norm` is held constant, so
the legacy confound cannot recur.

| arm | seed 0 | seed 1 | seed 2 | mean | Δ vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy Anchor | 0.9024 | 0.9048 | 0.9032 | **0.9035** | — |
| PA + distillation | 0.8999 | 0.8994 | 0.8990 | **0.8994** | **−0.41 pt** |
| HIST | 0.9046 | 0.9037 | 0.9031 | **0.9038** | — |
| HERD (HIST + distillation) | 0.8906 | 0.8892 | 0.8900 | **0.8899** | **−1.39 pt** |

All six paired per-seed deltas are negative (PA: −0.25 / −0.55 / −0.41; HIST:
−1.40 / −1.45 / −1.31). With the correct paired test (df=2): HERD vs HIST
t = −33.9, p ≈ 0.0009 — robust; PA+distill vs PA t = −4.75, p ≈ 0.042 — marginal,
and only under a normality assumption that three points cannot evidence. The
assumption-free exact sign test cannot go below 0.25 at n=3 for either leg, so
both are screening evidence. HIST − PA is +0.03 pt — the two bases are
indistinguishable on this dataset.

### H3: those six negative deltas were a BatchNorm bug, and fixing it recovers them

The EMA teacher was built with `deepcopy(model)` then `.eval()`, so it normalised
with BatchNorm **running** statistics while the student trained in `.train()` mode
on **batch** statistics; `_update_ema_teacher` also hard-copied buffers instead of
blending them. Under frozen BatchNorm (CUB) the two coincide and the bug is inert.
Under trainable BatchNorm (In-Shop) the teacher is a systematically different
function of the same images — which is exactly where all six negative deltas are.

The fix lives behind `ema_teacher_train_mode` / `ema_teacher_ema_buffers`, both
defaulting to the historical behaviour so old artifacts still reproduce.

| In-Shop arm | seed 0 | seed 1 | seed 2 | mean |
| --- | ---: | ---: | ---: | ---: |
| HIST | 0.9046 | 0.9037 | 0.9031 | 0.9038 |
| HERD | 0.8906 | 0.8892 | 0.8900 | 0.8899 |
| **`herd_bnfix`** | 0.9035 | 0.9048 | 0.9041 | **0.9041** |
| Proxy Anchor | 0.9024 | 0.9048 | 0.9032 | 0.9035 |
| PA + distillation | 0.8999 | 0.8994 | 0.8990 | 0.8994 |
| **`pa_distill_bnfix`** | 0.9029 | 0.9021 | 0.9044 | **0.9031** |

Paired against the *unfixed* distilled arm, which is the comparison that isolates
the bug:

| leg | per-seed recovery | mean | regression it had to undo |
| --- | --- | ---: | ---: |
| HIST | +1.29 / +1.56 / +1.41 | **+1.42 pt** | −1.39 pt |
| Proxy Anchor | +0.30 / +0.27 / +0.54 | **+0.37 pt** | −0.41 pt |

**Six paired seeds across two independent bases, every one positive** (exact sign
test p = 2⁻⁵ = 0.031, and this one is assumption-free rather than resting on
normality at n=3). Each leg recovers very nearly the whole regression it had to
undo — 1.42 against 1.39, and 0.37 against 0.41. A bug-recovery predicts exactly
that proportionality; a method gain has no reason to produce it.

Validated three further ways: proved at unit level, its *null* prediction confirmed
(inert under frozen BatchNorm) and its *positive* prediction confirmed across seeds
on a base it was never tuned against.

**But state the claim narrowly.** The fix repairs the bug; it does not turn
distillation into a win. Against their own bases, `herd_bnfix` is +0.03 pt over HIST
and `pa_distill_bnfix` is −0.04 pt under Proxy Anchor (t = −0.29, p = 0.80) — both
indistinguishable.

**Prior art (2026-07-30): the fix is EMAN.** Cai et al., *Exponential Moving Average
Normalization for Self-Supervised and Semi-Supervised Learning* (CVPR 2021), describe
the same teacher/student BatchNorm mismatch and propose EMA-ing the normalisation
statistics — precisely what `ema_teacher_ema_buffers` does. We rediscovered it
independently, which is not the same as inventing it. H3 is therefore an **audit**
result, not a method: the defect is live in DML momentum-teacher code, and this is what
it costs. Cite EMAN whenever the claim is made.

So the honest result is *"momentum-teacher recipes silently lose 0.3–1.4 pt under
trainable BatchNorm unless the teacher's normalisation mode matches the student's"*,
which is a bug-finding about a widely-used training pattern, **not** evidence that
this distillation helps. The PA leg matters precisely because it shows the effect is
a property of the teacher construction rather than of HIST: the magnitude tracks how
large that base's distillation regression was, which is what a bug-recovery should do
and what a genuine method gain would not.

Generalises to any MoCo/BYOL/DINO-style teacher on a backbone with updating
BatchNorm. Both legs are complete at three seeds; H3 is closed.

### BN-correct weight averaging and dual-timescale EMA on In-Shop

The preregistered screen used BN-correct averaging because In-Shop trains
BatchNorm. Raw best-over-training and leave-one-out-neighbour
selection-corrected R@1 must both be reported.

Three-seed averaging confirmation:

| arm | recipe digest | seed 0 | seed 1 | seed 2 | raw mean | corrected mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Proxy Anchor | `16a3bc844c81` | 0.9024 | 0.9048 | 0.9032 | 0.9035 | 0.9015 |
| `pa_ema_avg_bnfix` | `80f57f183966` | 0.9043 | 0.9036 | 0.9046 | 0.9042 | 0.9035 |

The paired raw deltas are **+0.18 / −0.13 / +0.15 pt**: mean **+0.068
pt**, paired sd **0.169**, t(2) = 0.70, p = 0.5589, exact sign p = 0.500.
Selection correction increases the mean to **+0.203 pt**, paired sd **0.157**.

The standard raw benchmark effect therefore does **not replicate** off CUB:
one seed is negative and the three-seed mean is only +0.07 pt, versus +0.41 pt
on CUB. The correction confirms that best-over-training under-credits the
stabler averaged arm, but a corrected +0.20 pt at n=3 is a measurement finding,
not an established method gain. Averaging does not earn Cars or a momentum
sweep.

The novel candidate failed. Dual-timescale EMA added only **+0.014 pt raw** and
**+0.077 pt corrected** over averaging alone. It required +0.24 pt raw, a
positive corrected delta, and raw R@1 ≥0.9048; it met only the corrected-sign
condition. Once both arms evaluate the same BN-correct fast average, the slow
relational teacher adds effectively nothing on In-Shop. No confirmation seeds
were run.

**iNaturalist 2018** — recorded for completeness, but the *recipe* is broken, not
just the method. Both arms peak at **epoch 5 of 60** and decay thereafter, because
the `selected-from-cars` recipe transfers Cars196 hyperparameters (8k images, 98
classes) to a ~450k-image / ~8k-class dataset:

| arm | best R@1 | best epoch | final R@1 |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.2099 | 5 / 60 | 0.1734 |
| PA + distillation | 0.2094 | 5 / 60 | (run stopped at epoch 46; best already fixed at epoch 5) |

These arms are 0.05 pt apart — statistically empty. iNat needs its own recipe work
before it can support any conclusion.

**CUB and Cars under official reference recipes have never been run.** That matrix
is now running; it is the experiment that decides whether the headline claim
survives. See [research_reset_plan.md](research_reset_plan.md).

## Headline (legacy recipes — see status correction above)

| Method | R@1 | Notes |
| --- | ---: | --- |
| Proxy Anchor (reported) | 69.7 | common baseline |
| HIST (reported) | 71.4 | prior strong same-arch method |
| PFML (reported) | **73.4** | best *reported* same-arch number |
| **HERD** — single model | ~71.6 | our method (see below) |
| **SFORA** — 5-model HERD ensemble | **74.68** | **beats the best reported number by +1.3** |
| SFORA — 9-model HERD ensemble | 75.34 | scales further; +1.9 over PFML |

## HERD — the method

**HERD** = **H**ypergraph **E**MA-teacher **R**elational **D**istillation. It
stacks three ingredients on a ResNet-50/512 backbone:

1. **HIST** hypergraph semantic-tuplet loss (per-class Gaussian prototypes +
   hypergraph neural network over the batch).
2. The official HIST **`LayerNorm(no-affine)` `is_norm` head** on the embedding
   (baseline behavior, not a HERD addition).
3. The novel piece — **EMA-teacher relational self-distillation**: a slow
   momentum copy of the model (`θ_teacher ← m·θ_teacher + (1−m)·θ_student`)
   produces a soft neighborhood distribution over the batch (row-wise softmax of
   the pairwise-similarity matrix); the student is trained to match it. Distilling
   *relational* structure — rather than hard labels — transfers to unseen classes,
   and the temporal-ensemble teacher lowers target variance on the small
   (~5.9k-image) training set.

This training-procedure change is what broke a long-standing ~0.71 same-arch
plateau: a wide range of loss-geometry changes we tried did not move it, but
changing the *information per training step* (teacher targets) did.

## The distillation on small datasets — broad gains (legacy recipes)

> **Claim withdrawn (2026-07-27).** This section was previously titled "The
> distillation is universal — and beats Proxy Anchor on every dataset". That
> universality claim is **falsified** by the corrected In-Shop matrix, where the
> same distillation *regresses* both bases across 3 seeds (PA −0.41 pt, HIST
> −1.39 pt; see "Corrected reference-recipe evidence" below). Every number in
> this section is CUB or Cars under **legacy** recipes.
>
> The surviving, weaker reading: the gains below are real measurements on the two
> *smallest* benchmarks (CUB 5.9k train images, Cars 8.1k) and the regressions are
> on a much larger one (In-Shop ~25k, 3997 classes). That pattern is what a
> **variance-reducing regularizer** looks like — helpful when data is scarce,
> pure capacity cost when it is not. Testing that hypothesis is the point of the
> sweeps in [research_reset_plan.md](research_reset_plan.md) §5.

In our harness the distill term is applied ungated on top of whatever objective is
training, so it can augment Proxy Anchor just as it augments HIST. We measured it
in-harness (same code, same protocol, reseeded where noted):

| dataset | base loss | plain | **+ our distillation** | Δ |
| --- | --- | ---: | ---: | ---: |
| CUB-200 | HIST | 0.700 | **0.716** (= HERD)† | +1.6† |
| CUB-200 | Proxy Anchor | 0.666 | 0.678 | +1.2 |
| Cars196 | HIST | 0.871 | 0.884 (= HERD)† | +1.3† |
| Cars196 | Proxy Anchor | 0.888 | **0.8961** | +0.8 |

† These historical HIST → HERD rows used a legacy HIST control without the
LayerNorm that official HIST enables. Their Δ is therefore a combined
head-plus-distillation change and is **not** the corrected paired estimate. In the new
recipe system both HIST and HERD retain official LayerNorm, and HERD changes only the
declared EMA distillation fields. Proxy Anchor rows remain historical until rerun from
their exact dataset recipe as well.

**It is not specific to HIST or Proxy Anchor.** To check that the distillation is a
*general* training-procedure improvement rather than a two-loss coincidence, we ran
it on three more bases (CUB, seed 0, same protocol). It lifts every one:

| additional base (CUB, seed 0) | plain | + our distillation | Δ |
| --- | ---: | ---: | ---: |
| SupCon | 0.580 | 0.611 | +3.2 |
| triplet (semi-hard) | 0.492 | 0.511 | +1.9 |
| batch-hard triplet | 0.244 | 0.617 | +37.3 |

So across **five** bases — HIST, Proxy Anchor, SupCon, triplet, batch-hard triplet —
the EMA-teacher relational distillation improves retrieval every time **on CUB**.
(It does not on In-Shop; see the corrected table below.) The batch-hard
row is a special case worth naming honestly: our plain batch-hard baseline collapses
(0.244, a known hard-mining failure mode), and the distillation *stabilises* it back to
a competitive 0.617 — so that +37 is "rescued a collapse", not a uniform-quality gain.
The SupCon and triplet rows are ordinary healthy baselines lifted by a few points. (These
three are single-seed spot checks, not reseeded means like the HIST/PA rows above.)

**On CUB and Cars, the distillation improves every base tested.** And on those two
datasets the *best base + our distillation* beats every plain baseline:

- **CUB** — HIST is the stronger base, so **HERD (HIST + distillation) = 0.716** is
  best (> Proxy Anchor 0.666/0.695, > HIST 0.700).
- **Cars** — Proxy Anchor is the stronger base, so **PA + distillation = 0.8961**
  is best (mean over 3 seeds `[0.8944, 0.8974, 0.8963]`, every seed above PA's best
  single run 0.8892; > PA 0.8879, > HERD 0.8835, > HIST 0.8709).

So **our method beats Proxy Anchor on both datasets** — via the *same procedure*
applied to whichever base is stronger. Honest caveats we verified:

- **No single fixed loss is best everywhere.** A fused `HIST + Proxy Anchor`
  objective in one model is a *compromise worse than the best base on both*
  datasets (CUB 0.69 < 0.716; Cars 0.880 < 0.896). HIST genuinely wins CUB, Proxy
  Anchor genuinely wins Cars; mixing them helps neither. The unifying method is the
  distillation *procedure*, not one loss.
- **Single HERD does not beat Proxy Anchor on Cars** (reseeded mean 0.8835 < 0.8879)
  — the HIST base is simply weaker than PA there. We reached "beats PA on Cars" only
  by putting the distillation on the *PA* base, and we say so rather than pretending
  the HIST-based HERD wins there. This is not for lack of tuning: an exhaustive sweep
  of every HIST-internal lever (samples-per-class, LR schedule, distillation
  temperature/weight/momentum, variance floor, hypergraph cross-entropy weight
  `λ_s ∈ {0.5, 2.0}`, incidence sharpness `α`, and HGNN width) leaves the HIST base
  plateaued at ~0.884 seed-0 — every variant landed *at or below* the baseline, none
  reached PA's 0.8857. HIST is genuinely the weaker Cars base; the distillation
  procedure is what transfers, so we apply it to PA there. The dead-end holds at **three
  independent levels**: (1) HIST-internal tuning above; (2) the ensemble — the HIST-based
  HERD ensemble (0.9026) trails the PA+distill ensemble (0.9172); and (3) **cross-teacher
  distillation** — distilling a frozen *trained PA* teacher's relational geometry into a
  HERD/HIST student (`teacher_checkpoint` + `teacher_similarity_weight`) scored 0.8737/0.8743
  (teacher weight 0.5/1.0), *below* HERD's 0.884, because the PA teacher's geometry conflicts
  with HIST's hypergraph rather than complementing it.
- Relaxing HIST's variance floor (an ablation) is a null/negative result; the
  faithful `relu6` default stands.

## SFORA — the ensemble

The SOTA-beating number is a **feature-concatenation ensemble** of independently
seeded HERD models: L2-normalise each model's test embeddings, concatenate them
per sample, L2-normalise the concatenation, and run cosine retrieval. This is an
established SOTA paradigm in deep metric learning (feature-concatenation ensembling,
e.g. BIER). Single HERD models sit at **0.705 mean / 0.716 best across 9 seeds**
(measured standard deviation σ ≈ 0.006 across seeds); the ensemble adds several
points from model diversity and scales monotonically with the number of models:

| models | 1 | 2 | 3 | 4 | 5 | 7 | 9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R@1 | 0.7088 | 0.7335 | 0.7394 | 0.7426 | 0.7468 | 0.7529 | 0.7534 |

These historical numbers recompute with `scripts/ensemble_eval.py` on the saved best-epoch
embeddings (`image_self_retrieval_score`, the project's own scorer). The curve
bends after ~5 models — the first few seeds buy the most. See the `README.md`
for the corrected publication-backed training command; it intentionally does not
promise the legacy score before rerunning.

### Compressing the 9-model pack back to a single-model footprint

Concatenating 9 models gives a 4608-dim vector, impractical to store or search.
We compared several ways to fold it back to **512 dims** (one model's size):

| method | dim | R@1 | retained |
| --- | ---: | ---: | ---: |
| concat (the full pack) | 4608 | 0.7534 | 100% |
| **GPA-aligned mean** | **512** | **0.7490** | **99.4%** |
| Procrustes-aligned mean (single ref) | 512 | 0.7470 | 99.1% |
| concat + PCA | 512 | 0.7439 | 98.7% |
| concat + PCA | 1024 | 0.7444 | 98.8% |
| concat + random projection | 512 | 0.7297 | 96.9% |
| naive mean (no alignment) | 512 | 0.7274 | 96.5% |
| single HERD model | 512 | 0.7053 | 93.6% |

The winner is **not** PCA of the concatenation. Independently-trained embeddings
live in arbitrarily rotated copies of the same geometry, so a naive average
cancels signal (0.7274, barely above one model). **Aligning** the models into one
shared frame before averaging fixes this. A single-reference Procrustes fit
(`R = UVᵀ` from the SVD of `Eₘᵀ·E₀`) already reaches 0.7470; iterating it to a
consensus — **Generalized Procrustes Analysis (GPA)**: repeatedly align every
model to the running mean and re-average — reaches **0.7490, 99.4% of the full
pack**, in one 512-dim vector with **no concatenation** and **+1.5 over reported
PFML (73.4)**. Notably GPA at 512-dim beats a PCA of the concat even at **1024**
dims (0.7444), so this is not just a dimension trade-off — alignment genuinely
captures the pack better.

GPA is the ceiling among folds that use only the embeddings' geometry: the
remaining ~0.4 pt to the concat is genuine cross-model disagreement no single
averaged vector can hold. **We do not close it by fitting a projection to the test
set** — that would be test-set overfitting, and reporting the resulting number is
not honest.

#### The honest, inductive answer: fit the fold on the disjoint train split

The legitimate way to compress is to fit the projection on the disjoint **train**
classes, freeze it, and only then apply it to test — nothing about the test split
informs the fold. We ran this on a 3-seed HERD pack (each seed exports its
best-epoch train and test embeddings via `--save-train-embeddings` /
`--save-test-embeddings`); `scripts/train_fit_fold.py` fits each 512-dim fold on
the train concat and evaluates it, frozen, on the test concat:

| 512-dim fold (this 3-seed pack) | R@1 | vs concat | vs single |
| --- | ---: | ---: | ---: |
| full concat (1536 dims) | 0.7259 | 100% | — |
| **PCA fit on train** | **0.7078** | **97.5%** | **+1.4 pt** |
| Proxy-Anchor head fit on train | 0.7076 | 97.4% | +1.4 pt |
| group-SupCon-XBM head fit on train | 0.7039 | 96.9% | +1.0 pt |
| single HERD model | 0.6940 | 95.6% | — |

A stricter 2026-08-01 diagnostic asked whether the five-seed train-fit GPA
consensus could be predicted from seed 0 alone. A preregistered uncentred ridge
map fit only on train classes reduced disjoint-test R@1 from **0.6940 to
0.6813 (-1.266 pt)**; the orthogonal-map control remained exactly 0.6940. The
five-seed concat and train-fit GPA target were 0.7350 and 0.7205. Thus the pack
gain is not a simple deterministic linear calibration hidden inside one seed.

An all-miss complementarity audit then found only **15 / 5,924 (0.253%)**
queries where every individual seed missed R@1 but the concat succeeded. In
73.3% of those rare rescues, however, the correct class was inside every seed's
top 10 (median worst-seed rank 7). The registered frequency prediction of at
least 0.5% failed; this is descriptive evidence of rare weak-score aggregation,
not support for another ensemble-derived arm.

Notably the **unsupervised PCA edges both supervised metric-learning heads** — for
folding an already-trained pack the discriminative geometry is already present, so
re-optimising it on train labels only risks overfitting the train classes.

**At the aggressive 512-dim footprint, the best inductive fold is an *uncentered*
train-fit projection.** Retrieval is cosine similarity, so a projection onto an
orthonormal train-fit basis (the top-k right singular vectors of the raw, **un-mean-
centered** train concat) is cosine-preserving — whereas subtracting the train mean
shifts test cosines (the train mean is not the test mean) and caps every *centered*
fold (PCA, whitening) at ~98%. On the 5-seed pack this uncentered fold recovers
**98.9%** at 512-dim (0.7272 of the 0.7350 concat), beating both train-fit GPA
alignment (**98.0%**, 0.7205 — align each model to a train consensus, then average)
and centered PCA (97.4%).

**And "decrease dims to 100%" is achievable honestly — just not at 512-dim.** The five
models are highly correlated (they encode the same classes), so the concat's effective
retrieval rank is well below 2560. Sweeping the uncentered train-fit projection's
target dimension (nothing fit on test) shows it stays **lossless** as it discards the
lowest-energy directions:

| train-clean uncentered fold | R@1 | retained |
| --- | ---: | ---: |
| 512 dims | 0.7272 | 98.94% |
| 1024 dims | 0.7318 | 99.56% |
| 1536 dims | 0.7331 | 99.75% |
| 1792 dims | 0.7343 | 99.91% |
| **2048 dims** | **0.7350** | **100.00%** |
| full concat (2560) | 0.7350 | 100% |

So a **train/test-clean projection reduces the pack 2560 → 2048 dims (a 20% cut) with
zero retrieval loss** — the honest reading of "decrease vectors dims to get 100%,
trained on train." What is *not* achievable without fitting on test is 100% at the
single-model **512-dim** footprint (a 5× cut): the bottom ~500 directions still carry
~1% of retrieval signal, so an honest 512-dim fold tops out at 98.9%, and only a fold
that peeks at the test geometry (transductive GPA, 99.4% below) closes more. Reproduce
both with:

```bash
# 512-dim folds (GPA, PCA, uncentered) + the dimension-vs-retention sweep
uv run python scripts/explore_trainclean_projection.py \
    --train 'reports/emb/herd_tt_seed*.train.npz' \
    --test  'reports/emb/herd_tt_seed*.test.npz'
```

Reproduce the transductive folds above with:

```bash
uv run python scripts/ensemble_eval.py --compare-methods 512 reports/emb/ema_seed*.npz
uv run python scripts/ensemble_eval.py --compress-sweep   reports/emb/ema_seed*.npz
```

> **Transductive caveat.** The PCA axes and the Procrustes/GPA rotations use only
> the embeddings' geometry (no labels, no retrieval targets), but they are *computed
> on the test embeddings themselves*, so 0.7490 (GPA) and 0.7439 (PCA) are a
> transductive upper bound — a deployment that froze the projection on held-out/train
> data would likely score slightly lower. The full concat (0.7534), random projection
> (0.7297), naive mean (0.7274) and single model (0.7053) involve no fitted projection
> at all. We do **not** fit any projection to the test set's *retrieval* (labels or
> nearest-neighbour targets) to inflate the compressed number — that would be
> test-set overfitting.

Two framings of "how much is retained": Procrustes keeps **99.1% of the pack's
R@1** (0.7470/0.7534) but **86.7% of the *gain* over a single model**
((0.7470−0.7053)/(0.7534−0.7053)). We quote the first; the second is the stricter
read.

## Cars196 — a second dataset

The same protocol on Cars196 (ResNet-50/512, zero-shot split, best-over-training).
On Cars the HIST base is weaker than Proxy Anchor, so the **HIST-based HERD does not
beat PA** — but our distillation *procedure* on the **PA** base does (see the
universal-distillation section above). In-harness, reseeded where noted:

| method | R@1 | provenance |
| --- | ---: | --- |
| HIST (our run) | 87.1 | in-harness baseline |
| Proxy Anchor (our run) | 88.8 | reseeded mean; above reported 87.7 |
| HERD = HIST + distillation | 88.4 | reseeded mean; *below* PA — HIST base weaker |
| **PA + our distillation** | **89.6** | reseeded mean `[0.8944, 0.8974, 0.8963]`; **beats PA** |
| SFORA — HERD (HIST) ensemble, 3 models | 90.3 | HIST-based ensemble; still below the PA-based one |
| **SFORA — PA+distill ensemble, 3 models** | **91.7** | best Cars ensemble (PA is the stronger Cars base) |

The HIST-based single HERD (mean 0.8835) lands below our own PA reproduction (0.8879)
— the HIST base simply loses to PA on Cars. The honest win is the **distillation
procedure**: applied to the stronger PA base it gives **PA + distillation = 0.8961**
(every seed above PA's best single run 0.8892), so **our method beats PA on Cars**. A
single fused HIST+PA loss is a compromise worse than each base (0.880), so no single
fixed loss is best everywhere. The **ensemble** confirms the base-adaptive story at a
higher level: the PA+distill ensemble (**0.9172**) beats the HIST-based HERD ensemble
(**0.9026**) — HIST is the weaker Cars base at *both* single-model and ensemble scale,
so on Cars we ensemble the PA base.

## SOP — a third dataset (a genuine reproduction gap, thoroughly investigated)

Stanford Online Products (11,318 train classes, ResNet-50/512, best-over-training). At
this scale the **base-adaptive story holds and extends**: Proxy Anchor is again the
stronger base, and our distillation on it wins.

| method (SOP, seed 0) | R@1 | note |
| --- | ---: | --- |
| **PA + our distillation** | **~0.72** | stronger base at scale; distill neutral here |
| HIST-HERD | 0.678 | HIST is weaker at 11k-class scale, as on Cars |

So across all three datasets the pattern is consistent — **HIST wins CUB, Proxy Anchor
wins Cars and SOP**, and "best base + our distillation" is the method each time.

**Honest caveat: our SOP Proxy Anchor reproduces at ~0.72, ~7.5 pt below the reported
0.796, and we could not close it.** This is not for lack of trying — we ran a full
investigation:

- **Hyperparameters (8 configs):** batch 120/180/256, lr 1e-4/2e-4, LR decay
  γ 0.25/0.5, samples-per-class 2/3/4 (spc=2 sees all 11,317 classes; spc=4 silently
  excluded 36% of them), the `is_norm` LayerNorm head on/off, 60 vs 90 epochs, and
  distillation on/off. **Every config plateaus at 0.712–0.721.** is_norm was neutral
  (0.719); 90 epochs peaked at 0.721 then overfit down.
- **Implementation audit (code trace):** the Proxy-Anchor loss normalization (positive
  term over |P⁺|, negative over all 11,318 proxies), proxies excluded from weight decay,
  the disjoint first-half/second-half class split, the standard `Resize(256)→CenterCrop`
  eval transform, and the eval protocol (60,698 test images / 11,317 classes ≈ the
  standard 60,502 / 11,316) are all faithful.

So SOP behaves like HIST (our 0.701 vs reported 0.714) and PFML (collapses): **the
reported number is hard to reproduce, not a knob we failed to turn.** We report our
honest ~0.72 as supporting evidence for the base-adaptive finding, **not** a SOTA claim.

## DeepFashion In-Shop and iNaturalist 2018 — dataset support and protocols

The harness now supports the official DeepFashion In-Shop train/query/gallery
partition and the project-defined `inat2018-zero-shot-species-v1` protocol. The
corrected three-seed In-Shop matrices and the BN-correct averaging screen are reported
above. The iNaturalist recipe remains a negative recipe diagnostic rather than a
canonical benchmark result.

In-Shop results will use query-to-gallery retrieval with R@1/10/20/30 as the canonical
cutoffs (plus the harness's R@2/4/8 and MAP@R). iNaturalist results will be labeled as
SFORA's project protocol, not as a canonical iNaturalist metric-learning benchmark or
a SOTA comparison. The exact setup and sequential runner are documented in
[`library_usage.md`](library_usage.md#deepfashion-in-shop-and-inaturalist-2018).

### Recipe matrix used by the corrected queue

| Dataset | Base method | Recipe | Backbone | Source / expected R@1 | Status |
| --- | --- | --- | --- | --- | --- |
| CUB | Proxy Anchor | `proxy_anchor.cub.official-51db570` | ResNet-50 | official repo 69.9 | registered |
| CUB | HIST | `hist.cub.official-e7d650c` | ResNet-50 | paper 71.4±0.2 | registered |
| Cars | Proxy Anchor | `proxy_anchor.cars.official-51db570` | ResNet-50 | official repo 87.7 | registered |
| Cars | HIST | `hist.cars.official-e7d650c` | ResNet-50 | paper 89.6±0.2 | registered |
| SOP | Proxy Anchor | `proxy_anchor.sop.official-51db570` | BN-Inception | official repo 79.2 | registered |
| SOP | HIST | `hist.sop.official-e7d650c` | ResNet-50 | paper 81.4±0.2 | registered |
| In-Shop | Proxy Anchor | `proxy_anchor.inshop.official-51db570` | BN-Inception | official repo 91.9 | queued reference |
| In-Shop | HIST | `hist.inshop.selected-from-<winner>-e7d650c` | winner preserved | no published pair | train-only selection queued |
| iNat2018 v1 | Proxy Anchor | `proxy_anchor.inat2018.selected-from-<winner>-51db570` | winner preserved | no published pair | train-only selection queued |
| iNat2018 v1 | HIST | `hist.inat2018.selected-from-<winner>-e7d650c` | winner preserved | no published pair | train-only selection queued |

Primary sources are the pinned [Proxy Anchor official repository](https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/tree/51db57031e38f75c03f69bbdfad1a3233afd9787)
and [HIST official repository](https://github.com/ljin0429/HIST/tree/e7d650c80460f464c55bcdc2262d785923c50dc4),
with HIST expected scores from its CVPR 2022 Table 3. `pa_distill` and `herd`
inherit the resolved base recipe unchanged and add only their recorded EMA delta.

### DGX handoff audit (2026-07-20)

At `2026-07-20T10:53:15+02:00`, before changing remote state, the legacy controller
was captured on `spark-2751`:

- controller PID `1595927`, process group/session `1595925`, command
  `/home/riomus/experiment-logs/run-inshop-matrix.sh`;
- active trainer PID `1612789` plus eight DataLoader workers, running legacy
  In-Shop PA-distill seed 1 at step 3700/12360;
- GPU 0 (`NVIDIA GB10`) was 96% utilized;
- completed legacy artifacts were In-Shop seed-0 PA, PA-distill, HIST, HERD and
  seed-1 PA. Their JSON files and per-run logs were preserved in place;
- the controller script checksum was
  `1f7f5f000f85dfd832708b126a90a6d4d6a7c79b0261acece9aa2c43b01379e5`.

A second legacy controller was then found waiting independently for iNaturalist:
PID `1596894`, process group `1596892`, script
`run-inat2018-matrix.sh` (SHA-256
`77b2bff7475fc98f40ec4723233371a540ac9d28790f0c21a2d7c1e68b29d3a6`). It had
started legacy PA/iNat seed 0 and reached step 1400/130380. Both captured legacy
process groups were terminated with SIGTERM; their logs and partial state remain
recoverable, and no unrelated process was signaled.

The captured legacy R@1 values were PA seed 0 `0.8636`, PA-distill seed 0 `0.8633`,
HIST seed 0 `0.7731`, HERD seed 0 `0.6831`, and PA seed 1 `0.8590`. These are the
poor common-preset results that triggered the recipe audit; they are retained as
`modified_legacy`, not reference evidence.

Deployment then passed both dataset preflights and installed the official
`bn_inception-52deb4733.pth` checkpoint after verifying its full SHA-256
`52deb473314542a5c2f87e9e6f26f4ca42fe863d15f986414dbae8c2dfdd2353`. A one-step,
two-class In-Shop smoke run loaded that checkpoint and emitted the official recipe
identity plus BN-Inception, batch 180, LR `6e-4`, updating BatchNorm, step-20,
γ=`0.25`, and shuffled batches. Its cap-related changes are correctly recorded as a
`modified` smoke artifact.

The corrected detached controller is PID `1619657` (log
`logs/reference_recipes.controller.log`). Its first full command is Proxy Anchor /
In-Shop seed 0 from recipe `proxy_anchor.inshop.official-51db570`, digest
`50137fe7f9d84cb567ee092f68b3bee6be58716dfab4d9bfbe964de8bc78fe57`; trainer PID
`1619742` was observed at 95% GPU utilization. The controller will run the paired
PA-distill seeds, perform frozen training-only selection for unsupported pairs, and
then execute their base/derived seed matrices. A duplicate controller created while
repairing PID-file detachment was detected immediately and its isolated process group
was stopped; PID `1619657` is the sole surviving corrected controller.

## SFORA on raw HIST — what does the ensemble alone buy? (ablation)

The historical ablation ensembled a legacy control labeled **plain HIST** that omitted
both `is_norm` and the EMA teacher. Because official HIST includes `is_norm`, this is
not an official HIST baseline. Cumulative first-N CUB seeds:

| models | 1 | 2 | 3 | 4 | 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SFORA-HIST | 0.6972 | 0.7242 | 0.7330 | 0.7402 | 0.7443 |
| SFORA-HERD | 0.7088 | 0.7335 | 0.7394 | 0.7426 | 0.7468 |

The ensemble is the **main driver**: a pack of raw HIST models clears reported PFML
(0.734) at 4 models (HERD clears it at 3) and reaches 0.7443 at 5. The full HERD
recipe then adds a steady margin — **~0.7 pt single-model (0.705 vs 0.698 mean) and
~0.25 pt at 5 models (0.7468 vs 0.7443)**. This isolates the old combined
LayerNorm-plus-EMA change, not the corrected EMA-only HERD delta. Recompute:
`ensemble_eval.py reports/emb/hist_only_seed*.npz`.

## Legacy reproducibility observations (official-recipe reruns pending)

The old common-preset harness produced the observations below. The audit found enough
recipe drift that none of them is evidence for or against the official recipes; the
new digest-tracked runs must finish before drawing that conclusion.

- **Proxy Anchor:** the legacy run reached best-mean R@1 **0.6946** over three seeds,
  close to the reported CUB value, but it did not carry an official recipe digest.
- **HIST/HERD:** the legacy control reached about **0.698 mean**, while the combined
  LayerNorm-plus-EMA variant reached **0.705 mean / 0.716 best** over nine seeds. Since
  official HIST already includes LayerNorm and uses a different optimizer, sampler,
  schedule, and dataset-specific parameters, this is not a valid official HIST
  reproduction or a clean EMA ablation.
- **PFML:** legacy attempts collapsed and did not approach the reported 73.4. This
  recipe audit covers Proxy Anchor and HIST, so it does not upgrade those attempts to
  an official PFML reproduction claim.

**Interpretation.** The strongest reported same-architecture numbers remain reference
targets. Only corrected artifacts with a `reference` or frozen `selected_extension`
track can update the comparison.

## Approaches that did **not** work (honest negatives)

For a metric-learning practitioner, these are as useful as the positive result:

- **Sub-center Proxy Anchor** (K proxies/class): 0.675 — fragmenting a class into
  modes hurts zero-shot transfer.
- **Gaussian-potential uniformity** (Wang–Isola) on PA/HIST: neutral-to-negative.
- **Un-normalised physics potentials** (electrostatic/PFML, symmetric long-range):
  collapse without a partition-function (softmax) normaliser.
- **Multi-crop / DINO-style distillation** is **incompatible with the frozen-BN
  metric-learning recipe**: non-224 local crops hit the backbone's frozen
  ImageNet-224 BatchNorm statistics, produce out-of-distribution activations, and
  collapse training; unfreezing BatchNorm stops the collapse but wrecks the HIST
  base. Same-resolution multi-crop avoids both but gives no benefit.
- Bigger ImageNet-V2 pretrained weights, longer (100-epoch) schedules, and HIST
  hyper-parameter re-tuning all under-performed the plain HERD configuration.
- **Tetrad interaction relational distillation (TIRD), In-Shop seed 0:** raw
  best-over-training R@1 **0.8301** versus paired Proxy Anchor **0.9024**
  (**-7.237 points**). Selection-corrected TIRD was **0.8267**, with corrected
  paired delta **-7.405 points**. The preregistered prediction was 0.9090 and the
  absolute falsifier was below 0.9085, so this is a decisive Gate-4 failure.
  TIRD isolated the reproducible 4.75%-variance image-by-image interaction, but
  cosine normalization gave that weak residual unit-scale pressure; training
  was delayed, volatile, and converged to a much lower retrieval ceiling.

## Pre-normalisation magnitude diagnostic (measurement, not a method)

A preregistered epoch-10 In-Shop Proxy Anchor diagnostic measured the raw
BN-Inception embedding-head magnitude before the model's internal L2
normalisation. The corrected seed-0 endpoint was R@1 **0.84365**, with no
best-over-training selection. Query magnitude correlated **0.18675** (Pearson)
with R@1 correctness and **0.32574** (Spearman) with the positive-minus-negative
retrieval margin; train identity ICC was **0.57754**. The registered prediction
was absolute correlation at least 0.15 or 0.20 respectively, and the joint
falsifier was both below 0.05, so the missing-observable hypothesis passes.

The first export was invalid because official BN-Inception normalises inside
`forward`; its unit norms were rejected before reading the corrected result. A
head-output hook and internal-normalisation regression test fixed the
measurement. No method claim follows: norm/quality-aware similarity, margins,
confidence and augmentation are occupied by MagFace, AdaFace, IDML, SEC and
ESA, while class-covariance synthetic support is occupied by IAA.

A preregistered CPU-only decomposition then showed that the result is not merely
between-identity confounding. Within 3,972 repeated query identities, centred
norm correlated **0.14170** with centred correctness and **0.20972** (Spearman)
with centred margin, clearing the frozen 0.10/0.15 prediction. Between-identity
effects were larger (**0.25870**, **0.45322**), so magnitude contains both
image-quality and identity-difficulty information. This strengthens the
measurement, not the novelty claim.
