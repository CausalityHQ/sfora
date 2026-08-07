# Pass 78 independent review — MCB-PA

**Verdict: DEAD; no GPU.** The reviewer found MCB-PA is a fourth instance of
the same deterministic per-block recoding of class labels already killed in
Pass 39 (FCS), Pass 46 (FPC), and Pass 51 (PSPL); PSPL even includes the same
equal-energy penalty.

The rank premise is not a causal deficit: for any loss depending on `(x,y)`,
the class-mean configuration rank is bounded by `C−1`, regardless of how the
descriptor is sliced. `J·G ≥ k` only covers the span of learned proxy vectors,
not label-bearing rank. A frozen code `π(y)` carries no information beyond the
original label. The claim that random coarse codes cannot become class
detectors is false: class-first lookup composed with `π` is a global minimizer,
and the fine Proxy Anchor term already trains that detector. This predicts
redundant blocks; the proposal checks only block energy and removes the
decorrelation mechanism used by A-BIER.

Closest primary prior art includes BIER/A-BIER (Opitz et al., ICCV 2017), DREML
(Xuan et al., ECCV 2018), ECOC (Dietterich & Bakiri, JAIR 1995), and split-space
DML. No Gate-1 repository measurement supports the claimed missing rank; the
existing DOIR null-space control and candidate 371 reject that premise. The
proposal therefore fails Gates 1 and 2 and does not cross the matched frontier.
