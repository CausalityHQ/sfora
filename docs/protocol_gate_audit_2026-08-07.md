# Protocol gate audit — why the search is producing almost no survivors

The current seven-gate protocol is good at preventing inflated benchmark claims,
but two gates are over-pruning discovery. This is a process audit, not a method
result; it does not retroactively revive any dead candidate.

## Gate 1 is too narrow for architecture and optimizer ideas

Requiring a prior repository measurement that directly identifies the causal
error mode is appropriate for a new supervision target. It is too strong for a
new activation, architecture, or optimizer: those mechanisms can be hypotheses
about inductive bias whose decisive evidence is a prospective matched control.
The present rule makes the search preferentially generate variants of whatever
has already been measured and rejects genuinely novel objects before they can
be falsified.

Proposed interpretation: retain Gate 1 as a requirement for a *quantitative
anchor*, but allow one of three anchors:

1. an existing repository measurement (strong provenance);
2. a reproducible zero-training/short-training diagnostic that measures the
   claimed defect on the exact corrected dataset; or
3. a theorem-level mechanism prediction with a cheap diagnostic that can kill
   it before a full run.

No headline result is allowed from (2) or (3) without the same controls and
replication. The untrained diagnostic must not be substituted for the operating
point, as the RSPG specification defect demonstrated.

## Gate 2 is conflating adjacency with occupation

The current ledger often treats a method as dead because a nearby paper uses
the same broad words—graph, topology, augmentation, normalization, or
distillation—even when the supervised object and decision rule differ. That
is safe for avoiding overclaims but systematically rejects combinations that
could be empirically novel. A mechanism-level prior should be a hard death only
when the cited work implements the same training object and the same causal
action. A nearby operator should instead produce **LIVE-NARROW** with a required
control that isolates the difference.

The distinction is visible in the repository itself: several independent cold
reviews returned LIVE narrowly, while local resolution later killed them for
provenance or degeneracy. Binary Gate-2 decisions erase that useful middle
state.

## Gate 4 must separate screening from a frontier claim

The corrected In-Shop reference is a paired, current-corpus control, not a
single absolute SOTA floor. A candidate can be worth a powered confirmation
because it improves that control by a preregistered effect while still falling
short of a higher-capacity literature number (for example, a 2048-D or
transformer lane). Conversely, using a literature frontier as a one-seed kill
threshold can discard a real, capacity-matched improvement before its variance
is measurable. Gate 4 should therefore kill only a prospectively large miss
against the paired control; Gate 5/7 decide whether a surviving delta is a
claim and whether it reaches the relevant capacity lane.

## Gates that should remain hard

- Gate 0 artifact and dataset validation: keep hard; a wrong corpus invalidates
  both positive and negative results.
- Gate 3 preregistration: keep hard; it is the protection against tuning after
  seeing results.
- Gate 4 corrected In-Shop screen: keep as the first benchmark, but one seed
  should only kill a large miss, never establish a small gain.
- Gate 5 out-of-sample confirmation, Gate 6 raw plus final/independently
  selected reporting, and Gate 7 second-dataset replication: keep hard for any
  SOTA or mechanism claim.

## Operating rule for the next search pass

I will not weaken the reporting standard or run an occupied method. I will,
however, classify a candidate as **LIVE-NARROW** when Gate 2 finds only an
adjacent operator, preregister the decisive matched control, and let a cheap
exact-dataset diagnostic decide whether GPU work is warranted. A candidate that
fails the diagnostic or its control is recorded dead plainly. This preserves
falsifiability while avoiding the current “every family is already occupied”
absorbing state.
