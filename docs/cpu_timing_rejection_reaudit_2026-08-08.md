# CPU timing rejection re-audit

## Trigger

Wall-clock CPU measurements collected while unrelated jobs ran concurrently are not
valid comparative cost evidence.  This audit asks whether any candidate was stopped
because of such a measurement and therefore needs to be reopened.  The replacement
cost criterion is theoretical operator complexity plus matched-compute benchmark
controls, not elapsed time on a contended host.

## Result

**No candidate is reopened.**  The ledger contains no current rejection whose
scientific disposition depends wholly or materially on concurrent CPU wall time.

* Hard CE-BN explicitly discards its CPU timing.  It remains closed because its paired
  In-Shop result is `-23.196` raw-best R@1 points and `-22.936` final points.
* Soft CE-BN remains closed at `-2.595` raw-best points; CEGT remains closed at
  `+0.007` raw-best points against a preregistered `+0.30`-point condition; ARSN
  remains closed because held-out moment prediction has aggregate R² `-0.0157`
  (linear) and `-0.0306` (MLP).  None is a timing decision.
* Proposals 50, 169, and 209 mention compute cost, but each also fails an independent
  mechanism/prior-art gate.  Their cost objections are theoretical (multiple
  teachers, second-order differentiation, or roughly sixteen views), not measured
  host throughput.
* Passes 79 and 80 use a `<=1 CPU-hour` bound only for a hypothetical future revival
  diagnostic.  GCNP and ETSD were already closed by algebra, measured mechanism, and
  prior art; the bound did not decide their status.
* Resource-serialization audit 276 labels CPU contention as an operational defect and
  explicitly states that no method or result changed.

## Process rule

Do not use CPU/GPU wall-clock gathered under uncontrolled concurrent load to rank or
reject methods.  Use symbolic asymptotic complexity, counted forward/backward passes,
stored-state size, and a prospectively matched-compute control.  Timing can be reported
only after resource isolation is verified.  This correction does not turn an
independent algebraic, novelty, or benchmark failure into a false negative.
