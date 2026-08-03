# SOP atomic evidence publication audit 277

Date: 2026-08-03. Repaired while corrected SOP was still training and before
the joint-audit or fragmentation output paths existed.

## Defect

The fragmentation controller used shell redirection directly to its final JSON:

```text
python measure_spectral_class_connectivity_275.py ... > final.json
```

The shell creates `final.json` before Python starts. Because this analysis makes
roughly 1.82 trillion multiply-accumulates, the path could exist empty or
partial for a long time. The independent verifier waited only for file
existence and could therefore consume invalid JSON before the producer
finished. The joint SOP verifier had the same logical race over a much shorter
window because it wrote its final JSON directly with `Path.write_text`.

The chains were fail-closed in the sense that a parse failure would prevent
In-Shop from starting, but they could discard a valid expensive measurement
and leave a misleading final-looking path. Existence is not completion.

## Repair

The joint verifier now renders to a sibling `.tmp` path and atomically replaces
the final output only after verification succeeds. The fragmentation controller
now:

1. waits for and parses a verified joint audit;
2. writes the locked diagnostic's stdout to a PID-specific temporary sibling;
3. parses that temporary JSON and requires the registered core fields;
4. renames it to the final result path only after success;
5. removes only its exact temporary sibling on failure.

The locked diagnostic itself and its preregistered SHA-256 are unchanged. The
superseded fragmentation controller was terminated after proving no final
result existed; replacement PID `3138562` was armed. The already-waiting joint
controller required no restart because it had not imported the verifier and now
resolves the replaced `/tmp/verify_sop_final_artifacts.py` only after packs
exist.

## Boundary

Atomic rename proves publication completeness, not scientific correctness. The
joint verifier still checks artifact identity/splits/hashes, and the independent
fragmentation support verifier still recomputes graph exposure. This repair is
an evidence-path correction, not a method candidate or metric change.
