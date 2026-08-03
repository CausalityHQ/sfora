# SOP structure evidence atomic-publication repair

**Status: repaired and deployed before any final SOP artifact existed.**

The final embedding exporters, joint verifier, fragmentation measurement, and
fragmentation support verifier already publish through same-filesystem temporary files
and atomic rename. A fresh audit of the actual deployed controller chain found that
`analyze_sop_official_structure.py` and `analyze_sop_proxy_clock.py` instead called
`Path.write_text` directly on their final paths.

A path-based waiter could therefore observe a final filename after it had been opened
but before it contained complete JSON. The long fragmentation analysis made premature
In-Shop launch unlikely, but timing is not evidence integrity: a reader could fail on
partial JSON, and the filename did not mean publication was complete.

Both scripts now render to a unique same-directory temporary file, flush and `fsync`
it, and atomically replace the requested final path. Cleanup is fail-safe. A unit test
proves that publication leaves one parseable final JSON and no temporary sibling. The
scientific calculations, thresholds, inputs, and output schema are unchanged.

