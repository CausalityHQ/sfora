# FGVC-Aircraft compact-metric evidence

`aircraft-compact-metric-gate.json` is the sealed summary for the fresh,
class-disjoint FGVC-Aircraft replication. The files in `scripts/` preserve the
exact experiment sources as byte evidence; their names end in `.txt` so normal
package and test discovery cannot execute scratch code with historical `/tmp`
paths. Their SHA-256 digests are recorded in the receipt.

`compact-metric-public-encode-gb10-v1.json` records the post-review public
`CompactMetricModule.encode()` measurement and an interleaved manual-operation
control. It binds both the release module and its exact benchmark source by
SHA-256.

The feature archive, learned checkpoint, raw result, and teacher checkpoint are
not vendored. Their exact byte lengths and SHA-256 authorities remain in the
receipt. Reproduction therefore requires independently acquiring and
authenticating those artifacts before adapting the recorded scratch paths.
