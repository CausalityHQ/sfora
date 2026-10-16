# Chunk500 checkpoint-I/O exactness falsifier: PASS

Source de08475359e6c8c1a278d86132c5ecd730c84139 (origin/devbox/full-chunk500-source-v2-20261015;
parent 289d18a3, 7 new v2 files, v1 bytes unchanged). Staged execution.json 49237c20.
Review: dual critique e356614702f7476b (engineering GO, no blocker; research critic lost to provider limit).

Unit: ONE Large v2 training unit 0->500 (CHUNK=500) from the s179032 v9r1 pilot source,
same systemd/flock/watchdog/lifetime guards, MemoryMax 8G, no swap. Lifetime check passed
(driver pid 2752800 in session-4381.scope). Service runtime 23min 17.380s, rc 0, admitted
terminal f7b0c0a6... . No official read.

Falsifier (CPU systemd unit, invocation 7dd6d384..., 45 s): `pass: true` against archived v1
chunk-0500 (terminal d48aec87...). Every numerical checkpoint component (vision, head, classifier,
bank, optimizer, scaler, counter, CPU/CUDA RNG, ...) equal by typed fingerprint; all step rows
1..500 equal; identity equal minus {schema, code}. raw_terminal_hash_equal=false as expected.

Timing, same arm/steps: v1 five 100-update units = 31:13 unit runtime (32:01 wall);
v2 one unit = 23:17, saving ~8.7 min per 500 steps (~131 s per removed checkpoint boundary).
Projected full paired campaign: 40 -> 8 units, 32 boundaries x ~131 s ~ 1.16 h saved
(4.61 h -> ~3.45 h), with bit-identical training.

Decision: future full-scale campaigns use the v2 CHUNK=500 tooling. No quality claim.
The receipt is stored gzipped; its original sha256 is in chunk-0500-receipt.json.sha256.
