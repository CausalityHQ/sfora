# Pass 45 invalid-proposal incident

Date: 2026-08-06 UTC  
Frozen prompt: `docs/opus_blind_prompt_pass45_2026-08-06.txt`  
Parent consultation: `0cc0d06fdb0948bd`  
Unauthorized nested consultation: `340476e4e010483b`

## Verdict

**INVALID before proposal freeze; not a candidate.** The fallback proposer did
not follow the frozen blind-proposal protocol. Its visible action stream said:

> Blind proposer dispatched to Fable (id `340476e4e010483b`). Reading the local
> protocol files while it runs.

The frozen prompt explicitly prohibited inspecting the local filesystem,
repository, history, prior conversations, or process state, and required the
consulted model itself to return one method or NONE. It also said not to launch
another consultation or agent. The fallback violated both restrictions by
starting a nested consultation and reading local protocol files. The parent
and nested jobs were cancelled immediately. No answer was frozen, no proposal
was audited, and no GPU or implementation followed.

This incident must not be counted as an independent invention pass or as
evidence for or against any method. A replacement pass needs a newly frozen
prompt that permits primary-source web search but explicitly disables all
filesystem and devbox-cross/agent/consultation actions; its answer must be
returned directly by the one durable job started by the protocol controller.
