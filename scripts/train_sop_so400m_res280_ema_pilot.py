#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@280 recipe plus an EMA shadow of the trainable weights (decay 0.996).

Training is byte-for-byte the res280 recipe (train_sop_so400m_res280_pilot.py, d0568eee): the EMA is a passive
post-step copy (torch global optimizer step post-hook) of vision + head parameters that require grad, so the
final checkpoint, receipt and embeddings are those of a normal res280 run. After main() returns, the EMA weights
are written to <output-dir>/ema_state.pt with a sidecar ema_receipt.json.

Run at seed 179025 so the final checkpoint is also a second-seed replicate of the res280 recipe (seed 179024
receipt 9e4a05db), reported but not gating.

Pre-registered decision (scored by screen_sop_so400m_res280_ema_v1.py; held 5,851 vs full 59,551 TRAIN gallery,
packed int8, paired product bootstrap 5000 seed 179019, EMA vs the same run's final weights):
  ADVANCE iff dR@1 lower95 > 0 AND dmAP@R upper95 > 0 (mAP not significantly harmed)
          => ONE official read of the EMA checkpoint at 280; retained iff official packed R@1 > 92.594.
  else KILL. Source: literature scan 2026-10-24 rank 3 (SOP-specific EMA evidence UNVERIFIED; expected < +0.3pp).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_res280_pilot as base  # noqa: E402

pair = base.pair
DECAY = 0.996
CAPTURED: dict[str, torch.nn.Module] = {}
EMA: dict[str, torch.Tensor] = {}
STEPS = [0]

_init_head = pair.initialize_head_and_classifier


def _capture_head(*args, **kwargs):
    head, classifier, pca_sha = _init_head(*args, **kwargs)
    CAPTURED["head"] = head
    return head, classifier, pca_sha


pair.initialize_head_and_classifier = _capture_head

from transformers import AutoModel  # noqa: E402

_from_pretrained = AutoModel.from_pretrained.__func__


def _capture_model(cls, *args, **kwargs):
    model = _from_pretrained(cls, *args, **kwargs)
    CAPTURED["vision"] = model.vision_model
    return model


AutoModel.from_pretrained = classmethod(_capture_model)


def _trainable():
    for prefix in ("vision", "head"):
        for name, param in CAPTURED[prefix].named_parameters():
            if param.requires_grad:
                yield f"{prefix}.{name}", param


@torch.no_grad()
def _ema_step(optimizer, args, kwargs):
    STEPS[0] += 1
    for name, param in _trainable():
        if name not in EMA:
            EMA[name] = param.detach().float().clone()
        else:
            EMA[name].lerp_(param.detach().float(), 1.0 - DECAY)


def main():
    out = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    updates = int(sys.argv[sys.argv.index("--updates") + 1])
    handle = torch.optim.optimizer.register_optimizer_step_post_hook(_ema_step)
    pair.main()
    handle.remove()
    assert STEPS[0] == updates, (STEPS[0], updates)
    state = {}
    for prefix in ("vision", "head"):
        full = {k: v.detach().cpu().clone() for k, v in CAPTURED[prefix].state_dict().items()}
        for name, value in EMA.items():
            if name.startswith(prefix + "."):
                key = name[len(prefix) + 1:]
                assert full[key].shape == value.shape and full[key].dtype == value.dtype
                full[key] = value.cpu()
        state[prefix] = full
    torch.save({**state, "decay": DECAY, "steps": STEPS[0]}, out / "ema_state.pt")
    receipt = json.loads((out / "receipt.json").read_text())
    sidecar = {"schema": "sfora-sop-ema-shadow-v1", "decay": DECAY, "steps": STEPS[0],
               "ema_parameters": len(EMA), "ema_elements": sum(v.numel() for v in EMA.values()),
               "ema_state_sha256": pair.sha256(out / "ema_state.pt"),
               "training_receipt_sha256": pair.sha256(out / "receipt.json"),
               "checkpoint_sha256": receipt["checkpoint_sha256"],
               "wrapper_sha256": pair.sha256(Path(__file__).resolve())}
    (out / "ema_receipt.json").write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n")
    print(json.dumps(sidecar), flush=True)


if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400_280"
    main()
