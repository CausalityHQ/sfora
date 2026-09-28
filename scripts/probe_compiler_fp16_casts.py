"""Synthetic fused residual/LN diagnostic; no images or retrieval claims."""

import json
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F


def residual_norm(a, b, c, d, weight, bias):
    return F.layer_norm(((a + b) + c) + d, (1024,), weight, bias, 1e-6)


@torch.inference_mode()
def main():
    output = Path(sys.argv[1])
    assert not output.exists()
    torch.manual_seed(179059001)
    torch.backends.cuda.matmul.allow_tf32 = False
    values = tuple(torch.randn((32, 257, 1024), dtype=torch.float16, device="cuda") for _ in range(4))
    weight = torch.ones(1024, dtype=torch.float16, device="cuda")
    bias = torch.zeros_like(weight)
    args = (*values, weight, bias)
    expected = residual_norm(*args)
    results = {}
    for name, options in (("default", {}), ("preserve_casts", {"emulate_precision_casts": True})):
        started = time.monotonic()
        actual = torch.compile(residual_norm, fullgraph=True, dynamic=False, options=options)(*args)
        torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        results[name] = {
            "mismatched_entries": int((actual != expected).sum()),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
            "rms_error": float((actual.float() - expected.float()).square().mean().sqrt()),
            "compile_and_call_seconds": time.monotonic() - started,
        }
    default, fixed = results["default"], results["preserve_casts"]
    promising = default["mismatched_entries"] > 0 and default["max_abs_error"] > 0 and fixed["rms_error"] <= .2 * default["rms_error"] and fixed["max_abs_error"] <= .2 * default["max_abs_error"]
    report = {"schema": "sfora-compiler-fp16-casts-v1", "seed": 179059001,
              "shape": list(values[0].shape), "entries": expected.numel(), "results": results,
              "decision": "CAST_MECHANISM_SUPPORTED_ONLY" if promising else "KILL_CAST_REPAIR",
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
              "limitations": "Synthetic diagnostic, not actual model/packed parity, recall or latency."}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
