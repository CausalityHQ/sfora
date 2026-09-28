"""Fresh paired direct128/256 mechanics only, with a100-second whole deadline."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import signal

import numpy as np
from score_inshop_crop_view_pair import sha256


def main():
    started = time.monotonic()
    def deadline(_signal, _frame):
        raise TimeoutError("external100-second direct-width deadline")
    signal.signal(signal.SIGTERM, deadline)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "model-snapshot", "features-dir", "preflight", "cached-gate", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.preflight) != "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034":
        raise ValueError("direct-width preflight authority differs")
    args.output_dir.mkdir(exist_ok=False)
    result = {"schema":"sfora-inshop-direct-width-mechanics-v1", "claim_eligible":False,
              "quality_measured":False, "serving_latency_measured":False, "arms":{},
              "source_sha256":sha256(Path(__file__))}
    try:
        for width in (128, 256):
            remaining = 100-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("paired direct-width deadline exceeded")
            destination = args.output_dir / str(width)
            command = [sys.executable, str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py"))]
            for key in ("dataset-root", "model-snapshot", "features-dir", "preflight"):
                command.extend([f"--{key}", str(getattr(args,key.replace('-','_')))])
            command.extend(["--preflight-sha256",sha256(args.preflight),"--direct-width-receipt",str(args.cached_gate),
                            "--arm","freeze_emb","--updates","17","--seed","179024","--workers","4",
                            "--training-width",str(width),"--output-dir",str(destination)])
            with (args.output_dir / f"{width}.log").open("wb") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=remaining)
            arm = json.loads((destination/"receipt.json").read_text())
            if (not arm["direct_width_smoke"] or arm["updates"]!=17 or arm["width_fold"] is not None
                or arm["held_values_sha256"] is not None or arm["quality"] is not None):
                raise ValueError("direct-width smoke performed the wrong experiment")
            result["arms"][str(width)] = arm
            print(json.dumps({"terminal_width":width,"elapsed":time.monotonic()-started}),flush=True)
        a,b=result["arms"]["128"],result["arms"]["256"]
        criteria={"same_pixels":a["first_input_batch_sha256"]==b["first_input_batch_sha256"] and len(a["first_input_batch_sha256"])==17,
                  "same_pca_rows":a["direct_initial_rows_sha256"]==b["direct_initial_rows_sha256"],
                  "median_step":np.median(b["step_seconds"])<=1.10*np.median(a["step_seconds"]),
                  "peak_training_cuda":b["training_peak_cuda_allocated_bytes"]<=a["training_peak_cuda_allocated_bytes"]+2**30,
                  "whole_budget":time.monotonic()-started<=100}
        criteria["same_protocol"]=all(a[k]==b[k] for k in (
            "source_files_sha256", "fit_rows_sha256", "held_rows_sha256", "preflight_sha256",
            "model_file_sha256", "executed_schedule_sha256", "features_sha256", "seed", "arm",
            "updates", "vision_lr", "batch_size", "rank_coefficient"))
        for width,arm in result["arms"].items():
            criteria[width+"_frozen"]=arm["mechanics_frozen_sha256"]==arm["mechanics_terminal_frozen_sha256"]
            criteria[width+"_trained"]=arm["mechanics_initial_trainable_sha256"]!=arm["mechanics_terminal_trainable_sha256"]
            criteria[width+"_geometry"]=all(arm["width_terminal_geometry"][k]>=.5*arm["width_initial_geometry"][k] for k in ("variance","effective_rank"))
            criteria[width+"_reload"]=all(arm["private_native_fp16_reload"]["checks"].values())
            criteria[width+"_stable"]=len(arm["all_step_losses"])==17 and bool(np.isfinite(arm["all_step_losses"]).all()) and len(arm["step_seconds"])==17
            grads=np.asarray([[step["group_gradient_norms"][name] for name in ("vision","head","classifier")] for step in arm["width_history"]])
            criteria[width+"_gradients"]=grads.shape==(17,3) and bool(np.isfinite(grads).all() and (grads>0).all())
            checkpoint=args.output_dir/width/"checkpoint.pt"
            criteria[width+"_checkpoint"]=sha256(checkpoint)==arm["checkpoint_sha256"]==arm["private_native_fp16_reload"]["checkpoint_sha256"]
            criteria[width+"_fixture"]=sha256(args.output_dir/width/"native_fp16_fit_fixture.pt")==arm["private_native_fp16_reload"]["fixture_sha256"]
        result["criteria"]={k:bool(v) for k,v in criteria.items()}
        result["decision"]="GO_FREEZE_REAL_100_GATE" if all(criteria.values()) else "KILL_DIRECT_WIDTH_MECHANICS"
    except Exception as error:
        result.update(decision="KILL_DIRECT_WIDTH_EXECUTION",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["whole_seconds"]=time.monotonic()-started
        (args.output_dir/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="arms"}),flush=True)


if __name__=="__main__":
    main()
