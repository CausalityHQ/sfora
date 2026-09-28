"""Frozen paired direct128/256 mechanics or qualified100-update quality gate."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import signal
import torch
from compare_inshop_sop_warmstart_100 import packed_quality
from score_inshop_crop_view_pair import roles, bootstrap_lower, QUERY_SHA, GALLERY_SHA, PARTITION_SHA
from preflight_inshop_siglip2_unseen_gallery import split
from sfora.unicom_inshop import parse_inshop_partition
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

import numpy as np
from score_inshop_crop_view_pair import sha256


def main():
    started = time.monotonic()
    budget = 100
    def deadline(_signal, _frame):
        raise TimeoutError(f"external{budget}-second direct-width deadline")
    signal.signal(signal.SIGTERM, deadline)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "model-snapshot", "features-dir", "preflight", "cached-gate", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--quality-qualification", type=Path)
    args = parser.parse_args()
    if sha256(args.preflight) != "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034":
        raise ValueError("direct-width preflight authority differs")
    args.output_dir.mkdir(exist_ok=False)
    quality_mode = args.quality_qualification is not None
    budget = 600 if quality_mode else 100
    if quality_mode and sha256(args.quality_qualification)!="5334209bf78c70b08e0dd20bd55572e57130f2b31ef5570ebdc2ce5c535b2333":
        raise ValueError("direct-width quality qualification differs")
    result = {"schema":"sfora-inshop-direct-width-mechanics-v1", "claim_eligible":False,
              "quality_measured":False, "serving_latency_measured":False, "arms":{},
              "source_sha256":sha256(Path(__file__))}
    if quality_mode:
        result.update(schema="sfora-inshop-direct-width-quality100-v1")
        torch.backends.cuda.matmul.allow_tf32=False
        assert sha256(args.dataset_root/"Eval/list_eval_partition.txt")==PARTITION_SHA
        train=tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split=="train")
        _,held=split(tuple(row.label for row in train))
        held_labels=tuple(train[i].label for i in held)
        query,gallery=roles(held_labels,tuple(train[i].image_path for i in held),args.dataset_root)
        assert (len(query),len(gallery),len(held))==(6354,6245,12599)
        import hashlib
        assert hashlib.sha256(np.asarray(query,dtype="<i4").tobytes()).hexdigest()==QUERY_SHA
        assert hashlib.sha256(np.asarray(gallery,dtype="<i4").tobytes()).hexdigest()==GALLERY_SHA
        qlabels=np.asarray([held_labels[i] for i in query])
    try:
        for width in (128, 256):
            remaining = budget-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError("paired direct-width deadline exceeded")
            destination = args.output_dir / str(width)
            command = [sys.executable, str(Path(__file__).with_name("train_inshop_siglip2_unseen_gallery.py"))]
            for key in ("dataset-root", "model-snapshot", "features-dir", "preflight"):
                command.extend([f"--{key}", str(getattr(args,key.replace('-','_')))])
            command.extend(["--preflight-sha256",sha256(args.preflight),
                            "--direct-width-qualification" if quality_mode else "--direct-width-receipt",
                            str(args.quality_qualification if quality_mode else args.cached_gate),
                            "--arm","freeze_emb","--updates","100" if quality_mode else "17","--seed","179024","--workers","4",
                            "--training-width",str(width),"--output-dir",str(destination)])
            with (args.output_dir / f"{width}.log").open("wb") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=min(280,remaining) if quality_mode else remaining)
            arm = json.loads((destination/"receipt.json").read_text())
            if (not arm["direct_width_smoke"] or arm["updates"]!=(100 if quality_mode else 17) or arm["width_fold"] is not None
                or (not quality_mode and (arm["held_values_sha256"] is not None or arm["quality"] is not None))):
                raise ValueError("direct-width smoke performed the wrong experiment")
            result["arms"][str(width)] = arm
            if quality_mode:
                values_path=destination/"held_values.npy"
                assert sha256(values_path)==arm["held_values_sha256"]
                values=np.load(values_path,allow_pickle=False)
                assert values.shape==(12599,width) and values.dtype==np.float32
                packed=pack_int8_unit_embeddings(torch.from_numpy(values))
                arm["held_packed_sha256"]={}
                for name,array in (("codes",packed.codes.numpy()),("inverse_norms",packed.inverse_norms.numpy())):
                    path=destination/f"held_{name}.npy"
                    np.save(path,array)
                    arm["held_packed_sha256"][name]=sha256(path)
                arm["asymmetric_quality"]=packed_quality(values,held_labels,query,gallery)
                result["quality_measured"]=True
                if width==128 and arm["asymmetric_quality"]["recall_at_1"]<.922:
                    raise ValueError("native100 fails recipe-validity floor before candidate")
            print(json.dumps({"terminal_width":width,"elapsed":time.monotonic()-started}),flush=True)
        a,b=result["arms"]["128"],result["arms"]["256"]
        steps=100 if quality_mode else 17
        criteria={"same_pixels":a["first_input_batch_sha256"]==b["first_input_batch_sha256"] and len(a["first_input_batch_sha256"])==steps,
                  "same_pca_rows":a["direct_initial_rows_sha256"]==b["direct_initial_rows_sha256"],
                  "median_step":np.median(b["step_seconds"])<=1.10*np.median(a["step_seconds"]),
                  "peak_training_cuda":b["training_peak_cuda_allocated_bytes"]<=a["training_peak_cuda_allocated_bytes"]+2**30,
                  "whole_budget":time.monotonic()-started<=budget}
        criteria["same_protocol"]=all(a[k]==b[k] for k in (
            "source_files_sha256", "fit_rows_sha256", "held_rows_sha256", "preflight_sha256",
            "model_file_sha256", "executed_schedule_sha256", "features_sha256", "seed", "arm",
            "updates", "vision_lr", "batch_size", "rank_coefficient"))
        for width,arm in result["arms"].items():
            criteria[width+"_frozen"]=arm["mechanics_frozen_sha256"]==arm["mechanics_terminal_frozen_sha256"]
            criteria[width+"_trained"]=arm["mechanics_initial_trainable_sha256"]!=arm["mechanics_terminal_trainable_sha256"]
            criteria[width+"_geometry"]=all(arm["width_terminal_geometry"][k]>=.5*arm["width_initial_geometry"][k] for k in ("variance","effective_rank"))
            criteria[width+"_reload"]=all(arm["private_native_fp16_reload"]["checks"].values())
            criteria[width+"_stable"]=len(arm["all_step_losses"])==steps and bool(np.isfinite(arm["all_step_losses"]).all()) and len(arm["step_seconds"])==steps
            grads=np.asarray([[step["group_gradient_norms"][name] for name in ("vision","head","classifier")] for step in arm["width_history"]])
            criteria[width+"_gradients"]=grads.shape==(steps,3) and bool(np.isfinite(grads).all() and (grads>0).all())
            checkpoint=args.output_dir/width/"checkpoint.pt"
            criteria[width+"_checkpoint"]=sha256(checkpoint)==arm["checkpoint_sha256"]==arm["private_native_fp16_reload"]["checkpoint_sha256"]
            criteria[width+"_fixture"]=sha256(args.output_dir/width/"native_fp16_fit_fixture.pt")==arm["private_native_fp16_reload"]["fixture_sha256"]
        if quality_mode:
            criteria["train_wall"]=b["training_wall_including_member_bank_init_seconds"]<=1.1*a["training_wall_including_member_bank_init_seconds"]
            result["deltas"]={}
            for name,field,floor in (("recall","per_query_r1",.005),("map","per_query_ap",.01)):
                delta=np.asarray(b["asymmetric_quality"][field])-np.asarray(a["asymmetric_quality"][field])
                lower=bootstrap_lower(delta,qlabels)
                result["deltas"][name]={"point":float(delta.mean()),"lower95":lower,"upper95":-bootstrap_lower(-delta,qlabels)}
                criteria[name]=delta.mean()>=floor and lower>0
        result["criteria"]={k:bool(v) for k,v in criteria.items()}
        result["decision"]=("GO_CONFIRM_DIRECT_WIDTH_QUALITY" if all(criteria.values()) else "KILL_DIRECT_WIDTH_QUALITY100") if quality_mode else ("GO_FREEZE_REAL_100_GATE" if all(criteria.values()) else "KILL_DIRECT_WIDTH_MECHANICS")
    except Exception as error:
        result.update(decision="KILL_DIRECT_WIDTH_EXECUTION",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["whole_seconds"]=time.monotonic()-started
        (args.output_dir/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="arms"}),flush=True)


if __name__=="__main__":
    main()
