#!/bin/bash
R=/home/riomus/runs/sfora-inshop-cutlass-serving-v1; X=/home/riomus/runs
run() { flock -n $X/.sfora-siglip2-gpu.lock flock -n /home/riomus/.sfora-siglip2-gpu.lock systemd-run --user --wait --collect --pipe -p MemoryMax=48G -p MemorySwapMax=0 -p RuntimeMaxSec=5400 --working-directory=$R/scripts --setenv=PYTHONPATH=$R/src:$R/scripts --setenv=HF_HUB_OFFLINE=1 --setenv=SFORA_CUTLASS_FP8_LIB=$R/libcutlass_fp8.so /home/riomus/group-learning/.venv/bin/python $R/scripts/check_inshop_serving_v3.py --fp8 --encoder $1 --run-dir $X/sfora-inshop-so400m-full-c384-179024-2000-v1 --model-snapshot /home/riomus/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f --dataset-root /home/riomus/datasets/inshop_official_standard --output $R/$1.json; }
run cutlass; run fused_mx
echo ICDONE
