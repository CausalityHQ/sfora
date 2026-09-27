#!/usr/bin/env bash
set -euo pipefail

run=/home/riomus/runs/sfora-inshop-equal-wall-v1
python=/home/riomus/group-learning/.venv/bin/python
preflight=/home/riomus/runs/inshop-unseen-gallery-preflight-179024-v2.json
model=/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c
dataset=/home/riomus/datasets/inshop_official_standard
export PYTHONPATH="$run:$run/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

test "$(sha256sum "$run/train_inshop_siglip2_unseen_gallery.py" | cut -d' ' -f1)" = de7a3fde54decd78c0c42aa29973c8c714d01c4a85a6a6ce1d19358642470801
test "$(sha256sum "$run/src/sfora/unicom_inshop.py" | cut -d' ' -f1)" = 635d300c66b2bebe7753707c9196d4735a97f07c7dd999f186f9b9b1251b7f1f
test "$(sha256sum "$preflight" | cut -d' ' -f1)" = f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034
test ! -e "$run/control-1000"
test ! -e "$run/freeze-1533"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

"$python" - "$dataset" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
lines = (root / "Eval/list_eval_partition.txt").read_text().splitlines()[2:]
digest = hashlib.sha256()
for line in lines:
    relative = line.split()[0]
    path = root / "Img" / relative
    with path.open("rb") as stream:
        image_digest = hashlib.file_digest(stream, "sha256").digest()
    digest.update(relative.encode() + b"\0" + path.stat().st_size.to_bytes(8, "little") + image_digest)
assert len(lines) == 52_712
assert digest.hexdigest() == "608373be84bc4e5b95e3c6f87e712e4d1ca53299e6f59d1627d033a70128f8fe"
print(f"standard In-Shop pixel manifest: {digest.hexdigest()}", flush=True)
PY

for spec in control:1000 freeze_emb:1533; do
    arm=${spec%%:*}
    updates=${spec#*:}
    output="$run/control-1000"
    if [[ $arm == freeze_emb ]]; then output="$run/freeze-1533"; fi
    echo "START arm=$arm updates=$updates" >&2
    "$python" "$run/train_inshop_siglip2_unseen_gallery.py" \
        --dataset-root "$dataset" --model-snapshot "$model" \
        --features-dir /home/riomus/runs/sfora-inshop-siglip2-train-features-v1 \
        --preflight "$preflight" --preflight-sha256 f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034 \
        --output-dir "$output" --seed 179024 --arm "$arm" --updates "$updates"
    echo "DONE arm=$arm receipt_sha256=$(sha256sum "$output/receipt.json" | cut -d' ' -f1)" >&2
done
