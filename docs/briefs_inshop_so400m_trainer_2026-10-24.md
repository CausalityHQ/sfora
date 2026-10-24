# Brief: In-Shop So400m 512-D TRAIN pilot trainer (source only)

Base: branch devbox/sfora-successor-20261013 at this file's commit. Make a new branch devbox/inshop-so400m-trainer-20261024, ONE commit, push, report SHA.
No GPU training. Run checks only on the DGX (ssh riomus@100.104.199.68; /home/riomus/group-learning/.venv/bin/python; CPU, CUDA_VISIBLE_DEVICES=""). Local host = git/editing only. No new deps, do not modify existing files, never read In-Shop query/gallery splits.

## Goal
A TRAIN-only In-Shop fine-tuning pilot for SigLIP2 So400m (google/siglip2-so400m-patch14-384) with 512-D head at crop 256 or 384, ported from the SOP pair trainer.

## Read
- scripts/train_sop_siglip2_substrate_pair.py — SOP trainer, hard-coded to SOP (59,551/53,700/5,851, UNICOM npz, bank-cost receipt). Never edit.
- scripts/train_sop_so400m_384_dim512_pilot.py, scripts/train_sop_so400m_res280_dim512_pilot.py, scripts/train_sop_so400m_res280_pilot.py — so400 arm, 512-D head, crop + pos-embed interpolation (only when crop != 384).
- scripts/screen_inshop_so400m_resolution_v1.py — exact rows/labels/partition: parse_inshop_partition(root) split=='train' (25,882 rows, 3,997 products); label = index in sorted unique label strings; products with >=2 images -> deterministic_class_partition(fit_fraction=0.5, seed=179019); single-image products always fit. Expect fit 12,761 / held 13,121. Ordered-rows sha 2d43ad5cee694b2f38d7e3d806c5f983b9161ede48b54bd9cc7841b31d53ee5c. DGX root /home/riomus/datasets/inshop_official_standard.
- Pretrained features (head PCA init + bank seeds), (25882,1152) f32, same row order: /home/riomus/runs/sfora-inshop-res-screen-v1/r256_train_features.npy (sha e119f6721210963fbbfcca7bf6e049d6ed80dd6e30454b5940acc68a9d95d5e1), r384_train_features.npy (sha b56aec2e3a2ff19860131b2ef71800f23c0a12fdb6d2fd8643eba3fda0cdf6d3).

## Deliverable: scripts/train_inshop_so400m_pilot.py
Parametrized copy allowed; keep the recipe identical (loss, member bank, optimizer/LR, so400 frozen blocks, batch composition, 1000 updates, fp16-autocast export, packed int8 scorer: held vs ALL 25,882 TRAIN rows, self excluded).
CLI: --crop {256,384} --seed (179024) --updates (1000) --dataset-root --model-snapshot --pretrained-features --output-dir.
Replace SOP-only pins with sha256 checks of the features file and the ordered-rows hash.
Outputs: checkpoint.pt {'vision','head'} loadable like src/sfora/so400m_compact_serving.py from_checkpoint; train_embeddings.npy (unit f32, all rows); receipt.json with quality.recall_at_1, quality.per_query_r1 (held order), quality.per_query_ap (mAP@R per held query), quality.output_width=512, substrate_config.crop, train wall seconds, peak CUDA bytes. Must fit one GB10 (SOP 384/512 peaked 35 GB).

## Check: scripts/test_train_inshop_so400m_pilot.py
Small CPU check (toy data / monkeypatched encoder): partition logic incl. singletons, head 512, scorer R@1 and mAP@R on a toy example. Run on DGX, paste output.

## Report
Commit SHA, files, test output, every recipe deviation from the SOP trainer listed explicitly.
