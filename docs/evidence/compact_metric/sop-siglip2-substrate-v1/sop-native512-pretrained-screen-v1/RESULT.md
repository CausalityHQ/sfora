# Pretrained SigLIP2 So400m native-512 (patch16, 1024 tokens) vs native 384 on SOP (proposals v3 #1 stage 0) — CLOSE
scripts/screen_sop_so400m_native512_v1.py (pre-registered 20beb2d6): packed PCA-128, SOP held split.
n384: full-gallery R@1 80.52, holdout-only 91.52; n512: 80.38 / 91.75. Δ full −0.14 pp (bar +1.0) → CLOSE.
SOP images: long side median 400 px, 0.9 % >= 512 px, so 512 input is upsampled and adds no detail.
