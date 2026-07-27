"""Is the antihub population a real, learnable target - or just noise?

5-8% of CUB test images appear in NOBODY's top-10. Mean R@1 hides them entirely.
Before proposing any method, three things have to be true for it to be worth doing:

  1. STABILITY  - are the same images antihubs across independently trained models?
                  If antihubs are random per model, there is nothing to learn.
  2. COST       - do antihubs actually fail retrieval themselves, or are they merely
                  unpopular as neighbours while still finding their own class?
  3. HEADROOM   - how much R@1 is sitting in that population?
"""

import glob

import numpy as np


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def analyse(emb, labels, k=10):
    x = unit(emb)
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    nn = np.argpartition(-s, k, axis=1)[:, :k]
    occurrence = np.bincount(nn.ravel(), minlength=len(x)).astype(int)
    top1 = np.argmax(s, axis=1)
    correct = labels[top1] == labels
    return occurrence, correct


packs = {
    "HIST": sorted(glob.glob("reports/emb/hist_only_seed*.npz")),
    "HERD": sorted(glob.glob("reports/emb/ema_seed*.npz")),
}

for name, files in packs.items():
    if len(files) < 3:
        continue
    occs, corrects, labels = [], [], None
    for f in files:
        d = np.load(f)
        o, c = analyse(d["embeddings"], d["labels"])
        occs.append(o)
        corrects.append(c)
        labels = d["labels"]
    occs = np.array(occs)
    corrects = np.array(corrects)
    n_models, n = occs.shape

    is_anti = occs == 0
    per_model_rate = is_anti.mean(axis=1)
    times_anti = is_anti.sum(axis=0)

    print(f"\n=== {name}  ({n_models} models, {n} images) ===")
    print(f"  antihub rate per model: {100 * per_model_rate.mean():.1f}%")

    # 1. STABILITY: how often is an antihub in one model an antihub in the others?
    ever = (times_anti > 0).mean()
    always = (times_anti == n_models).mean()
    expected_always = float(np.prod(per_model_rate))  # if independent across models
    print(f"  antihub in >=1 model: {100 * ever:.1f}%   in ALL {n_models}: {100 * always:.2f}%")
    print(
        f"     if independent, all-{n_models} would be {100 * expected_always:.4f}%"
        f"  -> {'STABLE population' if always > 10 * expected_always else 'looks random'}"
    )

    # 2. COST: do antihubs get their OWN retrieval right?
    anti_correct = corrects[is_anti].mean()
    hub_correct = corrects[~is_anti].mean()
    print(
        f"  R@1 of antihubs {anti_correct:.4f}   vs non-antihubs {hub_correct:.4f}"
        f"   (gap {100 * (hub_correct - anti_correct):+.1f} pt)"
    )

    # 3. HEADROOM: total R@1 recoverable if antihubs matched everyone else.
    share = is_anti.mean()
    headroom = share * (hub_correct - anti_correct)
    print(f"  headroom if antihubs matched the rest: {100 * headroom:+.2f} pt R@1")
