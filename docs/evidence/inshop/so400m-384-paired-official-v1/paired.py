import json, numpy as np
R = "/home/riomus/runs/"
lab = np.load(R + "sfora-inshop-so400m-full-c384-179024-1000-v1/official_packed.npz")["query_labels"]
q = lambda p: json.load(open(p))["quality"]
arms = {"s384_179024": q(R + "sfora-inshop-so400m-full-c384-179024-1000-v1/receipt.json"),
        "s384_179025": q(R + "sfora-inshop-so400m-full-c384-179025-1000-v1/receipt.json"),
        "large_256_128d": q(R + "sfora-inshop-paired-v1/large.json"), "so400_256_128d": q(R + "sfora-inshop-paired-v1/so400_256.json")}
_, inv = np.unique(lab, return_inverse=True); cnt = np.bincount(inv).astype(float)
rng = np.random.default_rng(179019); picks = rng.integers(0, len(cnt), size=(5000, len(cnt)))
def paired(a, b):
    d = np.bincount(inv, weights=np.asarray(a) - np.asarray(b)); boot = d[picks].sum(1) / cnt[picks].sum(1) * 100
    return [round(float((np.asarray(a) - np.asarray(b)).mean() * 100), 3), round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)]
out = {"means": {k: [v["recall_at_1"], v["map_at_r"]] for k, v in arms.items()},
       "ap_corr_large_vs_s384": float(np.corrcoef(arms["large_256_128d"]["per_query_ap"], arms["s384_179024"]["per_query_ap"])[0, 1])}
for c in ("s384_179024", "s384_179025"):
    for b in ("large_256_128d", "so400_256_128d"):
        out[f"{c}-{b}"] = {"r1": paired(arms[c]["per_query_r1"], arms[b]["per_query_r1"]), "map": paired(arms[c]["per_query_ap"], arms[b]["per_query_ap"])}
print(json.dumps(out, indent=1))
