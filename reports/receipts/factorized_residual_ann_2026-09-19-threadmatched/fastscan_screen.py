"""Screen Faiss IVFPQ fast-scan (4-bit SIMD) scan throughput against Sfora's.

Normalised by rows actually scanned so a 10M index screens a 100M claim.
"""
import json, os, resource, time
import faiss, numpy as np

BASE = "/home/riomus/datasets/bigann/base.first10m.raw.u8bin"
QUERY = "/home/riomus/datasets/bigann/query.public.10K.u8bin"
GT = "/home/riomus/datasets/bigann/gt.first10m.top100.npz"
ROWS, DIM = 10_000_000, 128
NLIST, NPROBE = 4096, 40          # ~1% of corpus scanned, matching Sfora's 640/65536
NQ, RETURN = 200, 100
THREADS = int(os.environ.get("THREADS", "20"))

faiss.omp_set_num_threads(THREADS)
base = np.memmap(BASE, np.uint8, "r", offset=8, shape=(ROWS, DIM))
queries = np.ascontiguousarray(base[:0].astype(np.float32))  # placeholder
q = np.memmap(QUERY, np.uint8, "r", offset=8, shape=(10_000, DIM))
queries = np.ascontiguousarray(q[:NQ], dtype=np.float32)
truth = np.load(GT)["arr_0"] if "arr_0" in np.load(GT) else None

quantizer = faiss.IndexFlatL2(DIM)
index = faiss.IndexIVFPQFastScan(quantizer, DIM, NLIST, 32, 4)
rng = np.random.default_rng(0)
sample = np.ascontiguousarray(base[rng.choice(ROWS, 400_000, replace=False)], dtype=np.float32)
t0 = time.perf_counter()
index.train(sample)
train_s = time.perf_counter() - t0
del sample

t0 = time.perf_counter()
CH = 500_000
for start in range(0, ROWS, CH):
    index.add(np.ascontiguousarray(base[start:start + CH], dtype=np.float32))
add_s = time.perf_counter() - t0
index.nprobe = NPROBE
index.parallel_mode = 1

for i in range(20):
    index.search(queries[i:i + 1], RETURN)

stats = faiss.cvar.indexIVF_stats
stats.reset()
lat = []
t0 = time.perf_counter()
cpu0 = resource.getrusage(resource.RUSAGE_SELF)
hits = 0
for i in range(NQ):
    s = time.perf_counter_ns()
    _, ids = index.search(queries[i:i + 1], RETURN)
    lat.append(time.perf_counter_ns() - s)
    if truth is not None:
        hits += len(set(ids[0].tolist()) & set(truth[i][:RETURN].tolist()))
wall_s = time.perf_counter() - t0
cpu1 = resource.getrusage(resource.RUSAGE_SELF)
cpu_s = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)

ndis = int(stats.ndis)
lat = np.asarray(lat, dtype=np.int64)
out = {
    "index": "IndexIVFPQFastScan PQ32x4",
    "threads": THREADS, "nlist": NLIST, "nprobe": NPROBE, "rows": ROWS,
    "train_seconds": round(train_s, 1), "add_seconds": round(add_s, 1),
    "queries": NQ,
    "rows_scanned_per_query": ndis // NQ,
    "mean_ms": round(float(lat.mean()) / 1e6, 4),
    "p99_ms": round(float(np.quantile(lat, 0.99, method="higher")) / 1e6, 4),
    "rows_per_second_aggregate": int(ndis / wall_s),
    "cpu_ms_per_query": round(cpu_s * 1000 / NQ, 3),
    "rows_per_cpu_second": int(ndis / cpu_s) if cpu_s > 0 else None,
    "compressed_only_recall_at_100": round(hits / (NQ * RETURN), 6) if truth is not None else None,
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
}
print(json.dumps(out, indent=2))
open(f"/tmp/fastscan-screen-t{THREADS}.json", "w").write(json.dumps(out, sort_keys=True) + "\n")
