"""
Layer A, experiment 5 - does workload size change the speedup?

Fixed costs of multiprocessing (spawning workers, setting up the pool) are paid
once per run regardless of how much work follows. On a small workload they are a
large share of the total; on a large one they amortise away. So the same worker
count should look better as the workload grows.

An early pilot measured this at n=1 and the presentation cites those numbers.
This runs it properly: 4 workload sizes x 3 configurations x 2 payloads x 3
repeats, with the page cache warmed and the configuration order randomised.

OMP_NUM_THREADS is pinned to 1, as in every other Layer A experiment, so numpy
cannot add hidden parallelism.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import gc
import glob
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
from PIL import Image

SIZES = [500, 2000, 8000, 27000]   # 27000 yields 25,500 after stratification
CONFIGS = [("sequential", 0), ("process", 4), ("process", 8)]
PAYLOADS = ["array", "scalar"]
DATASETS = ["RGB", "MS"]
REPEATS = 3
CHUNKSIZE = 64
SEED = 42
OUT = "results/layerA"
BAR = "=" * 78
_DS = None


def _decode(path):
    if _DS == "RGB":
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    else:
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
    a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-6)
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def decode_array(path):
    return _decode(path)


def decode_scalar(path):
    return float(_decode(path).mean())


FN = {"array": decode_array, "scalar": decode_scalar}


def find_dataset(ext):
    hits = sorted(glob.glob(os.path.join("data/ext", "**", "*." + ext), recursive=True))
    if not hits:
        return None, []
    root = os.path.dirname(os.path.dirname(hits[0]))
    return root, [h for h in hits if h.startswith(root + os.sep)]


def stratified(root, ext, classes, n):
    """Keep the class balance at every workload size."""
    per = n // len(classes)
    out = []
    for c in classes:
        out += sorted(glob.glob(os.path.join(root, c, "*." + ext)))[:per]
    return out


def prime(files):
    def rd(p):
        with open(p, "rb") as f:
            return len(f.read())
    with ThreadPoolExecutor(max_workers=8) as ex:
        return sum(ex.map(rd, files, chunksize=64))


def run_one(files, mode, workers, payload):
    fn = FN[payload]
    gc.collect()
    t0 = time.perf_counter()
    if mode == "sequential":
        for p in files:
            fn(p)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(fn, files, chunksize=CHUNKSIZE):
                pass
    return time.perf_counter() - t0


def main():
    global _DS
    print(BAR)
    print("LAYER A, EXPERIMENT 5 - speedup vs workload size")
    print(BAR)
    print(f"sizes={SIZES}  configs={CONFIGS}  payloads={PAYLOADS}  repeats={REPEATS}")
    print(f"cores: {os.cpu_count()} logical   OMP_NUM_THREADS=1")

    rows = []
    for ds in DATASETS:
        _DS = ds
        ext = "jpg" if ds == "RGB" else "tif"
        root, allf = find_dataset(ext)
        if not allf:
            print(f"skipping {ds}")
            continue
        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        print("")
        print(BAR)
        print(f"{ds}  ({root})")
        print(BAR)

        for n in SIZES:
            files = stratified(root, ext, classes, n)
            mb = prime(files) / 1024 ** 2
            plan = []
            for rep in range(REPEATS):
                block = [(m, w, p) for m, w in CONFIGS for p in PAYLOADS]
                random.Random(SEED + rep).shuffle(block)
                plan += [(m, w, p, rep) for m, w, p in block]
            print(f"  N={len(files):6d}  ({mb:7.1f} MB warmed)  {len(plan)} runs")
            for mode, w, payload, rep in plan:
                t = run_one(files, mode, w, payload)
                rows.append({"dataset": ds, "n_images": len(files), "mode": mode,
                             "workers": w, "payload": payload, "repeat": rep,
                             "total_s": round(t, 4),
                             "throughput_ips": round(len(files) / t, 1)})

        # ---- per-dataset summary ----
        g = defaultdict(list)
        for r in rows:
            if r["dataset"] != ds:
                continue
            g[(r["n_images"], r["mode"], r["workers"], r["payload"])].append(r["total_s"])
        print("")
        print(f"  speedup vs sequential at the SAME workload size and payload ({ds})")
        print(f"  {'N':>7}{'payload':>9}{'seq s':>9}{'p4 s':>8}{'p8 s':>8}"
              f"{'p4 spd':>9}{'p8 spd':>9}")
        # Use the sizes actually produced, not the requested ones. Stratified
        # sampling caps at the smallest class, so asking for 27,000 yields
        # 25,500 (Pasture has only 2,000 images).
        actual = sorted({k[0] for k in g})
        for n in actual:
            for payload in PAYLOADS:
                k_seq = (n, "sequential", 0, payload)
                if k_seq not in g:
                    continue
                sq = float(np.mean(g[k_seq]))
                p4 = float(np.mean(g[(n, "process", 4, payload)]))
                p8 = float(np.mean(g[(n, "process", 8, payload)]))
                print(f"  {n:>7}{payload:>9}{sq:>9.2f}{p4:>8.2f}{p8:>8.2f}"
                      f"{sq / p4:>8.2f}x{sq / p8:>8.2f}x")

    path = os.path.join(OUT, "workload_size.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("")
    print(f"wrote {path}  ({len(rows)} runs)")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")
