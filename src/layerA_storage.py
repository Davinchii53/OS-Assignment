"""
Layer A, experiment 4 - is storage ever the bottleneck?

Every earlier measurement was deliberately page-cache warm, so files were
served from RAM and storage never appeared. The assignment asks whether the
bottleneck is CPU, RAM, storage or GPU, so this measures the storage case.

Method: run the same decode matrix twice, once with every file already in the
page cache and once with the cache evicted. posix_fadvise(DONTNEED) evicts only
the dataset files, so nothing else in the machine is disturbed.

Pre-registered prediction, from the warm Layer A throughputs converted into the
read bandwidth they demand:

  MS  (104.73 KB/file): 8 workers need 690 MB/s = 81% of cold bandwidth
                        -> storage should become the bottleneck
  RGB (3.32 KB/file):   8 workers need  41 MB/s =  5% of cold bandwidth
                        -> storage should never matter

The scalar payload is used throughout, so inter-process transfer cost is held
out and the only variable is where the bytes come from.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import gc
import glob
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import psutil
from PIL import Image

QUICK = os.environ.get("LAYERA_QUICK") == "1"
SUBSET = 2000 if QUICK else None
REPEATS = 1 if QUICK else 3
CONFIGS = ([("sequential", 0), ("thread", 8)]
           + [("process", w) for w in ([1, 2, 4] if QUICK else [1, 2, 4, 8, 16])])
DATASETS = ["RGB", "MS"]
CHUNKSIZE = 64
SEED = 42
OUT = "results/layerA"
BAR = "=" * 78
_DS = None

os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(*a, flush=True)


def _decode(path):
    if _DS == "RGB":
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    else:
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
    a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-6)
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def decode_scalar(path):
    return float(_decode(path).mean())


def evict(files):
    """Drop these files from the page cache. No root needed."""
    for p in files:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def prime(files):
    def rd(p):
        with open(p, "rb") as f:
            return len(f.read())
    with ThreadPoolExecutor(max_workers=8) as ex:
        return sum(ex.map(rd, files, chunksize=64))


def measure_bandwidth(files):
    """Raw sequential read bandwidth, warm vs cold. The storage ceiling."""
    sample = files[:1500]
    prime(sample)
    t0 = time.perf_counter()
    n = prime(sample)
    warm = time.perf_counter() - t0
    evict(sample)
    t0 = time.perf_counter()
    n = prime(sample)
    cold = time.perf_counter() - t0
    mb = n / 1024 ** 2
    return mb / warm, mb / cold


def run_one(files, mode, workers, nbytes):
    gc.collect()
    cpu0 = psutil.cpu_percent(None)
    t0 = time.perf_counter()
    if mode == "sequential":
        for p in files:
            decode_scalar(p)
    elif mode == "thread":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(decode_scalar, files, chunksize=CHUNKSIZE):
                pass
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(decode_scalar, files, chunksize=CHUNKSIZE):
                pass
    total = time.perf_counter() - t0
    cpu = psutil.cpu_percent(None)
    return {"mode": mode, "workers": workers, "total_s": round(total, 4),
            "throughput_ips": round(len(files) / total, 1),
            "read_mb_s": round(nbytes / 1024 ** 2 / total, 1),
            "cpu_mean": round(cpu, 1),
            "busy_cores": round(cpu * os.cpu_count() / 100.0, 2)}


def main():
    global _DS
    log(BAR)
    log("LAYER A, EXPERIMENT 4 - storage as a bottleneck")
    log(BAR)
    log(f"cores: {os.cpu_count()} logical   eviction: posix_fadvise(DONTNEED)")
    log(f"mode: {'QUICK' if QUICK else 'FULL'}   repeats={REPEATS}")
    log(f"configs: {CONFIGS}")

    rows = []
    bw = {}
    for ds in DATASETS:
        _DS = ds
        ext = "jpg" if ds == "RGB" else "tif"
        hits = sorted(glob.glob(os.path.join("data/ext", "**", "*." + ext), recursive=True))
        if not hits:
            log(f"skipping {ds}")
            continue
        root = os.path.dirname(os.path.dirname(hits[0]))
        files = [h for h in hits if h.startswith(root + os.sep)]
        if SUBSET:
            files = files[:SUBSET]
        nbytes = sum(os.path.getsize(p) for p in files)

        log("")
        log(BAR)
        log(f"{ds}: {len(files)} files, {nbytes / 1024 ** 2:.0f} MB, "
            f"{nbytes / len(files) / 1024:.2f} KB each")
        log(BAR)
        w_bw, c_bw = measure_bandwidth(files)
        bw[ds] = {"warm_mb_s": round(w_bw, 0), "cold_mb_s": round(c_bw, 0)}
        log(f"raw read bandwidth: warm {w_bw:.0f} MB/s, cold {c_bw:.0f} MB/s "
            f"({w_bw / c_bw:.1f}x)")
        log("")

        for rep in range(REPEATS):
            for cache in ("warm", "cold"):
                block = list(CONFIGS)
                random.Random(SEED + rep * 10 + (cache == "cold")).shuffle(block)
                if cache == "warm":
                    prime(files)
                for mode, w in block:
                    if cache == "cold":
                        evict(files)
                    r = run_one(files, mode, w, nbytes)
                    r.update({"dataset": ds, "cache": cache, "repeat": rep})
                    rows.append(r)
                    log(f"  {cache:<5} {mode:<10} x{w:<3} r{rep}  {r['total_s']:7.2f}s  "
                        f"{r['throughput_ips']:8.1f} img/s  {r['read_mb_s']:7.1f} MB/s  "
                        f"cores {r['busy_cores']:5.2f}")

    path = os.path.join(OUT, "storage.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(OUT, "storage_bandwidth.json"), "w") as f:
        json.dump(bw, f, indent=2)

    # ------------------------------------------------------------- summary ----
    g = defaultdict(list)
    for r in rows:
        g[(r["dataset"], r["cache"], r["mode"], r["workers"])].append(r)

    log("")
    log(BAR)
    log("SUMMARY - warm vs cold, scalar payload, speedup vs sequential in the same state")
    log(BAR)
    for ds in DATASETS:
        if not any(k[0] == ds for k in g):
            continue
        log("")
        log(f"--- {ds} ---   cold storage ceiling {bw[ds]['cold_mb_s']:.0f} MB/s")
        log(f"{'config':<14}{'warm s':>9}{'cold s':>9}{'cold/warm':>11}"
            f"{'warm spd':>10}{'cold spd':>10}{'cold MB/s':>11}{'% of ceiling':>14}")
        base = {c: float(np.mean([r["total_s"] for r in g[(ds, c, "sequential", 0)]]))
                for c in ("warm", "cold")}
        for mode, w in CONFIGS:
            kw, kc = (ds, "warm", mode, w), (ds, "cold", mode, w)
            if kw not in g or kc not in g:
                continue
            tw = float(np.mean([r["total_s"] for r in g[kw]]))
            tc = float(np.mean([r["total_s"] for r in g[kc]]))
            mbs = float(np.mean([r["read_mb_s"] for r in g[kc]]))
            log(f"{mode + ' x' + str(w):<14}{tw:>9.2f}{tc:>9.2f}{tc / tw:>10.2f}x"
                f"{base['warm'] / tw:>9.2f}x{base['cold'] / tc:>9.2f}x{mbs:>11.1f}"
                f"{100 * mbs / bw[ds]['cold_mb_s']:>13.0f}%")

        warm_best = max((base["warm"] / float(np.mean([r["total_s"] for r in g[(ds, "warm", m, w)]]))
                         for m, w in CONFIGS if (ds, "warm", m, w) in g))
        cold_best = max((base["cold"] / float(np.mean([r["total_s"] for r in g[(ds, "cold", m, w)]]))
                         for m, w in CONFIGS if (ds, "cold", m, w) in g))
        log(f"  best speedup: warm {warm_best:.2f}x   cold {cold_best:.2f}x   "
            f"-> cold keeps {100 * cold_best / warm_best:.0f}% of the warm gain")
    log("")
    log(f"wrote {path}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    log(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")
