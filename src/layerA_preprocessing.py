"""
Layer A - preprocessing benchmark, no model, no GPU.

Layer B showed that on a GPU the data pipeline is not the bottleneck, so
nothing the loader does matters. This isolates the pipeline itself: the same
decode work done sequentially, with threads, and with processes.

Three experiments:

  1. GIL control. A pure-Python loop vs a SHA-256 loop that releases the GIL,
     each run with threads and with processes. This PROVES the GIL is the
     reason threads do not scale, instead of inferring it from the decode
     results.

  2. Decode matrix. Sequential / threads / processes x {1,2,4,8,16} on both
     EuroSAT variants, 3 repeats, randomised order.

  3. Payload ablation. Identical CPU work, but workers return either the full
     decoded array (212 KB for MS) or a single float. The CPU cost is the same,
     so any difference is the cost of moving data between processes. Threads are
     included as a control: they share memory, so payload size should not matter
     for them.

OMP_NUM_THREADS is pinned to 1. numpy's mean/std are thread-parallel, and
leaving them free makes "1 worker" secretly multi-core.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import gc
import glob
import hashlib
import json
import platform
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import psutil
from PIL import Image

# ------------------------------------------------------------------ config ----
QUICK = os.environ.get("LAYERA_QUICK") == "1"
SUBSET = 2000 if QUICK else None
REPEATS = 1 if QUICK else 3
WORKER_COUNTS = [1, 2, 4] if QUICK else [1, 2, 4, 8, 16]
DATASETS = ["RGB", "MS"]
CHUNKSIZE = 64
SEED = 42
COOLDOWN = 0.3
OUT = "results/layerA"
BAR = "=" * 78

os.makedirs(OUT, exist_ok=True)
_DS = None          # set before each matrix block; fork copies it to workers


def log(*a):
    print(*a, flush=True)


def cpu_topology():
    ids, pid, model = set(), None, None
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name") and model is None:
                model = line.split(":", 1)[1].strip()
            elif line.startswith("physical id"):
                pid = line.split(":", 1)[1].strip()
            elif line.startswith("core id") and pid is not None:
                ids.add((pid, line.split(":", 1)[1].strip()))
    except Exception:
        pass
    return model, (len(ids) or None), os.cpu_count()


CPU_MODEL, N_PHYS, N_LOG = cpu_topology()


# ------------------------------------------------------------- GIL control ----
def pure_python(n):
    """Bytecode only. Holds the GIL the entire time."""
    x = 0
    for i in range(n):
        x = (x * 31 + i) % 1000003
    return x


def sha256_work(n):
    """hashlib releases the GIL around the C hashing call."""
    buf = b"x" * 65536
    h = hashlib.sha256()
    for _ in range(n):
        h.update(buf)
    return h.hexdigest()[:8]


def run_gil_control():
    log("")
    log(BAR)
    log("EXPERIMENT 1 - GIL control")
    log(BAR)
    log("Same amount of work, split across N workers. If threads scale only on")
    log("the task that releases the GIL, the GIL is the cause.")
    log("")
    TASKS = 16
    SIZES = {"pure_python": 400_000, "sha256": 700}
    rows = []
    for task_name, fn in (("pure_python", pure_python), ("sha256", sha256_work)):
        size = SIZES[task_name]
        t0 = time.perf_counter()
        for _ in range(TASKS):
            fn(size)
        base = time.perf_counter() - t0
        log(f"  {task_name}: sequential {base:.2f}s for {TASKS} tasks")
        rows.append({"task": task_name, "mode": "sequential", "workers": 0,
                     "seconds": round(base, 4), "speedup": 1.0})
        for mode, Pool in (("thread", ThreadPoolExecutor), ("process", ProcessPoolExecutor)):
            for w in (2, 4, 8):
                t0 = time.perf_counter()
                with Pool(max_workers=w) as ex:
                    list(ex.map(fn, [size] * TASKS))
                dt = time.perf_counter() - t0
                rows.append({"task": task_name, "mode": mode, "workers": w,
                             "seconds": round(dt, 4), "speedup": round(base / dt, 3)})
                log(f"    {mode:<8} x{w:<3} {dt:6.2f}s   speedup {base / dt:5.2f}x")
        log("")
    with open(os.path.join(OUT, "gil_control.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def sp(task, mode, w):
        return next(r["speedup"] for r in rows
                    if r["task"] == task and r["mode"] == mode and r["workers"] == w)

    log(f"  threads on pure Python, 4 workers : {sp('pure_python', 'thread', 4):.2f}x")
    log(f"  threads on SHA-256,     4 workers : {sp('sha256', 'thread', 4):.2f}x")
    log(f"  processes on pure Python, 4 workers: {sp('pure_python', 'process', 4):.2f}x")
    log("  => threads help only when the work releases the GIL.")
    return rows


# --------------------------------------------------------------- decoding ----
def _decode(path):
    if _DS == "RGB":
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    else:
        import tifffile
        a = tifffile.imread(path).astype(np.float32)
    a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-6)
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def decode_array(path):
    """Returns the full tensor: a large payload to move between processes."""
    return _decode(path)


def decode_scalar(path):
    """Identical CPU work, 8-byte payload."""
    return float(_decode(path).mean())


PAYLOADS = {"array": decode_array, "scalar": decode_scalar}


class Monitor(threading.Thread):
    def __init__(self, hz=10.0):
        super().__init__(daemon=True)
        self.interval = 1.0 / hz
        self.stop_evt = threading.Event()
        self.cpu, self.ram = [], []
        self.ctx0 = psutil.cpu_stats().ctx_switches
        self.t0 = time.perf_counter()
        self.ram0 = psutil.virtual_memory().used

    def run(self):
        psutil.cpu_percent(None)
        while not self.stop_evt.wait(self.interval):
            self.cpu.append(psutil.cpu_percent(None))
            self.ram.append(psutil.virtual_memory().used)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=5)
        dur = max(time.perf_counter() - self.t0, 1e-9)
        cpu = float(np.mean(self.cpu)) if self.cpu else None
        return {
            "cpu_mean": round(cpu, 2) if cpu is not None else None,
            "busy_cores": round(cpu * N_LOG / 100.0, 2) if cpu is not None else None,
            "ram_delta_mb": round((max(self.ram) - self.ram0) / 1024 ** 2, 1) if self.ram else None,
            "ctx_switches_per_s": round((psutil.cpu_stats().ctx_switches - self.ctx0) / dur, 1),
        }


def run_one(files, mode, workers, payload):
    fn = PAYLOADS[payload]
    mon = Monitor()
    mon.start()
    t_start = time.perf_counter()
    try:
        if mode == "sequential":
            spawn = 0.0
            for p in files:
                fn(p)
        elif mode == "thread":
            t0 = time.perf_counter()
            ex = ThreadPoolExecutor(max_workers=workers)
            try:
                ex.submit(int).result()
                spawn = time.perf_counter() - t0
                for _ in ex.map(fn, files, chunksize=CHUNKSIZE):
                    pass
            finally:
                ex.shutdown(wait=True)
        else:
            t0 = time.perf_counter()
            ex = ProcessPoolExecutor(max_workers=workers)
            try:
                ex.submit(int).result()          # force workers to actually start
                spawn = time.perf_counter() - t0
                for _ in ex.map(fn, files, chunksize=CHUNKSIZE):
                    pass
            finally:
                ex.shutdown(wait=True)
        total = time.perf_counter() - t_start
    finally:
        res = mon.stop()
        gc.collect()
    row = {"mode": mode, "workers": workers, "payload": payload,
           "n_images": len(files), "total_s": round(total, 4),
           "spawn_s": round(spawn, 4),
           "throughput_ips": round(len(files) / total, 1)}
    row.update(res)
    return row


def find_root(ext):
    hits = sorted(glob.glob(os.path.join("data/ext", "**", "*." + ext), recursive=True))
    if not hits:
        return None, []
    root = os.path.dirname(os.path.dirname(hits[0]))
    return root, [h for h in hits if h.startswith(root + os.sep)]


def warm_cache(files):
    def rd(p):
        with open(p, "rb") as f:
            return len(f.read())
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        n = sum(ex.map(rd, files, chunksize=64))
    return n / 1024 ** 2, time.perf_counter() - t0


def run_matrix(ds, files):
    global _DS
    _DS = ds
    configs = [("sequential", 0, "array"), ("sequential", 0, "scalar")]
    for mode in ("thread", "process"):
        for payload in ("array", "scalar"):
            for w in WORKER_COUNTS:
                configs.append((mode, w, payload))

    plan = []
    for rep in range(REPEATS):
        block = list(configs)
        random.Random(SEED + rep).shuffle(block)
        plan += [(m, w, p, rep) for m, w, p in block]

    log("")
    log(BAR)
    log(f"EXPERIMENT 2+3 - decode matrix, {ds}, {len(files)} images")
    log(BAR)
    mb, secs = warm_cache(files)
    log(f"page-cache warm-up: read {mb:.0f} MB in {secs:.1f}s")
    log(f"{len(configs)} configs x {REPEATS} repeats = {len(plan)} runs")
    log("")

    path = os.path.join(OUT, f"runs_{ds.lower()}.csv")
    rows = []
    for i, (mode, w, payload, rep) in enumerate(plan):
        time.sleep(COOLDOWN)
        r = run_one(files, mode, w, payload)
        r.update({"dataset": ds, "repeat": rep, "order_idx": i})
        rows.append(r)
        if i == 0:
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=list(r)).writeheader()
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(rows[0])).writerow(r)
        log(f"[{i + 1:3d}/{len(plan)}] {mode:<10} x{w:<3} {payload:<7} r{rep}  "
            f"{r['total_s']:7.2f}s  {r['throughput_ips']:8.1f} img/s  "
            f"cores {r['busy_cores']:5.2f}  ctx/s {r['ctx_switches_per_s']:8.0f}  "
            f"spawn {r['spawn_s']:.2f}s")
    return rows


def summarise(ds, rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["mode"], r["workers"], r["payload"])].append(r)

    def agg(key, field):
        v = [x[field] for x in g[key] if x[field] is not None]
        if not v:
            return None, None
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)

    base = {p: agg(("sequential", 0, p), "total_s")[0] for p in ("array", "scalar")}
    order = {"sequential": 0, "thread": 1, "process": 2}
    keys = sorted(g, key=lambda k: (k[2], order[k[0]], k[1]))

    out = []
    for k in keys:
        t, sd = agg(k, "total_s")
        ips, _ = agg(k, "throughput_ips")
        b = base[k[2]]
        out.append({
            "dataset": ds, "mode": k[0], "workers": k[1], "payload": k[2], "n": len(g[k]),
            "total_s_mean": round(t, 3), "total_s_sd": round(sd, 3),
            "throughput_ips": round(ips, 1),
            "speedup": round(b / t, 3),
            "efficiency": round((b / t) / k[1], 3) if k[1] else None,
            "busy_cores": round(agg(k, "busy_cores")[0], 2),
            "ram_delta_mb": round(agg(k, "ram_delta_mb")[0], 1),
            "ctx_switches_per_s": round(agg(k, "ctx_switches_per_s")[0], 0),
            "spawn_s": round(agg(k, "spawn_s")[0], 3),
        })

    log("")
    log(BAR)
    log(f"SUMMARY - {ds}   (speedup is vs sequential with the SAME payload)")
    log(BAR)
    log(f"{'payload':<8}{'mode':<11}{'w':>3}{'total s':>15}{'img/s':>10}"
        f"{'speedup':>9}{'eff':>7}{'cores':>7}{'ctx/s':>9}{'RAM MB':>8}{'spawn':>7}")
    for s in out:
        log(f"{s['payload']:<8}{s['mode']:<11}{s['workers']:>3}"
            f"{s['total_s_mean']:>9.2f}+-{s['total_s_sd']:<5.2f}{s['throughput_ips']:>10.1f}"
            f"{s['speedup']:>9.2f}{(format(s['efficiency'], '.2f') if s['efficiency'] else '-'):>7}"
            f"{s['busy_cores']:>7.2f}{s['ctx_switches_per_s']:>9.0f}"
            f"{s['ram_delta_mb']:>8.0f}{s['spawn_s']:>7.2f}")
    return out


def main():
    log(BAR)
    log("LAYER A - preprocessing benchmark (no model, no GPU)")
    log(BAR)
    log(f"cpu: {CPU_MODEL}")
    log(f"cores: {N_PHYS} physical / {N_LOG} logical     OMP_NUM_THREADS="
        f"{os.environ['OMP_NUM_THREADS']}")
    log(f"python {platform.python_version()}   numpy {np.__version__}")
    log(f"mode: {'QUICK' if QUICK else 'FULL'}   repeats={REPEATS}   workers={WORKER_COUNTS}")

    env = {"cpu_model": CPU_MODEL, "cores_physical": N_PHYS, "cores_logical": N_LOG,
           "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
           "python": platform.python_version(), "numpy": np.__version__,
           "omp_num_threads": os.environ["OMP_NUM_THREADS"],
           "os": platform.system() + " " + platform.release(),
           "quick": QUICK, "repeats": REPEATS, "worker_counts": WORKER_COUNTS,
           "chunksize": CHUNKSIZE, "seed": SEED}
    with open(os.path.join(OUT, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    gil = run_gil_control()

    all_sum = []
    for ds in DATASETS:
        ext = "jpg" if ds == "RGB" else "tif"
        root, files = find_root(ext)
        if not files:
            log(f"skipping {ds}: no .{ext} files under data/ext")
            continue
        if SUBSET:
            files = files[:SUBSET]
        rows = run_matrix(ds, files)
        all_sum += summarise(ds, rows)

    if all_sum:
        with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(all_sum[0]))
            wr.writeheader()
            wr.writerows(all_sum)

    log("")
    log(BAR)
    log("FILES")
    log(BAR)
    for p in sorted(glob.glob(os.path.join(OUT, "*"))):
        log(f"  {p}  ({os.path.getsize(p) / 1024:.1f} KB)")
    return gil, all_sum


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    log(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")
