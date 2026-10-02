# ============================================================================
# Layer A on Kaggle - preprocessing benchmark, NO GPU NEEDED
#
# Companion to the sandbox run. Same experiments, same worker counts, different
# machine: 2 physical cores here against 4 in the sandbox. Two matched scaling
# curves are the cleanest way to show that parallel-computing results depend on
# the hardware they were measured on.
#
# HOW TO USE ON KAGGLE
#   1. Settings -> Accelerator -> **None**. This needs no GPU, so it costs no
#      GPU quota. Leaving a GPU attached only burns quota for nothing.
#   2. Add Data -> attach BOTH eurosat datasets (RGB and MS).
#   3. MODE = "SMOKE", run interactively (about 5 min). Check it ends clean.
#   4. MODE = "FULL", then Save Version -> Save & Run All (Commit).
#      Roughly 1.5 to 3 hours, server-side. Stop the interactive session after.
#
# Four experiments:
#   1. GIL control      - pure Python vs SHA-256, threads vs processes
#   2. Decode matrix    - sequential / threads / processes x {1,2,4,8,16}
#   3. Payload ablation - workers return the full array or a single float
#   4. Storage          - warm vs cold page cache
#
# OMP_NUM_THREADS is pinned to 1. numpy's mean/std are thread-parallel, and
# leaving them free makes a "1 worker" run secretly multi-core.
#
# PASTE-SAFE: there is no backslash character anywhere in this cell.
# ============================================================================
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

# ---------------------------------------------------------------- CONFIG ----
MODE = "SMOKE"          # "SMOKE" check run | "FULL" all 27,000 images
DATASETS = ["RGB", "MS"]
RUN_STORAGE = True      # experiment 4; set False to save about a third of the time
SEED = 42
CHUNKSIZE = 64
MAX_HOURS = 10.0        # stop cleanly before the Kaggle session limit
RESUME = True
HARNESS_VERSION = "1.0"

PRESETS = {
    "SMOKE": {"subset": 2000, "repeats": 1, "workers": [1, 2, 4], "cooldown": 0.3},
    "FULL": {"subset": None, "repeats": 3, "workers": [1, 2, 4, 8, 16], "cooldown": 0.3},
}
_P = PRESETS[MODE]
SUBSET, REPEATS, WORKER_COUNTS, COOLDOWN = _P["subset"], _P["repeats"], _P["workers"], _P["cooldown"]

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

T_START = time.time()
ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working" if ON_KAGGLE else "results/layerA_kaggle"
os.makedirs(OUT, exist_ok=True)
BAR = "=" * 78
NL = chr(10)
_DS = None              # set before each matrix block; fork copies it to workers


def log(*a):
    print(*a, flush=True)


def fmt(v, spec):
    """Format a value that may be None. Very short runs can finish before the
    monitor thread takes its first sample, leaving metrics unset."""
    if v is None:
        return "-".rjust(len(format(0, spec)))
    return format(v, spec)


def out_path(name):
    return os.path.join(OUT, "layerA_" + MODE.lower() + "_" + name)


RUNS_CSV = out_path("runs.csv")
STORAGE_CSV = out_path("storage.csv")
GIL_CSV = out_path("gil_control.csv")
SUMMARY_CSV = out_path("summary.csv")
SUMMARY_MD = out_path("summary.md")
CFG_PATH = out_path("config.json")
ENV_PATH = out_path("environment.json")

# Only resume from files written by this exact version, so results from
# different code versions never get mixed.
_stale = True
if RESUME and os.path.exists(CFG_PATH):
    try:
        with open(CFG_PATH) as _f:
            _stale = json.load(_f).get("version") != HARNESS_VERSION
    except Exception:
        _stale = True
if not RESUME or _stale:
    _n = 0
    for _p in (RUNS_CSV, STORAGE_CSV, GIL_CSV, SUMMARY_CSV, SUMMARY_MD, CFG_PATH, ENV_PATH):
        if os.path.exists(_p):
            os.remove(_p)
            _n += 1
    if _n:
        log("removed " + str(_n) + " old result files (different version, or RESUME = False)")


# -------------------------------------------------------------- topology ----
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

log(BAR)
log("LAYER A ON KAGGLE - preprocessing benchmark, no GPU")
log(BAR)
log("mode=" + MODE + "  repeats=" + str(REPEATS) + "  workers=" + str(WORKER_COUNTS))
log("cpu: " + str(CPU_MODEL))
log("cores: " + str(N_PHYS) + " physical / " + str(N_LOG) + " logical"
    + "   OMP_NUM_THREADS=" + os.environ["OMP_NUM_THREADS"])
log("python " + platform.python_version() + "   numpy " + np.__version__)
log("sandbox comparison: 4 physical / 8 logical, Xeon Platinum 8488C")


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
    log("Same work split across N workers. If threads scale only on the task")
    log("that releases the GIL, the GIL is the cause, not a correlation.")
    log("")
    TASKS = 16
    SIZES = {"pure_python": 400000, "sha256": 700}
    rows = []
    for name, fn in (("pure_python", pure_python), ("sha256", sha256_work)):
        size = SIZES[name]
        t0 = time.perf_counter()
        for _ in range(TASKS):
            fn(size)
        base = time.perf_counter() - t0
        log("  " + name + ": sequential " + format(base, ".2f") + "s for " + str(TASKS) + " tasks")
        rows.append({"task": name, "mode": "sequential", "workers": 0,
                     "seconds": round(base, 4), "speedup": 1.0})
        for mode, Pool in (("thread", ThreadPoolExecutor), ("process", ProcessPoolExecutor)):
            for w in (2, 4, 8):
                t0 = time.perf_counter()
                with Pool(max_workers=w) as ex:
                    list(ex.map(fn, [size] * TASKS))
                dt = time.perf_counter() - t0
                rows.append({"task": name, "mode": mode, "workers": w,
                             "seconds": round(dt, 4), "speedup": round(base / dt, 3)})
                log("    " + mode.ljust(8) + " x" + str(w).ljust(3) + format(dt, "6.2f")
                    + "s   speedup " + format(base / dt, "5.2f") + "x")
        log("")
    with open(GIL_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    def sp(task, mode, w):
        return next(r["speedup"] for r in rows
                    if r["task"] == task and r["mode"] == mode and r["workers"] == w)

    log("  threads, pure Python, 4 workers : " + format(sp("pure_python", "thread", 4), ".2f") + "x")
    log("  threads, SHA-256,     4 workers : " + format(sp("sha256", "thread", 4), ".2f") + "x")
    log("  sandbox measured 1.00x and 3.04x respectively")
    log("  => threads help only when the work releases the GIL")
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
        else:
            Pool = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
            t0 = time.perf_counter()
            ex = Pool(max_workers=workers)
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


# ---------------------------------------------------------------- dataset ----
def find_dataset(ext):
    """Kaggle nests dataset directories unpredictably, so discover rather than
    assume. Returns the class-parent directory and its files."""
    base = "/kaggle/input" if ON_KAGGLE else "data/ext"
    hits = sorted(glob.glob(os.path.join(base, "**", "*." + ext), recursive=True))
    if not hits:
        return None, []
    root = os.path.dirname(os.path.dirname(hits[0]))
    return root, [h for h in hits if h.startswith(root + os.sep)]


def stratified(root, ext, classes, n):
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


def evict(files):
    """Drop these files from the page cache. No root needed. May be a no-op on
    some filesystems, which the bandwidth check below will reveal."""
    for p in files:
        try:
            fd = os.open(p, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def append_rows(path, fields, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            wr.writeheader()
        for r in rows:
            wr.writerow(r)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


RUN_FIELDS = ["dataset", "mode", "workers", "payload", "repeat", "order_idx", "n_images",
              "total_s", "throughput_ips", "spawn_s", "cpu_mean", "busy_cores",
              "ram_delta_mb", "ctx_switches_per_s"]
STOR_FIELDS = ["dataset", "cache", "mode", "workers", "repeat", "n_images", "total_s",
               "throughput_ips", "read_mb_s", "cpu_mean", "busy_cores"]

# ------------------------------------------------------------- discover ----
FOUND = {}
for ds in DATASETS:
    ext = "jpg" if ds == "RGB" else "tif"
    root, files = find_dataset(ext)
    if not files:
        log("")
        log("  " + ds + ": NOT FOUND. Use Add Data to attach it.")
        continue
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if SUBSET:
        files = stratified(root, ext, classes, SUBSET)
    nbytes = sum(os.path.getsize(p) for p in files)
    FOUND[ds] = {"root": root, "files": files, "classes": classes, "bytes": nbytes}
    log("")
    log("  " + ds + ": " + str(len(files)) + " files, " + str(len(classes)) + " classes, "
        + format(nbytes / 1024 ** 2, ".0f") + " MB, "
        + format(nbytes / len(files) / 1024, ".2f") + " KB each")
    log("     root " + root)

assert FOUND, "no datasets found under /kaggle/input -- attach at least one"

if ON_KAGGLE and "MS" in FOUND:
    try:
        import tifffile
    except ImportError:
        log("")
        log("installing tifffile for the MS GeoTIFFs")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"],
                       capture_output=True)
        import tifffile

ENV = {"platform": "kaggle" if ON_KAGGLE else "sandbox", "cpu_model": CPU_MODEL,
       "cores_physical": N_PHYS, "cores_logical": N_LOG,
       "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
       "python": platform.python_version(), "numpy": np.__version__,
       "omp_num_threads": os.environ["OMP_NUM_THREADS"],
       "os": platform.system() + " " + platform.release()}
with open(ENV_PATH, "w") as f:
    json.dump(ENV, f, indent=2)
CFG = {"version": HARNESS_VERSION, "mode": MODE, "datasets": list(FOUND),
       "repeats": REPEATS, "worker_counts": WORKER_COUNTS, "chunksize": CHUNKSIZE,
       "seed": SEED, "run_storage": RUN_STORAGE,
       "n_images": {k: len(v["files"]) for k, v in FOUND.items()}}
with open(CFG_PATH, "w") as f:
    json.dump(CFG, f, indent=2)

GIL = run_gil_control()

# --------------------------------------------------- experiments 2 and 3 ----
CONFIGS = [("sequential", 0, "array"), ("sequential", 0, "scalar")]
for _m in ("thread", "process"):
    for _p in ("array", "scalar"):
        for _w in WORKER_COUNTS:
            CONFIGS.append((_m, _w, _p))

done = {(r["dataset"], r["mode"], int(r["workers"]), r["payload"], int(r["repeat"]))
        for r in read_rows(RUNS_CSV)}
if done:
    log("")
    log("resuming: " + str(len(done)) + " runs already recorded")

for ds in FOUND:
    globals()["_DS"] = ds
    files = FOUND[ds]["files"]
    log("")
    log(BAR)
    log("EXPERIMENT 2+3 - decode matrix, " + ds + ", " + str(len(files)) + " images")
    log(BAR)
    t0 = time.perf_counter()
    mb = prime(files) / 1024 ** 2
    log("page-cache warm-up: read " + format(mb, ".0f") + " MB in "
        + format(time.perf_counter() - t0, ".1f") + "s")

    plan = []
    for rep in range(REPEATS):
        block = list(CONFIGS)
        random.Random(SEED + rep).shuffle(block)
        plan += [(m, w, p, rep) for m, w, p in block]
    log(str(len(CONFIGS)) + " configs x " + str(REPEATS) + " repeats = " + str(len(plan)) + " runs")
    log("")

    for i, (mode, w, payload, rep) in enumerate(plan):
        if (ds, mode, w, payload, rep) in done:
            continue
        if time.time() - T_START > MAX_HOURS * 3600:
            log("time budget reached -- stopping cleanly, results so far are saved")
            break
        time.sleep(COOLDOWN)
        r = run_one(files, mode, w, payload)
        r.update({"dataset": ds, "repeat": rep, "order_idx": i})
        append_rows(RUNS_CSV, RUN_FIELDS, [r])
        log("[" + str(i + 1).rjust(3) + "/" + str(len(plan)) + "] " + mode.ljust(10)
            + " x" + str(w).ljust(3) + payload.ljust(7) + " r" + str(rep)
            + fmt(r["total_s"], "8.2f") + "s  " + fmt(r["throughput_ips"], "8.1f")
            + " img/s  cores " + fmt(r["busy_cores"], "5.2f")
            + "  ctx/s " + fmt(r["ctx_switches_per_s"], "8.0f")
            + "  RAM " + fmt(r["ram_delta_mb"], "6.0f") + " MB")

# ---------------------------------------------------------- experiment 4 ----
if RUN_STORAGE:
    S_CONFIGS = [("sequential", 0), ("thread", 8)] + [("process", w) for w in WORKER_COUNTS]
    sdone = {(r["dataset"], r["cache"], r["mode"], int(r["workers"]), int(r["repeat"]))
             for r in read_rows(STORAGE_CSV)}
    BW = {}
    for ds in FOUND:
        globals()["_DS"] = ds
        files = FOUND[ds]["files"]
        nbytes = FOUND[ds]["bytes"]
        log("")
        log(BAR)
        log("EXPERIMENT 4 - storage, " + ds)
        log(BAR)
        sample = files[:1500]
        prime(sample)
        t0 = time.perf_counter()
        n = prime(sample)
        warm_bw = (n / 1024 ** 2) / (time.perf_counter() - t0)
        evict(sample)
        t0 = time.perf_counter()
        n = prime(sample)
        cold_bw = (n / 1024 ** 2) / (time.perf_counter() - t0)
        BW[ds] = {"warm_mb_s": round(warm_bw, 1), "cold_mb_s": round(cold_bw, 1)}
        log("raw read bandwidth: warm " + format(warm_bw, ".0f") + " MB/s, cold "
            + format(cold_bw, ".0f") + " MB/s  (" + format(warm_bw / cold_bw, ".2f") + "x)")
        if warm_bw / cold_bw < 1.05:
            log("  NOTE: eviction barely changed read speed. Either the storage is fast")
            log("  enough that caching hardly matters, or posix_fadvise is a no-op on")
            log("  this filesystem. Treat the cold results as a weak manipulation.")
        log("")
        for rep in range(REPEATS):
            for cache in ("warm", "cold"):
                block = list(S_CONFIGS)
                random.Random(SEED + rep * 10 + (cache == "cold")).shuffle(block)
                if cache == "warm":
                    prime(files)
                for mode, w in block:
                    if (ds, cache, mode, w, rep) in sdone:
                        continue
                    if time.time() - T_START > MAX_HOURS * 3600:
                        log("time budget reached -- stopping cleanly")
                        break
                    if cache == "cold":
                        evict(files)
                    r = run_one(files, mode, w, "scalar")
                    r.update({"dataset": ds, "cache": cache, "repeat": rep,
                              "read_mb_s": round(nbytes / 1024 ** 2 / r["total_s"], 1)})
                    append_rows(STORAGE_CSV, STOR_FIELDS, [r])
                    log("  " + cache.ljust(5) + " " + mode.ljust(10) + " x" + str(w).ljust(3)
                        + " r" + str(rep) + fmt(r["total_s"], "8.2f") + "s  "
                        + fmt(r["throughput_ips"], "8.1f") + " img/s  "
                        + fmt(r["read_mb_s"], "7.1f") + " MB/s  cores "
                        + fmt(r["busy_cores"], "5.2f"))
    with open(out_path("storage_bandwidth.json"), "w") as f:
        json.dump(BW, f, indent=2)

# ---------------------------------------------------------------- summary ----
rows = read_rows(RUNS_CSV)
for r in rows:
    for k in ("workers", "repeat", "n_images"):
        r[k] = int(float(r[k]))
    for k in ("total_s", "throughput_ips", "spawn_s", "cpu_mean", "busy_cores",
              "ram_delta_mb", "ctx_switches_per_s"):
        r[k] = float(r[k]) if r[k] else None

g = defaultdict(list)
for r in rows:
    g[(r["dataset"], r["mode"], r["workers"], r["payload"])].append(r)

def safe_mean(rs, field, nd=2):
    """Mean over a field, skipping runs where the monitor recorded nothing."""
    v = [r[field] for r in rs if r.get(field) is not None]
    return round(float(np.mean(v)), nd) if v else None


ORDER = {"sequential": 0, "thread": 1, "process": 2}
SUM = []
for ds in FOUND:
    base = {}
    for p in ("array", "scalar"):
        k = (ds, "sequential", 0, p)
        if k in g:
            base[p] = float(np.mean([r["total_s"] for r in g[k]]))
    keys = sorted([k for k in g if k[0] == ds], key=lambda k: (k[3], ORDER[k[1]], k[2]))
    for k in keys:
        rs = g[k]
        t = float(np.mean([r["total_s"] for r in rs]))
        sd = float(np.std([r["total_s"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0
        b = base.get(k[3])
        SUM.append({
            "dataset": ds, "mode": k[1], "workers": k[2], "payload": k[3], "n": len(rs),
            "total_s_mean": round(t, 3), "total_s_sd": round(sd, 3),
            "throughput_ips": round(float(np.mean([r["throughput_ips"] for r in rs])), 1),
            "speedup": round(b / t, 3) if b else None,
            "efficiency": round((b / t) / k[2], 3) if b and k[2] else None,
            "busy_cores": safe_mean(rs, "busy_cores", 2),
            "ram_delta_mb": safe_mean(rs, "ram_delta_mb", 1),
            "ctx_switches_per_s": safe_mean(rs, "ctx_switches_per_s", 0),
            "spawn_s": safe_mean(rs, "spawn_s", 3),
        })

if SUM:
    with open(SUMMARY_CSV, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(SUM[0]))
        wr.writeheader()
        wr.writerows(SUM)

    md = ["| Dataset | Payload | Method | Workers | n | Total (s) | +- SD | Img/s | "
          "Speedup | Efficiency | Busy cores | ctx/s | RAM +MB |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SUM:
        md.append("| " + " | ".join([
            s["dataset"], s["payload"], s["mode"], str(s["workers"]), str(s["n"]),
            format(s["total_s_mean"], ".2f"), format(s["total_s_sd"], ".2f"),
            format(s["throughput_ips"], ".1f"),
            format(s["speedup"], ".2f") if s["speedup"] else "-",
            format(s["efficiency"], ".2f") if s["efficiency"] else "-",
            fmt(s["busy_cores"], ".2f"), fmt(s["ctx_switches_per_s"], ".0f"),
            fmt(s["ram_delta_mb"], ".0f")]) + " |")
    with open(SUMMARY_MD, "w") as f:
        f.write(NL.join(md) + NL)

    for ds in FOUND:
        log("")
        log(BAR)
        log("SUMMARY - " + ds + "   (speedup vs sequential with the SAME payload)")
        log(BAR)
        log("payload mode         w" + "        total s".rjust(15) + "     img/s"
            + "  speedup" + "    eff" + "  cores" + "     ctx/s" + "  RAM MB")
        for s in [x for x in SUM if x["dataset"] == ds]:
            log(s["payload"].ljust(8) + s["mode"].ljust(11) + str(s["workers"]).rjust(3)
                + format(s["total_s_mean"], "9.2f") + "+-" + format(s["total_s_sd"], "<5.2f")
                + format(s["throughput_ips"], "10.1f")
                + (format(s["speedup"], "9.2f") if s["speedup"] else "        -")
                + (format(s["efficiency"], "7.2f") if s["efficiency"] else "      -")
                + fmt(s["busy_cores"], "7.2f") + fmt(s["ctx_switches_per_s"], "10.0f")
                + fmt(s["ram_delta_mb"], "9.0f"))

        best = max([x for x in SUM if x["dataset"] == ds and x["speedup"]],
                   key=lambda x: x["speedup"], default=None)
        if best:
            log("  best: " + best["mode"] + " x" + str(best["workers"]) + " ("
                + best["payload"] + " payload) = " + format(best["speedup"], ".2f") + "x")
        th = [x for x in SUM if x["dataset"] == ds and x["mode"] == "thread"
              and x["payload"] == "scalar" and x["speedup"] is not None]
        if th:
            msg = "  threads peak at " + format(max(x["speedup"] for x in th), ".2f") + "x"
            cores = [x["busy_cores"] for x in th if x["busy_cores"] is not None]
            if cores:
                msg += " and cap at " + format(max(cores), ".2f") + " busy cores of " + str(N_LOG)
            log(msg)

log("")
log(BAR)
log("CROSS-HARDWARE COMPARISON")
log(BAR)
log("This machine : " + str(N_PHYS) + " physical / " + str(N_LOG) + " logical cores")
log("Sandbox      : 4 physical / 8 logical cores")
log("Sandbox results for reference, scalar payload, best process speedup:")
log("  RGB 5.03x at 8 workers, MS 4.54x at 8 workers")
log("  threads never beat sequential, capping at 2.4 busy cores of 8")
log("Expect the peak here to sit near this machine's logical core count, and")
log("the ceiling to be lower, because there are fewer physical cores to use.")

log("")
log("files written:")
for p in sorted(glob.glob(os.path.join(OUT, "layerA_" + MODE.lower() + "_*"))):
    log("  " + p + "  (" + format(os.path.getsize(p) / 1024, ".1f") + " KB)")
log("total wall time " + format((time.time() - T_START) / 60, ".1f") + " min")
