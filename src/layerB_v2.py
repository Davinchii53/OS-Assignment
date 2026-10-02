# ============================================================================
# EuroSAT - Layer B v2: Sequential vs Multithreading vs Multiprocessing
# Training pipeline benchmark, built for one long headless Kaggle run.
#
# HOW TO USE ON KAGGLE
#   1. MODE = "SMOKE", run it interactively (about 4 min). Check it ends clean.
#   2. MODE = "FULL", then Save Version -> Save & Run All (Commit).
#      Then stop the interactive session so it does not burn GPU quota.
#
# Changes from v1
#   - multithreading loader added (ThreadPoolExecutor), so all 3 methods run
#   - every method gets identical batches (shared seeded order, verified)
#   - 3 repeats, config order shuffled per repeat block, cooldown between runs
#   - page-cache warm-up pass before any timing
#   - startup, epoch 1 and steady-state epochs timed separately
#   - GPU clock, temperature, power and throttle reasons logged
#   - results written after every run; resumable; stops before MAX_HOURS
#   - the made-up oversubscription penalty is gone; only the overlap bound stays
#
# Changes in v2.1 (after the Kaggle SMOKE run)
#   - GPU heat soak before timing: the T4 clock fell 1044 -> 870 MHz as it
#     warmed from 51 to 72 C, as large as the differences between configs
#   - short cooldown, so the GPU stays at its steady temperature between runs
#   - extra config "sequential_nopin": sequential without pinned memory, to
#     test whether pinned memory is what lets sequential overlap with the GPU
#
# PASTE-SAFE: there is no backslash character anywhere in this cell.
# ============================================================================
import os

# ---------------------------------------------------------------- CONFIG ----
MODE = "SMOKE"          # "SMOKE" check run | "QUICK" 8k images | "FULL" all 27k
DATASET = "RGB"         # "RGB" or "MS"
THREAD_COUNTS = [1, 2, 4, 8]           # multithreading loader
PROCESS_COUNTS = [1, 2, 3, 4, 6, 8]    # multiprocessing DataLoader workers
BATCH = 64
TORCH_THREADS = 2       # Kaggle has 2 physical cores
GPU_INDEX = 0           # Kaggle gives 2x T4; use one
SEED = 42
SAMPLE_HZ = 4.0         # resource sampling rate
MAX_HOURS = 9.0         # stop cleanly before the Kaggle session limit
RESUME = True           # skip runs already in this session's CSV
LOADER_TIMEOUT = 300    # seconds; a hung worker raises instead of freezing

PIN_ABLATION = True     # also run sequential without pinned memory

# soak_min / soak_max: seconds of GPU heat soak before any timing
PRESETS = {
    "SMOKE": {"subset": 1280, "epochs": 2, "repeats": 1, "cooldown": 2, "soak_min": 20, "soak_max": 45},
    "QUICK": {"subset": 8000, "epochs": 2, "repeats": 3, "cooldown": 3, "soak_min": 60, "soak_max": 180},
    "FULL": {"subset": None, "epochs": 2, "repeats": 3, "cooldown": 3, "soak_min": 60, "soak_max": 180},
}
_P = PRESETS[MODE]
SUBSET, EPOCHS, REPEATS, COOLDOWN = _P["subset"], _P["epochs"], _P["repeats"], _P["cooldown"]
SOAK_MIN, SOAK_MAX = _P["soak_min"], _P["soak_max"]

# Must be set BEFORE torch / numpy are imported, or they are ignored.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"      # CUDA index == NVML index
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(TORCH_THREADS)

import sys, glob, time, json, csv, gc, random, platform, threading, subprocess, warnings
import multiprocessing as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models import resnet18
from PIL import Image

warnings.filterwarnings("ignore", message="This DataLoader will create")
torch.set_num_threads(TORCH_THREADS)
torch.backends.cudnn.benchmark = True

T_START = time.time()
ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working" if ON_KAGGLE else "results"
os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEV.type == "cuda"
PIN = CUDA              # same pin_memory setting for every method
NL = chr(10)
BAR = "=" * 78
TAG = "layerB_" + DATASET.lower() + "_" + MODE.lower()


def log(*a):
    print(*a, flush=True)


def out_path(name):
    return os.path.join(OUT, TAG + "_" + name)


RUNS_CSV = out_path("runs.csv")
EPOCHS_CSV = out_path("epochs.csv")
TS_CSV = out_path("timeseries.csv")
SUMMARY_CSV = out_path("summary.csv")
SUMMARY_MD = out_path("summary.md")

HARNESS_VERSION = "2.1"
# Only resume from files written by this exact harness version; anything older
# is removed so results from different code versions never get mixed.
_CFG_PATH = out_path("config.json")
_stale = True
if RESUME and os.path.exists(_CFG_PATH):
    try:
        with open(_CFG_PATH) as _fh:
            _stale = json.load(_fh).get("version") != HARNESS_VERSION
    except Exception:
        _stale = True
if not RESUME or _stale:
    _removed = 0
    for _p in (RUNS_CSV, EPOCHS_CSV, TS_CSV, SUMMARY_CSV, SUMMARY_MD, _CFG_PATH, out_path("environment.json")):
        if os.path.exists(_p):
            os.remove(_p)
            _removed += 1
    if _removed:
        log(f"removed {_removed} old result files (older harness version, or RESUME = False)")

log(BAR)
log(f"mode={MODE}  dataset={DATASET}  epochs={EPOCHS}  repeats={REPEATS}  batch={BATCH}")
log(f"platform={'kaggle' if ON_KAGGLE else 'sandbox'}  device={DEV}  torch_threads={torch.get_num_threads()}")


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
log(f"cpu: {CPU_MODEL}  cores {N_PHYS} physical / {N_LOG} logical")

# ------------------------------------------------------------ gpu probe ----
# NVML throttle-reason bits that actually cap performance
BIT_SW_POWER, BIT_HW_SLOW, BIT_SW_THERM, BIT_HW_THERM, BIT_HW_BRAKE = 4, 8, 32, 64, 128
LIMIT_MASK = BIT_SW_POWER | BIT_HW_SLOW | BIT_SW_THERM | BIT_HW_THERM | BIT_HW_BRAKE
THERMAL_MASK = BIT_HW_SLOW | BIT_SW_THERM | BIT_HW_THERM
POWER_MASK = BIT_SW_POWER | BIT_HW_BRAKE


class GpuProbe:
    def __init__(self):
        self.backend = "none"
        self.nv = self.h = self.reasons_fn = self.smi_fields = None
        if not CUDA:
            return
        nv = None
        try:
            import pynvml as nv
        except Exception:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nvidia-ml-py"],
                               capture_output=True, timeout=180)
                import pynvml as nv
            except Exception:
                nv = None
        if nv is not None:
            try:
                nv.nvmlInit()
                self.nv, self.h = nv, nv.nvmlDeviceGetHandleByIndex(GPU_INDEX)
                self.reasons_fn = (getattr(nv, "nvmlDeviceGetCurrentClocksEventReasons", None)
                                   or getattr(nv, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
                self.backend = "nvml"
                return
            except Exception:
                pass
        for rf in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
            fields = "utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw," + rf
            try:
                r = subprocess.run(["nvidia-smi", "--id=" + str(GPU_INDEX), "--query-gpu=" + fields,
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
                if r.returncode == 0 and r.stdout.strip():
                    self.smi_fields, self.backend = fields, "nvidia-smi"
                    return
            except Exception:
                pass

    def read(self):
        d = {}
        if self.backend == "nvml":
            nv, h = self.nv, self.h
            try:
                d["gpu_util"] = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                d["vram_mb"] = nv.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 2
            except Exception:
                pass
            try:
                d["sm_clock"] = float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
            except Exception:
                pass
            try:
                d["temp"] = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                pass
            try:
                if self.reasons_fn is not None:
                    d["reasons"] = int(self.reasons_fn(h))
            except Exception:
                pass
        elif self.backend == "nvidia-smi":
            try:
                r = subprocess.run(["nvidia-smi", "--id=" + str(GPU_INDEX), "--query-gpu=" + self.smi_fields,
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
                parts = [x.strip() for x in r.stdout.strip().split(",")]
                for name, val in zip(["gpu_util", "vram_mb", "sm_clock", "temp", "power_w"], parts[:5]):
                    try:
                        d[name] = float(val)
                    except ValueError:
                        pass
                try:
                    d["reasons"] = int(parts[5], 16)
                except Exception:
                    pass
            except Exception:
                pass
        return d


GPU = GpuProbe()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA else "none"
log(f"gpu: {GPU_NAME}  telemetry backend: {GPU.backend}")


class Monitor(threading.Thread):
    """Samples CPU, RAM, context switches and GPU telemetry during one run."""

    def __init__(self):
        super().__init__(daemon=True)
        hz = min(SAMPLE_HZ, 1.0) if GPU.backend == "nvidia-smi" else SAMPLE_HZ
        self.interval = 1.0 / hz
        self.stop_evt = threading.Event()
        self.samples = []
        self.proc = psutil.Process()
        self.t0 = time.perf_counter()
        self.ctx0 = psutil.cpu_stats().ctx_switches

    def run(self):
        psutil.cpu_percent(None)
        while not self.stop_evt.wait(self.interval):
            s = {"t": round(time.perf_counter() - self.t0, 3), "cpu": psutil.cpu_percent(None)}
            try:
                rss = self.proc.memory_info().rss
                for ch in self.proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                s["ram_proc_mb"] = rss / 1024 ** 2
            except Exception:
                pass
            s["ram_sys_mb"] = psutil.virtual_memory().used / 1024 ** 2
            s.update(GPU.read())
            self.samples.append(s)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=10)
        dur = max(time.perf_counter() - self.t0, 1e-9)
        ctx = (psutil.cpu_stats().ctx_switches - self.ctx0) / dur
        S = self.samples

        def col(k):
            return [s[k] for s in S if k in s]

        def agg(k, fn):
            v = col(k)
            return round(float(fn(v)), 2) if v else None

        a = {
            "n_samples": len(S),
            "cpu_mean": agg("cpu", np.mean), "cpu_max": agg("cpu", np.max),
            "ram_proc_max_mb": agg("ram_proc_mb", np.max), "ram_sys_max_mb": agg("ram_sys_mb", np.max),
            "ctx_switches_per_s": round(ctx, 1),
            "gpu_util_mean": agg("gpu_util", np.mean), "vram_max_mb": agg("vram_mb", np.max),
            "sm_clock_mean": agg("sm_clock", np.mean), "sm_clock_min": agg("sm_clock", np.min),
            "temp_mean": agg("temp", np.mean), "temp_max": agg("temp", np.max),
            "power_mean_w": agg("power_w", np.mean),
        }
        a["busy_cores"] = round(a["cpu_mean"] * N_LOG / 100.0, 2) if a["cpu_mean"] is not None else None
        a["gpu_idle_pct"] = round(100.0 - a["gpu_util_mean"], 2) if a["gpu_util_mean"] is not None else None
        rs = col("reasons")
        if rs:
            n = float(len(rs))
            a["throttle_limit_pct"] = round(100 * sum(1 for r in rs if r & LIMIT_MASK) / n, 1)
            a["throttle_thermal_pct"] = round(100 * sum(1 for r in rs if r & THERMAL_MASK) / n, 1)
            a["throttle_power_pct"] = round(100 * sum(1 for r in rs if r & POWER_MASK) / n, 1)
        return a, S


# --------------------------------------------------------------- dataset ----
EXT = "jpg" if DATASET == "RGB" else "tif"
IN_CH = 3 if DATASET == "RGB" else 13
if DATASET == "MS":
    try:
        import tifffile
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"], capture_output=True)
        import tifffile


def find_root(ext):
    base = "/kaggle/input" if ON_KAGGLE else "data/ext"
    hits = sorted(glob.glob(os.path.join(base, "**", "*." + ext), recursive=True))
    if not hits:
        return None, []
    root = os.path.dirname(os.path.dirname(hits[0]))
    return root, [h for h in hits if h.startswith(root + os.sep)]


ROOT, ALL_FILES = find_root(EXT)
assert ROOT, "no ." + EXT + " files found -- is the dataset attached to the notebook?"
CLASSES = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
C2I = {c: i for i, c in enumerate(CLASSES)}
if SUBSET:
    per = SUBSET // len(CLASSES)
    FILES = []
    for c in CLASSES:
        FILES += sorted(glob.glob(os.path.join(ROOT, c, "*." + EXT)))[:per]
else:
    FILES = ALL_FILES
N_IMG = len(FILES)
log(f"root={ROOT}")
log(f"files found={len(ALL_FILES)}  classes={len(CLASSES)}  using={N_IMG}")


def decode(path):
    if DATASET == "RGB":
        a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    else:
        a = tifffile.imread(path).astype(np.float32)
    a = (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 1e-6)
    return np.ascontiguousarray(a.transpose(2, 0, 1))


class EuroSAT(Dataset):
    def __init__(self, files):
        self.files = files
        self.labels = [C2I[os.path.basename(os.path.dirname(f))] for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return torch.from_numpy(decode(self.files[i])), self.labels[i]


DS = EuroSAT(FILES)


class EpochBatches:
    """Seeded batch order shared by every method: same batches, same order.
    set_epoch() is explicit because DataLoader may call iter() more than once."""

    def __init__(self, n, batch):
        self.n, self.batch, self.epoch = n, batch, 0

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n // self.batch

    def __iter__(self):
        perm = np.random.default_rng(SEED * 1000 + self.epoch).permutation(self.n)
        for i in range(len(self)):
            yield perm[i * self.batch:(i + 1) * self.batch].tolist()


class ThreadLoader:
    """Multithreading loader: each thread builds whole batches, like a
    DataLoader worker, but inside one process so it shares the GIL.
    Keeps 2 x threads batches in flight (same as prefetch_factor=2)."""

    def __init__(self, ds, batch_sampler, n_threads, pin):
        self.ds, self.bs, self.pin = ds, batch_sampler, pin
        self.depth = 2 * n_threads
        self.pool = ThreadPoolExecutor(max_workers=n_threads)

    def _make(self, idxs):
        items = [self.ds[i] for i in idxs]
        x = torch.stack([it[0] for it in items])
        y = torch.tensor([it[1] for it in items])
        if self.pin:
            x, y = x.pin_memory(), y.pin_memory()
        return x, y

    def __iter__(self):
        batches = iter(self.bs)
        q = deque()
        for _ in range(self.depth):          # prefetch starts immediately
            idx = next(batches, None)
            if idx is None:
                break
            q.append(self.pool.submit(self._make, idx))
        return self._drain(q, batches)

    def _drain(self, q, batches):
        while q:
            fut = q.popleft()
            idx = next(batches, None)
            if idx is not None:
                q.append(self.pool.submit(self._make, idx))
            yield fut.result()

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def make_loader(method, workers, bs):
    if method == "thread":
        return ThreadLoader(DS, bs, workers, PIN)
    pin = False if method == "sequential_nopin" else PIN
    kw = {"batch_sampler": bs, "num_workers": workers, "pin_memory": pin}
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=2, timeout=LOADER_TIMEOUT)
    return DataLoader(DS, **kw)


def close_loader(loader):
    if isinstance(loader, ThreadLoader):
        loader.close()


# ----------------------------------------------------------------- model ----
def make_model():
    m = resnet18(weights=None, num_classes=len(CLASSES))
    m.conv1 = nn.Conv2d(IN_CH, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m


# --------------------------------------------------- page-cache warm-up ----
log("")
log("--- warm-up: reading every file once so all runs see a warm page cache ---")


def _read(p):
    with open(p, "rb") as f:
        return len(f.read())


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as ex:
    nbytes = sum(ex.map(_read, FILES, chunksize=64))
log(f"read {nbytes / 1024 ** 2:.1f} MB in {time.perf_counter() - t0:.1f}s")

# ------------------------------------------------ batch equivalence check ----
log("")
log("--- checking that all three methods produce identical batches ---")


def first_batches(method, workers, k=2):
    bs = EpochBatches(N_IMG, BATCH)
    bs.set_epoch(0)
    if method == "thread":
        ld = ThreadLoader(DS, bs, workers, False)
    else:
        ld = DataLoader(DS, batch_sampler=bs, num_workers=workers)
    got = []
    for x, y in ld:
        got.append((x.clone(), y.clone()))
        if len(got) == k:
            break
    close_loader(ld)
    del ld
    return got


ref = first_batches("sequential", 0)
BATCHES_IDENTICAL = True
for m_, w_ in (("thread", 2), ("process", 2)):
    other = first_batches(m_, w_)
    same = len(other) == len(ref) and all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(ref, other))
    BATCHES_IDENTICAL = BATCHES_IDENTICAL and same
    log(f"  sequential vs {m_} x{w_}: {'IDENTICAL' if same else 'DIFFERENT'}")
assert BATCHES_IDENTICAL, "loaders produce different batches -- comparison would be unfair"
del ref, other
gc.collect()

# ------------------------------------------------------------- baselines ----
log("")
log("--- baselines (for the Amdahl bound and the ETA) ---")
sample = FILES[:300]
t0 = time.perf_counter()
for f in sample:
    decode(f)
DECODE_IPS = len(sample) / (time.perf_counter() - t0)


def heat_soak():
    """Run synthetic training until the GPU temperature levels off, so every
    timed run (and the baseline below) sees the same steady-state clock."""
    if not CUDA or SOAK_MAX <= 0:
        return None
    log("")
    log(f"--- GPU heat soak: until temperature is stable (min {SOAK_MIN}s, max {SOAK_MAX}s) ---")
    try:
        torch.manual_seed(SEED)
        m = make_model().to(DEV)
        o = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
        lf = nn.CrossEntropyLoss()
        x = torch.randn(BATCH, IN_CH, 64, 64, device=DEV)
        y = torch.randint(0, len(CLASSES), (BATCH,), device=DEV)
        g0 = GPU.read()
        t0_ = time.perf_counter()
        hist, next_log, el, temp, clk = [], 0.0, 0.0, g0.get("temp"), g0.get("sm_clock")
        log(f"     0s  temp {temp}  clock {clk}")
        while True:
            for _ in range(10):
                o.zero_grad()
                lf(m(x), y).backward()
                o.step()
            torch.cuda.synchronize()
            el = time.perf_counter() - t0_
            g = GPU.read()
            temp, clk = g.get("temp"), g.get("sm_clock")
            hist.append((el, temp))
            if el >= next_log + 15:
                next_log = el
                log(f"  {el:5.0f}s  temp {temp}  clock {clk}")
            if el >= SOAK_MAX:
                break
            if el >= SOAK_MIN and temp is not None:
                older = [tp for tt, tp in hist if tt <= el - 30 and tp is not None]
                if older and temp - older[-1] <= 1.0:
                    break
        del m, o, x, y
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  done after {el:.0f}s: temp {g0.get('temp')} -> {temp} C, clock {g0.get('sm_clock')} -> {clk} MHz")
        return {"seconds": round(el, 1), "temp_start": g0.get("temp"), "temp_end": temp,
                "clock_start": g0.get("sm_clock"), "clock_end": clk}
    except Exception as e:
        log("heat soak skipped: " + type(e).__name__ + ": " + str(e))
        return None


SOAK = heat_soak()

torch.manual_seed(SEED)
_m = make_model().to(DEV)
_o = torch.optim.SGD(_m.parameters(), lr=0.01, momentum=0.9)
_l = nn.CrossEntropyLoss()
_x = torch.randn(BATCH, IN_CH, 64, 64, device=DEV)
_y = torch.randint(0, len(CLASSES), (BATCH,), device=DEV)
for _ in range(5):
    _o.zero_grad()
    _l(_m(_x), _y).backward()
    _o.step()
if CUDA:
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _o.zero_grad()
    _l(_m(_x), _y).backward()
    _o.step()
if CUDA:
    torch.cuda.synchronize()
TRAIN_IPS = BATCH * 20 / (time.perf_counter() - t0)
del _m, _o, _x, _y
gc.collect()
if CUDA:
    torch.cuda.empty_cache()

_td, _tt = 1.0 / DECODE_IPS, 1.0 / TRAIN_IPS
LOAD_SHARE = _td / (_td + _tt)
S_MAX = 1.0 / (1.0 - LOAD_SHARE)
SEQ_IPS = 1.0 / (_td + _tt)
log(f"decode, 1 core      : {DECODE_IPS:8.1f} img/s")
log(f"train step (no I/O) : {TRAIN_IPS:8.1f} img/s")
log(f"loading share       : {LOAD_SHARE * 100:5.1f}%   ceiling vs a FULLY SERIAL loop {S_MAX:.2f}x")
log(f"per batch           : decode {1000 * BATCH / DECODE_IPS:.0f} ms on 1 core, GPU step {1000 * BATCH / TRAIN_IPS:.0f} ms")


def bound_ips(method, w):
    """Ideal bound: no GIL, no contention. An upper limit, not a forecast.
    Pinned + non_blocking copies let even the sequential loop overlap decode
    with GPU work, so only the pageable (nopin) loop is fully serial."""
    if method == "sequential_nopin":
        return SEQ_IPS
    if w == 0:
        return min(TRAIN_IPS, DECODE_IPS)
    return min(TRAIN_IPS, DECODE_IPS * min(w, N_PHYS or 1))


# ------------------------------------------------------------------- plan ----
CONFIGS = ([("sequential", 0)] + ([("sequential_nopin", 0)] if PIN_ABLATION and CUDA else [])
           + [("thread", n) for n in THREAD_COUNTS] + [("process", n) for n in PROCESS_COUNTS])
PLAN = []
for rep in range(REPEATS):
    block = list(CONFIGS)
    random.Random(SEED + rep).shuffle(block)        # randomised order per repeat block
    PLAN += [(m, w, rep) for m, w in block]


def eta_seconds(m, w):
    if m == "sequential_nopin":
        est = SEQ_IPS
    elif m == "thread":
        est = 0.85 * min(TRAIN_IPS, DECODE_IPS)
    else:
        est = 0.85 * bound_ips(m, w)
    return N_IMG * EPOCHS / est + COOLDOWN + 3


ETA = sum(eta_seconds(m, w) for m, w, _ in PLAN)
log("")
log(f"plan: {len(CONFIGS)} configs x {REPEATS} repeats = {len(PLAN)} runs")
log(f"rough ETA: {ETA / 60:.0f} min  (time budget {MAX_HOURS:.1f} h)")

# --------------------------------------------------- environment + config ----
ENV = {
    "platform": "kaggle" if ON_KAGGLE else "sandbox",
    "cpu_model": CPU_MODEL, "cpu_cores_physical": N_PHYS, "cpu_cores_logical": N_LOG,
    "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
    "gpu_name": GPU_NAME, "gpu_count_visible": torch.cuda.device_count() if CUDA else 0,
    "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2) if CUDA else None,
    "os": platform.system() + " " + platform.release(),
    "python": platform.python_version(),
    "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
    "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__,
    "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if CUDA else None,
    "mp_start_method": mp.get_start_method(allow_none=True) or mp.get_context().get_start_method(),
    "torch_threads": torch.get_num_threads(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
}
CFG = {
    "version": HARNESS_VERSION,
    "mode": MODE, "dataset": DATASET, "n_images": N_IMG, "classes": CLASSES, "epochs": EPOCHS,
    "repeats": REPEATS, "batch": BATCH, "seed": SEED, "cooldown_s": COOLDOWN,
    "thread_counts": THREAD_COUNTS, "process_counts": PROCESS_COUNTS,
    "model": "resnet18, conv1 3x3 stride 1, maxpool removed", "optimizer": "SGD lr 0.01 momentum 0.9",
    "pin_memory": PIN, "pin_ablation": PIN_ABLATION and CUDA, "heat_soak": SOAK,
    "prefetch_factor": 2, "persistent_workers": True, "cudnn_benchmark": True,
    "gpu_telemetry_backend": GPU.backend, "sample_hz": SAMPLE_HZ,
    "batches_identical_across_methods": BATCHES_IDENTICAL, "randomised_order": True,
    "baseline_decode_ips": round(DECODE_IPS, 2), "baseline_train_ips": round(TRAIN_IPS, 2),
    "loading_share": round(LOAD_SHARE, 4), "amdahl_ceiling": round(S_MAX, 3),
    "plan": [list(p) for p in PLAN],
}
with open(out_path("environment.json"), "w") as f:
    json.dump(ENV, f, indent=2)
with open(out_path("config.json"), "w") as f:
    json.dump(CFG, f, indent=2)

# -------------------------------------------------------------- csv helpers ----
RUN_FIELDS = [
    "dataset", "mode", "method", "workers", "repeat", "order_idx", "status",
    "n_images", "epochs", "batch", "torch_threads", "cores_physical", "cores_logical", "gpu_name",
    "total_s", "startup_s", "epoch1_s", "steady_epoch_s", "throughput_ips", "steady_ips",
    "loader_block_pct", "bound_ips", "train_acc_last",
    "cpu_mean", "cpu_max", "busy_cores", "ram_delta_mb", "ram_proc_max_mb", "ram_sys_max_mb",
    "ctx_switches_per_s",
    "gpu_util_mean", "gpu_idle_pct", "vram_max_mb", "sm_clock_mean", "sm_clock_min",
    "temp_start", "temp_mean", "temp_max", "power_mean_w",
    "throttle_limit_pct", "throttle_thermal_pct", "throttle_power_pct", "n_samples",
]
EPOCH_FIELDS = ["method", "workers", "repeat", "epoch", "epoch_s", "images", "ips", "block_s", "train_acc"]
TS_FIELDS = ["method", "workers", "repeat", "t", "cpu", "ram_proc_mb", "ram_sys_mb",
             "gpu_util", "vram_mb", "sm_clock", "temp", "power_w", "reasons"]


def append_rows(p, fields, rows):
    new = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def read_rows(p):
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- one run ----
def run_one(method, w, rep, order_idx):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = make_model().to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lossf = nn.CrossEntropyLoss()
    model.train()

    bs = EpochBatches(N_IMG, BATCH)
    loader = make_loader(method, w, bs)
    temp_start = GPU.read().get("temp")
    # System RAM in use before the run. Forked workers share pages with the
    # parent, so summing their RSS double-counts; the system delta does not.
    ram_sys_before = psutil.virtual_memory().used / 1024 ** 2
    epochs, block, n_seen, startup = [], 0.0, 0, None
    mon = None
    try:
        t_run = time.perf_counter()
        bs.set_epoch(0)
        it = iter(loader)                 # forks workers here, before the monitor thread
        mon = Monitor()
        mon.start()
        for ep in range(EPOCHS):
            if ep > 0:
                bs.set_epoch(ep)
                it = iter(loader)
            correct = torch.zeros((), dtype=torch.long, device=DEV)
            n_ep, blk_ep = 0, 0.0
            t_ep = time.perf_counter() if ep > 0 else t_run
            t_mark = t_ep
            for xb, yb in it:
                now = time.perf_counter()
                blk_ep += now - t_mark              # main process blocked on data
                if startup is None:
                    startup = now - t_run           # time to first batch (incl. worker start)
                xb = xb.to(DEV, non_blocking=True)
                yb = yb.to(DEV, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                lossf(out, yb).backward()
                opt.step()
                correct += (out.argmax(1) == yb).sum()   # no per-step sync
                n_ep += xb.size(0)
                t_mark = time.perf_counter()
            if CUDA:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t_ep
            acc = correct.item() / max(n_ep, 1)
            epochs.append({"epoch": ep, "epoch_s": round(dt, 3), "images": n_ep,
                           "ips": round(n_ep / dt, 2), "block_s": round(blk_ep, 3),
                           "train_acc": round(acc, 4)})
            block += blk_ep
            n_seen += n_ep
        total = time.perf_counter() - t_run
    finally:
        res, samples = mon.stop() if mon is not None else ({}, [])
        close_loader(loader)
        del loader, model, opt
        gc.collect()
        if CUDA:
            torch.cuda.empty_cache()

    steady = epochs[1:]
    steady_s = sum(e["epoch_s"] for e in steady)
    row = {
        "dataset": DATASET, "mode": MODE, "method": method, "workers": w, "repeat": rep,
        "order_idx": order_idx, "status": "ok", "n_images": N_IMG, "epochs": EPOCHS, "batch": BATCH,
        "torch_threads": torch.get_num_threads(), "cores_physical": N_PHYS, "cores_logical": N_LOG,
        "gpu_name": GPU_NAME,
        "total_s": round(total, 3), "startup_s": round(startup or 0.0, 3),
        "epoch1_s": epochs[0]["epoch_s"],
        "steady_epoch_s": round(steady_s / len(steady), 3) if steady else None,
        "throughput_ips": round(n_seen / total, 2),
        "steady_ips": round(sum(e["images"] for e in steady) / steady_s, 2) if steady else None,
        "loader_block_pct": round(100.0 * block / total, 2),
        "bound_ips": round(bound_ips(method, w), 2),
        "train_acc_last": epochs[-1]["train_acc"], "temp_start": temp_start,
    }
    row.update(res)
    if row.get("ram_sys_max_mb") is not None:
        row["ram_delta_mb"] = round(row["ram_sys_max_mb"] - ram_sys_before, 1)
    key = {"method": method, "workers": w, "repeat": rep}
    return row, [dict(key, **e) for e in epochs], [dict(key, **s) for s in samples]


# ----------------------------------------------------------------- sweep ----
done = {(r["method"], int(r["workers"]), int(r["repeat"]))
        for r in read_rows(RUNS_CSV) if r.get("status") == "ok"}
if done:
    log(f"resuming: {len(done)} runs already in {RUNS_CSV}")

log("")
log(BAR)
log("running sweep -> " + RUNS_CSV)
log(BAR)
durations = []
for order_idx, (method, w, rep) in enumerate(PLAN):
    if (method, w, rep) in done:
        continue
    elapsed = time.time() - T_START
    est = float(np.mean(durations)) if durations else eta_seconds(method, w)
    if elapsed + est > MAX_HOURS * 3600:
        log(f"time budget reached after {elapsed / 3600:.2f} h -- stopping cleanly, results so far are saved")
        break
    time.sleep(COOLDOWN)
    t_wall = time.time()
    try:
        row, ep_rows, ts_rows = run_one(method, w, rep, order_idx)
    except Exception as e:
        log(f"[{order_idx + 1:2d}/{len(PLAN)}] {method:<16s} w={w} r={rep}  ERROR: {type(e).__name__}: {e}")
        append_rows(RUNS_CSV, RUN_FIELDS, [{"dataset": DATASET, "mode": MODE, "method": method, "workers": w,
                                            "repeat": rep, "order_idx": order_idx,
                                            "status": "error: " + type(e).__name__}])
        continue
    durations.append(time.time() - t_wall)
    append_rows(RUNS_CSV, RUN_FIELDS, [row])
    append_rows(EPOCHS_CSV, EPOCH_FIELDS, ep_rows)
    append_rows(TS_CSV, TS_FIELDS, ts_rows)
    left = sum(1 for p in PLAN[order_idx + 1:] if p not in done)
    log(f"[{order_idx + 1:2d}/{len(PLAN)}] {method:<16s} w={w} r={rep}  "
        f"{row['throughput_ips']:7.1f} ips  total {row['total_s']:7.1f}s  "
        f"start {row['startup_s']:5.2f}s  block {row['loader_block_pct']:5.1f}%  "
        f"cpu {row['busy_cores']} cores  gpu {row['gpu_util_mean']}%  "
        f"clk {row['sm_clock_mean']}  temp {row['temp_mean']}  acc {row['train_acc_last']:.3f}  "
        f"| ~{left * float(np.mean(durations)) / 60:.0f} min left")

# --------------------------------------------------------------- summary ----
rows = [r for r in read_rows(RUNS_CSV) if r.get("status") == "ok"]
for r in rows:
    for k in list(r):
        if k not in ("dataset", "mode", "method", "status", "gpu_name"):
            r[k] = num(r[k])
    r["workers"] = int(r["workers"])


def stats(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


ORDER = {"sequential": 0, "sequential_nopin": 1, "thread": 2, "process": 3}
groups = {}
for r in rows:
    groups.setdefault((r["method"], r["workers"]), []).append(r)
keys = sorted(groups, key=lambda k: (ORDER[k[0]], k[1]))
seq_total = stats([r["total_s"] for r in groups.get(("sequential", 0), [])])[0]
seq_steady = stats([r["steady_epoch_s"] for r in groups.get(("sequential", 0), [])])[0]

SUM = []
for k in keys:
    g = groups[k]
    t_m, t_sd = stats([r["total_s"] for r in g])
    i_m, i_sd = stats([r["throughput_ips"] for r in g])
    s_m, s_sd = stats([r["steady_epoch_s"] for r in g])
    sp = seq_total / t_m if seq_total and t_m else None
    ssp = seq_steady / s_m if seq_steady and s_m else None

    def m_(field):
        return stats([r[field] for r in g])[0]

    SUM.append({
        "method": k[0], "workers": k[1], "n": len(g),
        "total_s_mean": t_m, "total_s_sd": t_sd, "steady_epoch_s_mean": s_m, "steady_epoch_s_sd": s_sd,
        "epoch1_s_mean": m_("epoch1_s"), "startup_s_mean": m_("startup_s"),
        "ips_mean": i_m, "ips_sd": i_sd, "cv_pct": 100 * i_sd / i_m if i_m else None,
        "speedup": sp, "steady_speedup": ssp,
        "efficiency": sp / k[1] if sp and k[1] > 0 else None,
        "loader_block_pct": m_("loader_block_pct"), "busy_cores": m_("busy_cores"),
        "cpu_mean": m_("cpu_mean"), "ram_delta_mb": m_("ram_delta_mb"),
        "ram_proc_max_mb": m_("ram_proc_max_mb"),
        "ctx_switches_per_s": m_("ctx_switches_per_s"),
        "gpu_util_mean": m_("gpu_util_mean"), "gpu_idle_pct": m_("gpu_idle_pct"),
        "vram_max_mb": m_("vram_max_mb"), "sm_clock_mean": m_("sm_clock_mean"),
        "temp_mean": m_("temp_mean"), "throttle_limit_pct": m_("throttle_limit_pct"),
        "train_acc_last": m_("train_acc_last"),
    })


def f(v, spec):
    return format(v, spec) if v is not None else "-"


log("")
log(BAR)
log(f"SUMMARY  {DATASET} / {MODE}  ({len(rows)} ok runs)  speedup = T_sequential / T_parallel")
log(BAR)
hdr = (f"{'method':<16} {'w':>2} {'n':>2} {'total s':>14} {'steady ep s':>12} {'img/s':>14} "
       f"{'spdup':>6} {'st.spd':>6} {'eff':>5} {'block%':>6} {'cores':>5} {'RAM+MB':>7} "
       f"{'gpu%':>5} {'clkMHz':>6} {'temp':>5} {'lim%':>5} {'acc':>5}")
log(hdr)
for s in SUM:
    log(f"{s['method']:<16} {s['workers']:>2} {s['n']:>2} "
        f"{f(s['total_s_mean'], '7.1f'):>7}+-{f(s['total_s_sd'], '<5.1f'):<5} "
        f"{f(s['steady_epoch_s_mean'], '12.1f'):>12} "
        f"{f(s['ips_mean'], '7.1f'):>7}+-{f(s['ips_sd'], '<5.1f'):<5} "
        f"{f(s['speedup'], '6.2f'):>6} {f(s['steady_speedup'], '6.2f'):>6} {f(s['efficiency'], '5.2f'):>5} "
        f"{f(s['loader_block_pct'], '6.1f'):>6} {f(s['busy_cores'], '5.2f'):>5} "
        f"{f(s['ram_delta_mb'], '7.0f'):>7} {f(s['gpu_util_mean'], '5.1f'):>5} "
        f"{f(s['sm_clock_mean'], '6.0f'):>6} {f(s['temp_mean'], '5.1f'):>5} "
        f"{f(s['throttle_limit_pct'], '5.1f'):>5} {f(s['train_acc_last'], '5.3f'):>5}")

SUM_FIELDS = list(SUM[0].keys()) if SUM else []
if SUM:
    with open(SUMMARY_CSV, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=SUM_FIELDS)
        wr.writeheader()
        wr.writerows(SUM)
    md = ["| Method | Workers | n | Total (s) | +- SD | Steady epoch (s) | Img/s | +- SD | Speedup | "
          "Efficiency | Busy cores | RAM +MB | GPU % | GPU idle % | SM clock | Temp | Acc |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SUM:
        md.append("| " + " | ".join([
            s["method"], str(s["workers"]), str(s["n"]), f(s["total_s_mean"], ".2f"), f(s["total_s_sd"], ".2f"),
            f(s["steady_epoch_s_mean"], ".2f"), f(s["ips_mean"], ".1f"), f(s["ips_sd"], ".1f"),
            f(s["speedup"], ".2f"), f(s["efficiency"], ".2f"), f(s["busy_cores"], ".2f"),
            f(s["ram_delta_mb"], ".0f"), f(s["gpu_util_mean"], ".1f"), f(s["gpu_idle_pct"], ".1f"),
            f(s["sm_clock_mean"], ".0f"), f(s["temp_mean"], ".1f"), f(s["train_acc_last"], ".3f")]) + " |")
    with open(SUMMARY_MD, "w") as fh:
        fh.write(NL.join(md) + NL)

# --------------------------------------------------------------- verdicts ----
if SUM and seq_total:
    log("")
    log("--- what the data says ---")
    ranked = sorted(SUM, key=lambda s: s["total_s_mean"])
    b = ranked[0]
    seq_ips = stats([r["throughput_ips"] for r in groups.get(("sequential", 0), [])])[0]
    log(f"fastest: {b['method']} x{b['workers']}  {b['total_s_mean']:.1f}s  speedup {b['speedup']:.2f}x  "
        f"(most any loader could gain over the measured sequential: {TRAIN_IPS / seq_ips:.2f}x)")
    nopin = groups.get(("sequential_nopin", 0), [])
    if nopin and seq_ips:
        np_ips = stats([r["throughput_ips"] for r in nopin])[0]
        np_gpu = stats([r["gpu_util_mean"] for r in nopin])[0]
        sq_gpu = stats([r["gpu_util_mean"] for r in groups[("sequential", 0)]])[0]
        log(f"pinned memory, sequential: {seq_ips:.1f} img/s pinned vs {np_ips:.1f} pageable "
            f"({seq_ips / np_ips:.2f}x); GPU util {f(sq_gpu, '.1f')}% vs {f(np_gpu, '.1f')}%; "
            f"fully serial prediction {SEQ_IPS:.1f}")
    if len(ranked) > 1:
        r2 = ranked[1]
        gap = r2["total_s_mean"] - b["total_s_mean"]
        noise = (b["total_s_sd"] or 0) + (r2["total_s_sd"] or 0)
        if min(b["n"], r2["n"]) < 2:
            verdict = "only 1 repeat -- cannot judge"
        elif gap > noise:
            verdict = "clearly faster"
        else:
            verdict = "NOT distinguishable at this n"
        log(f"runner-up: {r2['method']} x{r2['workers']}  {r2['total_s_mean']:.1f}s  -> gap {gap:.1f}s vs "
            f"combined SD {noise:.1f}s: " + verdict)
    for meth in ("thread", "process"):
        seq_ = sorted([s for s in SUM if s["method"] == meth], key=lambda s: s["workers"])
        if len(seq_) < 2:
            continue
        ips = [s["ips_mean"] for s in seq_]
        peak = seq_[int(np.argmax(ips))]
        mono = all(ips[i + 1] >= ips[i] for i in range(len(ips) - 1))
        log(f"{meth}: img/s by workers " + ", ".join(f"{s['workers']}:{s['ips_mean']:.0f}" for s in seq_)
            + f"  -> peak at {peak['workers']}; "
            + ("rises with every added worker" if mono else "more workers is NOT always faster"))

    gpu_runs = [r for r in rows if r["workers"] > 0 and r.get("sm_clock_mean") is not None]
    if len(gpu_runs) >= 3:
        def corr(a, b_):
            a, b_ = np.asarray(a, float), np.asarray(b_, float)
            if len(a) < 3 or a.std() == 0 or b_.std() == 0:
                return None
            return float(np.corrcoef(a, b_)[0, 1])

        ips_ = [r["throughput_ips"] for r in gpu_runs]
        r_clk = corr(ips_, [r["sm_clock_mean"] for r in gpu_runs])
        r_ord = corr(ips_, [r["order_idx"] for r in gpu_runs])
        r_tmp = corr([r["order_idx"] for r in gpu_runs], [r["temp_mean"] for r in gpu_runs])
        lim = [r["throttle_limit_pct"] for r in gpu_runs if r.get("throttle_limit_pct") is not None]
        log("")
        log("--- GPU throttling check (runs with workers >= 1) ---")
        log(f"corr(img/s, SM clock)   = {f(r_clk, '+.2f')}   strong positive -> clock explains the differences")
        log(f"corr(img/s, run order)  = {f(r_ord, '+.2f')}   strong negative -> later runs slower (drift)")
        log(f"corr(run order, temp)   = {f(r_tmp, '+.2f')}")
        log(f"SM clock range {min(r['sm_clock_mean'] for r in gpu_runs):.0f}-"
            f"{max(r['sm_clock_mean'] for r in gpu_runs):.0f} MHz; "
            f"perf-limit throttle active {f(float(np.mean(lim)) if lim else None, '.1f')}% of samples")

log("")
log("files written:")
for p in sorted(glob.glob(os.path.join(OUT, TAG + "_*"))):
    log(f"  {p}  ({os.path.getsize(p) / 1024:.1f} KB)")
log(f"total wall time {(time.time() - T_START) / 60:.1f} min")
