# ============================================================================
# Step 1 - environment and dataset verification
#
# Records the hardware and software specs required by the assignment, and
# checks the dataset is attached and decodes correctly. Fast, no training.
# Self-contained: defines everything it uses and leaves nothing for later cells.
#
# PASTE-SAFE: no backslash character anywhere in this cell.
# ============================================================================
import os, sys, glob, json, platform, subprocess, multiprocessing as mp

import numpy as np
import torch
import torchvision
import psutil
from PIL import Image

ON_KAGGLE = os.path.isdir("/kaggle/input")
OUT = "/kaggle/working" if ON_KAGGLE else "results"
os.makedirs(OUT, exist_ok=True)
CUDA = torch.cuda.is_available()
BAR = "=" * 70


def cpu_topology():
    """Physical vs logical cores. lscpu is unreliable inside containers, and
    the distinction matters: speedup plateaus near the PHYSICAL core count."""
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

SPECS = {
    "platform": "kaggle" if ON_KAGGLE else "sandbox",
    "cpu_model": CPU_MODEL,
    "cpu_cores_physical": N_PHYS,
    "cpu_cores_logical": N_LOG,
    "hyperthreading": bool(N_PHYS and N_LOG and N_LOG > N_PHYS),
    "ram_total_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
    "gpu_available": CUDA,
    "gpu_name": torch.cuda.get_device_name(0) if CUDA else None,
    "gpu_count": torch.cuda.device_count() if CUDA else 0,
    "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2) if CUDA else None,
    "gpu_capability": (lambda p: str(p.major) + "." + str(p.minor))(torch.cuda.get_device_properties(0)) if CUDA else None,
    "os": platform.system() + " " + platform.release(),
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "gil_enabled": bool(getattr(sys, "_is_gil_enabled", lambda: True)()),
    "framework": "pytorch " + torch.__version__,
    "torchvision": torchvision.__version__,
    "numpy": np.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if CUDA else None,
    "mp_start_method": mp.get_context().get_start_method(),
}

print(BAR)
print("ENVIRONMENT")
print(BAR)
for k, v in SPECS.items():
    print(f"  {k:24s} {v}")

if SPECS["hyperthreading"]:
    print("")
    print(f"  NOTE: {N_LOG} logical CPUs but only {N_PHYS} physical cores.")
    print(f"        Expect CPU-bound speedup to plateau near {N_PHYS}x, not {N_LOG}x.")
    print("        Report both numbers.")
if not CUDA:
    print("")
    print("  NOTE: no GPU. On Kaggle set Settings -> Accelerator -> GPU.")

# ---------------------------------------------------------------- dataset ----
print("")
print(BAR)
print("DATASET")
print(BAR)

EXPECTED_IMAGES = 27000
EXPECTED_CLASSES = 10
BASE = "/kaggle/input" if ON_KAGGLE else "data/ext"
DATASETS = {}

for name, ext, shape, dtype in (("RGB", "jpg", (64, 64, 3), "uint8"),
                                ("MS", "tif", (64, 64, 13), "uint16")):
    hits = sorted(glob.glob(os.path.join(BASE, "**", "*." + ext), recursive=True))
    if not hits:
        print(f"  {name:3s}: not attached")
        continue
    root = os.path.dirname(os.path.dirname(hits[0]))
    files = [h for h in hits if h.startswith(root + os.sep)]
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    if ext == "jpg":
        arr = np.asarray(Image.open(files[0]))
    else:
        try:
            import tifffile
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tifffile"],
                           capture_output=True)
            import tifffile
        arr = tifffile.imread(files[0])

    # Average over ALL files. Sampling the first N would only cover the first
    # class alphabetically and bias the result.
    total_bytes = sum(os.path.getsize(p) for p in files)
    info = {
        "root": root, "files": len(files), "classes": len(classes),
        "shape": list(arr.shape), "dtype": str(arr.dtype),
        "avg_file_kb": round(total_bytes / len(files) / 1024, 2),
        "total_mb": round(total_bytes / 1024 ** 2, 1),
    }
    DATASETS[name] = info

    print(f"  {name}")
    print(f"    root        {root}")
    print(f"    files       {len(files)}")
    print(f"    classes     {len(classes)}  {classes}")
    print(f"    sample      shape {arr.shape}  dtype {arr.dtype}")
    print(f"    file size   {info['avg_file_kb']} KB average, {info['total_mb']} MB total")
    per = {c: len(glob.glob(os.path.join(root, c, "*." + ext))) for c in classes}
    print(f"    per class   min {min(per.values())}  max {max(per.values())}")

    ok = (len(files) == EXPECTED_IMAGES and len(classes) == EXPECTED_CLASSES
          and tuple(arr.shape) == shape and str(arr.dtype) == dtype)
    print(f"    check       {'PASS' if ok else 'MISMATCH - expected ' + str(EXPECTED_IMAGES) + ' images, ' + str(shape) + ' ' + dtype}")

SPECS["datasets"] = DATASETS
with open(os.path.join(OUT, "specs.json"), "w") as f:
    json.dump(SPECS, f, indent=2)
print("")
print("wrote " + os.path.join(OUT, "specs.json"))
if not DATASETS:
    print("")
    print("NO DATASET FOUND. Use Add Data to attach eurosat-rgb (or eurosat-ms).")
