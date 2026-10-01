# Parallelism in a Deep Learning Data Pipeline

**Sequential vs Multithreading vs Multiprocessing** — EuroSAT land cover classification, PyTorch, Tesla T4.

Operating Systems assignment. The aim is not model accuracy, it is measuring how concurrency and parallelism affect data loading and training time, and explaining *why* each configuration is faster or slower.

[Notebook](notebooks/parallelism_benchmark.ipynb) · [RGB results](results/rgb_full/) · [Benchmark source](src/layerB_v2.py)

---

## Headline result (EuroSAT RGB, fp32)

**No loader configuration beats sequential.** All 12 configurations fall within **0.68%** of each other, with 3 repeats each.

The reason: the GPU was **99.3–99.8% busy in every configuration, including sequential**, leaving 0.2% of headroom for any loader to claim. Per batch of 64 images the GPU needs 207 ms while decoding on one core needs only 139 ms, and asynchronous CUDA already overlaps the two. Sequential is already GPU-bound.

Adding workers is not free. `thread x1` used **67% more CPU** for +0.0% throughput, and `process x8` added **159 MB of RAM** for −0.5%. Parallel efficiency falls as exactly 1/w.

Four configurations were **statistically slower** than sequential: `thread x4` (p=0.027), `thread x8` (p<0.001), `process x6` (p=0.034), `process x8` (p<0.001). Small, 0.3–0.7%, but real at this precision.

> So the assumption *"more threads or processes means faster"* is **rejected** for this workload. Extra workers only help when data loading is the bottleneck, and here it is not.

---

## Dataset

[EuroSAT](https://zenodo.org/records/7711810) (Helber et al.) — Sentinel-2 land use and land cover, **27,000 labelled patches**, 64x64 px, **10 classes**, 10 m/px. Both archives verified by MD5.

| | RGB | MS |
|---|---|---|
| Format | 3-band JPEG | 13-band GeoTIFF |
| Average file | **3.32 KB** | **104.73 KB** |
| Total | 87.6 MB | 2,761 MB |
| Tensor | `(64, 64, 3)` uint8 | `(64, 64, 13)` uint16 |

Classes are mildly imbalanced, 2,000–3,000 images each. The 31x difference in bytes per sample is the point of running both: it moves the bottleneck between the CPU and the GPU.

---

## Hardware

Two tiers were used deliberately, because the bottleneck differs between them. Every value below was captured programmatically rather than typed by hand:

| Tier | Authoritative source |
|---|---|
| Sandbox | [`results/specs_sandbox.json`](results/specs_sandbox.json), plus [`results/layerA/environment.json`](results/layerA/environment.json) written by the benchmark itself |
| Kaggle | [`results/rgb_full/environment.json`](results/rgb_full/environment.json) written by the benchmark during the recorded run; [`results/specs_kaggle.json`](results/specs_kaggle.json) is an earlier capture kept for the GPU driver and capability fields |

The two `specs_*.json` files use slightly different key names, because the Kaggle one predates the current verification script. The per-run `environment.json` files are the ones to trust, since each was written by the harness during the run it describes.

| | **Tier 1 — Sandbox** | **Tier 2 — Kaggle** |
|---|---|---|
| Role | Layer A, preprocessing only | Layer B, end-to-end training |
| CPU | Intel Xeon Platinum 8488C | Intel Xeon @ 2.00 GHz |
| **CPU cores / threads** | **4 physical / 8 logical** | **2 physical / 4 logical** |
| RAM | 30.8 GB | 31.4 GB |
| **GPU** | **none** | **Tesla T4 ×2**, 14.56 GB VRAM each, capability 7.5 |
| Disk | 125 GB NVMe, ~1.3 GB/s cold read | Kaggle managed storage |
| Operating system | Linux 6.1 (Amazon Linux 2023) | Linux 6.12 |
| `mp` start method | `fork` | `fork` |
| Python version | 3.11.15 | 3.12.13 |
| **Deep learning framework** | PyTorch 2.14.0+cpu | PyTorch 2.10.0+cu128, CUDA 12.8, cuDNN 9.1 |
| NumPy | 2.4.6 | 2.0.2 |
| **Dataset size** | 27,000 images (both variants) | 27,000 images |
| **Batch size** | n/a — no model | **64** |
| **Number of epochs** | n/a — no model | **2** |
| Thread pinning | `OMP_NUM_THREADS=1` | `torch_num_threads=2`, `OMP_NUM_THREADS=2` |
| Worker counts tested | 1, 2, 4, 8, 16 | 1, 2, 3, 4, 6, 8 |
| Repeats | 3 | 3 |

### Why two tiers, and why the specs matter

**Core count sets the ceiling.** The sandbox has twice the physical cores, which is why its scaling curve has five usable points (1→16) while Kaggle's saturates after two. Any worker-scaling result is meaningless without the core count beside it.

**Physical and logical counts are both reported on purpose.** `lscpu` fails inside containers, so both tiers parse `/proc/cpuinfo` instead. Reporting only the logical count would make a plateau near the physical count look like an unexplained failure.

**Single-core speed differs by 3.45×.** The sandbox decodes RGB at 2,585 img/s per core against Kaggle's ~590. Same code, same data. This is the single clearest demonstration that parallel-computing results cannot be separated from the hardware they were measured on.

**`OMP_NUM_THREADS` is not a detail.** NumPy's `mean`/`std` are thread-parallel, so leaving it unpinned makes a "1 worker" run secretly multi-core. It is set to 1 in Layer A to isolate the thread/process comparison, and to 2 in Layer B to match Kaggle's physical cores. Decode throughput measured 459 img/s at `OMP=2` versus 590 at `OMP=4` on the same machine — the thread setting must be reported with every figure.

---

## Experimental design

| Scenario | Implementation |
|---|---|
| Sequential | `num_workers=0` |
| Sequential, pageable memory | `pin_memory=False`, to isolate the effect of pinned memory |
| Multithreading | `ThreadPoolExecutor` building whole batches inside one process, so threads share the GIL. 1, 2, 4, 8 threads |
| Multiprocessing | `DataLoader` worker processes. 1, 2, 3, 4, 6, 8 workers |

Measured per run: total time, per-epoch time, data-loading block time, throughput, CPU, busy cores, RAM, context switches, GPU utilisation, VRAM, SM clock, temperature, throttle reasons, worker startup cost, training accuracy.

Derived: `speedup = T_sequential / T_parallel`, `efficiency = speedup / workers`.

### Controls, and why each is needed

- **Identical batches for every method** — shared seeded sampler, asserted at startup. Without it the comparison is unfair.
- **Page-cache warm-up** — every file is read once before timing. An earlier version produced a **false 1.5x speedup** purely because the sequential baseline ran first on a cold cache.
- **GPU heat soak** — runs until temperature levels off. On a fresh T4 the clock fell **1044 to 870 MHz** while warming, a bigger effect than the differences between configurations.
- **Randomised order within each repeat block** — so thermal drift cannot align with one method.
- **3 repeats, mean ± SD** — within-config variation is 0.11–0.67%, which is what makes 0.5% differences detectable.
- **Pinned thread counts** — left at default, PyTorch claims every core and worker effects become uninterpretable.
- **One pinned GPU** — Kaggle provides two T4s.

---

## Results: EuroSAT RGB, 27,000 images, 2 epochs, n=3

| Method | Workers | Total (s) | ± SD | Steady epoch (s) | Img/s | Speedup | Efficiency | Busy cores | RAM +MB | GPU % | Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sequential | 0 | 174.39 | 0.20 | 87.02 | 309.0 | 1.00 | – | 0.88 | −7 | 99.8 | 0.866 |
| sequential, pageable | 0 | 174.49 | 0.91 | 87.10 | 308.8 | 1.00 | – | 0.81 | 3 | 99.4 | 0.865 |
| thread | 1 | **174.36** | 0.43 | 87.04 | 309.1 | 1.00 | 1.00 | 1.47 | −12 | 99.8 | 0.866 |
| thread | 2 | 174.88 | 0.42 | 87.64 | 308.1 | 1.00 | 0.50 | 1.48 | 0 | 99.7 | 0.866 |
| thread | 4 | 174.86 | 0.31 | 87.36 | 308.2 | 1.00 | 0.25 | 1.49 | 0 | 99.7 | 0.866 |
| thread | 8 | 175.54 | 0.48 | 87.81 | 307.0 | 0.99 | 0.12 | 1.48 | −2 | 99.4 | 0.866 |
| process | 1 | 175.34 | 1.18 | 87.74 | 307.3 | 0.99 | 0.99 | 1.45 | −1 | 99.3 | 0.864 |
| process | 2 | 174.57 | 0.31 | 87.08 | 308.7 | 1.00 | 0.50 | 1.51 | 40 | 99.8 | 0.866 |
| process | 3 | 174.74 | 0.43 | 87.31 | 308.4 | 1.00 | 0.33 | 1.53 | 43 | 99.8 | 0.868 |
| process | 4 | 174.67 | 0.31 | 86.99 | 308.5 | 1.00 | 0.25 | 1.53 | 50 | 99.8 | 0.866 |
| process | 6 | 174.96 | 0.42 | 87.15 | 308.0 | 1.00 | 0.17 | 1.53 | 111 | 99.7 | 0.868 |
| process | 8 | 175.28 | 0.24 | 87.35 | 307.4 | 0.99 | 0.12 | 1.53 | 159 | 99.7 | 0.865 |

Accuracy is a sanity check only: identical across configurations, as it should be when the batches are identical.

Raw data: [`results/rgb_full/`](results/rgb_full/)

---

## Analysis

**Is multiprocessing always faster than sequential?** No. On this workload it is never faster, and at 6 or 8 workers it is measurably slower.

**Do more workers mean more speed?** No. Throughput peaks at 1 worker and declines. Efficiency falls as 1/w.

**Why can too many workers slow things down?** Each process worker carries its own interpreter and buffers, costing up to 159 MB, and all of them compete for 2 physical cores. When the queue is already full, that cost buys nothing.

**Does multithreading help?** No, and the context-switch counters show why. Threads share one GIL, so every handoff costs a context switch:

| Configuration | Context switches/s | vs sequential |
|---|---:|---:|
| sequential | 2,531 | – |
| thread x1 | 3,126 | +23.5% |
| thread x2 | 3,319 | +31.1% |
| thread x4 | 3,633 | +43.5% |
| **thread x8** | **4,213** | **+66.4%** |
| process x1 … x8 | 2,864–2,912 | +13–15%, flat |

Context switching **rises with thread count but is flat across process count**, because each worker process has its own interpreter and its own GIL, so they never contend for one. `thread x8` also had the lowest GPU utilisation of any configuration. On a shorter run where the GPU had slack, the same configuration was **13.8% slower** than sequential after normalising for clock speed.

**What does worker startup cost?** On Linux with `fork`, very little: 0.11 s sequential rising to 0.65 s for 8 worker processes, and the first epoch is only 0.1–0.8% slower than the steady-state epoch. This is **OS-dependent**: on Windows, which must `spawn`, a comparable published benchmark paid 5.5 s for 2 workers and 11.1 s for 4, enough to change which configuration wins a short training run. On `fork` that trade-off does not arise.

**What is the bottleneck: CPU, RAM, storage or GPU?** The **GPU**, decisively: 99.8% utilisation, 0.2% idle. RAM and storage are not constraints, and the CPU is only 0.88 cores busy in sequential out of 4 logical.

**Did the GPU wait for data?** Not here, 0.2% idle. Earlier runs appeared to show 37% idle, but that was traced to a cold OS page cache in the first-run baseline, not to the data pipeline.

**Best configuration?** Sequential. It matches the fastest configuration within noise while using the least CPU and RAM.

### A useful model

```
throughput = min(CPU supply rate, GPU rate)
```

Asynchronous CUDA already overlaps decoding with compute even with `num_workers=0`: the main thread queues kernels, returns immediately, and decodes the next batch while the GPU works. So this is a pipeline bottleneck problem, not Amdahl's serial sum. The naive serial model predicts a 1.68x ceiling here; the measured answer is 1.00x, and the data rejects the serial model.

The same model predicted RGB fp32 at exactly 1.00x before the run, which 36 runs confirmed.

---

## How to run

1. Open [`notebooks/parallelism_benchmark.ipynb`](notebooks/parallelism_benchmark.ipynb) on Kaggle (File -> Import Notebook).
2. Settings -> Accelerator -> **GPU**.
3. Add Data -> attach the EuroSAT dataset. Import it server-side from the Zenodo URL rather than uploading:
   - `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip`
   - `https://zenodo.org/records/7711810/files/EuroSAT_MS.zip`
4. Run **step 1** to confirm the environment and dataset.
5. **Step 2**, keep `MODE = "SMOKE"`, run once interactively (about 4 min) to confirm it completes.
6. Set `MODE = "FULL"`, pick `DATASET`, then **Save Version -> Save & Run All (Commit)**. About 2 hours, server-side. Stop the interactive session so it does not consume GPU quota in parallel.

To benchmark the other dataset, change `DATASET` and save another version. Each Kaggle version keeps its own outputs, so one benchmark cell is enough.

---

## Repository layout

```
.
├── README.md
├── notebooks/
│   └── parallelism_benchmark.ipynb   # the notebook to run on Kaggle
├── src/
│   ├── cell_verify.py                # step 1, identical to notebook cell 1
│   └── layerB_v2.py                  # step 2, identical to notebook cell 2
└── results/                          # see results/README.md
    ├── specs.json
    ├── rgb_full/                     # PRIMARY: RGB, 27k, 2 epochs, n=3
    ├── rgb_smoke/                    # the 1,280-image check run
    └── superseded/                   # earlier harness versions, do not cite
```

`src/` holds the same code as plain scripts so it can be reviewed and diffed. The notebook cells are byte-identical to these files.

All 36 runs completed with `status=ok` and no duplicates. `summary.csv` was independently recomputed from `runs.csv` with **0 mismatches**, and `config.json` confirms the controls were applied rather than merely intended. Details in [`results/README.md`](results/README.md).

---

## Limitations

- Results apply to **this** hardware, dataset and worker range. A machine with more physical cores, or a slower GPU, would move the bottleneck.
- Only **2 epochs**, so these are not converged-accuracy numbers. The aim is pipeline performance.
- Kaggle sessions vary. Single-core decode measured 459–576 img/s across sessions with identical settings, so **only compare numbers from the same session**.
- CUDA is not run in deterministic mode, so the same seed does not give bit-identical results.
- The T4 is power-limited and its clock moves between runs. On short runs this dominates; a 174 s run reaches steady state internally (clock spread 1.3%), which is why the full-length run is the trustworthy one.
