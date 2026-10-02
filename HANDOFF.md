# Session Handoff — Operating Systems Assignment

Full context for continuing this work in a new chat session. Read this first.

---

## 1. The assignment

**Course:** Operating Systems. **Topic:** measure how **multithreading** and **multiprocessing** affect the performance of a Deep Learning pipeline.

The grading emphasis is **not** model accuracy. It is understanding *why* a configuration is faster or slower.

### Required deliverables

| Requirement | Status |
|---|---|
| Choose a dataset and DL problem | ✅ EuroSAT, image classification |
| 3 scenarios: sequential, multithreading, multiprocessing (try 1/2/4/8 workers) | ✅ done, far beyond minimum |
| Measure total time, time/epoch, data-loading time, throughput, CPU, RAM, GPU | ✅ all measured |
| Speedup = T_sequential / T_parallel | ✅ |
| Results table | ✅ auto-generated `summary.md` per run |
| **At least 1 comparison graph** | ❌ **NOT STARTED** |
| Analysis answering 7 questions | ✅ drafted in `ANALYSIS.md`, needs 1 revision |
| Conclusion answering 2 main questions | ✅ drafted |
| Hardware specs (CPU, cores/threads, RAM, GPU, OS, Python, framework, dataset size, batch, epochs) | ✅ both tiers, side by side in README |
| Re-runnable code | ✅ 2 notebooks + 4 scripts |
| **Report, 5–8 pages** | ❌ **NOT STARTED** |
| **Presentation, every member understanding the code** | ❌ **NOT STARTED** |

### The two questions the conclusion must answer

1. Can multithreading or multiprocessing improve the DL pipeline on this dataset? If so, which configuration is best and why?
2. Do the results support or reject: *"more threads or processes means faster Deep Learning"*?

---

## 2. Team and logistics

- **3 people.** The user is the technical lead. Two teammates still need a code walkthrough before the presentation.
- **Report language: UNDECIDED.** `ANALYSIS.md` is English (primary), `ANALISIS.md` is the Indonesian translation. **Figures are blocked on this decision.**
- A progress report was already submitted to the course forum in Indonesian (`LAPORAN_PROGRES.md`).
- Roughly **day 6 of a 7-day window**. The optional experiments are finished; only required deliverables remain.

---

## 3. Dataset

**EuroSAT** (Helber et al., [Zenodo 7711810](https://zenodo.org/records/7711810)) — Sentinel-2 land use / land cover.

27,000 labelled patches, 64x64 px, 10 classes, 10 m/px. Classes mildly imbalanced (2,000–3,000 each). Both archives MD5-verified.

| | RGB | MS |
|---|---|---|
| Format | 3-band JPEG | 13-band GeoTIFF |
| Average file | **3.32 KB** | **104.73 KB** |
| Total | 87.6 MB | 2,761 MB |
| Tensor | `(64,64,3)` uint8 | `(64,64,13)` uint16 |

The 31.5x difference in bytes per file is what moves the bottleneck between regimes. (An earlier note said 2.6 KB and 40x — wrong, it came from a biased sample of one class.)

---

## 4. Hardware — three machines were involved

| | Sandbox | Kaggle (GPU runs) | Kaggle (Layer A run) |
|---|---|---|---|
| CPU | Xeon Platinum 8488C | Xeon @ 2.00GHz | **AMD EPYC 7B12** |
| Cores | **4 phys / 8 log** | **2 phys / 4 log** | 2 phys / 4 log |
| RAM | 30.8 GB | 31.4 GB | 31.4 GB |
| GPU | **none** | Tesla T4 x2, 14.56 GB | none (accelerator off) |
| Storage read, warm/cold | 1,407 / 1,315 MB/s (MS) | — | **468 / 117 MB/s** (MS) |
| Python / numpy | 3.11.15 / 2.4.6 | 3.12.13 / 2.0.2 | 3.12.13 / 2.0.2 |
| Framework | PyTorch 2.14.0+cpu | PyTorch 2.10.0+cu128 | — |

**Kaggle hands out different CPUs between sessions.** Never compare numbers across sessions — single-core decode measured 459–590 img/s on identical settings in different Kaggle sessions.

---

## 5. Experiment structure

**Layer A — preprocessing only, no model, no GPU.** Isolates the data pipeline.
- Experiment 1: GIL control — pure-Python loop (holds GIL) vs SHA-256 (releases it), threads vs processes
- Experiment 2: decode matrix — sequential / threads / processes x {1,2,4,8,16}
- Experiment 3: payload ablation — workers return the full array vs a single float. Same CPU work, so the difference is inter-process transfer cost. Threads are the control (shared memory, so payload should not matter)
- Experiment 4: storage — warm vs cold page cache via `posix_fadvise(DONTNEED)`

Run on **both** the sandbox (4 phys cores) and Kaggle (2 phys cores).

**Layer B — end-to-end training.** ResNet-18 with conv1 replaced by 3x3 stride 1 and maxpool removed for 64x64 input. Batch 64, 2 epochs, SGD lr 0.01 momentum 0.9.
- Configs: sequential, sequential_nopin, threads 1/2/4/8, processes 1/2/3/4/6/8
- Run on Kaggle with a T4, for **both** RGB and MS

All runs: **n=3**, randomised config order, 3 s cooldown, seed 42.

---

## 6. Results — the central finding

**Parallelism only helps at the bottleneck.** Identical decode code gives completely different answers depending on which stage is slowest.

| Regime | decode vs GPU per batch | Best speedup | What workers do | Data |
|---|---|---|---|---|
| **Layer A, sandbox** | no GPU, CPU-bound | **5.03x** (process x8) | large throughput gain | `results/layerA/` |
| **Layer A, Kaggle** | no GPU, partly I/O-bound | **5.96x** (process x16) | large gain, more from hiding I/O | `results/layerA_kaggle/` |
| **Layer B, MS** | 190.6 vs 198.5 ms (balanced) | 1.06x | small gain, **13x more stable** | `results/ms_full/` |
| **Layer B, RGB** | 108.5 vs 202.2 ms (GPU-bound) | **1.00x** | nothing at all | `results/rgb_full/` |

### Layer B RGB — the null result, and it is rigorous

All 12 configurations within **0.68%** of each other, n=3. GPU **99.3–99.8% busy in every configuration including sequential**, so only 0.2% of headroom existed. Measurement precision was high enough that 0.5% slowdowns are statistically detectable: `thread x4` (p=0.027), `thread x8` (p<0.001), `process x6` (p=0.034), `process x8` (p<0.001) were all **significantly slower** than sequential.

Cost of parallel loading that achieves nothing: `thread x1` used **67% more CPU** (1.47 vs 0.88 cores) for +0.0%; `process x8` added **159 MB** for −0.5%. Efficiency falls as exactly 1/w.

### Layer B MS — the balanced regime, where stability is the real gain

Decode 190.6 ms/batch against GPU 198.5 ms — within 4%. Sequential became **erratic**: three runs of 186.9, 169.2, 167.2 s (CV 6.2%, against 0.11% on RGB), with 6.3% GPU idle.

Every worker configuration collapsed that: **SD 10.86 s to 0.86 s, a 13x variance reduction** (`thread x8` reached SD 0.02 s), GPU idle to 0.3%.

**Important caveat:** the 1.06x throughput gain is **NOT statistically significant** at n=3 (p = 0.08–0.14) because sequential is so variable. Do not claim a significant speedup. The defensible claims are the variance reduction, plus two supporting observations: all 10 worker configs were faster, and 9 of 10 had a mean below sequential's single fastest run.

### Layer A — where the scaling curves live

Sandbox, RGB, scalar payload: **1.00x → 1.91x → 3.32x → 5.03x → 4.97x** for 1/2/4/8/16 workers. Peaks at the logical core count, declines at 16.

**Threads cap at 2.4 busy cores of 8** regardless of count (2, 4, 8, 16 identical). Processes reach **7.9**. Threads physically cannot use the machine.

Context switches invert: `thread x4` at **144,103/s** against `process x8` at 66,593/s, while being 5.6x slower.

### GIL — proven, not assumed

The control experiment is the strongest single piece of evidence, and it is **consistent on every machine**:

| Workers | pure Python threads | SHA-256 threads | pure Python processes |
|---|---|---|---|
| 4, sandbox | **1.00x** | 3.04x | 3.73x |
| 4, Kaggle EPYC | **0.87x** | **3.51x** | 1.81x |

Threads deliver nothing on work that holds the GIL and scale nearly linearly on work that releases it. Same pool, same machine, one variable.

### IPC payload — the dominant multiprocessing cost

Identical CPU work; only the returned payload differs:

| Dataset | Workers | array | scalar | gap |
|---|---|---|---|---|
| MS | 1 | **0.55x** | 0.96x | 1.75x |
| MS | 4 | 1.84x | 3.37x | **1.83x** |
| RGB | 8 | 4.23x | 5.03x | 1.19x |

Transfer cost consumes **up to 45%** of achievable speedup, and `process x1` on MS is **0.55x** — nearly half the speed of sequential, because one worker adds no parallelism but pays full serialisation cost. **Threads are unaffected** by payload (0.91x vs 0.90x), which proves the mechanism is inter-process transfer.

RAM confirms it: MS `process x16` with array payloads peaked at **+1,040 MB** on Kaggle, against +124 MB with scalars.

### Storage and I/O latency hiding

| | warm | cold | penalty |
|---|---|---|---|
| Sandbox MS | 1,407 MB/s | 1,315 MB/s | 1.07x |
| **Kaggle MS** | 468 MB/s | **117 MB/s** | **3.99x** |

`/kaggle/input` is a network mount. On Kaggle the cold-cache penalty **collapses with worker count**: sequential 2.34x down to `process x16` 1.16x, with threads effectively **immune** at 0.91x. Consequently **cold speedup exceeds warm speedup**: MS `process x16` reached **11.69x cold** against 5.77x warm.

Parallelism is worth *more* when there is latency to hide.

---

## 7. Corrections made during the project — keep these, they are good report material

1. **A false 1.50x speedup from the OS page cache.** The first Layer B run showed 1 worker giving 1.50x and the GPU 37.3% idle. Cause: the sequential baseline ran first, before files were cached, so it measured disk reads. Fixed by reading every file once before timing. The effect vanished.
2. **A wrong Amdahl formula.** Used a ratio of *rates* where a ratio of *times* was needed. Reported 51.4% / 2.06x; correct values 34.0% / 1.51x.
3. **Amdahl's serial model is wrong for this pipeline.** It predicted a 1.54x ceiling; measured 1.00x. The correct model is a bottleneck one: `throughput = min(CPU rate, GPU rate)`, because asynchronous CUDA already overlaps decode with compute even at `num_workers=0`. This model predicted RGB at exactly 1.00x before the run, confirmed by 36 runs.
4. **GPU thermal throttling.** A cold T4's clock fell 1044 → 870 MHz while heating 51 → 72 °C, larger than the differences between configurations. Fixed with a heat soak to stable temperature plus randomised order. Clock spread then fell from 24% to 1.3%.
5. **Pinned memory: rejected twice, then confirmed.** On RGB, pageable memory cost nothing (1.00x). On MS it cost 8% and left the GPU 15.1% idle, because 13-channel tensors make each host-to-device copy 4.3x larger and an unpinned copy is synchronous. The hypothesis was not wrong, it was untested in the regime where the mechanism operates.
6. **An n=1 result presented as solid.** The Kaggle Layer A smoke run showed `thread x8` beating `process x4` on a cold cache, described as a textbook reversal. At n=3 on 27,000 images, processes won in both states. It was an artifact.
7. **Biased file-size sample.** Averaged the first 2,000 sorted paths, which is entirely one class. RGB is 3.32 KB, not 2.6 KB; the MS:RGB ratio is 31.5x, not 40x.
8. **A wrong prediction about storage.** Predicted storage would saturate on MS at 8+ workers. It reached only 46% of the sandbox ceiling. The cold estimate came from a 400-file sample whose warm comparison was unrealistically fast.

---

## 8. OPEN ISSUE — needs fixing in `ANALYSIS.md`

`ANALYSIS.md` claims **"threads never beat sequential on CPU-bound decoding"**, based on sandbox data (max 1.06x).

**On Kaggle threads reached 2.00–2.01x.** Implied serial fraction from Amdahl moves from ~1.03 (sandbox, GIL dominates) to ~0.00 (Kaggle, decode runs outside it) — for identical code.

The GIL *control* experiment is consistent on both machines, so only that result is universal: **threads do not help work that holds the GIL**. Whether real decode holds it is platform- and library-dependent. Candidate causes, none verified:
- numpy 2.4.6 vs 2.0.2, different GIL-release behaviour in `mean`/`std`
- Kaggle sequential is partly I/O-bound so threads have waiting to overlap; the sandbox was compute-saturated (most likely, and consistent with the storage data)
- CPU microarchitecture changing the ratio of C-library work to interpreter work

**This section must be qualified before the report is written.**

---

## 9. Repository

**GitHub:** `Davinchii53/OS-Assignment`
**Open PR #1:** `cleanup/restructure-notebook-and-results` → `main`, mergeable, **not yet merged**

```
README.md                                    headline result, hardware both tiers, analysis, how to run
notebooks/
├── parallelism_benchmark.ipynb              Layer B, needs GPU
└── layerA_preprocessing_kaggle.ipynb        Layer A, accelerator = None
src/
├── cell_verify.py                           environment + dataset verification
├── layerB_v2.py                             Layer B harness (v2.1)
├── layerA_preprocessing.py                  Layer A, sandbox
├── layerA_storage.py                        Layer A experiment 4, sandbox
└── layerA_kaggle.py                         Layer A, Kaggle
results/
├── specs_sandbox.json  specs_kaggle.json
├── layerA/                                  sandbox: 132 decode + 84 storage runs
├── layerA_kaggle/                           Kaggle: 132 decode + 84 storage runs
├── rgb_full/                                Layer B RGB: 36 runs + 2 MB telemetry
├── ms_full/                                 Layer B MS: 36 runs + 2 MB telemetry
├── rgb_smoke/                               check run
├── superseded/                              v1 outputs, marked do-not-cite
└── README.md                                what each file holds, verification notes
```

Notebook code cells are asserted **byte-identical** to the `src/` scripts at build time. All committed Python contains **zero backslashes**, because a copy-paste once converted `\n` inside an f-string into a real newline and broke the cell on Kaggle.

### Sandbox-only files (NOT in the repo)

`PLAN.md`, `PROGRESS.md` (the full measurement log, every decision and bug), `ANALYSIS.md`, `ANALISIS.md`, `LAPORAN_PROGRES.md`, `KAGGLE_SETUP.md`, `HANDOFF.md`, plus ad-hoc analysis scripts (`analyse_rgb_full.py`, `amdahl_check.py`, `cost_probe.py`, `gate_test.py`, `gate_test2.py`, `analyse_kaggle_rgb.py`).

**`ANALYSIS.md` and `PROGRESS.md` should be pushed** — they are the most valuable documents and exist only in the sandbox.

---

## 10. Environment notes

- Sandbox venv at `/projects/sandbox/.venv` — Python 3.11.15, torch 2.14.0+cpu, numpy, pandas, matplotlib, psutil, tifffile, Pillow
- Datasets extracted at `/projects/sandbox/data/ext/EuroSAT_RGB` and `.../EuroSAT_MS`
- Repo clone at `/projects/sandbox/review-repo`
- **`/tmp` does not persist between bash calls.** Clone and inspect in the same call.
- **Background processes are reaped** when a bash call returns. Long jobs must run in the foreground of one call.
- Zenodo honours HTTP range requests — 8 parallel connections downloaded 2 GB in 8 minutes instead of 47.
- The sandbox has full capabilities, so `posix_fadvise` and `drop_caches` both work.

---

## 11. Kaggle workflow that works

1. Settings → Accelerator → **GPU** for Layer B, **None** for Layer A (no GPU quota consumed)
2. Add Data → attach the EuroSAT datasets. Import server-side from the Zenodo URL rather than uploading
3. `MODE = "SMOKE"` interactively first, ~4–7 min, confirm clean
4. `MODE = "FULL"` → **Save Version → Save & Run All (Commit)**, then **stop the interactive session**
5. Download outputs from the version's Output tab

GPU quota used so far: roughly **6 of ~30 hours per week**.

Harness safety features, all tested: results written after every run, resume support, a version guard that deletes results from a different code version rather than resuming from them, a `MAX_HOURS` guard that stops cleanly, and `DataLoader(timeout=300)`.

---

## 12. What to do next, in order

1. **Decide the report language.** Figures are blocked on this. English or Indonesian.
2. **Fix the threads claim in `ANALYSIS.md`** (see section 8).
3. **Build the figures.** None exist; at least one is required. Suggested set:
   - Speedup vs workers — threads flat vs processes rising to 5.03x then declining
   - Throughput vs workers, both datasets
   - Busy cores vs workers — threads capped at 2.4, processes at 7.9: the GIL, visually
   - GIL control — 1.00x pure Python vs 3.04x SHA-256
   - Payload ablation — array vs scalar
   - Layer A vs Layer B — 5.03x against 1.00x, the central thesis
   - Optional: warm vs cold cache, where cold speedup exceeds warm
4. **Write the report, 5–8 pages.** `ANALYSIS.md` already answers all 9 questions; the structure the assignment asks for is problem/dataset, model, hardware specs, sequential method, threading implementation, multiprocessing implementation, experimental setup, results, graphs, analysis, conclusion.
5. **Push `ANALYSIS.md` and `PROGRESS.md`** to the repo.
6. **Merge PR #1.**
7. **Slides, and a code walkthrough for the two teammates** — the assignment requires every member to understand the implementation.

### Optional, only if time allows

- Reduced sandbox Layer B (CPU-only training) to verify the predicted 1.01x Amdahl ceiling. ~40 min for 9 runs on a 4,000-image subset. The full matrix would be 17.7 hours and is not worth it.
- Mixed precision (AMP) on the T4 would roughly halve GPU step time and make the pipeline data-bound, producing a genuine scaling curve in Layer B. Needs a code change plus a new smoke test, ~2 GPU hours.
- A Colab tier (2 cores + T4) was planned and dropped. Its value has largely evaporated now that both datasets are known to be GPU-bound on a T4.

---

## 13. Working-style notes for the assistant

- **Work in small single steps.** Report after each. Long multi-tool turns with no output look like a stall. This was explicit user feedback.
- Save substantive answers as downloadable Markdown files.
- The user cannot see the filesystem or terminal — surface file contents and results in chat.
- Paste-safety matters: no backslashes in any code intended for copy-paste into Kaggle.
- The user values honest correction of earlier mistakes. Eight were caught and documented, and they strengthen the report rather than weakening it.
