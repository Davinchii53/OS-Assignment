# Results

Raw measurement data. Every number in the top-level README and the report comes from these files.

```
results/
├── specs_sandbox.json         sandbox hardware and software
├── specs_kaggle.json          Kaggle hardware and software
├── layerA/                    preprocessing benchmark, no GPU, 4 experiments
├── rgb_full/                  Layer B: EuroSAT RGB, 27,000 images, 2 epochs, n=3
├── ms_full/                   Layer B: EuroSAT MS, 27,000 images, 2 epochs, n=3
├── rgb_smoke/                 the 1,280-image check run that preceded rgb_full
└── superseded/                earlier harness versions, kept for the record only
```

## The three regimes

The same decode code gives different answers depending on which stage is slowest. This is the central result, and each row has its own data directory.

| Regime | decode vs GPU per batch | Best speedup | What workers do | Data |
|---|---|---|---|---|
| No GPU | CPU only | **5.03×** | large throughput gain | `layerA/` |
| Balanced | 1.04× (190.6 vs 198.5 ms) | 1.06× | small gain, **13× more stable** | `ms_full/` |
| GPU-bound | 1.86× (108.5 vs 202.2 ms) | 1.00× | nothing | `rgb_full/` |

## `layerA/` — preprocessing only, no GPU

| File | Contents |
|---|---|
| `gil_control.csv` | pure-Python vs SHA-256, threads vs processes. Isolates the GIL |
| `runs_rgb.csv`, `runs_ms.csv` | 66 runs each: sequential / threads / processes x {1,2,4,8,16} x {array, scalar payload} x 3 repeats |
| `storage.csv` | 84 runs, warm vs cold page cache, via `posix_fadvise(DONTNEED)` |
| `summary.csv` | aggregated per configuration, 44 rows |
| `environment.json`, `storage_bandwidth.json` | hardware and measured read bandwidth |

## `rgb_full/` and `ms_full/` — Layer B, end-to-end training

Identical file layout for both:

| File | Contents |
|---|---|
| `summary.md` | the results table, mean ± SD, ready for the report |
| `summary.csv` | the same aggregated per configuration, 12 rows |
| `runs.csv` | one row per run, 36 rows, every raw metric |
| `epochs.csv` | per-epoch times, separating cold start from steady state |
| `timeseries.csv` | CPU, RAM, GPU sampled 4x per second, ~23,000 rows |
| `environment.json` | hardware and library versions, written during that run |
| `config.json` | every setting, including the randomised order and heat-soak record |
| `run_log.txt` | the full Kaggle log (rgb_full only) |

**Read `ms_full/` with care.** Its sequential baseline is unusually variable: the three runs took 186.9, 169.2 and 167.2 s (CV 6.2%, against 0.11% for RGB). At the balance point sequential sometimes keeps up with the GPU and sometimes does not. The consequence is that the 1.06× worker speedup is **not statistically significant at n=3** (p = 0.08–0.14), even though all 10 worker configurations are faster. The defensible claim is the variance reduction, from 10.86 s to 0.86 s, not the throughput gain.

## `rgb_full/` — the primary result

| File | Contents |
|---|---|
| `summary.md` | the results table, mean ± SD, as pasted into the report |
| `summary.csv` | the same aggregated per configuration, 12 rows |
| `runs.csv` | one row per run, 36 rows, every raw metric |
| `epochs.csv` | per-epoch times, separating cold start from steady state |
| `timeseries.csv` | CPU, RAM, GPU sampled 4x per second, 23,817 rows |
| `environment.json` | hardware and library versions |
| `config.json` | every setting, including the randomised run order and the heat-soak record |
| `run_log.txt` | the full Kaggle log, 36 runs, 0 errors |

### Verified

- 36 rows, all `status=ok`, no duplicates, 3 repeats across 12 configurations.
- `summary.csv` recomputed from `runs.csv`: **0 mismatches**, so the aggregation is faithful.
- `timeseries.csv` covers all 36 runs, 658–670 samples each, no empty telemetry fields.
- `config.json` confirms the controls were actually applied: `batches_identical_across_methods: true`, `randomised_order: true`, `seed: 42`, `torch_threads: 2`, heat soak 60.3 s.

### Reproducing the headline numbers

```python
import csv, numpy as np
from collections import defaultdict

runs = list(csv.DictReader(open("rgb_full/runs.csv")))
g = defaultdict(list)
for r in runs:
    g[(r["method"], int(r["workers"]))].append(float(r["total_s"]))

seq = np.mean(g[("sequential", 0)])
for k in sorted(g, key=lambda k: np.mean(g[k])):
    t = np.mean(g[k])
    print(f"{k[0]:<18} x{k[1]:<2} {t:7.2f}s  speedup {seq / t:.3f}")
```

## `superseded/`

Results from earlier harness versions, kept only to document what changed. **Do not cite these.**

| File | Why it is superseded |
|---|---|
| `v1_layerB_rgb_quick.csv` | v1 harness. Its apparent 1.72× speedup was a cold OS page-cache artifact: the sequential baseline ran first, before the dataset was in the page cache. v2 reads every file once before timing, which removes the effect. |
| `v1_smoke_test_rgb.csv` | Contained a bug in the Amdahl calculation, a ratio of rates where it needed a ratio of times, so it reported an inflated loading share and ceiling. |
