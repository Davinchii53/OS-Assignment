# Results

Raw measurement data. Every number in the top-level README and the report comes from these files.

```
results/
├── specs.json                 hardware and software, from notebook step 1
├── rgb_full/                  PRIMARY RESULT: EuroSAT RGB, 27,000 images, 2 epochs, n=3
├── rgb_smoke/                 the 1,280-image check run that preceded it
└── superseded/                earlier harness versions, kept for the record only
```

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
