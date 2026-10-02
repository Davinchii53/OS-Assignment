# Analysis — Answers to the Assignment Questions

EuroSAT dataset, PyTorch, two hardware tiers. Every figure comes from `results/`.

> **Note:** the MS Layer B run is still in progress (n=3). MS Layer B figures below come from the smoke test (n=1, 1,280 images) and will be updated.

---

## The two tiers at a glance

| | Sandbox (no GPU) | Kaggle (Tesla T4) |
|---|---|---|
| Cores | 4 physical / 8 logical | 2 physical / 4 logical |
| What is measured | preprocessing only | preprocessing + training |
| Bottleneck | **CPU** | **GPU** (99.8% busy) |
| Best speedup | **5.03×** | **1.00×** |
| Best configuration | process ×8 | sequential |

The decode code is identical. The answers are opposite because the slowest stage is different.

---

## 1. Is multiprocessing always faster than sequential?

**No.** There are cases where multiprocessing is measurably **slower**, and we have both:

| Case | Result |
|---|---|
| Kaggle, `process ×6` | 0.99× — **statistically slower** (p = 0.034) |
| Kaggle, `process ×8` | 0.99× — **statistically slower** (p < 0.001) |
| Sandbox, MS, `process ×1`, array payload | **0.55×** — nearly half the speed of sequential |

The 0.55× case is the clearest: one worker adds no parallelism at all, yet still pays the full cost of serialising data between processes.

Multiprocessing only helps when **the stage being parallelised is actually the bottleneck**, and when its overhead (spawn, inter-process transfer, memory) is smaller than the gain.

---

## 2. Do more workers always mean more speed?

**No.** Both platforms show a peak followed by a decline.

Sandbox, RGB, scalar payload:

| Workers | 1 | 2 | 4 | **8** | 16 |
|---|---|---|---|---|---|
| Speedup | 1.00× | 1.91× | 3.32× | **5.03×** | 4.97× |

Kaggle, RGB, 27,000 images, n=3:

| Workers | 0 | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| Speedup | 1.00× | 0.99× | 1.00× | 1.00× | 0.99× | 0.99× |

Parallel efficiency (speedup ÷ workers) on Kaggle falls as exactly 1/w: **1.00 → 0.50 → 0.25 → 0.12**. That is the signature of workers that contribute nothing.

---

## 3. Why can too many workers slow a program down?

Four mechanisms, all measured:

**a. Oversubscription and context switching.** `thread ×4` performs **144,103 context switches per second** against `process ×8`'s 66,593 — twice as many while being 5.6× slower.

**b. Inter-process transfer cost (IPC).** Identical CPU work; only the returned payload differs:

| Dataset | Workers | array payload | scalar payload | gap |
|---|---|---|---|---|
| MS | 4 | 1.84× | 3.37× | **1.83×** |
| MS | 8 | 2.83× | 4.54× | 1.60× |
| RGB | 8 | 4.23× | 5.03× | 1.19× |

Returning data instead of a single number consumes **up to 45%** of the achievable speedup. Threads are the control: payload size makes no difference to them (0.91× vs 0.90×), because they share memory.

**c. Memory consumption.** MS `process ×16` with array payloads peaked at **+847 MB**, against +144 MB for scalar payloads and +34 MB for threads.

**d. Competition with the main process.** On Kaggle, `thread ×8` on MS dropped GPU utilisation to **78.5%** (sequential: 94%), because decode threads contend for the GIL with the thread issuing GPU commands.

---

## 4. Does multithreading improve performance?

**It depends on the kind of work — and we proved the rule.**

GIL control experiment (16 tasks split across N workers):

| Task | thread ×4 | process ×4 |
|---|---|---|
| Pure Python loop (holds the GIL) | **1.00×** | 3.73× |
| SHA-256 (releases the GIL) | **3.04×** | 3.58× |

Threads deliver **nothing** on work that holds the GIL, and scale nearly linearly on work that releases it. Same machine, same pool; the GIL is the only variable.

On real decoding:

| Condition | thread ×8 |
|---|---|
| Warm cache (CPU-bound) | **0.89×** — loses to sequential |
| Cold cache (I/O-bound) | **1.18×** — wins |

**Threads reverse sign** when the work shifts from CPU-bound to I/O-bound, because file reads release the GIL.

The most compact evidence: **threads plateau at 2.4 busy cores** out of 8 available, regardless of thread count (2, 4, 8 and 16 are identical). Processes reach **7.9 cores**. Threads physically cannot use the machine.

**Conclusion: threads help only when the work releases the GIL.** This single mechanism explains all four results above.

---

## 5. What is the main bottleneck: CPU, RAM, storage or GPU?

**It differs by regime**, and each has a number:

| Regime | Bottleneck | Evidence |
|---|---|---|
| Kaggle, T4 | **GPU** | 99.8% utilisation, 0.2% idle; all 12 configs within 0.68% |
| Sandbox, warm cache | **CPU** (GIL for threads) | threads capped at 2.4 cores, processes reach 7.9 |
| Sandbox, cold cache, few workers | **I/O latency** | sequential pays a 1.35× penalty |
| Sandbox, cold cache, many workers | **CPU again** | penalty falls to 1.09× |
| RAM | **never** | peak +847 MB of 30.8 GB |
| Storage bandwidth | **never saturated** | MS 605 of 1,315 MB/s (46%) |

Note: RGB with a cold cache at 16 workers reached **83% of its read ceiling**, closer to saturation than MS at 46%. That ceiling is an **operations-per-second** limit from 27,000 tiny files (3.32 KB each), not a bandwidth limit.

---

## 6. Was the GPU ever idle waiting for data from the CPU?

**In the final measurement: no.** GPU utilisation was 99.3–99.8% across **all** 12 configurations, including sequential. Idle time was 0.2–0.7%.

The reason is that CUDA is asynchronous. The main thread issues GPU commands, returns immediately, and decodes the next batch while the GPU works. So CPU and GPU already overlap **with no extra workers at all**. Per batch, the GPU needs 207 ms while one core decodes in 139 ms — decoding is already faster than training.

**Two points worth stating honestly in the report:**

1. Our earlier measurements appeared to show the GPU **37.3% idle** and a 1.50× speedup from one worker. On investigation this was a **page-cache ordering artifact**: the sequential baseline ran first, before the files were cached, so what we measured was disk read time rather than a pipeline problem. The final version reads every file once before timing, and the effect disappears.

2. The GPU **can** be starved — but not by a shortage of workers. `thread ×8` on MS dropped GPU utilisation to **78.5%**, because GIL contention delayed the commands being sent to the GPU. So adding threads **caused** the GPU to wait.

---

## 7. How many workers gives the best performance?

**It depends on the bottleneck:**

| Condition | Optimal workers | Speedup |
|---|---|---|
| Kaggle, GPU-bound | **0 (sequential)** | 1.00× |
| Sandbox, warm cache | **8 processes** = logical core count | 5.03× |
| Sandbox, cold cache | **16 processes** | 5.69× |

In the GPU-bound case, sequential matches the fastest configuration (`thread ×1`, a 0.03 s difference, p = 0.91) while using the **least CPU**: 0.88 cores against 1.47 cores (**+67%**) for a `thread ×1` that gains nothing.

With a cold cache the optimum shifts **higher** (16 rather than 8), because more workers hide more I/O latency.

---

# Conclusion

## Can multithreading or multiprocessing improve the deep learning pipeline? Which configuration is best, and why?

**It depends entirely on which stage is the bottleneck.**

**For the GPU training pipeline: no improvement is possible.** The GPU is already 99.8% busy, leaving 0.2% of headroom. All 12 configurations fall within 0.68% of each other, and four of them are **statistically slower** than sequential. The best configuration is **sequential**, because it matches the best performance while using the least CPU and memory.

**For the preprocessing pipeline (no GPU): yes, substantially.** **Multiprocessing with 8 workers gives 5.03×**, while multithreading never beats sequential.

Why 8 workers: it matches the machine's logical core count. Why processes and not threads: threads are capped at 2.4 cores by the GIL while processes reach 7.9. Why not 16: going beyond the core count adds context switching and memory without adding compute capacity.

**The principle: parallelism only helps at the bottleneck.** Parallelising a stage that is not the bottleneck still costs CPU and memory while returning nothing — demonstrated by `thread ×1` using 67% more CPU for a 0.0% gain.

## Do the results support or reject the assumption "more threads or processes means faster deep learning"?

**Reject.** Four independent pieces of evidence:

1. **Adding threads almost always made things worse.** Range 0.84–1.06×, and capped at 2.4 cores regardless of thread count.
2. **Processes peak then decline.** 8 workers 5.03×, 16 workers 4.97×.
3. **In the GPU pipeline, adding any worker gained nothing**, and 6–8 workers were **significantly slower** (p = 0.034 and p < 0.001).
4. **One worker can be slower than zero workers.** MS `process ×1` = **0.55×**.

The assumption holds only in a narrow band: when the parallelised stage really is the bottleneck, **and** the worker count has not exceeded the core count, **and** the data transfer cost is small. Outside those three conditions, adding workers is neutral or harmful.

---

## Methodology notes that changed the results

Four issues produced wrong numbers before they were controlled:

1. **OS page cache.** Produced a false 1.50× speedup. Fixed by reading every file once before timing.
2. **GPU clock dropping as it heats.** On a cold T4 the clock fell 1044 → 870 MHz while warming from 51 → 72 °C — larger than the differences between configurations. Fixed with a GPU warm-up to a stable temperature, plus randomised run order.
3. **PyTorch thread count.** The default claims every core, making worker effects uninterpretable. Pinned explicitly and reported.
4. **Variation between Kaggle sessions.** Single-core decode measured 459–590 img/s across sessions with identical settings. **Figures are only compared within one session.**
