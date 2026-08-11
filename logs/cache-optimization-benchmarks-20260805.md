# CUDA SSD Cache Optimization Benchmarks — 2026-08-05

Hardware: NVIDIA RTX 5080 Laptop GPU (16,303 MiB), Micron 2500 NVMe, WSL2 Ubuntu.

Common command: CUDA backend, `ds4flash.gguf`, 10 GB SSD expert-cache budget,
prefill chunk 128, context 4096, temperature 0, 12 output tokens, prompt
`Continue this sequence with only comma-separated numbers: 1, 2, 3,`.
Tracing is disabled for performance runs. Each row is the median of three fresh
processes. Expected output is `5, 8, 13, 21, `.

| Step | Configuration | Wall (s) | Prefill (t/s) | Decode (t/s) | Decode gain |
|---|---|---:|---:|---:|---:|
| 0 | grouped/sorted 4-reader baseline | 16.45 | 2.37 | 3.12 | — |
| 1 | decode-only LRU admission | 17.23 | 2.32 | 3.48 | +11.5% |
| 2a | 5 GiB host L2, baseline admission (screening run) | 18.48 | 2.23 | 2.18 | -30.1% |
| 2b | 5 GiB host L2 + decode LRU (screening run) | 18.84 | 2.31 | 1.95 | -44.0% vs step 1 |
| final | LRU + small-miss parallelism + persistent readers | 15.02 | 2.45 | 3.76 | +20.5% |

Raw baseline runs:

- 23.17 s, 1.99 prefill t/s, 3.08 decode t/s (cold-storage outlier)
- 16.45 s, 2.37 prefill t/s, 3.12 decode t/s
- 15.81 s, 2.39 prefill t/s, 3.13 decode t/s

Raw decode-LRU runs:

- 19.88 s, 2.01 prefill t/s, 3.48 decode t/s
- 17.23 s, 2.32 prefill t/s, 3.53 decode t/s
- 17.01 s, 2.37 prefill t/s, 3.40 decode t/s

Correctness/cache trace for step 1:
`cache-decode-lru-correctness-10gb-20260805.jsonl`.

- 516/516 route events match the baseline metadata, expert IDs, and weights.
- Baseline decode/final cache: 1,809 hits, 3,793 cumulative misses.
- Decode-LRU: 2,083 hits, 3,519 cumulative misses.
- Gain: 274 cache hits and 1,939,341,312 fewer SSD bytes (1.806 GiB).
- Generated output is identical.

Host-L2 screening:

- Baseline admission: the L2 served 1,029 spans / 2.26 GiB, but retaining
  4.52 GiB and copying every SSD miss into ordinary RAM reduced decode to
  2.18 t/s. Peak RSS was 5,988,424 KiB.
- With decode LRU, only 12 spans / 0.03 GiB hit the L2 because LRU already
  eliminated almost all repeated GPU misses. Decode fell to 1.95 t/s and peak
  RSS was 6,410,196 KiB.
- Result: reject the application-owned host L2. Its avoided SSD reads do not
  repay allocation and memory-copy overhead on this WSL configuration.

Compact IQ2 MMQ prefill screening:

The longer fixed prompt contains the instruction `Return only OK` followed by
eight repetitions of `one` through `ten`; one output token is generated. The
compact MMQ implementation consumes the SSD-streamed expert table and remapped
ids, so it no longer resolves the 256-expert slabs.

| Configuration | Median wall (s) | Median prefill (t/s) | Prefill gain |
|---|---:|---:|---:|
| MMQ disabled | 14.66 | 6.59 | — |
| Compact MMQ enabled | 15.36 | 6.60 | +0.2% |

- MMQ-off raw prefill: 6.53, 6.59, 6.76 t/s.
- Compact-MMQ raw prefill: 6.38, 6.87, 6.60 t/s.
- Both configurations generated `OK`; the original short prompt also retained
  the exact output `5, 8, 13, 21, `.
- Result: retain the safe compact integration as an opt-in path, but keep MMQ
  disabled in the measured best configuration because it provides no
  repeatable gain on this RTX 5080 / SSD-streaming workload.

CUDA prefill selected-load/shared-expert overlap:

This path replaces the device-wide selected-id barrier with a CUDA event and
uses the existing persistent selected-load service thread to read all batch
routes and prepare their compact expert table while the GPU evaluates the
shared expert. It is enabled with
`DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP=1` and is disabled automatically
when expert tracing is active.

| Configuration | Median wall (s) | Median prefill (t/s) | Prefill gain |
|---|---:|---:|---:|
| Synchronous selected batch load | 14.02 | 6.96 | — |
| Selected load/shared overlap | 13.89 | 6.94 | -0.3% |

- Synchronous raw runs: wall 13.69/14.98/14.02 s; prefill
  6.96/7.02/6.77 t/s.
- Overlap raw runs: wall 13.89/15.18/13.53 s; prefill
  6.90/6.94/6.99 t/s.
- All six runs generated the exact expected output `OK`.
- Result: retain as an experimental opt-in, but do not enable by default. The
  shared-expert interval is too short to hide a repeatable amount of SSD load
  time on this prompt; the small wall-time difference is within run variance.

Small-miss parallelism and persistent I/O readers:

These measurements return to the short 12-token decode prompt and keep the
decode LRU enabled. Small-miss parallelism uses three readers when a single
expert miss consists of only gate/up/down spans, instead of falling back to a
serial transfer because four readers were configured. The persistent-reader
step then reuses four sleeping threads instead of creating and joining them for
every parallel transfer group.

| Configuration | Median wall (s) | Median prefill (t/s) | Median decode (t/s) | Decode gain |
|---|---:|---:|---:|---:|
| Existing 4-reader threshold | 16.67 | 2.43 | 3.45 | — |
| Parallelize 3-span misses | 15.22 | 2.45 | 3.61 | +4.6% |
| + persistent reader pool | 15.02 | 2.45 | 3.76 | +9.0% cumulative |

- Existing threshold raw: wall 16.81/16.67/14.98 s; decode
  3.45/3.44/3.45 t/s.
- Small-miss parallel raw: wall 15.22/14.95/16.19 s; decode
  3.57/3.61/3.66 t/s.
- Persistent-reader raw: wall 15.02/16.26/14.80 s; decode
  3.76/3.71/3.76 t/s. An additional correctness screening run produced
  3.71 t/s.
- Every run generated exactly `5, 8, 13, 21, `.
- Result: enable both I/O changes in the machine launcher. Relative to the
  original grouped/sorted baseline at 3.12 t/s, the best measured decode rate
  is 20.5% higher; relative to the same-binary decode-LRU reference above, the
  two I/O changes add 9.0%.

Final correctness trace:
`cache-final-io-correctness-10gb-20260805.jsonl`.

- 516/516 route events match the decode-LRU reference exactly, including
  phase, layer, token positions, selected expert IDs, and routing weights.
- Final logical cache counters are also identical: 2,083 hits, 3,519 misses,
  2,402 insertions, 1,397 evictions, and 24,907,087,872 model bytes read.
- This confirms that small-miss parallelism and persistent readers change only
  I/O scheduling, not routing, admission, replacement, or generated output.
