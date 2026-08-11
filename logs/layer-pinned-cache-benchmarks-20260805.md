# Layer-pinned CUDA expert-cache benchmarks — 2026-08-05

Hardware: NVIDIA RTX 5080 Laptop GPU (16,303 MiB), Micron 2500 NVMe,
WSL2 Ubuntu. Model: DeepSeek V4 Flash 0731 GA Q2 GGUF.

All performance runs used the 10 GB SSD-streaming expert budget, prefill chunk
128, context 4096, decode LRU, four persistent direct-I/O readers, and a 256 MiB
CUDA weight-arena chunk. Tracing was disabled for performance measurements.
Each table reports medians of three fresh processes.

Profiles:

- `conservative`: six experts that were selected in all three source prefill
  workloads, spread across five layers.
- `full`: the two most-used experts in every routed layer, 86 entries total.
- Both use lazy admission and protect a pinned expert only after its first
  ordinary request/load.

## Short prefill plus 12-token decode

Prompt: `Continue this sequence with only comma-separated numbers: 1, 2, 3,`

| Policy | Wall (s) | Prefill (t/s) | Decode (t/s) |
|---|---:|---:|---:|
| Baseline layer-local cache | 16.26 | 2.43 | 3.64 |
| Conservative profile | 15.24 | 2.42 | 3.66 |
| Full profile | 15.05 | 2.42 | 3.64 |

All nine runs generated exactly `5, 8, 13, 21, `. Throughput is effectively
unchanged; lower profile wall times are startup/I/O variance rather than faster
prefill or decode.

The traced baseline finished with 2,083 hits, 3,519 misses, and
24,907,087,872 model bytes read. Conservative saved one miss (6.75 MiB); full
saved two misses (13.5 MiB). Both profile traces matched all 516 baseline route
events exactly, including expert ids and routing weights.

## Long isolated prefill

Prompt: `Return only OK` plus eight repetitions of `one` through `ten`, with
one output token.

| Policy | Wall (s) | Prefill (t/s) |
|---|---:|---:|
| Baseline layer-local cache | 14.40 | 6.91 |
| Conservative profile | 14.57 | 6.97 |
| Full profile | 14.73 | 6.84 |

Every run generated exactly `OK`. Baseline and full traces each had 3,344
misses, zero hits, and 23,668,457,472 model bytes read. A single batched prefill
loads each layer's unique routed experts once, so protecting them cannot create
an intra-prompt hit. All 43 routed-layer events were identical.

## Three-turn warm session

The same process answered a color prompt, `2+2`, and an Italian translation;
each turn generated one token. This is the workload most likely to benefit
from retaining stable experts between prompts.

| Policy | Wall (s) | Turn 1 prefill | Turn 2 prefill | Turn 3 prefill |
|---|---:|---:|---:|---:|
| Baseline layer-local cache | 41.18 | 1.54 t/s | 1.75 t/s | 0.83 t/s |
| Full profile | 43.58 | 1.41 t/s | 1.61 t/s | 0.84 t/s |

All six runs returned the same visible tokens: `orange`, `4`, `g`. The traced
full profile recorded 898 accesses to pinned residents but improved the net
cache result by only 15 hits / 101.25 MiB versus one baseline trace, because
almost all of those experts would already have remained resident under the
ordinary layer-local policy. A repeated baseline varied by 19 misses and
showed the same route divergence point as baseline-versus-profile, identifying
pre-existing CUDA multi-turn nondeterminism rather than a pinning regression.

## Decision

Keep `DS4_CUDA_PINNED_EXPERTS_FILE` as an opt-in profiling/experimentation
mechanism. Do not enable either profile by default on this machine: neither
provided a repeatable throughput gain, and the full profile was slower in the
warm-session median.

The upstream 1792 MiB weight-arena chunk caused a repeatable late allocation
failure near layer 38 after most model tensors and the 6.62 GiB expert cache
were resident. `DS4_CUDA_WEIGHT_ARENA_CHUNK_MB=256` removed the excessive late
reservation and completed every correctness and performance run, so the local
launcher now uses 256 MiB by default.
