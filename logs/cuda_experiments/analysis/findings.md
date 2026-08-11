# DeepSeek V4 Flash 0731 CUDA optimization findings

## Decision

The fastest output-preserving profile remains the existing optimized CUDA profile:

- 1,005 cached routed experts (`10GB` byte budget after the mandatory two-layer prefill reserve)
- 4 persistent SSD readers
- parallel three-span reads for isolated misses
- decode LRU enabled
- 64 MiB pinned transfer chunks
- 256 MiB CUDA weight-arena chunks
- MMQ prefill tier disabled
- Q8→F16 weight cache disabled
- dynamic decode expansion, layer quotas, prompt-aware policy, overlap, and expert pinning disabled

Three exact-output best-condition controls took 119.30, 120.15, and 120.29 seconds: mean 119.91 seconds, standard deviation 0.44 seconds. Their combined rates were approximately 4.42 input tokens/s and 3.06 decode tokens/s. The final complete-trace repeat took 121.91 seconds (4.40 prefill tokens/s, 2.97 decode tokens/s) and reproduced all ten answers exactly.

## Final ten-prompt cache trace

The repeat contains all ten per-prompt summaries and separates prefill from decode:

- Total expert requests: 70,510
- Hits: 35,936; misses: 34,574; overall hit rate: 50.97%
- SSD reads: 227.90 GiB
- Evictions: 22,659
- Prefill: 4,361 hits, 19,451 misses, 18.31% hit rate, 128.22 GiB SSD
- Decode: 31,575 hits, 15,123 misses, 67.62% hit rate, 99.68 GiB SSD
- The 1,005-slot cache was full and resident after every prompt.

The heaviest prompts by SSD traffic were sentiment (32.83 GiB), Python (30.67 GiB), practical advice (30.16 GiB), and summarization (28.29 GiB). The dominant remaining problem is prefill reuse: decode already obtains about two hits out of three, while prefill obtains fewer than one in five.

## A/B conclusions

All speed comparisons below use the 120.29-second controlled baseline where possible. A candidate is rejected if any answer, token count, or finish reason differs.

| Experiment | Result | Output | Decision |
|---|---:|---:|---|
| Flag-off cache refactor | 120.15 s | exact | Refactor is neutral and safe |
| +512 decode-only cache slots | 135.12 s (+12.3%) | exact | Reject |
| Trace-derived layer quotas | 125.40 s (+4.2%) | exact | Reject |
| Shared prefill overlap | 162.56 s (+35.1%) | changed | Reject |
| Prompt-aware cache defaults | 185.49 s (+54.2%) | exact | Reject |
| Aggressive two-expert-per-layer pinning | 180.84 s | changed | Reject |
| Conservative pinning | 168.08 s | exact | Reject; prefill improved but decode regressed |
| Prefill-only conservative pinning | 126.71 s | changed | Reject |
| Pinned transfer chunk 16 MiB | 123.03 s (+2.3%) | exact | Reject |
| Pinned transfer chunk 32 MiB | 123.97 s (+3.1%) | exact | Reject after sandwich control |
| Pinned transfer chunk 128 MiB | 130.58 s (+8.6%) | exact | Reject |
| Cache 1,024 experts | 121.21 s (+0.8%) | changed | Reject |
| Cache 1,040 experts | 128.08 s (+6.5%) | exact | Reject |
| 2 SSD reader threads | 129.93 s (+8.0%) | changed | Reject |
| 8 SSD reader threads | 124.19 s (+3.2%) | exact | Reject |
| Small-miss parallel reads disabled | 129.29 s (+7.5%) | changed | Reject; keep enabled |

The 1,040-expert test illustrates why maximum VRAM occupancy is not the same as maximum speed. It raised the nine-prompt hit rate from 49.84% to 50.48% and removed 2.52 GiB of SSD reads, but total time increased by 6.5%. The selected profile already peaked at 15,974 of 16,303 MiB (97.98% VRAM), leaving only the safety margin needed by CUDA workspaces and the selected-expert staging path.

## Implemented experimental facilities

- Phase-resizable cache with optional decode-only expert capacity (`DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS`)
- Per-layer capacity profiles (`DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE`)
- Prefill-only expert protection (`DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY`)
- Complete per-prompt cache summaries, including separate prefill/decode hit, miss, eviction, and SSD totals
- Thermally sampled A/B runner with isolated logs and exact-output checks
- Aggregate JSON, CSV, and Markdown experiment reports

The experimental paths remain opt-in. None is enabled in the interactive launcher because none passed both the speed and exact-output gates.

## Artifacts

- Full expert-by-layer selection trace: `logs/expert_trace_10prompts_20260806T110937Z/expert_trace.jsonl`
- Final complete trace: `logs/cuda_experiments/20260806T125902Z_final_best_complete_trace_repeat/`
- Machine-readable comparison: `logs/cuda_experiments/analysis/experiment_report.json`
- Spreadsheet-friendly comparison: `logs/cuda_experiments/analysis/experiment_report.csv`
- Full experiment table: `logs/cuda_experiments/analysis/experiment_report.md`
