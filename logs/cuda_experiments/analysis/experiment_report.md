# DS4 CUDA experiment report

Correctness reference: `logs/cuda_experiments/20260806T113610Z_baseline_before/results.json`
Performance control: `logs/cuda_experiments/20260806T123621Z_baseline_control_after_chunk32/results.json`

| Experiment | Change | Exact | Total s | vs control | Prefill t/s | Decode t/s | Hit rate | SSD GiB | Start/avg/max °C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_before | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_CACHE_SUMMARY=<unset>; DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS=<unset>; DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset>; cache_budget=<unset> | yes | 119.30 | -0.8% | 4.438 | 3.077 | 0.0% | 0.00 | -/-/- |
| refactor_flag_off | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_CACHE_SUMMARY=<unset>; DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 120.15 | -0.1% | 4.397 | 3.062 | 0.0% | 0.00 | -/-/- |
| decode_reclaim_512 | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_CACHE_SUMMARY=<unset>; DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS=512; DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | no | 0.00 | -100.0% | 0.000 | 0.000 | 0.0% | 0.00 | -/-/- |
| decode_reclaim_512_fixed | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_CACHE_SUMMARY=<unset>; DS4_CUDA_DECODE_CACHE_EXTRA_EXPERTS=512; DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 135.12 | +12.3% | 4.179 | 2.555 | 0.0% | 0.00 | -/-/- |
| layer_quota_replay_profile | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_LAYER_CACHE_CAPACITIES_FILE=/mnt/c/Users/gmart/OneDrive/ds4/logs/expert_trace_10prompts_20260806T110937Z/analysis/recommended_layer_capacities.txt; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 125.40 | +4.2% | 4.314 | 2.865 | 0.0% | 0.00 | -/-/- |
| prefill_shared_overlap | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset>; DS4_CUDA_STREAMING_PREFILL_SHARED_OVERLAP=1 | no | 162.56 | +35.1% | 3.344 | 2.199 | 0.0% | 0.00 | -/-/- |
| prompt_cache_default | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset>; DS4_CUDA_PROMPT_EXPERT_CACHE=1 | yes | 185.49 | +54.2% | 2.780 | 2.034 | 0.0% | 0.00 | -/-/- |
| baseline_control_mid | DS4_BENCH_MAX_GPU_TEMP=<unset>; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 160.92 | +33.8% | 3.276 | 2.290 | 49.9% | 197.72 | -/-/- |
| steady_baseline_78c | DS4_BENCH_MAX_GPU_TEMP=78; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 166.12 | +38.1% | 3.121 | 2.258 | 49.8% | 197.72 | 77.0/79.1/81.0 |
| pin_workload_2_per_layer | DS4_BENCH_MAX_GPU_TEMP=78; DS4_CUDA_PINNED_EXPERTS_FILE=/mnt/c/Users/gmart/OneDrive/ds4/logs/expert_trace_10prompts_20260806T110937Z/analysis/recommended_pinned_experts.txt; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | no | 180.84 | +50.3% | 3.009 | 1.996 | 50.3% | 197.79 | 72.0/74.2/78.0 |
| pin_conservative_6_layers | DS4_BENCH_MAX_GPU_TEMP=78; DS4_CUDA_PINNED_EXPERTS_FILE=/mnt/c/Users/gmart/OneDrive/ds4/profiles/ds4-flash-0731-prefill-pinned-conservative.txt; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=<unset> | yes | 168.08 | +39.7% | 3.414 | 2.023 | 49.8% | 197.91 | 73.0/75.3/78.0 |
| pin_conservative_prefill_only | DS4_BENCH_MAX_GPU_TEMP=78; DS4_CUDA_PINNED_EXPERTS_FILE=/mnt/c/Users/gmart/OneDrive/ds4/profiles/ds4-flash-0731-prefill-pinned-conservative.txt; DS4_CUDA_PINNED_EXPERTS_PREFILL_ONLY=1 | no | 126.71 | +5.3% | 4.209 | 2.907 | 50.0% | 198.76 | 65.0/71.1/76.0 |
| baseline_post_build_72 | baseline profile | yes | 126.83 | +5.4% | 4.219 | 2.864 | 49.9% | 197.70 | 68.0/72.7/76.0 |
| staging_chunk_128mb | DS4_CUDA_MODEL_COPY_CHUNK_MB=128 | yes | 130.58 | +8.6% | 4.275 | 2.670 | 49.9% | 197.68 | 70.0/73.1/76.0 |
| staging_chunk_32mb | DS4_CUDA_MODEL_COPY_CHUNK_MB=32 | yes | 123.97 | +3.1% | 4.437 | 2.851 | 49.8% | 197.71 | 70.0/70.7/73.0 |
| baseline_control_after_chunk32 | baseline profile | yes | 120.29 | +0.0% | 4.419 | 3.039 | 49.8% | 197.76 | 66.0/69.6/73.0 |
| staging_chunk_16mb | DS4_CUDA_MODEL_COPY_CHUNK_MB=16 | yes | 123.03 | +2.3% | 4.256 | 3.017 | 49.8% | 197.74 | 67.0/70.0/73.0 |
| cache_1040_experts | DS4_EXPERIMENT_CACHE_BUDGET=1040; cache_budget=1040 | yes | 128.08 | +6.5% | 4.167 | 2.843 | 50.5% | 195.24 | 65.0/69.2/74.0 |
| cache_1024_experts | DS4_EXPERIMENT_CACHE_BUDGET=1024; cache_budget=1024 | no | 121.21 | +0.8% | 4.372 | 3.060 | 50.4% | 197.17 | 69.0/71.0/73.0 |
| read_threads_8 | DS4_CUDA_STREAMING_READ_THREADS=8 | yes | 124.19 | +3.2% | 4.247 | 2.967 | 49.9% | 197.65 | 66.0/68.6/71.0 |
| read_threads_2 | DS4_CUDA_STREAMING_READ_THREADS=2 | no | 129.93 | +8.0% | 4.040 | 2.881 | 50.0% | 198.63 | 67.0/69.1/71.0 |
| small_miss_parallel_off | DS4_CUDA_STREAMING_SMALL_MISS_PARALLEL=0 | no | 129.29 | +7.5% | 4.224 | 2.784 | 50.0% | 198.72 | 63.0/68.9/74.0 |
| final_best_complete_trace | baseline profile | yes | 126.05 | +4.8% | 4.196 | 2.915 | 51.0% | 227.93 | 60.0/66.0/71.0 |
| final_best_complete_trace_repeat | baseline profile | yes | 121.91 | +1.3% | 4.401 | 2.971 | 51.0% | 227.90 | 58.0/65.4/70.0 |

Fastest exact-output run: **baseline_before**, 119.30 s.
