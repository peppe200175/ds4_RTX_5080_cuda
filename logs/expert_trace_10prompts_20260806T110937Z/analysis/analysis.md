# DwarfStar expert trace analysis

- Trace: `logs\expert_trace_10prompts_20260806T110937Z\expert_trace.jsonl`
- SHA-256: `12a1af34c61c4263db6c4765990da033971b62494446f67a60cdc37326510d9e`
- Prompts: 10
- Route events: 8213
- Cache-stat events: 8213
- Directly timed cache transactions: 430
- Expert selections: 115842
- Unique cache requests: 70510
- Unique cache hits: 35943
- Unique SSD misses: 34567
- Transaction hit rate: 50.98%
- Model bytes read: 227.86 GiB
- Evictions: 22646
- Integrity checks: PASS

## Prefill cache pressure

- Mean/median/p90/max unique experts per layer transaction: 55.4 / 50.0 / 74.1 / 163
- Transactions over their layer capacity: 430 / 430
- Single-use/repeated unique requests: 12429 / 11383
- Gross top-two-per-layer pinning upper bound: 961 SSD loads, 6.33 GiB
- Same-budget per-layer replay: 477 fewer misses (1.38%, 3.14 GiB); baseline match=yes
- Decode reuse of 3.38 GiB prefill reserve: 3680 fewer misses (10.65%, 24.26 GiB), before resize costs

## Phase summary

| Phase | Selections | Unique hits | Unique misses | Hit rate | SSD GiB | Evictions | Cache-load s |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode | 46698 | 31581 | 15117 | 67.63% | 99.65 | 15117 | 0.00 |
| prefill | 69144 | 4362 | 19450 | 18.32% | 128.21 | 7529 | 49.99 |

## Layers with most SSD traffic

| Layer | SSD GiB | Unique misses | Hit rate | Evictions | Top experts (id:prompt-count:selected) |
|---:|---:|---:|---:|---:|---|
| 1 | 13.14 | 1994 | 9.90% | 1216 | 93:10:33, 0:10:32, 8:10:30, 32:10:29, 214:10:28 |
| 0 | 12.88 | 1954 | 11.42% | 1207 | 245:10:38, 126:10:31, 85:10:27, 174:10:26, 171:10:23 |
| 2 | 12.74 | 1933 | 12.73% | 1173 | 128:10:43, 65:10:37, 115:10:32, 53:10:30, 135:10:26 |
| 19 | 7.09 | 1075 | 39.94% | 682 | 136:10:214, 81:10:213, 226:10:154, 67:10:109, 181:10:72 |
| 4 | 6.37 | 967 | 44.17% | 613 | 16:10:317, 119:10:216, 8:10:161, 161:10:93, 159:10:49 |
| 6 | 6.20 | 940 | 45.38% | 610 | 1:10:148, 111:10:118, 183:10:112, 138:10:94, 52:10:76 |
| 3 | 6.18 | 937 | 45.55% | 609 | 60:10:398, 74:10:263, 138:10:121, 119:10:80, 173:10:75 |
| 9 | 5.91 | 896 | 46.95% | 582 | 185:10:137, 216:10:132, 242:10:99, 191:10:87, 57:10:85 |
| 23 | 5.73 | 869 | 47.04% | 623 | 155:10:181, 213:10:115, 245:10:94, 170:10:88, 118:10:73 |
| 10 | 5.62 | 852 | 48.58% | 582 | 86:10:268, 40:10:259, 107:10:89, 11:10:87, 136:10:67 |

## Most stable layer/expert pairs

| Layer | Expert | Prompts | Selections | SSD loads | Cache hits | Selection hit rate |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 97 | 10 | 426 | 1 | 189 | 95.54% |
| 37 | 38 | 10 | 420 | 1 | 190 | 95.71% |
| 38 | 110 | 10 | 413 | 2 | 182 | 95.16% |
| 41 | 133 | 10 | 413 | 2 | 183 | 95.40% |
| 40 | 26 | 10 | 412 | 2 | 183 | 95.39% |
| 35 | 99 | 10 | 410 | 1 | 185 | 95.61% |
| 33 | 158 | 10 | 403 | 1 | 187 | 96.28% |
| 39 | 178 | 10 | 401 | 2 | 176 | 95.51% |
| 3 | 60 | 10 | 398 | 1 | 174 | 95.23% |
| 38 | 238 | 10 | 386 | 3 | 164 | 81.09% |
| 36 | 16 | 10 | 385 | 1 | 175 | 95.84% |
| 40 | 117 | 10 | 377 | 2 | 175 | 86.47% |
| 32 | 81 | 10 | 372 | 1 | 175 | 96.77% |
| 40 | 206 | 10 | 366 | 2 | 167 | 95.90% |
| 29 | 66 | 10 | 361 | 1 | 175 | 96.40% |
| 36 | 25 | 10 | 341 | 2 | 151 | 84.75% |
| 34 | 227 | 10 | 341 | 1 | 161 | 96.48% |
| 31 | 241 | 10 | 326 | 3 | 145 | 86.20% |
| 33 | 55 | 10 | 321 | 3 | 145 | 80.69% |
| 39 | 66 | 10 | 318 | 3 | 145 | 81.76% |

## Interpretation notes

- `expert_selections` counts every routed occurrence. A repeated expert in one prefill batch is counted repeatedly.
- `unique_cache_misses` and `disk_loads` count one load per expert per cache transaction.
- `model_bytes_read` is the authoritative CUDA counter; per-cell bytes are estimates using the traced expert size.
- All route/cache consistency checks passed only when `Integrity checks` is `PASS`.
- `observed_transactions` means directly timed transactions; `cache_stat_transactions` includes untimed decode snapshots.
- Tracing serializes the selected/shared path, so trace timing is diagnostic and must not replace the untraced performance benchmark.
- The generated pinned profile is a workload-specific candidate, not an automatically enabled setting.
