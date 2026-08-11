# DS4 RTX 5080 128K optimization report

Date: 2026-08-07

Model: DeepSeek V4 Flash 0731 IQ2/Q2, CUDA sm_120, RTX 5080 Laptop GPU 16 GB.

## Acceptance rule

Each experiment changed one variable at a time. A result was retained only
when it improved prefill and/or decode and preserved deterministic generated
answers, token counts, status, and finish reason. Microkernels additionally
required exact selected indices or bitwise-equal output, as appropriate.

## Recovered 128K baseline

Configuration: 7 GB expert budget, prefill chunk 128, context 131072.

Three runs produced a median of 5.006 prefill tok/s and 3.427 decode tok/s.
The third prefill run was a thermal outlier; all three produced identical
answers.

## Retained changes

### Exact streaming decode top-K at wide context

The existing exact 512-thread streaming top-K kernel is now used for
single-token decode when the compressed index exceeds 8192 rows. At 32768
rows the final regression measured 0.112 ms for the old tree versus 0.088 ms
for streaming: 1.273x faster with identical 512 selected indices. The measured
crossover sweep ranged from 1.27x to 1.83x in favor of streaming.

### Lazy, geometrically growing compressed KV allocations

With `DS4_CUDA_LAZY_KV_CACHE=1`, each layer initially allocates compressed KV
rows for 4096 tokens and grows losslessly up to the declared 128K capacity.
Live F32 rows are copied byte-for-byte, tensor-parallel mirrors are preserved,
and captured CUDA graphs are invalidated before old addresses are released.
The matched 7 GB experiment reduced sampled peak VRAM from 15775 MiB to
14029 MiB. A forced 8-token initial-capacity test crossed 42 growth events
across all 21 ratio-4 layers and still matched every baseline answer exactly.

The freed memory made the 8 GB expert budget useful. Two chunk-128 runs reached
5.295/5.107 prefill tok/s and 3.752/3.725 decode tok/s. Relative to the recovered
median, those runs improved prefill by 2.0-5.8% and decode by 8.7-9.5%.

### Prefill chunk 1024

Controlled long-prefill results:

| Input | Chunk | Prefill tok/s | Exact |
|---:|---:|---:|:---:|
| 276 tokens | 128 | 11.561 | yes |
| 276 tokens | 256 | 15.943 | yes |
| 276 tokens | 512 | 21.644 | yes |
| 820 tokens | 512 | 32.908 | yes |
| 820 tokens | 1024 | 49.108 | yes |

Chunk 1024 is 49.2% faster than 512 on the matched 820-token input. The full
128K-KV residency check also allocated and ran successfully with chunk 1024.

## Final pinned profile

- Context: 131072
- Expert budget: 8 GB
- Prefill chunk: 1024
- Lazy compressed KV initial context: 4096 tokens
- MMQ prefill tier: disabled, as in the recovered fastest profile

The freshly rebuilt and pinned ten-prompt run produced 3.687 decode tok/s,
7.6% above the recovered median. Short-prompt prefill varied from 4.770 to
5.006 tok/s with chunk 1024; the controlled long-prefill sweep above is the
relevant prefill gain. All five compared output fields matched the recovered
baseline for all ten prompts.

## Rejected changes

| Experiment | Result | Reason rejected |
|---|---:|---|
| Persistent packed indexer | 0.291 vs 0.292 ms | No speedup |
| Single-token grouped attention | 0.127 vs 0.209 ms | 40% slower and changed bits |
| Lossless compact attention rows | 0.110 vs 0.384 ms | 3.5x slower |
| MoE decode graph | 3.358 decode tok/s | About 2.0% below baseline |
| Split-score vec4 | 3.328 decode tok/s | Slower and changed prompt 9 answer |
| Split-score DIM2 | 3.142 decode tok/s | About 8.3% below baseline |
| Fused inverse-RoPE score path | 3.087 decode tok/s | About 9.9% below baseline |
| 8 GB experts without lazy KV | 3.436 decode tok/s, 15909 MiB | No repeatable gain; insufficient headroom |

Session checkpoint serialization was not changed because the existing format
already writes only live raw and compressed rows rather than reserved capacity.

## Validation

- Final ten-prompt output comparison: exact.
- Forced lazy-KV growth output comparison: exact.
- CPU build: passed.
- CUDA sm_120 full build: passed.
- Server/parser, agent, evaluator, layer-pack, prompt-cache-policy,
  placement, GPU argument, sampling, and CLI tests: passed.
- CUDA long-context top-K regression: passed.

The broad `make test` target initially stopped because its default
`ds4flash.gguf` path is absent from the workspace. Its non-model components
were then run explicitly and passed; model correctness is covered by the
actual 0731 ten-prompt and forced-growth integrations above.
