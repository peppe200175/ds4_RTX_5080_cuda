#include "../ds4_prompt_cache_policy.h"

#include <stdio.h>

#define CHECK(condition, message)                                           \
    do {                                                                    \
        if (!(condition)) {                                                 \
            fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int main(void) {
    const ds4_prompt_cache_config config = ds4_prompt_cache_default_config();

    const ds4_prompt_cache_partition normal =
        ds4_prompt_cache_partition_slots(23u, &config);
    CHECK(normal.prompt_slots == 14u, "23-slot prompt partition");
    CHECK(normal.session_slots == 3u, "23-slot session partition");
    CHECK(normal.decode_reserve_slots == 6u, "23-slot decode reserve");

    const ds4_prompt_cache_partition tiny =
        ds4_prompt_cache_partition_slots(1u, &config);
    CHECK(tiny.prompt_slots == 0u, "tiny prompt partition shrinks");
    CHECK(tiny.session_slots == 0u, "tiny session partition shrinks");
    CHECK(tiny.decode_reserve_slots == 1u, "tiny cache preserves decode slot");

    CHECK(!ds4_prompt_cache_should_admit_prefill(1u, &config),
          "one-use prefill expert remains probationary");
    CHECK(ds4_prompt_cache_should_admit_prefill(2u, &config),
          "second-use prefill expert is admitted");

    CHECK(ds4_prompt_cache_protection_active(
              DS4_PROMPT_CACHE_PROTECT_PROMPT,
              DS4_PROMPT_CACHE_PHASE_DECODE,
              8u,
              8u),
          "prompt protection includes its final token");
    CHECK(!ds4_prompt_cache_protection_active(
               DS4_PROMPT_CACHE_PROTECT_PROMPT,
               DS4_PROMPT_CACHE_PHASE_DECODE,
               9u,
               8u),
          "unused prompt protection expires");
    CHECK(ds4_prompt_cache_protection_active(
              DS4_PROMPT_CACHE_PROTECT_SESSION,
              DS4_PROMPT_CACHE_PHASE_DECODE,
              1000u,
              0u),
          "session-hot protection spans decode");
    CHECK(ds4_prompt_cache_protection_active(
              DS4_PROMPT_CACHE_PROTECT_SESSION,
              DS4_PROMPT_CACHE_PHASE_PREFILL,
              0u,
              0u),
          "session-hot protection spans the next prefill");
    CHECK(!ds4_prompt_cache_protection_active(
               DS4_PROMPT_CACHE_PROTECT_PROMPT,
               DS4_PROMPT_CACHE_PHASE_PREFILL,
               0u,
               8u),
          "prompt protection starts only after prefill");

    fprintf(stderr, "test_prompt_cache_policy PASS\n");
    return 0;
}
