#ifndef DS4_PROMPT_CACHE_POLICY_H
#define DS4_PROMPT_CACHE_POLICY_H

#include <stdbool.h>
#include <stdint.h>

/* Pure policy helpers shared by the CUDA cache and its CPU-only unit test.
 * The cache stores weights; this header only defines admission/protection
 * decisions, so policy invariants can be tested without a model or GPU. */
enum {
    DS4_PROMPT_CACHE_PHASE_IDLE = 0,
    DS4_PROMPT_CACHE_PHASE_PREFILL = 1,
    DS4_PROMPT_CACHE_PHASE_DECODE = 2,
};

enum {
    DS4_PROMPT_CACHE_PROTECT_PROMPT = 1u << 0,
    DS4_PROMPT_CACHE_PROTECT_SESSION = 1u << 1,
};

typedef struct ds4_prompt_cache_config {
    uint32_t prompt_percent;
    uint32_t session_percent;
    uint32_t prefill_min_frequency;
    uint32_t decode_protection_tokens;
} ds4_prompt_cache_config;

typedef struct ds4_prompt_cache_partition {
    uint32_t prompt_slots;
    uint32_t session_slots;
    uint32_t decode_reserve_slots;
} ds4_prompt_cache_partition;

static inline ds4_prompt_cache_config ds4_prompt_cache_default_config(void) {
    const ds4_prompt_cache_config config = {
        65u, /* prompt-hot */
        15u, /* session/global-hot */
        2u,  /* do not persist one-use prefill experts */
        8u,  /* extend protection when reused during early decode */
    };
    return config;
}

static inline ds4_prompt_cache_partition ds4_prompt_cache_partition_slots(
        uint32_t capacity,
        const ds4_prompt_cache_config *config) {
    ds4_prompt_cache_partition out = {0, 0, capacity};
    if (capacity == 0 || !config) return out;

    uint32_t prompt = (uint32_t)(
        ((uint64_t)capacity * config->prompt_percent) / 100u);
    uint32_t session = (uint32_t)(
        ((uint64_t)capacity * config->session_percent) / 100u);

    /* A decode/probation slot is a hard invariant. Small test caches therefore
     * shrink protection rather than allowing a fully pinned layer. */
    while ((uint64_t)prompt + session >= capacity) {
        if (session != 0) session--;
        else if (prompt != 0) prompt--;
        else break;
    }

    out.prompt_slots = prompt;
    out.session_slots = session;
    out.decode_reserve_slots = capacity - prompt - session;
    return out;
}

static inline bool ds4_prompt_cache_should_admit_prefill(
        uint64_t prompt_frequency,
        const ds4_prompt_cache_config *config) {
    return config &&
           prompt_frequency >= (uint64_t)config->prefill_min_frequency;
}

static inline bool ds4_prompt_cache_protection_active(
        uint8_t protection,
        uint32_t phase,
        uint64_t decode_token,
        uint64_t prompt_protection_until) {
    if ((protection & DS4_PROMPT_CACHE_PROTECT_SESSION) != 0 &&
        (phase == DS4_PROMPT_CACHE_PHASE_PREFILL ||
         phase == DS4_PROMPT_CACHE_PHASE_DECODE)) {
        return true;
    }
    return phase == DS4_PROMPT_CACHE_PHASE_DECODE &&
           (protection & DS4_PROMPT_CACHE_PROTECT_PROMPT) != 0 &&
           decode_token <= prompt_protection_until;
}

#endif
