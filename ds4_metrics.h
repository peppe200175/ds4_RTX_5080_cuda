#ifndef DS4_METRICS_H
#define DS4_METRICS_H

/* Small, self-contained, thread-safe metrics collector for ds4-server's web
 * monitoring UI.  No external dependencies; safe to call from any thread and
 * safe to call before ds4_metrics_init() (lazy init via pthread_once). */

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_METRICS_PROMPT_CAPACITY 64
#define DS4_METRICS_EXPERT_CAPACITY 384

typedef struct ds4_prompt_stat {
    uint64_t id;
    double ts;
    int prompt_tokens, completion_tokens;
    double prefill_s, ttft_s, gen_s, tok_s;
    int cache_read_tokens, cache_write_tokens;
    uint64_t expert_hits, expert_misses, disk_bytes;
    double disk_s;
} ds4_prompt_stat;

typedef struct ds4_metrics_snapshot {
    uint64_t expert_hits;
    uint64_t expert_misses;
    uint64_t disk_bytes;
    uint64_t disk_reads;
    double disk_read_s;
    double uptime_s;
    uint64_t prompt_count;  /* total prompts recorded so far (ring seq) */
    /* Process-level block-device reads sampled from /proc/self/io (Linux).
     * Unlike the hook-based counters above, these also capture mmap page
     * faults, so they report real disk traffic in CPU streaming mode too.
     * Zero when /proc is unavailable. */
    uint64_t proc_read_bytes;
    double proc_read_mbs;   /* MiB/s between the last two samples */
} ds4_metrics_snapshot;

/* Idempotent; also called lazily by every other entry point. */
void ds4_metrics_init(void);

/* Records one completed prompt into the ring buffer.  If stat->id is 0 a
 * monotonically increasing id is assigned. */
void ds4_metrics_record_prompt(const ds4_prompt_stat *stat);

void ds4_metrics_expert_hit(uint32_t expert);
void ds4_metrics_expert_miss(uint32_t expert);
void ds4_metrics_disk_read(uint64_t bytes, double seconds);
/* Clears activity counters after internal startup work such as GPU warmup. */
void ds4_metrics_reset_activity(void);

/* Copies cumulative cache decisions grouped by expert id (across layers).
 * Returns the number of expert slots copied. */
uint32_t ds4_metrics_get_experts(uint64_t *hits, uint64_t *misses,
                                 uint32_t capacity);

/* Copies the counters out under the lock.  uptime_s is seconds since the
 * first ds4_metrics_init(). */
void ds4_metrics_get_snapshot(ds4_metrics_snapshot *out);

/* Copies up to max newest-first prompt records into out; returns the number
 * of entries written. */
int ds4_metrics_get_prompts(ds4_prompt_stat *out, int max);

/* Copies the most recently recorded prompt into out.  Returns false when no
 * prompt has been recorded yet.  Intended for the end-of-stream stats frame:
 * trace_finish() records the prompt before the SSE tail is sent. */
bool ds4_metrics_latest_prompt(ds4_prompt_stat *out);

#ifdef __cplusplus
}
#endif

#endif /* DS4_METRICS_H */
