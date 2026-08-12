#if !defined(_GNU_SOURCE) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L  /* clock_gettime */
#endif

#include "ds4_metrics.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static pthread_once_t g_metrics_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_metrics_mu;

static double g_metrics_start;  /* CLOCK_MONOTONIC seconds at init */
static uint64_t g_expert_hits;
static uint64_t g_expert_misses;
static uint64_t g_expert_hits_by_id[DS4_METRICS_EXPERT_CAPACITY];
static uint64_t g_expert_misses_by_id[DS4_METRICS_EXPERT_CAPACITY];
static uint64_t g_disk_bytes;
static uint64_t g_disk_reads;
static double g_disk_read_s;

static ds4_prompt_stat g_prompts[DS4_METRICS_PROMPT_CAPACITY];
static int g_prompts_head;      /* next slot to write */
static int g_prompts_len;       /* valid entries, capped at capacity */
static uint64_t g_prompt_seq;   /* total prompts recorded */

/* /proc/self/io sampling state (guarded by g_metrics_mu). */
static uint64_t g_proc_read_bytes;      /* last sampled cumulative value */
static uint64_t g_proc_prev_bytes;
static double g_proc_prev_t;            /* CLOCK_MONOTONIC of previous sample */
static double g_proc_read_mbs;          /* last computed MiB/s rate */

/* Reads the cumulative block-device read counter from /proc/self/io.
 * Returns 0 when /proc is unavailable (non-Linux, restricted procfs). */
static uint64_t ds4_metrics_proc_io_read_bytes(void) {
#ifdef __linux__
    FILE *fp = fopen("/proc/self/io", "r");
    if (!fp) return 0;
    char line[256];
    uint64_t read_bytes = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long v;
        if (sscanf(line, "read_bytes: %llu", &v) == 1) {
            read_bytes = (uint64_t)v;
            break;
        }
    }
    fclose(fp);
    return read_bytes;
#else
    return 0;
#endif
}

static double ds4_metrics_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void ds4_metrics_init_once(void) {
    pthread_mutex_init(&g_metrics_mu, NULL);
    g_metrics_start = ds4_metrics_now();
}

void ds4_metrics_init(void) {
    pthread_once(&g_metrics_once, ds4_metrics_init_once);
}

void ds4_metrics_record_prompt(const ds4_prompt_stat *stat) {
    if (!stat) return;
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    ds4_prompt_stat copy = *stat;
    g_prompt_seq++;
    if (copy.id == 0) copy.id = g_prompt_seq;
    g_prompts[g_prompts_head] = copy;
    g_prompts_head = (g_prompts_head + 1) % DS4_METRICS_PROMPT_CAPACITY;
    if (g_prompts_len < DS4_METRICS_PROMPT_CAPACITY) g_prompts_len++;
    pthread_mutex_unlock(&g_metrics_mu);
}

void ds4_metrics_expert_hit(uint32_t expert) {
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    g_expert_hits++;
    if (expert < DS4_METRICS_EXPERT_CAPACITY)
        g_expert_hits_by_id[expert]++;
    pthread_mutex_unlock(&g_metrics_mu);
}

void ds4_metrics_expert_miss(uint32_t expert) {
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    g_expert_misses++;
    if (expert < DS4_METRICS_EXPERT_CAPACITY)
        g_expert_misses_by_id[expert]++;
    pthread_mutex_unlock(&g_metrics_mu);
}

uint32_t ds4_metrics_get_experts(uint64_t *hits, uint64_t *misses,
                                 uint32_t capacity) {
    if (!hits || !misses || capacity == 0) return 0;
    ds4_metrics_init();
    const uint32_t n = capacity < DS4_METRICS_EXPERT_CAPACITY ?
        capacity : DS4_METRICS_EXPERT_CAPACITY;
    pthread_mutex_lock(&g_metrics_mu);
    memcpy(hits, g_expert_hits_by_id, (size_t)n * sizeof(*hits));
    memcpy(misses, g_expert_misses_by_id, (size_t)n * sizeof(*misses));
    pthread_mutex_unlock(&g_metrics_mu);
    return n;
}

void ds4_metrics_disk_read(uint64_t bytes, double seconds) {
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    g_disk_bytes += bytes;
    g_disk_reads++;
    g_disk_read_s += seconds;
    pthread_mutex_unlock(&g_metrics_mu);
}

void ds4_metrics_reset_activity(void) {
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    g_expert_hits = 0;
    g_expert_misses = 0;
    memset(g_expert_hits_by_id, 0, sizeof(g_expert_hits_by_id));
    memset(g_expert_misses_by_id, 0, sizeof(g_expert_misses_by_id));
    g_disk_bytes = 0;
    g_disk_reads = 0;
    g_disk_read_s = 0.0;
    const uint64_t rb = ds4_metrics_proc_io_read_bytes();
    g_proc_read_bytes = rb;
    g_proc_prev_bytes = rb;
    g_proc_prev_t = ds4_metrics_now();
    g_proc_read_mbs = 0.0;
    pthread_mutex_unlock(&g_metrics_mu);
}

void ds4_metrics_get_snapshot(ds4_metrics_snapshot *out) {
    if (!out) return;
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    out->expert_hits = g_expert_hits;
    out->expert_misses = g_expert_misses;
    out->disk_bytes = g_disk_bytes;
    out->disk_reads = g_disk_reads;
    out->disk_read_s = g_disk_read_s;
    out->uptime_s = ds4_metrics_now() - g_metrics_start;
    out->prompt_count = g_prompt_seq;
    /* Sample /proc/self/io on every snapshot.  The rate is only recomputed
     * when at least 0.2 s elapsed since the previous sample, so callers
     * polling faster than that still see a stable value. */
    const uint64_t rb = ds4_metrics_proc_io_read_bytes();
    const double now = ds4_metrics_now();
    if (rb) {
        const double dt = now - g_proc_prev_t;
        if (g_proc_prev_t > 0.0 && dt >= 0.2 && rb >= g_proc_prev_bytes) {
            g_proc_read_mbs =
                (double)(rb - g_proc_prev_bytes) / dt / (1024.0 * 1024.0);
            g_proc_prev_bytes = rb;
            g_proc_prev_t = now;
        } else if (g_proc_prev_t == 0.0) {
            g_proc_prev_bytes = rb;
            g_proc_prev_t = now;
        }
        g_proc_read_bytes = rb;
    }
    out->proc_read_bytes = g_proc_read_bytes;
    out->proc_read_mbs = g_proc_read_mbs;
    pthread_mutex_unlock(&g_metrics_mu);
}

int ds4_metrics_get_prompts(ds4_prompt_stat *out, int max) {
    if (!out || max <= 0) return 0;
    ds4_metrics_init();
    pthread_mutex_lock(&g_metrics_mu);
    int n = g_prompts_len < max ? g_prompts_len : max;
    for (int i = 0; i < n; i++) {
        /* newest first: head-1, head-2, ... wrapping around the ring */
        int idx = g_prompts_head - 1 - i;
        if (idx < 0) idx += DS4_METRICS_PROMPT_CAPACITY;
        out[i] = g_prompts[idx];
    }
    pthread_mutex_unlock(&g_metrics_mu);
    return n;
}

bool ds4_metrics_latest_prompt(ds4_prompt_stat *out) {
    if (!out) return false;
    ds4_metrics_init();
    bool ok;
    pthread_mutex_lock(&g_metrics_mu);
    ok = g_prompts_len > 0;
    if (ok) {
        int idx = g_prompts_head - 1;
        if (idx < 0) idx += DS4_METRICS_PROMPT_CAPACITY;
        *out = g_prompts[idx];
    }
    pthread_mutex_unlock(&g_metrics_mu);
    return ok;
}
