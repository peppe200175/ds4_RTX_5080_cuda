#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr uint32_t kIq2Type = 16u;
constexpr uint32_t kQ2Type = 10u;
constexpr uint32_t kQk = 256u;
constexpr uint32_t kIn = 4096u;
constexpr uint32_t kRouterOut = 256u;
constexpr uint32_t kMid = 2048u;
constexpr uint32_t kOut = 4096u;
constexpr uint32_t kTotalExperts = 32u;
constexpr uint32_t kTop = 6u;
constexpr uint32_t kGuardBytes = 64u;
constexpr uint8_t kGuardByte = 0xd7u;
constexpr uint8_t kPoisonByte = 0xa5u;
constexpr float kClamp = 7.0f;

struct block_iq2_xxs_test {
    uint16_t d;
    uint16_t qs[kQk / 8u];
};

struct block_q2_k_test {
    uint8_t scales[kQk / 16u];
    uint8_t qs[kQk / 4u];
    uint16_t d;
    uint16_t dmin;
};

struct block_q8_k_test {
    float d;
    int8_t qs[kQk];
    int16_t bsums[kQk / 16u];
};

static_assert(sizeof(block_iq2_xxs_test) == 66u, "unexpected IQ2_XXS layout");
static_assert(sizeof(block_q2_k_test) == 84u, "unexpected Q2_K layout");
static_assert(sizeof(block_q8_k_test) == 292u, "unexpected Q8_K layout");

struct guarded_tensor {
    const char *name = nullptr;
    uint64_t payload_bytes = 0;
    uint64_t prefix_bytes = kGuardBytes;
    ds4_gpu_tensor *base = nullptr;
    ds4_gpu_tensor *view = nullptr;
};

struct tensors {
    guarded_tensor x;
    guarded_tensor selected;
    guarded_tensor weights;
    guarded_tensor gate;
    guarded_tensor up;
    guarded_tensor mid;
    guarded_tensor down;
    guarded_tensor out;
};

struct model_fixture {
    uint8_t *map = nullptr;
    uint64_t bytes = 0;
    uint64_t gate_offset = 0;
    uint64_t up_offset = 0;
    uint64_t down_offset = 0;
    uint64_t router_offset = 0;
    uint64_t gate_row_bytes = 0;
    uint64_t gate_expert_bytes = 0;
    uint64_t down_row_bytes = 0;
    uint64_t down_expert_bytes = 0;
};

struct capture {
    std::vector<uint8_t> xq;
    std::vector<uint8_t> midq;
    std::vector<float> mid;
    std::vector<float> out;
};

struct correctness_result {
    bool ok = true;
    uint32_t cases = 0;
    uint32_t calls = 0;
    uint32_t byte_checks = 0;
    uint32_t output_rows = 0;
};

struct graph_identity_result {
    bool ok = true;
    uint32_t byte_checks = 0;
    uint32_t output_rows = 0;
    bool pointer_update = false;
    ds4_gpu_decode_graph_stats before = {};
    ds4_gpu_decode_graph_stats after = {};
};

struct stream_offset_result {
    bool ok = true;
    uint32_t rows = 0;
    uint32_t byte_checks = 0;
    uint32_t negative_checks = 0;
    uint32_t cache_capacity = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t cache_evictions = 0;
    uint64_t prefetch_admitted = 0;
};

struct router_rows_result {
    bool ok = true;
    uint32_t cases = 0;
    uint32_t rows = 0;
    uint32_t byte_checks = 0;
    uint32_t canary_checks = 0;
};

struct direct_midq_result {
    bool ok = true;
    uint32_t cases = 0;
    uint32_t byte_checks = 0;
    uint32_t guarded_captures = 0;
};

struct moe_batch_exact_result {
    bool ok = true;
    uint32_t cases = 0;
    uint32_t rows = 0;
    uint32_t byte_checks = 0;
    uint32_t canary_checks = 0;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

bool cuda_ok(cudaError_t error, const char *what) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(error));
    return false;
}

bool guarded_init_offset(guarded_tensor *tensor, const char *name,
                         uint64_t bytes, uint64_t prefix_bytes) {
    if ((prefix_bytes != 0u && prefix_bytes < kGuardBytes) ||
        prefix_bytes > UINT64_MAX - bytes ||
        prefix_bytes + bytes > UINT64_MAX - kGuardBytes) {
        return false;
    }
    tensor->name = name;
    tensor->payload_bytes = bytes;
    tensor->prefix_bytes = prefix_bytes;
    const uint64_t total_bytes = prefix_bytes + bytes + kGuardBytes;
    tensor->base = ds4_gpu_tensor_alloc(total_bytes);
    if (!tensor->base) return false;
    tensor->view = ds4_gpu_tensor_view(tensor->base, prefix_bytes, bytes);
    if (!tensor->view) return false;
    std::vector<uint8_t> init((size_t)total_bytes, kGuardByte);
    return ds4_gpu_tensor_write(tensor->base, 0u, init.data(), init.size()) != 0;
}

bool guarded_init(guarded_tensor *tensor, const char *name, uint64_t bytes) {
    return guarded_init_offset(tensor, name, bytes, kGuardBytes);
}

void guarded_free(guarded_tensor *tensor) {
    ds4_gpu_tensor_free(tensor->view);
    ds4_gpu_tensor_free(tensor->base);
    tensor->view = nullptr;
    tensor->base = nullptr;
}

bool guarded_poison_payload(guarded_tensor *tensor) {
    std::vector<uint8_t> poison((size_t)tensor->payload_bytes, kPoisonByte);
    return ds4_gpu_tensor_write(tensor->view, 0u, poison.data(), poison.size()) != 0;
}

bool guarded_check(const guarded_tensor *tensor) {
    uint8_t prefix[kGuardBytes];
    uint8_t suffix[kGuardBytes];
    const bool have_prefix = tensor->prefix_bytes != 0u;
    if ((have_prefix &&
         !ds4_gpu_tensor_read(tensor->base,
                              tensor->prefix_bytes - kGuardBytes,
                              prefix, sizeof(prefix))) ||
        !ds4_gpu_tensor_read(tensor->base,
                             tensor->prefix_bytes + tensor->payload_bytes,
                             suffix, sizeof(suffix))) {
        std::fprintf(stderr, "FAIL: cannot read %s canary\n", tensor->name);
        return false;
    }
    for (uint32_t i = 0; i < kGuardBytes; i++) {
        if ((have_prefix && prefix[i] != kGuardByte) ||
            suffix[i] != kGuardByte) {
            std::fprintf(stderr,
                         "FAIL: %s canary changed at %s byte %u\n",
                         tensor->name,
                         have_prefix && prefix[i] != kGuardByte
                             ? "prefix" : "suffix",
                         i);
            return false;
        }
    }
    return true;
}

bool tensors_init_rows(tensors *t, uint32_t rows) {
    if (rows == 0u) return false;
    const uint64_t pairs = (uint64_t)rows * kTop;
    const uint64_t pair_mid = pairs * kMid;
    const uint64_t pair_out = pairs * kOut;
    return guarded_init(&t->x, "x", (uint64_t)rows * kIn * sizeof(float)) &&
           guarded_init(&t->selected, "selected", pairs * sizeof(int32_t)) &&
           guarded_init(&t->weights, "weights", pairs * sizeof(float)) &&
           guarded_init(&t->gate, "gate/xq-midq", pair_mid * sizeof(float)) &&
           guarded_init(&t->up, "up", pair_mid * sizeof(float)) &&
           guarded_init(&t->mid, "mid", pair_mid * sizeof(float)) &&
           guarded_init(&t->down, "down/xq", pair_out * sizeof(float)) &&
           guarded_init(&t->out, "out", (uint64_t)rows * kOut * sizeof(float));
}

bool tensors_init(tensors *t) {
    return tensors_init_rows(t, 1u);
}

void tensors_free(tensors *t) {
    guarded_free(&t->out);
    guarded_free(&t->down);
    guarded_free(&t->mid);
    guarded_free(&t->up);
    guarded_free(&t->gate);
    guarded_free(&t->weights);
    guarded_free(&t->selected);
    guarded_free(&t->x);
}

bool tensors_check_guards(const tensors *t) {
    return guarded_check(&t->x) && guarded_check(&t->selected) &&
           guarded_check(&t->weights) && guarded_check(&t->gate) &&
           guarded_check(&t->up) && guarded_check(&t->mid) &&
           guarded_check(&t->down) && guarded_check(&t->out);
}

bool reset_outputs(tensors *t) {
    return guarded_poison_payload(&t->gate) &&
           guarded_poison_payload(&t->up) &&
           guarded_poison_payload(&t->mid) &&
           guarded_poison_payload(&t->down) &&
           guarded_poison_payload(&t->out);
}

void fill_iq2(block_iq2_xxs_test *blocks, uint64_t count, uint32_t seed) {
    uint32_t state = seed;
    for (uint64_t i = 0; i < count; i++) {
        blocks[i].d = 0x1400u; /* 2^-10: enough range without clamp saturation. */
        for (uint32_t q = 0; q < kQk / 8u; q++) {
            blocks[i].qs[q] = (uint16_t)prng_next(&state);
        }
    }
}

void fill_q2(block_q2_k_test *blocks, uint64_t count, uint32_t seed) {
    uint32_t state = seed;
    for (uint64_t i = 0; i < count; i++) {
        for (uint32_t s = 0; s < kQk / 16u; s++) {
            const uint8_t scale = (uint8_t)(1u + (prng_next(&state) % 7u));
            const uint8_t minv = (uint8_t)(prng_next(&state) % 4u);
            blocks[i].scales[s] = (uint8_t)(scale | (uint8_t)(minv << 4u));
        }
        for (uint32_t q = 0; q < kQk / 4u; q++) {
            blocks[i].qs[q] = (uint8_t)prng_next(&state);
        }
        blocks[i].d = 0x1000u;
        blocks[i].dmin = 0x0c00u;
    }
}

void fill_router_f16(uint16_t *weights, uint64_t count) {
    uint32_t state = 0x7f4a7c15u;
    for (uint64_t i = 0; i < count; i++) {
        const uint16_t sign = (prng_next(&state) & 1u) ? 0x8000u : 0u;
        const uint16_t exponent =
            (uint16_t)(10u + prng_next(&state) % 5u);
        const uint16_t mantissa =
            (uint16_t)(1u + prng_next(&state) % 1023u);
        weights[i] = (uint16_t)(sign | (uint16_t)(exponent << 10u) |
                                mantissa);
    }
}

bool model_init(model_fixture *m) {
    m->gate_row_bytes = (kIn / kQk) * sizeof(block_iq2_xxs_test);
    m->gate_expert_bytes = (uint64_t)kMid * m->gate_row_bytes;
    m->down_row_bytes = (kMid / kQk) * sizeof(block_q2_k_test);
    m->down_expert_bytes = (uint64_t)kOut * m->down_row_bytes;
    const uint64_t gate_total = kTotalExperts * m->gate_expert_bytes;
    const uint64_t down_total = kTotalExperts * m->down_expert_bytes;
    m->gate_offset = 4096u;
    m->up_offset = align_up(m->gate_offset + gate_total, 4096u);
    m->down_offset = align_up(m->up_offset + gate_total, 4096u);
    m->router_offset = align_up(m->down_offset + down_total, 4096u);
    const uint64_t router_bytes =
        (uint64_t)kIn * kRouterOut * sizeof(uint16_t);
    m->bytes = align_up(m->router_offset + router_bytes, 4096u);
    void *map = mmap(nullptr, (size_t)m->bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        std::perror("mmap synthetic model");
        return false;
    }
    m->map = static_cast<uint8_t *>(map);
    fill_iq2(reinterpret_cast<block_iq2_xxs_test *>(m->map + m->gate_offset),
             gate_total / sizeof(block_iq2_xxs_test), 0x13579bdfu);
    fill_iq2(reinterpret_cast<block_iq2_xxs_test *>(m->map + m->up_offset),
             gate_total / sizeof(block_iq2_xxs_test), 0x2468ace1u);
    fill_q2(reinterpret_cast<block_q2_k_test *>(m->map + m->down_offset),
            down_total / sizeof(block_q2_k_test), 0x51a7f00du);
    fill_router_f16(
        reinterpret_cast<uint16_t *>(m->map + m->router_offset),
        (uint64_t)kIn * kRouterOut);
    return true;
}

void model_free(model_fixture *m) {
    if (m->map) (void)munmap(m->map, (size_t)m->bytes);
    m->map = nullptr;
}

void fill_input(float *x) {
    for (uint32_t b = 0; b < kIn / kQk; b++) {
        for (uint32_t i = 0; i < kQk; i++) {
            const int value = (int)((b * 71u + i * 29u + 11u) % 101u) - 50;
            x[(uint64_t)b * kQk + i] = (float)value;
        }
        /* A unique, exactly representable maximum makes the Q8_K oracle's
         * scale exactly one on both host and device. */
        x[(uint64_t)b * kQk + ((b * 13u + 17u) & 255u)] = -127.0f;
    }
}

void fill_router_input(float *x, uint32_t rows) {
    uint32_t state = 0xa341316cu ^ (rows * 0x9e3779b9u);
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t i = 0; i < kIn; i++) {
            const int32_t numerator =
                (int32_t)(prng_next(&state) % 2001u) - 1000;
            x[(uint64_t)row * kIn + i] =
                (float)numerator * (0.125f / 997.0f);
        }
        x[(uint64_t)row * kIn] = (float)(row + 1u) / 1009.0f;
    }
}

void quantize_q8_k_oracle(const float *x, uint32_t rows, uint32_t width,
                          block_q8_k_test *out) {
    const uint32_t blocks = width / kQk;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t b = 0; b < blocks; b++) {
            const float *src = x + (uint64_t)row * width + (uint64_t)b * kQk;
            float abs_part[kQk];
            float val_part[kQk];
            for (uint32_t i = 0; i < kQk; i++) {
                abs_part[i] = std::fabs(src[i]);
                val_part[i] = src[i];
            }
            for (uint32_t stride = kQk / 2u; stride > 0u; stride >>= 1u) {
                for (uint32_t i = 0; i < stride; i++) {
                    if (abs_part[i + stride] > abs_part[i]) {
                        abs_part[i] = abs_part[i + stride];
                        val_part[i] = val_part[i + stride];
                    }
                }
            }
            block_q8_k_test *dst = out + (uint64_t)row * blocks + b;
            if (abs_part[0] == 0.0f) {
                std::memset(dst, 0, sizeof(*dst));
                continue;
            }
            const float iscale = -127.0f / val_part[0];
            for (uint32_t i = 0; i < kQk; i++) {
                int q = (int)std::lrint(iscale * src[i]);
                q = std::max(-128, std::min(127, q));
                dst->qs[i] = (int8_t)q;
            }
            for (uint32_t group = 0; group < kQk / 16u; group++) {
                int sum = 0;
                for (uint32_t i = 0; i < 16u; i++) {
                    sum += dst->qs[group * 16u + i];
                }
                dst->bsums[group] = (int16_t)sum;
            }
            dst->d = 1.0f / iscale;
        }
    }
}

bool all_finite(const std::vector<float> &values, const char *what) {
    for (size_t i = 0; i < values.size(); i++) {
        uint32_t bits = 0;
        std::memcpy(&bits, &values[i], sizeof(bits));
        if (bits == 0xa5a5a5a5u) {
            std::fprintf(stderr, "FAIL: %s unwritten poison at %zu\n", what, i);
            return false;
        }
        if (!std::isfinite(values[i])) {
            std::fprintf(stderr, "FAIL: %s non-finite at %zu\n", what, i);
            return false;
        }
    }
    return true;
}

bool q8_blocks_valid(const std::vector<uint8_t> &bytes, const char *what) {
    if (bytes.empty() || bytes.size() % sizeof(block_q8_k_test) != 0u) {
        std::fprintf(stderr, "FAIL: %s invalid Q8_K byte count %zu\n",
                     what, bytes.size());
        return false;
    }
    for (size_t offset = 0; offset < bytes.size();
         offset += sizeof(block_q8_k_test)) {
        float d = 0.0f;
        uint32_t bits = 0;
        std::memcpy(&d, bytes.data() + offset, sizeof(d));
        std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
        if (!std::isfinite(d) || bits == 0xa5a5a5a5u) {
            std::fprintf(stderr, "FAIL: %s invalid Q8_K scale block=%zu\n",
                         what, offset / sizeof(block_q8_k_test));
            return false;
        }
    }
    return true;
}

bool bytes_equal(const char *what, const uint8_t *a, const uint8_t *b,
                 size_t bytes) {
    if (std::memcmp(a, b, bytes) == 0) return true;
    size_t first = 0;
    while (first < bytes && a[first] == b[first]) first++;
    std::fprintf(stderr,
                 "FAIL: %s byte mismatch at %zu: 0x%02x != 0x%02x\n",
                 what, first, (unsigned)a[first], (unsigned)b[first]);
    return false;
}

float add_rn_host(float a, float b) {
    volatile float rounded = a + b;
    return rounded;
}

bool launch_moe(const model_fixture &m, tensors *t,
                const int32_t *ids, const float *weights, uint32_t n_expert) {
    if (!ds4_gpu_tensor_write(t->selected.view, 0u, ids,
                              (uint64_t)n_expert * sizeof(*ids)) ||
        !ds4_gpu_tensor_write(t->weights.view, 0u, weights,
                              (uint64_t)n_expert * sizeof(*weights))) {
        return false;
    }
    return ds4_gpu_routed_moe_one_tensor(
        t->out.view, t->gate.view, t->up.view, t->mid.view, t->down.view,
        m.map, m.bytes, m.gate_offset, m.up_offset, m.down_offset,
        kIq2Type, kQ2Type, m.gate_expert_bytes, m.gate_row_bytes,
        m.down_expert_bytes, m.down_row_bytes, kIn, kMid, kOut,
        t->selected.view, t->weights.view, kTotalExperts, n_expert,
        kClamp, t->x.view, nullptr, 0u, true) != 0;
}

bool launch_moe_batch(const model_fixture &m, tensors *t,
                      const int32_t *ids, const float *weights,
                      uint32_t rows) {
    const uint64_t pairs = (uint64_t)rows * kTop;
    if (!ds4_gpu_tensor_write(t->selected.view, 0u, ids,
                              pairs * sizeof(*ids)) ||
        !ds4_gpu_tensor_write(t->weights.view, 0u, weights,
                              pairs * sizeof(*weights))) {
        return false;
    }
    bool mid_is_f16 = true;
    const bool ok = ds4_gpu_routed_moe_batch_tensor(
        t->out.view, t->gate.view, t->up.view, t->mid.view, t->down.view,
        m.map, m.bytes, m.gate_offset, m.up_offset, m.down_offset,
        kIq2Type, kQ2Type, m.gate_expert_bytes, m.gate_row_bytes,
        m.down_expert_bytes, m.down_row_bytes, kIn, kMid, kOut,
        t->selected.view, t->weights.view, kTotalExperts, kTop,
        kClamp, t->x.view, 0u, rows, &mid_is_f16, true) != 0;
    return ok && !mid_is_f16;
}

bool read_capture(tensors *t, uint32_t n_expert, capture *result) {
    const uint64_t xq_bytes = (kIn / kQk) * sizeof(block_q8_k_test);
    const uint64_t midq_bytes =
        (uint64_t)n_expert * (kMid / kQk) * sizeof(block_q8_k_test);
    if (n_expert == 3u || n_expert == 6u) {
        result->xq.resize((size_t)xq_bytes);
        if (!ds4_gpu_tensor_read(t->down.view, 0u, result->xq.data(), xq_bytes)) {
            return false;
        }
    }
    result->midq.resize((size_t)midq_bytes);
    result->mid.resize((size_t)n_expert * kMid);
    result->out.resize(kOut);
    if (!ds4_gpu_tensor_read(t->gate.view, 0u, result->midq.data(), midq_bytes) ||
        !ds4_gpu_tensor_read(t->mid.view, 0u, result->mid.data(),
                             result->mid.size() * sizeof(float)) ||
        !ds4_gpu_tensor_read(t->out.view, 0u, result->out.data(),
                             result->out.size() * sizeof(float))) {
        return false;
    }
    return (result->xq.empty() || q8_blocks_valid(result->xq, "xq")) &&
           q8_blocks_valid(result->midq, "midq") &&
           all_finite(result->mid, "mid") && all_finite(result->out, "out") &&
           tensors_check_guards(t);
}

bool run_capture(const model_fixture &m, tensors *t,
                 const int32_t *ids, const float *weights, uint32_t n_expert,
                 capture *result) {
    if (!reset_outputs(t) || !launch_moe(m, t, ids, weights, n_expert) ||
        !ds4_gpu_synchronize()) {
        std::fprintf(stderr, "FAIL: routed MoE call n_expert=%u\n", n_expert);
        return false;
    }
    return read_capture(t, n_expert, result);
}

int launch_moe_stream_slot_offset(const model_fixture &m, tensors *t,
                                  const int32_t ids[kTop],
                                  const float weights[kTop],
                                  uint64_t stream_slot_offset,
                                  bool force_resident) {
    if (!ds4_gpu_tensor_write(t->selected.view, 0u, ids,
                              kTop * sizeof(*ids)) ||
        !ds4_gpu_tensor_write(t->weights.view, 0u, weights,
                              kTop * sizeof(*weights))) {
        return 0;
    }
    return ds4_gpu_routed_moe_one_tensor_stream_slot_offset(
        t->out.view, t->gate.view, t->up.view, t->mid.view, t->down.view,
        m.map, m.bytes, m.gate_offset, m.up_offset, m.down_offset,
        kIq2Type, kQ2Type, m.gate_expert_bytes, m.gate_row_bytes,
        m.down_expert_bytes, m.down_row_bytes, kIn, kMid, kOut,
        t->selected.view, t->weights.view, kTotalExperts, kTop,
        kClamp, t->x.view, stream_slot_offset, nullptr, 0u,
        force_resident) != 0;
}

bool run_stream_capture(const model_fixture &m, tensors *t,
                        const int32_t ids[kTop],
                        const float weights[kTop],
                        uint64_t stream_slot_offset,
                        capture *result) {
    if (!reset_outputs(t) ||
        !launch_moe_stream_slot_offset(m, t, ids, weights,
                                       stream_slot_offset, false) ||
        !ds4_gpu_synchronize()) {
        std::fprintf(stderr,
                     "FAIL: streaming routed MoE slot offset=%llu\n",
                     (unsigned long long)stream_slot_offset);
        return false;
    }
    return read_capture(t, kTop, result);
}

bool check_case(const char *name, const model_fixture &m, tensors *t,
                const int32_t ids[kTop], const float weights[kTop],
                const std::vector<uint8_t> &xq_oracle,
                correctness_result *summary) {
    capture top6;
    capture first3;
    capture last3;
    capture top1[kTop];
    if (!run_capture(m, t, ids, weights, 6u, &top6) ||
        !run_capture(m, t, ids, weights, 3u, &first3) ||
        !run_capture(m, t, ids + 3u, weights + 3u, 3u, &last3)) {
        return false;
    }
    summary->calls += 3u;
    for (uint32_t slot = 0; slot < kTop; slot++) {
        if (!run_capture(m, t, ids + slot, weights + slot, 1u, &top1[slot])) {
            return false;
        }
        summary->calls++;
    }

    bool ok = true;
    ok = bytes_equal("xq top6/CPU", top6.xq.data(), xq_oracle.data(),
                     xq_oracle.size()) && ok;
    ok = bytes_equal("xq top6/first3", top6.xq.data(), first3.xq.data(),
                     top6.xq.size()) && ok;
    ok = bytes_equal("xq top6/last3", top6.xq.data(), last3.xq.data(),
                     top6.xq.size()) && ok;
    summary->byte_checks += 3u;

    const size_t slot_midq = (kMid / kQk) * sizeof(block_q8_k_test);
    ok = bytes_equal("midq top6/first3", top6.midq.data(), first3.midq.data(),
                     3u * slot_midq) && ok;
    ok = bytes_equal("midq top6/last3", top6.midq.data() + 3u * slot_midq,
                     last3.midq.data(), 3u * slot_midq) && ok;
    summary->byte_checks += 2u;
    for (uint32_t slot = 0; slot < kTop; slot++) {
        ok = bytes_equal("midq top6/top1",
                         top6.midq.data() + (size_t)slot * slot_midq,
                         top1[slot].midq.data(), slot_midq) && ok;
        summary->byte_checks++;
    }

    uint32_t mismatches = 0u;
    for (uint32_t row = 0; row < kOut; row++) {
        float sum_first = 0.0f;
        float sum_last = 0.0f;
        for (uint32_t slot = 0; slot < 3u; slot++) {
            sum_first = add_rn_host(sum_first, top1[slot].out[row]);
            sum_last = add_rn_host(sum_last, top1[slot + 3u].out[row]);
        }
        float sum_six = sum_first;
        for (uint32_t slot = 3u; slot < 6u; slot++) {
            sum_six = add_rn_host(sum_six, top1[slot].out[row]);
        }
        uint32_t got_first = 0, got_last = 0, got_six = 0;
        uint32_t want_first = 0, want_last = 0, want_six = 0;
        std::memcpy(&got_first, &first3.out[row], sizeof(got_first));
        std::memcpy(&got_last, &last3.out[row], sizeof(got_last));
        std::memcpy(&got_six, &top6.out[row], sizeof(got_six));
        std::memcpy(&want_first, &sum_first, sizeof(want_first));
        std::memcpy(&want_last, &sum_last, sizeof(want_last));
        std::memcpy(&want_six, &sum_six, sizeof(want_six));
        if (got_first != want_first || got_last != want_last ||
            got_six != want_six) {
            if (mismatches < 8u) {
                std::fprintf(stderr,
                             "FAIL: %s output row=%u top3a=%08x/%08x "
                             "top3b=%08x/%08x top6=%08x/%08x\n",
                             name, row, got_first, want_first, got_last,
                             want_last, got_six, want_six);
            }
            mismatches++;
        }
    }
    if (mismatches) {
        std::fprintf(stderr, "FAIL: %s output byte mismatches=%u/%u\n",
                     name, mismatches, kOut);
        ok = false;
    }
    summary->output_rows += kOut;
    summary->cases++;
    std::fprintf(stderr,
                 "cuda-flash-moe-decode correctness case=%s: %s\n",
                 name, ok ? "PASS" : "FAIL");
    return ok;
}

correctness_result run_correctness(const model_fixture &m, tensors *t,
                                   const std::vector<uint8_t> &xq_oracle) {
    static const int32_t unique_ids[kTop] = {31, 2, 19, 7, 28, 0};
    static const int32_t duplicate_ids[kTop] = {11, 11, 3, 25, 3, 11};
    static const float weights[kTop] = {
        0.375f, -0.21875f, 0.15625f, 0.296875f, -0.125f, 0.203125f,
    };
    correctness_result result;
    result.ok = check_case("unique-out-of-order", m, t, unique_ids, weights,
                           xq_oracle, &result) && result.ok;
    result.ok = check_case("duplicates", m, t, duplicate_ids, weights,
                           xq_oracle, &result) && result.ok;
    return result;
}

bool run_benchmark(const model_fixture &m, tensors *t,
                   double *median_ms, double *p95_ms) {
    static const int32_t ids[kTop] = {31, 2, 19, 7, 28, 0};
    static const float weights[kTop] = {
        0.375f, -0.21875f, 0.15625f, 0.296875f, -0.125f, 0.203125f,
    };
    if (!ds4_gpu_tensor_write(t->selected.view, 0u, ids, sizeof(ids)) ||
        !ds4_gpu_tensor_write(t->weights.view, 0u, weights, sizeof(weights))) {
        return false;
    }
    for (uint32_t i = 0; i < 100u; i++) {
        if (!launch_moe(m, t, ids, weights, kTop)) return false;
    }
    if (!ds4_gpu_synchronize()) return false;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (!cuda_ok(cudaEventCreate(&start), "create start event") ||
        !cuda_ok(cudaEventCreate(&stop), "create stop event")) {
        if (start) (void)cudaEventDestroy(start);
        if (stop) (void)cudaEventDestroy(stop);
        return false;
    }
    std::vector<double> block_ms(100u);
    bool ok = true;
    for (uint32_t block = 0; block < 100u && ok; block++) {
        ok = cuda_ok(cudaEventRecord(start, 0), "record block start");
        for (uint32_t iter = 0; iter < 10u && ok; iter++) {
            ok = launch_moe(m, t, ids, weights, kTop);
        }
        ok = ok && cuda_ok(cudaEventRecord(stop, 0), "record block stop") &&
             cuda_ok(cudaEventSynchronize(stop), "synchronize block stop");
        float elapsed = 0.0f;
        ok = ok && cuda_ok(cudaEventElapsedTime(&elapsed, start, stop),
                           "measure block");
        block_ms[block] = (double)elapsed / 10.0;
    }
    (void)cudaEventDestroy(stop);
    (void)cudaEventDestroy(start);
    if (!ok || !tensors_check_guards(t)) return false;
    std::sort(block_ms.begin(), block_ms.end());
    *median_ms = block_ms[49u];
    *p95_ms = block_ms[94u];
    return true;
}

bool graph_env_enabled() {
    const char *value = std::getenv("DS4_CUDA_FLASH_IQ2_DECODE_GRAPH");
    return value && value[0] == '1' && value[1] == '\0';
}

bool gpu_fixture_init(const model_fixture &model,
                      const std::vector<float> &input,
                      tensors *gpu) {
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_map(model.map, model.bytes) ||
        !tensors_init(gpu) ||
        !ds4_gpu_tensor_write(gpu->x.view, 0u, input.data(),
                              input.size() * sizeof(float))) {
        return false;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    return true;
}

router_rows_result run_router_rows_exact(const model_fixture &model) {
    static const uint32_t row_counts[] = {
        1u, 2u, 3u, 4u, 5u, 8u, 12u, 16u, 20u, 32u, 64u,
    };
    static const uint64_t view_offsets[] = {0u, 64u, 256u};
    router_rows_result result;
    if (!ds4_gpu_init() ||
        !ds4_gpu_set_model_map(model.map, model.bytes)) {
        std::fprintf(stderr, "FAIL: router rows GPU fixture initialization\n");
        result.ok = false;
        ds4_gpu_cleanup();
        return result;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    for (uint32_t n_rows : row_counts) {
      for (uint64_t view_offset : view_offsets) {
        const uint64_t x_bytes =
            (uint64_t)n_rows * kIn * sizeof(float);
        const uint64_t out_bytes =
            (uint64_t)n_rows * kRouterOut * sizeof(float);
        guarded_tensor x;
        guarded_tensor rowwise;
        guarded_tensor batched;
        std::vector<float> input((uint64_t)n_rows * kIn);
        std::vector<float> expected((uint64_t)n_rows * kRouterOut);
        std::vector<float> actual((uint64_t)n_rows * kRouterOut);
        fill_router_input(input.data(), n_rows);

        bool ok = guarded_init_offset(
                      &x, "router-x", x_bytes, view_offset) &&
                  guarded_init_offset(
                      &rowwise, "router-rowwise", out_bytes, view_offset) &&
                  guarded_init_offset(
                      &batched, "router-batched", out_bytes, view_offset) &&
                  guarded_poison_payload(&rowwise) &&
                  guarded_poison_payload(&batched) &&
                  ds4_gpu_tensor_write(
                      x.view, 0u, input.data(), x_bytes) != 0;
        for (uint32_t row = 0; ok && row < n_rows; row++) {
            ds4_gpu_tensor *x_row = ds4_gpu_tensor_view(
                x.view, (uint64_t)row * kIn * sizeof(float),
                (uint64_t)kIn * sizeof(float));
            ds4_gpu_tensor *out_row = ds4_gpu_tensor_view(
                rowwise.view,
                (uint64_t)row * kRouterOut * sizeof(float),
                (uint64_t)kRouterOut * sizeof(float));
            ok = x_row && out_row &&
                 ds4_gpu_matmul_f16_tensor(
                     out_row, model.map, model.bytes,
                     model.router_offset, kIn, kRouterOut, x_row, 1u) != 0;
            ds4_gpu_tensor_free(out_row);
            ds4_gpu_tensor_free(x_row);
        }
        if (ok) {
            ok = ds4_gpu_matmul_f16_router_rows_exact_tensor(
                     batched.view, model.map, model.bytes,
                     model.router_offset, x.view, n_rows) != 0 &&
                 ds4_gpu_synchronize() != 0 &&
                 ds4_gpu_tensor_read(
                     rowwise.view, 0u, expected.data(), out_bytes) != 0 &&
                 ds4_gpu_tensor_read(
                     batched.view, 0u, actual.data(), out_bytes) != 0;
        }
        if (ok) {
            char label[96];
            std::snprintf(label, sizeof(label),
                          "router rows exact n=%u offset=%llu", n_rows,
                          (unsigned long long)view_offset);
            ok = bytes_equal(
                label,
                reinterpret_cast<const uint8_t *>(expected.data()),
                reinterpret_cast<const uint8_t *>(actual.data()),
                out_bytes);
            result.byte_checks++;
        }
        const bool canaries_ok =
            x.base && rowwise.base && batched.base &&
            guarded_check(&x) && guarded_check(&rowwise) &&
            guarded_check(&batched);
        result.canary_checks += 3u;
        result.cases++;
        result.rows += n_rows;
        result.ok = result.ok && ok && canaries_ok;
        guarded_free(&batched);
        guarded_free(&rowwise);
        guarded_free(&x);
        if (!result.ok) break;
      }
      if (!result.ok) break;
    }
    ds4_gpu_cleanup();
    return result;
}

bool tuple_addresses_differ(const tensors &a, const tensors &b) {
    /* The public tensor is opaque, but both live base allocations have
     * non-zero, disjoint payloads; distinct handles therefore imply distinct
     * CUDA allocation (and view) addresses for every baked graph argument. */
    return a.x.base && b.x.base &&
        a.x.base != b.x.base &&
        a.selected.base != b.selected.base &&
        a.weights.base != b.weights.base &&
        a.gate.base != b.gate.base &&
        a.up.base != b.up.base &&
        a.mid.base != b.mid.base &&
        a.down.base != b.down.base &&
        a.out.base != b.out.base;
}

bool capture_payload_equal(const char *case_name,
                           const char *comparison,
                           const capture &reference,
                           const capture &actual,
                           uint32_t *byte_checks) {
    if (reference.xq.size() != actual.xq.size() ||
        reference.midq.size() != actual.midq.size() ||
        reference.mid.size() != actual.mid.size() ||
        reference.out.size() != actual.out.size()) {
        std::fprintf(stderr, "FAIL: %s capture sizes differ\n", case_name);
        return false;
    }
    char label[96];
    bool ok = true;
#define CHECK_CAPTURE_BYTES(field) do {                                      \
        std::snprintf(label, sizeof(label), "%s %s " #field,                \
                      case_name, comparison);                                 \
        ok = bytes_equal(label,                                               \
                         reinterpret_cast<const uint8_t *>(reference.field.data()),\
                         reinterpret_cast<const uint8_t *>(actual.field.data()),\
                         reference.field.size() * sizeof(reference.field[0])) && ok;\
        (*byte_checks)++;                                                      \
    } while (0)
    CHECK_CAPTURE_BYTES(xq);
    CHECK_CAPTURE_BYTES(midq);
    CHECK_CAPTURE_BYTES(mid);
    CHECK_CAPTURE_BYTES(out);
#undef CHECK_CAPTURE_BYTES
    return ok;
}

bool aligned_capture_equal(const char *case_name,
                           const capture &reference,
                           const capture &actual,
                           uint32_t *byte_checks) {
    return capture_payload_equal(
        case_name, "resident/streamed-aligned-cache",
        reference, actual, byte_checks);
}

bool capture_equal(const char *case_name,
                   const capture &eager,
                   const capture &graph,
                   graph_identity_result *result) {
    const bool ok = capture_payload_equal(
        case_name, "eager/graph", eager, graph, &result->byte_checks);
    result->output_rows += kOut;
    return ok;
}

direct_midq_result run_direct_midq_identity(const model_fixture &model,
                                            const std::vector<float> &input) {
    static const int32_t ids[2][kTop] = {
        {31, 2, 19, 7, 28, 0},
        {11, 11, 3, 25, 3, 11},
    };
    static const float weights[2][kTop] = {
        {0.375f, -0.21875f, 0.15625f, 0.296875f, -0.125f, 0.203125f},
        {-0.171875f, 0.34375f, 0.109375f, -0.265625f, 0.1875f, 0.234375f},
    };
    static const char *case_names[2] = {
        "direct-midq-unique",
        "direct-midq-duplicates",
    };
    direct_midq_result result;
    tensors gpu;
    (void)unsetenv("DS4_CUDA_FLASH_IQ2_DECODE_GRAPH");
    (void)unsetenv("DS4_CUDA_FLASH_IQ2_DIRECT_MIDQ");
    (void)unsetenv("DS4_CUDA_NO_FLASH_IQ2_DIRECT_MIDQ");
    (void)unsetenv("DS4_CUDA_MOE_WRITE_GATE_UP");
    bool ok = gpu_fixture_init(model, input, &gpu);
    for (uint32_t i = 0; ok && i < 2u; i++) {
        capture baseline;
        capture fused;
        capture rollback;
        bool case_ok = run_capture(
            model, &gpu, ids[i], weights[i], kTop, &baseline);
        case_ok = case_ok &&
            setenv("DS4_CUDA_FLASH_IQ2_DIRECT_MIDQ", "1", 1) == 0 &&
            run_capture(model, &gpu, ids[i], weights[i], kTop, &fused);
        case_ok = case_ok &&
            setenv("DS4_CUDA_NO_FLASH_IQ2_DIRECT_MIDQ", "1", 1) == 0 &&
            run_capture(model, &gpu, ids[i], weights[i], kTop, &rollback);
        if (case_ok) {
            case_ok = capture_payload_equal(
                case_names[i], "baseline/fused", baseline, fused,
                &result.byte_checks) && case_ok;
            case_ok = capture_payload_equal(
                case_names[i], "baseline/rollback", baseline, rollback,
                &result.byte_checks) && case_ok;
        }
        result.cases++;
        result.guarded_captures += 3u;
        result.ok = result.ok && case_ok;
        ok = case_ok;
        (void)unsetenv("DS4_CUDA_FLASH_IQ2_DIRECT_MIDQ");
        (void)unsetenv("DS4_CUDA_NO_FLASH_IQ2_DIRECT_MIDQ");
    }
    (void)unsetenv("DS4_CUDA_FLASH_IQ2_DIRECT_MIDQ");
    (void)unsetenv("DS4_CUDA_NO_FLASH_IQ2_DIRECT_MIDQ");
    tensors_free(&gpu);
    ds4_gpu_cleanup();
    result.ok = result.ok && ok;
    return result;
}

moe_batch_exact_result run_moe_batch_exact_identity(
        const model_fixture &model) {
    constexpr uint32_t rows = 128u;
    const uint64_t pairs = (uint64_t)rows * kTop;
    const uint64_t down_values = pairs * kOut;
    const uint64_t out_values = (uint64_t)rows * kOut;
    moe_batch_exact_result result;
    tensors gpu;
    std::vector<float> input((size_t)rows * kIn);
    std::vector<int32_t> ids((size_t)pairs);
    std::vector<float> weights((size_t)pairs);
    for (uint32_t row = 0; row < rows; row++) {
        fill_input(input.data() + (uint64_t)row * kIn);
        input[(uint64_t)row * kIn + (row * 37u) % kIn] +=
            (float)(row + 1u) / 509.0f;
        const uint32_t base = (row * 7u) % kTotalExperts;
        for (uint32_t slot = 0; slot < kTop; slot++) {
            const uint64_t pair = (uint64_t)row * kTop + slot;
            ids[pair] = (int32_t)((base + slot * 5u) % kTotalExperts);
            weights[pair] =
                (float)(slot + 1u) / (float)(37u + row % 5u);
        }
        if (row & 1u) ids[(uint64_t)row * kTop + 4u] = ids[(uint64_t)row * kTop];
    }

    (void)unsetenv("DS4_CUDA_MOE_BATCH_DECODE_EXACT");
    (void)unsetenv("DS4_CUDA_MOE_ATOMIC_DOWN");
    (void)setenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN", "1", 1);
    /* Keep this fixture on the same raw IQ2/Q2 sorted-pair path whose
     * n>=128 atomic-down policy is controlled by the exact-value switch. */
    (void)setenv("DS4_CUDA_MMQ", "0", 1);

    bool ok = ds4_gpu_init() &&
        ds4_gpu_set_model_map(model.map, model.bytes) &&
        tensors_init_rows(&gpu, rows) &&
        ds4_gpu_tensor_write(gpu.x.view, 0u, input.data(),
                             input.size() * sizeof(float));
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);

    std::vector<float> reference_down((size_t)down_values);
    std::vector<float> reference_out((size_t)out_values);
    std::vector<float> exact_down((size_t)down_values);
    std::vector<float> exact_out((size_t)out_values);
    if (ok) {
        ok = reset_outputs(&gpu) &&
            launch_moe_batch(model, &gpu, ids.data(), weights.data(), rows) &&
            ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(gpu.down.view, 0u, reference_down.data(),
                                down_values * sizeof(float)) &&
            ds4_gpu_tensor_read(gpu.out.view, 0u, reference_out.data(),
                                out_values * sizeof(float));
    }
    if (ok) {
        ok = tensors_check_guards(&gpu);
        result.canary_checks += 8u;
    }

    /* The exact flag must win even over an inherited explicit request for the
     * non-canonical atomic accumulation. */
    if (ok) {
        ok = unsetenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN") == 0 &&
            setenv("DS4_CUDA_MOE_ATOMIC_DOWN", "1", 1) == 0 &&
            setenv("DS4_CUDA_MOE_BATCH_DECODE_EXACT", "1", 1) == 0 &&
            reset_outputs(&gpu) &&
            launch_moe_batch(model, &gpu, ids.data(), weights.data(), rows) &&
            ds4_gpu_synchronize() &&
            ds4_gpu_tensor_read(gpu.down.view, 0u, exact_down.data(),
                                down_values * sizeof(float)) &&
            ds4_gpu_tensor_read(gpu.out.view, 0u, exact_out.data(),
                                out_values * sizeof(float));
    }
    if (ok) {
        ok = tensors_check_guards(&gpu);
        result.canary_checks += 8u;
    }
    if (ok) {
        ok = bytes_equal(
                 "MoE batch exact canonical per-slot down",
                 reinterpret_cast<const uint8_t *>(reference_down.data()),
                 reinterpret_cast<const uint8_t *>(exact_down.data()),
                 down_values * sizeof(float)) && ok;
        result.byte_checks++;
        ok = bytes_equal(
                 "MoE batch exact canonical sum",
                 reinterpret_cast<const uint8_t *>(reference_out.data()),
                 reinterpret_cast<const uint8_t *>(exact_out.data()),
                 out_values * sizeof(float)) && ok;
        result.byte_checks++;
        ok = all_finite(reference_out, "MoE batch exact reference out") &&
             all_finite(exact_out, "MoE batch exact out") && ok;
    }

    (void)unsetenv("DS4_CUDA_MOE_BATCH_DECODE_EXACT");
    (void)unsetenv("DS4_CUDA_MOE_ATOMIC_DOWN");
    (void)unsetenv("DS4_CUDA_MOE_NO_ATOMIC_DOWN");
    tensors_free(&gpu);
    ds4_gpu_cleanup();
    result.cases = 1u;
    result.rows = rows;
    result.ok = ok;
    return result;
}

bool run_stream_slot_offset_test(const model_fixture &model,
                                 tensors *gpu,
                                 stream_offset_result *result) {
    static const int32_t ids[4][kTop] = {
        {31, 2, 19, 7, 28, 0},
        {11, 11, 3, 25, 3, 11},
        {3, 25, 3, 25, 3, 25},
        {31, 2, 31, 2, 31, 2},
    };
    static const float weights[4][kTop] = {
        {0.375f, -0.21875f, 0.15625f, 0.296875f, -0.125f, 0.203125f},
        {-0.171875f, 0.34375f, 0.109375f, -0.265625f, 0.1875f, 0.234375f},
        {0.125f, -0.25f, 0.375f, -0.1875f, 0.21875f, 0.15625f},
        {-0.28125f, 0.1875f, 0.234375f, -0.125f, 0.3125f, 0.09375f},
    };
    capture resident[4];
    capture streamed[4];
    bool ok = true;
    for (uint32_t i = 0; i < 4u; i++) {
        ok = run_capture(model, gpu, ids[i], weights[i], kTop,
                         &resident[i]) && ok;
    }

    ds4_gpu_stream_expert_table table = {};
    table.model_map = model.map;
    table.model_size = model.bytes;
    table.layer = 0u;
    table.n_total_expert = kTotalExperts;
    table.gate_offset = model.gate_offset;
    table.up_offset = model.up_offset;
    table.down_offset = model.down_offset;
    table.gate_expert_bytes = model.gate_expert_bytes;
    table.down_expert_bytes = model.down_expert_bytes;
    table.gate_type = kIq2Type;
    table.down_type = kQ2Type;
    table.expert_in_dim = kIn;
    table.expert_mid_dim = kMid;
    table.out_dim = kOut;
    ds4_gpu_set_streaming_expert_cache_budget(4u);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(
        model.gate_expert_bytes * 2u + model.down_expert_bytes);
    ds4_gpu_stream_expert_cache_stats before = {};
    ds4_gpu_stream_expert_cache_stats after_batch = {};
    ds4_gpu_stream_expert_cache_stats after_hot = {};
    ds4_gpu_stream_expert_cache_stats after_prefetch = {};
    ds4_gpu_stream_expert_cache_get_stats(&before);
    ds4_gpu_set_ssd_streaming(true);
    if (ok) {
        ok = ds4_gpu_stream_expert_cache_prepare_selected_batch(
                 &table, &ids[0][0], 2u, kTop) != 0 &&
             run_stream_capture(model, gpu, ids[0], weights[0], 0u,
                                &streamed[0]) &&
             run_stream_capture(model, gpu, ids[1], weights[1], kTop,
                                &streamed[1]);
    }
    ds4_gpu_stream_expert_cache_get_stats(&after_batch);
    if (ok) {
        ok = aligned_capture_equal(
                 "stream-row0", resident[0], streamed[0],
                 &result->byte_checks) && ok;
        ok = aligned_capture_equal(
                 "stream-row1-duplicates", resident[1], streamed[1],
                 &result->byte_checks) && ok;
        result->rows = 2u;
    }

    const bool overflow_rejected =
        !launch_moe_stream_slot_offset(model, gpu, ids[1], weights[1],
                                       UINT64_MAX, false);
    result->negative_checks++;
    /* Twelve selected slots are loaded. Offset seven would require slots
     * [7,13), proving that the bound is offset+top6 rather than top6 alone. */
    const bool range_rejected =
        !launch_moe_stream_slot_offset(model, gpu, ids[1], weights[1],
                                       7u, false);
    result->negative_checks++;
    const bool resident_rejected =
        !launch_moe_stream_slot_offset(model, gpu, ids[1], weights[1],
                                       kTop, true);
    result->negative_checks++;
    if (!overflow_rejected || !range_rejected || !resident_rejected) {
        std::fprintf(stderr,
                     "FAIL: stream offset negatives overflow=%d range=%d "
                     "resident=%d\n",
                     overflow_rejected ? 1 : 0,
                     range_rejected ? 1 : 0,
                     resident_rejected ? 1 : 0);
        ok = false;
    }

    if (ok) {
        ok = ds4_gpu_stream_expert_cache_begin_selected_load(
                 &table, ids[2], kTop) != 0 &&
             run_stream_capture(model, gpu, ids[2], weights[2], 0u,
                                &streamed[2]);
    }
    ds4_gpu_stream_expert_cache_get_stats(&after_hot);
    if (ok) {
        ok = aligned_capture_equal(
                 "stream-hot-hit-duplicates", resident[2], streamed[2],
                 &result->byte_checks) && ok;
        result->rows++;
    }

    if (ok) {
        ds4_gpu_stream_expert_cache_release_resident();
        const int32_t seed_ids[2] = {31, 2};
        ok = ds4_gpu_stream_expert_cache_seed_selected(
                 &table, seed_ids, 2u) != 0 &&
             ds4_gpu_stream_expert_cache_begin_selected_load(
                 &table, ids[3], kTop) != 0 &&
             run_stream_capture(model, gpu, ids[3], weights[3], 0u,
                                &streamed[3]);
    }
    ds4_gpu_stream_expert_cache_get_stats(&after_prefetch);
    if (ok) {
        ok = aligned_capture_equal(
                 "stream-prefetch-hit-duplicates", resident[3], streamed[3],
                 &result->byte_checks) && ok;
        result->rows++;
    }

    const uint64_t batch_misses = after_batch.misses - before.misses;
    const uint64_t batch_evictions = after_batch.evictions - before.evictions;
    const uint64_t hot_hits = after_hot.hits - after_batch.hits;
    const uint64_t prefetch_hits = after_prefetch.hits - after_hot.hits;
    const uint64_t prefetch_admitted =
        after_prefetch.prefetch_admitted - after_hot.prefetch_admitted;
    const bool cache_stats_ok =
        after_batch.capacity == 3u && batch_misses >= 9u &&
        batch_evictions >= 6u && hot_hits >= 2u &&
        prefetch_admitted >= 2u && prefetch_hits >= 2u;
    if (!cache_stats_ok) {
        std::fprintf(stderr,
                     "FAIL: aligned cache lifecycle capacity=%u misses=%llu "
                     "evictions=%llu hot_hits=%llu prefetch=%llu/%llu\n",
                     after_batch.capacity,
                     (unsigned long long)batch_misses,
                     (unsigned long long)batch_evictions,
                     (unsigned long long)hot_hits,
                     (unsigned long long)prefetch_admitted,
                     (unsigned long long)prefetch_hits);
        ok = false;
    }
    result->cache_capacity = after_batch.capacity;
    result->cache_hits = after_prefetch.hits - before.hits;
    result->cache_misses = batch_misses;
    result->cache_evictions = batch_evictions;
    result->prefetch_admitted = prefetch_admitted;
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0u);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0u);
    result->ok = ok;
    return ok;
}

bool run_graph_identity(const model_fixture &model,
                        const std::vector<float> &input,
                        correctness_result *correctness,
                        graph_identity_result *identity,
                        stream_offset_result *stream_offset) {
    static const int32_t unique_ids[kTop] = {31, 2, 19, 7, 28, 0};
    static const int32_t duplicate_ids[kTop] = {11, 11, 3, 25, 3, 11};
    static const float weights[kTop] = {
        0.375f, -0.21875f, 0.15625f, 0.296875f, -0.125f, 0.203125f,
    };
    std::vector<uint8_t> xq_oracle(
        (kIn / kQk) * sizeof(block_q8_k_test));
    quantize_q8_k_oracle(
        input.data(), 1u, kIn,
        reinterpret_cast<block_q8_k_test *>(xq_oracle.data()));

    capture eager_unique;
    capture eager_duplicate;
    tensors eager_gpu;
    (void)unsetenv("DS4_CUDA_FLASH_IQ2_DECODE_GRAPH");
    bool ok = gpu_fixture_init(model, input, &eager_gpu);
    if (ok) {
        *correctness = run_correctness(model, &eager_gpu, xq_oracle);
        ok = correctness->ok &&
            run_capture(model, &eager_gpu, unique_ids, weights, kTop,
                        &eager_unique) &&
            run_capture(model, &eager_gpu, duplicate_ids, weights, kTop,
                        &eager_duplicate) &&
            run_stream_slot_offset_test(model, &eager_gpu, stream_offset);
    }
    tensors_free(&eager_gpu);
    ds4_gpu_cleanup();
    if (!ok) {
        std::fprintf(stderr, "FAIL: eager graph-identity fixture\n");
        return false;
    }

    tensors graph_a;
    tensors graph_b;
    (void)setenv("DS4_CUDA_FLASH_IQ2_DECODE_GRAPH", "1", 1);
    ok = gpu_fixture_init(model, input, &graph_a) &&
         tensors_init(&graph_b) &&
         ds4_gpu_tensor_write(graph_b.x.view, 0u, input.data(),
                              input.size() * sizeof(float));
    identity->pointer_update = ok && tuple_addresses_differ(graph_a, graph_b);
    ds4_gpu_decode_graphs_get_stats(&identity->before);
    capture graph_unique;
    capture graph_duplicate;
    if (ok) {
        ok = identity->pointer_update &&
            run_capture(model, &graph_a, unique_ids, weights, kTop,
                        &graph_unique) &&
            run_capture(model, &graph_b, duplicate_ids, weights, kTop,
                        &graph_duplicate) &&
            ds4_gpu_synchronize();
    }
    ds4_gpu_decode_graphs_get_stats(&identity->after);
    if (ok) {
        ok = capture_equal("unique-out-of-order", eager_unique, graph_unique,
                           identity) && ok;
        ok = capture_equal("duplicates", eager_duplicate, graph_duplicate,
                           identity) && ok;
    }
    const bool stats_ok =
        identity->after.moe_captures ==
            identity->before.moe_captures + 1u &&
        identity->after.moe_replays >= identity->before.moe_replays + 1u &&
        identity->after.moe_failed == identity->before.moe_failed;
    if (!stats_ok) {
        std::fprintf(stderr,
                     "FAIL: graph stats capture=%llu->%llu replay=%llu->%llu "
                     "failed=%llu->%llu\n",
                     (unsigned long long)identity->before.moe_captures,
                     (unsigned long long)identity->after.moe_captures,
                     (unsigned long long)identity->before.moe_replays,
                     (unsigned long long)identity->after.moe_replays,
                     (unsigned long long)identity->before.moe_failed,
                     (unsigned long long)identity->after.moe_failed);
    }
    identity->ok = ok && stats_ok;
    tensors_free(&graph_b);
    tensors_free(&graph_a);
    ds4_gpu_cleanup();
    return identity->ok;
}

} // namespace

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

int main(int argc, char **argv) {
    bool bench = false;
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
        bench = true;
    } else if (argc == 2 && std::strcmp(argv[1], "--correctness") == 0) {
        bench = false;
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--correctness|--bench]\n", argv[0]);
        return 2;
    }

    int devices = 0;
    if (!cuda_ok(cudaGetDeviceCount(&devices), "query device count") || devices < 1) {
        std::fprintf(stderr, "FAIL: CUDA device required\n");
        return 2;
    }
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1);
    (void)setenv("DS4_CUDA_SSD_ALIGNED_EXPERTS", "1", 1);
    (void)setenv("DS4_CUDA_ENABLE_STREAMING_EXPERT_PREFETCH", "1", 1);
    (void)unsetenv("DS4_CUDA_MOE_NO_IQ2_ALIGNED");
    (void)unsetenv("DS4_CUDA_MOE_NO_Q2K_ALIGNED");
    (void)unsetenv("DS4_CUDA_EXPERT_CACHE_2Q_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_EXPERT_CACHE_LAYER_QUOTA_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_EXPERT_CACHE_LAYER_QUOTA_DECODE_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_EXPERT_CACHE_LINEAR_SCAN_DIAGNOSTIC");
    (void)unsetenv("DS4_CUDA_PREFETCH_MAX_EXPERTS");
    (void)unsetenv("DS4_CUDA_DECODE_GRAPHS");
    (void)unsetenv("DS4_CUDA_MOE_PROFILE");
    (void)unsetenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE");
    (void)unsetenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6");

    model_fixture model;
    std::vector<float> input(kIn);
    fill_input(input.data());
    if (!model_init(&model)) {
        std::fprintf(stderr, "FAIL: fixture initialization\n");
        model_free(&model);
        return 1;
    }

    if (!bench) {
        correctness_result correctness;
        graph_identity_result identity;
        stream_offset_result stream_offset;
        const direct_midq_result direct_midq =
            run_direct_midq_identity(model, input);
        const router_rows_result router_rows =
            run_router_rows_exact(model);
        const moe_batch_exact_result moe_batch_exact =
            run_moe_batch_exact_identity(model);
        const bool graph_ok = run_graph_identity(
            model, input, &correctness, &identity, &stream_offset);
        const bool ok = direct_midq.ok && router_rows.ok &&
            moe_batch_exact.ok && graph_ok;
        std::printf(
            "{\"test\":\"cuda_flash_moe_decode\",\"mode\":\"correctness\","
            "\"shape\":{\"in\":%u,\"mid\":%u,\"out\":%u,"
            "\"experts\":%u,\"topk\":%u},\"cases\":%u,\"calls\":%u,"
            "\"byte_checks\":%u,\"output_rows\":%u,"
            "\"graph_identity\":{\"byte_checks\":%u,"
            "\"output_rows\":%u,\"pointer_update\":%s,"
            "\"moe_captures\":%llu,\"moe_replays\":%llu,"
            "\"moe_failed\":%llu},"
            "\"stream_slot_offset\":{\"rows\":%u,"
            "\"byte_checks\":%u,\"negative_checks\":%u,"
            "\"cache_capacity\":%u,\"cache_hits\":%llu,"
            "\"cache_misses\":%llu,\"cache_evictions\":%llu,"
            "\"prefetch_admitted\":%llu},"
            "\"router_rows_exact\":{\"cases\":%u,\"rows\":%u,"
            "\"byte_checks\":%u,\"canary_checks\":%u},"
            "\"moe_batch_decode_exact\":{\"cases\":%u,\"rows\":%u,"
            "\"byte_checks\":%u,\"canary_checks\":%u},"
            "\"flash_iq2_direct_midq\":{\"cases\":%u,"
            "\"byte_checks\":%u,\"guarded_captures\":%u},"
            "\"passed\":%s}\n",
            kIn, kMid, kOut, kTotalExperts, kTop, correctness.cases,
            correctness.calls, correctness.byte_checks,
            correctness.output_rows, identity.byte_checks,
            identity.output_rows,
            identity.pointer_update ? "true" : "false",
            (unsigned long long)(identity.after.moe_captures -
                                 identity.before.moe_captures),
            (unsigned long long)(identity.after.moe_replays -
                                 identity.before.moe_replays),
            (unsigned long long)(identity.after.moe_failed -
                                 identity.before.moe_failed),
            stream_offset.rows,
            stream_offset.byte_checks,
            stream_offset.negative_checks,
            stream_offset.cache_capacity,
            (unsigned long long)stream_offset.cache_hits,
            (unsigned long long)stream_offset.cache_misses,
            (unsigned long long)stream_offset.cache_evictions,
            (unsigned long long)stream_offset.prefetch_admitted,
            router_rows.cases,
            router_rows.rows,
            router_rows.byte_checks,
            router_rows.canary_checks,
            moe_batch_exact.cases,
            moe_batch_exact.rows,
            moe_batch_exact.byte_checks,
            moe_batch_exact.canary_checks,
            direct_midq.cases,
            direct_midq.byte_checks,
            direct_midq.guarded_captures,
            ok ? "true" : "false");
        model_free(&model);
        return ok ? 0 : 1;
    }

    const bool graph_bench = graph_env_enabled();
    tensors gpu;
    std::vector<uint8_t> xq_oracle(
        (kIn / kQk) * sizeof(block_q8_k_test));
    quantize_q8_k_oracle(input.data(), 1u, kIn,
                         reinterpret_cast<block_q8_k_test *>(xq_oracle.data()));
    bool initialized = gpu_fixture_init(model, input, &gpu);
    if (!initialized) {
        std::fprintf(stderr, "FAIL: benchmark fixture initialization\n");
        tensors_free(&gpu);
        ds4_gpu_cleanup();
        model_free(&model);
        return 1;
    }
    correctness_result correctness = run_correctness(model, &gpu, xq_oracle);
    int rc = correctness.ok ? 0 : 1;
    if (correctness.ok) {
        double median_ms = 0.0;
        double p95_ms = 0.0;
        bool perf_ok = run_benchmark(model, &gpu, &median_ms, &p95_ms);
        ds4_gpu_decode_graph_stats stats = {};
        ds4_gpu_decode_graphs_get_stats(&stats);
        if (graph_bench &&
            (stats.moe_captures == 0u || stats.moe_failed != 0u)) {
            perf_ok = false;
        }
        rc = perf_ok ? 0 : 1;
        std::printf(
            "{\"test\":\"cuda_flash_moe_decode\",\"mode\":\"bench\","
            "\"shape\":{\"in\":%u,\"mid\":%u,\"out\":%u,"
            "\"experts\":%u,\"topk\":%u},\"warmup_iterations\":100,"
            "\"blocks\":100,\"iterations_per_block\":10,"
            "\"iterations\":1000,\"median_ms\":%.6f,\"p95_ms\":%.6f,"
            "\"median_calls_per_second\":%.3f,\"correctness\":true,"
            "\"graph_enabled\":%s,\"moe_captures\":%llu,"
            "\"moe_replays\":%llu,\"moe_failed\":%llu,"
            "\"passed\":%s}\n",
            kIn, kMid, kOut, kTotalExperts, kTop, median_ms, p95_ms,
            median_ms > 0.0 ? 1000.0 / median_ms : 0.0,
            graph_bench ? "true" : "false",
            (unsigned long long)stats.moe_captures,
            (unsigned long long)stats.moe_replays,
            (unsigned long long)stats.moe_failed,
            perf_ok ? "true" : "false");
    } else {
        std::printf(
            "{\"test\":\"cuda_flash_moe_decode\",\"mode\":\"bench\","
            "\"correctness\":false,\"passed\":false}\n");
    }

    tensors_free(&gpu);
    ds4_gpu_cleanup();
    model_free(&model);
    return rc;
}
