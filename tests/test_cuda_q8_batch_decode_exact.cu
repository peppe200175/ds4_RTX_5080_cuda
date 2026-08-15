#include "ds4_gpu.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

namespace {

constexpr uint64_t kInDim = 1057u;
constexpr uint64_t kOutDim = 257u;
constexpr uint64_t kBlocks = (kInDim + 31u) / 32u;
constexpr uint64_t kWeightOffset = 4096u;
constexpr uint64_t kWeightBytes = kOutDim * kBlocks * 34u;
constexpr uint64_t kModelBytes =
    (kWeightOffset + kWeightBytes + 4095u) & ~4095ull;
constexpr uint64_t kGuardBytes = 64u;
constexpr uint8_t kGuard = 0xd7u;
constexpr uint8_t kPoison = 0xa5u;

struct block_q8_0_test {
    uint16_t d;
    int8_t qs[32];
};

static_assert(sizeof(block_q8_0_test) == 34u,
              "unexpected Q8_0 block layout");

struct guarded_tensor {
    const char *name = nullptr;
    uint64_t payload_bytes = 0u;
    ds4_gpu_tensor *base = nullptr;
    ds4_gpu_tensor *view = nullptr;
};

uint32_t prng_next(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

bool guarded_init(guarded_tensor *tensor, const char *name, uint64_t bytes) {
    tensor->name = name;
    tensor->payload_bytes = bytes;
    tensor->base = ds4_gpu_tensor_alloc(bytes + 2u * kGuardBytes);
    if (!tensor->base) return false;
    tensor->view = ds4_gpu_tensor_view(tensor->base, kGuardBytes, bytes);
    if (!tensor->view) return false;
    std::vector<uint8_t> init((size_t)bytes + 2u * kGuardBytes, kGuard);
    return ds4_gpu_tensor_write(
               tensor->base, 0u, init.data(), init.size()) != 0;
}

void guarded_free(guarded_tensor *tensor) {
    ds4_gpu_tensor_free(tensor->view);
    ds4_gpu_tensor_free(tensor->base);
    tensor->view = nullptr;
    tensor->base = nullptr;
}

bool guarded_poison(guarded_tensor *tensor) {
    std::vector<uint8_t> poison((size_t)tensor->payload_bytes, kPoison);
    return ds4_gpu_tensor_write(
               tensor->view, 0u, poison.data(), poison.size()) != 0;
}

bool guarded_check(const guarded_tensor &tensor) {
    uint8_t prefix[kGuardBytes];
    uint8_t suffix[kGuardBytes];
    if (!ds4_gpu_tensor_read(tensor.base, 0u, prefix, sizeof(prefix)) ||
        !ds4_gpu_tensor_read(tensor.base,
                             kGuardBytes + tensor.payload_bytes,
                             suffix,
                             sizeof(suffix))) {
        std::fprintf(stderr, "FAIL: cannot read %s canary\n", tensor.name);
        return false;
    }
    for (uint64_t i = 0u; i < kGuardBytes; i++) {
        if (prefix[i] != kGuard || suffix[i] != kGuard) {
            std::fprintf(stderr,
                         "FAIL: %s %s canary changed at byte %llu\n",
                         tensor.name,
                         prefix[i] != kGuard ? "prefix" : "suffix",
                         (unsigned long long)i);
            return false;
        }
    }
    return true;
}

void fill_weights(void *model) {
    block_q8_0_test *weights = reinterpret_cast<block_q8_0_test *>(
        static_cast<uint8_t *>(model) + kWeightOffset);
    uint32_t state = 0x91e10da5u;
    for (uint64_t i = 0u; i < kOutDim * kBlocks; i++) {
        weights[i].d = 0x2800u; /* IEEE binary16 2^-5. */
        for (uint32_t q = 0u; q < 32u; q++) {
            weights[i].qs[q] =
                static_cast<int8_t>((int32_t)(prng_next(&state) % 31u) - 15);
        }
    }
}

void fill_input(std::vector<float> *input, uint32_t n_rows) {
    input->resize((size_t)n_rows * kInDim);
    uint32_t state = 0x243f6a88u ^ n_rows;
    for (float &value : *input) {
        value = ((int32_t)(prng_next(&state) % 2001u) - 1000) / 997.0f;
    }
}

bool finite_and_written(const std::vector<float> &values, uint32_t n_rows) {
    for (size_t i = 0u; i < values.size(); i++) {
        uint32_t bits = 0u;
        std::memcpy(&bits, &values[i], sizeof(bits));
        if (bits == 0xa5a5a5a5u || !std::isfinite(values[i])) {
            std::fprintf(stderr,
                         "FAIL: rows=%u output invalid at value %zu\n",
                         n_rows,
                         i);
            return false;
        }
    }
    return true;
}

bool run_case(const void *model, uint32_t n_rows) {
    const uint64_t input_bytes =
        (uint64_t)n_rows * kInDim * sizeof(float);
    const uint64_t output_bytes =
        (uint64_t)n_rows * kOutDim * sizeof(float);
    guarded_tensor input;
    guarded_tensor rowwise;
    guarded_tensor batch;
    std::vector<float> host_input;
    std::vector<float> expected((size_t)n_rows * kOutDim);
    std::vector<float> actual((size_t)n_rows * kOutDim);
    fill_input(&host_input, n_rows);

    bool ok = guarded_init(&input, "input", input_bytes) &&
              guarded_init(&rowwise, "rowwise", output_bytes) &&
              guarded_init(&batch, "batch", output_bytes) &&
              guarded_poison(&rowwise) && guarded_poison(&batch) &&
              ds4_gpu_tensor_write(input.view,
                                   0u,
                                   host_input.data(),
                                   input_bytes) != 0;
    for (uint32_t row = 0u; ok && row < n_rows; row++) {
        ds4_gpu_tensor *input_row = ds4_gpu_tensor_view(
            input.view,
            (uint64_t)row * kInDim * sizeof(float),
            kInDim * sizeof(float));
        ds4_gpu_tensor *output_row = ds4_gpu_tensor_view(
            rowwise.view,
            (uint64_t)row * kOutDim * sizeof(float),
            kOutDim * sizeof(float));
        ok = input_row && output_row &&
             ds4_gpu_matmul_q8_0_tensor(output_row,
                                        model,
                                        kModelBytes,
                                        kWeightOffset,
                                        kInDim,
                                        kOutDim,
                                        input_row,
                                        1u) != 0;
        ds4_gpu_tensor_free(output_row);
        ds4_gpu_tensor_free(input_row);
    }
    if (ok) {
        ok = ds4_gpu_matmul_q8_0_tensor(batch.view,
                                       model,
                                       kModelBytes,
                                       kWeightOffset,
                                       kInDim,
                                       kOutDim,
                                       input.view,
                                       n_rows) != 0;
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(
                 rowwise.view, 0u, expected.data(), output_bytes) != 0 &&
             ds4_gpu_tensor_read(
                 batch.view, 0u, actual.data(), output_bytes) != 0;
    }
    if (ok && std::memcmp(expected.data(), actual.data(), output_bytes) != 0) {
        size_t first = 0u;
        const uint8_t *a = reinterpret_cast<const uint8_t *>(expected.data());
        const uint8_t *b = reinterpret_cast<const uint8_t *>(actual.data());
        while (first < output_bytes && a[first] == b[first]) first++;
        std::fprintf(stderr,
                     "FAIL: rows=%u byte mismatch at %zu: 0x%02x != 0x%02x\n",
                     n_rows,
                     first,
                     (unsigned)a[first],
                     (unsigned)b[first]);
        ok = false;
    }
    if (ok) {
        ok = finite_and_written(expected, n_rows) &&
             finite_and_written(actual, n_rows) &&
             guarded_check(input) && guarded_check(rowwise) &&
             guarded_check(batch);
    }

    guarded_free(&batch);
    guarded_free(&rowwise);
    guarded_free(&input);
    return ok;
}

} // namespace

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

int main() {
    static const uint32_t row_counts[] = {
        1u, 2u, 4u, 8u, 18u, 19u, 27u, 64u, 1024u,
    };
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) {
        std::fprintf(stderr, "FAIL: CUDA device required\n");
        return 2;
    }
    (void)setenv("DS4_CUDA_Q8_BATCH_DECODE_EXACT", "1", 1);
    (void)setenv("DS4_CUDA_COPY_MODEL", "1", 1);
    (void)setenv("DS4_CUDA_NO_DERIVED_WEIGHTS", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F16_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_NO_Q8_F32_CACHE", "1", 1);
    (void)setenv("DS4_CUDA_MMQ", "0", 1);

    void *model = mmap(nullptr,
                       (size_t)kModelBytes,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1,
                       0);
    if (model == MAP_FAILED) {
        std::perror("mmap synthetic Q8 model");
        return 1;
    }
    std::memset(model, 0, (size_t)kModelBytes);
    fill_weights(model);

    bool ok = ds4_gpu_init() != 0;
    ds4_gpu_set_quality(true);
    ok = ok && ds4_gpu_set_model_map(model, kModelBytes) != 0;
    uint32_t passed = 0u;
    for (uint32_t n_rows : row_counts) {
        if (!ok || !run_case(model, n_rows)) {
            ok = false;
            break;
        }
        passed++;
    }

    ds4_gpu_cleanup();
    (void)munmap(model, (size_t)kModelBytes);
    std::printf(
        "{\"test\":\"cuda_q8_batch_decode_exact\","
        "\"cases\":%u,\"byte_identity\":%s,\"canaries\":%s,"
        "\"passed\":%s}\n",
        passed,
        ok ? "true" : "false",
        ok ? "true" : "false",
        ok ? "true" : "false");
    return ok ? 0 : 1;
}
