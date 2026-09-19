/* Faithful A/B of the 6-bit code scan: current branchy extraction vs a
   branch-free unroll. 6 bits x 4 subquantizers = 24 bits = 3 bytes exactly, so
   the (byte, shift) pattern repeats every 4 and can be hoisted. */
#define _GNU_SOURCE
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SUBQ 32
#define BITS 6
#define CODEBOOK 64
#define CODE_SIZE 24

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

__attribute__((noinline))
static float scan_current(const uint8_t *codes, const uint8_t *norm_codes,
                          const float *lut, size_t rows, float constant, float scale) {
    float sink = 0.0f;
    for (size_t row = 0; row < rows; ++row) {
        float score = constant + scale * norm_codes[row];
        const uint8_t *packed_code = codes + row * CODE_SIZE;
        for (uint32_t m = 0; m < SUBQ; ++m) {
            size_t bit = (size_t)m * BITS;
            size_t byte = bit >> 3;
            uint32_t shift = (uint32_t)(bit & 7);
            uint32_t packed = packed_code[byte];
            if (shift + BITS > 8) packed |= (uint32_t)packed_code[byte + 1] << 8;
            uint32_t code = (packed >> shift) & (((uint32_t)1 << BITS) - 1);
            score += lut[(size_t)m * CODEBOOK + code];
        }
        sink += score;
    }
    return sink;
}

__attribute__((noinline))
static float scan_unrolled(const uint8_t *codes, const uint8_t *norm_codes,
                           const float *lut, size_t rows, float constant, float scale) {
    float sink = 0.0f;
    for (size_t row = 0; row < rows; ++row) {
        float score = constant + scale * norm_codes[row];
        const uint8_t *packed_code = codes + row * CODE_SIZE;
        /* eight groups of four 6-bit codes, each group exactly three bytes */
        for (uint32_t group = 0; group < SUBQ / 4; ++group) {
            const uint8_t *triple = packed_code + (size_t)group * 3;
            uint32_t word = (uint32_t)triple[0] | ((uint32_t)triple[1] << 8) |
                            ((uint32_t)triple[2] << 16);
            uint32_t m = group * 4;
            score += lut[(size_t)(m + 0) * CODEBOOK + ((word >> 0) & 63)];
            score += lut[(size_t)(m + 1) * CODEBOOK + ((word >> 6) & 63)];
            score += lut[(size_t)(m + 2) * CODEBOOK + ((word >> 12) & 63)];
            score += lut[(size_t)(m + 3) * CODEBOOK + ((word >> 18) & 63)];
        }
        sink += score;
    }
    return sink;
}

int main(void) {
    const size_t ROWS = 1036478;          /* rows Sfora scans per query at 100M */
    const int REPS = 40;
    uint8_t *codes = aligned_alloc(64, ROWS * CODE_SIZE);
    uint8_t *norms = aligned_alloc(64, ROWS);
    float *lut = aligned_alloc(64, SUBQ * CODEBOOK * sizeof(float));
    if (!codes || !norms || !lut) return 1;
    for (size_t i = 0; i < ROWS * CODE_SIZE; ++i) codes[i] = (uint8_t)(i * 31u);
    for (size_t i = 0; i < ROWS; ++i) norms[i] = (uint8_t)(i * 17u);
    for (int i = 0; i < SUBQ * CODEBOOK; ++i) lut[i] = (float)((i % 127) - 63) * 0.01f;

    float a = 0, b = 0;
    a += scan_current(codes, norms, lut, ROWS, 1.0f, 0.5f);
    b += scan_unrolled(codes, norms, lut, ROWS, 1.0f, 0.5f);

    double t0 = now_s();
    for (int r = 0; r < REPS; ++r) a += scan_current(codes, norms, lut, ROWS, 1.0f, 0.5f);
    double current_s = now_s() - t0;

    t0 = now_s();
    for (int r = 0; r < REPS; ++r) b += scan_unrolled(codes, norms, lut, ROWS, 1.0f, 0.5f);
    double unrolled_s = now_s() - t0;

    /* equality check on one pass, printed so the compiler cannot elide either */
    float one_a = scan_current(codes, norms, lut, ROWS, 1.0f, 0.5f);
    float one_b = scan_unrolled(codes, norms, lut, ROWS, 1.0f, 0.5f);

    printf("rows per pass          : %zu\n", ROWS);
    printf("current  rows/s/core   : %.0f  (%.3f ms per query-equivalent)\n",
           ROWS * REPS / current_s, current_s * 1000.0 / REPS);
    printf("unrolled rows/s/core   : %.0f  (%.3f ms per query-equivalent)\n",
           ROWS * REPS / unrolled_s, unrolled_s * 1000.0 / REPS);
    printf("speedup                : %.2fx\n", current_s / unrolled_s);
    printf("identical scores       : %s (%.6f vs %.6f)\n",
           one_a == one_b ? "yes" : "NO", one_a, one_b);
    printf("checksum               : %.3f\n", (double)(a + b));
    return 0;
}
