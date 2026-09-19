/* Is the candidate scan TLB-bound? Scan the REAL 2.4 GB codes file with the real
   access pattern: 640 random posting lists, each a sequential run of ~1620 rows
   of 24 bytes. Compare plain mapping against MADV_HUGEPAGE. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SUBQ 32
#define CODEBOOK 64
#define CODE_SIZE 24
#define LISTS 640
#define ROWS_PER_LIST 1620

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static float scan(const uint8_t *codes, const float *lut, size_t rows_total,
                  unsigned seed, size_t *scanned) {
    float sink = 0.0f;
    size_t count = 0;
    unsigned state = seed;
    for (int l = 0; l < LISTS; ++l) {
        state = state * 1103515245u + 12345u;
        size_t start = (size_t)((double)(state >> 1) / 2147483648.0 *
                                (double)(rows_total - ROWS_PER_LIST));
        for (size_t row = start; row < start + ROWS_PER_LIST; ++row) {
            const uint8_t *pc = codes + row * CODE_SIZE;
            float score = 0.0f;
            for (uint32_t m = 0; m < SUBQ; ++m) {
                size_t bit = (size_t)m * 6; size_t byte = bit >> 3;
                uint32_t shift = (uint32_t)(bit & 7);
                uint32_t packed = pc[byte];
                if (shift + 6 > 8) packed |= (uint32_t)pc[byte + 1] << 8;
                score += lut[(size_t)m * CODEBOOK + ((packed >> shift) & 63)];
            }
            sink += score; ++count;
        }
    }
    *scanned = count;
    return sink;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "codes.u8";
    int use_huge = argc > 2 ? atoi(argv[2]) : 0;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t bytes = (size_t)st.st_size;
    void *m = mmap(NULL, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (m == MAP_FAILED) { perror("mmap"); return 1; }
    if (use_huge) {
        if (madvise(m, bytes, MADV_HUGEPAGE) != 0) perror("MADV_HUGEPAGE");
        if (madvise(m, bytes, MADV_WILLNEED) != 0) perror("MADV_WILLNEED");
    }
    size_t rows_total = bytes / CODE_SIZE;
    float *lut = aligned_alloc(64, SUBQ * CODEBOOK * sizeof(float));
    for (int i = 0; i < SUBQ * CODEBOOK; ++i) lut[i] = (float)((i % 127) - 63) * 0.01f;

    size_t scanned = 0; float sink = 0;
    sink += scan((const uint8_t *)m, lut, rows_total, 7u, &scanned);   /* warm */
    double best = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        double t0 = now_s();
        sink += scan((const uint8_t *)m, lut, rows_total, 7u, &scanned);
        double el = now_s() - t0;
        if (el < best) best = el;
    }
    printf("%-10s rows %zu  best %.3f ms/query-equivalent  %.0f rows/s/core  %.2f GB/s  sink %.1f\n",
           use_huge ? "HUGEPAGE" : "baseline", scanned, best * 1000.0,
           scanned / best, scanned * 25.0 / best / 1e9, (double)sink);
    munmap(m, bytes);
    return 0;
}
