#define _GNU_SOURCE
#include <math.h>
#include <linux/io_uring.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    float score;
    int64_t id;
} Pair;

typedef struct {
    int fd;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_entries, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    void *sq_ring, *cq_ring;
    size_t sq_ring_size, cq_ring_size, sqes_size;
    int single_mmap;
} TinyRing;

static int tiny_ring_init(TinyRing *ring, unsigned entries) {
    struct io_uring_params params;
    memset(ring, 0, sizeof(*ring));
    memset(&params, 0, sizeof(params));
    ring->fd = (int)syscall(__NR_io_uring_setup, entries, &params);
    if (ring->fd < 0) return -1;
    ring->sq_ring_size = params.sq_off.array + params.sq_entries * sizeof(unsigned);
    ring->cq_ring_size = params.cq_off.cqes + params.cq_entries * sizeof(struct io_uring_cqe);
    ring->single_mmap = !!(params.features & IORING_FEAT_SINGLE_MMAP);
    if (ring->single_mmap && ring->cq_ring_size > ring->sq_ring_size)
        ring->sq_ring_size = ring->cq_ring_size;
    ring->sq_ring = mmap(NULL, ring->sq_ring_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQ_RING);
    if (ring->sq_ring == MAP_FAILED) goto fail;
    if (ring->single_mmap) {
        ring->cq_ring = ring->sq_ring;
        ring->cq_ring_size = ring->sq_ring_size;
    } else {
        ring->cq_ring = mmap(NULL, ring->cq_ring_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_CQ_RING);
        if (ring->cq_ring == MAP_FAILED) goto fail;
    }
    ring->sqes_size = params.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes = mmap(NULL, ring->sqes_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) goto fail;
    ring->sq_head = (unsigned *)((char *)ring->sq_ring + params.sq_off.head);
    ring->sq_tail = (unsigned *)((char *)ring->sq_ring + params.sq_off.tail);
    ring->sq_mask = (unsigned *)((char *)ring->sq_ring + params.sq_off.ring_mask);
    ring->sq_entries = (unsigned *)((char *)ring->sq_ring + params.sq_off.ring_entries);
    ring->sq_array = (unsigned *)((char *)ring->sq_ring + params.sq_off.array);
    ring->cq_head = (unsigned *)((char *)ring->cq_ring + params.cq_off.head);
    ring->cq_tail = (unsigned *)((char *)ring->cq_ring + params.cq_off.tail);
    ring->cq_mask = (unsigned *)((char *)ring->cq_ring + params.cq_off.ring_mask);
    ring->cqes = (struct io_uring_cqe *)((char *)ring->cq_ring + params.cq_off.cqes);
    return 0;
fail:
    if (ring->sqes && ring->sqes != MAP_FAILED) munmap(ring->sqes, ring->sqes_size);
    if (ring->cq_ring && ring->cq_ring != MAP_FAILED && !ring->single_mmap)
        munmap(ring->cq_ring, ring->cq_ring_size);
    if (ring->sq_ring && ring->sq_ring != MAP_FAILED) munmap(ring->sq_ring, ring->sq_ring_size);
    close(ring->fd);
    return -1;
}

static void tiny_ring_destroy(TinyRing *ring) {
    munmap(ring->sqes, ring->sqes_size);
    if (!ring->single_mmap) munmap(ring->cq_ring, ring->cq_ring_size);
    munmap(ring->sq_ring, ring->sq_ring_size);
    close(ring->fd);
}

void one_lut_decode_codes(
    const uint8_t *codes, int rows, int subquantizers, int bits_per_code, uint8_t *output
) {
    int code_size = (subquantizers * bits_per_code + 7) / 8;
    for (int row = 0; row < rows; ++row) {
        const uint8_t *code = codes + (size_t)row * code_size;
        for (int m = 0; m < subquantizers; ++m) {
            int bit = m * bits_per_code;
            int byte = bit >> 3;
            int shift = bit & 7;
            unsigned packed = code[byte];
            if (shift + bits_per_code > 8) packed |= (unsigned)code[byte + 1] << 8;
            output[(size_t)row * subquantizers + m] =
                (uint8_t)((packed >> shift) & ((1u << bits_per_code) - 1u));
        }
    }
}

static int worse(Pair a, Pair b) {
    return a.score > b.score || (a.score == b.score && a.id > b.id);
}

static void swap_pair(Pair *a, Pair *b) {
    Pair t = *a;
    *a = *b;
    *b = t;
}

static void heap_up(Pair *heap, int i) {
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!worse(heap[i], heap[p])) break;
        swap_pair(&heap[i], &heap[p]);
        i = p;
    }
}

static void heap_down(Pair *heap, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, w = i;
        if (l < n && worse(heap[l], heap[w])) w = l;
        if (r < n && worse(heap[r], heap[w])) w = r;
        if (w == i) break;
        swap_pair(&heap[i], &heap[w]);
        i = w;
    }
}

static void offer(Pair *heap, int *n, int cap, Pair value) {
    if (*n < cap) {
        heap[*n] = value;
        heap_up(heap, *n);
        ++*n;
    } else if (worse(heap[0], value)) {
        heap[0] = value;
        heap_down(heap, *n, 0);
    }
}

static int ascending_pair(const void *left, const void *right) {
    const Pair *a = (const Pair *)left;
    const Pair *b = (const Pair *)right;
    if (a->score < b->score) return -1;
    if (a->score > b->score) return 1;
    return (a->id > b->id) - (a->id < b->id);
}

int one_lut_search(
    const float *query,
    const float *coarse,
    const float *pq,
    const uint64_t *offsets,
    const void *ids,
    int ids_are_u32,
    const uint8_t *codes,
    const uint8_t *norm_codes,
    const float *norm_low,
    const float *norm_scale,
    int nlist,
    int dimensions,
    int subquantizers,
    int bits_per_code,
    int nprobe,
    int shortlist,
    int return_width,
    int exact_rerank,
    const void *base,
    int direct_fd,
    int base_is_u8,
    int64_t *output_ids,
    float *output_distances
) {
    if (!query || !coarse || !pq || !offsets || !ids || !codes || !norm_codes ||
        !norm_low || !norm_scale || (exact_rerank == 1 && !base) ||
        (exact_rerank == 2 && direct_fd < 0) || !output_ids || !output_distances ||
        nlist <= 0 || dimensions <= 0 || subquantizers <= 0 ||
        bits_per_code < 1 || bits_per_code > 8 ||
        dimensions % subquantizers || nprobe <= 0 || nprobe > nlist ||
        shortlist < return_width || return_width <= 0) return -1;

    int dsub = dimensions / subquantizers;
    int codebook_size = 1 << bits_per_code;
    int code_size = (subquantizers * bits_per_code + 7) / 8;
    Pair *list_heap = malloc((size_t)nprobe * sizeof(Pair));
    int thread_count = 1;
#ifdef _OPENMP
    thread_count = omp_get_max_threads();
#endif
    Pair *candidate_heaps = malloc((size_t)thread_count * shortlist * sizeof(Pair));
    int *candidate_counts = calloc((size_t)thread_count, sizeof(int));
    float *lut = malloc((size_t)subquantizers * codebook_size * sizeof(float));
    if (!list_heap || !candidate_heaps || !candidate_counts || !lut) {
        free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
        return -2;
    }

    int list_count = 0;
    float query_norm = 0.0f;
    for (int j = 0; j < dimensions; ++j) query_norm = fmaf(query[j], query[j], query_norm);
    for (int list = 0; list < nlist; ++list) {
        float distance = 0.0f;
        const float *centroid = coarse + (size_t)list * dimensions;
        for (int j = 0; j < dimensions; ++j) {
            float delta = query[j] - centroid[j];
            distance = fmaf(delta, delta, distance);
        }
        offer(list_heap, &list_count, nprobe, (Pair){distance, list});
    }
    qsort(list_heap, (size_t)list_count, sizeof(Pair), ascending_pair);

    for (int m = 0; m < subquantizers; ++m) {
        for (int code = 0; code < codebook_size; ++code) {
            float dot = 0.0f;
            const float *center = pq + ((size_t)m * codebook_size + code) * dsub;
            for (int j = 0; j < dsub; ++j) dot = fmaf(query[m * dsub + j], center[j], dot);
            lut[(size_t)m * codebook_size + code] = -2.0f * dot;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < list_count; ++p) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        Pair *local_heap = candidate_heaps + (size_t)thread_id * shortlist;
        int *local_count = &candidate_counts[thread_id];
        int list = (int)list_heap[p].id;
        const float *centroid = coarse + (size_t)list * dimensions;
        float coarse_dot = 0.0f;
        for (int j = 0; j < dimensions; ++j) coarse_dot = fmaf(query[j], centroid[j], coarse_dot);
        float constant = query_norm - 2.0f * coarse_dot + norm_low[list];
        float scale = norm_scale[list];
        for (uint64_t row = offsets[list]; row < offsets[list + 1]; ++row) {
            float score = constant + scale * norm_codes[row];
            const uint8_t *code = codes + row * (uint64_t)code_size;
            for (int m = 0; m < subquantizers; ++m) {
                int bit = m * bits_per_code;
                int byte = bit >> 3;
                int shift = bit & 7;
                unsigned packed = code[byte];
                if (shift + bits_per_code > 8) packed |= (unsigned)code[byte + 1] << 8;
                unsigned index = (packed >> shift) & ((1u << bits_per_code) - 1u);
                score += lut[(size_t)m * codebook_size + index];
            }
            int64_t vector_id = ids_are_u32
                ? (int64_t)((const uint32_t *)ids)[row]
                : ((const int64_t *)ids)[row];
            offer(local_heap, local_count, shortlist, (Pair){score, vector_id});
        }
    }

    Pair *candidate_heap = candidate_heaps;
    int candidate_count = candidate_counts[0];
    for (int thread_id = 1; thread_id < thread_count; ++thread_id) {
        Pair *local_heap = candidate_heaps + (size_t)thread_id * shortlist;
        for (int i = 0; i < candidate_counts[thread_id]; ++i) {
            offer(candidate_heap, &candidate_count, shortlist, local_heap[i]);
        }
    }

    if (exact_rerank == 2) {
        TinyRing ring;
        uint8_t *buffers = NULL;
        size_t *inside = calloc((size_t)candidate_count, sizeof(size_t));
        size_t *read_sizes = calloc((size_t)candidate_count, sizeof(size_t));
        if (!inside || !read_sizes || posix_memalign((void **)&buffers, 4096,
                                                     (size_t)candidate_count * 8192) != 0 ||
            tiny_ring_init(&ring, (unsigned)candidate_count) != 0) {
            free(inside); free(read_sizes); free(buffers);
            free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
            return -4;
        }
        unsigned tail = __atomic_load_n(ring.sq_tail, __ATOMIC_RELAXED);
        uint64_t row_bytes = (uint64_t)dimensions * (base_is_u8 ? 1u : 4u);
        for (int i = 0; i < candidate_count; ++i) {
            uint64_t byte_offset = 8u + (uint64_t)candidate_heap[i].id * row_bytes;
            uint64_t aligned_offset = byte_offset & ~4095ull;
            inside[i] = (size_t)(byte_offset - aligned_offset);
            read_sizes[i] = inside[i] + row_bytes <= 4096 ? 4096 : 8192;
            unsigned slot = tail & *ring.sq_mask;
            struct io_uring_sqe *sqe = &ring.sqes[slot];
            memset(sqe, 0, sizeof(*sqe));
            sqe->opcode = IORING_OP_READ;
            sqe->fd = direct_fd;
            sqe->off = aligned_offset;
            sqe->addr = (uint64_t)(uintptr_t)(buffers + (size_t)i * 8192);
            sqe->len = (unsigned)read_sizes[i];
            sqe->user_data = (uint64_t)i;
            ring.sq_array[slot] = slot;
            ++tail;
        }
        __atomic_store_n(ring.sq_tail, tail, __ATOMIC_RELEASE);
        int submitted = (int)syscall(__NR_io_uring_enter, ring.fd, candidate_count,
                                     candidate_count, IORING_ENTER_GETEVENTS, NULL, 0);
        int io_error = submitted < 0;
        unsigned completed = 0;
        while (!io_error && completed < (unsigned)candidate_count) {
            unsigned head = __atomic_load_n(ring.cq_head, __ATOMIC_RELAXED);
            unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
            while (head != cq_tail) {
                struct io_uring_cqe *cqe = &ring.cqes[head & *ring.cq_mask];
                unsigned i = (unsigned)cqe->user_data;
                if (i >= (unsigned)candidate_count || cqe->res < 0 ||
                    (size_t)cqe->res < inside[i] + row_bytes) io_error = 1;
                ++head; ++completed;
            }
            __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
            if (!io_error && completed < (unsigned)candidate_count &&
                syscall(__NR_io_uring_enter, ring.fd, 0,
                        candidate_count - completed, IORING_ENTER_GETEVENTS, NULL, 0) < 0)
                io_error = 1;
        }
        if (!io_error) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < candidate_count; ++i) {
                const uint8_t *raw = buffers + (size_t)i * 8192 + inside[i];
                float distance = 0.0f;
                if (base_is_u8) {
                    for (int j = 0; j < dimensions; ++j) {
                        float delta = query[j] - raw[j];
                        distance = fmaf(delta, delta, distance);
                    }
                } else {
                    const float *row = (const float *)raw;
                    for (int j = 0; j < dimensions; ++j) {
                        float delta = query[j] - row[j];
                        distance = fmaf(delta, delta, distance);
                    }
                }
                candidate_heap[i].score = distance;
            }
        }
        tiny_ring_destroy(&ring);
        free(inside); free(read_sizes); free(buffers);
        if (io_error) {
            free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
            return -4;
        }
    } else if (exact_rerank == 1) {
        for (int i = 0; i < candidate_count; ++i) {
            int64_t id = candidate_heap[i].id;
            float distance = 0.0f;
            if (base_is_u8) {
                const uint8_t *row = (const uint8_t *)base + (uint64_t)id * dimensions;
                for (int j = 0; j < dimensions; ++j) {
                    float delta = query[j] - row[j];
                    distance = fmaf(delta, delta, distance);
                }
            } else {
                const float *row = (const float *)base + (uint64_t)id * dimensions;
                for (int j = 0; j < dimensions; ++j) {
                    float delta = query[j] - row[j];
                    distance = fmaf(delta, delta, distance);
                }
            }
            candidate_heap[i].score = distance;
        }
    }
    qsort(candidate_heap, (size_t)candidate_count, sizeof(Pair), ascending_pair);
    if (candidate_count < return_width) {
        free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
        return -3;
    }
    for (int i = 0; i < return_width; ++i) {
        output_ids[i] = candidate_heap[i].id;
        output_distances[i] = candidate_heap[i].score;
    }

    free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
    return 0;
}

int exact_rerank_direct(
    const float *query,
    const int64_t *candidate_ids,
    int candidate_count,
    int dimensions,
    int return_width,
    int direct_fd,
    int base_is_u8,
    int64_t *output_ids,
    float *output_distances
) {
    if (!query || !candidate_ids || candidate_count <= 0 || dimensions <= 0 ||
        return_width <= 0 || return_width > candidate_count || direct_fd < 0 ||
        !output_ids || !output_distances) return -1;

    Pair *candidates = malloc((size_t)candidate_count * sizeof(Pair));
    size_t *inside = calloc((size_t)candidate_count, sizeof(size_t));
    size_t *read_sizes = calloc((size_t)candidate_count, sizeof(size_t));
    uint8_t *buffers = NULL;
    TinyRing ring;
    memset(&ring, 0, sizeof(ring));
    if (!candidates || !inside || !read_sizes ||
        posix_memalign((void **)&buffers, 4096, (size_t)candidate_count * 8192) != 0) {
        free(candidates); free(inside); free(read_sizes); free(buffers);
        return -2;
    }
    if (tiny_ring_init(&ring, (unsigned)candidate_count) != 0) {
        free(candidates); free(inside); free(read_sizes); free(buffers);
        return -2;
    }

    unsigned tail = __atomic_load_n(ring.sq_tail, __ATOMIC_RELAXED);
    uint64_t row_bytes = (uint64_t)dimensions * (base_is_u8 ? 1u : 4u);
    for (int i = 0; i < candidate_count; ++i) {
        if (candidate_ids[i] < 0) {
            tiny_ring_destroy(&ring);
            free(candidates); free(inside); free(read_sizes); free(buffers);
            return -1;
        }
        candidates[i].id = candidate_ids[i];
        uint64_t byte_offset = 8u + (uint64_t)candidate_ids[i] * row_bytes;
        uint64_t aligned_offset = byte_offset & ~4095ull;
        inside[i] = (size_t)(byte_offset - aligned_offset);
        read_sizes[i] = inside[i] + row_bytes <= 4096 ? 4096 : 8192;
        unsigned slot = tail & *ring.sq_mask;
        struct io_uring_sqe *sqe = &ring.sqes[slot];
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode = IORING_OP_READ;
        sqe->fd = direct_fd;
        sqe->off = aligned_offset;
        sqe->addr = (uint64_t)(uintptr_t)(buffers + (size_t)i * 8192);
        sqe->len = (unsigned)read_sizes[i];
        sqe->user_data = (uint64_t)i;
        ring.sq_array[slot] = slot;
        ++tail;
    }
    __atomic_store_n(ring.sq_tail, tail, __ATOMIC_RELEASE);
    int submitted = (int)syscall(__NR_io_uring_enter, ring.fd, candidate_count,
                                 candidate_count, IORING_ENTER_GETEVENTS, NULL, 0);
    int io_error = submitted < 0;
    unsigned completed = 0;
    while (!io_error && completed < (unsigned)candidate_count) {
        unsigned head = __atomic_load_n(ring.cq_head, __ATOMIC_RELAXED);
        unsigned cq_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        while (head != cq_tail) {
            struct io_uring_cqe *cqe = &ring.cqes[head & *ring.cq_mask];
            unsigned i = (unsigned)cqe->user_data;
            if (i >= (unsigned)candidate_count || cqe->res < 0 ||
                (size_t)cqe->res < inside[i] + row_bytes) io_error = 1;
            ++head; ++completed;
        }
        __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
        if (!io_error && completed < (unsigned)candidate_count &&
            syscall(__NR_io_uring_enter, ring.fd, 0, candidate_count - completed,
                    IORING_ENTER_GETEVENTS, NULL, 0) < 0) io_error = 1;
    }
    if (!io_error) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < candidate_count; ++i) {
            const uint8_t *raw = buffers + (size_t)i * 8192 + inside[i];
            float distance = 0.0f;
            if (base_is_u8) {
                for (int j = 0; j < dimensions; ++j) {
                    float delta = query[j] - raw[j];
                    distance = fmaf(delta, delta, distance);
                }
            } else {
                const float *row = (const float *)raw;
                for (int j = 0; j < dimensions; ++j) {
                    float delta = query[j] - row[j];
                    distance = fmaf(delta, delta, distance);
                }
            }
            candidates[i].score = distance;
        }
        qsort(candidates, (size_t)candidate_count, sizeof(Pair), ascending_pair);
        for (int i = 0; i < return_width; ++i) {
            output_ids[i] = candidates[i].id;
            output_distances[i] = candidates[i].score;
        }
    }
    tiny_ring_destroy(&ring);
    free(candidates); free(inside); free(read_sizes); free(buffers);
    return io_error ? -3 : 0;
}
