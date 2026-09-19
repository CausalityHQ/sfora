#define _GNU_SOURCE
#include <math.h>
#include <sched.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <linux/io_uring.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef STATX_DIOALIGN
#define STATX_DIOALIGN 0x00002000U
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    float score;
    uint32_t id;
} SforaPair;

typedef struct {
    double score;
    uint32_t id;
} SforaExactPair;

typedef struct {
    int fd;
    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_mask;
    unsigned *sq_array;
    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_mask;
    unsigned *cq_overflow;
    unsigned *sq_dropped;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
    void *sq_ring;
    void *cq_ring;
    size_t sq_ring_size;
    size_t cq_ring_size;
    size_t sqes_size;
    int single_mmap;
} SforaRing;

typedef struct {
    SforaRing ring;
    SforaExactPair *pairs;
    size_t *inside;
    size_t *read_sizes;
    uint8_t *terminal;
    uint8_t *buffers;
    void *buffer_mapping;
    size_t buffer_mapping_size;
    int data_fd;
} SforaDirectQuarantine;

static _Atomic int sfora_direct_state = 0;
static SforaDirectQuarantine sfora_direct_quarantine;
/* The per-query read buffer is 8 MiB at BigANN geometry, and mapping, faulting
   and unmapping it costs far more than every other fixed step combined. Keep it
   mapped between queries; the admission ledger already reserves these bytes for
   the one direct context this process admits. */
static void *sfora_direct_buffer_mapping = NULL;
static size_t sfora_direct_buffer_size = 0;

static void sfora_direct_buffer_forget(void) {
    sfora_direct_buffer_mapping = NULL;
    sfora_direct_buffer_size = 0;
}

static int checked_product(size_t left, size_t right, size_t *output);

uint32_t sfora_factorized_backend_abi_version(void) {
    return 2;
}

static void ring_destroy(SforaRing *ring) {
    if (ring->sqes && ring->sqes != MAP_FAILED) munmap(ring->sqes, ring->sqes_size);
    if (!ring->single_mmap && ring->cq_ring && ring->cq_ring != MAP_FAILED)
        munmap(ring->cq_ring, ring->cq_ring_size);
    if (ring->sq_ring && ring->sq_ring != MAP_FAILED)
        munmap(ring->sq_ring, ring->sq_ring_size);
    if (ring->fd >= 0) close(ring->fd);
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;
}

static int ring_init(SforaRing *ring, unsigned entries) {
    struct io_uring_params parameters;
    memset(ring, 0, sizeof(*ring));
    ring->fd = -1;
    memset(&parameters, 0, sizeof(parameters));
    ring->fd = (int)syscall(__NR_io_uring_setup, entries, &parameters);
    if (ring->fd < 0) return -1;
    ring->sq_ring_size = parameters.sq_off.array + parameters.sq_entries * sizeof(unsigned);
    ring->cq_ring_size = parameters.cq_off.cqes +
                         parameters.cq_entries * sizeof(struct io_uring_cqe);
    ring->single_mmap = !!(parameters.features & IORING_FEAT_SINGLE_MMAP);
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
    ring->sqes_size = parameters.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes = mmap(NULL, ring->sqes_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, ring->fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) goto fail;
    ring->sq_head = (unsigned *)((char *)ring->sq_ring + parameters.sq_off.head);
    ring->sq_tail = (unsigned *)((char *)ring->sq_ring + parameters.sq_off.tail);
    ring->sq_mask = (unsigned *)((char *)ring->sq_ring + parameters.sq_off.ring_mask);
    ring->sq_array = (unsigned *)((char *)ring->sq_ring + parameters.sq_off.array);
    ring->sq_dropped = (unsigned *)((char *)ring->sq_ring + parameters.sq_off.dropped);
    ring->cq_head = (unsigned *)((char *)ring->cq_ring + parameters.cq_off.head);
    ring->cq_tail = (unsigned *)((char *)ring->cq_ring + parameters.cq_off.tail);
    ring->cq_mask = (unsigned *)((char *)ring->cq_ring + parameters.cq_off.ring_mask);
    ring->cq_overflow = (unsigned *)((char *)ring->cq_ring + parameters.cq_off.overflow);
    ring->cqes = (struct io_uring_cqe *)((char *)ring->cq_ring + parameters.cq_off.cqes);
    return 0;
fail:
    {
        int saved_errno = errno;
        ring_destroy(ring);
        errno = saved_errno;
        return -1;
    }
}

int sfora_direct_io_probe(void) {
    SforaRing ring;
    if (ring_init(&ring, 1) != 0)
        return (errno == EPERM || errno == EACCES || errno == ENOSYS) ? -5 : -2;
    ring_destroy(&ring);
    return 0;
}

int sfora_direct_io_alignment(int fd, uint32_t *memory_alignment,
                              uint32_t *offset_alignment) {
    struct statx metadata;
    if (fd < 0 || !memory_alignment || !offset_alignment) return -1;
    memset(&metadata, 0, sizeof(metadata));
    if (syscall(__NR_statx, fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &metadata) != 0 ||
        !(metadata.stx_mask & STATX_DIOALIGN) || metadata.stx_dio_mem_align == 0 ||
        metadata.stx_dio_offset_align == 0)
        return -1;
    *memory_alignment = metadata.stx_dio_mem_align;
    *offset_alignment = metadata.stx_dio_offset_align;
    return 0;
}

static int ascending_exact_pair(const void *left, const void *right) {
    const SforaExactPair *a = (const SforaExactPair *)left;
    const SforaExactPair *b = (const SforaExactPair *)right;
    if (a->score < b->score) return -1;
    if (a->score > b->score) return 1;
    return (a->id > b->id) - (a->id < b->id);
}

static int align_up(size_t value, size_t alignment, size_t *output) {
    size_t remainder;
    if (alignment == 0) return -1;
    remainder = value % alignment;
    if (remainder == 0) {
        *output = value;
        return 0;
    }
    if (value > SIZE_MAX - (alignment - remainder)) return -1;
    *output = value + alignment - remainder;
    return 0;
}

static int checked_sum(size_t left, size_t right, size_t *output) {
    if (left > SIZE_MAX - right) return -1;
    *output = left + right;
    return 0;
}

int sfora_exact_rerank_context_bytes(
    uint32_t candidate_count, uint32_t dimensions, uint32_t base_is_u8,
    uint32_t memory_alignment, uint32_t offset_alignment, uint64_t *output
) {
    SforaRing ring;
    size_t row_bytes, maximum_read, buffer_stride, allocation_bytes;
    size_t allocation_alignment, mapping_bytes, total = 0, value;
    if (!output || candidate_count == 0 || candidate_count > 4096 || dimensions == 0 ||
        (base_is_u8 != 0 && base_is_u8 != 1) || memory_alignment == 0 ||
        offset_alignment == 0 || (memory_alignment & (memory_alignment - 1u)) != 0 ||
        (offset_alignment & (offset_alignment - 1u)) != 0)
        return -1;
    allocation_alignment = memory_alignment < sizeof(void *)
                               ? sizeof(void *)
                               : memory_alignment;
    if (checked_product(dimensions, base_is_u8 ? 1u : sizeof(float), &row_bytes) ||
        row_bytes > SIZE_MAX - (offset_alignment - 1u) ||
        align_up(row_bytes + offset_alignment - 1u, offset_alignment, &maximum_read) ||
        align_up(maximum_read, allocation_alignment, &buffer_stride) ||
        checked_product(candidate_count, buffer_stride, &allocation_bytes) ||
        allocation_bytes > SIZE_MAX - (allocation_alignment - 1u))
        return -1;
    mapping_bytes = allocation_bytes + allocation_alignment - 1u;
    if (ring_init(&ring, candidate_count) != 0)
        return (errno == EPERM || errno == EACCES || errno == ENOSYS) ? -5 : -2;
    if (checked_product(candidate_count, sizeof(SforaExactPair), &value) ||
        checked_sum(total, value, &total) ||
        checked_product(candidate_count, sizeof(size_t), &value) ||
        checked_sum(total, value, &total) || checked_sum(total, value, &total) ||
        checked_sum(total, candidate_count, &total) ||
        checked_sum(total, mapping_bytes, &total) ||
        checked_sum(total, ring.sqes_size, &total) ||
        checked_sum(total, ring.sq_ring_size, &total) ||
        (!ring.single_mmap && checked_sum(total, ring.cq_ring_size, &total))) {
        ring_destroy(&ring);
        return -1;
    }
    ring_destroy(&ring);
    *output = total;
    return 0;
}

int sfora_exact_rerank_direct(
    const float *query, size_t query_count,
    const uint32_t *candidate_ids, size_t candidate_count,
    uint32_t rows, uint32_t dimensions, uint32_t base_is_u8,
    uint32_t return_width, int direct_fd, uint64_t physical_bytes,
    uint32_t memory_alignment, uint32_t offset_alignment,
    uint32_t requested_threads,
    uint32_t *output_ids, double *output_distances, size_t output_capacity,
    uint64_t *physical_bytes_read, uint32_t *operations
) {
    SforaExactPair *pairs = NULL;
    size_t *inside = NULL;
    size_t *read_sizes = NULL;
    uint8_t *terminal = NULL;
    uint8_t *buffers = NULL;
    void *buffer_mapping = MAP_FAILED;
    SforaRing ring;
    size_t row_bytes, maximum_read, buffer_stride, allocation_bytes;
    size_t allocation_alignment, buffer_mapping_size;
    int data_fd = -1;
    int io_error = 0;
    int scoring_error = 0;
    int quiescence_unproven = 0;
    memset(&ring, 0, sizeof(ring));
    ring.fd = -1;
    if (!query || !candidate_ids || !output_ids || !output_distances ||
        !physical_bytes_read || !operations || candidate_count == 0 ||
        candidate_count > 4096 || rows == 0 || dimensions == 0 ||
        (base_is_u8 != 0 && base_is_u8 != 1) || return_width == 0 ||
        return_width > candidate_count || output_capacity < return_width || direct_fd < 0 ||
        memory_alignment == 0 || offset_alignment == 0 || query_count != dimensions ||
        (memory_alignment & (memory_alignment - 1u)) != 0 ||
        (offset_alignment & (offset_alignment - 1u)) != 0 ||
        requested_threads == 0 || requested_threads > 256)
        return -1;
    for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
        if (!isfinite(query[dimension]) ||
            (base_is_u8 &&
             (query[dimension] < 0.0f || query[dimension] > 255.0f ||
              query[dimension] != floorf(query[dimension]))))
            return -1;
    }
    allocation_alignment = memory_alignment < sizeof(void *)
                               ? sizeof(void *)
                               : memory_alignment;
    if (checked_product(dimensions, base_is_u8 ? 1u : sizeof(float), &row_bytes) ||
        row_bytes > SIZE_MAX - (offset_alignment - 1) ||
        align_up(row_bytes + offset_alignment - 1, offset_alignment, &maximum_read) ||
        align_up(maximum_read, allocation_alignment, &buffer_stride) ||
        checked_product(candidate_count, buffer_stride, &allocation_bytes) ||
        allocation_bytes > SIZE_MAX - (allocation_alignment - 1u))
        return -1;
    buffer_mapping_size = allocation_bytes + allocation_alignment - 1u;
    int expected_state = 0;
    if (!atomic_compare_exchange_strong(&sfora_direct_state, &expected_state, 1))
        return expected_state == 2 ? -4 : -7;
    pairs = malloc(candidate_count * sizeof(*pairs));
    inside = calloc(candidate_count, sizeof(*inside));
    read_sizes = calloc(candidate_count, sizeof(*read_sizes));
    terminal = calloc(candidate_count, sizeof(*terminal));
    data_fd = dup(direct_fd);
    if (sfora_direct_buffer_mapping != NULL &&
        sfora_direct_buffer_size == buffer_mapping_size) {
        buffer_mapping = sfora_direct_buffer_mapping;
    } else {
        if (sfora_direct_buffer_mapping != NULL) {
            munmap(sfora_direct_buffer_mapping, sfora_direct_buffer_size);
            sfora_direct_buffer_forget();
        }
        buffer_mapping = mmap(NULL, buffer_mapping_size, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (buffer_mapping != MAP_FAILED &&
            madvise(buffer_mapping, buffer_mapping_size, MADV_DONTFORK) == 0) {
            sfora_direct_buffer_mapping = buffer_mapping;
            sfora_direct_buffer_size = buffer_mapping_size;
        }
    }
    if (buffer_mapping != MAP_FAILED) {
        uintptr_t start = (uintptr_t)buffer_mapping;
        uintptr_t aligned = (start + allocation_alignment - 1u) &
                            ~(uintptr_t)(allocation_alignment - 1u);
        buffers = (uint8_t *)aligned;
    }
    if (!pairs || !inside || !read_sizes ||
        !terminal || data_fd < 0 || buffer_mapping == MAP_FAILED ||
        sfora_direct_buffer_mapping != buffer_mapping) {
        ring_destroy(&ring);
        if (data_fd >= 0) close(data_fd);
        if (buffer_mapping != MAP_FAILED) {
            munmap(buffer_mapping, buffer_mapping_size);
            sfora_direct_buffer_forget();
        }
        free(pairs); free(inside); free(read_sizes); free(terminal);
        atomic_store(&sfora_direct_state, 0);
        return -2;
    }
    if (ring_init(&ring, (unsigned)candidate_count) != 0) {
        int setup_errno = errno;
        if (data_fd >= 0) close(data_fd);
        munmap(buffer_mapping, buffer_mapping_size);
        sfora_direct_buffer_forget();
        free(pairs); free(inside); free(read_sizes); free(terminal);
        atomic_store(&sfora_direct_state, 0);
        return (setup_errno == EPERM || setup_errno == EACCES || setup_errno == ENOSYS)
                   ? -5
                   : -2;
    }
    *physical_bytes_read = 0;
    *operations = 0;
    for (size_t index = 0; index < candidate_count; ++index) {
        uint64_t byte_offset, aligned_offset;
        size_t required;
        if (candidate_ids[index] >= rows ||
            (uint64_t)candidate_ids[index] > (UINT64_MAX - 8u) / row_bytes) {
            io_error = 1;
            break;
        }
        pairs[index].id = candidate_ids[index];
        byte_offset = 8u + (uint64_t)candidate_ids[index] * row_bytes;
        aligned_offset = byte_offset - byte_offset % offset_alignment;
        inside[index] = (size_t)(byte_offset - aligned_offset);
        if (inside[index] > SIZE_MAX - row_bytes ||
            align_up(inside[index] + row_bytes, offset_alignment, &required) ||
            required > buffer_stride || aligned_offset > physical_bytes ||
            required > physical_bytes - aligned_offset || required > UINT_MAX) {
            io_error = 1;
            break;
        }
        read_sizes[index] = required;
        if (*physical_bytes_read > UINT64_MAX - required) {
            io_error = 1;
            break;
        }
        *physical_bytes_read += required;
    }
    unsigned tail = __atomic_load_n(ring.sq_tail, __ATOMIC_RELAXED);
    if (!io_error) {
        for (size_t index = 0; index < candidate_count; ++index) {
            uint64_t byte_offset = 8u + (uint64_t)candidate_ids[index] * row_bytes;
            uint64_t aligned_offset = byte_offset - byte_offset % offset_alignment;
            unsigned slot = tail & *ring.sq_mask;
            struct io_uring_sqe *entry = &ring.sqes[slot];
            memset(entry, 0, sizeof(*entry));
            entry->opcode = IORING_OP_READ;
            entry->fd = data_fd;
            entry->off = aligned_offset;
            entry->addr = (uint64_t)(uintptr_t)(buffers + index * buffer_stride);
            entry->len = (unsigned)read_sizes[index];
            entry->user_data = index;
            ring.sq_array[slot] = slot;
            ++tail;
        }
    }
    unsigned submitted_total = 0;
    if (!io_error) {
        unsigned initial_head = __atomic_load_n(ring.sq_head, __ATOMIC_ACQUIRE);
        __atomic_store_n(ring.sq_tail, tail, __ATOMIC_RELEASE);
        unsigned remaining = (unsigned)candidate_count;
        while (remaining) {
            int submitted = (int)syscall(__NR_io_uring_enter, ring.fd, remaining, 0, 0,
                                         NULL, 0);
            unsigned consumed = __atomic_load_n(ring.sq_head, __ATOMIC_ACQUIRE) -
                                initial_head;
            if (consumed > candidate_count || consumed < submitted_total) {
                quiescence_unproven = 1;
                break;
            }
            submitted_total = consumed;
            remaining = (unsigned)candidate_count - submitted_total;
            if (submitted < 0 && errno == EINTR) continue;
            if (submitted < 0 || (submitted == 0 && remaining != 0)) {
                io_error = 1;
                break;
            }
        }
    }
    unsigned completed = 0;
    while (!quiescence_unproven && completed < submitted_total) {
        if (__atomic_load_n(ring.cq_overflow, __ATOMIC_ACQUIRE) != 0 ||
            __atomic_load_n(ring.sq_dropped, __ATOMIC_ACQUIRE) != 0) {
            quiescence_unproven = 1;
            break;
        }
        unsigned head = __atomic_load_n(ring.cq_head, __ATOMIC_RELAXED);
        unsigned completion_tail = __atomic_load_n(ring.cq_tail, __ATOMIC_ACQUIRE);
        if (head == completion_tail) {
            int waited = (int)syscall(__NR_io_uring_enter, ring.fd, 0,
                                      1, IORING_ENTER_GETEVENTS,
                                      NULL, 0);
            if (waited < 0 && errno == EINTR) continue;
            if (waited < 0 && (errno == EAGAIN || errno == EBUSY)) {
                sched_yield();
                continue;
            }
            if (waited < 0) {
                quiescence_unproven = 1;
                break;
            }
            continue;
        }
        while (head != completion_tail) {
            struct io_uring_cqe *completion = &ring.cqes[head & *ring.cq_mask];
            size_t index = (size_t)completion->user_data;
            if (index >= submitted_total || terminal[index]) {
                quiescence_unproven = 1;
            } else {
                terminal[index] = 1;
                ++completed;
                if (completion->res < 0 ||
                    (size_t)completion->res != read_sizes[index])
                    io_error = 1;
            }
            ++head;
            if (quiescence_unproven) break;
        }
        __atomic_store_n(ring.cq_head, head, __ATOMIC_RELEASE);
    }
    if (quiescence_unproven) {
        sfora_direct_quarantine.ring = ring;
        sfora_direct_quarantine.pairs = pairs;
        sfora_direct_quarantine.inside = inside;
        sfora_direct_quarantine.read_sizes = read_sizes;
        sfora_direct_quarantine.terminal = terminal;
        sfora_direct_quarantine.buffers = buffers;
        sfora_direct_quarantine.buffer_mapping = buffer_mapping;
        sfora_direct_quarantine.buffer_mapping_size = buffer_mapping_size;
        sfora_direct_buffer_forget();
        sfora_direct_quarantine.data_fd = data_fd;
        atomic_store(&sfora_direct_state, 2);
        return -4;
    }
    if (!io_error && submitted_total == candidate_count) {
#pragma omp parallel for schedule(static) num_threads(requested_threads) reduction(|:scoring_error)
        for (size_t index = 0; index < candidate_count; ++index) {
            const uint8_t *raw = buffers + index * buffer_stride + inside[index];
            double distance = 0.0;
            if (base_is_u8) {
                uint64_t integer_distance = 0;
                for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
                    int64_t difference = (int64_t)query[dimension] - raw[dimension];
                    uint64_t square = (uint64_t)(difference * difference);
                    if (integer_distance > UINT64_MAX - square) scoring_error = 1;
                    else integer_distance += square;
                }
                distance = (double)integer_distance;
            } else {
                const float *row = (const float *)raw;
                for (uint32_t dimension = 0; dimension < dimensions; ++dimension) {
                    if (!isfinite(row[dimension])) scoring_error = 1;
                    double difference = (double)query[dimension] - row[dimension];
                    distance += difference * difference;
                }
                if (!isfinite(distance)) scoring_error = 1;
            }
            pairs[index].score = distance;
        }
        if (scoring_error) io_error = 2;
    }
    if (!io_error && submitted_total == candidate_count) {
        qsort(pairs, candidate_count, sizeof(*pairs), ascending_exact_pair);
        for (uint32_t index = 0; index < return_width; ++index) {
            output_ids[index] = pairs[index].id;
            output_distances[index] = pairs[index].score;
        }
        *operations = (uint32_t)candidate_count;
    }
    ring_destroy(&ring);
    close(data_fd);
    free(pairs); free(inside); free(read_sizes); free(terminal);
    atomic_store(&sfora_direct_state, 0);
    return io_error == 2 ? -6 : (io_error ? -3 : 0);
}

static int worse(SforaPair left, SforaPair right) {
    return left.score > right.score ||
           (left.score == right.score && left.id > right.id);
}

static void swap_pair(SforaPair *left, SforaPair *right) {
    SforaPair value = *left;
    *left = *right;
    *right = value;
}

static void heap_up(SforaPair *heap, size_t position) {
    while (position > 0) {
        size_t parent = (position - 1) / 2;
        if (!worse(heap[position], heap[parent])) return;
        swap_pair(&heap[position], &heap[parent]);
        position = parent;
    }
}

static void heap_down(SforaPair *heap, size_t count, size_t position) {
    for (;;) {
        size_t left = 2 * position + 1;
        size_t right = left + 1;
        size_t worst = position;
        if (left < count && worse(heap[left], heap[worst])) worst = left;
        if (right < count && worse(heap[right], heap[worst])) worst = right;
        if (worst == position) return;
        swap_pair(&heap[position], &heap[worst]);
        position = worst;
    }
}

static void offer(SforaPair *heap, size_t *count, size_t capacity, SforaPair value) {
    if (*count < capacity) {
        heap[*count] = value;
        heap_up(heap, *count);
        *count += 1;
    } else if (worse(heap[0], value)) {
        heap[0] = value;
        heap_down(heap, *count, 0);
    }
}

static int ascending_pair(const void *left, const void *right) {
    const SforaPair *a = (const SforaPair *)left;
    const SforaPair *b = (const SforaPair *)right;
    if (a->score < b->score) return -1;
    if (a->score > b->score) return 1;
    return (a->id > b->id) - (a->id < b->id);
}

static int checked_product(size_t left, size_t right, size_t *output) {
    if (right != 0 && left > SIZE_MAX / right) return -1;
    *output = left * right;
    return 0;
}

int sfora_direct_release_buffer(void) {
    int expected_state = 0;
    if (!atomic_compare_exchange_strong(&sfora_direct_state, &expected_state, 1))
        return expected_state == 2 ? -4 : -7;
    if (sfora_direct_buffer_mapping != NULL) {
        munmap(sfora_direct_buffer_mapping, sfora_direct_buffer_size);
        sfora_direct_buffer_forget();
    }
    atomic_store(&sfora_direct_state, 0);
    return 0;
}

int sfora_factorized_candidate_search(
    const float *query, size_t query_count,
    const float *coarse, size_t coarse_count,
    const float *pq, size_t pq_count,
    const uint64_t *offsets, size_t offsets_count,
    const uint32_t *ids, size_t ids_count,
    const uint8_t *codes, size_t codes_count,
    const uint8_t *norm_codes, size_t norm_codes_count,
    const float *norm_low, size_t norm_low_count,
    const float *norm_scale, size_t norm_scale_count,
    uint32_t nlist, uint32_t dimensions, uint32_t subquantizers,
    uint32_t bits_per_code, uint32_t nprobe, uint32_t shortlist,
    uint32_t requested_threads,
    uint32_t *output_ids, float *output_scores, size_t output_capacity,
    uint32_t *output_probe_lists, size_t probe_capacity,
    uint32_t *output_count,
    uint64_t *rows_scanned
) {
    size_t codebook_size, code_size, expected, rows;
    if (!query || !coarse || !pq || !offsets || !ids || !codes ||
        !norm_codes || !norm_low || !norm_scale || !output_ids ||
        !output_scores || !output_probe_lists || !output_count || !rows_scanned ||
        nlist == 0 || dimensions == 0 || subquantizers == 0 ||
        bits_per_code == 0 || bits_per_code > 8 ||
        dimensions % subquantizers != 0 || nprobe == 0 || nprobe > nlist ||
        shortlist == 0 || requested_threads == 0 || requested_threads > 256 ||
        output_capacity < shortlist || probe_capacity < nprobe)
        return -1;
    codebook_size = (size_t)1u << bits_per_code;
    code_size = ((size_t)subquantizers * bits_per_code + 7) / 8;
    rows = ids_count;
    if (query_count != dimensions || offsets_count != (size_t)nlist + 1 ||
        norm_low_count != nlist || norm_scale_count != nlist ||
        norm_codes_count != rows || offsets[0] != 0 || offsets[nlist] != rows ||
        checked_product(nlist, dimensions, &expected) || coarse_count != expected ||
        checked_product(subquantizers, codebook_size, &expected) ||
        checked_product(expected, dimensions / subquantizers, &expected) ||
        pq_count != expected || checked_product(rows, code_size, &expected) ||
        codes_count != expected)
        return -1;
    for (uint32_t list = 0; list < nlist; ++list) {
        if (offsets[list] > offsets[list + 1]) return -1;
    }

    SforaPair *list_heap = malloc((size_t)nprobe * sizeof(*list_heap));
    SforaPair *coarse_heaps =
        malloc((size_t)requested_threads * nprobe * sizeof(*coarse_heaps));
    size_t *coarse_counts = calloc((size_t)requested_threads, sizeof(*coarse_counts));
    int thread_count = (int)requested_threads;
    SforaPair *candidate_heaps = malloc((size_t)thread_count * shortlist * sizeof(*candidate_heaps));
    size_t *candidate_counts = calloc((size_t)thread_count, sizeof(*candidate_counts));
    float *lut = malloc((size_t)subquantizers * codebook_size * sizeof(*lut));
    if (!list_heap || !coarse_heaps || !coarse_counts ||
        !candidate_heaps || !candidate_counts || !lut) {
        free(list_heap); free(coarse_heaps); free(coarse_counts);
        free(candidate_heaps); free(candidate_counts); free(lut);
        return -2;
    }

    size_t list_count = 0;
    float query_norm = 0.0f;
    for (uint32_t j = 0; j < dimensions; ++j)
        query_norm = fmaf(query[j], query[j], query_norm);
    /* The coarse search streams every centroid, which is more bytes per query
       than the posting scan below it. Run it with the same per-thread heap and
       merge the posting scan uses; top-nprobe is partition-invariant, so the
       selection is identical to the serial order. */
    int coarse_error = 0;
#pragma omp parallel for schedule(static) num_threads(requested_threads)
    for (uint32_t list = 0; list < nlist; ++list) {
        int coarse_thread = 0;
#ifdef _OPENMP
        coarse_thread = omp_get_thread_num();
#endif
        float distance = 0.0f;
        const float *centroid = coarse + (size_t)list * dimensions;
        for (uint32_t j = 0; j < dimensions; ++j) {
            float delta = query[j] - centroid[j];
            distance = fmaf(delta, delta, distance);
        }
        if (!isfinite(distance)) {
#pragma omp atomic write
            coarse_error = 1;
        } else {
            offer(coarse_heaps + (size_t)coarse_thread * nprobe,
                  &coarse_counts[coarse_thread], nprobe,
                  (SforaPair){distance, list});
        }
    }
    if (coarse_error) goto invalid_score;
    for (uint32_t coarse_thread = 0; coarse_thread < requested_threads; ++coarse_thread) {
        SforaPair *source = coarse_heaps + (size_t)coarse_thread * nprobe;
        for (size_t index = 0; index < coarse_counts[coarse_thread]; ++index)
            offer(list_heap, &list_count, nprobe, source[index]);
    }
    qsort(list_heap, list_count, sizeof(*list_heap), ascending_pair);
    for (size_t index = 0; index < list_count; ++index)
        output_probe_lists[index] = list_heap[index].id;

    uint32_t dsub = dimensions / subquantizers;
    for (uint32_t m = 0; m < subquantizers; ++m) {
        for (size_t code = 0; code < codebook_size; ++code) {
            float dot = 0.0f;
            const float *center = pq + ((size_t)m * codebook_size + code) * dsub;
            for (uint32_t j = 0; j < dsub; ++j)
                dot = fmaf(query[(size_t)m * dsub + j], center[j], dot);
            lut[(size_t)m * codebook_size + code] = -2.0f * dot;
        }
    }

    int score_error = 0;
/* Probed lists are ordered by centroid distance and vary widely in length, so a
   static contiguous partition hands the densest lists to one thread and wall
   time tracks the unluckiest worker. The loop body is independent and the
   merged top-k is partition-invariant, so dynamic scheduling is result-neutral. */
#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (size_t probe = 0; probe < list_count; ++probe) {
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif
        SforaPair *local_heap = candidate_heaps + (size_t)thread_id * shortlist;
        size_t *local_count = &candidate_counts[thread_id];
        uint32_t list = list_heap[probe].id;
        const float *centroid = coarse + (size_t)list * dimensions;
        float coarse_dot = 0.0f;
        for (uint32_t j = 0; j < dimensions; ++j)
            coarse_dot = fmaf(query[j], centroid[j], coarse_dot);
        float constant = query_norm - 2.0f * coarse_dot + norm_low[list];
        for (uint64_t row = offsets[list]; row < offsets[list + 1]; ++row) {
            float score = constant + norm_scale[list] * norm_codes[row];
            const uint8_t *packed_code = codes + row * code_size;
            for (uint32_t m = 0; m < subquantizers; ++m) {
                size_t bit = (size_t)m * bits_per_code;
                size_t byte = bit >> 3;
                uint32_t shift = (uint32_t)(bit & 7);
                uint32_t packed = packed_code[byte];
                if (shift + bits_per_code > 8)
                    packed |= (uint32_t)packed_code[byte + 1] << 8;
                uint32_t code = (packed >> shift) & (((uint32_t)1 << bits_per_code) - 1);
                score += lut[(size_t)m * codebook_size + code];
            }
            if (isfinite(score)) {
                offer(local_heap, local_count, shortlist, (SforaPair){score, ids[row]});
            } else {
#pragma omp atomic write
                score_error = 1;
            }
        }
    }

    if (score_error) goto invalid_score;

    size_t candidate_count = candidate_counts[0];
    SforaPair *candidate_heap = candidate_heaps;
    for (int thread_id = 1; thread_id < thread_count; ++thread_id) {
        SforaPair *local_heap = candidate_heaps + (size_t)thread_id * shortlist;
        for (size_t index = 0; index < candidate_counts[thread_id]; ++index)
            offer(candidate_heap, &candidate_count, shortlist, local_heap[index]);
    }
    qsort(candidate_heap, candidate_count, sizeof(*candidate_heap), ascending_pair);
    for (size_t index = 0; index < candidate_count; ++index) {
        output_ids[index] = candidate_heap[index].id;
        output_scores[index] = candidate_heap[index].score;
    }
    *output_count = (uint32_t)candidate_count;
    *rows_scanned = 0;
    for (size_t index = 0; index < list_count; ++index) {
        uint32_t list = list_heap[index].id;
        *rows_scanned += offsets[list + 1] - offsets[list];
    }
    free(list_heap); free(coarse_heaps); free(coarse_counts);
    free(candidate_heaps); free(candidate_counts); free(lut);
    return 0;

invalid_score:
    free(list_heap); free(coarse_heaps); free(coarse_counts);
    free(candidate_heaps); free(candidate_counts); free(lut);
    return -4;
}
