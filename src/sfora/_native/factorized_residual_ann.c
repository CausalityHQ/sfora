#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
    float score;
    uint32_t id;
} SforaPair;

uint32_t sfora_factorized_backend_abi_version(void) {
    return 1;
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
    int thread_count = (int)requested_threads;
    SforaPair *candidate_heaps = malloc((size_t)thread_count * shortlist * sizeof(*candidate_heaps));
    size_t *candidate_counts = calloc((size_t)thread_count, sizeof(*candidate_counts));
    float *lut = malloc((size_t)subquantizers * codebook_size * sizeof(*lut));
    if (!list_heap || !candidate_heaps || !candidate_counts || !lut) {
        free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
        return -2;
    }

    size_t list_count = 0;
    float query_norm = 0.0f;
    for (uint32_t j = 0; j < dimensions; ++j)
        query_norm = fmaf(query[j], query[j], query_norm);
    for (uint32_t list = 0; list < nlist; ++list) {
        float distance = 0.0f;
        const float *centroid = coarse + (size_t)list * dimensions;
        for (uint32_t j = 0; j < dimensions; ++j) {
            float delta = query[j] - centroid[j];
            distance = fmaf(delta, delta, distance);
        }
        if (!isfinite(distance)) goto invalid_score;
        offer(list_heap, &list_count, nprobe, (SforaPair){distance, list});
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
#pragma omp parallel for schedule(static) num_threads(thread_count)
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
    free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
    return 0;

invalid_score:
    free(list_heap); free(candidate_heaps); free(candidate_counts); free(lut);
    return -4;
}
