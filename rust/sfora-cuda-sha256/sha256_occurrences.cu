// Standalone source hypothesis only; see docs/gpu_sha256_source_contract_2026-10-09.md.
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
__device__ __constant__ uint32_t K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};
__device__ __constant__ uint32_t IV[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
};

__device__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}
__device__ uint32_t small0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}
__device__ uint32_t small1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}
__device__ uint32_t big0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}
__device__ uint32_t big1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}
__device__ uint32_t choose(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}
__device__ uint32_t majority(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
__device__ uint64_t padded_blocks(uint64_t n) {
    return n / 64 + 1 + (n % 64 >= 56);
}
__device__ unsigned char padded_byte(const unsigned char* data, uint64_t n,
                                    uint64_t pos, uint64_t total) {
    if (pos < n) return data[pos];
    if (pos == n) return 0x80;
    if (pos >= total - 8) return (n * 8) >> ((total - 1 - pos) * 8);
    return 0;
}
__device__ uint32_t load_word(const unsigned char* data, uint64_t n,
                             uint64_t pos, uint64_t total) {
    return ((uint32_t(padded_byte(data, n, pos, total)) << 24) |
            (uint32_t(padded_byte(data, n, pos + 1, total)) << 16) |
            (uint32_t(padded_byte(data, n, pos + 2, total)) << 8) |
            uint32_t(padded_byte(data, n, pos + 3, total)));
}

__device__ void hash_leaf(const unsigned char* data, uint64_t n, unsigned char* output) {
    uint32_t state[8];
    for (uint32_t i = 0; i < 8; ++i) state[i] = IV[i];
    uint64_t blocks = padded_blocks(n);
    // shortcut: one serial chain per thread; reconsider layout only after authenticated timings.
    for (uint64_t block = 0; block < blocks; ++block) {
        uint32_t w[64];
        for (uint32_t i = 0; i < 16; ++i)
            w[i] = load_word(data, n, block * 64 + i * 4, blocks * 64);
        for (uint32_t i = 16; i < 64; ++i)
            w[i] = small1(w[i - 2]) + w[i - 7] + small0(w[i - 15]) + w[i - 16];
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (uint32_t i = 0; i < 64; ++i) {
            uint32_t t1 = h + big1(e) + choose(e, f, g) + K[i] + w[i];
            uint32_t t2 = big0(a) + majority(a, b, c);
            // round state
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        uint32_t working[8] = {a, b, c, d, e, f, g, h};
        for (uint32_t i = 0; i < 8; ++i) state[i] += working[i];
    }
    for (uint32_t i = 0; i < 32; ++i)
        output[i] = state[i / 4] >> (24 - 8 * (i % 4));
}

__global__ void sha256_occurrences(const unsigned char* const* device_ptrs,
                                   const uint64_t* device_byte_lengths,
                                   uint64_t occurrence_count, unsigned char* device_output) {
    uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= occurrence_count) return;
    hash_leaf(device_ptrs[index], device_byte_lengths[index], device_output + index * 32);
}
}  // namespace

// Enqueue only: callers retain every owner through checked completion, including failure paths.
extern "C" cudaError_t sfora_sha256_occurrences(const unsigned char* const* device_ptrs,
                                               const uint64_t* device_byte_lengths,
                                               uint64_t occurrence_count,
                                               unsigned char* device_output,
                                               cudaStream_t current_stream) {
    if (occurrence_count > UINT32_MAX) return cudaErrorInvalidValue;
    if (occurrence_count == 0) return cudaSuccess;
    if (!device_ptrs || !device_byte_lengths || !device_output) return cudaErrorInvalidValue;
    unsigned int blocks = (occurrence_count + 127) / 128;
    sha256_occurrences<<<blocks, 128, 0, current_stream>>>(
        device_ptrs, device_byte_lengths, occurrence_count, device_output);
    return cudaGetLastError();
}
