/* gps1 CUDA kernels, sort engine (msieve-shaped, CADO candidate space).
 * Three passes per batch of (q, rq) items:
 *   kernel_invqq  1/q^2 mod p^2 for every (prime, distinct q)
 *   kernel_fill   transformed roots expanded over [-M, M) as (key, root) pairs
 *   radix sort    all entries of the batch at once (CUB)
 *   kernel_scan   adjacent equal keys -> matches
 * See gps1.cpp for the host driver. */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "gps1.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

#define KEY_NONE UINT64_MAX     /* padding entry, sorts last */

__device__ uint64_t d_match_count;

static __device__ inline uint64_t mulmod64(uint64_t a, uint64_t b, uint64_t m)
{
    return (uint64_t)(((unsigned __int128) a * b) % m);
}

static __device__ inline uint64_t mod128(uint64_t hi, uint64_t lo, uint64_t m)
{
    return (uint64_t)(((((unsigned __int128) hi << 64) | lo)) % m);
}

/* inverse of a mod m, gcd(a,m)=1 */
static __device__ uint64_t invmod64(uint64_t a, uint64_t m)
{
    int64_t u = 1, v = 0;
    uint64_t g = a, h = m;
    while (h) {
        uint64_t qq = g / h;
        uint64_t t = g - qq * h; g = h; h = t;
        int64_t s = u - (int64_t) qq * v; u = v; v = s;
    }
    return u >= 0 ? (uint64_t) u : m - (uint64_t) (-u);
}

/* invqq[j * nprimes + i] = 1/qs[j]^2 mod primes[i]^2 */
__global__ void kernel_invqq(const uint32_t *primes, uint32_t nprimes,
                             const uint64_t *qs, uint32_t ndq, uint64_t *invqq)
{
    uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (uint64_t) nprimes * ndq) return;
    uint32_t i = t % nprimes, j = t / nprimes;
    uint64_t pp = (uint64_t) primes[i] * primes[i];
    uint64_t qm = qs[j] % pp;
    invqq[t] = invmod64(mulmod64(qm, qm, pp), pp);
}

/* One thread per (root, batch item). Computes u0 = (rp - rq)/q^2 mod p^2 and
 * writes the class over [-M, M) into the item's entry segment; unused slots
 * get KEY_NONE. */
__global__ void kernel_fill(const uint32_t *primes, const uint32_t *root_prime,
                            const uint64_t *roots, uint32_t nroots,
                            const uint32_t *entry_off, const uint32_t *qidx,
                            const uint64_t *rq_hi, const uint64_t *rq_lo,
                            const uint64_t *invqq, uint32_t nprimes,
                            uint32_t batch, int64_t M, uint32_t offset_bits,
                            uint64_t slots, uint64_t *keys, uint32_t *vals)
{
    uint64_t t = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (uint64_t) nroots * batch) return;
    uint32_t c = t / batch, b = t % batch;
    uint32_t pi = root_prime[c];
    uint64_t pp = (uint64_t) primes[pi] * primes[pi];
    uint64_t rqi = mod128(rq_hi[b], rq_lo[b], pp);
    uint64_t rp = roots[c];
    uint64_t u0 = rp >= rqi ? rp - rqi : rp + pp - rqi;
    u0 = mulmod64(u0, invqq[(uint64_t) qidx[b] * nprimes + pi], pp);

    /* smallest u >= -M in the class of u0 */
    int64_t u = (int64_t) u0 - (int64_t)(((uint64_t)((int64_t) u0 + M)) / pp) * (int64_t) pp;
    uint64_t e = (uint64_t) b * slots + entry_off[c];
    uint32_t n = entry_off[c + 1] - entry_off[c];
    for (uint32_t s = 0; s < n; s++, u += (int64_t) pp) {
        keys[e + s] = u < M ? ((uint64_t) b << offset_bits) | (uint64_t)(u + M)
                            : KEY_NONE;
        vals[e + s] = c;
    }
}

/* For each run of equal keys, emit every root pair with distinct primes,
 * like CADO's hash insert does. */
__global__ void kernel_scan(const uint64_t *keys, const uint32_t *vals,
                            uint64_t n, const uint32_t *root_prime,
                            const uint32_t *primes, int64_t M,
                            uint32_t offset_bits, match_t *matches)
{
    uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 1 >= n) return;
    uint64_t key = keys[i];
    if (key == KEY_NONE || (i > 0 && keys[i - 1] == key))
        return;                 /* not the first entry of a run */
    for (uint64_t j = i + 1; j < n && keys[j] == key; j++)
        for (uint64_t l = i; l < j; l++) {
            uint32_t p1 = primes[root_prime[vals[l]]];
            uint32_t p2 = primes[root_prime[vals[j]]];
            if (p1 == p2)
                continue;
            uint64_t m = atomicAdd((unsigned long long *) &d_match_count, 1ull);
            assert(m < GPS1_MAX_MATCHES);
            matches[m] = { p1, p2, (uint32_t)(key >> offset_bits), 0,
                           (int64_t)(key & (((uint64_t) 1 << offset_bits) - 1)) - M };
        }
}

extern "C" uint64_t gps1_run(const gpu_t *G, uint32_t nroots, uint32_t ndq,
                             uint32_t batch, int64_t M, uint64_t slots)
{
    /* the raw search runs on its own stream and synchronizes only it, so
     * a size-optimization batch on another stream keeps running */
    static cudaStream_t st;
    if (!st)
        CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    uint64_t zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbolAsync(d_match_count, &zero, 8, 0,
                                       cudaMemcpyHostToDevice, st));

    uint32_t offset_bits = gps1_offset_bits(M);
    int sort_bits = offset_bits + 64 - __builtin_clzll((uint64_t) batch);

    uint64_t nq_total = (uint64_t) G->nprimes * ndq;
    kernel_invqq<<<(nq_total + 255) / 256, 256, 0, st>>>(G->primes, G->nprimes,
                                                         G->qs, ndq, G->invqq);
    uint64_t total = (uint64_t) nroots * batch;
    kernel_fill<<<(total + 255) / 256, 256, 0, st>>>(G->primes, G->root_prime,
                                              G->roots,
                                              nroots, G->entry_off, G->qidx,
                                              G->rq_hi, G->rq_lo, G->invqq,
                                              G->nprimes, batch, M, offset_bits,
                                              slots, G->keys, G->vals);

    uint64_t nentries = slots * batch;
    assert(nentries <= GPS1_MAX_ENTRIES);
    cub::DoubleBuffer<uint64_t> dkeys(G->keys, G->keys_alt);
    cub::DoubleBuffer<uint32_t> dvals(G->vals, G->vals_alt);
    size_t tmp_bytes = G->sort_tmp_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(G->sort_tmp, tmp_bytes,
                                               dkeys, dvals, nentries, 0,
                                               sort_bits, st));
    kernel_scan<<<(nentries + 255) / 256, 256, 0, st>>>(dkeys.Current(),
                                                 dvals.Current(),
                                                 nentries, G->root_prime,
                                                 G->primes, M, offset_bits,
                                                 G->matches);
    uint64_t nmatches;
    CUDA_CHECK(cudaMemcpyFromSymbolAsync(&nmatches, d_match_count, 8, 0,
                                         cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));
    assert(nmatches < GPS1_MAX_MATCHES);
    return nmatches;
}

extern "C" size_t gps1_sort_tmp_bytes(uint64_t nentries)
{
    size_t bytes = 0;
    cub::DoubleBuffer<uint64_t> dkeys(NULL, NULL);
    cub::DoubleBuffer<uint32_t> dvals(NULL, NULL);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, bytes, dkeys, dvals,
                                               nentries, 0, 64));
    return bytes;
}
